import test from 'node:test';
import assert from 'node:assert/strict';
import {createConfigurationMaintenance} from '../scripts/config-maintenance.mjs';
let api={};try{api=await import('../scripts/fortress-rule-compat.mjs')}catch(error){if(error.code!=='ERR_MODULE_NOT_FOUND')throw error}

const NS='pf2e-third-party-automation';
const ALBATROSS='FTGjyYBZJ4JCpTZw',HANDS='fzUfZcstmQZGGwcJ';
const CURSE='Compendium.pf2e.spell-effects.Item.jSvpjSGnIVBAuFDu',SLOWED='Compendium.patreon-v3.effects.Item.xTHKw0VHuqu6SI93';
const HANDS_EFFECT='Compendium.pf2e.spell-effects.Item.lyLMiauxIVUM3oF1';
const OVERTURE='Compendium.pf2e.spell-effects.Item.ITErgFRfydm1xmnW';
const world=()=>({world:{id:'ujx5r8oipw7ercdr'},system:{id:'pf2e',version:'8.5.1'},user:{id:'gm'},users:{activeGM:{id:'gm'}},modules:new Map([['patreon-v3',{active:true,version:'3.2.28'}]])});
const original=()=>({
 [ALBATROSS]:{uuid:ALBATROSS,source:['Compendium.pf2e.spells-srd.Item.93SjFTGJUTTmAt6j'],isActive:true,baseRules:[{type:'base',triggerType:'spell-cast',target:'TargetEffect',value:'Compendium.pf2e.spell-effects.Item.b8BZHeuz5jH8E2SG',predicate:['item:albatross-curse']}],complexRules:[
  {type:'complex',triggerType:'saving-throw',extend:null,range:null,target:'SelfEffect',predicate:['origin:item:albatross-curse','outcome:success'],values:[{conditions:['stupefied'],effects:[],duration:{unit:'rounds',value:1}}]},
  {type:'complex',triggerType:'saving-throw',extend:null,range:null,target:'SelfEffect',predicate:['origin:item:albatross-curse','outcome:criticalFailure'],values:[{conditions:[],effects:[SLOWED],duration:{unit:'hours'}}]},
 ]},
 [HANDS]:{uuid:HANDS,source:['Compendium.pf2e.spells-srd.Item.zNN9212H2FGfM7VS'],isActive:true,baseRules:[
  {type:'base',triggerType:'saving-throw',extend:null,range:null,target:'TargetEffect',value:'Compendium.pf2e.spell-effects.Item.JhihziXQuoteftdd',predicate:['outcome:criticalFailure','item:slug:lay-on-hands','self:trait:undead']},
  {type:'base',triggerType:'damage-roll',extend:null,range:null,target:'TargetEffect',value:HANDS_EFFECT,predicate:['origin:item:lay-on-hands',{not:'target:trait:undead'},{not:{eq:['{actor|signature}','{target|signature}']}}]},
 ],complexRules:[]},
 custom:{uuid:'custom',isActive:false,baseRules:[{target:'TargetEffect',value:'custom-effect',predicate:['custom']}]},
});
const repair=(rules,game=world())=>api.buildFortressPatreonRepairs?.(rules,{game})??{rules:structuredClone(rules),changes:[]};
const predicate=(term,options,context)=>{
 if(typeof term==='string')return options.has(term);
 if(term.or)return term.or.some(t=>predicate(t,options,context));
 if(term.not)return !predicate(term.not,options,context);
 if(term.eq)return context.mainActor.id===context.targetActor.id;
 throw Error('Unsupported captured predicate');
};
// Boundary adapter for captured Patreon handlers. The native save actor and a
// deliberately unrelated selected target remain distinct throughout each test.
function dispatch(group,type,options,context){
 if(!group.isActive)return [];
 return [...group.baseRules??[],...group.complexRules??[]].filter(r=>r.triggerType===type&&r.predicate.every(p=>predicate(p,new Set(options),context))).flatMap(r=>{
  const recipient=r.target==='SelfEffect'?context.mainActor:context.targetActor;
  return (r.type==='complex'?r.values.flatMap(v=>v.effects):[r.value]).map(effect=>({recipient,effect}));
 });
}
// The exported unified effect has no choice: CreaturePF2e supplies self:mode.
const handsRules=[{key:'FlatModifier',selector:'ac',type:'status',value:2},{key:'AdjustModifier',mode:'downgrade',predicate:['self:mode:undead'],selector:'ac',slug:'lay-on-hands',value:-2}];
function handsAC(applied){
 if(!applied||applied.effect!==HANDS_EFFECT)return null;
 const options=new Set(['self:mode:'+applied.recipient.modeOfBeing]);
 const base=handsRules.find(r=>r.key==='FlatModifier').value;
 return handsRules.filter(r=>r.key==='AdjustModifier'&&r.predicate.every(p=>options.has(p))).reduce((value,r)=>Math.min(value,r.value),base);
}

test('Albatross critical failure resolves to the one-hour Will misfortune instead of Slowed, preserving other outcomes',()=>{
 const before=original(),result=repair(before),context={mainActor:{id:'saver'},targetActor:{id:'selected'}};
 const effects=dispatch(result.rules[ALBATROSS],'saving-throw',['origin:item:albatross-curse','outcome:criticalFailure'],context);
 assert.deepEqual(effects,[{recipient:context.mainActor,effect:CURSE}]);
 assert.deepEqual(result.rules[ALBATROSS].complexRules[1].values[0].duration,{unit:'hours'});
 assert.deepEqual(result.rules[ALBATROSS].complexRules[0],before[ALBATROSS].complexRules[0]);
 assert.deepEqual(before,original(),'planning a repair must not mutate live settings');
});

test('Lay on Hands failed saves penalize the saving undead even when a different living token is selected',()=>{
 const group=repair(original()).rules[HANDS],undead={id:'saver',modeOfBeing:'undead'},selected={id:'unrelated',modeOfBeing:'living'},context={mainActor:undead,targetActor:selected};
 for(const outcome of ['failure','criticalFailure']){
  const applied=dispatch(group,'saving-throw',['item:slug:lay-on-hands','self:trait:undead','outcome:'+outcome],context);
  assert.equal(applied.length,1,outcome);assert.equal(applied[0].recipient,undead);assert.equal(handsAC(applied[0]),-2);
 }
 for(const outcome of ['success','criticalSuccess'])assert.deepEqual(dispatch(group,'saving-throw',['item:slug:lay-on-hands','self:trait:undead','outcome:'+outcome],context),[]);
});

test('Lay on Hands keeps the existing living-recipient bonus and excludes self healing',()=>{
 const before=original(),result=repair(before),caster={id:'caster',modeOfBeing:'living'},ally={id:'ally',modeOfBeing:'living'};
 assert.deepEqual(result.rules[HANDS].baseRules[1],before[HANDS].baseRules[1]);
 const applied=dispatch(result.rules[HANDS],'damage-roll',['origin:item:lay-on-hands'],{mainActor:caster,targetActor:ally});
 assert.equal(handsAC(applied[0]),2);
 assert.deepEqual(dispatch(result.rules[HANDS],'damage-roll',['origin:item:lay-on-hands'],{mainActor:caster,targetActor:caster}),[]);
});

test('configuration maintenance backs up the exact changes, preserves disabled/custom rules, and is idempotent',async()=>{
 const game=world(),before=original();before[HANDS].isActive=false;before[ALBATROSS].complexRules[1].predicate.push('custom:confirmed');
 const data=new Map([['patreon-v3.rulesV3',before],[NS+'.configurationBackups',[]]]),writes=[];
 game.settings={get:(module,key)=>data.get(module+'.'+key),set:async(module,key,value)=>{writes.push(module+'.'+key);data.set(module+'.'+key,value)}};
 const maintain=createConfigurationMaintenance({game,repairs:[rules=>repair(rules,game)]});
 await maintain();await maintain();
 const result=data.get('patreon-v3.rulesV3');
 assert.equal(result[HANDS].isActive,false);assert.deepEqual(result.custom,before.custom);assert.ok(result[ALBATROSS].complexRules[1].predicate.includes('custom:confirmed'));
 assert.deepEqual(writes,[NS+'.configurationBackups','patreon-v3.rulesV3']);
 const changes=data.get(NS+'.configurationBackups')[0].changes;
 assert.equal(changes.length,4);
 for(const change of changes){
  const at=(value,path)=>path.split('.').reduce((node,key)=>node?.[key],value);
  assert.deepEqual(at(before,change.path),change.before);assert.deepEqual(at(result,change.path),change.after);assert.ok(change.reason);
 }
});

test('unknown worlds, systems and Patreon builds leave settings unchanged',()=>{
 for(const change of [g=>g.world.id='another-world',g=>g.system.version='8.6.0',g=>g.system.id='sf2e',g=>g.modules.get('patreon-v3').version='3.2.29',g=>g.modules.get('patreon-v3').active=false]){
  const game=world();change(game);const before=original(),result=repair(before,game);assert.deepEqual(result.rules,before);assert.deepEqual(result.changes,[]);
 }
});

test('edited rule identity, outcome, target, effect and duration are not overwritten',()=>{
 const alterations=[
  r=>r[HANDS].source.push('custom-source'),r=>r[HANDS].uuid='custom',r=>r[HANDS].baseRules[0].value='custom-effect',r=>r[HANDS].baseRules[0].target='OriginEffect',r=>r[HANDS].baseRules[0].predicate=['outcome:failure','item:slug:lay-on-hands','self:trait:undead'],
  r=>r[ALBATROSS].complexRules[1].values[0].effects=['custom-effect'],r=>r[ALBATROSS].complexRules[1].values[0].duration.value=2,r=>r[ALBATROSS].complexRules[1].target='TargetEffect',
 ];
 for(const alter of alterations){
  const before=original();alter(before);const result=repair(before);
  const changed=JSON.stringify(before[HANDS])!==JSON.stringify(original()[HANDS])?HANDS:ALBATROSS;
  assert.deepEqual(result.rules[changed],before[changed]);
 }
});

test('custom non-array rule containers are skipped without blocking other known repairs',()=>{
 for(const [id,key] of [[ALBATROSS,'complexRules'],[HANDS,'baseRules']]){
  const before=original();before[id][key]={custom:'editor-owned'};
  let result;assert.doesNotThrow(()=>{result=repair(before)});assert.deepEqual(result.rules[id],before[id]);assert.ok(result.changes.length>0);
 }
});

function overture(){return {_id:'effect',type:'effect',_stats:{compendiumSource:OVERTURE},flags:{},system:{duration:{unit:'rounds',value:1},rules:[
 {key:'RollOption',option:'uplifting-overture:origin:signature:{item|origin.signature}'},
 {adjustment:{success:'one-degree-better'},key:'AdjustDegreeOfSuccess',predicate:[{lte:['skill:performance:rank',3]}],selector:'performance'},
 {adjustment:{all:'to-critical-success'},key:'AdjustDegreeOfSuccess',predicate:[{gte:['skill:performance:rank',4]}],selector:'performance'},
 ]}};}
function adjustedDegree(effect,degree,rank){
 const outcomes=['criticalFailure','failure','success','criticalSuccess'];
 const rule=effect.system.rules.find(r=>r.key==='AdjustDegreeOfSuccess'&&r.selector==='performance'&&(r.predicate[0].lte?rank<=r.predicate[0].lte[1]:rank>=r.predicate[0].gte[1]));
 const adjustment=rule?.adjustment[degree]??rule?.adjustment.all;
 return adjustment==='to-critical-success'?'criticalSuccess':adjustment==='one-degree-better'?outcomes[Math.min(3,outcomes.indexOf(degree)+1)]:degree;
}
function actorFixture(){
 const game=world(),item=overture(),actor={uuid:'Actor.bard',flags:{},items:[],updates:[],async updateEmbeddedDocuments(type,updates){assert.equal(type,'Item');for(const update of updates){actor.updates.push(structuredClone(update));item.system.rules=structuredClone(update['system.rules']);item.flags[NS]={fortressRuleRepair:update['flags.'+NS+'.fortressRuleRepair']}}}};
 item.id=item._id;item.actor=actor;item.updateSource=update=>{item.system.rules=structuredClone(update['system.rules']);item.flags[NS]={fortressRuleRepair:update['flags.'+NS+'.fortressRuleRepair']}};actor.items=[item];
 return {game,item,actor};
}
test('existing Uplifting Overture changes failure to success while ordinary success and legendary behavior remain correct',async()=>{
 const f=actorFixture(),before=structuredClone(f.item.system.rules),provider=api.createFortressRuleCompatibility?.({game:f.game});
 await provider?.maintain(f.actor);await provider?.maintain(f.actor);
 assert.equal(adjustedDegree(f.item,'failure',2),'success');assert.equal(adjustedDegree(f.item,'success',2),'success');assert.equal(adjustedDegree(f.item,'criticalFailure',2),'criticalFailure');assert.equal(adjustedDegree(f.item,'failure',4),'criticalSuccess');
 assert.equal(f.actor.updates.length,1);assert.deepEqual(f.item.flags[NS].fortressRuleRepair.before['system.rules'],before);assert.deepEqual(f.item.system.rules[2],before[2]);
});

test('new Uplifting effects are corrected synchronously on the creating player client without a settings write',()=>{
 const f=actorFixture();f.game.user={id:'player'};const listeners=new Map(),Hooks={on:(name,fn)=>{listeners.set(name,fn);return name},off:(name,id)=>{assert.equal(name,id);listeners.delete(name)}};
 const provider=api.createFortressRuleCompatibility?.({game:f.game}),unregister=provider?.register({Hooks});
 listeners.get('preCreateItem')?.(f.item,{}, {},'player');
 assert.equal(adjustedDegree(f.item,'failure',2),'success');assert.equal(f.actor.updates.length,0);assert.equal(listeners.size,1);
 unregister();assert.equal(listeners.size,0);
});

test('unsupported worlds and system versions install no Uplifting item listener',()=>{
 for(const alter of [g=>g.world.id='another-world',g=>g.system.id='sf2e',g=>g.system.version='8.6.0']){
  const game=world(),calls=[];alter(game);
  const unregister=api.createFortressRuleCompatibility({game}).register({Hooks:{on:(...args)=>{calls.push(['on',...args]);return 1},off:(...args)=>calls.push(['off',...args])}});
  assert.equal(typeof unregister,'function');unregister();assert.deepEqual(calls,[]);
 }
});

test('Uplifting maintenance respects active GM, world/version, opt-out and exact source/bug shape',async()=>{
 for(const alter of [f=>f.game.user.id='player',f=>f.game.world.id='other',f=>f.game.system.version='8.6.0',f=>f.actor.flags[NS]={autoRepairDisabled:true},f=>f.item._stats.compendiumSource='custom-effect',f=>f.item.type='feat',f=>f.item.system.rules[1].adjustment={success:'to-critical-success'},f=>f.item.system.rules[1].predicate.push('custom:only')]){
  const f=actorFixture();alter(f);const before=structuredClone(f.item.system.rules),provider=api.createFortressRuleCompatibility?.({game:f.game});await provider?.maintain(f.actor);assert.deepEqual(f.item.system.rules,before);assert.equal(f.actor.updates.length,0);
 }
});
