import test from 'node:test';
import assert from 'node:assert/strict';
import {createConfigurationMaintenance} from '../scripts/config-maintenance.mjs';
import {Predicate} from './knowledge-pf2e-861-fixture.mjs';
let api={};try{api=await import('../scripts/bob2-swashbuckler.mjs')}catch(error){if(error.code!=='ERR_MODULE_NOT_FOUND')throw error}
const ID='pf2e-third-party-automation';
const S={panache:'Compendium.pf2e.feat-effects.Item.uBJsxCzNhje8m8jj',temporary:'Compendium.pf2e.feat-effects.Item.paKKHPds77fq5VTH',feature:'Compendium.pf2e.classfeatures.Item.LzYi0OuOoypNb6jd',flying:'Compendium.pf2e.feats-srd.Item.qgNc5XwjsaWET0Op',riposte:'Compendium.pf2e.actionspf2e.Item.EfjoIuDmtUn4yiow',charmed:'Compendium.pf2e.feats-srd.Item.DkoxNw9tsFFXrfJY'};
const thrown=['item:ranged','item:thrown','item:trait:agile','feat:flying-blade','target:range-increment:1','finisher','self:effect:panache'];
function set(object,path,value){const parts=path.split('.'),last=parts.pop();let node=object;for(const part of parts)node=node[part]??={};node[last]=structuredClone(value);}
function fixture({choice='use'}={}){
 const hooks=new Map(),gm={id:'gm',isGM:true,active:true},player={id:'player',isGM:false,active:true};
 const game={world:{id:'bob-2'},system:{id:'pf2e'},user:player,users:new Map([['gm',gm],['player',player]]),actors:new Map(),messages:new Map(),modules:new Map([['patreon-v3',{active:true}]]),time:{worldTime:120},combat:{id:'combat',started:true}};game.users.activeGM=gm;
 const actor={id:'hero',uuid:'Actor.hero',type:'character',flags:{},items:new Map(),testUserPermission:u=>[gm,player].includes(u),combatant:{initiative:20},update:async changes=>{for(const [p,v]of Object.entries(changes))set(actor,p,v)},toggleRollOption:async(domain,option,itemId,value)=>{const item=actor.items.get(itemId);item.system.rules.find(r=>r.option===option).value=value;cleared.push(option)},createEmbeddedDocuments:async(_type,data)=>data.map(d=>add(d)),deleteEmbeddedDocuments:async(_type,ids)=>{deleted.push(...ids);for(const id of ids)actor.items.delete(id)}};
 game.actors.set(actor.id,actor);const deleted=[],cleared=[];let counter=0,prompts=0;
 function add(data){const item={id:data.id??'item'+(++counter),...structuredClone(data),actor,update:async changes=>{for(const [p,v]of Object.entries(changes))set(item,p,v)}};item.uuid=actor.uuid+'.Item.'+item.id;actor.items.set(item.id,item);return item;}
 add({sourceId:S.feature,type:'feat',system:{rules:[]}});add({sourceId:S.flying,type:'feat',system:{rules:[]}});
 const riposte=add({sourceId:S.riposte,type:'action',system:{rules:[{key:'RollOption',option:'opportune-riposte',value:true}]}});
 const charmed=add({sourceId:S.charmed,type:'feat',system:{rules:[{key:'RollOption',option:'charmed-life',value:true}]}});
 const weapon=add({type:'weapon',system:{traits:{value:['agile','thrown-20']}},isMelee:false});
 const data={type:'effect',name:'Effect: Panache',_stats:{compendiumSource:S.panache},system:{slug:'effect-panache',rules:[],duration:{expiry:null,sustained:false,unit:'encounter',value:-1},start:{value:0,initiative:null}}};
 const fromUuid=async uuid=>uuid===S.panache?{toObject:()=>structuredClone(data)}:actor.items.get(uuid.split('.').at(-1));
 const provider=api.createBob2SwashbucklerProvider?.({game,fromUuid,choose:async()=>{prompts++;return choice},randomId:()=> 'nonce'+(++counter)})??{register:()=>{},interceptCheck:(native,...args)=>native(...args),processCheck:async()=>{}};
 provider.register({Hooks:{on:(name,fn)=>{hooks.set(name,fn);return name},off:name=>hooks.delete(name)}});
 function panache(duration={unit:'encounter',value:-1,expiry:null},extra={}){return add({type:'effect',sourceId:S.panache,system:{duration,...extra},flags:{}})}
 function card(options=thrown,outcome='success',type='attack-roll'){
  const message={id:'message'+(++counter),actor,author:player,speaker:{actor:actor.id},item:weapon,isCheckRoll:true,rolls:[{_evaluated:true,total:24}],flags:{pf2e:{origin:{uuid:weapon.uuid},context:{type,domains:type==='attack-roll'?['strike-attack-roll']:['saving-throw'],options:[...options],outcome}}},update:async changes=>{for(const[p,v]of Object.entries(changes))set(message,p,v)},updateSource:changes=>{for(const[p,v]of Object.entries(changes))set(message,p,v)}};
  hooks.get('preCreateChatMessage')?.(message,{}, {},player.id);game.messages.set(message.id,message);return message;
 }
 async function settle(message){game.user=gm;await provider.processCheck(message);game.user=player;}
 async function nativeCheck(options,type='attack-roll',outcome='success',{cancel=false}={}){
  let calls=0,callbacks=0,received,message;const context={actor,origin:{actor,item:weapon,self:true},type,domains:type==='attack-roll'?['strike-attack-roll']:['saving-throw'],options:new Set(options)};
  const result=await provider.interceptCheck(async(_check,ctx,event,callback)=>{calls++;received=ctx;if(cancel)return null;message=card(ctx.options,outcome,type);await callback?.(message.rolls[0],outcome,message,event);return message.rolls[0]}, {modifiers:[]},context,null,()=>{callbacks++});
  if(message)await settle(message);return {calls,callbacks,received,message,result};
 }
 return {provider,game,actor,weapon,riposte,charmed,hooks,add,panache,card,settle,nativeCheck,deleted,cleared,get prompts(){return prompts},effects:()=>[...actor.items.values()].filter(i=>i.type==='effect')};
}
function handlers(game,options,type){return repair(currentRules,game).rules.wmvqDMawtx5dcYHd.handlerRules.filter(r=>r.triggerType===type&&Predicate.test(r.predicate,options));}
for(const outcome of ['criticalSuccess','success','failure','criticalFailure'])test('Patreon owns the legal Flying Blade finisher consumption: '+outcome,()=>{
 const {game}=fixture(),options=[...thrown,'finisher:confident','outcome:'+outcome];
 assert.equal(handlers(game,options,'attack-roll').length,outcome==='failure'||outcome==='criticalFailure'?1:0);
 assert.equal(handlers(game,options,'damage-roll').length,1);
});
for(const missing of ['item:thrown','item:trait:agile','feat:flying-blade','target:range-increment:1','finisher'])test('unqualified thrown finisher never reaches Patreon deletePanache: missing '+missing,()=>{
 const {game}=fixture(),options=[...thrown.filter(o=>o!==missing),'finisher:confident','outcome:failure'];
 assert.equal(handlers(game,options,'attack-roll').length,0);assert.equal(handlers(game,options,'damage-roll').length,0);
});
test('the supplement never consumes a finisher before native damage or on a replay',async()=>{
 const f=fixture();f.panache();const message=f.card();await f.settle(message);await f.settle(message);assert.equal(f.effects().length,1);assert.equal(message.flags[ID]?.swashbuckler,undefined);
});
test('melee finishers, displayed cards, unknown outcomes and another world remain with their existing owners',async()=>{
 for(const change of [f=>f.game.world.id='old-world',(_f,m)=>m.isCheckRoll=false,(_f,m)=>m.flags.pf2e.context.outcome=null,(_f,m)=>m.flags.pf2e.context.options=['item:melee','finisher','self:effect:panache','item:trait:agile']]){
  const f=fixture();f.panache();const m=f.card();change(f,m);await f.settle(m);assert.equal(f.effects().length,1);
 }
});
test('ordinary Strike with saved Riposte toggle gets no panache without this check choice',async()=>{
 const f=fixture({choice:'decline'}),r=await f.nativeCheck(['opportune-riposte','item:melee','item:trait:bravado']);assert.equal(f.prompts,1);assert.equal(f.effects().length,0);assert.equal(r.calls,1);assert.equal(r.callbacks,1);assert.equal(f.riposte.system.rules[0].value,false);
});
for(const [outcome,unit]of [['success','encounter'],['criticalSuccess','encounter'],['failure','rounds'],['criticalFailure',null]])test('explicit Riposte Strike resolves native result: '+outcome,async()=>{
 const f=fixture();f.weapon.isMelee=true;const r=await f.nativeCheck(['opportune-riposte','item:melee','item:trait:bravado'],'attack-roll',outcome);assert.equal(r.calls,1);assert.equal(r.callbacks,1);assert.equal(f.effects()[0]?.system.duration.unit??null,unit);assert.equal(f.riposte.system.rules[0].value,false);
 if(unit){f.effects().forEach(i=>f.actor.items.delete(i.id));await f.settle(r.message);assert.equal(f.effects().length,0)}
});
test('Riposte does not treat a ranged attack or skill check as its melee Strike branch',async()=>{
 for(const [type,options]of [['attack-roll',['opportune-riposte','item:ranged']],['skill-check',['opportune-riposte','action:disarm','item:trait:bravado']]]){const f=fixture();await f.nativeCheck(options,type);assert.equal(f.effects().length,0);assert.equal(f.prompts,0)}
});
for(const outcome of ['success','criticalSuccess','failure','criticalFailure'])test('chosen Charmed Life preserves native save and clears the single-use toggle: '+outcome,async()=>{
 const f=fixture(),r=await f.nativeCheck(['charmed-life'],'saving-throw',outcome);assert.equal(r.calls,1);assert.equal(r.callbacks,1);assert.ok(r.received.options.has('charmed-life'));assert.equal(f.charmed.system.rules[0].value,false);assert.equal(f.effects().length,outcome.includes('Success')||outcome==='success'?1:0);if(f.effects().length)assert.deepEqual(f.effects()[0].system.duration,{unit:'rounds',value:1,expiry:'turn-end',sustained:false});
});
test('declining Charmed Life removes its bonus option; cancelling native check still clears toggle without panache',async()=>{
 const declined=fixture({choice:'decline'}),r=await declined.nativeCheck(['charmed-life'],'saving-throw');assert.equal(r.received.options.has('charmed-life'),false);assert.equal(declined.effects().length,0);
 const cancelled=fixture(),c=await cancelled.nativeCheck(['charmed-life'],'saving-throw','success',{cancel:true});assert.equal(c.result,null);assert.equal(cancelled.effects().length,0);assert.equal(cancelled.charmed.system.rules[0].value,false);
});
test('temporary results retain encounter panache, while Riposte success upgrades temporary panache',async()=>{
 const f=fixture(),existing=f.panache();await f.nativeCheck(['charmed-life'],'saving-throw');assert.equal(f.effects().length,1);assert.equal(f.effects()[0],existing);assert.equal(existing.system.duration.unit,'encounter');
 const g=fixture();g.panache({unit:'rounds',value:1,expiry:'turn-end'});await g.nativeCheck(['opportune-riposte','item:melee'],'attack-roll');assert.equal(g.effects().length,1);assert.equal(g.effects()[0].system.duration.unit,'encounter');
});
test('temporary GrantItem parent is removed before replacing its granted panache',async()=>{
 const f=fixture(),parent=f.add({type:'effect',sourceId:S.temporary,system:{duration:{unit:'rounds',value:1}}});f.panache(undefined);const child=f.effects().find(i=>i!==parent);child.flags={pf2e:{grantedBy:{id:parent.id}}};await f.nativeCheck(['opportune-riposte','item:melee']);assert.equal(f.actor.items.has(parent.id),false);assert.equal(f.effects().length,1);assert.equal(f.effects()[0].system.duration.unit,'encounter');
});
test('skill bravado and non-GM clients never acquire duplicate panache through the supplemental provider',async()=>{
 const f=fixture();const skill=f.card(['item:trait:bravado'],'success','skill-check');await f.settle(skill);assert.equal(f.effects().length,0);const m=f.card();f.panache();await f.provider.processCheck(m);assert.equal(f.effects().length,1);
});

// Captured bob-2 rulesV3 groups; unrelated settings are excluded.
const currentRules={
 "wmvqDMawtx5dcYHd": {
  "uuid": "wmvqDMawtx5dcYHd",
  "group": "feat",
  "name": "Panache",
  "labels": [
   "swashbuckler"
  ],
  "source": [
   "Compendium.pf2e.classfeatures.Item.LzYi0OuOoypNb6jd"
  ],
  "isActive": true,
  "baseRules": [
   {
    "type": "base",
    "triggerType": "skill-check",
    "extend": null,
    "range": null,
    "value": "Compendium.pf2e.feat-effects.Item.uBJsxCzNhje8m8jj",
    "target": "SelfEffect",
    "predicate": [
     "item:trait:bravado",
     {
      "not": "panache"
     },
     {
      "or": [
       "outcome:success",
       "outcome:criticalSuccess"
      ]
     }
    ]
   }
  ],
  "complexRules": [
   {
    "type": "complex",
    "triggerType": "skill-check",
    "extend": null,
    "range": null,
    "values": [
     {
      "conditions": [],
      "effects": [
       "Compendium.pf2e.feat-effects.Item.uBJsxCzNhje8m8jj"
      ],
      "duration": {
       "expiry": "turn-end",
       "sustained": false,
       "unit": "rounds",
       "value": 1
      }
     }
    ],
    "target": "SelfEffect",
    "predicate": [
     "item:trait:bravado",
     {
      "not": "panache"
     },
     "outcome:failure"
    ]
   }
  ],
  "handlerRules": [
   {
    "type": "handler",
    "triggerType": "damage-roll",
    "value": "deletePanache",
    "target": "none",
    "predicate": [
     "item:melee",
     "finisher",
     "self:effect:panache",
     {
      "or": [
       "finisher:confident",
       "agile",
       "finesse"
      ]
     }
    ]
   },
   {
    "type": "handler",
    "triggerType": "attack-roll",
    "value": "deletePanache",
    "target": "none",
    "predicate": [
     "item:melee",
     "finisher",
     "self:effect:panache",
     {
      "or": [
       "finisher:confident",
       "agile",
       "finesse"
      ]
     },
     {
      "or": [
       "outcome:failure",
       "outcome:criticalFailure"
      ]
     }
    ]
   }
  ]
 },
 "WJjRqP7GYCWlSnXw": {
  "uuid": "WJjRqP7GYCWlSnXw",
  "group": "feat",
  "name": "Extravagant Parry",
  "labels": [],
  "source": [
   "Compendium.pf2e.feats-srd.Item.WcY7H7mRiK2VDquV"
  ],
  "isActive": true,
  "baseRules": [
   {
    "triggerType": "postInfo",
    "range": null,
    "extend": null,
    "predicate": [
     "origin:item:extravagant-parry"
    ],
    "target": "SelfEffect",
    "type": "base",
    "value": "Compendium.pf2e.feat-effects.Item.1nCwQErK6hpkNvfw",
    "values": []
   }
  ],
  "complexRules": [],
  "handlerRules": []
 }
};

const repair=(rules,game)=>api.buildBob2SwashbucklerPatreonRepairs?.(rules,{game})??{rules:structuredClone(rules),changes:[]};
test('current Patreon repair removes only the exact obsolete Parry mapping and preserves all Panache owners',()=>{
 const {game}=fixture(),before=structuredClone(currentRules),result=repair(before,game);
 assert.equal(result.rules.WJjRqP7GYCWlSnXw.baseRules.length,0);assert.deepEqual(result.rules.wmvqDMawtx5dcYHd.baseRules,before.wmvqDMawtx5dcYHd.baseRules);assert.deepEqual(result.rules.wmvqDMawtx5dcYHd.complexRules,before.wmvqDMawtx5dcYHd.complexRules);assert.equal(result.rules.WJjRqP7GYCWlSnXw.isActive,true);assert.deepEqual(before,currentRules);assert.equal(result.changes.length,3);assert.equal(repair(result.rules,game).changes.length,0);
});
test('unknown/custom Parry rules and other worlds are not rewritten',()=>{
 for(const change of [(r,g)=>g.world.id='old-world',(r,g)=>g.system.id='sf2e',(r,g)=>g.modules.get('patreon-v3').active=false,r=>r.WJjRqP7GYCWlSnXw.source=['custom'],r=>r.WJjRqP7GYCWlSnXw.baseRules[0].predicate.push('custom'),r=>r.WJjRqP7GYCWlSnXw.baseRules[0].value='custom']){const {game}=fixture(),before=structuredClone(currentRules);before.wmvqDMawtx5dcYHd.handlerRules=[];change(before,game);assert.deepEqual(repair(before,game),{rules:before,changes:[]})}
});
test('existing configuration maintenance backs up the precise Parry change once and preserves disabled state',async()=>{
 const {game}=fixture();game.user=game.users.activeGM;const before=structuredClone(currentRules);before.WJjRqP7GYCWlSnXw.isActive=false;before.WJjRqP7GYCWlSnXw.baseRules.push({type:'base',triggerType:'use',value:'custom'});
 const data=new Map([['patreon-v3.rulesV3',before],[ID+'.configurationBackups',[]]]),writes=[];game.settings={get:(m,k)=>data.get(m+'.'+k),set:async(m,k,v)=>{writes.push(m+'.'+k);data.set(m+'.'+k,structuredClone(v))}};
 const maintain=createConfigurationMaintenance({game,repairs:[rules=>repair(rules,game)]});await maintain();await maintain();
 assert.deepEqual(writes,[ID+'.configurationBackups','patreon-v3.rulesV3']);const after=data.get('patreon-v3.rulesV3');assert.equal(after.WJjRqP7GYCWlSnXw.isActive,false);assert.deepEqual(after.WJjRqP7GYCWlSnXw.baseRules,[{type:'base',triggerType:'use',value:'custom'}]);assert.deepEqual(data.get(ID+'.configurationBackups')[0].changes[0].before,before.WJjRqP7GYCWlSnXw.baseRules);
});

test('temporary panache gained before the actor turn expires at that upcoming turn end',async()=>{
 const f=fixture();f.game.combat.combatant={actor:{id:'enemy'},initiative:25};await f.nativeCheck(['charmed-life'],'saving-throw');assert.equal(f.effects()[0].system.duration.value,0);assert.equal(f.effects()[0].system.start.initiative,20);
});
test('encounter end removes only this provider temporary panache and preserves unrelated effects',async()=>{
 const f=fixture();await f.nativeCheck(['charmed-life'],'saving-throw');const other=f.add({type:'effect',sourceId:'unrelated',system:{duration:{unit:'rounds',value:1}}});
 f.game.user=f.game.users.activeGM;await f.hooks.get('deleteCombat')?.({id:'combat',started:true,combatants:[{actor:f.actor}]});assert.deepEqual(f.effects(),[other]);
});
test('contextual native actor clones clear the actual owned toggle',async()=>{
 const f=fixture();const clone={...f.actor,toggleRollOption:async()=>assert.fail('must not update contextual clone')};
 await f.provider.interceptCheck(async()=>null,{modifiers:[]},{actor:clone,type:'saving-throw',options:new Set(['charmed-life'])});assert.equal(f.charmed.system.rules[0].value,false);
});

test('Charmed Life keeps native context mutations visible to the original save caller',async()=>{
 const f=fixture(),check={modifiers:[]},context={actor:f.actor,type:'saving-throw',options:new Set(['charmed-life'])};
 const result=await f.provider.interceptCheck(async(actual,ctx)=>{assert.equal(actual,check);ctx.outcome='criticalSuccess';return 31},check,context);
 assert.equal(result,31);assert.equal(context.outcome,'criticalSuccess');
});

test('Patreon keeps melee behavior and customized Panache consumers unchanged',()=>{
 const {game}=fixture(),before=structuredClone(currentRules);before.wmvqDMawtx5dcYHd.handlerRules[0].predicate.push('custom');
 assert.deepEqual(repair(before,game).rules.wmvqDMawtx5dcYHd.handlerRules[0],before.wmvqDMawtx5dcYHd.handlerRules[0]);
 for(const outcome of ['success','failure']){const options=['item:melee','finisher','self:effect:panache','agile','outcome:'+outcome];assert.equal(handlers(game,options,'attack-roll').length,outcome==='failure'?1:0);assert.equal(handlers(game,options,'damage-roll').length,1)}
});
