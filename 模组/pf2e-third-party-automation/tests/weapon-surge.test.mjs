import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createNextStrikeEffectFrame} from '../scripts/next-strike-effects.mjs';
import {createAttackSequence} from '../scripts/activity-attack-sequence.mjs';
import {createKnowledgeAutomation} from '../scripts/knowledge-automation.mjs';
import {beforeNativeRoll} from '../scripts/native-owner-operations.mjs';
let api={};try{api=await import('../scripts/weapon-surge.mjs')}catch(error){if(error.code!=='ERR_MODULE_NOT_FOUND')throw error}
const ID='pf2e-third-party-automation',SOURCE='Compendium.pf2e.spell-effects.Item.qlz0sJIvqc0FdUdr';
const clone=structuredClone;
function surge(id='surge',rank=1,weapon='sword'){
 return {_id:id,name:'Spell Effect: Weapon Surge',type:'effect',_stats:{compendiumSource:SOURCE},flags:{system:{rulesSelections:{spellEffectWeaponSurge:weapon}}},system:{slug:'spell-effect-weapon-surge',level:{value:rank},start:{value:100},duration:{value:1,unit:'rounds',expiry:'turn-start'},rules:[{key:'ChoiceSet',flag:'spellEffectWeaponSurge',selection:weapon},{key:'FlatModifier',selector:'{item|flags.system.rulesSelections.spellEffectWeaponSurge}-attack',type:'status',value:1},{key:'DamageDice',selector:'{item|flags.system.rulesSelections.spellEffectWeaponSurge}-damage',damageType:'spirit',diceNumber:'ternary(gte(@item.level,9),3,ternary(gte(@item.level,5),2,1))',dieSize:'d6'},{key:'AdjustStrike',definition:['item:id:{item|flags.system.rulesSelections.spellEffectWeaponSurge}'],mode:'add',property:'traits',value:'sanctified'}]}};
}
function fixture(){
 const writes=[],clones=[],actor={uuid:'Actor.hero',items:new Map(),_source:{items:[]},system:{actions:[]},async deleteEmbeddedDocuments(type,ids){writes.push(ids);for(const id of ids)this.items.delete(id);this._source.items=this._source.items.filter(item=>!ids.includes(item._id));}};
 const add=data=>{const item={...clone(data),id:data._id,uuid:`${actor.uuid}.Item.${data._id}`,actor,toObject(){const data=clone(this._source);data.system=clone(this.system);data.flags=clone(this.flags);return data},_source:clone(data)};actor.items.set(item.id,item);actor._source.items.push(clone(data));return item};
 const weapon={_id:'sword',type:'weapon',system:{traits:{value:[]}}};add(weapon);const effect=add(surge());
 const makeStrike=(owner,id='sword')=>({type:'strike',item:{id,uuid:`${actor.uuid}.Item.${id}`,actor:owner,type:'weapon',system:{traits:{value:[]}}},variants:[{roll:async()=>null}],damage:async params=>({items:owner._source.items,options:params.options}),critical:async params=>({items:owner._source.items,options:params.options})});
 const strike=makeStrike(actor),target={uuid:'Scene.scene.Token.target'},message={flags:{pf2e:{origin:{actor:actor.uuid,uuid:strike.item.uuid},context:{type:'attack-roll',outcome:'success',options:[],target:{token:target.uuid}}}}};actor.system.actions.push(strike);
 actor.clone=function(changes){clones.push(clone(changes));const copy={...actor,_source:clone(changes),items:new Map(changes.items.map(data=>[data._id,{...data,id:data._id}]))};copy.system={actions:[makeStrike(copy)]};return copy};
 const begin=()=>createNextStrikeEffectFrame({actor,strike,target});return {actor,effect,strike,target,message,writes,clones,add,begin};
}
for(const outcome of ['success','criticalSuccess','failure','criticalFailure'])test(`the next bound weapon ${outcome} consumes exactly its original Weapon Surge`,async()=>{
 const f=fixture(),frame=f.begin();f.message.flags.pf2e.context.outcome=outcome;assert.equal(frame.capture(f.message),true);await frame.consume();assert.deepEqual(f.writes,[['surge']]);assert.equal(f.actor.items.has('sword'),true);
});
test('a different weapon and an unresolved attack leave Weapon Surge for its bound weapon',async()=>{
 const f=fixture();f.strike.item.id='bow';f.strike.item.uuid='Actor.hero.Item.bow';f.message.flags.pf2e.origin.uuid=f.strike.item.uuid;const other=f.begin();other.capture(f.message);await other.consume();assert.deepEqual(f.writes,[]);
 const g=fixture(),unresolved=g.begin();g.message.flags.pf2e.context.outcome=null;assert.equal(unresolved.capture(g.message),false);await unresolved.consume();assert.deepEqual(g.writes,[]);
});
test('delayed damage uses only the old Surge and does not consume or borrow the new one',async()=>{
 const f=fixture(),frame=f.begin();frame.capture(f.message);await frame.consume();f.add(surge('new-surge',9));
 assert.equal(typeof frame.damage,'function');const damage=frame.damage(f.strike),items=damage.item.actor._source.items,effects=items.filter(item=>item.type==='effect'&&item.system.rules.some(rule=>rule.key==='DamageDice'));
 assert.equal(effects.length,1);assert.equal(effects[0].system.level.value,1);assert.equal(f.actor.items.get('new-surge').system.level.value,9);assert.deepEqual(f.writes,[['surge']]);
});
test('an attack without Surge cannot borrow an effect granted before its delayed damage',()=>{
 const f=fixture();f.actor.items.delete('surge');f.actor._source.items=f.actor._source.items.filter(item=>item._id!=='surge');const frame=f.begin();frame.capture(f.message);f.add(surge('later',9));
 const damage=frame.damage(f.strike);assert.equal(damage.item.actor._source.items.filter(item=>item.type==='effect'&&item.system.rules.some(rule=>rule.key==='DamageDice')).length,0);assert.ok(f.actor.items.has('later'));
});
test('same-ID refreshed Surge survives the old frame',async()=>{
 const f=fixture(),frame=f.begin();frame.capture(f.message);f.effect.system.start.value=101;await frame.consume();assert.deepEqual(f.writes,[]);assert.ok(f.actor.items.has('surge'));
});
test('native reroll pf2e context carries the original snapshot rather than the next Surge',async()=>{
 const f=fixture(),first=f.begin();first.capture(f.message);const saved=first.snapshot();await first.consume();f.add(surge('new',9));
 const reroll={flags:{pf2e:{...clone(f.message.flags.pf2e),context:{...clone(f.message.flags.pf2e.context),isReroll:true,weaponSurgeSnapshot:saved}}}},frame=f.begin();assert.equal(frame.capture(reroll),true);await frame.consume();assert.ok(f.actor.items.has('new'));assert.equal(frame.damage(f.strike).item.actor._source.items.find(item=>item.system?.rules?.some(rule=>rule.key==='DamageDice')).system.level.value,1);
});
test('attack sequence combines the original Surge snapshot with its own Forceful modifier once',async()=>{
 const f=fixture();f.strike.item.system.traits.value=['forceful'];const sequence=createAttackSequence({actor:f.actor}),first=sequence.begin(f.strike,f.target);first.capture(f.message);sequence.record(first,'success');await first.consume();
 f.add(surge('second',5));const second=sequence.begin(f.strike,f.target);second.capture(f.message);sequence.record(second,'success');await second.consume();const prepared=second.damage(f.strike).strike.item.actor._source.items;
 assert.equal(prepared.filter(item=>item.system?.rules?.some(rule=>rule.key==='DamageDice')).length,1);assert.equal(prepared.flatMap(item=>item.system?.rules??[]).filter(rule=>rule.slug==='activity-forceful-second').length,1);assert.equal(f.actor.items.has('second'),false);
});
test('owner RPC snapshot expansion excludes the current Surge and preserves other transient effects',async()=>{
 assert.equal(typeof api.prepareWeaponSurgeDamageSnapshotItems,'function');const f=fixture(),frame=f.begin();frame.capture(f.message);await frame.consume();f.add(surge('later',9));const extra=frame.transientItems();extra.push({_id:'activity',type:'effect',system:{rules:[{key:'FlatModifier',value:2}]}});
 const items=api.prepareWeaponSurgeDamageSnapshotItems(f.actor,extra);assert.ok(items.some(item=>item._id==='activity'));assert.equal(items.filter(item=>item.system?.rules?.some(rule=>rule.key==='DamageDice')).length,1);assert.equal(items.find(item=>item.system?.rules?.some(rule=>rule.key==='DamageDice')).system.level.value,1);
});
test('native Strike callbacks are awaited once and persist the snapshot before consuming the exact effect',async()=>{
 assert.equal(typeof api.createWeaponSurgeAutomation,'function');const f=fixture();let nativeCalls=0,callbacks=0;const game={user:{targets:new Set([f.target])},messages:new Map()};const provider=api.createWeaponSurgeAutomation({game});
 f.strike.variants[0].roll=async params=>{nativeCalls++;return provider.interceptCheck(async(_check,context,_event,callback)=>{f.message.flags.pf2e.context={...f.message.flags.pf2e.context,...context,options:[...context.options]};await callback({},'success',f.message);return 'native-roll'}, {},{type:'attack-roll',origin:{actor:f.actor,item:f.strike.item},options:params.options},null,params.callback)};
 provider.wrapStrike(f.strike,f.actor);assert.equal(await f.strike.variants[0].roll({callback:async()=>{callbacks++;assert.ok(f.message.flags.pf2e.context.weaponSurgeSnapshot)}}),'native-roll');assert.equal(nativeCalls,1);assert.equal(callbacks,1);assert.deepEqual(f.writes,[['surge']]);
});
test('a completed native attack still consumes its old Surge if the caller callback fails',async()=>{
 const f=fixture(),provider=api.createWeaponSurgeAutomation({game:{user:{targets:new Set([f.target])}}});
 f.strike.variants[0].roll=async params=>provider.interceptCheck(async(_check,context,_event,callback)=>{f.message.flags.pf2e.context={...f.message.flags.pf2e.context,...context,options:[...context.options]};await callback({},'success',f.message)}, {},{type:'attack-roll',origin:{actor:f.actor,item:f.strike.item},options:params.options},null,params.callback);
 provider.wrapStrike(f.strike,f.actor);await assert.rejects(f.strike.variants[0].roll({callback:async()=>{throw Error('downstream failure')}}),/downstream failure/);assert.deepEqual(f.writes,[['surge']]);
});
test('repeated native callback delivery settles and calls the activity callback once',async()=>{
 const f=fixture(),provider=api.createWeaponSurgeAutomation({game:{user:{targets:new Set()}}});let calls=0;
 f.strike.variants[0].roll=async params=>{await Promise.all([params.callback({},'success',f.message),params.callback({},'success',f.message)]);return 'native'};provider.wrapStrike(f.strike,f.actor);
 assert.equal(await f.strike.variants[0].roll({callback:async()=>{calls++}}),'native');assert.equal(calls,1);assert.deepEqual(f.writes,[['surge']]);
});
for(const method of ['damage','critical'])test(`ordinary native ${method} uses its exact old card context and leaves a later Surge untouched`,async()=>{
 const f=fixture(),first=f.begin();first.capture(f.message);const saved=first.snapshot();await first.consume();f.add(surge('new',9));f.add(surge('bow-surge',5,'bow'));
 const provider=api.createWeaponSurgeAutomation({game:{user:{targets:new Set()}}});provider.wrapStrike(f.strike,f.actor);
 const result=await f.strike[method]({checkContext:{...f.message.flags.pf2e.context,weaponSurgeSnapshot:saved},options:['self:effect:spell-effect-weapon-surge','other']});
 const old=result.items.filter(item=>item.system?.rules?.some(rule=>rule.key==='DamageDice')&&item.flags?.system?.rulesSelections?.spellEffectWeaponSurge==='sword');assert.equal(old.length,1);assert.equal(old[0].system.level.value,1);assert.ok(result.items.some(item=>item._id==='bow-surge'));assert.equal(result.options.has('self:effect:spell-effect-weapon-surge'),false);assert.ok(f.actor.items.has('new'));assert.deepEqual(f.writes,[['surge']]);
});
test('a cancelled native roll does not spend Surge or leak its open check context into another call',async()=>{
 const f=fixture(),provider=api.createWeaponSurgeAutomation({game:{user:{targets:new Set()}}});let marker;f.strike.variants[0].roll=async params=>{marker=[...params.options].find(value=>value.includes(':weapon-surge:'));return null};provider.wrapStrike(f.strike,f.actor);
 assert.equal(await f.strike.variants[0].roll(),null);assert.deepEqual(f.writes,[]);const context=await provider.interceptCheck(async(_check,context)=>context,{}, {type:'attack-roll',origin:{actor:f.actor,item:f.strike.item},options:[marker]});assert.equal(context.weaponSurgeSnapshot,undefined);
});
test('owner RPC expansion is idempotent and rejects a foreign actor or weapon record',()=>{
 const f=fixture(),frame=f.begin();frame.capture(f.message);const transients=frame.transientItems(),items=api.prepareWeaponSurgeDamageSnapshotItems(f.actor,transients),temporary=f.actor.clone({items}),second=api.prepareWeaponSurgeDamageSnapshotItems(temporary,items.filter(item=>!f.actor._source.items.some(live=>live._id===item._id)));
 assert.equal(second.filter(item=>item.system?.rules?.some(rule=>rule.key==='DamageDice')).length,1);assert.equal(second.filter(item=>item.flags?.[ID]?.weaponSurgeSnapshot).length,1);
 for(const field of ['actorUuid','weaponUuid']){const forged=clone(transients);forged[0].flags[ID].weaponSurgeSnapshot[field]='Actor.other.Item.sword';assert.throws(()=>api.prepareWeaponSurgeDamageSnapshotItems(f.actor,forged),/快照无效/)}assert.deepEqual(f.writes,[]);
});
test('a malformed saved card cannot spend or borrow the current weapon Surge',async()=>{
 const f=fixture(),frame=f.begin(),saved=frame.snapshot();saved.actorUuid='Actor.other';f.message.flags.pf2e.context.weaponSurgeSnapshot=saved;
 assert.equal(frame.capture(f.message),false);await frame.consume();assert.deepEqual(f.writes,[]);
 const provider=api.createWeaponSurgeAutomation({game:{user:{targets:new Set()}}});provider.wrapStrike(f.strike,f.actor);await assert.rejects(f.strike.damage({checkContext:f.message.flags.pf2e.context}),/快照无效/);
});
test('concurrent native attacks are distinct calls and only the first captures the same Surge',async()=>{
 const f=fixture(),provider=api.createWeaponSurgeAutomation({game:{user:{targets:new Set()}}}),records=[];let calls=0,release;
 const firstDialog=new Promise(resolve=>{release=resolve});
 f.strike.variants[0].roll=async params=>{const index=++calls;if(index===1)await firstDialog;return provider.interceptCheck(async(_check,context,_event,callback)=>{records.push(context.weaponSurgeSnapshot);const card=clone(f.message);card.flags.pf2e.context={...card.flags.pf2e.context,...context,options:[...context.options]};await callback({},'success',card);return index}, {},{type:'attack-roll',origin:{actor:f.actor,item:f.strike.item},options:params.options},null,params.callback)};
 provider.wrapStrike(f.strike,f.actor);const first=f.strike.variants[0].roll(),second=f.strike.variants[0].roll();for(let i=0;i<5;i++)await Promise.resolve();assert.equal(calls,1);release();assert.deepEqual(await Promise.all([first,second]),[1,2]);assert.deepEqual(records.map(record=>record.effects.length),[1,0]);assert.deepEqual(f.writes,[['surge']]);
});
test('the confirmed Strike releases its gate before an awaited callback starts another Strike',async()=>{
 const f=fixture(),provider=api.createWeaponSurgeAutomation({game:{user:{targets:new Set()}}}),records=[];
 f.strike.variants[0].roll=async params=>provider.interceptCheck(async(_check,context,_event,callback)=>{records.push(context.weaponSurgeSnapshot);const card=clone(f.message);card.flags.pf2e.context={...card.flags.pf2e.context,...context,options:[...context.options]};await callback({},'success',card);return 'native'}, {},{type:'attack-roll',origin:{actor:f.actor,item:f.strike.item},options:params.options},null,params.callback);
 provider.wrapStrike(f.strike,f.actor);await f.strike.variants[0].roll({callback:async()=>{await f.strike.variants[0].roll()}});assert.deepEqual(records.map(record=>record.effects.length),[1,0]);assert.deepEqual(f.writes,[['surge']]);
});
test('a queued Strike rebinds the current alternate usage and the exact native MAP variant',async()=>{
 const f=fixture(),provider=api.createWeaponSurgeAutomation({game:{user:{targets:new Set()}}}),calls=[];f.strike.item.altUsageType='thrown';f.strike.variants=[{roll:async()=>calls.push('old-0')},{roll:async()=>calls.push('old-1')}];provider.wrapStrike(f.strike,f.actor);
 const result=f.strike.variants[1].roll(),alternate={...f.strike,item:{...f.strike.item},variants:[{roll:async()=>calls.push('new-0')},{roll:async()=>calls.push('new-1')}]};
 f.actor.system.actions=[{...f.strike,item:{...f.strike.item,altUsageType:null},altUsages:[alternate]}];await result;assert.deepEqual(calls,['new-1']);assert.deepEqual(f.writes,[]);
});
for(const startsWithSurge of [true,false])test(`an actual evaluated native Strike without AC stores its original ${startsWithSurge?'Surge':'empty'} snapshot`,async()=>{
 const f=fixture(),provider=api.createWeaponSurgeAutomation({game:{user:{targets:new Set()}}});if(!startsWithSurge){f.actor.items.delete('surge');f.actor._source.items=f.actor._source.items.filter(item=>item._id!=='surge')}
 f.message.flags.pf2e.context.outcome=null;
 f.strike.variants[0].roll=async params=>provider.interceptCheck(async(_check,context,_event,callback)=>{f.message.flags.pf2e.context={...f.message.flags.pf2e.context,...context,options:[...context.options]};const roll={_evaluated:true,total:24};await callback(roll,null,f.message);return roll}, {},{type:'attack-roll',origin:{actor:f.actor,item:f.strike.item},options:params.options},null,params.callback);
 provider.wrapStrike(f.strike,f.actor);const rolled=await f.strike.variants[0].roll();assert.equal(rolled.total,24);assert.deepEqual(f.writes,startsWithSurge?[['surge']]:[]);assert.equal(f.message.flags.pf2e.context.weaponSurgeSnapshot.effects.length,startsWithSurge?1:0);
 f.add(surge('new',9));const result=await f.strike.damage({checkContext:f.message.flags.pf2e.context}),dice=result.items.filter(item=>item.system?.rules?.some(rule=>rule.key==='DamageDice'));
 assert.equal(dice.length,startsWithSurge?1:0);if(startsWithSurge)assert.equal(dice[0].system.level.value,1);assert.ok(f.actor.items.has('new'));
});
for(const roll of [{_evaluated:false,total:24},{_evaluated:true,total:NaN},{total:24},'attack-roll'])test(`an unresolved callback without evaluated native evidence cannot spend Surge (${JSON.stringify(roll)})`,async()=>{
 const f=fixture(),provider=api.createWeaponSurgeAutomation({game:{user:{targets:new Set()}}});f.message.flags.pf2e.context.outcome=null;f.strike.variants[0].roll=async params=>{await params.callback(roll,null,f.message);return roll};provider.wrapStrike(f.strike,f.actor);await f.strike.variants[0].roll();assert.deepEqual(f.writes,[]);
});
test('no-dialog payment reset cannot rewrap a Knowledge outer handler into a self-waiting Surge gate',async()=>{
 const f=fixture(),game={user:{targets:new Set()},actors:new Map(),scenes:new Map(),modules:new Map()},provider=api.createWeaponSurgeAutomation({game}),knowledge=createKnowledgeAutomation({game});let payments=0,nativeCalls=0,timer;
 const prepare=()=>{
  const current={...f.strike,item:{...f.strike.item},variants:[{roll:async params=>{nativeCalls++;return provider.interceptCheck(async(_check,context,_event,callback)=>{f.message.flags.pf2e.context={...f.message.flags.pf2e.context,...context,options:[...context.options]};const roll={_evaluated:true,total:24};await callback(roll,'success',f.message);return roll}, {},{type:'attack-roll',origin:{actor:f.actor,item:f.strike.item},options:params.options},null,params.callback)}}]};
  provider.wrapStrike(current,f.actor);knowledge.wrapStrike(current,f.actor);f.actor.system.actions=[current];return current;
 };
 const captured=prepare();
 try{
  const result=await Promise.race([beforeNativeRoll({showDialog:false,commit:async()=>{payments++;prepare()},native:commit=>captured.variants[0].roll({callback:commit})}),new Promise((_,reject)=>{timer=setTimeout(()=>reject(Error('Surge gate waited on its own rewrapped Knowledge handler')),100)})]);
  assert.equal(result.total,24);assert.equal(payments,1);assert.equal(nativeCalls,1);assert.deepEqual(f.writes,[['surge']]);
 }finally{clearTimeout(timer)}
});
test('outer damage handlers do not cause the same prepared Strike to receive a second Surge wrapper',()=>{
 const f=fixture(),provider=api.createWeaponSurgeAutomation({game:{user:{targets:new Set()}}});provider.wrapStrike(f.strike,f.actor);const native=f.strike.damage,outer=async params=>native(params);f.strike.damage=outer;
 provider.wrapStrike(f.strike,f.actor);assert.equal(f.strike.damage,outer);
});
const nativePath=process.env.PF2E_NATIVE_BUNDLE??'';
test('real PF2e DamageDice body prepares the frozen native spirit d6 at ranks 1, 5 and 9',{skip:!nativePath},()=>{
 assert.equal(typeof api.prepareWeaponSurgeDamageSnapshotItems,'function');const source=readFileSync(nativePath,'utf8'),classStart=source.indexOf('DamageDiceRuleElement = class'),start=source.indexOf('\tbeforePrepareData() {',classStart),end=source.indexOf('\n\t}\n',start)+4;assert.ok(classStart>=0&&start>classStart&&end>start);
 const body=source.slice(start,end).replace('beforePrepareData()', 'function()'),native=Function('CONFIG','objectHasKey','tupleHasValue','Jt','sluggify','extractDamageAlterations','hc','foundry',`return (${body})`)({PF2E:{damageTypes:{spirit:'Spirit'}}},(object,key)=>Object.hasOwn(object,key),(list,value)=>list.includes(value),['d6'],()=> 'weapon-surge',()=>[],class{constructor(data){Object.assign(this,data)}},{utils:{deepClone:clone}});
 for(const [rank,dice]of [[1,1],[5,2],[9,3]]){
  const f=fixture();f.effect.system.level.value=rank;const frame=f.begin();frame.capture(f.message);const data=frame.damage(f.strike).item.actor._source.items.find(item=>item.system?.rules?.some(rule=>rule.key==='DamageDice')),rule=data.system.rules.find(rule=>rule.key==='DamageDice'),actor={synthetics:{damageDice:{},damageAlterations:{}},getRollOptions:()=>[]};
  const resolve=value=>Array.isArray(value)?value.map(resolve):typeof value==='string'?value.replace('{item|flags.system.rulesSelections.spellEffectWeaponSurge}',data.flags.system.rulesSelections.spellEffectWeaponSurge):value;
  native.call({...rule,selector:[rule.selector],ignored:false,predicate:[],actor,parent:{name:data.name,getRollOptions:()=>[]},getReducedLabel:()=>data.name,resolveInjectedProperties:resolve,resolveValue:formula=>Function('gte','ternary',`return ${formula.replaceAll('@item.level',String(rank))}`)((a,b)=>a>=b,(condition,a,b)=>condition?a:b)});
  const prepared=actor.synthetics.damageDice['sword-damage'][0]({selectors:['sword-damage']});assert.equal(prepared.diceNumber,dice);assert.equal(prepared.damageType,'spirit');assert.equal(prepared.dieSize,'d6');assert.equal(data.system.rules.find(rule=>rule.key==='FlatModifier').value,1);assert.equal(data.system.rules.find(rule=>rule.key==='AdjustStrike').value,'sanctified');
 }
});
