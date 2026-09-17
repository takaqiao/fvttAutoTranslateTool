import test from 'node:test';
import assert from 'node:assert/strict';
let api={};try{api=await import('../scripts/metapower/rules.mjs')}catch(e){if(e.code!=='ERR_MODULE_NOT_FOUND')throw e}
const S='Compendium.battlezoo-eldamon-pf2e.',P=S+'powers.Item.';
const ids={surge:'veFrnrxYjlqca13w',anvil:'hQOa1yaP9C6wajNn',static:'KWQgx7RMeY3RKW6J',shot:'QIYppaP0zcGvb5Bd',chain:'fzV5Ly3a9nEsfcAJ',retributive:'geZCat82IOuShmmk',voltage:'9bElF2uVf5FCJtb9'};
const actor=()=>({uuid:'Actor.synthetic',level:5,flags:{pf2e:{eldamon:{element:{trait:'electricity',traitTwo:'electricity'}}}},items:[{sourceId:S+'feats.Item.kG0HSsDc6eHYjTU9'}]});
const power=id=>({uuid:'Actor.synthetic.Item.power',type:'feat',_stats:{compendiumSource:P+ids[id]},system:{traits:{value:['electricity','magical']}}});
const snapshot=(id,kind='siphoning',extra={})=>{assert.equal(typeof api.buildChannelSnapshot,'function');return api.buildChannelSnapshot({kind,item:power(id),actor:actor(),...extra})};

test('full source UUID identifies metapowers despite absent traits and renamed labels',()=>{
 assert.equal(typeof api.metapowerKind,'function');
 assert.equal(api.metapowerKind({sourceId:S+'actions.Item.4w72ljp4eLBqeZB2',name:'Synthetic renamed'}),'siphoning');
 assert.equal(api.metapowerKind({_stats:{compendiumSource:S+'feats.Item.3YasBiZw3N96rdUW'}}),'widen');
 assert.equal(api.metapowerKind({flags:{core:{sourceId:S+'feats.Item.3YasBiZw3N96rdUW'}}}),'widen');
 for(const sourceId of ['4w72ljp4eLBqeZB2',S+'feats.Item.4w72ljp4eLBqeZB2',S+'actions.Item.4w72ljp4eLBqeZB2.extra','__proto__'])assert.equal(api.metapowerKind({sourceId,name:'Siphoning Element'}),null);
 assert.equal(api.metapowerKind({name:'Widen Element',system:{slug:'widen-element',traits:{value:['metapower']}}}),null);
});
test('authoritative source takes precedence over conflicting legacy source',()=>{
 assert.equal(typeof api.sourceUuid,'function');
 assert.equal(api.sourceUuid({sourceId:'Actor.other.Item.fake',_stats:{compendiumSource:S+'feats.Item.3YasBiZw3N96rdUW'}}),'Actor.other.Item.fake');
 assert.equal(api.sourceUuid(null),null);
});
test('only the seven reviewed full power sources have profiles',()=>{
 assert.equal(typeof api.powerProfile,'function');
 for(const id of Object.keys(ids))assert.ok(api.powerProfile(power(id)));
 for(const sourceId of [ids.surge,P+ids.surge+'x','constructor','__proto__',S+'feats.Item.'+ids.surge])assert.equal(api.powerProfile({sourceId}),null);
 assert.equal(Object.keys(api.POWER_PROFILES).length,7);
});
test('widen geometry handles thresholds and rejects duration and nonarea',()=>{
 assert.equal(typeof api.widenDistance,'function');
 for(const [type,distance,expected] of [['burst',5,5],['burst',10,15],['burst',15,20],['cone',15,20],['cone',20,30],['line',15,20],['line',20,30],['emanation',20,20],['range',40,40]])assert.equal(api.widenDistance({type,distance,hasDuration:false}),expected);
 assert.equal(api.widenDistance({type:'line',distance:20,hasDuration:true}),20);
 assert.throws(()=>api.widenDistance({type:'line',distance:-5}),/distance/i);
});
test('surge uses selected level-legal base before Widen exactly once',()=>{
 assert.deepEqual(snapshot('surge','widen',{selection:{baseDistance:30}}).area,{type:'line',baseDistance:30,distance:40,hasDuration:false});
 assert.equal(snapshot('surge','widen',{selection:{discharge:true,baseDistance:60}}).area.distance,70);
 assert.equal(snapshot('surge','widen',{level:1,selection:{discharge:true}}).area.distance,50);
 assert.equal(snapshot('surge','widen',{level:17,selection:{baseDistance:60}}).area.distance,70);
 assert.throws(()=>snapshot('surge','widen',{level:5,selection:{baseDistance:40}}),/legal|distance/i);
 assert.throws(()=>snapshot('surge','widen',{level:17,selection:{baseDistance:70}}),/legal|distance/i);
});
test('anvil optional discharge cone is selected independently of secondary condition duration',()=>{
 assert.equal(snapshot('anvil','widen').area.distance,40);
 assert.equal(snapshot('anvil','widen',{selection:{discharge:true,baseDistance:50}}).area.distance,60);
 assert.equal(snapshot('anvil','widen',{selection:{discharge:true,baseDistance:60}}).area.distance,70);
 assert.throws(()=>snapshot('anvil','widen',{selection:{discharge:true,baseDistance:65}}),/legal|distance/i);
 for(const id of ['static','shot','chain','retributive','voltage'])assert.equal(snapshot(id,'widen').area,null);
});
test('snapshot freezes source traits, associated traits and exact Disruptive ownership without mutating actors/items',()=>{
 const a=actor(),item=power('surge'),before=JSON.stringify({a,item});
 const s=api.buildChannelSnapshot({kind:'siphoning',actor:a,item});
 assert.equal(JSON.stringify({a,item}),before);assert.deepEqual(s.associatedTraits,['electricity']);assert.equal(s.disruptive,true);
 assert.deepEqual(s.traits,['electricity','magical']);assert.ok(Object.isFrozen(s));assert.ok(Object.isFrozen(s.traits));assert.ok(Object.isFrozen(s.area));
 a.flags.pf2e.eldamon.element.trait='fire';item.system.traits.value.push('fire');assert.deepEqual(s.associatedTraits,['electricity']);assert.deepEqual(s.traits,['electricity','magical']);
 a.items=[{name:'Disruptive Siphon',sourceId:'kG0HSsDc6eHYjTU9'}];assert.equal(api.buildChannelSnapshot({kind:'siphoning',actor:a,item}).disruptive,false);
});
test('Disruptive coefficient matches target creature traits only and preserves save scaling independently',()=>{
 const s=snapshot('surge');assert.equal(typeof api.siphonMultiplier,'function');
 assert.equal(api.siphonMultiplier(s,['electricity']),1);assert.equal(api.siphonMultiplier(s,new Set(['electricity'])),1);
 assert.equal(api.siphonMultiplier(s,['shocked']),0.5);assert.equal(api.siphonMultiplier(s,{immunities:[{type:'electricity'}]}),0.5);
 assert.equal(api.siphonMultiplier({...s,disruptive:false},['electricity']),0.5);
 assert.equal(api.siphonMultiplier({...s,associatedTraits:['air','fire']},['fire']),1);
 assert.equal(api.siphonMultiplier(snapshot('surge','widen'),['electricity']),1);
 for(const outcome of [0,0.5,1,2]){assert.equal(20*outcome*api.siphonMultiplier(s,[]),10*outcome);assert.equal(20*outcome*api.siphonMultiplier(s,['electricity']),20*outcome)}
});
test('uncertain policies require explicit input only for dependent branches',()=>{
 assert.equal(typeof api.metapowerActionCost,'function');assert.equal(api.metapowerActionCost('siphoning'),1);
 assert.throws(()=>snapshot('voltage'),/policy|voltage/i);
 const unaffected=snapshot('voltage','siphoning',{policy:{highVoltage:'unaffected'}});assert.equal(unaffected.siphon.applies,false);assert.deepEqual(unaffected.suppressEffects,[]);assert.equal(api.siphonMultiplier(unaffected,[]),1);
 for(const id of ['surge','anvil','shot','retributive'])assert.throws(()=>snapshot(id,'siphoning',{selection:{discharge:true}}),/policy|discharge/i);
 assert.equal(snapshot('surge','siphoning',{selection:{discharge:true,baseDistance:60},policy:{dischargeNonDamage:'remove'}}).area.distance,30);
 assert.equal(snapshot('surge','siphoning',{selection:{discharge:true,baseDistance:60},policy:{dischargeNonDamage:'retain'}}).area.distance,60);
 assert.equal(snapshot('anvil','siphoning',{selection:{discharge:true,baseDistance:60},policy:{dischargeNonDamage:'remove'}}).area.distance,30);
 assert.equal(snapshot('shot','siphoning',{selection:{discharge:true},policy:{dischargeNonDamage:'remove'}}).range,40);
 assert.equal(snapshot('retributive','siphoning',{selection:{discharge:true},policy:{dischargeNonDamage:'retain'}}).saveDowngrade,1);
});
test('Widen costs one action and rejects policy overrides that conflict with its source',()=>{
 assert.equal(api.metapowerActionCost('widen'),1);
 assert.equal(api.metapowerActionCost('widen',{widenActionCost:1}),1);
 for(const cost of [0,2,3,'free',null])assert.throws(()=>api.metapowerActionCost('widen',{widenActionCost:cost}),/cost|one action/i);
});
test('Siphoning removes reviewed added effects while preserving discharge cost, native outcome branches and unknown source boundary',()=>{
 const s=snapshot('static','siphoning',{selection:{discharge:true}});assert.equal(s.dischargeCost,1);assert.equal(s.siphon.applies,true);assert.deepEqual(s.suppressEffects,['charged','shocked']);assert.equal(s.outcomeMode,'attack-with-fixed-failure');
 assert.equal(snapshot('retributive').outcomeMode,'special-save');assert.equal(snapshot('chain').damageBasis,'trigger-damage-halved');
 assert.throws(()=>api.buildChannelSnapshot({kind:'siphoning',actor:actor(),item:{sourceId:P+'unreviewed'}}),/unsupported|reviewed/i);
});
test('Electric Shot range advances at 7, 11, 15 and 19 with its level cap',()=>{
 for(const [level,expected] of [[1,40],[6,40],[7,60],[10,60],[11,80],[14,80],[15,100],[18,100],[19,120],[20,120]]){
  assert.equal(snapshot('shot','siphoning',{level}).range,expected,`normal range at level ${level}`);
  const widened=snapshot('shot','widen',{level});assert.equal(widened.range,expected);assert.equal(widened.area,null,'Widen does not increase single-target range');
 }
});
test('Electric Shot discharge policy scales the level-adjusted base without changing its cost',()=>{
 for(const [level,normal,discharged] of [[6,40,80],[7,60,120],[11,80,160],[15,100,200],[19,120,240],[20,120,240]]){
  const retain=snapshot('shot','siphoning',{level,selection:{discharge:true},policy:{dischargeNonDamage:'retain'}});
  const remove=snapshot('shot','siphoning',{level,selection:{discharge:true},policy:{dischargeNonDamage:'remove'}});
  assert.equal(retain.range,discharged,`retained discharge range at level ${level}`);assert.equal(remove.range,normal,`removed discharge range at level ${level}`);
  assert.equal(retain.dischargeCost,1);assert.equal(remove.dischargeCost,1);
  assert.equal(snapshot('shot','widen',{level,selection:{discharge:true}}).range,discharged);
 }
 assert.throws(()=>snapshot('shot','siphoning',{level:7,selection:{discharge:true}}),/policy|discharge/i);
});
test('Reactive Chain has no new Charged effect to suppress and discharge still costs one',()=>{
 const normal=snapshot('chain'),discharged=snapshot('chain','siphoning',{selection:{discharge:true}});
 assert.deepEqual(normal.suppressEffects,[]);assert.deepEqual(discharged.suppressEffects,[]);
 assert.equal(normal.dischargeCost,0);assert.equal(discharged.dischargeCost,1);assert.equal(discharged.damageBasis,'trigger-damage-halved');
});
