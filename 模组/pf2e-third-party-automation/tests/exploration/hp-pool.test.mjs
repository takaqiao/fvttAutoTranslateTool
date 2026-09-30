import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createHpPools,deduplicatePoolEffects} from '../../scripts/exploration/hp-pool.mjs';
function setup(){
  let middleware;const master={uuid:'Actor.M',isOwner:true},slave={uuid:'Actor.S',modules:{'pf2e-toolbelt':{shareData:{data:{health:true}}}}},armor={uuid:'Actor.A',modules:{'pf2e-toolbelt':{shareData:{data:{health:false}}}}};
  const game={settings:{get:()=>true},modules:new Map([['pf2e-toolbelt',{active:true}]]),toolbelt:{api:{shareData:{getMasterInMemory:a=>a===slave?master:null,getSlavesInMemory:()=>[slave,armor]}}}};
  const pools=createHpPools({game,actorUpdateEvents:{addActorUpdateMiddleware:f=>{middleware=f}}});
  return {pools,master,slave,armor,game,call:(actor,wrapped,changes,options)=>middleware.call(actor,wrapped,changes,options)};
}
test('runtime health gate, not armor sharing, owns pool identity',()=>{
  const s=setup();assert.deepEqual(s.pools.discover(s.slave).memberUUIDs,['Actor.M','Actor.S']);assert.equal(s.pools.discover(s.armor).poolUUID,'Actor.A');
});
test('unawaited slave forwarding waits actual master promise and cannot apply twice',async()=>{
  const s=setup();let finish,done=false,count=0;
  const delay=new Promise(resolve=>{finish=resolve});
  const result=s.pools.withNativeApplication({id:'A1'},s.slave,async scope=>{
    count++; await s.call(s.slave,async()=>{s.call(s.master,()=>delay,{'system.attributes.hp.value':20},{});return s.slave},{'system.attributes.hp.value':20},{explorationApplication:scope});
    return s.slave;
  }).then(r=>{done=true;return r});
  await new Promise(r=>setImmediate(r));assert.equal(done,false);finish(s.master);assert.equal((await result).poolReceipt.actorUUID,'Actor.M');assert.equal(count,1);
});
test('unrelated master write and socket-only ownership never prove application',async()=>{
  const s=setup();await assert.rejects(s.pools.withNativeApplication({id:'A1'},s.slave,async()=>{await s.call(s.master,async()=>s.master,{'system.attributes.hp.value':19},{});return s.slave}),/forward/);
  s.master.isOwner=false;await assert.rejects(s.pools.withNativeApplication({id:'A2'},s.slave,async()=>{throw Error('must not run')}),/owner/);
});
test('same effect selects larger healing once; different effects remain additive',()=>{
  const results=deduplicatePoolEffects([{poolUUID:'M',effectId:'one',amount:5},{poolUUID:'M',effectId:'one',amount:10},{poolUUID:'M',effectId:'two',amount:7}]);assert.equal(results.reduce((s,x)=>s+x.amount,0),17);
});
test('zero effective HP change is confirmed by exact saved native no-change receipt',async()=>{
  const s=setup();s.slave.id='S';const receipt={id:'R',speaker:{actor:'S'},flags:{pf2e:{appliedDamage:null,context:{type:'damage-taken',options:['pf2e-third-party-automation:exploration-apply:A1:D1:Actor.S']}}}};
  s.game.messages=new Map([['R',receipt]]);
  const result=await s.pools.withNativeApplication({id:'A1'},s.slave,async()=>({receipt}));assert.equal(result.poolReceipt.noChange,true);
});
test('actual asynchronous preUpdate boundary keeps the precise Toolbelt forwarding call',async()=>{
 const s=setup();let finish;const gate=new Promise(r=>finish=r);s.slave._preUpdate=async changes=>{s.call(s.master,()=>gate,{'system.attributes.hp.value':changes.system.attributes.hp.value},{});return true};
 const p=s.pools.withNativeApplication({id:'A'},s.slave,async()=>{await s.call(s.slave,async()=>{await Promise.resolve();await s.slave._preUpdate({system:{attributes:{hp:{value:20}}}},{},'G');return s.slave},{system:{attributes:{hp:{value:20}}}},{});return s.slave});p.catch(()=>{});await new Promise(r=>setImmediate(r));finish(s.master);assert.equal((await p).poolReceipt.actorUUID,'Actor.M');assert.deepEqual((await p).poolReceipt.fields,{'system.attributes.hp.value':20});
});
test('pool receipt retains prepared master before and requested after, separate from slave raw undo delta',async()=>{
 const s=setup();s.master.system={attributes:{hp:{value:50,max:73,temp:0,_modifiers:[{unneeded:true}]}}};
 const result=await s.pools.withNativeApplication({id:'A'},s.slave,async()=>{await s.call(s.slave,async()=>{await s.call(s.master,async()=>s.master,{'system.attributes.hp.value':73},{});return s.slave},{'system.attributes.hp.value':73},{});return s.slave});
 assert.deepEqual(result.poolReceipt.before,{value:50,max:73,temp:0});assert.equal(result.poolReceipt.after.value,73);assert.equal(result.poolReceipt.patientUUID,'Actor.S');
});
