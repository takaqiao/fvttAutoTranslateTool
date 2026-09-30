import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createHpPools,deduplicatePoolEffects} from '../../scripts/exploration/hp-pool.mjs';
function setup(){
  let middleware;const master={uuid:'Actor.M',isOwner:true},slave={uuid:'Actor.S',modules:{'pf2e-toolbelt':{shareData:{data:{health:true}}}}},armor={uuid:'Actor.A',modules:{'pf2e-toolbelt':{shareData:{data:{health:false}}}}};
  const game={settings:{get:()=>true},modules:new Map([['pf2e-toolbelt',{active:true}]]),toolbelt:{api:{shareData:{getMasterInMemory:a=>a===slave?master:null,getSlavesInMemory:()=>[slave,armor]}}}};
  const pools=createHpPools({game,actorUpdateEvents:{addActorUpdateMiddleware:f=>{middleware=f}}});
  return {pools,master,slave,armor,call:(actor,wrapped,changes,options)=>middleware.call(actor,wrapped,changes,options)};
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
