import {test} from 'node:test';import assert from 'node:assert/strict';
import {createExplorationOwnerOperations} from '../../scripts/exploration/owner-operations.mjs';
import {OWNER_TRANSPORT_CHANNEL,OWNER_TRANSPORT_PROTOCOL} from '../../scripts/exploration/owner-transport.mjs';
test('authenticated caller creates opaque context and durable owner claim',async()=>{
  const user={id:'G',isGM:true,active:true},actor={uuid:'Actor.H',isOwner:true,flags:{},testUserPermission:()=>true,update:async changes=>{actor.flags['pf2e-third-party-automation']={explorationExecutions:changes['flags.pf2e-third-party-automation.explorationExecutions']};return actor}};
  const game={user,users:{activeGM:user,get:id=>id==='G'?user:null},time:{worldTime:600}};
  const a={id:'A',actorUUID:actor.uuid,state:'completing',startedAt:0,endsAt:600};
  const ops=createExplorationOwnerOperations({game,fromUuid:async()=>actor,ledger:{getActivity:async()=>a}});let calls=0;
  ops.registerOperation('native',async(activity,ctx)=>{assert.equal(ops.isActivityContext(ctx,activity.id),true);assert.equal(ops.isActivityContext({...ctx},activity.id),false);calls++;return {checkId:'C'}});
  assert.deepEqual(await ops.runActivityWithOwner(a,'native'),{checkId:'C'});
  await assert.rejects(ops.runActivityWithOwner(a,'native'),/already/);assert.equal(calls,1);
  await assert.rejects(ops.ownerExecute({activity:a,operationId:'native'},'forged'),/gm/);
  await assert.rejects(ops.ownerExecute({activity:{...a,id:'B'},operationId:'Actor.update'},'G'),/unknown/);
});
test('concurrent duplicate owner requests cannot race an uncommitted actor claim',async()=>{
 const user={id:'G',isGM:true,active:true};let release,calls=0;const gate=new Promise(r=>release=r);const actor={uuid:'H',flags:{},testUserPermission:()=>true,update:async changes=>{await gate;actor.flags['pf2e-third-party-automation']={explorationExecutions:changes['flags.pf2e-third-party-automation.explorationExecutions']};return actor}};const game={user,users:{activeGM:user,get:()=>user},time:{worldTime:600}};const a={id:'A',actorUUID:'H',state:'completing',startedAt:0,endsAt:600};const ops=createExplorationOwnerOperations({game,fromUuid:async()=>actor,ledger:{getActivity:async()=>a}});ops.registerOperation('native',async()=>{calls++;return {}});
 const first=ops.ownerExecute({activity:a,operationId:'native'},'G');await new Promise(r=>setImmediate(r));
 const second=ops.ownerExecute({activity:a,operationId:'native'},'G');release();const results=await Promise.allSettled([first,second]);assert.equal(results[1].status,'rejected');assert.equal(calls,1);
});
test('external time, user stop and encounter invalidate the active native scope',async()=>{
 for(const event of ['time','stop','encounter']){const user={id:'G',isGM:true,active:true},actor={uuid:'H',flags:{},testUserPermission:()=>true,update:async()=>actor},game={user,users:{activeGM:user,get:()=>user},time:{worldTime:600}};const ops=createExplorationOwnerOperations({game,fromUuid:async()=>actor,ledger:{}});ops.registerOperation('native',async(a,ctx)=>{if(event==='time')game.time.worldTime=601;if(event==='stop')ops.cancelActivity(a);if(event==='encounter')game.combat={started:true};ctx.validate();return {}});await assert.rejects(ops.ownerExecute({activity:{id:'A',actorUUID:'H',state:'completing',startedAt:0,endsAt:600},operationId:'native'},'G'),/time|stop|encounter/)}
});

function atomicFixture(userId='G'){
 const gm={id:'G',isGM:true,active:true},owner={id:'O',active:true},users=Object.assign(new Map([[gm.id,gm],[owner.id,owner]]),{activeGM:gm}),listeners=new Map();let emits=0,reads=0;
 const identity={userId,clientNonce:'client'},game={user:users.get(userId),users,combat:{started:true},time:{worldTime:600},socket:{on(channel,fn){assert.equal(channel,OWNER_TRANSPORT_CHANNEL);listeners.set(channel,fn)},off(channel,fn){assert.equal(listeners.get(channel),fn);listeners.delete(channel)},emit(){emits++}}};
 const ledger={atomic:true,getActivity:async()=>{reads++;throw Error('encounter must reject before reading activities')},getSession:async()=>{reads++;throw Error('encounter must reject before reading sessions')},claimExecution:async()=>{reads++;throw Error('encounter must reject before granting execution')}};
 const ops=createExplorationOwnerOperations({game,ledger,fromUuid:async()=>{reads++;throw Error('encounter must reject before resolving native actors')},runtimeIdentity:()=>identity,getDriverScope:()=>({leaseNonce:'lease'})});
 return {game,identity,listeners,ops,counts:()=>({emits,reads})};
}
for(const userId of ['G','O'])test(`started encounter reload registers ${userId==='G'?'GM driver':'original owner'} transport while every exploration entry remains blocked`,async()=>{
 const f=atomicFixture(userId);assert.doesNotThrow(()=>f.ops.register());assert.equal(f.listeners.size,1,'ready must bind the real owner transport during an encounter');
 const activity={id:'A',sessionId:'S',actorUUID:'Actor.H',state:'completing',startedAt:0,endsAt:600};
 await assert.rejects(f.ops.runActivityWithOwner(activity,'native'),/encounter-started/);
 await assert.rejects(f.ops.createActivityContext({...activity,state:'planned'}),/encounter-started/);
 await assert.rejects(f.ops.reconcile(activity),/encounter-started/);
 const packet={protocol:OWNER_TRANSPORT_PROTOCOL,kind:userId==='G'?'claim':'offer',requestId:'request',receiverUserId:userId,rootUUID:'Journal.root',epoch:'epoch',sessionId:'S',activityId:'A',operationId:'native',actorUUID:'Actor.H',driverUserId:'G',ownerUserId:'O',offerId:'offer'};
 f.listeners.get(OWNER_TRANSPORT_CHANNEL)(packet,userId==='G'?'O':'G');await new Promise(resolve=>setImmediate(resolve));assert.deepEqual(f.counts(),{emits:0,reads:0},'transport delivery cannot claim or begin native exploration in combat');
 assert.throws(()=>f.ops.register(),/duplicate-exploration-socket/);f.ops.dispose();assert.equal(f.listeners.size,0);
});
test('encounter-safe registration still rejects an invalidated or disposed runtime before binding',()=>{
 for(const change of ['identity','user','dispose']){
  const f=atomicFixture();if(change==='identity')f.identity.clientNonce='replacement';if(change==='user')f.game.user=f.game.users.get('O');if(change==='dispose')f.ops.dispose();
  assert.throws(()=>f.ops.register(),/owner-runtime-invalidated/);assert.equal(f.listeners.size,0);assert.deepEqual(f.counts(),{emits:0,reads:0});
 }
});
