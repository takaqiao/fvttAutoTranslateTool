import {test} from 'node:test';import assert from 'node:assert/strict';
import {createExplorationOwnerOperations} from '../../scripts/exploration/owner-operations.mjs';
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
