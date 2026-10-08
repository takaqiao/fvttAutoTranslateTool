import {test} from 'node:test';import assert from 'node:assert/strict';
import {extension,completionState,createTreatmentProvider} from '../../scripts/exploration/treatment.mjs';
test('early native return cannot complete missing result or application',()=>{
  assert.equal(completionState({outcome:'criticalSuccess',checkId:'C1',expectedResults:2,persistedResultIds:['D1'],applicationReceiptIds:[],expectedApplications:2}),'awaiting-evidence');
  assert.equal(completionState({outcome:'failure',checkId:'C1',expectedResults:1,persistedResultIds:['cut'],applicationReceiptIds:['R1'],expectedApplications:1}),'confirmed');
});
test('extension adds the original rolled healing after fifty further minutes',()=>{
  assert.deepEqual(extension({startedAt:0,checkpointAt:600,outcome:'success',rolledHealing:19}),{endsAt:3600,additionalHealing:19});
  assert.equal(extension({startedAt:0,checkpointAt:600,outcome:'failure',rolledHealing:null}),null);
  assert.throws(()=>extension({startedAt:0,checkpointAt:500,outcome:'success',rolledHealing:19}),/checkpoint/);
});
test('a saved treatment estimate cannot dispatch native work after its runtime version changes',async()=>{
 const healer={systemVersion:'8.6.0',medicine:{rank:1},slugs:[],wardCapacity:1},patient={modeOfBeing:'living',pool:{ready:true}};let calls=0;
 const provider=createTreatmentProvider({capabilities:{discover:async()=>healer,snapshot:async()=>[patient]},nativeTreatment:{},ownerOperations:{runActivityWithOwner:async()=>{calls++;return {status:'confirmed'}}}});
 const activity={id:'A',actorUUID:'Actor.H',patientUUIDs:['Actor.P'],startedAt:0,options:{skill:'medicine',rank:'trained',estimateSource:{sourceVersion:'8.6.0'}}};
 assert.equal((await provider.begin(activity)).status,'started');healer.systemVersion='8.6.1';
 assert.deepEqual(await provider.complete(activity,{}),{status:'blocked',reason:'treatment-estimate-runtime-changed'});assert.equal(calls,0);
});
