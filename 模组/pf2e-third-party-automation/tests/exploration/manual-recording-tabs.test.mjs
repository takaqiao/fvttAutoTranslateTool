import {test} from 'node:test';
import assert from 'node:assert/strict';
import {authorityFixture} from './authority-fixture.mjs';
import {manualEvidenceFixture} from './manual-evidence-fixture.mjs';
import {createManualEvents} from '../../scripts/exploration/manual-events.mjs';
import {createCoordinator} from '../../scripts/exploration/coordinator.mjs';

async function fixture(){
 const evidence=manualEvidenceFixture(),storage=await authorityFixture(),issuer=storage.client('issuer'),peer=storage.client('peer');
 await issuer.createSession({id:'S',manual:true,status:'recording',startedAt:0,budgetEndsAt:600,actorUUIDs:['Actor.H','Actor.P']});
 const createRecorder=ledger=>createManualEvents({...evidence.options,ledger,isAuthority:undefined});
 return {...evidence,storage,issuer,peer,createRecorder};
}

test('a restored same-account peer has ordinary recording authority without the pool issuer nonce',async()=>{
 const f=await fixture(),coordinator=createCoordinator({ledger:f.peer,providers:[],isAuthority:()=>true});
 const restored=await coordinator.restore('S');assert.equal(restored.status,'recording');assert.deepEqual(restored.manualPoolIssuer,{userId:'G',clientNonce:'issuer'});
 await f.createRecorder(f.peer).observe(f.event);
 assert.equal((await f.issuer.snapshot('S')).activities.length,1);
});

test('two GM tabs observing the first source concurrently save one activity without rejecting either observer',async()=>{
 const f=await fixture();let entered=0,release;const ready=new Promise(resolve=>{release=resolve});
 const synchronized=ledger=>({...ledger,snapshot:async sid=>{const saved=await ledger.snapshot(sid);if(++entered===2)release();await ready;return saved}});
 const observations=await Promise.allSettled([f.createRecorder(synchronized(f.issuer)).observe(f.event),f.createRecorder(synchronized(f.peer)).observe(f.event)]);
 assert.deepEqual(observations.map(result=>result.status),['fulfilled','fulfilled']);
 const saved=await f.issuer.snapshot('S');assert.deepEqual(saved.session.activityIds,['manual:W']);assert.equal(saved.activities.length,1);
 assert.deepEqual(saved.activities[0].proof.resultIds,['W']);
});

test('a stale peer observation preserves private source and application proof added before its transaction',async()=>{
 const f=await fixture(),recorder=f.createRecorder(f.peer);await recorder.observe(f.event);
 const source={sourceNonce:'issued'},claim={state:'applying',request:{resultId:'W'}};
 // Seed private protocol state at the storage boundary; the real source and
 // claim protocols are exercised separately by manual-pool-registry/provider.
 await f.storage.storage('issuer').transact(state=>{const a=state.activities['manual:W'];a.proof.manualPoolSource=source;a.proof.poolApplications={batch:claim};a.proof.receiptIds=['R'];a.options.missing=['native-immunity-receipt']});
 await recorder.observe(f.event);
 const a=await f.issuer.getActivity('manual:W');assert.deepEqual(a.proof.manualPoolSource,source);assert.deepEqual(a.proof.poolApplications,{batch:claim});assert.deepEqual(a.proof.receiptIds,['R']);assert.deepEqual(a.options.missing,['native-immunity-receipt']);
});

test('manual observation merges against the latest proof and keeps the first source order',async()=>{
 const f=await fixture();await f.createRecorder(f.issuer).observe(f.event);const a=await f.issuer.getActivity('manual:W');
 await f.peer.insertActivity({...a,order:12,proof:{...a.proof,resultIds:['D'],receiptIds:['R']}},{manualObservation:true});
 const saved=await f.issuer.getActivity(a.id);assert.equal(saved.order,0);assert.deepEqual(saved.proof.resultIds,['W','D']);assert.deepEqual(saved.proof.receiptIds,['R']);assert.deepEqual((await f.issuer.getSession('S')).activityIds,[a.id]);
 await assert.rejects(f.peer.insertActivity(a),/duplicate-activity/);
});

test('manual observation rejects a reused ID with changed provenance and cannot inject private proof',async()=>{
 const f=await fixture();await f.createRecorder(f.issuer).observe(f.event);const a=await f.issuer.getActivity('manual:W');
 for(const change of [{actorUUID:'Actor.P'},{patientUUIDs:['Actor.H']},{source:{...a.source,type:'native-action'}},{proof:{...a.proof,useId:'OTHER'}},{durationSeconds:1}])await assert.rejects(f.peer.insertActivity({...a,...change},{manualObservation:true}),/manual-source-conflict/);
 for(const proof of [{...a.proof,manualPoolSource:{}},{...a.proof,poolApplications:{}}])await assert.rejects(f.peer.insertActivity({...a,proof},{manualObservation:true}),/manual-pool-source-required|manual-pool-claim-required/);
 assert.deepEqual(await f.issuer.getActivity(a.id),a);
});

test('manual observation does not append to a stopped recording',async()=>{
 const f=await fixture();await f.createRecorder(f.issuer).observe(f.event);const a=await f.issuer.getActivity('manual:W');await f.issuer.updateSession('S',{status:'stopped'});
 await assert.rejects(f.peer.insertActivity(a,{manualObservation:true}),/manual-recording-required/);
 assert.deepEqual(await f.issuer.getActivity(a.id),a);
});

test('another tab cannot enroll a copied message from the same invocation as a second activity',async()=>{
 const f=await fixture();await f.createRecorder(f.issuer).observe(f.event);const a=await f.issuer.getActivity('manual:W');
 await assert.rejects(f.peer.insertActivity({...a,id:'manual:COPY',groupId:'manual:COPY',source:{...a.source,messageId:'COPY'}},{manualObservation:true}),/manual-source-already-enrolled/);
 assert.deepEqual((await f.issuer.getSession('S')).activityIds,[a.id]);
});
