import {test} from 'node:test';
import assert from 'node:assert/strict';
import {authorityFixture} from './authority-fixture.mjs';

const activity={id:'A',sessionId:'S',providerId:'manual',kind:'treatment',actorUUID:'Actor.H',patientUUIDs:['Actor.P'],hpPoolUUIDs:['Actor.M'],startedAt:0,endsAt:600,state:'awaiting-evidence',source:{manual:true,type:'native-action'},proof:{useId:'U',checkIds:['C'],resultIds:['D']}};
async function fixture(){const f=await authorityFixture(),ledger=f.client('issuer');await ledger.createSession({id:'S',manual:true,status:'recording',startedAt:0,budgetEndsAt:600,actorUUIDs:['Actor.H','Actor.P','Actor.M']});return {...f,ledger}}

test('generic insertion cannot mint a private observed manual source',async()=>{
 const f=await fixture();await assert.rejects(f.ledger.insertActivity({...activity,proof:{...activity.proof,manualPoolSource:{sourceNonce:'forged'}}}),/manual-pool-source/);
});
test('generic transitions cannot replace the private observed manual source',async()=>{
 const f=await fixture();await f.ledger.insertActivity(activity);const old=await f.ledger.getActivity('A');
 await assert.rejects(f.ledger.transitionActivity('A',{expected:['awaiting-evidence'],patch:{proof:{...old.proof,manualPoolSource:{sourceNonce:'forged'}}}}),/manual-pool-source/);
});
test('ordinary recording binds its private pool issuer to the creation ACK tab',async()=>{
 const f=await fixture(),s=await f.ledger.getSession('S');assert.deepEqual(s.manualPoolIssuer,{userId:'G',clientNonce:'issuer'});
 await assert.rejects(f.client('peer').updateSession('S',{manualPoolIssuer:{userId:'G',clientNonce:'peer'}}),/immutable-session/);
});
for(const state of ['confirmed','awaiting-evidence'])test(`a persisted legacy ${state} source stays readable and cannot issue another source write`,async()=>{
 const f=await fixture();await f.ledger.insertActivity(activity);const seed=await f.ledger.all();
 const legacy={version:1,sessionId:'S',activityId:'A',actorUUID:'Actor.H',patientUUID:'Actor.P',sourceType:'native-action',useId:'U',checkId:'C',resultId:'D',rollIndex:0,worldTime:0,sourceUserId:'G',sourceClientNonce:'issuer',sourceNonce:'old-source',provider:{id:'pf2e',version:'8.5.1'},documentsDigest:'a'.repeat(64)};
 seed.activities.A.state=state;seed.activities.A.proof.manualPoolSource=legacy;
 const restored=await authorityFixture(seed),ledger=restored.client('issuer'),pages=restored.raw.pages.length;
 assert.equal((await ledger.getActivity('A')).state,state);assert.deepEqual((await ledger.snapshot('S')).activities[0].proof.manualPoolSource,legacy);
 assert.throws(()=>ledger.recordManualPoolSource(legacy,{evidenceGuard:()=>true}),/invalid-manual-pool-source/);
 assert.equal(restored.raw.pages.length,pages);
});
