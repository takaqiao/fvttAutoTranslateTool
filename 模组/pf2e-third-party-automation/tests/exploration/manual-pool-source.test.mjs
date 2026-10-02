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
