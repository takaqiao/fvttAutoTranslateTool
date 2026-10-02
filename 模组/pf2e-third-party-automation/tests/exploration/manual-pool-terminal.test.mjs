import {test} from 'node:test';
import assert from 'node:assert/strict';
import {authorityFixture} from './authority-fixture.mjs';

async function fixture(){
 const f=await authorityFixture(),ledger=f.client('issuer');await ledger.createSession({id:'S',manual:true,status:'recording',actorUUIDs:['Actor.H','Actor.P'],startedAt:0,budgetEndsAt:600});
 const claims=[];
 for(const id of ['A','B']){
  await ledger.insertActivity({id,sessionId:'S',providerId:'manual',kind:'treatment',actorUUID:'Actor.H',patientUUIDs:['Actor.P'],hpPoolUUIDs:['Actor.M'],startedAt:0,endsAt:600,state:'awaiting-evidence',source:{manual:true,type:'native-action'},proof:{useId:id,checkIds:['C'],resultIds:['R']}});
  const request={sessionId:'S',activityId:id,actorUUID:'Actor.H',sourceType:'native-action',useId:id,checkId:'C',resultId:'R',rollIndex:0,stage:'healing',poolUUID:'Actor.M',patientUUIDs:['Actor.P'],batchId:id,ownerClientNonce:'O1',attemptNonce:id};
  const permit=await ledger.claimManualPoolApplication({request,effectId:'R',selectedPatientUUID:'Actor.P',sourceDigest:'a'.repeat(64),ownerUserId:'O',permitNonce:id,worldTime:0},{evidenceGuard:()=>true});claims.push({request,permit});
 }
 return {...f,ledger,claims};
}
test('independent use claims cannot begin simultaneous writes to their shared pool',async()=>{
 const f=await fixture();assert.equal(typeof f.ledger.beginManualPoolApplication,'function');
 const result=await Promise.allSettled(f.claims.map((claim,i)=>f.client(`gm${i}`).beginManualPoolApplication(claim,{evidenceGuard:()=>true})));
 assert.equal(result.filter(x=>x.status==='fulfilled').length,1);
});
test('noChange settlement releases the write lock but never grants through lookup',async()=>{
 const f=await fixture(),first=f.claims[0];assert.equal(typeof f.ledger.beginManualPoolApplication,'function');
 first.permit=await f.ledger.beginManualPoolApplication(first,{evidenceGuard:()=>true});
 await f.ledger.recordManualPoolTerminal({...first,receiptId:'RECEIPT',noChange:true,master:null},{evidenceGuard:()=>true});
 const saved=await f.ledger.lookupManualPoolProof(first.request,'O');assert.equal(saved.status,'settled');assert.equal(saved.permitNonce,undefined);
 assert.equal((await f.ledger.beginManualPoolApplication(f.claims[1],{evidenceGuard:()=>true})).state,'applying');
});
test('unknown begin acknowledgement preserves the pool lock without executing again',async()=>{
 const f=await fixture();assert.equal(typeof f.ledger.beginManualPoolApplication,'function');f.setAcknowledgement(()=>null);
 await assert.rejects(f.ledger.beginManualPoolApplication(f.claims[0],{evidenceGuard:()=>true}));f.setAcknowledgement(ack=>ack);
 await assert.rejects(f.ledger.beginManualPoolApplication(f.claims[1],{evidenceGuard:()=>true}),/busy/);
 assert.equal((await f.ledger.lookupManualPoolProof(f.claims[0].request,'O')).status,'reserved');
});

function terminal(permit){return {binding:{permitNonce:permit.permitNonce,applicationNonce:permit.applicationNonce,ownerUserId:'O',patientUUID:'Actor.P',poolUUID:'Actor.M'},poolUUID:'Actor.M',writerUserId:'O',fields:{'system.attributes.hp.value':20},before:{value:1,max:30,temp:0},terminal:'fulfilled'}}
for(const variant of ['unknown terminal field','unknown binding field','wrong writer','wrong nonce','missing master'])test(`terminal contract rejects ${variant}`,async()=>{
 const f=await fixture(),first=f.claims[0];first.permit=await f.ledger.beginManualPoolApplication(first,{evidenceGuard:()=>true});let master=terminal(first.permit);
 if(variant==='unknown terminal field')master.execute=true;
 if(variant==='unknown binding field')master.binding.mark=true;
 if(variant==='wrong writer')master.writerUserId='OTHER';
 if(variant==='wrong nonce')master.binding.applicationNonce='OTHER';
 if(variant==='missing master')master=null;
 await assert.rejects(f.ledger.recordManualPoolTerminal({...first,receiptId:'RECEIPT',noChange:false,master},{evidenceGuard:()=>true}),/invalid-manual-pool-terminal/);
 assert.equal((await f.ledger.lookupManualPoolProof(first.request,'O')).status,'reserved');
});

test('an exact original master terminal settles once and admits a different use',async()=>{
 const f=await fixture(),first=f.claims[0];first.permit=await f.ledger.beginManualPoolApplication(first,{evidenceGuard:()=>true});const input={...first,receiptId:'RECEIPT',noChange:false,master:terminal(first.permit)};
 await f.ledger.recordManualPoolTerminal(input,{evidenceGuard:()=>true});await assert.rejects(f.ledger.recordManualPoolTerminal(input,{evidenceGuard:()=>true}));
 assert.equal((await f.ledger.beginManualPoolApplication(f.claims[1],{evidenceGuard:()=>true})).state,'applying');
});
