import {test} from 'node:test';
import assert from 'node:assert/strict';
import {fixture} from './manual-pool-provider-fixture.mjs';

for(const remote of [false,true])test(`private broker joins ${remote?'GM original socket':'local original'} master Promise and original application receipt`,async t=>{
 const f=await fixture(remote);t.after(()=>f.close());const grant=await f.owner.broker.claim(f.request);
 const pending=f.owner.broker.withApplication(grant,f.owner.patient,f.operation);pending.catch(()=>{});await Promise.race([f.writeStarted,pending]);
 assert.equal(f.writes.length,1);assert.notEqual(Object.values((await f.ledger.getActivity('A')).proof.poolApplications)[0].state,'settled');f.finish();await pending;
 assert.equal((await f.owner.broker.lookup(f.request)).status,'settled');assert.equal(f.applications(),1);assert.equal(f.writes.length,1);
 await assert.rejects(f.owner.broker.withApplication(grant,f.owner.patient,f.operation),/private-grant/);assert.equal(f.applications(),1);
});

test('a lost terminal acknowledgement permits a settled read, not another native call',async t=>{
 const f=await fixture(false);t.after(()=>f.close());const grant=await f.owner.broker.claim(f.request);f.loseTerminalAck();
 const pending=f.owner.broker.withApplication(grant,f.owner.patient,f.operation);pending.catch(()=>{});await Promise.race([f.writeStarted,pending]);f.finish();await assert.rejects(pending,/timeout/);
 const saved=await f.owner.broker.lookup(f.request);assert.equal(saved.status,'settled');assert.equal(saved.permitNonce,undefined);assert.equal(f.writes.length,1);
});

for(const reason of ['stop','deleted source','revoked patient','remote writer revoked','master rejection'])test(`authenticated forward remains unknown after ${reason}`,async t=>{
 const f=await fixture(true);t.after(()=>f.close());const grant=await f.owner.broker.claim(f.request);
 const pending=f.owner.broker.withApplication(grant,f.owner.patient,f.operation);pending.catch(()=>{});await Promise.race([f.writeStarted,pending]);assert.equal(f.writes.length,1);
 if(reason==='stop'){await f.ledger.updateSession('S',{status:'closed'});f.invalid()}
 if(reason==='deleted source')f.messages.delete('R');
 if(reason==='revoked patient'){f.owner.patient.testUserPermission=()=>false;f.gm.patient.testUserPermission=()=>false}
 if(reason==='remote writer revoked')f.gm.master.isOwner=false;
 if(reason==='master rejection')f.reject(Error('master-rejected'));else f.finish();
 await assert.rejects(pending);const claim=Object.values((await f.ledger.getActivity('A')).proof.poolApplications)[0];assert.equal(claim.state,'applying');assert.equal(claim.terminal,undefined);
 await assert.rejects(f.owner.broker.withApplication(grant,f.owner.patient,f.operation),/private-grant/);assert.equal(f.writes.length,1);assert.equal(f.applications(),1);
});

test('noChange is sealed only from the original unchanged native receipt',async t=>{
 const f=await fixture(true);t.after(()=>f.close());f.receipt.flags.pf2e.appliedDamage=null;const grant=await f.owner.broker.claim(f.request);
 await f.owner.broker.withApplication(grant,f.owner.patient,async()=>{f.messages.set(f.receipt.id,f.receipt);return {receipt:f.receipt}});
 assert.equal(f.writes.length,0);assert.equal((await f.owner.broker.lookup(f.request)).noChange,true);
});

test('unknown begin ACK prevents the original application from starting',async t=>{
 const f=await fixture(true);t.after(()=>f.close());const grant=await f.owner.broker.claim(f.request);f.setAcknowledgement(()=>null);
 await assert.rejects(f.owner.broker.withApplication(grant,f.owner.patient,f.operation));assert.equal(f.applications(),0);assert.equal(f.writes.length,0);
 f.setAcknowledgement(ack=>ack);assert.equal((await f.owner.broker.lookup(f.request)).status,'reserved');await assert.rejects(f.owner.broker.withApplication(grant,f.owner.patient,f.operation),/private-grant/);
});

test('the original forwarding fields cannot be replaced by an edited socket payload',async t=>{
 const f=await fixture(true);t.after(()=>f.close());f.tamperForward(payload=>({...payload,'system.attributes.hp.value':99}));
 const grant=await f.owner.broker.claim(f.request),pending=f.owner.broker.withApplication(grant,f.owner.patient,f.operation);pending.catch(()=>{});
 await f.turn();f.finish();await assert.rejects(pending);assert.equal(f.writes.length,0);
});

test('two independent original uses in one pool cannot write concurrently',async t=>{
 const f=await fixture(true);t.after(()=>f.close());
 const activity=await f.ledger.getActivity('A');await f.ledger.insertActivity({...activity,id:'B',proof:{...activity.proof,useId:'U2'}});
 const second={...f.request,activityId:'B',useId:'U2',batchId:'second',attemptNonce:'second'};
 const grants=await Promise.all([f.owner.broker.claim(f.request),f.owner.broker.claim(second)]);
 const pending=grants.map(grant=>f.owner.broker.withApplication(grant,f.owner.patient,f.operation));for(const p of pending)p.catch(()=>{});
 await f.writeStarted;assert.equal(f.writes.length,1);f.finish();const results=await Promise.allSettled(pending);
 assert.equal(results.filter(row=>row.status==='fulfilled').length,1);assert.equal(f.applications(),1);assert.equal(f.writes.length,1);
});
