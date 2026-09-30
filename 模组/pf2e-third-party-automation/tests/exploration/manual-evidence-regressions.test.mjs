import {test} from 'node:test';
import assert from 'node:assert/strict';
import {manualEvidenceFixture,flush,M} from './manual-evidence-fixture.mjs';
test('a matching source receipt authored without patient ownership cannot confirm manual HP application',async()=>{
 const f=manualEvidenceFixture(),r=f.createRecorder();r.start();await r.observe({...f.event,missing:['native-application-receipt']});
 await f.fire(f.receipt('FORGED','OUTSIDER'));
 const a=await f.ledger.getActivity('manual:W');assert.equal(a.state,'awaiting-evidence');assert.deepEqual(a.proof.receiptIds,[]);assert.deepEqual(a.options.missing,['native-application-receipt']);r.stop();
});
test('an owned receipt object must be the exact persisted ChatMessage, not a copied shape',async()=>{
 const f=manualEvidenceFixture(),r=f.createRecorder();r.start();await r.observe({...f.event,missing:['native-application-receipt']});
 const saved=f.receipt();f.messages.set(saved.id,saved);const copy={...saved};f.handlers.get('createChatMessage')(copy);await flush();
 assert.deepEqual((await f.ledger.getActivity('manual:W')).proof.receiptIds,[]);r.stop();
});
for(const author of ['PUSER','G'])test(`an exact source-bound native receipt from ${author} confirms its selected patient`,async()=>{
 const f=manualEvidenceFixture(),r=f.createRecorder();r.start();await r.observe({...f.event,missing:['native-application-receipt']});await f.fire(f.receipt('R',author));
 const a=await f.ledger.getActivity('manual:W');assert.equal(a.state,'confirmed');assert.deepEqual(a.proof.receiptIds,['R']);assert.deepEqual(f.errors,[]);r.stop();
});
test('wrong patient, reverted, or ambiguous-source receipts stay unconfirmed',async()=>{
 const f=manualEvidenceFixture(),r=f.createRecorder();r.start();await r.observe({...f.event,missing:['native-application-receipt']});
 const wrong=f.receipt('WRONG');wrong.flags.pf2e.appliedDamage.uuid='Actor.Q';await f.fire(wrong);
 const reverted=f.receipt('REVERTED');reverted.flags.pf2e.appliedDamage.isReverted=true;await f.fire(reverted);
 const ambiguous=f.receipt('AMBIGUOUS');ambiguous.flags.pf2e.context.options.push(`${M}:source:C:0`);await f.fire(ambiguous);
 assert.deepEqual((await f.ledger.getActivity('manual:W')).proof.receiptIds,[]);r.stop();
});
test('patient and author changes during UUID resolution cannot borrow prior receipt authorization',async()=>{
 for(const change of [message=>{message.speaker.actor='Q'},message=>{message.author={id:'OUTSIDER'}}]){
  const f=manualEvidenceFixture();let release;const gate=new Promise(resolve=>{release=resolve});f.options.fromUuid=async uuid=>uuid==='Actor.P'?gate:f.healer;
  const r=f.createRecorder();r.start();await r.observe({...f.event,missing:['native-application-receipt']});const receipt=f.receipt();f.messages.set(receipt.id,receipt);f.handlers.get('createChatMessage')(receipt);await flush();change(receipt);release(f.patient);await flush();
  assert.deepEqual((await f.ledger.getActivity('manual:W')).proof.receiptIds,[]);r.stop();
 }
});
test('GM recorder reconstruction reads an existing source-bound Workbench immunity without repeating native work',async()=>{
 const f=manualEvidenceFixture(),old=f.createRecorder();old.start();await old.observe(f.event);await f.fire(f.receipt());old.stop();
 const item=f.immunity();f.patient.items.set(item.id,item);const fresh=f.createRecorder();fresh.start();await flush();
 const a=await f.ledger.getActivity('manual:W');assert.equal(a.state,'confirmed');assert.deepEqual(a.proof.immunityIds,['Actor.P.Item.I']);assert.deepEqual(a.proof.receiptIds,['R']);
 fresh.stop();const again=f.createRecorder();again.start();await flush();assert.deepEqual((await f.ledger.getActivity('manual:W')).proof.immunityIds,['Actor.P.Item.I']);assert.deepEqual(f.errors,[]);again.stop();
});
test('immunity reconstruction ignores mismatched creators and reads only matching patient actors',async()=>{
 const f=manualEvidenceFixture(),old=f.createRecorder();old.start();await old.observe({...f.event,missing:['native-immunity-receipt']});old.stop();
 const item=f.immunity();item.flags[M].explorationManualImmunity.creatorId='OUTSIDER';f.patient.items.set(item.id,item);
 const original=f.options.fromUuid;f.options.fromUuid=async uuid=>{assert.equal(uuid,'Actor.P');return original(uuid)};
 const fresh=f.createRecorder();fresh.start();await flush();assert.equal((await f.ledger.getActivity('manual:W')).state,'awaiting-evidence');assert.deepEqual((await f.ledger.getActivity('manual:W')).proof.immunityIds,[]);fresh.stop();
});
for(const delivery of ['restored','live'])for(const invalidation of ['reverted','deleted','ownership-revoked'])test(`${delivery} immunity cannot confirm a ${invalidation} prior HP receipt`,async()=>{
 const f=manualEvidenceFixture(),old=f.createRecorder();old.start();await old.observe(f.event);const receipt=f.receipt();await f.fire(receipt);
 assert.deepEqual((await f.ledger.getActivity('manual:W')).options.missing,['native-immunity-receipt']);
 if(invalidation==='reverted')receipt.flags.pf2e.appliedDamage.isReverted=true;
 if(invalidation==='deleted')f.messages.delete(receipt.id);
 if(invalidation==='ownership-revoked')f.patient.testUserPermission=user=>user?.isGM===true;
 const item=f.immunity();item.flags[M].explorationManualImmunity.creatorId='G';f.patient.items.set(item.id,item);
 let active=old;
 if(delivery==='restored'){old.stop();active=f.createRecorder();active.start()}
 else f.handlers.get('createItem')(item,{},'G');
 await flush();const a=await f.ledger.getActivity('manual:W');
 assert.equal(a.state,'awaiting-evidence');assert.deepEqual(a.options.missing,['native-application-receipt']);
 assert.deepEqual(a.proof.receiptIds,['R']);assert.deepEqual(a.proof.immunityIds,['Actor.P.Item.I']);assert.deepEqual(f.errors,[]);active.stop();
});
test('immunity arrival cannot treat a deleted native result as an empty completed HP requirement',async()=>{
 const f=manualEvidenceFixture(),r=f.createRecorder();f.messages.set('D',{...f.messages.get('W'),id:'D'});
 r.start();await r.observe({...f.event,resultIds:['D','W']});await f.fire(f.receipt());
 const stageReceipt=f.receipt('RD');stageReceipt.flags.pf2e.context.options=[`${M}:source:D:0`];await f.fire(stageReceipt);
 assert.deepEqual((await f.ledger.getActivity('manual:W')).options.missing,['native-immunity-receipt']);f.messages.delete('D');
 const item=f.immunity();f.patient.items.set(item.id,item);f.handlers.get('createItem')(item,{},'PUSER');await flush();
 const a=await f.ledger.getActivity('manual:W');assert.equal(a.state,'awaiting-evidence');assert.deepEqual(a.options.missing,['native-application-receipt']);assert.deepEqual(a.proof.receiptIds,['R','RD']);r.stop();
});
test('all stored HP receipts are checked together after asynchronous patient resolution',async()=>{
 const f=manualEvidenceFixture(),original=f.options.fromUuid,receipt=f.receipt();let checking=false,reads=0;
 f.options.fromUuid=async uuid=>{if(checking&&uuid==='Actor.P'&&++reads===2)receipt.flags.pf2e.appliedDamage.isReverted=true;return original(uuid)};
 f.messages.set('D',{...f.result,id:'D'});const r=f.createRecorder();r.start();await r.observe({...f.event,resultIds:['D','W']});await f.fire(receipt);
 const stageReceipt=f.receipt('RD');stageReceipt.flags.pf2e.context.options=[`${M}:source:D:0`];await f.fire(stageReceipt);
 checking=true;const item=f.immunity();f.patient.items.set(item.id,item);f.handlers.get('createItem')(item,{},'PUSER');await flush();
 const a=await f.ledger.getActivity('manual:W');assert.equal(a.state,'awaiting-evidence');assert.deepEqual(a.options.missing,['native-application-receipt']);assert.deepEqual(f.errors,[]);r.stop();
});
