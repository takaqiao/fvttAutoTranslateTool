import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createLedger} from '../../scripts/exploration/ledger.mjs';
import * as schema from '../../scripts/exploration/schema.mjs';
import {authorityFixture} from './authority-fixture.mjs';

const H='Actor.H',P='Actor.P',MEDIC=H+'.Item.Medic',BM=H+'.Item.BM';
const baseline=(patch={})=>({id:'review-1',actorUUID:H,itemUUID:MEDIC,source:'gm-reviewed',nativeCounter:false,remaining:1,period:'hourly',checkedAt:0,reviewedBy:'G',previousBaselineId:null,renewal:'initial',...patch});
const finite=(patch={})=>({version:1,secondsPerUse:6,battleMedicine:{enabled:true,maxUsesByActor:{[H]:2},rankByActor:{[H]:'trained'}},medicBypass:{enabled:true,maxUsesByActor:{[H]:1},baselineByActor:{[H]:baseline()}},...patch});
const session=(id='S',medicine=finite())=>({id,actorUUIDs:[H,P],startedAt:0,cursorAt:0,budgetEndsAt:7200,goalsByPool:[{poolUUID:P,targetHP:20}],activityIds:[],status:'running',finiteMedicine:medicine});
const activity=(id='A',patch={})=>({id,sessionId:'S',providerId:'battle-medicine',actorUUID:H,patientUUIDs:[P],hpPoolUUIDs:[P],startedAt:0,endsAt:6,state:'planned',source:{type:'coordinator'},options:{itemUUID:BM,rank:'trained',medicBaselineId:'review-1',medicReceiptId:'medic-bypass:'+id},...patch});
const permitInput=(s,patch={})=>({leaseNonce:s.driver.leaseNonce,operationId:'battle-medicine',ownerUserId:'G',ownerClientNonce:'owner',attemptNonce:'attempt',permitNonce:'permit',...patch});
async function fixture({medicine=finite(),now=0,readTime=true,userId='G'}={}){
 const server=await authorityFixture();let time=now,valid=true,finalHook;
 const make=(clientNonce='driver',who=userId)=>{
  const store=server.storage(clientNonce,who),transact=(fn,options)=>store.transact(fn,options?.validateCommit?{...options,validateCommit:()=>{finalHook?.();return options.validateCommit()}}:options);
  return createLedger({...store,transact,isAuthority:()=>true,identity:()=>({userId:who,clientNonce}),...readTime?{now:()=>time}:{}});
 };
 const ledger=make(),s=await ledger.createSession(session('S',medicine),{guard:()=>valid});
 const scope={leaseNonce:s.driver.leaseNonce},insert=async(id='A',patch={})=>ledger.insertActivity(activity(id,patch),scope);
 const reserve=async(id='A',guard=()=>valid)=>{assert.equal(typeof ledger.reserveBattleMedicine,'function','atomic BM reservation is not implemented');return ledger.reserveBattleMedicine(id,{...scope,guard})};
 const completing=async(id='A')=>{await ledger.transitionActivity(id,{...scope,expected:['planned'],patch:{state:'started'}});await ledger.transitionActivity(id,{...scope,expected:['started'],patch:{state:'completing'}})};
 const view=async()=>{assert.equal(typeof ledger.finiteMedicineView,'function','whole-root Medic view is not implemented');return ledger.finiteMedicineView({actorUUID:H,itemUUID:MEDIC,baselineId:'review-1',now:time})};
 return {server,ledger,s,scope,insert,reserve,completing,view,make,time:value=>time=value,valid:()=>valid,invalid:()=>valid=false,onFinal:fn=>finalHook=fn};
}

test('finite session capture rejects accessors before clone and fixes selected actors',()=>{
 assert.equal(typeof schema.captureFiniteMedicine,'function','finite input capture is not implemented');let reads=0;
 const config=Object.defineProperty({actorUUIDs:[H]},'finiteMedicine',{enumerable:true,get(){reads++;return finite()}});
 assert.throws(()=>schema.captureFiniteMedicine(config),/invalid-finite-medicine/);assert.equal(reads,0);
 assert.throws(()=>schema.captureFiniteMedicine({actorUUIDs:[P],finiteMedicine:finite()}),/invalid-finite-medicine/);
 const input={actorUUIDs:[H,P],finiteMedicine:finite()},captured=schema.captureFiniteMedicine(input);input.finiteMedicine.medicBypass.baselineByActor[H].remaining=0;assert.equal(captured.medicBypass.baselineByActor[H].remaining,1);
});

test('fresh baseline review requires actual finite world time and authenticated GM identity',async()=>{
 for(const [medicine,options] of [[finite(),{readTime:false}],[finite(),{now:NaN}],[finite(),{now:Infinity}],[finite(),{now:1}],[finite({medicBypass:{enabled:true,maxUsesByActor:{[H]:1},baselineByActor:{[H]:baseline({reviewedBy:'Other'})}}}),{}]]){
  await assert.rejects(fixture({medicine,...options}),/finite.*time|baseline.*time|baseline.*review|review.*identity|baseline.*identity/);
 }
 const f=await fixture({now:-10.5,medicine:finite({medicBypass:{enabled:true,maxUsesByActor:{[H]:1},baselineByActor:{[H]:baseline({checkedAt:-10.5})}}})});assert.equal(f.s.finiteMedicine.medicBypass.baselineByActor[H].checkedAt,-10.5);
});

test('ordinary legacy sessions do not require a finite time reader or gain a baseline',async()=>{
 const server=await authorityFixture(),ledger=server.client('ordinary'),input=session();delete input.finiteMedicine;
 const s=await ledger.createSession(input);assert.equal(Object.hasOwn(s,'finiteMedicine'),false);
});

test('BM reservation is durable before started and binds both attempt and independent Medic claim',async()=>{
 const f=await fixture();await f.insert();await f.reserve();const a=await f.ledger.getActivity('A');
 assert.equal(a.state,'planned');assert.equal(a.executor,undefined);assert.equal(a.proof.battleMedicineAttempt.state,'reserved');assert.equal(a.proof.medicBypass.state,'reserved');assert.equal(a.proof.medicBypass.baselineId,'review-1');
 assert.equal((await f.view()).remaining,0);assert.equal(Object.keys((await f.server.read()).clocks).length,0);
 const repeat=await f.reserve();assert.deepEqual(repeat.proof,a.proof);
});

test('ordinary BM reserves its per-session budget without consuming Medic',async()=>{
 const medicine=finite({medicBypass:{enabled:false,maxUsesByActor:{},baselineByActor:{}}}),f=await fixture({medicine});await f.insert('A',{options:{itemUUID:BM,rank:'trained'}});await f.reserve();
 const a=await f.ledger.getActivity('A');assert.equal(a.proof.battleMedicineAttempt.state,'reserved');assert.equal(Object.hasOwn(a.proof,'medicBypass'),false);
});

for(const [name,reader] of [['missing',false],['invalid',true]])
test(`repeat reservation rejects a ${name} actual time reader even for ordinary BM`,async()=>{
 const medicine=finite({medicBypass:{enabled:false,maxUsesByActor:{},baselineByActor:{}}}),f=await fixture({medicine});
 await f.insert('A',{options:{itemUUID:BM,rank:'trained'}});await f.reserve();const before=await f.server.read();
 const peer=createLedger({...f.server.storage('driver','G'),isAuthority:()=>true,identity:()=>({userId:'G',clientNonce:'driver'}),...reader?{now:()=>NaN}:{}});
 await assert.rejects(peer.reserveBattleMedicine('A',{...f.scope,guard:()=>true}),/finite-medicine-time-required/);
 assert.deepEqual(await f.server.read(),before);
});

for(const bypass of [false,true])
test(`repeat reservation rechecks actual time at final commit with Medic bypass ${bypass}`,async()=>{
 const medicine=bypass?finite():finite({medicBypass:{enabled:false,maxUsesByActor:{},baselineByActor:{}}}),f=await fixture({medicine});
 await f.insert('A',bypass?{}:{options:{itemUUID:BM,rank:'trained'}});await f.reserve();const before=await f.server.read();
 f.onFinal(()=>f.time(NaN));await assert.rejects(f.reserve(),/finite-medicine-time-required/);
 assert.deepEqual(await f.server.read(),before);
});

test('two clients competing for one baseline commit one reservation with no orphan losing attempt',async()=>{
 const f=await fixture();await f.insert();await f.insert('B',{startedAt:6,endsAt:12});const peer=f.make();
 assert.equal(typeof peer.reserveBattleMedicine,'function');const results=await Promise.allSettled([f.reserve('A'),peer.reserveBattleMedicine('B',{...f.scope,guard:()=>true})]);
 assert.equal(results.filter(r=>r.status==='fulfilled').length,1);const rows=Object.values((await f.server.read()).activities);assert.equal(rows.filter(a=>a.proof.battleMedicineAttempt).length,1);assert.equal(rows.filter(a=>a.proof.medicBypass).length,1);
});

test('per-session attempt budget includes occupied reservations and cannot be exceeded',async()=>{
 const medicine=finite({battleMedicine:{enabled:true,maxUsesByActor:{[H]:1},rankByActor:{[H]:'trained'}},medicBypass:{enabled:false,maxUsesByActor:{},baselineByActor:{}}}),f=await fixture({medicine});
 await f.insert('A',{options:{itemUUID:BM,rank:'trained'}});await f.reserve();await f.insert('B',{startedAt:6,endsAt:12,options:{itemUUID:BM,rank:'trained'}});
 await assert.rejects(f.reserve('B'),/battle-medicine-budget-exhausted/);assert.equal((await f.ledger.getActivity('B')).proof.battleMedicineAttempt,undefined);
});

test('wrong lease and final source invalidation cannot leave BM claims',async()=>{
 const f=await fixture();await f.insert();assert.equal(typeof f.ledger.reserveBattleMedicine,'function');
 await assert.rejects(f.ledger.reserveBattleMedicine('A',{leaseNonce:'wrong',guard:()=>true}),/session-driver-required/);
 f.onFinal(()=>f.invalid());await assert.rejects(f.reserve(),/finite.*changed|battle.*changed|source.*changed/);assert.equal((await f.ledger.getActivity('A')).proof.battleMedicineAttempt,undefined);
});

test('BM execution requires its original reservation and binds the permit atomically',async()=>{
 const f=await fixture();await f.insert();await assert.rejects(f.ledger.transitionActivity('A',{...f.scope,expected:['planned'],patch:{state:'started'}}),/battle-medicine-reservation-required/);
 const g=await fixture();await g.insert();await g.reserve();await g.completing();const permit=await g.ledger.claimExecution('A',permitInput(g.s,{offerId:'offer',requestId:'request',commandDigest:'command-digest'}),{guard:()=>true});
 assert.deepEqual(Object.keys(permit).sort(),['protocol','rootUUID','epoch','revision','sessionId','activityId','operationId','actorUUID','ownerUserId','ownerClientNonce','attemptNonce','permitNonce','leaseNonce','offerId','requestId','commandDigest','state'].sort());
 const a=await g.ledger.getActivity('A');assert.equal(a.proof.medicBypass.state,'claimed');assert.equal(a.proof.medicBypass.permitNonce,permit.permitNonce);assert.equal(a.proof.battleMedicineAttempt.permitNonce,permit.permitNonce);
 await assert.rejects(g.ledger.claimExecution('A',permitInput(g.s,{attemptNonce:'second'}),{guard:()=>true}),/activity-already-executed/);
});

test('execution source guard runs again before the server permit write',async()=>{
 const f=await fixture();await f.insert();await f.reserve();await f.completing();f.onFinal(()=>f.invalid());
 await assert.rejects(f.ledger.claimExecution('A',permitInput(f.s),{guard:f.valid}),/finite.*changed|battle.*changed|source.*changed/);assert.equal((await f.ledger.getActivity('A')).executor,undefined);
});

test('exact original failure witness uses the allowance without claiming HP or immunity completion',async()=>{
 const f=await fixture();await f.insert();await f.reserve();await f.completing();const permit=await f.ledger.claimExecution('A',permitInput(f.s),{guard:()=>true});f.time(6);
 assert.equal(typeof f.ledger.recordMedicUseWitness,'function');await f.ledger.recordMedicUseWitness('A',{permit,checkId:'original-failure',usedAt:0,guard:()=>true});
 const a=await f.ledger.getActivity('A');assert.equal(a.state,'completing');assert.equal(a.executionResult,undefined);assert.equal(a.proof.medicBypass.state,'used');assert.equal(a.proof.medicBypass.checkId,'original-failure');assert.equal((await f.view()).state,'spent');
 await f.ledger.recordMedicUseWitness('A',{permit,checkId:'original-failure',usedAt:0,guard:()=>true});
 await assert.rejects(f.ledger.recordMedicUseWitness('A',{permit,checkId:'another-check',usedAt:0,guard:()=>true}),/witness-conflict/);
});

test('unknown execution retains occupied allowance and never releases without original evidence',async()=>{
 const f=await fixture();await f.insert();await f.reserve();await f.completing();const permit=await f.ledger.claimExecution('A',permitInput(f.s),{guard:()=>true});
 await f.ledger.recordExecutionResult('A',{permit,result:{status:'uncertain',reason:'immunity-unconfirmed',proof:{useId:'A',checkIds:[],resultIds:[],receiptIds:[],immunityIds:[]}}});
 assert.equal((await f.view()).remaining,0);assert.equal(typeof f.ledger.cancelBattleMedicineReservation,'function');await assert.rejects(f.ledger.cancelBattleMedicineReservation('A',f.scope),/reservation.*release|reservation.*cancel|reservation.*unsafe/);
});

test('BM result consumption requires a synchronous exact witness guard and preserves GM proof',async()=>{
 const f=await fixture();await f.insert();await f.reserve();await f.completing();const permit=await f.ledger.claimExecution('A',permitInput(f.s),{guard:()=>true});f.time(6);
 const result={status:'confirmed',effectiveOutcome:'failure',sourceDegree:1,proof:{useId:'A',checkIds:['original-failure'],resultIds:[],receiptIds:[],immunityIds:['Actor.P.Item.Immunity']},resourceReceiptIds:['medic-bypass:A']};
 await assert.rejects(f.ledger.recordExecutionResult('A',{permit,result}),/witness.*guard/);
 await f.ledger.recordExecutionResult('A',{permit,result,witnessGuard:()=>true});const a=await f.ledger.getActivity('A');assert.equal(a.proof.medicBypass.state,'used');assert.equal(a.proof.battleMedicineAttempt.state,'used');assert.deepEqual(a.executionResult,result);assert.equal(a.executor.state,'settled');
 await f.ledger.transitionActivity('A',{...f.scope,expected:['completing'],patch:{...result,state:'confirmed',proof:a.proof}});assert.equal((await f.ledger.getActivity('A')).state,'confirmed');
});

test('a final witness change leaves use and result uncommitted',async()=>{
 const f=await fixture();await f.insert();await f.reserve();await f.completing();const permit=await f.ledger.claimExecution('A',permitInput(f.s),{guard:()=>true});f.time(6);f.onFinal(()=>f.invalid());
 const result={status:'confirmed',effectiveOutcome:'failure',proof:{useId:'A',checkIds:['C'],resultIds:[],receiptIds:[],immunityIds:['I']},resourceReceiptIds:['medic-bypass:A']};
 await assert.rejects(f.ledger.recordExecutionResult('A',{permit,result,witnessGuard:f.valid}),/witness.*changed|finite.*changed/);const a=await f.ledger.getActivity('A');assert.equal(a.proof.medicBypass.state,'claimed');assert.equal(a.executionResult,undefined);
});

test('safe planned cancellation releases both reservations while a claimed clock forbids release',async()=>{
 const f=await fixture();await f.insert();await f.reserve();assert.equal(typeof f.ledger.cancelBattleMedicineReservation,'function');await f.ledger.cancelBattleMedicineReservation('A',f.scope);assert.equal((await f.view()).remaining,1);
 const g=await fixture();await g.insert();await g.reserve();await g.ledger.upsertClockCommit({id:'clock',sessionId:'S',from:0,to:6,gmId:'G',state:'started'},g.scope);
 await assert.rejects(g.ledger.cancelBattleMedicineReservation('A',g.scope),/reservation.*release|reservation.*cancel|reservation.*unsafe/);assert.equal((await g.view()).remaining,0);
});

test('generic insertion, transition and session update cannot forge or overwrite finite declarations',async()=>{
 const f=await fixture();await assert.rejects(f.insert('forged',{proof:{battleMedicineAttempt:{state:'used'}}}),/finite.*claim|battle.*claim|protected.*proof/);await f.insert();await f.reserve();
 await assert.rejects(f.ledger.transitionActivity('A',{...f.scope,expected:['planned'],patch:{proof:{useId:null,checkIds:[],resultIds:[],receiptIds:[],immunityIds:[]}}}),/finite.*claim|battle.*claim|protected.*proof/);
 await assert.rejects(f.ledger.updateSession('S',{finiteMedicine:finite({secondsPerUse:0})}),/immutable/);
 await assert.rejects(f.ledger.updateSession('S',{finiteMedicineReviewChecks:[]}),/immutable|finite.*review/);
 await assert.rejects(f.ledger.transitionActivity('A',{...f.scope,expected:['planned'],patch:{options:{itemUUID:BM,rank:'expert'}}}),/immutable/);
 await assert.rejects(f.ledger.transitionActivity('A',{...f.scope,expected:['planned'],patch:{proof:null}}),/finite.*claim|protected.*proof/);
 await assert.rejects(f.ledger.transitionActivity('A',{...f.scope,expected:['planned'],patch:{options:null}}),/immutable/);
});

test('an acknowledged exact baseline reference in a new session retains spent root-wide usage',async()=>{
 const f=await fixture();await f.insert();await f.reserve();await f.completing();const permit=await f.ledger.claimExecution('A',permitInput(f.s),{guard:()=>true});f.time(6);await f.ledger.recordMedicUseWitness('A',{permit,checkId:'C',usedAt:0,guard:()=>true});
 await f.ledger.recordExecutionResult('A',{permit,result:{status:'uncertain',reason:'HP-unconfirmed'}});await f.ledger.transitionActivity('A',{...f.scope,expected:['completing'],patch:{state:'uncertain'}});await f.ledger.updateSession('S',{status:'paused'});
 const before=await f.view();assert.equal(before.state,'spent');await assert.rejects(f.ledger.createSession(session('T'),{guard:()=>true}),/unresolved-evidence-no-replay/);
});

test('baseline initial re-review and unknown predecessor cannot refill an occupied allowance',async()=>{
 const f=await fixture();await f.insert();await f.reserve();await f.ledger.updateSession('S',{status:'paused'});
 const fresh=finite({medicBypass:{enabled:true,maxUsesByActor:{[H]:1},baselineByActor:{[H]:baseline({id:'review-2'})}}});
 await assert.rejects(f.ledger.createSession(session('T',fresh),{guard:()=>true}),/baseline.*predecessor|baseline.*initial|unresolved-evidence/);
});

test('protected external-use events invalidate root-wide availability and exact duplicate events are idempotent',async()=>{
 const f=await fixture();f.time(2);assert.equal(typeof f.ledger.noteExternalBattleMedicineUse,'function');const input={sessionId:'S',actorUUID:H,checkId:'external-BM',observedAt:2};
 await assert.rejects(async()=>f.ledger.noteExternalBattleMedicineUse(input,{guard:()=>false}),/external.*changed|finite.*changed/);
 await f.ledger.noteExternalBattleMedicineUse(input,{guard:()=>true});const first=await f.ledger.getSession('S');await f.ledger.noteExternalBattleMedicineUse(input,{guard:()=>true});assert.deepEqual((await f.ledger.getSession('S')).finiteMedicineReviewChecks,first.finiteMedicineReviewChecks);assert.equal((await f.view()).state,'uncertain');
});

test('an external exact BM use between reservation and completion prevents a native permit',async()=>{
 const f=await fixture();await f.insert();await f.reserve();await f.completing();f.time(2);assert.equal(typeof f.ledger.noteExternalBattleMedicineUse,'function');await f.ledger.noteExternalBattleMedicineUse({sessionId:'S',actorUUID:H,checkId:'external',observedAt:2},{guard:()=>true});
 await assert.rejects(f.ledger.claimExecution('A',permitInput(f.s),{guard:()=>true}),/medic.*unavailable|medic.*uncertain/);assert.equal((await f.ledger.getActivity('A')).executor,undefined);
});

async function settleFailure(f){
 await f.insert();await f.reserve();await f.completing();const permit=await f.ledger.claimExecution('A',permitInput(f.s),{guard:()=>true});f.time(6);
 const result={status:'confirmed',effectiveOutcome:'failure',sourceDegree:1,proof:{useId:'A',checkIds:['C'],resultIds:[],receiptIds:[],immunityIds:['I']},resourceReceiptIds:['medic-bypass:A']};
 await f.ledger.recordExecutionResult('A',{permit,result,witnessGuard:()=>true});await f.ledger.transitionActivity('A',{...f.scope,expected:['completing'],patch:{...result,state:'confirmed'}});await f.ledger.updateSession('S',{status:'paused'});
 return permit;
}

test('an exact acknowledged reference in a fresh session preserves a completed spent baseline',async()=>{
 const f=await fixture();await settleFailure(f);const t=await f.ledger.createSession({...session('T'),startedAt:6,cursorAt:6},{guard:()=>true});
 assert.deepEqual(t.finiteMedicine.medicBypass.baselineByActor[H],f.s.finiteMedicine.medicBypass.baselineByActor[H]);assert.equal((await f.view()).state,'spent');
 await f.ledger.insertActivity({...activity('B',{sessionId:'T',startedAt:6,endsAt:12})},{leaseNonce:t.driver.leaseNonce});
 await assert.rejects(f.ledger.reserveBattleMedicine('B',{leaseNonce:t.driver.leaseNonce,guard:()=>true}),/medic-bypass-unavailable/);
 assert.equal((await f.ledger.getActivity('B')).proof.battleMedicineAttempt,undefined);
});

test('same actor cannot gain another initial baseline or alter an acknowledged baseline',async()=>{
 const f=await fixture();await f.ledger.updateSession('S',{status:'paused'});f.time(6);
 const replacement=patch=>finite({medicBypass:{enabled:true,maxUsesByActor:{[H]:1},baselineByActor:{[H]:baseline(patch)}}});
 await assert.rejects(f.ledger.createSession(session('T',replacement({id:'review-2',checkedAt:6}))),/medic-baseline-predecessor-required/);
 await assert.rejects(f.ledger.createSession(session('T',replacement({remaining:0}))),/medic-baseline-reference-conflict/);
 assert.equal((await f.ledger.getSession('T')),null);assert.equal((await f.view()).remaining,1);
});

test('hourly renewal requires exact predecessor and elapsed window, never automatically refills',async()=>{
 const f=await fixture();await settleFailure(f);f.time(7200);assert.equal((await f.view()).state,'spent');
 const next=at=>finite({medicBypass:{enabled:true,maxUsesByActor:{[H]:1},baselineByActor:{[H]:baseline({id:'review-2',checkedAt:at,previousBaselineId:'review-1',renewal:'hour-window'})}}});
 f.time(3599.5);await assert.rejects(f.ledger.createSession(session('T',next(3599.5)),{guard:()=>true}),/medic-baseline-hour-window-required/);
 f.time(3600);const t=await f.ledger.createSession(session('T',next(3600)),{guard:()=>true});
 assert.equal(t.finiteMedicine.medicBypass.baselineByActor[H].previousBaselineId,'review-1');assert.equal((await f.ledger.finiteMedicineView({actorUUID:H,itemUUID:MEDIC,baselineId:'review-2'})).remaining,1);assert.equal((await f.view()).state,'spent');
});

test('daily renewal requires an explicit GM new-preparation marker, not time passage',async()=>{
 const daily=finite({medicBypass:{enabled:true,maxUsesByActor:{[H]:1},baselineByActor:{[H]:baseline({period:'daily',remaining:0})}}}),f=await fixture({medicine:daily});await f.ledger.updateSession('S',{status:'paused'});f.time(86400);
 assert.equal((await f.view()).state,'spent');
 const medicine=finite({medicBypass:{enabled:true,maxUsesByActor:{[H]:1},baselineByActor:{[H]:baseline({id:'review-2',period:'daily',checkedAt:86400,previousBaselineId:'review-1',renewal:'new-preparation'})}}});
 const t=await f.ledger.createSession(session('T',medicine),{guard:()=>true});assert.equal(t.finiteMedicine.medicBypass.baselineByActor[H].remaining,1);
});

test('a pending old claim prevents renewal even after the hourly window',async()=>{
 const f=await fixture();await f.insert();await f.reserve();await f.ledger.updateSession('S',{status:'paused'});f.time(3600);
 const medicine=finite({medicBypass:{enabled:true,maxUsesByActor:{[H]:1},baselineByActor:{[H]:baseline({id:'review-2',checkedAt:3600,previousBaselineId:'review-1',renewal:'hour-window'})}}});
 await assert.rejects(f.ledger.createSession(session('T',medicine),{guard:()=>true}),/medic-baseline-unknown-no-renewal/);assert.equal((await f.view()).remaining,0);
});

test('an external same-tick check prevents reuse and cannot be cleared by a fresh baseline',async()=>{
 const f=await fixture();await f.ledger.noteExternalBattleMedicineUse({sessionId:'S',actorUUID:H,checkId:'external',observedAt:0},{guard:()=>true});assert.equal((await f.view()).state,'uncertain');
 await f.ledger.updateSession('S',{status:'paused'});f.time(3600);
 const medicine=finite({medicBypass:{enabled:true,maxUsesByActor:{[H]:1},baselineByActor:{[H]:baseline({id:'review-2',checkedAt:3600,previousBaselineId:'review-1',renewal:'hour-window'})}}});
 await assert.rejects(f.ledger.createSession(session('T',medicine),{guard:()=>true}),/medic-baseline-unknown-no-renewal/);
});

test('an exact baseline copied from another root cannot be referenced in this root',async()=>{
 const foreign={...session(),status:'paused',protocol:{version:1,rootUUID:'JournalEntry.OTHER00000000001',epoch:'epoch'}},server=await authorityFixture({sessions:{S:foreign},activities:{},clocks:{}});
 const ledger=createLedger({...server.storage('driver'),isAuthority:()=>true,identity:()=>({userId:'G',clientNonce:'driver'}),now:()=>6});
 await assert.rejects(ledger.createSession(session('T'),{guard:()=>true}),/medic-baseline-scope-mismatch/);
});

test('unknown server acknowledgement keeps the saved reservation occupied across reconstructed clients',async()=>{
 const f=await fixture();await f.insert();f.server.setAcknowledgement(ack=>({...ack,result:[]}));await assert.rejects(f.reserve(),/revision-acknowledgement/);
 f.server.setAcknowledgement(ack=>ack);const reconstructed=f.make();assert.equal((await reconstructed.finiteMedicineView({actorUUID:H,itemUUID:MEDIC,baselineId:'review-1'})).remaining,0);
 const saved=await reconstructed.getActivity('A');await reconstructed.reserveBattleMedicine('A',{...f.scope,guard:()=>true});assert.deepEqual((await reconstructed.getActivity('A')).proof,saved.proof);
});

test('two reconstructed clients cannot grant two native permits for one reserved BM attempt',async()=>{
 const f=await fixture();await f.insert();await f.reserve();await f.completing();const peer=f.make();
 const results=await Promise.allSettled([f.ledger.claimExecution('A',permitInput(f.s),{guard:()=>true}),peer.claimExecution('A',permitInput(f.s,{permitNonce:'peer',attemptNonce:'peer-attempt'}),{guard:()=>true})]);
 assert.equal(results.filter(result=>result.status==='fulfilled').length,1);const a=await f.ledger.getActivity('A');assert.equal(a.proof.medicBypass.permitNonce,a.executor.permitNonce);assert.equal(a.proof.battleMedicineAttempt.permitNonce,a.executor.permitNonce);
});

test('final time changes reject baseline review, reservation and original-check consumption atomically',async()=>{
 const server=await authorityFixture(),store=server.storage('driver');let time=0;
 const ledger=createLedger({...store,transact:(fn,options)=>store.transact(fn,{...options,validateCommit:()=>{time=1;return options.validateCommit()}}),isAuthority:()=>true,identity:()=>({userId:'G',clientNonce:'driver'}),now:()=>time});
 await assert.rejects(ledger.createSession(session()),/medic-baseline-time-changed/);assert.equal((await server.read()).sessions.S,undefined);
 const f=await fixture();await f.insert();f.onFinal(()=>f.time(1));await assert.rejects(f.reserve(),/finite-medicine-time-changed/);assert.equal((await f.ledger.getActivity('A')).proof.battleMedicineAttempt,undefined);
 const g=await fixture();await g.insert();await g.reserve();await g.completing();const permit=await g.ledger.claimExecution('A',permitInput(g.s),{guard:()=>true});g.time(6);g.onFinal(()=>g.time(0));
 await assert.rejects(g.ledger.recordMedicUseWitness('A',{permit,checkId:'C',usedAt:0,guard:()=>true}),/battle-medicine-witness-time-conflict/);assert.equal((await g.ledger.getActivity('A')).proof.medicBypass.state,'claimed');
});

test('BM native grants and known-check consumption reject asynchronous guards',async()=>{
 const f=await fixture();await f.insert();await f.reserve();await f.completing();await assert.rejects(f.ledger.claimExecution('A',permitInput(f.s)),/guard-required/);
 await assert.rejects(f.ledger.claimExecution('A',permitInput(f.s),{guard:()=>Promise.resolve(true)}),/synchronous-evidence-guard-required/);
 const permit=await f.ledger.claimExecution('A',permitInput(f.s),{guard:()=>true});f.time(6);
 await assert.rejects(async()=>f.ledger.recordMedicUseWitness('A',{permit,checkId:'C',usedAt:0,guard:()=>Promise.resolve(true)}),/synchronous-evidence-guard-required/);
 assert.equal((await f.ledger.getActivity('A')).proof.medicBypass.state,'claimed');
});

test('an adjacent preceding clock does not own a later planned BM reservation',async()=>{
 const f=await fixture();await f.insert('A',{startedAt:6,endsAt:12});await f.reserve();await f.ledger.upsertClockCommit({id:'earlier',sessionId:'S',from:0,to:6,gmId:'G',state:'started'},f.scope);
 await f.ledger.cancelBattleMedicineReservation('A',f.scope);assert.equal((await f.view()).remaining,1);
});

test('an atomic clock cannot advance through an unreserved automatic BM activity',async()=>{
 const f=await fixture();await f.insert();await assert.rejects(f.ledger.upsertClockCommit({id:'clock',sessionId:'S',from:0,to:6,gmId:'G',state:'started'},f.scope),/battle-medicine-reservation-required/);
 assert.equal((await f.ledger.getClockCommit('clock')),null);await f.reserve();await f.ledger.upsertClockCommit({id:'clock',sessionId:'S',from:0,to:6,gmId:'G',state:'started'},f.scope);assert.equal((await f.ledger.getClockCommit('clock')).state,'started');
});

test('another GM takeover keeps used allowance and rejects the former driver lease',async()=>{
 const f=await fixture();await settleFailure(f);const successor=f.make('successor','G2');await successor.takeoverSession('S');
 assert.equal((await successor.finiteMedicineView({actorUUID:H,itemUUID:MEDIC,baselineId:'review-1'})).state,'spent');
 const t=await successor.createSession({...session('T'),startedAt:6,cursorAt:6},{guard:()=>true});
 assert.equal(t.driver.userId,'G2');await successor.insertActivity(activity('B',{sessionId:'T',startedAt:6,endsAt:12}),{leaseNonce:t.driver.leaseNonce});
 await assert.rejects(f.ledger.reserveBattleMedicine('B',{leaseNonce:t.driver.leaseNonce,guard:()=>true}),/session-driver-required/);assert.equal((await f.view()).state,'spent');
});

test('resume cannot launder a finite baseline copied from another root or epoch',async()=>{
 for(const protocol of [{version:1,rootUUID:'JournalEntry.OTHER00000000001',epoch:'epoch'},{version:1,rootUUID:'JournalEntry.ROOT000000000001',epoch:'old-epoch'}]){
  const foreign={...session(),status:'paused',protocol},server=await authorityFixture({sessions:{S:foreign},activities:{},clocks:{}}),ledger=createLedger({...server.storage('driver'),isAuthority:()=>true,identity:()=>({userId:'G',clientNonce:'driver'}),now:()=>6});
  await assert.rejects(ledger.resumeSession('S',{cursorAt:0}),/medic-baseline-scope-mismatch/);assert.deepEqual((await server.read()).sessions.S.protocol,protocol);
 }
});

test('an owner DTO from a non-BM provider cannot manufacture protected finite proof',async()=>{
 const f=await fixture();await f.insert('A',{providerId:'lay-on-hands',options:{itemUUID:H+'.Item.LoH'}});await f.completing();const permit=await f.ledger.claimExecution('A',permitInput(f.s,{operationId:'lay-on-hands'}));
 await assert.rejects(f.ledger.recordExecutionResult('A',{permit,result:{status:'uncertain',proof:{useId:'A',checkIds:[],resultIds:[],receiptIds:[],immunityIds:[],medicBypass:{state:'used'}}}}),/finite-medicine-protected-proof/);
 assert.equal(Object.hasOwn((await f.ledger.getActivity('A')).proof,'medicBypass'),false);
});

test('zero-duration BM still needs a reservation before a clock covering its starting instant',async()=>{
 const f=await fixture({medicine:finite({secondsPerUse:0})});await f.insert('A',{endsAt:0});await assert.rejects(f.ledger.upsertClockCommit({id:'clock',sessionId:'S',from:0,to:6,gmId:'G',state:'started'},f.scope),/battle-medicine-reservation-required/);
 await f.reserve();await f.ledger.upsertClockCommit({id:'clock',sessionId:'S',from:0,to:6,gmId:'G',state:'started'},f.scope);await assert.rejects(f.ledger.cancelBattleMedicineReservation('A',f.scope),/battle-medicine-reservation-unsafe-release/);
});
