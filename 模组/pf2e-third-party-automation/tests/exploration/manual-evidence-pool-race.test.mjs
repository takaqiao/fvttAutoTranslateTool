import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createLedger} from '../../scripts/exploration/ledger.mjs';
import {createManualEvents} from '../../scripts/exploration/manual-events.mjs';
import {createPatreonManualImmunity} from '../../scripts/exploration/patreon-manual-immunity.mjs';
import {manualEvidenceFixture,flush,M} from './manual-evidence-fixture.mjs';

function fixture({shared=true}={}){
 const f=manualEvidenceFixture();f.messages.clear();f.game.time.worldTime=0;f.game.system={version:'8.5.1'};
 const binding={invocationId:'INV',messageId:'C',useId:'U',tag:'exploration-manual:U',actorUUID:'Actor.H',patientUUID:'Actor.P',sourceUserId:'HUSER',startedAt:0,recordingSessionId:'S'};
 const descriptor={version:1,providerId:'patreon-v3',providerVersion:'3.2.29',baseSourceSHA256:'89ded325b92fa6b03dcf9257337ae2d628b3e99c987fe22cef9fff4e1837f4e9',pf2eSourceSHA256:'d63da8312831b84905e6866b1dd3f9d93e95c1012955b0177ad2ce8ccf246157'};
 f.game.modules=new Map([['patreon-v3',{active:true,version:'3.2.29',api:{explorationManualImmunity:{descriptor}}}]]);
 f.check={id:'C',author:f.users.get('HUSER'),speaker:{actor:'H'},isCheckRoll:true,isReroll:false,rolls:[{_evaluated:true,total:24}],flags:{[M]:{explorationManualNative:{...binding,riskySurgery:false,continualRecovery:false,patreonImmunity:binding}},pf2e:{modifiers:[],context:{type:'skill-check',origin:{actor:'Actor.H'},target:{actor:'Actor.P'},options:[binding.tag,'action:treat-wounds'],outcome:'success'}}}};
 f.child={id:'D',author:f.users.get('HUSER'),speaker:{actor:'H'},isCheckRoll:false,rolls:[{_evaluated:true,total:17}],flags:{[M]:{explorationManualNative:{useId:'U',tag:binding.tag,patientUUID:null}},pf2e:{origin:{messageId:'C'},context:{type:'skill-check',origin:{actor:'Actor.H'},options:[binding.tag,`${M}:source:D:0`]}}}};
 f.receipt=(id='R')=>({id,author:f.users.get('PUSER'),speaker:{actor:'P'},flags:{pf2e:{context:{type:'damage-taken',options:[`${M}:source:D:0`]},appliedDamage:{uuid:'Actor.P',isHealing:true,isReverted:false}}}});
 f.item={id:'I',uuid:'Actor.P.Item.I',actor:f.patient,parent:f.patient,type:'effect',sourceId:'Compendium.pf2e.feat-effects.Item.Lb4q2bBAgxamtix5',system:{start:{value:0},duration:{value:1,unit:'hours',expiry:'turn-start',sustained:false}},flags:{[M]:{explorationManualPatreonImmunity:{...binding,creatorId:'PUSER'}}}};
 f.patient.items.set('I',f.item);f.messages.set('C',f.check);f.messages.set('D',f.child);
 f.terminal={descriptor,binding,creatorId:'PUSER',itemUUID:f.item.uuid,start:0,duration:structuredClone(f.item.system.duration),expiresAt:3600};
 f.source={version:1,sourceNonce:'observed-source'};
 f.claim={state:'applying',terminal:{receiptId:'R'},selectedPatientUUID:'Actor.P',poolUUID:'Actor.M',request:{resultId:'D'}};
 let state={sessions:{S:{id:'S',manual:true,status:'recording',actorUUIDs:['Actor.H','Actor.P'],activityIds:['manual:C']}},activities:{'manual:C':{id:'manual:C',sessionId:'S',providerId:'manual',kind:'treatment',actorUUID:'Actor.H',patientUUIDs:['Actor.P'],hpPoolUUIDs:[shared?'Actor.M':'Actor.P'],state:'awaiting-evidence',startedAt:0,endsAt:600,durationSeconds:600,source:{manual:true,type:'native-action',messageId:'C',tag:binding.tag},options:{effectiveOutcome:'success',missing:['native-immunity-receipt','native-application-receipt',...shared?['shared-hp-completion-unavailable']:[]]},proof:{useId:'U',checkIds:['C'],resultIds:['D'],receiptIds:[],immunityIds:[]}}},clocks:{}};
 let interleave,commits=0;const ledger=createLedger({read:async()=>structuredClone(state),isAuthority:()=>true,identity:()=>({userId:'G',clientNonce:'issuer'}),transact:async(fn,options)=>{const change=interleave;interleave=null;change?.(state);const next=structuredClone(state),result=fn(next,{rootUUID:'JournalEntry.abcdefghijklmnop',epoch:'E',revision:++commits});options?.validateCommit?.();state=next;return result}});
 f.ledger=ledger;f.options.ledger=ledger;f.options.fromUuid=async uuid=>uuid===f.item.uuid?f.item:f.actors.get(uuid);
 f.options.patreonImmunity=createPatreonManualImmunity({game:f.game,fromUuid:f.options.fromUuid});
 f.options.hpPools={discover:actor=>({ready:true,poolUUID:shared?'Actor.M':actor.uuid,memberUUIDs:shared?['Actor.P','Actor.M']:[actor.uuid]})};
 // The pool claim is a fixture-only already validated private terminal. This
 // does not stand in for receipt trust: the recorder still checks real docs.
 f.options.manualPoolProof={evidence:async activity=>Object.values(activity.proof.poolApplications??{}).some(c=>c.state==='settled'&&activity.proof.receiptIds.includes(c.terminal.receiptId))?{poolUUID:'Actor.M',isCurrent:current=>Object.values(current.proof.poolApplications??{}).some(c=>c.state==='settled'&&current.proof.receiptIds.includes(c.terminal.receiptId))}:null};
 f.recorder=createManualEvents(f.options);f.activity=()=>ledger.getActivity('manual:C');f.setInterleave=fn=>{interleave=fn};f.edit=fn=>fn(state);f.start=async()=>{f.recorder.start();await flush()};
 return f;
}

test('native immunity receipt survives a source registration between validation and commit',async()=>{
 const f=fixture();f.setInterleave(state=>{state.activities['manual:C'].proof.manualPoolSource=f.source});
 await f.recorder.observeNativeImmunity(f.terminal);
 const activity=await f.activity();assert.deepEqual(activity.proof.nativeImmunity,f.terminal);assert.deepEqual(activity.proof.immunityIds,['Actor.P.Item.I']);assert.deepEqual(activity.proof.manualPoolSource,f.source);
});

test('HP receipt survives applying to settled between validation and commit',async()=>{
 const f=fixture();f.edit(s=>{s.activities['manual:C'].proof.manualPoolSource=f.source;s.activities['manual:C'].proof.poolApplications={effect:structuredClone(f.claim)}});await f.start();
 f.setInterleave(s=>{s.activities['manual:C'].proof.poolApplications.effect.state='settled'});
 await f.fire(f.receipt());const a=await f.activity();f.recorder.stop();
 assert.deepEqual(f.errors,[]);assert.deepEqual(a.proof.receiptIds,['R']);assert.equal(a.proof.poolApplications.effect.state,'settled');
});

test('settled notification collects its existing receipt read-only and confirms after native immunity',async()=>{
 const f=fixture();await f.recorder.observeNativeImmunity(f.terminal);
 f.edit(s=>{const a=s.activities['manual:C'];a.proof.manualPoolSource=f.source;a.proof.poolApplications={effect:{...f.claim,state:'settled'}}});
 f.messages.set('R',f.receipt());await f.recorder.refreshPoolEvidence('manual:C');const a=await f.activity();
 assert.deepEqual(a.proof.receiptIds,['R']);assert.deepEqual(a.proof.immunityIds,['Actor.P.Item.I']);assert.equal(a.state,'confirmed');assert.deepEqual(a.options.missing,[]);assert.equal(f.game.time.worldTime,0);
});

for(const change of ['stopped','revoked','deleted-result','foreign-receipt','reverted-receipt'])test('settled collection rejects '+change,async()=>{
 const f=fixture();await f.recorder.observeNativeImmunity(f.terminal);f.edit(s=>{s.activities['manual:C'].proof.poolApplications={effect:{...f.claim,state:'settled'}}});f.messages.set('R',f.receipt());
 if(change==='stopped')f.edit(s=>{s.sessions.S.status='paused'});
 if(change==='revoked')f.patient.testUserPermission=()=>false;
 if(change==='deleted-result')f.messages.delete('D');
 if(change==='foreign-receipt')f.messages.get('R').flags.pf2e.context.options=[`${M}:source:FOREIGN:0`];
 if(change==='reverted-receipt')f.messages.get('R').flags.pf2e.appliedDamage.isReverted=true;
 await f.recorder.refreshPoolEvidence('manual:C').catch(error=>assert.match(error.message,/session-state-conflict|manual-evidence-changed/));
 const a=await f.activity();assert.equal(a.state,'awaiting-evidence');assert.deepEqual(a.proof.receiptIds,[]);
});

test('ordinary nonshared native receipts still confirm without a pool claim',async()=>{
 const f=fixture({shared:false});f.messages.set('R',f.receipt());await f.recorder.observeNativeImmunity(f.terminal);const a=await f.activity();assert.equal(a.state,'confirmed');assert.deepEqual(a.proof.receiptIds,['R']);assert.equal(a.proof.poolApplications,undefined);
});

test('ordinary append cannot write private source or application claims',async()=>{
 const f=fixture(),activity=await f.activity();
 for(const proof of [{manualPoolSource:f.source},{poolApplications:{effect:f.claim}},{executor:{state:'settled'}}])assert.throws(()=>f.ledger.appendManualEvidence(activity.id,{activity,proof,resolveOptions:a=>a.options}),/invalid-ordinary-evidence/);
 assert.equal((await f.activity()).proof.poolApplications,undefined);
});

for(const changed of ['source','patientUUIDs','sessionId','checkpoint','automatic'])test('ordinary append rejects latest '+changed+' identity',async()=>{
 const f=fixture(),activity=await f.activity();f.edit(state=>{const a=state.activities[activity.id];if(changed==='checkpoint')a.temporalSource={type:'checkpoint-reservation'};else if(changed==='automatic')a.source.manual=false;else if(changed==='source')a.source.tag='foreign';else if(changed==='sessionId')a.sessionId='other';else a.patientUUIDs=['Actor.OTHER']});
 await assert.rejects(f.ledger.appendManualEvidence(activity.id,{activity,proof:{receiptIds:['R']},resolveOptions:a=>a.options}),/ordinary-evidence-source-changed/);
 assert.deepEqual((await f.activity()).proof.receiptIds,[]);
});

test('ordinary append does not accept asynchronous final evidence validation',async()=>{
 const f=fixture(),activity=await f.activity();await assert.rejects(f.ledger.appendManualEvidence(activity.id,{activity,proof:{receiptIds:['R']},resolveOptions:async a=>a.options}),/invalid-ordinary-evidence/);assert.deepEqual((await f.activity()).proof.receiptIds,[]);
});

test('ordinary append preserves concurrently added IDs and private state without replacing either',async()=>{
 const f=fixture(),activity=await f.activity();f.edit(s=>{const a=s.activities[activity.id];a.proof.receiptIds=['PRIOR'];a.proof.manualPoolSource=f.source;a.proof.poolApplications={effect:f.claim}});
 await f.ledger.appendManualEvidence(activity.id,{activity,proof:{receiptIds:['R']},resolveOptions:a=>a.options});const a=await f.activity();
 assert.deepEqual(a.proof.receiptIds,['PRIOR','R']);assert.deepEqual(a.proof.manualPoolSource,f.source);assert.deepEqual(a.proof.poolApplications,{effect:f.claim});
});

test('the original generic transition still rejects replacing private claim records',async()=>{
 const f=fixture();f.edit(s=>{s.activities['manual:C'].proof.poolApplications={effect:f.claim}});const activity=await f.activity();
 await assert.rejects(f.ledger.transitionActivity(activity.id,{expected:['awaiting-evidence'],patch:{proof:{...activity.proof,poolApplications:{}}}}),/manual-pool-claim-required/);
});

test('Workbench immunity survives concurrent shared source registration through the same ordinary append',async()=>{
 const f=fixture(),wb=manualEvidenceFixture();f.messages.clear();f.patient.items.clear();
 const check={...wb.check,id:'WC'},result={...wb.result,id:'C'};result.flags[M].explorationManual.checkIds=['WC'];
 f.messages.set('WC',check);f.messages.set('C',result);
 f.edit(s=>{const a=s.activities['manual:C'];a.source={type:'workbench',manual:true,messageId:'C'};a.proof.checkIds=['WC'];a.proof.resultIds=['C']});
 await f.start();const item=wb.immunity();item.actor=f.patient;item.parent=f.patient;item.flags[M].explorationManualImmunity.messageId='C';f.patient.items.set(item.id,item);
 f.setInterleave(s=>{s.activities['manual:C'].proof.manualPoolSource=f.source});
 f.handlers.get('createItem')(item,{},'PUSER');await flush();const a=await f.activity();f.recorder.stop();
 assert.deepEqual(f.errors,[]);assert.deepEqual(a.proof.immunityIds,['Actor.P.Item.I']);assert.deepEqual(a.proof.manualPoolSource,f.source);assert.equal(a.state,'awaiting-evidence');
});
