import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createLedger} from '../../scripts/exploration/ledger.mjs';
import {authorityFixture} from './authority-fixture.mjs';

const empty=()=>({sessions:{},activities:{},clocks:{}});
function server(seed=empty()) {
 let state=structuredClone(seed),revision=0,unknown=false;
 function client(clientNonce,userId='G') {
  const read=async()=>structuredClone(state);
  const transact=async fn=>{
   for(;;){
    const expected=revision,next=await read();
    const result=fn(next,{rootUUID:'JournalEntry.ROOT000000000001',epoch:'epoch',revision:expected+1});
    await Promise.resolve();
    if(revision!==expected)continue;
    state=structuredClone(next);revision++;
    if(unknown)throw Error('acknowledgement-unknown');
    return structuredClone(result);
   }
  };
  return createLedger({read,transact,write:async next=>{state=structuredClone(next)},isAuthority:()=>true,identity:()=>({userId,clientNonce})});
 }
 return {client,read:()=>structuredClone(state),loseAck:()=>{unknown=true}};
}
const session=(id='S',patch={})=>({id,actorUUIDs:['Actor.H','Actor.P'],startedAt:0,cursorAt:0,budgetEndsAt:7200,goalsByPool:[{poolUUID:'Actor.P',targetHP:20}],activityIds:[],status:'running',...patch});
const activity=(id='A',patch={})=>({id,sessionId:'S',providerId:'treat-wounds',actorUUID:'Actor.H',patientUUIDs:['Actor.P'],hpPoolUUIDs:['Actor.P'],startedAt:0,endsAt:600,state:'planned',source:{type:'coordinator'},...patch});
const clock=(id='C',patch={})=>({id,sessionId:'S',from:0,to:600,gmId:'G',state:'started',...patch});
const options=s=>({leaseNonce:s.driver.leaseNonce});
async function begun(f){const ledger=f.client('driver'),s=await ledger.createSession(session());const a=await ledger.insertActivity(activity(),options(s));await ledger.transitionActivity(a.id,{...options(s),expected:['planned'],patch:{state:'started'}});return {ledger,s}}

test('independent clients starting together commit exactly one automatic session',async()=>{
 const f=server(),results=await Promise.allSettled([f.client('one').createSession(session('S1')),f.client('two').createSession(session('S2'))]);
 assert.equal(results.filter(r=>r.status==='fulfilled').length,1);
 assert.equal(Object.values(f.read().sessions).filter(s=>s.status==='running').length,1);
});

test('automatic sessions persist this runtime identity and a fresh continuation lease',async()=>{
 const f=server(),s=await f.client('one').createSession(session());
 assert.equal(s.driver.userId,'G');assert.equal(s.driver.clientNonce,'one');assert.ok(s.driver.leaseNonce);
 assert.deepEqual(s.protocol,{version:1,rootUUID:'JournalEntry.ROOT000000000001',epoch:'epoch'});
});

test('a peer cannot use the saved driver lease as its own local identity',async()=>{
 const f=server(),{ledger,s}=await begun(f),peer=f.client('peer');
 await assert.rejects(peer.transitionActivity('A',{...options(s),expected:['started'],patch:{state:'completing'}}),/session-driver-required/);
 assert.equal((await ledger.getActivity('A')).state,'started');
});

test('driver transitions require the exact private continuation lease',async()=>{
 const f=server(),{ledger}=await begun(f);
 await assert.rejects(ledger.transitionActivity('A',{leaseNonce:'wrong',expected:['started'],patch:{state:'completing'}}),/session-driver-required/);
 await assert.rejects(ledger.transitionActivity('A',{expected:['started'],patch:{state:'completing'}}),/session-driver-required/);
});

test('a stopped session cannot obtain a completing claim',async()=>{
 const f=server(),{ledger,s}=await begun(f);await f.client('peer').updateSession('S',{status:'paused',stopReason:'user-stopped'});
 await assert.rejects(ledger.transitionActivity('A',{...options(s),expected:['started'],patch:{state:'completing'}}),/recovery-session-stopped/);
});

test('unresolved clock domains remain blocked after a review closes their session',async()=>{
 const f=server({sessions:{Old:session('Old',{status:'closed',review:{note:'review'}})},activities:{},clocks:{OldClock:clock('OldClock',{sessionId:'Old',state:'uncertain',review:{note:'review'}})}});
 await assert.rejects(f.client('new').createSession(session()),/unresolved-evidence-no-replay/);
});

test('unresolved actor and patient domains remain blocked after reviewed closure',async()=>{
 for(const patch of [{actorUUID:'Actor.H'},{actorUUID:'Actor.Other',patientUUIDs:['Actor.P']},{actorUUID:'Actor.Other',patientUUIDs:[],hpPoolUUIDs:['Actor.P']}]){
  const f=server({sessions:{Old:session('Old',{status:'closed'})},activities:{OldActivity:activity('OldActivity',{sessionId:'Old',state:'uncertain',review:{note:'review'},...patch})},clocks:{}});
  await assert.rejects(f.client('new').createSession(session()),/unresolved-evidence-no-replay/);
 }
});

test('manual sessions and unrelated evidence survive concurrent automatic mutations',async()=>{
 const f=server(),driver=f.client('driver'),s=await driver.createSession(session());
 await Promise.all([driver.insertActivity(activity(),options(s)),f.client('other').createSession(session('Manual',{manual:true,status:'recording'}))]);
 const state=f.read();assert.equal(state.activities.A.state,'planned');assert.equal(state.sessions.Manual.status,'recording');
});

test('automatic activity insertion checks enrolled actors and overlap inside the transaction',async()=>{
 const f=server(),driver=f.client('driver'),s=await driver.createSession(session());
 await assert.rejects(driver.insertActivity(activity('Wrong',{actorUUID:'Actor.Outsider'}),options(s)),/session-actor-required/);
 await driver.insertActivity(activity(),options(s));
 await assert.rejects(driver.insertActivity(activity('Overlap'),options(s)),/actor-activity-overlap/);
 const next=await driver.insertActivity(activity('Next',{startedAt:600,endsAt:1200}),options(s));assert.equal(next.id,'Next');
});

test('stopped and stale-cursor clocks fail before a server commit',async()=>{
 const f=server(),driver=f.client('driver'),s=await driver.createSession(session());
 await assert.rejects(driver.upsertClockCommit(clock('Stale',{from:20}),options(s)),/world-time-conflict/);
 await driver.updateSession('S',{status:'paused'});
 await assert.rejects(driver.upsertClockCommit(clock(),options(s)),/recovery-session-stopped/);
});

test('two independently constructed ledgers cannot claim overlapping clock checkpoints',async()=>{
 const f=server(),driver=f.client('driver'),s=await driver.createSession(session());
 const results=await Promise.allSettled([driver.upsertClockCommit(clock('One'),options(s)),f.client('driver').upsertClockCommit(clock('Two'),options(s))]);
 assert.equal(results.filter(r=>r.status==='fulfilled').length,1);assert.equal(Object.keys(f.read().clocks).length,1);
});

test('clock commits bind the authenticated driver and atomic protocol revision',async()=>{
 const f=server(),driver=f.client('driver'),s=await driver.createSession(session());
 await assert.rejects(driver.upsertClockCommit(clock('Wrong',{gmId:'Other'}),options(s)),/clock-driver-mismatch/);
 const c=await driver.upsertClockCommit(clock(),options(s));
 assert.equal(c.claim.rootUUID,s.protocol.rootUUID);assert.equal(c.claim.epoch,'epoch');assert.equal(c.claim.leaseNonce,s.driver.leaseNonce);assert.ok(c.claim.revision>0);
});

test('automatic native claims persist one executor before granting a continuation',async()=>{
 const f=server(),{ledger,s}=await begun(f);await ledger.transitionActivity('A',{...options(s),expected:['started'],patch:{state:'completing'}});
 const claim=attemptNonce=>ledger.claimExecution('A',{...options(s),operationId:'treat-wounds',ownerUserId:'G',ownerClientNonce:'owner',attemptNonce,permitNonce:`permit-${attemptNonce}`});
 const results=await Promise.allSettled([claim('one'),claim('two')]);
 assert.equal(results.filter(r=>r.status==='fulfilled').length,1);
 const executor=f.read().activities.A.executor;assert.equal(executor.state,'granted');assert.equal(executor.epoch,'epoch');assert.equal(executor.rootUUID,s.protocol.rootUUID);
});

test('persisted unknown execution never grants another attempt or replay',async()=>{
 const f=server(),{ledger,s}=await begun(f);await ledger.transitionActivity('A',{...options(s),expected:['started'],patch:{state:'completing'}});f.loseAck();let native=0;
 const claim={...options(s),operationId:'treat-wounds',ownerUserId:'G',ownerClientNonce:'owner',attemptNonce:'one',permitNonce:'permit'};
 await assert.rejects(ledger.claimExecution('A',claim).then(()=>{native++}),/acknowledgement-unknown/);
 await assert.rejects(f.client('driver').claimExecution('A',{...claim,attemptNonce:'two'}),/activity-already-executed/);
 assert.equal(native,0);assert.equal(f.read().activities.A.executor.attemptNonce,'one');
});

test('native grant checks operation, owner and stopped session atomically',async()=>{
 const f=server(),{ledger,s}=await begun(f);await ledger.transitionActivity('A',{...options(s),expected:['started'],patch:{state:'completing'}});
 const claim={...options(s),operationId:'treat-wounds',ownerUserId:'G',ownerClientNonce:'owner',attemptNonce:'attempt',permitNonce:'permit'};
 await assert.rejects(ledger.claimExecution('A',{...claim,operationId:'refocus'}),/native-operation-mismatch/);
 await assert.rejects(ledger.claimExecution('A',{...claim,ownerUserId:'Other'}),/native-owner-mismatch/);
 await ledger.updateSession('S',{status:'paused'});await assert.rejects(ledger.claimExecution('A',claim),/recovery-session-stopped/);
});

test('generic patches cannot manufacture a driver, executor or running session',async()=>{
 const f=server(),{ledger,s}=await begun(f);
 await assert.rejects(ledger.updateSession('S',{driver:{...s.driver,clientNonce:'peer'}}),/immutable-session/);
 await assert.rejects(ledger.transitionActivity('A',{...options(s),expected:['started'],patch:{executor:{state:'granted'}}}),/immutable-provenance/);
 await ledger.updateSession('S',{status:'paused'});await assert.rejects(ledger.updateSession('S',{status:'running'}),/explicit-resume-required/);
});

test('explicit resume rotates the driver lease and stale callbacks remain invalid',async()=>{
 const f=server(),driver=f.client('driver'),s=await driver.createSession(session());await driver.updateSession('S',{status:'paused'});
 const resumed=await driver.resumeSession('S',{cursorAt:0});assert.notEqual(resumed.driver.leaseNonce,s.driver.leaseNonce);
 await assert.rejects(driver.insertActivity(activity(),options(s)),/session-driver-required/);
 await driver.insertActivity(activity(),options(resumed));
});

test('explicit takeover quarantines pending work without issuing another native grant',async()=>{
 const f=server(),{ledger}=await begun(f),peer=f.client('peer');
 const paused=await peer.takeoverSession('S');assert.equal(paused.status,'paused');assert.equal((await ledger.getActivity('A')).state,'uncertain');
 await assert.rejects(peer.resumeSession('S',{cursorAt:0}),/unresolved-evidence-no-replay/);
});

test('manual evidence can be appended by a different GM without adopting the driver',async()=>{
 const f=server(),driver=f.client('driver'),s=await driver.createSession(session()),peer=f.client('peer','OtherGM');
 const a=await peer.insertActivity(activity('Manual',{state:'awaiting-evidence',source:{manual:true}}));
 await peer.transitionActivity(a.id,{expected:['awaiting-evidence'],patch:{state:'confirmed'}});
 assert.equal((await driver.getSession('S')).driver.clientNonce,s.driver.clientNonce);
});

test('strict transactional ledgers require a runtime identity at construction',()=>{
 assert.throws(()=>createLedger({read:async()=>empty(),transact:async fn=>fn(empty()),isAuthority:()=>true}),/runtime-identity-required/);
});

test('actual revision storage grants one session across independent GM runtimes',async()=>{
 const f=await authorityFixture(),results=await Promise.allSettled([f.client('one').createSession(session('One')),f.client('two','OtherGM').createSession(session('Two'))]);
 assert.equal(results.filter(r=>r.status==='fulfilled').length,1);assert.equal(Object.keys((await f.read()).sessions).length,1);assert.equal(f.raw.pages.length,2);
});

test('generic mode changes cannot turn an automatic session into a driverless recorder',async()=>{
 for(const patch of [{manual:true},{status:'recording'}]){
  const f=await authorityFixture(),{ledger}=await begun(f),peer=f.client('peer');await ledger.updateSession('S',{status:'paused'});
  await assert.rejects(peer.updateSession('S',patch),/immutable-session|invalid-session-mode/);
  await assert.rejects(peer.transitionActivity('A',{expected:['started'],patch:{state:'completing'}}),/session-driver-required/);
 }
});

test('automatic activity insertion cannot prepopulate completion or executor evidence',async()=>{
 const f=await authorityFixture(),ledger=f.client('driver'),s=await ledger.createSession(session());
 for(const patch of [{state:'completing'},{state:'confirmed'},{executor:{state:'settled'}}])await assert.rejects(ledger.insertActivity(activity('Forged',patch),options(s)),/initial-activity-claim-required/);
 assert.equal(Object.keys((await f.read()).activities).length,0);
});

test('clock creation cannot prepopulate a confirmation or native receipt',async()=>{
 const f=await authorityFixture(),ledger=f.client('driver'),s=await ledger.createSession(session());
 for(const patch of [{state:'confirmed'},{nativeResolved:true},{effectsSettled:true},{evidence:[{worldTime:600}]}])await assert.rejects(ledger.upsertClockCommit(clock('Forged',patch),options(s)),/initial-clock-claim-required/);
 assert.equal(Object.keys((await f.read()).clocks).length,0);
});

test('a review annotation cannot manufacture clock confirmation or release its domain',async()=>{
 const f=await authorityFixture(),ledger=f.client('driver'),s=await ledger.createSession(session());await ledger.upsertClockCommit(clock(),options(s));await ledger.updateSession('S',{status:'paused'});
 const peer=f.client('peer');await assert.rejects(peer.transitionClockCommit('C',{expected:['started'],patch:{state:'confirmed',review:{note:'close'}}}),/session-driver-required|clock-completion-required/);
 await peer.transitionClockCommit('C',{expected:['started'],patch:{review:{note:'unknown result'}}});
 await assert.rejects(peer.resumeSession('S',{cursorAt:0}),/unresolved-evidence-no-replay/);assert.equal((await ledger.getClockCommit('C')).state,'started');
});

test('clock confirmation requires native resolution, settled effects and an exact hook receipt',async()=>{
 const f=await authorityFixture(),ledger=f.client('driver'),s=await ledger.createSession(session());await ledger.upsertClockCommit(clock(),options(s));
 const receipt={worldTime:600,dt:600,userId:'G',options:{pf2eThirdPartyAutomation:{exploration:{sessionId:'S',checkpointId:'C',expectedFrom:0,expectedTo:600,gmId:'G'}}}};
 for(const patch of [{state:'confirmed'},{state:'confirmed',nativeResolved:true,effectsSettled:true,evidence:[{...receipt,userId:'Other'}]},{state:'confirmed',nativeResolved:true,evidence:[receipt]}])await assert.rejects(ledger.transitionClockCommit('C',{...options(s),expected:['started'],patch}),/clock-completion-required/);
 const saved=await ledger.transitionClockCommit('C',{...options(s),expected:['started'],patch:{state:'confirmed',nativeResolved:true,effectsSettled:true,evidence:[receipt]}});assert.equal(saved.state,'confirmed');
});

test('reconciliation cannot roll a session cursor back to an older confirmed checkpoint',async()=>{
 const f=await authorityFixture(),ledger=f.client('driver'),s=await ledger.createSession(session());
 for(const [key,from,to] of [['One',0,600],['Two',600,1200]]){
  await ledger.upsertClockCommit(clock(key,{from,to}),options(s));
  const receipt={worldTime:to,dt:to-from,userId:'G',options:{pf2eThirdPartyAutomation:{exploration:{sessionId:'S',checkpointId:key,expectedFrom:from,expectedTo:to,gmId:'G'}}}};
  await ledger.transitionClockCommit(key,{...options(s),expected:['started'],patch:{state:'confirmed',nativeResolved:true,effectsSettled:true,evidence:[receipt]}});
  await ledger.updateSession('S',{cursorAt:to},options(s));
 }
 await assert.rejects(f.client('peer').updateSession('S',{cursorAt:600},{reconcile:true}),/session-cursor-regression/);assert.equal((await ledger.getSession('S')).cursorAt,1200);
});

test('a native result settles only the exact persisted executor permit',async()=>{
 const f=await authorityFixture(),{ledger,s}=await begun(f);await ledger.transitionActivity('A',{...options(s),expected:['started'],patch:{state:'completing'}});
 const permit=await ledger.claimExecution('A',{...options(s),operationId:'treat-wounds',ownerUserId:'G',ownerClientNonce:'owner',attemptNonce:'attempt',permitNonce:'permit',offerId:'offer',requestId:'request',commandDigest:'digest'});
 const result={status:'confirmed',proof:{useId:'A',checkIds:['Check'],resultIds:['Result'],receiptIds:['Receipt']}};
 for(const field of ['rootUUID','epoch','activityId','operationId','ownerClientNonce','attemptNonce','permitNonce','commandDigest'])await assert.rejects(ledger.recordExecutionResult('A',{permit:{...permit,[field]:'wrong'},result}),/native-permit-mismatch/);
 await assert.rejects(ledger.transitionActivity('A',{...options(s),expected:['completing'],patch:{state:'confirmed'}}),/native-completion-required/);
 await ledger.recordExecutionResult('A',{permit,result});assert.equal((await ledger.getActivity('A')).executor.state,'settled');
 const saved=await ledger.transitionActivity('A',{...options(s),expected:['completing'],patch:{...result,state:'confirmed'}});assert.equal(saved.state,'confirmed');
});

test('a delayed internal pause cannot stop a session under a newer driver lease',async()=>{
 const f=await authorityFixture(),ledger=f.client('driver'),s=await ledger.createSession(session());await ledger.updateSession('S',{status:'paused'});
 const resumed=await ledger.resumeSession('S',{cursorAt:0});
 await assert.rejects(ledger.updateSession('S',{status:'paused',stopReason:'old-step-failed'},options(s)),/session-driver-required/);
 assert.equal((await ledger.getSession('S')).status,'running');assert.equal((await ledger.getSession('S')).driver.leaseNonce,resumed.driver.leaseNonce);
});

test('an internal pause requires the session to remain running inside the transaction',async()=>{
 for(const status of ['paused','closed','complete']){
  const f=await authorityFixture(),ledger=f.client('driver'),s=await ledger.createSession(session());
  await ledger.updateSession('S',{status},status==='complete'?options(s):{});
  const before=await ledger.getSession('S'),pages=f.raw.pages.length;
  await assert.rejects(ledger.updateSession('S',{status:'paused',stopReason:'stale-failure'},{...options(s),expectedStatus:'running'}),/session-state-conflict/);
  assert.deepEqual(await ledger.getSession('S'),before);assert.equal(f.raw.pages.length,pages);
 }
 const f=await authorityFixture(),ledger=f.client('driver'),s=await ledger.createSession(session());
 assert.equal((await ledger.updateSession('S',{status:'paused'},{...options(s),expectedStatus:'running'})).status,'paused');
});

test('clock claims never acquire the manual activity exemption',async()=>{
 const f=await authorityFixture(),ledger=f.client('driver'),s=await ledger.createSession(session());
 await assert.rejects(ledger.upsertClockCommit(clock('Manual',{source:{manual:true}}),options(s)),/initial-clock-claim-required/);
 await ledger.upsertClockCommit(clock(),options(s));
 await assert.rejects(ledger.transitionClockCommit('C',{...options(s),expected:['started'],patch:{source:{manual:true}}}),/immutable-provenance/);
 await assert.rejects(f.client('peer').transitionClockCommit('C',{expected:['started'],patch:{state:'confirmed'}}),/session-driver-required/);
});

test('an exact permit result cannot erase or relabel its uncertain execution domains',async()=>{
 const f=await authorityFixture(),{ledger,s}=await begun(f);await ledger.transitionActivity('A',{...options(s),expected:['started'],patch:{state:'completing'}});
 const permit=await ledger.claimExecution('A',{...options(s),operationId:'treat-wounds',ownerUserId:'G',ownerClientNonce:'owner',attemptNonce:'attempt',permitNonce:'permit'});
 for(const patch of [{patientUUIDs:[]},{hpPoolUUIDs:[]},{groupId:'Other'},{kind:'manual'},{mode:'recording'}])await assert.rejects(ledger.recordExecutionResult('A',{permit,result:{status:'uncertain',...patch}}),/immutable-provenance/);
 const a=await ledger.getActivity('A');assert.deepEqual(a.patientUUIDs,['Actor.P']);assert.deepEqual(a.hpPoolUUIDs,['Actor.P']);
});

test('a proven ordinary treatment failure completes without an HP application receipt',async()=>{
 const f=await authorityFixture(),{ledger,s}=await begun(f);await ledger.transitionActivity('A',{...options(s),expected:['started'],patch:{state:'completing'}});
 const permit=await ledger.claimExecution('A',{...options(s),operationId:'treat-wounds',ownerUserId:'G',ownerClientNonce:'owner',attemptNonce:'attempt',permitNonce:'permit'});
 const result={status:'confirmed',effectiveOutcome:'failure',proof:{useId:'A',checkIds:['Check'],resultIds:[],receiptIds:[],immunityIds:['Immunity']}};
 await assert.rejects(ledger.recordExecutionResult('A',{permit,result:{...result,effectiveOutcome:'criticalFailure'}}),/native-completion-required/);
 await assert.rejects(ledger.recordExecutionResult('A',{permit,result:{...result,effectiveOutcome:'success'}}),/native-completion-required/);
 await ledger.recordExecutionResult('A',{permit,result});assert.equal((await ledger.getActivity('A')).executor.state,'settled');
});

test('risky surgery failure still requires evidence of its damage application',async()=>{
 const f=await authorityFixture(),ledger=f.client('driver'),s=await ledger.createSession(session());await ledger.insertActivity(activity('A',{options:{riskySurgery:true}}),options(s));
 await ledger.transitionActivity('A',{...options(s),expected:['planned'],patch:{state:'started'}});await ledger.transitionActivity('A',{...options(s),expected:['started'],patch:{state:'completing'}});
 const permit=await ledger.claimExecution('A',{...options(s),operationId:'treat-wounds',ownerUserId:'G',ownerClientNonce:'owner',attemptNonce:'attempt',permitNonce:'permit'});
 await assert.rejects(ledger.recordExecutionResult('A',{permit,result:{status:'confirmed',effectiveOutcome:'failure',proof:{useId:'A',checkIds:['Check'],resultIds:[],receiptIds:[]}}}),/native-completion-required/);
});

test('legacy quarantine preserves unknown domains and manual history without creating a driver',async()=>{
 const old=session('Old'),manual=session('Manual',{manual:true,status:'recording'}),paused=session('Paused',{status:'paused'});
 const seed={sessions:{Old:old,Manual:manual,Paused:paused},activities:{
  Planned:activity('Planned',{sessionId:'Old'}),Started:activity('Started',{sessionId:'Old',state:'started',proof:{useId:'Started',checkIds:['Check'],receiptIds:['Receipt']}}),Completing:activity('Completing',{sessionId:'Old',state:'completing'}),
  Unknown:activity('Unknown',{sessionId:'Old',state:'uncertain'}),Manual:activity('Manual',{sessionId:'Manual',state:'awaiting-evidence',source:{manual:true,messageId:'Message'}})
 },clocks:{OldClock:clock('OldClock',{sessionId:'Old'}),UnknownClock:clock('UnknownClock',{sessionId:'Old',state:'uncertain',evidence:[{id:'Receipt'}]})}};
 const f=server(seed),ledger=f.client('migrator');const result=await ledger.quarantineLegacySessions();
 assert.deepEqual(result.quarantinedSessionIds,['Old']);const state=await ledger.all();assert.equal(state.sessions.Old.status,'paused');assert.equal(state.sessions.Old.driver,undefined);assert.equal(state.sessions.Old.protocol,undefined);
 assert.equal(state.activities.Planned.state,'cancelled');assert.equal(state.activities.Started.state,'uncertain');assert.equal(state.activities.Completing.state,'uncertain');
 assert.deepEqual(state.activities.Started.proof,seed.activities.Started.proof);assert.deepEqual(state.activities.Started.patientUUIDs,seed.activities.Started.patientUUIDs);assert.deepEqual(state.activities.Unknown,seed.activities.Unknown);
 assert.equal(state.clocks.OldClock.state,'uncertain');assert.deepEqual(state.clocks.UnknownClock,seed.clocks.UnknownClock);assert.deepEqual(state.sessions.Manual,manual);assert.deepEqual(state.activities.Manual,seed.activities.Manual);assert.deepEqual(state.sessions.Paused,paused);
 await assert.rejects(ledger.resumeSession('Old',{cursorAt:0}),/unresolved-evidence-no-replay/);
 await ledger.updateSession('Old',{status:'closed',review:{note:'checked'}});await assert.rejects(ledger.createSession(session('Next')),/unresolved-evidence-no-replay/);
});

test('stale administrative quarantine cannot pause a current protocol driver',async()=>{
 const f=server({sessions:{Old:session('Old')},activities:{},clocks:{}}),old=f.client('migrator');await old.quarantineLegacySessions();
 const driver=f.client('driver'),resumed=await driver.resumeSession('Old',{cursorAt:0});
 const result=await old.quarantineLegacySessions();assert.deepEqual(result.quarantinedSessionIds,[]);assert.deepEqual(await driver.getSession('Old'),resumed);assert.equal(resumed.status,'running');assert.ok(resumed.driver.leaseNonce);
});

test('administrative quarantine never adopts partial protocol identities',async()=>{
 for(const patch of [{protocol:{version:1}},{driver:{userId:'G',clientNonce:'other',leaseNonce:'lease'}}]){
  const state={sessions:{S:session('S',patch)},activities:{},clocks:{}},ledger=server(state).client('setup');
  assert.deepEqual((await ledger.quarantineLegacySessions()).quarantinedSessionIds,[]);assert.deepEqual(await ledger.all(),state);
 }
});
