import {test} from 'node:test';
import assert from 'node:assert/strict';
import {authorityFixture} from './authority-fixture.mjs';
import {createLedger} from '../../scripts/exploration/ledger.mjs';
import {createCoordinator} from '../../scripts/exploration/coordinator.mjs';
import {createClock} from '../../scripts/exploration/clock.mjs';
import {createTreatmentProvider} from '../../scripts/exploration/treatment.mjs';

const session=(id='S',patch={})=>({id,actorUUIDs:['Actor.H'],startedAt:-1200,cursorAt:-1200,budgetEndsAt:6000,activityIds:[],goalsByPool:[{poolUUID:'Actor.H',targetHP:20}],status:'paused',...patch});
function oldClock(patch={}){
 const c={id:'C',sessionId:'S',from:-1200,to:-600,gmId:'G',state:'uncertain',nativeIssued:true,nativeResolved:true,reason:'time-effects-unconfirmed',...patch};
 c.evidence??=[{worldTime:-600,dt:600,userId:'G',options:{pf2eThirdPartyAutomation:{exploration:{checkpointId:'C',sessionId:'S',gmId:'G',expectedFrom:-1200,expectedTo:-600}}}}];return c;
}
const activity=(patch={})=>({id:'A',sessionId:'S',actorUUID:'Actor.H',patientUUIDs:['Actor.H'],hpPoolUUIDs:['Actor.H'],providerId:'treat-wounds',startedAt:-1200,endsAt:-600,state:'confirmed',source:{type:'coordinator'},proof:{},...patch});
const seed=()=>({sessions:{S:session()},activities:{A:activity()},clocks:{C:oldClock()}});
async function fixture(state=seed()){
 const initial=structuredClone(state);for(const row of Object.values(initial.sessions))if(row.status==='running')row.status='paused';
 const server=await authorityFixture(initial);if(Object.values(state.sessions).some(row=>row.status==='running'))await server.storage('seed').transact(next=>{Object.assign(next,structuredClone(state));return true});
 let at=-300,authority=true,beforeCommit=()=>{};
 const client=(nonce='client')=>{
  const store=server.storage(nonce);
  return createLedger({...store,transact:(fn,options)=>store.transact(fn,{validateCommit:()=>{beforeCommit();return options?.validateCommit?.()??true}}),isAuthority:()=>authority,identity:()=>({userId:'G',clientNonce:nonce}),now:()=>at});
 };
 return {server,client,ledger:client(),now:()=>at,setTime:value=>{at=value},setAuthority:value=>{authority=value},beforeCommit:fn=>{beforeCommit=fn}};
}
const accept=(ledger,options={})=>ledger.acceptCurrentTime('S',{expectedWorldTime:-300,note:'  Checked current actors and time.  ',guard:()=>true,...options});

test('current-time acceptance closes a terminal session without rewriting its clocks or activities',async()=>{
 const f=await fixture(),before=await f.server.read(),saved=await accept(f.ledger);
 assert.equal(saved.status,'closed');assert.equal(saved.cursorAt,-1200);
 assert.deepEqual(saved.recoveryReview,{disposition:'accepted-current-state',userId:'G',at:-300,clockIds:['C'],note:'Checked current actors and time.'});
 const after=await f.server.read();assert.deepEqual(after.clocks,before.clocks);assert.deepEqual(after.activities,before.activities);assert.equal(f.now(),-300);
 await assert.rejects(accept(f.ledger),/already-accepted/);
});

test('accepted clock occupancy permits a fresh start, declaration checkpoint, and real clock claim',async()=>{
 const f=await fixture();await accept(f.ledger);
 const fresh=await f.ledger.createSession(session('Next',{status:'running',startedAt:-300,cursorAt:-300})),scope={leaseNonce:fresh.driver.leaseNonce};
 const binding={id:'D',sessionId:'Next',rootUUID:fresh.protocol.rootUUID,epoch:fresh.protocol.epoch,observationNonce:'O',from:-300};
 await f.ledger.openActivityCheckpoint(binding,{...scope,guard:()=>true});
 await f.ledger.enrollCheckpointActivity(binding,{registrationId:'search',checkpointBinding:binding,actorUUID:'Actor.H',label:'Search',durationSeconds:600},{...scope,authenticatedCaller:'G',guard:()=>true});
 await f.ledger.sealActivityCheckpoint(binding,{...scope,registrations:(await f.ledger.getSession('Next')).activityCheckpoint.registrations,guard:()=>true});
 await f.ledger.upsertClockCommit({id:'NextClock',sessionId:'Next',from:-300,to:300,gmId:'G',state:'started',evidence:[]},scope);
 const evidence=[{worldTime:300,dt:600,userId:'G',options:{pf2eThirdPartyAutomation:{exploration:{checkpointId:'NextClock',sessionId:'Next',gmId:'G',expectedFrom:-300,expectedTo:300}}}}];
 await f.ledger.transitionClockCommit('NextClock',{...scope,expected:['started'],patch:{state:'confirmed',nativeResolved:true,effectsSettled:true,evidence}});
 await f.ledger.updateSession('Next',{cursorAt:300},scope);await f.ledger.advanceActivityCheckpoint(binding,{...scope,at:300,guard:()=>true});
 assert.equal((await f.ledger.snapshot('Next')).activities[0].state,'confirmed');
 assert.equal((await f.ledger.getClockCommit('C')).state,'uncertain');
});

for(const patch of [{nativeResolved:false},{evidence:[]},{evidence:[{worldTime:-600,dt:600,userId:'G'}]},{state:'started'},...['explicit-driver-takeover','client-context-lost','legacy-migration-quarantine','time-effects-timeout',undefined].map(reason=>({reason}))])test(`unproven or possibly active clock remains blocking: ${JSON.stringify(patch)}`,async()=>{
 const state=seed();state.clocks.C=oldClock(patch);const f=await fixture(state),before=await f.server.read();
 await assert.rejects(accept(f.ledger),/unresolved-clock/);assert.deepEqual(await f.server.read(),before);
});
for(const field of ['checkpointId','sessionId','gmId','expectedFrom','expectedTo','userId','worldTime','dt'])test(`clock receipt ${field} must match the original native request`,async()=>{
 const state=seed(),r=state.clocks.C.evidence[0];if(field in r)r[field]='wrong';else r.options.pf2eThirdPartyAutomation.exploration[field]='wrong';
 const f=await fixture(state);await assert.rejects(accept(f.ledger),/unresolved-clock/);
});
for(const patch of [
 {state:'planned'},{state:'started'},{state:'completing'},{state:'uncertain'},{state:'awaiting-evidence'},
 {executor:{state:'granted'}},{executor:{state:'uncertain'}},{proof:{poolApplications:{P:{state:'applying'}}}},
 ...['reserved','claimed','uncertain'].flatMap(state=>[{proof:{battleMedicineAttempt:{state}}},{proof:{medicBypass:{state}}}])
])test(`unresolved activity or claim cannot be discarded: ${JSON.stringify(patch)}`,async()=>{
 const state=seed();state.activities.A=activity(patch);const f=await fixture(state),before=await f.server.read();
 await assert.rejects(accept(f.ledger),/unresolved-activity/);assert.deepEqual(await f.server.read(),before);
});
for(const status of ['confirmed','cancelled','blocked'])test(`terminal ${status} activities and settled claims can be accepted`,async()=>{
 const state=seed();state.activities.A=activity({state:status,executor:{state:'settled'},proof:{poolApplications:{P:{state:'settled'}},battleMedicineAttempt:{state:'used'},medicBypass:{state:'released'}}});
 const f=await fixture(state);assert.equal((await accept(f.ledger)).status,'closed');
});
test('a terminal session with only confirmed clocks can close after external time changes',async()=>{
 const state=seed();state.clocks.C.state='confirmed';const f=await fixture(state);
 assert.deepEqual((await accept(f.ledger)).recoveryReview.clockIds,[]);
});
for(const change of ['authority','time-before','time-before-commit','guard-before-commit','running-peer','manual-session','running-session'])test(`acceptance refuses ${change} without writing a revision`,async()=>{
 const state=seed();if(change==='running-peer')state.sessions.Peer=session('Peer',{status:'running'});if(change==='manual-session')state.sessions.S.manual=true;if(change==='running-session')state.sessions.S.status='running';
 const f=await fixture(state);let valid=true;
 if(change==='authority')f.setAuthority(false);if(change==='time-before')f.setTime(-299);if(change==='time-before-commit')f.beforeCommit(()=>f.setTime(-299));if(change==='guard-before-commit')f.beforeCommit(()=>{valid=false});
 const pages=f.server.raw.pages.length;
 await assert.rejects(accept(f.ledger,{guard:()=>valid}),/active-gm|world-time|recovery-state|already-running|automatic-session/);assert.equal(f.server.raw.pages.length,pages);
});
test('only one concurrent GM acceptance can commit the protected review',async()=>{
 const f=await fixture(),results=await Promise.allSettled([accept(f.client('one')),accept(f.client('two'))]);
 assert.equal(results.filter(r=>r.status==='fulfilled').length,1);assert.equal((await f.ledger.getSession('S')).recoveryReview.clockIds.length,1);
});
test('ordinary review and arbitrary session patches cannot create the recovery disposition',async()=>{
 const f=await fixture();await f.ledger.updateSession('S',{status:'closed',review:{disposition:'accepted-current-state',clockIds:['C']}});
 await assert.rejects(f.ledger.createSession(session('Next',{status:'running'})),/unresolved-evidence/);
 await assert.rejects(f.ledger.updateSession('S',{recoveryReview:{disposition:'accepted-current-state',clockIds:['C']}}),/immutable-session/);
 await assert.rejects(f.ledger.createSession(session('Injected',{status:'running',recoveryReview:{disposition:'accepted-current-state',clockIds:['C']}})),/recovery-review-required/);
});

const actor={actorUUID:'Actor.H',medicine:{rank:0},slugs:[],items:[],assuranceSkills:[],pool:{poolUUID:'Actor.H',ready:true},hp:{value:1,max:20},focus:{value:0,max:0},cooldownExpiresAt:0};
function coordinator(f,{clock={advanceTo:()=>assert.fail('acceptance must not advance time')},capabilities={snapshot:async()=>[actor]}}={}){
 return createCoordinator({ledger:f.ledger,capabilities,providers:[],clock,policy:()=>({activities:[],checkpointAt:300}),isAuthority:()=>true,now:f.now,ownerOperations:{}});
}
test('coordinator exposes eligibility, closes only the old session, and starts a fresh round from current state',async()=>{
 const f=await fixture(),c=coordinator(f);assert.equal((await c.snapshot('S')).canAcceptCurrentTime,true);assert.equal((await c.snapshot('S')).worldTime,-300);
 await c.acceptCurrentTime('S',{expectedWorldTime:-300,note:'Reviewed'});assert.equal(c.executionScope('S'),undefined);
 const all=await f.ledger.all();assert.deepEqual(Object.keys(all.sessions),['S']);assert.equal((await c.snapshot('S')).canAcceptCurrentTime,false);
 const fresh=await c.start({id:'Next',actorUUIDs:['Actor.H'],autoRun:false});assert.equal(fresh.startedAt,-300);assert.equal(fresh.cursorAt,-300);assert.equal(fresh.budgetEndsAt,6900);
});
test('a stale confirmation of world time does not close the session',async()=>{
 const f=await fixture(),c=coordinator(f),view=await c.snapshot('S');f.setTime(-299);
 await assert.rejects(c.acceptCurrentTime('S',{expectedWorldTime:view.worldTime,note:'Reviewed'}),/world-time/);assert.equal((await f.ledger.getSession('S')).status,'paused');
});

test('current-time acceptance waits for a local step even after a peer pauses its session',async()=>{
 const f=await fixture({sessions:{},activities:{},clocks:{}});let hold=false,entered,release;
 const ready=new Promise(resolve=>{entered=resolve}),wait=new Promise(resolve=>{release=resolve});
 const c=coordinator(f,{capabilities:{snapshot:async()=>{if(hold){entered();await wait}return [actor]}}});
 await c.start({id:'S',actorUUIDs:['Actor.H'],autoRun:false});hold=true;const pending=c.step('S');await ready;
 await f.ledger.updateSession('S',{status:'paused'});const pages=f.server.raw.pages.length;
 await assert.rejects(c.acceptCurrentTime('S',{expectedWorldTime:-300,note:'Reviewed'}),/session-in-flight/);assert.equal(f.server.raw.pages.length,pages);
 hold=false;release();await pending;assert.equal((await c.snapshot('S')).canAcceptCurrentTime,true);
 assert.equal((await c.acceptCurrentTime('S',{expectedWorldTime:-300,note:'Reviewed'})).status,'closed');
});

for(const boundary of ['receipt','completing-claim'])test(`external time after ${boundary} pauses before treatment and keeps the exact committed cursor`,async()=>{
 const f=await fixture({sessions:{},activities:{},clocks:{}});f.setTime(-1200);let calls=0,completions=0;const hooks=new Set();
 const game={user:{id:'G'},time:{get worldTime(){return f.now()},advance:async(dt,options)=>{calls++;f.setTime(f.now()+dt);for(const fn of hooks)fn(f.now(),dt,options,'G')}}};
 const clock=createClock({game,Hooks:{on:(_name,fn)=>{hooks.add(fn);return fn},off:(_name,fn)=>hooks.delete(fn)},ledger:f.ledger,isAuthority:()=>true,timeEffects:{beforeAdvance:async()=>({status:'ready'}),settle:async()=>({status:'ready'})}});
 const advanceTo=clock.advanceTo;clock.advanceTo=async(...args)=>{const receipt=await advanceTo(...args);if(boundary==='receipt')f.setTime(-500);return receipt};
 const transition=f.ledger.transitionActivity;f.ledger.transitionActivity=async(...args)=>{const result=await transition(...args);if(boundary==='completing-claim'&&args[1].patch.state==='completing')f.setTime(-500);return result};
 const healer={...actor,medicine:{rank:1},wardCapacity:1,modeOfBeing:'living',cooldownExpiresAt:null},capabilities={snapshot:async()=>[healer],discover:async()=>healer},cancelled=new Set();
 const ownerOperations={createActivityContext:async()=>({}),runActivityWithOwner:async()=>{completions++;return {status:'confirmed'}},cancelActivity:async activity=>{cancelled.add(activity.id)}};
 const provider=createTreatmentProvider({ledger:f.ledger,capabilities,ownerOperations,nativeTreatment:{reconcile:()=>{}}});
 const c=createCoordinator({ledger:f.ledger,capabilities,providers:[provider],clock,policy:()=>({activities:[],checkpointAt:-600}),isAuthority:()=>true,now:f.now,ownerOperations});
 await c.start({id:'Next',actorUUIDs:['Actor.H'],autoRun:false});await c.addActivity('Next',{providerId:'treat-wounds',actorUUID:'Actor.H',patientUUIDs:['Actor.H'],hpPoolUUIDs:['Actor.H'],startedAt:-1200,endsAt:-600,options:{}});
 await c.step('Next');const saved=await f.ledger.getSession('Next');assert.equal(saved.status,'paused');assert.equal(saved.stopReason,'external-world-time-change');assert.equal(saved.cursorAt,-600);assert.equal(f.now(),-500);assert.equal(calls,1);assert.equal(completions,0);
 const stopped=(await f.ledger.snapshot('Next')).activities[0];assert.equal(stopped.state,boundary==='completing-claim'?'blocked':'cancelled');assert.equal(stopped.executor,undefined);assert.equal((await c.snapshot('Next')).canAcceptCurrentTime,true);
 assert.equal((await provider.complete(stopped,{})).reason,'activity-begin-context-lost');assert.equal(cancelled.has(stopped.id),true);assert.equal(completions,0);
});
