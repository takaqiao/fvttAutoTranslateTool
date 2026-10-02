import {test} from 'node:test';
import assert from 'node:assert/strict';
import {authorityFixture} from './authority-fixture.mjs';
import {createCoordinator} from '../../scripts/exploration/coordinator.mjs';
import {createLedger} from '../../scripts/exploration/ledger.mjs';
import {createClock} from '../../scripts/exploration/clock.mjs';
import {createManualRecordBridge} from '../../scripts/exploration/manual-record.mjs';
import {chooseNext} from '../../scripts/exploration/policy.mjs';
import {createRecoveryPanel} from '../../scripts/exploration/panel.mjs';

async function fixture({time=0,budget=1800,maxActivities=1,deficit=false,nativeOwnerByActor,autoRun=false,waitForActivityFirstRound=false,configure}={}){
 const store=await authorityFixture(),storage=store.storage('driver');let boundary=()=>{};const ledger=createLedger({...storage,transact:(fn,options)=>storage.transact((state,context)=>{const value=fn(state,context);queueMicrotask(()=>boundary(state,context));return value},options),isAuthority:()=>true,identity:()=>({userId:'G',clientNonce:'driver'})}),calls={advance:[],begin:[],complete:[],context:0};let serial=0;const hooks=new Map();
 const users=new Map(['G','P'].map(id=>[id,{id,active:true,isGM:id==='G'}]));users.activeGM=users.get('G');
 const actors=new Map(['H','P','A','B'].map(id=>['Actor.'+id,{id,uuid:'Actor.'+id,testUserPermission:u=>u?.active===true,hp:{value:id==='P'&&deficit?1:20,max:20},focus:{value:1,max:1}}]));
 const game={user:users.get('G'),users,actors:new Map([...actors.values()].map(actor=>[actor.id,actor])),time:{worldTime:time,advance:async(dt,options)=>{calls.advance.push(dt);game.time.worldTime+=dt;for(const fn of hooks.values())fn(game.time.worldTime,dt,options,'G')}}};
 const capabilities={snapshot:async uuids=>uuids.map(uuid=>{const a=actors.get(uuid);return {actorUUID:uuid,hp:{...a.hp},focus:{...a.focus},pool:{poolUUID:uuid,ready:true},medicine:{rank:a.rank??0},modeOfBeing:'living',slugs:[],items:[],assuranceSkills:[],refocusUnsupported:[],unsupported:[],isDead:false,unconscious:false}})};
 const timeEffects={beforeAdvance:async()=>({status:'ready'}),settle:async()=>({status:'ready',proof:[]})};
 const Hooks={on:(_,fn)=>{hooks.set(++serial,fn);return serial},off:(_,id)=>hooks.delete(id)},clock=createClock({game,Hooks,ledger,isAuthority:()=>true,confirmationTimeoutMs:30,timeEffects});
 let c;const providers=['treat-wounds','refocus'].map(id=>({id,begin:async a=>{calls.begin.push(a.id);return {status:'started'}},complete:async a=>{calls.complete.push(a.id);const permit=await ledger.claimExecution(a.id,{...c.executionScope('S'),operationId:id,ownerUserId:'G',ownerClientNonce:'driver',attemptNonce:a.id,permitNonce:a.id});const result={status:'confirmed',proof:{useId:a.id,checkIds:[],resultIds:[],receiptIds:['receipt-'+a.id],immunityIds:[]}};await ledger.recordExecutionResult(a.id,{permit,result});for(const uuid of a.patientUUIDs)actors.get(uuid).hp.value=20;if(id==='refocus')actors.get(a.actorUUID).focus.value=1;return result},cancel:async()=>{}}));
 const options={ledger,game,fromUuid:async uuid=>actors.get(uuid),capabilities,providers,clock,policy:chooseNext,isAuthority:()=>true,now:()=>game.time.worldTime,getHpPool:actor=>({ready:true,poolUUID:actor.uuid}),ownerOperations:{createActivityContext:async()=>{calls.context++;return {}},cancelActivity:async()=>{}}};c=createCoordinator(options);
 const handlers=new Map(),bridge=createManualRecordBridge({game,fromUuid:options.fromUuid,getSession:()=>ledger.getSession('S'),checkpointContext:()=>({sessionId:'S',worldTime:game.time.worldTime,...c.executionScope('S')}),enrollCheckpointActivity:(...args)=>ledger.enrollCheckpointActivity(...args),lookupCheckpointActivity:(...args)=>ledger.lookupCheckpointActivity(...args)});bridge.register({register:(name,fn)=>handlers.set(name,fn)});
 const enroll=async(binding,registrationId,actorUUID,durationSeconds,extra={})=>{const response=await handlers.get('exploration:record').call({socketdata:{userId:'P'}},{registrationId,checkpointBinding:binding,actorUUID,label:registrationId,durationSeconds,...extra});assert.equal(response.ok,true,response.error);return response.value};
 const native=async(id,actorUUID,patientUUIDs=[])=>c.addActivity('S',{id,providerId:patientUUIDs.length?'treat-wounds':'refocus',actorUUID,patientUUIDs,hpPoolUUIDs:[...patientUUIDs],startedAt:game.time.worldTime,endsAt:game.time.worldTime+600});
 const f={store,ledger,game,actors,c,clock,calls,enroll,native,options,timeEffects,Hooks,providers,handlers,setBoundary:fn=>boundary=fn};configure?.(f);
 await c.start({id:'S',actorUUIDs:[...actors.keys()],budgetSeconds:budget,maxActivities,nativeOwnerByActor,autoRun,waitForActivityFirstRound});return f;
}

test('an open activity registration window waits without completing goals or advancing',async()=>{
 const f=await fixture(),s=await f.ledger.getSession('S'),binding={id:'C',sessionId:'S',rootUUID:s.protocol.rootUUID,epoch:s.protocol.epoch,observationNonce:'N',from:0};await f.ledger.openActivityCheckpoint(binding,{...f.c.executionScope('S'),guard:()=>true});await f.enroll(binding,'search','Actor.A',900);
 const result=await f.c.step('S');assert.equal(result.status,'waiting-activities');assert.equal((await f.ledger.getSession('S')).status,'running');assert.deepEqual(f.calls.advance,[]);
});
test('future declaration reservation blocks a crossing native action and creates a start boundary',()=>{
 const declaration={id:'D',state:'planned',actorUUID:'Actor.A',patientUUIDs:[],hpPoolUUIDs:[],startedAt:300,endsAt:900,source:{manual:true},temporalSource:{type:'checkpoint-declaration'}};
 const proposal={providerId:'refocus',actorUUID:'Actor.A',patientUUIDs:[],hpPoolUUIDs:[],durationSeconds:600,earliestStart:0};const r=chooseNext({snapshot:{activities:[declaration]},proposals:[proposal],session:{budgetEndsAt:1800},now:0});assert.deepEqual(r.activities,[]);assert.equal(r.checkpointAt,300);
});
const opened=async f=>f.c.openActivityCheckpoint('S');
const declarations=async f=>(await f.ledger.snapshot('S')).activities.filter(a=>a.temporalSource?.type==='checkpoint-declaration');
test('600 native treatment and parallel 900 declaration advance 600 then 300 with no manual provider',async()=>{
 const f=await fixture({deficit:true});await f.native('treat','Actor.H',['Actor.P']);const b=await opened(f);await f.enroll(b,'search','Actor.A',900);await f.c.closeActivityCheckpoint(b,{autoRun:false});
 assert.equal((await f.c.step('S')).status,'running');assert.equal(f.game.time.worldTime,600);assert.deepEqual(f.calls.complete,['treat']);assert.equal((await declarations(f))[0].state,'started');
 assert.equal((await f.c.step('S')).status,'running');assert.equal(f.game.time.worldTime,900);assert.equal((await declarations(f))[0].state,'confirmed');assert.deepEqual(f.calls.advance,[600,300]);assert.equal(f.calls.context,1);assert.equal((await f.c.step('S')).status,'complete');
 const a=(await declarations(f))[0];assert.equal(a.source.type,'user-record');assert.equal(a.temporalSource.type,'checkpoint-declaration');assert.equal(a.executor,undefined);assert.deepEqual(a.proof,{useId:null,checkIds:[],resultIds:[],receiptIds:[],immunityIds:[]});assert.deepEqual(a.patientUUIDs,[]);
});
test('one executor 300 plus 300 and another 600 share 600; order controls only same actor',async()=>{
 const f=await fixture(),b=await opened(f);await f.enroll(b,'second','Actor.A',300,{order:2});await f.enroll(b,'first','Actor.A',300,{order:1});await f.enroll(b,'parallel','Actor.B',600);await f.c.closeActivityCheckpoint(b,{autoRun:false});
 const schedule=Object.fromEntries((await declarations(f)).map(a=>[a.label,[a.startedAt,a.endsAt]]));assert.deepEqual(schedule,{first:[0,300],second:[300,600],parallel:[0,600]});await f.c.step('S');await f.c.step('S');assert.deepEqual(f.calls.advance,[300,300]);assert.ok((await declarations(f)).every(a=>a.state==='confirmed'));assert.equal(f.calls.context,0);
});
test('patient Refocus overlaps receiving treatment, healer search waits and cross actor dependency waits',async()=>{
 const f=await fixture({deficit:true,maxActivities:2});await f.native('treat','Actor.H',['Actor.P']);f.actors.get('Actor.P').focus.value=0;await f.native('refocus','Actor.P');const b=await opened(f);await f.enroll(b,'healer-search','Actor.H',300);await f.enroll(b,'dependent','Actor.B',300,{dependsOn:['treat']});await f.c.closeActivityCheckpoint(b,{autoRun:false});assert.deepEqual((await declarations(f)).map(a=>[a.startedAt,a.endsAt]),[[600,900],[600,900]]);await f.c.step('S');assert.deepEqual(f.calls.complete,['treat','refocus']);assert.equal(f.actors.get('Actor.P').focus.value,1);await f.c.step('S');assert.deepEqual(f.calls.advance,[600,300]);
});
test('zero duration at a negative epoch completes once without advancing; new window rejects old binding',async()=>{
 const f=await fixture({time:-900}),b=await opened(f);await f.enroll(b,'zero','Actor.A',0);await f.c.closeActivityCheckpoint(b,{autoRun:false});assert.equal((await declarations(f))[0].state,'confirmed');assert.deepEqual(f.calls.advance,[]);const next=await opened(f);assert.notEqual(next.id,b.id);await assert.rejects(f.c.closeActivityCheckpoint(b,{autoRun:false}),/checkpoint/);await f.enroll(next,'negative','Actor.B',300);await f.c.closeActivityCheckpoint(next,{autoRun:false});await f.c.step('S');assert.equal(f.game.time.worldTime,-600);assert.deepEqual(f.calls.advance,[300]);await assert.rejects(f.c.closeActivityCheckpoint(next,{autoRun:false}),/checkpoint/);
});
test('budget conflicts leave the window open and all declarations unplanned',async()=>{
 const f=await fixture({budget:600}),b=await opened(f);await f.enroll(b,'long','Actor.A',900);await assert.rejects(f.c.closeActivityCheckpoint(b,{autoRun:false}),/budget/);assert.equal((await f.ledger.getSession('S')).activityCheckpoint.phase,'open');assert.deepEqual(await declarations(f),[]);assert.deepEqual(f.calls.advance,[]);
});
test('goals already met still wait for selected notBefore work and native budget does not consume it',async()=>{
 const f=await fixture({maxActivities:1}),b=await opened(f);await f.enroll(b,'later','Actor.A',300,{notBefore:600});await f.c.closeActivityCheckpoint(b,{autoRun:false});await f.c.step('S');assert.equal(f.game.time.worldTime,600);await f.c.step('S');assert.equal(f.game.time.worldTime,900);assert.equal((await f.c.step('S')).status,'complete');assert.deepEqual(f.calls.begin,[]);
});
for(const reason of ['stop','takeover'])test(`${reason} interrupts started declaration, cancels future and never resumes its work`,async()=>{
 const f=await fixture(),b=await opened(f);await f.enroll(b,'started','Actor.A',900);await f.enroll(b,'future','Actor.A',300);await f.c.closeActivityCheckpoint(b,{autoRun:false});if(reason==='stop')await f.c.stop('S');else await f.store.client('peer').takeoverSession('S');const d=await f.ledger.snapshot('S');assert.equal(d.session.activityCheckpoint.phase,'interrupted');assert.ok(d.activities.every(a=>a.state==='cancelled'));assert.equal(d.session.activityCheckpoint.registrations.started.status,'interrupted');assert.equal(d.session.activityCheckpoint.registrations.future.status,'cancelled');await f.c.resume('S',{autoRun:false});await assert.rejects(f.c.closeActivityCheckpoint(b,{autoRun:false}),/checkpoint/);await f.c.step('S');assert.deepEqual(f.calls.advance,[]);
});
test('reloaded coordinator observes selected declarations without obtaining a driver or reissuing time',async()=>{
 const f=await fixture(),b=await opened(f);await f.enroll(b,'search','Actor.A',900);await f.c.closeActivityCheckpoint(b,{autoRun:false});const peer=createCoordinator({...f.options,ledger:f.store.client('peer')});assert.equal((await peer.step('S')).status,'observing');await peer.restore('S');assert.deepEqual(f.calls.advance,[]);assert.equal((await declarations(f))[0].state,'started');
});
for(const change of ['permission','offline','time','membership'])test(`final seal revision guard rejects ${change} with zero planned activities`,async()=>{
 const f=await fixture(),b=await opened(f);await f.enroll(b,'search','Actor.A',900);const before=f.store.raw.pages.length;
 if(change==='membership')await f.ledger.updateSession('S',{actorUUIDs:['Actor.H','Actor.P','Actor.B']});else f.setBoundary(()=>{if(change==='permission')f.actors.get('Actor.A').testUserPermission=()=>false;if(change==='offline')f.game.users.get('P').active=false;if(change==='time')f.game.time.worldTime++});
 await assert.rejects(f.c.closeActivityCheckpoint(b,{autoRun:false}),/checkpoint-changed|session-actor/);assert.deepEqual(await declarations(f),[]);assert.equal(f.store.raw.pages.length,before+(change==='membership'?1:0));assert.deepEqual(f.calls.advance,[]);
});
test('registration arriving after permission snapshot cannot be silently sealed',async()=>{
 const f=await fixture(),b=await opened(f);await f.enroll(b,'first','Actor.A',300);const seal=f.ledger.sealActivityCheckpoint;f.ledger.sealActivityCheckpoint=async(...args)=>{await f.enroll(b,'late','Actor.B',300);return seal(...args)};await assert.rejects(f.c.closeActivityCheckpoint(b,{autoRun:false}),/registrations-changed/);assert.equal(Object.keys((await f.ledger.getSession('S')).activityCheckpoint.registrations).length,2);assert.deepEqual(await declarations(f),[]);
});
test('Stop during seal await rejects the original binding without adding activities',async()=>{
 const f=await fixture(),b=await opened(f);await f.enroll(b,'search','Actor.A',900);const seal=f.ledger.sealActivityCheckpoint;f.ledger.sealActivityCheckpoint=async(...args)=>{await f.c.stop('S');return seal(...args)};await assert.rejects(f.c.closeActivityCheckpoint(b,{autoRun:false}),/driver|required|stopped/);assert.deepEqual(await declarations(f),[]);assert.deepEqual(f.calls.advance,[]);
});
test('unknown seal ACK retains the saved plan but drops local drive authority and never retries',async()=>{
 const f=await fixture(),b=await opened(f);await f.enroll(b,'search','Actor.A',900);f.store.setAcknowledgement(()=>undefined);const pages=f.store.raw.pages.length;await assert.rejects(f.c.closeActivityCheckpoint(b,{autoRun:false}),/acknowledgement-unknown/);assert.equal(f.store.raw.pages.length,pages+1);assert.equal((await f.ledger.getSession('S')).activityCheckpoint.phase,'sealed');assert.equal((await f.c.step('S')).status,'observing');assert.deepEqual(f.calls.advance,[]);assert.equal(f.store.raw.pages.length,pages+1);
});
test('Stop during clock preflight cancels declarations before any native time call',async()=>{
 const f=await fixture(),b=await opened(f);await f.enroll(b,'search','Actor.A',900);await f.c.closeActivityCheckpoint(b,{autoRun:false});f.timeEffects.beforeAdvance=async()=>{await f.c.stop('S');return {status:'ready'}};await f.c.step('S');assert.deepEqual(f.calls.advance,[]);assert.equal((await declarations(f))[0].state,'cancelled');assert.equal((await f.ledger.snapshot('S')).clocks.length,0);
});
test('unknown native clock never completes a declaration or advances again on reload',async()=>{
 const f=await fixture(),b=await opened(f);await f.enroll(b,'search','Actor.A',900);await f.c.closeActivityCheckpoint(b,{autoRun:false});f.game.time.advance=async dt=>{f.calls.advance.push(dt);f.game.time.worldTime+=dt};await f.c.step('S');const data=await f.ledger.snapshot('S');assert.equal(data.clocks[0].state,'uncertain');assert.notEqual(data.activities[0].state,'confirmed');await assert.rejects(f.c.resume('S',{autoRun:false}),/unresolved/);assert.deepEqual(f.calls.advance,[900]);
});
test('queue cap stays 128 independent from the one-native-activity budget',async()=>{
 const f=await fixture({maxActivities:1}),b=await opened(f);await f.enroll(b,'r0','Actor.A',0);
 const seed=await f.store.read(),checkpoint=seed.sessions.S.activityCheckpoint;
 // A saved, source-normalized queue avoids generating 128 unrelated historical revisions here.
 for(let i=1;i<128;i++)checkpoint.registrations['r'+i]={...structuredClone(checkpoint.registrations.r0),registrationId:'r'+i,registrationOrder:i};
 await f.store.storage('driver').transact(state=>{state.sessions.S.activityCheckpoint=structuredClone(checkpoint);return true});
 const ledger=f.ledger,scope={...f.c.executionScope('S'),guard:()=>true};
 await assert.rejects(ledger.enrollCheckpointActivity(b,{registrationId:'over',checkpointBinding:b,actorUUID:'Actor.A',label:'over',durationSeconds:0},{...scope,authenticatedCaller:'P'}),/registration-limit/);
 await ledger.sealActivityCheckpoint(b,{...scope,registrations:checkpoint.registrations});const data=await ledger.snapshot('S');assert.equal(data.activities.length,128);assert.ok(data.activities.every(a=>a.state==='confirmed'&&a.executor===undefined));assert.equal(data.session.maxActivities,1);assert.deepEqual(data.clocks,[]);
});
test('ledger refuses generic provenance insertion, native permits, direct terminal changes and clock boundary skipping',async()=>{
 const f=await fixture(),b=await opened(f);await f.enroll(b,'first','Actor.A',300);await f.enroll(b,'second','Actor.A',300);await f.c.closeActivityCheckpoint(b,{autoRun:false});const a=(await declarations(f))[0],scope=f.c.executionScope('S');await assert.rejects(f.ledger.transitionActivity(a.id,{expected:['started'],patch:{state:'completing'},...scope}),/lifecycle/);await assert.rejects(f.ledger.insertActivity({...a,id:'forged',state:'planned'},scope),/seal-required/);await assert.rejects(f.ledger.claimExecution(a.id,{...scope}),/state-conflict|forbidden/);await assert.rejects(f.ledger.upsertClockCommit({id:'skip',sessionId:'S',gmId:'G',from:0,to:600,state:'started',evidence:[]},scope),/time-boundary/);assert.deepEqual(f.calls.advance,[]);
});
test('patient identity alone never blocks ordinary Refocus policy',()=>{
 const treatment={state:'started',providerId:'treat-wounds',actorUUID:'Actor.H',patientUUIDs:['Actor.P'],hpPoolUUIDs:['Actor.P'],startedAt:0,endsAt:600};const p={providerId:'refocus',actorUUID:'Actor.P',patientUUIDs:[],hpPoolUUIDs:[],earliestStart:0,durationSeconds:600,patientTreatmentExclusive:false};const selected=chooseNext({snapshot:{activities:[treatment]},proposals:[p],session:{budgetEndsAt:600},now:0});assert.equal(selected.activities.length,1);assert.equal(selected.activities[0].actorUUID,'Actor.P');assert.equal(selected.checkpointAt,600);
});
test('a dependency cancelled after enrollment prevents seal without accepting its planned end as evidence',async()=>{
 const f=await fixture();await f.native('prior','Actor.H');const b=await opened(f);await f.enroll(b,'after','Actor.B',300,{dependsOn:['prior']});await f.ledger.transitionActivity('prior',{expected:['started'],patch:{state:'cancelled'},...f.c.executionScope('S')});await assert.rejects(f.c.closeActivityCheckpoint(b,{autoRun:false}),/invalid-manual-dependency/);assert.deepEqual(await declarations(f),[]);assert.deepEqual(f.calls.advance,[]);
});
test('Stop after a confirmed shorter checkpoint retains elapsed time without claiming declaration completion',async()=>{
 const f=await fixture({deficit:true});await f.native('treat','Actor.H',['Actor.P']);const b=await opened(f);await f.enroll(b,'search','Actor.A',900);await f.c.closeActivityCheckpoint(b,{autoRun:false});await f.c.step('S');await f.c.stop('S');const a=(await declarations(f))[0];assert.equal(a.elapsedSeconds,600);assert.equal(a.state,'cancelled');assert.equal(a.executionResult,undefined);assert.deepEqual(f.calls.advance,[600]);assert.equal((await f.ledger.getSession('S')).cursorAt,600);
});
test('sealing an empty window consumes no time and reopening requires a fresh binding',async()=>{
 const f=await fixture(),b=await opened(f);await f.c.closeActivityCheckpoint(b,{autoRun:false});await assert.rejects(f.ledger.openActivityCheckpoint(b,{...f.c.executionScope('S'),guard:()=>true}),/binding-reused/);const next=await opened(f);assert.notEqual(next.observationNonce,b.observationNonce);assert.deepEqual(f.calls.advance,[]);await assert.rejects(f.enroll(b,'late','Actor.A',300),/actor-not-allowed|checkpoint/);
});
test('a thrown transport error after seal creation also removes local execution authority',async()=>{
 const f=await fixture(),b=await opened(f);await f.enroll(b,'search','Actor.A',900);f.store.setAcknowledgement(()=>{throw Error('transport-disconnected')});await assert.rejects(f.c.closeActivityCheckpoint(b,{autoRun:false}),/transport-disconnected/);assert.equal(f.c.executionScope('S'),undefined);assert.deepEqual(f.calls.advance,[]);
});
test('closing declarations before native selection schedules treatment and patient Refocus together',async()=>{
 const f=await fixture({deficit:true,maxActivities:2});f.actors.get('Actor.H').rank=1;f.actors.get('Actor.P').focus.value=0;await f.ledger.updateSession('S',{requireFullFocus:true});const b=await opened(f);await f.enroll(b,'search','Actor.A',900);await f.c.closeActivityCheckpoint(b,{autoRun:false});assert.deepEqual(f.calls.begin,[]);await f.c.step('S');const at600=await f.ledger.snapshot('S'),native=at600.activities.filter(a=>!a.source.manual);assert.equal(f.game.time.worldTime,600);assert.deepEqual(native.map(a=>[a.providerId,a.actorUUID,a.startedAt,a.endsAt]).sort(),[['refocus','Actor.P',0,600],['treat-wounds','Actor.H',0,600]]);assert.ok(native.every(a=>a.state==='confirmed'));await f.c.step('S');assert.deepEqual(f.calls.advance,[600,300]);assert.equal(f.calls.complete.length,2);assert.equal((await f.c.step('S')).status,'complete');
});

async function selectedDeclaration(options){
 const f=await fixture(options),binding=await opened(f);await f.enroll(binding,'search','Actor.A',900);await f.c.closeActivityCheckpoint(binding,{autoRun:false});return f;
}
const preflightChanges={
 permission:f=>{f.actors.get('Actor.A').testUserPermission=()=>false},
 offline:f=>{f.game.users.get('P').active=false},
 'canonical actor':f=>{f.game.actors.set('A',{...f.actors.get('Actor.A')})},
 'external time':f=>{f.game.time.worldTime=1},
 membership:f=>f.ledger.updateSession('S',{actorUUIDs:['Actor.H','Actor.P','Actor.B']}),
 takeover:f=>f.store.client('peer').takeoverSession('S')
};
for(const [change,apply] of Object.entries(preflightChanges))test(`clock preflight rejects ${change} before advancing a selected declaration`,async()=>{
 const f=await selectedDeclaration();f.timeEffects.beforeAdvance=async()=>{await apply(f);return {status:'ready'}};
 await f.c.step('S');const saved=await f.ledger.snapshot('S');assert.deepEqual(f.calls.advance,[]);assert.equal(f.game.time.worldTime,change==='external time'?1:0);assert.equal(saved.session.cursorAt,0);assert.deepEqual(saved.clocks,[]);assert.ok(saved.activities.every(a=>a.state!=='confirmed'));
});

for(const [change,apply] of Object.entries(preflightChanges).filter(([name])=>['permission','offline','canonical actor','external time'].includes(name)))test(`clock atomic final write rejects ${change} after preparing its revision`,async()=>{
 const f=await selectedDeclaration();let reached=false;
 f.setBoundary(state=>{if(!reached&&Object.keys(state.clocks).length){reached=true;apply(f)}});
 await f.c.step('S');assert.equal(reached,true);assert.deepEqual(f.calls.advance,[]);assert.deepEqual((await f.ledger.snapshot('S')).clocks,[]);assert.equal((await f.ledger.getSession('S')).cursorAt,0);
});

test('permission loss during the post-claim session read preserves an unissued clock without advancing',async()=>{
 const f=await selectedDeclaration(),get=f.ledger.getSession;let changed=false;
 f.ledger.getSession=async id=>{const session=await get(id);if(!changed&&Object.keys((await f.store.read()).clocks).length){changed=true;preflightChanges.permission(f)}return session};
 await f.c.step('S');const saved=await f.ledger.snapshot('S');assert.equal(changed,true);assert.deepEqual(f.calls.advance,[]);assert.equal(saved.clocks.length,1);assert.equal(saved.clocks[0].nativeIssued,false);assert.equal(saved.clocks[0].state,'uncertain');assert.equal(saved.session.cursorAt,0);
});

test('permission loss while installing the native time observer is checked immediately before the call',async()=>{
 const f=await selectedDeclaration(),on=f.Hooks.on;let installed=false;
 f.Hooks.on=(...args)=>{const result=on(...args);installed=true;preflightChanges.permission(f);return result};
 await f.c.step('S');const saved=await f.ledger.snapshot('S');assert.equal(installed,true);assert.deepEqual(f.calls.advance,[]);assert.equal(saved.clocks[0].nativeIssued,false);assert.equal(saved.clocks[0].state,'uncertain');assert.equal(saved.session.cursorAt,0);
});

test('native actor ownership is rechecked in the same clock preflight',async()=>{
 const f=await fixture({deficit:true,nativeOwnerByActor:{'Actor.H':'P'}});await f.native('treat','Actor.H',['Actor.P']);
 f.timeEffects.beforeAdvance=async()=>{f.actors.get('Actor.H').testUserPermission=()=>false;return {status:'ready'}};
 await f.c.step('S');assert.deepEqual(f.calls.advance,[]);assert.deepEqual(f.calls.complete,[]);assert.deepEqual((await f.ledger.snapshot('S')).clocks,[]);
});

test('an already-issued clock retains its exact receipt and settlement after late declaration permission loss',async()=>{
 const f=await selectedDeclaration(),advance=f.game.time.advance;let settled=0;
 f.game.time.advance=async(...args)=>{preflightChanges.permission(f);return advance(...args)};
 f.timeEffects.settle=async()=>{settled++;return {status:'ready',proof:[]}};
 await f.c.step('S');const saved=await f.ledger.snapshot('S'),clock=saved.clocks[0];
 assert.deepEqual(f.calls.advance,[900]);assert.equal(settled,1);assert.equal(f.game.time.worldTime,900);assert.equal(saved.session.cursorAt,900);assert.equal(clock.state,'confirmed');assert.equal(clock.nativeIssued,true);assert.equal(clock.nativeResolved,true);assert.equal(clock.effectsSettled,true);assert.equal(clock.evidence[0].options.pf2eThirdPartyAutomation.exploration.checkpointId,clock.id);assert.equal(saved.activities[0].state,'cancelled');
 await f.c.step('S');const peer=createCoordinator({...f.options,ledger:f.store.client('peer')});await peer.restore('S');assert.deepEqual(f.calls.advance,[900]);
});

const deferred=()=>{let resolve;return {promise:new Promise(yes=>{resolve=yes}),resolve:(...args)=>resolve(...args)}};
async function heldAutomatic({completion=false,maxActivities=1}={}){
 const entered=deferred(),release=deferred(),clockReturned=deferred(),f=await fixture({deficit:true,autoRun:true,maxActivities,configure:f=>{
  f.actors.get('Actor.H').rank=1;
  const advance=f.clock.advanceTo;f.clock.advanceTo=async(...args)=>{try{return await advance(...args)}finally{clockReturned.resolve()}};
  if(completion){const original=f.providers[0].complete;f.providers[0].complete=async(...args)=>{entered.resolve();await release.promise;return original(...args)}}
  else f.timeEffects.beforeAdvance=async()=>{entered.resolve();await release.promise;return {status:'ready'}};
 }});await entered.promise;return {...f,release,clockReturned};
}
for(const completion of [false,true])test(`ordinary autoRun opens only after the original ${completion?'provider':'clock'} and final step lock settle`,async()=>{
 const f=await heldAutomatic({completion}),scope=f.c.executionScope('S');let returned=false;
 const pending=f.c.openActivityCheckpoint('S');pending.then(()=>{returned=true},()=>{});
 const second=f.c.openActivityCheckpoint('S');second.catch(()=>{});await f.ledger.snapshot('S');
 assert.equal(returned,false);assert.equal((await f.ledger.getSession('S')).activityCheckpoint,undefined);assert.equal(f.game.time.worldTime,completion?600:0);
 f.release.resolve();const binding=await pending;assert.deepEqual(await second,binding);
 const saved=await f.ledger.snapshot('S');assert.equal(binding.from,600);assert.equal(saved.session.cursorAt,600);assert.equal(saved.session.activityCheckpoint.phase,'open');assert.deepEqual(f.c.executionScope('S'),scope);assert.equal(saved.clocks[0].state,'confirmed');assert.equal(saved.activities[0].state,'confirmed');assert.deepEqual(f.calls.advance,[600]);assert.equal(f.calls.complete.length,1);
 await f.c.closeActivityCheckpoint(binding,{autoRun:false});assert.equal((await f.c.step('S')).status,'complete');assert.deepEqual(f.calls.advance,[600]);
});
for(const reason of ['stop','takeover','encounter','offline','root-reset'])test(`a pending automatic window cannot survive ${reason}`,async()=>{
 const f=await heldAutomatic(),pending=f.c.openActivityCheckpoint('S');pending.catch(()=>{});
 if(reason==='stop')await f.c.stop('S');if(reason==='takeover')await f.store.client('peer').takeoverSession('S');if(reason==='encounter')f.game.combat={started:true};if(reason==='offline')f.game.user.active=false;if(reason==='root-reset')f.c.invalidate('revision-root-changed');
 f.release.resolve();await assert.rejects(pending);const saved=await f.ledger.snapshot('S');assert.equal(saved.session.activityCheckpoint,undefined);assert.deepEqual(f.calls.advance,[]);
});
test('an unknown original clock rejects pending enrollment without another advance',async()=>{
 const f=await heldAutomatic(),pending=f.c.openActivityCheckpoint('S');pending.catch(()=>{});f.game.time.advance=async dt=>{f.calls.advance.push(dt);f.game.time.worldTime+=dt};f.release.resolve();await assert.rejects(pending);
 const saved=await f.ledger.snapshot('S');assert.equal(saved.session.activityCheckpoint,undefined);assert.equal(saved.clocks[0].state,'uncertain');assert.deepEqual(f.calls.advance,[600]);assert.equal(f.calls.complete.length,0);
});
test('an unknown open ACK preserves its one saved window without recovering drive permission',async()=>{
 const f=await heldAutomatic(),open=f.ledger.openActivityCheckpoint;f.ledger.openActivityCheckpoint=(...args)=>{f.store.setAcknowledgement(()=>undefined);return open(...args)};
 const pending=f.c.openActivityCheckpoint('S');pending.catch(()=>{});f.release.resolve();await assert.rejects(pending,/acknowledgement-unknown/);
 const saved=await f.ledger.snapshot('S'),pages=f.store.raw.pages.length;assert.equal(saved.session.activityCheckpoint.phase,'open');assert.equal(f.c.executionScope('S'),undefined);assert.deepEqual(f.calls.advance,[600]);assert.equal(saved.activities[0].state,'confirmed');assert.equal((await f.c.step('S')).status,'observing');await assert.rejects(f.c.openActivityCheckpoint('S'),/driver-required/);assert.equal(f.store.raw.pages.length,pages);
});
test('the initial generic registration choice opens before ordinary autoRun emits time or native work',async()=>{
 const f=await fixture({deficit:true,autoRun:true,waitForActivityFirstRound:true,configure:f=>{f.actors.get('Actor.H').rank=1}}),session=await f.ledger.getSession('S');assert.equal(session.activityCheckpoint.phase,'open');assert.equal(session.activityCheckpoint.from,0);assert.deepEqual(f.calls.advance,[]);assert.deepEqual(f.calls.begin,[]);assert.equal(typeof f.c.executionScope('S').leaseNonce,'string');
});
test('a late old runner cannot cancel the new lease checkpoint request after Stop and resume',async()=>{
 const f=await heldAutomatic({maxActivities:2}),old=f.c.openActivityCheckpoint('S');old.catch(()=>{});await f.c.stop('S');await assert.rejects(old);const entered=deferred(),release=deferred();f.timeEffects.beforeAdvance=async()=>{entered.resolve();await release.promise;return {status:'ready'}};
 await f.c.resume('S');await entered.promise;const pending=f.c.openActivityCheckpoint('S');pending.catch(()=>{});f.release.resolve();await f.clockReturned.promise;await f.ledger.snapshot('S');release.resolve();const binding=await pending;assert.equal(binding.from,600);assert.equal((await f.ledger.getSession('S')).activityCheckpoint.phase,'open');assert.deepEqual(f.calls.advance,[600]);
});

const activityForm={actor:'Actor.A',label:'Search',duration:'10',unit:'60',durationSource:'user-declared',durationDetail:'',notBefore:'',order:'',dependsOn:[]};
function ownerDeclarationPanel(f){
 const counts={records:0,queries:0,lookups:0},owner=createManualRecordBridge({game:{...f.game,user:f.game.users.get('P')}});
 owner.register({register:()=>{},executeAsGM:async(name,...args)=>f.handlers.get(name).call({socketdata:{userId:'P'}},...args)});
 const panel=createRecoveryPanel({game:{user:f.game.users.get('P')},getActivityCheckpoint:uuid=>{counts.queries++;return owner.getActivityCheckpoint(uuid)},record:event=>{counts.records++;return owner.record(event)},lookupCheckpointActivity:(...args)=>{counts.lookups++;return owner.lookupCheckpointActivity(...args)}});
 return {panel,counts};
}
function declarationDialog(t,wait){const previous=globalThis.foundry;t.after(()=>{globalThis.foundry=previous});globalThis.foundry={applications:{api:{DialogV2:{wait}}}}}
test('OWNER declaration rejection before enrollment permits a new window after Stop and resume',async t=>{
 const f=await fixture(),old=await opened(f),{panel,counts}=ownerDeclarationPanel(f);let entered,release;const ready=new Promise(resolve=>entered=resolve);
 declarationDialog(t,()=>{entered();return new Promise(resolve=>release=resolve)});const pending=panel.openActivityDeclaration('Actor.A');await ready;
 await f.c.stop('S');await f.c.resume('S',{autoRun:false});const current=await opened(f);assert.notEqual(current.id,old.id);release({...activityForm});
 await assert.rejects(pending,/manual-actor-not-allowed|recovery-session-stopped|activity-checkpoint/);const pages=f.store.raw.pages.length;
 globalThis.foundry.applications.api.DialogV2.wait=async()=>({...activityForm});const saved=await panel.openActivityDeclaration('Actor.A');
 assert.equal(saved.checkpointBinding.id,current.id);assert.deepEqual(counts,{queries:2,records:2,lookups:0});assert.equal(f.store.raw.pages.length,pages+1);
});
test('OWNER declaration with unknown acknowledgement is looked up after time completes without another record',async t=>{
 const f=await fixture(),binding=await opened(f),{panel,counts}=ownerDeclarationPanel(f);declarationDialog(t,async()=>({...activityForm}));
 f.store.setAcknowledgement(()=>undefined);await assert.rejects(panel.openActivityDeclaration('Actor.A'),/acknowledgement-unknown/);
 const saved=Object.values((await f.ledger.getSession('S')).activityCheckpoint.registrations)[0];assert.ok(saved);f.store.setAcknowledgement(ack=>ack);
 await f.c.closeActivityCheckpoint(binding,{autoRun:false});await f.c.step('S');assert.equal(f.game.time.worldTime,600);
 const pages=f.store.raw.pages.length,result=await panel.openActivityDeclaration('Actor.A');assert.equal(result.registrationId,saved.registrationId);assert.equal(result.status,'completed');
 assert.deepEqual(counts,{queries:1,records:1,lookups:1});assert.equal(f.store.raw.pages.length,pages);assert.deepEqual(f.calls.advance,[600]);
});
test('a caller rejection flag is rejected before enrollment and cannot forge a definitive result',async()=>{
 const f=await fixture(),binding=await opened(f),owner=createManualRecordBridge({game:{...f.game,user:f.game.users.get('P')}});
 owner.register({register:()=>{},executeAsGM:async(name,...args)=>f.handlers.get(name).call({socketdata:{userId:'P'}},...args)});f.store.setAcknowledgement(()=>undefined);
 await assert.rejects(owner.record({registrationId:'saved',checkpointBinding:binding,actorUUID:'Actor.A',label:'Search',durationSeconds:600,declarationRejected:true}),/invalid-checkpoint-declaration/);
 assert.deepEqual((await f.ledger.getSession('S')).activityCheckpoint.registrations,{});
});
for(const drift of ['OWNER','offline','canonical','root','source-user'])test(`completed registration readback rejects ${drift} drift without a new record`,async()=>{
 const f=await fixture(),binding=await opened(f);await f.enroll(binding,'saved','Actor.A',600);await f.c.closeActivityCheckpoint(binding,{autoRun:false});await f.c.step('S');
 const input={...binding};if(drift==='OWNER')f.actors.get('Actor.A').testUserPermission=()=>false;if(drift==='offline')f.game.users.get('P').active=false;if(drift==='canonical')f.game.actors.set('A',{...f.actors.get('Actor.A')});if(drift==='root')input.rootUUID='JournalEntry.OTHER';
 const pages=f.store.raw.pages.length,response=await f.handlers.get('exploration:lookupCheckpointActivity').call({socketdata:{userId:drift==='source-user'?'G':'P'}},input,'saved','Actor.A');
 assert.equal(response.ok,false);assert.equal(f.store.raw.pages.length,pages);assert.deepEqual(f.calls.advance,[600]);
});
test('OWNER resolves an unknown archived registration before entering the next window without replay',async t=>{
 const f=await fixture(),old=await opened(f),{panel,counts}=ownerDeclarationPanel(f);declarationDialog(t,async()=>({...activityForm}));f.store.setAcknowledgement(()=>undefined);
 await assert.rejects(panel.openActivityDeclaration('Actor.A'),/acknowledgement-unknown/);const registered=Object.values((await f.ledger.getSession('S')).activityCheckpoint.registrations)[0];f.store.setAcknowledgement(ack=>ack);
 await f.c.closeActivityCheckpoint(old,{autoRun:false});await f.c.step('S');const next=await opened(f),session=await f.ledger.getSession('S');assert.notEqual(next.id,old.id);assert.equal(session.activityCheckpointHistory[0].registrations[registered.registrationId].status,'completed');
 const pages=f.store.raw.pages.length,saved=await panel.openActivityDeclaration('Actor.A');assert.equal(saved.registrationId,registered.registrationId);assert.deepEqual(saved.checkpointBinding,old);assert.equal(saved.status,'completed');assert.equal(f.store.raw.pages.length,pages);assert.deepEqual(counts,{queries:1,records:1,lookups:1});
 const fresh=await panel.openActivityDeclaration('Actor.A');assert.deepEqual(fresh.checkpointBinding,next);assert.notEqual(fresh.registrationId,saved.registrationId);assert.equal(f.store.raw.pages.length,pages+1);assert.deepEqual(counts,{queries:2,records:2,lookups:1});assert.deepEqual(f.calls.advance,[600]);
});
