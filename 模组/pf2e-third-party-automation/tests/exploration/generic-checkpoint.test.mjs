import {test} from 'node:test';
import assert from 'node:assert/strict';
import {authorityFixture} from './authority-fixture.mjs';
import {createManualRecordBridge} from '../../scripts/exploration/manual-record.mjs';
import {createLedger} from '../../scripts/exploration/ledger.mjs';
import {promptActivityDeclaration} from '../../scripts/exploration/panel.mjs';

async function fixture({beforeOpen}={}){
 const store=await authorityFixture(),ledger=store.client('driver');
 const session=await ledger.createSession({id:'S',actorUUIDs:['Actor.A','Actor.B'],startedAt:-600,cursorAt:-600,budgetEndsAt:1200});
 const binding={id:'C',sessionId:'S',rootUUID:store.rootUUID,epoch:'epoch',observationNonce:'observation',from:-600};
 const context={sessionId:'S',worldTime:-600,leaseNonce:session.driver.leaseNonce};
 await beforeOpen?.({ledger,session});
 await ledger.openActivityCheckpoint(binding,{leaseNonce:context.leaseNonce,guard:()=>true});
 const users=new Map([['G',{id:'G',isGM:true,active:true}],['P',{id:'P',active:true}]]);users.activeGM=users.get('G');
 const game={user:users.get('G'),users},actor={uuid:'Actor.A',testUserPermission:user=>user?.id==='P'};
 const handlers=new Map();let resolveActor=async()=>actor;
 const makeBridge=(writer=ledger)=>{
  const bridge=createManualRecordBridge({game,fromUuid:uuid=>resolveActor(uuid),getSession:()=>writer.getSession(context.sessionId),checkpointContext:()=>({...context}),
   enrollCheckpointActivity:(...args)=>writer.enrollCheckpointActivity(...args),lookupCheckpointActivity:(...args)=>writer.lookupCheckpointActivity(...args),observe:()=>{throw Error('recording-route-used')}});
  const local=new Map();bridge.register({register:(name,fn)=>local.set(name,fn)});return local;
 };
 for(const [key,value] of makeBridge())handlers.set(key,value);
 const input=()=>({registrationId:'registration',checkpointBinding:{...binding},actorUUID:'Actor.A',label:'搜索',durationSeconds:900,durationSource:{type:'user-declared',detail:'约定时间'},dependsOn:[]});
 const send=(value=input(),map=handlers)=>map.get('exploration:record').call({socketdata:{userId:'P'}},value);
 return {store,ledger,binding,context,game,actor,users,handlers,makeBridge,input,send,setResolve:fn=>{resolveActor=fn}};
}

test('authenticated running declaration reaches the atomic ledger without treatment credentials',async()=>{
 const f=await fixture(),result=await f.send();assert.equal(result.ok,true,result.error);
 const saved=await f.ledger.getSession('S'),rows=Object.values(saved.activityCheckpoint.registrations);
 assert.equal(rows.length,1);assert.equal(rows[0].source.userId,'P');assert.equal(rows[0].temporalSource.type,'checkpoint-declaration');
 assert.equal(rows[0].declaration.durationSeconds,900);assert.deepEqual(rows[0].checkpointBinding,f.binding);
 assert.deepEqual(saved.activityIds,[]);const state=await f.store.read();assert.deepEqual(state.activities,{});assert.deepEqual(state.clocks,{});
 assert.equal(rows[0].executor,undefined);assert.equal(rows[0].proof,undefined);
});

test('two authenticated bridge instances enroll the same registration only once',async()=>{
 const f=await fixture(),other=f.makeBridge(f.store.client('driver'));
 const results=await Promise.all([f.send(),f.send(f.input(),other)]);
 assert.ok(results.every(r=>r.ok));assert.deepEqual(results[0].value,results[1].value);
 assert.equal(Object.keys((await f.ledger.getSession('S')).activityCheckpoint.registrations).length,1);
 const changed=await f.send({...f.input(),durationSeconds:600});assert.equal(changed.ok,false);assert.match(changed.error,/registration-conflict/);
 const second=await f.send({...f.input(),registrationId:'independent'});assert.equal(second.ok,true);
 assert.equal(Object.keys((await f.ledger.getSession('S')).activityCheckpoint.registrations).length,2);
});

test('registration ownership is authenticated and cannot be changed by a second OWNER',async()=>{
 const f=await fixture();f.actor.testUserPermission=()=>true;assert.equal((await f.send()).ok,true);
 const response=await f.handlers.get('exploration:record').call({socketdata:{userId:'G'}},f.input());
 assert.equal(response.ok,false);assert.match(response.error,/registration-conflict/);
 const missing=await f.handlers.get('exploration:record').call({},f.input());assert.equal(missing.ok,false);
});

for(const field of ['source','authenticatedCaller','proof','patientUUIDs','hpPoolUUIDs','executor','observedStart','observedEnd','sessionId'])test(`checkpoint rejects public ${field} before reading an actor`,async()=>{
 const f=await fixture();let reads=0;f.setResolve(async()=>{reads++;return f.actor});const before=f.store.raw.pages.length;
 const result=await f.send({...f.input(),[field]:'forged'});assert.equal(result.ok,false);assert.equal(reads,0);assert.equal(f.store.raw.pages.length,before);
});

test('normalization clones every accepted field before the first await',async()=>{
 const f=await fixture();let release,entered;const ready=new Promise(resolve=>entered=resolve);
 f.setResolve(()=>{entered();return new Promise(resolve=>release=()=>resolve(f.actor))});
 const input=f.input(),pending=f.send(input);await ready;input.durationSeconds=1;input.durationSource.detail='mutated';input.checkpointBinding.id='foreign';input.dependsOn.push('foreign');release();
 const result=await pending;assert.equal(result.ok,true,result.error);assert.equal(result.value.declaration.durationSeconds,900);assert.equal(result.value.declaration.durationSource.detail,'约定时间');assert.deepEqual(result.value.declaration.dependsOn,[]);
});

for(const mutation of ['stop','revoke','offline','gm-change','session','time','lease','membership'])test(`awaited actor resolution followed by ${mutation} creates no registration`,async()=>{
 const f=await fixture();let release,entered;const ready=new Promise(resolve=>entered=resolve);
 f.setResolve(()=>{entered();return new Promise(resolve=>release=()=>resolve(f.actor))});const pending=f.send();await ready;
 if(mutation==='stop')await f.ledger.updateSession('S',{status:'paused'});
 if(mutation==='revoke')f.actor.testUserPermission=()=>false;
 if(mutation==='offline')f.users.get('P').active=false;
 if(mutation==='gm-change')f.game.users.activeGM={id:'other'};
 if(mutation==='session')f.context.sessionId='another';
 if(mutation==='time')f.context.worldTime++;
 if(mutation==='lease')f.context.leaseNonce='another';
 if(mutation==='membership')await f.ledger.updateSession('S',{actorUUIDs:['Actor.B']});
 const before=f.store.raw.pages.length;release();const result=await pending;assert.equal(result.ok,false);assert.equal(f.store.raw.pages.length,before);
 assert.deepEqual((await f.ledger.getSession('S')).activityCheckpoint.registrations,{});
});

for(const mutation of ['revoke','time','session'])test(`final transaction guard rejects ${mutation} during revision serialization`,async()=>{
 const f=await fixture(),storage=f.store.storage('driver');
 const writer=createLedger({...storage,transact:(fn,options)=>storage.transact((state,context)=>{
  const result=fn(state,context);queueMicrotask(()=>{if(mutation==='revoke')f.actor.testUserPermission=()=>false;if(mutation==='time')f.context.worldTime++;if(mutation==='session')f.context.sessionId='other'});return result;
 },options),isAuthority:()=>true,identity:()=>({userId:'G',clientNonce:'driver'})});
 const before=f.store.raw.pages.length,result=await f.send(f.input(),f.makeBridge(writer));assert.equal(result.ok,false);assert.equal(f.store.raw.pages.length,before);
});

test('unknown commit acknowledgement permits only authenticated readback, without another revision or grant',async()=>{
 const f=await fixture();f.store.setAcknowledgement(()=>undefined);const before=f.store.raw.pages.length,result=await f.send();
 assert.equal(result.ok,false);assert.match(result.error,/acknowledgement-unknown/);assert.equal(f.store.raw.pages.length,before+1);
 const lookup=f.handlers.get('exploration:lookupCheckpointActivity'),saved=await lookup.call({socketdata:{userId:'P'}},f.binding,'registration','Actor.A');
 assert.equal(saved.ok,true);assert.equal(saved.value.status,'registered');assert.equal(saved.value.permit,undefined);assert.equal(saved.value.executor,undefined);assert.equal(f.store.raw.pages.length,before+1);
 f.actor.testUserPermission=()=>true;const foreign=await lookup.call({socketdata:{userId:'G'}},f.binding,'registration','Actor.A');assert.equal(foreign.ok,false);
 assert.deepEqual((await f.store.read()).clocks,{});
});

test('lost local lease permits only the known registration readback, not a new window or enrollment',async()=>{
 const f=await fixture();assert.equal((await f.send()).ok,true);delete f.context.leaseNonce;
 const before=await f.store.read(),pages=f.store.raw.pages.length;
 const saved=await f.handlers.get('exploration:lookupCheckpointActivity').call({socketdata:{userId:'P'}},f.binding,'registration','Actor.A');
 assert.equal(saved.ok,true,saved.error);assert.equal(saved.value.status,'registered');assert.equal(saved.value.permit,undefined);assert.equal(saved.value.executor,undefined);
 const window=await f.handlers.get('exploration:activityCheckpoint').call({socketdata:{userId:'P'}},'Actor.A');assert.equal(window.ok,false);assert.match(window.error,/session-driver-required/);
 assert.equal((await f.send({...f.input(),registrationId:'late'})).ok,false);assert.equal(f.store.raw.pages.length,pages);assert.deepEqual(await f.store.read(),before);
});

test('a form opened before Stop cannot enroll or switch to a later binding',async t=>{
 const f=await fixture(),previous=globalThis.foundry;let entered,release;const ready=new Promise(resolve=>entered=resolve);
 t.after(()=>{globalThis.foundry=previous});globalThis.foundry={applications:{api:{DialogV2:{wait:()=>{entered();return new Promise(resolve=>release=resolve)}}}}};
 const window={binding:{...f.binding},phase:'open'};let submissions=0;
 const pending=promptActivityDeclaration({actors:[{actorUUID:'Actor.A',name:'A'}],window,record:async value=>{submissions++;assert.deepEqual(value.checkpointBinding,f.binding);const result=await f.send(value);if(!result.ok)throw Error(result.error);return result.value}});
 await ready;await f.ledger.updateSession('S',{status:'paused'});window.binding.id='later-window';window.binding.from=0;const pages=f.store.raw.pages.length;
 release({actor:'Actor.A',label:'Search',duration:'5',unit:'60',durationSource:'user-declared',durationDetail:'',notBefore:'',order:'',dependsOn:[]});
 await assert.rejects(pending,/recovery-session-stopped/);assert.equal(submissions,1);assert.equal(f.store.raw.pages.length,pages);assert.deepEqual((await f.ledger.getSession('S')).activityCheckpoint.registrations,{});assert.deepEqual((await f.store.read()).clocks,{});
});

test('checkpoint declaration supports zero duration and negative epoch, but not retroactive or invalid values',async()=>{
 const f=await fixture();const zero=await f.send({...f.input(),durationSeconds:0,notBefore:-600});assert.equal(zero.ok,true);assert.equal(zero.value.temporalSource.registeredAt,-600);
 for(const patch of [{notBefore:-601},{durationSeconds:-1},{durationSeconds:Infinity},{durationSeconds:NaN},{order:-1},{durationSource:{type:'observed'}},{durationSource:{type:'user-declared',sourceUUID:'fake'}}])assert.equal((await f.send({...f.input(),registrationId:'bad',...patch})).ok,false);
 assert.equal(Object.keys((await f.ledger.getSession('S')).activityCheckpoint.registrations).length,1);
});

test('dependencies must be existing usable activities in the same session',async()=>{
 const f=await fixture({beforeOpen:({ledger,session})=>ledger.insertActivity({id:'dependency',sessionId:'S',actorUUID:'Actor.B',providerId:'refocus',patientUUIDs:[],hpPoolUUIDs:[],startedAt:-600,endsAt:0,state:'planned'},{leaseNonce:session.driver.leaseNonce})});
 assert.equal((await f.send({...f.input(),dependsOn:['dependency']})).ok,true);
 const unknown=await f.send({...f.input(),registrationId:'unknown',dependsOn:['not-yet-created']});assert.equal(unknown.ok,false);
 await f.ledger.transitionActivity('dependency',{expected:['planned'],patch:{state:'cancelled'}});
 assert.equal((await f.send({...f.input(),registrationId:'cancelled',dependsOn:['dependency']})).ok,false);
});

test('open registration window cannot admit native work, start existing work, complete the session or advance time',async()=>{
 const activity={id:'existing',sessionId:'S',actorUUID:'Actor.B',providerId:'refocus',patientUUIDs:[],hpPoolUUIDs:[],startedAt:-600,endsAt:0,state:'planned'};
 const f=await fixture({beforeOpen:({ledger,session})=>ledger.insertActivity(activity,{leaseNonce:session.driver.leaseNonce})}),options={leaseNonce:f.context.leaseNonce};
 const before=f.store.raw.pages.length;
 await assert.rejects(f.ledger.insertActivity({...activity,id:'new',actorUUID:'Actor.A'},options),/activity-checkpoint-open/);
 await assert.rejects(f.ledger.transitionActivity('existing',{...options,expected:['planned'],patch:{state:'started'}}),/activity-checkpoint-open/);
 await assert.rejects(f.ledger.upsertClockCommit({id:'clock',sessionId:'S',gmId:'G',from:-600,to:0,state:'started'},options),/activity-checkpoint-open/);
 await assert.rejects(f.ledger.updateSession('S',{status:'complete'},options),/activity-checkpoint-open/);
 assert.equal(f.store.raw.pages.length,before);
});

test('checkpoint authority is private and separate from session patching and WB600',async()=>{
 const f=await fixture();
 await assert.rejects(f.ledger.updateSession('S',{activityCheckpoint:{...f.binding,phase:'open',registrations:{}}}),/immutable-session/);
 await assert.rejects(f.ledger.updateSession('S',{manualCheckpoint:{...f.binding,to:0,phase:'open'}},{leaseNonce:f.context.leaseNonce}),/activity-checkpoint-active/);
 await assert.rejects(f.ledger.createSession({id:'forged',startedAt:0,budgetEndsAt:600,activityCheckpoint:f.binding}),/activity-checkpoint-open-required/);
 const other=f.makeBridge(f.store.client('other-tab'));assert.equal((await f.send(f.input(),other)).ok,false);
 await f.ledger.updateSession('S',{status:'paused'});assert.equal((await f.ledger.getSession('S')).activityCheckpoint.phase,'interrupted');assert.equal((await f.send()).ok,false);
});

test('takeover preserves registrations and rejects the old checkpoint after a new driver resumes',async()=>{
 const f=await fixture();assert.equal((await f.send()).ok,true);
 const registered=(await f.ledger.getSession('S')).activityCheckpoint.registrations;
 const successor=f.store.client('other-tab'),paused=await successor.takeoverSession('S');
 assert.equal(paused.status,'paused');assert.equal(paused.activityCheckpoint.phase,'interrupted');
 assert.deepEqual(paused.activityCheckpoint.registrations,registered);
 const resumed=await successor.resumeSession('S',{cursorAt:f.context.worldTime});
 f.context.leaseNonce=resumed.driver.leaseNonce;
 const pages=f.store.raw.pages.length,result=await f.send({...f.input(),registrationId:'late-registration'},f.makeBridge(successor));
 assert.equal(result.ok,false);assert.equal(result.error,'activity-checkpoint-closed');
 const after=await successor.getSession('S');
 assert.equal(after.status,'running');assert.equal(after.activityCheckpoint.phase,'interrupted');
 assert.deepEqual(after.activityCheckpoint.registrations,registered);assert.equal(f.store.raw.pages.length,pages);
});

test('a player submits and looks up only through the authenticated socket, never the private ledger',async()=>{
 const f=await fixture(),playerGame={...f.game,user:f.users.get('P')};let socketCalls=0;
 const player=createManualRecordBridge({game:playerGame,fromUuid:()=>{throw Error('player-private-read')},getSession:()=>{throw Error('player-private-read')}});
 player.register({register:()=>{},executeAsGM:async(name,...args)=>{socketCalls++;return f.handlers.get(name).call({socketdata:{userId:'P'}},...args)}});
 const registered=await player.record(f.input()),before=f.store.raw.pages.length;
 assert.deepEqual(await player.lookupCheckpointActivity(f.binding,'registration','Actor.A'),registered);
 assert.equal(socketCalls,2);assert.equal(f.store.raw.pages.length,before);
});

test('malformed binding and accessor payloads do not invoke their getters or read actors',async()=>{
 const f=await fixture();let reads=0,getters=0;f.setResolve(async()=>{reads++;return f.actor});
 const accessor=f.input();Object.defineProperty(accessor,'durationSeconds',{enumerable:true,get(){getters++;return 900}});
 const nested=f.input();Object.defineProperty(nested.durationSource,'detail',{enumerable:true,get(){getters++;return 'fake'}});
 const sparse=f.input();sparse.dependsOn=Array(1);
 for(const value of [accessor,nested,sparse,{...f.input(),checkpointBinding:{...f.binding,to:0}},{...f.input(),checkpointBinding:{...f.binding,epoch:'foreign'}},{...f.input(),registrationId:'__proto__'}])assert.equal((await f.send(value)).ok,false);
 assert.equal(getters,0);assert.equal(reads,1);assert.deepEqual((await f.ledger.getSession('S')).activityCheckpoint.registrations,{});
});

test('a rejected or asynchronous private final guard does not append a revision',async()=>{
 const f=await fixture(),before=f.store.raw.pages.length;
 for(const guard of [undefined,()=>false,async()=>true])assert.throws(()=>f.ledger.enrollCheckpointActivity(f.binding,f.input(),{authenticatedCaller:'P',leaseNonce:f.context.leaseNonce,guard}),/checkpoint/);
 assert.equal(f.store.raw.pages.length,before);
});

test('a checkpoint cannot open over unknown native time or execution, or an existing WB window',async()=>{
 for(const mode of ['clock','executor','workbench']){
  const store=await authorityFixture(),ledger=store.client('driver'),session=await ledger.createSession({id:'S',actorUUIDs:['Actor.A'],startedAt:0,cursorAt:0,budgetEndsAt:1200}),options={leaseNonce:session.driver.leaseNonce};
  const binding={id:'C',sessionId:'S',rootUUID:store.rootUUID,epoch:'epoch',observationNonce:'observation',from:0};
  if(mode==='clock')await ledger.upsertClockCommit({id:'clock',sessionId:'S',gmId:'G',from:0,to:600,state:'started'},options);
  if(mode==='executor'){
   await ledger.insertActivity({id:'activity',sessionId:'S',providerId:'refocus',actorUUID:'Actor.A',patientUUIDs:[],hpPoolUUIDs:[],startedAt:0,endsAt:600,state:'planned'},options);
   await ledger.transitionActivity('activity',{...options,expected:['planned'],patch:{state:'started'}});
   await ledger.transitionActivity('activity',{...options,expected:['started'],patch:{state:'completing'}});
  }
  if(mode==='workbench')await ledger.updateSession('S',{manualCheckpoint:{...binding,to:600,phase:'open'}},options);
  const before=store.raw.pages.length;await assert.rejects(ledger.openActivityCheckpoint(binding,{...options,guard:()=>true}),/unresolved-evidence|manual-checkpoint-active/);assert.equal(store.raw.pages.length,before);
 }
});
