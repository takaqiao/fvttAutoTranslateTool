import {test} from 'node:test';
import assert from 'node:assert/strict';
import {authorityFixture} from './authority-fixture.mjs';
import {createCoordinator} from '../../scripts/exploration/coordinator.mjs';
import {createClock} from '../../scripts/exploration/clock.mjs';
import {createManualEvents,WORKBENCH_SOURCE_SHA,registerWorkbenchObservation} from '../../scripts/exploration/manual-events.mjs';
import {createManualRecordBridge} from '../../scripts/exploration/manual-record.mjs';
import {createRefocusProvider} from '../../scripts/exploration/refocus.mjs';
import {chooseNext} from '../../scripts/exploration/policy.mjs';
import {readFile} from 'node:fs/promises';

const M='pf2e-third-party-automation',CD='aa3aa174524021b06e38f9128fd29196ac5f5da863bd818068a9b2fa0e699d20';
const drain=async()=>{for(let i=0;i<8;i++)await new Promise(resolve=>setImmediate(resolve))};
async function fixture({continualRecovery=true,beforeLookup=async()=>{}}={}){
 const store=await authorityFixture(),ledger=store.client('driver'),users=new Map(['G','HUSER','PUSER','OUT'].map(id=>[id,{id,active:true,isGM:id==='G'}]));users.activeGM=users.get('G');
 const actors=new Map(),hooks=new Map(),messages=new Map(),updates=[],counts={clock:0,healing:0,immunity:0,refocus:0,automaticTreatment:0};let serial=0,c,r;
 for(const [id,owner] of [['H','HUSER'],['P','PUSER'],['F','HUSER']])actors.set(`Actor.${id}`,{id,uuid:`Actor.${id}`,name:id,items:new Map(),owner,testUserPermission:u=>u?.isGM===true||u?.id===owner,hp:{value:id==='P'?1:20,max:20},focus:{value:id==='F'?0:1,max:1}});
 const healer=actors.get('Actor.H'),patient=actors.get('Actor.P'),focusActor=actors.get('Actor.F'),token={id:'T',actor:patient},scenes=new Map();scenes.active={tokens:new Map([['T',token]])};
 const fire=(name,...args)=>{for(const h of hooks.values())if(h.name===name)h.fn(...args)};
 const Hooks={on:(name,fn)=>{const id=++serial;hooks.set(id,{name,fn});return id},off:(_name,id)=>hooks.delete(id)};
 const game={user:users.get('G'),users,actors,messages,scenes,time:{worldTime:0,advance:async(dt,options)=>{counts.clock++;game.time.worldTime+=dt;fire('updateWorldTime',game.time.worldTime,dt,options,'G');for(const item of patient.items.values())if(item.system.start.value+item.system.duration.value*60<=game.time.worldTime){item.isExpired=true;item.remainingDuration={expired:true}}}}};
 const fromUuid=async uuid=>{await beforeLookup(uuid);return actors.get(uuid)??[...patient.items.values()].find(item=>item.uuid===uuid)};
 const hpPools={discover:actor=>({poolUUID:actor.uuid,ready:true})};
 const capabilities={discover:async uuid=>(await capabilities.snapshot([uuid]))[0],snapshot:async uuids=>uuids.map(uuid=>{const a=actors.get(uuid);return {actorUUID:uuid,name:a.name,hp:{...a.hp},focus:{...a.focus},pool:hpPools.discover(a),isDead:false,unconscious:false,modeOfBeing:'living',medicine:{rank:0},slugs:[],items:[],assuranceSkills:[],continualRecovery:uuid===healer.uuid&&continualRecovery,refocusUnsupported:[],hasActiveToken:true}})};
 const ownerOperations={createActivityContext:async()=>({}),isActivityContext:()=>true,isExecutionContext:()=>true,cancelActivity:async()=>{}};
 const refocus=createRefocusProvider({game,ledger,capabilities,ownerOperations,refocusEvents:{available:()=>true,complete:async a=>{assert.equal(game.time.worldTime,600);counts.refocus++;const before=focusActor.focus.value;focusActor.focus.value=1;return {id:a.id,focusBefore:before,focusAfter:1}}}});
 const providers=[{id:'treat-wounds',begin:async()=>{counts.automaticTreatment++;return {status:'started'}},complete:()=>assert.fail('manual source must not call automatic treatment')},{...refocus,complete:async(a,ctx)=>{const scope=c.executionScope('S'),permit=await ledger.claimExecution(a.id,{...scope,operationId:'refocus',ownerUserId:'G',ownerClientNonce:'driver',attemptNonce:'attempt',permitNonce:'permit'}),result=await refocus.complete(a,ctx);await ledger.recordExecutionResult(a.id,{permit,result});return result}}];
 const clock=createClock({game,Hooks,ledger,isAuthority:()=>true,timeEffects:{beforeAdvance:async()=>({status:'ready'}),settle:async()=>({status:'ready',proof:[]})},confirmationTimeoutMs:100});
 const bridge=createManualRecordBridge({game,fromUuid,getSession:()=>ledger.getSession('S'),observe:e=>r.observe(e),reserveSource:(binding,intent,callerId)=>c.reserveManualSource(binding,intent,callerId)});
 r=createManualEvents({game,Hooks,ledger,fromUuid,hpPools,isAuthority:()=>true,sessionId:()=> 'S',onChange:value=>updates.push(value),reserveSource:(binding,intent)=>bridge.reserveSource(binding,intent),checkpointOptions:binding=>c.manualCheckpointOptions(binding)});
 c=createCoordinator({game,fromUuid,getHpPool:hpPools.discover,ledger,capabilities,providers,clock,policy:chooseNext,isAuthority:()=>true,now:()=>game.time.worldTime,ownerOperations,manualEvents:r});r.start();
 const config={id:'S',actorUUIDs:[...actors.keys()],budgetSeconds:600,requireFullFocus:true,waitForManualFirstRound:true,autoRun:false};
 const start=()=>c.start(config),reserve=(binding,intent={})=>c.reserveManualSource(binding,{sourceType:'workbench',kind:'treatment',useId:'U',actorUUID:healer.uuid,patientUUID:patient.uuid,...intent},'HUSER');
 async function nativeTreatment(binding,{duration=continualRecovery?10:60,start=0}={}){
  const reservation=await r.beginCheckpointSource({sourceType:'workbench',kind:'treatment',useId:'U',actorUUID:healer.uuid,patientUUID:patient.uuid});assert.deepEqual(reservation.checkpointBinding,binding);
  const check={id:'C',author:users.get('HUSER'),speaker:{actor:'H'},rolls:[{_evaluated:true}],flags:{pf2e:{context:{options:['exploration-manual-use:U']}}}};
  const result={id:'W',author:users.get('HUSER'),speaker:{actor:'H'},rolls:[{_evaluated:true,formula:'{2d8[healing]}'}],flags:{[M]:{explorationManual:{lexicalSource:true,sourceSHA:WORKBENCH_SOURCE_SHA,useId:'U',actorUUID:healer.uuid,patientUUID:patient.uuid,kind:'treatment',continualRecovery,checkIds:['C'],stageIds:[],checkpointReservation:reservation}},treat_wounds_battle_medicine:{id:'T',healerId:'H',dos:2,healing:19}}};
  messages.set('C',check);messages.set('W',result);fire('createChatMessage',result);await drain();
  counts.healing++;patient.hp.value=20;const receipt={id:'R',author:users.get('PUSER'),speaker:{actor:'P'},flags:{pf2e:{context:{type:'damage-taken',options:[`${M}:source:W:0`]},appliedDamage:{uuid:patient.uuid,isHealing:true,isReverted:false}}}};messages.set('R',receipt);fire('createChatMessage',receipt);await drain();
  patient.createEmbeddedDocuments=async(type,data)=>{counts.immunity++;assert.equal(type,'Item');const item={...data[0],id:'I',uuid:'Actor.P.Item.I',actor:patient,parent:patient};patient.items.set('I',item);fire('createItem',item,{},'G');return [item]};
  const observer=r.bindImmunity({message:result,token,kind:'treatment',sourceSHA:CD});await observer.createEmbeddedDocuments('Item',[{type:'effect',system:{start:{value:start,initiative:null},duration:{unit:'minutes',value:duration,expiry:'turn-start',sustained:false}},flags:{core:{sourceId:'Compendium.pf2e.feat-effects.Lb4q2bBAgxamtix5'}}}]);await drain();
  return {reservation,check,result,receipt,item:patient.items.get('I')};
 }
 return {store,ledger,game,c,r,bridge,actors,healer,patient,counts,start,reserve,nativeTreatment,fire,drain,hpPools,updates};
}

test('a first manual window waits at zero with no automatic begin or native time',async()=>{
 const f=await fixture(),s=await f.start();assert.equal((await f.c.step(s.id)).status,'waiting-manual');assert.equal(f.game.time.worldTime,0);assert.deepEqual(f.counts,{clock:0,healing:0,immunity:0,refocus:0,automaticTreatment:0});assert.equal(s.manualCheckpoint.phase,'open');
});
for(const continualRecovery of [true,false])test(`source-bound treatment and ordinary Refocus share one 600-second checkpoint (${continualRecovery?'Continual':'ordinary'} immunity)`,async()=>{
 const f=await fixture({continualRecovery}),s=await f.start(),binding=await f.c.openManualCheckpoint(s.id),native=await f.nativeTreatment(binding);
 const before=await f.ledger.getActivity(native.reservation.activityId);assert.equal(before.state,'awaiting-evidence');assert.deepEqual(before.options.missing,['checkpoint-time-confirmation']);assert.equal(before.startedAt,0);assert.equal(before.endsAt,600);
 const result=await f.c.closeManualCheckpoint(binding,{autoRun:false});assert.equal(result.status,'running',JSON.stringify(result));assert.equal(f.game.time.worldTime,600);assert.deepEqual(f.counts,{clock:1,healing:1,immunity:1,refocus:1,automaticTreatment:0});
 const data=await f.ledger.snapshot(s.id),manual=data.activities.find(a=>a.source.manual),refocus=data.activities.find(a=>a.providerId==='refocus');assert.equal(manual.state,'confirmed');assert.deepEqual(manual.proof.receiptIds,['R']);assert.deepEqual(manual.proof.immunityIds,['Actor.P.Item.I']);assert.deepEqual(manual.options.missing,[]);assert.equal(manual.proof.checkpointImmunity.expiresAt,continualRecovery?600:3600);assert.equal(refocus.startedAt,0);assert.equal(refocus.endsAt,600);assert.equal(refocus.state,'confirmed');assert.equal(data.clocks.length,1);assert.equal(data.clocks[0].id,binding.id);assert.equal(data.session.manualCheckpoint.phase,'settled');
 await assert.rejects(f.c.closeManualCheckpoint(binding,{autoRun:false}),/closed|settled/);assert.equal(f.counts.clock,1);assert.equal((await f.c.step(s.id)).status,'complete');
});
for(const change of ['wrong-start','wrong-duration','unknown-start','unknown-duration','wrong-flag-deadline','expired-early','deleted-check','reverted-HP','duplicate-HP','deleted-immunity','changed-pool'])test(`${change} keeps manual evidence waiting and does not advance`,async()=>{
 const f=await fixture(),s=await f.start(),binding=await f.c.openManualCheckpoint(s.id),native=await f.nativeTreatment(binding);
 if(change==='wrong-start')native.item.system.start.value=1;
 if(change==='wrong-duration')native.item.system.duration.value=11;
 if(change==='unknown-start')delete native.item.system.start;
 if(change==='unknown-duration')native.item.system.duration.value='10';
 if(change==='wrong-flag-deadline')native.item.flags[M].salubriousKiss={expiresAt:1};
 if(change==='expired-early')native.item.isExpired=true;
 if(change==='deleted-check')f.game.messages.delete('C');
 if(change==='reverted-HP')native.receipt.flags.pf2e.appliedDamage.isReverted=true;
 if(change==='duplicate-HP'){const duplicate={...native.receipt,id:'R2'};f.game.messages.set('R2',duplicate);f.fire('createChatMessage',duplicate);await f.drain()}
 if(change==='deleted-immunity')f.patient.items.delete('I');
 if(change==='changed-pool')f.hpPools.discover=actor=>({poolUUID:actor.uuid==='Actor.P'?'Actor.M':actor.uuid,ready:true});
 const result=await f.c.closeManualCheckpoint(binding,{autoRun:false});assert.equal(result.status,'awaiting-evidence');assert.equal(f.counts.clock,0);assert.equal((await f.ledger.getActivity(native.reservation.activityId)).state,'awaiting-evidence');
});
test('a second HP receipt arriving during immunity lookup remains evidence and prevents sealing',async()=>{
 let armed=false,release,entered;const gate=new Promise(resolve=>{release=resolve}),reached=new Promise(resolve=>{entered=resolve});
 const f=await fixture({beforeLookup:async uuid=>{if(armed&&uuid==='Actor.P.Item.I'){armed=false;entered();await gate}}}),s=await f.start(),binding=await f.c.openManualCheckpoint(s.id),native=await f.nativeTreatment(binding);
 armed=true;const pending=f.r.flushCheckpoint(binding).catch(error=>({error:error.message}));await reached;
 const duplicate={...native.receipt,id:'R2'};f.game.messages.set('R2',duplicate);
 const current=await f.ledger.getActivity(native.reservation.activityId);await f.ledger.transitionActivity(current.id,{expected:['awaiting-evidence'],patch:{proof:{...current.proof,receiptIds:[...current.proof.receiptIds,'R2']}},...f.c.executionScope(s.id),checkpointBinding:binding});
 release();const flushed=await pending;assert.equal(flushed.error,'manual-evidence-regression');
 const a=await f.ledger.getActivity(native.reservation.activityId);assert.deepEqual(a.proof.receiptIds,['R','R2'],JSON.stringify(f.updates));
 const result=await f.c.closeManualCheckpoint(binding,{autoRun:false});assert.equal(result.status,'awaiting-evidence');assert.equal(f.counts.clock,0);assert.equal((await f.ledger.getActivity(a.id)).state,'awaiting-evidence');
});
test('the atomic ledger refuses stale proof that drops an observed HP receipt',async()=>{
 const f=await fixture(),s=await f.start(),binding=await f.c.openManualCheckpoint(s.id),native=await f.nativeTreatment(binding),old=await f.ledger.getActivity(native.reservation.activityId);
 const duplicate={...native.receipt,id:'R2'};f.game.messages.set('R2',duplicate);f.fire('createChatMessage',duplicate);await f.r.flushCheckpoint(binding);assert.deepEqual((await f.ledger.getActivity(old.id)).proof.receiptIds,['R','R2']);
 await assert.rejects(f.ledger.transitionActivity(old.id,{expected:['awaiting-evidence'],patch:{proof:old.proof},...f.c.executionScope(s.id),checkpointBinding:binding}),/evidence-regression/);
 assert.deepEqual((await f.ledger.getActivity(old.id)).proof.receiptIds,['R','R2']);assert.equal(f.counts.clock,0);
});
test('only the private driver and a selected actor owner may reserve a source',async()=>{
 const f=await fixture(),s=await f.start(),binding=await f.c.openManualCheckpoint(s.id),peerLedger=f.store.client('peer');
 const peer=createCoordinator({ledger:peerLedger,capabilities:{snapshot:async()=>[]},providers:[],clock:{},policy:chooseNext,isAuthority:()=>true,now:()=>0,ownerOperations:{}});
 await assert.rejects(peer.reserveManualSource(binding,{sourceType:'workbench',kind:'treatment',useId:'U',actorUUID:'Actor.H',patientUUID:'Actor.P'},'HUSER'),/driver/);
 await assert.rejects(f.c.reserveManualSource(binding,{sourceType:'workbench',kind:'treatment',useId:'U',actorUUID:'Actor.H',patientUUID:'Actor.P'},'OUT'),/owner|allowed/);
 await assert.rejects(f.reserve({...binding,observationNonce:'old'}),/checkpoint/);
 const reservation=await f.reserve(binding),a=await f.ledger.getActivity(reservation.activityId);
 await assert.rejects(peerLedger.transitionActivity(a.id,{expected:['awaiting-evidence'],patch:{options:{...a.options,missing:[]}},leaseNonce:s.driver.leaseNonce,checkpointBinding:binding}),/driver/);
 await assert.rejects(f.reserve(binding),/reserved|duplicate/);assert.equal((await f.ledger.snapshot(s.id)).activities.length,1);
});
test('Stop preserves an unknown manual reservation and a new session cannot replay its domain',async()=>{
 const f=await fixture(),s=await f.start(),binding=await f.c.openManualCheckpoint(s.id),reservation=await f.reserve(binding);await f.c.stop(s.id);
 assert.equal((await f.ledger.getActivity(reservation.activityId)).state,'awaiting-evidence');await assert.rejects(f.c.start({id:'next',actorUUIDs:[...f.actors.keys()],autoRun:false}),/unresolved/);assert.equal(f.counts.clock,0);
});
test('external time interrupts an open window without a second native advance',async()=>{
 const f=await fixture(),s=await f.start(),binding=await f.c.openManualCheckpoint(s.id);await f.reserve(binding);f.game.time.worldTime=1;assert.equal((await f.c.step(s.id)).status,'paused');await assert.rejects(f.reserve(binding,{useId:'late'}),/stopped|driver|checkpoint/);assert.equal(f.counts.clock,0);
});
test('reservation socket uses its authenticated caller and rejects supplied execution credentials',async()=>{
 const f=await fixture(),s=await f.start(),binding=await f.c.openManualCheckpoint(s.id),handlers=new Map();f.bridge.register({register:(name,fn)=>handlers.set(name,fn)});
 const handler=handlers.get('exploration:reserveSource');assert.equal(typeof handler,'function');const intent={sourceType:'workbench',kind:'treatment',useId:'U',actorUUID:'Actor.H',patientUUID:'Actor.P'};
 assert.equal((await handler.call({socketdata:{userId:'OUT'}},binding,intent)).ok,false);
 assert.equal((await handler.call({socketdata:{userId:'HUSER'}},binding,{...intent,leaseNonce:s.driver.leaseNonce,proof:{receiptIds:['forged']}})).ok,false);
 const reply=await handler.call({socketdata:{userId:'HUSER'}},binding,intent);assert.equal(reply.ok,true);assert.equal(reply.value.leaseNonce,undefined);assert.equal(reply.value.checkpointBinding.id,binding.id);
});
test('a late duplicate source at 600 cannot turn the completed manual activity into another ten minutes',async()=>{
 const f=await fixture(),s=await f.start(),binding=await f.c.openManualCheckpoint(s.id),native=await f.nativeTreatment(binding);await f.c.closeManualCheckpoint(binding,{autoRun:false});f.fire('createChatMessage',native.result);await f.drain();
 const rows=(await f.ledger.snapshot(s.id)).activities.filter(a=>a.source.manual);assert.equal(rows.length,1);assert.equal(rows[0].state,'confirmed');assert.equal(rows[0].startedAt,0);assert.equal(rows[0].endsAt,600);assert.equal(f.counts.clock,1);
 await f.c.step(s.id);assert.equal(await f.bridge.reserveSource(null,{sourceType:'workbench',kind:'treatment',useId:'fresh',actorUUID:'Actor.H',patientUUID:'Actor.P'}),null);
});
test('normal Continual expiry may remove the item at 600 while retaining its sealed native creation proof',async()=>{
 const f=await fixture(),s=await f.start(),binding=await f.c.openManualCheckpoint(s.id);await f.nativeTreatment(binding);const advance=f.game.time.advance;
 f.game.time.advance=async(...args)=>{await advance(...args);f.patient.items.clear()};await f.c.closeManualCheckpoint(binding,{autoRun:false});const a=(await f.ledger.snapshot(s.id)).activities.find(a=>a.source.manual);assert.equal(a.state,'confirmed');assert.equal(a.proof.checkpointImmunity.expiresAt,600);assert.equal(f.counts.immunity,1);
});
test('an advancing checkpoint refuses a new reservation and a repeated close',async()=>{
 const f=await fixture(),s=await f.start(),binding=await f.c.openManualCheckpoint(s.id);await f.nativeTreatment(binding);let release,entered;
 const gate=new Promise(resolve=>{release=resolve}),reached=new Promise(resolve=>{entered=resolve}),advance=f.game.time.advance;f.game.time.advance=async(...args)=>{entered();await gate;return advance(...args)};
 const pending=f.c.closeManualCheckpoint(binding,{autoRun:false});await reached;await assert.rejects(f.reserve(binding,{useId:'late'}),/closed/);await assert.rejects(f.c.closeManualCheckpoint(binding,{autoRun:false}),/closed|flight/);release();await pending;assert.equal(f.counts.clock,1);
});
test('bound manual rows cannot bypass driver, clock confirmation, or immutable source provenance',async()=>{
 const f=await fixture(),s=await f.start(),binding=await f.c.openManualCheckpoint(s.id),reservation=await f.reserve(binding),a=await f.ledger.getActivity(reservation.activityId),scope=f.c.executionScope(s.id);
 await assert.rejects(f.ledger.transitionActivity(a.id,{expected:['awaiting-evidence'],patch:{proof:{...a.proof,receiptIds:['fake']}}}),/driver/);
 await assert.rejects(f.ledger.transitionActivity(a.id,{expected:['awaiting-evidence'],patch:{proof:{...a.proof,useId:'changed'}},...scope,checkpointBinding:binding}),/immutable/);
 await assert.rejects(f.ledger.transitionActivity(a.id,{expected:['awaiting-evidence'],patch:{state:'confirmed',options:{...a.options,missing:[]}},...scope,checkpointBinding:binding}),/completion/);
 await assert.rejects(f.ledger.insertActivity({...a,id:'unbound',temporalSource:{type:'user-declared'},checkpointBinding:undefined},scope),/reservation-required/);
 await assert.rejects(f.c.addActivity(s.id,{providerId:'refocus',actorUUID:'Actor.F',patientUUIDs:[],hpPoolUUIDs:[],startedAt:0,endsAt:600,options:{}}),/manual-checkpoint-activity/);
});
test('a public start cannot manufacture a prebound checkpoint or leave a running session after an insufficient budget',async()=>{
 const f=await fixture();await assert.rejects(f.ledger.createSession({id:'fake',actorUUIDs:['Actor.H'],startedAt:0,cursorAt:0,budgetEndsAt:600,status:'running',manualCheckpoint:{phase:'open'}}),/checkpoint-open-required/);
 await assert.rejects(f.c.start({id:'short',actorUUIDs:[...f.actors.keys()],budgetSeconds:60,waitForManualFirstRound:true,autoRun:false}),/checkpoint-unavailable/);assert.deepEqual((await f.ledger.all()).sessions,{});assert.equal(f.counts.clock,0);
});
for(const unknown of [false,true])test(`the pinned Workbench target awaits its reservation before the original check (${unknown?'unknown ACK':'confirmed ACK'})`,async()=>{
 const command=await readFile('C:/Users/Taka/Desktop/fvtt/output/automation-native-20260930/treat-wounds-actual-command.txt','utf8'),trace=[];let release;
 const gate=new Promise((resolve,reject)=>{release=()=>unknown?reject(Error('acknowledgement-unknown')):resolve({reservationId:'A',activityId:'A',checkpointBinding:{id:'C'}})}),skill={roll:async()=>{trace.push('native-check')}},healer={uuid:'Actor.H',items:[]},target={actor:{uuid:'Actor.P'}};
 const make=()=>({name:'XDY DO_NOT_IMPORT Treat Wounds and Battle Medicine',command,clone:changes=>({...make(),...changes}),execute:async function(input){const scope=await input.explorationManualTarget({target,bmtw:'Treat Wounds',skillUsed:skill,isRiskySurgery:false,healer},{ChatMessage:{create:async()=>({})},DamageRoll:class{},CheckRoll:class{}});return scope.skill.roll({})}}),pack={getDocuments:async()=>[make()],getDocument:async()=>make()};
 await registerWorkbenchObservation({game:{packs:new Map([['xdy-pf2e-workbench.asymonous-benefactor-macros-internal',pack]])},recorder:{beginCheckpointSource:async intent=>{assert.equal(intent.sourceType,'workbench');trace.push('reserve');return gate}}});
 const macro=await pack.getDocument('M'),pending=macro.execute({});await drain();assert.deepEqual(trace,['reserve']);release();if(unknown)await assert.rejects(pending,/unknown/);else await pending;assert.deepEqual(trace,unknown?['reserve']:['reserve','native-check']);assert.equal(macro.command,command);
});
