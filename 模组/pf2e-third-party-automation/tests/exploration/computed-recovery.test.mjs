import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createComputedRecovery,computedSummary} from '../../scripts/exploration/computed-recovery.mjs';
import {createComputedDocuments} from '../../scripts/exploration/computed-documents.mjs';
import {createClock} from '../../scripts/exploration/clock.mjs';
import {createRecoverySimulation} from '../../scripts/exploration/computed-plan.mjs';
import {authorityFixture} from './authority-fixture.mjs';

async function fixture({passive=false,writeFailure=false,onPause,simulationFactory:customFactory}={}){
 const store=await authorityFixture(),ledger=store.client('one'),order=[],listeners=new Map();let serial=0,settled=false,samples=0;
 const actor={id:'A',uuid:'Actor.A',name:'Patient',isOwner:true,items:[],flags:{},hasCondition:()=>false,system:{attributes:{hp:{value:1,max:20,temp:0}},resources:{focus:{value:0,max:0}}}},
  game={user:{id:'G',active:true},users:{activeGM:{id:'G'}},time:{worldTime:0},modules:new Map(),actors:new Map([['A',actor]])};
 actor.update=async changes=>{order.push('hp');for(const [path,value]of Object.entries(changes)){const keys=path.split('.'),last=keys.pop();let object=actor;for(const key of keys)object=object[key]??={};object[last]=structuredClone(value)}if(writeFailure)throw Error('lost-ack');return actor};
 const Hooks={on:(name,fn)=>{const id=++serial;listeners.set(id,{name,fn});return id},off:(name,id)=>listeners.delete(id)};
 game.time.advance=async(delta,options)=>{order.push('time');game.time.worldTime+=delta;if(passive)actor.system.attributes.hp.value+=5;for(const {name,fn}of listeners.values())if(name==='updateWorldTime')fn(game.time.worldTime,delta,options,'G')};
 const clock=createClock({game,Hooks,ledger,isAuthority:()=>true,timeEffects:{beforeAdvance:async()=>({status:'ready'}),settle:async()=>{order.push('effects');return {status:'ready',proof:[]}}}});
 const capabilities={snapshot:async()=>[{actorUUID:actor.uuid,name:actor.name,pool:{ready:true,poolUUID:actor.uuid},hp:structuredClone(actor.system.attributes.hp),focus:{value:0,max:0},wounded:false}]};
 const simulationFactory=customFactory??(({actors,session,state})=>{
  let current=structuredClone(actors),at=session.cursorAt,done=false,history=[];
  return {next:()=>done?{done:true,reason:'goals-met'}:{from:at,to:at+600,activities:[]},settle:({actors:fresh}={})=>{samples++;order.push('sample');const beforeActors=structuredClone(fresh??current);current=structuredClone(beforeActors);current[0].hp.value=Math.min(20,current[0].hp.value+14);const from=at;at+=600;done=true;const event={providerId:'focus-healing',actorUUID:'Actor.A',patientUUID:'Actor.A',poolUUID:'Actor.A',healing:14,startedAt:from,endsAt:at};history.push(event);return {from,to:at,beforeActors,afterActors:structuredClone(current),events:[event]}},checkpoint:()=>({cursorAt:at,actors:[],active:[],history:[],activityCount:done?1:0}),snapshot:()=>({cursorAt:at,actors:structuredClone(current),active:[],history,activityCount:done?1:0,stopReason:done?'goals-met':null})};
 });
 const documents=createComputedDocuments({game,fromUuid:async()=>actor});
 const runtime=createComputedRecovery({game,ledger,clock,capabilities,documents,prepareActors:capabilities.snapshot,pauseGame:async()=>{order.push('pause');onPause?.(actor)},isAuthority:()=>true,simulationFactory,timeBoundary:async()=>({segmented:passive,until:Infinity}),onChange:()=>{}});
 return {runtime,ledger,game,actor,documents,order,samples:()=>samples};
}

test('stable recovery persists real samples once and applies one result without native action messages',async()=>{
 const f=await fixture();const session=await f.runtime.start({actorUUIDs:['Actor.A'],goalsByPool:[{poolUUID:'Actor.A',targetHP:15}],autoRun:false});await f.runtime.drive(session.id);
 const saved=await f.ledger.getSession(session.id);assert.equal(saved.status,'complete');assert.equal(f.actor.system.attributes.hp.value,15);assert.equal(f.game.time.worldTime,600);
 assert.equal(f.samples(),1);assert.equal(saved.computedRecovery.batches.length,1);assert.equal(saved.computedRecovery.batches[0].state,'confirmed');
 assert.equal(f.order.filter(event=>event==='pause').length,1);assert.equal(f.order.filter(event=>event==='hp').length,1);
});

test('time effects settle before sampling and healing uses the refreshed HP',async()=>{
 const f=await fixture({passive:true});const session=await f.runtime.start({actorUUIDs:['Actor.A'],autoRun:false});await f.runtime.drive(session.id);
 assert.equal(f.actor.system.attributes.hp.value,20);assert.ok(f.order.indexOf('effects')<f.order.indexOf('sample'));assert.ok(f.order.indexOf('sample')<f.order.indexOf('hp'));
});

test('unknown actor acknowledgement preserves samples and blocks resume or a replacement session',async()=>{
 const f=await fixture({writeFailure:true});const session=await f.runtime.start({actorUUIDs:['Actor.A'],autoRun:false});await f.runtime.drive(session.id);
 assert.equal(f.actor.system.attributes.hp.value,15);assert.equal((await f.ledger.getSession(session.id)).status,'paused');assert.equal(f.samples(),1);
 await assert.rejects(f.runtime.resume(session.id),/unresolved|uncertain/);
 await assert.rejects(f.runtime.start({actorUUIDs:['Actor.A'],autoRun:false}),/unresolved/);
 assert.equal(f.samples(),1);assert.equal(f.game.time.worldTime,600);
});

test('legacy sessions cannot resume through the computed driver',async()=>{
 const f=await fixture();await f.ledger.createSession({id:'old',actorUUIDs:['Actor.A'],startedAt:0,cursorAt:0,budgetEndsAt:600,goalsByPool:[]});await f.ledger.updateSession('old',{status:'paused'});
 await assert.rejects(f.runtime.resume('old'),/legacy/);assert.deepEqual(f.order,[]);
});

test('ambient encounter stops preserve an already paused computed session',async()=>{
 const f=await fixture(),session=await f.runtime.start({actorUUIDs:['Actor.A'],autoRun:false});
 await f.runtime.stop(session.id);const before=await f.ledger.getSession(session.id);
 await f.runtime.stopActive(session.id,'encounter-started');assert.deepEqual(await f.ledger.getSession(session.id),before);
});

test('recovery targets use the HP maximum after the pause has settled',async()=>{
 const f=await fixture({onPause:actor=>{actor.system.attributes.hp.max=30}}),session=await f.runtime.start({actorUUIDs:['Actor.A'],autoRun:false});
 assert.equal(session.goalsByPool[0].targetHP,30);assert.equal(session.recoveryGoals.patientTargets[0].basisMaxHP,30);
});

test('a summary does not report a character finished while required focus is missing',()=>{
 const actor={actorUUID:'Actor.A',name:'A',pool:{poolUUID:'Actor.A'},hp:{value:20,max:20},focus:{value:0,max:1},wounded:false};
 const summary=computedSummary({startedAt:0,computedInitialActors:[actor],goalsByPool:[{poolUUID:'Actor.A',targetHP:20}],recoveryGoals:{requireNoWounded:false},requireFullFocus:true},{cursorAt:600,history:[]},[actor],'budget');
 assert.equal(summary.actors[0].status,'remaining');
});

test('one confirmed transition stores both receipts and omits accumulated events from the session',async()=>{
 const f=await fixture(),changes=[],transition=f.ledger.transitionComputedBatch;
 f.ledger.transitionComputedBatch=async(...args)=>{changes.push(args[2]);return transition(...args)};
 const session=await f.runtime.start({actorUUIDs:['Actor.A'],autoRun:false});await f.runtime.drive(session.id);
 assert.equal(changes.length,1);assert.equal(changes[0].state,'confirmed');assert.deepEqual(Object.keys(changes[0].receipts),['actors','items']);
 const saved=await f.ledger.getSession(session.id);assert.equal('events' in saved.computedRecovery.summary,false);
 const shown=await f.runtime.snapshot(session.id);assert.equal(shown.session.computedRecovery.summary.events.length,1);assert.equal(shown.activities.length,0);
});

test('a peer stop between actor and item writes prevents items and records only the actor receipt',async()=>{
 const f=await fixture(),apply=f.documents.applyActors;let itemCalls=0;
 const session=await f.runtime.start({actorUUIDs:['Actor.A'],autoRun:false});
 f.documents.applyActors=async(...args)=>{const result=await apply(...args);await f.ledger.updateSession(session.id,{status:'paused',stopReason:'peer-stop'});return result};
 f.documents.applyItems=async()=>{itemCalls++;return {items:[],wounded:[]}};
 await f.runtime.drive(session.id);assert.equal(itemCalls,0);
 const saved=await f.ledger.getSession(session.id),batch=await f.ledger.getComputedBatch(session.id,saved.computedRecovery.batches[0].id);
 assert.equal(batch.state,'uncertain');assert.deepEqual(Object.keys(batch.receipts),['actors']);
});

test('panel projection hydrates compact batch indexes from the same ledger read',async()=>{
 const f=await fixture(),session=await f.runtime.start({actorUUIDs:['Actor.A'],autoRun:false});await f.runtime.drive(session.id);
 const state=await f.ledger.all(),data={session:state.sessions[session.id],activities:Object.values(state.activities),clocks:Object.values(state.clocks)};
 const shown=f.runtime.describeSnapshot(data,[]);
 assert.equal(shown.activities.length,0);assert.equal(shown.session.computedRecovery.summary.events.length,1);
 assert.equal('events' in state.sessions[session.id].computedRecovery.summary,false);
});

test('uncertain batch reconciliation loads its independent persisted document plan',async()=>{
 const f=await fixture({writeFailure:true}),session=await f.runtime.start({actorUUIDs:['Actor.A'],autoRun:false});await f.runtime.drive(session.id);
 const readback=await f.runtime.reconcile(session.id);assert.equal(readback.computedReadbacks[0].actors[0].state,'confirmed');assert.equal(f.samples(),1);
});

test('changed maximum HP stops before advancing beyond the saved recovery goal basis',async()=>{
 const f=await fixture(),session=await f.runtime.start({actorUUIDs:['Actor.A'],autoRun:false});f.actor.system.attributes.hp.max=30;
 await f.runtime.drive(session.id);const saved=await f.ledger.getSession(session.id);
 assert.equal(saved.status,'paused');assert.equal(saved.stopReason,'recovery-max-changed');assert.equal(f.game.time.worldTime,0);assert.equal(f.samples(),0);
});

test('resume rechecks current HP maximum before acquiring a fresh driver',async()=>{
 const f=await fixture(),session=await f.runtime.start({actorUUIDs:['Actor.A'],autoRun:false});await f.runtime.stop(session.id);
 f.actor.system.attributes.hp.max=30;const before=await f.ledger.getSession(session.id);
 await assert.rejects(f.runtime.resume(session.id,{autoRun:false}),/recovery-max-changed/);assert.deepEqual(await f.ledger.getSession(session.id),before);
});

for(const stage of ['prepare','claim'])test(`a ${stage} failure after time advances cannot discard the boundary and reroll on resume`,async()=>{
 const f=await fixture({passive:true}),session=await f.runtime.start({actorUUIDs:['Actor.A'],autoRun:false});
 if(stage==='prepare')f.documents.prepare=async()=>{throw Error('immunity-source-unavailable')};
 else f.ledger.claimComputedBatch=async()=>{throw Error('claim-rejected')};
 await f.runtime.drive(session.id);const saved=await f.ledger.getSession(session.id);
 assert.equal(f.game.time.worldTime,600);assert.equal(f.samples(),1);assert.equal(saved.computedRecovery.batches.length,0);
 assert.deepEqual(saved.computedRecovery.pendingBoundary,{from:0,to:600});
 assert.equal((await f.runtime.snapshot(session.id)).canResumeComputed,false);
 await assert.rejects(f.runtime.resume(session.id,{autoRun:false}),/unresolved-computed-batch/);
 await assert.rejects(f.runtime.start({actorUUIDs:['Actor.A'],autoRun:false}),/unresolved/);
 assert.equal(f.samples(),1);assert.equal(f.actor.system.attributes.hp.value,6);
});

function qualifiedSimulation(input,random){
 const actors=input.actors.map(actor=>({...actor,type:'character',modeOfBeing:'living',canAct:true,slugs:[],computed:{version:1,
  patient:{healingReady:true,damageReady:true,vitalityReady:true},refocus:{ready:true,recovery:1,durationSeconds:600},
  treatments:[{ready:true,skill:'medicine',rank:'trained',dc:15,modifier:5,assurance:true,riskySurgery:false,outcomes:[2],bonus:0,durationSeconds:600,cooldownSeconds:600,capacity:1}]}}));
 return createRecoverySimulation({...input,actors,random});
}

test('stopping an in-memory segment commits its sampled prefix and resumes without rerolling it',async()=>{
 let f,session,samples=0,stopping;
 f=await fixture({simulationFactory:input=>qualifiedSimulation(input,()=>{if(++samples===1)setTimeout(()=>{stopping=f.runtime.stop(session.id)},0);return 0})});
 session=await f.runtime.start({actorUUIDs:['Actor.A'],autoRun:false});await f.runtime.drive(session.id);await stopping;
 const paused=await f.runtime.snapshot(session.id),prefix=structuredClone(paused.session.computedRecovery.summary.events);
 assert.equal(paused.session.status,'paused');assert.ok(prefix.length>0);assert.equal(samples,prefix.length*2);assert.ok(f.game.time.worldTime>0);assert.equal(paused.session.computedRecovery.pendingCalculation,undefined);
 await f.runtime.resume(session.id,{autoRun:false});await f.runtime.drive(session.id);
 const final=await f.runtime.snapshot(session.id);assert.equal(final.session.status,'complete');assert.equal(samples,20);assert.equal(f.game.time.worldTime,6000);assert.deepEqual(final.session.computedRecovery.summary.events.slice(0,prefix.length),prefix);
});

test('takeover during in-memory sampling preserves its intent and cannot restart the discarded segment',async()=>{
 let f,session,samples=0,taking;
 f=await fixture({simulationFactory:input=>qualifiedSimulation(input,()=>{if(++samples===1)setTimeout(()=>{taking=f.runtime.takeover(session.id)},0);return 0})});
 session=await f.runtime.start({actorUUIDs:['Actor.A'],autoRun:false});await f.runtime.drive(session.id);await taking;
 const paused=await f.runtime.snapshot(session.id);assert.equal(paused.session.status,'paused');assert.ok(samples>0);
 assert.deepEqual(paused.session.computedRecovery.pendingCalculation,{from:0});assert.equal(f.game.time.worldTime,0);assert.equal(f.actor.system.attributes.hp.value,1);assert.equal(paused.canResumeComputed,false);
 await assert.rejects(f.runtime.resume(session.id,{autoRun:false}),/unresolved/);await assert.rejects(f.runtime.start({actorUUIDs:['Actor.A'],autoRun:false}),/unresolved/);
});

for(const persisted of [false,true])test(`an unknown computation intent acknowledgement samples nothing (persisted=${persisted})`,async()=>{
 let samples=0;const f=await fixture({simulationFactory:input=>qualifiedSimulation(input,()=>{samples++;return 0})});
 const begin=f.ledger.beginComputedCalculation;
 f.ledger.beginComputedCalculation=async(...args)=>{if(persisted)await begin(...args);throw Error('revision-acknowledgement-unknown')};
 const session=await f.runtime.start({actorUUIDs:['Actor.A'],autoRun:false});await f.runtime.drive(session.id);
 assert.equal(samples,0);assert.equal(f.game.time.worldTime,0);assert.equal(f.actor.system.attributes.hp.value,1);
 if(persisted)await assert.rejects(f.runtime.resume(session.id,{autoRun:false}),/unresolved/);
});
