import test from 'node:test';
import assert from 'node:assert/strict';
import {createClock} from '../../scripts/exploration/clock.mjs';
import {createLedger} from '../../scripts/exploration/ledger.mjs';
import {createCoordinator} from '../../scripts/exploration/coordinator.mjs';

const deferred=()=>{let resolve;return {promise:new Promise(r=>resolve=r),resolve:(...args)=>resolve(...args)}};
const commit={id:'C',sessionId:'S',from:0,to:600};
function fixture(){
  let data={sessions:{S:{id:'S',status:'running',startedAt:0,activityIds:[]}},activities:{},clocks:{}},calls=0;
  const listeners=new Map(),user={id:'G',isGM:true,active:true};
  const game={user,users:{activeGM:user,get:()=>user},time:{worldTime:0,advance:async(dt,options)=>{calls++;game.time.worldTime+=dt;listeners.get('updateWorldTime')?.(game.time.worldTime,dt,options,'G')}}};
  const ledger=createLedger({read:async()=>structuredClone(data),write:async next=>{data=structuredClone(next)},isAuthority:()=>true});
  return {game,ledger,Hooks:{on:(name,fn)=>{listeners.set(name,fn);return name},off:name=>listeners.delete(name)},isAuthority:()=>true,timeEffects:{beforeAdvance:async()=>({status:'ready'}),settle:async()=>({status:'ready',proof:[]})},calls:()=>calls,emit:(dt,options)=>listeners.get('updateWorldTime')?.(game.time.worldTime,dt,options,'G')};
}

test('coordinator Stop during passive preflight cancels before issuing native time',async()=>{
  const f=fixture(),entered=deferred(),release=deferred();
  f.timeEffects.beforeAdvance=async()=>{entered.resolve();await release.promise;return {status:'ready'}};
  const clock=createClock(f),actor={actorUUID:'Actor.H',slugs:[],pool:{poolUUID:'Actor.P'},hp:{value:1,max:20},focus:{value:0,max:0}};
  const coordinator=createCoordinator({ledger:f.ledger,capabilities:{snapshot:async()=>[actor]},providers:[{id:'treat-wounds',begin:async()=>({status:'started'}),complete:async()=>assert.fail('cancelled activity was completed')}],clock,policy:()=>({activities:[],checkpointAt:600}),isAuthority:()=>true,now:()=>f.game.time.worldTime,ownerOperations:{createActivityContext:async()=>({})}});
  await f.ledger.updateSession('S',{status:'complete'});
  await coordinator.start({id:'RUN',actorUUIDs:['Actor.H'],autoRun:false});
  await coordinator.addActivity('RUN',{id:'A',providerId:'treat-wounds',actorUUID:'Actor.H',patientUUIDs:['Actor.P'],hpPoolUUIDs:['Actor.P'],startedAt:0,endsAt:600,options:{},source:{}});
  const running=coordinator.step('RUN');await entered.promise;
  await coordinator.stop('RUN');release.resolve();await running;
  assert.equal(f.calls(),0);assert.equal(f.game.time.worldTime,0);
  assert.equal((await f.ledger.getActivity('A')).state,'cancelled');
  assert.equal((await f.ledger.getSession('RUN')).status,'paused');
});

test('clock cancellation while persistent claim is pending does not issue advance',async()=>{
  const f=fixture(),entered=deferred(),release=deferred(),original=f.ledger.upsertClockCommit;
  f.ledger.upsertClockCommit=async input=>{entered.resolve();await release.promise;return original(input)};
  const clock=createClock(f),running=clock.advanceTo(commit);await entered.promise;
  clock.stop();release.resolve();const result=await running;
  assert.equal(f.calls(),0);assert.equal(f.game.time.worldTime,0);
  assert.notEqual(result.status,'confirmed');assert.equal((await f.ledger.getClockCommit('C')).nativeIssued,false);
});

for(const scope of [undefined,{sessionId:'S',leaseNonce:'lease'}])test(`clock ${scope?'targeted':'global'} cancellation covers the initial asynchronous lookup`,async()=>{
  const f=fixture(),entered=deferred(),release=deferred(),original=f.ledger.getClockCommit;let pending=true;
  f.ledger.getClockCommit=async id=>{if(pending){pending=false;entered.resolve();await release.promise}return original(id)};
  const clock=createClock(f),running=clock.advanceTo(commit,{leaseNonce:'lease'});await entered.promise;
  clock.stop('lookup-stopped',scope);release.resolve();const result=await running;
  assert.equal(result.reason,'lookup-stopped');assert.equal(f.calls(),0);assert.equal(await original('C'),null);
});

test('a paused persistent session cannot start a new clock attempt',async()=>{
  const f=fixture();await f.ledger.updateSession('S',{status:'paused'});
  const result=await createClock(f).advanceTo(commit);
  assert.equal(f.calls(),0);assert.equal(f.game.time.worldTime,0);assert.notEqual(result.status,'confirmed');
});

test('an encounter that begins during preflight blocks a fresh advance without the runtime hook',async()=>{
  const f=fixture(),entered=deferred(),release=deferred();
  f.timeEffects.beforeAdvance=async()=>{entered.resolve();await release.promise;return {status:'ready'}};
  const clock=createClock(f),running=clock.advanceTo(commit);await entered.promise;
  f.game.combat={started:true};release.resolve();const result=await running;
  assert.equal(f.calls(),0);assert.equal(f.game.time.worldTime,0);assert.notEqual(result.status,'confirmed');
});

test('Stop after a native request is issued preserves elapsed time and never replays the attempt',async()=>{
  const f=fixture(),entered=deferred(),release=deferred();let issued=0;
  f.game.time.advance=async(dt,options)=>{issued++;entered.resolve();await release.promise;f.game.time.worldTime+=dt;f.emit(dt,options)};
  const clock=createClock({...f,confirmationTimeoutMs:1000}),running=clock.advanceTo(commit);await entered.promise;
  clock.stop();release.resolve();const first=await running;
  await clock.advanceTo(commit);assert.equal(issued,1);assert.equal(f.game.time.worldTime,600);
  assert.equal(first.status,'uncertain');assert.equal((await f.ledger.getClockCommit('C')).nativeIssued,true);
});
