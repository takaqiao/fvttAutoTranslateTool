import {test} from 'node:test';
import assert from 'node:assert/strict';
import {authorityFixture} from './authority-fixture.mjs';
import {createCoordinator} from '../../scripts/exploration/coordinator.mjs';
import {createClock} from '../../scripts/exploration/clock.mjs';
import {chooseNext} from '../../scripts/exploration/policy.mjs';

async function fixture(){
 const server=await authorityFixture(),ledger=server.client('driver'),listeners=new Map();let pauseEffect=()=>{},authority=true,healed=false;
 const game={user:{id:'G',active:true},paused:false,time:{worldTime:0,advance:async(dt,options)=>{assert.equal(game.paused,true);game.time.worldTime+=dt;listeners.get('updateWorldTime')?.(game.time.worldTime,dt,options,'G')}}};
 const clock=createClock({game,ledger,Hooks:{on:(event,fn)=>{listeners.set(event,fn);return event},off:event=>listeners.delete(event)},isAuthority:()=>authority,timeEffects:{beforeAdvance:async()=>({status:'ready'}),settle:async()=>({status:'ready',proof:[]})}});
 const actor={actorUUID:'Actor.H',pool:{poolUUID:'Actor.H',ready:true},hp:{value:1,max:20},focus:{value:0,max:0},slugs:[],assuranceSkills:[],items:[],medicine:{rank:0}};
 const provider={id:'treat-wounds',begin:async()=>({status:'started'}),complete:async activity=>{
  assert.equal(game.paused,true);const permit=await ledger.claimExecution(activity.id,{...coordinator.executionScope(activity.sessionId),operationId:'treat-wounds',ownerUserId:'G',ownerClientNonce:'driver',attemptNonce:activity.id,permitNonce:activity.id});
  const result={status:'confirmed',proof:{useId:activity.id,checkIds:['check'],receiptIds:['receipt']}};await ledger.recordExecutionResult(activity.id,{permit,result});healed=true;return result;
 }};
 const coordinator=createCoordinator({game,ledger,clock,capabilities:{snapshot:async()=>[{...actor,hp:{value:healed?20:1,max:20}}]},providers:[provider],policy:chooseNext,isAuthority:()=>authority,now:()=>game.time.worldTime,pauseGame:async()=>{await pauseEffect();game.paused=true},ownerOperations:{createActivityContext:async()=>({})}});
 return {coordinator,ledger,game,setPauseEffect:fn=>{pauseEffect=fn},setAuthority:value=>{authority=value}};
}
const start=f=>f.coordinator.start({id:'S',actorUUIDs:['Actor.H'],autoRun:false});

test('automatic start pauses before capturing its initial world time',async()=>{
 const f=await fixture();f.setPauseEffect(async()=>{await new Promise(resolve=>setImmediate(resolve));f.game.time.worldTime=45});
 const s=await start(f);assert.equal(f.game.paused,true);assert.equal(s.startedAt,45);assert.equal(s.cursorAt,45);assert.equal(s.budgetEndsAt,7245);
});

test('automatic steps re-pause and retain explicit time advancement through completion',async()=>{
 const f=await fixture(),s=await start(f);
 await f.coordinator.addActivity(s.id,{providerId:'treat-wounds',actorUUID:'Actor.H',patientUUIDs:['Actor.H'],hpPoolUUIDs:['Actor.H'],startedAt:0,endsAt:600,options:{}});
 f.game.paused=false;
 const advanced=await f.coordinator.step(s.id);assert.equal(advanced.status,'running',advanced.reason);assert.equal(f.game.time.worldTime,600);
 assert.equal((await f.ledger.snapshot(s.id)).clocks[0].state,'confirmed');
 assert.equal((await f.coordinator.step(s.id)).status,'complete');assert.equal(f.game.paused,true);
});

test('resume pauses the game while stop leaves it paused',async()=>{
 const f=await fixture(),s=await start(f);await f.coordinator.stop(s.id);assert.equal(f.game.paused,true);
 f.game.paused=false;assert.equal((await f.coordinator.resume(s.id,{autoRun:false})).status,'running');assert.equal(f.game.paused,true);
});

test('manual recording and invalid starts do not pause the game',async()=>{
 const f=await fixture();f.setPauseEffect(()=>assert.fail('unneeded pause'));
 await assert.rejects(f.coordinator.start({actorUUIDs:['Actor.H'],goalsByPool:[{poolUUID:'Actor.H',targetHP:30}],autoRun:false}),/invalid-recovery-goals/);
 const s=await f.coordinator.start({actorUUIDs:['Actor.H'],manual:true});assert.equal(s.status,'recording');assert.equal(f.game.paused,false);
});

test('pause failure prevents session creation and new activity execution',async()=>{
 const f=await fixture();f.setPauseEffect(()=>{throw Error('game-pause-failed')});
 await assert.rejects(start(f),/game-pause-failed/);assert.deepEqual((await f.ledger.all()).sessions,{});
 f.setPauseEffect(()=>{});const s=await start(f);f.game.paused=false;f.setPauseEffect(()=>{throw Error('game-pause-failed')});
 assert.equal((await f.coordinator.step(s.id)).reason,'game-pause-failed');
 const data=await f.ledger.snapshot(s.id);assert.equal(data.session.status,'paused');assert.equal(data.activities.length,0);assert.equal(data.clocks.length,0);
 await assert.rejects(f.coordinator.resume(s.id,{autoRun:false}),/game-pause-failed/);assert.equal((await f.ledger.getSession(s.id)).status,'paused');
});

test('resume rechecks time after the pause hook before granting a new driver',async()=>{
 const f=await fixture(),s=await start(f);await f.coordinator.stop(s.id);f.game.paused=false;f.setPauseEffect(()=>{f.game.time.worldTime=1});
 await assert.rejects(f.coordinator.resume(s.id,{autoRun:false}),/external-world-time-change/);assert.equal((await f.ledger.getSession(s.id)).status,'paused');assert.equal(f.coordinator.executionScope(s.id),undefined);
});

test('pause hooks cannot retain a revoked authority or invalidated execution lease',async()=>{
 const f=await fixture();f.setPauseEffect(()=>f.setAuthority(false));await assert.rejects(start(f),/active-gm-required/);assert.deepEqual((await f.ledger.all()).sessions,{});
 f.setAuthority(true);f.setPauseEffect(()=>{});const s=await start(f);f.setPauseEffect(()=>f.coordinator.invalidate('gm-changed'));
 assert.equal((await f.coordinator.step(s.id)).status,'observing');const data=await f.ledger.snapshot(s.id);assert.equal(data.activities.length,0);assert.equal(data.clocks.length,0);
});
