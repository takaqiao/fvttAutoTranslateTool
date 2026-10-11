import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createClock} from '../../scripts/exploration/clock.mjs';
import {createTimeEffects} from '../../scripts/exploration/time-effects.mjs';
import {authorityFixture} from './authority-fixture.mjs';

const checkpoint={id:'C',sessionId:'S',from:-3703909144,to:-3703906843};
const passiveProof={adapterId:'passive',checkpointId:'C',receiptId:'effect-receipt'};
const deferred=()=>{let resolve;return {promise:new Promise(r=>resolve=r),resolve:value=>resolve(value)}};

async function fixture(){
 const storage=await authorityFixture(),ledger=storage.client('driver'),listeners=new Map();let calls=0;
 const session=await ledger.createSession({id:'S',actorUUIDs:['Actor.P'],startedAt:checkpoint.from,cursorAt:checkpoint.from,budgetEndsAt:checkpoint.to+600,status:'running'});
 const emit=(dt,options)=>listeners.get('updateWorldTime')?.(game.time.worldTime,dt,options,'G');
 const game={user:{id:'G'},time:{worldTime:checkpoint.from,advance:async(dt,options)=>{calls++;game.time.worldTime+=dt;emit(dt,options)}}};
 const adapter={matches:()=>true,beforeAdvance:async()=>({status:'ready'}),settle:async()=>({status:'ready',proof:[passiveProof]})};
 const timeEffects=createTimeEffects({capabilities:{activePassiveRules:async()=>[{actorUUID:'Actor.P',key:'FastHealing',passing:true}]},completionAdapters:[adapter]});
 return {game,ledger,timeEffects,adapter,scope:{leaseNonce:session.driver.leaseNonce},isAuthority:()=>true,confirmationTimeoutMs:1000,
  Hooks:{on:(name,fn)=>{listeners.set(name,fn);return name},off:name=>listeners.delete(name)},calls:()=>calls,
  externalAdvance:dt=>{game.time.worldTime+=dt;emit(dt,{})}};
}

test('external time during passive settlement preserves the confirmed checkpoint and its proof',async()=>{
 const f=await fixture(),entered=deferred(),release=deferred();
 f.adapter.settle=async()=>{entered.resolve();await release.promise;return {status:'ready',proof:[passiveProof]}};
 const running=createClock(f).advanceTo(checkpoint,f.scope);await entered.promise;
 f.externalAdvance(60);release.resolve();const result=await running;
 assert.equal(result.status,'confirmed');
 const saved=await f.ledger.getClockCommit('C');
 assert.equal(saved.state,'confirmed');assert.equal(saved.from,-3703909144);assert.equal(saved.to,-3703906843);
 assert.equal(saved.nativeIssued,true);assert.equal(saved.nativeResolved,true);assert.equal(saved.effectsSettled,true);
 assert.equal(saved.evidence[0].worldTime,-3703906843);assert.equal(saved.evidence[0].dt,2301);assert.deepEqual(saved.evidence[1],passiveProof);
 assert.deepEqual(result.commit,saved);assert.equal(f.game.time.worldTime,-3703906783);
 const reloaded=createClock(f);
 assert.deepEqual((await reloaded.reconcile(checkpoint)).commit,saved);
 assert.deepEqual((await reloaded.advanceTo(checkpoint,f.scope)).commit,saved);
 assert.equal(f.calls(),1);assert.equal(f.game.time.worldTime,-3703906783);
});

test('external time cannot confirm a checkpoint whose passive effects remain unproven',async()=>{
 const f=await fixture(),entered=deferred(),release=deferred();
 f.adapter.settle=async()=>{entered.resolve();await release.promise;return {status:'uncertain',reason:'passive-native-pending'}};
 const running=createClock(f).advanceTo(checkpoint,f.scope);await entered.promise;
 f.externalAdvance(60);release.resolve();const result=await running;
 assert.equal(result.status,'uncertain');assert.equal(result.reason,'passive-native-pending');
 const saved=await f.ledger.getClockCommit('C');
 assert.equal(saved.state,'uncertain');assert.equal(saved.nativeResolved,true);assert.notEqual(saved.effectsSettled,true);
 assert.equal(saved.evidence.length,1);assert.equal(saved.evidence[0].worldTime,-3703906843);
 const reloaded=createClock(f);
 assert.equal((await reloaded.reconcile(checkpoint)).status,'uncertain');
 assert.equal((await reloaded.advanceTo(checkpoint,f.scope)).status,'uncertain');
 assert.equal(f.calls(),1);assert.equal(f.game.time.worldTime,-3703906783);
});

test('a foreign hook before native resolution still prevents confirmation despite an earlier exact receipt',async()=>{
 const f=await fixture(),entered=deferred(),release=deferred(),advance=f.game.time.advance;
 f.game.time.advance=async(dt,options)=>{await advance(dt,options);entered.resolve();await release.promise};
 f.adapter.settle=async()=>assert.fail('unconfirmed native time reached passive settlement');
 const running=createClock(f).advanceTo(checkpoint,f.scope);await entered.promise;
 f.externalAdvance(60);release.resolve();const result=await running;
 assert.equal(result.status,'uncertain');assert.equal(result.reason,'world-time-source-unconfirmed');
 const saved=await f.ledger.getClockCommit('C');
 assert.equal(saved.state,'uncertain');assert.notEqual(saved.nativeResolved,true);assert.notEqual(saved.effectsSettled,true);
 const reloaded=createClock(f);
 assert.equal((await reloaded.reconcile(checkpoint)).status,'uncertain');
 assert.equal((await reloaded.advanceTo(checkpoint,f.scope)).status,'uncertain');
 assert.equal(f.calls(),1);assert.equal(f.game.time.worldTime,-3703906783);
});
