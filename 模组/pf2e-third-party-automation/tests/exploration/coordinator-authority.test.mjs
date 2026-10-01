import nodeTest from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {createCoordinator} from '../../scripts/exploration/coordinator.mjs';
import {createClock} from '../../scripts/exploration/clock.mjs';
import {createLedger} from '../../scripts/exploration/ledger.mjs';
import {createRevisionStore} from '../../scripts/exploration/revision-store.mjs';
import {canonicalJSON} from '../../scripts/exploration/revision-codec.mjs';
const M='pf2e-third-party-automation',root='JournalEntry.ROOT000000000001';
const test=(name,fn)=>nodeTest(name,{timeout:1000},fn);
const deferred=()=>{let resolve;return {promise:new Promise(r=>resolve=r),resolve:value=>resolve(value)}};
const input={providerId:'treat-wounds',actorUUID:'H',patientUUIDs:['H'],hpPoolUUIDs:['H'],startedAt:0,endsAt:600,options:{}};
async function fixture(){
  const state={sessions:{},activities:{},clocks:{}};
  const raw={_id:'ROOT000000000001',ownership:{default:0,G:3},pages:[],flags:{[M]:{explorationLedger:state}}};
  let sequence=0,calls=0;const hooks=new Map();
  const game={user:{id:'G'},time:{worldTime:0,advance:async(dt,options)=>{calls++;game.time.worldTime+=dt;for(const fn of hooks.values())fn(game.time.worldTime,dt,options,'G')}}};
  function store(client){return createRevisionStore({getRootUUID:()=>root,readRoot:async()=>structuredClone(raw),isAuthority:()=>true,canUseRoot:()=>true,writerUserId:()=> 'G',writerClientId:client,nonce:()=>`n${++sequence}`,
    createPage:async({rootUUID,page})=>{
      const ack={type:'JournalEntryPage',action:'create',userId:'G',broadcast:false,operation:{parentUuid:rootUUID}};
      if(raw.pages.some(p=>p._id===page._id))return {...ack,error:{class:'ServerError',message:`The _id [${page._id}] already exists within the parent collection: JournalEntry [${raw._id}] pages`}};
      raw.pages.push(structuredClone(page));return {...ack,result:[structuredClone(page)]};
    }})}
  await store('setup').initialize({epoch:'epoch',expectedSourceDigest:createHash('sha256').update(canonicalJSON(state)).digest('hex')});
  function client(id,options={}){
    const ledger=createLedger({...store(id),isAuthority:()=>true,identity:()=>({userId:'G',clientNonce:id})});
    const actor={actorUUID:'H',medicine:{rank:0},slugs:[],items:[],assuranceSkills:[],pool:{poolUUID:'H',ready:true},hp:{value:1,max:20},focus:{value:0,max:0},cooldownExpiresAt:0};
    let begins=0,completes=0,snapshots=0;
    const capabilities={snapshot:async()=>{snapshots++;if(options.snapshot)await options.snapshot();return [actor]}};
    const clock=createClock({game,Hooks:{on:(name,fn)=>{const key=++sequence;hooks.set(key,fn);return key},off:(name,key)=>hooks.delete(key)},ledger,isAuthority:()=>true,confirmationTimeoutMs:100,timeEffects:{beforeAdvance:options.preflight??(async()=>({status:'ready'})),settle:async()=>({status:'ready',proof:[]})}});
    const provider={id:'treat-wounds',begin:async()=>{begins++;if(options.begin)await options.begin();return {status:'started'}},complete:async()=>{completes++;return {status:'uncertain',reason:'fixture-native-not-granted'}},cancel:async()=>{}};
    const c=createCoordinator({ledger,capabilities,providers:[provider],clock,policy:options.policy??(()=>({activities:[],checkpointAt:600})),isAuthority:()=>true,now:()=>game.time.worldTime,ownerOperations:{createActivityContext:async()=>{if(options.context)await options.context();return {}}}});
    return {c,ledger,clock,counts:()=>({begins,completes,snapshots})};
  }
  return {client,game,calls:()=>calls,raw};
}
const start=c=>c.start({id:'S',actorUUIDs:['H'],autoRun:false});

for(const status of ['paused','closed','complete'])test(`late failure preserves a peer ${status} session and a newer session clock`,async()=>{
  const f=await fixture(),oldEntered=deferred(),oldRelease=deferred(),newEntered=deferred(),newRelease=deferred();let armed=false;
  const a=f.client('A',{snapshot:async()=>{if(armed){armed=false;oldEntered.resolve();await oldRelease.promise;throw Error('old-snapshot-failed')}},preflight:async()=>{newEntered.resolve();await newRelease.promise;return {status:'ready'}}}),b=f.client('B');
  await start(a.c);armed=true;const oldStep=a.c.step('S');await oldEntered.promise;await b.c.stop('S');
  if(status==='closed')await b.c.review('S',{note:'reviewed',userId:'G'});
  if(status==='complete')await a.ledger.updateSession('S',{status:'complete'},a.c.executionScope('S'));
  const before=await a.ledger.getSession('S');await a.c.start({id:'T',actorUUIDs:['H'],autoRun:false});const newStep=a.c.step('T');await newEntered.promise;
  oldRelease.resolve();await oldStep;newRelease.resolve();await newStep;
  assert.deepEqual(await a.ledger.getSession('S'),before);assert.equal((await a.ledger.getSession('T')).status,'running');assert.equal(f.calls(),1);
});

test('peer Stop and review between internal pause read and transaction are preserved',async()=>{
  const f=await fixture(),a=f.client('A',{snapshot:async()=>{if(armed)throw Error('snapshot-failed')}}),b=f.client('B');let armed=false;
  await start(a.c);const update=a.ledger.updateSession;let intercepted=false,before;
  a.ledger.updateSession=async(id,patch,options)=>{
    if(patch.status==='paused'&&options?.leaseNonce&&!intercepted){intercepted=true;await b.c.stop(id);before=await b.c.review(id,{note:'closed during pause',userId:'G'})}
    return update(id,patch,options);
  };
  armed=true;await a.c.step('S');assert.equal(intercepted,true);assert.deepEqual(await a.ledger.getSession('S'),before);assert.equal(f.calls(),0);
});

test('targeted cancellation for another session cannot cancel this clock preflight',async()=>{
  const f=await fixture(),entered=deferred(),release=deferred(),a=f.client('A',{preflight:async()=>{entered.resolve();await release.promise;return {status:'ready'}}});
  await start(a.c);const pending=a.c.step('S');await entered.promise;a.clock.stop('old-session-stopped',{sessionId:'Old'});release.resolve();
  assert.equal((await pending).status,'running');assert.equal(f.calls(),1);assert.equal((await a.ledger.getSession('S')).status,'running');
});

test('targeted cancellation with an obsolete lease cannot cancel this clock preflight',async()=>{
  const f=await fixture(),entered=deferred(),release=deferred(),a=f.client('A',{preflight:async()=>{entered.resolve();await release.promise;return {status:'ready'}}});
  await start(a.c);const pending=a.c.step('S');await entered.promise;a.clock.stop('old-lease-stopped',{sessionId:'S',leaseNonce:'obsolete'});release.resolve();
  assert.equal((await pending).status,'running');assert.equal(f.calls(),1);
});

test('only the create ACK recipient holds an execution scope; a peer cannot drive or pause it',async()=>{
  const f=await fixture(),a=f.client('A'),b=f.client('B'),s=await start(a.c);
  const scope=a.c.executionScope('S');assert.equal(scope.leaseNonce,s.driver.leaseNonce);scope.leaseNonce='changed-copy';assert.equal(a.c.executionScope('S').leaseNonce,s.driver.leaseNonce);
  assert.equal(b.c.executionScope('S'),undefined);assert.equal((await b.c.step('S')).status,'observing');assert.equal((await a.ledger.getSession('S')).status,'running');assert.equal(f.calls(),0);assert.equal(b.counts().begins,0);assert.equal(b.counts().completes,0);
});
test('restore and recover never adopt a persisted lease or pause another runtime',async()=>{
  const f=await fixture(),a=f.client('A'),b=f.client('B');await start(a.c);await a.c.addActivity('S',{...input,id:'A'});
  assert.equal((await b.c.restore('S')).status,'running');assert.equal((await b.c.recover('S')).status,'running');assert.equal(b.c.executionScope('S'),undefined);
  assert.equal((await a.ledger.getActivity('A')).state,'started');assert.equal((await b.c.step('S')).status,'observing');assert.equal(f.calls(),0);
});
test('a same-identity reconstructed coordinator does not reconstruct the private continuation',async()=>{
  const f=await fixture(),a=f.client('A');await start(a.c);const reconstructed=f.client('A');await reconstructed.c.restore('S');
  assert.equal(reconstructed.c.executionScope('S'),undefined);assert.equal((await reconstructed.c.step('S')).status,'observing');assert.equal((await a.ledger.getSession('S')).status,'running');
});
test('the owning coordinator propagates its lease through clock claim and cursor update',async()=>{
  const f=await fixture(),a=f.client('A');await start(a.c);const result=await a.c.step('S');
  assert.equal(result.status,'running');assert.equal(f.calls(),1);assert.equal((await a.ledger.getSession('S')).cursorAt,600);
  const clocks=(await a.ledger.snapshot('S')).clocks;assert.equal(clocks.length,1);assert.equal(clocks[0].state,'confirmed');assert.equal(clocks[0].claim.leaseNonce,a.c.executionScope('S').leaseNonce);
});
test('explicit peer takeover pauses and quarantines but grants no private execution scope',async()=>{
  const f=await fixture(),a=f.client('A'),b=f.client('B');await start(a.c);await a.c.addActivity('S',{...input,id:'A'});
  const taken=await b.c.takeover('S');assert.equal(taken.status,'paused');assert.equal((await a.ledger.getActivity('A')).state,'uncertain');assert.equal(b.c.executionScope('S'),undefined);
  await assert.rejects(b.c.resume('S',{autoRun:false}),/unresolved/);assert.equal(f.calls(),0);
});
test('resume rotates the driver lease and an old client cannot step or add an activity',async()=>{
  const f=await fixture(),a=f.client('A'),b=f.client('B'),s=await start(a.c);await a.c.stop('S');const resumed=await b.c.resume('S',{autoRun:false});
  assert.notEqual(resumed.driver.leaseNonce,s.driver.leaseNonce);assert.equal(b.c.executionScope('S').leaseNonce,resumed.driver.leaseNonce);
  assert.equal((await a.c.step('S')).status,'observing');await assert.rejects(a.c.addActivity('S',input),/driver/);assert.equal((await b.ledger.getSession('S')).status,'running');assert.equal(f.calls(),0);
});
test('an old step exception after peer resume cannot stop the new driver',async()=>{
  const f=await fixture(),entered=deferred(),release=deferred();let armed=false;
  const a=f.client('A',{snapshot:async()=>{if(armed){entered.resolve();await release.promise;throw Error('old-capabilities-failed')}}}),b=f.client('B');await start(a.c);armed=true;
  const old=a.c.step('S');await entered.promise;await b.c.stop('S');const resumed=await b.c.resume('S',{autoRun:false});release.resolve();
  assert.equal((await old).status,'observing');const current=await b.ledger.getSession('S');assert.equal(current.status,'running');assert.equal(current.driver.leaseNonce,resumed.driver.leaseNonce);assert.equal(f.calls(),0);
});
test('context completion after Stop cannot invoke provider.begin',async()=>{
  const f=await fixture(),entered=deferred(),release=deferred(),a=f.client('A',{context:async()=>{entered.resolve();await release.promise}});await start(a.c);
  const pending=a.c.addActivity('S',{...input,id:'A'}).then(value=>({value}),error=>({error}));await Promise.race([entered.promise,pending.then(()=>assert.fail('activity did not reach context boundary'))]);await a.c.stop('S');release.resolve();await pending;
  assert.equal(a.counts().begins,0);assert.equal((await a.ledger.getActivity('A')).state,'cancelled');assert.equal(f.calls(),0);
});
test('clock refuses a copied peer lease and performs no native advance',async()=>{
  const f=await fixture(),a=f.client('A'),b=f.client('B'),s=await start(a.c);
  const result=await b.clock.advanceTo({id:'C',sessionId:'S',from:0,to:600},{leaseNonce:s.driver.leaseNonce});assert.notEqual(result.status,'confirmed');assert.equal(f.calls(),0);assert.equal((await a.ledger.snapshot('S')).clocks.length,0);
});
test('clock preflight paused by another GM never issues native time when released',async()=>{
  const f=await fixture(),entered=deferred(),release=deferred(),a=f.client('A',{preflight:async()=>{entered.resolve();await release.promise;return {status:'ready'}}}),b=f.client('B');await start(a.c);
  const running=a.c.step('S');await entered.promise;await b.c.stop('S');release.resolve();await running;assert.equal(f.calls(),0);assert.equal((await a.ledger.getSession('S')).status,'paused');
});
test('a stopped preflight cannot occupy the clock slot of a new lease in the same coordinator',async()=>{
  const f=await fixture(),entered=deferred(),release=deferred();let first=true;
  const a=f.client('A',{preflight:async()=>{if(first){first=false;entered.resolve();await release.promise}return {status:'ready'}}});await start(a.c);
  const old=a.c.step('S');await entered.promise;await a.c.stop('S');const resumed=await a.c.resume('S',{autoRun:false});
  const next=await a.c.step('S');release.resolve();await old;
  assert.equal(next.status,'running');assert.equal(f.calls(),1);const saved=await a.ledger.getSession('S');assert.equal(saved.status,'running');assert.equal(saved.driver.leaseNonce,resumed.driver.leaseNonce);
});
test('two independent starts expose a private scope only to the acknowledged winning runtime',async()=>{
  const f=await fixture(),a=f.client('A'),b=f.client('B');
  const outcomes=await Promise.allSettled([start(a.c),b.c.start({id:'T',actorUUIDs:['H'],autoRun:false})]);assert.equal(outcomes.filter(r=>r.status==='fulfilled').length,1);
  assert.equal([a.c.executionScope('S'),b.c.executionScope('T')].filter(Boolean).length,1);assert.equal(Object.values((await a.ledger.all()).sessions).filter(s=>s.status==='running').length,1);assert.equal(f.calls(),0);
});
test('a provider begin finishing after stop and resume cannot claim started or stop the new lease',async()=>{
  const f=await fixture(),entered=deferred(),release=deferred(),a=f.client('A',{begin:async()=>{entered.resolve();await release.promise}});await start(a.c);
  const old=a.c.addActivity('S',{...input,id:'A'}).then(value=>({value}),error=>({error}));await entered.promise;
  await a.c.stop('S');const resumed=await a.c.resume('S',{autoRun:false});release.resolve();assert.ok((await old).error);
  assert.equal((await a.ledger.getActivity('A')).state,'cancelled');assert.equal((await a.ledger.getSession('S')).status,'running');assert.equal(a.c.executionScope('S').leaseNonce,resumed.driver.leaseNonce);
});
test('clock reconciliation uses only existing exact saved evidence and issues no native time',async()=>{
  const f=await fixture(),a=f.client('A'),b=f.client('B');await start(a.c);const scope=a.c.executionScope('S');
  const commit={id:'C',sessionId:'S',gmId:'G',from:0,to:600,state:'started',evidence:[]};await a.ledger.upsertClockCommit(commit,scope);
  const evidence={worldTime:600,dt:600,userId:'G',options:{pf2eThirdPartyAutomation:{exploration:{sessionId:'S',checkpointId:'C',gmId:'G',expectedFrom:0,expectedTo:600}}}};
  await a.ledger.transitionClockCommit('C',{expected:['started'],patch:{state:'uncertain',nativeIssued:true,nativeResolved:true,effectsSettled:true,evidence:[evidence]},...scope});await a.c.stop('S');
  assert.equal((await b.clock.reconcile(commit)).status,'confirmed');assert.equal(f.calls(),0);assert.equal(b.c.executionScope('S'),undefined);
});
test('a cancelled old preflight finishing cannot clear a new attempt clock slot',async()=>{
  const f=await fixture(),firstEntered=deferred(),firstRelease=deferred(),secondEntered=deferred(),secondRelease=deferred();let count=0;
  const a=f.client('A',{preflight:async()=>{if(++count===1){firstEntered.resolve();await firstRelease.promise}else{secondEntered.resolve();await secondRelease.promise}return {status:'ready'}}});await start(a.c);
  const old=a.c.step('S');await firstEntered.promise;await a.c.stop('S');await a.c.resume('S',{autoRun:false});
  const next=a.c.step('S');await secondEntered.promise;firstRelease.resolve();await old;
  const third=await a.clock.advanceTo({id:'THIRD',sessionId:'S',from:0,to:600},a.c.executionScope('S'));assert.equal(third.reason,'clock-in-flight');assert.equal(f.calls(),0);
  secondRelease.resolve();assert.equal((await next).status,'running');assert.equal(f.calls(),1);
});
