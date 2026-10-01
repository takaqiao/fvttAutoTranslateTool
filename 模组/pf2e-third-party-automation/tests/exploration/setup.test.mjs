import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createLedgerSetup} from '../../scripts/exploration/setup.mjs';
import {canonicalJSON,decodeRevision} from '../../scripts/exploration/revision-codec.mjs';
import {createHash} from 'node:crypto';

const confirmations={issuersStopped:true,clientsReloaded:true,recoveryDisabled:true};
function fixture({configured=true,initialized=false}={}){
 let rootUUID=configured?'JournalEntry.ROOT000000000001':null,ready=initialized,authority=true;
 const calls=[],seed={sessions:{Old:{id:'Old',status:'paused'}},activities:{},clocks:{}};
 const game={user:{id:'G',isGM:true},users:{activeGM:{id:'G'}}};
 const store={status:async()=>{calls.push('status');if(!rootUUID)throw Error('root-not-configured');return {initialized:ready,rootUUID,epoch:ready?'existing-epoch':null}},read:async()=>{calls.push('read');return structuredClone(seed)},provision:async input=>{calls.push(['provision',input]);rootUUID='JournalEntry.ROOT000000000001';return {rootUUID,requiresReload:false}},initialize:async input=>{calls.push(['initialize',input]);ready=true;return {initialized:true,rootUUID,epoch:input.epoch}},select:async input=>{calls.push(['select',input]);return {rootUUID:input.rootUUID,requiresReload:true}}};
 const setup=createLedgerSetup({game,store,writerClientId:()=> 'client'});
 return {game,store,setup,calls,seed,loseAuthority:()=>{authority=false;game.users.activeGM={id:'Other'}},hasAuthority:()=>authority};
}

test('setup status describes a missing root without provisioning or starting recovery',async()=>{
 const f=fixture({configured:false});assert.deepEqual(await f.setup.status(),{state:'unconfigured'});assert.deepEqual(f.calls,['status']);
});

test('legacy initialization preserves the exact source fingerprint and never starts recovery',async()=>{
 const f=fixture();const result=await f.setup.initialize(confirmations),input=f.calls.find(c=>Array.isArray(c)&&c[0]==='initialize')[1];
 assert.equal(result.state,'ready');assert.equal(input.expectedSourceDigest,createHash('sha256').update(canonicalJSON(f.seed)).digest('hex'));assert.ok(input.epoch);assert.deepEqual(input.issuersStopped,true);assert.equal(f.calls.some(c=>Array.isArray(c)&&c[0]==='provision'),false);
});

test('new-world provisioning is an explicit separate setup action',async()=>{
 const f=fixture({configured:false});await assert.rejects(f.setup.initialize(confirmations),/root-not-configured/);assert.deepEqual(f.calls,['status']);
 const provisioned=await f.setup.provision(confirmations);assert.equal(provisioned.rootUUID,'JournalEntry.ROOT000000000001');assert.equal(f.calls.some(c=>Array.isArray(c)&&c[0]==='initialize'),false);
});

test('already initialized storage is read-only when the setup action is invoked again',async()=>{
 const f=fixture({initialized:true});const status=await f.setup.initialize(confirmations);assert.equal(status.state,'ready');assert.equal(status.epoch,'existing-epoch');assert.deepEqual(f.calls,['status','read']);
});

test('setup requires all controlled-migration prerequisites before any read or write',async()=>{
 for(const action of ['initialize','provision','select'])for(const key of Object.keys(confirmations)){
  const f=fixture();await assert.rejects(f.setup[action]({...confirmations,[key]:false,rootUUID:'JournalEntry.ROOT000000000001'}),/setup-confirmations-required/);assert.equal(f.calls.length,0);
 }
});

test('authority loss after a source read prevents genesis submission',async()=>{
 const f=fixture(),read=f.store.read;f.store.read=async()=>{const state=await read();f.loseAuthority();return state};
 await assert.rejects(f.setup.initialize(confirmations),/active-gm-required/);assert.equal(f.calls.some(c=>Array.isArray(c)&&c[0]==='initialize'),false);
});

test('nonactive GMs cannot provision, select or initialize storage',async()=>{
 for(const action of ['initialize','provision','select']){const f=fixture();f.loseAuthority();await assert.rejects(f.setup[action]({...confirmations,rootUUID:'JournalEntry.ROOT000000000001'}),/active-gm-required/);assert.equal(f.calls.length,0)}
});

test('configuration selection retains its explicit reload requirement',async()=>{
 const f=fixture();const result=await f.setup.select({...confirmations,rootUUID:'JournalEntry.OTHER00000000001'});assert.equal(result.requiresReload,true);assert.equal(f.calls.length,1);
});

test('status preserves configured-root and protocol errors as blocked diagnostics',async()=>{
 for(const reason of ['root-not-found','root-not-approved','revision-root-changed','revision-digest-mismatch']){const f=fixture();f.store.status=async()=>{throw Error(reason)};assert.deepEqual(await f.setup.status(),{state:'blocked',reason})}
});

test('setup does not convert an unknown genesis acknowledgement into a fresh attempt',async()=>{
 const f=fixture();let writes=0;f.store.initialize=async()=>{writes++;throw Error('revision-acknowledgement-unknown')};
 await assert.rejects(f.setup.initialize(confirmations),/revision-acknowledgement-unknown/);assert.equal(writes,1);assert.equal(f.calls.filter(c=>c==='read').length,1);
});

// The raw server owns page uniqueness; each setup uses independent production storage and ledger objects.
async function migrationFixture(){
 const {createDocumentStore}=await import('../../scripts/exploration/document-store.mjs');
 const {createLedger}=await import('../../scripts/exploration/ledger.mjs');
 const M='pf2e-third-party-automation',rootUUID='JournalEntry.ROOT000000000001';
 const seed={sessions:{Old:{id:'Old',actorUUIDs:['Actor.H','Actor.P'],startedAt:0,cursorAt:0,budgetEndsAt:600,goalsByPool:[{poolUUID:'Actor.P',targetHP:20}],activityIds:['A'],status:'running'},Manual:{id:'Manual',manual:true,status:'recording',activityIds:[]}},activities:{A:{id:'A',sessionId:'Old',actorUUID:'Actor.H',patientUUIDs:['Actor.P'],hpPoolUUIDs:['Actor.P'],state:'started',source:{messageId:'Check'},proof:{checkIds:['Check'],receiptIds:[]}}},clocks:{C:{id:'C',sessionId:'Old',state:'started',from:0,to:600,evidence:[]}}};
 const raw={_id:'ROOT000000000001',ownership:{default:0,G:3},flags:{[M]:{explorationLedger:structuredClone(seed)}},pages:[]};
 let sequence=0,alter=ack=>ack;const requests=[];
 function client(name){
  const user={id:'G',isGM:true},users=new Map([['G',user]]);users.activeGM=user;
  const game={user,users,settings:{get:()=>rootUUID},socket:{id:name,connected:true,on:()=>{},off:()=>{},emit:(_event,request,send)=>{
   requests.push(structuredClone(request));const {type,action,operation}=request;let result=[],error;
   if(type==='JournalEntry'&&action==='get')result=[structuredClone(raw)];
   else if(type==='JournalEntryPage'&&action==='create'){
    const page=structuredClone(operation.data[0]);
    if(raw.pages.some(p=>p._id===page._id))error={class:'ServerError',message:`The _id [${page._id}] already exists within the parent collection: JournalEntry [${raw._id}] pages`};
    else{raw.pages.push(page);result=[structuredClone(page)]}
   }else throw Error('unexpected-server-write');
   const ack={type,action,operation:structuredClone(operation),userId:'G',broadcast:false,result,...(error?{error}:{})};
   queueMicrotask(()=>send(action==='create'?alter(ack):ack));
  }}};
  const store=createDocumentStore({game,writerClientId:name,nonce:()=>`nonce-${++sequence}`}),ledger=createLedger({...store,isAuthority:()=>true,identity:()=>({userId:'G',clientNonce:name})});
  return {store,ledger,setup:createLedgerSetup({game,store,ledger,writerClientId:name}),game};
 }
 return {seed,raw,requests,client,setACK:fn=>{alter=fn},M};
}

test('controlled migration preserves genesis and flags then quarantines legacy work in a separate revision',async()=>{
 const f=await migrationFixture(),c=f.client('setup');const result=await c.setup.initialize(confirmations);
 assert.equal(result.state,'ready');assert.equal(f.raw.pages.length,2);
 const genesis=decodeRevision(f.raw.pages[0].flags[f.M].explorationRevision);assert.deepEqual(genesis.seed,f.seed);assert.equal(genesis.sourceDigest,createHash('sha256').update(canonicalJSON(f.seed)).digest('hex'));assert.deepEqual(f.raw.flags[f.M].explorationLedger,f.seed);
 const state=await c.store.read();assert.equal(state.sessions.Old.status,'paused');assert.equal(state.sessions.Old.driver,undefined);assert.equal(state.activities.A.state,'uncertain');assert.equal(state.clocks.C.state,'uncertain');
 assert.deepEqual(state.activities.A.source,f.seed.activities.A.source);assert.deepEqual(state.activities.A.patientUUIDs,f.seed.activities.A.patientUUIDs);assert.deepEqual(state.activities.A.proof,f.seed.activities.A.proof);assert.deepEqual(state.sessions.Manual,f.seed.sessions.Manual);
 assert.equal((await c.setup.status()).state,'ready');await assert.rejects(c.ledger.resumeSession('Old',{cursorAt:0}),/unresolved-evidence-no-replay/);
});

test('unknown genesis outcome remains migration-required until explicit administrative continuation',async()=>{
 const f=await migrationFixture(),c=f.client('setup');f.setACK(ack=>({...ack,result:[]}));
 await assert.rejects(c.setup.initialize(confirmations),/acknowledgement/);assert.equal(f.raw.pages.length,1);assert.equal((await c.store.read()).sessions.Old.status,'running');
 const restarted=f.client('restart');assert.equal((await restarted.setup.status()).state,'migration-required');assert.equal(f.raw.pages.length,1);
 f.setACK(ack=>ack);assert.equal((await restarted.setup.initialize(confirmations)).state,'ready');assert.equal(f.raw.pages.length,2);assert.equal((await restarted.store.read()).sessions.Old.driver,undefined);
});

test('unknown quarantine outcome is not retried or converted to a native continuation',async()=>{
 const f=await migrationFixture(),c=f.client('setup');f.setACK(ack=>decodeRevision(ack.result[0]?.flags[f.M].explorationRevision).revision===1?{...ack,result:[]}:ack);
 await assert.rejects(c.setup.initialize(confirmations),/acknowledgement/);assert.equal(f.raw.pages.length,2);
 const state=await c.store.read();assert.equal(state.sessions.Old.status,'paused');assert.equal(state.activities.A.executor,undefined);assert.equal(state.sessions.Old.driver,undefined);
 const restarted=f.client('restart');assert.equal((await restarted.setup.status()).state,'ready');f.setACK(ack=>ack);
 assert.equal((await restarted.setup.initialize(confirmations)).state,'ready');assert.equal(f.raw.pages.length,2);
});

test('simultaneous controlled setup clients with different epochs cannot both initialize',async()=>{
 const f=await migrationFixture(),one=f.client('one'),two=f.client('two');
 const outcomes=await Promise.allSettled([one.setup.initialize(confirmations),two.setup.initialize(confirmations)]);
 assert.equal(outcomes.filter(r=>r.status==='fulfilled').length,1);assert.equal(outcomes.filter(r=>r.status==='rejected'&&r.reason.message==='genesis-epoch-conflict').length,1);
 assert.equal((await one.store.read()).sessions.Old.status,'paused');assert.equal(f.raw.pages.length,2);assert.deepEqual(f.raw.flags[f.M].explorationLedger,f.seed);
});

test('already initialized legacy running history is not reported ready or automatically quarantined by status',async()=>{
 const f=await migrationFixture(),c=f.client('setup');
 await c.store.initialize({...confirmations,epoch:'epoch',expectedSourceDigest:createHash('sha256').update(canonicalJSON(f.seed)).digest('hex'),allowStoppedLegacy:true});
 assert.equal((await c.setup.status()).state,'migration-required');assert.equal(f.raw.pages.length,1);assert.equal((await c.store.read()).sessions.Old.status,'running');
 await assert.rejects(c.setup.initialize({...confirmations,clientsReloaded:false}),/setup-confirmations-required/);assert.equal(f.raw.pages.length,1);
});

test('a setup delayed before quarantine cannot pause a newly resumed protocol session',async()=>{
 const f=await migrationFixture(),legacy=f.raw.flags[f.M].explorationLedger;legacy.activities={};legacy.clocks={};legacy.sessions.Old.activityIds=[];
 const stale=f.client('stale');await stale.store.initialize({...confirmations,allowStoppedLegacy:true,epoch:'epoch',expectedSourceDigest:createHash('sha256').update(canonicalJSON(legacy)).digest('hex')});
 let enter,release;const entered=new Promise(resolve=>{enter=resolve}),gate=new Promise(resolve=>{release=resolve}),quarantine=stale.ledger.quarantineLegacySessions;
 stale.ledger.quarantineLegacySessions=async()=>{enter();await gate;return quarantine()};
 const pending=stale.setup.initialize(confirmations);await entered;
 const current=f.client('current');await current.setup.initialize(confirmations);const resumed=await current.ledger.resumeSession('Old',{cursorAt:0});
 release();assert.equal((await pending).state,'ready');assert.deepEqual(await current.ledger.getSession('Old'),resumed);assert.equal(resumed.driver.clientNonce,'current');assert.equal(resumed.status,'running');
});

test('controlled migration keeps manual recording appendable without adopting an automatic driver',async()=>{
 const f=await migrationFixture(),c=f.client('setup');await c.setup.initialize(confirmations);
 const a=await c.ledger.insertActivity({id:'ManualEntry',sessionId:'Manual',providerId:'manual',actorUUID:'Actor.H',patientUUIDs:[],hpPoolUUIDs:[],startedAt:0,endsAt:600,state:'awaiting-evidence',source:{manual:true,type:'user-record'}});
 assert.equal(a.state,'awaiting-evidence');const manual=await c.ledger.getSession('Manual');assert.equal(manual.status,'recording');assert.equal(manual.driver,undefined);assert.deepEqual(manual.activityIds,['ManualEntry']);
});
