import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {createRevisionStore} from '../../scripts/exploration/revision-store.mjs';
import {canonicalJSON,normalizeLedger,revisionId,createGenesisRevision,createSuccessorRevision,decodeRevision} from '../../scripts/exploration/revision-codec.mjs';
import {MODULE_ID as M} from '../../scripts/exploration/schema.mjs';

const ROOT='JournalEntry.ROOT000000000001';
const seed=()=>({sessions:{S:{id:'S',status:'paused'}},activities:{A:{id:'A',state:'started'}},clocks:{}});
const sourceDigest=state=>createHash('sha256').update(canonicalJSON(normalizeLedger(state))).digest('hex');
function fixture(state=seed()) {
 const raw={_id:'ROOT000000000001',ownership:{default:0,G:3},pages:[],flags:{[M]:{explorationLedger:structuredClone(state)}}};
 let creates=0,reads=0,afterCreate;
 const requests=[];
 const readRoot=async uuid=>{assert.equal(uuid,ROOT);reads++;return structuredClone(raw)};
 async function createPage({rootUUID,page}) {
  creates++;requests.push(structuredClone({rootUUID,page}));
  const operation={parentUuid:rootUUID};
  if(raw.pages.some(p=>p._id===page._id))return {type:'JournalEntryPage',action:'create',broadcast:false,operation,userId:'G',error:{class:'ServerError',message:`The _id [${page._id}] already exists within the parent collection: JournalEntry [${raw._id}] pages`}};
  // This check+insert is one server operation, shared by otherwise independent clients.
  raw.pages.push(structuredClone(page));
  const ack={type:'JournalEntryPage',action:'create',broadcast:false,operation,userId:'G',result:[structuredClone(page)]};
  return afterCreate?afterCreate(ack):ack;
 }
 let sequence=0;
 function client(options={}){const clientId=`client-${++sequence}`;return createRevisionStore({getRootUUID:()=>ROOT,readRoot,createPage,isAuthority:()=>true,canUseRoot:()=>true,writerUserId:()=> 'G',writerClientId:()=>clientId,nonce:()=>`nonce-${++sequence}`,...options})}
 async function initialized(){const store=client();await store.initialize({epoch:'epoch-1',expectedSourceDigest:sourceDigest(state)});return store}
 return {raw,client,initialized,createPage,readRoot,requests,counts:()=>({creates,reads}),setAfterCreate:fn=>{afterCreate=fn}};
}

test('legacy state remains readable without creating a root or genesis',async()=>{
 const f=fixture(),s=f.client();assert.deepEqual(await s.read(),seed());assert.equal((await s.status()).initialized,false);assert.deepEqual(f.counts(),{creates:0,reads:2});
 await assert.rejects(s.transact(state=>{state.sessions.S.status='running'}),/protocol-not-initialized/);assert.equal(f.counts().creates,0);
});

test('explicit genesis pins the legacy source and successor returns a detached committed value',async()=>{
 const f=fixture(),s=await f.initialized();let returned;
 const result=await s.transact(state=>{state.activities.A.state='completing';returned=state.activities.A;return returned});
 result.state='wrong';returned.state='also-wrong';assert.equal((await s.read()).activities.A.state,'completing');
 const status=await s.status();assert.equal(status.initialized,true);assert.equal(status.rootUUID,ROOT);assert.equal(status.epoch,'epoch-1');assert.equal(status.revision,1);assert.equal(status.sourceDigest,sourceDigest(seed()));
 assert.equal(f.raw.pages.length,2);assert.equal(f.requests[1].page._id,'er00000000000001');
 assert.deepEqual(f.raw.flags[M].explorationLedger,seed());
});

test('two independent clients preserve both concurrent pure field changes',async()=>{
 const f=fixture();await f.initialized();const a=f.client(),b=f.client();
 await Promise.all([a.transact(s=>{s.sessions.X={id:'X',status:'recording'}}),b.transact(s=>{s.activities.B={id:'B',state:'confirmed'}})]);
 const state=await a.read();assert.equal(state.sessions.X.status,'recording');assert.equal(state.activities.B.state,'confirmed');assert.equal(f.raw.pages.length,3);assert.equal(f.counts().creates,4);
});

test('the same expected activity transition grants only one independent continuation',async()=>{
 const f=fixture();await f.initialized();let native=0;
 const acquire=store=>store.transact(s=>{if(s.activities.A.state!=='started')throw Error('state-conflict');s.activities.A.state='completing';return {activityId:'A'}}).then(()=>{native++});
 const outcomes=await Promise.allSettled([acquire(f.client()),acquire(f.client())]);
 assert.equal(outcomes.filter(r=>r.status==='fulfilled').length,1);assert.equal(outcomes.filter(r=>r.status==='rejected'&&r.reason.message==='state-conflict').length,1);assert.equal(native,1);assert.equal(f.raw.pages.length,2);
});

test('an unknown persisted claim never grants execution or rereads its nonce into success',async()=>{
 const f=fixture(),s=await f.initialized();f.setAfterCreate(()=>{throw Error('connection-lost')});const before=f.counts();let native=0;
 await assert.rejects(s.transact(state=>{state.activities.A.state='completing';return 'grant'}).then(()=>{native++}),/connection-lost/);
 assert.equal(native,0);assert.equal(f.counts().creates,before.creates+1);assert.equal(f.counts().reads,before.reads+1);assert.equal(f.raw.pages.length,2);
 f.setAfterCreate(null);await assert.rejects(f.client().transact(state=>{if(state.activities.A.state!=='started')throw Error('state-conflict');state.activities.A.state='completing'}),/state-conflict/);assert.equal(native,0);
});

function changeSavedMetadata(page,change){
 const metadata=decodeRevision(page.flags[M].explorationRevision);change(metadata);page.flags[M].explorationRevision=canonicalJSON(metadata);
}
const badAcks={
 empty:ack=>({...ack,result:[]}),multiple:ack=>({...ack,result:[ack.result[0],ack.result[0]]}),
 type:ack=>({...ack,type:'Actor'}),action:ack=>({...ack,action:'update'}),broadcast:ack=>({...ack,broadcast:true}),
 parent:ack=>({...ack,operation:{parentUuid:'JournalEntry.OTHER00000000001'}}),
 user:ack=>({...ack,userId:'OTHER'}),
 id:ack=>{ack.result[0]._id='er00000000000009';return ack},
 nonce:ack=>{changeSavedMetadata(ack.result[0],metadata=>{metadata.nonce='other'});return ack},
 digest:ack=>{changeSavedMetadata(ack.result[0],metadata=>{metadata.digest='0'.repeat(64)});return ack},
 content:ack=>{changeSavedMetadata(ack.result[0],metadata=>{metadata.changes.activities.A.state='confirmed'});return ack},
 error:ack=>({...ack,error:{class:'ServerError',message:'database-write-failed'}})
};
for(const [name,alter] of Object.entries(badAcks))test(`${name} acknowledgement cannot grant or trigger a retry`,async()=>{
 const f=fixture(),s=await f.initialized();f.setAfterCreate(alter);let native=0;const before=f.counts();
 await assert.rejects(s.transact(state=>{state.activities.A.state='completing';return 'grant'}).then(()=>{native++}));
 assert.equal(native,0);assert.equal(f.counts().creates,before.creates+1);assert.equal(f.counts().reads,before.reads+1);
});

test('an exact duplicate with the wrong parent is unknown instead of a pure retry',async()=>{
 const f=fixture(),s=await f.initialized();const before=f.counts();
 f.setAfterCreate(ack=>({...ack,result:[],error:{class:'ServerError',message:`The _id [${ack.result[0]._id}] already exists within the parent collection: JournalEntry [OTHER00000000001] pages`}}));
 await assert.rejects(s.transact(state=>{state.activities.A.state='completing'}));assert.equal(f.counts().creates,before.creates+1);
});

test('same-source genesis races converge without granting a transaction result',async()=>{
 const f=fixture(),options={epoch:'epoch-1',expectedSourceDigest:sourceDigest(seed())};const results=await Promise.all([f.client().initialize(options),f.client().initialize(options)]);
 assert.equal(results.every(r=>r.initialized===true&&r.revision===0),true);assert.equal(f.raw.pages.length,1);
});

test('genesis refuses an unexpected source, active automatic session, or conflicting epoch',async()=>{
 const f=fixture(),s=f.client();await assert.rejects(s.initialize({epoch:'epoch-1',expectedSourceDigest:'0'.repeat(64)}),/source/);assert.equal(f.counts().creates,0);
 f.raw.flags[M].explorationLedger.sessions.S.status='running';await assert.rejects(s.initialize({epoch:'epoch-1',expectedSourceDigest:sourceDigest(f.raw.flags[M].explorationLedger)}),/running/);assert.equal(f.counts().creates,0);
 f.raw.flags[M].explorationLedger=seed();await s.initialize({epoch:'epoch-1',expectedSourceDigest:sourceDigest(seed())});await assert.rejects(f.client().initialize({epoch:'other-epoch',expectedSourceDigest:sourceDigest(seed())}),/epoch/);
});

test('unknown genesis creation is not retried or converted to an executable result',async()=>{
 const f=fixture();f.setAfterCreate(()=>{throw Error('initialization-unknown')});await assert.rejects(f.client().initialize({epoch:'epoch-1',expectedSourceDigest:sourceDigest(seed())}),/initialization-unknown/);
 assert.equal(f.counts().creates,1);assert.equal(f.counts().reads,1);assert.equal(f.raw.pages.length,1);
});

for(const problem of ['missing-config','different-document','public-default','unapproved-root','absent-root-validator'])test(`${problem} blocks before any document create`,async()=>{
 const f=fixture(),options={};
 if(problem==='missing-config')options.getRootUUID=()=>null;
 if(problem==='different-document')f.raw._id='OTHER00000000001';
 if(problem==='public-default')f.raw.ownership.default=1;
 if(problem==='unapproved-root')options.canUseRoot=()=>false;
 if(problem==='absent-root-validator')options.canUseRoot=undefined;
 const reasons={'missing-config':/root-not-configured/,'different-document':/root-document-mismatch/,'public-default':/root-not-private/,'unapproved-root':/root-not-approved/,'absent-root-validator':/root-not-approved/};
 await assert.rejects(f.client(options).initialize({epoch:'epoch-1',expectedSourceDigest:sourceDigest(seed())}),reasons[problem]);assert.equal(f.counts().creates,0);
});

test('a pinned root cannot silently follow a changed configuration',async()=>{
 const f=fixture();let root=ROOT;const s=f.client({getRootUUID:()=>root});await s.read();root='JournalEntry.OTHER00000000001';
 await assert.rejects(s.read(),/root.*changed/);await assert.rejects(s.transact(()=>{}),/root.*changed/);assert.equal(f.counts().creates,0);
});

test('configuration and authority are checked again after awaiting the server read',async()=>{
 const f=fixture();await f.initialized();let authorized=true;
 const s=f.client({isAuthority:()=>authorized,readRoot:async uuid=>{const raw=await f.readRoot(uuid);authorized=false;return raw}});
 const before=f.counts().creates;await assert.rejects(s.transact(state=>{state.activities.A.state='completing'}),/authority|active-gm/);assert.equal(f.counts().creates,before);
});

test('authority lost after an exact acknowledgement withholds the continuation',async()=>{
 const f=fixture();await f.initialized();let authorized=true,native=0;const s=f.client({isAuthority:()=>authorized});f.setAfterCreate(ack=>{authorized=false;return ack});
 await assert.rejects(s.transact(state=>{state.activities.A.state='completing'}).then(()=>native++),/authority|active-gm/);assert.equal(native,0);assert.equal(f.raw.pages.length,2);
});

test('known head rollback and altered committed pages block reads and execution',async()=>{
 const f=fixture(),s=await f.initialized();await s.transact(state=>{state.activities.A.state='completing'});const last=f.raw.pages.pop();
 await assert.rejects(s.read(),/regression/);f.raw.pages.push(last);changeSavedMetadata(last,metadata=>{metadata.changes.activities.A.state='confirmed'});await assert.rejects(s.read(),/digest/);
});

test('missing all revision pages cannot make a previously initialized client see legacy state',async()=>{
 const f=fixture(),s=await f.initialized();f.raw.pages=[];await assert.rejects(s.read(),/regression|initialized/);assert.equal(f.counts().creates,1);
});

test('malformed reserved pages cannot be treated as an uninitialized legacy root',async()=>{
 const f=fixture();f.raw.pages.push({_id:revisionId(1),flags:{}});await assert.rejects(f.client().read(),/invalid-revision-page/);assert.equal(f.counts().creates,0);
});

test('asynchronous mutation callbacks cannot obtain a grant',async()=>{
 const f=fixture(),s=await f.initialized();const before=f.counts().creates;await assert.rejects(s.transact(async state=>{state.activities.A.state='completing'}),/synchronous/);assert.equal(f.counts().creates,before);
});

test('pure retries receive the current immutable commit identity for persisted permits',async()=>{
 const f=fixture();await f.initialized();const seen=[];
 const change=(name)=>(state,context)=>{
  assert.equal(Object.isFrozen(context),true);assert.equal(context.rootUUID,ROOT);assert.equal(context.epoch,'epoch-1');assert.match(context.previousDigest,/^[0-9a-f]{64}$/);
  seen.push(context.revision);state.activities[name]={state:'completing',permit:{...context}};return context.revision;
 };
 const committed=await Promise.all([f.client().transact(change('B')),f.client().transact(change('C'))]);
 assert.deepEqual(committed.sort(),[1,2]);assert.deepEqual(seen.sort(),[1,1,2]);const state=await f.client().read();
 assert.deepEqual([state.activities.B.permit.revision,state.activities.C.permit.revision].sort(),[1,2]);
});

test('a successful server acknowledgement may include native defaults without changing signed content',async()=>{
 const f=fixture(),s=await f.initialized();f.setAfterCreate(ack=>{Object.assign(ack.result[0],{_stats:{lastModifiedBy:'G'},sort:0});ack.result[0].ownership.G=3;return ack});
 assert.equal(await s.transact(state=>{state.activities.A.state='completing';return 'grant'}),'grant');
});

test('a transport timeout latches rejection even if a matching success arrives later',async()=>{
 const f=fixture(),s=await f.initialized();let late,native=0;
 f.setAfterCreate(ack=>new Promise((resolve,reject)=>{late=()=>resolve(ack);reject(Error('transport-timeout'))}));
 await assert.rejects(s.transact(state=>{state.activities.A.state='completing'}).then(()=>native++),/transport-timeout/);late();await new Promise(resolve=>setImmediate(resolve));
 assert.equal(native,0);assert.equal(f.raw.pages.length,2);assert.equal((await s.read()).activities.A.state,'completing');
});

test('Promise-like callback results reject before any revision submission',async()=>{
 const f=fixture(),s=await f.initialized();const before=f.counts().creates;
 await assert.rejects(s.transact(()=>Promise.resolve('grant')),/synchronous/);assert.equal(f.counts().creates,before);
});

test('a configuration change while awaiting create cannot authorize the continuation',async()=>{
 const f=fixture();await f.initialized();let configured=ROOT,native=0;
 const s=f.client({getRootUUID:()=>configured});f.setAfterCreate(ack=>{configured='JournalEntry.OTHER00000000001';return ack});
 await assert.rejects(s.transact(state=>{state.activities.A.state='completing'}).then(()=>native++),/root.*changed/);assert.equal(native,0);assert.equal(f.raw.pages.length,2);
});

test('conflict exhaustion withholds the caller result while preserving other committed changes',async()=>{
 const f=fixture();await f.initialized();const rival=f.client();let native=0,calls=0;
 const s=f.client({maxConflicts:1,createPage:async request=>{await rival.transact(state=>{state.sessions.R={count:(state.sessions.R?.count??0)+1}});return f.createPage(request)}});
 await assert.rejects(s.transact(state=>{calls++;state.activities.A.state='completing'}).then(()=>native++),/conflict-limit/);
 assert.equal(native,0);assert.equal(calls,2);const state=await rival.read();assert.equal(state.sessions.R.count,2);assert.equal(state.activities.A.state,'started');
});

test('root approval revoked while awaiting create withholds the committed continuation',async()=>{
 const f=fixture();await f.initialized();let approved=true,native=0;
 const s=f.client({canUseRoot:()=>approved});f.setAfterCreate(ack=>{approved=false;return ack});
 await assert.rejects(s.transact(state=>{state.activities.A.state='completing'}).then(()=>native++),/root-not-approved/);assert.equal(native,0);assert.equal(f.raw.pages.length,2);
});

function deferred(){let resolve;const promise=new Promise(done=>{resolve=done});return {promise,resolve}}
async function overlappingFixture(){
 const f=fixture();let readGate,ackGate;
 const store=f.client({readRoot:async uuid=>{const snapshot=await f.readRoot(uuid),gate=readGate;readGate=null;if(gate){gate.captured.resolve();await gate.release.promise}return snapshot}});
 f.setAfterCreate(async ack=>{const gate=ackGate;ackGate=null;if(gate){gate.captured.resolve();await gate.release.promise}return ack});
 await store.initialize({epoch:'epoch-1',expectedSourceDigest:sourceDigest(seed())});
 const gate=()=>({captured:deferred(),release:deferred()});
 return {...f,store,holdRead:()=>readGate=gate(),holdACK:()=>ackGate=gate()};
}

for(const operation of ['read','status','transact'])test(`an overlapping ${operation} may complete from its earlier valid read without lowering the known head`,async()=>{
 const f=await overlappingFixture(),gate=f.holdRead(),attempts=[];
 const request=operation==='transact'?f.store.transact((state,context)=>{attempts.push(context.revision);state.activities.SLOW={state:'planned'};return 'slow-committed'}):f.store[operation]();
 const pending=request.then(value=>({value}),error=>({error}));await gate.captured.promise;
 await f.store.transact(state=>{state.sessions.FAST={status:'recording'}});gate.release.resolve();const outcome=await pending;
 assert.equal(outcome.error,undefined);
 if(operation==='read')assert.deepEqual(outcome.value,seed());
 if(operation==='status')assert.equal(outcome.value.revision,0);
 if(operation==='transact'){assert.equal(outcome.value,'slow-committed');assert.deepEqual(attempts,[1,2]);assert.equal((await f.store.read()).activities.SLOW.state,'planned')}
 assert.equal((await f.store.read()).sessions.FAST.status,'recording');
 f.raw.pages.pop();await assert.rejects(f.store.read(),/known-head-regression/);
});

test('an exact earlier successful ACK remains valid when a later commit proves its ancestry',async()=>{
 const f=await overlappingFixture(),gate=f.holdACK();let granted=0;
 const pending=f.store.transact(state=>{state.activities.A.state='completing';return 'grant-A'}).then(value=>{granted++;return {value}},error=>({error}));
 await gate.captured.promise;assert.equal(await f.store.transact(state=>{state.activities.B={state:'planned'};return 'commit-B'}),'commit-B');
 gate.release.resolve();const outcome=await pending;assert.equal(outcome.error,undefined);assert.equal(outcome.value,'grant-A');assert.equal(granted,1);assert.equal(f.raw.pages.length,3);
 f.raw.pages.pop();await assert.rejects(f.store.status(),/known-head-regression/);
});

test('a projection already hashing its snapshot tolerates a concurrently validated higher head',async t=>{
 const f=await overlappingFixture(),gate={captured:deferred(),release:deferred()},original=globalThis.crypto.subtle.digest;let armed=true;
 globalThis.crypto.subtle.digest=async function(...args){const result=await original.apply(this,args);if(armed){armed=false;gate.captured.resolve();await gate.release.promise}return result};
 t.after(()=>{globalThis.crypto.subtle.digest=original});
 const pending=f.store.read().then(value=>({value}),error=>({error}));await gate.captured.promise;
 await f.store.transact(state=>{state.sessions.FAST={status:'recording'}});gate.release.resolve();const outcome=await pending;
 assert.equal(outcome.error,undefined);assert.deepEqual(outcome.value,seed());f.raw.pages.pop();await assert.rejects(f.store.read(),/known-head-regression/);
});

test('a delayed cold snapshot cannot replace a different genesis validated while it was in flight',async()=>{
 const f=fixture();await f.initialized();const captured=deferred(),release=deferred();let held=true;
 const s=f.client({readRoot:async uuid=>{const snapshot=await f.readRoot(uuid);if(held){held=false;captured.resolve();await release.promise}return snapshot}});
 const pending=s.read().then(value=>({value}),error=>({error}));await captured.promise;
 const alternate=await createGenesisRevision({rootUUID:ROOT,epoch:'epoch-1',nonce:'replacement-genesis',writerUserId:'G',writerClientId:'other-client',seed:seed()});
 f.raw.pages[0].flags[M].explorationRevision=alternate;await s.read();release.resolve();const outcome=await pending;assert.match(outcome.error?.message??'',/known-head-replaced/);
});

test('an earlier ACK does not grant when its revision differs from the already validated chain',async()=>{
 const f=await overlappingFixture(),gate=f.holdACK();let granted=0;
 const pending=f.store.transact(state=>{state.activities.A.state='completing';return 'grant-A'}).then(value=>{granted++;return {value}},error=>({error}));await gate.captured.promise;
 const alternateState=seed();alternateState.sessions.OTHER={status:'recording'};
 const alternate=await createSuccessorRevision({previous:decodeRevision(f.raw.pages[0].flags[M].explorationRevision),state:seed(),nextState:alternateState,nonce:'replacement-revision',writerUserId:'G',writerClientId:'other-client'});
 f.raw.pages[1].flags[M].explorationRevision=alternate;await f.store.read();gate.release.resolve();const outcome=await pending;
 assert.match(outcome.error?.message??'',/known-head-replaced/);assert.equal(granted,0);
});

test('controlled stopped-issuer migration preserves the exact running seed without granting execution',async()=>{
 const legacy=seed();legacy.sessions.S.status='running';const f=fixture(legacy),store=f.client();
 await assert.rejects(store.initialize({epoch:'epoch-1',expectedSourceDigest:sourceDigest(legacy)}),/legacy-automatic-session-running/);
 const status=await store.initialize({epoch:'epoch-1',expectedSourceDigest:sourceDigest(legacy),allowStoppedLegacy:true});
 assert.equal(status.initialized,true);assert.equal(status.revision,0);assert.equal(status.sourceDigest,sourceDigest(legacy));
 assert.deepEqual(await store.read(),legacy);assert.deepEqual(f.raw.flags[M].explorationLedger,legacy);
 assert.deepEqual(decodeRevision(f.raw.pages[0].flags[M].explorationRevision).seed,legacy);assert.equal(f.raw.pages.length,1);
});

test('controlled legacy genesis with different epochs still accepts only one initializer',async()=>{
 const legacy=seed();legacy.sessions.S.status='running';const f=fixture(legacy);
 const outcomes=await Promise.allSettled(['one','two'].map(epoch=>f.client().initialize({epoch,expectedSourceDigest:sourceDigest(legacy),allowStoppedLegacy:true})));
 assert.equal(outcomes.filter(r=>r.status==='fulfilled').length,1);assert.equal(outcomes.filter(r=>r.status==='rejected'&&r.reason.message==='genesis-epoch-conflict').length,1);
 assert.deepEqual(f.raw.flags[M].explorationLedger,legacy);assert.equal(f.raw.pages.length,1);
});
