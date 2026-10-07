import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {createDocumentStore,isActiveGM} from '../../scripts/exploration/document-store.mjs';
import {canonicalJSON,normalizeLedger} from '../../scripts/exploration/revision-codec.mjs';

const M='pf2e-third-party-automation',ROOT='JournalEntry.ROOT000000000001',OTHER='JournalEntry.OTHER00000000001';
const confirmations={issuersStopped:true,clientsReloaded:true,recoveryDisabled:true};
const seed=()=>({sessions:{S:{id:'S',status:'paused'}},activities:{},clocks:{}});
const fingerprint=s=>createHash('sha256').update(canonicalJSON(normalizeLedger(s))).digest('hex');
function fixture({configured=ROOT,state=seed(),timeoutMs=100,player=false}={}){
  let setting=configured,sequence=0,alterACK=null,cacheCalls=0,setHook=null;
  const users=new Map([['G',{id:'G',isGM:true}],['G2',{id:'G2',isGM:true}],['P',{id:'P',isGM:false}]]);users.activeGM=users.get('G');
  const docs=new Map();const document=id=>({_id:id,ownership:{default:0,G:3},pages:[],flags:{[M]:{explorationLedger:structuredClone(state)}}});
  docs.set(ROOT,document(ROOT.split('.')[1]));docs.set(OTHER,document(OTHER.split('.')[1]));
  const requests=[],settingsWrites=[],listeners=new Map();
  const game={user:users.get(player?'P':'G'),users,settings:{get:()=>setting,set:async(m,k,value)=>{settingsWrites.push(value);if(setHook)return setHook(value);setting=value;return value}},socket:{id:'socket-1',connected:true,
    on:(event,fn)=>{if(!listeners.has(event))listeners.set(event,new Set());listeners.get(event).add(fn)},
    off:(event,fn)=>listeners.get(event)?.delete(fn),
    emit:(event,request,callback)=>{
      assert.equal(event,'modifyDocument');requests.push(structuredClone(request));const userId=game.user.id;
      queueMicrotask(()=>{
        const {type,action,operation:op}=request;let result=[],error;
        if(type==='JournalEntry'&&action==='get'){const found=docs.get('JournalEntry.'+op.query._id);if(found)result=[structuredClone(found)]}
        else if(type==='JournalEntry'&&action==='create'){
          const data=structuredClone(op.data[0]);data._id='NEWROOT000000001';data.ownership[userId]=3;data.pages??=[];docs.set('JournalEntry.'+data._id,data);result=[structuredClone(data)];
        }else if(type==='JournalEntryPage'&&action==='create'){
          const root=docs.get(op.parentUuid);assert.ok(root);assert.equal(op.keepId,true);assert.equal(op.data.length,1);
          const data=structuredClone(op.data[0]);
          if(root.pages.some(p=>p._id===data._id))error={class:'ServerError',message:`The _id [${data._id}] already exists within the parent collection: JournalEntry [${root._id}] pages`};
          else{data.ownership[userId]=3;root.pages.push(data);result=[structuredClone(data)]}
        }else throw Error('Unexpected document mutation');
        const ack={type,action,operation:structuredClone(op),broadcast:op.broadcast,userId,result,...(error?{error}:{})};
        if(alterACK)alterACK(ack,callback,request);else callback(ack);
      });
    }}};
  const store=createDocumentStore({game,writerClientId:'client-1',nonce:()=>`nonce-${++sequence}`,timeoutMs,
    JournalEntry:{create:()=>{cacheCalls++;throw Error('native-cache-create-forbidden')}},fromUuid:()=>{cacheCalls++;throw Error('cached-read-forbidden')}});
  const initialize=(options={})=>store.initialize({...confirmations,epoch:'epoch-1',expectedSourceDigest:fingerprint(state),...options});
  return {game,store,docs,requests,settingsWrites,initialize,getSetting:()=>setting,setSetting:value=>{setting=value},cacheCalls:()=>cacheCalls,
    setACK:fn=>{alterACK=fn},setSettingHook:fn=>{setHook=fn},disconnect:()=>{game.socket.connected=false;for(const fn of listeners.get('disconnect')??[])fn('transport close')},listenerCount:()=>[...listeners.values()].reduce((n,set)=>n+set.size,0)};
}

test('missing configuration is readonly empty but never exposes mutable write or creates a root',async()=>{
  const f=fixture({configured:''});assert.deepEqual(await f.store.read(),{sessions:{},activities:{},clocks:{}});assert.equal(f.store.write,undefined);
  await assert.rejects(f.store.status(),/root-not-configured/);await assert.rejects(f.store.transact(()=>{}),/root-not-configured/);
  assert.equal(f.requests.length,0);assert.equal(f.settingsWrites.length,0);assert.equal(f.cacheCalls(),0);
});
test('legacy reads use exact authenticated server query and return detached records',async()=>{
  const f=fixture();const state=await f.store.read();assert.deepEqual(state,seed());state.sessions.S.status='wrong';
  assert.equal(f.docs.get(ROOT).flags[M].explorationLedger.sessions.S.status,'paused');assert.equal(f.cacheCalls(),0);
  assert.deepEqual(f.requests,[{type:'JournalEntry',action:'get',operation:{query:{_id:'ROOT000000000001'},broadcast:false}}]);
});
test('configured missing document diagnoses an absent root instead of returning legacy empty',async()=>{
  const f=fixture();f.docs.delete(ROOT);await assert.rejects(f.store.read(),/root-not-found/);assert.equal(f.requests.length,1);
});
for(const [name,alter] of Object.entries({type:a=>{a.type='Actor'},action:a=>{a.action='create'},user:a=>{a.userId='P'},broadcast:a=>{a.broadcast=true},query:a=>{a.operation.query._id='OTHER00000000001'},id:a=>{a.result[0]._id='OTHER00000000001'},multiple:a=>{a.result.push(a.result[0])},error:a=>{a.error={message:'denied'}},batch:a=>{a.results=[]}}))test(`server read rejects ${name} ACK provenance`,async()=>{
  const f=fixture();f.setACK((ack,send)=>{alter(ack);send(ack)});await assert.rejects(f.store.read(),/acknowledgement|root-document/);
});
for(const [name,ownership] of [['public-default',{default:1,G:3}],['player-owner',{default:0,G:3,P:3}],['player-observer',{default:0,G:3,P:2}],['unknown-owner',{default:0,missing:3}]])test(`${name} root is refused`,async()=>{
  const f=fixture();f.docs.get(ROOT).ownership=ownership;await assert.rejects(f.store.read(),/root-not-private|root-not-approved/);
});
test('other GM ownership and explicit player NONE are permitted write restrictions',async()=>{
  const f=fixture();f.docs.get(ROOT).ownership={default:0,G:3,G2:3,P:0};assert.deepEqual(await f.store.read(),seed());
});
test('root removal or replacement after a read invalidates the pinned runtime',async()=>{
  for(const changed of ['',OTHER]){const f=fixture();await f.store.read();f.setSetting(changed);await assert.rejects(f.store.read(),/root.*changed/);await assert.rejects(f.store.transact(()=>{}),/root.*changed/)}
});
test('only the active GM can write or provision',async()=>{
  const f=fixture({player:true});assert.equal(isActiveGM(f.game),false);
  const before=f.requests.length;await assert.rejects(f.initialize(),/active-gm/);await assert.rejects(f.store.transact(()=>{}),/active-gm/);await assert.rejects(f.store.provision(confirmations),/active-gm/);assert.equal(f.requests.length,before);
  f.game.user=f.game.users.get('G2');await assert.rejects(f.initialize(),/active-gm/);
});
for(const configured of [ROOT,''])for(const operation of ['read','status','inspect'])test(`player ${operation} is refused before reading ${configured?'configured':'unconfigured'} storage`,async()=>{
  const f=fixture({player:true,configured});await assert.rejects(f.store[operation](),/gm-read-required/);
  assert.equal(f.requests.length,0);assert.equal(f.settingsWrites.length,0);assert.equal(f.cacheCalls(),0);
});
test('a nonactive GM can read, describe and inspect the private ledger',async()=>{
  const f=fixture();f.game.user=f.game.users.get('G2');assert.equal(isActiveGM(f.game),false);
  assert.deepEqual(await f.store.read(),seed());assert.equal((await f.store.status()).initialized,false);assert.deepEqual((await f.store.inspect()).state,seed());
  assert.equal(f.requests.length,3);assert.equal(f.requests.every(request=>request.action==='get'),true);
  await assert.rejects(f.store.transact(()=>{}),/active-gm-required/);assert.equal(f.requests.length,3);
});
test('truthy nonboolean GM role cannot read the private ledger',async()=>{
  const f=fixture();f.game.user.isGM=1;await assert.rejects(f.store.read(),/gm-read-required/);assert.equal(f.requests.length,0);
});
for(const operation of ['read','status','inspect'])for(const timing of ['before-ack','after-ack','after-read'])test(`${operation} refuses same-user GM demotion ${timing}`,async()=>{
  const f=fixture();f.setACK((ack,send)=>{
    if(timing==='before-ack')f.game.user.isGM=false;
    send(ack);
    if(timing==='after-ack')f.game.user.isGM=false;
    if(timing==='after-read')queueMicrotask(()=>{f.game.user.isGM=false});
  });
  await assert.rejects(f.store[operation](),/gm-read-required/);assert.equal(f.requests.length,1);assert.equal(f.listenerCount(),0);
});
for(const operation of ['read','status','inspect'])for(const change of ['demotion','other-gm','same-id-replacement'])test(`${operation} rejects ${change} during revision projection after its ACK`,async t=>{
  const f=fixture();await f.initialize();const before=f.requests.length,subtle=globalThis.crypto.subtle,digest=subtle.digest;
  let entered,release;const boundary=new Promise(resolve=>{entered=resolve}),wait=new Promise(resolve=>{release=resolve});
  t.after(()=>{release();subtle.digest=digest});subtle.digest=async(...args)=>{entered();await wait;return digest.apply(subtle,args)};
  const pending=f.store[operation]();pending.catch(()=>{});await boundary;assert.equal(f.requests.length,before+1);assert.equal(f.listenerCount(),0);
  if(change==='demotion')f.game.user.isGM=false;else if(change==='other-gm')f.game.user=f.game.users.get('G2');else f.game.user={...f.game.user};
  release();await assert.rejects(pending,change==='demotion'?/gm-read-required/:/document-user-changed/);
});
for(const key of Object.keys(confirmations))test(`setup requires explicit ${key} confirmation`,async()=>{
  const f=fixture({configured:''});await assert.rejects(f.store.provision({...confirmations,[key]:false}),/setup-confirmations/);
  await assert.rejects(f.store.select({rootUUID:ROOT,...confirmations,[key]:false}),/setup-confirmations/);
  f.setSetting(ROOT);await assert.rejects(f.initialize({[key]:false}),/setup-confirmations/);assert.equal(f.requests.length,0);assert.equal(f.settingsWrites.length,0);
});
test('initialization and transactions create one immutable page with native request options',async()=>{
  const f=fixture();const status=await f.initialize();assert.equal(status.initialized,true);assert.equal(status.epoch,'epoch-1');
  const value=await f.store.transact((state,context)=>{state.activities.A={state:'completing',revision:context.revision};return 'grant'});assert.equal(value,'grant');
  assert.equal((await f.store.read()).activities.A.state,'completing');assert.deepEqual(f.docs.get(ROOT).flags[M].explorationLedger,seed());
  const creates=f.requests.filter(r=>r.action==='create');assert.equal(creates.length,2);
  for(const request of creates){assert.equal(request.type,'JournalEntryPage');assert.equal(request.operation.parentUuid,ROOT);assert.equal(request.operation.keepId,true);assert.equal(request.operation.broadcast,false);assert.equal(request.operation.render,false);assert.equal(request.operation.renderSheet,false);assert.equal(request.operation.data.length,1)}
  assert.equal(creates[0].operation.data[0]._id,'er00000000000000');assert.equal(creates[1].operation.data[0]._id,'er00000000000001');assert.equal(f.cacheCalls(),0);assert.equal(f.listenerCount(),0);
});
test('migration refuses a running automatic legacy session and unexpected source hash',async()=>{
  const state=seed();state.sessions.S.status='running';const f=fixture({state});await assert.rejects(f.initialize(),/running/);
  const g=fixture();await assert.rejects(g.initialize({expectedSourceDigest:'0'.repeat(64)}),/source/);assert.equal([...f.requests,...g.requests].filter(r=>r.action==='create').length,0);
});
test('explicit provisioning creates a fresh marked root and configures it without starting or initializing',async()=>{
  const f=fixture({configured:''});const result=await f.store.provision(confirmations);assert.equal(result.rootUUID,'JournalEntry.NEWROOT000000001');assert.equal(result.initialized,false);
  assert.equal(f.getSetting(),result.rootUUID);assert.equal(f.docs.get(result.rootUUID).pages.length,0);
  const creates=f.requests.filter(r=>r.action==='create');assert.equal(creates.length,1);assert.equal(creates[0].type,'JournalEntry');assert.equal(creates[0].operation.broadcast,false);assert.equal(creates[0].operation.keepId,undefined);assert.equal(creates[0].operation.data[0]._id,undefined);assert.equal(creates[0].operation.data[0].ownership.default,0);
  assert.equal(creates[0].operation.data[0].flags[M].explorationLedgerRoot.schemaVersion,1);assert.deepEqual(await f.store.read(),{sessions:{},activities:{},clocks:{}});assert.equal(f.cacheCalls(),0);
});
test('provision refuses existing configuration and preserves a competing newly configured root',async()=>{
  const existing=fixture();await assert.rejects(existing.store.provision(confirmations),/root-already-configured/);assert.equal(existing.requests.length,0);
  const f=fixture({configured:''});f.setACK((ack,send,request)=>{if(request.action==='create')f.setSetting(OTHER);send(ack)});
  await assert.rejects(f.store.provision(confirmations),error=>{assert.match(error.message,/root-config-conflict/);assert.equal(error.rootUUID,'JournalEntry.NEWROOT000000001');return true});
  assert.equal(f.getSetting(),OTHER);assert.equal(f.settingsWrites.length,0);assert.ok(f.docs.has('JournalEntry.NEWROOT000000001'));
});
test('select validates an existing root then explicitly configures it without creating documents',async()=>{
  const f=fixture({configured:''});const result=await f.store.select({rootUUID:ROOT,...confirmations});assert.equal(result.rootUUID,ROOT);assert.equal(result.requiresReload,false);assert.equal(f.getSetting(),ROOT);assert.equal(f.requests.every(r=>r.action==='get'),true);
  const rejected=fixture({configured:''});rejected.docs.get(ROOT).ownership.P=3;await assert.rejects(rejected.store.select({rootUUID:ROOT,...confirmations}),/root-not-approved/);assert.equal(rejected.settingsWrites.length,0);
});
test('selecting a different configured root requires a new runtime and cannot silently move execution',async()=>{
  const f=fixture();await f.store.read();const selected=await f.store.select({rootUUID:OTHER,...confirmations});assert.equal(selected.requiresReload,true);assert.equal(f.getSetting(),OTHER);
  await assert.rejects(f.store.read(),/root.*changed|reload-required/);await assert.rejects(f.store.transact(()=>{}),/root.*changed|reload-required/);
});
test('read timeout rejects permanently and removes its disconnect listener',async()=>{
  const f=fixture({timeoutMs:10});let late;f.setACK((ack,send)=>{late=()=>send(ack)});await assert.rejects(f.store.read(),/timeout/);late();assert.equal(f.listenerCount(),0);assert.equal(f.requests.length,1);
});
test('persisted page timeout never releases a grant when its exact ACK arrives later',async()=>{
  const f=fixture({timeoutMs:10});await f.initialize();let late,native=0;
  f.setACK((ack,send,request)=>{if(request.type==='JournalEntryPage')late=()=>send(ack);else send(ack)});
  await assert.rejects(f.store.transact(state=>{state.activities.A={state:'completing'}}).then(()=>native++),/timeout/);
  late();await new Promise(resolve=>setImmediate(resolve));assert.equal(native,0);assert.equal(f.docs.get(ROOT).pages.length,2);assert.equal(f.listenerCount(),0);
});
test('disconnect expires a persisted request even after reconnect and a late ACK',async()=>{
  const f=fixture();await f.initialize();let late,native=0;
  f.setACK((ack,send,request)=>{if(request.type==='JournalEntryPage'){late=()=>send(ack);f.disconnect()}else send(ack)});
  await assert.rejects(f.store.transact(state=>{state.activities.A={state:'completing'}}).then(()=>native++),/disconnect/);
  f.game.socket.connected=true;late();await new Promise(resolve=>setImmediate(resolve));assert.equal(native,0);assert.equal(f.listenerCount(),0);
});
test('socket identity change before an ACK cannot authorize its continuation',async()=>{
  const f=fixture();await f.initialize();let native=0;
  f.setACK((ack,send,request)=>{if(request.type==='JournalEntryPage')f.game.socket.id='socket-2';send(ack)});
  await assert.rejects(f.store.transact(state=>{state.activities.A={state:'completing'}}).then(()=>native++),/socket.*changed|connection.*changed/);assert.equal(native,0);
});
test('authority and configured root changes during create cannot return a transaction result',async()=>{
  for(const change of [f=>{f.game.users.activeGM=f.game.users.get('G2')},f=>f.setSetting(OTHER)]){
    const f=fixture();await f.initialize();let native=0;f.setACK((ack,send,request)=>{if(request.type==='JournalEntryPage')change(f);send(ack)});
    await assert.rejects(f.store.transact(state=>{state.activities.A={state:'completing'}}).then(()=>native++),/active-gm|root.*changed/);assert.equal(native,0);
  }
});
test('empty page ACK stays unknown and never retries or consults cached documents',async()=>{
  const f=fixture();await f.initialize();f.setACK((ack,send,request)=>{if(request.type==='JournalEntryPage')ack.result=[];send(ack)});
  const before=f.requests.length;await assert.rejects(f.store.transact(s=>{s.activities.A={state:'completing'}}),/acknowledgement/);assert.equal(f.requests.length,before+2);assert.equal(f.cacheCalls(),0);
});
test('select refuses a damaged revision chain before changing configuration',async()=>{
  const f=fixture({configured:''});f.docs.get(ROOT).pages.push({_id:'er00000000000000',flags:{}});
  await assert.rejects(f.store.select({rootUUID:ROOT,...confirmations}),/revision/);assert.equal(f.settingsWrites.length,0);
});
test('read ACK without an exact query fails with a transport diagnostic',async()=>{
  const f=fixture();f.setACK((ack,send)=>{delete ack.operation.query;send(ack)});
  await assert.rejects(f.store.read(),/acknowledgement/);
});
test('provisioning timeout leaves its possible artifact unconfigured and does not retry after a late ACK',async()=>{
  const f=fixture({configured:'',timeoutMs:10});let late;f.setACK((ack,send)=>{late=()=>send(ack)});
  await assert.rejects(f.store.provision(confirmations),/timeout/);late();await new Promise(resolve=>setImmediate(resolve));
  assert.equal(f.getSetting(),'');assert.equal(f.settingsWrites.length,0);assert.equal(f.requests.length,1);assert.equal(f.docs.get('JournalEntry.NEWROOT000000001').pages.length,0);
});
test('a mismatching root creation marker cannot configure the returned document',async()=>{
  const f=fixture({configured:''});f.setACK((ack,send)=>{ack.result[0].flags[M].explorationLedgerRoot.setupNonce='unrelated';send(ack)});
  await assert.rejects(f.store.provision(confirmations),/acknowledgement-mismatch/);assert.equal(f.settingsWrites.length,0);
});
test('selection preserves competing settings changes discovered after its read',async()=>{
  const f=fixture({configured:''});f.setACK((ack,send)=>{f.setSetting(OTHER);send(ack)});
  await assert.rejects(f.store.select({rootUUID:ROOT,...confirmations}),/root-config-conflict/);assert.equal(f.getSetting(),OTHER);assert.equal(f.settingsWrites.length,0);
});
test('a rejected settings write never authorizes or deletes the created root',async()=>{
  const f=fixture({configured:''});f.setSettingHook(()=>{throw Error('setting-outcome-unknown')});
  await assert.rejects(f.store.provision(confirmations),error=>{assert.equal(error.message,'setting-outcome-unknown');assert.equal(error.rootUUID,'JournalEntry.NEWROOT000000001');return true});
  assert.equal(f.docs.has('JournalEntry.NEWROOT000000001'),true);assert.equal(f.requests.filter(r=>r.action==='create').length,1);
});

test('the wrapper permits stopped legacy migration only with every explicit setup prerequisite',async()=>{
 const state=seed();state.sessions.S.status='running';
 for(const key of Object.keys(confirmations)){
  const f=fixture({state});await assert.rejects(f.initialize({allowStoppedLegacy:true,[key]:false}),/setup-confirmations-required/);assert.equal(f.requests.length,0);
 }
 const f=fixture({state});await assert.rejects(f.initialize(),/legacy-automatic-session-running/);
 const initialized=await f.initialize({allowStoppedLegacy:true});assert.equal(initialized.revision,0);
 assert.deepEqual(await f.store.read(),state);assert.deepEqual(f.docs.get(ROOT).flags[M].explorationLedger,state);
});
