import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createExplorationRuntime} from '../../scripts/exploration/runtime.mjs';
import {createDocumentStore} from '../../scripts/exploration/document-store.mjs';
import {createGenesisRevision,encodeRevision,revisionId} from '../../scripts/exploration/revision-codec.mjs';
import {MODULE_ID} from '../../scripts/exploration/schema.mjs';

const rootUUID='JournalEntry.ROOT000000000001';
const seed=()=>({sessions:{S:{id:'S',status:'paused',manual:false,actorUUIDs:[],startedAt:0,cursorAt:0,budgetEndsAt:7200,maxActivities:100,activityIds:['A'],goalsByPool:[{poolUUID:'Actor.Patient',targetHP:20}],revisionMarker:0}},activities:{A:{id:'A',sessionId:'S',providerId:'manual',actorUUID:'Actor.Healer',patientUUIDs:['Actor.Patient'],hpPoolUUIDs:['Actor.Patient'],startedAt:0,endsAt:600,durationSeconds:600,state:'confirmed',source:{manual:true,type:'user-record'},proof:{checkIds:[],resultIds:[],receiptIds:[],immunityIds:[]},options:{label:'手动行动'}}},clocks:{}});
function deferred(){let resolve;const promise=new Promise(done=>{resolve=done});return {promise,resolve}}
async function fixture(t,{actorUUIDs=[],selectedUUIDs=actorUUIDs,renderContext=false}={}){
 const previousFoundry=globalThis.foundry;t.after(()=>{globalThis.foundry=previousFoundry});
 const renders=[],renderTasks=[],renderWaiters=[];let renderQueue=Promise.resolve();
 globalThis.foundry={applications:{api:{ApplicationV2:class{
  render(){if(!renderContext)return this;
   const task=renderQueue.then(async()=>{const context=await this._prepareContext(),html=await this._renderHTML(context);renders.push({context,html});this.rendered=true;return this});
   renderQueue=task.catch(()=>{});renderTasks.push(task);for(const waiter of renderWaiters.splice(0))waiter.resolve(task);return task;
  }
  async close(){this.rendered=false;return this}
 }}}};
 async function awaitRenders(){let completed=0;while(completed<renderTasks.length){const pending=renderTasks.slice(completed);completed=renderTasks.length;await Promise.all(pending)}}
 const state=seed();state.sessions.S.actorUUIDs=actorUUIDs;const genesis=await createGenesisRevision({rootUUID,epoch:'epoch',nonce:'genesis',writerUserId:'G',writerClientId:'setup',seed:state});
 const raw={_id:'ROOT000000000001',ownership:{default:0,G:3},flags:{[MODULE_ID]:{explorationLedger:structuredClone(state)}},pages:[{_id:revisionId(0),type:'text',name:'Exploration revision 0',ownership:{default:0},flags:{[MODULE_ID]:{explorationRevision:encodeRevision(genesis)}}}]};
 const gm={id:'G',isGM:true,active:true,getFlag:(_module,key)=>key==='explorationSession'?'S':null,setFlag:async()=>{}};
 const other={id:'Other',isGM:true,active:true},player={id:'P',isGM:false,active:true},users=new Map([['G',gm],['Other',other],['P',player]]);users.activeGM=gm;
 let reads=0,writes=0,gate,serial=0;const actorReads=[],actors=new Map([...new Set([...actorUUIDs,...selectedUUIDs])].map(uuid=>{const id=uuid.split('.')[1];return [id,{id,uuid,name:id,items:[],system:{attributes:{hp:{value:10,max:30,temp:0}},resources:{focus:{value:0,max:0}}},getStatistic:()=>({rank:1,mod:7})}]}));
 const game={user:gm,users,settings:{get:()=>rootUUID},modules:new Map(),packs:new Map(),messages:new Map(),actors,scenes:new Map(),time:{worldTime:0},system:{id:'pf2e',version:'8.5.1'},pf2e:{actions:new Map()},socket:{id:'primary-socket',connected:true,on(){},off(){},emit(event,request,respond){
  assert.equal(event,'modifyDocument');const {type,action,operation}=request;
  let result=[],error;
  if(type==='JournalEntry'&&action==='get'){reads++;assert.deepEqual(operation.query,{_id:raw._id});result=[structuredClone(raw)]}
  else if(type==='JournalEntryPage'&&action==='create'){
   writes++;assert.equal(operation.parentUuid,rootUUID);assert.equal(operation.keepId,true);const page=structuredClone(operation.data[0]);
   if(raw.pages.some(saved=>saved._id===page._id))error={class:'ServerError',message:`The _id [${page._id}] already exists within the parent collection: JournalEntry [${raw._id}] pages`};
   else{raw.pages.push(page);result=[structuredClone(page)]}
  }else assert.fail('Unexpected document action');
  const ack={type,action,operation:structuredClone(operation),broadcast:false,userId:game.user.id,result,...error?{error}:{}};
  const current=action==='get'?gate:null;if(current){gate=null;current.captured.resolve();void current.release.promise.then(()=>respond(ack))}else queueMicrotask(()=>respond(ack));
 }}};
 const nativeCasts={addMatcher(){},addCapture(){},addActorUpdateMiddleware(){},addConsumePolicy(){},addObserver(){}};
 const hooks=new Map(),errors=[];let hookId=0;const Hooks={on(event,callback){if(!hooks.has(event))hooks.set(event,new Map());const id=++hookId;hooks.get(event).set(id,callback);return id},off(event,id){hooks.get(event)?.delete(id)}};
 const runtime=createExplorationRuntime({game,Hooks,fromUuid:async uuid=>{actorReads.push(uuid);return actors.get(uuid.split('.')[1])},nativeCasts,onError:error=>errors.push(error)});await runtime.bind({});
 const panel=await runtime.api.open(selectedUUIDs),peer=createDocumentStore({game,writerClientId:'peer',nonce:()=>`peer-${++serial}`});
 await awaitRenders();
 return {game,gm,other,raw,runtime,panel,peer,renders,errors,awaitRenders,resetRenders:()=>{renders.length=0;renderTasks.length=0},nextRender(){const waiter=deferred();renderWaiters.push(waiter);return waiter.promise},emitHook(event,...args){for(const callback of hooks.get(event)?.values()??[])callback(...args)},actorReads:()=>[...actorReads],counts:()=>({reads,writes}),resetCounts:()=>{reads=0;writes=0;actorReads.length=0},holdNextRead(){const pending={captured:deferred(),release:deferred()};gate=pending;return pending}};
}

test('public GM open prepares the first panel once with goals and history',async t=>{
 const f=await fixture(t,{selectedUUIDs:['Actor.Healer'],renderContext:true});
 assert.equal(f.renders.length,1);const {context,html}=f.renders[0];assert.equal(context.ledgerStatus.state,'ready');
 assert.deepEqual(context.actors.map(actor=>actor.actorUUID),['Actor.Healer']);assert.equal(context.data.session.goalsByPool[0].targetHP,20);
 assert.match(html,/data-recovery-activity-id="A"/);assert.match(html,/手动行动/);assert.equal(f.counts().writes,0);
});

test('public GM open prepares a closed panel once from the current server revision',async t=>{
 const f=await fixture(t,{selectedUUIDs:['Actor.Healer'],renderContext:true});await f.panel.close();
 await f.peer.transact(state=>{state.sessions.S.revisionMarker=1;state.sessions.S.goalsByPool[0].targetHP=25});f.resetCounts();f.resetRenders();
 assert.equal(await f.runtime.api.open(['Actor.Healer']),f.panel);await f.awaitRenders();
 assert.equal(f.renders.length,1);assert.deepEqual(f.actorReads(),['Actor.Healer']);assert.equal(f.counts().writes,0);
 const {context,html}=f.renders[0];assert.equal(context.ledgerStatus.revision,1);assert.equal(context.data.session.revisionMarker,1);assert.equal(context.data.session.goalsByPool[0].targetHP,25);
 assert.match(html,/data-recovery-activity-id="A"/);assert.match(html,/手动行动/);
});

test('public GM open prepares only the new selection when the old actor is outside the session roster',async t=>{
 const f=await fixture(t,{selectedUUIDs:['Actor.Healer','Actor.Selected'],renderContext:true});
 await f.runtime.api.open(['Actor.Healer']);await f.awaitRenders();f.resetCounts();f.resetRenders();
 assert.equal(await f.runtime.api.open(['Actor.Selected']),f.panel);await f.awaitRenders();
 assert.equal(f.renders.length,1);assert.deepEqual(f.actorReads(),['Actor.Selected']);assert.equal(f.counts().writes,0);
 const {context,html}=f.renders[0];assert.deepEqual(context.actors.map(actor=>actor.actorUUID),['Actor.Selected']);assert.deepEqual(context.data.session.actorUUIDs,[]);
 assert.match(html,/data-actor="Actor.Selected"/);assert.doesNotMatch(html,/data-actor="Actor.Healer"/);
});

for(const event of ['userConnected','updateUser'])test(`${event} still refreshes an existing panel from the current server revision`,{timeout:5000},async t=>{
 const f=await fixture(t,{selectedUUIDs:['Actor.Healer'],renderContext:true});await f.runtime.register({socket:null});
 await f.peer.transact(state=>{state.sessions.S.revisionMarker=1;state.sessions.S.goalsByPool[0].targetHP=25});f.resetCounts();f.resetRenders();
 const refreshed=f.nextRender();f.emitHook(event,f.gm);await refreshed;await f.awaitRenders();
 assert.equal(f.renders.length,1);assert.equal(f.renders[0].context.ledgerStatus.revision,1);assert.equal(f.renders[0].context.data.session.goalsByPool[0].targetHP,25);
 assert.deepEqual(f.actorReads(),['Actor.Healer']);assert.equal(f.counts().writes,0);assert.deepEqual(f.errors,[]);
});

test('public GM open retains another atomic runtime session and pending activity without taking over',async t=>{
 const f=await fixture(t,{selectedUUIDs:['Actor.Healer'],renderContext:true});
 const driver={userId:'G',clientNonce:'other-runtime',leaseNonce:'persisted-lease'};
 await f.peer.transact(state=>{state.sessions.S.status='running';state.sessions.S.driver=driver;state.activities.A.state='started'});f.resetCounts();f.resetRenders();
 assert.equal(await f.runtime.api.open(['Actor.Healer']),f.panel);await f.awaitRenders();
 const {context,html}=f.renders.at(-1);assert.equal(context.data.session.status,'running');assert.deepEqual(context.data.session.driver,driver);
 assert.equal(context.data.activities[0].state,'started');assert.match(html,/data-recovery-activity-id="A"/);
 const stored=await f.peer.read();assert.equal(stored.sessions.S.status,'running');assert.deepEqual(stored.sessions.S.driver,driver);assert.equal(stored.activities.A.state,'started');assert.equal(f.counts().writes,0);
});

test('public GM open rechecks authority after an awaited server read and performs no recovery write',async t=>{
 const f=await fixture(t,{selectedUUIDs:['Actor.Healer'],renderContext:true});f.resetCounts();f.resetRenders();
 const pending=f.holdNextRead(),opening=f.runtime.api.open(['Actor.Healer']);await pending.captured.promise;f.game.users.activeGM=f.other;pending.release.resolve();
 const panel=await opening;await f.awaitRenders();assert.equal(panel,f.panel);assert.equal(f.counts().writes,0);
 await assert.rejects(panel.act('resume',{}),/active-gm-required/);
});

test('one panel preparation reads one authenticated revision snapshot and retains goals and history',async t=>{
 const f=await fixture(t);f.resetCounts();const context=await f.panel._prepareContext();
 assert.equal(context.ledgerStatus.state,'ready');assert.equal(context.ledgerStatus.revision,0);assert.deepEqual(context.data.session.goalsByPool,[{poolUUID:'Actor.Patient',targetHP:20}]);assert.equal(context.data.activities[0].id,'A');
 const html=await f.panel._renderHTML(context);assert.match(html,/data-recovery-activity-id="A"/);assert.match(html,/手动行动/);
 assert.deepEqual(f.counts(),{reads:1,writes:0});
});

test('one UI context discovers overlapping selected and session actors once without changing either roster',async t=>{
 const f=await fixture(t,{actorUUIDs:['Actor.Healer','Actor.Patient'],selectedUUIDs:['Actor.Patient','Actor.Selected']});f.resetCounts();const context=await f.panel._prepareContext();
 assert.deepEqual(context.actors.map(actor=>actor.actorUUID),['Actor.Patient','Actor.Selected']);assert.deepEqual(context.data.actors.map(actor=>actor.actorUUID),['Actor.Healer','Actor.Patient']);
 assert.deepEqual(f.actorReads().sort(),['Actor.Healer','Actor.Patient','Actor.Selected']);
});

test('a concurrent successor cannot mix status and session revisions within one panel context',async t=>{
 const f=await fixture(t),pending=f.holdNextRead(),preparing=f.panel._prepareContext();await pending.captured.promise;
 await f.peer.transact(state=>{state.sessions.S.revisionMarker=1;state.sessions.S.goalsByPool[0].targetHP=25});pending.release.resolve();
 const earlier=await preparing;assert.equal(earlier.ledgerStatus.revision,0);assert.equal(earlier.data.session.revisionMarker,0);assert.equal(earlier.data.session.goalsByPool[0].targetHP,20);
 const next=await f.panel._prepareContext();assert.equal(next.ledgerStatus.revision,1);assert.equal(next.data.session.revisionMarker,1);assert.equal(next.data.session.goalsByPool[0].targetHP,25);
});

test('mutating a returned panel context cannot change the next server-backed context',async t=>{
 const f=await fixture(t),context=await f.panel._prepareContext();context.data.session.goalsByPool[0].targetHP=99;context.data.activities[0].state='started';context.ledgerStatus.revision=99;
 const next=await f.panel._prepareContext();assert.equal(next.ledgerStatus.revision,0);assert.equal(next.data.session.goalsByPool[0].targetHP,20);assert.equal(next.data.activities[0].state,'confirmed');
});

test('a previous UI snapshot cannot authorize resume after the active GM changes',async t=>{
 const f=await fixture(t);await f.panel._prepareContext();f.game.users.activeGM=f.other;f.resetCounts();
 await assert.rejects(f.panel.act('resume',{}),/active-gm-required/);assert.deepEqual(f.counts(),{reads:0,writes:0});
});

test('a transaction after preparing UI reads the current server revision and rechecks authority',async t=>{
 const f=await fixture(t);await f.panel._prepareContext();await f.peer.transact(state=>{state.sessions.S.revisionMarker=1});
 await f.peer.transact(state=>{assert.equal(state.sessions.S.revisionMarker,1);state.sessions.S.revisionMarker=2});assert.equal((await f.panel._prepareContext()).data.session.revisionMarker,2);
 const pending=f.holdNextRead(),transaction=f.peer.transact(state=>{state.sessions.S.revisionMarker=3});await pending.captured.promise;f.game.users.activeGM=f.other;const writes=f.counts().writes;pending.release.resolve();
 await assert.rejects(transaction,/active-gm-required/);assert.equal(f.counts().writes,writes);
});

test('resume after displaying an earlier UI snapshot rejects a session closed by another client',async t=>{
 const f=await fixture(t);await f.panel._prepareContext();await f.peer.transact(state=>{state.sessions.S.status='closed'});f.resetCounts();
 await assert.rejects(f.runtime.api.resume('S',{autoRun:false}),/paused-automatic-session-required/);assert.equal(f.counts().writes,0);assert.ok(f.counts().reads>0);
});

test('GM demotion while revision hashing is awaited rejects the prepared view',async t=>{
 const f=await fixture(t),entered=deferred(),release=deferred(),subtle=globalThis.crypto.subtle,original=subtle.digest;let first=true;
 t.after(()=>{release.resolve();subtle.digest=original});subtle.digest=async function(...args){if(first){first=false;entered.resolve();await release.promise}return original.apply(this,args)};
 f.resetCounts();const preparing=f.panel._prepareContext();preparing.catch(()=>{});await entered.promise;f.gm.isGM=false;release.resolve();
 await assert.rejects(preparing,/gm-read-required/);assert.deepEqual(f.counts(),{reads:1,writes:0});
});

for(const [label,ownership]of [['public default',{default:1,G:3}],['player observer',{default:0,G:3,P:2}]])test(`a new panel context refuses ${label} permissions instead of reusing its prior view`,async t=>{
 const f=await fixture(t);await f.panel._prepareContext();f.raw.ownership=ownership;f.resetCounts();const next=await f.panel._prepareContext();
 assert.equal(next.ledgerStatus.state,'blocked');assert.match(next.ledgerStatus.reason,/root-not-private|root-not-approved/);assert.equal(next.data,null);assert.equal(f.counts().writes,0);
});

test('known-head rollback remains blocked after preparing a newer panel context',async t=>{
 const f=await fixture(t);await f.peer.transact(state=>{state.sessions.S.revisionMarker=1});assert.equal((await f.panel._prepareContext()).ledgerStatus.revision,1);
 f.raw.pages.pop();const next=await f.panel._prepareContext();assert.equal(next.ledgerStatus.state,'blocked');assert.match(next.ledgerStatus.reason,/known-head-regression/);assert.equal(next.data,null);
});

test('changed historical revision content cannot be hidden by the previous panel view',async t=>{
 const f=await fixture(t);await f.panel._prepareContext();const metadata=JSON.parse(f.raw.pages[0].flags[MODULE_ID].explorationRevision);metadata.seed.sessions.S.goalsByPool[0].targetHP=99;
 f.raw.pages[0].flags[MODULE_ID].explorationRevision=encodeRevision(metadata);const next=await f.panel._prepareContext();assert.equal(next.ledgerStatus.state,'blocked');assert.match(next.ledgerStatus.reason,/revision-digest-mismatch/);assert.equal(next.data,null);
});
