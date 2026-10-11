import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createExplorationRuntime} from '../../scripts/exploration/runtime.mjs';
import {MODULE_ID} from '../../scripts/exploration/schema.mjs';

function fixture(){
 const gm={id:'G',isGM:true,active:true,getFlag:()=>null},calls=[],pauses=[];
 const users=new Map([['G',gm],['G2',{...gm,id:'G2'}],['P',{...gm,id:'P',isGM:false}]]);users.activeGM=gm;
 const game={user:gm,users,settings:{get:()=>''},modules:new Map(),packs:new Map(),messages:new Map(),actors:{},time:{worldTime:0},system:{id:'pf2e',version:'8.5.1'},socket:{id:'socket',connected:true,on(){},off(){},emit(){calls.push('emit');throw Error('unexpected-socket-write')}},pf2e:{actions:new Map()}};
 const nativeCasts={addMatcher(){},addCapture(){},addActorUpdateMiddleware(){},addConsumePolicy(){},addObserver(){}};
 game.paused=false;game.togglePause=(paused,options)=>{pauses.push({paused,options});game.paused=paused;return paused};
 return {game,calls,pauses,nativeCasts,Hooks:{on(){},off(){}},fromUuid:async()=>null};
}
function documentFixture(){
 const f=fixture(),hooks=new Map(),listeners=new Map();let configured='',raw;
 f.Hooks.on=(event,handler)=>{const list=hooks.get(event)??[];list.push(handler);hooks.set(event,list);return handler};
 f.game.settings={get:()=>configured,set:async(namespace,key,value)=>{configured=value;for(const handler of hooks.get('updateSetting')??[])handler({key:`${namespace}.${key}`})}};
 f.game.user.setFlag=async()=>{};
 f.game.socket.on=(event,handler)=>{const list=listeners.get(event)??new Set();list.add(handler);listeners.set(event,list)};
 f.game.socket.off=(event,handler)=>listeners.get(event)?.delete(handler);
 f.game.socket.emit=(event,request,...args)=>{
  if(event!== 'modifyDocument')return;
  f.calls.push(structuredClone(request));const respond=args[0],envelope={type:request.type,action:request.action,operation:request.operation,broadcast:false,userId:f.game.user.id};
  if(request.action==='get')return respond({...envelope,result:raw?[structuredClone(raw)]:[]});
  if(request.type==='JournalEntry'){
   raw={...structuredClone(request.operation.data[0]),_id:'ROOT000000000001',pages:[]};return respond({...envelope,result:[structuredClone(raw)]});
  }
  const page=request.operation.data[0];
  if(raw.pages.some(p=>p._id===page._id))return respond({...envelope,error:{class:'ServerError',message:`The _id [${page._id}] already exists within the parent collection: JournalEntry [${raw._id}] pages`}});
  raw.pages.push(structuredClone(page));respond({...envelope,result:[structuredClone(page)]});
 };
 return {...f,seed:state=>{raw={_id:'ROOT000000000001',ownership:{default:0},pages:[],flags:{[MODULE_ID]:{explorationLedger:state}}};configured='JournalEntry.ROOT000000000001'},emitHook:(event,...args)=>hooks.get(event)?.forEach(handler=>handler(...args))};
}

test('a new runtime binds its transaction identity and exposes controlled setup',async()=>{
 const f=fixture(),runtime=createExplorationRuntime(f);await runtime.bind({});
 assert.equal(typeof runtime.api.storage.status,'function');assert.equal(typeof runtime.api.storage.provision,'function');assert.equal(typeof runtime.api.storage.initialize,'function');assert.equal(typeof runtime.api.storage.select,'function');assert.equal(typeof runtime.api.takeover,'function');
 assert.equal(runtime.api.diagnostic().status,'setup-required');assert.equal((await runtime.api.storage.status()).state,'unconfigured');assert.equal(f.calls.length,0);
});

test('binding and restoring an unconfigured runtime do not create storage or issue native work',async()=>{
 const f=fixture(),runtime=createExplorationRuntime(f);await runtime.bind({});
 await assert.rejects(runtime.api.start({actorUUIDs:[],manual:true}),/exploration-ledger-setup-required/);
 await assert.rejects(runtime.api.storage.initialize({issuersStopped:true,clientsReloaded:true,recoveryDisabled:true}),/root-not-configured/);assert.equal(f.calls.length,0);
});

test('first root provisioning keeps the new runtime available for owner evidence',async()=>{
 const f=documentFixture(),runtime=createExplorationRuntime(f);await runtime.bind({});await runtime.register({socket:null});
 const result=await runtime.api.storage.provision({issuersStopped:true,clientsReloaded:true,recoveryDisabled:true});assert.equal(result.requiresReload,false);
 await runtime.api.storage.initialize({issuersStopped:true,clientsReloaded:true,recoveryDisabled:true});assert.equal(runtime.api.diagnostic().status,'ready');
 await assert.doesNotReject(runtime.ownerOperations.reconcile({id:'Absent',actorUUID:'Actor.None',providerId:'treat-wounds',options:{}}));
});

test('the panel resumes through the runtime encounter and PF2e identity guards',async t=>{
 const before=globalThis.foundry;t.after(()=>{globalThis.foundry=before});globalThis.foundry={applications:{api:{ApplicationV2:class{render(){return this}}}}};
 for(const mode of ['system','encounter']){
  const f=documentFixture();f.seed({sessions:{S:{id:'S',status:'paused',actorUUIDs:[],startedAt:0,cursorAt:0,budgetEndsAt:600,activityIds:[],goalsByPool:[]}},activities:{},clocks:{}});f.game.user.getFlag=()=> 'S';
  const runtime=createExplorationRuntime(f);await runtime.bind({});await runtime.api.storage.initialize({issuersStopped:true,clientsReloaded:true,recoveryDisabled:true});
  if(mode==='system')f.game.system.id='other';else f.game.combat={started:true};
  const panel=await runtime.api.open([]);await assert.rejects(panel.act('resume',{}),/encounter-or-system-unavailable/);
 }
});

test('PF2e 8.6 runtime can start and resume automatic recovery with its actual diagnostic version',async()=>{
 const f=documentFixture();f.game.system.version='8.6.0';
 const actor={id:'H',uuid:'Actor.H',name:'Healer',type:'character',items:[],testUserPermission:()=>true,system:{attributes:{hp:{value:20,max:20,temp:0}},resources:{focus:{value:0,max:0}}}};
 f.fromUuid=async uuid=>uuid===actor.uuid?actor:null;f.game.actors=new Map([['H',actor]]);
 const runtime=createExplorationRuntime(f);await runtime.bind({});
 await runtime.api.storage.provision({issuersStopped:true,clientsReloaded:true,recoveryDisabled:true});
 await runtime.api.storage.initialize({issuersStopped:true,clientsReloaded:true,recoveryDisabled:true});
 const session=await runtime.api.start({actorUUIDs:[actor.uuid],budgetSeconds:600,autoRun:false});assert.ok(session.id);
 assert.equal(runtime.api.diagnostic().nativeVersion,'8.6.0');
 await runtime.api.stop(session.id);await assert.doesNotReject(runtime.api.resume(session.id,{autoRun:false}));
});

async function automaticFixture(){
 const f=documentFixture(),actor={id:'H',uuid:'Actor.H',name:'Healer',type:'character',items:[],testUserPermission:()=>true,system:{attributes:{hp:{value:20,max:20,temp:0}},resources:{focus:{value:0,max:0}}}};
 f.fromUuid=async uuid=>uuid===actor.uuid?actor:null;f.game.actors=new Map([['H',actor]]);
 const runtime=createExplorationRuntime(f);await runtime.bind({});await runtime.api.storage.provision({issuersStopped:true,clientsReloaded:true,recoveryDisabled:true});await runtime.api.storage.initialize({issuersStopped:true,clientsReloaded:true,recoveryDisabled:true});
 return {...f,runtime,config:{actorUUIDs:[actor.uuid],budgetSeconds:600,autoRun:false}};
}

async function waitForSession(runtime,id,status){
 const deadline=performance.now()+2000;
 while(performance.now()<deadline){const data=await runtime.api.snapshot(id);if(data.session.status===status)return data;await new Promise(resolve=>setImmediate(resolve))}
 assert.fail(`runtime session did not become ${status}`);
}

async function afterRuntimeHooks(runtime,id){
 await new Promise(resolve=>setImmediate(resolve));
 for(let i=0;i<4;i++)await runtime.api.snapshot(id);
 return runtime.api.snapshot(id);
}

for(const status of ['complete','paused'])test(`ambient encounter and rest hooks preserve a ${status} automatic session`,async()=>{
 const f=await automaticFixture(),combat={started:false,previous:{round:0},round:0};f.game.combats=[combat];
 await f.runtime.register({socket:null});
 const s=await f.runtime.api.start({...f.config,autoRun:status==='complete'});
 if(status==='paused')await f.runtime.api.stop(s.id);
 const before=await waitForSession(f.runtime,s.id,status),writes=()=>f.calls.filter(request=>request.action==='create').length,originalWrites=writes();
 assert.equal(before.session.stopReason,status==='complete'?'goals-met':'user-stopped');assert.equal(f.game.paused,true);
 const events=[()=>{combat.started=true;combat.round=1;f.emitHook('updateCombat',combat)},()=>{combat.previous.round=1;combat.round=2;f.emitHook('updateCombat',combat)},()=>f.emitHook('pf2e.restForTheNight',f.game.actors.get('H'))];
 for(const emit of events){
  emit();const after=await afterRuntimeHooks(f.runtime,s.id);
  assert.deepEqual(after.session,before.session);assert.deepEqual(after.activities,before.activities);assert.deepEqual(after.clocks,before.clocks);assert.equal(writes(),originalWrites);
 }
 assert.equal(f.game.paused,true);assert.equal(f.game.time.worldTime,0);assert.equal(f.pauses.length,1);
});

test('an automatic recovery stops once when its encounter starts and ignores later combat rounds',async()=>{
 const f=await automaticFixture(),combat={started:false,previous:{round:0},round:0};f.game.combats=[combat];f.game.combat=combat;
 let timeAdvances=0,actorWrites=0;f.game.time.advance=async()=>{timeAdvances++};f.game.actors.get('H').update=async()=>{actorWrites++};
 await f.runtime.register({socket:null});
 const s=await f.runtime.api.start({...f.config,autoRun:true,waitForActivityFirstRound:true}),binding=await f.runtime.api.openActivityCheckpoint(s.id);
 assert.equal((await f.runtime.api.snapshot(s.id)).session.status,'running');assert.equal(f.game.paused,true);
 const writes=()=>f.calls.filter(request=>request.action==='create').length,initialWrites=writes();
 f.emitHook('updateCombat',combat);assert.equal((await afterRuntimeHooks(f.runtime,s.id)).session.status,'running');assert.equal(writes(),initialWrites);
 combat.started=true;combat.round=1;f.emitHook('updateCombat',combat);f.emitHook('updateCombat',combat);
 const stopped=await waitForSession(f.runtime,s.id,'paused');assert.equal(stopped.session.stopReason,'encounter-started');assert.equal(writes(),initialWrites+1);
 const beforeRounds=f.calls.length;
 for(const round of [2,3,4]){combat.previous.round=round-1;combat.round=round;f.emitHook('updateCombat',combat)}
 await new Promise(resolve=>setImmediate(resolve));assert.equal(f.calls.length,beforeRounds);
 f.emitHook('pf2e.restForTheNight',f.game.actors.get('H'));const after=await afterRuntimeHooks(f.runtime,s.id);
 assert.deepEqual(after.session,stopped.session);assert.equal(writes(),initialWrites+1);
 await assert.rejects(f.runtime.api.closeActivityCheckpoint(binding),/session-driver-required|recovery-session-stopped/);
 await assert.rejects(f.runtime.api.resume(s.id),/encounter-or-system-unavailable/);
 assert.deepEqual(after.activities,[]);assert.deepEqual(after.clocks,[]);assert.equal(timeAdvances,0);assert.equal(actorWrites,0);assert.equal(f.game.time.worldTime,0);assert.equal(f.game.paused,true);assert.equal(f.pauses.length,1);
});

test('automatic runtime broadcasts pause once and leaves stop and resume paused',async()=>{
 const f=await automaticFixture(),session=await f.runtime.api.start(f.config);
 assert.equal(f.game.paused,true);assert.deepEqual(f.pauses,[{paused:true,options:{broadcast:true}}]);
 await f.runtime.api.stop(session.id);await f.runtime.api.resume(session.id,{autoRun:false});assert.equal(f.pauses.length,1);assert.equal(f.game.paused,true);
 await f.runtime.api.stop(session.id);f.game.paused=false;await f.runtime.api.resume(session.id,{autoRun:false});assert.equal(f.pauses.length,2);assert.equal(f.game.paused,true);
});

test('opening the panel and manual recording preserve the current pause state',async t=>{
 const before=globalThis.foundry;t.after(()=>{globalThis.foundry=before});globalThis.foundry={applications:{api:{ApplicationV2:class{render(){return this}}}}};
 const f=await automaticFixture();await f.runtime.api.open(['Actor.H']);assert.equal(f.game.paused,false);assert.deepEqual(f.pauses,[]);
 assert.equal((await f.runtime.api.start({...f.config,manual:true})).status,'recording');assert.equal(f.game.paused,false);assert.deepEqual(f.pauses,[]);
});

for(const mode of ['throws','unconfirmed'])test(`runtime refuses to create a session when pause ${mode}`,async()=>{
 const f=await automaticFixture();f.game.togglePause=()=>{if(mode==='throws')throw Error('socket disconnected')};f.calls.length=0;
 await assert.rejects(f.runtime.api.start(f.config),/game-pause-failed/);assert.equal(f.calls.some(request=>request.action==='create'),false);
});

test('nonactive GMs cannot pause the game through automatic recovery',async()=>{
 const f=await automaticFixture();f.game.user=f.game.users.get('G2');
 await assert.rejects(f.runtime.api.start(f.config),/active-gm-required/);assert.equal(f.game.paused,false);assert.deepEqual(f.pauses,[]);
});

test('a user change inside the pause hook prevents session creation',async()=>{
 const f=await automaticFixture(),toggle=f.game.togglePause;f.game.togglePause=(...args)=>{toggle(...args);f.game.user=f.game.users.get('G2')};f.calls.length=0;
 await assert.rejects(f.runtime.api.start(f.config),/active-gm-required|exploration-user-changed/);assert.equal(f.calls.some(request=>request.action==='create'),false);
});

test('runtime initialization quarantines legacy automatic sessions without adopting their driver',async()=>{
 const f=documentFixture();f.seed({sessions:{S:{id:'S',status:'running',manual:false,actorUUIDs:[],startedAt:0,cursorAt:0,budgetEndsAt:600,activityIds:[],goalsByPool:[]}},activities:{},clocks:{}});f.game.user.getFlag=()=> 'S';
 const runtime=createExplorationRuntime(f);await runtime.bind({});assert.equal((await runtime.api.storage.status()).state,'legacy');
 const prepared=await runtime.api.storage.initialize({issuersStopped:true,clientsReloaded:true,recoveryDisabled:true});assert.equal(prepared.state,'ready');
 const data=await runtime.api.snapshot('S');assert.equal(data.session.status,'paused');assert.equal(data.session.stopReason,'legacy-migration-quarantine');assert.equal(data.session.driver,undefined);
});

test('public current-time recovery uses the active GM and the current world time',async()=>{
 const f=documentFixture();f.game.time.worldTime=-600;
 f.seed({sessions:{S:{id:'S',status:'paused',actorUUIDs:[],startedAt:-1200,cursorAt:-1200,budgetEndsAt:600,activityIds:[],goalsByPool:[]}},activities:{},clocks:{}});
 const runtime=createExplorationRuntime(f);await runtime.bind({});await runtime.api.storage.initialize({issuersStopped:true,clientsReloaded:true,recoveryDisabled:true});
 const view=await runtime.api.snapshot('S');assert.equal(view.canAcceptCurrentTime,true);assert.equal(view.worldTime,-600);
 f.game.user=f.game.users.get('G2');const count=f.calls.length;
 await assert.rejects(runtime.api.acceptCurrentTime('S',{expectedWorldTime:-600,note:'Reviewed'}),/active-gm-required/);assert.equal(f.calls.length,count);
 f.game.user=f.game.users.get('G');const result=await runtime.api.acceptCurrentTime('S',{expectedWorldTime:-600,note:'Reviewed',userId:'forged'});
 assert.equal(result.status,'closed');assert.equal(result.recoveryReview.userId,'G');assert.equal(result.recoveryReview.at,-600);assert.equal(f.game.time.worldTime,-600);
});

const privateState=()=>({sessions:{S:{id:'S',status:'paused',actorUUIDs:[],startedAt:0,cursorAt:0,budgetEndsAt:600,activityIds:[],goalsByPool:[],label:'private-session-marker'}},activities:{},clocks:{}});
for(const configured of [true,false])for(const previous of ['S',null])test(`player binding skips ${configured?'configured':'unconfigured'} storage and ${previous?'a prior':'an empty'} local session flag`,async()=>{
 const f=documentFixture();if(configured)f.seed(privateState());f.game.user=f.game.users.get('P');let sessionFlags=0;f.game.user.getFlag=(namespace,key)=>{if(key==='explorationSession')sessionFlags++;return previous};
 const runtime=createExplorationRuntime(f);await runtime.bind({});assert.equal(f.calls.length,0);assert.equal(sessionFlags,0);
 assert.equal(JSON.stringify(runtime.api.diagnostic()).includes('private-session-marker'),false);
});
for(const configured of [true,false])test(`a player ledger setting update only invalidates ${configured?'configured':'unconfigured'} local state`,async()=>{
 const f=documentFixture(),errors=[];if(configured)f.seed(privateState());f.game.user=f.game.users.get('P');f.game.user.getFlag=()=> 'S';
 const runtime=createExplorationRuntime({...f,onError:error=>errors.push(error)});await runtime.bind({});await runtime.register({socket:null});f.calls.length=0;
 await f.game.settings.set(MODULE_ID,'explorationLedgerUUID','JournalEntry.ROOT000000000001');await new Promise(resolve=>setImmediate(resolve));
 assert.equal(f.calls.length,0);assert.deepEqual(errors,[]);
});
for(const configured of [true,false])for(const operation of ['snapshot','status'])test(`player public ${operation} rejects without a ${configured?'configured':'unconfigured'} private GET`,async()=>{
 const f=documentFixture();if(configured)f.seed(privateState());f.game.user=f.game.users.get('P');f.game.user.getFlag=()=> 'S';
 const runtime=createExplorationRuntime(f);await runtime.bind({});assert.equal(f.calls.length,0);
 await assert.rejects(operation==='snapshot'?runtime.api.snapshot('S'):runtime.api.storage.status(),/gm-read-required/);assert.equal(f.calls.length,0);
});
test('a nonactive GM retains public storage and snapshot reads',async()=>{
 const f=documentFixture();f.seed(privateState());f.game.user=f.game.users.get('G2');f.game.user.getFlag=()=> 'S';
 const runtime=createExplorationRuntime(f);await runtime.bind({});assert.equal((await runtime.api.storage.status()).state,'legacy');
 assert.equal((await runtime.api.snapshot('S')).session.label,'private-session-marker');await assert.rejects(runtime.api.start({actorUUIDs:[],manual:true}),/active-gm-required/);
 assert.equal(f.calls.every(request=>request.action==='get'),true);
});
for(const change of ['demotion','other-gm','same-id-replacement'])test(`public snapshot rejects ${change} while resolving its actors`,async()=>{
 const f=documentFixture(),state=privateState();state.sessions.S.actorUUIDs=['Actor.H'];f.seed(state);
 let entered,release;const boundary=new Promise(resolve=>{entered=resolve}),wait=new Promise(resolve=>{release=resolve});
 const actor={uuid:'Actor.H',name:'Private actor',items:[],system:{attributes:{hp:{value:20,max:20}},resources:{focus:{value:0,max:0}}}};
 f.fromUuid=async()=>{entered();await wait;return actor};const runtime=createExplorationRuntime(f);await runtime.bind({});
 const pending=runtime.api.snapshot('S');pending.catch(()=>{});await boundary;
 if(change==='demotion')f.game.user.isGM=false;else if(change==='other-gm')f.game.user=f.game.users.get('G2');else f.game.user={...f.game.user};
 release();await assert.rejects(pending,change==='demotion'?/gm-read-required/:/exploration-user-changed/);
});
test('player diagnostic cannot expose previously collected GM time evidence',async()=>{
 const f=documentFixture(),runtime=createExplorationRuntime(f);await runtime.bind({});await runtime.register({socket:null});
 f.emitHook('pf2e.restForTheNight',{uuid:'Actor.private-rest-marker'});f.game.user.isGM=false;
 const diagnostic=runtime.api.diagnostic();assert.equal(JSON.stringify(diagnostic).includes('private-rest-marker'),false);assert.equal(diagnostic.ledger,undefined);assert.equal(diagnostic.time,undefined);
});
