// Public owner selection contracts; effects are offline provider fixtures.
import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
const M=new URL('../../',import.meta.url);
const moduleURL=name=>new URL(name,M);
const {authorityFixture}=await import(moduleURL('tests/exploration/authority-fixture.mjs'));
const {createCoordinator}=await import(moduleURL('scripts/exploration/coordinator.mjs'));
const {chooseNext}=await import(moduleURL('scripts/exploration/policy.mjs'));
const {createHpPools}=await import(moduleURL('scripts/exploration/hp-pool.mjs'));
const H='Actor.HEALER0000000001',P='Actor.PATIENT000000001',MASTER='Actor.MASTER0000000001';
const proposal={id:'activity',actorUUID:H,providerId:'treat-wounds',patientUUIDs:[P],hpPoolUUIDs:[P],startedAt:0,endsAt:600,durationSeconds:600,options:{}};
const deferred=()=>{let resolve;return {promise:new Promise(r=>resolve=r),resolve:value=>resolve(value)}};
async function fixture({snapshotGate=null,shared=false,resolveHook=null,beginHook=null,seed,policy=chooseNext,poolHook=null}={}){
 const f=await authorityFixture(seed),ledger=f.client('driver'),G={id:'G',active:true,isGM:true},O={id:'O',active:true,isGM:false},N={id:'N',active:true,isGM:false},users=new Map([['G',G],['O',O],['N',N]]);users.activeGM=G;
 const documents=new Map([H,P,MASTER].map(uuid=>[uuid,{uuid,testUserPermission:user=>user?.isGM||user?.id==='O'}]));
 const stats=[H,P].map((actorUUID,i)=>({actorUUID,medicine:{rank:i?0:1},slugs:[],items:[],assuranceSkills:[],pool:{poolUUID:shared&&i?MASTER:actorUUID,ready:true},hp:{value:i?1:20,max:20},focus:{value:0,max:0},cooldownExpiresAt:0,modeOfBeing:'living'}));
 const game={user:G,users,time:{worldTime:0}},calls={begin:[],clock:0,complete:0},ownerOperations={createActivityContext:async()=>({validate(){}}),cancelActivity(){}};
 const capabilities={snapshot:async()=>{if(snapshotGate)await snapshotGate();return structuredClone(stats)}};
 const provider={id:'treat-wounds',begin:async activity=>{calls.begin.push(structuredClone(activity));return beginHook?beginHook(activity):{status:'blocked',reason:'offline-probe-boundary'}},cancel:async()=>{},complete:async()=>{calls.complete++;return {status:'uncertain'}}};
 const clock={advanceTo:async()=>{calls.clock++;return {status:'uncertain'}},stop(){}};
 const c=createCoordinator({game,fromUuid:async uuid=>{if(resolveHook)await resolveHook(uuid);return documents.get(uuid)},getHpPool:actor=>poolHook?poolHook(actor):stats.find(s=>s.actorUUID===actor.uuid)?.pool,ledger,capabilities,providers:[provider,{...provider,id:'refocus'}],clock,ownerOperations,isAuthority:()=>true,now:()=>0,policy});
 const config={id:'session',actorUUIDs:[H,P],autoRun:false,budgetSeconds:600,nativeOwnerByActor:{[H]:'O'}};
 return {f,ledger,c,config,game,documents,calls,stats,start:input=>c.start(input??config)};
}

test('default automatic start stores an empty owner map and keeps local source',async()=>{const f=await fixture(),config={...f.config};delete config.nativeOwnerByActor;const s=await f.start(config);assert.deepEqual(s.nativeOwnerByActor,{});await f.c.addActivity(s.id,proposal);assert.equal(f.calls.begin[0].source.ownerId,undefined)});
test('explicit valid active OWNER is preserved in common automatic activity insertion',async()=>{const f=await fixture(),s=await f.start();await f.c.addActivity(s.id,proposal);assert.equal(f.calls.begin[0].source.ownerId,'O')});
test('policy-generated public-start activity receives persisted source owner',async()=>{const f=await fixture();await f.start();await f.c.step('session');assert.equal(f.calls.begin.length,1);assert.equal(f.calls.begin[0].source.ownerId,'O')});
test('extension insertion uses the same actor map',async()=>{const f=await fixture(),s=await f.start();await f.c.addActivity(s.id,{...proposal,options:{extensionOf:'original'}});assert.equal(f.calls.begin[0].source.ownerId,'O')});
test('proposal source cannot override the saved owner choice',async()=>{const f=await fixture(),s=await f.start();const outcome=await f.c.addActivity(s.id,{...proposal,source:{type:'coordinator',ownerId:'N'}}).catch(error=>({error}));assert.ok(outcome.error||f.calls.begin[0]?.source.ownerId==='O');assert.equal(f.calls.begin.some(a=>a.source.ownerId==='N'),false)});

for(const [name,map] of [['array',[]],['null',null],['number',7],['empty owner',{[H]:''}],['unknown actor',{'Actor.UNRELATED0000001':'O'}],['unknown user',{[H]:'missing'}],['nonowner',{[H]:'N'}]])test('start rejects '+name+' before session/reservation/time',async()=>{const f=await fixture();await assert.rejects(f.start({...f.config,nativeOwnerByActor:map}));assert.equal(Object.keys((await f.ledger.all()).sessions).length,0);assert.equal(f.calls.begin.length,0);assert.equal(f.calls.clock,0)});
test('start rejects an inactive explicit owner',async()=>{const f=await fixture();f.game.users.get('O').active=false;await assert.rejects(f.start());assert.equal(Object.keys((await f.ledger.all()).sessions).length,0)});
test('manual recording cannot carry an execution owner map',async()=>{const f=await fixture();await assert.rejects(f.start({...f.config,manual:true}));assert.equal(Object.keys((await f.ledger.all()).sessions).length,0)});
test('caller map mutation during initial asynchronous snapshot cannot retarget start',async()=>{const entered=deferred(),release=deferred(),f=await fixture({snapshotGate:async()=>{entered.resolve();await release.promise}});const pending=f.start();await entered.promise;f.config.nativeOwnerByActor[H]='N';release.resolve();const s=await pending;assert.equal(s.nativeOwnerByActor[H],'O')});
test('generic session patch cannot change a persisted owner map',async()=>{const f=await fixture(),s=await f.start();await assert.rejects(f.ledger.updateSession(s.id,{nativeOwnerByActor:{[H]:'N'}},f.c.executionScope(s.id)));assert.equal((await f.ledger.getSession(s.id)).nativeOwnerByActor[H],'O')});
test('atomic activity insertion rejects a source contradicting session map',async()=>{const f=await fixture(),s=await f.start();await assert.rejects(f.ledger.insertActivity({...proposal,sessionId:s.id,state:'planned',source:{type:'coordinator',ownerId:'N'}},f.c.executionScope(s.id)));assert.equal((await f.ledger.snapshot(s.id)).activities.length,0)});
for(const which of ['executor','patient','master','offline'])test(which+' eligibility loss is blocked before provider.begin and time',async()=>{const f=await fixture({shared:which==='master'}),s=await f.start();if(which==='offline')f.game.users.get('O').active=false;else f.documents.get(which==='executor'?H:which==='patient'?P:MASTER).testUserPermission=user=>user?.isGM;await f.c.addActivity(s.id,{...proposal,hpPoolUUIDs:[which==='master'?MASTER:P]}).catch(()=>{});assert.equal(f.calls.begin.length,0);assert.equal(f.calls.clock,0);assert.equal(f.calls.complete,0)});
test('remote Three Pecks cannot reserve provider begin',async()=>{const f=await fixture(),s=await f.start();await f.c.addActivity(s.id,{...proposal,providerId:'refocus',options:{threePecks:true}}).catch(()=>{});assert.equal(f.calls.begin.length,0);assert.equal(f.calls.clock,0)});
test('ordinary remote Refocus retains its owner and has no fake patients',async()=>{const f=await fixture(),s=await f.start();await f.c.addActivity(s.id,{...proposal,providerId:'refocus',patientUUIDs:[],hpPoolUUIDs:[]});assert.equal(f.calls.begin[0].source.ownerId,'O');assert.deepEqual(f.calls.begin[0].patientUUIDs,[])});
test('explicit current-GM map entry is preserved as an explicit choice',async()=>{const f=await fixture(),s=await f.start({...f.config,nativeOwnerByActor:{[H]:'G'}});await f.c.addActivity(s.id,proposal);assert.equal(f.calls.begin[0].source.ownerId,'G')});
test('readonly restore leaves saved map intact and performs no new work',async()=>{const f=await fixture(),s=await f.start();const before=await f.ledger.getSession(s.id);await f.c.restore(s.id);assert.deepEqual(await f.ledger.getSession(s.id),before);assert.equal(f.calls.begin.length,0);assert.equal(f.calls.clock,0)});
test('exact public runtime start captures the map before its storage await',async()=>{
 // Execute only the existing start closure; bind its environment without initializing runtime/native providers.
 const source=fs.readFileSync(moduleURL('scripts/exploration/runtime.mjs'),'utf8'),begin=source.indexOf('const start=async config=>'),end=source.indexOf('\n async function refreshStorage',begin);assert.ok(begin>0&&end>begin);
 const schema=await import(moduleURL('scripts/exploration/schema.mjs')),release=deferred(),game={user:{setFlag:async()=>{}},system:{id:'pf2e',version:'8.5.1'}},captured=[];
 const names=Object.keys(schema),start=new Function('isActiveGM','game','refreshStorage','coordinator','ledger',...names,'let lastSessionId=null;'+source.slice(begin,end)+';return start')(()=>true,game,()=>release.promise,{start:async config=>{captured.push(structuredClone(config));return {id:'session'}}},{getSession:async()=>null},...names.map(name=>schema[name]));
 const config={actorUUIDs:[H,P],nativeOwnerByActor:{[H]:'O'}},pending=start(config);config.nativeOwnerByActor[H]='N';release.resolve({state:'ready'});await pending;assert.equal(captured[0].nativeOwnerByActor[H],'O');
});

const schema=await import(moduleURL('scripts/exploration/schema.mjs'));
test('pure normalizer copies own entries and defaults missing map',()=>{
 assert.equal(typeof schema.normalizeNativeOwnerMap,'function');const input={[H]:'O'};
 assert.deepEqual(schema.normalizeNativeOwnerMap(undefined,[H]),{});
 const normalized=schema.normalizeNativeOwnerMap(input,[H]);input[H]='N';assert.deepEqual(normalized,{[H]:'O'});
});
for(const [name,make]of [
 ['inherited entry',()=>Object.create({[H]:'O'})],
 ['getter',()=>Object.defineProperty({},H,{enumerable:true,get(){throw Error('getter-executed')}})],
 ['hidden entry',()=>Object.defineProperty({},H,{value:'O'})],
 ['symbol entry',()=>({[Symbol('actor')]:'O'})],
 ['forbidden key',()=>JSON.parse('{"__proto__":"O"}')],
 ['blank user',()=>({[H]:'  '})]
])test('normalizer rejects '+name+' without invoking getters',()=>{
 assert.equal(typeof schema.normalizeNativeOwnerMap,'function');assert.throws(()=>schema.normalizeNativeOwnerMap(make(),[H]),error=>error.message!=='getter-executed');
});
test('direct ledger creation validates and copies the map before queued transaction',async()=>{
 const f=await fixture(),input={id:'direct',actorUUIDs:[H,P],nativeOwnerByActor:{[H]:'O'},startedAt:0,cursorAt:0,budgetEndsAt:600};
 const pending=f.ledger.createSession(input);input.nativeOwnerByActor[H]='N';assert.equal((await pending).nativeOwnerByActor[H],'O');
});
test('direct ledger creation refuses a malformed map',async()=>{const f=await fixture();await assert.rejects(async()=>f.ledger.createSession({id:'direct',actorUUIDs:[H],nativeOwnerByActor:[],startedAt:0,budgetEndsAt:600}));});
test('permission revoked while resolving the shared master fails before begin',async()=>{
 const entered=deferred(),release=deferred();let armed=false;
 const f=await fixture({shared:true,resolveHook:async uuid=>{if(armed&&uuid===MASTER){entered.resolve();await release.promise}}});const s=await f.start();armed=true;
 const pending=f.c.addActivity(s.id,{...proposal,hpPoolUUIDs:[MASTER]});assert.equal(await Promise.race([entered.promise.then(()=>true),pending.then(()=>false)]),true,'Master must resolve before provider.begin');f.documents.get(H).testUserPermission=u=>u.isGM;release.resolve();
 await assert.rejects(pending);assert.equal(f.calls.begin.length,0);assert.equal(f.calls.clock,0);
});
test('permission lost during begin prevents the first clock submission',async()=>{
 let f;f=await fixture({beginHook:async()=>{f.documents.get(P).testUserPermission=u=>u.isGM;return {status:'started'}}});await f.start();
 assert.equal((await f.c.step('session')).status,'paused');assert.equal(f.calls.begin.length,1);assert.equal(f.calls.clock,0);assert.equal(f.calls.complete,0);
});
test('existing started work is rechecked before a later step advances time',async()=>{
 const f=await fixture({beginHook:async()=>({status:'started'})}),s=await f.start();await f.c.addActivity(s.id,proposal);f.game.users.get('O').active=false;
 assert.equal((await f.c.step(s.id)).status,'paused');assert.equal(f.calls.clock,0);assert.equal(f.calls.complete,0);
});
test('resume rechecks saved owner before issuing a new driver lease',async()=>{
 const f=await fixture(),s=await f.start();await f.c.stop(s.id);const prior=await f.ledger.getSession(s.id);f.game.users.get('O').active=false;
 await assert.rejects(f.c.resume(s.id,{autoRun:false}));assert.deepEqual(await f.ledger.getSession(s.id),prior);
});
test('resume cannot accept a replacement owner map',async()=>{
 const f=await fixture(),s=await f.start();await f.c.stop(s.id);const resumed=await f.c.resume(s.id,{autoRun:false,nativeOwnerByActor:{[H]:'N'}});
 assert.deepEqual(resumed.nativeOwnerByActor,{[H]:'O'});await f.c.addActivity(s.id,proposal);assert.equal(f.calls.begin[0].source.ownerId,'O');
});
test('seeded legacy session keeps historical sources and defaults new work local',async()=>{
 const history={...proposal,id:'history',sessionId:'legacy',state:'confirmed',source:{type:'coordinator',ownerId:'old-owner'},executor:{state:'settled',ownerUserId:'old-owner'}};
 const seed={sessions:{legacy:{id:'legacy',actorUUIDs:[H,P],activityIds:['history'],goalsByPool:[{poolUUID:P,targetHP:20}],startedAt:0,cursorAt:600,budgetEndsAt:1800,maxActivities:3,status:'paused'}},activities:{history},clocks:{}};
 const f=await fixture({seed});assert.equal(Object.hasOwn(await f.ledger.getSession('legacy'),'nativeOwnerByActor'),false);assert.deepEqual(await f.ledger.getActivity('history'),history);
 // Its historical cursor is external-time-sensitive; use an explicit zero-time legacy seed for future work.
 seed.sessions.legacy.cursorAt=0;seed.activities.history.endsAt=0;seed.activities.history.startedAt=-600;
 const g=await fixture({seed});await g.c.resume('legacy',{autoRun:false});await g.c.addActivity('legacy',{...proposal,id:'new'});
 assert.equal(g.calls.begin[0].source.ownerId,undefined);assert.deepEqual(await g.ledger.getActivity('history'),seed.activities.history);
});
test('a shared pool changed during begin cannot advance using the saved old pool',async()=>{
 let f;f=await fixture({beginHook:async()=>{f.stats[1].pool.poolUUID=MASTER;f.documents.get(MASTER).testUserPermission=u=>u.isGM;return {status:'started'}}});await f.start();
 assert.equal((await f.c.step('session')).status,'paused');assert.equal(f.calls.clock,0);assert.equal(f.calls.complete,0);
});
for(const operation of ['start','resume'])test(operation+' rechecks ownership after the eligibility Promise returns',async()=>{
 const f=await fixture();if(operation==='resume'){await f.start();await f.c.stop('session')}
 const before=await f.ledger.all();f.documents.get(H).testUserPermission=user=>{queueMicrotask(()=>{f.game.users.get('O').active=false});return user.id==='O'||user.isGM};
 await assert.rejects(operation==='start'?f.start():f.c.resume('session',{autoRun:false}));assert.deepEqual(await f.ledger.all(),before);
});
test('permission loss during a different provider begin still blocks the shared clock',async()=>{
 const other='Actor.OTHER00000000001';let f;
 f=await fixture({policy:()=>({activities:[proposal,{...proposal,id:'second',actorUUID:other,providerId:'refocus',patientUUIDs:[],hpPoolUUIDs:[]}],checkpointAt:600}),beginHook:async activity=>{if(activity.actorUUID===other)f.documents.get(P).testUserPermission=u=>u.isGM;return {status:'started'}}});
 f.config.actorUUIDs.push(other);f.config.nativeOwnerByActor[other]='O';f.documents.set(other,{uuid:other,testUserPermission:u=>u.id==='O'||u.isGM});f.stats.push({...structuredClone(f.stats[0]),actorUUID:other,pool:{poolUUID:other,ready:true}});
 await f.start();assert.equal((await f.c.step('session')).status,'paused');assert.equal(f.calls.begin.length,2);assert.equal(f.calls.clock,0);assert.equal(f.calls.complete,0);
});
test('Stop during asynchronous mapped resolution cannot insert or begin',async()=>{
 const entered=deferred(),release=deferred();let armed=false;
 const f=await fixture({shared:true,resolveHook:async uuid=>{if(armed&&uuid===MASTER){entered.resolve();await release.promise}}});const s=await f.start();armed=true;
 const pending=f.c.addActivity(s.id,{...proposal,hpPoolUUIDs:[MASTER]});await entered.promise;await f.c.stop(s.id);release.resolve();await assert.rejects(pending);
 assert.equal((await f.ledger.snapshot(s.id)).activities.length,0);assert.equal(f.calls.begin.length,0);assert.equal(f.calls.clock,0);
});
test('map-selected actor cannot be directly inserted with an omitted source owner',async()=>{
 const f=await fixture(),s=await f.start();await assert.rejects(f.ledger.insertActivity({...proposal,sessionId:s.id,state:'planned',source:{type:'coordinator'}},f.c.executionScope(s.id)),/native-owner-selection-mismatch/);
});
test('new default-local session cannot be directly inserted with an explicit remote owner',async()=>{
 const f=await fixture(),s=await f.start({...f.config,nativeOwnerByActor:{}});await assert.rejects(f.ledger.insertActivity({...proposal,sessionId:s.id,state:'planned',source:{type:'coordinator',ownerId:'O'}},f.c.executionScope(s.id)),/native-owner-selection-mismatch/);
});
test('waiting for cooldown also rechecks the saved owner before spending time',async()=>{
 const f=await fixture({policy:()=>({activities:[],checkpointAt:600})});await f.start();f.game.users.get('O').active=false;
 assert.equal((await f.c.step('session')).status,'paused');assert.equal(f.calls.clock,0);
});
for(const boundary of ['document-resolution','final-owned-read'])test('actual Toolbelt relink during '+boundary+' prevents clock submission',async()=>{
 let f,pools,armed=false;
 const relink=()=>{f.documents.get(P).modules={'pf2e-toolbelt':{shareData:{data:{health:true}}}};f.documents.get(MASTER).testUserPermission=u=>u.isGM;armed=false};
 f=await fixture({poolHook:actor=>pools.discover(actor),snapshotGate:async()=>{if(pools)f.stats[1].pool=pools.discover(f.documents.get(P))},resolveHook:async uuid=>{if(armed&&boundary==='document-resolution'&&uuid===P)relink()},beginHook:async()=>({status:'started'})});
 f.game.modules=new Map([['pf2e-toolbelt',{active:true}]]);f.game.settings={get:()=>true};f.game.toolbelt={api:{shareData:{getMasterInMemory:()=>f.documents.get(MASTER),getSlavesInMemory:()=>[]}}};pools=createHpPools({game:f.game});
 await f.start();await f.c.addActivity('session',proposal);
 let snapshotReads=0;const originalSnapshot=f.ledger.snapshot,originalGet=f.ledger.getSession;
 f.ledger.snapshot=async id=>{const value=await originalSnapshot(id);if(value.activities.some(a=>a.state==='started')&&++snapshotReads===2)armed=true;return value};
 f.ledger.getSession=async id=>{const value=await originalGet(id);if(armed&&boundary==='final-owned-read')relink();return value};
 const result=await f.c.step('session');assert.equal(pools.discover(f.documents.get(P)).poolUUID,MASTER);assert.equal(f.documents.get(MASTER).testUserPermission(f.game.users.get('O')),false);assert.equal(result.status,'paused');assert.equal(f.calls.clock,0);assert.equal(f.calls.complete,0);
});
