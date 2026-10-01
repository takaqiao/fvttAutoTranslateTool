// Offline interface/acceptance-boundary tests, not real Foundry/native-RNG QA.
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createSalubriousCheckScope} from '../../scripts/salubrious-kiss-check-scope.mjs';
import {beforeNativeRoll} from '../../scripts/native-owner-operations.mjs';
import {authorityFixture} from './authority-fixture.mjs';
import {createExplorationOwnerOperations} from '../../scripts/exploration/owner-operations.mjs';
import {OWNER_TRANSPORT_CHANNEL} from '../../scripts/exploration/owner-transport.mjs';
import {createHpPools} from '../../scripts/exploration/hp-pool.mjs';
const ID='pf2e-third-party-automation';
const flush=async()=>{for(let i=0;i<25;i++)await Promise.resolve()};
function hooks(){let serial=0;const listeners=new Map();return {on:(name,fn)=>{const id=++serial;listeners.set(id,{name,fn});return id},off:(_name,id)=>listeners.delete(id),fire:(name,...args)=>{for(const row of [...listeners.values()])if(row.name===name)row.fn(...args)},get size(){return listeners.size}}}
function checkFixture({remote=true,show=true,explicit}={}){
 const Hooks=hooks(),controller=new AbortController();let live=true,rng=0,windows=0,app;
 const ctx=Object.freeze({nativeDialogMode:remote?'owner-preference':'automatic',executionSignal:controller.signal,validate:()=>{if(!live)throw Error('activity-stopped')}}),healer={},activity={id:'activity',options:{skill:'medicine',rank:'trained'}};
 const game={user:{id:remote?'O':'G',settings:{showCheckDialogs:show}}};
 const scope=createSalubriousCheckScope({game,Hooks,isExplorationContext:c=>c===ctx});
 const context={actor:healer,type:'skill-check',domains:['medicine'],dc:{value:15},options:new Set(['action:treat-wounds','exploration-activity:activity']),...explicit===undefined?{}:{skipDialog:explicit}};
 // This mirrors the observed PF2e Check.roll resolver boundary, not its dice engine.
 const wrapped=async(_check,c)=>{c.skipDialog??=!game.user.settings.showCheckDialogs;if(!c.skipDialog){windows++;const accepted=await new Promise(resolve=>{app={context:c,resolve,close:async()=>{}};Hooks.fire('renderCheckModifiersDialog',app)});if(!accepted)return null}rng++;return 'offline-roll-boundary'};
 const run=()=>scope.runExploration({ctx,healer,activity},()=>scope.interceptCheck(wrapped,{},context));
 return {ctx,scope,context,Hooks,run,revoke:()=>{live=false},cancel:()=>{live=false;controller.abort(Error('activity-stopped'))},get app(){return app},get rng(){return rng},get windows(){return windows}};
}
test('remote private context preserves the selected owner check dialog preference',async()=>{const f=checkFixture(),pending=f.run().then(value=>({value}),error=>({error}));await flush();try{assert.equal(f.windows,1);assert.equal(f.rng,0)}finally{if(f.app)await f.app.resolve(false);await pending}});
test('remote preference-disabled native check is not forced to display a window',async()=>{const f=checkFixture({show:false});assert.equal(await f.run(),'offline-roll-boundary');assert.equal(f.windows,0);assert.equal(f.rng,1)});
test('default local GM automatic checks still skip the window',async()=>{const f=checkFixture({remote:false});assert.equal(await f.run(),'offline-roll-boundary');assert.equal(f.windows,0);assert.equal(f.rng,1)});
test('explicit native skipDialog false is preserved for the remote context',async()=>{const f=checkFixture({show:false,explicit:false}),pending=f.run().catch(error=>error);await flush();try{assert.equal(f.windows,1);assert.equal(f.rng,0)}finally{if(f.app)await f.app.resolve(false);await pending}});
test('Stop while exact owner window is pending cannot cross final roll boundary',async()=>{const f=checkFixture(),pending=f.run().then(value=>({value}),error=>({error}));await flush();f.cancel();if(f.app)await f.app.resolve(true);await pending;assert.equal(f.rng,0)});
test('permission loss without an event must be rechecked at native final acceptance',async()=>{const f=checkFixture(),pending=f.run().then(value=>({value}),error=>({error}));await flush();f.revoke();if(f.app)await f.app.resolve(true);await pending;assert.equal(f.rng,0)});
test('one authorized remote native confirmation crosses its boundary exactly once',async()=>{const f=checkFixture(),pending=f.run().then(value=>({value}),error=>({error}));await flush();try{assert.equal(f.windows,1);assert.equal(f.rng,0);await f.app.resolve(true);await f.app.resolve(true);await pending;assert.equal(f.rng,1)}finally{if(f.app)await f.app.resolve(false);await pending}});
test('manual native window cancellation yields zero dice work',async()=>{const f=checkFixture(),pending=f.run().catch(error=>error);await flush();if(f.app)await f.app.resolve(false);await pending;assert.equal(f.rng,0)});
test('copied private context cannot acquire exploration dialog or dice scope',async()=>{const f=checkFixture();await assert.rejects(f.scope.runExploration({ctx:{...f.ctx},healer:{},activity:{id:'activity',options:{}}},async()=>{}),/context/);assert.equal(f.rng,0)});
test('unrelated ordinary native context keeps its window argument untouched',async()=>{const f=checkFixture(),context={options:new Set(),skipDialog:false};await f.scope.interceptCheck(async(_check,actual)=>{assert.equal(actual,context);assert.equal(actual.skipDialog,false)}, {},context);assert.equal(f.rng,0)});
test('optional execution abort signal closes only its exact native window',async()=>{const Hooks=hooks(),controller=new AbortController();let exact,unrelated,closes=0;const pending=beforeNativeRoll({Hooks,marker:'exact-marker',showDialog:true,signal:controller.signal,commit:async()=>{},assertLive:()=>{if(controller.signal.aborted)throw Error('activity-stopped')},native:()=>new Promise(resolve=>{exact={context:{options:['exact-marker']},resolve,close:async()=>{closes++}};unrelated={context:{options:['other-marker']},resolve:()=>assert.fail('unrelated dialog resolved'),close:async()=>assert.fail('unrelated dialog closed')};Hooks.fire('renderCheckModifiersDialog',unrelated);Hooks.fire('renderCheckModifiersDialog',exact)})}).catch(error=>error);await flush();controller.abort(Error('activity-stopped'));await flush();try{assert.equal(closes,1)}finally{await exact.resolve(false);await pending}assert.equal(Hooks.size,0)});

async function brokerFixture({owner='O',timeoutMs=1000,onNative,configureGame=()=>{},configureActors=()=>{},poolDiscovery=false,operation='treat-wounds',patientUUIDs=operation==='refocus'?[]:['Actor.P'],hpPoolUUIDs=patientUUIDs}={}){
 const f=await authorityFixture(),ledger=f.client('driver'),G={id:'G',isGM:true,active:true,settings:{showCheckDialogs:true}},O={id:'O',active:true,settings:{showCheckDialogs:true}},users=new Map([['G',G],['O',O]]);users.activeGM=G;
 const actors=new Map(['H','P'].map(id=>{const actor={id,uuid:'Actor.'+id,flags:{},testUserPermission:user=>user?.isGM||user?.id==='O',update:async data=>{actor.flags[ID]??={};for(const [key,value]of Object.entries(data))if(key===`flags.${ID}.explorationExecutions`)actor.flags[ID].explorationExecutions=structuredClone(value);return actor}};return [actor.uuid,actor]}));
 configureActors(actors);
 const s=await ledger.createSession({id:'session',actorUUIDs:[...actors.keys()],nativeOwnerByActor:{'Actor.H':owner},startedAt:0,cursorAt:0,budgetEndsAt:600,status:'running'}),lease={leaseNonce:s.driver.leaseNonce};
 let activity=await ledger.insertActivity({id:'activity',sessionId:s.id,providerId:operation,actorUUID:'Actor.H',patientUUIDs,hpPoolUUIDs,startedAt:0,endsAt:600,state:'planned',source:{type:'coordinator',ownerId:owner},options:{}},lease);
 await ledger.transitionActivity(activity.id,{...lease,expected:['planned'],patch:{state:'started'}});activity=await ledger.transitionActivity(activity.id,{...lease,expected:['started'],patch:{state:'completing'}});
 const tabs=[];let enteredResolve,release;const entered=new Promise(r=>enteredResolve=r),held=new Promise(r=>release=r);let captured;
 function tab(userId,clientNonce){const listeners=new Set(),game={user:users.get(userId),users,time:{worldTime:600},messages:new Map(),socket:{on:(channel,fn)=>{assert.equal(channel,OWNER_TRANSPORT_CHANNEL);listeners.add(fn)},off:(_channel,fn)=>listeners.delete(fn),emit:(_channel,packet,routing,ack)=>{for(const target of tabs.filter(t=>t.userId===routing.recipients[0]))for(const fn of target.listeners)fn(structuredClone(packet),userId);ack?.()}}};
  configureGame(game);
  const pools=poolDiscovery?createHpPools({game}):null;
  const ops=createExplorationOwnerOperations({game,fromUuid:async uuid=>actors.get(uuid),getHpPool:pools?actor=>pools.discover(actor):undefined,ledger:userId==='G'?ledger:{atomic:true},runtimeIdentity:()=>({userId,clientNonce}),getDriverScope:()=>userId==='G'?lease:null,timeoutMs});
  ops.registerOperation(operation,async(a,ctx)=>{captured={a,ctx,ops};enteredResolve(captured);if(onNative)return onNative(a,ctx,ops);await held;ctx.validate();throw Error('offline-probe-complete')});ops.register({});const t={userId,listeners,ops};tabs.push(t);return t;
 }
 const driver=tab('G','driver'),receiver=tab('O','owner');
 return {driver,receiver,activity,entered,ledger,actors,users,get captured(){return captured},release,dispose:()=>tabs.forEach(t=>t.ops.dispose())};
}
for(const owner of ['G','O'])test('real atomic broker derives private dialog mode for '+owner,async()=>{const f=await brokerFixture({owner}),pending=f.driver.ops.runActivityWithOwner(f.activity,'treat-wounds').catch(error=>error);try{const {a,ctx,ops}=await f.entered;assert.equal(a.source.ownerId,undefined);assert.equal(Object.isFrozen(ctx),true);assert.equal(ops.isExecutionContext({...ctx},f.activity.id),false);assert.equal(ctx.nativeDialogMode,owner==='O'?'owner-preference':'automatic')}finally{f.release();await pending;f.dispose()}});
test('real owner broker Stop invalidates the private execution abort signal',async()=>{const f=await brokerFixture(),pending=f.driver.ops.runActivityWithOwner(f.activity,'treat-wounds').catch(error=>error);try{const {ctx}=await f.entered;assert.ok(ctx.executionSignal instanceof AbortSignal);await f.driver.ops.cancelActivity(f.activity);await flush();assert.equal(ctx.executionSignal.aborted,true);assert.throws(()=>ctx.validate())}finally{f.release();await pending;f.dispose()}});

test('an already aborted signal never starts native preparation even without assertLive',async()=>{
 const controller=new AbortController();controller.abort(Error('cancelled-before-native'));let nativeCalls=0;
 await assert.rejects(beforeNativeRoll({Hooks:hooks(),marker:'A',showDialog:true,signal:controller.signal,commit:async()=>{},native:async()=>nativeCalls++}),/cancelled-before-native/);
 assert.equal(nativeCalls,0);
});

test('a late exact window stays cancelled after the helper has rejected',async()=>{
 const Hooks=hooks(),controller=new AbortController();let release,dice=0,closed=0;
 const gate=new Promise(resolve=>release=resolve);
 const task=beforeNativeRoll({Hooks,marker:'late',showDialog:true,signal:controller.signal,commit:async()=>{},native:async()=>{
  await gate;const accepted=await new Promise(resolve=>Hooks.fire('renderCheckModifiersDialog',{context:{options:['late']},resolve,close:()=>closed++}));if(accepted)dice++;
 }});
 await flush();controller.abort(Error('expired'));await assert.rejects(task,/expired/);
 assert.equal(Hooks.size,1);release();await flush();assert.equal(dice,0);assert.equal(closed,1);assert.equal(Hooks.size,0);
});

test('aborting during awaited acceptance payment closes the resolver without rolling',async()=>{
 const Hooks=hooks(),controller=new AbortController();let app,release,commits=0,dice=0,closed=0;
 const gate=new Promise(resolve=>release=resolve);
 const task=beforeNativeRoll({Hooks,marker:'pay',showDialog:true,signal:controller.signal,commit:async()=>{commits++;await gate},native:async()=>{
  if(await new Promise(resolve=>{app={context:{options:['pay']},resolve,close:()=>closed++};Hooks.fire('renderCheckModifiersDialog',app)}))dice++;
 }});
 await flush();const accepted=app.resolve(true);await flush();controller.abort(Error('stop-during-payment'));await assert.rejects(task,/stop-during-payment/);
 release();await accepted;await flush();assert.equal(commits,1);assert.equal(dice,0);assert.equal(closed,1);assert.equal(Hooks.size,0);
});

test('normal settlement removes the signal listener and leaves unrelated windows alone',async()=>{
 const Hooks=hooks(),controller=new AbortController();let added=0,removed=0;
 const add=controller.signal.addEventListener.bind(controller.signal),remove=controller.signal.removeEventListener.bind(controller.signal);
 controller.signal.addEventListener=(...args)=>{added++;return add(...args)};controller.signal.removeEventListener=(...args)=>{removed++;return remove(...args)};
 await beforeNativeRoll({Hooks,marker:'settled',showDialog:true,signal:controller.signal,commit:async()=>{},native:async()=>42});
 assert.equal(added,1);assert.equal(removed,1);assert.equal(Hooks.size,0);controller.abort();assert.equal(Hooks.size,0);
});

test('eventless revocation before deferred remote native entry prevents dice work',async()=>{
 const f=checkFixture({explicit:true});
 queueMicrotask(()=>f.revoke());
 await assert.rejects(f.run(),/activity-stopped/);
 assert.equal(f.rng,0);assert.equal(f.Hooks.size,0);
});

for(const showDialog of [false,true])test('eventless revocation after payment prevents '+(showDialog?'final window acceptance':'direct native entry'),async()=>{
 const Hooks=hooks();let live=true,paid=false,queued=false,app,commits=0,dice=0,closed=0,resolved;
 const task=beforeNativeRoll({Hooks,marker:'final-acceptance',showDialog,
  commit:async()=>{commits++;paid=true},
  assertLive:()=>{
   if(!live)throw Error('owner-revoked');
   if(paid&&!queued){queued=true;queueMicrotask(()=>{live=false})}
  },
  native:async()=>{
   if(showDialog){
    const accepted=await new Promise(resolve=>{
     app={context:{options:['final-acceptance']},resolve:value=>{resolved=value;resolve(value)},close:()=>closed++};
     Hooks.fire('renderCheckModifiersDialog',app);
    });
    if(!accepted)return;
   }
   dice++;
  }
 }).catch(error=>error);
 if(showDialog){await flush();await app.resolve(true);await app.resolve(true)}
 const error=await task;
 assert.match(error?.message??'',/owner-revoked/);
 assert.equal(commits,1);assert.equal(dice,0);assert.equal(Hooks.size,0);
 if(showDialog){assert.equal(resolved,false);assert.equal(closed,1)}
});

test('settled broker scopes leave the active cancellation set',async()=>{
 const f=await brokerFixture({onNative:async()=>({status:'blocked',reason:'known-offline-fixture-result'})});
 try{
  const result=await f.driver.ops.runActivityWithOwner(f.activity,'treat-wounds');assert.equal(result.status,'blocked');
  const {ctx,ops}=f.captured;assert.equal(ops.isExecutionContext(ctx,f.activity.id),false);assert.equal(ctx.executionSignal.aborted,false);
  f.receiver.ops.invalidate('later-runtime-generation');assert.equal(ctx.executionSignal.aborted,false);
 }finally{f.dispose()}
});

for(const action of ['stop','timeout','invalidate','dispose','permission-without-event','owner-offline-without-event'])test('atomic remote '+action+' aborts the exact pending window and preserves the unresolved grant',async()=>{
 const Hooks=hooks();let app,rng=0,closed=0,opened;
 const ready=new Promise(resolve=>opened=resolve);
 const f=await brokerFixture({timeoutMs:150,onNative:async(activity,ctx,ops)=>{
  const scope=createSalubriousCheckScope({game:{user:{id:'O'}},Hooks,isExplorationContext:(value,id)=>ops.isExecutionContext(value,id)}),healer=ctx.actor;
  const context={actor:healer,type:'skill-check',domains:['medicine'],dc:{value:15},options:new Set(['action:treat-wounds','exploration-activity:'+activity.id])};
  await scope.runExploration({ctx,healer,activity},()=>scope.interceptCheck(async()=>{
   const accepted=await new Promise(resolve=>{app={context,resolve,close:()=>closed++};Hooks.fire('renderCheckModifiersDialog',app);opened()});if(accepted)rng++;
  },{},context));
  throw Error('unexpected-native-continuation');
 }});
 const pending=f.driver.ops.runActivityWithOwner(f.activity,'treat-wounds').catch(error=>error);
 try{
  await ready;const {ctx,ops}=f.captured;
  assert.equal(ctx.nativeDialogMode,'owner-preference');assert.equal(Object.isFrozen(ctx),true);assert.throws(()=>{ctx.nativeDialogMode='automatic'},TypeError);
  if(action==='stop')await f.driver.ops.cancelActivity(f.activity);
  else if(action==='invalidate')f.receiver.ops.invalidate('owner-disconnected');
  else if(action==='dispose')f.receiver.ops.dispose();
  else if(action==='permission-without-event'){f.actors.get('Actor.P').testUserPermission=user=>user?.isGM;await app.resolve(true)}
  else if(action==='owner-offline-without-event'){f.users.get('O').active=false;await app.resolve(true)}
  if(!ctx.executionSignal.aborted)await new Promise(resolve=>ctx.executionSignal.addEventListener('abort',resolve,{once:true}));
  await pending;await flush();await app.resolve(true);assert.equal(rng,0);assert.equal(closed,1);assert.equal(Hooks.size,0);
  assert.equal(ops.isExecutionContext(ctx,f.activity.id),false);assert.equal((await f.ledger.getActivity(f.activity.id)).executor.state,'granted');
  assert.equal(f.actors.get('Actor.H').flags[ID].explorationExecutions[f.activity.id].state,'uncertain');
 }finally{f.release();f.dispose();await pending}
});

for(const change of ['unowned-master','owned-master','unavailable-master','unchanged'])test('private native window rechecks actual Toolbelt pool: '+change,async()=>{
 const Hooks=hooks();let f,app,opened,dice=0,closed=0;
 const ready=new Promise(resolve=>opened=resolve);
 const master={uuid:'Actor.MASTER',testUserPermission:user=>user.isGM||change==='owned-master'};
 const fOptions={poolDiscovery:true,configureGame:game=>{
  game.modules=new Map([['pf2e-toolbelt',{active:true}]]);game.settings={get:()=>true};
  game.toolbelt={api:{shareData:{getMasterInMemory:()=>change==='unavailable-master'?null:master,getSlavesInMemory:()=>[]}}};
 },onNative:async(activity,ctx,ops)=>{
  const scope=createSalubriousCheckScope({game:{user:{id:'O'}},Hooks,isExplorationContext:(value,id)=>ops.isExecutionContext(value,id)});
  const context={actor:ctx.actor,type:'skill-check',domains:['medicine'],dc:{value:15},options:new Set(['action:treat-wounds','exploration-activity:'+activity.id])};
  await scope.runExploration({ctx,healer:ctx.actor,activity},()=>scope.interceptCheck(async()=>{
   const accepted=await new Promise(resolve=>{app={context,resolve,close:()=>closed++};Hooks.fire('renderCheckModifiersDialog',app);opened()});if(accepted)dice++;
  },{},context));
  return {status:'blocked',reason:'offline-native-boundary'};
 }};
 f=await brokerFixture(fOptions);
 const pending=f.driver.ops.runActivityWithOwner(f.activity,'treat-wounds').catch(error=>error);
 try{
  await ready;
  if(change!=='unchanged')f.actors.get('Actor.P').modules={'pf2e-toolbelt':{shareData:{data:{health:true}}}};
  await app.resolve(true);await pending;await flush();
  assert.equal(dice,change==='unchanged'?1:0);assert.equal(closed,change==='unchanged'?0:1);assert.equal(Hooks.size,0);
  if(change!=='unchanged'){
   assert.throws(()=>f.captured.ctx.validate(),/pool|stopped/);
   assert.equal((await f.ledger.getActivity(f.activity.id)).executor.state,'granted');
   assert.equal(f.actors.get('Actor.H').flags[ID].explorationExecutions[f.activity.id].state,'uncertain');
  }
 }finally{f.release();f.dispose();await pending}
});

for(const enabled of [true,false,undefined])test('missing pool discovery with Toolbelt setting '+enabled+' is bounded',async()=>{
 let nativeCalls=0;
 const f=await brokerFixture({configureGame:game=>{game.modules=new Map([['pf2e-toolbelt',{active:true}]]);game.settings={get:()=>enabled}},onNative:async()=>{nativeCalls++;return {status:'blocked',reason:'offline-native-boundary'}}});
 try{
  const result=await f.driver.ops.runActivityWithOwner(f.activity,'treat-wounds').catch(error=>error);
  assert.equal(nativeCalls,enabled===false?1:0);
  if(enabled!==false){assert.ok(result instanceof Error);assert.equal((await f.ledger.getActivity(f.activity.id)).executor.state,'granted')}
 }finally{f.dispose()}
});

test('two patients sharing the same claimed master pass the private pool check',async()=>{
 let calls=0;
 const master={uuid:'Actor.MASTER',testUserPermission:user=>user.isGM||user.id==='O'};
 const f=await brokerFixture({poolDiscovery:true,patientUUIDs:['Actor.P','Actor.P2'],hpPoolUUIDs:[master.uuid],
  configureActors:actors=>{
   actors.set(master.uuid,master);actors.set('Actor.P2',{...actors.get('Actor.P'),uuid:'Actor.P2'});
   for(const uuid of ['Actor.P','Actor.P2'])actors.get(uuid).modules={'pf2e-toolbelt':{shareData:{data:{health:true}}}};
  },
  configureGame:game=>{
   game.modules=new Map([['pf2e-toolbelt',{active:true}]]);game.settings={get:()=>true};
   game.toolbelt={api:{shareData:{getMasterInMemory:()=>master,getSlavesInMemory:()=>[]}}};
  },onNative:async(_activity,ctx)=>{ctx.validate();calls++;return {status:'blocked',reason:'offline-native-boundary'}}
 });
 try{await f.driver.ops.runActivityWithOwner(f.activity,'treat-wounds');assert.equal(calls,1)}finally{f.dispose()}
});

test('ordinary Refocus empty HP domains need no discovery even when Toolbelt is enabled',async()=>{
 let calls=0;
 const f=await brokerFixture({operation:'refocus',configureGame:game=>{game.modules=new Map([['pf2e-toolbelt',{active:true}]]);game.settings={get:()=>true}},onNative:async(_activity,ctx)=>{ctx.validate();calls++;return {status:'blocked',reason:'offline-native-boundary'}}});
 try{await f.driver.ops.runActivityWithOwner(f.activity,'refocus');assert.equal(calls,1)}finally{f.dispose()}
});
