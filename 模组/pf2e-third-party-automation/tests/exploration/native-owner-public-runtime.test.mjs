import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createExplorationRuntime} from '../../scripts/exploration/runtime.mjs';
import {MODULE_ID} from '../../scripts/exploration/schema.mjs';
import {OWNER_TRANSPORT_CHANNEL} from '../../scripts/exploration/owner-transport.mjs';
import {samePermit} from '../../scripts/exploration/owner-command.mjs';

function publicRuntimeFixture() {
 const tabs=[],packets=[],documents=[],effects=[],clocks=[],errors=[];
 const users=new Map(['G','O','P'].map(id=>[id,{id,isGM:id==='G',active:true,flags:{},getFlag(scope,key){return this.flags[scope]?.[key]},async setFlag(scope,key,value){(this.flags[scope]??={})[key]=value}}]));
 users.activeGM=users.get('G');
 let configured='',raw=null,worldTime=0,nativeGate=null;
 const actorSource={id:'H',uuid:'Actor.H',name:'Offline Refocus',type:'character',level:1,modeOfBeing:'living',flags:{},system:{attributes:{hp:{value:20,max:20,negativeHealing:false}},resources:{focus:{value:0,max:1}}}};
 function apply(document,data){for(const [path,value]of Object.entries(data)){const keys=path.split('.');let at=document;for(const key of keys.slice(0,-1))at=at[key]??={};at[keys.at(-1)]=structuredClone(value)}}
 function emitHook(event,...args){for(const tab of tabs)for(const [name,handler]of tab.hooks.values())if(name===event)handler(...args)}
 function documentRequest(tab,request,ack){
  documents.push({tab:tab.id,userId:tab.game.user.id,request:structuredClone(request)});
  const envelope={type:request.type,action:request.action,operation:structuredClone(request.operation),broadcast:false,userId:tab.game.user.id};
  if(request.action==='get')return ack({...envelope,result:raw?[structuredClone(raw)]:[]});
  assert.equal(tab.game.user.isGM,true,'the fixture server authenticates document writers');
  if(request.type==='JournalEntry'){
   assert.equal(raw,null);raw={...structuredClone(request.operation.data[0]),_id:'ROOT000000000001',pages:[]};
   return ack({...envelope,result:[structuredClone(raw)]});
  }
  assert.equal(request.type,'JournalEntryPage');assert.equal(request.operation.parentUuid,`JournalEntry.${raw._id}`);
  const page=structuredClone(request.operation.data[0]);
  if(raw.pages.some(p=>p._id===page._id))return ack({...envelope,error:{class:'ServerError',message:`The _id [${page._id}] already exists within the parent collection: JournalEntry [${raw._id}] pages`}});
  raw.pages.push(page);ack({...envelope,result:[structuredClone(page)]});
 }
 async function addTab(id,userId){
  const hooks=new Map(),listeners=new Map();let hookId=0;
  const actor={...structuredClone(actorSource),items:[],rules:[],synthetics:{},attributes:{immunities:[]},hasCondition:()=>false,getStatistic:()=>null,getRollOptions:()=>[],testUserPermission:user=>user.isGM||user.id==='O',getActiveTokens:()=>[{actor}],
   async update(data){
    apply(actorSource,data);for(const peer of tabs)apply(peer.actor,data);
    if(Object.hasOwn(data,'system.resources.focus.value'))effects.push({tab:id,userId,changes:structuredClone(data)});
    emitHook('updateActor',actor,data,{},userId);return actor;
   }};
  const tab={id,hooks,listeners,actor};
  const game={user:users.get(userId),users,actors:new Map([['H',actor]]),messages:new Map(),packs:new Map(),modules:new Map(),system:{version:'8.5.1'},pf2e:{actions:new Map()},
   settings:{get:(scope,key)=>scope===MODULE_ID&&key==='explorationLedgerUUID'?configured:'',async set(scope,key,value){assert.equal(userId,'G');configured=value;emitHook('updateSetting',{key:`${scope}.${key}`});return value}},
   time:{get worldTime(){return worldTime},async advance(dt,options){assert.equal(userId,'G');worldTime+=dt;clocks.push({tab:id,worldTime,dt,options:structuredClone(options),userId});emitHook('updateWorldTime',worldTime,dt,options,userId);return worldTime}},
   socket:{id:`socket-${id}`,connected:true,on(event,handler){const set=listeners.get(event)??new Set();set.add(handler);listeners.set(event,set)},off(event,handler){listeners.get(event)?.delete(handler)},emit(event,packet,...args){
    if(event==='modifyDocument')return documentRequest(tab,packet,args[0]);
    assert.equal(event,OWNER_TRANSPORT_CHANNEL);const [routing,ack]=args;
    assert.deepEqual(routing,{recipients:[packet.receiverUserId]});
    const recipients=tabs.filter(peer=>routing.recipients.includes(peer.game.user.id));
    packets.push({senderTab:id,senderId:userId,recipients:recipients.map(peer=>peer.id),packet:structuredClone(packet)});
    for(const peer of recipients)for(const handler of peer.listeners.get(event)??[])queueMicrotask(()=>handler(structuredClone(packet),userId));
    ack?.();
   }}
  };
  tab.game=game;tabs.push(tab);
  const canvas={tokens:{controlled:[{actor}]}};
  const previousCanvas=globalThis.canvas;globalThis.canvas=canvas;
  const nativeCasts={addMatcher(){},addCapture(){},addActorUpdateMiddleware(){},addConsumePolicy(){},addObserver(){}};
  const Hooks={on(event,handler){const id=++hookId;hooks.set(id,[event,handler]);return id},off(event,id){hooks.delete(id)}};
  const runtime=createExplorationRuntime({game,Hooks,fromUuid:async uuid=>uuid===actor.uuid?actor:null,nativeCasts,onError:error=>errors.push(error)});
  globalThis.canvas=previousCanvas;tab.runtime=runtime;tab.boundaries=[];
  await runtime.bind({});await runtime.register({socket:null});
  // Observe the real adapter call without replacing the registered provider or its context.
  const complete=runtime.refocusAdapter.complete;
  runtime.refocusAdapter.complete=async(activity,ctx)=>{
   assert.equal(runtime.ownerOperations.isExecutionContext(ctx,activity.id),true);
   assert.equal(runtime.ownerOperations.isExecutionContext({...ctx},activity.id),false);
   for(const peer of tabs.filter(peer=>peer!==tab))assert.equal(peer.runtime.ownerOperations.isActivityContext(ctx,activity.id),false);
   tab.boundaries.push({activity:structuredClone(activity),mode:ctx.nativeDialogMode,signal:ctx.executionSignal,ctx});
   return complete(activity,ctx);
  };
  // Offline Workbench boundary: production Refocus scope authorizes this one intent/focus update.
  game.PF2eWorkbench={async refocus(selected){
   assert.deepEqual(selected,[actor]);const activity=runtime.refocusAdapter.getCurrent(actor);assert.ok(activity);
   if(nativeGate){nativeGate.entered();await nativeGate.wait}
   const before=actor.system.resources.focus.value,after=runtime.refocusAdapter.commitValue(activity,actor,actor.system.resources.focus.max);
   await actor.update({'system.resources.focus.value':after,[`flags.${MODULE_ID}.avRefocusIntent`]:{nonce:activity.id,actorUuid:actor.uuid,userId,startedAt:activity.startedAt,before,after}});
  }};
  return tab;
 }
 async function provision(tab){
  const options={issuersStopped:true,clientsReloaded:true,recoveryDisabled:true};
  const result=await tab.runtime.api.storage.provision(options);assert.equal(result.requiresReload,false);
  assert.equal((await tab.runtime.api.storage.initialize(options)).state,'ready');
 }
 async function finished(tab,id){
  const deadline=performance.now()+3000;
  while(performance.now()<deadline){const data=await tab.runtime.api.snapshot(id);if(data.session.status!=='running')return data;await new Promise(resolve=>setImmediate(resolve))}
  throw Error('offline-public-runtime-did-not-settle');
 }
 return {tabs,packets,documents,effects,clocks,errors,addTab,provision,finished,get raw(){return raw},holdNative(){
  let entered,release;const boundary=new Promise(resolve=>{entered=resolve}),wait=new Promise(resolve=>{release=resolve});nativeGate={entered,wait};return {release,async waitForBoundary(){
   let timer;try{await Promise.race([boundary,new Promise((resolve,reject)=>{timer=setTimeout(()=>reject(Error('offline-native-boundary-not-reached')),3000)})])}finally{clearTimeout(timer)}
  }};
 },dispose(){for(const tab of tabs)tab.runtime.ownerOperations.dispose()}};
}

for(const mapped of [true,false])test(`public runtime ${mapped?'mapped OWNER':'default GM'} start settles one ordinary Refocus through its native adapter`,async t=>{
 const f=publicRuntimeFixture();t.after(()=>f.dispose());
 const driver=await f.addTab('driver','G'),peer=await f.addTab('peer','G'),ownerOne=await f.addTab('owner-one','O'),ownerTwo=await f.addTab('owner-two','O'),unrelated=await f.addTab('unrelated','P');
 await f.provision(driver);
 const gate=f.holdNative();t.after(()=>gate.release());
 const config={actorUUIDs:['Actor.H'],requireFullFocus:true,budgetSeconds:1200,maxActivities:1,...mapped?{nativeOwnerByActor:{'Actor.H':'O'}}:{}};
 const session=await driver.runtime.api.start(config);
 await assert.rejects(peer.runtime.api.start(config),/recovery-session-already-running/);
 await assert.rejects(ownerOne.runtime.api.start(config),/active-gm-required/);
 await gate.waitForBoundary();
 const beforeRestore=structuredClone(f.raw.pages);
 const restored=await f.addTab('restored-gm','G'),observed=await restored.runtime.api.snapshot(session.id);
 assert.equal(observed.session.status,'running');assert.equal(observed.activities[0].state,'completing');assert.equal(observed.activities[0].executor.state,'granted');
 assert.equal(restored.boundaries.length,0);assert.deepEqual(f.raw.pages,beforeRestore);assert.equal(f.effects.length,0);assert.equal(f.clocks.length,1);
 gate.release();
 const data=await f.finished(driver,session.id);
 assert.equal(data.session.status,'complete');assert.equal(data.session.stopReason,'goals-met');
 assert.deepEqual(data.session.nativeOwnerByActor,mapped?{'Actor.H':'O'}:{});
 assert.equal(data.activities.length,1);const activity=data.activities[0];
 assert.equal(activity.providerId,'refocus');assert.equal(activity.state,'confirmed');assert.deepEqual(activity.patientUUIDs,[]);assert.deepEqual(activity.hpPoolUUIDs,[]);
 assert.deepEqual(activity.source,mapped?{type:'coordinator',ownerId:'O'}:{type:'coordinator'});
 assert.equal(activity.executor.state,'settled');assert.equal(activity.executor.ownerUserId,mapped?'O':'G');
 assert.equal(data.clocks.length,1);assert.equal(data.clocks[0].state,'confirmed');assert.equal(data.clocks[0].evidence[0].options.pf2eThirdPartyAutomation.exploration.sessionId,session.id);
 assert.equal(f.clocks.length,1);assert.equal(f.clocks[0].dt,600);assert.equal(f.effects.length,1);
 const winner=f.tabs.find(tab=>tab.id===f.effects[0].tab);assert.ok(winner);
 assert.equal(winner.game.user.id,mapped?'O':'G');assert.equal(winner.boundaries.length,1);
 assert.equal(winner.boundaries[0].mode,mapped?'owner-preference':'automatic');assert.equal(winner.boundaries[0].signal.aborted,false);
 assert.equal(f.tabs.reduce((n,tab)=>n+tab.boundaries.length,0),1);
 assert.equal(winner.actor.system.resources.focus.value,1);assert.equal(winner.actor.system.attributes.hp.value,20);
 assert.deepEqual(activity.proof.receiptIds,[activity.id]);assert.deepEqual(activity.proof.checkIds,[]);assert.deepEqual(activity.proof.resultIds,[]);
 const saved=winner.actor.flags[MODULE_ID].explorationExecutions[activity.id];assert.equal(saved.state,'done');assert.equal(samePermit(saved,activity.executor),true);
 assert.equal(f.effects[0].changes[`flags.${MODULE_ID}.avRefocusIntent`].nonce,activity.id);
 assert.equal(f.packets.some(row=>row.recipients.includes(unrelated.id)),false);
 if(mapped){
  const claims=f.packets.filter(row=>row.packet.kind==='claim'),grants=f.packets.filter(row=>row.packet.kind==='grant');
  assert.equal(claims.length,2);assert.equal(new Set(claims.map(row=>row.packet.ownerClientNonce)).size,2);assert.equal(new Set(claims.map(row=>row.packet.attemptNonce)).size,2);
  assert.equal(grants.length,1);assert.equal(grants[0].senderTab,driver.id);assert.equal(grants[0].packet.ownerClientNonce,activity.executor.ownerClientNonce);
  const winningClaim=claims.find(row=>row.packet.ownerClientNonce===activity.executor.ownerClientNonce);
  assert.equal(winningClaim.senderTab,winner.id);assert.equal(winningClaim.packet.attemptNonce,activity.executor.attemptNonce);assert.equal(winningClaim.packet.requestId,activity.executor.requestId);
  assert.equal(f.packets.filter(row=>row.packet.kind==='completion').length,1);
 }else assert.equal(f.packets.length,0);
 const pages=structuredClone(f.raw.pages),effectCount=f.effects.length,clockCount=f.clocks.length;
 const snapshot=await peer.runtime.api.snapshot(session.id);assert.equal(snapshot.session.status,'complete');assert.deepEqual(f.raw.pages,pages);assert.equal(f.effects.length,effectCount);assert.equal(f.clocks.length,clockCount);
 assert.deepEqual(f.errors,[]);
});

test('public runtime manual start stays recording without an automatic permit or offline native effect',async t=>{
 const f=publicRuntimeFixture();t.after(()=>f.dispose());const gm=await f.addTab('driver','G');await f.provision(gm);
 const session=await gm.runtime.api.start({actorUUIDs:['Actor.H'],manual:true,requireFullFocus:true});
 const data=await gm.runtime.api.snapshot(session.id);
 assert.equal(data.session.status,'recording');assert.deepEqual(data.session.nativeOwnerByActor,{});assert.deepEqual(data.activities,[]);assert.deepEqual(data.clocks,[]);
 assert.equal(f.effects.length,0);assert.equal(f.clocks.length,0);assert.equal(f.packets.length,0);assert.deepEqual(f.errors,[]);
});
