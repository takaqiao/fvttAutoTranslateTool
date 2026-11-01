import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createExplorationRuntime} from '../../scripts/exploration/runtime.mjs';
import {MODULE_ID} from '../../scripts/exploration/schema.mjs';
import {OWNER_TRANSPORT_CHANNEL} from '../../scripts/exploration/owner-transport.mjs';

function publicRuntimeFixture({sharedPool=false}={}) {
 const tabs=[],packets=[],documents=[],effects=[],clocks=[],errors=[],manualPackets=[],pauses=[];
 const users=new Map(['G','O','P'].map(id=>[id,{id,isGM:id==='G',active:true,flags:{},getFlag(scope,key){return this.flags[scope]?.[key]},async setFlag(scope,key,value){(this.flags[scope]??={})[key]=value}}]));
 users.activeGM=users.get('G');
 let configured='',raw=null,worldTime=0,actorGate=null,paused=false,actorAckUnknown=false,claimAckUnknown=false;
 const actorSource={id:'H',uuid:'Actor.H',name:'Offline Refocus',type:'character',level:1,canAct:true,isDead:false,hardness:0,isOwner:true,modeOfBeing:'living',flags:{},system:{attributes:{hp:{value:20,max:20,temp:0,negativeHealing:false}},resources:{focus:{value:0,max:1}}}};
 const sources=new Map([['H',actorSource],...sharedPool?[['E',{...structuredClone(actorSource),id:'E',uuid:'Actor.E',name:'Shared patient'}]]:[]]);
 if(sharedPool)sources.get('E').system.attributes.hp.value=7;
 function apply(document,data){for(const [path,value]of Object.entries(data)){const keys=path.split('.');let at=document;for(const key of keys.slice(0,-1))at=at[key]??={};at[keys.at(-1)]=structuredClone(value)}}
 function emitHook(event,...args){for(const tab of tabs)for(const [name,handler]of tab.hooks.values())if(name===event)handler(...args)}
 function documentRequest(tab,request,ack){
  documents.push({tab:tab.id,userId:tab.game.user.id,request:structuredClone(request)});
  const envelope={type:request.type,action:request.action,operation:structuredClone(request.operation),broadcast:false,userId:tab.game.user.id};
  if(request.action==='get'){if(raw&&!tab.game.user.isGM)return ack({...envelope,error:{class:'PermissionError',message:'private-ledger'}});return ack({...envelope,result:raw?[structuredClone(raw)]:[]})}
  assert.equal(tab.game.user.isGM,true,'the fixture server authenticates document writers');
  if(request.type==='JournalEntry'){
   assert.equal(raw,null);raw={...structuredClone(request.operation.data[0]),_id:'ROOT000000000001',pages:[]};
   return ack({...envelope,result:[structuredClone(raw)]});
  }
  assert.equal(request.type,'JournalEntryPage');assert.equal(request.operation.parentUuid,`JournalEntry.${raw._id}`);
  const page=structuredClone(request.operation.data[0]);
  if(raw.pages.some(p=>p._id===page._id))return ack({...envelope,error:{class:'ServerError',message:`The _id [${page._id}] already exists within the parent collection: JournalEntry [${raw._id}] pages`}});
  raw.pages.push(page);
  const revision=JSON.parse(page.flags[MODULE_ID].explorationRevision);
  if(claimAckUnknown&&Object.values(revision.changes?.sessions??{}).some(session=>session.computedRecovery?.batches.some(batch=>batch.state==='claimed'))){claimAckUnknown=false;return ack({})}
  ack({...envelope,result:[structuredClone(page)]});
 }
 async function addTab(id,userId){
  const hooks=new Map(),listeners=new Map();let hookId=0;
  const actors=new Map();
  for(const [actorId,source]of sources){
   const actor={...structuredClone(source),isOwner:userId!=='P',items:[],rules:[],synthetics:{},hasCondition:()=>false,getStatistic:()=>null,getRollOptions:()=>[],testUserPermission:user=>user.isGM||user.id==='O',getActiveTokens:()=>[{actor}],
    async update(data){
     assert.equal(userId,'G');apply(source,data);for(const peer of tabs)apply(peer.actors.get(actorId),data);
     effects.push({tab:id,userId,actorUUID:actor.uuid,changes:structuredClone(data)});emitHook('updateActor',actor,data,{},userId);
     if(actorGate){const gate=actorGate;actorGate=null;gate.entered();await gate.wait}
     if(actorAckUnknown){actorAckUnknown=false;throw Error('actor-acknowledgement-unknown')}
     return actor;
    }};
   actor.attributes={hp:actor.system.attributes.hp,immunities:[],resistances:[],weaknesses:[]};
   if(sharedPool&&actorId==='E')actor.modules={'pf2e-toolbelt':{shareData:{data:{health:true}}}};
   actors.set(actorId,actor);
  }
  const actor=actors.get('H'),tab={id,hooks,listeners,actor,actors};
  const game={user:users.get(userId),users,actors,messages:new Map(),packs:new Map(),modules:new Map(sharedPool?[['pf2e-toolbelt',{active:true}]]:[]),system:{id:'pf2e',version:'8.6.1'},pf2e:{actions:new Map()},
   get paused(){return paused},togglePause(state,{broadcast=false}={}){assert.equal(userId,'G');assert.equal(broadcast,true);pauses.push({tab:id,state});paused=state;emitHook('pauseGame',state);return state},
   settings:{get:(scope,key)=>scope===MODULE_ID&&key==='explorationLedgerUUID'?configured:scope==='pf2e-toolbelt'&&key==='shareData.enabled'?sharedPool:'',async set(scope,key,value){assert.equal(userId,'G');configured=value;emitHook('updateSetting',{key:`${scope}.${key}`});return value}},
   time:{get worldTime(){return worldTime},async advance(dt,options){assert.equal(userId,'G');assert.equal(game.paused,true);worldTime+=dt;clocks.push({tab:id,worldTime,dt,options:structuredClone(options),userId});emitHook('updateWorldTime',worldTime,dt,options,userId);return worldTime}},
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
  if(sharedPool)game.toolbelt={api:{shareData:{getMasterInMemory:patient=>patient.id==='E'?actor:null,getSlavesInMemory:master=>master.id==='H'?[actors.get('E')]:[]}}};
  tab.game=game;tabs.push(tab);
  const canvas={tokens:{controlled:[{actor}]}};
  const previousCanvas=globalThis.canvas;globalThis.canvas=canvas;
  const nativeCasts={addMatcher(){},addCapture(){},addActorUpdateMiddleware(){},addConsumePolicy(){},addObserver(){}};
  const Hooks={on(event,handler){const id=++hookId;hooks.set(id,[event,handler]);return id},off(event,id){hooks.delete(id)}};
  const runtime=createExplorationRuntime({game,Hooks,fromUuid:async uuid=>[...actors.values()].find(actor=>actor.uuid===uuid)??null,nativeCasts,onError:error=>errors.push(error)});
  globalThis.canvas=previousCanvas;tab.runtime=runtime;tab.boundaries=[];
  tab.manualHandlers=new Map();const socket={register:(name,handler)=>tab.manualHandlers.set(name,handler),executeAsGM:async(name,...args)=>{const gm=tabs.find(peer=>peer.game.user.id===users.activeGM.id&&peer.id==='driver');manualPackets.push({senderId:userId,name,args:structuredClone(args)});return gm.manualHandlers.get(name).apply({socketdata:{userId}},structuredClone(args))}};
  await runtime.bind({});await runtime.register({socket});
  const complete=runtime.refocusAdapter.complete;
  runtime.refocusAdapter.complete=async(...args)=>{tab.boundaries.push('refocus-adapter');return complete(...args)};
  game.PF2eWorkbench={async refocus(){tab.boundaries.push('Workbench.refocus');throw Error('unexpected-native-refocus')}};
  return tab;
 }
 async function provision(tab,seed){
  const options={issuersStopped:true,clientsReloaded:true,recoveryDisabled:true};
  const result=await tab.runtime.api.storage.provision(options);assert.equal(result.requiresReload,false);
  if(seed)raw.flags[MODULE_ID].explorationLedger=structuredClone(seed);
  assert.equal((await tab.runtime.api.storage.initialize(options)).state,'ready');
 }
 async function finished(tab,id){
  const deadline=performance.now()+3000;
  while(performance.now()<deadline){const data=await tab.runtime.api.snapshot(id);if(data.session.status!=='running')return data;await new Promise(resolve=>setImmediate(resolve))}
  throw Error('offline-public-runtime-did-not-settle');
 }
 return {tabs,packets,documents,effects,clocks,errors,manualPackets,pauses,addTab,provision,finished,emitHook,get raw(){return raw},loseActorAcknowledgement:()=>actorAckUnknown=true,loseClaimAcknowledgement:()=>claimAckUnknown=true,disconnect(tab){tab.game.socket.connected=false;for(const handler of [...tab.listeners.get('disconnect')??[]])handler()},holdActorAcknowledgement(){
  let entered,release;const boundary=new Promise(resolve=>{entered=resolve}),wait=new Promise(resolve=>{release=resolve});actorGate={entered,wait};return {release,async waitForBoundary(){
   let timer;try{await Promise.race([boundary,new Promise((resolve,reject)=>{timer=setTimeout(()=>reject(Error('actor-write-boundary-not-reached')),3000)})])}finally{clearTimeout(timer)}
  }};
 },dispose(){for(const tab of tabs){tab.runtime.ownerOperations.dispose();tab.runtime.hpPools.dispose()}}};
}

function fixture(t,options){
 const f=publicRuntimeFixture(options),previousChat=globalThis.ChatMessage;f.chatCalls=[];
 globalThis.ChatMessage={async create(data){f.chatCalls.push(data);throw Error('unexpected-native-chat')}};
 t.after(()=>{f.dispose();globalThis.ChatMessage=previousChat});return f;
}
const config={actorUUIDs:['Actor.H'],requireFullFocus:true,budgetSeconds:1200};
const recoveryPackets=f=>f.packets.filter(row=>row.packet.operationId!=='manual-record-driver');
function noNativeCalls(f){assert.deepEqual(f.chatCalls,[]);assert.ok(f.tabs.every(tab=>tab.boundaries.length===0));assert.deepEqual(recoveryPackets(f),[])}

for(const strategy of ['success-first','time-first'])test(`public computed ${strategy} restores focus with one batch and no owner operation`,async t=>{
 const f=fixture(t),gm=await f.addTab('driver','G'),peer=await f.addTab('peer','G'),owner=await f.addTab('owner','O');await f.provision(gm);
 const session=await gm.runtime.api.start({...config,recoveryStrategy:strategy,nativeOwnerByActor:{'Actor.H':'O'}}),data=await f.finished(gm,session.id);
 assert.equal(data.session.status,'complete',data.session.stopReason);assert.equal(data.session.executionMode,'computed-v1');assert.equal(data.session.stopReason,'goals-met');
 assert.equal(data.activities.length,0);assert.equal(data.session.computedRecovery.batches.length,1);
 const batch=data.session.computedRecovery.batches[0];assert.equal(batch.state,'confirmed');assert.deepEqual(batch.events.map(event=>event.providerId),['refocus']);
 assert.equal(batch.receipts.actors.actors[0].actorUUID,'Actor.H');assert.equal(batch.receipts.actors.actors[0].batchId,batch.id);
 assert.equal(gm.actor.system.resources.focus.value,1);assert.equal(owner.actor.system.resources.focus.value,1);assert.equal(gm.actor.system.attributes.hp.value,20);
 assert.equal(f.effects.length,1);assert.equal(f.effects[0].tab,'driver');assert.equal(f.effects[0].changes[`flags.${MODULE_ID}.computedRecoveryBatch`].id,batch.id);
 assert.equal(data.session.computedRecovery.summary.actors[0].focusAfter,1);assert.equal(data.clocks.length,1);assert.equal(data.clocks[0].state,'confirmed');assert.deepEqual(f.clocks.map(row=>row.dt),[600]);
 assert.equal(owner.game.paused,true);assert.deepEqual(f.pauses,[{tab:'driver',state:true}]);noNativeCalls(f);assert.deepEqual(f.packets,[]);assert.equal(gm.game.messages.size,0);
 const pages=structuredClone(f.raw.pages);assert.equal((await peer.runtime.api.snapshot(session.id)).session.status,'complete');assert.deepEqual(f.raw.pages,pages);assert.equal(f.effects.length,1);
 await assert.rejects(owner.runtime.api.snapshot(session.id),/gm-read-required/);await assert.rejects(owner.runtime.api.start(config),/active-gm-required/);assert.equal(f.documents.some(row=>row.userId!=='G'),false);assert.deepEqual(f.errors,[]);
});

test('two public GM tabs start together but only one session writes Actor focus and time',async t=>{
 const f=fixture(t),one=await f.addTab('driver','G'),two=await f.addTab('peer','G');await f.provision(one);
 const gate=f.holdActorAcknowledgement();t.after(()=>gate.release());
 const started=await Promise.allSettled([one.runtime.api.start({...config,id:'one'}),two.runtime.api.start({...config,id:'two'})]);
 assert.equal(started.filter(row=>row.status==='fulfilled').length,1);assert.match(started.find(row=>row.status==='rejected').reason.message,/recovery-session-already-running/);
 const winner=started[0].status==='fulfilled'?one:two,session=started.find(row=>row.status==='fulfilled').value;await gate.waitForBoundary();
 const restored=await f.addTab('restored','G'),before=structuredClone(f.raw.pages);assert.equal((await restored.runtime.api.snapshot(session.id)).session.computedRecovery.batches[0].state,'claimed');assert.deepEqual(f.raw.pages,before);
 gate.release();assert.equal((await f.finished(winner,session.id)).session.status,'complete');assert.equal(f.effects.length,1);assert.equal(f.effects[0].tab,winner.id);assert.equal(f.clocks.length,1);noNativeCalls(f);
});

test('public snapshots use the shared HP master while each member refocuses once',async t=>{
 const f=fixture(t,{sharedPool:true}),gm=await f.addTab('driver','G');await f.provision(gm);
 const session=await gm.runtime.api.start({...config,actorUUIDs:['Actor.H','Actor.E']}),data=await f.finished(gm,session.id);
 assert.equal(data.session.status,'complete',data.session.stopReason);assert.deepEqual(data.session.goalsByPool,[{poolUUID:'Actor.H',targetHP:20}]);
 assert.deepEqual(data.actors.map(actor=>[actor.actorUUID,actor.pool.poolUUID,actor.hp.value,actor.focus.value]),[['Actor.H','Actor.H',20,1],['Actor.E','Actor.H',20,1]]);
 assert.equal(gm.actors.get('E').system.attributes.hp.value,7);assert.deepEqual(f.effects.map(row=>row.actorUUID).sort(),['Actor.E','Actor.H']);assert.ok(f.effects.every(row=>!Object.hasOwn(row.changes,'system.attributes.hp.value')));
 assert.deepEqual(f.clocks.map(row=>row.dt),[600]);assert.equal(data.session.computedRecovery.batches[0].events.length,2);noNativeCalls(f);
});

test('public runtime refuses a manual start without a session, an automatic permit or an offline native effect',async t=>{
 const f=fixture(t),gm=await f.addTab('driver','G');await f.provision(gm);const pages=structuredClone(f.raw.pages);
 await assert.rejects(gm.runtime.api.start({...config,manual:true}),/manual-recording-unavailable/);
 assert.deepEqual(f.raw.pages,pages);assert.equal(f.effects.length,0);assert.equal(f.clocks.length,0);assert.equal(f.packets.length,0);assert.deepEqual(f.errors,[]);assert.equal(gm.game.paused,false);assert.deepEqual(f.pauses,[]);
});

test('unknown Actor acknowledgement keeps its original samples and refuses public replay',async t=>{
 const f=fixture(t),gm=await f.addTab('driver','G');await f.provision(gm);f.loseActorAcknowledgement();
 const session=await gm.runtime.api.start(config),data=await f.finished(gm,session.id),batch=data.session.computedRecovery.batches[0];
 assert.equal(data.session.status,'paused');assert.equal(batch.state,'uncertain');assert.equal(gm.actor.system.resources.focus.value,1);assert.equal(f.effects.length,1);assert.equal(f.clocks.length,1);
 const restored=await f.addTab('restored','G'),before=structuredClone(f.raw.pages);assert.deepEqual((await restored.runtime.api.snapshot(session.id)).session.computedRecovery.batches,[batch]);assert.deepEqual(f.raw.pages,before);
 await assert.rejects(restored.runtime.api.resume(session.id),/unresolved-computed-batch/);await assert.rejects(restored.runtime.api.start(config),/unresolved-evidence-no-replay/);
 assert.equal(f.effects.length,1);assert.equal(f.clocks.length,1);noNativeCalls(f);
});

test('unknown ledger claim acknowledgement does not advance time or write sampled focus',async t=>{
 const f=fixture(t),gm=await f.addTab('driver','G');await f.provision(gm);f.loseClaimAcknowledgement();
 const session=await gm.runtime.api.start(config),data=await f.finished(gm,session.id);
 assert.equal(data.session.status,'paused');assert.equal(data.session.computedRecovery.batches.length,1);assert.equal(data.session.computedRecovery.batches[0].state,'claimed');
 assert.equal(data.session.computedRecovery.batches[0].events[0].providerId,'refocus');assert.equal(gm.actor.system.resources.focus.value,0);assert.equal(f.effects.length,0);assert.equal(f.clocks.length,0);
 const restored=await f.addTab('restored','G');await assert.rejects(restored.runtime.api.resume(session.id),/unresolved-computed-batch/);await assert.rejects(restored.runtime.api.start(config),/unresolved-evidence-no-replay/);assert.equal(f.effects.length,0);assert.equal(f.clocks.length,0);
});

test('disconnect after an Actor write leaves a claimed batch for explicit takeover without replay',async t=>{
 const f=fixture(t),gm=await f.addTab('driver','G');await f.provision(gm);const gate=f.holdActorAcknowledgement();t.after(()=>gate.release());
 const session=await gm.runtime.api.start(config);await gate.waitForBoundary();f.disconnect(gm);gate.release();
 const restored=await f.addTab('restored','G'),data=await restored.runtime.api.snapshot(session.id),batch=data.session.computedRecovery.batches[0];
 assert.equal(batch.state,'claimed');assert.equal(restored.actor.system.resources.focus.value,1);
 await restored.runtime.api.takeover(session.id);await assert.rejects(restored.runtime.api.resume(session.id),/unresolved-computed-batch/);
 const saved=await restored.runtime.api.snapshot(session.id);assert.equal(saved.session.computedRecovery.batches[0].state,'uncertain');assert.deepEqual(saved.session.computedRecovery.batches[0].events,batch.events);
 assert.equal(f.effects.length,1);assert.equal(f.clocks.length,1);noNativeCalls(f);
});

test('computed recovery refuses new registration windows and keeps player ledger access private',async t=>{
 const f=fixture(t),gm=await f.addTab('driver','G'),owner=await f.addTab('owner','O');await f.provision(gm);const gate=f.holdActorAcknowledgement();t.after(()=>gate.release());
 const session=await gm.runtime.api.start(config);await gate.waitForBoundary();
 await assert.rejects(gm.runtime.api.openActivityCheckpoint(session.id),/computed-manual-checkpoint-unavailable/);
 await assert.rejects(owner.runtime.api.getActivityCheckpoint('Actor.H'),/activity-checkpoint-closed/);
 await assert.rejects(owner.runtime.api.openActivityCheckpoint(session.id),/active-gm-required/);await assert.rejects(owner.runtime.api.snapshot(session.id),/gm-read-required/);
 await assert.rejects(owner.runtime.api.record({registrationId:'late',actorUUID:'Actor.H',label:'Search',durationSeconds:300}),/activity-checkpoint-closed/);
 gate.release();const data=await f.finished(gm,session.id);assert.equal(data.session.status,'complete');assert.equal(data.session.activityCheckpoint,undefined);assert.deepEqual(data.activities,[]);
 assert.equal(f.documents.some(row=>row.userId!=='G'),false);noNativeCalls(f);assert.equal(f.effects.length,1);assert.equal(f.clocks.length,1);
});

test('legacy declarations remain readable through the authenticated GM bridge without player ledger reads',async t=>{
 const f=fixture(t),gm=await f.addTab('driver','G'),owner=await f.addTab('owner','O'),other=await f.addTab('other','P');
 const binding={id:'old-window',sessionId:'old',rootUUID:'JournalEntry.ROOT000000000001',epoch:'old-epoch',observationNonce:'old-observation',from:0};
 const row={registrationId:'search',source:{userId:'O'},declaration:{actorUUID:'Actor.H',label:'Search',durationSeconds:600},checkpointBinding:binding,status:'completed'};
 const session={id:'old',status:'paused',actorUUIDs:['Actor.H'],startedAt:0,cursorAt:600,budgetEndsAt:1200,activityIds:[],goalsByPool:[{poolUUID:'Actor.H',targetHP:20}],activityCheckpoint:{...binding,phase:'settled',registrations:{search:row}}};
 await f.provision(gm,{sessions:{old:session},activities:{},clocks:{}});const before=structuredClone(f.raw.pages);
 assert.equal((await gm.runtime.api.snapshot('old')).session.executionMode,undefined);assert.deepEqual(await owner.runtime.api.lookupCheckpointActivity(binding,'search','Actor.H'),row);
 await assert.rejects(other.runtime.api.lookupCheckpointActivity(binding,'search','Actor.H'),/manual-actor-not-allowed/);await assert.rejects(owner.runtime.api.snapshot('old'),/gm-read-required/);await assert.rejects(gm.runtime.api.resume('old'),/legacy-recovery-view-only/);
 assert.deepEqual(f.raw.pages,before);assert.equal(f.documents.some(request=>request.userId!=='G'),false);assert.deepEqual(f.manualPackets.map(packet=>packet.name),['exploration:lookupCheckpointActivity','exploration:lookupCheckpointActivity']);
 assert.equal(f.effects.length,0);assert.equal(f.clocks.length,0);noNativeCalls(f);assert.deepEqual(f.pauses,[]);
});
