import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {createHash} from 'node:crypto';
import {authorityFixture} from './authority-fixture.mjs';
import {createManualRecordBridge} from '../../scripts/exploration/manual-record.mjs';

const channel='module.pf2e-third-party-automation';
const socketlibSource=fs.readFileSync(process.env.FVTT_SOCKETLIB_SOURCE??'C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/socketlib/src/socketlib.js','utf8');
assert.equal(createHash('sha256').update(socketlibSource).digest('hex'),'5643449d1e355b4b773b4675eea99f09003c8aad254dca5bf74008386aec7fd8');
const turn=async()=>{for(let i=0;i<40;i++)await new Promise(resolve=>setImmediate(resolve))};

async function fixture({timeoutMs=1000,manualWindow=false,beforeActor=async()=>{},editRequest=()=>{}}={}){
 const store=await authorityFixture(),ledger=store.client('driver');
 const session=await ledger.createSession({id:'S',actorUUIDs:['Actor.A'],startedAt:0,cursorAt:0,budgetEndsAt:900});
 const binding={id:'C',sessionId:'S',rootUUID:store.rootUUID,epoch:'epoch',observationNonce:'observation',from:0,...manualWindow?{to:600}:{}};
 if(manualWindow)await ledger.updateSession('S',{manualCheckpoint:{...binding,phase:'open'}},{leaseNonce:session.driver.leaseNonce});else await ledger.openActivityCheckpoint(binding,{leaseNonce:session.driver.leaseNonce,guard:()=>true});
 const actor={id:'A',uuid:'Actor.A',name:'A',testUserPermission:user=>['G','P'].includes(user?.id)},clients=[],packets=[],replies=[];let playerReads=0,reservations=0;
 function make(role,userId){
  const users=new Map(['G','P','X'].map(id=>[id,{id,active:true,isGM:id==='G',isSelf:id===userId}]));users.activeGM=users.get('G');
  const listeners=new Map(),game={user:users.get(userId),userId,users,actors:new Map([['A',actor]]),time:{worldTime:0}};
  const socket={on(name,fn){if(!listeners.has(name))listeners.set(name,new Set());listeners.get(name).add(fn)},off(name,fn){listeners.get(name)?.delete(fn)},emit(name,packet,options){
   packet=structuredClone(packet);if(packet.kind==='continuation')editRequest(packet,role);
   const saved={role,name,packet:structuredClone(packet),options:structuredClone(options)};packets.push(saved);
   if(packet.type===3||['continuation-ack','denied'].includes(packet.kind)){replies.push(saved);return}
   for(const client of clients)if(!options?.recipients||options.recipients.includes(client.game.user.id))for(const fn of client.listeners.get(name)??[])fn(structuredClone(packet),userId);
  }};game.socket=socket;
  const context=vm.createContext({game,Hooks:{on(){},once(){}},window:{},console,Function,Array,foundry:{utils:{randomID:()=>crypto.randomUUID()}},errors:new Proxy({},{get:()=>Error})});
  vm.runInContext(socketlibSource.replace(/^import \* as errors from "\.\/errors\.js";\s*/,'')+'\nglobalThis.NativeSocketlibSocket=SocketlibSocket;',context);
  const nativeSocket=new context.NativeSocketlibSocket('pf2e-third-party-automation','module'),writer=store.client(role),scope={sessionId:'S',worldTime:0,...role==='driver'?{leaseNonce:session.driver.leaseNonce}:{}};
  const privateRead=fn=>{if(userId==='P'){playerReads++;throw Error('player-private-read')}return fn()};
  const bridge=createManualRecordBridge({game,timeoutMs,fromUuid:async()=>{await beforeActor(role);return privateRead(()=>actor)},getSession:()=>privateRead(()=>writer.getSession('S')),getActivities:async()=>privateRead(()=>writer.snapshot('S').then(saved=>saved.activities)),checkpointContext:()=>({...scope}),ownsSession:(...args)=>writer.ownsSession(...args),enrollCheckpointActivity:(...args)=>writer.enrollCheckpointActivity(...args),lookupCheckpointActivity:(...args)=>writer.lookupCheckpointActivity(...args),observe:()=>assert.fail('recording-route-used'),reserveSource:async captured=>{if(!writer.ownsSession(await writer.getSession('S'),scope))throw Error('session-driver-required');reservations++;return {activityId:'reserved',reservationId:'reserved',checkpointBinding:captured}}});
  const client={role,game,listeners,nativeSocket,bridge,scope,writer};clients.push(client);bridge.register(nativeSocket);return client;
 }
 const driver=make('driver','G'),peer=make('peer','G'),player=make('player','P');
 async function request(call,order){
  const first=replies.length,pending=call().then(value=>({ok:true,value}),error=>({ok:false,error})),deadline=performance.now()+2000;
  while(replies.length===first&&performance.now()<deadline)await new Promise(resolve=>setImmediate(resolve));await turn();
  const received=replies.slice(first).sort((a,b)=>a.role===b.role?0:a.role===(order==='peer-first'?'peer':'driver')?-1:1);
  for(const reply of received)for(const client of clients)for(const fn of client.listeners.get(reply.name)??[])fn(structuredClone(reply.packet),'G');
  const result=await pending;return {...result,replies:received};
 }
 const input=()=>({registrationId:'registration',checkpointBinding:{...binding},actorUUID:'Actor.A',label:'Search',durationSeconds:300,durationSource:{type:'user-declared',detail:'agreed'},dependsOn:[]});
 const deliver=(reply,senderId='G')=>{for(const client of clients)for(const fn of client.listeners.get(reply.name)??[])fn(structuredClone(reply.packet),senderId)};
 return {store,ledger,binding,driver,peer,player,actor,packets,replies,request,input,deliver,reservations:()=>reservations,playerReads:()=>playerReads,dispose(){for(const client of clients)client.bridge.invalidate?.('test-finished')}};
}

for(const order of ['peer-first','driver-first'])test(`Player checkpoint window accepts only the driver response with ${order} delivery`,async t=>{
 const f=await fixture();t.after(()=>f.dispose());const before=f.store.raw.pages.length,result=await f.request(()=>f.player.bridge.getActivityCheckpoint('Actor.A'),order);
 assert.equal(result.ok,true,result.error?.message);assert.deepEqual(result.value.binding,f.binding);assert.equal(result.value.phase,'open');
 assert.equal(result.replies.length,1);assert.equal(result.replies[0].role,'driver');assert.equal(f.store.raw.pages.length,before);assert.equal(f.playerReads(),0);
});

for(const order of ['peer-first','driver-first'])test(`Player checkpoint declaration accepts only the driver ACK with ${order} delivery`,async t=>{
 const f=await fixture();t.after(()=>f.dispose());const before=f.store.raw.pages.length,result=await f.request(()=>f.player.bridge.record(f.input()),order);
 assert.equal(result.ok,true,result.error?.message);assert.equal(result.value.registrationId,'registration');assert.equal(result.replies.length,1);assert.equal(result.replies[0].role,'driver');
 const saved=await f.ledger.getSession('S');assert.equal(Object.keys(saved.activityCheckpoint.registrations).length,1);assert.equal(f.store.raw.pages.length,before+1);
 const state=await f.store.read();assert.deepEqual(state.activities,{});assert.deepEqual(state.clocks,{});assert.equal(f.playerReads(),0);
});

test('a second tab of the active GM routes its window request to the local session driver',async t=>{
 const f=await fixture();t.after(()=>f.dispose());const result=await f.request(()=>f.peer.bridge.getActivityCheckpoint('Actor.A'),'peer-first');
 assert.equal(result.ok,true,result.error?.message);assert.deepEqual(result.value.binding,f.binding);assert.equal(result.replies.length,1);assert.equal(result.replies[0].role,'driver');
});

test('the driver rejects a revoked OWNER without another tab supplying an ACK',async t=>{
 const f=await fixture();t.after(()=>f.dispose());f.actor.testUserPermission=user=>user?.id==='G';const before=f.store.raw.pages.length,result=await f.request(()=>f.player.bridge.record(f.input()),'peer-first');
 assert.equal(result.ok,false);assert.equal(result.error.message,'manual-actor-not-allowed');assert.equal(result.error.declarationRejected,true);assert.equal(result.replies.length,1);assert.equal(result.replies[0].role,'driver');assert.equal(f.store.raw.pages.length,before);assert.equal(f.playerReads(),0);
});

test('a caller cannot turn an unknown commit into a pre-write rejection through echoed proof',async t=>{
 const f=await fixture({editRequest:(packet,role)=>{if(role==='player')packet.proof={declarationRejected:true}}});t.after(()=>f.dispose());f.store.setAcknowledgement(()=>undefined);
 const before=f.store.raw.pages.length,result=await f.request(()=>f.player.bridge.record(f.input()),'peer-first');
 assert.equal(result.ok,false);assert.match(result.error.message,/acknowledgement-unknown/);assert.notEqual(result.error.declarationRejected,true);
 assert.equal(f.store.raw.pages.length,before+1);assert.equal(Object.keys((await f.ledger.getSession('S')).activityCheckpoint.registrations).length,1);assert.equal(f.playerReads(),0);
});

for(const order of ['peer-first','driver-first'])test(`a bound manual reservation reaches only the driver with ${order} delivery`,async t=>{
 const f=await fixture({manualWindow:true});t.after(()=>f.dispose());const intent={sourceType:'workbench',kind:'treatment',useId:'use',actorUUID:'Actor.A',patientUUID:'Actor.A'};
 const result=await f.request(()=>f.player.bridge.reserveSource(f.binding,intent),order);
 assert.equal(result.ok,true,result.error?.message);assert.deepEqual(result.value.checkpointBinding,f.binding);assert.equal(result.replies.length,1);assert.equal(result.replies[0].role,'driver');assert.equal(f.reservations(),1);assert.equal(f.playerReads(),0);
});

test('a normal unbound treatment has no reservation when no manual checkpoint is open',async t=>{
 const f=await fixture();t.after(()=>f.dispose());delete f.driver.scope.leaseNonce;
 const result=await f.request(()=>f.player.bridge.reserveSource(null,{sourceType:'workbench',kind:'treatment',useId:'use',actorUUID:'Actor.A',patientUUID:'Actor.A'}),'peer-first');
 assert.equal(result.ok,true,result.error?.message);assert.equal(result.value,null);assert.equal(f.reservations(),0);assert.equal(f.packets.filter(row=>row.packet.operationId==='manual-record-driver').length,0);assert.equal(f.playerReads(),0);
});

test('an absent session driver leaves one window request unknown without a retry',async t=>{
 const f=await fixture({timeoutMs:100});t.after(()=>f.dispose());delete f.driver.scope.leaseNonce;const before=f.store.raw.pages.length;
 await assert.rejects(f.player.bridge.getActivityCheckpoint('Actor.A'),/timeout-unknown-no-retry/);
 assert.equal(f.packets.filter(row=>row.packet.kind==='continuation').length,1);assert.deepEqual(f.replies,[]);assert.equal(f.store.raw.pages.length,before);assert.equal(f.playerReads(),0);
});

test('a lost declaration ACK permits saved readback after scope loss without resubmitting',async t=>{
 const f=await fixture({timeoutMs:100});t.after(()=>f.dispose());const before=f.store.raw.pages.length;
 await assert.rejects(f.player.bridge.record(f.input()),/timeout-unknown-no-retry/);
 assert.equal(f.replies.length,1);assert.equal(f.store.raw.pages.length,before+1);delete f.driver.scope.leaseNonce;
 f.deliver(f.replies[0]);f.deliver(f.replies[0]);
 const result=await f.request(()=>f.player.bridge.lookupCheckpointActivity(f.binding,'registration','Actor.A'),'peer-first');
 assert.equal(result.ok,true,result.error?.message);assert.equal(result.value.status,'registered');assert.equal(f.store.raw.pages.length,before+1);
 assert.equal(f.packets.filter(row=>row.packet.kind==='continuation'&&row.packet.status==='declaration').length,1);assert.equal(f.playerReads(),0);
});

test('replayed driver requests and duplicate ACKs cannot append another declaration',async t=>{
 const f=await fixture();t.after(()=>f.dispose());const before=f.store.raw.pages.length,result=await f.request(()=>f.player.bridge.record(f.input()),'peer-first');assert.equal(result.ok,true,result.error?.message);
 const request=f.packets.find(row=>row.packet.kind==='continuation'),reply=f.replies[0];f.player.game.socket.emit(request.name,request.packet,request.options);await turn();f.deliver(reply);f.deliver(reply);
 assert.equal(f.replies.length,1);assert.equal(f.store.raw.pages.length,before+1);assert.equal(Object.keys((await f.ledger.getSession('S')).activityCheckpoint.registrations).length,1);
});

test('a retained old lease cannot answer after the persisted driver changes',async t=>{
 const f=await fixture();t.after(()=>f.dispose());await f.peer.writer.takeoverSession('S');const successor=await f.peer.writer.resumeSession('S',{cursorAt:0});f.peer.scope.leaseNonce=successor.driver.leaseNonce;
 Object.assign(f.binding,{id:'next-window',observationNonce:'next-observation'});await f.peer.writer.openActivityCheckpoint(f.binding,{leaseNonce:successor.driver.leaseNonce,guard:()=>true});
 const result=await f.request(()=>f.player.bridge.getActivityCheckpoint('Actor.A'),'driver-first');assert.equal(result.ok,true,result.error?.message);assert.deepEqual(result.value.binding,f.binding);assert.equal(result.replies.length,1);assert.equal(result.replies[0].role,'peer');assert.equal(f.playerReads(),0);
});

test('a driver superseded during an actor lookup sends no terminal reply',async t=>{
 let began,release;const ready=new Promise(resolve=>began=resolve),gate=new Promise(resolve=>release=resolve);
 const f=await fixture({timeoutMs:100,beforeActor:async role=>{if(role==='driver'){began();await gate}}});t.after(()=>f.dispose());
 const rejected=assert.rejects(f.player.bridge.getActivityCheckpoint('Actor.A'),/timeout-unknown-no-retry/);await ready;await f.peer.writer.takeoverSession('S');release();await rejected;
 assert.deepEqual(f.replies,[]);assert.equal(f.packets.filter(row=>row.packet.kind==='continuation').length,1);assert.equal(f.playerReads(),0);
});

test('invalidation closes a pending declaration and ignores its old ACK without a resend',async t=>{
 const f=await fixture();t.after(()=>f.dispose());const pending=f.player.bridge.record(f.input()),rejected=assert.rejects(pending,/disposed/);while(!f.replies.length)await new Promise(resolve=>setImmediate(resolve));
 f.player.bridge.invalidate('root-changed');await rejected;f.deliver(f.replies[0]);await turn();assert.equal(f.packets.filter(row=>row.packet.kind==='continuation').length,1);assert.equal(Object.keys((await f.ledger.getSession('S')).activityCheckpoint.registrations).length,1);
});

test('checkpoint requests and replies carry no private lease, ledger or activity payload',async t=>{
 const f=await fixture();t.after(()=>f.dispose());const session=await f.ledger.getSession('S'),result=await f.request(()=>f.player.bridge.getActivityCheckpoint('Actor.A'),'peer-first');assert.equal(result.ok,true,result.error?.message);
 const wire=JSON.stringify(f.packets.map(row=>row.packet));assert.equal(wire.includes(session.driver.leaseNonce),false);assert.doesNotMatch(wire,/"(?:leaseNonce|driverClientNonce|ledger|activities|clocks|history|providerId)"/);assert.equal(f.playerReads(),0);
});

test('checkpoint native routing cannot fall back to socketlib when the native transport is unavailable',async()=>{
 const users=new Map([['G',{id:'G',isGM:true,active:true}],['P',{id:'P',active:true}]]);users.activeGM=users.get('G');let fallback=0;
 const bridge=createManualRecordBridge({game:{user:users.get('P'),users},getSession:()=>assert.fail('player-private-read')});bridge.register({register(){},executeAsGM(){fallback++;assert.fail('unsafe-fallback')}});
 await assert.rejects(bridge.getActivityCheckpoint('Actor.A'),/manual-record-native-unavailable/);assert.equal(fallback,0);
});

for(const status of ['declaration','reservation'])test(`a ${status} request cannot substitute another inner actor`,async t=>{
 const f=await fixture({manualWindow:status==='reservation',editRequest:packet=>{if(packet.status===status)(status==='declaration'?packet.command.event:packet.command.intent).actorUUID='Actor.OTHER'}});t.after(()=>f.dispose());const before=f.store.raw.pages.length;
 const call=status==='declaration'?()=>f.player.bridge.record(f.input()):()=>f.player.bridge.reserveSource(f.binding,{sourceType:'workbench',kind:'treatment',useId:'use',actorUUID:'Actor.A',patientUUID:'Actor.A'}),result=await f.request(call,'peer-first');
 assert.equal(result.ok,false);assert.equal(result.error.message,'manual-driver-request-mismatch');assert.equal(f.store.raw.pages.length,before);assert.equal(f.reservations(),0);assert.equal(f.playerReads(),0);
});
