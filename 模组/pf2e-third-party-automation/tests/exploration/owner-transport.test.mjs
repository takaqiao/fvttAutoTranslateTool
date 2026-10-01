import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import vm from 'node:vm';
import {createOwnerTransport,OWNER_TRANSPORT_PROTOCOL as protocol,OWNER_TRANSPORT_CHANNEL as channel} from '../../scripts/exploration/owner-transport.mjs';

const turn=async()=>{await Promise.resolve();await Promise.resolve()};
const claim=(requestId='Q',patch={})=>({protocol,kind:'claim',requestId,receiverUserId:'G',offerId:'OFFER',ownerClientNonce:'OWNER-1',attemptNonce:'ATTEMPT-1',activityId:'A',...patch});
const grant=(requestId='Q',patch={})=>({...claim(requestId),kind:'grant',receiverUserId:'O',permitNonce:'PERMIT',...patch});
function fixture(userId='O',options={}){
 const listeners=new Set(),sent=[],packets=[],errors=[],deliveries=[],users=new Map(['G','O','OTHER'].map(id=>[id,{id,isGM:id==='G'}]));
 const socket={on:(name,listener)=>{assert.equal(name,channel);listeners.add(listener)},off:(name,listener)=>{assert.equal(name,channel);listeners.delete(listener)},emit:(...args)=>sent.push(args)};
 const game={user:users.get(userId),users,socket};
 const transport=createOwnerTransport({game,onPacket:(...args)=>packets.push(args),onError:error=>errors.push(error),timeoutMs:100,...options});
 const deliver=(packet,senderId='G')=>{deliveries.push([packet,senderId]);for(const listener of [...listeners])listener(packet,senderId)};
 return {transport,game,socket,users,listeners,sent,packets,errors,deliveries,deliver};
}
const waiting=(f,packet=claim(),options={})=>f.transport.request(packet,{expectedSenderId:'G',matches:reply=>reply.kind==='grant'||reply.kind==='denied',...options});

test('send uses exactly one server recipient, detaches JSON and never converts relay ack into a reply',()=>{
 const f=fixture(),packet=claim();packet.command={patientUUIDs:['Actor.P']};f.transport.send(packet);assert.equal(f.sent.length,1);
 const [event,saved,options,ack]=f.sent[0];assert.equal(event,channel);assert.deepEqual(options,{recipients:['G']});assert.notEqual(saved,packet);packet.command.patientUUIDs.push('Actor.Q');assert.deepEqual(saved.command.patientUUIDs,['Actor.P']);
 assert.equal(typeof ack,'function');ack({ok:true});assert.deepEqual(f.packets,[]);f.transport.dispose();
});
for(const patch of [{receiverUserId:''},{receiverUserId:'MISSING'},{receiverUserId:['G']},{receiverUserId:' G'},{protocol:'other'},{kind:'arbitrary'},{requestId:''},{requestId:'q'.repeat(129)},{id:'Q'},{type:'claim'},{handlerName:'private'},{args:[]},{recipient:['G']},{result:{}},{activity:{id:'A'}},{ledger:{}},{command:{sessions:{S:{}}}},{command:{id:'A',sessionId:'S',providerId:'treat-wounds',actorUUID:'Actor.H',proof:{}}}])test(`invalid send envelope cannot fall back to broadcast: ${Object.keys(patch).join(',')} ${JSON.stringify(patch).slice(0,70)}`,()=>{
 const f=fixture();assert.throws(()=>f.transport.send(claim('Q',patch)));assert.deepEqual(f.sent,[]);f.transport.dispose();
});
test('non-JSON, cyclic, deep, oversized and getter payloads fail before emit',()=>{
 const f=fixture(),cycle={};cycle.self=cycle;const getter={get value(){throw Error('getter should not run')}};let deep={};for(let i=0;i<20;i++)deep={nested:deep};
 const disguisedSparse=Array(1);disguisedSparse['4294967295']=1;
 for(const command of [{n:NaN},{n:Infinity},{n:1n},{f:()=>{}},{u:undefined},[undefined],Array(2),disguisedSparse,new Date(),cycle,getter,deep,{text:'x'.repeat(5000)},{values:Array.from({length:257},()=>1)}])assert.throws(()=>f.transport.send(claim('Q',{command})));
 assert.deepEqual(f.sent,[]);f.transport.dispose();
});
test('onPacket receives detached valid requests and only the native authenticated sender',()=>{
 const f=fixture('G'),packet=claim();f.deliver(packet,'O');packet.attemptNonce='changed';assert.equal(f.packets.length,1);assert.equal(f.packets[0][1],'O');assert.equal(f.packets[0][0].attemptNonce,'ATTEMPT-1');
 f.deliver(claim('Q',{receiverUserId:'O'}),'O');f.deliver(claim(),'UNKNOWN');f.deliver({...claim(),protocol:'socketlib',type:1,id:'OTHER'},'O');f.deliver(null,'O');assert.equal(f.packets.length,1);assert.deepEqual(f.sent,[]);f.transport.dispose();
});
test('a non-driver handler returning undefined or a value emits no automatic response',()=>{
 let calls=0;const f=fixture('G',{onPacket:()=>{calls++;return calls===1?undefined:{denied:true}}});f.deliver(claim('one'),'O');f.deliver(claim('two'),'O');assert.equal(calls,2);assert.deepEqual(f.sent,[]);f.transport.dispose();
});
test('request is pending before emit and relay ack has no authority',async()=>{
 const f=fixture();let settled=false;const pending=waiting(f).then(value=>{settled=true;return value});assert.equal(f.sent.length,1);f.sent[0][3]({ok:true,grant:true});await turn();assert.equal(settled,false);
 f.deliver(grant());assert.equal((await pending).permitNonce,'PERMIT');assert.deepEqual(f.packets,[]);f.transport.dispose();
});
test('a synchronous business response during emit finds the already-installed private pending',async()=>{
 const f=fixture();f.socket.emit=(...args)=>{f.sent.push(args);f.deliver(grant())};const result=await waiting(f);assert.equal(result.permitNonce,'PERMIT');f.transport.dispose();
});
test('outbound socket data cannot mutate the private expected request identity',async()=>{
 const f=fixture();f.socket.emit=(...args)=>{f.sent.push(args);args[1].attemptNonce='WIRE-CHANGED'};
 let settled=false;const pending=waiting(f).then(value=>{settled=true;return value});f.deliver(grant('Q',{attemptNonce:'WIRE-CHANGED'}));await turn();assert.equal(settled,false);
 f.deliver(grant());assert.equal((await pending).attemptNonce,'ATTEMPT-1');f.transport.dispose();
});
for(const [label,reply,sender] of [['sender',grant(),'OTHER'],['receiver',grant('Q',{receiverUserId:'OTHER'}),'G'],['request',grant('OTHER'),'G'],['nonce',grant('Q',{attemptNonce:'WRONG'}),'G'],['runtime',grant('Q',{ownerClientNonce:'WRONG'}),'G'],['offer',grant('Q',{offerId:'WRONG'}),'G'],['activity',grant('Q',{activityId:'B'}),'G'],['predicate',grant('Q',{kind:'completion-ack'}),'G']])test(`wrong ${label} cannot consume a request or reach onPacket`,async()=>{
 const f=fixture();let settled=false;const pending=waiting(f).then(value=>{settled=true;return value});f.deliver(reply,sender);await turn();assert.equal(settled,false);assert.deepEqual(f.packets,[]);f.deliver(grant());assert.equal((await pending).permitNonce,'PERMIT');f.transport.dispose();
});
test('denied is a matched business response without creating a grant',async()=>{
 const f=fixture();const pending=waiting(f);f.deliver({...claim(),kind:'denied',receiverUserId:'O',errorCode:'claim-conflict'});assert.equal((await pending).errorCode,'claim-conflict');assert.deepEqual(f.sent.length,1);f.transport.dispose();
});
test('timeout expires synchronously; late and duplicate replies never reach a callback',async t=>{
 t.mock.timers.enable({apis:['setTimeout']});const f=fixture(),pending=waiting(f);const rejected=assert.rejects(pending,/timeout.*unknown|unknown.*timeout/);t.mock.timers.tick(101);f.deliver(grant());await rejected;f.deliver(grant());assert.deepEqual(f.packets,[]);assert.equal(f.sent.length,1);f.transport.dispose();
});
test('an elapsed deadline rejects a response even before the delayed timer callback can run',async()=>{
 const f=fixture('O',{timeoutMs:5}),pending=waiting(f),rejected=assert.rejects(pending,/timeout.*unknown|unknown.*timeout/);
 const until=performance.now()+10;while(performance.now()<until){}
 f.deliver(grant());await rejected;assert.deepEqual(f.packets,[]);f.transport.dispose();
});
test('consumed request ignores a duplicate or unexpected response',async()=>{
 const f=fixture(),pending=waiting(f);f.deliver(grant());await pending;f.deliver(grant());f.deliver(grant('NOT-PENDING'));assert.deepEqual(f.packets,[]);assert.equal(f.sent.length,1);f.transport.dispose();
});
test('dispose closes requests and removes only its own listener',async()=>{
 const f=fixture(),other=()=>{};f.listeners.add(other);const pending=waiting(f),rejected=assert.rejects(pending,/disposed/);f.transport.dispose();f.transport.dispose();await rejected;assert.deepEqual([...f.listeners],[other]);f.deliver(grant());assert.deepEqual(f.packets,[]);assert.throws(()=>f.transport.send(claim()),/disposed/);await assert.rejects(waiting(f),/disposed/);
});
test('duplicate pending IDs and capacity overflow reject without emitting another request',async()=>{
 const f=fixture(),pending=[];for(let i=0;i<128;i++)pending.push(waiting(f,claim(`Q${i}`)).catch(error=>error));assert.equal(f.sent.length,128);await assert.rejects(waiting(f,claim('Q0')),/pending|duplicate/);await assert.rejects(waiting(f,claim('OVERFLOW')),/limit|capacity/);assert.equal(f.sent.length,128);f.transport.dispose();await Promise.all(pending);
});
test('invalid request configuration and a thrown socket emit do not leave live pending',async()=>{
 const f=fixture();await assert.rejects(waiting(f,claim(),{expectedSenderId:'OTHER'}));await assert.rejects(waiting(f,claim(),{matches:null}));await assert.rejects(waiting(f,claim(),{matches:async()=>true}));assert.deepEqual(f.sent,[]);
 f.socket.emit=()=>{throw Error('socket-disconnected')};await assert.rejects(waiting(f),/socket-disconnected/);f.transport.dispose();
});
test('handler exceptions are reported locally and never generate a network reply',async()=>{
 const f=fixture('G',{onPacket:async()=>{throw Error('business-failed')}});f.deliver(claim(),'O');await turn();assert.equal(f.errors[0]?.message,'business-failed');assert.deepEqual(f.sent,[]);f.transport.dispose();
});
test('two owner tabs share recipient delivery but only one private nonce can consume the grant',async()=>{
 const first=fixture(),second=fixture(),a=waiting(first),b=waiting(second,claim('Q2',{ownerClientNonce:'OWNER-2',attemptNonce:'ATTEMPT-2'})).catch(error=>error);first.deliver(grant());second.deliver(grant());assert.equal((await a).permitNonce,'PERMIT');assert.deepEqual(second.packets,[]);second.transport.dispose();assert.match((await b).message,/disposed/);first.transport.dispose();
});
test('installed socketlib keeps an existing pending request intact beside the production helper',async()=>{
 const f=fixture(),diagnostics=[],source=await fs.readFile(process.env.FVTT_SOCKETLIB_SOURCE??'C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/socketlib/src/socketlib.js','utf8');
 const context=vm.createContext({game:{userId:'O',users:{get:id=>({id,isGM:id==='G'})},socket:f.socket},Hooks:{once(){},on(){}},window:{},errors:{},console:{error:(...args)=>diagnostics.push(args),warn:(...args)=>diagnostics.push(args),info:(...args)=>diagnostics.push(args)}});
 vm.runInContext(source.replace(/^import \* as errors from "\.\/errors.js";\s*/,'')+';globalThis.SocketClass=SocketlibSocket',context);const socketlib=new context.SocketClass('pf2e-third-party-automation','module');let resolved=false;
 socketlib.pendingRequests.set('Q',{recipient:vm.runInContext('["G"]',context),handlerName:'existing',resolve:()=>{resolved=true},reject:()=>{throw Error('existing request corrupted')}});
 const pending=waiting(f);f.deliver(grant());await pending;assert.equal(socketlib.pendingRequests.size,1);assert.equal(resolved,false);assert.deepEqual(diagnostics,[]);f.deliver({id:'Q',type:3,result:'existing'},'G');assert.equal(resolved,true);assert.deepEqual(f.packets,[]);f.transport.dispose();assert.equal(f.listeners.size,1);
});
test('installed native server routes production helpers across two GM and owner tabs without unrelated payloads',async()=>{
 let driver;const firstGM=fixture('G',{onPacket:(packet,senderId)=>{assert.equal(senderId,'O');driver.send({...packet,kind:packet.ownerClientNonce==='OWNER-1'?'grant':'denied',receiverUserId:'O',...packet.ownerClientNonce==='OWNER-1'?{permitNonce:'PERMIT'}:{errorCode:'claim-conflict'}})}});driver=firstGM.transport;
 const secondGM=fixture('G',{onPacket:()=>undefined}),firstOwner=fixture(),secondOwner=fixture(),unrelated=fixture('OTHER'),clients=[firstGM,secondGM,firstOwner,secondOwner,unrelated];
 const users=['G','O','OTHER'].map(id=>({id,sockets:clients.filter(client=>client.game.user.id===id).map(client=>({emit:(event,packet,senderId)=>{assert.equal(event,channel);client.deliver(packet,senderId)}}))}));
 const source=await fs.readFile(process.env.FVTT_FOUNDRY_SOCKET_SOURCE??'C:/Program Files/Foundry Virtual Tabletop/resources/app/dist/server/sockets.mjs','utf8'),start=source.indexOf('export function handleCustomSocket('),end=source.indexOf('export function handleMigrateDocumentData',start);assert(start>=0&&end>start);
 const context=vm.createContext({game:{users}});vm.runInContext(source.slice(start,end).replace(/^export /,'')+';globalThis.relay=handleCustomSocket',context);
 for(const client of clients)client.socket.emit=(event,packet,options,ack)=>{client.sent.push([event,packet,options,ack]);context.relay.call({user:client.game.user,broadcast:{emit:()=>{throw Error('private broadcast forbidden')}}},event,packet,options,ack)};
 const first=await waiting(firstOwner),second=await waiting(secondOwner,claim('Q2',{ownerClientNonce:'OWNER-2',attemptNonce:'ATTEMPT-2'}));
 assert.equal(first.kind,'grant');assert.equal(second.kind,'denied');assert.equal(firstGM.sent.length,2);assert.equal(secondGM.sent.length,0);assert.equal(unrelated.deliveries.length,0);assert.equal(secondGM.deliveries.length,2);
 assert.deepEqual(firstOwner.packets,[]);assert.deepEqual(secondOwner.packets,[]);for(const client of clients)client.transport.dispose();
});

for(const change of ['sender','requestId','attemptNonce','permitNonce','kind'])test(`continuation requires the private pending identity: ${change}`,async()=>{
 const f=fixture(),packet=claim('CONTINUE',{kind:'continuation',permitNonce:'PERMIT',proof:{permit:{permitNonce:'PERMIT'}}});let settled=false;
 const pending=f.transport.request(packet,{expectedSenderId:'G',matches:reply=>reply.kind==='continuation-ack'&&reply.proof?.permit?.permitNonce==='PERMIT'}).then(value=>{settled=true;return value});
 const reply={...packet,kind:'continuation-ack',receiverUserId:'O',status:'accepted'},wrong={...reply};if(change!=='sender')wrong[change]=change==='kind'?'grant':'WRONG';
 f.deliver(wrong,change==='sender'?'OTHER':'G');await turn();assert.equal(settled,false);f.sent[0][3]({accepted:true});await turn();assert.equal(settled,false);
 f.deliver(reply);assert.equal((await pending).kind,'continuation-ack');f.deliver(reply);assert.deepEqual(f.packets,[]);f.transport.dispose();
});
test('late continuation ACK and targeted cancel have no automatic response',async t=>{
 t.mock.timers.enable({apis:['setTimeout']});const f=fixture(),packet=claim('CONTINUE',{kind:'continuation',permitNonce:'PERMIT'}),pending=f.transport.request(packet,{expectedSenderId:'G',matches:()=>true}),rejected=assert.rejects(pending,/timeout.*unknown/);
 t.mock.timers.tick(101);await rejected;f.deliver({...packet,kind:'continuation-ack',receiverUserId:'O'});assert.deepEqual(f.packets,[]);
 f.deliver({...packet,kind:'cancel',receiverUserId:'O'},'G');assert.equal(f.packets.length,1);assert.equal(f.packets[0][1],'G');assert.equal(f.sent.length,1);f.transport.dispose();
});
test('cancel waits for the exact retired ACK and never consumes another business response',async()=>{
 const f=fixture(),packet=claim('STOP',{kind:'cancel',permitNonce:'PERMIT',proof:{permit:{permitNonce:'PERMIT'}}});let settled=false;
 const pending=f.transport.request(packet,{expectedSenderId:'G',matches:reply=>reply.kind==='cancel-ack'&&reply.status==='retired'&&reply.proof?.permit?.permitNonce==='PERMIT'}).then(value=>{settled=true;return value});
 const reply={...packet,kind:'cancel-ack',receiverUserId:'O',status:'retired'};
 for(const [wrong,sender]of [[{...reply,kind:'continuation-ack'},'G'],[{...reply,kind:'denied'},'G'],[reply,'OTHER'],[{...reply,proof:{permit:{permitNonce:'WRONG'}}},'G']])f.deliver(wrong,sender);
 f.sent[0][3]({retired:true});await turn();assert.equal(settled,false);f.deliver(reply);assert.equal((await pending).status,'retired');f.deliver(reply);assert.deepEqual(f.packets,[]);f.transport.dispose();
});