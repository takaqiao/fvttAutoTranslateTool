import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createManualEvents} from '../../scripts/exploration/manual-events.mjs';
import {createPatreonManualImmunity} from '../../scripts/exploration/patreon-manual-immunity.mjs';
import {createPatreonManualImmunityOwner} from '../../scripts/exploration/patreon-manual-immunity-owner.mjs';
import {createHpPools} from '../../scripts/exploration/hp-pool.mjs';
import {OWNER_TRANSPORT_CHANNEL} from '../../scripts/exploration/owner-transport.mjs';
import {manualEvidenceFixture,flush,M} from './manual-evidence-fixture.mjs';

function senderHub(){
 const clients=new Map(),packets=[],held=[];
 const hub={packets,held,holdTerminal:false,dropNextCompletionAck:false};
 hub.deliver=(packet,senderId)=>{
  const client=clients.get(packet.receiverUserId);
  for(const listener of client?.listeners??[])listener(structuredClone(packet),senderId);
 };
 hub.socket=userId=>{
  const listeners=new Set();clients.set(userId,{listeners});
  return {on:(channel,fn)=>{assert.equal(channel,OWNER_TRANSPORT_CHANNEL);listeners.add(fn)},
   off:(channel,fn)=>{assert.equal(channel,OWNER_TRANSPORT_CHANNEL);listeners.delete(fn)},
   emit(channel,packet,options,ack){
    assert.equal(channel,OWNER_TRANSPORT_CHANNEL);assert.deepEqual(options,{recipients:[packet.receiverUserId]});
    const saved=structuredClone(packet);packets.push({senderId:userId,packet:saved});ack?.({ok:true});
    if(hub.holdTerminal&&saved.kind==='completion'&&saved.status==='terminal'){held.push({senderId:userId,packet:saved});return}
    if(hub.dropNextCompletionAck&&saved.kind==='completion-ack'){hub.dropNextCompletionAck=false;return}
    queueMicrotask(()=>hub.deliver(saved,userId));
   }};
 };
 return hub;
}

function provider(){
 const observers=new Set();
 const descriptor={version:1,providerId:'patreon-v3',providerVersion:'3.2.29',
  baseSourceSHA256:'89ded325b92fa6b03dcf9257337ae2d628b3e99c987fe22cef9fff4e1837f4e9',
  pf2eSourceSHA256:'d63da8312831b84905e6866b1dd3f9d93e95c1012955b0177ad2ce8ccf246157'};
 return {descriptor,subscribe:fn=>{observers.add(fn);return ()=>observers.delete(fn)},
  emit(event){for(const observer of observers)observer(event)}};
}

function ownerFixture(){
 const f=manualEvidenceFixture();f.messages.clear();f.game.time.worldTime=100;
 f.game.system={version:'8.5.1'};f.game.release={generation:14};
 f.healer.type='character';f.patient.type='character';
 f.patient.testUserPermission=user=>user?.isGM===true||user?.id==='HUSER';
 f.binding={invocationId:'INV',messageId:'C',useId:'U',tag:'exploration-manual:U',actorUUID:'Actor.H',patientUUID:'Actor.P',sourceUserId:'HUSER',startedAt:100,recordingSessionId:'S'};
 const native={tag:'exploration-manual:U',useId:'U',patientUUID:'Actor.P',continualRecovery:false,startedAt:100,riskySurgery:false,recordingSessionId:'S',patreonImmunity:{...f.binding}};
 f.check={id:'C',isCheckRoll:true,isReroll:false,author:f.users.get('HUSER'),speaker:{actor:'H'},rolls:[{_evaluated:true}],
  flags:{[M]:{explorationManualNative:{...native}},pf2e:{modifiers:[],context:{type:'skill-check',origin:{actor:'Actor.H'},target:{actor:'Actor.P'},outcome:'success',options:['action:treat-wounds','exploration-manual:U']}}}};
 f.child={id:'D',isCheckRoll:false,author:f.users.get('HUSER'),speaker:{actor:'H'},rolls:[{_evaluated:true,formula:'{2d8[healing]}'}],
  flags:{[M]:{explorationManualNative:{...native}},pf2e:{origin:{messageId:'C'},context:{origin:{actor:'Actor.H'},target:{actor:'Actor.P'},options:['exploration-manual:U']}}},
  async update(changes){this.flags.pf2e.context.options=changes['flags.pf2e.context.options'];return this}};
 f.receipt={id:'R',author:f.users.get('HUSER'),speaker:{actor:'P'},flags:{pf2e:{context:{type:'damage-taken',options:[`${M}:source:D:0`]},appliedDamage:{uuid:'Actor.P',isHealing:true,isReverted:false}}}};
 f.item={id:'I',uuid:'Actor.P.Item.I',type:'effect',actor:f.patient,parent:f.patient,sourceId:'Compendium.pf2e.feat-effects.Item.Lb4q2bBAgxamtix5',
  system:{start:{value:100},duration:{value:60,unit:'minutes',expiry:'turn-start',sustained:false},context:{origin:{actor:'Actor.H'}}},
  flags:{[M]:{explorationManualPatreonImmunity:{...f.binding,creatorId:'HUSER'}}}};
 f.options.fromUuid=async uuid=>f.actors.get(uuid)??f.patient.items.get(uuid.split('.Item.')[1]);
 f.hub=senderHub();f.game.socket=f.hub.socket('G');
 const gmProvider=provider(),ownerProvider=provider();f.ownerProvider=ownerProvider;
 f.game.modules=new Map([['patreon-v3',{active:true,version:'3.2.29',api:{explorationManualImmunity:gmProvider}}]]);
 f.options.hpPools=createHpPools({game:f.game});
 f.ownerGame={...f.game,user:f.users.get('HUSER'),socket:f.hub.socket('HUSER'),
  modules:new Map([['patreon-v3',{active:true,version:'3.2.29',api:{explorationManualImmunity:ownerProvider}}]])};
 const gmImmunity=createPatreonManualImmunity({game:f.game,fromUuid:f.options.fromUuid});
 const ownerImmunity=createPatreonManualImmunity({game:f.ownerGame,fromUuid:f.options.fromUuid});
 f.recorder=createManualEvents({...f.options,patreonImmunity:gmImmunity,observePatreonTerminal:false});
 f.commits=[];f.bridgeErrors=[];
 const callbacks={fromUuid:f.options.fromUuid,getSession:()=>f.ledger.getSession('S'),getActivity:id=>f.ledger.getActivity(id),timeoutMs:30,onError:error=>f.bridgeErrors.push(error)};
 f.gmBridge=createPatreonManualImmunityOwner({...callbacks,game:f.game,patreonImmunity:gmImmunity,
  record:async proof=>{const activity=await f.recorder.observeNativeImmunity(proof);if(activity)f.commits.push(structuredClone(proof));return activity}});
 const privateLedger=async()=>{throw Error('private-ledger-read-forbidden')};
 f.ownerBridge=createPatreonManualImmunityOwner({...callbacks,game:f.ownerGame,patreonImmunity:ownerImmunity,
  getSession:privateLedger,getActivity:privateLedger,record:async()=>{throw Error('only active GM may record native immunity')}});
 f.terminal=()=>({descriptor:ownerProvider.descriptor,binding:{...f.binding},creatorId:'HUSER',itemUUID:'Actor.P.Item.I',start:100,
  duration:{value:60,unit:'minutes',expiry:'turn-start',sustained:false},expiresAt:3700});
 f.emit=async terminalPromise=>{ownerProvider.emit({descriptor:ownerProvider.descriptor,binding:f.binding,terminalPromise});await flush()};
 f.prepare=async()=>{
  f.recorder.start();f.gmBridge.start();f.ownerBridge.start();
  f.source=await f.ownerBridge.bindSource('Actor.H');
  await f.fire(f.check);await f.fire(f.child);await f.fire(f.receipt);
  f.patient.items.set('I',f.item);f.handlers.get('createItem')?.(f.item,{},'HUSER');await flush();
  assert.equal((await f.ledger.getActivity('manual:C')).state,'awaiting-evidence');
 };
 f.stop=()=>{f.recorder.stop();f.gmBridge.stop();f.ownerBridge.stop()};
 f.activity=()=>f.ledger.getActivity('manual:C');
 return f;
}

test('the patient owner binds its healer to the current recording through an authenticated GM query',async()=>{
 const f=ownerFixture();await f.prepare();
 try{
  assert.deepEqual(f.source,{sessionId:'S',startedAt:100,actorUUID:'Actor.H',userId:'HUSER'});
 }finally{f.stop()}
});

test('an actual creator local terminal reaches the GM only after its original Promise fulfills',async()=>{
 const f=ownerFixture();await f.prepare();let resolve;const pending=new Promise(done=>{resolve=done});
 try{
  const before=f.hub.packets.length;await f.emit(pending);
  assert.equal(f.hub.packets.length,before);assert.equal((await f.activity()).state,'awaiting-evidence');
  resolve(f.terminal());await flush();const activity=await f.activity();
  assert.equal(activity.state,'confirmed');assert.deepEqual(activity.proof.immunityIds,['Actor.P.Item.I']);
  assert.deepEqual(activity.proof.receiptIds,['R']);assert.deepEqual(activity.proof.nativeImmunity,f.terminal());
  assert.equal(f.commits.length,1);assert.equal(f.patient.items.size,1);assert.equal(f.game.time.worldTime,100);
 }finally{f.stop()}
});

test('an authenticated peer cannot borrow the actual creator terminal',async()=>{
 const f=ownerFixture();await f.prepare();f.hub.holdTerminal=true;
 try{
  await f.emit(Promise.resolve(f.terminal()));const held=f.hub.held[0];
  assert.ok(held,'the original owner terminal must reach the transport before testing a peer sender');
  f.hub.deliver(held.packet,'PUSER');await flush();
  assert.equal((await f.activity()).state,'awaiting-evidence');assert.equal(f.commits.length,0);
 }finally{f.stop()}
});

test('a local noncreator cannot relay a different creator terminal',async()=>{
 const f=ownerFixture();await f.prepare();const proof=f.terminal();proof.creatorId='PUSER';
 f.item.flags[M].explorationManualPatreonImmunity.creatorId='PUSER';
 try{
  await f.emit(Promise.resolve(proof));assert.equal((await f.activity()).state,'awaiting-evidence');
  assert.equal(f.commits.length,0);
 }finally{f.stop()}
});

test('a creator without current patient OWNER permission cannot relay its native terminal',async()=>{
 const f=ownerFixture();await f.prepare();f.patient.testUserPermission=user=>user?.isGM===true;
 try{
  await f.emit(Promise.resolve(f.terminal()));assert.equal((await f.activity()).state,'awaiting-evidence');
  assert.equal(f.commits.length,0);
 }finally{f.stop()}
});

test('a revoked source owner cannot relay a terminal while its original Promise was pending',async()=>{
 const f=ownerFixture();await f.prepare();let resolve;const pending=new Promise(done=>{resolve=done});await f.emit(pending);
 f.healer.testUserPermission=user=>user?.isGM===true;
 try{
  resolve(f.terminal());await flush();assert.equal((await f.activity()).state,'awaiting-evidence');assert.equal(f.commits.length,0);
 }finally{f.stop()}
});

test('an owner terminal without the actual saved immunity Item cannot complete recording',async()=>{
 const f=ownerFixture();await f.prepare();f.patient.items.clear();
 try{
  await f.emit(Promise.resolve(f.terminal()));assert.equal((await f.activity()).state,'awaiting-evidence');assert.equal(f.commits.length,0);
 }finally{f.stop()}
});

test('owned check and Item markers without a provider terminal produce no completion relay',async()=>{
 const f=ownerFixture();await f.prepare();
 try{
  await flush();assert.equal((await f.activity()).state,'awaiting-evidence');
  assert.equal(f.hub.packets.filter(({packet})=>packet.kind==='completion').length,0);assert.equal(f.commits.length,0);
 }finally{f.stop()}
});

test('a repeated terminal and authenticated duplicate packet read one GM seal without recording twice',async()=>{
 const f=ownerFixture();await f.prepare();
 try{
  await f.emit(Promise.resolve(f.terminal()));assert.equal((await f.activity()).state,'confirmed');
  const before=await f.activity(),packet=f.hub.packets.find(entry=>entry.senderId==='HUSER'&&entry.packet.kind==='completion'&&entry.packet.status==='terminal')?.packet;
  assert.ok(packet);await f.emit(Promise.resolve(f.terminal()));f.hub.deliver(packet,'HUSER');await flush();
  assert.deepEqual(await f.activity(),before);assert.equal(f.commits.length,1);assert.equal(f.patient.items.size,1);
 }finally{f.stop()}
});

test('an owner terminal cannot cross the bound recording time after external time advancement',async()=>{
 const f=ownerFixture();await f.prepare();f.game.time.worldTime=101;
 try{
  await f.emit(Promise.resolve(f.terminal()));assert.equal((await f.activity()).state,'awaiting-evidence');assert.equal(f.commits.length,0);
 }finally{f.stop()}
});

test('an owner terminal cannot complete after its bound recording session stops',async()=>{
 const f=ownerFixture();await f.prepare();await f.ledger.updateSession('S',{status:'stopped'});
 try{
  await f.emit(Promise.resolve(f.terminal()));assert.equal((await f.activity()).state,'awaiting-evidence');assert.equal(f.commits.length,0);
 }finally{f.stop()}
});

test('a lost terminal ACK uses only a same-proof lookup of the existing GM seal',async t=>{
 t.mock.timers.enable({apis:['setTimeout']});const f=ownerFixture();await f.prepare();f.hub.dropNextCompletionAck=true;
 try{
  await f.emit(Promise.resolve(f.terminal()));const saved=await f.activity();assert.equal(saved.state,'confirmed');
  t.mock.timers.tick(31);await flush();
  const completions=f.hub.packets.filter(entry=>entry.senderId==='HUSER'&&entry.packet.kind==='completion').map(entry=>entry.packet);
  assert.equal(completions.filter(packet=>packet.status==='terminal').length,1);
  assert.ok(completions.some(packet=>packet.status==='lookup'&&packet.activityId==='manual:C'&&packet.sessionId==='S'));
  assert.deepEqual(await f.activity(),saved);assert.equal(f.commits.length,1);assert.equal(f.patient.items.size,1);
 }finally{f.stop()}
});

test('an owner with no private ledger read access completes from its authenticated source binding',async()=>{
 const f=ownerFixture();await f.prepare();
 try{
  assert.deepEqual(f.source,{sessionId:'S',startedAt:100,actorUUID:'Actor.H',userId:'HUSER'});
  await f.emit(Promise.resolve(f.terminal()));const activity=await f.activity();
  assert.equal(activity.state,'confirmed');assert.deepEqual(activity.proof.nativeImmunity,f.terminal());
  assert.equal(f.commits.length,1);assert.deepEqual(f.bridgeErrors,[]);
 }finally{f.stop()}
});

test('a held owner terminal is rejected if healer ownership ends before the GM receives it',async()=>{
 const f=ownerFixture();await f.prepare();f.hub.holdTerminal=true;
 try{
  await f.emit(Promise.resolve(f.terminal()));const held=f.hub.held[0];assert.ok(held);
  f.healer.testUserPermission=user=>user?.isGM===true;f.hub.deliver(held.packet,'HUSER');await flush();
  assert.equal((await f.activity()).state,'awaiting-evidence');assert.equal(f.commits.length,0);
 }finally{f.stop()}
});
