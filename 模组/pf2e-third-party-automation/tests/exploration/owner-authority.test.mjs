import {createNativeTreatment} from '../../scripts/exploration/native-treatment.mjs';
import {createCoordinator} from '../../scripts/exploration/coordinator.mjs';
import {recoveryProposals} from '../../scripts/exploration/policy.mjs';
import {TREAT_WOUNDS_IMMUNITY} from '../../scripts/salubrious-kiss-rules.mjs';
import {OWNER_TRANSPORT_PROTOCOL} from '../../scripts/exploration/owner-transport.mjs';
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {authorityFixture} from './authority-fixture.mjs';
import {createExplorationOwnerOperations} from '../../scripts/exploration/owner-operations.mjs';
import {MODULE_ID} from '../../scripts/exploration/schema.mjs';
import {OWNER_TRANSPORT_CHANNEL} from '../../scripts/exploration/owner-transport.mjs';
import {createRefocusAdapter,createRefocusProvider} from '../../scripts/exploration/refocus.mjs';
import {validateNativeResult,nativeParameters,projectCommand} from '../../scripts/exploration/owner-command.mjs';
import {fixture as kissFixture} from '../salubrious-kiss-fixture.mjs';
import {identityKeys,marker} from '../../scripts/salubrious-kiss-context.mjs';

const tick=()=>new Promise(resolve=>setImmediate(resolve));
async function setup({owner='O',timeoutMs=1000,operation='treat-wounds',options={},onNative}={}){
 const f=await authorityFixture(),ledger=f.client('driver'),userG={id:'G',isGM:true,active:true},userO={id:'O',isGM:false,active:true},none={id:'N',active:true},users=new Map([['G',userG],['O',userO],['N',none]]);users.activeGM=userG;
 const messages=new Map(),actors=new Map();let calls=0,intercept=packet=>packet;const packets=[],tabs=[];
 for(const id of ['H','P']){const actor={id,uuid:`Actor.${id}`,flags:{},system:{resources:{focus:{value:0,max:3}}},testUserPermission:user=>user?.isGM||user?.id==='O',update:async data=>{for(const [key,value] of Object.entries(data))if(key===`flags.${MODULE_ID}.explorationExecutions`)actor.flags[MODULE_ID]={...actor.flags[MODULE_ID],explorationExecutions:structuredClone(value)};return actor}};actors.set(actor.uuid,actor)}
 const session=await ledger.createSession({id:'S',actorUUIDs:[...actors.keys()],nativeOwnerByActor:{'Actor.H':owner},startedAt:0,cursorAt:0,budgetEndsAt:7200,status:'running'}),scope={leaseNonce:session.driver.leaseNonce};
 let a=await ledger.insertActivity({id:'A',sessionId:'S',providerId:operation,actorUUID:'Actor.H',patientUUIDs:operation==='refocus'&&!options.threePecks?[]:['Actor.P'],hpPoolUUIDs:operation==='refocus'&&!options.threePecks?[]:['Actor.P'],startedAt:0,endsAt:600,state:'planned',source:{ownerId:owner},options},scope);
 await ledger.transitionActivity('A',{...scope,expected:['planned'],patch:{state:'started'}});a=await ledger.transitionActivity('A',{...scope,expected:['started'],patch:{state:'completing'}});
 function make(userId,clientNonce,driver=false){const listeners=new Set(),user=users.get(userId),identity={userId,clientNonce};let currentLedger=f.client(clientNonce),privateScope=driver?scope:null;const game={user,users,time:{worldTime:600},messages,socket:{on:(channel,fn)=>{assert.equal(channel,OWNER_TRANSPORT_CHANNEL);listeners.add(fn)},off:(channel,fn)=>listeners.delete(fn),emit:(channel,payload,routing,ack)=>{assert.deepEqual(routing,{recipients:[payload.receiverUserId]});packets.push({sender:userId,packet:structuredClone(payload)});const wire=intercept(structuredClone(payload),userId);if(wire)for(const tab of tabs.filter(t=>t.userId===routing.recipients[0]))for(const listener of tab.listeners)listener(structuredClone(wire),userId);ack?.()}}};
  const ops=createExplorationOwnerOperations({game,fromUuid:async uuid=>actors.get(uuid),ledger:userId==='G'?new Proxy({},{get:(_,key)=>typeof currentLedger[key]==='function'?currentLedger[key].bind(currentLedger):currentLedger[key]}):{atomic:true,getActivity:async()=>{throw Error('owner-must-not-read-ledger')}},runtimeIdentity:()=>identity,getDriverScope:()=>privateScope,timeoutMs});
  ops.registerOperation(operation,async(activity,ctx)=>{ctx.validate();assert.equal(ops.isActivityContext(ctx,activity.id),true);assert.equal(ops.isActivityContext({...ctx},activity.id),false);calls++;if(onNative)return onNative(activity,ctx);if(operation==='refocus'){const actor=actors.get('Actor.H');actor.system.resources.focus.value=1;actor.flags[MODULE_ID]={...actor.flags[MODULE_ID],avRefocusIntent:{nonce:activity.id,actorUuid:actor.uuid,userId,startedAt:activity.startedAt,before:0,after:1}};return {status:'confirmed',proof:{useId:activity.id,checkIds:[],resultIds:[],receiptIds:[activity.id],immunityIds:[],poolReceipts:[]},focusBefore:0,focusAfter:1}}
   const check={id:'C',author:user,speaker:{actor:'H'},flags:{pf2e:{context:{options:['exploration-activity:A'],outcome:'success'}},[MODULE_ID]:{exploration:{activityId:'A',patientUUID:'Actor.P'}}},rolls:[{_evaluated:true,total:20}]};
   const result={id:'R',author:user,speaker:{actor:'H'},flags:{pf2e:{origin:{messageId:'C'},context:{options:['exploration-activity:A']}},[MODULE_ID]:{exploration:{activityId:'A',patientUUID:'Actor.P'}}},rolls:[{_evaluated:true,total:8,toJSON:()=>({formula:'{2d8[healing]}'})}]};
   const receipt={id:'HP',author:user,speaker:{actor:'P'},flags:{pf2e:{context:{type:'damage-taken',options:[`${MODULE_ID}:source:R:0`,`${MODULE_ID}:exploration-apply:A:R:Actor.P`]},appliedDamage:{uuid:'Actor.P',isReverted:false}}}};
   for(const m of [check,result,receipt])messages.set(m.id,m);
   return {status:'confirmed',proof:{useId:'A',checkIds:['C'],resultIds:['R'],receiptIds:['HP'],immunityIds:[],poolReceipts:[]},effectiveOutcome:'success',rolledHealing:8,patientUUID:'Actor.P'};
  });ops.register({});const tab={userId,clientNonce,listeners,ops,game,get ledger(){return currentLedger},adopt:(next,leaseNonce)=>{identity.clientNonce=next;currentLedger=f.client(next);privateScope=leaseNonce?{leaseNonce}:null}};tabs.push(tab);return tab;
 }
 const driver=make('G','driver',true),peer=make('G','peer'),one=make('O','owner-one'),two=make('O','owner-two'),outsider=make('N','none');
 return {f,ledger,a,driver,peer,one,two,outsider,actors,messages,packets,tabs,make,get calls(){return calls},intercept:fn=>{intercept=fn},dispose:()=>tabs.forEach(t=>t.ops.dispose?.())};
}

test('two GM tabs and two owner tabs execute one persisted atomic grant',async()=>{const s=await setup();try{const result=await s.driver.ops.runActivityWithOwner(s.a,'treat-wounds');assert.equal(result.status,'confirmed');assert.equal(s.calls,1);assert.equal((await s.ledger.getActivity('A')).executor.state,'settled');assert.equal(s.packets.filter(p=>p.packet.kind==='grant').length,1);assert.equal(s.packets.filter(p=>p.sender==='N').length,0)}finally{s.dispose()}});
test('local GM execution also persists a one-use grant',async()=>{const s=await setup({owner:'G'});try{await s.driver.ops.runActivityWithOwner(s.a,'treat-wounds');assert.equal(s.calls,1);assert.equal((await s.ledger.getActivity('A')).executor.ownerClientNonce,'driver');await assert.rejects(s.driver.ops.runActivityWithOwner(s.a,'treat-wounds'));assert.equal(s.calls,1)}finally{s.dispose()}});
test('a peer knowing the persisted lease cannot start native work',async()=>{const s=await setup();try{await assert.rejects(s.peer.ops.runActivityWithOwner(s.a,'treat-wounds'),/driver/);assert.equal(s.calls,0)}finally{s.dispose()}});
test('the public old ownerExecute cannot consume even a saved atomic permit',async()=>{const s=await setup({owner:'G'});try{await assert.rejects(s.driver.ops.ownerExecute({activity:s.a,operationId:'treat-wounds'},'G'),/private|atomic/);assert.equal(s.calls,0)}finally{s.dispose()}});
test('wrong authenticated sender cannot turn an offer into native work',async()=>{const s=await setup({timeoutMs:30});try{s.intercept((p,sender)=>p.kind==='offer'?{...p,driverUserId:'O'}:p);await assert.rejects(s.driver.ops.runActivityWithOwner(s.a,'treat-wounds'));assert.equal(s.calls,0);assert.equal((await s.ledger.getActivity('A')).executor,undefined)}finally{s.dispose()}});
test('a changed actor or OWNER permission cannot obtain a grant',async()=>{for(const change of ['actor','permission']){const s=await setup({timeoutMs:30});try{if(change==='actor')s.intercept(p=>p.kind==='claim'?{...p,actorUUID:'Actor.P'}:p);else s.actors.get('Actor.H').testUserPermission=user=>user?.isGM;await assert.rejects(s.driver.ops.runActivityWithOwner(s.a,'treat-wounds'));assert.equal(s.calls,0);assert.equal((await s.ledger.getActivity('A')).executor,undefined)}finally{s.dispose()}}});
test('duplicate offer and grant deliveries do not open another native window',async()=>{const s=await setup();try{await s.driver.ops.runActivityWithOwner(s.a,'treat-wounds');for(const {sender,packet} of s.packets.filter(p=>['offer','grant'].includes(p.packet.kind)))for(const t of s.tabs.filter(t=>t.userId===packet.receiverUserId))for(const fn of t.listeners)fn(structuredClone(packet),sender);await tick();assert.equal(s.calls,1)}finally{s.dispose()}});
test('an independent fresh owner and NONE knowing saved nonces cannot replay a grant',async()=>{const s=await setup();try{await s.driver.ops.runActivityWithOwner(s.a,'treat-wounds');const fresh=s.make('O','fresh'),grant=s.packets.find(p=>p.packet.kind==='grant');for(const t of [fresh,s.outsider])for(const fn of t.listeners)fn({...structuredClone(grant.packet),receiverUserId:t.userId},'G');await assert.rejects(fresh.ops.ownerExecute({permit:(await s.ledger.getActivity('A')).executor,activity:s.a,operationId:'treat-wounds'},'G'));await tick();assert.equal(s.calls,1)}finally{s.dispose()}});
test('timeout followed by a late grant does not execute or acquire a second permit',async()=>{const s=await setup({timeoutMs:30});let grant;try{s.intercept(p=>{if(p.kind==='grant'){grant=structuredClone(p);return null}return p});await assert.rejects(s.driver.ops.runActivityWithOwner(s.a,'treat-wounds'));assert.equal(s.calls,0);assert.ok((await s.ledger.getActivity('A')).executor);for(const t of [s.one,s.two])for(const fn of t.listeners)fn(grant,'G');await tick();assert.equal(s.calls,0);await assert.rejects(s.driver.ops.runActivityWithOwner(s.a,'treat-wounds'));assert.equal(s.calls,0)}finally{s.dispose()}});
test('dispose invalidates pending claims and removes only its own listener',async()=>{const s=await setup({timeoutMs:30});try{s.intercept(p=>{if(p.kind==='grant'){s.one.ops.dispose();s.two.ops.dispose()}return p});await assert.rejects(s.driver.ops.runActivityWithOwner(s.a,'treat-wounds'));assert.equal(s.calls,0);assert.equal(s.driver.listeners.size,1)}finally{s.dispose()}});
test('ordinary refocus settles its opaque receipt without requiring a ChatMessage',async()=>{const s=await setup({operation:'refocus'});try{const r=await s.driver.ops.runActivityWithOwner(s.a,'refocus');assert.equal(r.status,'confirmed');assert.equal(s.calls,1);assert.equal(s.messages.size,0);assert.equal((await s.ledger.getActivity('A')).executor.state,'settled')}finally{s.dispose()}});
test('remote three pecks is explicitly blocked before a grant',async()=>{const s=await setup({operation:'refocus',options:{threePecks:true}});try{await assert.rejects(s.driver.ops.runActivityWithOwner(s.a,'refocus'),/three-pecks/);assert.equal(s.calls,0);assert.equal((await s.ledger.getActivity('A')).executor,undefined)}finally{s.dispose()}});
test('a patient losing OWNER permission is rejected before native work',async()=>{const s=await setup({timeoutMs:30});try{s.actors.get('Actor.P').testUserPermission=user=>user?.isGM;await assert.rejects(s.driver.ops.runActivityWithOwner(s.a,'treat-wounds'));assert.equal(s.calls,0)}finally{s.dispose()}});
test('runtime invalidation while grant is held prevents its consumption',async()=>{const s=await setup({timeoutMs:30});try{s.intercept(p=>{if(p.kind==='grant'){s.one.ops.invalidate('disconnect');s.two.ops.invalidate('disconnect')}return p});await assert.rejects(s.driver.ops.runActivityWithOwner(s.a,'treat-wounds'));assert.equal(s.calls,0)}finally{s.dispose()}});
test('revoked native HP receipt prevents reconciliation from saved owner completion',async()=>{const s=await setup();try{await s.driver.ops.runActivityWithOwner(s.a,'treat-wounds');s.messages.get('HP').flags.pf2e.appliedDamage.isReverted=true;const saved=await s.ledger.getActivity('A');const result=await s.driver.ops.reconcile(saved);assert.equal(result.status,'uncertain');assert.equal(s.calls,1)}finally{s.dispose()}});
test('ordinary remote refocus consumes the private permit without reading a GM ledger',async()=>{
 let calls=0;const ctx={},activity={id:'A',actorUUID:'Actor.H',startedAt:0,endsAt:600,options:{}},ownerOperations={isActivityContext:()=>true,isExecutionContext:value=>value===ctx},provider=createRefocusProvider({game:{},capabilities:{discover:async()=>({})},ledger:{getActivity:async()=>{throw Error('private-ledger-read')}},ownerOperations,refocusEvents:{complete:async()=>{calls++;return {id:'A',focusBefore:0,focusAfter:1}}}});
 assert.equal((await provider.complete(activity,ctx)).status,'confirmed');assert.equal(calls,1);
});
test('remote refocus resolves only the exact native update intent on its own tab',async()=>{
 const user={id:'O'},actor={uuid:'Actor.H',flags:{},system:{resources:{focus:{value:0,max:3}}}},activity={id:'A',actorUUID:actor.uuid,startedAt:0,endsAt:600,options:{}},ctx={validate(){}},canvas={tokens:{controlled:[{actor}]}},game={user,time:{worldTime:600},PF2eWorkbench:{refocus:async()=>{actor.system.resources.focus.value=1;actor.flags[MODULE_ID]={avRefocusIntent:{nonce:'A',actorUuid:actor.uuid,userId:'O',startedAt:0,before:0,after:1}}}}},ops={isActivityContext:()=>true};
 const adapter=createRefocusAdapter({game,canvas,fromUuid:async()=>actor,ownerOperations:ops,timeoutMs:20});assert.deepEqual(await adapter.complete(activity,ctx),{id:'A',focusBefore:0,focusAfter:1});
});
test('a saved refocus result without its native intent is uncertain',async()=>{const s=await setup({operation:'refocus',timeoutMs:30});try{s.intercept(p=>{if(p.kind==='completion')delete s.actors.get('Actor.H').flags[MODULE_ID].avRefocusIntent;return p});await assert.rejects(s.driver.ops.runActivityWithOwner(s.a,'refocus'));assert.equal((await s.ledger.getActivity('A')).executor.state,'granted')}finally{s.dispose()}});
test('GM local three pecks consumes the original private begin context',async()=>{
 let ctx;const s=await setup({owner:'G',operation:'refocus',timeoutMs:100,onNative:async(a,received)=>{assert.equal(received,ctx);assert.equal(a.source.type,'coordinator');return {status:'blocked',reason:'native-declined'}}});try{
  const leaseNonce=(await s.ledger.getSession('S')).driver.leaseNonce;
  const b=await s.ledger.insertActivity({...s.a,id:'B',patientUUIDs:['Actor.P'],hpPoolUUIDs:['Actor.P'],startedAt:600,endsAt:1200,state:'planned',options:{threePecks:true},proof:undefined},{leaseNonce});ctx=await s.driver.ops.createActivityContext(b);
  await s.ledger.transitionActivity('B',{leaseNonce:(await s.ledger.getSession('S')).driver.leaseNonce,expected:['planned'],patch:{state:'started'}});const completing=await s.ledger.transitionActivity('B',{leaseNonce:(await s.ledger.getSession('S')).driver.leaseNonce,expected:['started'],patch:{state:'completing'}});s.driver.game.time.worldTime=1200;
  assert.equal((await s.driver.ops.runActivityWithOwner(completing,'refocus')).status,'blocked');assert.equal(s.calls,1);assert.equal(s.driver.ops.isActivityContext(ctx,'B'),false);
 }finally{s.dispose()}
});
test('remote extension applies the stored healing source with no new check',async()=>{
 const s=await setup();try{
  await s.driver.ops.runActivityWithOwner(s.a,'treat-wounds');const original=await s.ledger.getActivity('A'),leaseNonce=(await s.ledger.getSession('S')).driver.leaseNonce;
  await s.ledger.transitionActivity('A',{leaseNonce,expected:['completing'],patch:{state:'confirmed'}});
  await s.ledger.insertActivity({...s.a,id:'B',startedAt:600,endsAt:3600,state:'planned',options:{extensionOf:'A'}},{leaseNonce});await s.ledger.transitionActivity('B',{leaseNonce,expected:['planned'],patch:{state:'started'}});const b=await s.ledger.transitionActivity('B',{leaseNonce,expected:['started'],patch:{state:'completing'}});
  let applications=0;for(const tab of [s.driver,s.one,s.two]){tab.game.time.worldTime=3600;tab.ops.registerOperation('treatment-extension',async(a,ctx)=>{assert.equal(ctx.extensionOriginal.id,'A');assert.equal(ctx.extensionOriginal.results[0].rolledHealing,8);applications++;const m=structuredClone(s.messages.get('HP'));m.id='HP2';m.flags.pf2e.context.options=[`${MODULE_ID}:source:R:0`,`${MODULE_ID}:exploration-apply:B:R:Actor.P`];s.messages.set(m.id,m);return {status:'confirmed',proof:{useId:'A',checkIds:['C'],resultIds:['R'],receiptIds:['HP2'],immunityIds:[],poolReceipts:[]},effectiveOutcome:'success',rolledHealing:8}})}
  assert.equal((await s.driver.ops.runActivityWithOwner(b,'treatment-extension')).status,'confirmed');assert.equal(applications,1);assert.equal(s.calls,1);assert.equal(s.messages.size,4);assert.equal((await s.ledger.getActivity('B')).executor.state,'settled');assert.equal(original.rolledHealing,8);
 }finally{s.dispose()}
});
test('unknown revision acknowledgement leaves a saved grant without native work',async()=>{const s=await setup({timeoutMs:30});try{s.f.setAcknowledgement(()=>undefined);await assert.rejects(s.driver.ops.runActivityWithOwner(s.a,'treat-wounds'));assert.equal(s.calls,0);assert.ok((await s.ledger.getActivity('A')).executor);assert.equal(s.packets.filter(p=>p.packet.kind==='grant').length,0)}finally{s.dispose()}});
test('mismatched request and owner attempt responses cannot consume native authority',async()=>{for(const field of ['requestId','ownerClientNonce','attemptNonce','receiverUserId']){const s=await setup({timeoutMs:30});try{s.intercept(p=>p.kind==='grant'?{...p,[field]:'N'}:p);await assert.rejects(s.driver.ops.runActivityWithOwner(s.a,'treat-wounds'));assert.equal(s.calls,0)}finally{s.dispose()}}});
test('completion payload cannot replace the current saved native proof',async()=>{const s=await setup({timeoutMs:30});try{s.intercept(p=>{if(p.kind==='completion'){s.messages.delete('HP');p.results=[{status:'confirmed',proof:{receiptIds:['invented']}}]}return p});await assert.rejects(s.driver.ops.runActivityWithOwner(s.a,'treat-wounds'));assert.equal((await s.ledger.getActivity('A')).executor.state,'granted');assert.equal(s.calls,1)}finally{s.dispose()}});
test('invalidate retires old scopes but a new explicit resume ACK can drive new work',async()=>{
 const s=await setup({owner:'G',operation:'refocus'});try{
  await s.driver.ops.runActivityWithOwner(s.a,'refocus');let leaseNonce=(await s.ledger.getSession('S')).driver.leaseNonce;await s.ledger.transitionActivity('A',{leaseNonce,expected:['completing'],patch:{state:'confirmed'}});await s.ledger.updateSession('S',{status:'paused'});
  s.driver.adopt('new-driver');s.driver.ops.invalidate('disconnect');const resumed=await s.driver.ledger.resumeSession('S',{cursorAt:0});leaseNonce=resumed.driver.leaseNonce;s.driver.adopt('new-driver',leaseNonce);
  await s.driver.ledger.insertActivity({...s.a,id:'B',startedAt:600,endsAt:1200,state:'planned'},{leaseNonce});await s.driver.ledger.transitionActivity('B',{leaseNonce,expected:['planned'],patch:{state:'started'}});const b=await s.driver.ledger.transitionActivity('B',{leaseNonce,expected:['started'],patch:{state:'completing'}});s.driver.game.time.worldTime=1200;
  assert.equal((await s.driver.ops.runActivityWithOwner(b,'refocus')).status,'confirmed');assert.equal(s.calls,2);assert.equal((await s.driver.ledger.getActivity('B')).executor.ownerClientNonce,'new-driver');assert.equal(s.driver.listeners.size,1);
 }finally{s.dispose()}
});
test('local Three Pecks validates its actual saved Salubrious check and refocus intent',async()=>{
 const f=kissFixture(),claim={...f.claim,nonce:'A',userId:f.gm.id,state:'done',result:{checkId:'C',damageId:null,degree:1},immunityIds:[]};f.setClaim(claim);f.game.time.worldTime=700;f.actor.flags[MODULE_ID].avRefocusIntent={...f.proof,nonce:'A',userId:f.gm.id};
 const roll={_evaluated:true,total:15,options:{degreeOfSuccess:1},toJSON:()=>({evaluated:true})},check={id:'C',isCheckRoll:true,blind:false,whisper:[],author:f.gm,speaker:{actor:f.actor.id,scene:f.scene.id,token:f.token.id},rolls:[roll],flags:{pf2e:{origin:{actor:f.actor.uuid,uuid:f.item.uuid},context:{messageMode:'public',origin:{actor:f.actor.uuid,token:f.token.uuid},type:'skill-check',action:'treat-wounds',dc:{value:30},domains:['occultism'],options:[marker('check',claim),'skip-handling-message'],outcome:'failure'},suppressDamageButtons:true},[MODULE_ID]:{salubriousKiss:{kind:'check',...Object.fromEntries(identityKeys.map(key=>[key,claim[key]]))}}}};f.game.messages.set('C',check);
 const treatment={status:'confirmed',proof:{useId:'A',checkIds:['C'],resultIds:[],receiptIds:[],immunityIds:[],poolReceipts:[]},sourceDegree:1,effectiveOutcome:'failure',rolledHealing:null},result={status:'confirmed',proof:{...treatment.proof,receiptIds:['A']},focusBefore:2,focusAfter:3,treatment};
 assert.equal((await validateNativeResult({game:f.game,fromUuid:f.fromUuid,activity:{id:'A',actorUUID:f.actor.uuid,patientUUIDs:[f.patient.uuid],hpPoolUUIDs:[f.patient.uuid],startedAt:100,endsAt:700,options:{threePecks:true}},permit:{operationId:'refocus',ownerUserId:f.gm.id},result})).status,'confirmed');
 check.author=f.user;await assert.rejects(validateNativeResult({game:f.game,fromUuid:f.fromUuid,activity:{id:'A',actorUUID:f.actor.uuid,patientUUIDs:[f.patient.uuid],hpPoolUUIDs:[f.patient.uuid],startedAt:100,endsAt:700,options:{threePecks:true}},permit:{operationId:'refocus',ownerUserId:f.gm.id},result}));
});
test('focus healing confirmation requires the actual used cast payment receipt',async()=>{
 const s=await setup();try{
  const actor=s.actors.get('Actor.H'),item={id:'F',uuid:'Actor.H.Item.F',actor,sourceId:'Compendium.pf2e.spells-srd.Item.zNN9212H2FGfM7VS'};s.actors.set(item.uuid,item);actor.items=new Map([['F',item]]);
  actor.flags[MODULE_ID]={explorationFocusCommits:{A:{activityId:'A',castNonce:'PAY',itemUuid:item.uuid,before:1,after:0,cost:1}}};
  const card={id:'CAST',author:s.one.game.user,speaker:{actor:'H'},flags:{pf2e:{origin:{uuid:item.uuid,actor:actor.uuid}},[MODULE_ID]:{explorationFocus:{activityId:'A'},nativeCast:{id:'PAY',actorUuid:actor.uuid,itemUuid:item.uuid,userId:'O'}}}},damage={id:'D',author:s.one.game.user,speaker:{actor:'H'},flags:{pf2e:{origin:{uuid:item.uuid,messageId:'CAST'},context:{options:['exploration-activity:A']}},[MODULE_ID]:{exploration:{activityId:'A',patientUUID:'Actor.P'}}},rolls:[{_evaluated:true,total:8,toJSON:()=>({formula:'{8[healing]}'})}]},receipt={id:'HP',author:s.one.game.user,speaker:{actor:'P'},flags:{pf2e:{context:{type:'damage-taken',options:[`${MODULE_ID}:source:D:0`,`${MODULE_ID}:exploration-apply:A:D:Actor.P`]},appliedDamage:{uuid:'Actor.P',isReverted:false}}}};
  for(const m of [card,damage,receipt])s.messages.set(m.id,m);
  const result={status:'confirmed',proof:{useId:'A',checkIds:[],resultIds:['CAST','D'],receiptIds:['HP'],immunityIds:[],poolReceipts:[]},resourceReceiptIds:['PAY'],rolledHealing:8},args={game:s.driver.game,fromUuid:async uuid=>s.actors.get(uuid),activity:{...s.a,options:{itemUUID:item.uuid}},permit:{operationId:'focus-healing',ownerUserId:'O'},result};
  await assert.rejects(validateNativeResult(args),/payment/);
  actor.flags[MODULE_ID].nativeCasts=[{id:'PAY',state:'used',messageId:'CAST',userId:'O',actorUuid:actor.uuid,itemUuid:item.uuid,focusPoints:1}];assert.equal((await validateNativeResult(args)).status,'confirmed');
  actor.flags[MODULE_ID].nativeCasts[0].state='paid';await assert.rejects(validateNativeResult(args),/payment/);
 }finally{s.dispose()}
});
test('native command decoding rejects nonprojected fields and mismatched domains',()=>{
 const a={id:'A',providerId:'treat-wounds',actorUUID:'Actor.H',patientUUIDs:['Actor.P'],hpPoolUUIDs:['Actor.P'],startedAt:0,endsAt:600,options:{}},permit={activityId:'A',sessionId:'S',operationId:'treat-wounds',actorUUID:'Actor.H'},command=projectCommand(a,'treat-wounds');
 assert.equal(nativeParameters(command,permit).actorUUID,'Actor.H');
 for(const malformed of [{...command,other:true},{...command,parameters:{...command.parameters,extra:true}},{...command,parameters:{...command.parameters,actorUUID:'Actor.Other'}},{...command,parameters:{...command.parameters,endsAt:-1}},{...command,parameters:{...command.parameters,patientUUIDs:[]}}])assert.throws(()=>nativeParameters(malformed,permit),/command/);
});
test('a new active GM can reconcile an exact older permit without granting another native call',async()=>{
 const s=await setup({timeoutMs:30});try{
  s.intercept(p=>p.kind==='completion'?null:p);await assert.rejects(s.driver.ops.runActivityWithOwner(s.a,'treat-wounds'));const saved=await s.ledger.getActivity('A');assert.equal(s.calls,1);
  const gm={id:'OtherGM',isGM:true,active:true};s.driver.game.users.set(gm.id,gm);s.driver.game.users.activeGM=gm;
  const ops=createExplorationOwnerOperations({game:{...s.driver.game,user:gm},fromUuid:async uuid=>s.actors.get(uuid),ledger:s.f.client('new-gm',gm.id),runtimeIdentity:()=>({userId:gm.id,clientNonce:'new-gm'}),getDriverScope:()=>null});
  assert.equal((await ops.reconcile(saved)).status,'confirmed');assert.equal((await s.ledger.getActivity('A')).executor.state,'settled');assert.equal(s.calls,1);ops.dispose();
 }finally{s.dispose()}
});
test('patient OWNER is checked again after saving the started execution',async()=>{
 const s=await setup({timeoutMs:30});try{const actor=s.actors.get('Actor.H'),update=actor.update;actor.update=async data=>{const saved=await update(data);if(data[`flags.${MODULE_ID}.explorationExecutions`]?.A?.state==='started')s.actors.get('Actor.P').testUserPermission=user=>user?.isGM;return saved};await assert.rejects(s.driver.ops.runActivityWithOwner(s.a,'treat-wounds'));assert.equal(s.calls,0)}finally{s.dispose()}
});

const deferred=()=>{let resolve;return {promise:new Promise(r=>resolve=r),resolve:v=>resolve(v)}};

async function nativeBrokerFixture({activity,seed={sessions:{},activities:{},clocks:{}},owner='G'}={}){
 const f=await authorityFixture(seed),ledger=f.client('driver'),gm={id:'G',isGM:true,active:true},player={id:'O',isGM:false,active:true},users=new Map([['G',gm],['O',player]]);users.activeGM=gm;
 const documents=new Map(),messages=new Map(),tabs=[],hooks=new Map();let hookId=0,afterUpdate=async()=>{};
 for(const id of ['H','P','Q']){
  const actor={id,uuid:`Actor.${id}`,flags:{},items:[],system:{resources:{focus:{value:0,max:3}}},testUserPermission:()=>true,getSelfRollOptions:()=>[],getRollOptions:()=>[],getActiveTokens:()=>[],getContextualClone(){return this},update:async data=>{for(const [key,value]of Object.entries(data)){const keys=key.split('.');let at=actor;for(const k of keys.slice(0,-1))at=at[k]??={};at[keys.at(-1)]=structuredClone(value)}await afterUpdate(actor,data);return actor}};
  documents.set(actor.uuid,actor);
 }
 const session=await ledger.createSession({id:'S',actorUUIDs:[...documents.keys()],nativeOwnerByActor:{[activity.actorUUID]:owner},startedAt:activity.startedAt,cursorAt:activity.startedAt,budgetEndsAt:7200,activityIds:[],goalsByPool:[],status:'running'});let privateScope={leaseNonce:session.driver.leaseNonce};
 const planned=await ledger.insertActivity({...activity,id:'B',sessionId:'S',state:'planned',source:{ownerId:owner}},privateScope);
 const Hooks={on(event,fn){const id=++hookId;hooks.set(id,{event,fn});return id},off(event,id){hooks.delete(id)}};
 const emitHook=(event,...args)=>{for(const row of hooks.values())if(row.event===event)row.fn(...args)};
 function make(userId,clientNonce){
  const listeners=new Set(),game={user:users.get(userId),users,messages,time:{worldTime:activity.startedAt},socket:{on(event,fn){listeners.add(fn)},off(event,fn){listeners.delete(fn)},emit(event,packet,routing,ack){for(const tab of tabs.filter(t=>routing.recipients.includes(t.game.user.id)))for(const fn of tab.listeners)queueMicrotask(()=>fn(structuredClone(packet),userId));ack?.()}}};
  const ops=createExplorationOwnerOperations({game,ledger:userId==='G'?ledger:{atomic:true},runtimeIdentity:()=>({userId,clientNonce}),getDriverScope:()=>userId==='G'?privateScope:null,fromUuid:async uuid=>documents.get(uuid),timeoutMs:80});
  const tab={game,ops,listeners};tabs.push(tab);ops.register();return tab;
 }
 const driver=make('G','driver'),receiver=owner==='G'?driver:make('O','owner');
 const ctx=await driver.ops.createActivityContext(planned);
 await ledger.transitionActivity('B',{...privateScope,expected:['planned'],patch:{state:'started'}});
 const completing=await ledger.transitionActivity('B',{...privateScope,expected:['started'],patch:{state:'completing'}});
 for(const tab of tabs)tab.game.time.worldTime=activity.endsAt;
 return {f,ledger,driver,receiver,documents,messages,Hooks,emitHook,ctx,activity:completing,scope:privateScope,setAfterUpdate(fn){afterUpdate=fn},clearDriver(){privateScope=null},dispose(){for(const tab of tabs)tab.ops.dispose()}};
}

test('the ordinary refocus proposal reaches the granted native handler',async()=>{
 const actor={actorUUID:'Actor.H',pool:{ready:true,poolUUID:'Actor.H'},hp:{value:20,max:20},focus:{value:0,max:1},items:[],slugs:[],assuranceSkills:[],refocusUnsupported:[]};
 const proposal=recoveryProposals({actors:[actor],activities:[],session:{goalsByPool:[{poolUUID:'Actor.H',targetHP:20}],requireFullFocus:true},now:0,providerIds:['refocus']})[0];
 assert.deepEqual(proposal.patientUUIDs,[]);assert.deepEqual(proposal.hpPoolUUIDs,[]);
 const s=await nativeBrokerFixture({activity:{...proposal,startedAt:0,endsAt:600}});let native=0,error,result;
 try{
  s.receiver.ops.registerOperation('refocus',async(a,ctx)=>{ctx.validate();native++;return {status:'blocked',reason:'probe-stops-at-native-boundary'}});
  try{result=await s.driver.ops.runActivityWithOwner(s.activity,'refocus')}catch(e){error=e}
  console.log(JSON.stringify({case:'ordinary-policy-refocus',nativeCalls:native,error:error?.message,executor:(await s.ledger.getActivity('B')).executor?.state}));
  assert.equal(error,undefined);assert.equal(native,1);assert.equal(result.status,'blocked');
 }finally{s.dispose()}
});

test('native extension of a selected Ward Medic patient confirms the already-applied healing',async()=>{
 const rows=['P','Q'].map(patient=>({status:'confirmed',patientUUID:`Actor.${patient}`,effectiveOutcome:'success',rolledHealing:8,medicBonus:0,proof:{useId:'A',checkIds:[`C${patient}`],resultIds:[`R${patient}`]}}));
 const original={id:'A',sessionId:'Old',providerId:'treat-wounds',actorUUID:'Actor.H',patientUUIDs:['Actor.P','Actor.Q'],hpPoolUUIDs:['Actor.P','Actor.Q'],startedAt:0,endsAt:600,state:'confirmed',effectiveOutcome:'success',rolledHealing:8,results:rows,proof:{useId:'A',checkIds:['CP','CQ'],resultIds:['RP','RQ'],receiptIds:['OLDP','OLDQ'],immunityIds:[]}};
 const s=await nativeBrokerFixture({activity:{providerId:'treat-wounds',actorUUID:'Actor.H',patientUUIDs:['Actor.P'],hpPoolUUIDs:['Actor.P'],startedAt:600,endsAt:3600,options:{extensionOf:'A'}},seed:{sessions:{Old:{id:'Old',status:'complete'}},activities:{A:original},clocks:{}}});let applications=0,error,result;
 try{
  for(const patient of ['P','Q']){
   const common={author:s.driver.game.user,speaker:{actor:'H'},flags:{[MODULE_ID]:{exploration:{activityId:'A',patientUUID:`Actor.${patient}`}},pf2e:{context:{options:['exploration-activity:A'],outcome:'success'}}}};
   s.messages.set(`C${patient}`,{...structuredClone(common),id:`C${patient}`});
   s.messages.set(`R${patient}`,{...structuredClone(common),id:`R${patient}`,flags:{...structuredClone(common.flags),pf2e:{...structuredClone(common.flags.pf2e),origin:{messageId:`C${patient}`}}},rolls:[{_evaluated:true,total:8,toJSON:()=>({formula:'{2d8[healing]}'})}]});
  }
  const native=createNativeTreatment({game:s.driver.game,Hooks:s.Hooks,fromUuid:async uuid=>s.documents.get(uuid),ownerOperations:s.driver.ops,damageGuard:{authorizeExploration:async()=>()=>{}},hpPools:{withNativeApplication:async(a,p,operation)=>({result:await operation()})},apply:async request=>{
   applications++;const receipt={id:'NEWP',author:s.driver.game.user,speaker:{actor:'P'},flags:{pf2e:{context:{type:'damage-taken',options:[request.source,request.application]},appliedDamage:{uuid:'Actor.P',isReverted:false}}}};s.messages.set(receipt.id,receipt);s.emitHook('createChatMessage',receipt);
  }});
  s.driver.ops.registerOperation('treatment-extension',(a,ctx)=>native.extend(a,ctx.extensionOriginal,ctx));
  try{result=await s.driver.ops.runActivityWithOwner(s.activity,'treatment-extension')}catch(e){error=e}
  console.log(JSON.stringify({case:'ward-selected-extension',applications,error:error?.message,executor:(await s.ledger.getActivity('B')).executor?.state,ownerRecord:s.documents.get('Actor.H').flags[MODULE_ID]?.explorationExecutions?.B?.state}));
  assert.equal(applications,1);assert.equal(error,undefined);assert.equal(result.status,'confirmed');assert.deepEqual(result.proof.checkIds,['CP']);assert.deepEqual(result.proof.resultIds,['RP']);s.messages.delete('CQ');s.messages.delete('RQ');assert.equal((await s.driver.ops.reconcile(await s.ledger.getActivity('B'))).status,'confirmed');assert.equal(applications,1);
 }finally{s.dispose()}
});

test('remote Stop before handler entry prevents a new native operation',async()=>{
 const s=await nativeBrokerFixture({owner:'O',activity:{providerId:'treat-wounds',actorUUID:'Actor.H',patientUUIDs:['Actor.P'],hpPoolUUIDs:['Actor.P'],startedAt:0,endsAt:600,options:{}}}),entered=deferred(),release=deferred();let calls=0;
 try{
  s.receiver.ops.registerOperation('treat-wounds',async()=>{calls++;return {status:'blocked',reason:'probe-stops-at-native-boundary'}});
  s.setAfterUpdate(async(actor,data)=>{if(data[`flags.${MODULE_ID}.explorationExecutions`]?.B?.state==='started'){entered.resolve();await release.promise}});
  const running=s.driver.ops.runActivityWithOwner(s.activity,'treat-wounds').catch(e=>e);await entered.promise;
  const peer=createCoordinator({ledger:s.ledger,capabilities:{},providers:[],clock:{stop(){}},ownerOperations:s.driver.ops,isAuthority:()=>true});
  await peer.stop('S');release.resolve();await running;
  console.log(JSON.stringify({case:'remote-stop-before-handler',nativeCalls:calls,session:(await s.ledger.getSession('S')).status,executor:(await s.ledger.getActivity('B')).executor?.state}));
  assert.equal(calls,0);
 }finally{s.dispose()}
});

for(const owner of ['G','O'])test(`actual native treatment handler accepts its ${owner==='G'?'local':'remote'} private execution context`,async()=>{
 const s=await nativeBrokerFixture({owner,activity:{providerId:'treat-wounds',actorUUID:'Actor.H',patientUUIDs:['Actor.P'],hpPoolUUIDs:['Actor.P'],startedAt:0,endsAt:600,options:{}}});let checks=0;
 try{
  const game=s.receiver.game,healer=s.documents.get('Actor.H'),patient=s.documents.get('Actor.P');
  s.documents.set(TREAT_WOUNDS_IMMUNITY,{toObject:()=>({system:{},flags:{}})});
  patient.createEmbeddedDocuments=async(type,[data])=>{const effect={...data,uuid:'Actor.P.Item.Immunity',actor:patient};s.documents.set(effect.uuid,effect);return [effect]};
  game.pf2e={actions:new Map([['treat-wounds',{use:async input=>{
   checks++;const message={id:'CHECK',actor:healer,author:game.user,speaker:{actor:'H'},flags:{pf2e:{context:{options:input.rollOptions,outcome:'failure'}}}};
   s.emitHook('preCreateChatMessage',message,message);s.messages.set(message.id,message);s.emitHook('createChatMessage',message);return [{actor:healer,message,outcome:'failure'}];
  }}]])};
  const native=createNativeTreatment({game,Hooks:s.Hooks,fromUuid:async uuid=>s.documents.get(uuid),ownerOperations:s.receiver.ops,checkScope:{runExploration:async(input,operation)=>{input.ctx.validate();return operation()}}});
  s.receiver.ops.registerOperation('treat-wounds',native.run);
  const result=await s.driver.ops.runActivityWithOwner(s.activity,'treat-wounds');assert.equal(result.status,'confirmed');assert.equal(checks,1);assert.equal((await s.ledger.getActivity('B')).executor.state,'settled');
  const gm={id:'NewGM',isGM:true,active:true};game.users.set(gm.id,gm);game.users.activeGM=gm;
  const fresh=createExplorationOwnerOperations({game:{...game,user:gm},fromUuid:async uuid=>s.documents.get(uuid),ledger:s.f.client('new-gm',gm.id),runtimeIdentity:()=>({userId:gm.id,clientNonce:'new-gm'}),getDriverScope:()=>null});
  assert.equal((await fresh.reconcile(await s.ledger.getActivity('B'))).status,'confirmed');assert.equal(checks,1);
  s.messages.delete('CHECK');assert.equal((await fresh.reconcile(await s.ledger.getActivity('B'))).status,'uncertain');assert.equal(checks,1);fresh.dispose();
 }finally{s.dispose()}
});

test('completed denied offers do not exhaust the owner tab across later sessions',async()=>{
 const gm={id:'G',isGM:true,active:true},owner={id:'O',isGM:false,active:true},users=new Map([['G',gm],['O',owner]]);users.activeGM=gm;const listeners=new Set();let claims=0;
 const game={user:owner,users,time:{worldTime:600},socket:{on(event,fn){listeners.add(fn)},off(event,fn){listeners.delete(fn)},emit(event,packet,routing,ack){claims++;queueMicrotask(()=>{for(const fn of listeners)fn({...packet,kind:'denied',receiverUserId:'O',errorCode:'execution-unavailable'},'G')});ack?.()}}};
 const ops=createExplorationOwnerOperations({game,ledger:{atomic:true},fromUuid:async()=>({testUserPermission:()=>true}),runtimeIdentity:()=>({userId:'O',clientNonce:'owner'}),timeoutMs:80});ops.register();
 try{
  for(let index=0;index<129;index++){
   const packet={protocol:OWNER_TRANSPORT_PROTOCOL,kind:'offer',requestId:`offer-request-${index}`,receiverUserId:'O',rootUUID:'JournalEntry.ROOT000000000001',epoch:'epoch',sessionId:`session-${index}`,activityId:`activity-${index}`,operationId:'treat-wounds',actorUUID:'Actor.H',driverUserId:'G',ownerUserId:'O',offerId:`offer-${index}`};
   for(const fn of listeners)fn(packet,'G');await new Promise(resolve=>setImmediate(resolve));
  }
  console.log(JSON.stringify({case:'owner-tab-lifetime-offer-cap',offers:129,claims}));assert.equal(claims,129);
 }finally{ops.dispose()}
});

test('an unknown continuation ACK never enters the remote native handler',async()=>{const s=await setup({timeoutMs:30});try{s.intercept(p=>p.kind==='continuation-ack'?null:p);await assert.rejects(s.driver.ops.runActivityWithOwner(s.a,'treat-wounds'));assert.equal(s.calls,0);assert.equal((await s.ledger.getActivity('A')).executor.state,'granted')}finally{s.dispose()}});
test('a continuation ACK with another permit cannot enter native work',async()=>{const s=await setup({timeoutMs:30});try{s.intercept(p=>p.kind==='continuation-ack'?{...p,proof:{permit:{...p.proof.permit,permitNonce:'wrong'}}}:p);await assert.rejects(s.driver.ops.runActivityWithOwner(s.a,'treat-wounds'));assert.equal(s.calls,0)}finally{s.dispose()}});
test('lost cancel ACK reports Stop as unknown and keeps the native permit blocked',async()=>{const s=await setup({timeoutMs:30});let release,entered;const gate=new Promise(r=>release=r),started=new Promise(r=>entered=r);try{const actor=s.actors.get('Actor.H'),update=actor.update;actor.update=async data=>{const value=await update(data);if(data[`flags.${MODULE_ID}.explorationExecutions`]?.A?.state==='started'){entered();await gate}return value};s.intercept(p=>p.kind==='cancel'?null:p);const running=s.driver.ops.runActivityWithOwner(s.a,'treat-wounds').catch(e=>e);await started;const coordinator=createCoordinator({ledger:s.ledger,capabilities:{},providers:[],clock:{stop(){}},ownerOperations:s.driver.ops,isAuthority:()=>true});await assert.rejects(coordinator.stop('S'),/stop-ack-unknown/);release();await running;assert.equal(s.calls,0);assert.equal((await s.ledger.getActivity('A')).executor.state,'granted');assert.equal((await s.ledger.getSession('S')).status,'paused')}finally{release?.();s.dispose()}});
test('a retired timed-out offer cannot revive its persisted unknown grant',async()=>{const s=await setup({timeoutMs:30});try{s.intercept(p=>p.kind==='grant'?null:p);await assert.rejects(s.driver.ops.runActivityWithOwner(s.a,'treat-wounds'));await new Promise(r=>setTimeout(r,40));const offer=s.packets.find(p=>p.packet.kind==='offer').packet,permit=(await s.ledger.getActivity('A')).executor;s.intercept(p=>p);for(const tab of [s.one,s.two])for(const fn of tab.listeners)fn(structuredClone(offer),'G');await tick();assert.equal(s.calls,0);assert.equal(s.packets.filter(p=>p.packet.kind==='grant').length,1);assert.deepEqual((await s.ledger.getActivity('A')).executor,permit)}finally{s.dispose()}});
test('forged cancel sender or permit cannot retire a legitimate private context',async()=>{for(const change of ['sender','permit']){const s=await setup();try{s.intercept(p=>{if(p.kind==='continuation'){const cancel={...p,kind:'cancel',receiverUserId:'O',proof:structuredClone(p.proof)};if(change==='permit')cancel.proof.permit.attemptNonce='wrong';for(const tab of [s.one,s.two])for(const fn of tab.listeners)fn(cancel,change==='sender'?'N':'G')}return p});assert.equal((await s.driver.ops.runActivityWithOwner(s.a,'treat-wounds')).status,'confirmed');assert.equal(s.calls,1)}finally{s.dispose()}}});
test('owner offer capacity counts active attempts and is available after timeout',async t=>{
 t.mock.timers.enable({apis:['setTimeout']});const gm={id:'G',isGM:true,active:true},owner={id:'O',active:true},users=new Map([['G',gm],['O',owner]]);users.activeGM=gm;const listeners=new Set();let claims=0;
 const game={user:owner,users,socket:{on:(event,fn)=>listeners.add(fn),off:(event,fn)=>listeners.delete(fn),emit:()=>claims++}},ops=createExplorationOwnerOperations({game,ledger:{atomic:true},fromUuid:async()=>({testUserPermission:()=>true}),runtimeIdentity:()=>({userId:'O',clientNonce:'owner'}),timeoutMs:80});ops.register();
 const packet=index=>({protocol:OWNER_TRANSPORT_PROTOCOL,kind:'offer',requestId:`offer-request-${index}`,receiverUserId:'O',rootUUID:'JournalEntry.ROOT000000000001',epoch:'epoch',sessionId:`session-${index}`,activityId:`activity-${index}`,operationId:'treat-wounds',actorUUID:'Actor.H',driverUserId:'G',ownerUserId:'O',offerId:`offer-${index}`});
 try{for(let index=0;index<129;index++)for(const fn of listeners)fn(packet(index),'G');await tick();assert.equal(claims,128);t.mock.timers.tick(81);await tick();for(const fn of listeners)fn(packet(129),'G');await tick();assert.equal(claims,129);ops.invalidate('disconnect');await tick();for(const fn of listeners)fn(packet(130),'G');await tick();assert.equal(claims,130)}finally{ops.dispose()}
});
test('an expired offer waiting for actor resolution cannot rebuild its old pending claim',async t=>{
 t.mock.timers.enable({apis:['setTimeout']});const gm={id:'G',isGM:true,active:true},owner={id:'O',active:true},users=new Map([['G',gm],['O',owner]]);users.activeGM=gm;const listeners=new Set(),actor={testUserPermission:()=>true},gate=deferred();let delayed=true,claims=0;
 const game={user:owner,users,socket:{on:(event,fn)=>listeners.add(fn),off:(event,fn)=>listeners.delete(fn),emit:()=>claims++}},ops=createExplorationOwnerOperations({game,ledger:{atomic:true},fromUuid:async()=>delayed?gate.promise:actor,runtimeIdentity:()=>({userId:'O',clientNonce:'owner'}),timeoutMs:80});ops.register();
 const packet=index=>({protocol:OWNER_TRANSPORT_PROTOCOL,kind:'offer',requestId:`offer-request-${index}`,receiverUserId:'O',rootUUID:'JournalEntry.ROOT000000000001',epoch:'epoch',sessionId:`session-${index}`,activityId:`activity-${index}`,operationId:'treat-wounds',actorUUID:'Actor.H',driverUserId:'G',ownerUserId:'O',offerId:`offer-${index}`});
 try{for(let index=0;index<128;index++)for(const fn of listeners)fn(packet(index),'G');await tick();t.mock.timers.tick(81);delayed=false;for(const fn of listeners)fn(packet(129),'G');await tick();assert.equal(claims,1);gate.resolve(actor);await tick();assert.equal(claims,1)}finally{gate.resolve(actor);ops.dispose()}
});
for(const owner of ['G','O'])test(`independent GM Stop invalidates a pending ${owner==='G'?'local':'remote'} continuation read`,async()=>{
 const s=await nativeBrokerFixture({owner,activity:{providerId:'treat-wounds',actorUUID:'Actor.H',patientUUIDs:['Actor.P'],hpPoolUUIDs:['Actor.P'],startedAt:0,endsAt:600,options:{}}}),entered=deferred(),release=deferred();let armed=false,calls=0;
 const peerLedger=s.f.client('peer'),peerOps=createExplorationOwnerOperations({game:s.driver.game,ledger:peerLedger,fromUuid:async uuid=>s.documents.get(uuid),runtimeIdentity:()=>({userId:'G',clientNonce:'peer'}),getDriverScope:()=>null});
 try{
  const original=s.ledger.getSession;
  s.setAfterUpdate(async(actor,data)=>{if(data['flags.pf2e-third-party-automation.explorationExecutions']?.B?.state==='started')armed=true});
  s.ledger.getSession=async id=>{const state=await original(id);if(armed){armed=false;entered.resolve();await release.promise}return state};
  s.receiver.ops.registerOperation('treat-wounds',async()=>{calls++;return {status:'blocked',reason:'probe-native-boundary'}});
  const running=s.driver.ops.runActivityWithOwner(s.activity,'treat-wounds').catch(error=>error);await entered.promise;
  const peer=createCoordinator({ledger:peerLedger,capabilities:{},providers:[],clock:{stop(){}},ownerOperations:peerOps,isAuthority:()=>true});
  await peer.stop('S');assert.equal((await peerLedger.getSession('S')).status,'paused');release.resolve();await running;
  console.log(JSON.stringify({case:'separate-peer-stop-during-continuation-read',owner,nativeCalls:calls,status:(await peerLedger.getSession('S')).status}));
  assert.equal(calls,0);
 }finally{release.resolve();peerOps.dispose();s.dispose()}
});

test('peer Stop retires a persisted claim whose grant has not been delivered',async()=>{
 const s=await nativeBrokerFixture({owner:'O',activity:{providerId:'treat-wounds',actorUUID:'Actor.H',patientUUIDs:['Actor.P'],hpPoolUUIDs:['Actor.P'],startedAt:0,endsAt:600,options:{}}}),entered=deferred(),release=deferred();let calls=0;
 const peerLedger=s.f.client('peer'),peerOps=createExplorationOwnerOperations({game:s.driver.game,ledger:peerLedger,fromUuid:async uuid=>s.documents.get(uuid),runtimeIdentity:()=>({userId:'G',clientNonce:'peer'}),getDriverScope:()=>null,timeoutMs:30});
 try{const claim=s.ledger.claimExecution;s.ledger.claimExecution=async(...args)=>{const permit=await claim(...args);entered.resolve();await release.promise;return permit};s.receiver.ops.registerOperation('treat-wounds',async()=>{calls++;return {status:'blocked',reason:'native-boundary'}});const running=s.driver.ops.runActivityWithOwner(s.activity,'treat-wounds').catch(e=>e);await entered.promise;const peer=createCoordinator({ledger:peerLedger,capabilities:{},providers:[],clock:{stop(){}},ownerOperations:peerOps,isAuthority:()=>true});await peer.stop('S');release.resolve();await running;await tick();assert.equal(calls,0);assert.equal((await peerLedger.getActivity('B')).executor.state,'granted')}finally{release.resolve();peerOps.dispose();s.dispose()}
});
test('Stop preserves an already-verified same-permit done result for explicit reconciliation',async()=>{
 const s=await setup(),entered=deferred(),release=deferred();try{const actor=s.actors.get('Actor.H'),update=actor.update;actor.update=async data=>{const value=await update(data);if(data[`flags.${MODULE_ID}.explorationExecutions`]?.A?.state==='done'){entered.resolve();await release.promise}return value};const running=s.driver.ops.runActivityWithOwner(s.a,'treat-wounds').catch(e=>e);await entered.promise;const coordinator=createCoordinator({ledger:s.ledger,capabilities:{},providers:[],clock:{stop(){}},ownerOperations:s.driver.ops,isAuthority:()=>true});await coordinator.stop('S');release.resolve();await running;await tick();assert.equal((await s.driver.ops.reconcile(await s.ledger.getActivity('A'))).status,'confirmed');assert.equal(s.calls,1);assert.equal(s.messages.size,3)}finally{release.resolve();s.dispose()}
});