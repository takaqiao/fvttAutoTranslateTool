import {MODULE_ID,clone} from './schema.mjs';
import {isActiveGM} from './document-store.mjs';
import {createOwnerTransport,OWNER_TRANSPORT_PROTOCOL} from './owner-transport.mjs';
import {commandDigest,projectCommand,nativeParameters,projectPermit,samePermit,projectResult,validateNativeResult,validateExtensionOriginal} from './owner-command.mjs';
import {canonicalJSON} from './revision-codec.mjs';
export function createExplorationOwnerOperations(options) {
  return options.ledger?.atomic===true?createAtomicOwnerOperations(options):createLegacyOwnerOperations(options);
}
function createLegacyOwnerOperations({game,fromUuid,ledger,sharedOwnerOperations,timeoutMs=60000}) {
  const contexts=new WeakMap(),operations=new Map(),inflightActors=new Set(),cancelled=new Set();let socket;
  const authority=caller=>{if(caller!==game.users.activeGM?.id||!game.users.get(caller)?.isGM||!game.users.get(caller)?.active)throw Error('active-gm-required');if(game.combat?.started)throw Error('encounter-started')};
  async function ownerExecute(payload,callerId){
    authority(callerId);const key=payload?.activity?.actorUUID;if(typeof key!=='string')throw Error('actor-required');
    if(inflightActors.has(key))throw Error('owner-activity-already-in-flight');inflightActors.add(key);
    try{return await executeClaim(payload,callerId)}finally{inflightActors.delete(key)}
  }
  async function executeClaim(payload,callerId) {
    authority(callerId);const handler=operations.get(payload?.operationId);if(!handler)throw Error('unknown-native-operation');
    const activity=clone(payload.activity),actor=await fromUuid(activity?.actorUUID);authority(callerId);
    if(!actor?.testUserPermission(game.user,'OWNER'))throw Error('original-owner-required');
    if(activity.state!=='completing'||game.time.worldTime<activity.endsAt||activity.endsAt<activity.startedAt)throw Error('invalid-activity-time');
    const records=clone(actor.flags?.[MODULE_ID]?.explorationExecutions??{});
    if(records[activity.id])throw Error('activity-already-executed');
    const identity={activityId:activity.id,operationId:payload.operationId,gmId:callerId,userId:game.user.id,actorUUID:actor.uuid};
    records[activity.id]={...identity,state:'started'};
    await actor.update({[`flags.${MODULE_ID}.explorationExecutions`]:records});authority(callerId);
    const ctx=Object.freeze({actor,validate:()=>{authority(callerId);if(cancelled.has(activity.id))throw Error('activity-stopped');if(game.time.worldTime!==activity.endsAt)throw Error('external-world-time-change');if(!actor.testUserPermission(game.user,'OWNER'))throw Error('owner-changed')}});contexts.set(ctx,activity.id);
    try{
      ctx.validate();const result=await handler(activity,ctx);ctx.validate();
      const current=clone(actor.flags?.[MODULE_ID]?.explorationExecutions??{});
      if(current[activity.id]?.state!=='started')throw Error('owner-claim-changed');current[activity.id]={...identity,state:'done',result:clone(result)};
      await actor.update({[`flags.${MODULE_ID}.explorationExecutions`]:current});authority(callerId);return result;
    }catch(error){
      const current=clone(actor.flags?.[MODULE_ID]?.explorationExecutions??{});current[activity.id]={...identity,state:'uncertain',reason:String(error.message),...error.proof?{proof:clone(error.proof)}:{}};
      if(game.user.id===identity.userId)await actor.update({[`flags.${MODULE_ID}.explorationExecutions`]:current}).catch(()=>{});throw error;
    }finally{contexts.delete(ctx)}
  }
  async function runActivityWithOwner(activity,operationId) {
    if(!isActiveGM(game))throw Error('active-gm-required');
    const saved=await ledger.getActivity(activity.id);if(JSON.stringify(saved)!==JSON.stringify(activity)||saved.state!=='completing')throw Error('activity-claim-changed');
    if(sharedOwnerOperations?.runExploration)return sharedOwnerOperations.runExploration(activity,operationId);
    const actor=await fromUuid(activity.actorUUID);if(!isActiveGM(game))throw Error('gm-changed');
    // An active GM is a native OWNER; shared HP forwarding must stay on a client
    // that owns the master. Remote execution is used only for explicit ownerId.
    const ownerId=activity.source?.ownerId??game.user.id;
    if(ownerId===game.user.id)return ownerExecute({activity,operationId},game.user.id);
    if(!actor?.testUserPermission(game.users.get(ownerId),'OWNER')||!game.users.get(ownerId)?.active||!socket)throw Error('original-owner-offline');
    let timer;try{
      const response=await Promise.race([socket.executeAsUser('exploration:execute',ownerId,{activity,operationId}),new Promise((_,reject)=>{timer=setTimeout(()=>reject(Error('owner-response-uncertain-no-retry')),timeoutMs)})]);
      if(!isActiveGM(game))throw Error('gm-changed');if(!response?.ok)throw Error(response?.error??'owner-response-uncertain-no-retry');return response.value;
    }finally{clearTimeout(timer)}
  }
  async function reconcile(activity){
    if(!isActiveGM(game))throw Error('active-gm-required');const actor=await fromUuid(activity.actorUUID),record=actor?.flags?.[MODULE_ID]?.explorationExecutions?.[activity.id],operation=activity.options?.extensionOf?'treatment-extension':activity.providerId;
    if(record?.state!=='done'||record.activityId!==activity.id||record.actorUUID!==activity.actorUUID||record.operationId!==operation||record.result?.status!=='confirmed')return {status:'uncertain',reason:'saved-owner-completion-unavailable'};
    const result=clone(record.result),proof=result.proof;if(!proof||proof.useId!==(activity.options?.extensionOf??activity.id))return {status:'uncertain',reason:'saved-owner-proof-mismatch'};
    for(const key of ['checkIds','resultIds'])if(!Array.isArray(proof[key])||proof[key].some(id=>!game.messages.get(id)))return {status:'uncertain',reason:'saved-native-message-unavailable'};
    if((proof.poolReceipts??[]).some(p=>!activity.hpPoolUUIDs.includes(p.actorUUID)||p.activityId&&p.activityId!==activity.id))return {status:'uncertain',reason:'saved-pool-proof-mismatch'};
    if(!isActiveGM(game))throw Error('active-gm-required');return result;
  }
  return {ownerExecute,runActivityWithOwner,reconcile,cancelActivity:activity=>cancelled.add(activity.id),isActivityContext:(ctx,id)=>contexts.get(ctx)===id&&!cancelled.has(id),
    createActivityContext:async activity=>{
      if(!isActiveGM(game))throw Error('active-gm-required');const stored=await ledger.getActivity(activity.id);
      if(stored?.state!=='planned'||JSON.stringify(stored)!==JSON.stringify(activity)||game.time.worldTime!==activity.startedAt)throw Error('activity-begin-claim-required');
      const ctx=Object.freeze({validate:()=>{if(!isActiveGM(game))throw Error('gm-changed')}});contexts.set(ctx,activity.id);return ctx;
    },
    registerOperation:(id,handler)=>{if(!/^[-a-z0-9]+$/.test(id)||operations.has(id)||typeof handler!=='function')throw Error('invalid-operation');operations.set(id,handler)},
    register:({socket:api})=>{if(socket)throw Error('duplicate-exploration-socket');socket=api;socket?.register('exploration:execute',async function(payload){try{return {ok:true,value:await ownerExecute(payload,this.socketdata?.userId)}}catch(e){return {ok:false,error:String(e.message)}}})}
  };
}

function createAtomicOwnerOperations({game,fromUuid,ledger,getHpPool,getDriverScope=()=>null,runtimeIdentity,timeoutMs=60000}){
 if(typeof runtimeIdentity!=='function')throw Error('runtime-identity-required');
 const contexts=new WeakMap(),beginnings=new Map(),activeScopes=new Set(),operations=new Map(),offers=new Map(),ownerAttempts=new Map(),cancelled=new Set();let identity=clone(runtimeIdentity()),transport,disposed=false,generation=0,registered=false;
 const nonce=()=>crypto.randomUUID();
 const liveRuntime=(expected=generation)=>{if(disposed||expected!==generation||JSON.stringify(runtimeIdentity())!==JSON.stringify(identity)||game.user.id!==identity.userId)throw Error('owner-runtime-invalidated')};
 const live=(expected=generation)=>{liveRuntime(expected);if(game.combat?.started)throw Error('encounter-started')};
 const gm=id=>id===game.users.activeGM?.id&&game.users.get(id)?.isGM&&game.users.get(id)?.active;
 function driver(sessionId,leaseNonce){live();const scope=getDriverScope(sessionId);if(!gm(identity.userId)||!scope||scope.leaseNonce!==leaseNonce)throw Error('session-driver-required')}
 const route=packet=>({protocol:OWNER_TRANSPORT_PROTOCOL,rootUUID:packet.rootUUID,epoch:packet.epoch,sessionId:packet.sessionId,activityId:packet.activityId,operationId:packet.operationId,actorUUID:packet.actorUUID,driverUserId:packet.driverUserId,ownerUserId:packet.ownerUserId,offerId:packet.offerId});
 const actorRecord=actor=>clone(actor.flags?.[MODULE_ID]?.explorationExecutions??{});
 function validOwner(actor,permit){live();if(game.users.get(identity.userId)!==game.user||!game.user.active||!gm(permit.driverUserId)||!actor?.testUserPermission(game.user,'OWNER'))throw Error('original-owner-required');if(cancelled.has(permit.activityId))throw Error('activity-stopped')}
 function createScope(data){
  const scope={...data,controller:new AbortController()};
  scope.ctx=Object.freeze({
   get actor(){return scope.actor},
   validate:()=>validateScope(scope),
   get extensionOriginal(){return scope.extensionOriginal},
   get executionSignal(){return scope.controller.signal},
   get nativeDialogMode(){return scope.permit&&scope.permit.ownerUserId!==scope.driverUserId?'owner-preference':'automatic'}
  });
  contexts.set(scope.ctx,scope);activeScopes.add(scope);return scope;
 }
 function releaseScope(scope){contexts.delete(scope.ctx);activeScopes.delete(scope)}
 function abortScope(scope,reason){
  if(!scope)return;
  scope.controller.abort(reason instanceof Error?reason:Error(reason??'activity-stopped'));
  releaseScope(scope);
 }
 function retireAttempt(attempt,reason){
  if(!attempt)return;
  attempt.active=false;attempt.cancelled=true;
  attempt.controller.abort(reason instanceof Error?reason:Error(reason??'activity-stopped'));
  abortScope(attempt.scope,reason);
 }
 function finish(offer,error,value){
  if(!offer.active)return;offer.active=false;
  if(error)retireAttempt(offer.localAttempt,error);
  const beginning=beginnings.get(offer.activity.id);
  if(beginning){error?abortScope(beginning,error):releaseScope(beginning);beginnings.delete(offer.activity.id)}
  clearTimeout(offer.timer);offers.delete(offer.offerId);error?offer.reject(error):offer.resolve(value);
 }
 function validateOffer(offer){live(offer.generation);driver(offer.activity.sessionId,offer.leaseNonce);if(cancelled.has(offer.activity.id))throw Error('activity-stopped');if(!offer.active||performance.now()>=offer.deadline)throw Error('owner-response-uncertain-no-retry');if(game.time.worldTime!==offer.activity.endsAt)throw Error('external-world-time-change')}
 async function claim(packet,sender){
  const offer=offers.get(packet.offerId);if(!offer)return;
  try{validateOffer(offer)}catch{return}
  if(sender!==offer.ownerId||packet.ownerUserId!==sender||packet.driverUserId!==identity.userId||Object.entries(route(offer.packet)).some(([key,value])=>packet[key]!==value)||!packet.ownerClientNonce||!packet.attemptNonce||offer.requests.has(packet.requestId)||offer.requests.size>=128)return;
  offer.requests.add(packet.requestId);
  try{
   const actor=await fromUuid(offer.activity.actorUUID);validateOffer(offer);if(!actor?.testUserPermission(game.users.get(sender),'OWNER')||!game.users.get(sender)?.active)throw Error('original-owner-required');
   const permit=await ledger.claimExecution(offer.activity.id,{leaseNonce:offer.leaseNonce,operationId:offer.operationId,ownerUserId:sender,ownerClientNonce:packet.ownerClientNonce,attemptNonce:packet.attemptNonce,permitNonce:nonce(),offerId:offer.offerId,requestId:packet.requestId,commandDigest:offer.commandDigest});
   validateOffer(offer);if(permit.ownerClientNonce!==packet.ownerClientNonce||permit.attemptNonce!==packet.attemptNonce||permit.requestId!==packet.requestId||permit.commandDigest!==offer.commandDigest)throw Error('native-permit-mismatch');
   offer.permit=projectPermit(permit);
   const response={...route(packet),kind:'grant',requestId:packet.requestId,receiverUserId:sender,ownerClientNonce:packet.ownerClientNonce,attemptNonce:packet.attemptNonce,permitNonce:permit.permitNonce,commandDigest:permit.commandDigest,command:{...clone(offer.command),permit:projectPermit(permit)}};
   // A successful commit is issued only once. A lost relay acknowledgement is unknown.
   if(offer.localResolve)offer.localResolve(response);else transport.send(response);
  }catch(error){
   if(offer.localReject)offer.localReject(error);
   else if(offer.active){try{transport.send({...route(packet),kind:'denied',requestId:packet.requestId,receiverUserId:sender,ownerClientNonce:packet.ownerClientNonce,attemptNonce:packet.attemptNonce,errorCode:'execution-unavailable'})}catch{}}
  }
 }
 function scopeFor(activity,actor,permit,driverUserId,attempt){
  let scope=beginnings.get(activity.id);if(scope&&permit.ownerUserId!==identity.userId)scope=null;
  if(!scope)scope=createScope({activity,actor,permit,driverUserId,generation});
  scope.activity=activity;scope.actor=actor;scope.permit=permit;scope.driverUserId=driverUserId;scope.attempt=attempt;attempt.scope=scope;
  if(attempt.controller.signal.aborted)abortScope(scope,attempt.controller.signal.reason);
  beginnings.delete(activity.id);return scope;
 }
 function currentPool(actor){
  if(typeof getHpPool==='function')return getHpPool(actor);
  // Native-only callers need no pool service. An enabled or unreadable shared
  // HP setting cannot establish that the patient's own actor is its HP domain.
  if(game.modules?.get('pf2e-toolbelt')?.active===true){
   let disabled=false;try{disabled=game.settings?.get?.('pf2e-toolbelt','shareData.enabled')===false}catch{}
   if(!disabled)throw Error('original-patient-pool-unavailable');
  }
  return {ready:true,poolUUID:actor.uuid};
 }
 function validateReceivingPools(scope){
  const actual=new Set(),expected=new Set(scope.activity.hpPoolUUIDs);
  for(const uuid of scope.activity.patientUUIDs){
   const patient=scope.receiving?.find(actor=>actor.uuid===uuid);
   if(!patient)throw Error('original-patient-pool-unavailable');
   const pool=currentPool(patient);
   if(pool?.ready!==true||typeof pool.poolUUID!=='string')throw Error('original-patient-pool-unavailable');
   actual.add(pool.poolUUID);
  }
  if(actual.size!==expected.size||[...actual].some(uuid=>!expected.has(uuid)))throw Error('original-patient-pool-changed');
 }
 function validateScope(scope){
  live(scope.generation);if(scope.controller.signal.aborted||cancelled.has(scope.activity.id)||scope.attempt?.cancelled)throw Error('activity-stopped');
  if(scope.permit){validOwner(scope.actor,{...scope.permit,driverUserId:scope.driverUserId});if(scope.receiving?.some(actor=>!actor.testUserPermission(game.user,'OWNER')))throw Error('original-patient-owner-required');validateReceivingPools(scope);if(game.time.worldTime!==scope.activity.endsAt)throw Error('external-world-time-change')}
  else driver(scope.activity.sessionId,scope.leaseNonce);
 }
 async function executeGranted(response,attempt){
  live(attempt.generation);if(!attempt.active||response.kind!=='grant'||response.ownerClientNonce!==identity.clientNonce||response.attemptNonce!==attempt.attemptNonce||response.requestId!==attempt.requestId)throw Error('private-owner-attempt-required');
  attempt.active=false;
  const {permit,...command}=response.command??{};
  if(!permit||permit.ownerUserId!==identity.userId||permit.ownerClientNonce!==identity.clientNonce||permit.attemptNonce!==attempt.attemptNonce||permit.requestId!==attempt.requestId||permit.offerId!==response.offerId||permit.protocol!==OWNER_TRANSPORT_PROTOCOL||permit.rootUUID!==response.rootUUID||permit.epoch!==response.epoch||permit.activityId!==response.activityId||permit.actorUUID!==response.actorUUID||permit.operationId!==response.operationId||permit.permitNonce!==response.permitNonce||permit.commandDigest!==response.commandDigest||await commandDigest(command)!==permit.commandDigest)throw Error('native-permit-mismatch');
  if(permit.sessionId!==response.sessionId||permit.ownerUserId!==response.ownerUserId||command.parameters?.actorUUID!==permit.actorUUID)throw Error('native-permit-mismatch');
  live(attempt.generation);const activity=nativeParameters(command,permit),actor=await fromUuid(activity.actorUUID);live(attempt.generation);validOwner(actor,response);
  const receiving=await Promise.all([...new Set([...activity.patientUUIDs,...activity.hpPoolUUIDs])].map(fromUuid));live(attempt.generation);
  if(receiving.some(document=>!document?.testUserPermission(game.user,'OWNER')))throw Error('original-patient-owner-required');
  if(activity.options.threePecks&&response.driverUserId!==identity.userId)throw Error('remote-three-pecks-reservation-unadapted');
  if(command.extensionOriginal)validateExtensionOriginal({game,original:command.extensionOriginal,actorUUID:activity.actorUUID,patientUUIDs:activity.patientUUIDs});
  const handler=operations.get(permit.operationId);if(!handler)throw Error('unknown-native-operation');
  const records=actorRecord(actor);if(records[activity.id])throw Error('saved-native-execution-unresolved');
  const scope=scopeFor(activity,actor,projectPermit(permit),response.driverUserId,attempt);scope.extensionOriginal=command.extensionOriginal;scope.receiving=receiving;attempt.permit=projectPermit(permit);
  const savedIdentity={...projectPermit(permit),driverUserId:response.driverUserId};records[activity.id]={...savedIdentity,state:'started'};
  try{
   validateScope(scope);await actor.update({[`flags.${MODULE_ID}.explorationExecutions`]:records});validateScope(scope);
   // An awaited document write may outlive Stop. This ACK continues the same permit;
   // it never reserves another executor or revives a timed-out private attempt.
   if(response.driverUserId!==identity.userId){
    const continued=await transport.request({...route(response),kind:'continuation',requestId:nonce(),receiverUserId:response.driverUserId,ownerClientNonce:permit.ownerClientNonce,attemptNonce:permit.attemptNonce,permitNonce:permit.permitNonce,commandDigest:permit.commandDigest,proof:{permit:projectPermit(permit)}},{expectedSenderId:response.driverUserId,matches:reply=>reply.kind==='denied'||reply.kind==='continuation-ack'&&samePermit(reply.proof?.permit,permit)});
    validateScope(scope);if(continued.kind!=='continuation-ack'||continued.status!=='accepted')throw Error('owner-continuation-unavailable');
   }else{const saved=await ledger.getActivity(activity.id),session=await ledger.getSession(activity.sessionId);validateScope(scope);if(saved?.state!=='completing'||!samePermit(saved.executor,permit)||session?.status!=='running')throw Error('owner-continuation-unavailable')}
   const result=projectResult(await handler(activity,scope.ctx));validateScope(scope);
   await validateNativeResult({game,fromUuid,activity,permit,result,extensionOriginal:command.extensionOriginal});validateScope(scope);
   scope.verifiedResult=result;
   const current=actorRecord(actor);if(current[activity.id]?.state!=='started'||!samePermit(current[activity.id],permit))throw Error('owner-claim-changed');
   current[activity.id]={...savedIdentity,state:'done',result};await actor.update({[`flags.${MODULE_ID}.explorationExecutions`]:current});validateScope(scope);
   return {permit:projectPermit(permit),result,activity};
  }catch(error){
   const current=actorRecord(actor),record=current[activity.id];
   // Stop can retire the continuation while an already-verified done write is
   // awaiting its reply. Keep that evidence for an explicit source recheck.
   const done=record?.state==='done'&&scope.verifiedResult&&canonicalJSON(record.result)===canonicalJSON(scope.verifiedResult);
   if(samePermit(record,permit)&&!done){current[activity.id]={...savedIdentity,state:'uncertain',reason:String(error.message),...error.proof?{proof:clone(error.proof)}:{}};await actor.update({[`flags.${MODULE_ID}.explorationExecutions`]:current}).catch(()=>{})}throw error;
  }finally{releaseScope(scope)}
 }
 async function acceptOffer(packet,sender){
  try{live()}catch{return}
  const acceptedGeneration=generation;
  if(!gm(sender)||sender!==packet.driverUserId||packet.ownerUserId!==identity.userId||ownerAttempts.has(packet.offerId)||ownerAttempts.size>=128)return;
  const attempt={active:true,controller:new AbortController(),attemptNonce:nonce(),requestId:nonce(),generation:acceptedGeneration,packet:route(packet)};ownerAttempts.set(packet.offerId,attempt);
  attempt.timer=setTimeout(()=>{retireAttempt(attempt,'owner-response-uncertain-no-retry');if(ownerAttempts.get(packet.offerId)===attempt)ownerAttempts.delete(packet.offerId)},timeoutMs);
  try{
   const actor=await fromUuid(packet.actorUUID);live(acceptedGeneration);if(attempt.cancelled||!actor?.testUserPermission(game.user,'OWNER')||!game.user.active)return;
   const response=await transport.request({...route(packet),kind:'claim',requestId:attempt.requestId,receiverUserId:sender,ownerClientNonce:identity.clientNonce,attemptNonce:attempt.attemptNonce},{expectedSenderId:sender,matches:reply=>reply.kind==='denied'||reply.kind==='grant'&&reply.command?.permit?.ownerClientNonce===identity.clientNonce});
   if(response.kind!=='grant'){attempt.active=false;return}
   const execution=await executeGranted(response,attempt);live();
   const completionId=nonce();await transport.request({...route(packet),kind:'completion',requestId:nonce(),receiverUserId:sender,ownerClientNonce:identity.clientNonce,attemptNonce:attempt.attemptNonce,permitNonce:execution.permit.permitNonce,commandDigest:execution.permit.commandDigest,completionId,status:execution.result.status,proof:{permit:execution.permit}},{expectedSenderId:sender,matches:reply=>reply.kind==='completion-ack'});
  }catch(error){retireAttempt(attempt,error)}finally{clearTimeout(attempt.timer);if(ownerAttempts.get(packet.offerId)===attempt)ownerAttempts.delete(packet.offerId)}
 }
 async function continueExecution(packet,sender){
  const offer=offers.get(packet.offerId);if(!offer?.permit)return;
  try{validateOffer(offer)}catch{return}
  if(sender!==offer.ownerId||Object.entries(route(offer.packet)).some(([key,value])=>packet[key]!==value)||!samePermit(packet.proof?.permit,offer.permit)||packet.ownerClientNonce!==offer.permit.ownerClientNonce||packet.attemptNonce!==offer.permit.attemptNonce||packet.permitNonce!==offer.permit.permitNonce||packet.commandDigest!==offer.commandDigest)return;
  const [saved,session]=await Promise.all([ledger.getActivity(offer.activity.id),ledger.getSession(offer.activity.sessionId)]);validateOffer(offer);
  if(saved?.state!=='completing'||!samePermit(saved.executor,offer.permit)||session?.status!=='running')return;
  transport.send({...route(packet),kind:'continuation-ack',requestId:packet.requestId,receiverUserId:sender,ownerClientNonce:packet.ownerClientNonce,attemptNonce:packet.attemptNonce,permitNonce:packet.permitNonce,commandDigest:packet.commandDigest,status:'accepted',proof:{permit:offer.permit}});
 }
 function cancelExecution(packet,sender){
  const attempt=ownerAttempts.get(packet.offerId),offer=offers.get(packet.offerId),permit=packet.proof?.permit;if(!gm(sender)||!permit)return;
  let retired=false;
  const pendingPermit=offer&&!offer.permit&&offer.requests.has(permit.requestId)&&permit.leaseNonce===offer.leaseNonce&&permit.commandDigest===offer.commandDigest&&Object.entries(route(offer.packet)).filter(([key])=>key!=='driverUserId').every(([key,value])=>permit[key]===value);
  if(offer&&(samePermit(offer.permit,permit)||pendingPermit)&&Object.entries(route(offer.packet)).every(([key,value])=>packet[key]===value)){cancelled.add(offer.activity.id);offer.localReject?.(Error('activity-stopped'));finish(offer,Error('activity-stopped'));retired=true}
  if(attempt&&Object.entries(attempt.packet).every(([key,value])=>packet[key]===value)&&permit.ownerClientNonce===identity.clientNonce&&permit.attemptNonce===attempt.attemptNonce&&permit.requestId===attempt.requestId&&permit.ownerUserId===identity.userId&&permit.offerId===packet.offerId&&(!attempt.permit||samePermit(attempt.permit,permit))){retireAttempt(attempt,'activity-stopped');retired=true}
  if(retired)transport?.send({...route(packet),kind:'cancel-ack',requestId:packet.requestId,receiverUserId:sender,ownerClientNonce:permit.ownerClientNonce,attemptNonce:permit.attemptNonce,permitNonce:permit.permitNonce,commandDigest:permit.commandDigest,status:'retired',proof:{permit}});
 }
 async function cancelActivity(activity){
  if(!gm(identity.userId))throw Error('active-gm-required');cancelled.add(activity.id);abortScope(beginnings.get(activity.id),'activity-stopped');beginnings.delete(activity.id);
  const captured=generation,[saved,session]=await Promise.all([ledger.getActivity(activity.id),ledger.getSession(activity.sessionId)]);
  if(disposed||captured!==generation||game.user.id!==identity.userId||JSON.stringify(runtimeIdentity())!==JSON.stringify(identity))throw Error('owner-runtime-invalidated');if(!gm(identity.userId))throw Error('active-gm-required');
  const permit=saved?.executor,offer=[...offers.values()].find(row=>row.activity.id===activity.id);
  if(!permit){if(offer){offer.localReject?.(Error('activity-stopped'));finish(offer,Error('activity-stopped'))}return}
  if(session?.protocol?.rootUUID!==permit.rootUUID||session.protocol.epoch!==permit.epoch||session.driver?.leaseNonce!==permit.leaseNonce||permit.activityId!==activity.id||permit.sessionId!==activity.sessionId)throw Error('native-permit-mismatch');
  const packet=route({...permit,driverUserId:session.driver.userId});
  const temporary=!transport,relay=transport??createOwnerTransport({game,timeoutMs,onError:()=>{}});
  try{
   // Stop may be requested by a peer GM. Both the real driver and the owner must
   // retire their matching private continuation before Stop reports completion.
   for(const receiverUserId of new Set([session.driver.userId,permit.ownerUserId]))await relay.request({...route(packet),kind:'cancel',requestId:nonce(),receiverUserId,ownerClientNonce:permit.ownerClientNonce,attemptNonce:permit.attemptNonce,permitNonce:permit.permitNonce,commandDigest:permit.commandDigest,proof:{permit:projectPermit(permit)}},{expectedSenderId:receiverUserId,matches:reply=>reply.kind==='cancel-ack'&&reply.status==='retired'&&samePermit(reply.proof?.permit,permit)});
  }catch{throw Error('owner-stop-ack-unknown-no-retry')}finally{if(temporary)relay.dispose()}
 }
 async function savedResult(activity,permit,extensionOriginal){
  const actor=await fromUuid(activity.actorUUID),record=actor?.flags?.[MODULE_ID]?.explorationExecutions?.[activity.id],session=await ledger.getSession(activity.sessionId);
  if(record?.state!=='done'||!samePermit(record,permit)||record.driverUserId!==session?.driver?.userId||session.driver.leaseNonce!==permit.leaseNonce||session.protocol?.rootUUID!==permit.rootUUID||session.protocol?.epoch!==permit.epoch)throw Error('saved-owner-completion-unavailable');
  return validateNativeResult({game,fromUuid,activity,permit,result:projectResult(record.result),extensionOriginal});
 }
 async function complete(packet,sender){
  const offer=offers.get(packet.offerId);if(!offer||!offer.permit)return;
  try{driver(offer.activity.sessionId,offer.leaseNonce)}catch{return}
  if(sender!==offer.ownerId||packet.ownerUserId!==sender||!samePermit(packet.proof?.permit,offer.permit)||packet.ownerClientNonce!==offer.permit.ownerClientNonce||packet.attemptNonce!==offer.permit.attemptNonce||packet.permitNonce!==offer.permit.permitNonce||packet.commandDigest!==offer.commandDigest||offer.completionId)return;
  offer.completionId=packet.completionId;
  try{const result=await savedResult(offer.activity,offer.permit,offer.command.extensionOriginal);driver(offer.activity.sessionId,offer.leaseNonce);await ledger.recordExecutionResult(offer.activity.id,{permit:offer.permit,result});driver(offer.activity.sessionId,offer.leaseNonce);
   transport.send({...route(packet),kind:'completion-ack',requestId:packet.requestId,receiverUserId:sender,ownerClientNonce:packet.ownerClientNonce,attemptNonce:packet.attemptNonce,permitNonce:packet.permitNonce,commandDigest:packet.commandDigest,completionId:packet.completionId,status:'accepted'});finish(offer,null,result);
  }catch(error){finish(offer,error)}
 }
 function onPacket(packet,sender){if(packet.kind==='offer')return acceptOffer(packet,sender);if(packet.kind==='cancel')return cancelExecution(packet,sender);if(packet.kind==='continuation')return continueExecution(packet,sender);if(packet.kind==='claim'){const offer=offers.get(packet.offerId);if(!offer)return;try{driver(offer.activity.sessionId,offer.leaseNonce)}catch{return}return claim(packet,sender)}if(packet.kind==='completion')return complete(packet,sender)}
 async function runActivityWithOwner(activity,operationId){
  live();const runGeneration=generation,scope=getDriverScope(activity.sessionId),check=()=>{live(runGeneration);driver(activity.sessionId,scope?.leaseNonce)};check();
  const [saved,session]=await Promise.all([ledger.getActivity(activity.id),ledger.getSession(activity.sessionId)]);check();
  if(canonicalJSON(saved)!==canonicalJSON(activity)||saved.state!=='completing'||saved.executor||!ledger.ownsSession(session,scope)||session.status!=='running')throw Error('activity-claim-changed');
  const ownerId=activity.source?.ownerId??identity.userId,actor=await fromUuid(activity.actorUUID);check();
  if(!actor?.testUserPermission(game.users.get(ownerId),'OWNER')||!game.users.get(ownerId)?.active)throw Error('original-owner-offline');
  if(activity.options?.threePecks&&(ownerId!==identity.userId||!beginnings.has(activity.id)))throw Error('three-pecks-private-begin-reservation-required');
  if(!transport&&ownerId!==identity.userId)throw Error('owner-transport-unavailable');if(offers.size>=128)throw Error('owner-offer-capacity-limit');
  const original=activity.options?.extensionOf?await ledger.getActivity(activity.options.extensionOf):null,command=projectCommand(activity,operationId,original),digest=await commandDigest(command);check();
  const offerId=nonce(),packet={protocol:OWNER_TRANSPORT_PROTOCOL,rootUUID:session.protocol.rootUUID,epoch:session.protocol.epoch,sessionId:activity.sessionId,activityId:activity.id,operationId,actorUUID:activity.actorUUID,driverUserId:identity.userId,ownerUserId:ownerId,offerId,kind:'offer',requestId:nonce(),receiverUserId:ownerId};
  return new Promise((resolve,reject)=>{
   const offer={offerId,generation:runGeneration,activity:clone(activity),operationId,ownerId,leaseNonce:scope.leaseNonce,packet,command,commandDigest:digest,resolve,reject,requests:new Set(),active:true,deadline:performance.now()+timeoutMs};offers.set(offerId,offer);offer.timer=setTimeout(()=>finish(offer,Error('owner-response-uncertain-no-retry')),timeoutMs);
   if(ownerId===identity.userId){const attempt={active:true,controller:new AbortController(),attemptNonce:nonce(),requestId:nonce(),generation:runGeneration};offer.localAttempt=attempt;const granted=new Promise((r,j)=>{offer.localResolve=r;offer.localReject=j});granted.then(response=>executeGranted(response,attempt)).then(async execution=>{validateOffer(offer);const result=await savedResult(activity,execution.permit,command.extensionOriginal);validateOffer(offer);await ledger.recordExecutionResult(activity.id,{permit:execution.permit,result});validateOffer(offer);finish(offer,null,result)}).catch(error=>finish(offer,error));claim({...route(packet),kind:'claim',requestId:attempt.requestId,receiverUserId:identity.userId,ownerClientNonce:identity.clientNonce,attemptNonce:attempt.attemptNonce},identity.userId);
   }else try{transport.send(packet)}catch(error){finish(offer,error)}
  });
 }
 const connect=()=>{transport=createOwnerTransport({game,onPacket,timeoutMs,onError:()=>{}})};
 const invalidate=reason=>{generation++;transport?.dispose();transport=undefined;for(const offer of [...offers.values()]){offer.localReject?.(Error(reason??'owner-runtime-invalidated'));finish(offer,Error(reason??'owner-runtime-invalidated'))}for(const scope of [...activeScopes])abortScope(scope,reason??'owner-runtime-invalidated');beginnings.clear();for(const attempt of ownerAttempts.values()){retireAttempt(attempt,reason??'owner-runtime-invalidated');clearTimeout(attempt.timer)}ownerAttempts.clear();identity=clone(runtimeIdentity());if(registered&&!disposed)connect()};
 return {
  runActivityWithOwner,ownerExecute:async()=>{throw Error('private-atomic-owner-attempt-required')},
  reconcile:async activity=>{live();if(!isActiveGM(game))throw Error('active-gm-required');try{const saved=await ledger.getActivity(activity.id);if(!saved?.executor||!samePermit(saved.executor,activity.executor))throw Error('native-permit-mismatch');const original=saved.options?.extensionOf?await ledger.getActivity(saved.options.extensionOf):null;const result=await savedResult(saved,saved.executor,original);await ledger.recordExecutionResult(saved.id,{permit:saved.executor,result});return result}catch(error){return {status:'uncertain',reason:String(error.message)}}},
  cancelActivity,
  isActivityContext:(ctx,id)=>{const scope=contexts.get(ctx);if(!scope||scope.activity.id!==id)return false;try{validateScope(scope);return true}catch{return false}},
  isExecutionContext:(ctx,id)=>{const scope=contexts.get(ctx);if(!scope?.permit||scope.activity.id!==id)return false;try{validateScope(scope);return true}catch{return false}},
  createActivityContext:async activity=>{const captured=generation,leaseNonce=getDriverScope(activity.sessionId)?.leaseNonce;driver(activity.sessionId,leaseNonce);const saved=await ledger.getActivity(activity.id);live(captured);driver(activity.sessionId,leaseNonce);if(saved?.state!=='planned'||canonicalJSON(saved)!==canonicalJSON(activity)||game.time.worldTime!==activity.startedAt)throw Error('activity-begin-claim-required');abortScope(beginnings.get(activity.id),'activity-context-replaced');const scope=createScope({activity:clone(activity),leaseNonce,generation:captured});beginnings.set(activity.id,scope);return scope.ctx},
  registerOperation:(id,handler)=>{if(!/^[-a-z0-9]+$/.test(id)||operations.has(id)||typeof handler!=='function')throw Error('invalid-operation');operations.set(id,handler)},
  register:()=>{liveRuntime();if(registered)throw Error('duplicate-exploration-socket');registered=true;connect()},invalidate,
  dispose:()=>{if(disposed)return;disposed=true;invalidate('owner-operations-disposed')}
 };
}
