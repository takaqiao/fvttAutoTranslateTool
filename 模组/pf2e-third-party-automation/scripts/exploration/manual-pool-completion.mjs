import {manualPoolRequest,MANUAL_POOL_OPERATION} from './schema.mjs';
import {canonicalJSON} from './revision-codec.mjs';
import {createOwnerTransport,OWNER_TRANSPORT_PROTOCOL} from './owner-transport.mjs';
import {manualHpBaseline,assertManualHpBaseline} from './hp-pool.mjs';

const copy=value=>JSON.parse(canonicalJSON(value));
const same=(a,b)=>canonicalJSON(a)===canonicalJSON(b);
const author=message=>message?.author?.id??message?.author??message?.user?.id??message?.user;
const packetFields=['protocol','operationId','kind','status','requestId','receiverUserId','driverUserId','ownerUserId','ownerClientNonce','attemptNonce','actorUUID','sessionId','activityId','command'];

/** Internal source admission and native Promise correlation. The application
 * method consumes an exact private grant; lookup never returns a permit. */
export function createManualPoolCompletion({game,ledger,fromUuid,hpPools,resolveSource,isIssuer,clientNonce,onSettled=()=>{},getProvider=()=>game.modules?.get('pf2e-toolbelt')?.api?.explorationManualPool,timeoutMs=10000,onError=()=>{}}){
 if(typeof clientNonce!=='string'||!clientNonce||typeof isIssuer!=='function')throw Error('manual-pool-runtime-required');
 let transport,started=false,disposeProvider;const submitted=new Set(),grants=new WeakMap(),applications=new Map(),userId=game.user.id;
 const activeGM=()=>game.users?.activeGM;
 const gm=id=>activeGM()?.id===id&&game.users.get(id)?.active===true&&game.users.get(id)?.isGM===true;
 const current=()=>started&&game.user.id===userId;
 const issuer=()=>current()&&gm(userId)&&isIssuer()===true;
 function route(request){return {protocol:OWNER_TRANSPORT_PROTOCOL,operationId:MANUAL_POOL_OPERATION,driverUserId:activeGM()?.id,ownerUserId:userId,
  ownerClientNonce:clientNonce,attemptNonce:request.attemptNonce,actorUUID:request.actorUUID,sessionId:request.sessionId,activityId:request.activityId}}
 async function qualify(request,senderId,{owner=false}={}){
  const gmId=activeGM()?.id,live=()=>owner?current()&&senderId===userId&&gm(gmId):issuer();
  if(!live()||typeof resolveSource!=='function'||typeof fromUuid!=='function'||!hpPools)throw Error('manual-pool-source-unavailable');
  const user=game.users.get(senderId),source=await resolveSource(copy(request),{callerId:senderId,owner});
  if(!source||typeof source.isCurrent!=='function')throw Error('manual-pool-source-unavailable');
  const binding=copy(source.binding),expected={...request};delete expected.ownerClientNonce;delete expected.attemptNonce;
  if(Object.keys(binding).length!==Object.keys(expected).length+3||Object.entries(expected).some(([key,value])=>!same(binding[key],value))||!request.patientUUIDs.includes(binding.selectedPatientUUID)||typeof binding.effectId!=='string'||!binding.effectId||!Number.isFinite(binding.worldTime))throw Error('manual-pool-source-mismatch');
  const healer=await fromUuid(request.actorUUID),patients=await Promise.all(request.patientUUIDs.map(uuid=>fromUuid(uuid))),master=await fromUuid(request.poolUUID);
  const check=game.messages.get(request.checkId),result=game.messages.get(request.resultId);
  if(!check||!result||typeof check.toObject!=='function'||typeof result.toObject!=='function')throw Error('manual-pool-source-unavailable');
  const documents=canonicalJSON([check.toObject(true),result.toObject(true)]);
  const validate=()=>{
   if(!live()||!user?.active||game.users.get(senderId)!==user||game.time.worldTime!==binding.worldTime||source.isCurrent()!==true||!same(source.binding,binding))return false;
   if(healer?.uuid!==request.actorUUID||healer.testUserPermission?.(user,'OWNER')!==true||patients.some((patient,i)=>patient?.uuid!==request.patientUUIDs[i]||patient.testUserPermission?.(user,'OWNER')!==true))return false;
   if(master?.uuid!==request.poolUUID||master.testUserPermission?.(user,'OWNER')!==true&&master.testUserPermission?.(game.users.get(gmId),'OWNER')!==true)return false;
   if(patients.some(patient=>{const pool=hpPools.discover(patient);return !pool?.ready||pool.poolUUID!==request.poolUUID||!pool.memberUUIDs?.includes(patient.uuid)}))return false;
   if(game.messages.get(request.checkId)!==check||game.messages.get(request.resultId)!==result||check.rolls?.[0]?._evaluated!==true||result.rolls?.[request.rollIndex]?._evaluated!==true||!Number.isFinite(result.rolls[request.rollIndex].total))return false;
   const sourceUser=game.users.get(author(check));
   return !!sourceUser&&author(result)===sourceUser.id&&healer.testUserPermission?.(sourceUser,'OWNER')===true&&check.speaker?.actor===healer.id&&result.speaker?.actor===healer.id&&canonicalJSON([check.toObject(true),result.toObject(true)])===documents;
  };
  if(!validate())throw Error('manual-pool-evidence-changed');
  const bytes=await crypto.subtle.digest('SHA-256',new TextEncoder().encode(canonicalJSON({binding,documents})));
  if(!validate())throw Error('manual-pool-evidence-changed');
  return {binding,validate,master,sourceDigest:Array.from(new Uint8Array(bytes),byte=>byte.toString(16).padStart(2,'0')).join('')};
 }
 function reply(packet,kind,status,proof){
  if(!issuer())return;
  transport.send({...packet,kind,status,receiverUserId:packet.ownerUserId,...proof?{proof}:{}});
 }
 function application(packet,senderId){
  const permit=packet.proof?.permit,scope=applications.get(permit?.applicationNonce);
  if(!scope||scope.permit.ownerUserId!==senderId||!same(scope.permit,permit)||!same(scope.request,packet.command)||!scope.source.validate())throw Error('manual-pool-application-unavailable');return scope;
 }
 function receiptFor(request,permit,proof){
  const receipt=game.messages.get(proof.receiptId),pf=receipt?.flags?.pf2e;
  return receipt&&author(receipt)===permit.ownerUserId&&receipt.speaker?.actor===permit.selectedPatientUUID.split('.').at(-1)&&pf?.context?.type==='damage-taken'
   &&pf.context.options?.includes(`pf2e-third-party-automation:source:${request.resultId}:${request.rollIndex}`)
   &&(proof.noChange?pf.appliedDamage===null:pf.appliedDamage?.uuid===permit.selectedPatientUUID&&pf.appliedDamage.isHealing===true&&!pf.appliedDamage.isReverted)?receipt:null;
 }
 function onProvider(event){
  if(!issuer())return;
  if(event.phase==='authorize')return (async()=>{
   const scope=applications.get(event.binding?.applicationNonce);
   if(!scope||!scope.fields||!same(event.fields,scope.fields)||scope.authorized||event.senderId!==scope.permit.ownerUserId||event.binding.permitNonce!==scope.permit.permitNonce||event.binding.poolUUID!==scope.permit.poolUUID||event.binding.patientUUID!==scope.permit.selectedPatientUUID||event.master.uuid!==scope.permit.poolUUID)throw Error('manual-pool-native-authorization-required');
   const session=await ledger.getSession(scope.request.sessionId);
   if(session?.status!=='recording'||!scope.source.validate()||scope.authorized||event.master!==scope.source.master)throw Error('manual-pool-evidence-changed');assertManualHpBaseline(event.master,scope.hpBaseline);scope.authorized=true;
   return {validate:()=>issuer()&&scope.source.validate(),beforeWrite:()=>{
    if(!issuer()||!scope.source.validate()||scope.writeStarted)throw Error('manual-pool-evidence-changed');
    assertManualHpBaseline(event.master,scope.hpBaseline);scope.writeStarted=true;return true;
   }};
  })();
  if(event.phase==='write'){
   const scope=applications.get(event.binding?.applicationNonce);
   if(!scope?.authorized||scope.masterObserved||event.master.uuid!==scope.permit.poolUUID||!same(event.fields,scope.fields))return;
   scope.masterObserved=true;event.terminalPromise.then(proof=>{if(!scope.source.validate())throw Error('manual-pool-evidence-changed');scope.resolveMaster(copy(proof))}).catch(scope.rejectMaster);
  }
 }
 async function onPacket(packet,senderId){
  if(packet.operationId!==MANUAL_POOL_OPERATION||!issuer()||packet.driverUserId!==userId)return;
  // Only the authenticated socket sender can request its own private tab's claim.
  const extra=['begin','forward','master','terminal'].includes(packet.status)?['proof']:[];
  if(packet.ownerUserId!==senderId||Object.keys(packet).length!==packetFields.length+extra.length||Object.keys(packet).some(key=>![...packetFields,...extra].includes(key)))return;
  try{
   const request=manualPoolRequest(packet.command);
   if(['sessionId','activityId','actorUUID','ownerClientNonce','attemptNonce'].some(key=>packet[key]!==request[key]))throw Error('manual-pool-request-mismatch');
   if(packet.kind==='claim'&&packet.status==='claim'){
    const source=await qualify(request,senderId);
    const grant=await ledger.claimManualPoolApplication({request,effectId:source.binding.effectId,selectedPatientUUID:source.binding.selectedPatientUUID,sourceDigest:source.sourceDigest,
     ownerUserId:senderId,permitNonce:crypto.randomUUID(),worldTime:source.binding.worldTime},{evidenceGuard:source.validate});
    if(!source.validate())throw Error('manual-pool-evidence-changed');
    reply(packet,'grant','granted',grant);
   }else if(packet.kind==='continuation'&&packet.status==='begin'){
    const source=await qualify(request,senderId);if(source.sourceDigest!==packet.proof?.permit?.sourceDigest)throw Error('manual-pool-source-mismatch');
    const hpBaseline=manualHpBaseline(source.master);
    const permit=await ledger.beginManualPoolApplication({request,permit:packet.proof.permit},{evidenceGuard:source.validate});
    assertManualHpBaseline(source.master,hpBaseline);
    let resolveMaster,rejectMaster;const masterPromise=new Promise((resolve,reject)=>{resolveMaster=resolve;rejectMaster=reject});masterPromise.catch(()=>{});
    applications.set(permit.applicationNonce,{request,permit,source,hpBaseline,masterPromise,resolveMaster,rejectMaster});
    if(!source.validate())throw Error('manual-pool-evidence-changed');reply(packet,'continuation-ack','applying',permit);
   }else if(packet.kind==='continuation'&&packet.status==='forward'){
    const scope=application(packet,senderId),fields=packet.proof.fields;
    if(scope.fields||!same(packet.proof.hpBaseline,scope.hpBaseline)||!fields||!Object.keys(fields).length||Object.entries(fields).some(([key,value])=>!['system.attributes.hp.value','system.attributes.hp.sp.value','system.attributes.hp.temp'].includes(key)||!Number.isFinite(value)||value<0))throw Error('manual-pool-forward-mismatch');
    const session=await ledger.getSession(request.sessionId);
    if(session?.status!=='recording'||!scope.source.validate()||scope.fields)throw Error('manual-pool-evidence-changed');assertManualHpBaseline(scope.source.master,scope.hpBaseline);scope.fields=copy(fields);
    reply(packet,'continuation-ack','forward-prepared',{fields:scope.fields,hpBaseline:scope.hpBaseline});
   }else if(packet.kind==='continuation'&&packet.status==='master'){
    const scope=application(packet,senderId),proof=await scope.masterPromise;
    if(!scope.source.validate())throw Error('manual-pool-evidence-changed');reply(packet,'continuation-ack','master-fulfilled',proof);
   }else if(packet.kind==='completion'&&packet.status==='terminal'){
    const scope=application(packet,senderId),proof=packet.proof,receipt=receiptFor(request,scope.permit,proof);
    if(!receipt||proof.noChange&&(scope.fields||scope.authorized||scope.masterObserved))throw Error('manual-pool-receipt-unavailable');
    if(!proof.noChange&&proof.master?.writerUserId!==senderId){const master=await scope.masterPromise;if(!same(master,proof.master))throw Error('manual-pool-terminal-mismatch')}
    const guard=()=>scope.source.validate()&&game.messages.get(proof.receiptId)===receipt&&receiptFor(request,scope.permit,proof)===receipt;
    await ledger.recordManualPoolTerminal({request,permit:scope.permit,receiptId:proof.receiptId,noChange:proof.noChange,master:proof.master},{evidenceGuard:guard});
    await onSettled(copy(request));
    reply(packet,'completion-ack','settled',{receiptId:proof.receiptId,noChange:proof.noChange});
   }else if(packet.kind==='continuation'&&packet.status==='lookup'){
    const source=await qualify(request,senderId),proof=await ledger.lookupManualPoolProof(request,senderId);
    if(!source.validate()||!proof||proof.sourceDigest!==source.sourceDigest)throw Error('manual-pool-proof-unavailable');
    reply(packet,'continuation-ack',proof.status,proof);
   }else throw Error('invalid-manual-pool-operation');
  }catch(error){onError(error);reply(packet,'denied','unavailable')}
 }
 async function send(input,lookup){
  const request=manualPoolRequest(input);
  if(!current()||request.ownerClientNonce!==clientNonce||!game.user.active||!gm(activeGM()?.id))throw Error('manual-pool-client-unavailable');
  const key=canonicalJSON(request);
  if(!lookup){if(submitted.has(key))throw Error('manual-pool-unknown-no-retry');submitted.add(key)}
  const gmId=activeGM().id,packet={...route(request),kind:lookup?'continuation':'claim',status:lookup?'lookup':'claim',requestId:crypto.randomUUID(),receiverUserId:gmId,command:request};
  const response=await transport.request(packet,{expectedSenderId:gmId,matches:reply=>{
   if(!same(reply.command,request))return false;
   if(reply.kind==='denied')return true;
   if(lookup)return reply.kind==='continuation-ack'&&['reserved','settled'].includes(reply.status)&&reply.proof?.status===reply.status&&Object.keys(reply.proof).sort().join(',')===(reply.status==='settled'?'effectKey,noChange,poolUUID,receiptId,sourceDigest,status':'effectKey,poolUUID,sourceDigest,status')&&reply.proof.poolUUID===request.poolUUID;
   const grant=reply.proof;
   return reply.kind==='grant'&&reply.status==='granted'&&grant?.state==='granted'&&grant.operationId===MANUAL_POOL_OPERATION&&grant.issuerUserId===gmId&&grant.ownerUserId===userId&&grant.ownerClientNonce===clientNonce&&grant.attemptNonce===request.attemptNonce&&grant.sessionId===request.sessionId&&grant.activityId===request.activityId&&grant.actorUUID===request.actorUUID&&grant.poolUUID===request.poolUUID&&same(grant.patientUUIDs,request.patientUUIDs)&&typeof grant.permitNonce==='string'&&!!grant.permitNonce;
  }});
  if(!current()||!game.user.active||!gm(gmId)||response.kind==='denied')throw Error('manual-pool-unavailable');
  const proof=copy(response.proof);if(!lookup)grants.set(proof,{request,original:copy(proof),used:false});return proof;
 }
 async function applicationRequest(request,status,proof,expectedStatus){
  const gmId=activeGM()?.id,packet={...route(request),kind:status==='terminal'?'completion':'continuation',status,requestId:crypto.randomUUID(),receiverUserId:gmId,command:request,proof};
  const response=await transport.request(packet,{expectedSenderId:gmId,matches:reply=>same(reply.command,request)&&(reply.kind==='denied'||reply.kind===(status==='terminal'?'completion-ack':'continuation-ack')&&reply.status===expectedStatus)});
  if(!current()||!gm(gmId)||response.kind==='denied')throw Error('manual-pool-application-unavailable');return copy(response.proof);
 }
 async function withApplication(grant,patient,operation,{updateActor=patient}={}){
  const local=grants.get(grant);if(!local||local.used||!same(local.original,grant)||typeof hpPools?.withManualApplication!=='function')throw Error('manual-pool-private-grant-required');local.used=true;
  const provider=getProvider();if(provider?.descriptor?.sourceSHA256!=='2946fa27eaf0963098f9f48ee48f777c5cf65166b56b043de9404f09813c240f'||provider.descriptor.hpBaselineGuardVersion!==1)throw Error('manual-pool-provider-unavailable');
  const source=await qualify(local.request,userId,{owner:true});
  const permit=await applicationRequest(local.request,'begin',{permit:grant},'applying');
  const {applicationNonce,state,...claimed}=permit;
  if(state!=='applying'||typeof applicationNonce!=='string'||!applicationNonce||!same({...claimed,state:'granted'},grant))throw Error('manual-pool-permit-mismatch');
  if(!source.validate())throw Error('manual-pool-evidence-changed');
  const result=await hpPools.withManualApplication({permit,provider,validate:source.validate,request:local.request,updateActor,
   prepareForward:async(fields,hpBaseline)=>{const ack=await applicationRequest(local.request,'forward',{permit,fields,hpBaseline},'forward-prepared');if(!same(ack,{fields,hpBaseline})||!source.validate())throw Error('manual-pool-forward-mismatch')},
   remoteCompletion:()=>applicationRequest(local.request,'master',{permit},'master-fulfilled')},patient,operation);
  if(!source.validate())throw Error('manual-pool-evidence-changed');const proof=result.poolReceipt;
  const expected={receiptId:proof.receiptId,noChange:proof.noChange};
  const ack=await applicationRequest(local.request,'terminal',{permit,...expected,master:proof.master??null},'settled');if(!same(ack,expected))throw Error('manual-pool-terminal-mismatch');return result;
 }
 return {
  start(){if(started)return;started=true;transport=createOwnerTransport({game,onPacket,timeoutMs,onError});const provider=getProvider();if(provider?.descriptor?.sourceSHA256==='2946fa27eaf0963098f9f48ee48f777c5cf65166b56b043de9404f09813c240f')disposeProvider=provider.subscribe(onProvider)},
  stop(){started=false;disposeProvider?.();disposeProvider=null;for(const scope of applications.values())scope.rejectMaster(Error('manual-pool-stopped-unknown'));applications.clear();transport?.dispose();transport=null},
  claim:input=>send(input,false),lookup:input=>send(input,true),withApplication
 };
}
