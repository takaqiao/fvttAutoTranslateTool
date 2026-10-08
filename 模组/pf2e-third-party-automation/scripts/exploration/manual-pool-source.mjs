import {MODULE_ID,manualPoolRequest} from './schema.mjs';
import {canonicalJSON} from './revision-codec.mjs';
import {createOwnerTransport,OWNER_TRANSPORT_PROTOCOL} from './owner-transport.mjs';
import {WORKBENCH_SOURCE_SHA} from './manual-events.mjs';
import {manualPoolBatchModel,MANUAL_POOL_BATCH_SOURCE_SHA} from './manual-pool-model.mjs';
import {isPatreonSourceQualified} from './patreon-source-qualification.mjs';

const operationId='manual-pool-source';
const copy=value=>JSON.parse(canonicalJSON(value));
const same=(a,b)=>canonicalJSON(a)===canonicalJSON(b);
const author=message=>message?.author?.id??message?.author??message?.user?.id??message?.user;
const actorCurrent=(game,actor)=>{const token=actor?.token?.document??actor?.token;return !!actor&&(token?token.actor===actor&&token.parent?.tokens?.get(token.id)===token:game.actors?.get(actor.id)===actor)};
const documents=(check,result)=>canonicalJSON([check.toObject(true),result.toObject(true)]);
async function digest(value){return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(value))),byte=>byte.toString(16).padStart(2,'0')).join('')}

/** Private original-use observations. Public card markers only prevent fallback;
 * authenticated owners remain responsible for the history of their local JS. */
export function createManualPoolSources({game,Hooks,ledger,fromUuid,hpPools,clientNonce,getSession,isIssuer,onEnroll,onSettled,onError=()=>{},timeoutMs=10000,getBatchProvider=()=>game.pf2e?.thirdPartyManualPoolBatch}){
 let transport,started=false,batchDispose,updateHook,broker;
 const localUses=new WeakMap(),checks=new Map(),pendingResults=new Map(),sources=new Map(),tickets=new Map(),batches=new Map(),submitted=new Set(),workbenchProviders=new Map();
 const userId=game.user.id,system=game.system,sourceVersion=String(system?.version??''),activeGM=()=>game.users.activeGM;
 const gm=id=>game.users.get(id)?.active===true&&game.users.get(id)?.isGM===true&&activeGM()?.id===id;
 const local=()=>started&&game.user.id===userId&&game.system===system&&system?.id==='pf2e'&&sourceVersion!==''&&system.version===sourceVersion;
 const issuer=()=>local()&&gm(userId)&&isIssuer()===true;
 const route=(ownerId,actorUUID)=>({protocol:OWNER_TRANSPORT_PROTOCOL,operationId,driverUserId:activeGM()?.id,ownerUserId:ownerId,actorUUID});
 const provider=type=>type==='native-action'?{id:'pf2e',version:sourceVersion,sourceSHA:getBatchProvider()?.descriptor?.baseSourceSHA256??MANUAL_POOL_BATCH_SOURCE_SHA}:{id:'xdy-pf2e-workbench',sourceSHA:WORKBENCH_SOURCE_SHA};
 function providerCurrent(source){
  if(!local()||source.version!==2||source.sourceVersion!==sourceVersion||!same(source.provider,provider(source.sourceType)))return false;
  if(source.sourceType!=='native-action')return true;
  return manualPoolBatchModel(getBatchProvider()?.descriptor);
 }
 function providerWitness(type){
  const batch=getBatchProvider();if(!manualPoolBatchModel(batch?.descriptor))return ()=>false;
  const action=type==='native-action'?Array.from(game.pf2e?.actions?.values?.()??[]).find(value=>value.slug==='treat-wounds'):null;
  const workbench=type==='workbench'?Array.from(workbenchProviders.values()).find(value=>value.isCurrent()===true):null,use=action?.use,factory=action?.toActionVariant;
  return ()=>getBatchProvider()===batch&&manualPoolBatchModel(batch.descriptor)&&(type==='native-action'?!!action&&Array.from(game.pf2e.actions.values()).includes(action)&&action.use===use&&action.toActionVariant===factory:!!workbench&&workbench.isCurrent()===true);
 }
 function observeWorkbenchProvider(witness){if(witness?.macro&&typeof witness.isCurrent==='function'&&witness.isCurrent()===true)workbenchProviders.set(witness.macro,witness)}
 function reply(packet,status,proof){if(issuer())transport.send({...packet,kind:status==='accepted'?'continuation-ack':'denied',status,receiverUserId:packet.ownerUserId,...proof?{proof}:{}})}
 async function sessionTicket(intent,callerId,ownerClientNonce){
  const input=copy(intent),user=game.users.get(callerId),actor=await fromUuid(input.actorUUID),session=await getSession();
  if(session?.manual!==true||session.status!=='recording')throw Error('manual-pool-session-inactive');
  if(!issuer()||!user?.active||actor?.uuid!==input.actorUUID||actor.testUserPermission?.(user,'OWNER')!==true||session?.manual!==true||session.status!=='recording'||!session.actorUUIDs.includes(actor.uuid)||game.time.worldTime!==input.worldTime||session.startedAt>input.worldTime||!['native-action','workbench'].includes(input.sourceType)||input.kind!=='treatment'||input.riskySurgery===true)throw Error('manual-pool-source-unavailable');
  const ticket={sourceNonce:crypto.randomUUID(),sessionId:session.id,actorUUID:actor.uuid,useId:input.useId,sourceType:input.sourceType,sourceUserId:callerId,sourceClientNonce:ownerClientNonce,worldTime:input.worldTime};
  tickets.set(ticket.sourceNonce,ticket);return ticket;
 }
 async function begin(input,isCurrent,extra){
  if(!local())return null;const intent=copy({...input,worldTime:game.time.worldTime});let ticket;
  const unavailable=()=>{const handle=Object.freeze({});localUses.set(handle,{ticket:{useId:intent.useId},isCurrent:()=>false,...extra});return handle};
  if(issuer())try{ticket=await sessionTicket(intent,userId,clientNonce)}catch(error){if(error.message==='manual-pool-session-inactive')return null;onError(error);return unavailable()}
  else{
   if(!transport||!gm(activeGM()?.id))return null;
   let response;try{response=await transport.request({...route(userId,intent.actorUUID),kind:'continuation',status:'session',requestId:crypto.randomUUID(),receiverUserId:activeGM().id,ownerClientNonce:clientNonce,command:intent},{expectedSenderId:activeGM().id,matches:packet=>packet.kind==='denied'||packet.kind==='continuation-ack'&&packet.status==='accepted'})}catch(error){onError(error);return unavailable()}
   if(response.kind!=='continuation-ack')return response.status==='inactive'?null:unavailable();ticket=copy(response.proof);
  }
  let current=false;if(local())try{current=isCurrent()===true}catch(error){onError(error)}
  if(!current||ticket?.sourceClientNonce!==clientNonce||ticket.sourceUserId!==userId||ticket.useId!==intent.useId||ticket.worldTime!==game.time.worldTime)return unavailable();
  const handle=Object.freeze({}),entry={ticket,isCurrent,...extra,check:null,result:null,publication:null,source:null};localUses.set(handle,entry);return handle;
 }
 // A closed original use still denies fallback; this marker grants no execution.
 function poolMarker(handle){const entry=localUses.get(handle);return entry?{...entry.ticket.sessionId?{sessionId:entry.ticket.sessionId}:{},useId:entry.ticket.useId,pending:true}:null}
 async function beginNative(scope,{useId,tag}){
  if(scope.slug!=='treat-wounds'||scope.actors.length!==1||scope.user!==game.user)return null;
  const target=scope.params.target?.actor??scope.params.target,pool=target?.uuid&&hpPools.discover(target);if(pool?.ready&&pool.poolUUID===target.uuid&&pool.memberUUIDs?.length===1)return null;
  const action=scope.action,variant=scope.variant,use=variant.use;
  return begin({sourceType:'native-action',kind:'treatment',actorUUID:scope.actors[0].uuid,useId,riskySurgery:!!scope.params.selection?.feats?.['risky-surgery']},()=>local()&&game.user===scope.user&&Array.from(game.pf2e.actions.values()).includes(action)&&variant.use===use,{scope,tag});
 }
 async function beginWorkbench({context,healer,target,isCurrent}){
  if(context.sourceSHA!==WORKBENCH_SOURCE_SHA||context.kind!=='treatment'||context.riskySurgery||healer.uuid!==context.actorUUID||target.actor.uuid!==context.patientUUID)return null;
  const pool=hpPools.discover(target.actor);if(pool?.ready&&pool.poolUUID===target.actor.uuid&&pool.memberUUIDs?.length===1)return null;
  return begin({sourceType:'workbench',kind:context.kind,actorUUID:healer.uuid,useId:context.useId,riskySurgery:context.riskySurgery},()=>local()&&isCurrent()===true,{context,healer,target});
 }
 function checkCurrent(entry,message){
  if(!entry||!entry.isCurrent()||game.messages.get(message?.id)!==message||author(message)!==userId||message.speaker?.actor!==entry.ticket.actorUUID.split('.').at(-1)||message.rolls?.[0]?._evaluated!==true)return false;
  const healer=entry.healer??entry.scope?.actors?.[0];if(!game.user.active||!actorCurrent(game,healer)||healer?.testUserPermission?.(game.user,'OWNER')!==true)return false;
  if(entry.ticket.sourceType==='workbench')return message.flags?.pf2e?.context?.options?.includes(`exploration-manual-use:${entry.ticket.useId}`)||message.flags?.[MODULE_ID]?.explorationManual?.useId===entry.ticket.useId;
  const meta=message.flags?.[MODULE_ID]?.explorationManualNative,c=message.flags?.pf2e?.context;
  return message.isCheckRoll===true&&!message.isReroll&&c?.type==='skill-check'&&c.origin?.actor===entry.ticket.actorUUID&&c.options?.includes(entry.tag)&&meta?.useId===entry.ticket.useId&&meta.riskySurgery===false;
 }
 async function nativeCheck(handle,row){
  const entry=localUses.get(handle);if(!entry||row.actor!==entry.scope.actors[0]||!checkCurrent(entry,row.message))return;
  entry.check=row.message;checks.set(row.message.id,entry);const result=pendingResults.get(row.message.id);if(result){entry.result=result;pendingResults.delete(row.message.id)}await publish(entry);
 }
 async function nativeResult(message){
  if(game.messages.get(message?.id)!==message||message.isCheckRoll!==false||message.rolls?.[0]?._evaluated!==true)return;
  const parentId=message.flags?.pf2e?.origin?.messageId,entry=checks.get(parentId);if(!entry){if(parentId)pendingResults.set(parentId,message);return}
  if(entry.result&&entry.result!==message){entry.invalid=true;return}entry.result=message;await publish(entry);
 }
 async function workbenchCheck(handle,message){const entry=localUses.get(handle);if(!checkCurrent(entry,message))return;if(entry.check&&entry.check!==message){entry.invalid=true;return}entry.check=message;checks.set(message.id,entry);await publish(entry)}
 async function workbenchResult(handle,message){const entry=localUses.get(handle);if(!entry||game.messages.get(message?.id)!==message||message.rolls?.[0]?._evaluated!==true)return;if(entry.result&&entry.result!==message){entry.invalid=true;return}entry.result=message;await publish(entry)}
 function patientUUID(entry){
  if(entry.context)return entry.target.actor.uuid;
  const c=entry.check?.flags?.pf2e?.context,meta=entry.check?.flags?.[MODULE_ID]?.explorationManualNative,b=meta?.patreonImmunity;
  const paid=game.modules?.get('patreon-v3'),descriptor=paid?.api?.explorationManualImmunity?.descriptor;
  // The original provider saves patient metadata before its immunity writer.
  // Wait for that source snapshot, independently of the later create terminal.
  if(paid?.active===true&&(isPatreonSourceQualified(descriptor)||paid.version==='3.2.29'&&descriptor?.version===1&&descriptor.providerVersion==='3.2.29'&&descriptor.pf2eSourceSHA256===MANUAL_POOL_BATCH_SOURCE_SHA)&&!b)return null;
  if(c?.target?.actor)return c.target.actor===meta?.patientUUID?c.target.actor:null;
  return c?.target==null&&b?.messageId===entry.check.id&&b.useId===entry.ticket.useId&&b.actorUUID===entry.ticket.actorUUID&&b.patientUUID===meta.patientUUID&&b.targetSnapshot?.type==='patreon-single-target'&&b.targetSnapshot.actorUUID===meta.patientUUID?meta.patientUUID:null;
 }
 async function publish(entry){
  if(entry.publication)return entry.publication;if(!entry.check||!entry.result||!patientUUID(entry))return;
  const result=entry.result;entry.publication=(async()=>{
   if(entry.invalid||!checkCurrent(entry,entry.check)||author(result)!==userId||result.speaker?.actor!==entry.check.speaker.actor||!result.rolls?.[0]?._evaluated||!Number.isFinite(result.rolls[0].total))throw Error('manual-pool-source-unavailable');
   const patient=await fromUuid(patientUUID(entry)),pool=patient&&hpPools.discover(patient);
   if(!pool?.ready||!pool.memberUUIDs?.includes(patient.uuid))return;
   if(pool.poolUUID===patient.uuid&&pool.memberUUIDs.length===1){if(result.update)await result.update({[`flags.${MODULE_ID}.explorationManualPoolParticipation`]:null});return}
   if(entry.ticket.sourceType==='native-action'&&result.flags?.pf2e?.origin?.messageId!==entry.check.id)throw Error('manual-pool-source-unavailable');
   if(result.update)await result.update({[`flags.${MODULE_ID}.explorationManualPoolParticipation`]:{sessionId:entry.ticket.sessionId,useId:entry.ticket.useId,poolUUID:pool.poolUUID}});
   const source={version:2,sourceVersion,sessionId:entry.ticket.sessionId,activityId:`manual:${entry.ticket.sourceType==='native-action'?entry.check.id:result.id}`,actorUUID:entry.ticket.actorUUID,patientUUID:patient.uuid,sourceType:entry.ticket.sourceType,useId:entry.ticket.useId,checkId:entry.check.id,resultId:result.id,rollIndex:0,worldTime:entry.ticket.worldTime,sourceUserId:userId,sourceClientNonce:clientNonce,sourceNonce:entry.ticket.sourceNonce,provider:provider(entry.ticket.sourceType),documentsDigest:await digest(documents(entry.check,result))};
   entry.source=source;entry.checkSource=documents(entry.check,result);entry.patient=patient;entry.poolUUID=pool.poolUUID;
   const witness=providerWitness(source.sourceType),current=()=>local()&&!entry.invalid&&witness()&&entry.isCurrent()&&checkCurrent(entry,entry.check)&&game.messages.get(result.id)===result&&documents(entry.check,result)===entry.checkSource&&game.time.worldTime===source.worldTime&&hpPools.discover(patient).poolUUID===entry.poolUUID;
   if(!current())throw Error('manual-pool-source-changed');sources.set(result.id,{source,check:entry.check,result,patient,poolUUID:pool.poolUUID,isCurrent:current,ready:()=>entry.publication});
   if(issuer())await accept(source,userId);else{
    const key=source.sourceNonce;if(submitted.has(key))return;submitted.add(key);
    const response=await transport.request({...route(userId,source.actorUUID),kind:'continuation',status:'source',requestId:crypto.randomUUID(),receiverUserId:activeGM().id,ownerClientNonce:clientNonce,sessionId:source.sessionId,activityId:source.activityId,command:copy(source)},{expectedSenderId:activeGM().id,matches:packet=>packet.kind==='denied'||packet.kind==='continuation-ack'&&packet.status==='accepted'&&same(packet.proof,source)});
    if(response.kind!=='continuation-ack'||!current())throw Error('manual-pool-source-unconfirmed');
   }
  })();entry.publication.catch(onError);return entry.publication;
 }
 async function qualified(source,callerId,{saved=false}={}){
  const captured=copy(source),session=await getSession(),healer=await fromUuid(captured.actorUUID),patient=await fromUuid(captured.patientUUID),user=game.users.get(callerId),sourceUser=game.users.get(captured.sourceUserId),check=game.messages.get(captured.checkId),result=game.messages.get(captured.resultId);
  if(!check||!result||!patient||!healer||typeof check.toObject!=='function'||typeof result.toObject!=='function')throw Error('manual-pool-source-unavailable');
  const before=documents(check,result),pool=hpPools.discover(patient),poolUUID=pool.poolUUID,witness=providerWitness(captured.sourceType);
  const target=check.flags?.[MODULE_ID]?.explorationManualNative?.patreonImmunity?.targetSnapshot,targetToken=target?.tokenUUID?await fromUuid(target.tokenUUID):null;
  const validate=()=>{
   if(!local()||!gm(activeGM()?.id)||!saved&&!issuer()||!witness()||!actorCurrent(game,healer)||!actorCurrent(game,patient)||session?.id!==captured.sessionId||session.status!=='recording'||!session.actorUUIDs.includes(patient.uuid)||!session.actorUUIDs.includes(healer.uuid)||game.time.worldTime!==captured.worldTime||!providerCurrent(captured)||!user?.active||!sourceUser?.active||game.users.get(callerId)!==user||healer.testUserPermission?.(sourceUser,'OWNER')!==true||healer.testUserPermission?.(user,'OWNER')!==true||patient.testUserPermission?.(user,'OWNER')!==true)return false;
   const now=hpPools.discover(patient);if(!now?.ready||now.poolUUID!==poolUUID||!now.memberUUIDs?.includes(patient.uuid))return false;
   if(game.messages.get(check.id)!==check||game.messages.get(result.id)!==result||author(check)!==captured.sourceUserId||author(result)!==captured.sourceUserId||check.speaker?.actor!==healer.id||result.speaker?.actor!==healer.id||check.rolls?.[0]?._evaluated!==true||result.rolls?.[0]?._evaluated!==true||!Number.isFinite(result.rolls[0].total)||documents(check,result)!==before)return false;
   if(captured.sourceType==='workbench'){const m=result.flags?.[MODULE_ID]?.explorationManual;return m?.sourceSHA===WORKBENCH_SOURCE_SHA&&m.useId===captured.useId&&m.patientUUID===patient.uuid&&m.actorUUID===healer.uuid&&m.riskySurgery===false&&m.checkIds?.includes(check.id)}
   const m=check.flags?.[MODULE_ID]?.explorationManualNative,c=check.flags?.pf2e?.context;
   const actualTarget=c?.target?.actor===patient.uuid||c?.target==null&&target?.type==='patreon-single-target'&&target.actorUUID===patient.uuid&&targetToken?.uuid===target.tokenUUID&&targetToken.actor===patient&&targetToken.parent?.tokens?.get(targetToken.id)===targetToken;
   return actualTarget&&check.isCheckRoll===true&&!check.isReroll&&m?.useId===captured.useId&&m.patientUUID===patient.uuid&&m.riskySurgery===false&&c?.origin?.actor===healer.uuid&&c.options?.includes(m.tag)&&result.flags?.pf2e?.origin?.messageId===check.id;
  };
  if(!validate()||await digest(before)!==captured.documentsDigest||!validate())throw Error('manual-pool-source-changed');return {source:captured,check,result,healer,patient,poolUUID,isCurrent:validate};
 }
 async function accept(source,callerId){
  const input=copy(source),ticket=tickets.get(input.sourceNonce);
  if(!ticket||['sessionId','actorUUID','useId','sourceType','worldTime','sourceUserId','sourceClientNonce'].some(key=>ticket[key]!==input[key])||callerId!==ticket.sourceUserId)throw Error('manual-pool-source-unavailable');
  const entry=await qualified(input,callerId);
  try{await onEnroll(input)}catch(error){
   if(error.message!=='duplicate-activity')throw error;
   if(typeof ledger?.getActivity!=='function'||typeof ledger.appendManualEvidence!=='function')throw error;
   const native=input.sourceType==='native-action',meta=(native?entry.check:entry.result).flags?.[MODULE_ID]?.[native?'explorationManualNative':'explorationManual'];
   const activityId=`manual:${native?input.checkId:input.resultId}`,resultIds=native?[input.resultId]:[...meta.stageIds??[],input.resultId];
   const validate=current=>{
    if(!entry.isCurrent()||!current||current.id!==input.activityId||current.id!==activityId||current.sessionId!==input.sessionId||current.providerId!=='manual'||current.kind!=='treatment'||current.state!=='awaiting-evidence'||current.source?.manual!==true||current.temporalSource?.type==='checkpoint-reservation'
      ||current.source.type!==input.sourceType||current.source.messageId!==(native?input.checkId:input.resultId)||current.proof?.useId!==input.useId||!same(current.proof.checkIds,[input.checkId])||!Array.isArray(current.proof.resultIds)||current.proof.resultIds.some(id=>!resultIds.includes(id))
      ||current.actorUUID!==entry.healer.uuid||!same(current.patientUUIDs,[entry.patient.uuid])||!same(current.hpPoolUUIDs,[entry.poolUUID])||current.startedAt!==input.worldTime||current.endsAt!==input.worldTime+600||current.durationSeconds!==600||current.treatmentImmunitySeconds!==(meta.continualRecovery?600:3600)
      ||current.groupId!==(native?activityId:meta.groupId??activityId)||current.groupProof!==(native?undefined:meta.groupProof))throw Error('manual-pool-source-mismatch');
    if(native?current.source.tag!==meta.tag:current.source.sourceSHA!==WORKBENCH_SOURCE_SHA||current.source.lexicalSource!==true||!same(meta.checkIds,[input.checkId])||current.source.adapter!==undefined&&current.source.adapter!=='target-callback-instrumentation-v1'||current.source.adapterSHA!==undefined&&current.source.adapterSHA!==meta.adapterSHA)throw Error('manual-pool-source-mismatch');
    return current.options;
   };
   const activity=await ledger.getActivity(input.activityId);validate(activity);
   await ledger.appendManualEvidence(input.activityId,{activity,proof:{useId:input.useId,checkIds:[input.checkId],resultIds:[input.resultId]},resolveOptions:validate});
  }
  if(!entry.isCurrent())throw Error('manual-pool-source-changed');
  await ledger.recordManualPoolSource(input,{evidenceGuard:entry.isCurrent});const observed=sources.get(input.resultId);sources.set(input.resultId,observed?{...entry,isCurrent:()=>entry.isCurrent()&&observed.isCurrent()}:entry);return input;
 }
 async function onPacket(packet,senderId){
  if(packet.operationId!==operationId||!local()||!gm(userId)||packet.driverUserId!==userId||packet.ownerUserId!==senderId||packet.kind!=='continuation')return;
  if(!issuer()){
   if(packet.status!=='session')return;const session=await getSession(),hint=game.user.getFlag?.(MODULE_ID,'explorationSession');
   // A peer tab with a recording scope stays silent. It cannot race the tab
   // which owns the persistent creation ACK or answer from stale user flags.
   if(session?.manual===true&&session.status==='recording'||hint&&session?.id!==hint)return;
   transport.send({...packet,kind:'denied',status:'inactive',receiverUserId:packet.ownerUserId});return;
  }
  try{
   if(packet.status==='session'){const ticket=await sessionTicket(packet.command,senderId,packet.ownerClientNonce);reply(packet,'accepted',ticket)}
   else if(packet.status==='source'){const source=await accept(packet.command,senderId);reply(packet,'accepted',source)}
   else if(packet.status==='batch'){
    const input=copy(packet.command),entry=sources.get(input.resultId);
    if(!entry?.isCurrent()||entry.source.sourceNonce!==input.sourceNonce||entry.source.sourceUserId!==senderId||entry.source.sourceClientNonce!==packet.ownerClientNonce||input.patientUUID!==entry.patient.uuid||input.poolUUID!==entry.poolUUID)throw Error('manual-pool-batch-source-unavailable');
    batches.set(input.batchId,{entry,callerId:senderId,clientNonce:packet.ownerClientNonce});reply(packet,'accepted',{batchId:input.batchId});
   }
  }catch(error){onError(error);reply(packet,error.message==='manual-pool-session-inactive'?'inactive':'unavailable')}
 }
 async function resolveSource(request,{callerId=userId,owner=false,readOnly=false}={}){
  const input=manualPoolRequest(copy(request)),batch=batches.get(input.batchId);let entry=sources.get(input.resultId);
  if(readOnly&&!entry){const activity=await ledger.getActivity(input.activityId),source=activity?.proof.manualPoolSource;if(source)try{entry=await qualified(source,callerId,{saved:true})}catch{return null}}
  if(!entry||!entry.isCurrent()||!readOnly&&(!batch||batch.entry!==entry||batch.callerId!==callerId||batch.clientNonce!==input.ownerClientNonce)||entry.source.sessionId!==input.sessionId||entry.source.activityId!==input.activityId||entry.source.actorUUID!==input.actorUUID||entry.source.sourceType!==input.sourceType||entry.source.useId!==input.useId||entry.source.checkId!==input.checkId||entry.source.resultId!==input.resultId||input.rollIndex!==0||input.stage!=='healing'||input.poolUUID!==entry.poolUUID||input.patientUUIDs.length!==1||input.patientUUIDs[0]!==entry.patient.uuid)return null;
  const binding=copy(input);delete binding.ownerClientNonce;delete binding.attemptNonce;
  return {binding:{...binding,effectId:input.resultId,selectedPatientUUID:entry.patient.uuid,worldTime:entry.source.worldTime},isCurrent:entry.isCurrent};
 }
 async function authorizeBatch({phase,batch}){
  const entry=sources.get(batch.message.id),marker=batch.message.flags?.[MODULE_ID]?.explorationManualPoolParticipation;
  if(phase==='admit'){
   if(!entry&&!marker)return {status:'unregistered'};
   if(!entry)return {status:'participating',sourceBinding:{unavailable:true},isCurrent:()=>false};
   return {status:'participating',sourceBinding:copy(entry.source),isCurrent:entry.isCurrent};
  }
  await entry?.ready?.();
  if(!entry||!entry.isCurrent()||!broker||batch.targets.length!==1||batch.targets[0].patient!==entry.patient||batch.candidates?.length!==1||batch.candidates[0].patient!==entry.patient)throw Error('manual-pool-single-patient-source-required');
  const request={sessionId:entry.source.sessionId,activityId:entry.source.activityId,actorUUID:entry.source.actorUUID,sourceType:entry.source.sourceType,useId:entry.source.useId,checkId:entry.source.checkId,resultId:entry.source.resultId,rollIndex:batch.rollIndex,stage:'healing',poolUUID:entry.poolUUID,patientUUIDs:[entry.patient.uuid],batchId:batch.batchId,ownerClientNonce:clientNonce,attemptNonce:crypto.randomUUID()};
  batches.set(batch.batchId,{entry,callerId:userId,clientNonce});
  if(!issuer()){
   const response=await transport.request({...route(userId,request.actorUUID),kind:'continuation',status:'batch',requestId:crypto.randomUUID(),receiverUserId:activeGM().id,ownerClientNonce:clientNonce,command:{sourceNonce:entry.source.sourceNonce,resultId:request.resultId,batchId:batch.batchId,patientUUID:entry.patient.uuid,poolUUID:entry.poolUUID}},{expectedSenderId:activeGM().id,matches:packet=>packet.kind==='continuation-ack'&&packet.status==='accepted'&&packet.proof?.batchId===batch.batchId});
   if(response.status!=='accepted'||!entry.isCurrent())throw Error('manual-pool-source-changed');
  }
  const grant=await broker.claim(request);if(!entry.isCurrent())throw Error('manual-pool-source-changed');
  return {status:'selected',batchId:batch.batchId,sourceDigest:grant.sourceDigest,selections:[{poolUUID:entry.poolUUID,effectKey:grant.effectKey,patientUUIDs:[entry.patient.uuid],selectedOrdinal:0,grant}]};
 }
 function start(completion){if(started)return;started=true;broker=completion;if(['on','off','emit'].every(key=>typeof game.socket?.[key]==='function'))transport=createOwnerTransport({game,onPacket,timeoutMs,onError});
  const batchProvider=getBatchProvider();if(manualPoolBatchModel(batchProvider?.descriptor)&&typeof batchProvider.subscribe==='function')batchDispose=batchProvider.subscribe(()=>{},{authorizeBatch:event=>event.phase==='admit'?authorizeBatchSync(event):authorizeBatch(event)});
  updateHook=Hooks?.on('updateChatMessage',message=>{const entry=checks.get(message.id);if(entry&&entry.check===message&&!entry.publication)void publish(entry).catch(onError)});
 }
 function authorizeBatchSync(event){const entry=sources.get(event.batch.message.id),marker=event.batch.message.flags?.[MODULE_ID]?.explorationManualPoolParticipation;if(!entry&&!marker)return {status:'unregistered'};return {status:'participating',sourceBinding:entry?copy(entry.source):{unavailable:true},isCurrent:()=>!!entry&&entry.isCurrent()}}
 function stop(){if(!started)return;started=false;batchDispose?.();transport?.dispose();if(updateHook)Hooks.off('updateChatMessage',updateHook);batches.clear();sources.clear();tickets.clear()}
 return {start,stop,beginNative,nativeCheck,nativeResult,beginWorkbench,workbenchCheck,workbenchResult,poolMarker,observeWorkbenchProvider,resolveSource,onSettled:request=>onSettled?.(request.activityId)};
}
