import {isActiveGM} from './document-store.mjs';
import {createOwnerTransport,OWNER_TRANSPORT_PROTOCOL} from './owner-transport.mjs';
import {canonicalJSON} from './revision-codec.mjs';

const operationId='patreon-manual-immunity';
const same=(a,b)=>{try{return canonicalJSON(a)===canonicalJSON(b)}catch{return false}};
const copy=value=>JSON.parse(canonicalJSON(value));

/** Authenticated owners report the fixed provider's local terminal. This does
 * not attest arbitrary client code or turn ordinary document flags into proof. */
export function createPatreonManualImmunityOwner({game,fromUuid,patreonImmunity,getSession,getActivity,record,timeoutMs=10000,onError=()=>{}}){
 let transport,dispose,started=false;const received=new Map(),submitted=new Set(),observedSources=new Map();
 const activeGM=()=>game.users?.activeGM;
 const gm=id=>activeGM()?.id===id&&game.users.get(id)?.isGM===true&&game.users.get(id)?.active===true;
 const route=(callerId,actorUUID)=>({protocol:OWNER_TRANSPORT_PROTOCOL,operationId,driverUserId:activeGM()?.id,ownerUserId:callerId,actorUUID});
 const reply=(packet,kind,status,proof)=>transport?.send({...route(packet.ownerUserId,packet.actorUUID),kind,status,requestId:packet.requestId,receiverUserId:packet.ownerUserId,
  ...packet.sessionId?{sessionId:packet.sessionId}:{},...packet.activityId?{activityId:packet.activityId}:{},...packet.completionId?{completionId:packet.completionId}:{},...proof?{proof}:{}});
 async function sessionSource(actorUUID,callerId,startedAt){
  if(!isActiveGM(game)||!gm(game.user.id)||game.time.worldTime!==startedAt)throw Error('native-recording-source-unavailable');
  const user=game.users.get(callerId),actor=await fromUuid(actorUUID),session=await getSession();
  if(!isActiveGM(game)||!gm(game.user.id)||!user?.active||actor?.uuid!==actorUUID||actor.testUserPermission?.(user,'OWNER')!==true
   ||session?.status!=='recording'||!session.actorUUIDs.includes(actorUUID)||game.time.worldTime!==startedAt
   ||Number.isFinite(session.startedAt)&&session.startedAt>startedAt)throw Error('native-recording-source-unavailable');
  return {sessionId:session.id,startedAt,actorUUID,userId:callerId};
 }
 async function validate(proof,callerId){
  const binding=proof?.binding,user=game.users.get(callerId),healer=binding&&await fromUuid(binding.actorUUID),patient=binding&&await fromUuid(binding.patientUUID);
  const observed=binding&&observedSources.get(binding.actorUUID);
  const session=isActiveGM(game)?await getSession():observed&&{id:observed.sessionId,status:'recording',startedAt:observed.startedAt,actorUUIDs:[binding.actorUUID,binding.patientUUID]};
  if(!binding||!user?.active||proof.creatorId!==callerId||callerId!==binding.sourceUserId&&!gm(callerId)
   ||!isActiveGM(game)&&observed?.startedAt!==binding.startedAt
   ||healer?.uuid!==binding.actorUUID||patient?.uuid!==binding.patientUUID||healer.testUserPermission?.(user,'OWNER')!==true||patient.testUserPermission?.(user,'OWNER')!==true
   ||session?.status!=='recording'||session.id!==binding.recordingSessionId||!session.actorUUIDs.includes(binding.actorUUID)||!session.actorUUIDs.includes(binding.patientUUID)
   ||game.time.worldTime!==binding.startedAt||Number.isFinite(session.startedAt)&&session.startedAt>binding.startedAt)throw Error('native-owner-terminal-unavailable');
  const activity={sessionId:session.id,actorUUID:binding.actorUUID,patientUUIDs:[binding.patientUUID],kind:'treatment',source:{type:'native-action',tag:binding.tag},startedAt:binding.startedAt,
   proof:{useId:binding.useId,checkIds:[binding.messageId]}};
  if(!await patreonImmunity.evidence(proof,activity))throw Error('native-owner-terminal-unavailable');
  const current=isActiveGM(game)?await getSession():session;
  if(!user.active||current?.id!==session.id||current.status!=='recording'||game.time.worldTime!==binding.startedAt
   ||healer.testUserPermission?.(user,'OWNER')!==true||patient.testUserPermission?.(user,'OWNER')!==true)throw Error('native-owner-terminal-unavailable');
  return session;
 }
 async function saved(proof,callerId){
  if(!isActiveGM(game)||proof?.creatorId!==callerId||!game.users.get(callerId)?.active)return null;
  const user=game.users.get(callerId),healer=await fromUuid(proof.binding?.actorUUID),patient=await fromUuid(proof.binding?.patientUUID),activity=await getActivity(`manual:${proof.binding?.messageId}`);
  return isActiveGM(game)&&user.active&&healer?.testUserPermission?.(user,'OWNER')===true&&patient?.testUserPermission?.(user,'OWNER')===true
   &&activity?.sessionId===proof.binding?.recordingSessionId&&same(activity.proof?.nativeImmunity,proof)?activity:null;
 }
 async function accept(proof,callerId){
  if(!isActiveGM(game)||!gm(game.user.id))throw Error('active-gm-required');
  const existing=await saved(proof,callerId);if(existing)return existing;
  await validate(proof,callerId);if(!isActiveGM(game)||!gm(game.user.id))throw Error('active-gm-required');
  const key=`${proof.binding.recordingSessionId}:${proof.binding.invocationId}`,previous=received.get(key);
  if(previous){if(!same(previous.proof,proof))throw Error('native-owner-terminal-mismatch');return previous.task}
  const entry={proof:copy(proof)};entry.task=(async()=>{await record(proof);const activity=await saved(proof,callerId);if(!activity)throw Error('native-owner-seal-unavailable');return activity})();
  received.set(key,entry);return entry.task;
 }
 async function onPacket(packet,sender){
  if(packet.operationId!==operationId||!isActiveGM(game)||packet.driverUserId!==game.user.id||packet.ownerUserId!==sender)return;
  if(packet.kind==='continuation'&&packet.status==='session-query'){
   try{const source=await sessionSource(packet.actorUUID,sender,packet.proof?.startedAt);reply(packet,'continuation-ack','accepted',source)}catch{reply(packet,'denied','unavailable')}
   return;
  }
  if(packet.kind!=='completion'||!['terminal','lookup'].includes(packet.status))return;
  const proof=packet.proof,binding=proof?.binding;
  if(proof?.creatorId!==sender||packet.sessionId!==binding?.recordingSessionId||packet.actorUUID!==binding?.actorUUID
   ||packet.activityId!==`manual:${binding?.messageId}`||packet.completionId!==binding?.invocationId){reply(packet,'denied','unavailable');return}
  try{
   const activity=packet.status==='lookup'?await saved(proof,sender):await accept(proof,sender);
   if(!activity)throw Error('native-owner-seal-unavailable');reply(packet,'completion-ack','accepted',copy(activity.proof.nativeImmunity));
  }catch{reply(packet,'denied','unavailable')}
 }
 async function sendTerminal(proof){
  if(!started||proof?.creatorId!==game.user.id)return;
  if(isActiveGM(game)){await accept(proof,game.user.id);return}
  if(!transport||!gm(activeGM()?.id))return;await validate(proof,game.user.id);
  const gmId=activeGM().id,key=`${proof.binding.recordingSessionId}:${proof.binding.invocationId}`;if(submitted.has(key))return;submitted.add(key);
  const packet={...route(game.user.id,proof.binding.actorUUID),kind:'completion',status:'terminal',requestId:crypto.randomUUID(),receiverUserId:gmId,
   sessionId:proof.binding.recordingSessionId,activityId:`manual:${proof.binding.messageId}`,completionId:proof.binding.invocationId,proof:copy(proof)};
  const matches=reply=>reply.kind==='denied'||reply.kind==='completion-ack'&&reply.status==='accepted'&&same(reply.proof,proof);
  try{const response=await transport.request(packet,{expectedSenderId:gmId,matches});if(response.kind==='denied')throw Error('native-owner-terminal-unavailable')}
  catch(error){
   if(!started||!gm(gmId)||error.message!=='owner-response-timeout-unknown-no-retry')throw error;
   // Completion is never resent after an unknown ACK. Only the exact GM seal
   // can answer this read; missing proof leaves the activity incomplete.
   const response=await transport.request({...packet,status:'lookup',requestId:crypto.randomUUID()},{expectedSenderId:gmId,matches});
   if(response.kind==='denied')throw Error('native-owner-seal-unavailable');
  }
 }
 return {
  start(){if(started)return;started=true;
   if(['on','off','emit'].every(method=>typeof game.socket?.[method]==='function'))transport=createOwnerTransport({game,onPacket,timeoutMs,onError});
   dispose=patreonImmunity.subscribe(proof=>{void sendTerminal(proof).catch(onError)});
  },
  stop(){if(!started)return;started=false;dispose?.();dispose=null;transport?.dispose();transport=null},
  async bindSource(actorUUID){
   if(!started||typeof actorUUID!=='string'||!gm(activeGM()?.id))return null;
   const startedAt=game.time.worldTime;
   try{
    if(isActiveGM(game)){const source=await sessionSource(actorUUID,game.user.id,startedAt);observedSources.set(actorUUID,source);return source}
    if(!transport)return null;const gmId=activeGM().id;
    const response=await transport.request({...route(game.user.id,actorUUID),kind:'continuation',status:'session-query',requestId:crypto.randomUUID(),receiverUserId:gmId,proof:{startedAt}},
     {expectedSenderId:gmId,matches:reply=>reply.kind==='denied'||reply.kind==='continuation-ack'&&reply.status==='accepted'&&reply.proof?.actorUUID===actorUUID&&reply.proof.userId===game.user.id&&reply.proof.startedAt===startedAt});
    if(!started||!gm(gmId)||game.time.worldTime!==startedAt||response.kind!=='continuation-ack')return null;
    const source=copy(response.proof);observedSources.set(actorUUID,source);return source;
   }catch{return null}
  }
 };
}
