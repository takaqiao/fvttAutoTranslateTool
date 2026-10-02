import {isActiveGM} from './document-store.mjs';
import {normalizeManualDeclaration} from './manual-time.mjs';
import {manualSourceIntent,checkpointBinding,sameCheckpoint,activityCheckpointBinding,sameActivityCheckpoint,id} from './schema.mjs';
/** Only a human's generic declaration crosses this bridge, never treatment credentials. */
export function createManualRecordBridge({game,fromUuid,getSession,observe,reserveSource,checkpointContext,enrollCheckpointActivity,lookupCheckpointActivity}){
 let socket;
 async function checkpointActor(binding,actorUUID,callerId){
  if(typeof checkpointContext!=='function')throw Error('activity-checkpoint-unavailable');
  const current={...checkpointContext()},user=game.users.get(callerId);
  if(!isActiveGM(game)||!game.user.active||!user?.active||current?.sessionId!==binding.sessionId||current.worldTime!==binding.from)throw Error('activity-checkpoint-changed');
  const actor=await fromUuid(actorUUID),session=await getSession();
  const guard=()=>{
   const now=checkpointContext();
   return isActiveGM(game)&&game.user.active===true&&game.users.get(callerId)===user&&user.active===true&&actor?.uuid===actorUUID&&actor.testUserPermission?.(user,'OWNER')===true&&now?.sessionId===binding.sessionId&&now.worldTime===binding.from&&now.leaseNonce===current.leaseNonce;
  };
  if(!guard()||session?.id!==binding.sessionId||!session.actorUUIDs.includes(actorUUID)||!sameActivityCheckpoint(session.activityCheckpoint,binding))throw Error('manual-actor-not-allowed');
  return {authenticatedCaller:callerId,leaseNonce:current.leaseNonce,guard};
 }
 async function receive(event,callerId){
  event=normalizeManualDeclaration(event);
  if(event.checkpointBinding){
   const options=await checkpointActor(event.checkpointBinding,event.actorUUID,callerId);
   if(typeof enrollCheckpointActivity!=='function')throw Error('activity-checkpoint-unavailable');
   return enrollCheckpointActivity(event.checkpointBinding,event,options);
  }
  if(!isActiveGM(game))throw Error('active-gm-required');const user=game.users.get(callerId);if(!user?.active)throw Error('caller-offline');
  const actor=await fromUuid(event?.actorUUID),session=await getSession();
  if(!isActiveGM(game)||!user.active||!actor?.testUserPermission(user,'OWNER')||session?.status!=='recording'||!session.actorUUIDs.includes(actor.uuid))throw Error('manual-actor-not-allowed');
  if(event.sessionId!==undefined&&event.sessionId!==session.id)throw Error('manual-session-changed');
  return observe({...event,id:crypto.randomUUID(),expectedSessionId:session.id,actorUUID:actor.uuid,patientUUIDs:[],kind:'activity',source:{type:'user-record',userId:callerId,unverified:true},missing:['manual-source-requires-review']});
 }
 async function receiveReservation(binding,intent,callerId){
  if(!isActiveGM(game))throw Error('active-gm-required');const session=await getSession();
  if(!session?.manualCheckpoint||session.status!=='running'){if(binding)throw Error('manual-checkpoint-unavailable');return null}
  if(session.status!=='running'||session.manualCheckpoint.phase!=='open')throw Error('manual-checkpoint-closed');
  intent=manualSourceIntent(intent);const user=game.users.get(callerId),actor=await fromUuid(intent.actorUUID);
  if(!isActiveGM(game)||!user?.active||actor?.uuid!==intent.actorUUID||actor.testUserPermission?.(user,'OWNER')!==true||!session.actorUUIDs.includes(actor.uuid))throw Error('manual-actor-owner-required');
  if(binding&&!sameCheckpoint(session.manualCheckpoint,binding))throw Error('manual-checkpoint-mismatch');
  if(typeof reserveSource!=='function')throw Error('session-driver-required');return reserveSource(binding??checkpointBinding(session.manualCheckpoint),intent,callerId);
 }
 async function lookup(binding,registrationId,actorUUID,callerId){
  binding=activityCheckpointBinding(binding);id(registrationId,'registration');id(actorUUID,'actor');
  const options=await checkpointActor(binding,actorUUID,callerId);
  if(typeof lookupCheckpointActivity!=='function')throw Error('activity-checkpoint-unavailable');
  return lookupCheckpointActivity(binding,registrationId,{...options,actorUUID});
 }
 return {register(api){socket=api;socket?.register('exploration:lookupCheckpointActivity',async function(binding,registrationId,actorUUID){try{return {ok:true,value:await lookup(binding,registrationId,actorUUID,this.socketdata?.userId)}}catch(error){return {ok:false,error:error.message}}});socket?.register('exploration:reserveSource',async function(binding,intent){try{return {ok:true,value:await receiveReservation(binding,intent,this.socketdata?.userId)}}catch(error){return {ok:false,error:error.message}}});socket?.register('exploration:record',async function(event){try{return {ok:true,value:await receive(event,this.socketdata?.userId)}}catch(error){return {ok:false,error:error.message}}})},async reserveSource(binding,intent){
  if(isActiveGM(game))return receiveReservation(binding,intent,game.user.id);if(!game.users.activeGM?.active&&!binding)return null;if(!socket)throw Error('manual-record-socket-unavailable');const response=await socket.executeAsGM('exploration:reserveSource',binding,intent);if(!response?.ok)throw Error(response?.error??'manual-reservation-unconfirmed');return response.value;
 },async lookupCheckpointActivity(binding,registrationId,actorUUID){
  binding=activityCheckpointBinding(binding);id(registrationId,'registration');id(actorUUID,'actor');
  if(isActiveGM(game))return lookup(binding,registrationId,actorUUID,game.user.id);if(!socket)throw Error('manual-record-socket-unavailable');
  const response=await socket.executeAsGM('exploration:lookupCheckpointActivity',binding,registrationId,actorUUID);if(!response?.ok)throw Error(response?.error??'manual-lookup-unconfirmed');return response.value;
 },async record(event){event=normalizeManualDeclaration(event);if(isActiveGM(game))return receive(event,game.user.id);if(!socket)throw Error('manual-record-socket-unavailable');const response=await socket.executeAsGM('exploration:record',event);if(!response?.ok)throw Error(response?.error??'manual-record-unconfirmed');return response.value}};
}
