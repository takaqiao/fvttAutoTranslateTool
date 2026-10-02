import {isActiveGM} from './document-store.mjs';
import {normalizeManualDeclaration} from './manual-time.mjs';
import {manualSourceIntent,checkpointBinding,sameCheckpoint,activityCheckpointBinding,sameActivityCheckpoint,id} from './schema.mjs';
/** Only a human's generic declaration crosses this bridge, never treatment credentials. */
export function createManualRecordBridge({game,fromUuid,getSession,getActivities,observe,reserveSource,checkpointContext,enrollCheckpointActivity,lookupCheckpointActivity}){
 let socket;
 async function checkpointActor(binding,actorUUID,callerId){
  if(typeof checkpointContext!=='function')throw Error('activity-checkpoint-unavailable');
  const current={...checkpointContext()},user=game.users.get(callerId);
  if(!isActiveGM(game)||!game.user.active||!user?.active||current?.sessionId!==binding.sessionId||current.worldTime!==binding.from)throw Error('activity-checkpoint-changed');
  const actor=await fromUuid(actorUUID),session=await getSession();
  const guard=()=>{
   const now=checkpointContext();
   return isActiveGM(game)&&game.user.active===true&&game.users.get(callerId)===user&&user.active===true&&actor?.uuid===actorUUID&&actor.testUserPermission?.(user,'OWNER')===true&&(!game.actors?.get||game.actors.get(actor.id)===actor)&&now?.sessionId===binding.sessionId&&now.worldTime===binding.from&&now.leaseNonce===current.leaseNonce;
  };
  if(!guard()||session?.id!==binding.sessionId||!session.actorUUIDs.includes(actorUUID)||!sameActivityCheckpoint(session.activityCheckpoint,binding))throw Error('manual-actor-not-allowed');
  return {authenticatedCaller:callerId,leaseNonce:current.leaseNonce,guard};
 }
 async function currentWindow(actorUUID,callerId){
  id(actorUUID,'actor');const session=await getSession(),checkpoint=session?.activityCheckpoint;
  if(session?.status!=='running'||checkpoint?.phase!=='open')throw Error('activity-checkpoint-closed');
  const binding=activityCheckpointBinding(Object.fromEntries(['id','sessionId','rootUUID','epoch','observationNonce','from'].map(key=>[key,checkpoint[key]]))),options=await checkpointActor(binding,actorUUID,callerId);
  if(typeof options.leaseNonce!=='string'||session.driver?.leaseNonce!==options.leaseNonce)throw Error('session-driver-required');
  const actor=await fromUuid(actorUUID),activities=await getActivities?.()??[],latest=await getSession();
  if(!options.guard()||latest?.status!=='running'||latest.activityCheckpoint?.phase!=='open'||!sameActivityCheckpoint(latest.activityCheckpoint,binding)||latest.driver?.leaseNonce!==options.leaseNonce||!latest.actorUUIDs.includes(actorUUID))throw Error('activity-checkpoint-changed');
  return {binding,phase:'open',actor:{actorUUID,name:actor.name??actorUUID},budgetEndsAt:latest.budgetEndsAt,dependencies:activities.filter(a=>a.actorUUID===actorUUID&&a.sessionId===binding.sessionId&&['planned','started','confirmed'].includes(a.state)&&(!a.executor||a.executor.state==='settled')).slice(-128).map(a=>({id:a.id,label:a.label??a.options?.label??a.providerId,state:a.state}))};
 }
 async function receive(event,callerId){
  event=normalizeManualDeclaration(event);
  if(event.checkpointBinding){
   let options;
   try{
    options=await checkpointActor(event.checkpointBinding,event.actorUUID,callerId);
    if(typeof enrollCheckpointActivity!=='function')throw Error('activity-checkpoint-unavailable');
   }catch(error){error.declarationRejected=true;throw error}
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
  const gm=game.user,user=game.users.get(callerId);
  if(!isActiveGM(game)||!gm?.active||!user?.active)throw Error('manual-actor-not-allowed');
  const actor=await fromUuid(actorUUID);
  const guard=()=>game.user===gm&&isActiveGM(game)&&gm.active===true&&game.users.get(callerId)===user&&user.active===true&&actor?.uuid===actorUUID&&actor.testUserPermission?.(user,'OWNER')===true&&(!game.actors?.get||game.actors.get(actor.id)===actor);
  if(!guard())throw Error('manual-actor-not-allowed');
  if(typeof lookupCheckpointActivity!=='function')throw Error('activity-checkpoint-unavailable');
  return lookupCheckpointActivity(binding,registrationId,{authenticatedCaller:callerId,actorUUID,guard});
 }
 return {register(api){socket=api;socket?.register('exploration:activityCheckpoint',async function(actorUUID){try{return {ok:true,value:await currentWindow(actorUUID,this.socketdata?.userId)}}catch(error){return {ok:false,error:error.message}}});socket?.register('exploration:lookupCheckpointActivity',async function(binding,registrationId,actorUUID){try{return {ok:true,value:await lookup(binding,registrationId,actorUUID,this.socketdata?.userId)}}catch(error){return {ok:false,error:error.message}}});socket?.register('exploration:reserveSource',async function(binding,intent){try{return {ok:true,value:await receiveReservation(binding,intent,this.socketdata?.userId)}}catch(error){return {ok:false,error:error.message}}});socket?.register('exploration:record',async function(event){try{return {ok:true,value:await receive(event,this.socketdata?.userId)}}catch(error){return {ok:false,error:error.message,...error.declarationRejected===true?{declarationRejected:true}:{}}}})},async getActivityCheckpoint(actorUUID){
  id(actorUUID,'actor');if(isActiveGM(game))return currentWindow(actorUUID,game.user.id);if(!socket)throw Error('manual-record-socket-unavailable');const response=await socket.executeAsGM('exploration:activityCheckpoint',actorUUID);if(!response?.ok)throw Error(response?.error??'manual-window-unconfirmed');return response.value;
 },async reserveSource(binding,intent){
  if(isActiveGM(game))return receiveReservation(binding,intent,game.user.id);if(!game.users.activeGM?.active&&!binding)return null;if(!socket)throw Error('manual-record-socket-unavailable');const response=await socket.executeAsGM('exploration:reserveSource',binding,intent);if(!response?.ok)throw Error(response?.error??'manual-reservation-unconfirmed');return response.value;
 },async lookupCheckpointActivity(binding,registrationId,actorUUID){
  binding=activityCheckpointBinding(binding);id(registrationId,'registration');id(actorUUID,'actor');
  if(isActiveGM(game))return lookup(binding,registrationId,actorUUID,game.user.id);if(!socket)throw Error('manual-record-socket-unavailable');
  const response=await socket.executeAsGM('exploration:lookupCheckpointActivity',binding,registrationId,actorUUID);if(!response?.ok)throw Error(response?.error??'manual-lookup-unconfirmed');return response.value;
 },async record(event){event=normalizeManualDeclaration(event);if(isActiveGM(game))return receive(event,game.user.id);if(!socket)throw Error('manual-record-socket-unavailable');const response=await socket.executeAsGM('exploration:record',event);if(!response?.ok){const error=Error(response?.error??'manual-record-unconfirmed');if(response?.declarationRejected===true)error.declarationRejected=true;throw error}return response.value}};
}
