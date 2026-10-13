import {isActiveGM} from './document-store.mjs';
import {normalizeManualDeclaration} from './manual-time.mjs';
import {activityCheckpointBinding,sameActivityCheckpoint,id} from './schema.mjs';
import {createOwnerTransport,OWNER_TRANSPORT_PROTOCOL} from './owner-transport.mjs';
const driverOperation='manual-record-driver';
const driverCommands=new Set(['window','declaration']);
/** Only a human's generic declaration crosses this bridge, never treatment credentials. */
export function createManualRecordBridge({game,fromUuid,getSession,getActivities,checkpointContext,ownsSession,enrollCheckpointActivity,lookupCheckpointActivity,timeoutMs=10000}){
 let socket,transport,registered=false,generation=0,clientNonce=crypto.randomUUID();const handled=new Map();
 function localScope(){if(!isActiveGM(game)||game.user.active!==true)return null;const scope=checkpointContext?.();return typeof scope?.leaseNonce==='string'?{...scope,user:game.user,generation}:null}
 async function currentDriver(scope=localScope()){
  if(!scope)return null;const session=await getSession(),now=localScope();
  if(!now||scope.generation!==generation||scope.user!==game.user||now.sessionId!==scope.sessionId||now.leaseNonce!==scope.leaseNonce||session?.id!==scope.sessionId||session.status!=='running'||session.driver?.userId!==game.user.id||session.driver?.leaseNonce!==scope.leaseNonce||ownsSession?.(session,scope)!==true)return null;
  return scope;
 }
 async function onDriverPacket(packet,senderId){
  if(packet.operationId!==driverOperation||packet.kind!=='continuation'||!driverCommands.has(packet.status)||packet.ownerUserId!==senderId||packet.driverUserId!==game.user.id||!packet.ownerClientNonce||!localScope())return;
  const key=`${senderId}:${packet.ownerClientNonce}:${packet.requestId}`;if(handled.has(key))return;
  const scope=await currentDriver();if(!scope||handled.has(key))return;
  if(handled.size>=256){const oldest=[...handled].find(([,done])=>done);if(!oldest)return;handled.delete(oldest[0])}handled.set(key,false);
  let value,error;
  try{
   const command=packet.command,binding=packet.status==='declaration'?command?.event?.checkpointBinding:null;
   if(!command||command.actorUUID!==packet.actorUUID||packet.status!=='window'&&!binding||packet.status==='declaration'&&command.event.actorUUID!==packet.actorUUID||binding&&(packet.sessionId!==binding.sessionId||packet.rootUUID!==binding.rootUUID||packet.epoch!==binding.epoch))throw Error('manual-driver-request-mismatch');
   if(packet.status==='window')value=await currentWindow(command.actorUUID,senderId);
   else value=await receive(command.event,senderId);
  }catch(failure){error=failure}
  if(scope.generation!==generation)return;handled.set(key,true);
  // A restored or superseded tab must never win the reply race, even when its
  // earlier private read or attempted declaration produced an error.
  if(!await currentDriver(scope))return;
  const route=Object.fromEntries(['protocol','operationId','requestId','driverUserId','ownerUserId','ownerClientNonce','actorUUID','sessionId','rootUUID','epoch'].filter(field=>Object.hasOwn(packet,field)).map(field=>[field,packet[field]]));
  transport.send({...route,kind:error?'denied':'continuation-ack',status:packet.status,receiverUserId:senderId,...error?{errorCode:error.message,...error.declarationRejected===true?{proof:{declarationRejected:true}}:{}}:{proof:value}});
 }
 function registerDriverTransport(){
  const native=game.socket;if(native&&['on','off','emit'].every(method=>typeof native[method]==='function'))transport=createOwnerTransport({game,onPacket:onDriverPacket,timeoutMs});
 }
 async function requestDriver(status,actorUUID,command,binding){
  if(!transport)throw Error('manual-record-native-unavailable');const gm=game.users.activeGM,nonce=clientNonce,version=generation;
  if(!gm?.active||gm.isGM!==true)throw Error('active-gm-required');
  const response=await transport.request({protocol:OWNER_TRANSPORT_PROTOCOL,kind:'continuation',operationId:driverOperation,status,requestId:crypto.randomUUID(),receiverUserId:gm.id,driverUserId:gm.id,ownerUserId:game.user.id,ownerClientNonce:nonce,actorUUID,command,...binding?{sessionId:binding.sessionId,rootUUID:binding.rootUUID,epoch:binding.epoch}:{}},{expectedSenderId:gm.id,matches:packet=>version===generation&&nonce===clientNonce&&game.users.activeGM===gm&&gm.active===true&&packet.operationId===driverOperation&&packet.status===status&&(packet.kind==='denied'||packet.kind==='continuation-ack')});
  if(response.kind==='denied'){const error=Error(response.errorCode??'manual-driver-unconfirmed');if(response.proof?.declarationRejected===true)error.declarationRejected=true;throw error}return response.proof;
 }
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
 // A declaration is only accepted for a registration window; there is no other record to add it to.
 function windowDeclaration(event){event=normalizeManualDeclaration(event);if(!event.checkpointBinding)throw Error('activity-checkpoint-closed');return event}
 async function receive(event,callerId){
  event=windowDeclaration(event);let options;
  try{
   options=await checkpointActor(event.checkpointBinding,event.actorUUID,callerId);
   if(typeof enrollCheckpointActivity!=='function')throw Error('activity-checkpoint-unavailable');
  }catch(error){error.declarationRejected=true;throw error}
  return enrollCheckpointActivity(event.checkpointBinding,event,options);
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
 return {register(api){socket=api;if(!registered){registered=true;registerDriverTransport()}socket?.register('exploration:activityCheckpoint',async function(actorUUID){try{return {ok:true,value:await currentWindow(actorUUID,this.socketdata?.userId)}}catch(error){return {ok:false,error:error.message}}});socket?.register('exploration:lookupCheckpointActivity',async function(binding,registrationId,actorUUID){try{return {ok:true,value:await lookup(binding,registrationId,actorUUID,this.socketdata?.userId)}}catch(error){return {ok:false,error:error.message}}});socket?.register('exploration:record',async function(event){try{return {ok:true,value:await receive(event,this.socketdata?.userId)}}catch(error){return {ok:false,error:error.message,...error.declarationRejected===true?{declarationRejected:true}:{}}}})},invalidate(){generation++;clientNonce=crypto.randomUUID();transport?.dispose();transport=null;handled.clear();if(registered)registerDriverTransport()},async getActivityCheckpoint(actorUUID){
  id(actorUUID,'actor');if(await currentDriver())return currentWindow(actorUUID,game.user.id);return requestDriver('window',actorUUID,{actorUUID});
 },async lookupCheckpointActivity(binding,registrationId,actorUUID){
  binding=activityCheckpointBinding(binding);id(registrationId,'registration');id(actorUUID,'actor');
  if(isActiveGM(game))return lookup(binding,registrationId,actorUUID,game.user.id);if(!socket)throw Error('manual-record-socket-unavailable');
  const response=await socket.executeAsGM('exploration:lookupCheckpointActivity',binding,registrationId,actorUUID);if(!response?.ok)throw Error(response?.error??'manual-lookup-unconfirmed');return response.value;
 },async record(event){event=windowDeclaration(event);if(await currentDriver())return receive(event,game.user.id);return requestDriver('declaration',event.actorUUID,{actorUUID:event.actorUUID,event},event.checkpointBinding)}};
}
