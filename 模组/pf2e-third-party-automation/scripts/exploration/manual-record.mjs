import {isActiveGM} from './document-store.mjs';
/** Only a human's generic declaration crosses this bridge, never treatment credentials. */
export function createManualRecordBridge({game,fromUuid,getSession,observe}){
 let socket;
 async function receive(event,callerId){
  if(!isActiveGM(game))throw Error('active-gm-required');const user=game.users.get(callerId);if(!user?.active)throw Error('caller-offline');
  const actor=await fromUuid(event?.actorUUID),session=await getSession();
  if(!isActiveGM(game)||!actor?.testUserPermission(user,'OWNER')||session?.status!=='recording'||!session.actorUUIDs.includes(actor.uuid))throw Error('manual-actor-not-allowed');
  const label=typeof event.label==='string'?event.label.trim():'';if(!label||label.length>200||!Number.isFinite(event.durationSeconds)||event.durationSeconds<=0)throw Error('invalid-manual-activity');
  return observe({id:crypto.randomUUID(),actorUUID:actor.uuid,patientUUIDs:[],kind:'activity',label,durationSeconds:event.durationSeconds,source:{type:'user-record',userId:callerId,unverified:true},missing:['manual-source-requires-review']});
 }
 return {register(api){socket=api;socket?.register('exploration:record',async function(event){try{return {ok:true,value:await receive(event,this.socketdata?.userId)}}catch(error){return {ok:false,error:error.message}}})},async record(event){if(isActiveGM(game))return receive(event,game.user.id);if(!socket)throw Error('manual-record-socket-unavailable');const response=await socket.executeAsGM('exploration:record',{actorUUID:event.actorUUID,label:event.label,durationSeconds:event.durationSeconds});if(!response?.ok)throw Error(response?.error??'manual-record-unconfirmed');return response.value}};
}
