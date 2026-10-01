import {MODULE_ID} from './schema.mjs';

export const OWNER_TRANSPORT_PROTOCOL=`${MODULE_ID}.exploration-owner.v1`;
export const OWNER_TRANSPORT_CHANNEL=`module.${MODULE_ID}`;
const kinds=new Set(['offer','claim','grant','denied','continuation','continuation-ack','cancel','cancel-ack','completion','completion-ack']);
const responses=new Set(['grant','denied','continuation-ack','cancel-ack','completion-ack']);
const fields=new Set(['protocol','kind','requestId','receiverUserId','rootUUID','epoch','sessionId','activityId','operationId','actorUUID','driverUserId','ownerUserId','offerId','ownerClientNonce','attemptNonce','permitNonce','commandDigest','command','completionId','status','errorCode','proof','results','focusBefore','focusAfter','rolledHealing','expiresAt','resourceReceiptIds']);
const identityFields=['rootUUID','epoch','sessionId','activityId','operationId','actorUUID','driverUserId','ownerUserId','offerId','ownerClientNonce','attemptNonce','permitNonce','commandDigest','completionId'];
const privateFields=new Set(['__proto__','constructor','prototype','activity','ledger','sessions','activities','clocks','history','driverClientNonce']);
const identifier=value=>typeof value==='string'&&value.length>0&&value.length<=128&&value.trim()===value&&!/[\u0000-\u001f\u007f]/.test(value);

function snapshotPacket(packet){
 const ancestors=new Set();let nodes=0;
 function copy(value,depth){
  if(++nodes>2048||depth>12)throw Error('owner-packet-too-large');
  if(value===null||typeof value==='boolean')return value;
  if(typeof value==='string'){if(value.length>4096)throw Error('owner-packet-too-large');return value}
  if(typeof value==='number'&&Number.isFinite(value))return value;
  if(!value||typeof value!=='object'||ancestors.has(value))throw Error('invalid-owner-packet-json');
  const array=Array.isArray(value),prototype=Object.getPrototypeOf(value);
  if(array?prototype!==Array.prototype:prototype!==Object.prototype&&prototype!==null)throw Error('invalid-owner-packet-json');
  ancestors.add(value);
  try{
   const keys=Reflect.ownKeys(value),result=array?[]:{};
   if(array&&(value.length>256||keys.length!==value.length+1)||!array&&keys.length>64)throw Error('owner-packet-too-large');
   if(!array&&Object.hasOwn(value,'id')&&Object.hasOwn(value,'sessionId')&&Object.hasOwn(value,'providerId'))throw Error('private-activity-payload-forbidden');
   for(const key of keys){
    if(array&&key==='length')continue;
    if(typeof key!=='string'||key.length>128||privateFields.has(key))throw Error('private-owner-packet-field');
    const descriptor=Object.getOwnPropertyDescriptor(value,key);
    if(!descriptor.enumerable||!Object.hasOwn(descriptor,'value')||array&&(!/^(0|[1-9][0-9]*)$/.test(key)||Number(key)>=value.length))throw Error('invalid-owner-packet-json');
    result[key]=copy(descriptor.value,depth+1);
   }
   return result;
  }finally{ancestors.delete(value)}
 }
 const saved=copy(packet,0);
 if(!saved||Array.isArray(saved)||saved.protocol!==OWNER_TRANSPORT_PROTOCOL||!kinds.has(saved.kind)||!identifier(saved.requestId)||!identifier(saved.receiverUserId))throw Error('invalid-owner-packet-envelope');
 if(Object.keys(saved).some(key=>!fields.has(key)))throw Error('invalid-owner-packet-field');
 for(const key of identityFields)if(Object.hasOwn(saved,key)&&!identifier(saved[key]))throw Error('invalid-owner-packet-identity');
 if(new TextEncoder().encode(JSON.stringify(saved)).length>65536)throw Error('owner-packet-too-large');
 return saved;
}

/** Native recipient routing and response lifetimes only; grants belong to the broker. */
export function createOwnerTransport({game,onPacket=()=>{},onError=error=>console.error(error),timeoutMs=10000}) {
 const socket=game?.socket,userId=game?.user?.id,pending=new Map();let disposed=false;
 if(!identifier(userId)||game.users?.get(userId)?.id!==userId||!socket||['on','off','emit'].some(method=>typeof socket[method]!=='function')||typeof onPacket!=='function'||typeof onError!=='function')throw Error('invalid-owner-transport');
 if(!Number.isSafeInteger(timeoutMs)||timeoutMs<=0||timeoutMs>2147483647)throw Error('invalid-owner-timeout');
 const knownUser=id=>identifier(id)&&game.users.get(id)?.id===id;
 function live(){if(disposed)throw Error('owner-transport-disposed');if(game.user?.id!==userId||game.socket!==socket)throw Error('owner-transport-connection-changed')}
 function report(error){try{onError(error)}catch(failure){console.error(failure)}}
 function finish(entry,error,value){
  if(pending.get(entry.packet.requestId)!==entry)return;
  pending.delete(entry.packet.requestId);clearTimeout(entry.timer);
  if(error)entry.reject(error);else entry.resolve(value);
 }
 function expire(entry){finish(entry,Error('owner-response-timeout-unknown-no-retry'))}
 function emit(packet){live();if(!knownUser(packet.receiverUserId))throw Error('unknown-owner-recipient');socket.emit(OWNER_TRANSPORT_CHANNEL,structuredClone(packet),{recipients:[packet.receiverUserId]},()=>{})}
 function send(packet){live();emit(snapshotPacket(packet))}
 function receive(packet,senderId){
  if(disposed||game.user?.id!==userId||game.socket!==socket||!knownUser(senderId))return;
  if(!packet||typeof packet!=='object'||Object.getOwnPropertyDescriptor(packet,'protocol')?.value!==OWNER_TRANSPORT_PROTOCOL)return;
  let saved;try{saved=snapshotPacket(packet)}catch{return}
  if(saved.receiverUserId!==userId)return;
  if(responses.has(saved.kind)){
   const entry=pending.get(saved.requestId);if(!entry||senderId!==entry.expectedSenderId)return;
   if(performance.now()>=entry.deadline){expire(entry);return}
   const allowed=entry.packet.kind==='claim'?['grant','denied']:entry.packet.kind==='continuation'?['continuation-ack','denied']:entry.packet.kind==='cancel'?['cancel-ack']:['completion-ack','denied'];
   if(!allowed.includes(saved.kind)||identityFields.some(key=>Object.hasOwn(entry.packet,key)&&entry.packet[key]!==saved[key]))return;
   let matched;try{matched=entry.matches(saved)}catch(error){finish(entry,error);return}
   if(matched&&typeof matched.then==='function'){Promise.resolve(matched).catch(report);finish(entry,Error('synchronous-owner-matcher-required'));return}
   if(performance.now()>=entry.deadline){expire(entry);return}
   if(matched===true)finish(entry,null,saved);
   return;
  }
  try{const task=onPacket(saved,senderId);if(task&&typeof task.then==='function')Promise.resolve(task).catch(report)}catch(error){report(error)}
 }
 async function request(packet,{expectedSenderId,matches}={}){
  live();const saved=snapshotPacket(packet);
  if(!['claim','continuation','cancel','completion'].includes(saved.kind)||!knownUser(expectedSenderId)||expectedSenderId!==saved.receiverUserId||typeof matches!=='function'||matches.constructor?.name==='AsyncFunction')throw Error('invalid-owner-request');
  if(pending.has(saved.requestId))throw Error('duplicate-owner-request-pending');
  if(pending.size>=128)throw Error('owner-request-capacity-limit');
  return new Promise((resolve,reject)=>{
   const entry={packet:saved,expectedSenderId,matches,resolve,reject,deadline:performance.now()+timeoutMs};
   pending.set(saved.requestId,entry);entry.timer=setTimeout(()=>expire(entry),timeoutMs);
   try{emit(saved)}catch(error){finish(entry,error)}
  });
 }
 socket.on(OWNER_TRANSPORT_CHANNEL,receive);
 return {send,request,dispose(){if(disposed)return;disposed=true;socket.off(OWNER_TRANSPORT_CHANNEL,receive);for(const entry of [...pending.values()])finish(entry,Error('owner-transport-disposed'))}};
}
