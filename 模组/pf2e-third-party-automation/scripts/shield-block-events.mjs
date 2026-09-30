import {MODULE_ID,hasSource} from './rules.mjs';
import {SerialActions} from './runtime.mjs';
import {isActiveGM} from './native-context.mjs';
import {classifyNativeShieldBlock,shieldEncounter} from './reaction-budget.mjs';

const SOURCE='Compendium.pf2e.feats-srd.Item.dSSwRyuhKTq1VubX';
const worlds=new Set(['-','sog','pnvfcgjbf2cjp7gz','ujx5r8oipw7ercdr','team-automation-qa2']);
const prefix=`${MODULE_ID}:shield-event:`,budgetPrefix=`${MODULE_ID}:native-shield:`;
const fields=['actorUuid','tokenUuid','attackerActorUuid','attackerTokenUuid','attackItemUuid','weaponUuid','damageMessageId','rollIndex'];
const values=c=>Array.from(c?.values?.()??c??[]),records=actor=>actor?.flags?.[MODULE_ID]?.shieldBlockEvents?.records??[];
const messageOptions=message=>Array.isArray(message?.flags?.pf2e?.context?.options)?message.flags.pf2e.context.options:[];
const authorId=message=>message?.author?.id??message?.user?.id??message?.user;
const feature=actor=>values(actor?.items).some(item=>hasSource(item,SOURCE));
const ready=actor=>{const s=actor?.attributes?.shield;return !!(actor?.hitPoints&&s?.itemId&&s.raised&&!s.broken&&!s.destroyed)};
const random=()=>globalThis.foundry?.utils?.randomID?.(24)??globalThis.crypto.randomUUID();
async function fingerprint(message){
 const pf=message.flags.pf2e,applied=structuredClone(pf.appliedDamage??null);
 // Undo is not another trigger and does not undo an already used free action.
 if(applied)delete applied.isReverted;
 const input=JSON.stringify({author:authorId(message),speaker:message.speaker,content:message.content,context:pf.context,applied,custom:message.flags[MODULE_ID]?.shieldBlock??null});
 const digest=await globalThis.crypto.subtle.digest('SHA-256',new TextEncoder().encode(input));
 return [...new Uint8Array(digest)].map(byte=>byte.toString(16).padStart(2,'0')).join('');
}

/** Observe a native block independently from its optional reaction accounting. */
export function createShieldBlockEvents({game,fromUuid=globalThis.fromUuid,resolveSource,validateSource,onConfirmed=async()=>{},onError=console.error}={}){
 const queue=new SerialActions(),scopes=new Map(),dispatches=new Map(),liveClaims=new Map(),scopeKey=Symbol('shield-block-event');let socket;
 const eligible=actor=>worlds.has(game.world?.id)&&actor?.type==='character'&&feature(actor);
 const owner=(actor,user)=>{if(!isActiveGM(game)||!user||!actor?.testUserPermission?.(user,'OWNER'))throw Error('格挡事件需要当前主GM验证原操作者。')};
 async function documents(payload,user){
  const actor=await fromUuid(payload.actorUuid),token=await fromUuid(payload.tokenUuid);owner(actor,user);
  if(token?.documentName!=='Token'||token.actor?.uuid!==actor.uuid)throw Error('格挡事件的角色与Token不匹配。');
  return {actor,token};
 }
 async function persist(actor,record){
  if(!isActiveGM(game))throw Error('主GM已改变，不能保存格挡事件。');
  const prior=records(actor),next=[...prior.filter(r=>r.nonce!==record.nonce),record];
  const unfinished=next.filter(r=>['pending','uncertain'].includes(r.status)||r.status==='confirmed'&&!r.delivered);
  const history=next.filter(r=>!unfinished.includes(r)).slice(-64);
  await actor.update({[`flags.${MODULE_ID}.shieldBlockEvents.records`]:[...unfinished,...history]});
 }
 async function begin(payload,user){
  const {actor,token}=await documents(payload,user);
  return queue.run(actor.uuid,async()=>{
   owner(actor,user);
   if(!eligible(actor)||!ready(actor)||actor.attributes.shield.itemId!==payload.shieldId||!/^[A-Za-z0-9-]{8,80}$/.test(payload.nonce??''))throw Error('本次格挡事件的专长、盾牌或编号已失效。');
   if(records(actor).some(r=>r.nonce===payload.nonce))throw Error('本次格挡事件已登记。');
   const source=await validateSource({game,fromUuid,snapshot:payload.sourceSnapshot});owner(actor,user);
   if(!source?.verified||fields.some(field=>source[field]!==payload[field]))throw Error('格挡来源快照未通过主GM验证。');
   if(!ready(actor)||actor.attributes.shield.itemId!==payload.shieldId||token.actor?.uuid!==actor.uuid)throw Error('保存事件前盾牌状态已改变。');
   const record={...Object.fromEntries(fields.map(field=>[field,source[field]])),sourceSnapshot:structuredClone(source.sourceSnapshot??payload.sourceSnapshot),nonce:payload.nonce,shieldId:payload.shieldId,userId:user.id,epoch:shieldEncounter(actor,token,game)?.epoch??null,status:'pending',createdAt:Date.now()};
   liveClaims.set(record.nonce,user.id);
   try{await persist(actor,record);owner(actor,user);return record}
   catch(error){liveClaims.delete(record.nonce);throw error}
  });
 }
 function matchingMessage(message,record){
  const options=messageOptions(message);
  return !!(message?.id&&game.messages.get(message.id)===message&&authorId(message)===record.userId&&message.speaker?.actor===record.actorUuid.split('.').at(-1)&&`Scene.${message.speaker?.scene}.Token.${message.speaker?.token}`===record.tokenUuid&&message.flags?.pf2e?.context?.type==='damage-taken'&&Array.isArray(options)&&options.filter(o=>typeof o==='string'&&o.startsWith(prefix)).length===1&&options.includes(prefix+record.nonce));
 }
 async function prove(actor,token,record,message,user){
  const source=await validateSource({game,fromUuid,snapshot:record.sourceSnapshot});owner(actor,user);
  if(!source?.verified||fields.some(field=>source[field]!==record[field]))throw Error('格挡完成时来源证据已改变。');
  const markers=message.flags.pf2e.context.options.filter(o=>typeof o==='string'&&o.startsWith(budgetPrefix));
  if(markers.length>1)throw Error('格挡结果带有多个反应认领。');
  const blocked=classifyNativeShieldBlock(message,{game,token,shieldId:record.shieldId,nativeBlockNonce:markers[0]?.slice(budgetPrefix.length)??null});
  const next={...record,status:blocked===true?'confirmed':blocked===false?'not-blocked':'uncertain',blockMessageId:message.id,blockFingerprint:await fingerprint(message)};
  // Awaited source/digest work must not validate a card that changed in between.
  owner(actor,user);if(!matchingMessage(message,record)||game.messages.get(message.id)!==message||await fingerprint(message)!==next.blockFingerprint)throw Error('格挡结果卡在验证期间已改变。');
  await persist(actor,next);return blocked===true?next:null;
 }
 async function finish(payload,user){
  const {actor,token}=await documents(payload,user);
  const event=await queue.run(actor.uuid,async()=>{
   owner(actor,user);const record=records(actor).find(r=>r.nonce===payload.nonce);
   if(!record||record.userId!==user.id||record.tokenUuid!==token.uuid)throw Error('格挡事件认领不存在或所有者不符。');
   if(record.status==='confirmed')return record;
   if(!['pending','uncertain'].includes(record.status))return null;
   if(!payload.applied||payload.uncertain||!payload.entered||!payload.messageId){await persist(actor,{...record,status:payload.entered||payload.uncertain?'uncertain':'unobserved'});return null;}
   const message=game.messages.get(payload.messageId);if(!matchingMessage(message,record))throw Error('格挡结果卡与认领不匹配。');
   if(values(game.messages).filter(m=>matchingMessage(m,record)).length!==1)throw Error('同一格挡事件存在多张结果卡，不能选择其一。');
   return prove(actor,token,record,message,user);
  }).finally(()=>liveClaims.delete(payload.nonce));
  if(event)await dispatch(actor,event,user);
  return !!event;
 }
 async function dispatch(actor,event,user){
  if(event.delivered)return;
  if(dispatches.has(event.nonce))return dispatches.get(event.nonce);
  // The native pipeline and both short persistence locks have been released.
  const work=(async()=>{
   const result=await onConfirmed(structuredClone(event));
   if(['offered','claimed','uncertain'].includes(result?.status))return result;
   await queue.run(actor.uuid,async()=>{owner(actor,user);const latest=records(actor).find(r=>r.nonce===event.nonce);if(latest?.status==='confirmed')await persist(actor,{...latest,delivered:true})});
   return result;
  })();
  dispatches.set(event.nonce,work);
  try{return await work}finally{if(dispatches.get(event.nonce)===work)dispatches.delete(event.nonce)}
 }
 async function rpc(method,payload){
  if(isActiveGM(game))return (method==='begin'?begin:finish)(payload,game.user);
  if(!socket||!game.users.activeGM)throw Error('格挡后的自动卸武需要在线主GM。');
  const result=await socket.executeAsUser(`shield-block-events:${method}`,game.users.activeGM.id,payload);
  if(!result?.ok)throw Error(result?.error??'格挡事件验证失败。');return result.value;
 }
 async function beforeDamage(actor,params){
  const damage=typeof params.damage==='number'?params.damage:params.damage?.total,token=params.token?.document??params.token;
  if(!eligible(actor)||!params.shieldBlockRequest||params.final||!Number.isFinite(damage)||damage<=0||!ready(actor)||token?.actor?.uuid!==actor.uuid)return null;
  if(!actor.testUserPermission?.(game.user,'OWNER'))throw Error('无权使用这个角色的格挡。');
  const source=await resolveSource({game,fromUuid,actor,token,params});if(!source?.verified)return null;
  const payload={...Object.fromEntries(fields.map(field=>[field,source[field]])),sourceSnapshot:source.sourceSnapshot,nonce:random(),shieldId:actor.attributes.shield.itemId};
  const record=await rpc('begin',payload),scope={record,actor,token,entered:false,messageId:null};scopes.set(record.nonce,scope);
  return {params:{...params,[scopeKey]:record.nonce},receipt:{nonce:record.nonce}};
 }
 function wrapNativeDamage(actor,params,apply){
  const scope=scopes.get(params[scopeKey]);if(!scope)return apply(params);
  const live=scope.token.actor;
  if(actor.uuid!==scope.record.actorUuid||live?.uuid!==actor.uuid||!ready(live)||live.attributes.shield.itemId!==scope.record.shieldId||actor.attributes?.shield?.itemId!==scope.record.shieldId)return apply(params);
  scope.entered=true;return apply({...params,rollOptions:new Set([...params.rollOptions??[],prefix+scope.record.nonce])});
 }
 function capture(message,_options,creator){
  const options=messageOptions(message),markers=options.filter(o=>typeof o==='string'&&o.startsWith(prefix));if(markers.length!==1)return;
  const scope=scopes.get(markers[0].slice(prefix.length));if(!scope||!scope.entered||scope.messageId||creator!==game.user.id||!matchingMessage(message,scope.record))return;
  scope.messageId=message.id;
 }
 async function afterDamage(receipt,{applied,uncertain}){
  const scope=scopes.get(receipt?.nonce);if(!scope)return;
  try{await rpc('finish',{nonce:scope.record.nonce,actorUuid:scope.record.actorUuid,tokenUuid:scope.record.tokenUuid,entered:scope.entered,messageId:scope.messageId,applied:!!applied,uncertain:!!uncertain})}
  finally{scopes.delete(scope.record.nonce)}
 }
 async function validateConfirmed(event){
  if(!isActiveGM(game)||!event?.nonce)return false;
  const actor=await fromUuid(event.actorUuid),record=records(actor).find(r=>r.nonce===event.nonce);
  if(record?.status!=='confirmed'||[...fields,'blockMessageId','shieldId','userId','epoch'].some(field=>event[field]!==record[field]))return false;
  const message=game.messages.get(record.blockMessageId);if(!matchingMessage(message,record)||await fingerprint(message)!==record.blockFingerprint)return false;
  const source=await validateSource({game,fromUuid,snapshot:record.sourceSnapshot});return !!source?.verified&&fields.every(field=>source[field]===record[field]);
 }
 async function maintain(actor){
  if(!isActiveGM(game)||!actor)return;
  const pending=records(actor).filter(r=>['pending','uncertain'].includes(r.status)&&!liveClaims.has(r.nonce));
  // Native 8.5 creates its damage card after HP, shield and persistent updates.
  // A fresh coordinator can use that exact card if its owner lost the finish RPC.
  // Index this actor's candidate nonces once, never rescan chat per record.
  if(pending.length){
   const candidates=new Map(pending.map(r=>[r.nonce,[]]));
   for(const message of values(game.messages)){
    const markers=messageOptions(message).filter(o=>typeof o==='string'&&o.startsWith(prefix));
    if(markers.length===1)candidates.get(markers[0].slice(prefix.length))?.push(message);
   }
   for(const prior of pending)await queue.run(actor.uuid,async()=>{
    if(!isActiveGM(game))return;
    const record=records(actor).find(r=>r.nonce===prior.nonce);if(!record||!['pending','uncertain'].includes(record.status)||liveClaims.has(record.nonce))return;
    const user=game.users.get(record.userId);if(!user||!actor.testUserPermission?.(user,'OWNER'))return;
    const token=await fromUuid(record.tokenUuid);owner(actor,user);
    if(token?.actor?.uuid!==actor.uuid)return;
    const matches=(candidates.get(record.nonce)??[]).filter(m=>matchingMessage(m,record));
    if(matches.length!==1){if(record.status==='pending')await persist(actor,{...record,status:'uncertain'});return;}
    await prove(actor,token,record,matches[0],user);
   });
  }
  for(const event of records(actor).filter(record=>record.status==='confirmed'&&!record.delivered)){
   if(!isActiveGM(game))return;
   const user=game.users.get(event.userId);if(!user||!actor.testUserPermission?.(user,'OWNER'))continue;
   if(await validateConfirmed(event))await dispatch(actor,event,user);
  }
 }
 function register({Hooks,socket:api}={}){
  socket=api;
  for(const method of ['begin','finish'])socket?.register(`shield-block-events:${method}`,async function(payload){try{return {ok:true,value:await (method==='begin'?begin:finish)(payload,game.users.get(this.socketdata.userId))}}catch(error){return {ok:false,error:error.message}}});
  const id=Hooks.on('createChatMessage',capture),userId=Hooks.on('updateUser',user=>{
   if(!isActiveGM(game))liveClaims.clear();
   else if(user?.active===false)for(const [nonce,ownerId]of liveClaims)if(ownerId===user.id)liveClaims.delete(nonce);
  });return()=>{Hooks.off('createChatMessage',id);Hooks.off('updateUser',userId)};
 }
 return {beforeDamage,afterDamage,wrapNativeDamage,validateConfirmed,maintain,register};
}
