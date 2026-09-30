import {MODULE_ID} from './rules.mjs';
import {isActiveGM,markUnappliedDamageError} from './native-context.mjs';
import {isCurrentDisruptToken} from './disrupt-prey-rules.mjs';
import {withReactionReservation} from './reaction-budget.mjs';

const sourcePrefix=`${MODULE_ID}:source:`,damagePrefix=`${MODULE_ID}:disrupt-damage:`,applyPrefix=`${MODULE_ID}:disrupt-apply:`;
const stateOf=actor=>actor?.flags?.[MODULE_ID]?.disruptPrey??{events:[],reactions:[]};
const authorId=message=>message?.author?.id??message?.user?.id??message?.user;
const speakerToken=message=>`Scene.${message?.speaker?.scene}.Token.${message?.speaker?.token}`;
const strings=value=>{try{return Array.from(value??[]).filter(v=>typeof v==='string');}catch{return [];}};
const automaticOption=option=>option.startsWith(damagePrefix)||option.startsWith(applyPrefix);
const fail=reason=>{throw markUnappliedDamageError(Error(`扰乱狩猎自动伤害未再次应用：${reason}`));};
const serializedRoll=roll=>{
 if(!roll||roll._evaluated!==true||!Number.isFinite(roll.total)||typeof roll.toJSON!=='function')fail('缺少已评估原生伤害骰子。');
 const value=roll.toJSON();if(!value||value.evaluated!==true||!Number.isFinite(value.total))fail('原生伤害骰子无法核验。');
 return JSON.stringify(value);
};

/**
 * One guard instance is shared by the executor and the outermost applyDamage
 * wrapper. Authorization durably burns the claim before issuing a local,
 * nonserializable permission tied to the exact params and contextual actor.
 * Neither native failure, revoke, reconnect nor Undo rearms that permission.
 */
export function createDisruptPreyDamageGuard({game,getRollContext=()=>null,DamageRoll}={}){
 const grants=new WeakMap(),protectedIds=new Set();
 const gm=()=>{if(!isActiveGM(game)||!game.user?.isGM||game.users.get(game.user.id)?.isGM!==true)fail('当前客户端不是本次执行的主 GM。');};
 const rollClass=()=>DamageRoll??game?.pf2e?.DamageRoll??globalThis.CONFIG?.Dice?.rolls?.find(cls=>typeof cls==='function'&&cls.name==='DamageRoll');
 const stampOf=claim=>({userId:claim.applicationUserId,damageMessageId:claim.damageMessageId,targetTokenUuid:claim.targetUuid});
 const sameStamp=(a,b)=>!!a&&['userId','damageMessageId','targetTokenUuid'].every(key=>a[key]===b[key]);
 function sourceIds(params){
  const tracked=getRollContext(params?.damage),ids=new Set(tracked?.messageId?[tracked.messageId]:[]);
  for(const option of strings(params?.rollOptions))if(option.startsWith(sourcePrefix)){
   const match=/^([^:]+):(0|[1-9]\d*)$/.exec(option.slice(sourcePrefix.length));if(match)ids.add(match[1]);
  }
  return ids;
 }
 function protectedCard(message,id){
  if(protectedIds.has(id))return true;
  const marked=strings(message?.flags?.pf2e?.context?.options).some(automaticOption);
  const claimed=stateOf(message?.actor).reactions?.some(claim=>claim.damageMessageId===id&&claim.actorUuid===message.actor.uuid&&claim.claimKey===`disrupt:${claim.nonce}`);
  if(marked||claimed){protectedIds.add(id);return true;}return false;
 }
 function protectedApplication(params){
  if(strings(params?.rollOptions).some(automaticOption))return true;
  for(const id of sourceIds(params)){
   if(protectedCard(game.messages?.get(id),id))return true;
   // A pending native call may still hold its weapon after the source chat card
   // was removed. Inspect that one source actor, never scan all world actors.
   if(stateOf(params?.item?.actor).reactions?.some(claim=>claim.damageMessageId===id&&claim.actorUuid===params.item.actor.uuid&&claim.claimKey===`disrupt:${claim.nonce}`))return true;
  }
  return false;
 }
 function inspect({reactor,actor,message,target,claim:requested,params}){
  gm();
  if(!reactor?.uuid||game.messages?.get(message?.id)!==message||message.actor!==reactor||!isCurrentDisruptToken(target,game)||target.actor.uuid!==actor?.uuid)fail('角色、聊天卡或目标不是当前准确文档。');
  const claim=stateOf(reactor).reactions?.find(r=>r.nonce===requested?.nonce&&r.eventId===requested?.eventId);
  if(!claim||claim.actorUuid!==reactor.uuid||claim.actorId!==reactor.id||claim.claimKey!==`disrupt:${claim.nonce}`||!/^[A-Za-z0-9_-]{1,80}$/.test(claim.nonce)||claim.state!=='applying'||claim.applicationUserId!==game.user.id||![2,3].includes(claim.degree)||!claim.checkId||claim.damageMessageId!==message.id||claim.targetUuid!==target.uuid||claim.targetActorUuid!==target.actor.uuid||claim.targetActorId!==target.actor.id)fail('本次持久伤害认领不处于当前 GM 的应用阶段。');
  const pf=message.flags?.pf2e,context=pf?.context,options=strings(context?.options),author=game.users.get(claim.userId);
  if(authorId(message)!==claim.userId||!author||!reactor.testUserPermission?.(author,'OWNER')||message.speaker?.actor!==claim.actorId||speakerToken(message)!==claim.tokenUuid||context?.type!=='damage-roll'||pf.origin?.actor!==reactor.uuid||pf.origin?.uuid!==claim.itemUuid||context.target?.token!==target.uuid||context.target?.actor!==target.actor.uuid||options.filter(o=>o.startsWith(damagePrefix)).length!==1||!options.includes(damagePrefix+claim.nonce)||!options.includes(`${MODULE_ID}:bear-attack:${claim.checkId}`)||options.includes('action:reaction')||options.includes('trait:reaction'))fail('原生伤害卡来源或原攻击标识不符。');
  const sourceToken=game.scenes?.get(message.speaker.scene)?.tokens?.get(message.speaker.token);
  if(!isCurrentDisruptToken(sourceToken,game)||sourceToken.actor!==reactor||sourceToken.parent!==target.parent)fail('原攻击 Token 已失效。');
  const NativeRoll=rollClass(),roll=message.rolls?.[0];
  if(typeof NativeRoll!=='function'||!(roll instanceof NativeRoll)||!(params?.damage instanceof NativeRoll)||serializedRoll(roll)!==JSON.stringify(claim.damageRoll)||params.damage.total!==roll.total)fail('伤害骰子与已确认的原生结果不符。');
  serializedRoll(params.damage);
  const actualOptions=strings(params.rollOptions),token=params.token?.document??params.token,tracked=getRollContext(params.damage);
  if(token!==target||params.item?.uuid!==claim.itemUuid||params.item?.actor?.uuid!==reactor.uuid||params.skipIWR!==false||params.final===true||params.shieldBlockRequest!==false||params.outcome!==context.outcome||actualOptions.filter(o=>o.startsWith(sourcePrefix)).length!==1||!actualOptions.includes(`${sourcePrefix}${message.id}:0`)||actualOptions.filter(o=>o.startsWith(applyPrefix)).length!==1||!actualOptions.includes(applyPrefix+claim.nonce)||tracked&&(tracked.messageId!==message.id||tracked.rollIndex!==0))fail('本次应用参数、免抗设置或来源标记不符。');
  return claim;
 }
 function snapshot(params,message){
  return {entries:Reflect.ownKeys(params).map(key=>[key,params[key]]),damage:serializedRoll(params.damage),options:JSON.stringify(strings(params.rollOptions)),originalRoll:message.rolls[0]};
 }
 function unchanged(params,message,before){
  if(Reflect.ownKeys(params).length!==before.entries.length||before.entries.some(([key,value])=>params[key]!==value)||serializedRoll(params.damage)!==before.damage||JSON.stringify(strings(params.rollOptions))!==before.options||message.rolls[0]!==before.originalRoll)fail('签发后应用参数发生变化。');
 }
 async function authorize(input){
  // Mutable performer state is only an identity selector. Every permission and
  // mechanical field is read from the real actor flag inside the shared lock.
  const request={...input,claim:{nonce:input.claim?.nonce,eventId:input.claim?.eventId}};
  return withReactionReservation(request.reactor,game,async()=>{
   const claim=inspect(request);if(Object.hasOwn(claim,'applicationGuard'))fail('本次伤害已经签发过一次性应用许可。');
   const before=snapshot(request.params,request.message),stamp=stampOf(claim),state=structuredClone(stateOf(request.reactor));
   state.reactions.find(r=>r.nonce===claim.nonce&&r.eventId===claim.eventId).applicationGuard=stamp;
   gm();await request.reactor.update({[`flags.${MODULE_ID}.disruptPrey`]:state});
   const current=inspect(request);unchanged(request.params,request.message,before);if(!sameStamp(current.applicationGuard,stamp))fail('持久应用许可发生变化。');
   const grant={request,before,stamp};grants.set(request.params,grant);protectedIds.add(request.message.id);
   return()=>{if(grants.get(request.params)===grant)grants.delete(request.params);};
  });
 }
 async function applyDamage(actor,params,apply){
  const grant=params&&typeof params==='object'?grants.get(params):null;
  if(grant){
   // Consume synchronously before any provider, RPC, callback or native await.
   // Even a failed validation cannot be retried with the old local permission.
   grants.delete(params);if(actor!==grant.request.actor)fail('一次性许可属于另一份伤害上下文。');
   const claim=inspect(grant.request);unchanged(params,grant.request.message,grant.before);if(!sameStamp(claim.applicationGuard,grant.stamp))fail('持久应用许可已失效。');
   let open=true,enteredNative=false;
   const assertNative=()=>{
    if(!open)fail('本次原生伤害执行范围已经结束，许可失效。');
    if(enteredNative)fail('本次原生伤害只能进入一次，不能重复调用。');
    enteredNative=true;
    const current=inspect(grant.request);unchanged(params,grant.request.message,grant.before);
    if(!sameStamp(current.applicationGuard,grant.stamp))fail('持久应用许可已失效。');
   };
   // The main wrapper calls this private callback immediately before wrapped
   // native damage, after all provider/defense awaits, without another wrapper.
   try{return await apply(params,assertNative);}finally{open=false;}
  }
  if(protectedApplication(params))fail('这张自动伤害卡只能由本次内部结算使用，不能手动或重复应用。');
  return apply(params,()=>{});
 }
 return {authorize,applyDamage};
}
