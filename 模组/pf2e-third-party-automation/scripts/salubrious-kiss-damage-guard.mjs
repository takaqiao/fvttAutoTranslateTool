import {MODULE_ID} from './rules.mjs';
import {SerialActions} from './runtime.mjs';
import {markUnappliedDamageError} from './native-context.mjs';
import {assertSource,assertPatient,assertClaimPrivacy,claimOf,sameClaim,marker} from './salubrious-kiss-context.mjs';
import {kissState,treatmentTiers,treatmentOutcome} from './salubrious-kiss-rules.mjs';
import {validateSalubriousCard} from './salubrious-kiss-executor.mjs';

const queue=new SerialActions();
const sourcePrefix=`${MODULE_ID}:source:`,damagePrefix=`${MODULE_ID}:salubrious-damage:`,applyPrefix=`${MODULE_ID}:salubrious-apply:`;
const offGuard=`${MODULE_ID}:offguard-before-damage`;
const strings=value=>Array.from(value??[]).filter(v=>typeof v==='string');
const options=value=>JSON.stringify([...new Set(strings(value))].sort());
const automatic=value=>value.startsWith(damagePrefix)||value.startsWith(applyPrefix);
const fail=reason=>{throw markUnappliedDamageError(Error(`仙露三吻未再次应用：${reason}`));};
const rollData=roll=>{
 if(!roll||roll._evaluated!==true||!Number.isFinite(roll.total)||typeof roll.toJSON!=='function')fail('缺少已评估的原生骰子');
 const data=roll.toJSON();if(data?.evaluated!==true||data.total!==roll.total)fail('原生骰子序列化不符');return data;
};
const resultEqual=(a,b)=>!!a&&!!b&&['checkId','damageId'].every(key=>a[key]===b[key]);
const stampEqual=(a,b)=>!!a&&!!b&&['schema','nonce','userId','actorUuid','checkId','damageId','targetUuid','itemUuid'].every(key=>a[key]===b[key]);

/** One instance belongs to the executor and the existing outer damage wrapper.
 * No wrapper registration, additional HP writes, healing computation or IWR.
 * The input capability is private to the exact params and contextual receiver;
 * its final assertion validates the actual native params after provider awaits. */
export function createSalubriousDamageGuard({game,messagePrivacy,getRollContext=()=>null,DamageRoll,runExclusive=(key,operation)=>queue.run(key,operation)}={}){
 const grants=new WeakMap(),issued=new Set(),protectedIds=new Set();
 const gm=()=>{if(!game.user?.isGM||!game.user.active||game.users.activeGM!==game.user||game.users.get(game.user.id)!==game.user)fail('当前客户端不是本次主 GM');return game.user;};
 const nativeRoll=()=>DamageRoll??game.pf2e?.DamageRoll??globalThis.CONFIG?.Dice?.rolls?.find(cls=>cls.name==='DamageRoll');
 function sourceIds(params){
  const ids=new Set(),tracked=getRollContext(params?.damage);if(typeof tracked?.messageId==='string')ids.add(tracked.messageId);
  for(const option of strings(params?.rollOptions))if(option.startsWith(sourcePrefix)){const match=/^([^:]+):(0|[1-9]\d*)$/.exec(option.slice(sourcePrefix.length));if(match)ids.add(match[1]);}
  return ids;
 }
 function protectedApplication(params){
  if(strings(params?.rollOptions).some(automatic))return true;
  for(const id of sourceIds(params)){
   if(protectedIds.has(id))return true;
   const card=game.messages.get(id),cardProof=card?.flags?.[MODULE_ID]?.salubriousKiss;
   if(cardProof?.kind==='damage'||strings(card?.flags?.pf2e?.context?.options).some(automatic)){protectedIds.add(id);return true;}
   // Exact one source actor only. An already deleted card still cannot be
   // replayed using its persisted claim; do not search actors or recent cards.
   const supplied=params?.item?.actor??card?.actor,actor=game.actors.get(supplied?.id);
   if(actor&&actor.uuid===supplied.uuid&&kissState(actor).claims?.some(c=>c.actorUuid===actor.uuid&&c.result?.damageId===id))return true;
  }
  return false;
 }
 function inspect(request,params=request.params){
  const user=gm(),{reactor,actor,item,token,target,check,message}=request,claim=claimOf(reactor,request.nonce);
  if(!claim||claim.state!=='applying'||claim.actorUuid!==reactor?.uuid||claim.tokenUuid!==token?.uuid||claim.itemUuid!==item?.uuid||claim.targetUuid!==target?.uuid||claim.targetActorUuid!==target?.actor?.uuid||!/^[A-Za-z0-9_-]{1,80}$/.test(claim.nonce??'')||claim.skill!=='occultism'||!Number.isFinite(claim.startedAt)||game.time.worldTime<claim.startedAt||game.time.worldTime>=claim.startedAt+3600)fail('没有准确且仍有效的应用认领');
  assertSource({game,actor:reactor,item,token,user:game.users.get(claim.userId),privacy:claim.privacy});assertPatient({game,actor:reactor,token,target,allowImmune:true,user:game.users.get(claim.userId)});assertClaimPrivacy({game,claim,token,item,target,user:game.users.get(claim.userId)});
  if(actor?.uuid!==target.actor.uuid||actor.id!==target.actor.id||!treatmentTiers(reactor).some(t=>t.tier===claim.tier&&t.dc===claim.dc))fail('接收上下文或原神秘档位不符');
  const application=claim.application,execution=reactor.flags?.[MODULE_ID]?.salubriousKissExecutions?.find(entry=>entry.nonce===claim.nonce),pending=kissState(target.actor).pending;
  if(application?.state!=='started'||application.userId!==user.id||application.damageId!==message?.id||application.targetUuid!==target.uuid||pending?.nonce!==claim.nonce||pending.actorUuid!==reactor.uuid||execution?.state!=='done'||!resultEqual(execution.result,claim.result)||claim.result?.checkId!==check?.id||claim.result?.damageId!==message?.id)fail('患者预留、拥有者骰子或本次持久开始记录不符');
  if(message.actor!==reactor||check.actor!==reactor)fail('原生卡不是当前真实施治者');
  const degree=validateSalubriousCard({game,message:check,claim});if(validateSalubriousCard({game,message,claim,damage:true})!==degree||claim.result.degree!==undefined&&degree!==claim.result.degree||degree===1||message.flags.pf2e.origin.messageId!==check.id)fail('本次检定与治疗骰子来源不一致');
  const roll=message.rolls[0],NativeRoll=nativeRoll(),data=rollData(roll),expected=treatmentOutcome({degree,tier:claim.tier});
  if(typeof NativeRoll!=='function'||!(roll instanceof NativeRoll)||data.formula?.replace(/\s/g,'')!==expected.formula.replace(/\s/g,''))fail('不是本次原生医疗公式');
  if(degree===0){if(!(params?.damage instanceof NativeRoll)||rollData(params.damage).formula!==data.formula||params.damage.total!==roll.total)fail('大失败伤害必须保留确切原生 DamageRoll');}
  else if(typeof params?.damage!=='number'||!Number.isFinite(params.damage)||params.damage!==-roll.total)fail('治疗必须保留原生负数基础结果');
  const actual=strings(params.rollOptions),tracked=getRollContext(params.damage);
  if(params.token!==target||params.item!==item||params.skipIWR!==(degree!==0)||params.final!==false||params.shieldBlockRequest!==false||params.outcome!==message.flags.pf2e.context.outcome||actual.includes('skip-handling-message')||actual.filter(o=>o.startsWith(sourcePrefix)).length!==1||!actual.includes(`${sourcePrefix}${message.id}:0`)||actual.filter(o=>o.startsWith(applyPrefix)).length!==1||!actual.includes(marker('apply',claim))||actual.filter(o=>o.startsWith(damagePrefix)).length!==1||!actual.includes(marker('damage',claim))||!actual.includes(marker('check',claim))||tracked&&(tracked.messageId!==message.id||tracked.rollIndex!==0))fail('本次实际应用参数或来源标识不符');
  return claim;
 }
 function cardEvidence(card){return JSON.stringify({author:card.author?.id??card.author,speaker:card.speaker,blind:card.blind,whisper:card.whisper,pf:card.flags.pf2e,proof:card.flags[MODULE_ID]?.salubriousKiss,roll:rollData(card.rolls[0])});}
 function snapshot(request){const {params,check,message}=request;return {user:game.user,targetActor:request.target.actor,damage:params.damage,damageData:typeof params.damage==='number'?null:JSON.stringify(rollData(params.damage)),options:options(params.rollOptions),checkRoll:check.rolls[0],damageRoll:message.rolls[0],check:cardEvidence(check),message:cardEvidence(message),extra:Reflect.ownKeys(params).filter(k=>k!=='rollOptions'&&k!=='damage').map(k=>[k,params[k]])};}
 function unchanged(request,before,params,{final=false}={}){
  if(game.user!==before.user||request.target.actor!==before.targetActor||request.check.rolls[0]!==before.checkRoll||request.message.rolls[0]!==before.damageRoll||cardEvidence(request.check)!==before.check||cardEvidence(request.message)!==before.message||params.damage!==before.damage||before.damageData!==null&&JSON.stringify(rollData(params.damage))!==before.damageData)fail('签发后来源、患者或真实骰子已改变');
  if(before.extra.some(([key,value])=>params[key]!==value))fail('签发后实际原生参数已改变');
  // main/cycle legitimately spread params and rebuild rollOptions. The only
  // added option allowed here is the proven campaign off-guard observation.
  const actual=strings(params.rollOptions),baseline=JSON.parse(before.options),added=actual.filter(value=>!baseline.includes(value));
  if(options(actual.filter(value=>baseline.includes(value)))!==before.options||added.some(value=>!final||value!==offGuard||!request.actor.hasCondition?.('off-guard')))fail('签发后原生检定选项已改变');
  if(final&&!(params.rollOptions instanceof Set))fail('最终原生 rollOptions 不是 Set');
  // Unknown additional public fields could change the native call in a future
  // system version. Private provider symbols do not alter this signature.
  const known=new Set(['damage','rollOptions',...before.extra.map(([key])=>key)]);
  if(Object.keys(params).some(key=>!known.has(key)))fail('出现未验证的原生应用参数');
 }
 function validateGrant(grant,actor,params,final){
  if(actor!==grant.request.actor)fail('一次性许可属于另一份接收上下文');
  const current=inspect(grant.request,params);unchanged(grant.request,grant.before,params,{final});
  if(!sameClaim(current,grant.claim)||!stampEqual(current.applicationGuard,grant.stamp))fail('持久应用许可或认领已改变');
 }
 async function authorize(input){
  const request={...input,nonce:input.claim?.nonce};delete request.claim;
  return runExclusive(request.reactor?.uuid,async()=>{
   try{
    const claim=inspect(request),key=`${request.reactor.uuid}:${claim.nonce}`;
    if(issued.has(key)||Object.hasOwn(claim,'applicationGuard'))fail('本次应用已经签发或尝试过一次性许可');
    issued.add(key);const before=snapshot(request),stamp={schema:1,nonce:claim.nonce,userId:game.user.id,actorUuid:request.reactor.uuid,checkId:request.check.id,damageId:request.message.id,targetUuid:request.target.uuid,itemUuid:request.item.uuid},identity=structuredClone(claim);
    const claims=structuredClone(kissState(request.reactor).claims);claims.find(c=>c.nonce===claim.nonce).applicationGuard=stamp;
    await request.reactor.update({[`flags.${MODULE_ID}.salubriousKiss.claims`]:claims});
    const grant={request,before,stamp,claim:identity};validateGrant(grant,request.actor,request.params,false);
    grants.set(request.params,grant);protectedIds.add(request.message.id);
    return()=>{if(grants.get(request.params)===grant)grants.delete(request.params);};
   }catch(error){throw markUnappliedDamageError(error);}
  });
 }
 async function applyDamage(actor,params,continuation){
  const grant=params&&typeof params==='object'?grants.get(params):null;
  if(!grant){if(protectedApplication(params))fail('这张自动医疗卡只能由本次内部结算使用');return continuation(params,()=>{});}
  grants.delete(params); // consume synchronously before any provider/async wait
  let open=true,entered=false;
  try{
   validateGrant(grant,actor,params,false);
   const assertNative=(nativeActor,finalParams)=>{
    if(!open||entered)fail('本次原生入口已使用或已关闭');entered=true;
    try{validateGrant(grant,nativeActor,finalParams,true);}catch(error){throw markUnappliedDamageError(error);}
   };
   if(grant.claim.privacy){if(!messagePrivacy?.withNativeApplication)fail('缺少原生私密回执scope');return await messagePrivacy.withNativeApplication(grant.request,grant.claim,()=>continuation(params,assertNative));}
   return await continuation(params,assertNative);
  }catch(error){if(!entered)throw markUnappliedDamageError(error);throw error;}
  finally{open=false;}
 }
 return {authorize,applyDamage};
}
