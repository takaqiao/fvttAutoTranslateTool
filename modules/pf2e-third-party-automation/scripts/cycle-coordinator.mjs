import {MODULE_ID} from './rules.mjs';
import {SerialActions,requireOwner} from './runtime.mjs';
import {getCycleContext,resolveCycleDamageMessage} from './cycle-automation.mjs';

const identity=['actorUuid','tokenUuid','messageId','rollIndex'];
const same=(a,b)=>identity.every(key=>(a?.[key]??null)===(b?.[key]??null));
export function createCycleCoordinator({game,fromUuid=globalThis.fromUuid,getContext=getCycleContext,
 resolveMessage=resolveCycleDamageMessage,randomID=()=>foundry.utils.randomID(32),onEffect,chooseTrait}={}){
 const queue=new SerialActions(),messageQueue=new SerialActions();
 const gm=()=>{if(game.user!==game.users.activeGM)throw Error('循环能量必须由当前主GM统一结算。');};
 const actorFor=async(context,user)=>{gm();const actor=await fromUuid(context.actorUuid);requireOwner(actor,user);return actor;};
 const tokenFor=async context=>{
  if(!context.tokenUuid)return null;
  const token=await fromUuid(context.tokenUuid);
  if(token?.actor?.uuid!==context.actorUuid)throw Error('触发伤害的Token与角色不符。');
  return token;
 };
 const save=(actor,pending)=>actor.update({[`flags.${MODULE_ID}.cyclePending`]:pending});
 const expired=pending=>{
  const combat=game.combat,timing=pending?.timing;
  if(!timing||timing.combatId!==combat?.id||!combat.started)return true;
  const index=combat.turns.findIndex(c=>c.id===timing.combatantId);
  return index<0||combat.round>timing.endRound||(combat.round===timing.endRound&&combat.turn>index);
 };
 const record=(context,applied)=>messageQueue.run(context.messageId,async()=>{
  const message=game.messages.get(context.messageId);if(!message?.isDamageRoll||!message.rolls?.[context.rollIndex])return;
  const prior=message.flags?.[MODULE_ID]?.cycleApplications??[];
  await message.update({[`flags.${MODULE_ID}.cycleApplications`]:[
   ...prior.filter(entry=>!same(entry,context)),
   ...[{...Object.fromEntries(identity.map(key=>[key,context[key]??null])),applied}],
  ]});
 });
 return {
  async use(actor,usageMessage,user){
   gm();requireOwner(actor,user);
   return queue.run(actor.uuid,async()=>{
    const explicit=usageMessage?.flags?.[MODULE_ID]?.cycleDamage;
    const token=explicit?.tokenUuid?await tokenFor({actorUuid:actor.uuid,tokenUuid:explicit.tokenUuid}):actor.getActiveTokens?.(true,true)?.[0]??null;
    let context=explicit?getContext(actor,game.messages.get(explicit.messageId),{token,rollIndex:explicit.rollIndex??0,trait:explicit.trait??null}):resolveMessage(actor,game.messages,{token});
    if(!context)throw Error('没有找到以你为目标、符合调谐特征的待结算伤害。请从触发伤害卡使用循环能量。');
    if(context.status==='choice-required'){
     if(!chooseTrait)throw Error('此效果同时具有虚能和命能，请从伤害卡选择要循环的特征。');
     const trait=await chooseTrait(actor,user,context.choices);
     if(!trait)throw Error('已取消循环能量。');
     context=getContext(actor,game.messages.get(context.messageId),{token,rollIndex:context.rollIndex,trait});
    }
    if(context.status==='already-applied')throw Error('这次伤害已经结算。请先使用原伤害结果的撤销，再使用循环能量。');
    if(context.status!=='ready')throw Error('该伤害不符合循环能量的触发条件，或你不是其目标。');
    const combat=game.combat,combatant=combat?.combatants.find(c=>c.actor?.uuid===actor.uuid);
    if(!combat?.started||combatant?.initiative==null)throw Error('循环能量需要已掷先攻的遭遇，以追踪反应效果到期。');
    const index=combat.turns.findIndex(c=>c.id===combatant.id),rounds=index>combat.turn?0:1;
    const previous=actor.flags?.[MODULE_ID]?.cyclePending;
    if(previous?.status==='claimed')throw Error('上一次循环能量的伤害正在结算，请等待完成。');
    if(previous?.status==='armed'&&!expired(previous)&&same(previous,context))return '这次伤害的循环能量已就绪，无需重复使用。';
    const pending={...context,status:'armed',reactionId:randomID(),nonce:randomID(),armedAt:Date.now(),
     timing:{combatId:combat.id,combatantId:combatant.id,endRound:combat.round+rounds,worldTime:game.time.worldTime,initiative:combatant.initiative,rounds}};
    await save(actor,pending);
    return `循环能量已就绪：本次伤害自动应用抗力${actor.level}，随后自动获得打击加伤。`;
   });
  },
  async claim(context,user){
   const actor=await actorFor(context,user);
   return queue.run(actor.uuid,async()=>{
    const pending=actor.flags?.[MODULE_ID]?.cyclePending;
    if(pending?.status!=='armed'||!same(pending,context))return null;
    if(expired(pending)){await save(actor,{...pending,status:'expired'});return null;}
    const current=getContext(actor,game.messages.get(context.messageId),{token:await tokenFor(context),rollIndex:context.rollIndex,trait:pending.trait});
    if(current.status!=='ready'||current.trait!==pending.trait||current.damageType!==pending.damageType)return null;
    const claim={...pending,status:'claimed',level:actor.level};await save(actor,claim);return claim;
   });
  },
  async complete({context,claim,applied,uncertain=false,error},user){
   const actor=await actorFor(context,user);
   return queue.run(actor.uuid,async()=>{
    const pending=actor.flags?.[MODULE_ID]?.cyclePending;
    if(claim&&(!pending||pending.reactionId!==claim.reactionId||pending.nonce!==claim.nonce||pending.status!=='claimed'||!same(pending,context)))return;
    if(applied)await record(context,true);
    if(!claim)return;
    // Consume before effect creation: a failed multi-document update must never replay the reaction.
    await save(actor,{...pending,status:applied?'done':'error',uncertain,error});
    if(applied)await onEffect(actor,pending,user);
   });
  },
  async undo(message,user){
   gm();if(!message.flags?.pf2e?.appliedDamage?.isReverted)return;
   const options=message.flags.pf2e.context?.options??[];
   for(const option of options){
    const prefix=`${MODULE_ID}:source:`;if(!option.startsWith(prefix))continue;
    const [messageId,index]=option.slice(prefix.length).split(':');
    const source=game.messages.get(messageId);if(!source)continue;
    const entries=source.flags?.[MODULE_ID]?.cycleApplications??[];
    for(const entry of entries.filter(entry=>entry.rollIndex===Number(index)&&entry.actorUuid===message.flags.pf2e.appliedDamage.uuid)){
     const actor=await actorFor(entry,user);await queue.run(actor.uuid,()=>record(entry,false));
    }
   }
  },
 };
}
