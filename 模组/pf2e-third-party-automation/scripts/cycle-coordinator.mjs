import {MODULE_ID} from './rules.mjs';
import {SerialActions,requireOwner} from './runtime.mjs';
import {getCycleContext,resolveCycleDamageMessage} from './cycle-automation.mjs';

const identity=['actorUuid','tokenUuid','messageId','rollIndex'];
const same=(a,b)=>identity.every(key=>(a?.[key]??null)===(b?.[key]??null));
export function createCycleCoordinator({game,fromUuid=globalThis.fromUuid,getContext=getCycleContext,
 resolveMessage=resolveCycleDamageMessage,randomID=()=>foundry.utils.randomID(32),onEffect,chooseTrait}={}){
 const queue=new SerialActions(),messageQueue=new SerialActions();
 const demand=(ok)=>{if(!ok)throw Error('循环能量的当前主GM、所有者或原始来源已改变；已有认领与伤害不会回滚或重试。');};
 const equal=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
 function actorProof(actor){
  if(!actor?.isToken)return {actor,uuid:actor?.uuid};
  const token=actor.token;return {actor,uuid:actor.uuid,token,scene:token?.parent,baseActor:token?.baseActor,actorId:token?.actorId};
 }
 function liveActor(p){
  if(!p.actor||p.actor.uuid!==p.uuid)return false;
  if(!p.token)return !p.actor.isToken&&game.actors?.get(p.actor.id)===p.actor;
  return p.actor.isToken&&p.actor.token===p.token&&p.token.documentName==='Token'&&p.token.actorLink===false&&p.token.actor===p.actor&&p.token.parent===p.scene&&game.scenes?.get(p.scene?.id)===p.scene&&p.scene.tokens.get(p.token.id)===p.token&&p.token.baseActor===p.baseActor&&p.token.actorId===p.actorId&&!!p.baseActor&&game.actors.get(p.actorId)===p.baseActor;
 }
 const liveToken=(token,actor)=>token?.documentName==='Token'&&token.actor===actor&&game.scenes?.get(token.parent?.id)===token.parent&&token.parent.tokens.get(token.id)===token;
 function frame(user,context=null){
  const scope={gm:game.user,user,context};scope.gmId=scope.gm?.id;scope.userId=user?.id;
  if(context){scope.message=game.messages.get(context.messageId);scope.roll=scope.message?.rolls?.[context.rollIndex];}
  guard(scope);return scope;
 }
 function guard(scope,{pending=true}={}){
  demand(scope.gm?.active===true&&scope.gm.isGM===true&&game.user===scope.gm&&scope.gm.id===scope.gmId&&game.users.get(scope.gmId)===scope.gm&&game.users.activeGM===scope.gm&&scope.user?.active===true&&scope.user.id===scope.userId&&game.users.get(scope.userId)===scope.user);
  if(scope.proof){demand(liveActor(scope.proof));requireOwner(scope.actor,scope.user);}
  if(scope.message)demand(game.messages.get(scope.message.id)===scope.message&&scope.message.isDamageRoll&&scope.roll&&scope.message.rolls?.[scope.context.rollIndex]===scope.roll);
  else if(scope.context)demand(false);
  if(scope.token)demand(liveToken(scope.token,scope.actor));
  if(scope.usage)demand(game.messages.get(scope.usage.id)===scope.usage);
  if(pending&&scope.pending!==undefined)demand(equal(scope.actor.flags?.[MODULE_ID]?.cyclePending,scope.pending));
  if(scope.qualification){const current=getContext(scope.actor,scope.message,{token:scope.token??null,rollIndex:scope.context.rollIndex,trait:scope.qualification.trait});demand(['ready','already-applied'].includes(current.status)&&['trait','damageType','level'].every(key=>current[key]===scope.qualification[key]));}
 }
 async function actorFor(context,user){
  const scope=frame(user,context),actor=await fromUuid(context.actorUuid);
  guard(scope);scope.actor=actor;scope.proof=actorProof(actor);demand(actor?.uuid===context.actorUuid);guard(scope);return scope;
 }
 async function tokenFor(scope,context){
  if(!context.tokenUuid)return null;
  const token=await fromUuid(context.tokenUuid);guard(scope);demand(liveToken(token,scope.actor)&&token.uuid===context.tokenUuid);
  scope.token=token;guard(scope);return token;
 }
 async function save(scope,pending){
  guard(scope);const result=await scope.actor.update({[`flags.${MODULE_ID}.cyclePending`]:pending},{render:false});
  guard(scope,{pending:false});demand(result===scope.actor&&equal(scope.actor.flags?.[MODULE_ID]?.cyclePending,pending));scope.pending=structuredClone(pending);guard(scope);
 }
 const expired=pending=>{
  const combat=game.combat,timing=pending?.timing;
  if(!timing||timing.combatId!==combat?.id||!combat.started)return true;
  const index=combat.turns.findIndex(c=>c.id===timing.combatantId);
  return index<0||combat.round>timing.endRound||(combat.round===timing.endRound&&combat.turn>index);
 };
 const record=(scope,applied)=>messageQueue.run(scope.context.messageId,async()=>{
  guard(scope);const {message,context}=scope;
  const prior=message.flags?.[MODULE_ID]?.cycleApplications??[];
  const next=[
   ...prior.filter(entry=>!same(entry,context)),
   ...[{...Object.fromEntries(identity.map(key=>[key,context[key]??null])),applied}],
  ];guard(scope);const result=await message.update({[`flags.${MODULE_ID}.cycleApplications`]:next});guard(scope);demand(result===message&&equal(message.flags?.[MODULE_ID]?.cycleApplications,next));
 });
 return {
  async use(actor,usageMessage,user){
   const scope=frame(user);scope.actor=actor;scope.proof=actorProof(actor);scope.usage=usageMessage;guard(scope);
   return queue.run(actor.uuid,async()=>{
    guard(scope);
    const explicit=usageMessage?.flags?.[MODULE_ID]?.cycleDamage;
    const token=explicit?.tokenUuid?await tokenFor(scope,{actorUuid:actor.uuid,tokenUuid:explicit.tokenUuid}):actor.getActiveTokens?.(true,true)?.[0]??null;
    if(token){scope.token=token.document??token;guard(scope);}
    let context=explicit?getContext(actor,game.messages.get(explicit.messageId),{token,rollIndex:explicit.rollIndex??0,trait:explicit.trait??null}):resolveMessage(actor,game.messages,{token});
    if(!context)throw Error('没有找到以你为目标、符合调谐特征的待结算伤害。请从触发伤害卡使用循环能量。');
    scope.context=context;scope.message=game.messages.get(context.messageId);scope.roll=scope.message?.rolls?.[context.rollIndex];guard(scope);
    if(context.status==='choice-required'){
     if(!chooseTrait)throw Error('此效果同时具有虚能和命能，请从伤害卡选择要循环的特征。');
     const trait=await chooseTrait(actor,user,context.choices);
     guard(scope);
     if(!trait)throw Error('已取消循环能量。');
     context=getContext(actor,game.messages.get(context.messageId),{token,rollIndex:context.rollIndex,trait});
    }
    if(context.status==='already-applied')throw Error('这次伤害已经结算。请先使用原伤害结果的撤销，再使用循环能量。');
    if(context.status!=='ready')throw Error('该伤害不符合循环能量的触发条件，或你不是其目标。');
    scope.qualification=Object.fromEntries(['trait','damageType','level'].map(key=>[key,context[key]]));guard(scope);
    const combat=game.combat,combatant=combat?.combatants.find(c=>c.actor?.uuid===actor.uuid);
    if(!combat?.started||combatant?.initiative==null)throw Error('循环能量需要已掷先攻的遭遇，以追踪反应效果到期。');
    const index=combat.turns.findIndex(c=>c.id===combatant.id),rounds=index>combat.turn?0:1;
    const previous=actor.flags?.[MODULE_ID]?.cyclePending;
    if(previous?.status==='claimed')throw Error('上一次循环能量的伤害正在结算，请等待完成。');
    if(previous?.status==='armed'&&!expired(previous)&&same(previous,context))return '这次伤害的循环能量已就绪，无需重复使用。';
    const pending={...context,status:'armed',reactionId:randomID(),nonce:randomID(),armedAt:Date.now(),
     timing:{combatId:combat.id,combatantId:combatant.id,endRound:combat.round+rounds,worldTime:game.time.worldTime,initiative:combatant.initiative,rounds}};
    await save(scope,pending);
    return `循环能量已就绪：本次伤害自动应用抗力${actor.level}，随后自动获得打击加伤。`;
   });
  },
  async claim(context,user){
   const scope=await actorFor(context,user),actor=scope.actor;guard(scope);
   return queue.run(actor.uuid,async()=>{
    guard(scope);
    const pending=actor.flags?.[MODULE_ID]?.cyclePending;
    if(pending?.status!=='armed'||!same(pending,context))return null;
    scope.pending=structuredClone(pending);
    if(expired(pending)){await save(scope,{...pending,status:'expired'});return null;}
    const current=getContext(actor,scope.message,{token:await tokenFor(scope,context),rollIndex:context.rollIndex,trait:pending.trait});guard(scope);
    if(current.status!=='ready'||current.trait!==pending.trait||current.damageType!==pending.damageType)return null;
    scope.qualification=Object.fromEntries(['trait','damageType','level'].map(key=>[key,current[key]]));
    const claim={...pending,status:'claimed',level:actor.level};await save(scope,claim);return claim;
   });
  },
  async complete({context,claim,applied,uncertain=false,error},user){
   const scope=await actorFor(context,user),actor=scope.actor;await tokenFor(scope,context);guard(scope);
   return queue.run(actor.uuid,async()=>{
    guard(scope);
    const pending=actor.flags?.[MODULE_ID]?.cyclePending;
    if(claim&&(!pending||pending.reactionId!==claim.reactionId||pending.nonce!==claim.nonce||pending.status!=='claimed'||!same(pending,context)))return;
    if(claim)scope.pending=structuredClone(pending);
    if(applied)await record(scope,true);
    if(!claim)return;
    // Consume before effect creation: a failed multi-document update must never replay the reaction.
    await save(scope,{...pending,status:applied?'done':'error',uncertain,error});
    if(applied){guard(scope);await onEffect(actor,pending,user);guard(scope);}
   });
  },
  async undo(message,user){
   const authority=frame(user);demand(game.messages.get(message.id)===message);if(!message.flags?.pf2e?.appliedDamage?.isReverted)return;
   const options=message.flags.pf2e.context?.options??[];
   for(const option of options){
    const prefix=`${MODULE_ID}:source:`;if(!option.startsWith(prefix))continue;
    const [messageId,index]=option.slice(prefix.length).split(':');
    const source=game.messages.get(messageId);if(!source)continue;
    const entries=source.flags?.[MODULE_ID]?.cycleApplications??[];
    for(const entry of entries.filter(entry=>entry.rollIndex===Number(index)&&entry.actorUuid===message.flags.pf2e.appliedDamage.uuid)){
     guard(authority);demand(game.messages.get(message.id)===message&&message.flags?.pf2e?.appliedDamage?.isReverted);
     const scope=await actorFor(entry,user);guard(authority);await queue.run(scope.actor.uuid,()=>record(scope,false));guard(authority);
    }
   }
  },
 };
}
