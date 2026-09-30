import {MODULE_ID} from './rules.mjs';
import {SerialActions} from './runtime.mjs';
import {getSourceId,isActiveGM,resolveMessageTargets,upsertOwnedEffect} from './native-context.mjs';
import {SCARE_SOURCE,scareState,scareResult,scareLanguagePenalty,scareTemporarilyImmune,scareImmunityExpired,scareEffectData,scareDeathImmune,scareSensePair,assertScareTokens} from './scare-to-death-rules.mjs';
import {createScareExecutor,validateScareCard} from './scare-to-death-executor.mjs';

const values=c=>Array.from(c?.values?.()??c??[]);
const languages=actor=>[...new Set(actor.system?.details?.languages?.value??[])];
const author=m=>m.author?.id??m.user?.id??m.user;

export function createScareToDeath({game,fromUuid=globalThis.fromUuid,choose,onError=console.error,sense=scareSensePair,
 makeEffect=data=>new globalThis.CONFIG.Item.documentClass(data),executor=createScareExecutor({game,fromUuid})}={}){
 const queue=new SerialActions(),tracked=new Map();
 const gm=()=>{if(!isActiveGM(game))throw Error('肝胆俱裂必须由当前主GM结算。');};
 const now=()=>game.time.worldTime;
 const resolveAction=item=>item?.type==='feat'&&getSourceId(item)===SCARE_SOURCE?'scare-to-death':undefined;
 const write=async operation=>{gm();const value=await operation();gm();return value;};
 const saveClaim=(message,claim)=>write(()=>message.update({[`flags.${MODULE_ID}.scare.claim`]:structuredClone(claim)}));
 const pending=actor=>scareState(actor).pending;
 const forget=actor=>{if(actor&&tracked.get(actor.uuid)===actor)tracked.delete(actor.uuid);};
 function liveActor(actor){
  if(!actor?.uuid)return false;
  if(!actor.isToken)return game.actors?.get?.(actor.id)===actor;
  const token=actor.token,scene=token?.parent;
  return !!scene&&game.scenes?.get?.(scene.id)===scene&&scene.tokens?.get?.(token.id)===token&&token.actorLink===false&&
   !!token.baseActor&&game.actors?.get?.(token.actorId)===token.baseActor&&token.actor===actor;
 }
 function track(actor){
  if(liveActor(actor)&&scareTemporarilyImmune(actor,now()))tracked.set(actor.uuid,actor);else forget(actor);
 }
 // PF2e registers owned effects during actor preparation and removes expired
 // ones through its primary updater. Never race that request, or override the
 // user's removeEffects=false preference. The timestamp fallback is only for
 // effects without the native duration/tracker interface.
 const nativeExpiry=item=>item.type==='effect'&&typeof item.remainingDuration?.expired==='boolean'&&typeof game.pf2e?.effectTracker?.refresh==='function';
 async function clearCompletedReservation(actor){
  const reservation=pending(actor),claim=reservation&&scareState(game.messages.get(reservation.usageId)).claim;
  if(claim?.state==='done'&&claim.nonce===reservation.nonce&&claim.targetActorUuid===actor.uuid&&claim.actorUuid===reservation.sourceActorUuid)await write(()=>actor.update({[`flags.${MODULE_ID}.scare.pending`]:null}));
 }
 function eligible(actor,item,origin,target){
  gm();assertScareTokens(origin,target);
  if(origin.actor!==actor||item.actor!==actor||actor.items.get(item.id)!==item||resolveAction(item)!=='scare-to-death')throw Error('肝胆俱裂来源专长或角色已改变。');
  if(actor.isDead||actor.hasCondition?.('unconscious')||(actor.getStatistic?.('intimidation')?.rank??0)<4)throw Error('肝胆俱裂需要能行动且威吓传奇。');
  if(target.actor?.modeOfBeing!=='living'||target.actor.isDead)throw Error('肝胆俱裂需要一个活物目标。');
  const distance=origin.object.distanceTo?.(target.object);if(!Number.isFinite(distance)||distance<0||distance>30)throw Error('肝胆俱裂目标必须在30尺内。');
  const senses=sense(origin,target);
  if(!senses.originSensesTarget||!senses.targetSensesOrigin)throw Error('肝胆俱裂双方必须能够感知到彼此。');
  return senses;
 }
 async function executeUsage({actor,item,message,user,action}){
  gm();if(game.messages.get(message?.id)!==message||author(message)!==user?.id||!user?.active||!actor?.testUserPermission?.(user,'OWNER')||resolveAction(item)!==action||message.actor?.uuid!==actor.uuid)throw Error('肝胆俱裂的原使用卡、来源或操作者无效。');
  const targets=await resolveMessageTargets(message,{fromUuid});if(targets.length!==1)throw Error('肝胆俱裂需要选中一个目标。');
  const target=targets[0],origin=game.scenes.get(message.speaker?.scene)?.tokens.get(message.speaker?.token);
  return queue.run(target.actor.uuid,async()=>{
   gm();await clearCompletedReservation(target.actor);const previous=scareState(message).claim;
   if(previous){if(previous.state==='done')return '本次肝胆俱裂已经结算。';throw Error('本次肝胆俱裂已有未确认的执行记录，不会重复投骰。');}
   if(pending(target.actor))throw Error('该目标有一项进行中或未确认的肝胆俱裂，不会再次投骰。');
   eligible(actor,item,origin,target);if(scareTemporarilyImmune(target.actor,now()))throw Error('目标仍在1分钟肝胆俱裂暂时免疫中。');
   const spoken=languages(actor),choices=spoken.map(value=>({value,label:game.i18n?.localize?.(globalThis.CONFIG?.PF2E?.languages?.[value]??value)??value}));
   const language=choices.length>1?await choose({actor,user,title:'肝胆俱裂：本次使用的语言',choices}):choices[0]?.value??null;
   if(choices.length>1&&language==null)return '已取消肝胆俱裂。';
   if(language!==null&&!spoken.includes(language))throw Error('肝胆俱裂语言选择无效。');
   const senses=eligible(actor,item,origin,target);
   if(!actor.testUserPermission(user,'OWNER')||!user.active)throw Error('肝胆俱裂操作者已改变。');
   const saveUser=values(game.users).find(u=>u.active&&!u.isGM&&target.actor.testUserPermission?.(u,'OWNER'))??game.users.activeGM;
   const claim={usageId:message.id,actorUuid:actor.uuid,originUuid:origin.uuid,targetUuid:target.uuid,targetActorUuid:target.actor.uuid,itemUuid:item.uuid,userId:user.id,saveUserId:saveUser.id,
    language,penalty:scareLanguagePenalty({heard:senses.targetHearsOrigin,understood:language!==null&&languages(target.actor).includes(language)}),nonce:globalThis.foundry?.utils?.randomID?.()??globalThis.crypto.randomUUID(),state:'intimidation-ready'};
   await saveClaim(message,claim);
   await write(()=>target.actor.update({[`flags.${MODULE_ID}.scare.pending`]:{usageId:message.id,nonce:claim.nonce,sourceActorUuid:actor.uuid}}));
   try{
    const check=await executor.roll(claim,'intimidation');gm();const degree=validateScareCard({game,message:check,claim,stage:'intimidation'});
    Object.assign(claim,{checkId:check.id,checkDegree:degree,state:'intimidation-done',effectStart:now()});await saveClaim(message,claim);
    const effectData=kind=>scareEffectData({kind,item,origin,now:claim.effectStart,usageId:message.id,checkId:check.id});
    if(typeof target.actor.isImmuneTo!=='function')throw Error('缺少原生恐惧/心智免疫接口。');
    const effectImmune=target.actor.isImmuneTo(makeEffect(effectData('test')));let fortitude;
    if(scareResult({check:degree,effectImmune}).save){
     claim.state='fortitude-ready';await saveClaim(message,claim);
     const card=await executor.roll(claim,'fortitude');gm();fortitude=validateScareCard({game,message:card,claim,stage:'fortitude'});
     Object.assign(claim,{fortitudeId:card.id,fortitudeDegree:fortitude,state:'fortitude-done'});await saveClaim(message,claim);
    }
    const deathImmune=degree===3&&fortitude===0&&scareDeathImmune(target.actor,makeEffect(effectData('death')).getRollOptions('item'),game.pf2e.settings.iwr);
    const result=scareResult({check:degree,fortitude,effectImmune,deathImmune});
    Object.assign(claim,{state:'applying',result,effectImmune,deathImmune:!!deathImmune});await saveClaim(message,claim);
    if(pending(target.actor)?.nonce!==claim.nonce)throw Error('肝胆俱裂目标保留记录已变，尚未应用结果。');
    await write(()=>upsertOwnedEffect(target.actor,'scare-to-death:immunity',effectData('immunity')));track(target.actor);
    if(result.frightened>(target.actor.getCondition?.('frightened')?.value??0))await write(()=>target.actor.increaseCondition('frightened',{value:result.frightened,max:result.frightened}));
    if(result.fleeing)await write(()=>upsertOwnedEffect(target.actor,`scare-to-death:fleeing:${message.id}`,effectData('fleeing')));
    if(result.death){
     await write(()=>target.actor.toggleStatusEffect('dead',{active:true,overlay:true}));
     if(target.combatant?.actor?.uuid===target.actor.uuid)await write(()=>target.combatant.toggleDefeated({to:true,overlayIcon:false}));
     if(!target.actor.isDead)throw Error('原生死亡状态未确认。');
    }
    claim.state='done';await saveClaim(message,claim);
    if(pending(target.actor)?.nonce===claim.nonce)await write(()=>target.actor.update({[`flags.${MODULE_ID}.scare.pending`]:null}));
    return '已完成肝胆俱裂的原生检定、对应结果与1分钟暂时免疫。';
   }catch(error){
    if(isActiveGM(game)&&scareState(message).claim?.nonce===claim.nonce&&claim.state!=='done')await saveClaim(message,{...claim,state:'uncertain',lastStage:claim.state,error:String(error.message??error)}).catch(()=>{});
    throw error;
   }
  });
 }
 const lifecycle=()=>JSON.stringify([now(),game.combat?.id,game.combat?.started,game.combat?.round,game.combat?.turn]);
 async function maintain(actor,state=lifecycle()){
  if(!liveActor(actor)){forget(actor);return;}if(!isActiveGM(game))return;
  return queue.run(actor.uuid,async()=>{
   if(!liveActor(actor)){forget(actor);return;}if(!isActiveGM(game)||state!==lifecycle())return;
   await clearCompletedReservation(actor);
   if(!liveActor(actor)){forget(actor);return;}
   if(state!==lifecycle())return;
   const expired=values(actor.items).filter(i=>scareState(i).kind==='immunity'&&scareImmunityExpired(i,now())&&!nativeExpiry(i));
   if(expired.length)await write(()=>{
    if(!liveActor(actor)){forget(actor);return;}if(state!==lifecycle())return;
    const ids=expired.filter(i=>actor.items.get(i.id)===i&&scareImmunityExpired(i,now())&&!nativeExpiry(i)).map(i=>i.id);
    if(ids.length)return actor.deleteEmbeddedDocuments('Item',ids);
   });
   track(actor);
  });
 }
 function register({Hooks,socket}={}){
  executor.register({socket});for(const actor of new Set([...values(game.actors),...values(game.pf2e?.effectTracker?.effects).map(i=>i.actor).filter(Boolean)]))track(actor);
  const registrations=[];for(const event of ['updateWorldTime','updateCombat','deleteCombat'])registrations.push([event,Hooks.on(event,()=>{
   // PF2e also defers combat-linked time changes to the committed encounter:
   // core.time can arrive while game.combat still points at the previous turn.
   if(event==='updateWorldTime'&&game.combat?.started)return;
   const state=lifecycle();if(isActiveGM(game))for(const actor of tracked.values())maintain(actor,state).catch(onError);
  })]);
  const on=(event,fn)=>registrations.push([event,Hooks.on(event,fn)]);
  on('deleteActor',actor=>{for(const current of tracked.values())if(current===actor||current.isToken&&current.token?.actorId===actor.id)forget(current);});
  on('deleteToken',token=>{for(const actor of tracked.values())if(actor.isToken&&actor.token===token)forget(actor);});
  on('deleteScene',scene=>{for(const actor of tracked.values())if(actor.isToken&&actor.token?.parent===scene)forget(actor);});
  on('deleteItem',item=>{if(scareState(item).kind==='immunity')track(item.actor);});
  return()=>{for(const[name,id]of registrations)Hooks.off(name,id);tracked.clear();};
 }
 return {resolveAction,executeUsage,maintain,register};
}
