import {MODULE_ID,hasSource} from './rules.mjs';
import {SerialActions} from './runtime.mjs';
import {isActiveGM,publicTargetName,resolveMessageTargets,upsertOwnedEffect} from './native-context.mjs';
import {degreeForSharedCheck} from './social-automation.mjs';
import {genericReactionAvailable,withReactionReservation,reactionEpoch as epoch} from './reaction-budget.mjs';
import {reactionPermitted} from './reaction-restriction.mjs';
import {createFearOwner} from './fear-owner.mjs';
import {pinnedMedicTarget} from './medic-native.mjs';
export {genericReactionAvailable,withReactionReservation} from './reaction-budget.mjs';

export const FEAR_SOURCES=Object.freeze({battle:'Compendium.pf2e.feats-srd.Item.ePObIpaJDgDb9CQj',knowledge:'Compendium.pf2e.feats-srd.Item.hkSuxXOc9qBleJbd'});
const DEMORALIZE_IMMUNITY='Compendium.patreon-v3.effects.Item.DFLW2gzu0PGeX6zu',CONFUSED='Compendium.pf2e.conditionitems.Item.yblD8fOR1J8rDwEQ';
const OUTCOMES=['criticalFailure','failure','success','criticalSuccess'],TRAITS=['emotion','fear','mental'];
const values=c=>Array.from(c?.values?.()??c??[]),own=d=>d?.flags?.[MODULE_ID]?.fear??{},feature=(a,k)=>values(a?.items).find(i=>hasSource(i,FEAR_SOURCES[k]));
const skill=(a,k)=>a.getStatistic?.(k)??a.skills?.[k],rank=(a,k)=>skill(a,k)?.rank??0;
const conscious=a=>a&&a.isDead!==true&&!a.hasCondition?.('unconscious');
export const battleCryReactionAvailable=(actor,game,{reactionRestriction}={})=>reactionPermitted(actor,reactionRestriction)&&(!epoch(actor,game)||genericReactionAvailable(actor,game));

const foe=(actor,target)=>target?.actor&&target.actor.uuid!==actor.uuid&&conscious(target.actor)&&!(actor.isAllyOf?.(target.actor)??false);
const intimidateImmune=(target,actor)=>values(target.items).some(i=>hasSource(i,DEMORALIZE_IMMUNITY)&&i.isExpired!==true&&i.system?.context?.origin?.actor===actor.uuid);
function adjustments(game,raw,options){
 const map={};for(const entry of raw){if(entry.predicate&&!(typeof entry.predicate.test==='function'?entry.predicate.test(options):new game.pf2e.Predicate(entry.predicate).test(options)))continue;for(const key of ['all',...OUTCOMES])if(entry.adjustments?.[key])map[key]=structuredClone(entry.adjustments[key]);}return map;
}
const equivalent=(a,b)=>['all',...OUTCOMES].every(key=>a?.[key]?.amount===b?.[key]?.amount&&a?.[key]?.label===b?.[key]?.label);

export function createFearAutomation({game,reactionRestriction,fromUuid=globalThis.fromUuid,choose,onError=console.error}={}){
 const queue=new SerialActions(),tracked=new Map();
 const gm=()=>{if(!isActiveGM(game))throw Error('恐惧能力必须由当前主GM结算。');};
 const now=()=>game.time?.worldTime??0;
 const forget=actor=>{if(actor?.uuid&&tracked.get(actor.uuid)===actor)tracked.delete(actor.uuid);};
 function liveActor(actor){
  if(!actor?.uuid)return false;if(!actor.isToken)return game.actors?.get?.(actor.id)===actor;
  const token=actor.token,scene=token?.parent;
  return !!scene&&game.scenes?.get?.(scene.id)===scene&&scene.tokens?.get?.(token.id)===token&&token.actorLink===false&&!!token.baseActor&&game.actors?.get?.(token.actorId)===token.baseActor&&token.actor===actor;
 }
 // Native timed effects are registered and expired by PF2e's primary updater.
 // A parallel module deletion races that request and ignores removeEffects=false.
 const nativeExpiry=item=>item.type==='effect'&&typeof item.remainingDuration?.expired==='boolean'&&typeof game.pf2e?.effectTracker?.refresh==='function';
 const fallbackExpired=item=>own(item).kind==='knowledge-immunity'&&own(item).expiresAt<=now()&&!nativeExpiry(item);
 const track=actor=>{if(!liveActor(actor)){forget(actor);return;}if(values(actor.items).some(i=>own(i).kind==='knowledge-immunity'))tracked.set(actor.uuid,actor);else forget(actor);};
 const resolveAction=item=>hasSource(item,FEAR_SOURCES.knowledge)?'fear:disturbing-knowledge':undefined;
 const sourceFor=(actor,message)=>{const t=game.scenes.get(message.speaker?.scene)?.tokens.get(message.speaker?.token);if(!t||t.actor?.uuid!==actor.uuid)throw Error('请从场景中的角色Token使用此能力。');return t;};
 const mark=async(actor,key,data)=>{gm();const result=await upsertOwnedEffect(actor,key,data);track(actor);return result;};
 const frightened=async actor=>{gm();if((actor.getCondition?.('frightened')?.value??0)<1)await actor.increaseCondition('frightened',{value:1});};
 const immunity=actor=>values(actor.items).some(i=>own(i).kind==='knowledge-immunity'&&own(i).expiresAt>now());
 const owner=createFearOwner({game,context:async(message,user,kind)=>{
  const actor=message.actor??await fromUuid(`Actor.${message.speaker?.actor}`),item=feature(actor,kind==='knowledge'?'knowledge':'battle'),origin=sourceFor(actor,message);
  if(!item||game.messages.get(message.id)!==message||!actor.testUserPermission?.(user,'OWNER')||(message.author?.id??message.user?.id??message.user)!==user.id||!conscious(actor))throw Error('原恐惧能力来源或操作者已改变。');
  let target;
  if(kind==='knowledge'){
   if(message.flags?.pf2e?.origin?.uuid!==item.uuid||rank(actor,'occultism')<3||!own(actor).knowledgeUses?.includes(message.id))throw Error('原惊世胡言使用回执已改变。');
   target=(await resolveMessageTargets(message,{fromUuid})).find(t=>t.uuid===message.flags?.[MODULE_ID]?.fearNative?.knowledge?.targetUuid);
  }else{
   const state=own(message).battleCry;if(state?.status!=='rolling')throw Error('战吼原生阶段已改变。');target=await fromUuid(state.targetUuid);
   if(rank(actor,'intimidation')<(state.reaction?4:3))throw Error('战吼资格已改变。');
  }
  if(!foe(actor,target)||target.parent!==origin.parent)throw Error('原恐惧能力目标已改变。');
  return {actor,item,message,user,origin,target};
 },execute:async({actor,item,message,origin,target,payload,marker,validate})=>{
  if(payload.kind==='knowledge'){
   if(!Number.isFinite(payload.dc)||payload.dc<1)throw Error('缺少真实目标意志DC。');
   let check;
   await skill(actor,'occultism').roll({token:origin,item,target:pinnedMedicTarget(target),action:'disturbing-knowledge',dc:{value:payload.dc,visible:false},traits:TRAITS,extraRollOptions:['action:disturbing-knowledge',marker,...TRAITS.map(t=>`item:trait:${t}`)],skipDialog:!(game.user.settings?.showCheckDialogs??true),createMessage:true,callback:(_roll,_outcome,card)=>{validate();check=card;}});
   return check??null;
  }
  const native=game.pf2e.actions.get('demoralize');if(!native?.toActionVariant)throw Error('缺少原生Demoralize动作。');
  if(!target.object)throw Error('原目标Token尚未在原操作者客户端就绪。');
  const reaction=own(message).battleCry.reaction,result=await native.toActionVariant({cost:reaction?'reaction':'free'}).use({actors:[pinnedMedicTarget(origin)],target:target.object,rollOptions:[`${MODULE_ID}:battle-cry:${message.id}`,marker,...reaction?['action:reaction']:[]],event:{ctrlKey:false,metaKey:false,shiftKey:game.user.settings?.showCheckDialogs??true}});
  validate();return result?.[0]?.message??null;
 }});

 async function executeUsage({actor,item,message,user,action}){
  gm();if(!actor?.testUserPermission?.(user,'OWNER')||item.actor?.uuid!==actor.uuid||resolveAction(item)!==action||game.messages.get(message?.id)!==message)throw Error('惊世胡言使用来源或权限无效。');
  return queue.run('fear:knowledge',async()=>{
   gm();if(own(actor).knowledgeUses?.includes(message.id))throw Error('此惊世胡言消息已经结算。');
   if(!conscious(actor)||rank(actor,'occultism')<3)throw Error('惊世胡言需要神秘大师且能行动。');
   const origin=sourceFor(actor,message),targets=await resolveMessageTargets(message,{fromUuid});
   if(!targets.length||rank(actor,'occultism')<4&&targets.length!==1)throw Error('惊世胡言需要一个敌人目标；神秘传奇时可以选中多个。');
   if(targets.some(t=>!foe(actor,t)||t.parent!==origin.parent))throw Error('惊世胡言需要同场景中的敌人目标。');
   const skipped=targets.filter(t=>immunity(t.actor)),eligible=targets.filter(t=>!skipped.includes(t));
   if(!eligible.length)throw Error('目标仍在24小时惊世胡言暂时免疫中。');
   const unique=[...new Map(eligible.map(t=>[t.actor.uuid,t])).values()],dcs=unique.map(t=>skill(t.actor,'will')?.dc?.value),statistic=skill(actor,'occultism');
   const bindings=unique.map(target=>({target,actorUuid:target.actor.uuid}));
   if(dcs.some(d=>!Number.isFinite(d))||!statistic?.roll)throw Error('缺少原生神秘检定或目标意志DC。');
   const domains=statistic.check?.domains??statistic.domains??['occultism','skill-check'],raw=domains.flatMap(d=>actor.synthetics?.degreeOfSuccessAdjustments?.[d]??[]).map(r=>({...r,adjustments:structuredClone(r.adjustments)}));
   gm();await actor.update({[`flags.${MODULE_ID}.fear.knowledgeUses`]:[...(own(actor).knowledgeUses??[]).slice(-127),message.id]});
   const checked=await owner.run({actor,item,message,user,target:unique[0]},'knowledge',{dc:dcs[0]});
   if(checked.status==='cancelled'){await message.update({[`flags.${MODULE_ID}.fear.knowledge`]:{status:'cancelled'}});return {status:'cancelled',result:'已取消惊世胡言检定。'};}
   if(bindings.some(({target,actorUuid})=>target.actor?.uuid!==actorUuid||target.parent!==origin.parent||origin.parent.tokens.get(target.id)!==target))throw Error('惊世胡言的原目标在检定期间已改变。');
   const card=checked.check,roll=card.rolls[0],context=card.flags?.pf2e?.context??{},natural=roll.isDeterministic?roll.terms?.find(t=>t.constructor?.name==='NumericTerm')?.total:roll.dice?.find(d=>d.faces===20)?.total;
   const dice={total:roll.total,natural},base=[...(context.options??[]),...(context.contextualOptions?.postRoll??[])].filter(o=>!o.startsWith('check:total:delta:'));
   const adjustmentFor=dc=>adjustments(game,raw,new Set([...base,`check:total:delta:${roll.total-dc}`])),degree=dc=>degreeForSharedCheck(dice,dc,adjustmentFor(dc)).value;
   const reference=adjustmentFor(dcs[0]),nativeDegree=degreeForSharedCheck(dice,dcs[0],reference);
   if(!equivalent(reference,context.dosAdjustments)||OUTCOMES[nativeDegree.value]!==context.outcome||OUTCOMES[nativeDegree.unadjusted]!==context.unadjustedOutcome)throw Error('惊世胡言成功度无法与原生检定对照，尚未修改目标。');
   for(const [index,target]of unique.entries()){
    gm();const value=degree(dcs[index]),start=now();
    await mark(target.actor,'fear:knowledge-immunity',{name:'惊世胡言：暂时免疫',type:'effect',img:item.img,system:{duration:{value:24,unit:'hours',expiry:'turn-start',sustained:false},start:{value:start,initiative:null},context:{origin:{actor:actor.uuid,item:item.uuid,token:origin.uuid}},rules:[],tokenIcon:{show:false}},flags:{[MODULE_ID]:{fear:{kind:'knowledge-immunity',expiresAt:start+86400,usageId:message.id,checkId:card.id}}}});
    if(value===0)await frightened(actor);
    if(target.actor.isImmuneTo?.(item))continue;
    if(value>=2)await frightened(target.actor);
    if(value===3)await mark(target.actor,`fear:knowledge-confused:${actor.uuid}`,{name:'惊世胡言：困惑',type:'effect',img:item.img,system:{duration:{value:1,unit:'rounds',expiry:'turn-start',sustained:false},start:{value:start,initiative:actor.combatant?.initiative??null},context:{origin:{actor:actor.uuid,item:item.uuid,token:origin.uuid}},rules:[{key:'GrantItem',uuid:CONFUSED,onDeleteActions:{granter:'cascade'}}]},flags:{[MODULE_ID]:{fear:{kind:'knowledge-confused',usageId:message.id,checkId:card.id}}}});
   }
   gm();await message.update({[`flags.${MODULE_ID}.fear.knowledge`]:{checkId:card.id,targets:unique.map(t=>t.uuid),skipped:skipped.map(t=>t.uuid)}});
   return `已完成惊世胡言的一次神秘检定、对应恐惧效果与24小时暂时免疫。${skipped.length?`已略过 ${skipped.length} 个暂时免疫目标。`:''}`;
  });
 }

 async function battleCry(message,creatingUserId){
  if(!isActiveGM(game)||game.messages.get(message?.id)!==message||!message.rolls?.length||own(message).battleCry)return;
  const context=message.flags?.pf2e?.context??{},type=context.type,actor=message.actor??await fromUuid(`Actor.${message.speaker?.actor}`),item=feature(actor,'battle');
  if(!item||actor.type!=='character'||!conscious(actor)||rank(actor,'intimidation')<3)return;
  const reaction=type==='attack-roll'&&context.outcome==='criticalSuccess'&&rank(actor,'intimidation')>=4;
  if(type!=='initiative'&&!reaction||context.isReroll)return;
  const author=message.author??game.users.get(message.user?.id??message.user);
  if(!author||creatingUserId&&creatingUserId!==author.id&&creatingUserId!==game.users.activeGM?.id||!actor.testUserPermission?.(author,'OWNER'))return;
  const user=author;
  return queue.run(`fear:battle:${actor.uuid}`,async()=>{
   gm();if(own(message).battleCry||!conscious(actor)||reaction&&!battleCryReactionAvailable(actor,game,{reactionRestriction}))return;
   const triggerEpoch=epoch(actor,game),origin=sourceFor(actor,message),raw=reaction?[await fromUuid(context.target?.token)]:values(origin.parent.tokens);
   // Hidden documents are a recipient permission boundary, not a sight test.
   const canTarget=t=>foe(actor,t)&&t.parent===origin.parent&&!intimidateImmune(t.actor,actor)&&(!t.hidden||user.isGM||t.actor.testUserPermission?.(user,'OWNER'));
   const candidates=raw.filter(canTarget);if(!candidates.length)return;
   gm();await message.update({[`flags.${MODULE_ID}.fear.battleCry`]:{status:'offered',reaction}});
   const choices=[...candidates.map(t=>({value:t.uuid,label:`${reaction?'使用反应':'自由动作'}：挫败${publicTargetName(t,{game,user})}的士气`})),{value:'decline',label:'不使用战吼'}];
   const selected=await choose({actor,user,title:reaction?'战吼：攻击大成功后的反应':'战吼：先攻后的自由动作',choices});
   if(selected==null||selected==='decline'){gm();await message.update({[`flags.${MODULE_ID}.fear.battleCry`]:{status:'declined',reaction}});return;}
   const target=candidates.find(t=>t.uuid===selected);if(!target)throw Error('战吼的目标选择无效。');
   gm();if(!conscious(actor)||!feature(actor,'battle')||rank(actor,'intimidation')<(reaction?4:3)||!canTarget(target)||reaction&&(epoch(actor,game)!==triggerEpoch||!battleCryReactionAvailable(actor,game,{reactionRestriction})))throw Error('战吼选择期间角色、目标或反应资源已改变。');
   let claim;
   if(reaction)await withReactionReservation(actor,game,async()=>{
    gm();if(!conscious(actor)||epoch(actor,game)!==triggerEpoch||!battleCryReactionAvailable(actor,game,{reactionRestriction}))throw Error('战吼确认期间角色或反应资源已改变。');
    claim={id:message.id,epoch:epoch(actor,game),checkId:null};await actor.update({[`flags.${MODULE_ID}.fear.reactions`]:[...(own(actor).reactions??[]).filter(r=>r.epoch===claim.epoch),claim]});
   });
   await message.update({[`flags.${MODULE_ID}.fear.battleCry`]:{status:'rolling',reaction,targetUuid:target.uuid}});
   const result=await owner.run({actor,item,message,user,target},'battleCry');
   if(result.status==='cancelled'){
    gm();if(claim)await actor.update({[`flags.${MODULE_ID}.fear.reactions`]:(own(actor).reactions??[]).filter(r=>r.id!==claim.id)});
    await message.update({[`flags.${MODULE_ID}.fear.battleCry`]:{status:'cancelled',reaction,targetUuid:target.uuid}});return;
   }
   const checkId=result.check.id;
   gm();if(claim)await actor.update({[`flags.${MODULE_ID}.fear.reactions`]:(own(actor).reactions??[]).map(r=>r.id===claim.id?{...r,checkId}:r)});
   await message.update({[`flags.${MODULE_ID}.fear.battleCry`]:{status:'done',reaction,checkId,targetUuid:target.uuid}});
  });
 }
 async function maintain(actor){
  if(!liveActor(actor)){forget(actor);return;}if(!isActiveGM(game))return;
  return queue.run('fear:knowledge',async()=>{
   gm();if(!liveActor(actor)){forget(actor);return;}
   const expired=values(actor.items).filter(fallbackExpired);
   if(expired.length){gm();if(!liveActor(actor)){forget(actor);return;}const ids=expired.filter(i=>values(actor.items).includes(i)&&fallbackExpired(i)).map(i=>i.id);if(ids.length)await actor.deleteEmbeddedDocuments('Item',ids);}
   track(actor);
  });
 }
 function register({Hooks,socket}={}){
  owner.register({Hooks,socket});
  const registrations=[],on=(name,fn)=>registrations.push([name,Hooks.on(name,(...args)=>Promise.resolve().then(()=>fn(...args)).catch(onError))]);
  on('createChatMessage',async(m,_options,userId)=>{
   try{await battleCry(m,userId);}catch(error){
    if(isActiveGM(game)&&['offered','rolling'].includes(own(m).battleCry?.status))await m.update({[`flags.${MODULE_ID}.fear.battleCry`]:{...own(m).battleCry,status:'uncertain',result:'本次战吼尚需 GM 核对。'}});
    throw error;
   }
  });
  for(const actor of values(game.actors))track(actor);
  for(const event of ['createItem','updateItem','deleteItem'])on(event,item=>{if(item.actor&&(own(item).kind==='knowledge-immunity'||tracked.has(item.actor.uuid)))track(item.actor);});
  on('createActor',track);on('deleteActor',forget);on('createToken',token=>token.actor&&track(token.actor));
  on('deleteToken',token=>{for(const actor of tracked.values())if(actor.isToken&&actor.token===token)forget(actor);});
  on('deleteScene',scene=>{for(const actor of tracked.values())if(actor.isToken&&actor.token?.parent===scene)forget(actor);});
  on('updateWorldTime',async()=>{if(isActiveGM(game))for(const actor of tracked.values())await maintain(actor);});
  return()=>{for(const[name,id]of registrations)Hooks.off(name,id);};
 }
 return {resolveAction,executeUsage,battleCry,maintain,register};
}
