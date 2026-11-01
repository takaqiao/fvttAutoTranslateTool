import {MODULE_ID} from './rules.mjs';
import {SerialActions,requireOwner} from './runtime.mjs';
import {getSourceId,isActiveGM,resolveMessageTargets,publicTargetName} from './native-context.mjs';
import {isActualUseMessage} from './usage-events.mjs';
import {knowledgeNativeLabel} from './knowledge-display.mjs';
import {BOB2_AUTOMATION_WORLDS} from './world-gates.mjs';

export const MAGUS_ANALYSIS_SOURCE='Compendium.pf2e.feats-srd.Item.0yPbPVEESwB6Bdfw';
const spellstrikeSource='Compendium.pf2e.actionspf2e.Item.QDW9H8XLIjuW2fE4';
const values=value=>Array.from(value?.values?.()??value??[]),own=document=>document?.flags?.[MODULE_ID]??{};
const active=item=>item&&!item.suppressed&&!item.isSuppressed&&!item.system?.suppressed;
const summary='已尝试回忆知识，法术打击已充能；目标临时免疫魔战士分析1日。';

export function createMagusAnalysis({game,fromUuid=globalThis.fromUuid,choose,runRecallKnowledge}={}){
 const queue=new SerialActions();
 const resolveAction=item=>BOB2_AUTOMATION_WORLDS.has(game.world?.id)&&active(item)&&item.type==='feat'&&getSourceId(item)===MAGUS_ANALYSIS_SOURCE?'magus-analysis':null;
 const save=(message,state)=>message.update({[`flags.${MODULE_ID}.magusAnalysis`]:state});
 function authorize({actor,item,message,user,action}){
  if(!isActiveGM(game))throw Error('魔战士分析必须由当前主GM结算。');
  requireOwner(actor,user);
  if(game.users.get(user.id)!==user||game.messages.get(message.id)!==message||message.author?.id!==user.id||message.actor?.uuid!==actor.uuid||item.actor!==actor||actor.items.get(item.id)!==item||resolveAction(item)!==action)throw Error('魔战士分析的原始使用来源不匹配。');
  if(!isActualUseMessage(message))throw Error('请使用魔战士分析；展示聊天卡不会触发检定。');
  if(!values(actor.items).some(feature=>active(feature)&&feature.type==='action'&&getSourceId(feature)===spellstrikeSource))throw Error('魔战士分析需要法术打击。');
 }
 async function pick(actor,user,title,choices){
  const selection=await choose({actor,user,title,choices});
  if(selection===null||selection===undefined||selection===false)return null;
  if(!choices.some(choice=>choice.value===selection))throw Error('魔战士分析收到无效的规则选择。');
  return selection;
 }
 function checkReceipt(message,state,actor,user){
  const recall=own(message).workbenchRecall,context=message?.flags?.pf2e?.context;
  if(!message||game.messages.get(message.id)!==message||message.actor?.uuid!==actor.uuid||message.author?.id!==user.id||message.blind!==true||context?.type!=='skill-check'||!context.options?.includes('action:recall-knowledge')||!message.rolls?.length||!Number.isFinite(message.rolls[0].total)||recall?.schema!==1||recall.requestId!==state.requestId||recall.actorUuid!==actor.uuid||recall.userId!==user.id||recall.tokenUuid!==state.tokenUuid||recall.origin?.messageId!==state.messageId||recall.origin.itemUuid!==state.itemUuid||recall.statistic!==state.statistic||recall.targetUuids?.length!==1||recall.targetUuids[0]!==state.targetUuid||recall.targetActors?.length!==1||recall.targetActors[0].tokenUuid!==state.targetUuid||recall.targetActors[0].actorUuid!==state.targetActorUuid)throw Error('魔战士分析的回忆知识暗骰回执不匹配。');
  // The saved native attempt is sufficient; success and a resolved DC are not required.
  return message;
 }
 function savedCheck(state){
  if(state.checkMessageId)return game.messages.get(state.checkMessageId);
  return values(game.messages).find(message=>own(message).workbenchRecall?.requestId===state.requestId&&own(message).workbenchRecall.actorUuid===state.actorUuid);
 }
 async function executeUsage(ctx){
  authorize(ctx);
  const {actor,item,message,user}=ctx;
  if(own(message).magusAnalysis?.status==='done')return summary;
  const targets=await resolveMessageTargets(message,{fromUuid});
  if(targets.length!==1)throw Error('请为魔战士分析选定一个目标。');
  const target=targets[0],targetActor=target.actor;
  return queue.run(targetActor.uuid,async()=>{
   authorize(ctx);
   if(own(message).magusAnalysis?.status==='done')return summary;
   const assertLive=async()=>{
    authorize(ctx);
    if(await fromUuid(target.uuid)!==target||target.actor!==targetActor)throw Error('魔战士分析的目标已经改变。');
   };
   await assertLive();
   let state=own(message).magusAnalysis,check;
   if(state){
    if(state.actorUuid!==actor.uuid||state.itemUuid!==item.uuid||state.userId!==user.id||state.targetUuid!==target.uuid||state.targetActorUuid!==targetActor.uuid)throw Error('魔战士分析保存的使用来源不匹配。');
    check=savedCheck(state);
    if(!check)throw Error('本次回忆知识已开始但未完成；不会重复投骰。');
    checkReceipt(check,state,actor,user);
   }else{
    if(own(targetActor).magusAnalysisImmunity?.until>game.time.worldTime)throw Error('该目标仍临时免疫魔战士分析。');
    const hit=await pick(actor,user,`魔战士分析 · 本轮此前是否已用打击命中${publicTargetName(target,{game,user})}？`,[{value:'no',label:'未命中，不加值'},{value:'yes',label:'已命中，本次检定+1环境加值'}]);
    if(hit===null)return {status:'cancelled',result:'已取消魔战士分析。'};
    const skills=new Set(['arcana','crafting','medicine','nature','occultism','religion','society']);
    const choices=Object.entries(actor.skills??{}).filter(([slug,skill])=>skills.has(slug)||skill.lore).map(([slug,skill])=>({value:slug,label:knowledgeNativeLabel(game,slug,skill.label)}));
    if(!choices.length)throw Error('没有可用于回忆知识的原生技能。');
    const statistic=await pick(actor,user,'魔战士分析 · 选择回忆知识技能',choices);
    if(statistic===null)return {status:'cancelled',result:'已取消魔战士分析。'};
    await assertLive();
    const token=values(actor.getActiveTokens?.(true,true)).map(value=>value.document??value).find(value=>value.id===message.speaker?.token&&value.parent?.id===message.speaker?.scene);
    if(!token)throw Error('本次魔战士分析缺少原始角色Token。');
    if(typeof runRecallKnowledge!=='function')throw Error('回忆知识原生入口不可用。');
    const requestId=`magus-analysis:${message.id}`;
    state={status:'rolling',messageId:message.id,actorUuid:actor.uuid,itemUuid:item.uuid,userId:user.id,targetUuid:target.uuid,targetActorUuid:targetActor.uuid,tokenUuid:token.uuid,requestId,statistic,hit:hit==='yes'};
    // Workbench's existing owner RPC authenticates this source-card binding.
    await message.update({[`flags.${MODULE_ID}.magusAnalysis`]:state,[`flags.${MODULE_ID}.knowledge.recall`]:{actorUuid:actor.uuid,targetUuid:target.uuid,targetUuids:[target.uuid],itemId:item.id,action:ctx.action,userId:user.id,requestId,statistic,assurance:false,dc:null}});
    const rollOptions=[...(item.getOriginData?.().rollOptions??[]).filter(option=>option!=='maguss-analysis'),'action:recall-knowledge',...(hit==='yes'?['maguss-analysis']:[])];
    const result=await runRecallKnowledge({actor,token,user,targetUuids:[target.uuid],requestId,statistic,origin:{messageId:message.id,itemUuid:item.uuid,rollOptions}});
    check=checkReceipt(game.messages.get(result?.messageId),state,actor,user);
   }
   await assertLive();
   state={...state,status:'rolled',checkMessageId:check.id,attemptedAt:state.attemptedAt??game.time.worldTime};
   await save(message,state);
   const immunity=own(targetActor).magusAnalysisImmunity;
   if(immunity?.messageId!==message.id)await targetActor.update({[`flags.${MODULE_ID}.magusAnalysisImmunity`]:{until:state.attemptedAt+86400,messageId:message.id,actorUuid:actor.uuid}});
   await assertLive();
   if(own(actor).spellstrike?.messageId!==message.id)await actor.update({[`flags.${MODULE_ID}.spellstrike`]:{charged:true,messageId:message.id}});
   await save(message,{...state,status:'done'});
   return summary;
  });
 }
 return {resolveAction,requiresActualUse:(_item,action)=>action==='magus-analysis',executeUsage};
}
