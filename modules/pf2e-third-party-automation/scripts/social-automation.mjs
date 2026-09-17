import {MODULE_ID} from './rules.mjs';
import {SerialActions} from './runtime.mjs';
import {getSourceId,isActiveGM,upsertOwnedEffect} from './native-context.mjs';

const SOURCE='Compendium.pf2e.feats-srd.Item.6ON8DjFXSMITZleX';
const OUTCOMES=['criticalFailure','failure','success','criticalSuccess'];
const TRAITS=['auditory','concentrate','emotion','linguistic','mental'];
const IMMUNITY='social:no-cause-for-alarm:immunity';
const values=c=>Array.from(c?.values?.()??c??[]),own=d=>d?.flags?.[MODULE_ID]??{};
const escape=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const clamp=n=>Math.max(0,Math.min(3,n));

/** PF2e 8.5 degree order: threshold, natural die, then first applicable adjustment.
 * Inputs come from the actual native check; this never creates or rerolls a die.
 */
export function degreeForSharedCheck({total,natural},dc,adjustments={}){
 if(!Number.isFinite(total)||!Number.isFinite(dc)||!Number.isInteger(natural)||natural<1||natural>20)throw Error('原生检定总值、有效d20或DC无效。');
 const threshold=total>=dc+10?3:total<=dc-10?0:total>=dc?2:1;
 const unadjusted=clamp(threshold+(natural===20?1:natural===1?-1:0));
 for(const key of ['all',...OUTCOMES]){
  const entry=adjustments[key],amount=entry?.amount;
  if(!entry?.label||!amount||(key!=='all'&&key!==OUTCOMES[unadjusted]))continue;
  if((unadjusted===3&&amount===1)||(unadjusted===0&&amount===-1))continue;
  const value=typeof amount==='string'?OUTCOMES.indexOf(amount):clamp(unadjusted+amount);
  if(value<0||!Number.isInteger(value))throw Error('未识别的原生成功度修正。');
  return {value,unadjusted,adjustment:{label:entry.label,amount}};
 }
 return {value:unadjusted,unadjusted,adjustment:null};
}

function adjustmentMap(game,raw,options){
 const result={};
 for(const entry of raw){
  const predicate=entry.predicate;
  if(predicate&&!(typeof predicate.test==='function'?predicate.test(options):new game.pf2e.Predicate(predicate).test(options)))continue;
  for(const key of ['all',...OUTCOMES])if(entry.adjustments?.[key])result[key]=structuredClone(entry.adjustments[key]);
 }
 return result;
}
const equivalent=(a,b)=>['all',...OUTCOMES].every(key=>a?.[key]?.amount===b?.[key]?.amount&&a?.[key]?.label===b?.[key]?.label);
const languages=a=>Array.from(new Set(a?.system?.details?.languages?.value??[]));
const immune=(actor,time)=>values(actor.items).some(i=>own(i).kind==='social-alarm-immunity'&&own(i).expiresAt>time);
function soundBlocked(origin,target){
 const backend=globalThis.CONFIG?.Canvas?.polygonBackends?.sound,level=origin.parent?.levels?.get(origin._source?.level??origin.level);
 if(!backend?.testCollision||!level)throw Error('无法确认场景层级的原生声音传播。');
 // Foundry 14 Token.checkCollision explicitly rejects type:"sound"; its native
 // polygon backend accepts an explicit level and elevations without a source.
 return backend.testCollision({...origin.object.center,elevation:origin.elevation??0},{...target.object.center,elevation:target.elevation??0},{type:'sound',mode:'any',level});
}

export function createSocialAutomation({game,fromUuid=globalThis.fromUuid,choose,onError=()=>{}}={}){
 const queue=new SerialActions();
 const resolveAction=item=>item?.type==='feat'&&getSourceId(item)===SOURCE?'social:no-cause-for-alarm':null;
 async function choice(actor,user,title,choices){if(choices.length===1)return choices[0].value;const result=await choose({actor,user,title,choices});if(result==null)return null;if(!choices.some(c=>c.value===result))throw Error('无效的无需惊慌规则选择。');return result;}
 async function executeUsage({actor,item,message,user,action}){
  if(!isActiveGM(game)||!actor?.testUserPermission(user,'OWNER')||item?.actor?.uuid!==actor.uuid||resolveAction(item)!==action)throw Error('无权执行此来源的无需惊慌。');
  return queue.run('social:no-cause-for-alarm',async()=>{
   if(own(actor).socialUses?.includes(message.id))return '本次无需惊慌已结算。';
   if(message.flags?.pf2e?.flatCheck?.result==='fail')return '原生平检失败，本次无需惊慌不产生效果。';
   const {scene:sceneId,token:tokenId}=message.speaker??{},scene=game.scenes.get(sceneId),origin=scene?.tokens.get(tokenId);
   if(!origin?.object||origin.actor?.uuid!==actor.uuid)throw Error('请从场景中的角色使用无需惊慌，以确定10尺散发起点。');
   if(actor.isDead||actor.hasCondition?.('unconscious'))throw Error('当前无法发声使用无需惊慌。');
   const now=game.time.worldTime,candidates=new Map(),skipped=[];
   for(const target of values(scene.tokens)){
    const recipient=target.actor,object=target.object;if(!recipient||!object||!(recipient.getCondition?.('frightened')?.value>0))continue;
    const distance=origin.object.distanceTo(object);if(!Number.isFinite(distance)||distance>10)continue;
    if(immune(recipient,now)){skipped.push({actorUuid:recipient.uuid,reason:'temporary-immunity'});continue;}
    if(recipient.hasCondition?.('deafened')||recipient.isImmuneTo?.(item)){skipped.push({actorUuid:recipient.uuid,reason:'trait-immunity-or-deafened'});continue;}
    if(target.uuid!==origin.uuid){
     if(typeof origin.object.checkCollision!=='function')throw Error('场景缺少原生阻挡检测，无法确认散发范围。');
     if(origin.object.checkCollision(object.center,{origin:origin.object.center,type:'move',mode:'any'})||soundBlocked(origin,target))continue;
    }
    if(!candidates.has(recipient.uuid))candidates.set(recipient.uuid,target);
   }
   if(!candidates.size)return '10尺内没有可受影响且未免疫的惊惧生物。';
   if(candidates.has(actor.uuid)){
    const include=await choice(actor,user,'无需惊慌：散发是否包括自身',[{value:'exclude',label:'不影响自身'},{value:'include',label:'包括自身'}]);if(include===null)return '已取消。';if(include==='exclude')candidates.delete(actor.uuid);
   }
   const spoken=languages(actor);
   if(!spoken.length)return '施术者没有已记录的可用语言。';
   const language=await choice(actor,user,'无需惊慌：本次使用的语言',spoken.map(value=>({value,label:game.i18n?.localize?.(globalThis.CONFIG?.PF2E?.languages?.[value]??value)??value})));if(language===null)return '已取消。';
   const recipients=values(candidates).filter(t=>languages(t.actor).includes(language));if(!recipients.length)return '没有理解本次语言的可受影响生物。';
   const targets=recipients.map(token=>({token,actor:token.actor,dc:token.actor.getStatistic?.('will')?.dc?.value,fear:token.actor.getCondition('frightened').value}));
   if(targets.some(t=>!Number.isFinite(t.dc)))throw Error('目标缺少原生意志DC，尚未检定。');
   const statistic=actor.getStatistic?.('diplomacy')??actor.skills?.diplomacy,domains=statistic?.check?.domains??statistic?.domains;
   if(!statistic?.roll||!Array.isArray(domains))throw Error('未找到原生交涉检定。');
   const raw=domains.flatMap(domain=>actor.synthetics?.degreeOfSuccessAdjustments?.[domain]??[]).map(r=>({...r,adjustments:structuredClone(r.adjustments)}));
   await actor.update({[`flags.${MODULE_ID}.socialUses`]:[...(own(actor).socialUses??[]).slice(-127),message.id]});
   if(actor.hasCondition?.('deafened')){
    const pf=message.flags?.pf2e??{},postInfo=Object.keys(pf).length===1&&pf.origin&&!pf.origin.sourceId;
    const patreonGate=game.modules?.get('patreon-v3')?.active&&postInfo&&['all','attack'].includes(game.settings?.get('patreon-v3','flatCheck'));
    if(!patreonGate){
     if(!game.pf2e.Check?.roll||!game.pf2e.CheckModifier)throw Error('无法执行耳聋的原生DC 5听觉动作平检。');
     let total;
     await game.pf2e.Check.roll(new game.pf2e.CheckModifier('no-cause-for-alarm-deafened',{modifiers:[]},[]),{actor,token:origin,type:'flat-check',domains:['flat-check'],dc:{value:5},options:new Set(['check:type:flat','action:no-cause-for-alarm']),skipDialog:true,createMessage:true},null,async roll=>{total=roll.total});
     if(!Number.isFinite(total))throw Error('耳聋的听觉动作平检未完成；本条使用不会自动重掷。');
     if(total<5)return '耳聋的DC 5听觉动作平检失败，本次无需惊慌不产生效果。';
    }
   }
   let checked;
   // No dc.slug/statistic and no target argument: the shared check cannot inherit
   // whichever creature the executing GM happened to select.
   await statistic.roll({token:origin,item,action:'no-cause-for-alarm',dc:{value:targets[0].dc,visible:false},traits:TRAITS,extraRollOptions:['action:no-cause-for-alarm',...TRAITS.map(t=>`item:trait:${t}`)],skipDialog:true,createMessage:true,callback:async(roll,_outcome,card)=>{checked={roll,card}}});
   if(!checked)throw Error('交涉检定未完成；本条使用不会自动重掷。');
   const {roll,card}=checked,context=card.flags?.pf2e?.context;
   const natural=roll.isDeterministic?roll.terms?.find(t=>t.constructor?.name==='NumericTerm')?.total:roll.dice?.find(d=>d.faces===20)?.total;
   const result={total:roll.total,natural},baseOptions=[...(context?.options??[]),...(context?.contextualOptions?.postRoll??[])].filter(o=>!o.startsWith('check:total:delta:'));
   const optionsFor=dc=>new Set([...baseOptions,`check:total:delta:${result.total-dc}`]);
   const reference=adjustmentMap(game,raw,optionsFor(targets[0].dc)),nativeDegree=degreeForSharedCheck(result,targets[0].dc,reference);
   if(!equivalent(reference,context?.dosAdjustments)||OUTCOMES[nativeDegree.value]!==context?.outcome||OUTCOMES[nativeDegree.unadjusted]!==context?.unadjustedOutcome)throw Error('此检定的原生成功度修正无法完整对照，尚未更改目标状态；请GM查看原检定。');
   const outcomes=[];
   for(const target of targets){
    const degree=degreeForSharedCheck(result,target.dc,adjustmentMap(game,raw,optionsFor(target.dc))),reduction=degree.value===3?2:degree.value===2?1:0;
    await upsertOwnedEffect(target.actor,IMMUNITY,{name:'无需惊慌：暂时免疫',type:'effect',img:item.img??'icons/svg/aura.svg',system:{slug:'no-cause-for-alarm-immunity',duration:{value:1,unit:'hours',expiry:'turn-start',sustained:false},start:{value:now,initiative:null},rules:[],tokenIcon:{show:false}},flags:{[MODULE_ID]:{kind:'social-alarm-immunity',sourceId:SOURCE,usageMessageId:message.id,checkMessageId:card.id,expiresAt:now+3600}}});
    for(let i=0;i<reduction&&target.actor.getCondition('frightened')?.value>0;i++)await target.actor.decreaseCondition('frightened');
    outcomes.push({actorUuid:target.actor.uuid,tokenUuid:target.token.uuid,dc:target.dc,degree:OUTCOMES[degree.value],before:target.fear,after:target.actor.getCondition('frightened')?.value??0});
   }
   await globalThis.ChatMessage.create({speaker:globalThis.ChatMessage.getSpeaker({actor,token:origin}),whisper:values(game.users).filter(u=>u.isGM).map(u=>u.id),content:`<p>无需惊慌：一次交涉检定（${roll.total}），${escape(language)}。</p><ul>${outcomes.map(o=>`<li>${escape(targets.find(t=>t.actor.uuid===o.actorUuid).actor.name)}：${escape(o.degree)}，惊惧 ${o.before} → ${o.after}；暂时免疫1小时。</li>`).join('')}</ul>`,flags:{[MODULE_ID]:{usageGenerated:true,socialAlarm:{usageMessageId:message.id,checkMessageId:card.id,language,outcomes,skipped}}}});
   return '已完成无需惊慌的交涉检定、惊惧调整与1小时暂时免疫。';
  });
 }
 async function maintain(actor){
  if(!isActiveGM(game))return;
  return queue.run('social:no-cause-for-alarm',async()=>{
   const ids=values(actor.items).filter(i=>own(i).kind==='social-alarm-immunity'&&own(i).expiresAt<=game.time.worldTime).map(i=>i.id);if(ids.length)await actor.deleteEmbeddedDocuments('Item',ids);
  });
 }
 return {resolveAction,executeUsage,maintain,register:()=>()=>{}};
}
