import {MODULE_ID} from './rules.mjs';
import {confirmManualFlatCheck} from './manual-native-roll.mjs';
import {SerialActions} from './runtime.mjs';
import {getSourceId,isActiveGM} from './native-context.mjs';
import {electricityRollWitness} from './eldamon-electricity.mjs';

const SOURCE='Compendium.pf2e.feats-srd.Item.6ON8DjFXSMITZleX';
const SQUAWK='Compendium.pf2e.feats-srd.Item.CCmiEmS7ZgyQUfhn';
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
const author=message=>message?.author?.id??message?.author??message?.user?.id??message?.user;
const ordered=value=>Array.isArray(value)?value.map(ordered):value&&typeof value==='object'?Object.fromEntries(Object.keys(value).sort().map(key=>[key,ordered(value[key])])):value;
function paidSquawk(game,actor,user,card,roll,nativeDegree){
 const proof=own(card).reactionChecks;if(proof?.reaction!=='squawk')return false;
 const records=own(actor).reactionChecks?.reactions??[],record=records.find(r=>r.nonce===proof.nonce&&r.kind==='squawk'),payment=game.messages.get(record?.checkId),feature=values(actor.items).find(i=>i.type==='feat'&&getSourceId(i)===SQUAWK),paid=own(payment).reactionChecks,previous=proof.previousRoll,current=structuredClone(roll.toJSON()),context=card.flags?.pf2e?.context;
 if(current.options)current.options.degreeOfSuccess=0;
 const sameRoll=previous?.evaluated===true&&previous.options?.degreeOfSuccess===0&&JSON.stringify(ordered(electricityRollWitness(previous)))===JSON.stringify(ordered(electricityRollWitness(current)));
 if(proof.kind!=='check-reaction-result'||proof.actorUuid!==actor.uuid||!proof.nonce||!record||!['claimed','used'].includes(record.state)||record.userId!==user.id||record.state==='used'&&record.resultMessageId!==card.id||game.messages.get(card.id)!==card||author(card)!==user.id||card.speaker?.actor!==actor.id||!actor.testUserPermission(user,'OWNER')||!game.users.get(author(payment))?.isGM||payment?.speaker?.actor!==actor.id||paid?.kind!=='reaction-use'||paid.reaction!=='squawk'||paid.nonce!==proof.nonce||!feature||payment.flags?.pf2e?.origin?.uuid!==feature.uuid||payment.flags.pf2e.origin.actor!==actor.uuid||context?.type!=='skill-check'||!context.options?.includes('action:no-cause-for-alarm')||nativeDegree.value!==0||context.outcome!=='failure'||roll.options?.degreeOfSuccess!==1||!sameRoll)throw Error('喀咯！的已付款反应或原检定回执不匹配，尚未更改目标状态。');
 return true;
}
function soundBlocked(origin,target){
 const backend=globalThis.CONFIG?.Canvas?.polygonBackends?.sound,level=origin.parent?.levels?.get(origin._source?.level??origin.level);
 if(!backend?.testCollision||!level)throw Error('无法确认场景层级的原生声音传播。');
 // Foundry 14 Token.checkCollision explicitly rejects type:"sound"; its native
 // polygon backend accepts an explicit level and elevations without a source.
 return backend.testCollision({...origin.object.center,elevation:origin.elevation??0},{...target.object.center,elevation:target.elevation??0},{type:'sound',mode:'any',level});
}

function captureTokenActor(token,game){
 const actor=token?.actor,baseActor=token?.baseActor,actorId=token?.actorId,actorLink=token?.actorLink,synthetic=actor?.isToken===true;
 return {actor,current:()=>{
  if(!actor||!token||token.actor!==actor||token.baseActor!==baseActor||token.actorId!==actorId||token.actorLink!==actorLink||(actor.isToken===true)!==synthetic)return false;
  if(synthetic)return token.documentName==='Token'&&actor.token===token&&actor.parent===token&&actorLink===false&&!!baseActor&&actorId===baseActor.id&&actor.id===baseActor.id&&game.actors.get(actorId)===baseActor;
  return baseActor===undefined||baseActor===actor&&actorId===actor.id&&actorLink===true&&game.actors.get(actorId)===baseActor;
 }};
}

export function createSocialAutomation({game,fromUuid=globalThis.fromUuid,choose,runNative,onError=()=>{}}={}){
 const queue=new SerialActions();
 const resolveAction=item=>item?.type==='feat'&&getSourceId(item)===SOURCE?'social:no-cause-for-alarm':null;
 async function choice(actor,user,title,choices){if(choices.length===1)return choices[0].value;const result=await choose({actor,user,title,choices});if(result==null)return null;if(!choices.some(c=>c.value===result))throw Error('无效的无需惊慌规则选择。');return result;}
 async function executeUsage({actor,item,message,user,action}){
  if(!isActiveGM(game)||!actor?.testUserPermission(user,'OWNER')||item?.actor?.uuid!==actor.uuid||resolveAction(item)!==action)throw Error('无权执行此来源的无需惊慌。');
   const originalGM=game.user,sourceOrigin=structuredClone(message.flags?.pf2e?.origin??{}),sourceSpeaker=structuredClone(message.speaker??{});
   const scene=game.scenes.get(sourceSpeaker.scene),origin=scene?.tokens.get(sourceSpeaker.token);
   const sourceBinding=captureTokenActor(origin,game);
   const assertSource=()=>{
    if(sourceBinding.actor!==actor||!sourceBinding.current())throw Error('无需惊慌的原始Token与基础Actor已改变；不会重掷。');
    if(game.user!==originalGM||game.users.activeGM!==originalGM||game.users.get(originalGM.id)!==originalGM)throw Error('无需惊慌的原始主GM文档已改变；不会重掷。');
    if(!isActiveGM(game)||game.users.get(user.id)!==user||!actor.testUserPermission(user,'OWNER')||actor.items.get(item.id)!==item||item.actor!==actor||resolveAction(item)!==action||item.suppressed===true||item.isSuppressed||item.system?.suppressed||game.messages.get(message.id)!==message||author(message)!==user.id||message.flags?.pf2e?.origin?.uuid!==item.uuid||message.flags.pf2e.origin.actor!==actor.uuid||JSON.stringify(message.flags.pf2e.origin)!==JSON.stringify(sourceOrigin)||JSON.stringify(message.speaker)!==JSON.stringify(sourceSpeaker)||!scene||!origin||game.scenes.get(scene.id)!==scene||scene.tokens.get(origin.id)!==origin||origin.actor!==actor||!origin.object||!actor.isToken&&game.actors.get(actor.id)!==actor||actor.isDead||actor.hasCondition?.('unconscious'))throw Error('无需惊慌的原始来源、权限或Token已改变；不会重掷。');
   };
   assertSource();
   const upsert=async(recipient,data,guard)=>{
    const existing=values(recipient.items).filter(i=>i.type==='effect'&&own(i).nativeEffectKey===IMMUNITY),next=structuredClone(data);delete next._id;
    next.flags={...next.flags,[MODULE_ID]:{...next.flags?.[MODULE_ID],nativeEffectKey:IMMUNITY}};
    const embedded=item=>item?.type==='effect'&&item.actor===recipient&&(!item.parent||item.parent===recipient)&&recipient.items.get(item.id)===item;
    const saved=item=>embedded(item)&&Object.entries(next.flags[MODULE_ID]).every(([key,value])=>own(item)[key]===value);
    const original=()=>{if(existing.some(item=>!embedded(item)))throw Error('无需惊慌的原始免疫效果文档已改变；保留已有检定。');};
    guard();
    if(existing.length){original();await existing[0].update(next);guard();if(!saved(existing[0]))throw Error('无需惊慌的免疫更新未能确认；不会继续降低惊惧。');if(existing.length>1){original();await recipient.deleteEmbeddedDocuments('Item',existing.slice(1).map(i=>i.id));guard();if(existing.slice(1).some(item=>recipient.items.get(item.id)))throw Error('无需惊慌的重复免疫清理尚未确认。');}if(!saved(existing[0]))throw Error('无需惊慌的原始免疫效果已改变。');return existing[0];}
    const result=await recipient.createEmbeddedDocuments('Item',[next]);guard();if(!Array.isArray(result)||result.length!==1||!saved(result[0]))throw Error('无需惊慌的免疫创建未能确认；不会继续降低惊惧。');return result[0];
   };
  return queue.run('social:no-cause-for-alarm',async()=>{
    assertSource();
   if(own(actor).socialUses?.includes(message.id))return '本次无需惊慌已结算。';
   if(message.flags?.pf2e?.flatCheck?.result==='fail')return '原生平检失败，本次无需惊慌不产生效果。';
   if(!origin?.object||origin.actor?.uuid!==actor.uuid)throw Error('请从场景中的角色使用无需惊慌，以确定10尺散发起点。');
   if(actor.isDead||actor.hasCondition?.('unconscious'))throw Error('当前无法发声使用无需惊慌。');
    const now=game.time.worldTime,candidates=new Map(),candidateActors=new Map(),candidateBindings=new Map(),skipped=[];
   for(const target of values(scene.tokens)){
    const recipient=target.actor,object=target.object;if(!recipient||!object||!(recipient.getCondition?.('frightened')?.value>0))continue;
    const distance=origin.object.distanceTo(object);if(!Number.isFinite(distance)||distance>10)continue;
    if(immune(recipient,now)){skipped.push({actorUuid:recipient.uuid,reason:'temporary-immunity'});continue;}
    if(recipient.hasCondition?.('deafened')||recipient.isImmuneTo?.(item)){skipped.push({actorUuid:recipient.uuid,reason:'trait-immunity-or-deafened'});continue;}
    if(target.uuid!==origin.uuid){
     if(typeof origin.object.checkCollision!=='function')throw Error('场景缺少原生阻挡检测，无法确认散发范围。');
     if(origin.object.checkCollision(object.center,{origin:origin.object.center,type:'move',mode:'any'})||soundBlocked(origin,target))continue;
    }
     if(!candidates.has(recipient.uuid)){candidates.set(recipient.uuid,target);candidateActors.set(target,recipient);candidateBindings.set(target,captureTokenActor(target,game));}
   }
   if(!candidates.size)return '10尺内没有可受影响且未免疫的惊惧生物。';
   if(candidates.has(actor.uuid)){
    const include=await choice(actor,user,'无需惊慌：散发是否包括自身',[{value:'exclude',label:'不影响自身'},{value:'include',label:'包括自身'}]);if(include===null)return '已取消。';if(include==='exclude')candidates.delete(actor.uuid);
     assertSource();
   }
   const spoken=languages(actor);
   if(!spoken.length)return '施术者没有已记录的可用语言。';
   const language=await choice(actor,user,'无需惊慌：本次使用的语言',spoken.map(value=>({value,label:game.i18n?.localize?.(globalThis.CONFIG?.PF2E?.languages?.[value]??value)??value})));if(language===null)return '已取消。';
    assertSource();
   const recipients=values(candidates).filter(t=>languages(t.actor).includes(language));if(!recipients.length)return '没有理解本次语言的可受影响生物。';
    const targets=recipients.map(token=>{const recipient=candidateActors.get(token);return {token,actor:recipient,dc:recipient.getStatistic?.('will')?.dc?.value,fear:recipient.getCondition('frightened')?.value??0};});
   if(targets.some(t=>!Number.isFinite(t.dc)))throw Error('目标缺少原生意志DC，尚未检定。');
    const assertRecipient=(target,checkImmunity=false)=>{
     assertSource();const recipient=target.actor,token=target.token,distance=origin.object.distanceTo(token.object);
     if(!candidateBindings.get(token)?.current())throw Error('无需惊慌的原始目标Token与基础Actor已改变。');
     if(scene.tokens.get(token.id)!==token||token.parent!==scene||token.actor!==recipient||!recipient.isToken&&game.actors.get(recipient.id)!==recipient||!token.object||!Number.isFinite(distance)||distance<0||distance>10||!languages(actor).includes(language)||!languages(recipient).includes(language)||recipient.hasCondition?.('deafened')||recipient.isImmuneTo?.(item)||recipient.getStatistic?.('will')?.dc?.value!==target.dc||(recipient.getCondition('frightened')?.value??0)!==(target.expectedFear??target.fear)||checkImmunity&&immune(recipient,game.time.worldTime))throw Error('无需惊慌的原始目标、范围或状态已改变；保留已有检定与效果。');
     if(token!==origin&&(origin.object.checkCollision(token.object.center,{origin:origin.object.center,type:'move',mode:'any'})||soundBlocked(origin,token)))throw Error('无需惊慌的声音范围已改变。');
    };
    for(const target of targets)assertRecipient(target,true);
   const statistic=actor.getStatistic?.('diplomacy')??actor.skills?.diplomacy,domains=statistic?.check?.domains??statistic?.domains;
   if(!statistic?.roll||!Array.isArray(domains))throw Error('未找到原生交涉检定。');
   const raw=domains.flatMap(domain=>actor.synthetics?.degreeOfSuccessAdjustments?.[domain]??[]).map(r=>({...r,adjustments:structuredClone(r.adjustments)}));
   await actor.update({[`flags.${MODULE_ID}.socialUses`]:[...(own(actor).socialUses??[]).slice(-127),message.id]});
    assertSource();for(const target of targets)assertRecipient(target,true);
   if(actor.hasCondition?.('deafened')){
    const pf=message.flags?.pf2e??{},postInfo=Object.keys(pf).length===1&&pf.origin&&!pf.origin.sourceId;
    const patreonGate=game.modules?.get('patreon-v3')?.active&&postInfo&&['all','attack'].includes(game.settings?.get('patreon-v3','flatCheck'));
    if(!patreonGate){
     let total;
     if(runNative){
      const native=await runNative({actor,item,message,user},{type:'flat',itemUuid:item.uuid,tokenUuid:origin.uuid,dc:{value:5},label:'无需惊慌 · 耳聋听觉动作平检',action:'no-cause-for-alarm',options:['check:type:flat','action:no-cause-for-alarm']});
       assertSource();
      if(native.status!=='rolled')return '已取消投骰，本次无需惊慌不产生效果。';total=native.check.rolls[0].total;
     }else{
      if(!await confirmManualFlatCheck({label:'无需惊慌 · 耳聋听觉动作平检',dc:5}))return '已取消投骰，本次无需惊慌不产生效果。';
       assertSource();
      if(!game.pf2e.Check?.roll||!game.pf2e.CheckModifier)throw Error('无法执行耳聋的原生DC 5听觉动作平检。');
      await game.pf2e.Check.roll(new game.pf2e.CheckModifier('no-cause-for-alarm-deafened',{modifiers:[]},[]),{actor,token:origin,type:'flat-check',domains:['flat-check'],dc:{value:5},options:new Set(['check:type:flat','action:no-cause-for-alarm']),skipDialog:false,event:null,createMessage:true},null,async roll=>{total=roll.total});
       assertSource();
     }
     if(!Number.isFinite(total))throw Error('耳聋的听觉动作平检未完成；本条使用不会自动重掷。');
     if(total<5)return '耳聋的DC 5听觉动作平检失败，本次无需惊慌不产生效果。';
    }
   }
   let checked;
   // No dc.slug/statistic and no target argument: the shared check cannot inherit
   // whichever creature the executing GM happened to select.
   if(runNative){
    const native=await runNative({actor,item,message,user},{type:'check',statistic:'diplomacy',itemUuid:item.uuid,tokenUuid:origin.uuid,action:'no-cause-for-alarm',dc:{value:targets[0].dc,visible:false},traits:TRAITS,options:['action:no-cause-for-alarm',...TRAITS.map(t=>`item:trait:${t}`)]});
     assertSource();
    if(native.status!=='rolled')return '已取消交涉投骰，本次无需惊慌不产生效果。';checked={roll:native.check.rolls[0],card:native.check};
   }else await statistic.roll({token:origin,item,action:'no-cause-for-alarm',dc:{value:targets[0].dc,visible:false},traits:TRAITS,extraRollOptions:['action:no-cause-for-alarm',...TRAITS.map(t=>`item:trait:${t}`)],skipDialog:false,event:null,createMessage:true,callback:async(roll,_outcome,card)=>{checked={roll,card}}});
   if(!checked)throw Error('交涉检定未完成；本条使用不会自动重掷。');
   const {roll,card}=checked,context=card.flags?.pf2e?.context;
   const natural=roll.isDeterministic?roll.terms?.find(t=>t.constructor?.name==='NumericTerm')?.total:roll.dice?.find(d=>d.faces===20)?.total;
    const contextProof=JSON.stringify(ordered(context)),totalProof=roll.total;
    const assertCheck=()=>{
     assertSource();const currentNatural=roll.isDeterministic?roll.terms?.find(t=>t.constructor?.name==='NumericTerm')?.total:roll.dice?.find(d=>d.faces===20)?.total;
     if(game.messages.get(card.id)!==card||author(card)!==user.id||card.speaker?.actor!==actor.id||card.rolls?.[0]!==roll||roll.total!==totalProof||currentNatural!==natural||JSON.stringify(ordered(card.flags?.pf2e?.context))!==contextProof)throw Error('无需惊慌的原始检定回执已改变；不会重掷。');
    };
   const result={total:roll.total,natural},baseOptions=[...(context?.options??[]),...(context?.contextualOptions?.postRoll??[])].filter(o=>!o.startsWith('check:total:delta:'));
   const optionsFor=dc=>new Set([...baseOptions,`check:total:delta:${result.total-dc}`]);
   const reference=adjustmentMap(game,raw,optionsFor(targets[0].dc)),nativeDegree=degreeForSharedCheck(result,targets[0].dc,reference),squawk=paidSquawk(game,actor,user,card,roll,nativeDegree);
    assertCheck();
   if(!equivalent(reference,context?.dosAdjustments)||OUTCOMES[squawk?1:nativeDegree.value]!==context?.outcome||OUTCOMES[nativeDegree.unadjusted]!==context?.unadjustedOutcome)throw Error('此检定的原生成功度修正无法完整对照，尚未更改目标状态；请GM查看原检定。');
   const outcomes=[];
   for(const target of targets){
     assertCheck();assertRecipient(target,true);
    const degree=degreeForSharedCheck(result,target.dc,adjustmentMap(game,raw,optionsFor(target.dc)));if(squawk&&degree.value===0)degree.value=1;const reduction=degree.value===3?2:degree.value===2?1:0;
     await upsert(target.actor,{name:'无需惊慌：暂时免疫',type:'effect',img:item.img??'icons/svg/aura.svg',system:{slug:'no-cause-for-alarm-immunity',duration:{value:1,unit:'hours',expiry:'turn-start',sustained:false},start:{value:now,initiative:null},rules:[],tokenIcon:{show:false}},flags:{[MODULE_ID]:{kind:'social-alarm-immunity',sourceId:SOURCE,usageMessageId:message.id,checkMessageId:card.id,expiresAt:now+3600}}},()=>{assertCheck();assertRecipient(target);});
     for(let i=0;i<reduction&&target.actor.getCondition('frightened')?.value>0;i++){
      assertCheck();assertRecipient(target);const before=target.expectedFear??target.fear;
      await target.actor.decreaseCondition('frightened');target.expectedFear=Math.max(0,before-1);assertCheck();assertRecipient(target);
     }
    outcomes.push({actorUuid:target.actor.uuid,tokenUuid:target.token.uuid,dc:target.dc,degree:OUTCOMES[degree.value],before:target.fear,after:target.actor.getCondition('frightened')?.value??0});
   }
    assertCheck();for(const target of targets)assertRecipient(target);
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
