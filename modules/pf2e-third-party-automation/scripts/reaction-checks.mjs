import {MODULE_ID} from './rules.mjs';
import {SerialActions} from './runtime.mjs';
import {getSourceId,isActiveGM,resolveMessageTargets} from './native-context.mjs';
import {genericReactionAvailable,withReactionReservation} from './fear-automation.mjs';
import {createEatFortune} from './eat-fortune.mjs';

export const REACTION_CHECK_SOURCES=Object.freeze({pointed:'Compendium.pf2e.actionspf2e.Item.xccOiNL2W1EtfUYl',pointedEffect:'Compendium.pf2e.feat-effects.Item.SScln8qRQgVC6Brz',clock:'Compendium.pf2e.feats-srd.Item.3aG0gkHulBIHqqGE',clockEffect:'Compendium.pf2e.feat-effects.Item.LbICHKe5jLMxhaOw',squawk:'Compendium.pf2e.feats-srd.Item.CCmiEmS7ZgyQUfhn',eat:'Compendium.pf2e.feats-srd.Item.rFmJVDdB313EibTs'});
const values=c=>Array.from(c?.values?.()??c??[]),own=d=>d?.flags?.[MODULE_ID]?.reactionChecks??{},OUTCOMES=['criticalFailure','failure','success','criticalSuccess'];
const escape=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const languages=a=>a?.system?.details?.languages?.value??[];
const POINTED_TRAITS=['auditory','concentrate','investigator','linguistic','mental'];

/** Keep the ordinary native modifier dialog, dice and DC calculation. Deliver only
 * the final result with the caller's publication choice, then its callback once.
 */
export async function runCheckReactionPipeline({game,check,context,event=null,callback,native,decide,beforeReroll,publish=data=>globalThis.ChatMessage.create(data)}){
 const createMessage=context.createMessage!==false;
 let captured;const collect=async(roll,outcome,card,callbackEvent)=>{captured={roll,outcome,card,event:callbackEvent}};
 const options=context.options instanceof Set?context.options:new Set(context.options??[]),draftContext={...context,options,createMessage:false};
 const originalReturn=await native(check,draftContext,event,collect);if(!captured)return originalReturn;
 const original=captured,originalRollData=captured.roll.toJSON(),decision=await decide({...original,check,context:draftContext}),reaction=typeof decision==='string'?decision:decision?.reaction??null;
 const disruption=reaction==='clock'?await beforeReroll?.({...original,check,context:draftContext,decision}):null;
 const reroll=(reaction==='clock'||reaction==='halfling-luck')&&!disruption?.disrupted;
 if(reroll){
  const rerollCheck=reaction==='clock'?new game.pf2e.CheckModifier(check.slug,{modifiers:check.modifiers},[new game.pf2e.Modifier({slug:'turn-back-the-clock',label:'倒转光阴',modifier:1,type:'circumstance'})]):check;
  const options=new Set(draftContext.options);options.add('fortune');options.add('check:reroll');
  captured=null;await native(rerollCheck,{...draftContext,options,isReroll:true,skipDialog:true,rollTwice:false,substitutions:[],createMessage:false},null,collect);
  if(!captured)throw Error(`${reaction==='clock'?'倒转光阴':'半身人幸运'}的原生重掷未完成；次数已使用，不会自动重试。`);
 }
 const data=captured.card.toObject();delete data._id;
 if(reaction==='squawk'){
  captured.roll.options.degreeOfSuccess=1;captured.outcome='failure';data.flags.pf2e.context.outcome='failure';
  const root=globalThis.document?.createElement?.('div');if(root){root.innerHTML=data.flavor??'';const result=root.querySelector('.degree-of-success');if(result)result.textContent='喀咯！：大失败 → 失败';root.querySelector('ul.notes')?.remove();data.flavor=root.innerHTML;}else data.flavor=(data.flavor??'')+'<p>喀咯！：大失败 → 失败。</p>';
  // The original context keeps the raw degree; only this committed result is changed.
  data.content=await captured.roll.render();
 }
 if(reroll)data.content=`<div class="reroll-discard">${await game.pf2e.Check.renderReroll(original.roll,{isOld:true})}</div><div class="reroll-second">${await game.pf2e.Check.renderReroll(captured.roll,{isOld:false})}</div>`;
 if(reaction){data.flags??={};data.flags[MODULE_ID]={...data.flags[MODULE_ID],reactionChecks:{kind:'check-reaction-result',reaction,nonce:decision?.nonce??null,actorUuid:decision?.actorUuid??context.actor?.uuid??null,previousRoll:originalRollData,...(disruption?.disrupted?{disrupted:true}:{})}};}
 if(disruption?.disrupted){const options=new Set(data.flags.pf2e.context.options??[]);for(const option of ['fortune','misfortune',`${MODULE_ID}:eat:${disruption.nonce}`])options.add(option);data.flags.pf2e.context.options=[...options];data.flags.pf2e.context.eatFortune=disruption;data.flavor=(data.flavor??'')+'<p>倒转光阴被吞噬福祸打断；保留原检定，双方反应和次数已使用。</p>';}
 const finalContext=data.flags.pf2e.context;for(const key of ['outcome','unadjustedOutcome','isReroll','rollTwice','substitutions'])if(key in finalContext)context[key]=finalContext[key];
 for(const option of finalContext.options??[])options.add(option);context.options=options;
 data.rolls=[captured.roll.toJSON()];let message=captured.card;
 if(createMessage){message=await publish(data);if(!message)throw Error('原生检定结果未能发布；不会自动重掷。');}
 // Native Check supplies a ChatMessage draft even when it is not published.
 // Keep that document (and its privacy fields) for Toolbelt's original callback.
 else message.updateSource(data);
 if(callback)await callback(captured.roll,captured.outcome,message,captured.event);return captured.roll;
}

export function createReactionChecks({game,fromUuid=globalThis.fromUuid,choose,onError=()=>{},nativeCheckMiddleware,halflingLuck}={}){
 const queue=new SerialActions(),tracked=new Map(),reactors=new Map(),nativeInvocations=new Map();let socket;
 const eatFortune=createEatFortune({game,fromUuid,choose,onError});
 const requireReactionGM=()=>{if(!isActiveGM(game))throw Error('主GM已交接，本次反应已停止；已有认领或费用不会自动回滚或重试。')};
 // A request already sent may have committed. Stop on handoff without undoing it.
 const asReactionGM=async operation=>{requireReactionGM();const result=await operation();requireReactionGM();return result};
 const now=()=>game.time.worldTime??0;
 const epoch=actor=>{const c=game.combat,index=c?.turns?.findIndex(t=>t.actor?.uuid===actor.uuid)??-1;return c?.started&&index>=0?`${c.id}:${c.round-(index>c.turn?1:0)}`:null};
 const reactionAvailable=actor=>!epoch(actor)||genericReactionAvailable(actor,game);
 const canReact=actor=>!actor.isDead&&actor.canAct!==false&&!actor.hasCondition?.('unconscious')&&!actor.hasCondition?.('stunned');
 const resolveAction=item=>item?.type==='action'&&getSourceId(item)===REACTION_CHECK_SOURCES.pointed?'reaction-checks:pointed-question':null;
 const immune=(actor,kind)=>values(actor.items).some(i=>own(i).kind===kind&&(own(i).expiresAt??0)>now());
 const feature=(actor,key)=>values(actor?.items).find(i=>i.type==='feat'&&getSourceId(i)===REACTION_CHECK_SOURCES[key]);
 const forget=actor=>{if(!actor?.uuid)return;if(tracked.get(actor.uuid)===actor)tracked.delete(actor.uuid);if(reactors.get(actor.uuid)===actor)reactors.delete(actor.uuid)};
 function liveActor(actor){
  if(!actor?.uuid)return false;if(!actor.isToken)return game.actors?.get?.(actor.id)===actor;
  const token=actor.token,scene=token?.parent;
  return !!scene&&game.scenes?.get?.(scene.id)===scene&&scene.tokens?.get?.(token.id)===token&&token.actorLink===false&&!!token.baseActor&&game.actors?.get?.(token.actorId)===token.baseActor&&token.actor===actor;
 }
 // PF2e's primary updater already dispatches expiration on combat/world time
 // changes. Never send a competing delete or override removeEffects=false.
 const nativeExpiry=item=>item.type==='effect'&&typeof item.remainingDuration?.expired==='boolean'&&typeof game.pf2e?.effectTracker?.refresh==='function';
 const fallbackExpired=item=>!nativeExpiry(item)&&((own(item).expiresAt&&own(item).expiresAt<=now())||own(item).turnEnd&&ended(own(item).turnEnd));
 const track=actor=>{if(!liveActor(actor)){forget(actor);return;}if(values(actor.items).some(i=>own(i).expiresAt||own(i).turnEnd))tracked.set(actor.uuid,actor);else tracked.delete(actor.uuid);if(feature(actor,'clock')||feature(actor,'squawk'))reactors.set(actor.uuid,actor);else reactors.delete(actor.uuid)};
 const pick=async(actor,user,title,choices)=>{const selected=choices.length===1?choices[0].value:await choose({actor,user,title,choices});if(selected!=null&&!choices.some(c=>c.value===selected))throw Error('无效的规则选择。');return selected};
 function turnEnd(actor){const c=game.combat,index=c?.turns?.findIndex(t=>t.actor?.uuid===actor.uuid)??-1;return c?.started&&index>=0?{combatId:c.id,combatantId:c.turns[index].id,round:c.round+(index<c.turn?1:0),initiative:c.turns[index].initiative}:null}
 function ended(t){const c=game.combat,index=c?.turns?.findIndex(x=>x.id===t.combatantId)??-1;return !c?.started||c.id!==t.combatId||index<0||c.round>t.round||c.round===t.round&&c.turn>index}
 async function mark(actor,key,data){
  const existing=values(actor.items).filter(i=>i.type==='effect'&&i.flags?.[MODULE_ID]?.nativeEffectKey===key),next=structuredClone(data);delete next._id;
  next.flags={...next.flags,[MODULE_ID]:{...next.flags?.[MODULE_ID],nativeEffectKey:key}};
  // The shared upsert can issue a second write for duplicates: guard both writes.
  if(existing.length){await asReactionGM(()=>existing[0].update(next));if(existing.length>1)await asReactionGM(()=>actor.deleteEmbeddedDocuments('Item',existing.slice(1).map(i=>i.id)))}
  else await asReactionGM(()=>actor.createEmbeddedDocuments('Item',[next]));
  track(actor);
 }
 async function reactionCard(actor,item,nonce,kind){
  // AAT annotates the native card with its exact frequency update. Its wrapper
  // cannot annotate unsaved drafts, so this is a regular posted feature card.
  const card=await asReactionGM(()=>item.toMessage());if(!card?.id)throw Error('原生反应卡未创建，不能重试已认领反应。');
  await asReactionGM(()=>card.update({[`flags.${MODULE_ID}.usageGenerated`]:true,[`flags.${MODULE_ID}.reactionChecks`]:{kind:'reaction-use',reaction:kind,nonce}}));return card;
 }
 async function squawkWitnesses(actor,origin,target){
  const visible=values(origin.parent.tokens).filter(t=>t.actor&&t.actor.uuid!==actor.uuid&&t.actor.canSee!==false&&!t.actor.hasCondition?.('unconscious')&&t.object&&typeof t.object.checkCollision==='function'&&!t.object.checkCollision(origin.object.center,{origin:t.object.center,type:'sight',mode:'any'}));
  const unique=[...new Map(visible.map(t=>[t.actor.uuid,t])).values()];if(unique.length<=1)return unique;
  const mode=await asReactionGM(()=>pick(actor,game.user,'喀咯！：实际目击本次表现的生物',[{value:'target',label:'只有本次检定目标目击'},{value:'visible',label:`场景内这 ${unique.length} 位有视线的生物均目击`},{value:'select',label:'逐一选择实际目击者'}]));
  if(mode==='target')return unique.filter(t=>t.uuid===target.uuid);if(mode==='visible')return unique;if(mode!== 'select')throw Error('尚未确定喀咯的目击者；反应未消费。');
  const selected=[];while(unique.length){const value=await asReactionGM(()=>pick(actor,game.user,'喀咯！：选择目击者',[{value:'done',label:'完成'},...unique.map(t=>({value:t.uuid,label:t.name??t.actor.name}))]));if(!value||value==='done')break;const index=unique.findIndex(t=>t.uuid===value);selected.push(unique.splice(index,1)[0]);}return selected;
 }
 async function decideCheckReaction(payload,user){
  if(!isActiveGM(game)||!user||typeof payload?.nonce!=='string'||!/^[A-Za-z0-9-]{8,80}$/.test(payload.nonce))throw Error('反应检定回执或GM权限无效。');
  const actor=await fromUuid(payload.actorUuid);if(!actor?.testUserPermission?.(user,'OWNER'))throw Error('无权处理这个角色的检定反应。');
  return queue.run(`reaction:${actor.uuid}`,async()=>{
   requireReactionGM();
   if(own(actor).reactions?.some(r=>r.nonce===payload.nonce))return null;
   if(!['skill-check','saving-throw'].includes(payload.type)||![0,1].includes(payload.degree)||!canReact(actor)||!reactionAvailable(actor))return null;
   const triggerEpoch=epoch(actor);
   const origin=payload.tokenUuid?await fromUuid(payload.tokenUuid):null;if(payload.tokenUuid&&(!origin||origin.actor?.uuid!==actor.uuid||origin.documentName!=='Token'))throw Error('反应的原检定Token来源不匹配。');
   const clock=feature(actor,'clock'),squawk=feature(actor,'squawk'),choices=[];
   if(clock&&!payload.isReroll&&payload.rerollable!==false&&!payload.fortune&&(clock.system.frequency?.value??clock.system.frequency?.max??0)>0)choices.push({value:'clock',label:'使用倒转光阴（反应；每日1次）'});
   const target=payload.targetUuid?await fromUuid(payload.targetUuid):null;
   if(squawk&&payload.degree===0&&payload.type==='skill-check'&&payload.domains?.some(d=>['deception','diplomacy','intimidation'].includes(d))&&origin?.object&&target?.object&&target.parent?.id===origin.parent?.id&&target.actor?.uuid!==actor.uuid&&!values(target.actor?.traits??target.actor?.system?.traits?.value).includes('tengu')&&!immune(target.actor,'squawk-immunity'))choices.push({value:'squawk',label:'使用喀咯！（反应；大失败视为失败）'});
   if(!choices.length)return null;
   const selected=await asReactionGM(()=>pick(actor,user,'本次检定失败：是否使用反应',[...choices,{value:'decline',label:'不使用反应'}]));if(!selected||selected==='decline')return null;
   const item=selected==='clock'?clock:squawk,witnesses=selected==='squawk'?await squawkWitnesses(actor,origin,target):[];
   const reserved=await withReactionReservation(actor,game,async()=>{
    requireReactionGM();
    if(epoch(actor)!==triggerEpoch||!canReact(actor)||feature(actor,selected)?.id!==item.id||!reactionAvailable(actor)||own(actor).reactions?.some(r=>r.nonce===payload.nonce))return false;
    const uses=selected==='clock'?(clock.system.frequency?.value??clock.system.frequency.max):null;if(selected==='clock'&&uses<1)return false;
    await asReactionGM(()=>actor.update({[`flags.${MODULE_ID}.reactionChecks.reactions`]:[...(own(actor).reactions??[]),{nonce:payload.nonce,kind:selected,state:'claimed',time:now(),epoch:epoch(actor),userId:user.id}]}));
    if(selected==='clock')await asReactionGM(()=>clock.update({'system.frequency.value':uses-1},{[MODULE_ID]:{usageInternal:true}}));return true;
   });if(!reserved)return null;
   for(const t of witnesses)await mark(t.actor,'reaction-checks:squawk-immunity',{name:'喀咯！：暂时免疫',type:'effect',img:item.img??'icons/creatures/birds/corvid-watchful-glowing-red.webp',system:{slug:'squawk-immunity',rules:[],duration:{value:24,unit:'hours',expiry:'turn-start',sustained:false},start:{value:now(),initiative:null},tokenIcon:{show:false}},flags:{[MODULE_ID]:{reactionChecks:{kind:'squawk-immunity',expiresAt:now()+86400,nonce:payload.nonce}}}});
   const card=await reactionCard(actor,item,payload.nonce,selected);await asReactionGM(()=>actor.update({[`flags.${MODULE_ID}.reactionChecks.reactions`]:(own(actor).reactions??[]).map(r=>r.nonce===payload.nonce?{...r,checkId:card.id}:r)}));return selected;
  });
 }
 const decide=async(state,targetSnapshot=null,userSnapshot=null)=>{
  const {context,roll,card}=state,actor=context.actor??(context.origin?.self?context.origin?.actor:context.target?.actor),pf=card.flags?.pf2e?.context??{},opts=new Set(pf.options??context.options??[]);
  const payload={actorUuid:actor.uuid,tokenUuid:context.token?.uuid??(context.origin?.self?context.origin?.token?.uuid:context.target?.token?.uuid)??null,targetUuid:(context.origin?.self?context.target?.token?.uuid:context.origin?.token?.uuid)??pf.target?.token??targetSnapshot,nonce:globalThis.foundry?.utils?.randomID?.()??globalThis.crypto.randomUUID(),type:context.type,domains:context.domains??[],degree:roll.options?.degreeOfSuccess??OUTCOMES.indexOf(pf.outcome),isReroll:!!(context.isReroll||roll.options?.isReroll),rerollable:roll.isRerollable,fortune:opts.has('fortune')||pf.substitutions?.some(s=>s.selected&&s.effectType==='fortune')||pf.rollTwice==='keep-higher',misfortune:opts.has('misfortune')||pf.substitutions?.some(s=>s.selected&&s.effectType==='misfortune')||pf.rollTwice==='keep-lower'};
  let reaction;if(isActiveGM(game))reaction=await decideCheckReaction(payload,userSnapshot??game.user);
  else {if(!socket||!game.users.activeGM)return null;const response=await socket.executeAsUser('reaction-checks:decide',game.users.activeGM.id,payload);if(!response.ok)throw Error(response.error);reaction=response.value;}
  return reaction?{reaction,nonce:payload.nonce,actorUuid:actor.uuid}:null;
 };
 async function settleReaction(message){
  const proof=own(message),context=message.flags?.pf2e?.context;if(!isActiveGM(game)||proof.kind!=='check-reaction-result'||!proof.nonce||!proof.actorUuid||!message.id||game.messages.get(message.id)!==message||!message.rolls?.length||!['skill-check','saving-throw'].includes(context?.type)||proof.reaction==='squawk'&&context.outcome!=='failure')return;
  if(proof.reaction==='clock'&&!context.isReroll){
   const eat=context.eatFortune,options=new Set(context.options??[]);if(!proof.disrupted||!eat?.disrupted||eat.sourceKind!=='clock'||eat.clockNonce!==proof.nonce||eat.sourceActorUuid!==proof.actorUuid||!options.has('fortune')||!options.has('misfortune')||!options.has(`${MODULE_ID}:eat:${eat.nonce}`))return;
   const reactor=await fromUuid(eat.reactorActorUuid);requireReactionGM();if(!own(reactor).reactions?.some(r=>r.kind==='eat'&&r.nonce===eat.nonce&&r.clockNonce===proof.nonce&&r.sourceActorUuid===proof.actorUuid&&r.sourceItemUuid===eat.sourceItemUuid&&['claimed','used'].includes(r.state)))return;
  }
  const actor=await fromUuid(proof.actorUuid),user=message.author??game.users.get(message.user?.id??message.user);if(!actor?.testUserPermission?.(user,'OWNER'))return;
  return queue.run(`reaction:${actor.uuid}`,async()=>{requireReactionGM();const records=own(actor).reactions??[],record=records.find(r=>r.nonce===proof.nonce&&r.kind===proof.reaction);if(!record||record.state==='used')return;const next=records.map(r=>r===record?{...r,state:'used',resultMessageId:message.id}:r),used=next.filter(r=>r.state==='used').slice(-128),unresolved=next.filter(r=>r.state!=='used');await asReactionGM(()=>actor.update({[`flags.${MODULE_ID}.reactionChecks.reactions`]:[...unresolved,...used]}))});
 }
 async function pointed({actor,item,message,user}){
  requireReactionGM();
  if(own(actor).uses?.includes(message.id))return '本次针对讯问已处理；不会重复检定。';
  const targetList=await resolveMessageTargets(message,{fromUuid});
  if(targetList.length!==1)throw Error('针对讯问需要选中一个能看见的非盟友生物。');
  const target=targetList[0],origin=game.scenes.get(message.speaker?.scene)?.tokens.get(message.speaker?.token),recipient=target.actor;
  if(!origin?.object||origin.actor?.uuid!==actor.uuid||!target.object||target.parent?.id!==origin.parent?.id)throw Error('针对讯问需要明确的同场景来源和目标Token。');
  if(recipient?.uuid===actor.uuid||recipient?.isAllyOf?.(actor)||!recipient)throw Error('针对讯问的目标必须是非盟友生物。');
  if(immune(recipient,'pointed-immunity'))return '该目标仍暂时免疫针对讯问（1小时）。';
  if(actor.canSee===false||actor.isDead||actor.hasCondition?.('unconscious')||target.hidden&&!user.isGM)throw Error('目前无法看见目标并向其提问。');
  if(typeof origin.object.checkCollision!=='function'||origin.object.checkCollision(target.object.center,{origin:origin.object.center,type:'sight',mode:'any'}))throw Error('目标不在本次针对讯问的视线内。');
  if(recipient.hasCondition?.('deafened')||recipient.isImmuneTo?.(item))throw Error('目标无法受到本次听觉、语言或心灵效果影响。');
  const spoken=languages(actor);if(!spoken.length)throw Error('角色尚未记录可用语言。');
  const language=await asReactionGM(()=>pick(actor,user,'针对讯问：使用的语言',spoken.map(value=>({value,label:game.i18n?.localize?.(globalThis.CONFIG?.PF2E?.languages?.[value]??value)??value}))));if(language==null)return '已取消针对讯问。';
  if(!languages(recipient).includes(language))throw Error('目标不理解本次提问使用的语言。');
  const stat=actor.getStatistic?.('diplomacy'),dc=recipient.getStatistic?.('will')?.dc?.value;
  if(!stat?.check?.roll||!Number.isFinite(dc))throw Error('缺少原生交涉检定或目标意志DC。');
  await asReactionGM(()=>actor.update({[`flags.${MODULE_ID}.reactionChecks.uses`]:[...(own(actor).uses??[]).slice(-127),message.id]}));
  if(message.flags?.pf2e?.flatCheck?.result==='fail')return '原生听觉动作平检失败，针对讯问未生效。';
  if(actor.hasCondition?.('deafened')){
   const pf=message.flags?.pf2e??{},alreadyGated=game.modules?.get('patreon-v3')?.active&&Object.keys(pf).length===1&&pf.origin&&!pf.origin.sourceId&&['all','attack'].includes(game.settings?.get('patreon-v3','flatCheck'));
   if(!alreadyGated){let total;if(!game.pf2e.Check?.roll||!game.pf2e.CheckModifier)throw Error('无法进行耳聋的原生听觉动作平检。');await asReactionGM(()=>game.pf2e.Check.roll(new game.pf2e.CheckModifier('pointed-question-deafened',{modifiers:[]},[]),{actor,token:origin,type:'flat-check',dc:{value:5},domains:['flat-check'],options:new Set(['action:pointed-question']),skipDialog:true},null,async r=>{total=r.total}));if(!Number.isFinite(total))throw Error('听觉动作平检未完成，不能重试本条使用。');if(total<5)return '耳聋的DC 5听觉动作平检失败。';}
  }
  const marker=`${MODULE_ID}:pointed-question:${globalThis.foundry?.utils?.randomID?.()??globalThis.crypto.randomUUID()}`;
  let result;nativeInvocations.set(marker,{actorUuid:actor.uuid,tokenUuid:origin.uuid,targetUuid:target.uuid,user,usageId:message.id});try{await asReactionGM(()=>stat.check.roll({token:origin,target:recipient,item,action:'pointed-question',dc:{value:dc,visible:false},traits:POINTED_TRAITS,extraRollOptions:['action:pointed-question',marker,...POINTED_TRAITS.map(t=>`item:trait:${t}`)],skipDialog:true,createMessage:true,callback:async(roll,outcome,card)=>{result={roll,card,degree:roll.options?.degreeOfSuccess??OUTCOMES.indexOf(outcome)}}}))}finally{nativeInvocations.delete(marker)};
  if(!result||!Number.isInteger(result.degree)||result.degree<0||result.degree>3)throw Error('交涉检定结果无法确认；本次使用不会自动重掷。');
  const start=now(),{degree,card}=result;
  await mark(recipient,'reaction-checks:pointed-immunity',{type:'effect',name:'针对讯问：暂时免疫',img:item.img??'icons/magic/symbols/question-stone-yellow.webp',system:{slug:'pointed-question-immunity',duration:{value:1,unit:'hours',expiry:'turn-start',sustained:false},start:{value:start,initiative:null},rules:[],tokenIcon:{show:false}},flags:{[MODULE_ID]:{reactionChecks:{kind:'pointed-immunity',sourceId:REACTION_CHECK_SOURCES.pointed,usageId:message.id,checkId:card.id,expiresAt:start+3600}}}});
  if(degree>=2){
   const native=await fromUuid(REACTION_CHECK_SOURCES.pointedEffect);if(!native?.toObject)throw Error('缺少原生针对讯问效果。');const data=native.toObject();delete data._id;const end=turnEnd(actor),bonus=degree===3?4:2;
   for(const r of data.system.rules){
    if(r.key==='TokenMark'&&r.slug==='pointed-question')r.uuid=target.uuid;
    if(r.key==='ChoiceSet'&&r.flag==='pointedQuestion')r.selection=bonus;
    if(r.key==='EphemeralEffect')r.predicate=['self:mark:pointed-question','self:mark:devise-a-stratagem','devise-a-stratagem:attack',{or:['item:trait:agile','item:trait:finesse',{nor:['item:melee','item:thrown']}]},{not:'misfortune'}];
   }
   data.system.start={value:start,initiative:end?.initiative??null};data.system.duration=end?{value:end.round-game.combat.round,unit:'rounds',expiry:'turn-end',sustained:false}:{value:1,unit:'rounds',expiry:'turn-end',sustained:false};
   data.system.context={origin:{actor:actor.uuid,item:item.uuid,token:origin.uuid},target:{actor:recipient.uuid,token:target.uuid},roll:{total:result.roll.total,degreeOfSuccess:degree}};
   data.flags??={};data.flags.system={...data.flags.system,rulesSelections:{...data.flags.system?.rulesSelections,pointedQuestion:bonus}};data.flags[MODULE_ID]={...data.flags[MODULE_ID],reactionChecks:{kind:'pointed-benefit',targetUuid:target.uuid,usageId:message.id,checkId:card.id,turnEnd:end,expiresAt:end?null:start+6}};
   await mark(actor,`reaction-checks:pointed-benefit:${target.uuid}`,data);
  }
  const narrative=degree>=2?'目标必须直接回答（可以说谎）；已建立察觉DC及本回合指定策略打击效果。':degree===0?'目标可以拒绝回答，并且对提问者的态度降低一阶。':'目标可以拒绝回答。';
  await asReactionGM(()=>globalThis.ChatMessage.create({speaker:globalThis.ChatMessage.getSpeaker({actor,token:origin}),whisper:values(game.users).filter(u=>u.isGM).map(u=>u.id),content:`<p>${escape(recipient.name)}：${OUTCOMES[degree]}。${narrative} 暂时免疫针对讯问1小时。</p>`,flags:{[MODULE_ID]:{usageGenerated:true,reactionChecks:{kind:'pointed-result',usageId:message.id,checkId:card.id,degree,targetUuid:target.uuid}}}}));
  return '已完成针对讯问的交涉检定、对应效果与1小时暂时免疫。';
 }
 async function executeUsage(ctx){if(!isActiveGM(game)||!ctx.actor?.testUserPermission(ctx.user,'OWNER')||ctx.item?.actor?.uuid!==ctx.actor.uuid||resolveAction(ctx.item)!==ctx.action)throw Error('针对讯问的来源或操作权限无效。');return queue.run('reaction-checks:pointed',()=>pointed(ctx))}
 async function maintain(actor){
  if(!liveActor(actor)){forget(actor);return;}if(!isActiveGM(game))return;
  return queue.run('reaction-checks:pointed',async()=>{
   requireReactionGM();if(!liveActor(actor)){forget(actor);return;}
   const expired=values(actor.items).filter(fallbackExpired);
   if(expired.length)await asReactionGM(()=>{if(!liveActor(actor)){forget(actor);return;}const ids=expired.filter(i=>values(actor.items).includes(i)&&fallbackExpired(i)).map(i=>i.id);if(ids.length)return actor.deleteEmbeddedDocuments('Item',ids)});
   track(actor);
  });
 }
 function register({Hooks,libWrapper,socket:socketApi}={}){
  socket=socketApi;
  const unregisterEat=eatFortune.register({Hooks,socket:socketApi});
  for(const a of values(game.actors))track(a);for(const s of values(game.scenes))for(const t of values(s.tokens))if(t.actor)track(t.actor);
  const registrations=[],on=(name,fn)=>registrations.push([name,Hooks.on(name,(...args)=>Promise.resolve().then(()=>fn(...args)).catch(onError))]);
  on('createItem',i=>i.actor&&track(i.actor));on('deleteItem',i=>i.actor&&track(i.actor));
  on('updateItem',i=>i.actor&&track(i.actor));
  on('createActor',track);on('createToken',t=>t.actor&&track(t.actor));
  on('deleteActor',forget);
  on('deleteToken',token=>{for(const actor of new Set([...tracked.values(),...reactors.values()]))if(actor.isToken&&actor.token===token)forget(actor)});
  on('deleteScene',scene=>{for(const actor of new Set([...tracked.values(),...reactors.values()]))if(actor.isToken&&actor.token?.parent===scene)forget(actor)});
  on('createChatMessage',settleReaction);
  if(socket)socket.register('reaction-checks:decide',async function(payload){try{return {ok:true,value:await decideCheckReaction(payload,game.users.get(this.socketdata.userId))}}catch(e){return {ok:false,error:e.message}}});
  const checkPath='game.pf2e.Check.roll';if(libWrapper)libWrapper.register(MODULE_ID,checkPath,async function(wrapped,check,context={},event=null,callback){
   const existingEntry=(check,context={},event=null,callback)=>{
   const native=(...args)=>eatFortune.interceptCheck(wrapped,...args);
   const actor=context.actor??(context.origin?.self?context.origin?.actor:context.target?.actor);
   // PF2e can supply a contextual clone for opposed checks. Resolve only its
   // exact world actor for routing; keep the native context for full validation.
   const luckActor=game.actors?.get(actor?.id);
   if(actor&&luckActor?.uuid===actor.uuid&&!reactors.has(actor.uuid)&&!context.isReroll&&['skill-check','saving-throw'].includes(context.type)&&halflingLuck?.handlesActor(luckActor))return halflingLuck.interceptCheck(native,check,context,event,callback);
   if(!actor||!reactors.has(actor.uuid)||!['skill-check','saving-throw'].includes(context.type)||context.createMessage===false||context.isReroll)return native(check,context,event,callback);
   // Numeric DCs drop native targets. Only this invocation's random marker and
   // exact source actor/token can recover the verified GM-proxied Use target.
   const tokenUuid=context.token?.uuid??(context.origin?.self?context.origin?.token?.uuid:context.target?.token?.uuid),invocation=values(context.options).map(o=>nativeInvocations.get(o)).find(i=>i?.actorUuid===actor.uuid&&i.tokenUuid===tokenUuid);
   const selected=values(game.user?.targets),targetSnapshot=invocation?.targetUuid??(selected.length===1?(selected[0].document??selected[0]).uuid:null);
   // Preserve ordinary player rolls. A pipeline started by the primary GM must
   // stop on handoff before its next roll, publication or mechanical callback.
   const run=isActiveGM(game)?asReactionGM:operation=>operation();
   return runCheckReactionPipeline({game,check,context,event,callback:callback?(...args)=>run(()=>callback(...args)):undefined,native:(...args)=>run(()=>native(...args)),beforeReroll:state=>run(()=>eatFortune.beforeReroll(state)),decide:state=>run(()=>decide(state,targetSnapshot,invocation?.user)),publish:data=>run(()=>globalThis.ChatMessage.create(data))});
   };
   return nativeCheckMiddleware?nativeCheckMiddleware(existingEntry,check,context,event,callback):existingEntry(check,context,event,callback);
  // Eat's authenticated preparation probe returns before the native die.
  // MIXED explicitly permits that branch to omit the wrapped call.
  },'MIXED');
  for(const event of ['updateCombat','deleteCombat','updateWorldTime'])on(event,async()=>{if(isActiveGM(game))for(const a of tracked.values())await maintain(a)});
  return()=>{unregisterEat();for(const[name,id]of registrations)Hooks.off(name,id);if(libWrapper)libWrapper.unregister(MODULE_ID,checkPath)};
 }
 return {resolveAction,executeUsage,maintain,register,decideCheckReaction,settleReaction,eatFortune};
}
