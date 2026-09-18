import {MODULE_ID} from './rules.mjs';
import {SerialActions} from './runtime.mjs';
import {getSourceId,isActiveGM} from './native-context.mjs';
import {genericReactionAvailable,withReactionReservation,reactionEpoch} from './reaction-budget.mjs';
import {reactionPermitted} from './reaction-restriction.mjs';

export const EAT_FORTUNE_SOURCES=Object.freeze({eat:'Compendium.pf2e.feats-srd.Item.rFmJVDdB313EibTs',assurance:'Compendium.pf2e.feats-srd.Item.W6Gl9ePmItfDHji0',chrono:'Compendium.pf2e.feats-srd.Item.ygdbkfPPgSoWxaBa',devise:'Compendium.pf2e.feat-effects.Item.XQpTyjXFYYNexyOk',clock:'Compendium.pf2e.feats-srd.Item.3aG0gkHulBIHqqGE'});
const values=c=>Array.from(c?.values?.()??c??[]),own=d=>d?.flags?.[MODULE_ID]?.reactionChecks??{};
const actorOf=context=>context.actor??(context.origin?.self?context.origin.actor:context.target?.actor);
const tokenOf=context=>context.token??(context.origin?.self?context.origin.token:context.target?.token);
const opposite=trait=>trait==='fortune'?'misfortune':'fortune';
const asSet=value=>value instanceof Set?value:new Set(value??[]);
const selectedSubstitution=context=>(context.substitutions??[]).find(s=>s.required&&s.selected)??(context.substitutions??[]).find(s=>s.selected);
const framePrefix=`${MODULE_ID}:eat-strike:`,probePrefix=`${MODULE_ID}:eat-probe:`,strikeFrames=new Map(),probeParams=new WeakSet();
export const isEatFortuneProbe=params=>!!params&&probeParams.has(params);
/** The original prepared actor may be Spellstrike's temporary actor. Retain it,
 * and the exact native invocation, only until that invocation finishes. */
export async function withEatStrikeFrame({actor,strike,variant,params={}},native){
 if(isEatFortuneProbe(params)||!values(actor?.items).some(i=>getSourceId(i)===EAT_FORTUNE_SOURCES.devise))return native(params);
 const id=globalThis.foundry?.utils?.randomID?.()??globalThis.crypto.randomUUID(),marker=framePrefix+id,frame={actor,strike,variantIndex:strike.variants.indexOf(variant),params};strikeFrames.set(marker,frame);
 const framed={...params,options:new Set([...params.options??[],marker]),extraRollOptions:[...params.extraRollOptions??[],marker]};
 try{return await native(framed)}finally{strikeFrames.delete(marker)}
}
export function occupiedTraits(context){
 const options=asSet(context.options),actor=actorOf(context),occupied=new Set(['fortune','misfortune'].filter(t=>options.has(t))),selected=selectedSubstitution(context);
 if(selected?.effectType)occupied.add(selected.effectType);
 if(context.rollTwice)occupied.add(context.rollTwice==='keep-higher'?'fortune':'misfortune');
 // extractRollTwice folds two opposing sources into false before Check.roll.
 const synthetics=(context.domains??[]).flatMap(domain=>actor?.synthetics?.rollTwice?.[domain]??[]).filter(s=>s.predicate?.test(options)??true);
 if(synthetics.some(s=>s.keep==='higher')&&synthetics.some(s=>s.keep==='lower')){occupied.add('fortune');occupied.add('misfortune')}
 return occupied;
}

/** Recover the original live rule, not a label or an anonymous rollTwice option. */
export function selectedFortuneSources(context){
 const actor=actorOf(context),selected=selectedSubstitution(context),options=asSet(context.options);
 if(!actor||!selected||context.isReroll||occupiedTraits(context).size>1)return [];
 const token=tokenOf(context),result=[];
 for(const rule of values(actor.rules)){
  const item=rule.item,kind=['assurance','chrono','devise'].find(key=>getSourceId(item)===EAT_FORTUNE_SOURCES[key]);
  if(!kind||rule.key!=='SubstituteRoll'||rule.ignored||item.isExpired||!(rule.test?.(options)??rule.predicate?.test(options)??true))continue;
  const selector=rule.resolveInjectedProperties(rule.selector);
  if(!(context.domains??[]).includes(selector))continue;
  if(kind==='devise'&&(context.type!=='attack-roll'||!options.has('devise-a-stratagem:attack')||options.has('devise-a-stratagem:skill')||!options.has('target:mark:devise-a-stratagem')||!(context.action==='strike'||selector==='strike-attack-roll')))continue;
  const slug=rule.slug??globalThis.game?.pf2e?.system?.sluggify?.(item.name),value=Math.min(20,Math.max(1,Math.trunc(Number(rule.resolveValue(rule.value))))),effectType=rule.effectType??'fortune';
  if(slug!==selected.slug||value!==selected.value||effectType!==selected.effectType||!!rule.required!==!!selected.required||rule.label!=null&&selected.label!==rule.label)continue;
  const origin=item.type==='effect'?item.system.context?.origin:null;
  result.push({kind,sourceItemUuid:item.uuid,sourceActorUuid:origin?.actor??actor.uuid,sourceTokenUuid:origin?.token??(origin?.actor&&origin.actor!==actor.uuid?null:token?.uuid)??null,rollerActorUuid:actor.uuid,rollerTokenUuid:token?.uuid??null,selector,slug,value,effectType,required:!!rule.required,label:rule.label??null});
 }
 return result;
}

export function createEatFortune({game,reactionRestriction,fromUuid=globalThis.fromUuid,choose,onError=()=>{}}={}){
 const reactors=new Map(),queue=new SerialActions(),dialogs=new WeakMap(),wrappedDialogs=new WeakSet(),modifierInputs=new WeakSet(),probes=new Map();let socket;
 const requireGM=()=>{if(!isActiveGM(game))throw Error('主GM已交接，吞噬福祸停止；已认领或已付的资源不会自动回滚或重试。')};
 const guarded=async operation=>{requireGM();const result=await operation();requireGM();return result};
 const feature=actor=>values(actor?.items).find(i=>i.type==='feat'&&getSourceId(i)===EAT_FORTUNE_SOURCES.eat);
 const track=actor=>{if(feature(actor))reactors.set(actor.uuid,actor);else reactors.delete(actor?.uuid)};
 const canUse=actor=>!actor.isDead&&actor.canAct!==false&&!actor.hasCondition?.('unconscious')&&!actor.hasCondition?.('stunned');
 const available=actor=>reactionPermitted(actor,reactionRestriction)&&(!reactionEpoch(actor,game)||genericReactionAvailable(actor,game));
 const uses=item=>item.system.frequency?.value??item.system.frequency?.max??0;
 const range=(source,target)=>{const distance=source?.object&&target?.object&&source.parent?.id===target.parent?.id?target.object.distanceTo?.(source.object):null;return Number.isFinite(distance)&&distance>=0&&distance<=60};
 const chooser=actor=>values(game.users).find(u=>u.active&&!u.isGM&&u.character?.uuid===actor.uuid&&actor.testUserPermission(u,'OWNER'))??values(game.users).find(u=>u.active&&!u.isGM&&actor.testUserPermission(u,'OWNER'))??game.user;
 const writeRecord=(actor,nonce,change)=>guarded(()=>actor.update({[`flags.${MODULE_ID}.reactionChecks.reactions`]:(own(actor).reactions??[]).map(r=>r.nonce===nonce?{...r,...change}:r)}));
 async function resolveTrigger(payload,user){
  requireGM();
  if(!user||typeof payload?.nonce!=='string'||!/^[A-Za-z0-9-]{8,80}$/.test(payload.nonce))throw Error('吞噬福祸检定凭据无效。');
  const actor=await fromUuid(payload.rollerActorUuid),token=await fromUuid(payload.rollerTokenUuid),item=await fromUuid(payload.sourceItemUuid),source=await fromUuid(payload.sourceTokenUuid);
  requireGM();if(!actor?.testUserPermission?.(user,'OWNER')||token?.documentName!=='Token'||token.actor?.uuid!==actor.uuid||source?.documentName!=='Token'||source.actor?.uuid!==payload.sourceActorUuid||token.parent?.id!==source.parent?.id||item?.actor?.uuid!==actor.uuid)throw Error('吞噬福祸的检定使用者、来源或Token不匹配。');
  if(payload.kind==='clock'){
   const claim=own(actor).reactions?.find(r=>r.kind==='clock'&&r.nonce===payload.clockNonce&&r.state==='claimed'),card=claim&&game.messages.get(claim.checkId);
   if(getSourceId(item)!==EAT_FORTUNE_SOURCES.clock||payload.effectType!=='fortune'||!claim||card?.item?.uuid!==item.uuid||own(card).kind!=='reaction-use'||own(card).nonce!==payload.clockNonce||!['skill-check','saving-throw'].includes(payload.type))throw Error('Clock 已付反应来源凭据无法确认。');
   if(asSet(payload.options).has('misfortune'))return null;return {actor,token,item,source,matched:payload};
  }
  const context={actor,token,type:payload.type,action:payload.action,domains:payload.domains,options:new Set(payload.options??[]),rollTwice:payload.rollTwice,substitutions:payload.substitutions};
  const matched=selectedFortuneSources(context).find(s=>s.kind===payload.kind&&s.sourceItemUuid===payload.sourceItemUuid&&s.sourceActorUuid===payload.sourceActorUuid&&s.sourceTokenUuid===payload.sourceTokenUuid&&s.slug===payload.slug&&s.value===payload.value&&s.effectType===payload.effectType);
  if(!matched)return null;return {actor,token,item,source,matched};
 }
 async function decide(payload,user){
  const initial=await resolveTrigger(payload,user);if(!initial)return null;
  return queue.run(`source:${payload.sourceItemUuid}`,async()=>{
   requireGM();
   const existing=values(reactors.values()).flatMap(a=>(own(a).reactions??[]).filter(r=>r.kind==='eat'&&(r.nonce===payload.nonce||payload.clockNonce&&r.clockNonce===payload.clockNonce&&r.sourceActorUuid===payload.sourceActorUuid)));
   if(existing.length)throw Error('吞噬福祸已认领本次检定，不能重放或再次收费。');
   const candidates=values(game.scenes.get(initial.source.parent.id)?.tokens).filter(t=>reactors.has(t.actor?.uuid)&&range(initial.source,t));
   const seen=new Set();
   for(const reactorToken of candidates){
    const actor=reactorToken.actor,item=feature(actor);if(seen.has(actor.uuid))continue;seen.add(actor.uuid);
    if(!item||uses(item)<1||!canUse(actor)||!available(actor))continue;
    const triggerEpoch=reactionEpoch(actor,game),owner=chooser(actor),answer=await guarded(()=>choose({actor,user:owner,title:`吞噬福祸：${initial.source.actor.name??'生物'}使用${payload.effectType==='fortune'?'幸运':'厄运'}效果`,choices:[{value:'eat',label:'使用吞噬福祸（反应；每日一次）'},{value:'decline',label:'不使用'}]}));
    if(answer==null||answer==='decline')continue;if(answer!=='eat')throw Error('无效的吞噬福祸选择。');
    const proof=await withReactionReservation(actor,game,async()=>{
     requireGM();const current=await resolveTrigger(payload,user);requireGM();
     if(!current||reactionEpoch(actor,game)!==triggerEpoch||!range(current.source,reactorToken)||!canUse(actor)||feature(actor)?.id!==item.id||uses(item)<1||!available(actor))return null;
     const proof={nonce:payload.nonce,kind:'eat',state:'claimed',epoch:triggerEpoch,time:game.time.worldTime??0,userId:user.id,reactorUserId:owner.id,reactorActorUuid:actor.uuid,reactorTokenUuid:reactorToken.uuid,sourceItemUuid:payload.sourceItemUuid,sourceActorUuid:payload.sourceActorUuid,sourceTokenUuid:payload.sourceTokenUuid,rollerActorUuid:payload.rollerActorUuid,rollerTokenUuid:payload.rollerTokenUuid,sourceKind:payload.kind,effectType:payload.effectType,oppositeTrait:opposite(payload.effectType),disrupted:true,...(payload.clockNonce?{clockNonce:payload.clockNonce}:{})};
     await guarded(()=>actor.update({[`flags.${MODULE_ID}.reactionChecks.reactions`]:[...(own(actor).reactions??[]),proof]}));
     await guarded(()=>item.update({'system.frequency.value':uses(item)-1},{[MODULE_ID]:{usageInternal:true}}));
     return proof;
    });if(!proof)continue;
    if(proof.sourceKind==='devise'){
     // The native if-enabled afterRoll deletes this whole attack-stratagem
     // effect. Cancellation must consume it as well, even though a d20 is rolled.
     const sourceItem=await fromUuid(proof.sourceItemUuid);requireGM();if(!sourceItem||getSourceId(sourceItem)!==EAT_FORTUNE_SOURCES.devise)throw Error('已认领的攻击策略来源消失；不会重试。');
     await guarded(()=>sourceItem.actor.deleteEmbeddedDocuments('Item',[sourceItem.id]));await writeRecord(actor,proof.nonce,{sourceConsumed:true});
    }
    const card=await guarded(()=>item.toMessage());if(!card?.id)throw Error('吞噬福祸已认领，但原生反应卡未确认；不会重试。');
    await guarded(()=>card.update({[`flags.${MODULE_ID}.usageGenerated`]:true,[`flags.${MODULE_ID}.reactionChecks`]:{kind:'reaction-use',reaction:'eat',nonce:proof.nonce}}));
    await writeRecord(actor,proof.nonce,{checkId:card.id});return {...proof,checkId:card.id};
   }
   return null;
  });
 }
 async function request(context,user=game.user){
  const sources=[...new Map(selectedFortuneSources(context).map(source=>[source.sourceItemUuid,source])).values()];if(!sources.length)return null;
  // Native selected substitution is unique; identical rules require a real source choice.
  let selected=sources[0];if(sources.length>1){
   const actor=actorOf(context),startedAsGM=isActiveGM(game),select=()=>choose({actor,user:chooser(actor),title:'选择本次使用的幸运／厄运来源',choices:sources.map(source=>({value:source.sourceItemUuid,label:values(actor.items).find(item=>item.uuid===source.sourceItemUuid)?.name??source.sourceItemUuid}))});
   const answer=startedAsGM?await guarded(select):await select();if(answer==null)return null;selected=sources.find(source=>source.sourceItemUuid===answer);if(!selected)throw Error('选择的幸运／厄运来源不属于本次检定。');
  }
  if(selected.kind==='devise'&&!values(context.options).some(option=>strikeFrames.has(option)))throw Error('攻击策略缺少本次原生Strike来源，无法安全重建；吞噬福祸尚未消费。');
  const payload={...selected,nonce:globalThis.foundry?.utils?.randomID?.()??globalThis.crypto.randomUUID(),type:context.type,action:context.action??null,domains:context.domains??[],options:[...asSet(context.options)],rollTwice:context.rollTwice??false,substitutions:(context.substitutions??[]).map(s=>({slug:s.slug,label:s.label,value:s.value,effectType:s.effectType,selected:!!s.selected,required:!!s.required}))};
  return arbitrate(payload,user);
 }
 async function arbitrate(payload,user=game.user){
  if(isActiveGM(game))return decide(payload,user);
  if(!socket||!game.users.activeGM)return null;const response=await socket.executeAsUser('eat-fortune:decide',game.users.activeGM.id,payload);if(!response.ok)throw Error(response.error);return response.value;
 }
 async function beforeReroll({context,decision}){
  if(!reactors.size||asSet(context.options).has('misfortune'))return null;
  const actor=actorOf(context),token=tokenOf(context),item=values(actor?.items).find(i=>getSourceId(i)===EAT_FORTUNE_SOURCES.clock);if(!item||!token?.uuid)return null;
  return arbitrate({kind:'clock',nonce:globalThis.foundry?.utils?.randomID?.()??globalThis.crypto.randomUUID(),clockNonce:decision?.nonce,sourceItemUuid:item.uuid,sourceActorUuid:actor.uuid,sourceTokenUuid:token.uuid,rollerActorUuid:actor.uuid,rollerTokenUuid:token.uuid,type:context.type,options:[...asSet(context.options)],effectType:'fortune'});
 }
 async function rebuildDevise(check,context,proof,baseline){
  const frame=values(context.options).map(option=>strikeFrames.get(option)).find(Boolean),sourceToken=tokenOf(context),target=context.target?.token;
  if(!frame||frame.actor.uuid!==proof.rollerActorUuid||frame.strike.item.id!==context.item?.id||frame.variantIndex<0||!target?.object||!sourceToken?.uuid)throw Error('攻击策略已被打断，但原生攻击重建来源不完整；不会补掷或重试。');
  const sourceId=proof.sourceItemUuid.split('.').at(-1),items=structuredClone(frame.actor._source.items).filter(i=>i._id!==sourceId);
  for(const item of items)if(getSourceId(item)==='Compendium.pf2e.actionspf2e.Item.m0f2B7G9eaaTmhFL')item.system.rules=item.system.rules.filter(r=>!(r.key==='FlatModifier'&&r.ability==='int'&&r.selector==='strike-attack-roll'));
  const clone=frame.actor.clone({items},{keepId:true}),strikes=values(clone.system.actions).flatMap(s=>[s,...s.altUsages??[]]),weapon=frame.strike.item,strike=strikes.find(s=>s.item?.id===weapon.id&&s.item?.isMelee===weapon.isMelee&&s.item?.isThrown===weapon.isThrown),variant=strike?.variants?.[frame.variantIndex];
  if(!variant?.roll)throw Error('攻击策略已被打断，但无法重建同一武器和MAP；不会补掷。');
  const id=probePrefix+(globalThis.foundry?.utils?.randomID?.()??globalThis.crypto.randomUUID()),capture={actorUuid:clone.uuid,itemId:weapon.id,sourceTokenUuid:sourceToken.uuid,targetTokenUuid:target.uuid,value:null};probes.set(id,capture);
  const options=new Set([...frame.params.options??[],...frame.params.extraRollOptions??[]].filter(o=>!o.startsWith('devise-a-stratagem')&&!o.startsWith('target:mark:devise-a-stratagem')));for(const option of ['fortune','misfortune',id])options.add(option);
  const params={...frame.params,target:target.object,options,extraRollOptions:[...options],consumeAmmo:false,createMessage:false,skipDialog:true,event:null,callback:undefined,dc:undefined};probeParams.add(params);
  try{await variant.roll(params)}finally{probeParams.delete(params);probes.delete(id)}
  if(!capture.value)throw Error('攻击策略已打断，原生上下文探针未完成；不会继续投骰。');
  const rebuilt=capture.value,identity=m=>`${m.slug}|${m.type}|${m.rule?.item?.uuid??''}|${m.label??''}`,overrides=new Map(),added=[];
  for(const modifier of check.modifiers){const initial=baseline?.get(modifier);if(initial&&initial.ignored!==modifier.ignored)overrides.set(identity(modifier),modifier.ignored);else if(baseline&&!initial)added.push(modifier)}
  for(const modifier of check.modifiers)check.delete(modifier.slug);for(const modifier of rebuilt.check.modifiers)check.push(modifier.clone?.()??modifier);for(const modifier of added)if(!check.modifiers.some(m=>identity(m)===identity(modifier)))check.push(modifier);
  const optionsRef=asSet(context.options);optionsRef.clear();for(const option of rebuilt.context.options)if(option!==id)optionsRef.add(option);for(const option of ['fortune','misfortune',`${MODULE_ID}:eat:${proof.nonce}`])optionsRef.add(option);context.options=optionsRef;
  for(const key of ['origin','target','dc']){if(context[key]&&rebuilt.context[key])Object.assign(context[key],rebuilt.context[key]);else context[key]=rebuilt.context[key]}
  if(Array.isArray(context.domains))context.domains.splice(0,context.domains.length,...rebuilt.context.domains);else context.domains=rebuilt.context.domains;
  for(const key of ['actor','token','item','traits','notes','dosAdjustments','damaging'])context[key]=rebuilt.context[key];
  const calculate=check.calculateTotal;check.calculateTotal=function(options){calculate.call(this,options);if(overrides.size){for(const modifier of this.modifiers)if(overrides.has(identity(modifier)))modifier.ignored=overrides.get(identity(modifier));calculate.call(this)}return this.totalModifier};check.calculateTotal(optionsRef);
  return()=>{check.calculateTotal=calculate};
 }
 function recoverAssurance(check,context,slot){
  const calculate=check.calculateTotal,slug=selectedSubstitution(context)?.slug,marker=`substitute:${slug}`;
  // PF2e's suppress adjustment sets ignored=true, but an empty-predicate
  // Modifier.test never clears it when fortune/misfortune removes Assurance.
  check.calculateTotal=function(options){
   const cancelled=options?.has('fortune')&&options.has('misfortune')&&!options.has(marker);
   if(cancelled)for(const modifier of this.modifiers){
    const after=[...options,...modifier.getRollOptions?.()??[]],before=[...after,marker];
    if(modifier.adjustments?.some(adjustment=>adjustment.suppress&&adjustment.test(before)&&!adjustment.test(after))){
     const initial=slot.initial.get(modifier);modifier.ignored=slot.manualIgnored.get(modifier)??(initial?.ignored&&!slot.initialOptions.has(marker));
    }
   }
   calculate.call(this,options);
   if(cancelled&&slot.manualIgnored.size){for(const modifier of this.modifiers)if(slot.manualIgnored.has(modifier))modifier.ignored=slot.manualIgnored.get(modifier);calculate.call(this)}
   return this.totalModifier;
  };
  return()=>{check.calculateTotal=calculate};
 }
 async function interceptCheck(native,check,context={},event=null,callback){
  const probe=values(context.options).map(option=>probes.get(option)).find(Boolean);
  if(probe){if(probe.value||actorOf(context)?.uuid!==probe.actorUuid||context.item?.id!==probe.itemId||tokenOf(context)?.uuid!==probe.sourceTokenUuid||context.target?.token?.uuid!==probe.targetTokenUuid)throw Error('原生攻击重建探针来源不匹配。');probe.value={check,context};return null}
  if(!reactors.size||context.isReroll||!(context.substitutions?.length)||!values(actorOf(context)?.rules).some(r=>r.key==='SubstituteRoll'&&['assurance','chrono','devise'].some(kind=>getSourceId(r.item)===EAT_FORTUNE_SOURCES[kind])))return native(check,context,event,callback);
  let restore;const initial=new Map(check.modifiers.map(m=>[m,{ignored:m.ignored}])),slot={runPreRoll:null,error:null,baseline:initial,initial,initialOptions:new Set(context.options),manualIgnored:new Map()};
  const startedAsGM=isActiveGM(game),nativeCallback=callback?((...args)=>{if(startedAsGM)requireGM();return callback(...args)}):undefined,runNative=async()=>{const result=await native(check,context,event,nativeCallback);if(startedAsGM)requireGM();return result},runPreRoll=async()=>{
   if(startedAsGM)requireGM();const proof=await request(context);if(startedAsGM)requireGM();
   if(proof){const options=asSet(context.options);options.add(proof.effectType);options.add(proof.oppositeTrait);options.add(`${MODULE_ID}:eat:${proof.nonce}`);context.options=options;context.eatFortune=proof;if(proof.sourceKind==='devise')restore=await rebuildDevise(check,context,proof,slot.baseline);else if(proof.sourceKind==='assurance')restore=recoverAssurance(check,context,slot);if(startedAsGM)requireGM()}
  };
  const defaultSkip=!game.user.settings?.showCheckDialogs,relevant=event&&typeof event==='object'&&['ctrlKey','metaKey','shiftKey'].every(key=>key in event),skip=event?(relevant&&event.shiftKey?!defaultSkip:defaultSkip):context.skipDialog??defaultSkip;
  if(skip||context.type==='flat-check'){try{await runPreRoll();return await runNative()}finally{restore?.()}}
  slot.runPreRoll=runPreRoll;dialogs.set(context,slot);
  try{const result=await runNative();if(slot.error)throw slot.error;return result}finally{dialogs.delete(context);restore?.()}
 }
 async function settle(message){
  const context=message.flags?.pf2e?.context,proof=context?.eatFortune;if(!isActiveGM(game)||!message.id||game.messages.get(message.id)!==message||!message.rolls?.length||!proof?.nonce||!asSet(context.options).has(`${MODULE_ID}:eat:${proof.nonce}`))return;
  const actor=await fromUuid(proof.reactorActorUuid),roller=await fromUuid(proof.rollerActorUuid),author=message.author??game.users.get(message.user?.id??message.user);requireGM();
  if(!roller?.testUserPermission?.(author,'OWNER')||!['skill-check','saving-throw','attack-roll','initiative'].includes(context.type)||!asSet(context.options).has('fortune')||!asSet(context.options).has('misfortune'))return;
  const nativeSource=[context.origin,context.target].some(endpoint=>endpoint?.actor===proof.rollerActorUuid&&endpoint?.token===proof.rollerTokenUuid),speaker=message.speaker,directSource=context.actor===roller.id&&context.token===proof.rollerTokenUuid.split('.').at(-1)&&speaker?.actor===roller.id&&`Scene.${speaker.scene}.Token.${speaker.token}`===proof.rollerTokenUuid;if(!nativeSource&&!directSource)return;
  return withReactionReservation(actor,game,async()=>{
   requireGM();const record=own(actor).reactions?.find(r=>r.kind==='eat'&&r.nonce===proof.nonce);if(!record||record.state==='used'||record.userId!==author.id&&!author.isGM||['sourceItemUuid','sourceActorUuid','sourceTokenUuid','rollerActorUuid','rollerTokenUuid','reactorActorUuid','reactorTokenUuid','sourceKind','effectType','oppositeTrait','clockNonce','disrupted','checkId'].some(key=>record[key]!==proof[key]))return;
   const next=(own(actor).reactions??[]).map(r=>r===record?{...r,state:'used',resultMessageId:message.id}:r),settled=next.filter(r=>r.state==='used').slice(-128),pending=next.filter(r=>r.state!=='used');
   await guarded(()=>actor.update({[`flags.${MODULE_ID}.reactionChecks.reactions`]:[...pending,...settled]}));
  });
 }
 function register({Hooks,socket:socketApi}={}){
  socket=socketApi;for(const actor of values(game.actors))track(actor);for(const scene of values(game.scenes))for(const token of values(scene.tokens))if(token.actor)track(token.actor);
  const registrations=[],on=(name,fn)=>registrations.push([name,Hooks.on(name,fn)]);
  for(const event of ['createItem','updateItem','deleteItem'])on(event,item=>item.actor&&track(item.actor));on('createActor',track);on('createToken',token=>token.actor&&track(token.actor));on('deleteActor',actor=>reactors.delete(actor.uuid));
  // The hook itself is synchronous: suspend the Promise Check.roll awaits by
  // wrapping this exact application's resolver, after its final UI choices.
  on('renderCheckModifiersDialog',app=>{
   const slot=dialogs.get(app.context);if(!slot)return;
   const root=app.element?.[0]??app.element;for(const input of root?.querySelectorAll?.('.modifier-container input[type=checkbox]')??[])if(!modifierInputs.has(input)){modifierInputs.add(input);input.addEventListener('click',()=>{const modifier=app.check.modifiers[Number(input.dataset.modifierIndex)];if(modifier)slot.manualIgnored.set(modifier,!input.checked)})}
   if(wrappedDialogs.has(app))return;wrappedDialogs.add(app);slot.baseline=new Map(app.check.modifiers.map(m=>[m,{ignored:m.ignored}]));const resolve=app.resolve;let submitted=false;
   app.resolve=accepted=>{if(submitted)return;submitted=true;if(!accepted)return resolve(false);return slot.runPreRoll().then(()=>resolve(true),error=>{slot.error=error;return resolve(false)})};
  });
  on('createChatMessage',message=>settle(message).catch(onError));
  if(socket)socket.register('eat-fortune:decide',async function(payload){try{return {ok:true,value:await decide(payload,game.users.get(this.socketdata.userId))}}catch(error){return {ok:false,error:error.message}}});
  return()=>{for(const[name,id]of registrations)Hooks.off(name,id)};
 }
 return {interceptCheck,beforeReroll,register,decide,settle};
}
