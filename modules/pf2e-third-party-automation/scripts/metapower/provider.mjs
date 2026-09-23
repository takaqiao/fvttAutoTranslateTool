import {METAPOWER_SOURCES,metapowerKind,powerProfile,sourceUuid,buildChannelSnapshot,siphonMultiplier} from './rules.mjs';
import {MODULE_ID,createMetapowerLedger,ledgerState,chargedEffect} from './lifecycle.mjs';
import {createMetapowerObserver} from './observer.mjs';
import {renderMetapowerCard} from './card.mjs';
import {installActionEntrances,wrapSheetHandlers,createToolbeltEntrance,patchHudController,installLegacyActionBoundary,ensureNativeUseControls} from './entrances.mjs';
import {convertSiphonRoll,applyNativeOutcomeInPlace} from './damage.mjs';
import {showNativeChoice} from '../native-context.mjs';
import {createActorStateIndex} from '../actor-state-index.mjs';
import {ELEMENTAL_POWERS_SOURCE} from '../eldamon-voltage.mjs';
const values=c=>Array.from(c?.values?.()??c??[]);
const prefix=`${MODULE_ID}:metapower:`;
/** Ordinary native actions need no transaction while the metapower window is idle.
 * Read live flags: cloning the growing receipt ledger is itself unnecessary here. */
export function needsMetapowerObservation({actor,item},supportsOriginalUse=()=>false){
 const state=actor?.flags?.[MODULE_ID]?.metapower;
 return !!(metapowerKind(item)||powerProfile(item)||sourceUuid(item)===ELEMENTAL_POWERS_SOURCE||item&&supportsOriginalUse(item)||state?.armed||state?.pending||
  Object.values(state?.receipts??{}).some(receipt=>receipt.delivery&&receipt.delivery.status!=='done'));
}
export function adjustMetapowerCheckContext(snapshot,context){
 if(context.type!=='saving-throw'||snapshot.saveDowngrade!==1)return context;
 return {...context,dosAdjustments:[...(context.dosAdjustments??[]),{adjustments:{all:{label:'Retributive Shock · Discharge',amount:-1}}}]};
}
export function preserveMetapowerOnAlter(original,result){
 const proof=original?.options?.[MODULE_ID]?.metapowerDamage;
 if(proof&&result?.options)result.options[MODULE_ID]={...result.options[MODULE_ID],metapowerDamage:structuredClone(proof)};
 const failure=original?.options?.[MODULE_ID]?.metapowerShotFailure;
 if(failure&&result?.options)result.options[MODULE_ID]={...result.options[MODULE_ID],metapowerShotFailure:structuredClone(failure)};
 return result;
}

export function createMetapowerProvider({game,fromUuid,onError=console.error,selectChoice=showNativeChoice,onCommittedChannel=async()=>{},beforeChannel,validateSelection,supportsOriginalUse=()=>false,interceptDamageMessage=(_roll,data,options,native)=>native(data,options)}){
 const ledger=createMetapowerLedger({game,fromUuid,validateSelection}),deliveries=new Map();let socket;
 const activeActors=createActorStateIndex({game,matches:actor=>{const state=actor.flags?.[MODULE_ID]?.metapower;return !!(state?.armed||state?.pending||Object.values(state?.receipts??{}).some(r=>r.delivery&&r.delivery.status!=='done'))}});
 const eligible=actor=>actor?.type==='character'&&values(actor.items).some(item=>metapowerKind(item));
 const request=async(method,payload)=>{
  if(!socket||!game.users.activeGM)throw Error('Metapower automation requires an active GM and socketlib.');
  const response=await socket.executeAsUser(`metapower:${method}`,game.users.activeGM.id,payload);
  if(!response?.ok)throw Error(response?.error??'Metapower coordinator did not respond.');
  return response.value;
 };
 async function deliverCommitted({actorUuid,nonce}){
  if(game.user?.id!==game.users.activeGM?.id)throw Error('Committed follow-up delivery requires the active GM.');
  const key=`${actorUuid}:${nonce}`;if(deliveries.has(key))return deliveries.get(key);
  const task=(async()=>{
   const actor=await fromUuid(actorUuid),initial=ledgerState(actor).receipts[nonce];
   if(initial?.status!=='committed'||!initial.delivery||initial.delivery.status==='done')return initial;
   try{
    const receipt=await ledger.delivery({actorUuid,nonce,status:'started'},game.user),message=await fromUuid(receipt.messageUuid),user=game.users.get(receipt.userId);
    if(!user||!message||message.flags?.[MODULE_ID]?.metapowerUse?.nonce!==nonce||message.speaker?.actor!==actor.id)throw Error('Committed original card or author is unavailable; GM reconciliation is required.');
    await message.update({[`flags.${MODULE_ID}.metapowerUse.status`]:'committed'});
    await onCommittedChannel({receipt,message,user});
    const result=await ledger.delivery({actorUuid,nonce,status:'done'},game.user);
    await message.update({[`flags.${MODULE_ID}.metapowerUse.deliveryStatus`]:'done'});return result;
   }catch(error){
    const result=await ledger.delivery({actorUuid,nonce,status:'pending',error:error.message},game.user).catch(()=>ledgerState(actor).receipts[nonce]);onError(error);return result;
   }
  })().finally(()=>deliveries.delete(key));deliveries.set(key,task);return task;
 }
 async function select(item,input={}){
  const profile=powerProfile(item),armed=ledgerState(item.actor).armed;
  if(!profile)return {};
  let selection={discharge:false};
  if(profile.id!=='high-voltage'){
   const choices=[];
   for(const discharge of [false,true]){
    if(discharge&&!chargedEffect(item.actor))continue;
    if(discharge&&armed?.kind==='siphoning'&&profile.id==='reactive-chain')continue;
    const distances=profile.areaType?Array.from({length:25},(_,i)=>5+i*5):[null];
    for(const baseDistance of distances){
     const candidate={discharge,...(baseDistance?{baseDistance}:{})};let snapshot;
     try{snapshot=buildChannelSnapshot({kind:armed?.kind??'normal',item,selection:candidate,policy:{dischargeNonDamage:'remove',dischargeArea:'retain',dischargeRange:'retain',dischargeSaveDowngrade:'retain',highVoltage:'convert'}})}catch{continue}
     choices.push({value:JSON.stringify(candidate),label:`${discharge?'放电：Charged −1':'普通分支'}${snapshot.area?` · ${snapshot.area.distance} 尺` : ''}${armed?.kind==='siphoning'?' · 虹吸':''}`});
    }
   }
   const result=await selectChoice({title:item.name,choices});if(!result)return null;selection=JSON.parse(result);
  }
  if(beforeChannel){selection=await beforeChannel({item,selection:{...selection,...input},kind:armed?.kind??'normal'});if(!selection||profile.id==='reactive-chain')return selection;}
  if(profile.reaction){
   const chain=profile.id==='reactive-chain';
   const result=await globalThis.foundry.applications.api.DialogV2.wait({window:{title:item.name},content:`<p>${chain?'确认：30尺内生物实际受到电击伤害；所选目标在该生物30尺内，未受同一效果电击伤害，且已Shocked。虹吸不允许用放电放宽此资格。':'确认本次真实反应触发符合原威能条件。'}</p>${chain?'<label>触发生物实际受到的电击伤害 <input name="triggerDamage" type="number" min="1" step="1" required></label>':''}`,buttons:[{action:'confirm',label:'确认实际触发',callback:(_event,button)=>({triggerConfirmed:true,eligibleTargetConfirmed:chain,triggerDamage:chain?Number(new FormData(button.form).get('triggerDamage')):null})},{action:'cancel',label:'取消',callback:()=>null}],rejectClose:false});
   if(!result)return null;Object.assign(selection,result);
  }
  return selection;
 }
 const observer=createMetapowerObserver({request,select,captureInput:()=>({targetUuids:values(game.user.targets).map(t=>t.document?.uuid??t.uuid).filter(Boolean)}),onError});
 const observe=(context,native)=>eligible(context.actor)&&needsMetapowerObservation(context,supportsOriginalUse)?observer.observe(context,native):native();
 async function validateDamageProof(proof){
  const card=game.messages.get(proof?.cardId),actor=await fromUuid(proof?.actorUuid),receipt=actor?.flags?.[MODULE_ID]?.metapower?.receipts?.[proof?.nonce];
  if(!card||!receipt||receipt.status!=='committed'||receipt.messageUuid!==card.uuid||card.flags?.[MODULE_ID]?.metapowerUse?.nonce!==proof.nonce||receipt.snapshot?.itemUuid!==card.flags?.pf2e?.origin?.uuid)throw Error('Metapower damage source/card binding is invalid.');
  return receipt.snapshot;
 }
 async function beforeDamage(actor,params){
  const failure=params.damage?.options?.[MODULE_ID]?.metapowerShotFailure;
  if(failure){const snapshot=await validateDamageProof(failure),source=await fromUuid(failure.actorUuid),targets=ledgerState(source).receipts[failure.nonce]?.selection?.targetUuids,token=await fromUuid(failure.targetTokenUuid);if(snapshot.powerId!=='electric-shot'||failure.targetActorUuid!==actor.uuid||targets?.length!==1||targets[0]!==failure.targetTokenUuid||token?.actor?.uuid!==actor.uuid||(params.token?.document??params.token)?.uuid!==failure.targetTokenUuid)throw Error('Electric Shot Shocked failure damage is bound to its original recipient.');}
  const proof=params.damage?.options?.[MODULE_ID]?.metapowerDamage;if(!proof)return null;
  const snapshot=await validateDamageProof(proof),multiplier=siphonMultiplier(snapshot,actor.traits??actor.system?.traits?.value??[]);
  return {params:multiplier===1?params:{...params,damage:params.damage.alter(multiplier,0)}};
 }
 async function maintain(actor){
  if(game.user?.id!==game.users.activeGM?.id||!actor?.testUserPermission(game.user,'OWNER'))return;
  for(const item of values(actor.items))if(sourceUuid(item)===METAPOWER_SOURCES.widen&&(item.system.actionType?.value!=='action'||item.system.actions?.value!==1))await item.update({'system.actionType.value':'action','system.actions.value':1});
 }
 async function interceptCheck(wrapped,check,context={},...args){
  const option=values(context.options).find(o=>o.startsWith(prefix));if(!option)return wrapped(check,context,...args);
  const [cardId,nonce]=option.slice(prefix.length).split(':'),card=game.messages.get(cardId),proof={cardId,nonce,actorUuid:card?.flags?.[MODULE_ID]?.metapowerUse?.actorUuid};
  const snapshot=await validateDamageProof(proof);
  if(context.item?.uuid!==snapshot.itemUuid&&context.origin?.item?.uuid!==snapshot.itemUuid)throw Error('Native check origin differs from the bound power.');
  return wrapped(check,adjustMetapowerCheckContext(snapshot,context),...args);
 }
 function register({Hooks,libWrapper,socket:api}){
  socket=api;activeActors.register(Hooks);
  for(const method of ['begin','start','finish','clear','expire','reconcile'])socket.register(`metapower:${method}`,async function(payload){try{const result=await ledger[method](payload,game.users.get(this.socketdata.userId));return {ok:true,value:method==='finish'&&result?.delivery?await deliverCommitted({actorUuid:payload.actorUuid,nonce:result.nonce}):result}}catch(error){return {ok:false,error:error.message}}});
  socket.register('metapower:deliver',async function(payload){try{const actor=await fromUuid(payload.actorUuid),user=game.users.get(this.socketdata.userId);if(!actor?.testUserPermission(user,'OWNER'))throw Error('Actor owner permission is required.');return {ok:true,value:await deliverCommitted(payload)}}catch(error){return {ok:false,error:error.message}}});
  socket.register('metapower:ack-delivery',async function(payload){try{const user=game.users.get(this.socketdata.userId);if(user!==game.users.activeGM||payload.confirmation!=='gm-manual-effects-settled'||deliveries.has(`${payload.actorUuid}:${payload.nonce}`))throw Error('Only the active GM may acknowledge manually settled follow-up after automatic delivery stops.');return {ok:true,value:await ledger.delivery({...payload,status:'done'},user)}}catch(error){return {ok:false,error:error.message}}});
  const recoverDeliveries=()=>{if(game.user?.id!==game.users.activeGM?.id)return;for(const actor of activeActors.values())for(const r of Object.values(actor.flags?.[MODULE_ID]?.metapower?.receipts??{}))if(r.delivery&&r.delivery.status!=='done')deliverCommitted({actorUuid:actor.uuid,nonce:r.nonce}).catch(onError)};
  Hooks.on('userConnected',recoverDeliveries);Hooks.on('updateUser',recoverDeliveries);Promise.resolve().then(recoverDeliveries);
  const wrap=(path,fn,type='WRAPPER')=>libWrapper.register(MODULE_ID,path,fn,type);
  for(const actor of values(game.actors))if(eligible(actor))maintain(actor).catch(onError);
  Hooks.on('createActor',actor=>maintain(actor).catch(onError));
  Hooks.on('createToken',token=>{if(token.actor)maintain(token.actor).catch(onError)});
  Hooks.on('createItem',item=>{if(sourceUuid(item)===METAPOWER_SOURCES.widen&&item.actor)maintain(item.actor).catch(onError)});
  // Endpoint is synchronous admission evidence collection, not an async create
  // hook; original batch/options and the actual native document objects survive.
  wrap('CONFIG.ChatMessage.documentClass.createDocuments',async function(wrapped,data,options){const created=await wrapped(data.map(d=>observer.decorate(d)),options);observer.record(created);return created});
  // Public sheet controllers return the same handler map their listener awaits.
  // Find each most-derived registered class; inherited implementations are
  // wrapped once per class and only modify its actual use-action handler.
  const seen=new Set();
  for(const [key,definition]of Object.entries(globalThis.CONFIG.Actor.sheetClasses.character??{})){
   if(!definition.cls?.prototype?.activateClickListener||seen.has(definition.cls))continue;seen.add(definition.cls);
   wrap(`CONFIG.Actor.sheetClasses.character[${JSON.stringify(key)}].cls.prototype.activateClickListener`,function(wrapped,...args){return wrapSheetHandlers(this,wrapped(...args),observe,eligible)});
  }
  // Toolbelt 3.56.2 freezes its API (non-configurable descriptor). Its owned DOM
  // entrance calls the same captured helper; never try to replace that API.
  const toolbeltNative=game.toolbelt?.api?.actionable?.useAction;
  const useToolbelt=toolbeltNative?createToolbeltEntrance({native:toolbeltNative,eligible,observe}):null;
  if(useToolbelt&&game.modules.get('pf2e-hud')?.version==='2.55.2'&&game.modules.get('pf2e-toolbelt')?.version==='3.56.2'){
   const patchHUD=(app,kind)=>{
    // HUD 2.55.2 preserves these class names. Both collections also contain
    // strikes, stances, spells and other controls with different use contracts.
    const className=kind==='sidebar'?'ActionsSidebarAction':'ActionShortcut';
    for(const controller of values(kind==='sidebar'?app.sidebarItems:app.shortcuts)){
     if(!controller?.item||Object.getPrototypeOf(controller)?.constructor?.name!==className||kind==='persistent'&&controller.type!=='action')continue;
     patchHudController(controller,{kind,eligible,useToolbelt}).catch(onError);
    }
   };
   Hooks.on('renderActionsSidebarPF2eHUD',app=>patchHUD(app,'sidebar'));
   Hooks.on('renderPersistentShortcutsPF2eHUD',app=>patchHUD(app,'persistent'));
  }
  const roots=new WeakSet();
  const renderSheet=(app,html)=>{
   const root=html?.[0]??html;if(!root?.addEventListener||!app.actor?.isOwner||app.isEditable===false)return;
   ensureNativeUseControls(root,app.actor,item=>!!metapowerKind(item)||!!powerProfile(item)||supportsOriginalUse(item));
   root.querySelector('.metapower-reconcile')?.remove();
   const pending=ledgerState(app.actor).pending;
   root.querySelector('.metapower-delivery-recovery')?.remove();
   const undelivered=Object.values(ledgerState(app.actor).receipts).find(r=>r.delivery&&r.delivery.status!=='done');
   if(undelivered&&game.user.id===game.users.activeGM?.id){
    const button=root.ownerDocument.createElement('button');button.type='button';button.className='metapower-delivery-recovery';button.textContent='恢复已提交动作的后续结算';button.addEventListener('click',async event=>{event.preventDefault();event.stopPropagation();try{
     const result=await globalThis.foundry.applications.api.DialogV2.wait({window:{title:'恢复原始动作后续结算'},content:'<p>重试会使用原始回执与各能力的幂等记录。若原卡已删除或只能手动结算，请先由GM核对并完成频次、Charged、延迟触发等全部后续，再选择手动结算完成；不会退款或补发效果。</p>',buttons:[{action:'retry',label:'重试原始后续',callback:()=> 'retry'},{action:'manual',label:'GM已手动完成全部后续',callback:()=> 'manual'},{action:'cancel',label:'取消',callback:()=>null}],rejectClose:false});
     if(result)await request(result==='retry'?'deliver':'ack-delivery',{actorUuid:app.actor.uuid,nonce:undelivered.nonce,...(result==='manual'?{confirmation:'gm-manual-effects-settled'}:{})});
    }catch(error){onError(error)}});(root.querySelector('.tab.actions, [data-tab="actions"].tab')??root).append(button);
   }
   if(pending&&game.user.id===game.users.activeGM?.id){
    const button=root.ownerDocument.createElement('button');button.type='button';button.className='metapower-reconcile';button.textContent='处理未完成的威能动作';
    button.addEventListener('click',async event=>{event.preventDefault();event.stopPropagation();try{
     const yes=await globalThis.foundry.applications.api.DialogV2.confirm({window:{title:'处理未完成的原生动作'},content:'<p>仅在原操作客户端已停止或断线后使用。先核对原卡、频次和 Charged；此操作将回执保留为不确定并释放动作锁，不退款、不重试、不补发伤害或效果。需要的规则结算由GM核对后手动完成。</p>'});
     if(yes)await request('reconcile',{actorUuid:app.actor.uuid,nonce:pending,confirmation:'archive-uncertain'});
    }catch(error){onError(error)}});
    (root.querySelector('.tab.actions, [data-tab="actions"].tab')??root).append(button);
   }
   if(roots.has(root)||!eligible(app.actor))return;roots.add(root);
   root.addEventListener('click',event=>{
    const button=event.target?.closest?.('button.use-action:not([data-action])'),item=app.actor.items.get(button?.closest?.('[data-item-id]')?.dataset.itemId);
    if(!item||!useToolbelt||button.disabled)return;
    event.preventDefault();event.stopImmediatePropagation();Promise.resolve(useToolbelt(event,item)).catch(onError);
   },true);
  };
  for(const hook of ['renderActorSheetPF2e','renderCharacterSheetPF2e','renderActorSheetV2'])Hooks.on(hook,renderSheet);
  installActionEntrances({game,eligible,observe,continuation:(actor,proof)=>{
   const card=game.messages.get(proof.cardId),state=card?.flags?.[MODULE_ID]?.medic;
   return proof.actorUuid===actor.uuid&&state?.actorUuid===actor.uuid&&state.nonce===proof.nonce&&state.status==='treatment'&&card.flags?.pf2e?.origin?.uuid===state.itemUuid;
  }});
  installLegacyActionBoundary({game,blocked:actor=>eligible(actor)&&!!(ledgerState(actor).armed||ledgerState(actor).pending),onError});
  const index=globalThis.CONFIG.Dice.rolls.findIndex(C=>C.name==='DamageRoll');
  wrap(`CONFIG.Dice.rolls.${index}.prototype.toMessage`,async function(wrapped,data={},options={}){
   const nativeMessage=async(data,options)=>{
   const option=data.flags?.pf2e?.context?.options?.find(o=>o.startsWith(prefix));if(!option)return wrapped(data,options);
   const [cardId,nonce]=option.slice(prefix.length).split(':'),card=game.messages.get(cardId),proof={cardId,nonce,actorUuid:card?.flags?.[MODULE_ID]?.metapowerUse?.actorUuid};
   const snapshot=await validateDamageProof(proof);
   if(data.flags?.pf2e?.origin?.uuid!==snapshot.itemUuid)throw Error('Native damage origin differs from the bound power.');
   if(data.flags?.pf2e?.context?.options?.includes(`${MODULE_ID}:electric-shot-failure-half`)){
    if(snapshot.powerId!=='electric-shot')throw Error('The half-base failure branch belongs only to Electric Shot.');
    const targets=values(game.user.targets),target=targets.length===1?targets[0].actor:null,targetTokenUuid=(targets[0]?.document??targets[0])?.uuid;
    const source=await fromUuid(proof.actorUuid),originalTargets=ledgerState(source).receipts[proof.nonce]?.selection?.targetUuids;
    if(originalTargets?.length!==1||originalTargets[0]!==targetTokenUuid)throw Error('Select the original Electric Shot recipient for its already-Shocked failure branch.');
    if(!target||!values(target.items).some(i=>['Compendium.battlezoo-eldamon-pf2e.conditions.1fZbuJEbVmE3J4XL','Compendium.battlezoo-eldamon-pf2e.conditions.Item.1fZbuJEbVmE3J4XL'].includes(sourceUuid(i))))throw Error('Electric Shot half-base failure requires one selected already-Shocked recipient.');
    const prior=this.options?.[MODULE_ID]?.metapowerShotFailure;
    if(prior&&(prior.nonce!==proof.nonce||prior.targetTokenUuid!==targetTokenUuid))throw Error('Electric Shot failure roll is already bound to another source or recipient.');
    if(!prior)applyNativeOutcomeInPlace(this,.5);this.options[MODULE_ID]={...this.options[MODULE_ID],metapowerShotFailure:{...proof,targetActorUuid:target.uuid,targetTokenUuid}};
   }
   if(snapshot.siphon?.applies){convertSiphonRoll(this,{rejectMixedPartitions:true});this.options[MODULE_ID]={...this.options[MODULE_ID],metapowerDamage:proof};}
   return wrapped(data,options);
   };
   return interceptDamageMessage(this,data,options,nativeMessage);
  });
  Hooks.on('renderChatMessageHTML',(message,html)=>{
   const proof=message.flags?.[MODULE_ID]?.metapowerUse;if(!proof)return;
   const actor=message.speakerActor??message.actor??game.actors.get(message.speaker.actor),receipt=actor?.flags?.[MODULE_ID]?.metapower?.receipts?.[proof.nonce];
   if(receipt?.messageUuid!==message.uuid||receipt.status!=='committed')return;
   renderMetapowerCard(message,html,{receipt,onClear:r=>request('clear',{actorUuid:r.actorUuid,activationNonce:r.nonce}),onRetryDelivery:r=>request('deliver',{actorUuid:r.actorUuid,nonce:r.nonce}),onError});
  });
  const expire=()=>{if(game.user?.id===game.users.activeGM?.id)for(const actor of activeActors.values())if(actor.flags?.[MODULE_ID]?.metapower?.armed)ledger.expire({actorUuid:actor.uuid},game.user).catch(onError)};
  Hooks.on('updateCombat',expire);Hooks.on('deleteCombat',expire);
 }
 function wrapStrike(strike,actor){
  if(!eligible(actor))return strike;
  for(const variant of strike?.variants??[]){const native=variant.roll;if(typeof native!=='function'||native.metapowerWrapped)continue;const wrapped=function(...args){return observe({actor,entry:'native-check'},()=>native.apply(this,args))};wrapped.metapowerWrapped=true;variant.roll=wrapped;}
  if(strike?.variants?.[0])strike.roll=strike.attack=strike.variants[0].roll;
  return strike;
 }
 return {register,maintain,beforeDamage,wrapStrike,observe,interceptCheck,validateDamageProof,deliverCommitted,diagnostic:{unsupported:['legacy callback-only actions','direct frozen Toolbelt API macro calls','custom macros','automated reaction providers without actual-use entrance'],highVoltage:'delegated-native-voltage-executor'}};
}
