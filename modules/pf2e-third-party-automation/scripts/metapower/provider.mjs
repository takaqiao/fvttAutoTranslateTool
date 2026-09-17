import {METAPOWER_SOURCES,metapowerKind,powerProfile,sourceUuid,buildChannelSnapshot,siphonMultiplier} from './rules.mjs';
import {MODULE_ID,createMetapowerLedger,ledgerState,chargedEffect} from './lifecycle.mjs';
import {createMetapowerObserver} from './observer.mjs';
import {renderMetapowerCard} from './card.mjs';
import {installActionEntrances,wrapSheetHandlers} from './entrances.mjs';
import {convertSiphonRoll} from './damage.mjs';
import {showNativeChoice} from '../native-context.mjs';
const values=c=>Array.from(c?.values?.()??c??[]);
const prefix=`${MODULE_ID}:metapower:`;
export function preserveMetapowerOnAlter(original,result){
 const proof=original?.options?.[MODULE_ID]?.metapowerDamage;
 if(proof&&result?.options)result.options[MODULE_ID]={...result.options[MODULE_ID],metapowerDamage:structuredClone(proof)};
 return result;
}

export function createMetapowerProvider({game,fromUuid,onError=console.error,selectChoice=showNativeChoice,onCommittedChannel=async()=>{}}){
 const ledger=createMetapowerLedger({game,fromUuid});let socket;
 const eligible=actor=>actor?.type==='character'&&values(actor.items).some(item=>metapowerKind(item));
 const request=async(method,payload)=>{
  if(!socket||!game.users.activeGM)throw Error('Metapower automation requires an active GM and socketlib.');
  const response=await socket.executeAsUser(`metapower:${method}`,game.users.activeGM.id,payload);
  if(!response?.ok)throw Error(response?.error??'Metapower coordinator did not respond.');
  if(method==='finish'&&response.value?.messageUuid){const m=await fromUuid(response.value.messageUuid);await m?.update({[`flags.${MODULE_ID}.metapowerUse.status`]:response.value.status});if(response.value.status==='committed')await onCommittedChannel({receipt:response.value,message:m});}
  return response.value;
 };
 async function select(item){
  const profile=powerProfile(item),armed=ledgerState(item.actor).armed;
  if(!profile||!armed&&!profile.reaction)return {};
  let selection={discharge:false};
  if(armed&&profile.id!=='high-voltage'){
   const choices=[];
   for(const discharge of [false,true]){
    if(discharge&&!chargedEffect(item.actor))continue;
    if(discharge&&armed.kind==='siphoning'&&profile.id==='reactive-chain')continue;
    const distances=profile.areaType?Array.from({length:25},(_,i)=>5+i*5):[null];
    for(const baseDistance of distances){
     const candidate={discharge,...(baseDistance?{baseDistance}:{})};let snapshot;
     try{snapshot=buildChannelSnapshot({kind:armed.kind,item,selection:candidate,policy:{dischargeNonDamage:'remove',dischargeRange:'retain',dischargeSaveDowngrade:'retain',highVoltage:'convert'}})}catch{continue}
     choices.push({value:JSON.stringify(candidate),label:`${discharge?'放电：Charged −1':'普通分支'}${snapshot.area?` · ${snapshot.area.distance} 尺` : ''}${armed.kind==='siphoning'?' · 虹吸':''}`});
    }
   }
   const result=await selectChoice({title:item.name,choices});if(!result)return null;selection=JSON.parse(result);
  }
  if(profile.reaction){
   const chain=profile.id==='reactive-chain';
   const result=await globalThis.foundry.applications.api.DialogV2.wait({window:{title:item.name},content:`<p>${chain?'确认：30尺内生物实际受到电击伤害；所选目标在该生物30尺内，未受同一效果电击伤害，且已Shocked。虹吸不允许用放电放宽此资格。':'确认本次真实反应触发符合原威能条件。'}</p>${chain?'<label>触发生物实际受到的电击伤害 <input name="triggerDamage" type="number" min="1" step="1" value="1"></label>':''}`,buttons:[{action:'confirm',label:'确认实际触发',callback:(_event,_button,dialog)=>({triggerConfirmed:true,eligibleTargetConfirmed:chain,triggerDamage:chain?Number(dialog.element.querySelector('[name="triggerDamage"]').value):null})},{action:'cancel',label:'取消',callback:()=>null}],rejectClose:false});
   if(!result)return null;Object.assign(selection,result);
  }
  return selection;
 }
 const observer=createMetapowerObserver({request,select,onError});
 const observe=(context,native)=>eligible(context.actor)?observer.observe(context,native):native();
 async function validateDamageProof(proof){
  const card=game.messages.get(proof?.cardId),actor=await fromUuid(proof?.actorUuid),receipt=actor?.flags?.[MODULE_ID]?.metapower?.receipts?.[proof?.nonce];
  if(!card||!receipt||receipt.status!=='committed'||receipt.messageUuid!==card.uuid||card.flags?.[MODULE_ID]?.metapowerUse?.nonce!==proof.nonce||receipt.snapshot?.itemUuid!==card.flags?.pf2e?.origin?.uuid)throw Error('Metapower damage source/card binding is invalid.');
  return receipt.snapshot;
 }
 async function beforeDamage(actor,params){
  const proof=params.damage?.options?.[MODULE_ID]?.metapowerDamage;if(!proof)return null;
  const snapshot=await validateDamageProof(proof),multiplier=siphonMultiplier(snapshot,actor.traits??actor.system?.traits?.value??[]);
  return {params:multiplier===1?params:{...params,damage:params.damage.alter(multiplier,0)}};
 }
 async function maintain(actor){
  if(game.user?.id!==game.users.activeGM?.id||!actor?.testUserPermission(game.user,'OWNER'))return;
  for(const item of values(actor.items))if(sourceUuid(item)===METAPOWER_SOURCES.widen&&(item.system.actionType?.value!=='action'||item.system.actions?.value!==1))await item.update({'system.actionType.value':'action','system.actions.value':1});
 }
 function register({Hooks,libWrapper,socket:api}){
  socket=api;
  for(const method of ['begin','start','finish','clear','expire'])socket.register(`metapower:${method}`,async function(payload){try{return {ok:true,value:await ledger[method](payload,game.users.get(this.socketdata.userId))}}catch(error){return {ok:false,error:error.message}}});
  const wrap=(path,fn,type='WRAPPER')=>libWrapper.register(MODULE_ID,path,fn,type);
  // Endpoint is synchronous admission evidence collection, not an async create
  // hook; original batch/options and the actual native document objects survive.
  wrap('CONFIG.ChatMessage.documentClass.createDocuments',async function(wrapped,data,options){const created=await wrapped(data.map(d=>observer.decorate(d)),options);observer.record(created);return created});
  wrap('game.pf2e.rollItemMacro',async function(wrapped,uuid,event){
   const item=typeof uuid==='string'&&uuid.includes('.')?await fromUuid(uuid):game.user.character?.items.get(uuid);
   return item?.actor&&['feat','action'].includes(item.type)?observe({actor:item.actor,item},()=>wrapped(uuid,event)):wrapped(uuid,event);
  });
  // Public sheet controllers return the same handler map their listener awaits.
  // Find each most-derived registered class; inherited implementations are
  // wrapped once per class and only modify its actual use-action handler.
  const seen=new Set();
  for(const [key,definition]of Object.entries(globalThis.CONFIG.Actor.sheetClasses.character??{})){
   if(!definition.cls?.prototype?.activateClickListener||seen.has(definition.cls))continue;seen.add(definition.cls);
   wrap(`CONFIG.Actor.sheetClasses.character[${JSON.stringify(key)}].cls.prototype.activateClickListener`,function(wrapped,...args){return wrapSheetHandlers(this,wrapped(...args),observe,eligible)});
  }
  if(game.toolbelt?.api?.actionable?.useAction)wrap('game.toolbelt.api.actionable.useAction',async function(wrapped,event,item,virtual){
   if(!eligible(item?.actor))return wrapped(event,item,virtual);
   if(virtual||item.flags?.['pf2e-toolbelt']?.actionable?.linked)throw Error('Virtual/linked-macro action needs manual metapower resolution; use the owned original item.');
   return observe({actor:item.actor,item},()=>wrapped(event,item,virtual));
  });
  const roots=new WeakSet();
  const renderSheet=(app,html)=>{
   const root=html?.[0]??html;if(!root?.addEventListener||roots.has(root)||!eligible(app.actor))return;roots.add(root);
   root.addEventListener('click',event=>{
    const button=event.target?.closest?.('button.use-action:not([data-action])'),item=app.actor.items.get(button?.closest?.('[data-item-id]')?.dataset.itemId);
    if(!item||!game.toolbelt?.api?.actionable?.useAction||button.disabled)return;
    event.preventDefault();event.stopImmediatePropagation();Promise.resolve(game.toolbelt.api.actionable.useAction(event,item)).catch(onError);
   },true);
  };
  for(const hook of ['renderActorSheetPF2e','renderCharacterSheetPF2e','renderActorSheetV2'])Hooks.on(hook,renderSheet);
  installActionEntrances({game,eligible,observe});
  // Native spell entrances expire a pending metapower before starting the cast.
  // Consume/message-false calls are known continuations handled by cast owner.
  wrap('CONFIG.PF2E.Item.documentClasses.spellcastingEntry.prototype.cast',function(wrapped,item,options={}){return options.consume===false||options.message===false?wrapped(item,options):observe({actor:item.actor,entry:'spell'},()=>wrapped(item,options))});
  const index=globalThis.CONFIG.Dice.rolls.findIndex(C=>C.name==='DamageRoll');
  wrap(`CONFIG.Dice.rolls.${index}.prototype.toMessage`,async function(wrapped,data={},options={}){
   const option=data.flags?.pf2e?.context?.options?.find(o=>o.startsWith(prefix));if(!option)return wrapped(data,options);
   const [cardId,nonce]=option.slice(prefix.length).split(':'),card=game.messages.get(cardId),proof={cardId,nonce,actorUuid:card?.flags?.[MODULE_ID]?.metapowerUse?.actorUuid};
   const snapshot=await validateDamageProof(proof);
   if(data.flags?.pf2e?.origin?.uuid!==snapshot.itemUuid)throw Error('Native damage origin differs from the bound power.');
   if(snapshot.siphon?.applies){convertSiphonRoll(this,{rejectMixedPartitions:true});this.options[MODULE_ID]={...this.options[MODULE_ID],metapowerDamage:proof};}
   return wrapped(data,options);
  });
  Hooks.on('renderChatMessageHTML',(message,html)=>{
   const proof=message.flags?.[MODULE_ID]?.metapowerUse;if(!proof)return;
   const actor=message.speakerActor??message.actor??game.actors.get(message.speaker.actor),receipt=actor?.flags?.[MODULE_ID]?.metapower?.receipts?.[proof.nonce];
   if(receipt?.messageUuid!==message.uuid||receipt.status!=='committed')return;
   renderMetapowerCard(message,html,{receipt,onClear:r=>request('clear',{actorUuid:r.actorUuid,activationNonce:r.nonce}),onError});
  });
  const expire=()=>{if(game.user?.id===game.users.activeGM?.id)for(const actor of values(game.actors))if(eligible(actor)&&ledgerState(actor).armed)ledger.expire({actorUuid:actor.uuid},game.user).catch(onError)};
  Hooks.on('updateCombat',expire);Hooks.on('deleteCombat',expire);
 }
 function wrapStrike(strike,actor){
  if(!eligible(actor))return strike;
  for(const variant of strike?.variants??[]){const native=variant.roll;if(typeof native!=='function'||native.metapowerWrapped)continue;const wrapped=function(...args){return observe({actor,entry:'native-check'},()=>native.apply(this,args))};wrapped.metapowerWrapped=true;variant.roll=wrapped;}
  return strike;
 }
 return {register,maintain,beforeDamage,wrapStrike,observe,validateDamageProof,diagnostic:{unsupported:['legacy callback-only actions','HUD cached closures','custom macros','automated reaction providers without actual-use entrance'],highVoltage:'snapshot-only; delayed executor required'}};
}
