import {MODULE_ID,SOURCES,hasSource} from './rules.mjs';

export const USE_ACTION_OPTION='origin:action:slug:use-action';
const entries=[['breath','breath'],['circadian','rest'],['cycle','cycle']];
const values=collection=>Array.from(collection?.values?.()??collection??[]);
const authorId=message=>message.author?.id??message.user?.id??message.user;
export const defaultUsageAction=item=>entries.find(([key])=>hasSource(item,SOURCES[key]))?.[1];
const defaultFrequencyMatch=item=>hasSource(item,SOURCES.breath);
const escapeHTML=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

export function usageStatusHTML(usage){
 const label={pending:'正在自动结算…',done:'已自动结算',error:'自动结算未完成'}[usage?.status];
 if(!label)return '';
 const detail=usage.status==='error'?usage.error:usage.result;
 return `<div class="third-party-usage-result" role="status" style="border-top:1px solid var(--color-border-light-primary);margin-top:8px;padding-top:6px"><strong>${label}</strong>${detail?`：${escapeHTML(detail)}`:''}</div>`;
}

/** Only original feature cards are usage events. Rolls and our result messages are never inputs. */
export function parseUsageMessage(message,item,{resolveAction=defaultUsageAction}={}){
 const flags=message?.flags??{},own=flags[MODULE_ID]??{},pf=flags.pf2e??{};
 if(!item?.actor||!['feat','action','spell'].includes(item.type)||own.usage||own.usageGenerated||message.isCheckRoll||message.isRoll||message.rolls?.length)return null;
 if(pf.context&&pf.context.type!=='self-effect'&&!(item.type==='spell'&&pf.context.type==='spell-cast'))return null;
 const actor=item.actor,origin=pf.origin;
 if(origin){
  if(origin.uuid!==item.uuid||(origin.actor&&origin.actor!==actor.uuid)||!['feat','action','spell'].includes(origin.type))return null;
 }else if(pf.context?.type!=='self-effect'||pf.context.item!==item.id)return null;
 if(message.speaker?.actor!==actor.id)return null;
 const action=resolveAction(item);if(!action)return null;
 return {action,itemUuid:item.uuid,actorUuid:actor.uuid,userId:authorId(message),actualUse:own.usageInput?.actualUse===true||origin?.rollOptions?.includes(USE_ACTION_OPTION)===true,frequencyReceiptId:own.usageInput?.frequencyReceiptId??null};
}

/** Mirror committed frequency values so concurrent stale writes cannot pay for two usages. */
export function createFrequencyTracker({now=Date.now,ttl=5000,matches=defaultFrequencyMatch}={}){
 const observed=new Map(),receipts=new Map(),consumed=new Set();
 return {
  seed(item){if(matches(item))observed.set(item.uuid,item.system.frequency?.value??item.system.frequency?.max);},
  observe(item,options,userId){
   if(!matches(item))return null;
   const before=observed.get(item.uuid),after=item.system.frequency?.value;
   observed.set(item.uuid,after);
   const proof=options?.[MODULE_ID]?.frequencyReceipt;
   for(const[id,entry]of receipts)if(now()-entry.observedAt>ttl)receipts.delete(id);
   if(!proof||typeof proof.id!=='string'||proof.itemUuid!==item.uuid||proof.userId!==userId||proof.before!==before||proof.after!==after||!Number.isInteger(before)||before<1||after!==before-1||receipts.has(proof.id)||consumed.has(proof.id))return null;
   const receipt={id:proof.id,itemUuid:item.uuid,userId,before,after,createdAt:proof.createdAt};
   receipts.set(proof.id,{receipt,observedAt:now()});return receipt;
  },
  claim(id,{itemUuid,userId}){
   const entry=receipts.get(id);
   if(!entry||entry.receipt.itemUuid!==itemUuid||entry.receipt.userId!==userId||now()-entry.observedAt>ttl)throw Error('技能使用次数回执无效或已经结算。');
   receipts.delete(id);consumed.add(id);return entry.receipt;
  },
 };
}

/** Install on every client at ready; only activeGM calls executeUsage. Returns an unregister function. */
export function registerUsageEvents({game,Hooks,executeUsage,resolveAction=defaultUsageAction,captureUsage=()=>null,onMessageOutcome=()=>false,tracksFrequency=defaultFrequencyMatch,fromUuid=globalThis.fromUuid,libWrapper=globalThis.libWrapper,canvas=globalThis.canvas,onError=error=>console.error(MODULE_ID,error),now=Date.now}){
 const tracker=createFrequencyTracker({now,matches:tracksFrequency}),pending=new Map(),scopes=new Map(),inFlight=new Set(),processing=new Set(),registrations=[],wrappers=[],listeners=[],capturedElements=new WeakSet();
 const on=(name,callback)=>registrations.push([name,Hooks.on(name,callback)]);
 const seedActor=actor=>{for(const item of values(actor?.items))tracker.seed(item);};
 const seedScene=scene=>{for(const token of values(scene?.tokens))seedActor(token.actor);};
 for(const actor of values(game.actors))seedActor(actor);
 for(const scene of values(game.scenes))seedScene(scene);
 seedScene(canvas?.scene);
 const scopeActive=item=>{const scope=scopes.get(item.uuid);return scope&&(scope.running||scope.expires>now());};
 const selectedTargets=()=>[...new Set(values(game.user.targets).map(t=>t.document?.uuid??t.uuid).filter(uuid=>typeof uuid==='string'))];
 const takeReceipt=(item,actualUse)=>{
  const key=item.uuid+':'+game.user.id,receipts=(pending.get(key)??[]).filter(r=>now()-r.createdAt<=5000);
  const receipt=actualUse?receipts.shift()??null:null;
  if(receipts.length)pending.set(key,receipts);else pending.delete(key);
  return receipt;
 };
 const randomId=()=>globalThis.foundry?.utils?.randomID?.()??globalThis.crypto.randomUUID();
 on('createItem',item=>tracker.seed(item));
 // Actor imports and token actors arrive with embedded items already present.
 on('createActor',seedActor);
 on('createToken',token=>seedActor(token.actor));
 on('canvasReady',view=>seedScene(view?.scene));
 on('preUpdateItem',(item,changes,options,userId)=>{
  if(!tracksFrequency(item)||processing.has(item.uuid)||options?.[MODULE_ID]?.usageInternal)return;
  const before=item.system.frequency?.value,after=changes['system.frequency.value']??changes.system?.frequency?.value;
  if(!Number.isInteger(before)||before<1||after!==before-1)return;
  const receipt={id:randomId(),itemUuid:item.uuid,userId,before,after,createdAt:now()};
  options[MODULE_ID]={...options[MODULE_ID],frequencyReceipt:receipt};
 });
 on('updateItem',(item,changes,options,userId)=>{
  const receipt=tracker.observe(item,options,userId);
  if(receipt&&userId===game.user.id){const key=item.uuid+':'+userId;pending.set(key,[...(pending.get(key)??[]),receipt]);}
 });
 // PF2e's private native Use builder bypasses toMessage for selfEffect items.
 // This hook runs on the creating client and preserves that use boundary.
 on('preCreateChatMessage',message=>{
  const pf=message.flags?.pf2e,own=message.flags?.[MODULE_ID];
  if(pf?.context?.type!=='self-effect'||own?.usageInput||own?.usage||own?.usageGenerated)return;
  const actor=message.actor??game.actors.get?.(message.speaker?.actor),item=actor?.items?.get?.(pf.context.item);
  if(!item||!resolveAction(item))return;
  const scope=scopeActive(item)?scopes.get(item.uuid):null;
  const receipt=takeReceipt(item,true),actualUse=!!scope||!!receipt;
  const targetUuids=scope?.targetUuids??selectedTargets();scopes.delete(item.uuid);
  message.updateSource({[`flags.${MODULE_ID}.usageInput`]:{actualUse,frequencyReceiptId:receipt?.id??null,targetUuids}});
 });

 // Original native sheet Use buttons call a private PF2e function; capture its actual DOM boundary.
 const captureNativeUse=(app,html)=>{
  const element=html?.[0]??html,actor=app.actor??app.document;
  if(!element?.addEventListener||!actor?.items||capturedElements.has(element))return;
  capturedElements.add(element);
  const listener=event=>{
   const button=event.target?.closest?.('[data-action="use-action"],button.use-action');
   const id=button?.closest?.('[data-item-id]')?.dataset.itemId;
   const item=actor.items.get?.(id);if(!item||!resolveAction(item))return;
   scopes.set(item.uuid,{expires:now()+5000,running:false,targetUuids:selectedTargets()});
  };
  element.addEventListener('click',listener,true);listeners.push([element,listener]);
 };
 on('renderActorSheetPF2e',captureNativeUse);
 on('renderCharacterSheetPF2e',captureNativeUse);
 on('renderActorSheetV2',captureNativeUse);

 if(libWrapper){
  const wrap=(path,callback,type='WRAPPER')=>{libWrapper.register(MODULE_ID,path,callback,type);wrappers.push(path)};
  // Auto Action Tracker 0.19.1 calls setFlag even when toMessage(create:false)
  // returned an unsaved draft. Keep its exact item-usage annotation in the draft.
  wrap('CONFIG.ChatMessage.documentClass.prototype.setFlag',async function(wrapped,namespace,key,value){
   if(!this.id&&namespace==='pf2e-auto-action-tracker'&&key==='itemUsage'&&value?.uuid===this.flags?.pf2e?.origin?.uuid){
    const item=await fromUuid(value.uuid);
    if(item&&resolveAction(item)){this.updateSource({[`flags.${namespace}.${key}`]:value});return this;}
   }
   return wrapped(namespace,key,value);
  },'MIXED');
  // This public native hotbar entry encloses its frequency update and private message builder.
  wrap('game.pf2e.rollItemMacro',async function(wrapped,uuid,event){
   let item=typeof uuid==='string'&&uuid.includes('.')?await fromUuid(uuid):null;
   if(!item&&typeof uuid==='string'){
    const speaker=globalThis.ChatMessage?.getSpeaker?.()??{};
    const actor=globalThis.canvas?.tokens?.get?.(speaker.token)?.actor??game.actors.get?.(speaker.actor);
    item=actor?.items.get?.(uuid);
   }
   if(!item||!resolveAction(item))return wrapped(uuid,event);
   const scope={running:true,expires:now()+5000,targetUuids:selectedTargets()};scopes.set(item.uuid,scope);
   try{return await wrapped(uuid,event)}finally{if(scopes.get(item.uuid)===scope)scopes.delete(item.uuid)}
  });
  for(const type of ['feat','action','spell'])wrap(`CONFIG.PF2E.Item.documentClasses.${type}.prototype.toMessage`,async function(wrapped,event,options={}){
   const route=resolveAction(this),captured=captureUsage(this,{options,event});
   if(!route&&!captured?.nativeCast)return wrapped(event,options);
   const scope=scopeActive(this)?scopes.get(this.uuid):null;
   const targetUuids=scope?.targetUuids??selectedTargets();
   const actualUse=options.actualUse===true||!!scope;
   scopes.delete(this.uuid);
   const castNonce=captured?.nativeCast?.id,settle=outcome=>castNonce?onMessageOutcome(this,options,{...outcome,castNonce}):false;
   try{
   // Building a draft first preserves Toolbelt's own actualUse roll option and its normal card template.
   const message=await wrapped(event,{...options,create:false});if(!message){settle({error:Error('原生施法未生成消息。')});return message;}
   if(captured)message.updateSource(Object.fromEntries(Object.entries(captured).map(([key,value])=>[`flags.${MODULE_ID}.${key}`,value])));
   if(route&&message.flags?.pf2e?.origin)message.updateSource({'flags.pf2e.origin.rollOptions':[...new Set([...(message.flags.pf2e.origin.rollOptions??[]),`${MODULE_ID}:usage:${route}`])]});
   const receipt=takeReceipt(this,actualUse);
   message.updateSource({[`flags.${MODULE_ID}.usageInput`]:{...message.flags?.[MODULE_ID]?.usageInput,actualUse,frequencyReceiptId:receipt?.id??null,targetUuids:message.flags?.[MODULE_ID]?.usageInput?.targetUuids??targetUuids}});
   if(options.create===false)return message;
   const cls=globalThis.getDocumentClass('ChatMessage');
   const created=await cls.create(message.toObject(),{renderSheet:false});
   if(created)settle({message:created});else settle({error:Error('原生施法未生成消息。')});
   return created;
   }catch(error){if(settle({error})===true)return null;throw error;}
  });
 }

 on('renderChatMessageHTML',(message,html)=>{
  const element=html?.[0]??html,markup=usageStatusHTML(message.flags?.[MODULE_ID]?.usage);
  element?.querySelector?.('.third-party-usage-result')?.remove();
  if(markup)(element?.querySelector?.('.message-content')??element)?.insertAdjacentHTML?.('beforeend',markup);
 });

 on('createChatMessage',async(message,_options,creatingUserId)=>{
  if(game.user?.id!==game.users.activeGM?.id||inFlight.has(message.id)||message.flags?.[MODULE_ID]?.usage)return;
  inFlight.add(message.id);
  let event;
  try{
   const pf=message.flags?.pf2e??{},itemUuid=pf.origin?.uuid;
   const item=itemUuid?await fromUuid(itemUuid):message.item??message.actor?.items?.get?.(pf.context?.item)??game.actors.get?.(message.speaker?.actor)?.items.get?.(pf.context?.item);
   event=parseUsageMessage(message,item,{resolveAction});if(!event)return;
   const user=game.users.get(event.userId);
   if(!user||(creatingUserId&&event.userId!==creatingUserId)||!item.actor.testUserPermission(user,'OWNER'))return;
   if(game.user?.id!==game.users.activeGM?.id||message.flags?.[MODULE_ID]?.usage)return;
   await message.update({[`flags.${MODULE_ID}.usage`]:{status:'pending',action:event.action,userId:user.id,processedBy:game.user.id}});
   const frequencyReceipt=event.frequencyReceiptId?tracker.claim(event.frequencyReceiptId,{itemUuid:item.uuid,userId:user.id}):null;
   processing.add(item.uuid);
   try{
    const result=await executeUsage({actor:item.actor,item,message,user,action:event.action,frequencyReceipt});
    await message.update({[`flags.${MODULE_ID}.usage`]:{status:'done',action:event.action,userId:user.id,processedBy:game.user.id,result:typeof result==='string'?result:undefined}});
   }finally{processing.delete(item.uuid)}
  }catch(error){
   if(event)await message.update({[`flags.${MODULE_ID}.usage`]:{status:'error',action:event.action,error:error.message,processedBy:game.user.id}}).catch(onError);
   onError(error);
  }finally{inFlight.delete(message.id)}
 });
 return ()=>{
  for(const[name,id]of registrations)Hooks.off(name,id);
  for(const path of wrappers)libWrapper.unregister(MODULE_ID,path);
  for(const[element,listener]of listeners)element.removeEventListener('click',listener,true);
 };
}
