import {MODULE_ID} from './rules.mjs';
import {requireOwner} from './runtime.mjs';

export const getSourceId=item=>item?.sourceId??item?._stats?.compendiumSource??item?.flags?.core?.sourceId??null;
export const isActiveGM=game=>!!game?.users?.activeGM?.id&&game.user?.id===game.users.activeGM.id;
const values=collection=>Array.from(collection?.values?.()??collection??[]);
const escapeHTML=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const unappliedDamageErrors=new WeakSet();
/** Brand only failures known to precede the native damage call; never serialize. */
export function markUnappliedDamageError(error){
 if(error!==null&&(['object','function'].includes(typeof error)))unappliedDamageErrors.add(error);
 return error;
}
export const isUnappliedDamageError=error=>unappliedDamageErrors.has(error);

/** Resolve recorded targets only; the executing GM's current selection is unrelated. */
export async function resolveMessageTargets(message,{fromUuid=globalThis.fromUuid}={}){
 const target=message?.flags?.pf2e?.context?.target;
 const nativeToken=typeof target?.token==='string'?target.token:target?.token?.uuid;
 const input=message?.flags?.[MODULE_ID]?.usageInput;
 const candidates=nativeToken?[nativeToken]:Array.isArray(input?.targetUuids)?input.targetUuids:
  message?.flags?.['pf2e-toolbelt']?.targetHelper?.targets??[];
 const result=[];
 for(const uuid of new Set(candidates)){
  if(typeof uuid!=='string'||!/^Scene\.[^.]+\.Token\.[^.]+$/.test(uuid))continue;
  const doc=await fromUuid(uuid);
  if(doc?.documentName==='Token'&&doc.uuid===uuid&&doc.actor)result.push(doc);
 }
 return result;
}

/** Key ownership prevents name collisions with system and third-party effects. */
export async function upsertOwnedEffect(actor,key,data){
 if(typeof key!=='string'||!key||data?.type!=='effect')throw Error('需要有效的自动化效果及唯一标识。');
 const existing=values(actor.items).filter(i=>i.type==='effect'&&i.flags?.[MODULE_ID]?.nativeEffectKey===key);
 const next=structuredClone(data);delete next._id;
 next.flags={...next.flags,[MODULE_ID]:{...next.flags?.[MODULE_ID],nativeEffectKey:key}};
 if(existing.length){
  await existing[0].update(next);
  if(existing.length>1)await actor.deleteEmbeddedDocuments('Item',existing.slice(1).map(i=>i.id));
  return existing[0];
 }
 return (await actor.createEmbeddedDocuments('Item',[next]))[0];
}

export function validateNativeChoices(choices){
 if(!Array.isArray(choices)||choices.length<1||choices.length>100||choices.some(c=>typeof c?.value!=='string'||!c.value||c.value.length>256||typeof c.label!=='string'||c.label.length>500)||new Set(choices.map(c=>c.value)).size!==choices.length)throw Error('规则选择列表无效。');
 return choices.map(({value,label})=>({value,label}));
}
export async function showNativeChoice({title,choices}){
 const options=validateNativeChoices(choices);
 if(options.length===1)return options[0].value;
 return globalThis.foundry.applications.api.DialogV2.wait({
  window:{title:String(title??'规则选择').slice(0,200)},content:'<p>请选择本次能力的规则分支。</p>',
  buttons:options.map((option,index)=>({action:`choice-${index}`,label:escapeHTML(option.label),callback:()=>option.value})),rejectClose:false,
 });
}

/** The active GM requests only a bounded rule decision from the original owner. */
export function createNativeChooser({game,send,show=showNativeChoice}){
 return async({actor,user,title,choices})=>{
  if(!isActiveGM(game))throw Error('规则选择必须由当前主GM请求。');
  requireOwner(actor,user);
  const options=validateNativeChoices(choices);
  if(options.length===1)return options[0].value;
  const payload={actorUuid:actor.uuid,title:String(title??'规则选择').slice(0,200),choices:options};
  const selected=user.id===game.user.id?await show(payload):await send(user.id,payload);
  if(selected===null||selected===undefined||selected===false)return null;
  if(!options.some(option=>option.value===selected))throw Error('收到无效的规则选择。');
  return selected;
 };
}

/** One native application, with independently claimed provider effects around it. */
export async function runDamagePipeline({actor,params,providers,apply,onError=()=>{}}){
 const receipts=[];let actual=params,enteredNative=false,applied=false;
 try{
  for(const provider of providers){
   if(!provider.beforeDamage)continue;
   const prepared=await provider.beforeDamage(actor,actual);
   if(!prepared)continue;
   actual=prepared.params??actual;
   if(Object.hasOwn(prepared,'receipt'))receipts.push([provider,prepared.receipt]);
  }
  enteredNative=true;
  const result=await apply(actual);applied=true;return result;
 }catch(error){
  if(isUnappliedDamageError(error))enteredNative=false;
  throw error;
 }finally{
  for(const [provider,receipt]of receipts){
   try{await provider.afterDamage?.(receipt,{applied,uncertain:enteredNative&&!applied});}
   catch(error){try{onError(error)}catch{/* Never replay or mask native damage. */}}
  }
 }
}
