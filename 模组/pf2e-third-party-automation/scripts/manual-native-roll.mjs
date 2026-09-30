import {MODULE_ID} from './rules.mjs';
const damageAudiences=new WeakMap();
const damageOwners=new WeakMap();

/** Read immediately after the manual roll, before conversions create a new
 * Roll. toMessage must receive messageMode as its options argument too. */
export function manualDamagePrivacy(roll,minimum){
 const privacy=damageAudiences.get(roll);
 if(!minimum)return privacy?{...privacy,whisper:[...privacy.whisper]}:null;
 const selected=privacy??{blind:false,whisper:[],messageMode:'public'},before=[...minimum.whisper??[]];
 const whisper=before.length?(selected.whisper.length?before.filter(id=>selected.whisper.includes(id)):before):[...selected.whisper];
 const blind=selected.blind||!!minimum.blind;
 if(before.length&&selected.whisper.length&&!whisper.length||blind&&!whisper.length)throw Error('本次原生伤害窗口与原来源受众不相容，未发布伤害。');
 // Foundry's self mode rewrites whisper to the current executing user. A GM
 // continuing another owner's self roll must use explicit recipients instead.
 const messageMode=blind?'blind':!whisper.length?'public':whisper.length===1&&whisper[0]===damageOwners.get(roll)?'self':'gm';
 return {messageMode,blind,whisper};
}

/** PF2e Shift toggles the owner's preference; it is not an absolute skip flag. */
export function nativeRollEvent(game,kind='check'){
 const setting=kind==='damage'?'showDamageDialogs':'showCheckDialogs';
 return {shiftKey:!game.user.settings?.[setting],ctrlKey:false,metaKey:false};
}

/** Native PF2e flat checks deliberately omit modifier windows. Keep their check
 * type and mechanics, with a Foundry confirmation before the real roll. */
export async function confirmManualFlatCheck({label='平检',dc,Dialog=globalThis.foundry?.applications?.api?.DialogV2}={}){
 if(typeof Dialog?.wait!=='function')throw Error('无法打开原生投骰窗口，尚未投骰。');
 return await Dialog.wait({window:{title:label},content:Number.isFinite(dc)?`<p>DC ${dc}</p>`:'',buttons:[{action:'roll',label:'投骰',callback:()=>true},{action:'cancel',label:'取消',callback:()=>false}],rejectClose:false})===true;
}

/** Use PF2e's real inline DamageModifierDialog with no actor/item ancestors.
 * This preserves the already selected base formula without applying unrelated
 * item/actor synthetics. The one local nonce captures the native evaluated roll
 * before publication; the caller keeps its original provenance and settlement. */
export async function manualDamageRoll({game,roll,messageMode,assertLive,Hooks=globalThis.Hooks,TextEditor=game?.pf2e?.TextEditor}={}){
 const json=roll?.toJSON?.(),data=typeof json==='string'?JSON.parse(json):json,formula=data?.formula;
 if(roll?._evaluated!==false||typeof formula!=='string'||!formula.trim()||typeof TextEditor?._onClickInlineRoll!=='function'||typeof Hooks?.on!=='function'||typeof Hooks?.off!=='function'||typeof globalThis.document?.createElement!=='function')throw Error('缺少原生伤害窗口接口，尚未投骰。');
 const nonce=globalThis.foundry?.utils?.randomID?.()??globalThis.crypto?.randomUUID?.();
 if(typeof nonce!=='string'||!nonce)throw Error('无法生成本次原生伤害标记，尚未投骰。');
 const marker=`${MODULE_ID}:manual-damage:${nonce}`,anchor=document.createElement('a');
 Object.assign(anchor.dataset,{formula,baseFormula:formula,damageRoll:'',immutable:'',overrideTraits:'',rollOptions:marker});
 let captured,error,nativeTask,nativeSettled=false,rejectAborted;const dialogs=new Map(),listeners=[];
 assertLive?.();
 const aborted=assertLive?new Promise((_resolve,reject)=>rejectAborted=reject):null;
 const abort=caught=>{error??=caught;for(const app of dialogs.keys()){app.resolve(false);void Promise.resolve(app.close?.()).catch(()=>{});}rejectAborted?.(error);};
 if(messageMode!==undefined&&!['public','gm','blind','self'].includes(messageMode))throw Error('本次原生伤害窗口的固定受众无效。');
 const dialogHook=messageMode===undefined&&!assertLive?undefined:Hooks.on('renderDamageModifierDialog',(app,html)=>{
  if(!Array.from(app.context?.options??[]).includes(marker))return;
  if(messageMode!==undefined){app.context.messageMode=messageMode;const select=(html?.[0]??html)?.querySelector?.('select[name=messageMode]');if(select){select.value=messageMode;select.disabled=true;}}
  if(assertLive&&!dialogs.has(app)&&typeof app.resolve==='function'){
   const original=app.resolve,resolve=original.bind(app);let submitted=false;
   const wrapped=accepted=>{if(submitted)return;submitted=true;if(!accepted||error)return resolve(false);try{assertLive();return resolve(true)}catch(caught){abort(caught);return resolve(false)}};
   dialogs.set(app,{original,wrapped});app.resolve=wrapped;
  }
  if(error){app.resolve(false);void Promise.resolve(app.close?.()).catch(()=>{});}
 });
 const hook=Hooks.on('preCreateChatMessage',message=>{
  const context=message.flags?.pf2e?.context;
  if(context?.type!=='damage-roll'||!context.options?.includes(marker))return;
  try{if(error)throw error;assertLive?.();}catch(caught){abort(caught);return false;}
  const result=message.rolls?.[0];
  if(captured||messageMode!==undefined&&context.messageMode!==messageMode||(message.author?.id??message.author??message.user?.id??message.user)!==game.user.id||message.rolls?.length!==1||!(result instanceof roll.constructor)||result._evaluated!==true||!Number.isFinite(result.total))error=Error('本次原生伤害结果不唯一或无效，不能重复投骰。');
  else {
   captured=result;
   damageOwners.set(result,game.user.id);
   damageAudiences.set(result,{messageMode:context.messageMode??(message.blind?'blind':message.whisper?.length?'gm':'public'),blind:!!message.blind,whisper:[...message.whisper??[]]});
  }
  return false;
 });
 if(assertLive)for(const event of ['updateUser','userConnected','updateActor','deleteActor','deleteItem','updateChatMessage','deleteChatMessage','deleteToken'])listeners.push([event,Hooks.on(event,()=>{try{assertLive()}catch(caught){abort(caught)}})]);
 const cleanup=()=>{Hooks.off('preCreateChatMessage',hook);if(dialogHook!==undefined)Hooks.off('renderDamageModifierDialog',dialogHook);for(const [app,{original,wrapped}]of dialogs)if(app.resolve===wrapped)app.resolve=original;dialogs.clear();};
 try{
  nativeTask=Promise.resolve().then(()=>{if(error)throw error;return TextEditor._onClickInlineRoll({...nativeRollEvent(game,'damage'),target:anchor});});
  nativeTask.then(()=>nativeSettled=true,()=>nativeSettled=true);
  await (aborted?Promise.race([nativeTask,aborted]):nativeTask);
  if(error)throw error;
  assertLive?.();
  if(captured)Object.assign(captured.options,roll.options);
  return captured??null;
 }catch(caught){error??=caught;throw caught;}
 finally{
  for(const [event,id]of listeners)Hooks.off(event,id);
  // Preparing a native dialog can outlive an aborted owner operation. Keep
  // only its exact nonce hooks until it resolves, so a late window remains
  // cancelled even if the old owner's role later becomes valid again.
  if(error&&nativeTask&&!nativeSettled)void nativeTask.finally(cleanup).catch(()=>{});else cleanup();
 }
}
