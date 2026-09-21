import {MODULE_ID} from './rules.mjs';
import {salubriousFeat} from './salubrious-kiss-rules.mjs';
import {assertSource,currentToken,marker} from './salubrious-kiss-context.mjs';
import {captureSalubriousPrivacy,validateSalubriousPrivacy,sameSalubriousPrivacy,salubriousPrivacyData,mergeSalubriousAudience,treatmentPrivacyForPatient} from './salubrious-privacy.mjs';
const verifiedProfiles=new WeakSet();
const hashText=async text=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(text))),n=>n.toString(16).padStart(2,'0')).join('');
export const SALUBRIOUS_WORKBENCH_PROFILE=Object.freeze({coreGeneration:14,system:'8.5.0',workbench:'7.7.5',sourceSHA256:'8ba67f06ed216b024a866861dcdd64da1bec3f4210367994156bcb2c9b2092cb',refocusSHA256:'9ee3738b87618ffba2ecb47b5e787d76b06bd018f34710a626c4e4392f7e2080'});
// 8.5.1's native treatment callback is unchanged; its numeric-DC Check path
// retains the exact boundary used here. Both profiles still hash Workbench's
// complete bundle and actual Ec function before granting a private capability.
export const SALUBRIOUS_WORKBENCH_PROFILES=Object.freeze({
 '8.5.0':SALUBRIOUS_WORKBENCH_PROFILE,
 '8.5.1':Object.freeze({...SALUBRIOUS_WORKBENCH_PROFILE,system:'8.5.1'}),
});
export async function verifySalubriousWorkbench({game,source,hash=hashText}){
 const p=SALUBRIOUS_WORKBENCH_PROFILES[game.system?.version],dependency=game.modules.get('xdy-pf2e-workbench'),fn=game.PF2eWorkbench?.refocus;
 const current=()=>game.modules.get('xdy-pf2e-workbench')===dependency&&dependency?.active&&dependency.version===p.workbench&&game.release?.generation===p.coreGeneration&&game.system?.version===p.system&&game.PF2eWorkbench?.refocus===fn;
 if(!p||!current()||typeof fn!=='function'||typeof source!=='string')return Object.freeze({ready:false,reason:'unknown-workbench-profile'});
 const hashes=await Promise.all([hash(source),hash(Function.prototype.toString.call(fn))]);
 if(!current()||hashes[0]!==p.sourceSHA256||hashes[1]!==p.refocusSHA256)return Object.freeze({ready:false,reason:'unknown-or-changed-workbench-source'});
 const result=Object.freeze({ready:true,profile:p});verifiedProfiles.add(result);return result;
}
/** Optional startup verification must not download an absent dependency or
 * indefinitely hold up every unrelated provider while waiting for its body. */
export async function loadSalubriousWorkbench({game,fetch=globalThis.fetch,timeoutMs=10000}){
 const profile=SALUBRIOUS_WORKBENCH_PROFILES[game.system?.version],dependency=game.modules.get('xdy-pf2e-workbench');
 if(!dependency?.active)return {ready:false,reason:'workbench-inactive'};
 if(!profile||dependency.version!==profile.workbench||game.release?.generation!==profile.coreGeneration||typeof game.PF2eWorkbench?.refocus!=='function')return {ready:false,reason:'unknown-workbench-profile'};
 const controller=new AbortController();let timer;
 try{
  const timeout=new Promise((_,reject)=>{timer=setTimeout(()=>{controller.abort();reject(Error('workbench-verification-timeout'));},timeoutMs)});
  const source=await Promise.race([Promise.resolve().then(async()=>{
   const response=await fetch('modules/xdy-pf2e-workbench/xdy-pf2e-workbench.js',{signal:controller.signal});
   if(!response.ok)throw Error('workbench-source-unavailable');return response.text();
  }),timeout]);
  if(game.modules.get('xdy-pf2e-workbench')!==dependency)return {ready:false,reason:'workbench-dependency-changed'};
  return await verifySalubriousWorkbench({game,source});
 }catch(error){return {ready:false,reason:controller.signal.aborted?'workbench-verification-timeout':String(error.message??error)};}
 finally{clearTimeout(timer);}
}
const author=m=>m?.author?.id??m?.author??m?.user?.id??m?.user;
const sourceToken=m=>`Scene.${m?.speaker?.scene}.Token.${m?.speaker?.token}`;
const receiptKeys=['nonce','actorUuid','itemUuid','userId','before','after','tokenUuid','startedAt'];
const sameRefocus=(a,b)=>a&&b&&receiptKeys.every(k=>a[k]===b[k])&&sameSalubriousPrivacy(a.privacy,b.privacy);
const fail=message=>{throw Error(`Salubrious message privacy: ${message}`)};
const waitable=()=>{let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b});promise.catch(()=>{});return {promise,resolve,reject}};

/** Used only by the existing Refocus/Actor.update and ChatMessage.create
 * wrappers, plus the private damage guard. No new global wrapper or scan.
 * The native Ec calls detached Tc: retain the source scope until its exact
 * update THEN exact completion-note creation have both committed. */
export function createSalubriousMessagePrivacy({game,Hooks=globalThis.Hooks,constants=globalThis.CONST,timeoutMs=60000}={}){
 const sources=new Map(),handles=new WeakMap(),notes=new Map(),waiters=new Map(),applications=new Map();let workbenchReady=false;
 const noteFlag=data=>data?.flags?.[MODULE_ID]?.avRefocusNote;
 const noteValid=(message,proof)=>{try{return typeof proof?.actorUuid==='string'&&game.messages.get(message?.id)===message&&author(message)===proof.userId&&message.speaker?.actor===proof.actorUuid.split('.').at(-1)&&sourceToken(message)===proof.tokenUuid&&sameRefocus(noteFlag(message),proof)&&noteFlag(message).kind==='completion'&&JSON.stringify({blind:!!message.blind,whisper:[...(message.whisper??[])].sort()})===JSON.stringify(salubriousPrivacyData({userId:proof.userId,privacy:proof.privacy}));}catch{return false}};
 const hook=Hooks?.on?.('createChatMessage',message=>{
  const proof=noteFlag(message);if(!proof?.nonce||!noteValid(message,proof))return;
  const waiter=waiters.get(proof.nonce);if(waiter){if(!sameRefocus(waiter.proof,proof))waiter.reject(Error('Refocus note identity changed'));else waiter.resolve(message);}
  else {notes.set(proof.nonce,message);if(notes.size>128)notes.delete(notes.keys().next().value);}
 });
 function enableWorkbench(result){if(!verifiedProfiles.has(result)||!result.ready)fail('unverified Workbench capability');workbenchReady=true;}
 function beginRefocus({actor,token,user=game.user}){
  if(!workbenchReady)fail('Workbench note source is not verified');
  if(sources.has(actor.uuid))fail('the previous native Refocus has not settled');
  const item=salubriousFeat(actor),privacy=captureSalubriousPrivacy({game,user,token,item});assertSource({game,actor,item,token,user,privacy});
  const scope={actor,item,token,user,privacy,completion:waitable(),committed:false,entered:false};
  const handle=Object.freeze({privacy:structuredClone(privacy)});handles.set(handle,scope);sources.set(actor.uuid,scope);return handle;
 }
 function bindRefocusUpdate(handle,proof,task){
  const scope=handles.get(handle);if(!scope||scope.proof||proof.actorUuid!==scope.actor.uuid||proof.userId!==scope.user.id||proof.tokenUuid!==scope.token.uuid||!sameSalubriousPrivacy(proof.privacy,scope.privacy))fail('wrong native Refocus update');
  scope.proof=structuredClone(proof);
  return Promise.resolve(task).then(result=>{
   if(result!==scope.actor||scope.actor.system.resources.focus.value!==proof.after||!sameRefocus(scope.actor.flags?.[MODULE_ID]?.avRefocusIntent,proof))fail('native focus update did not commit');
   scope.committed=true;return result;
  }).catch(error=>{
   // The verified Workbench Ec discards Tc's Promise. Relay the failure to the
   // awaited outer Refocus instead, and retain this scope until Tc's following
   // exact note attempt is blocked. Ordinary Actor.update is never intercepted.
   scope.updateRejected=true;scope.completion.reject(error);return undefined;
  });
 }
 function abortRefocus(handle,error){const scope=handles.get(handle);if(!scope)return;handles.delete(handle);scope.completion.reject(error);if(!scope.proof){if(sources.get(scope.actor.uuid)===scope)sources.delete(scope.actor.uuid);}}
 async function finishRefocus(handle){
  const scope=handles.get(handle);if(!scope?.proof)fail('no native focus update was observed');
  let timer;try{return await Promise.race([scope.completion.promise,new Promise((_,reject)=>{timer=setTimeout(()=>reject(Error('Native Refocus note is uncertain; no treatment retry')),timeoutMs)})]);}
  // On timeout keep its minimal exact source scope: a delayed native note may
  // still arrive and must not become public or authorize a newer Refocus.
  finally{clearTimeout(timer);handles.delete(handle)}
 }
 async function waitRefocusNote({proof}){
  if(!proof.privacy)return null;const cached=notes.get(proof.nonce);if(cached){if(!noteValid(cached,proof))fail('wrong committed Refocus note');notes.delete(proof.nonce);return cached.id;}
  if(waiters.has(proof.nonce))return waiters.get(proof.nonce).promise.then(m=>m.id);
  const waiter={...waitable(),proof:structuredClone(proof)};waiters.set(proof.nonce,waiter);let timer;
  try{const message=await Promise.race([waiter.promise,new Promise((_,reject)=>{timer=setTimeout(()=>reject(Error('Exact native Refocus note did not synchronize')),timeoutMs)})]);if(!noteValid(message,proof))fail('Refocus note identity changed');return message.id;}
  finally{clearTimeout(timer);if(waiters.get(proof.nonce)===waiter)waiters.delete(proof.nonce)}
 }
 function validateRefocusNote({proof}){return !proof.privacy||noteValid(game.messages.get(proof.noteId),proof);}
 function protect(data,args,privacy){
  if(args[0]?.messageMode!==undefined)fail('unexpected native messageMode override');
  const actual={...data,...salubriousPrivacyData({userId:privacy.userId,privacy})};
  return actual;
 }
 async function createMessageMiddleware(wrapped,data,...args){
  // Exact original native completion structure and a committed live source,
  // never the last card, a translated title, or a generic actor update.
  const source=sources.get(`Actor.${data?.speaker?.actor}`);
  if(source&&data?.style===constants?.CHAT_MESSAGE_STYLES?.EMOTE&&data.flavor===`<strong><img src="systems/${game.system.id}/icons/actions/Passive.webp" width="10" height="10" style="border: 0; margin-right: 3px;" alt="Passive">Refocus</strong>`){
   const scope=source,proof=scope.proof;
   try{
    if(!proof||!scope.committed||scope.entered||sourceToken(data)!==scope.token.uuid||author(data)&&author(data)!==scope.user.id||game.user!==scope.user||data.content!==game.i18n.format('xdy-pf2e-workbench.macros.refocus.regains',{focus:proof.after-proof.before}))fail('native Refocus note does not match the active source');
    scope.entered=true;
    assertSource({game,actor:scope.actor,item:scope.item,token:scope.token,user:scope.user,privacy:scope.privacy});
    const actual=protect(data,args,scope.privacy);actual.flags={...actual.flags,[MODULE_ID]:{...actual.flags?.[MODULE_ID],avRefocusNote:{...structuredClone(proof),kind:'completion'}}};
    const message=await wrapped(actual,...args);if(!noteValid(message,proof))fail('native completion note did not persist');scope.completion.resolve(message);return message;
   // Only the exact detached Tc branch terminates with null. Its failure is
   // already owned by finishRefocus; rethrowing here would create a second,
   // unhandled Promise. All unrelated create calls still reject normally.
   }catch(error){scope.completion.reject(error);return null}
   finally{if(sources.get(scope.actor.uuid)===scope)sources.delete(scope.actor.uuid)}
  }
  const rawOptions=data?.flags?.pf2e?.context?.options,options=Array.isArray(rawOptions)?rawOptions:rawOptions instanceof Set?[...rawOptions]:[],matches=options.filter(o=>applications.has(o));
  if(matches.length>1)fail('conflicting native application scopes');
  if(matches.length){
   const scope=applications.get(matches[0]),{request,claim}=scope,{target,item,message}=request;
   if(scope.entered||game.user!==scope.gm||game.users.activeGM!==scope.gm||!scope.gm.active||data.flags.pf2e.context.type!=='damage-taken'||sourceToken(data)!==target.uuid||data.speaker.actor!==target.actor.id||author(data)&&author(data)!==scope.gm.id||data.flags.pf2e.origin?.uuid!==item.uuid||data.flags.pf2e.origin?.actor!==claim.actorUuid||!options.includes(`${MODULE_ID}:source:${message.id}:0`)||!currentToken(target,game)||target.actor!==scope.targetActor)fail('wrong native treatment receipt');
   // Native applyDamage awaits HP writes before it creates the receipt. A token
   // or secret-damage setting may have become more private in that interval.
   // Intersect all three audiences; never widen a prior self/GM/blind receipt.
   const user=game.users.get(claim.userId),current=treatmentPrivacyForPatient({game,user,token:request.token,item,target,privacy:claim.privacy});
   validateSalubriousPrivacy({game,user,token:request.token,item,privacy:current});
   const sourceBound=mergeSalubriousAudience(current,salubriousPrivacyData(claim));
   scope.entered=true;const receiptPrivacy=mergeSalubriousAudience(sourceBound,data),actual=protect(data,args,receiptPrivacy);actual.flags={...actual.flags,pf2e:{...actual.flags.pf2e,context:{...actual.flags.pf2e.context,messageMode:receiptPrivacy.mode}},[MODULE_ID]:{...actual.flags?.[MODULE_ID],salubriousKiss:{kind:'receipt',nonce:claim.nonce,actorUuid:claim.actorUuid,userId:claim.userId,damageId:message.id,targetUuid:target.uuid,privacy:structuredClone(receiptPrivacy)}}};
   return wrapped(actual,...args);
  }
  return wrapped(data,...args);
 }
 async function withNativeApplication(request,claim,operation){
  if(!claim.privacy)return operation();const key=marker('apply',claim);if(applications.has(key))fail('duplicate application scope');
  const scope={request,claim:structuredClone(claim),gm:game.user,targetActor:request.target.actor,entered:false};applications.set(key,scope);
  try{const result=await operation();if(!scope.entered)fail('native private application receipt was not observed');return result;}
  finally{if(applications.get(key)===scope)applications.delete(key)}
 }
 return {enableWorkbench,beginRefocus,bindRefocusUpdate,finishRefocus,abortRefocus,waitRefocusNote,validateRefocusNote,createMessageMiddleware,withNativeApplication,dispose(){Hooks?.off?.('createChatMessage',hook);}};
}
