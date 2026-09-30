import {sameSalubriousPrivacy} from './salubrious-privacy.mjs';
import {MODULE_ID} from './rules.mjs';
import {SerialActions} from './runtime.mjs';
import {WAVE_SOURCE,buildAvWaveRepair,hasAvWaveEnergy} from './av-wave-repair.mjs';
const source=item=>item?.sourceId??item?._stats?.compendiumSource??item?.flags?.core?.sourceId;
const values=collection=>Array.from(collection?.values?.()??collection??[]);
const waveOf=actor=>values(actor?.items).find(item=>source(item)===WAVE_SOURCE);
const get=(object,path)=>Object.hasOwn(object??{},path)?object[path]:path.split('.').reduce((value,key)=>value?.[key],object);
const clone=value=>globalThis.foundry?.utils?.deepClone?.(value)??structuredClone(value);
const receiptKeys=['nonce','actorUuid','itemUuid','userId','before','after','tokenUuid','startedAt'];
const same=(a,b)=>!!a&&!!b&&receiptKeys.every(key=>a[key]===b[key])&&sameSalubriousPrivacy(a.privacy,b.privacy);
const focusPath='system.resources.focus.value',intentPath=`flags.${MODULE_ID}.avRefocusIntent`;

/** Observe the actual public Workbench Refocus call, not arbitrary resource
 * increases. The nonce rides on the one native update started by that call. */
export function registerAvRefocusEvents({game,Hooks,libWrapper,registerActorUpdate,canvas=globalThis.canvas,onError=console.error,runExclusive,actorMatchers=[],onRefocus,refocusPrivacy}={}){
 const scopes=new Map(),queue=new SerialActions();
 const run=runExclusive??((actor,fn)=>queue.run(actor.uuid,fn));
 const activeGM=()=>!!game.user?.id&&game.user.id===game.users.activeGM?.id;
 const owner=(actor,user)=>actor?.type==='character'&&!!user&&actor.testUserPermission?.(user,'OWNER');
 const matched=actor=>actorMatchers.some(test=>test(actor));
 const eventRecords=actor=>actor.flags?.[MODULE_ID]?.refocusEvents??[];
 const saveEvent=(actor,proof,state)=>actor.update({[`flags.${MODULE_ID}.refocusEvents`]:[...eventRecords(actor).filter(r=>r.nonce!==proof.nonce).filter((r,index,list)=>r.state!=='done'||index>=list.length-99),{...proof,state}]});
 const hook=Hooks.on('updateActor',(actor,changes,options={},userId)=>{
  if(!activeGM())return;
  const proof=options?.[MODULE_ID]?.refocusReceipt,committed=clone(actor.flags?.[MODULE_ID]?.avRefocusIntent??null),user=game.users.get(userId),wave=waveOf(actor);
  if(!proof||typeof proof.nonce!=='string'||!proof.nonce||proof.userId!==userId||!owner(actor,user)||proof.actorUuid!==actor.uuid||proof.itemUuid!==(wave?.uuid??null)||!wave&&!matched(actor))return;
  if(get(changes,intentPath)?.nonce!==proof.nonce||!same(proof,committed)||!Number.isFinite(proof.before)||!Number.isFinite(proof.after)||proof.before<0||proof.after<proof.before)return;
  const focusChange=get(changes,focusPath),committedFocus=get(actor,focusPath);
  if(committedFocus!==proof.after||focusChange!==undefined&&focusChange!==proof.after)return;
  return run(actor,async()=>{
   if(!activeGM()||!owner(actor,user)||(waveOf(actor)?.uuid??null)!==(wave?.uuid??null))return;
   const notify=typeof onRefocus==='function'&&matched(actor);
   if(notify){if(eventRecords(actor).some(r=>r.nonce===proof.nonce))return;await saveEvent(actor,proof,'claimed');}
   if(wave){const receipts=wave.flags?.[MODULE_ID]?.avRefocusReceipts??[];
   if(!receipts.some(receipt=>receipt.nonce===proof.nonce)){
   // Claim before any awaited rule change: replay or a new active GM cannot
   // erase a later spell's choice with the same Refocus event.
   await wave.update({[`flags.${MODULE_ID}.avRefocusReceipts`]:[...receipts,{nonce:proof.nonce,userId,state:'claimed'}]});
   const repair=buildAvWaveRepair(wave);if(repair)await wave.update(repair);
   await actor.toggleRollOption('all','conservation-of-energy',wave.id,true,'none');
   if(!hasAvWaveEnergy(actor,'none'))throw Error('未能将热流谐摆恢复中性；本次再聚能回执不会重复处理。');
   await wave.update({[`flags.${MODULE_ID}.avRefocusReceipts`]:(wave.flags?.[MODULE_ID]?.avRefocusReceipts??[]).map(receipt=>receipt.nonce===proof.nonce?{...receipt,state:'done'}:receipt)});
   }}
   return notify?{actor,user,proof:clone(proof)}:null;
  }).then(async event=>{
   if(!event)return;
   // A target/DC choice must not hold the existing resource queue. Subscribers
   // persist their own treatment stages before invoking any dice or HP writes.
   try{if(!activeGM())throw Error('Refocus leader changed before subscriber');if(event.proof.privacy){if(!refocusPrivacy)throw Error('Missing exact Refocus note observer');event.proof.noteId=await refocusPrivacy.waitRefocusNote(event);if(!activeGM())throw Error('Refocus leader changed during native note');}await onRefocus(event);await run(actor,()=>{if(!activeGM())throw Error('Refocus leader changed after subscriber');return saveEvent(actor,proof,'done');});}
   catch(error){if(activeGM())await run(actor,()=>saveEvent(actor,proof,'uncertain'));throw error;}
  }).catch(onError);
 });
 const paths=[];let unregisterActorUpdate;
 if(libWrapper&&typeof game.PF2eWorkbench?.refocus==='function'){
  const register=(path,fn)=>{libWrapper.register(MODULE_ID,path,fn,'WRAPPER');paths.push(path);};
  register('game.PF2eWorkbench.refocus',async function(wrapped,...args){
   const controlled=canvas?.tokens?.controlled??[],actors=args[0]??controlled.map(token=>token.actor),actor=controlled[0]?.actor,wave=waveOf(actor);
   // Workbench Ec checks its argument count but operates controlled[0].actor.
   if(controlled.length!==1||actors?.length!==1||!wave&&!matched(actor)||!owner(actor,game.user))return wrapped(...args);
   if(scopes.has(actor.uuid))throw Error('该角色有尚未确认的原生再聚能。');
   const privacyHandle=refocusPrivacy&&matched(actor)?refocusPrivacy.beginRefocus({actor,token:controlled[0].document,user:game.user}):null;
   const scope={actor,wave,privacyHandle,userId:game.user.id,tokenUuid:controlled[0].document?.uuid??null,startedAt:game.time?.worldTime};scopes.set(actor.uuid,scope);
   try{const result=await wrapped(...args);if(scope.updateTask)await scope.updateTask;if(privacyHandle)await refocusPrivacy.finishRefocus(privacyHandle);return result;}
   catch(error){if(privacyHandle)refocusPrivacy.abortRefocus(privacyHandle,error);throw error;}
   finally{if(scopes.get(actor.uuid)===scope)scopes.delete(actor.uuid);}
  });
  const observeActorUpdate=function(wrapped,changes,options={}){
   const scope=scopes.get(this.uuid);
   if(!scope||scope.used||scope.actor!==this||Object.keys(changes??{}).length!==1||!Object.hasOwn(changes,focusPath))return wrapped(changes,options);
   scope.used=true;if(!scope.privacyHandle)scopes.delete(this.uuid);
   const before=this.system?.resources?.focus?.value,after=changes[focusPath],max=this.system?.resources?.focus?.max;
   if(!Number.isFinite(before)||!Number.isFinite(after)||after<before||after>max)return wrapped(changes,options);
   const proof={nonce:globalThis.foundry?.utils?.randomID?.(32)??globalThis.crypto.randomUUID(),actorUuid:this.uuid,itemUuid:scope.wave?.uuid??null,userId:scope.userId,before,after,tokenUuid:scope.tokenUuid,startedAt:scope.startedAt,...scope.privacyHandle?{privacy:structuredClone(scope.privacyHandle.privacy)}:{}};
   // A changed intent also makes a full-focus Refocus an acknowledged update,
   // rather than an empty diff that Foundry may discard without updateActor.
   const task=wrapped({...changes,[intentPath]:proof},{...options,[MODULE_ID]:{...options?.[MODULE_ID],refocusReceipt:proof}});
   scope.updateTask=scope.privacyHandle?refocusPrivacy.bindRefocusUpdate(scope.privacyHandle,proof,task):Promise.resolve(task);return scope.updateTask;
  };
  if(registerActorUpdate)unregisterActorUpdate=registerActorUpdate(observeActorUpdate);
  else register('CONFIG.Actor.documentClass.prototype.update',observeActorUpdate);
 }
 return()=>{Hooks.off('updateActor',hook);unregisterActorUpdate?.();for(const path of paths)libWrapper.unregister(MODULE_ID,path);scopes.clear();};
}
