// Version-bound adapter for the original Patreon checkCall callback. It keeps
// the libWrapper entry and order, and never edits third-party files or settings.
// The private treatment scope proves the native context before this callback;
// the original core and all later wrappers retain their normal privacy rules.
export const PATREON_TREATMENT_PROFILE=Object.freeze({
 coreGeneration:14,system:'8.5.0',patreon:'3.2.28',libWrapper:'1.13.5.1',
 checkCallSHA256:'3d55438160d1c13a392029635e53392d1cbae3f596347372623e1457935e64a5',
 holderSHA256:Object.freeze({
  get_fn_data:'0d5c5cd558cd12e5ba4066e88373287cec96005c89388a6b27ee2cb23c2a6d03',
  clear_static_dispatch_chain_cache:'d623b99e2afdb9ae9f36ad0b9b3f68068c8c4e6150f3539e7b6681e7af97d702',
  get_static_dispatch_chain:'9f6ff954fef895b9d40818594cb361a7ee93b9a42c35d44420117d24aa86655b',
  call_wrapper:'487132e8435f496b84c9ab239c164e5ec21fc611391d78239af24ceb4259945a',
 }),
});
// Explicitly audited system profiles: 8.5.1 changes Check.roll only by moving
// the unchanged Treat Wounds callback import. It does not change this entry,
// checkCall or the four libWrapper methods, whose real source hashes stay required.
export const PATREON_TREATMENT_PROFILES=Object.freeze({
 '8.5.0':PATREON_TREATMENT_PROFILE,
 '8.5.1':Object.freeze({...PATREON_TREATMENT_PROFILE,system:'8.5.1'}),
});
const installations=new WeakMap();
const sha256=async source=>Array.from(new Uint8Array(await globalThis.crypto.subtle.digest('SHA-256',new TextEncoder().encode(source))),byte=>byte.toString(16).padStart(2,'0')).join('');
const sourceOf=fn=>typeof fn==='function'?Function.prototype.toString.call(fn):null;
const unavailable=reason=>Object.freeze({installed:false,reason,dispose:()=>false});
const entryState=entry=>({entry,fn:entry.fn,packageInfo:entry.package_info,packageId:entry.package_info?.id,target:entry.target,setter:entry.setter,type:entry.type,typeName:entry.type?.name,priority:entry.priority,chain:entry.chain,bind:entry.bind,bound:Array.isArray(entry.bind)?entry.bind.slice():null,wrapper:entry.wrapper});
const sameEntry=(entry,state)=>entry===state.entry&&entry.fn===state.fn&&entry.package_info===state.packageInfo&&entry.package_info?.id===state.packageId&&entry.target===state.target&&entry.setter===state.setter&&entry.type===state.type&&entry.type?.name===state.typeName&&entry.priority===state.priority&&entry.chain===state.chain&&entry.bind===state.bind&&entry.wrapper===state.wrapper&&(!state.bound||entry.bind.length===state.bound.length&&state.bound.every((value,index)=>entry.bind[index]===value));

export function wrapPatreonTreatmentCheck(original,scope){
 return function treatmentPatreonCheck(next,check,context,...rest){
  const authorization=(scope.acquirePatreonModeScope??scope.acquirePatreonPublicScope).call(scope,check,context);
  if(!authorization)return original.call(this,next,check,context,...rest);
  const args=[check,context,...rest],mode=context.messageMode;
  if(mode!==(authorization.mode??'public')||typeof authorization.revalidate!=='function')throw Error('Invalid treatment source-mode authorization');
  let entered=false;
  const continuation=function(...nextArgs){
   if(entered)throw Error('A treatment may enter the native continuation only once');
   entered=true;
   if(nextArgs.length!==args.length||nextArgs.some((value,index)=>value!==args[index]))throw Error('Patreon treatment continuation arguments changed');
   // The exact checkCall build forces gm for a GM source, blind otherwise.
   // The private capability fixes the originally requested native audience;
   // its recheck rejects any hidden/secret change requiring a stricter mode.
   if(![mode,authorization.forcedMode??'blind'].includes(context.messageMode))throw Error('Unexpected private treatment mode');
   if(authorization.revalidate()!==true)throw Error('Treatment source-mode authorization changed');
   context.messageMode=mode;
   return next.call(this,...nextArgs);
  };
  return original.call(this,continuation,...args);
 };
}

/** Internal libWrapper data are used only after hashing the actual methods and
 * Patreon callback. Unknown versions/entry layouts are untouched. `profile`
 * and `hash` are injectable only to exercise these guards in isolated tests. */
export async function installPatreonTreatmentCompatibility({game,libWrapper=globalThis.libWrapper,scope,profile=PATREON_TREATMENT_PROFILES[game?.system?.version],hash=sha256}={}){
 const dependency=game?.modules?.get('patreon-v3');
 if(!dependency?.active)return unavailable('patreon-inactive');
 if(!profile||game.release?.generation!==profile.coreGeneration||game.system?.version!==profile.system||dependency.version!==profile.patreon||libWrapper?.version!==profile.libWrapper)return unavailable('unknown-dependency-version');
 if(typeof scope?.acquirePatreonPublicScope!=='function')return unavailable('missing-private-treatment-scope');
 const descriptor=Object.getOwnPropertyDescriptor(game.pf2e?.Check??{},'roll'),holder=descriptor?.get?._lib_wrapper;
 if(!holder||holder.name!=='game.pf2e.Check.roll'||holder.is_property!==false||holder.active!==true||holder._outstanding_wrappers!==0||!Array.isArray(holder.getter_data))return unavailable('unsafe-wrapper-holder');
 const already=installations.get(holder);
 if(already)return already.scope===scope&&already.entry.fn===already.wrapped?already.api:unavailable('wrapper-already-changed');
 const entries=holder.getter_data,entryStates=entries.map(entryState),candidates=entries.filter(entry=>entry.package_info?.id==='patreon-v3');
 if(candidates.length!==1)return unavailable('nonunique-patreon-callback');
 const entry=candidates[0],original=entry.fn,fnDescriptor=Object.getOwnPropertyDescriptor(entry,'fn');
 if(entry.target!=='game.pf2e.Check.roll'||entry.setter!==false||entry.type?.name!=='WRAPPER'||entry.chain!==true||entry.bind!==null&&(!Array.isArray(entry.bind)||entry.bind.length)||original?.name!=='checkCall'||!fnDescriptor?.writable)return unavailable('unknown-patreon-entry-shape');
 const methods=Object.keys(profile.holderSHA256),functions=Object.fromEntries(methods.map(name=>[name,holder[name]]));
 if(methods.some(name=>typeof functions[name]!=='function'))return unavailable('missing-wrapper-method');
 const expected=[profile.checkCallSHA256,...methods.map(name=>profile.holderSHA256[name])];
 const sources=[sourceOf(original),...methods.map(name=>sourceOf(functions[name]))];
 const actual=await Promise.all(sources.map(source=>hash(source)));
 if(actual.some((value,index)=>value!==expected[index]))return unavailable('unknown-callback-source');
 const concurrent=installations.get(holder);
 if(concurrent)return concurrent.scope===scope&&concurrent.entry.fn===concurrent.wrapped?concurrent.api:unavailable('wrapper-already-changed');
 // Hashing is asynchronous. Never install against a stale entry or a changed
 // holder, and do not mutate any in-progress dispatch chain.
 if(Object.getOwnPropertyDescriptor(game.pf2e.Check,'roll')?.get!==descriptor.get||holder.name!=='game.pf2e.Check.roll'||holder.active!==true||holder.is_property!==false||holder._outstanding_wrappers!==0||holder.getter_data!==entries||holder.get_fn_data(false)!==entries||entries.length!==entryStates.length||entries.some((value,index)=>!sameEntry(value,entryStates[index]))||!entries.includes(entry)||methods.some(name=>holder[name]!==functions[name]))return unavailable('wrapper-changed-during-verification');
 const wrapped=wrapPatreonTreatmentCheck(original,scope);
 const api=Object.freeze({installed:true,dependency:{patreon:profile.patreon,libWrapper:profile.libWrapper,callbackSHA256:profile.checkCallSHA256},dispose(){
  if(entry.fn!==wrapped||holder._outstanding_wrappers!==0)return false;
  entry.fn=original;holder.clear_static_dispatch_chain_cache();installations.delete(holder);return true;
 }});
 entry.fn=wrapped;
 try{holder.clear_static_dispatch_chain_cache();}
 catch(error){entry.fn=original;throw error;}
 installations.set(holder,{scope,entry,original,wrapped,api});
 return api;
}
