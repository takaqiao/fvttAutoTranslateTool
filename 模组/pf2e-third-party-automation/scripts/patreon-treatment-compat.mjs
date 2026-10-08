// Keep the installed callback and wrapper order; third-party versions are unrestricted.
export const PATREON_TREATMENT_PROFILE=Object.freeze({coreGeneration:14});
const installations=new WeakMap();
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

export async function installPatreonTreatmentCompatibility({game,libWrapper=globalThis.libWrapper,scope,profile=PATREON_TREATMENT_PROFILE}={}){
 const dependency=game?.modules?.get('patreon-v3');
 if(!dependency?.active)return unavailable('patreon-inactive');
 if(!profile||game.release?.generation!==profile.coreGeneration||game.system?.id!=='pf2e')return unavailable('unknown-system-interface');
 if(typeof scope?.acquirePatreonPublicScope!=='function')return unavailable('missing-private-treatment-scope');
 const descriptor=Object.getOwnPropertyDescriptor(game.pf2e?.Check??{},'roll'),holder=descriptor?.get?._lib_wrapper;
 if(!holder||holder.name!=='game.pf2e.Check.roll'||holder.is_property!==false||holder.active!==true||holder._outstanding_wrappers!==0||!Array.isArray(holder.getter_data))return unavailable('unsafe-wrapper-holder');
 const already=installations.get(holder);
 if(already)return already.scope===scope&&already.entry.fn===already.wrapped?already.api:unavailable('wrapper-already-changed');
 const entries=holder.getter_data,entryStates=entries.map(entryState),candidates=entries.filter(entry=>entry.package_info?.id==='patreon-v3');
 if(candidates.length!==1)return unavailable('nonunique-patreon-callback');
 const entry=candidates[0],original=entry.fn,fnDescriptor=Object.getOwnPropertyDescriptor(entry,'fn');
 if(entry.target!=='game.pf2e.Check.roll'||entry.setter!==false||entry.type?.name!=='WRAPPER'||entry.chain!==true||entry.bind!==null&&(!Array.isArray(entry.bind)||entry.bind.length)||original?.name!=='checkCall'||!fnDescriptor?.writable)return unavailable('unknown-patreon-entry-shape');
 const methods=['get_fn_data','clear_static_dispatch_chain_cache','get_static_dispatch_chain','call_wrapper'],functions=Object.fromEntries(methods.map(name=>[name,holder[name]]));
 if(methods.some(name=>typeof functions[name]!=='function'))return unavailable('missing-wrapper-method');
 const concurrent=installations.get(holder);
 if(concurrent)return concurrent.scope===scope&&concurrent.entry.fn===concurrent.wrapped?concurrent.api:unavailable('wrapper-already-changed');
 // Keep the live entry identity and leave any in-progress dispatch chain alone.
 if(Object.getOwnPropertyDescriptor(game.pf2e.Check,'roll')?.get!==descriptor.get||holder.name!=='game.pf2e.Check.roll'||holder.active!==true||holder.is_property!==false||holder._outstanding_wrappers!==0||holder.getter_data!==entries||holder.get_fn_data(false)!==entries||entries.length!==entryStates.length||entries.some((value,index)=>!sameEntry(value,entryStates[index]))||!entries.includes(entry)||methods.some(name=>holder[name]!==functions[name]))return unavailable('wrapper-changed-during-verification');
 const wrapped=wrapPatreonTreatmentCheck(original,scope);
 const api=Object.freeze({installed:true,dependency:{patreon:dependency.version,libWrapper:libWrapper?.version,callbackSHA256:null},dispose(){
  if(entry.fn!==wrapped||holder._outstanding_wrappers!==0)return false;
  entry.fn=original;holder.clear_static_dispatch_chain_cache();installations.delete(holder);return true;
 }});
 entry.fn=wrapped;
 try{holder.clear_static_dispatch_chain_cache();}
 catch(error){entry.fn=original;throw error;}
 installations.set(holder,{scope,entry,original,wrapped,api});
 return api;
}
