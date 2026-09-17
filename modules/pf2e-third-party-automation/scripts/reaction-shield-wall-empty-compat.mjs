// Local draft. Only protects the empty Shield Wall candidate branch in the
// three audited campaigns. It does not implement the Shield Wall reaction.
const MODULE='pf2e-reaction',EVENT='createItem';
const TARGETS=new Set(['-','sog','pnvfcgjbf2cjp7gz','team-automation-qa2']);
const BUNDLE_SHA256='4a81322796ce1c6ed545edc09e1aa3a96a9c8a96dfd034403bf657068ed7036c';
const CALLBACK_SHA256='6cb70e38abbdc59e441e56f085144eaa54f69763e8649e7ea942e32326ad4841';
// The 8.5.1 NPC sheet calls the unchanged native Raise a Shield macro. Both
// audited profiles retain the exact Reaction bundle/callback proof and empty
// candidate scope; an installed wrapper remains bound to its original profile.
const PROFILES=Object.freeze({
 '8.5.0':Object.freeze({coreGeneration:14,system:'8.5.0',reaction:'1.4.3',bundleSHA256:BUNDLE_SHA256,callbackSHA256:CALLBACK_SHA256}),
 '8.5.1':Object.freeze({coreGeneration:14,system:'8.5.1',reaction:'1.4.3',bundleSHA256:BUNDLE_SHA256,callbackSHA256:CALLBACK_SHA256}),
});
const installations=new WeakMap();
const sourceOf=fn=>typeof fn==='function'?Function.prototype.toString.call(fn):'';
const defaultHash=async source=>Array.from(new Uint8Array(await globalThis.crypto.subtle.digest('SHA-256',new TextEncoder().encode(source))),b=>b.toString(16).padStart(2,'0')).join('');
const defaultFetch=async()=>{
 const route=globalThis.foundry?.utils?.getRoute?.('modules/pf2e-reaction/pf2e-reaction.js')??'/modules/pf2e-reaction/pf2e-reaction.js';
 const response=await fetch(route,{cache:'no-store'});if(!response.ok)throw Error('Reaction Checker source unavailable');return response.text();
};
const unsupported=reason=>Object.freeze({status:'unsupported',reason,dispose(){}});

function emptyCandidateRaise(game,item,userId){
 if(userId!==game.userId||item?.type!=='effect'||item.slug!=='effect-raise-a-shield')return false;
 const actor=item.actor,turns=game.combat?.turns;
 if(!actor||!['character','npc','familiar'].includes(actor.type)||actor.alliance!=='party'||!Array.isArray(turns))return false;
 const source=turns.find(row=>row?.actorId===actor.id);
 if(!source||source.actor!==actor)return false;
 // Do not invent candidates from names, linked action UUIDs or cached party
 // membership. Any actual Shield Wall feat leaves the upstream path intact.
 return turns.every(row=>!row?.actor||Array.isArray(row.actor.itemTypes?.feat)&&!row.actor.itemTypes.feat.some(feat=>feat.slug==='shield-wall'));
}

export function registerReactionShieldWallEmptyCompatibility({game,Hooks,fetchSource=defaultFetch,hashSource=defaultHash}={}){
 if(!Hooks||!['object','function'].includes(typeof Hooks))return Promise.resolve(unsupported('missing-hooks'));
 const existing=installations.get(Hooks);if(existing)return existing.promise;
 const module=game?.modules?.get(MODULE),world=game?.world,worldId=world?.id,profile=PROFILES[game?.system?.version];
 const profileCurrent=()=>!!profile&&game?.modules?.get(MODULE)===module&&module?.active===true&&module.version===profile.reaction&&game.release?.generation===profile.coreGeneration&&game.system?.version===profile.system&&game.world===world&&world?.id===worldId&&TARGETS.has(worldId);
 if(!profileCurrent())return Promise.resolve(unsupported('unknown-dependency-or-world-profile'));
 const entries=Hooks.events?.[EVENT];
 if(!Array.isArray(entries))return Promise.resolve(unsupported('missing-create-item-hooks'));
 const matching=entries.filter(entry=>{const source=sourceOf(entry?.fn);return source.includes('"shield-wall"')&&source.includes('"effect-raise-a-shield"');});
 if(matching.length!==1)return Promise.resolve(unsupported('ambiguous-create-item-handler'));
 const entry=matching[0],original=entry.fn,id=entry.id,index=entries.indexOf(entry),callbackSource=sourceOf(original);
 const entryCurrent=()=>Hooks.events?.[EVENT]===entries&&entries[index]===entry&&entry.id===id&&entry.hook===EVENT&&entry.once===false&&Number.isSafeInteger(id)&&id>0&&entry.fn===original&&Object.getOwnPropertyDescriptor(entry,'fn')?.writable===true;
 if(!entryCurrent())return Promise.resolve(unsupported('unknown-hook-entry-profile'));
 const state={promise:null};
 // Reserve before the asynchronous source check. Repeated initializers share
 // that work; no source fetch, digest or function serialization occurs per event.
 installations.set(Hooks,state);
 state.promise=Promise.resolve().then(async()=>{
  try{
   const bundleHash=await hashSource(await fetchSource());
   if(bundleHash!==profile.bundleSHA256)return unsupported('unknown-reaction-bundle');
   if(await hashSource(callbackSource)!==profile.callbackSHA256)return unsupported('unknown-create-item-callback');
   if(!profileCurrent()||!entryCurrent())return unsupported('identity-changed-during-verification');
   let active=true;
   const wrapped=function(item,options,userId,...rest){
    if(active&&profileCurrent()&&Hooks.events?.[EVENT]?.includes(entry)&&entry.id===id&&emptyCandidateRaise(game,item,userId))return Promise.resolve();
    return Reflect.apply(original,this,[item,options,userId,...rest]);
   };
   entry.fn=wrapped;
   return Object.freeze({status:'installed',scope:'empty-shield-wall-candidates-only',sourceSHA256:bundleHash,callbackSHA256:profile.callbackSHA256,hook:{event:EVENT,id,index},dispose(){
    active=false;
    if(Hooks.events?.[EVENT]?.includes(entry)&&entry.id===id&&entry.fn===wrapped&&Object.getOwnPropertyDescriptor(entry,'fn')?.writable===true)entry.fn=original;
    if(installations.get(Hooks)===state)installations.delete(Hooks);
   }});
  }catch(error){return unsupported(String(error.message??error));}
 }).then(result=>{
  if(result.status==='unsupported'&&installations.get(Hooks)===state)installations.delete(Hooks);
  return result;
 });
 return state.promise;
}
