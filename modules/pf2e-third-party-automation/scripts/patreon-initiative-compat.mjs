const PATREON='patreon-v3',BATTLE_CRY='Compendium.pf2e.feats-srd.Item.ePObIpaJDgDb9CQj';
const SOURCE_SHA256='dc85cbea3a111b832e66dea58bdbfe1a591ef34c9975b4730eae9ecffd13970d';
const HANDLER_SOURCE='async function hn(a){R.createChatMessage.forEach(e=>{e.listen(a)})}';
const FA_FILTER_SOURCE='o=>En.includes(o.sourceId)',EVENTS=['createChatMessage','patreon-v3.processMessage'];
const installedByHooks=new WeakMap();
const functionSource=fn=>typeof fn==='function'?Function.prototype.toString.call(fn):null;
const defaultFetch=async()=>{const path=globalThis.foundry?.utils?.getRoute?.('modules/patreon-v3/src/index.js')??'/modules/patreon-v3/src/index.js';const response=await fetch(path,{cache:'no-store'});if(!response.ok)throw Error('Patreon source fetch failed');return response.text();};
const defaultHash=async source=>Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(source))),b=>b.toString(16).padStart(2,'0')).join('');

// A separate target is required: Foundry defines actor.items as a non-configurable,
// readonly own property, whose value a Proxy over the actual Actor cannot replace.
function readonlyView(real,overrides){
 const bound=new Map();
 return new Proxy(Object.create(Object.getPrototypeOf(real)),{
  get(_target,key){
   if(Object.hasOwn(overrides,key))return overrides[key];
   const value=Reflect.get(real,key,real);if(typeof value!=='function')return value;
   if(key==='constructor')return value;
   if(!bound.has(value))bound.set(value,value.bind(real));return bound.get(value);
  },
  has:(_target,key)=>Reflect.has(real,key),ownKeys:()=>Reflect.ownKeys(real),
  getOwnPropertyDescriptor(_target,key){const descriptor=Reflect.getOwnPropertyDescriptor(real,key);return descriptor?{...descriptor,configurable:true}:undefined;},
  set:()=>false,defineProperty:()=>false,deleteProperty:()=>false
 });
}

function nativePair(Hooks){
 const custom=Hooks.events?.['patreon-v3.processMessage'];if(!Array.isArray(custom)||custom.length!==1)return null;
 const original=custom[0];if(!original||typeof original!=='object')return null;
 const descriptor=Object.getOwnPropertyDescriptor(original,'fn');if(descriptor?.writable!==true)return null;
 const fn=descriptor.value;
 if(original.once||fn?.name!=='processMessage'||functionSource(fn)!==HANDLER_SOURCE)return null;
 const chatEvents=Hooks.events?.createChatMessage;if(!Array.isArray(chatEvents))return null;
 const chat=chatEvents.filter(entry=>entry?.fn===fn);
 if(chat?.length!==1||chat[0].once)return null;
 const registrations=[chat[0],original].map((entry,index)=>({event:EVENTS[index],entry,id:entry.id,index:Hooks.events[EVENTS[index]].indexOf(entry)}));
 if(registrations.some(({event,entry,index})=>entry.hook!==event||!Number.isSafeInteger(entry.id)||entry.id<1||entry.once!==false||Hooks.events[event][index]!==entry||Object.getOwnPropertyDescriptor(entry,'fn')?.writable!==true))return null;
 if(chat[0].id===original.id)return null;
 return {fn,registrations};
}

/** Decorate the exact paired public hook entry; never replace a private Patreon handler. */
export async function registerPatreonInitiativeCompatibility({game,Hooks,isProviderReady,fetchSource=defaultFetch,hashSource=defaultHash}={}){
 if(installedByHooks.has(Hooks))return installedByHooks.get(Hooks);
 const unsupported=reason=>({status:'unsupported',reason,dispose(){}});
 if(!game?.modules?.get(PATREON)?.active)return unsupported('Patreon is inactive');
 if(game.modules.get(PATREON).version!=='3.2.28')return unsupported('Unknown Patreon version');
 if(game.release?.generation!==14)return unsupported('Unknown Foundry hook profile');
 let sourceHash;try{sourceHash=await hashSource(await fetchSource());}catch(error){return unsupported(String(error.message??error));}
 if(sourceHash!==SOURCE_SHA256)return unsupported('Unknown Patreon source SHA256');
 // Another concurrent registration may have finished while the source was read.
 if(installedByHooks.has(Hooks))return installedByHooks.get(Hooks);
 const pair=nativePair(Hooks);if(!pair)return unsupported('Unknown or ambiguous Patreon hook signature');
 let active=true;
 const canHandle=(message,actor,gm,token)=>{
  if(!active||isProviderReady?.()!==true||!gm?.active||!gm.isGM||game.users.activeGM!==gm)return false;
  const context=message?.flags?.pf2e?.context;
  if(game.messages.get(message?.id)!==message||context?.type!=='initiative'||context.isReroll||!message.isCheckRoll||!message.rolls?.length||!message.isAuthor)return false;
  if(message.actor!==actor||actor?.type!=='character'||actor.isDead===true||actor.canAct===false||actor.hasCondition?.('unconscious'))return false;
  const author=message.author;if(!author||game.users.get(author.id)!==author||game.user.id!==author.id||!actor.testUserPermission?.(author,'OWNER'))return false;
  const statistic=actor.getStatistic?.('intimidation')??actor.skills?.intimidation;if((statistic?.rank??0)<3)return false;
  if(!actor.items?.filter(item=>item.sourceId===BATTLE_CRY).length)return false;
  const speaker=message.speaker,scene=game.scenes.get(speaker?.scene);
  return speaker?.actor===actor.id&&token?.id===speaker.token&&scene?.tokens.get(token.id)===token&&token.parent===scene&&token.actor===actor&&token.object?.document===token&&(!gm.viewedScene||gm.viewedScene===speaker.scene);
 };
 const wrapper=function(message,...args){
  const actor=message?.actor,gm=game.users.activeGM,token=game.scenes.get(message?.speaker?.scene)?.tokens.get(message?.speaker?.token);
  if(!canHandle(message,actor,gm,token))return pair.fn.call(this,message,...args);
  const items=actor.items,filter=items.filter;
  const itemsView=readonlyView(items,{filter:function(predicate,...rest){
   if(!canHandle(message,actor,gm,token)||functionSource(predicate)!==FA_FILTER_SOURCE)return filter.call(items,predicate,...rest);
   return filter.call(items,(item,...tail)=>item.sourceId!==BATTLE_CRY&&predicate(item,...tail),...rest);
  }});
  const actorView=readonlyView(actor,{items:itemsView}),messageView=readonlyView(message,{actor:actorView});
  // hn intentionally fire-and-forgets its internal listeners. Preserve that exact
  // return value; the per-call view remains reachable across their own awaits.
  return pair.fn.call(this,messageView,...args);
 };
 const {registrations}=pair;
 // Foundry V14 stores the same HookedFunction in events and its private ID
 // map. Change only that entry's writable callback, preserving IDs and order.
 if(registrations.some(({event,entry,index})=>Hooks.events[event]?.[index]!==entry||entry.fn!==pair.fn||Object.getOwnPropertyDescriptor(entry,'fn')?.writable!==true))return unsupported('Patreon hook changed before registration');
 for(const {entry}of registrations)entry.fn=wrapper;
 const result={status:'installed',sourceSHA256:sourceHash,version:'3.2.28',foundryVersion:game.version,hookEntries:registrations.map(({event,entry,index})=>({event,id:entry.id,index})),dispose(){
  active=false;
  for(const {event,entry,id}of registrations){
   // A later wrapper may retain ours. Deactivate the closed-over view, but do
   // not remove or overwrite that later registration.
   if(!Hooks.events[event]?.includes(entry)||entry.id!==id||entry.fn!==wrapper||Object.getOwnPropertyDescriptor(entry,'fn')?.writable!==true)continue;
   entry.fn=pair.fn;
  }
  if(installedByHooks.get(Hooks)===result)installedByHooks.delete(Hooks);
 }};
 installedByHooks.set(Hooks,result);return result;
}
