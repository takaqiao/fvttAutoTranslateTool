const instances=new WeakMap();
const values=c=>Array.from(c?.values?.()??c??[]);

/** One owner for actual collection/factory action variants, shared by consumers.
 * A returned native Use row proves neither completion nor freedom from disruption. */
export function getNativeActionEvents({game}={}){
 let events=instances.get(game);
 if(!events){events=createNativeActionEvents(game);instances.set(game,events)}
 return events;
}

function createNativeActionEvents(game){
 const middlewares=new Set();let installation;
 function cleanup(){
  const state=installation;if(!state)return;installation=null;state.active=false;
  for(const restore of state.restores.reverse())restore();
 }
 function addMiddleware(fn){
  if(typeof fn!=='function')throw TypeError('Native action middleware must be a function.');
  middlewares.add(fn);let subscribed=true;
  return()=>{if(!subscribed)return;subscribed=false;middlewares.delete(fn);if(!middlewares.size)cleanup()};
 }
 function register(){
  const state=installation??={active:true,actions:new WeakSet(),variants:new WeakMap(),restores:[]};
  const current=action=>state.active&&values(game.pf2e?.actions).includes(action);
  function patch(action,variant){
   if(!variant||typeof variant!=='object'||typeof variant.use!=='function'||state.variants.has(variant))return variant;
   const native=variant.use,descriptor=Object.getOwnPropertyDescriptor(variant,'use');
   state.variants.set(variant,action);
   const use=async function(params={}){
    if(this!==variant||!current(action)||params.message?.create===false||!middlewares.size)return native.call(this,params);
    const selected=params.actors?(Array.isArray(params.actors)?params.actors:[params.actors]):values(game.user.getActiveTokens?.()).map(t=>(t.document??t).actor);
    const actors=[...new Set(selected)];if(!actors.length&&game.user.character)actors.push(game.user.character);
    // Snapshot containers only. Foundry documents and the exact native input
    // remain live and unfrozen; consumers verify their own returned evidence.
    const input={...params};
    if(Array.isArray(params.actors))input.actors=Object.freeze([...params.actors]);
    if(params.message&&typeof params.message==='object')input.message=Object.freeze({...params.message});
    const scope=Object.freeze({action,variant,slug:action.slug,actors:Object.freeze(actors),params:Object.freeze(input),user:game.user});
    const handlers=[...middlewares];let closed=false;
    const invoke=async index=>{
     if(closed)throw Error('Native action continuation is closed.');
     if(index===handlers.length)return native.call(variant,params);
     let called=false;
     return handlers[index](scope,async()=>{
      if(closed||called)throw Error('Native action continuation can only run once and while open.');
      called=true;return invoke(index+1);
     });
    };
    try{return await invoke(0)}finally{closed=true}
   };
   Object.defineProperty(variant,'use',{configurable:true,writable:true,value:use});
   state.restores.push(()=>{if(variant.use===use){if(descriptor)Object.defineProperty(variant,'use',descriptor);else delete variant.use}});
   return variant;
  }
  for(const action of values(game.pf2e?.actions)){
   if(!action||typeof action.toActionVariant!=='function')continue;
   for(const variant of values(action.variants))patch(action,variant);
   if(state.actions.has(action))continue;state.actions.add(action);
   const original=action.toActionVariant,descriptor=Object.getOwnPropertyDescriptor(action,'toActionVariant');
   const factory=function(...args){const variant=original.apply(this,args);return this===action&&current(action)?patch(action,variant):variant};
   Object.defineProperty(action,'toActionVariant',{configurable:true,writable:true,value:factory});
   state.restores.push(()=>{if(action.toActionVariant===factory){if(descriptor)Object.defineProperty(action,'toActionVariant',descriptor);else delete action.toActionVariant}});
  }
  return cleanup;
 }
 return {addMiddleware,register,cleanup};
}
