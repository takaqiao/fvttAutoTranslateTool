const SELECTOR='.chat-message[data-message-id] .message-timestamp';
const TARGET='foundry.applications.sidebar.tabs.ChatLog.prototype.updateTimestamps';
const installed=new WeakSet();

// Read on every refresh, never cached: an accessor another module puts on
// Node.prototype later is the one to delegate to, not one to route around.
function inheritedText(runtime){
  const prototype=runtime.Node?.prototype,descriptor=prototype&&Object.getOwnPropertyDescriptor(prototype,'textContent');
  if(typeof descriptor?.get==='function'&&typeof descriptor.set==='function')return {prototype,descriptor};
}
function textOwner(element){
  for(let p=element;p;p=Object.getPrototypeOf(p))if(Object.hasOwn(p,'textContent'))return p;
}

/**
 * Core reassigns every displayed timestamp on a timer, and an assignment of the
 * same label still replaces the Text node. The loop stays core's: while one
 * refresh runs, each stamp carries an own textContent accessor that drops equal
 * writes and forwards everything else to the inherited setter.
 */
export function installTimestampPatch({moduleId='av-v14-hotfix',runtime=globalThis,report=()=>{}}={}){
  const stats={skipped:0,written:0};
  const finish=status=>{const result={feature:'timestamps',status,stats};report(result);return result;};
  const prototype=runtime.foundry?.applications?.sidebar?.tabs?.ChatLog?.prototype;
  if(typeof runtime.libWrapper?.register!=='function'||typeof prototype?.updateTimestamps!=='function'
    ||typeof runtime.document?.querySelectorAll!=='function'||!inheritedText(runtime))return finish('unsupported-runtime');
  if(installed.has(prototype))return finish('already-installed');
  let asyncReported=false;
  // Stay synchronous: core binds this method for its timer at the first ChatLog
  // render, and a registration made after that may never be called.
  runtime.libWrapper.register(moduleId,TARGET,function(wrapped,...args){
    const guarded=[];let own;
    // libWrapper unregisters a WRAPPER that returns without calling wrapped, so
    // no failure while attaching may keep the refresh below from running.
    try{
      const inherited=inheritedText(runtime);
      if(inherited){
        const {get,set:write,enumerable}=inherited.descriptor;
        own={configurable:true,enumerable,get,set(value){
          let same=false;
          try{
            // An empty string removes the child natively, so it is never an equal write.
            if(typeof value==='string'&&value){const child=this.firstChild;same=!!child&&child===this.lastChild&&child.nodeType===3&&child.data===value;}
          }catch{/* Unreadable children are not a proven match. */}
          if(same){stats.skipped++;return;}
          stats.written++;Reflect.apply(write,this,[value]);
        }};
        for(const element of runtime.document.querySelectorAll(SELECTOR)){
          // An own textContent is another module's or an outer re-entrant call's;
          // an accessor layered over Node.prototype's is not ours to shadow.
          try{
            if(textOwner(element)!==inherited.prototype||!Object.isExtensible(element))continue;
            Object.defineProperty(element,'textContent',own);guarded.push(element);
          }catch{/* This stamp keeps its ordinary write. */}
        }
      }
    }catch{/* An unguarded refresh is still a correct one. */}
    try{
      const result=wrapped(...args);
      if(!asyncReported)try{
        // An async refresh assigns after the accessors are gone, so nothing is skipped.
        if(typeof result?.then==='function'){asyncReported=true;report({feature:'timestamps',status:'ineffective',reason:'async-refresh',stats});}
      }catch{/* Diagnostics never replace the refresh result. */}
      return result;
    }finally{
      for(const element of guarded)try{
        // A redefinition made during the refresh belongs to its author.
        const current=Object.getOwnPropertyDescriptor(element,'textContent');
        if(current?.get===own.get&&current.set===own.set)Reflect.deleteProperty(element,'textContent');
      }catch{/* Cleanup never replaces the refresh result or error. */}
    }
  },'WRAPPER');
  installed.add(prototype);
  return finish('installed');
}
