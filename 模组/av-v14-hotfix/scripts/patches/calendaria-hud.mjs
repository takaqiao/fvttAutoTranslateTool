const TARGET='CALENDARIA.apps.HUD.prototype._onRender';
const EVENTS=['calendaria.visualTick','calendaria.worldTimeUpdated'];
const LABELS=[
  ['.time',[':scope > .center > .time-wrapper > span.time',':scope > .center > .slice > .slice-overlay > span.time']],
  ['.date',[':scope > .section.left > .info-panel > span.date']]
];
const installed=new WeakMap(),renderWrappers=new WeakMap(),functionToString=Function.prototype.toString;

function fingerprint(fn){
  if(typeof fn!=='function')return null;
  let hash=2166136261;
  for(const character of Reflect.apply(functionToString,fn,[]))hash=Math.imul(hash^character.charCodeAt(0),16777619);
  return (hash>>>0).toString(16).padStart(8,'0');
}
function sameDescriptor(left,right){
  return !!left&&!!right&&Reflect.ownKeys(left).length===Reflect.ownKeys(right).length
    &&Reflect.ownKeys(left).every(key=>left[key]===right[key]);
}
function sameDescriptors(object,descriptors){
  const current=Object.getOwnPropertyDescriptors(object),keys=Reflect.ownKeys(descriptors);
  return Reflect.ownKeys(current).length===keys.length&&keys.every(key=>sameDescriptor(current[key],descriptors[key]));
}
function originalRender(prototype){
  const descriptor=Object.getOwnPropertyDescriptor(prototype,'_onRender');
  if(descriptor?.value)return descriptor.value;
  // unregister keeps libWrapper's accessor in current builds. Only reuse the
  // accessor installed here, after both registration lists have become empty.
  const known=renderWrappers.get(prototype),chain=descriptor?.get?._lib_wrapper;
  if(!known||chain!==known.chain||!sameDescriptor(descriptor,known.descriptor)
    ||chain.get_fn_data!==known.getFnData||chain._wrapped!==known.original)return;
  const wrappers=chain.get_fn_data(false),listeners=chain.get_fn_data(false,true);
  if(Array.isArray(wrappers)&&!wrappers.length&&Array.isArray(listeners)&&!listeners.length)return chain._wrapped;
}
function knownHooks(Hooks){
  const events=Object.getOwnPropertyDescriptor(Hooks,'events');
  // Hooks.off may belong to native-hook-adapter. HUD closes by numeric ID, so
  // it needs no function alias and never changes any Hooks method.
  return fingerprint(Hooks)==='05559ab9'&&fingerprint(events?.get)==='d22fc400'&&!events.set
    &&fingerprint(Object.getOwnPropertyDescriptor(Hooks,'on')?.value)==='82df7097'
    &&fingerprint(Object.getOwnPropertyDescriptor(Hooks,'callAll')?.value)==='782db789';
}
function methodChain(prototype,name){
  const chain=[];
  for(let owner=prototype;owner;owner=Object.getPrototypeOf(owner)){
    const descriptor=Object.getOwnPropertyDescriptor(owner,name);
    if(descriptor)chain.push({owner,descriptor});
  }
  return chain;
}
function unchangedChain(prototype,name,chain){
  const current=methodChain(prototype,name);
  return current.length===chain.length&&current.every((entry,i)=>entry.owner===chain[i].owner&&sameDescriptor(entry.descriptor,chain[i].descriptor));
}
function snapshot(Hooks){
  const events=Hooks.events;
  if(events!==Hooks.events)return null;
  return {events,groups:EVENTS.map(hook=>{
    const descriptor=Object.getOwnPropertyDescriptor(events,hook),list=descriptor?.value;
    if(descriptor&&!Array.isArray(list))throw new TypeError('Unsupported hook list');
    return {hook,descriptor,list,entries:list?Array.from(list,entry=>({entry,descriptors:Object.getOwnPropertyDescriptors(entry)})):[]};
  })};
}
function newRecords(Hooks,before){
  if(Hooks.events!==before.events)return null;
  const records=[];
  for(const group of before.groups){
    const {hook,descriptor,entries}=group,current=Object.getOwnPropertyDescriptor(before.events,hook),list=current?.value;
    if(!Array.isArray(list)||descriptor&&!sameDescriptor(current,descriptor)||list.length!==entries.length+1)return null;
    if(entries.some(({entry,descriptors},i)=>list[i]!==entry||!sameDescriptors(entry,descriptors)))return null;
    const entry=list.at(-1),descriptors=Object.getOwnPropertyDescriptors(entry),fn=descriptors.fn;
    if(descriptors.hook?.value!==hook||!Number.isSafeInteger(descriptors.id?.value)||descriptors.id.value<=0
      ||descriptors.once?.value!==false||!fn?.writable||typeof fn.value!=='function')return null;
    records.push({entry,descriptor:fn,fn:fn.value});
  }
  return records;
}
function inheritedText(runtime){
  const prototype=runtime.Node?.prototype,descriptor=prototype&&Object.getOwnPropertyDescriptor(prototype,'textContent');
  if(typeof descriptor?.get==='function'&&typeof descriptor.set==='function')return {prototype,descriptor};
}
function textOwner(element){
  for(let owner=element;owner;owner=Object.getPrototypeOf(owner))if(Object.hasOwn(owner,'textContent'))return owner;
}

function withHudText(runtime,hud,callback,receiver,args,stats){
  const guarded=[];let own;
  try{
    const inherited=inheritedText(runtime),root=hud.element,bar=root?.querySelector('.bar');
    if(inherited&&bar){
      const {get,set:write,enumerable}=inherited.descriptor;
      own={configurable:true,enumerable,get,set(value){
        let same=false;
        try{
          if(typeof value==='string'&&value){const child=this.firstChild;same=!!child&&child===this.lastChild&&child.nodeType===3&&child.data===value;}
        }catch{/* An unreadable child is not a proven match. */}
        if(same){stats.skipped++;return;}
        stats.written++;Reflect.apply(write,this,[value]);
      }};
      for(const [selector,templates] of LABELS){
        // Match the same first result as Calendaria; a widget may insert another
        // .time or .date before the template's own span.
        const element=root.querySelector(selector);
        if(!element||!templates.some(template=>bar.querySelector(template)===element)
          ||textOwner(element)!==inherited.prototype||!Object.isExtensible(element))continue;
        try{Object.defineProperty(element,'textContent',own);guarded.push(element);}catch{/* Keep this node's ordinary setter. */}
      }
    }
  }catch{/* Selection failures must not prevent the native callback. */}
  try{return Reflect.apply(callback,receiver,args);}
  finally{
    for(const element of guarded)try{
      if(sameDescriptor(Object.getOwnPropertyDescriptor(element,'textContent'),own))Reflect.deleteProperty(element,'textContent');
    }catch{/* Cleanup must preserve the callback's result or exception. */}
  }
}

/** Install during setup, before HUD's first render; existing callbacks stay native. */
export function installCalendariaHudPatch({moduleId='av-v14-hotfix',runtime=globalThis,report=()=>{}}={}){
  const stats={adapted:0,skipped:0,written:0},reported=new Set();
  const emit=result=>{try{report(result);}catch{/* Diagnostics do not change HUD behavior. */}};
  const finish=(status,reason)=>{const result={feature:'calendariaHud',status,stats,restore(){}};if(reason)result.reason=reason;emit(result);return result;};
  const fallback=reason=>{if(!reported.has(reason)){reported.add(reason);emit({feature:'calendariaHud',status:'skipped',reason,stats});}};
  if(!runtime.game?.modules?.get('calendaria')?.active)return finish('skipped','module-inactive');
  const HUD=runtime.CALENDARIA?.apps?.HUD,prototype=HUD?.prototype,Hooks=runtime.Hooks,libWrapper=runtime.libWrapper;
  if(installed.has(prototype))return finish('already-installed');
  if(!prototype||!Hooks||typeof libWrapper?.register!=='function'||typeof libWrapper.unregister!=='function'
    ||typeof runtime.WeakRef!=='function'&&typeof WeakRef!=='function'||!inheritedText(runtime))return finish('unsupported-runtime');
  let original;
  try{original=originalRender(prototype);}catch{return finish('skipped','hud-source-mismatch');}
  if(fingerprint(HUD)!=='66ad2b08'||fingerprint(original)!=='ba49bfc8'
    ||fingerprint(Object.getOwnPropertyDescriptor(prototype,'_onClose')?.value)!=='59073946')return finish('skipped','hud-source-mismatch');
  if(!knownHooks(Hooks))return finish('skipped','core-hooks-fingerprint-mismatch');
  const hookDescriptors=Object.fromEntries(['events','on','callAll'].map(key=>[key,Object.getOwnPropertyDescriptor(Hooks,key)]));
  const parent=Object.getPrototypeOf(prototype),parents=methodChain(parent,'_onRender'),close=methodChain(prototype,'_onClose');
  const owned=new WeakMap(),instances=new WeakMap(),references=new Set(),WeakReference=runtime.WeakRef??WeakRef;
  let registration,renderDescriptor,active=true;
  function knownChain(hud){
    if(!active||runtime.CALENDARIA?.apps?.HUD!==HUD||runtime.Hooks!==Hooks||Object.getPrototypeOf(hud)!==prototype
      ||Object.hasOwn(hud,'_onRender')||Object.hasOwn(hud,'_onClose')||Object.getPrototypeOf(prototype)!==parent
      ||!sameDescriptor(Object.getOwnPropertyDescriptor(prototype,'_onRender'),renderDescriptor)
      ||!unchangedChain(parent,'_onRender',parents)||!unchangedChain(prototype,'_onClose',close)
      ||!Object.entries(hookDescriptors).every(([key,descriptor])=>sameDescriptor(Object.getOwnPropertyDescriptor(Hooks,key),descriptor)))return false;
    const chain=renderDescriptor.get?._lib_wrapper,wrappers=chain?.get_fn_data?.(false),listeners=chain?.get_fn_data?.(false,true);
    return chain?._wrapped===original&&Array.isArray(wrappers)&&wrappers.length===1&&wrappers[0].fn===onRender
      &&Array.isArray(listeners)&&listeners.length===0;
  }
  function restoreRecord(entry){
    const record=owned.get(entry);if(!record)return;
    try{if(sameDescriptor(Object.getOwnPropertyDescriptor(entry,'fn'),{...record.descriptor,value:record.replacement}))Object.defineProperty(entry,'fn',record.descriptor);}
    catch{/* Later frozen or replaced descriptors belong to their author. */}
    owned.delete(entry);
  }
  function onRender(wrapped,...args){
    let before;
    try{if(knownChain(this))before=snapshot(Hooks);else fallback('render-chain-changed');}catch{fallback('unsupported-hook-records');}
    const result=wrapped(...args);
    try{
      if(!before)return result;
      if(typeof result?.then==='function'){fallback('async-render');return result;}
      if(!knownChain(this)){fallback('render-chain-changed');return result;}
      const existing=instances.get(this);
      if(existing&&existing.every(reference=>{
        const entry=reference.deref(),record=entry&&owned.get(entry);
        return record&&Object.getOwnPropertyDescriptor(entry,'fn')?.value===record.replacement
          &&Hooks.events[entry.hook]?.includes(entry);
      }))return result;
      const records=newRecords(Hooks,before);
      if(!records){
        const unchanged=before.groups.every(group=>group.list?.length===group.entries.length);
        fallback(unchanged?'existing-unowned-hud-callbacks':'ambiguous-hud-registration');return result;
      }
      const hud=new WeakReference(this),refs=[];
      try{
        for(const record of records){
          record.replacement=function(...callbackArgs){
            const instance=hud.deref();let supported=false;
            try{supported=instance&&knownChain(instance);}catch{/* Unknown later chains remain native. */}
            return supported?withHudText(runtime,instance,record.fn,this,callbackArgs,stats):Reflect.apply(record.fn,this,callbackArgs);
          };
          owned.set(record.entry,record);
          Object.defineProperty(record.entry,'fn',{...record.descriptor,value:record.replacement});
          refs.push(new WeakReference(record.entry));
        }
      }catch{
        for(const record of records)restoreRecord(record.entry);
        fallback('record-adaptation-failed');return result;
      }
      for(const reference of references)if(!reference.deref())references.delete(reference);
      for(const reference of refs)references.add(reference);
      instances.set(this,refs);stats.adapted+=records.length;
    }catch{fallback('unsupported-hook-records');}
    return result;
  }
  try{
    registration=libWrapper.register(moduleId,TARGET,onRender,'WRAPPER');
    renderDescriptor=Object.getOwnPropertyDescriptor(prototype,'_onRender');
    const chain=renderDescriptor?.get?._lib_wrapper;
    if(chain?._wrapped===original&&typeof chain.get_fn_data==='function'){
      renderWrappers.set(prototype,{descriptor:renderDescriptor,chain,getFnData:chain.get_fn_data,original});
    }
  }catch{return finish('skipped','render-wrapper-registration-failed');}
  const result={feature:'calendariaHud',status:'installed',stats,restore(){
    if(!active)return;active=false;
    for(const reference of references){const entry=reference.deref();if(entry)restoreRecord(entry);}
    references.clear();installed.delete(prototype);
    libWrapper.unregister(moduleId,registration??TARGET);
  }};
  installed.set(prototype,result);emit(result);return result;
}
