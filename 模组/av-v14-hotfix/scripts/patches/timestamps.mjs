import {hashSource} from '../source-hash.mjs';

const SELECTOR='.chat-message[data-message-id] .message-timestamp';
const HOOKS=['renderChatMessageHTML','renderChatLog','renderChatPopout'];
const installations=new WeakMap();
const functionSource=Function.prototype.toString;
const nativeFunction=fn=>typeof fn==='function'&&/^function(?:\s+(?:(?:get|set)\s+)?[\w$]+)?\s*\(\s*\)\s*\{\s*\[native code\]\s*\}$/.test(Reflect.apply(functionSource,fn,[]));
const descriptor=(target,key)=>target&&Object.getOwnPropertyDescriptor(target,key);
const sameDescriptor=(a,b)=>!!a&&!!b&&['value','get','set','writable','enumerable','configurable'].every(key=>a[key]===b[key]);
function inherited(target,key){
  for(let p=target;p;p=Object.getPrototypeOf(p)){
    const d=descriptor(p,key);if(d)return d;
  }
}

function coreSources(runtime){
  const applications=runtime.foundry?.applications;
  return [
    [applications?.api?.ApplicationV2?.prototype,'_doEvent','d1a7e27de29f5fe11e75d98263dbc65efad5efd3529eafb61447c9d037d3e823'],
    [applications?.api?.ApplicationV2,'inheritanceChain','e860f36e7fe2c9b95d185e6f454b2bb039a29807d824f94dc3f11c92cc0e25b0'],
    [applications?.sidebar?.tabs?.ChatLog,'renderMessage','82023d47f625284e0670d20b6468b74b6abb90a8dce7c8d69e405d8ba6317381'],
    [applications?.sidebar?.apps?.ChatPopout?.prototype,'_renderHTML','234da83c8ac9c5d1d870f681b008b40992a32da9350fd90f1eb22ae34acce5b2']
  ].map(([target,key,sha])=>({target,key,sha,descriptor:descriptor(target,key)}));
}

function nativeDOM(runtime){
  const constructors=['Node','Element','Document','CharacterData','Text'].map(key=>[key,runtime[key]]);
  if(constructors.some(([,value])=>!nativeFunction(value)))return;
  const properties=[
    [runtime.Node.prototype,'textContent','accessor'],
    [runtime.Node.prototype,'firstChild','getter'],
    [runtime.Node.prototype,'nextSibling','getter'],
    [runtime.Node.prototype,'nodeType','getter'],
    [runtime.CharacterData.prototype,'data','accessor'],
    [runtime.Element.prototype,'matches','method'],
    [runtime.Element.prototype,'querySelectorAll','method'],
    [runtime.Document.prototype,'querySelectorAll','method']
  ].map(([target,key,kind])=>({target,key,kind,descriptor:descriptor(target,key)}));
  if(properties.some(({kind,descriptor:d})=>!d||(kind==='method'?!nativeFunction(d.value):!nativeFunction(d.get)||(kind==='accessor'&&!nativeFunction(d.set)))))return;
  const [text,first,next,type,data,matches,query,documentQuery]=properties.map(p=>p.descriptor);
  return {text,first,next,type,data,matches,query,documentQuery,
    unchanged:()=>constructors.every(([key,value])=>runtime[key]===value)&&properties.every(p=>sameDescriptor(descriptor(p.target,p.key),p.descriptor))};
}

/**
 * Avoid redundant plaintext replacement on actual chat timestamps only. A skipped
 * write intentionally preserves the sole Text node and emits no childList mutation.
 * Changed text and all other DOM shapes still use the inherited setter.
 */
export async function installTimestampPatch({runtime=globalThis,hash=hashSource,report=()=>{}}={}){
  const finish=(status,extra={})=>{const result={feature:'timestamps',status,...extra};report(result);return result;};
  const document=runtime.document,Hooks=runtime.Hooks,dom=nativeDOM(runtime);
  if(!dom||!document||typeof Hooks?.on!=='function'||typeof Hooks?.off!=='function'||
    typeof runtime.WeakRef!=='function'||typeof runtime.FinalizationRegistry!=='function'||typeof runtime.queueMicrotask!=='function')return finish('unsupported-runtime');
  const existing=installations.get(document);if(existing)return existing;
  const sources=coreSources(runtime);
  if(sources.some(s=>typeof s.descriptor?.value!=='function'))return finish('unsupported-source');
  // renderChatMessageHTML is a public subscription, not a replaced renderer.
  // Its native emitter is recorded in the fixture; libWrapper may expose that
  // renderer through an accessor, which this installer must never evaluate.
  const hashes=await Promise.all(sources.map(s=>hash(Reflect.apply(functionSource,s.descriptor.value,[]))));
  if(hashes.some((value,i)=>value!==sources[i].sha))return finish('unsupported-source');
  const currentSources=coreSources(runtime);
  if(runtime.document!==document||runtime.Hooks!==Hooks||!dom.unchanged()||
    sources.some((s,i)=>s.target!==currentSources[i].target||!sameDescriptor(s.descriptor,currentSources[i].descriptor)))return finish('source-changed-during-validation');
  const concurrent=installations.get(document);if(concurrent)return concurrent;

  let active=true,attached=0;
  const records=new WeakMap(),references=new Set(),hooks=[];
  const finalizer=new runtime.FinalizationRegistry(reference=>references.delete(reference));
  const owns=(element,record)=>sameDescriptor(descriptor(element,'textContent'),record.descriptor);
  const isElement=value=>value instanceof runtime.Element;
  const nativeProperty=(value,key,native)=>sameDescriptor(inherited(value,key),native);

  function makeAccessors(reference){
    return {
      get(){
        if(this!==reference.deref())return Reflect.apply(dom.text.get,this,[]);
        const current=inherited(Object.getPrototypeOf(this),'textContent');
        return current&&('value' in current?current.value:current.get?Reflect.apply(current.get,this,[]):undefined);
      },
      set(value){
        if(this!==reference.deref())return Reflect.apply(dom.text.set,this,[value]);
        const current=inherited(Object.getPrototypeOf(this),'textContent');
        if(active&&typeof value==='string'&&sameDescriptor(current,dom.text)&&dom.unchanged()&&
          nativeProperty(this,'matches',dom.matches)&&Reflect.apply(dom.matches.value,this,[SELECTOR])&&
          nativeProperty(this,'firstChild',dom.first)){
          const child=Reflect.apply(dom.first.get,this,[]);
          if(child&&nativeProperty(child,'nodeType',dom.type)&&Reflect.apply(dom.type.get,child,[])===3&&
            nativeProperty(child,'nextSibling',dom.next)&&Reflect.apply(dom.next.get,child,[])===null&&
            nativeProperty(child,'data',dom.data)&&Reflect.apply(dom.data.get,child,[])===value)return;
        }
        // A later inherited accessor belongs to its author; do not bypass it.
        if(current?.set)return Reflect.apply(current.set,this,[value]);
        if(current&&!('value' in current))throw new TypeError('Cannot set textContent which has only a getter');
        // A later inherited data property (or deletion) needs ordinary assignment.
        // Peel only our unchanged descriptor, then let Reflect.set enforce writability.
        const record=records.get(this);
        if(!record||!owns(this,record)||!Reflect.deleteProperty(this,'textContent'))throw new TypeError('Cannot replace textContent');
        if(!Reflect.set(this,'textContent',value))throw new TypeError('Cannot assign textContent');
      }
    };
  }

  function attach(element){
    if(!active||!isElement(element)||records.has(element)||Object.hasOwn(element,'textContent')||!Object.isExtensible(element)||
      !nativeProperty(element,'textContent',dom.text)||!nativeProperty(element,'matches',dom.matches)||
      !Reflect.apply(dom.matches.value,element,[SELECTOR]))return;
    const reference=new runtime.WeakRef(element);
    const owned={...makeAccessors(reference),enumerable:dom.text.enumerable,configurable:true};
    try{Object.defineProperty(element,'textContent',owned);}catch{return;}
    records.set(element,{reference,descriptor:owned});references.add(reference);
    finalizer.register(element,reference,reference);attached++;
  }

  function attachTree(root){
    if(!active||!dom.unchanged()||!isElement(root))return;
    attach(root);
    for(const element of Reflect.apply(dom.query.value,root,[SELECTOR]))attach(element);
  }

  // Do not close over the hook's application/ChatMessage or the HTML root. This
  // second local pass covers later synchronous render listeners replacing a stamp.
  function queueLocal(reference){runtime.queueMicrotask(()=>{const root=reference.deref();if(active&&root)attachTree(root);});}
  function onRender(_application,html){
    if(!active||!isElement(html))return;
    attachTree(html);queueLocal(new runtime.WeakRef(html));
  }

  function restore(){
    if(!active)return;
    active=false;
    for(const {hook,id}of hooks)try{Hooks.off(hook,id);}catch{/* Continue retiring our other hooks and descriptors. */}
    for(const reference of references){
      const element=reference.deref(),record=element&&records.get(element);
      if(record&&owns(element,record))Reflect.deleteProperty(element,'textContent');
      finalizer.unregister(reference);
      if(element)records.delete(element);
    }
    references.clear();
    installations.delete(document);
  }

  try{
    for(const hook of HOOKS)hooks.push({hook,id:Hooks.on(hook,onRender)});
    for(const element of Reflect.apply(dom.documentQuery.value,document,[SELECTOR]))attach(element);
  }catch(error){restore();return finish('installation-failed',{reason:String(error)});}
  const result={feature:'timestamps',status:'installed',attached,hooks,restore};
  installations.set(document,result);report(result);return result;
}
