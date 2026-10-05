import {sha256Fallback} from '../source-hash.mjs';
import {dsnCompatibility} from './dsn-runtime.mjs';

const hashes={
  loadModel:'8de674019b0ec5e363828d7b5507d2fff16e88f057de77bca8152590e13a7d64',
  loaderLoad:'ddf401a139f0a607d7239ec18d6cdc62747b6abb7925d7bf080894d75bdd2209',
  loaderParse:'8fc6916e53608f400cd4737cec076bfe9239fe19c603e27982cdacb616ca0095'
};
const digest=fn=>typeof fn==='function'?sha256Fallback(Function.prototype.toString.call(fn)):null;
const installations=new WeakMap();

export function installDsnModelRecovery({g=globalThis}={}){
  const module=g.game?.modules?.get('dice-so-nice');
  if(!module?.active)return {status:'inactive'};
  const reason=dsnCompatibility(g);if(reason)return {status:reason};
  const systems=g.game.dice3d?.DiceFactory?.systems,preset=systems?.get('standard')?.dice?.get('d20');
  if(!preset)return {status:'waiting-dsn'};
  const proto=Object.getPrototypeOf(preset),prior=installations.get(proto);
  if(prior)return prior;
  const descriptor=Object.getOwnPropertyDescriptor(proto,'loadModel'),native=descriptor?.value;
  if(Object.hasOwn(preset,'loadModel')||digest(native)!==hashes.loadModel)return {status:'unsupported-source'};
  if(!descriptor.writable&&!descriptor.configurable)return {status:'unsupported-runtime'};
  let active=true,pendingNativeLoads=0;
  for(const system of systems.values())for(const die of system.dice.values())
    if(Object.getPrototypeOf(die)===proto&&die.modelFile&&!die.modelLoaded&&typeof die.modelLoading?.then==='function')pendingNativeLoads++;
  const compatible=()=>active&&g.game.modules.get('dice-so-nice')===module&&!dsnCompatibility(g)
    &&Object.getOwnPropertyDescriptor(proto,'loadModel')?.value===wrapper;
  function wrapper(...args){
    if(!compatible()||Object.getPrototypeOf(this)!==proto||Object.hasOwn(this,'loadModel')
      ||!Object.getOwnPropertyDescriptor(this,'modelLoading')?.writable
      ||!Object.getOwnPropertyDescriptor(this,'modelLoaded')?.writable
      ||this.modelLoaded||this.modelLoading!==false)return native.apply(this,args);
    const loader=args[0],loaderProto=loader&&Object.getPrototypeOf(loader);
    const load=loaderProto&&Object.getOwnPropertyDescriptor(loaderProto,'load')?.value;
    const parse=loaderProto&&Object.getOwnPropertyDescriptor(loaderProto,'parse')?.value;
    if(Object.hasOwn(loader??{},'load')||Object.hasOwn(loader??{},'parse')
      ||digest(load)!==hashes.loaderLoad||digest(parse)!==hashes.loaderParse)return native.apply(this,args);
    let reject;
    const failed=new Promise((resolve,onError)=>{reject=onError;});
    // Keep the audited preset's success callback intact and invoke GLTFLoader
    // with its real receiver. Its error callback also covers parse/onLoad throws.
    const view={load(url,onLoad){return load.call(loader,url,onLoad,undefined,reject);}};
    const promise=Promise.race([native.call(this,view,...args.slice(1)),failed]);
    this.modelLoading=promise;
    promise.catch(()=>{
      // unloadModel or a later attempt may already own the cache. Never clear it.
      const loading=Object.getOwnPropertyDescriptor(this,'modelLoading');
      if(loading?.value!==promise||!loading.writable)return;
      this.modelLoading=false;
      if(Object.getOwnPropertyDescriptor(this,'modelLoaded')?.writable)this.modelLoaded=false;
    });
    return promise;
  }
  Object.defineProperty(proto,'loadModel',{...descriptor,value:wrapper});
  const result={status:'installed',pendingNativeLoads,restore(){
    if(!active)return;
    active=false;
    const current=Object.getOwnPropertyDescriptor(proto,'loadModel');
    if(current?.value===wrapper&&['writable','configurable','enumerable'].every(key=>current[key]===descriptor[key]))
      Object.defineProperty(proto,'loadModel',descriptor);
    if(installations.get(proto)===result)installations.delete(proto);
  }};
  installations.set(proto,result);
  return result;
}
