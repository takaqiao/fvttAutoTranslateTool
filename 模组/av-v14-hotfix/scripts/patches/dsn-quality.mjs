import {sha256Fallback} from '../source-hash.mjs';
import {bbmmCompatibility,readBbmmHardRules} from './bbmm-locks.mjs';

const fields={
  shadowQuality:{type:String,value:'low',name:'DsN 阴影质量',choices:{none:'关闭',low:'低',high:'高'}},
  useHighDPI:{type:Boolean,value:false,name:'DsN 高 DPI'},
  glow:{type:Boolean,value:false,name:'DsN 辉光'},
  advancedGlass:{type:Boolean,value:false,name:'DsN 高级玻璃'},
  antialiasing:{type:String,value:'none',name:'DsN 抗锯齿',choices:{none:'关闭',msaa:'MSAA',smaa:'SMAA'}}
};
// Audited native DiceConfig consumers, DsN 6.4.1. Validate synchronously in
// setup so neither the first renderer nor BBMM's ready sync can race hashing.
const hashes={
  parseInputs:'32f903886b97b79c8887400fe1bdf71ff47830832f603e4bd68e89ae043e6713',
  _updateObject:'8387b2486be46725baddb47fbb8efc8df60518609e70028408fc7ce144cee3f3',
  _clearUserRecord:'4791bc44f2e96a89047af8390d198bc4ee3a446be9f1bc285b8d3495031157f4',
  _prepareContext:'d237b5c2ddd181bab469927aeeffc48b737413b180117cd9db4a2572cb09f2d3',
  getShowcaseAppearance:'a34fd3108315b3e3e215bd8c915c6bb96ae36519784061fdb8ca87c6229f2740',
  onApply:'1274e9de8fb72ea6a9f0a776e2bb45e1ce440b6533dae2611ad5f5340b89189a',
  onReset:'78809ca815488f9a3bf2ca6ab20d771473c74be6adb1dcbb46fd027433e95b5f'
};
const factoryHash='e78f559590c60dc25fc14b110ded6778213fad64fe33011c591f5a2cc09aa149';
const registered=new WeakSet(),installed=new WeakMap();
const compatibility=runtime=>bbmmCompatibility(runtime)
  ||(!runtime.game.modules.get('dice-so-nice')?.active?'inactive-dsn':null)
  ||(runtime.game.modules.get('dice-so-nice').version!=='6.4.1'?'unsupported-dsn':null);

export function registerDsnQualitySettings({runtime=globalThis,report=()=>{}}={}){
  const finish=status=>{const result={feature:'dsnQualityLocks',status};report(result);return result;};
  const reason=compatibility(runtime);if(reason)return finish(reason);
  const settings=runtime.game.settings;
  if(registered.has(settings))return finish('already-registered');
  if(Object.keys(fields).some(key=>settings.settings.has(`bbmm.dsnQuality.${key}`)))return finish('settings-conflict');
  for(const[key,def]of Object.entries(fields))settings.register('bbmm',`dsnQuality.${key}`,{
    name:`${def.name}（硬锁适配）`,
    hint:'由 FVTT v14 Local Hotfix 适配。通过 BBMM 锁定时强制玩家画质；未锁定时沿用个人设置。GM 保留管理权限。',
    scope:'client',config:true,type:def.type,default:def.value,
    ...(def.choices?{choices:def.choices}:{}),requiresReload:true
  });
  registered.add(settings);return finish('registered');
}

function patchFields(node,path,locked,operators){
  const {ForcedDeletion,ForcedReplacement}=operators;
  const replace=node instanceof ForcedDeletion||node instanceof ForcedReplacement;
  const raw=node instanceof ForcedReplacement?ForcedReplacement.get(node):node;
  const value=raw&&typeof raw==='object'&&!(raw instanceof ForcedDeletion)?{...raw}:{};
  if(path.length){
    const[key,...rest]=path,deleting=Object.hasOwn(value,`-=${key}`);
    delete value[`-=${key}`];
    value[key]=patchFields(deleting?new ForcedDeletion():value[key],rest,locked,operators);
  }else for(const[key,target]of Object.entries(locked)){delete value[`-=${key}`];value[key]=target;}
  return replace?new ForcedReplacement(value):value;
}

export function installDsnQualityLocks({moduleId='av-v14-hotfix',runtime=globalThis,report=()=>{}}={}){
  const finish=status=>{const result={feature:'dsnQualityLocks',status};report(result);return result;};
  const reason=compatibility(runtime);if(reason)return finish(reason);
  const {game,Hooks,libWrapper}=runtime,settings=game.settings;
  if(installed.has(settings))return finish('already-installed');
  if(!registered.has(settings))return finish('settings-conflict');
  const prototype=settings.menus.get('dice-so-nice.dice-so-nice')?.type?.prototype;
  const userPrototype=runtime.CONFIG?.User?.documentClass?.prototype;
  const operators=runtime.foundry?.data?.operators;
  if(!prototype||typeof userPrototype?.getFlag!=='function'||typeof libWrapper?.register!=='function'
    ||typeof libWrapper?.unregister!=='function'||typeof Hooks?.off!=='function'
    ||typeof operators?.ForcedDeletion!=='function'||typeof operators?.ForcedReplacement?.get!=='function')return finish('unsupported-runtime');
  const descriptors=Object.fromEntries(Object.keys(hashes).map(key=>[key,Object.getOwnPropertyDescriptor(prototype,key)]));
  for(const[key,hash]of Object.entries(hashes)){
    const descriptor=descriptors[key];
    if(typeof descriptor?.value!=='function'||!descriptor.writable
      ||sha256Fallback(Function.prototype.toString.call(descriptor.value))!==hash)return finish('unsupported-source');
  }
  let active=true;const hooks=[],factoryPatches=new Map(),methods={};
  const currentFactory=()=>game.dice3d?.box?.dicefactory??game.dice3d?.DiceFactory;
  const changed=()=>{
    if(Object.entries(descriptors).some(([key,descriptor])=>prototype[key]!== (methods[key]??descriptor.value)))return true;
    for(const[target,row]of factoryPatches)if(target.setQualitySettings!==row.wrapper)return true;
    const factory=currentFactory(),row=factory&&factoryPatches.get(Object.getPrototypeOf(factory));
    return row&&factory.setQualitySettings!==row.wrapper;
  };
  const locks=(user=game.user)=>{
    if(!active)return {};
    if(compatibility(runtime)||changed()){
      active=false;finish('source-changed');return {};
    }
    const rules=readBbmmHardRules(runtime,user),values={};
    for(const[key,def]of Object.entries(fields)){
      const row=rules[`bbmm.dsnQuality.${key}`];if(!row||row.soft===true)continue;
      if(def.type===Boolean?typeof row.value!=='boolean':!Object.hasOwn(def.choices,row.value))continue;
      values[key]=row.value;
    }
    return values;
  };
  // DsN feeds parsed values directly to its renderer after saving, so a flag
  // write hook alone would leave the current page rendering unlocked quality.
  const controls=(root,locked)=>{
    for(const[key,value]of Object.entries(locked))for(const field of root?.querySelectorAll?.(`[name="${key}"]`)??[]){
      field.disabled=true;
      if(typeof value==='boolean')field.checked=value;else field.value=value;
      field.title='由 GM 通过 BBMM 强制锁定';
    }
  };
  const attachFactory=factory=>{
    if(!active||!factory)return;
    const target=Object.getPrototypeOf(factory),previous=factoryPatches.get(target);
    if(previous){if(factory.setQualitySettings!==previous.wrapper){active=false;finish('source-changed');}return;}
    const descriptor=Object.getOwnPropertyDescriptor(target,'setQualitySettings');
    if(typeof descriptor?.value!=='function'||!descriptor.writable||factory.setQualitySettings!==descriptor.value
      ||sha256Fallback(Function.prototype.toString.call(descriptor.value))!==factoryHash){active=false;finish('unsupported-factory-source');return;}
    const wrapper=function(options,...args){
      const locked=locks();
      return descriptor.value.call(this,Object.keys(locked).length?{...options,...locked}:options,...args);
    };
    Object.defineProperty(target,'setQualitySettings',{...descriptor,value:wrapper});
    factoryPatches.set(target,{descriptor,wrapper});
  };
  methods.parseInputs=function(...args){
    const value=descriptors.parseInputs.value.apply(this,args);
    if(this.isUser)Object.assign(value,locks(this.document));
    return value;
  };
  methods.getShowcaseAppearance=function(...args){
    attachFactory(currentFactory());
    const value=descriptors.getShowcaseAppearance.value.apply(this,args);
    if(this.isUser){const locked=locks(this.document);Object.assign(value,locked);controls(this.element,locked);}
    return value;
  };
  methods._prepareContext=async function(...args){
    // Reset renders use ALL_DEFAULT_OPTIONS rather than CONFIG. The shared
    // factory must be protected before that native async render begins.
    attachFactory(currentFactory());
    const value=await descriptors._prepareContext.value.apply(this,args);
    if(this.isUser)Object.assign(value,locks(this.document));
    return value;
  };
  const flagTarget='CONFIG.User.documentClass.prototype.getFlag';
  let registration;
  const restoreMethods=()=>{
    for(const[key,method]of Object.entries(methods))if(prototype[key]===method)Object.defineProperty(prototype,key,descriptors[key]);
    for(const[target,row]of factoryPatches)if(target.setQualitySettings===row.wrapper)Object.defineProperty(target,'setQualitySettings',row.descriptor);
  };
  for(const[key,value]of Object.entries(methods))Object.defineProperty(prototype,key,{...descriptors[key],value});
  try{
    registration=libWrapper.register(moduleId,flagTarget,function(wrapped,...args){
      const value=wrapped(...args);
      if(args[0]!=='dice-so-nice'||args[1]!=='settings')return value;
      const locked=locks(this);return Object.keys(locked).length?{...value,...locked}:value;
    },'WRAPPER');
  }catch(error){
    restoreMethods();
    throw error;
  }
  const on=(name,fn,once=false)=>hooks.push([name,Hooks[once?'once':'on'](name,fn)]);
  // First board construction already reads locked CONFIG. This hook also
  // covers the common factory prototype used after a native board rebuild.
  on('diceSoNiceReady',dice=>attachFactory(dice?.box?.dicefactory??dice?.DiceFactory));
  on('preUpdateUser',(user,change)=>{
    if(!change||!Object.hasOwn(change,'flags'))return;
    const locked=locks(user);if(!Object.keys(locked).length)return;
    // Preserve native v14 deletion/replacement semantics at every ancestor.
    change.flags=patchFields(change.flags,['dice-so-nice','settings'],locked,operators);
  });
  on('ready',async()=>{
    const locked=locks();
    if(!Object.entries(locked).some(([key,value])=>game.user.flags?.['dice-so-nice']?.settings?.[key]!==value))return;
    try{await game.user.update({flags:{'dice-so-nice':{settings:locked}}});}
    catch(error){runtime.console?.warn?.(`${moduleId} | DsN quality lock persistence failed`,error);}
  },true);
  on('renderDiceConfig',(app,html)=>{
    if(!app.isUser)return;
    const root=html?.querySelectorAll?html:html?.[0];
    controls(root,locks(app.document));
  });
  const restore=()=>{
    active=false;for(const[name,id]of hooks)Hooks.off(name,id);
    restoreMethods();
    libWrapper.unregister(moduleId,registration??flagTarget);installed.delete(settings);
  };
  const result={feature:'dsnQualityLocks',status:'installed',restore};installed.set(settings,result);report(result);
  attachFactory(currentFactory());return active?result:{...result,status:'unsupported-factory-source'};
}
