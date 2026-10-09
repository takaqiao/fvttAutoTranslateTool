import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {pathToFileURL} from 'node:url';

const bbmmFixture=JSON.parse(fs.readFileSync(new URL('./fixtures/bbmm-rules-native.json',import.meta.url)));
export const fixture=JSON.parse(fs.readFileSync(new URL('./fixtures/bbmm-dsn-native.json',import.meta.url)));
const app=process.env.FVTT_NATIVE_APP??'C:/Program Files/Foundry Virtual Tabletop/resources/app';
const operatorsURL=pathToFileURL(`${app}/common/data/operators.mjs`);
const currentCore=JSON.parse(fs.readFileSync(new URL('./fixtures/foundry-14.369-native.json',import.meta.url)));
assert.ok([fixture.provenance.files.operators,currentCore.files.operators.sha256].includes(createHash('sha256').update(fs.readFileSync(operatorsURL)).digest('hex')));
export const operators=await import(operatorsURL.href);
export const quality={shadowQuality:'low',useHighDPI:false,glow:false,advancedGlass:false,antialiasing:'none'};
export const high={shadowQuality:'high',useHighDPI:true,glow:true,advancedGlass:true,antialiasing:'msaa'};
export const plain=v=>JSON.parse(JSON.stringify(v));
export function apply(before,patch){
  if(patch===undefined)return before;
  if(patch instanceof operators.ForcedDeletion)return undefined;
  if(patch instanceof operators.ForcedReplacement)return apply({},operators.ForcedReplacement.get(patch));
  if(!patch||typeof patch!=='object'||Array.isArray(patch))return patch;
  const out={...before};
  for(const[k,v]of Object.entries(patch)){
    if(k.startsWith('-='))delete out[k.slice(2)];
    else{const val=apply(out[k],v);if(val===undefined)delete out[k];else out[k]=val;}
  }
  return out;
}

export function harness({gm=false,sync=true,bbmmActive=true,dsnActive=true,bbmmVersion='1.4.11',dsnVersion='6.4.2',generation=14,soft=false,scope='client',target=false}={}){
  const callbacks=new Map(),registry=new Map(),values=new Map(),timers=[],warnings=[],errors=[],updates=[],rendered=[],wrappers=new Map();
  let sequence=0;
  const Hooks={on(name,fn){const id=++sequence;const list=callbacks.get(name)??new Map();list.set(id,fn);callbacks.set(name,list);return id;},once(name,fn){const id=this.on(name,(...args)=>{this.off(name,id);return fn(...args);});return id;},off(name,id){callbacks.get(name)?.delete(id);},callAll(name,...args){for(const fn of [...(callbacks.get(name)?.values()??[])])fn(...args);}};
  const call=async(name,...args)=>{for(const fn of [...(callbacks.get(name)?.values()??[])])await fn(...args);};
  const rules=Object.fromEntries(Object.entries(quality).map(([k,value])=>['bbmm.dsnQuality.'+k,{namespace:'bbmm',key:'dsnQuality.'+k,value,...soft?{soft:true}:{}}]));
  rules['example.selected']={namespace:'example',key:'selected',value:target,...soft?{soft:true}:{}};
  class User {
    static database={getFlagScopes:()=>['dice-so-nice','world']};
    constructor(id='player',isGM=gm){this.id=id;this.isGM=isGM;this.flags={'dice-so-nice':{settings:{...high,soundsVolume:.7,rollingArea:{left:80},protectPersistent:true},appearance:{global:{colorset:'custom'}},roleAppearance:{basic:{}},saved:{appearance:true},sfxList:[]},world:{keep:1}};}
    getFlag(ns,key){return this.flags[ns]?.[key];}
    async update(change){await call('preUpdateUser',this,change,{},this.id);updates.push(change);if(Object.hasOwn(change,'flags'))this.flags=apply(this.flags,change.flags);return this;}
    async setFlag(ns,key,value){return this.update({flags:{[ns]:{[key]:value}}});}
  }
  const game={user:new User(),release:{generation},version:`${generation}.368`,system:{id:'pf2e',version:'8.5.1'},modules:new Map([['bbmm',{active:bbmmActive,version:bbmmVersion}],['dice-so-nice',{active:dsnActive,version:dsnVersion}]]),i18n:{localize:k=>k},socket:{emit(){}},dice3d:{update:value=>rendered.push(value)}};
  values.set('bbmm.enableUserSettingSync',sync);values.set('bbmm.userSettingSync',rules);
  registry.set('example.selected',{id:'example.selected',namespace:'example',key:'selected',scope});
  const storage={getItem(k){return this[k];},setItem(k,v){this[k]=v;}};
  class Setting {constructor({key,value}){this.key=key;this.value=JSON.parse(value);}updateSource({value}){const next=JSON.parse(value);if(JSON.stringify(this.value)===JSON.stringify(next))return{};this.value=next;return{value};}}
  const foundry={data:{operators},utils:{duplicate:structuredClone,deepClone:structuredClone,isEmpty:o=>!Object.keys(o).length,equals:(a,b)=>JSON.stringify(a)===JSON.stringify(b),mergeObject:(a,b)=>apply(a,b)},applications:{settings:{SettingsConfig:{reloadConfirm:()=>rendered.push('reload')}}}};
  const runtime={game,Hooks,foundry,CONFIG:{User:{documentClass:User}},setTimeout:fn=>{timers.push(fn);return fn;},clearTimeout:fn=>{const i=timers.indexOf(fn);if(i>=0)timers.splice(i,1);},console:{warn:(...a)=>errors.push(a),error:(...a)=>errors.push(a),info(){}},ui:{notifications:{warn:s=>warnings.push(s),info(){}}}};
  game.canvas={app:{renderer:{context:{webGLVersion:2}}}};
  const context=vm.createContext({...runtime,Object,Boolean,Setting,_del:new operators.ForcedDeletion(),document:{createElement:()=>({})},Co:{foyer_1k:{}},Yo:{},bo:{init(){},EXTRA_TRIGGER_TYPE:[],EXTRA_TRIGGER_RESULTS:{}},Utils:{RELOAD_REQUIRED_IF_MODIFIED:Object.keys(quality),localize:v=>v,prepareTextureList:()=>({}),prepareFontList:()=>({}),prepareColorsetList:()=>({})},SFXFormulaMatcher:{validate:()=>({valid:true})},DiceLibrary:{getLibraryForUser:()=>[]},DiceScene:class{async initialize(){}setupBloomPipeline(){}updateRenderSettings(){}},ShowcaseView:class{diceList=[];async showcase(value){this.last=value;}}});
  vm.runInContext('globalThis.NativeClient=class {'+fixture.core.client.replace('#setClient','setClient').replace('this.#cleanJSON','this.cleanJSON')+'}; globalThis.NativeUser=class {'+fixture.core.unset+'}; class DsnSettings {static DEFAULT_OPTIONS='+JSON.stringify({...high,soundsVolume:.5,rollingArea:null,protectPersistent:false})+';static DEFAULT_APPEARANCE(){return {global:{colorset:"custom"}};}static ALL_DEFAULT_OPTIONS(){return {...this.DEFAULT_OPTIONS,appearance:this.DEFAULT_APPEARANCE()};}static ALL_CONFIG(){return {...this.CONFIG(),appearance:this.DEFAULT_APPEARANCE()};}static interactiveThrowsAvailable(){return false;}static DEFAULT_SFX(){return [];}static SFX(){return [];}static ROLE_CONTEXT(){return {};}'+fixture.dsn.CONFIG+'} globalThis.DsnSettings=DsnSettings; globalThis.DiceConfig=class DiceConfig {'+['parseInputs','_updateObject','_clearUserRecord','_prepareContext','getShowcaseAppearance','onApply','onReset'].map(k=>fixture.dsn[k]).join('')+'};globalThis.DiceFactory=class {'+fixture.dsn.setQualitySettings+'};',context);
  const factory=new context.DiceFactory();factory.systems=new Map([['standard',{dice:new Map(),loadSettings(){}}]]);factory.getHiddenDieTypes=()=>new Set();factory.disposeCachedMaterials=()=>{};factory.preloadPresets=async()=>{};
  game.dice3d.DiceFactory=factory;game.dice3d.box={dicefactory:factory};
  context.DiceConfig._parseLibraryDieValue=()=>({libraryDieId:''});
  User.prototype.unsetFlag=context.NativeUser.prototype.unsetFlag;
  const native=new context.NativeClient();native.storage=new Map([['client',storage]]);native.cleanJSON=(_cfg,v)=>JSON.stringify(v);
  let writes=0;
  game.settings={settings:registry,menus:new Map([['dice-so-nice.dice-so-nice',{type:context.DiceConfig}]]),register(ns,key,data){const id=ns+'.'+key;assert.ok(!registry.has(id),'duplicate setting '+id);registry.set(id,{...data,id,namespace:ns,key});},get(ns,key){const id=ns+'.'+key,cfg=registry.get(id);if(cfg?.scope==='client')return storage.getItem(id)===undefined?cfg.default:JSON.parse(storage.getItem(id));return values.has(id)?values.get(id):cfg?.default;},async set(ns,key,value){writes++;const id=ns+'.'+key,cfg=registry.get(id);if(cfg?.scope==='client')return native.setClient(cfg,value,{}).value;const exists=values.has(id);values.set(id,structuredClone(value));Hooks.callAll(exists?'updateSetting':'createSetting',{key:id,user:game.user.id,value},{value},{},game.user.id);return value;}};
  vm.runInContext('const BBMM_ID="bbmm";const LT={sync:{EnableName:()=>"",EnableHint:()=>""}};'+Object.values(bbmmFixture.registrations).join('\n'),context);
  runtime.libWrapper={register(moduleId,path,fn,type){assert.equal(moduleId,'av-v14-hotfix');assert.equal(type,'WRAPPER');assert.equal(path,'CONFIG.User.documentClass.prototype.getFlag');const original=User.prototype.getFlag;const wrapped=function(...args){return fn.call(this,original.bind(this),...args);};User.prototype.getFlag=wrapped;wrappers.set(path,{original,wrapped});return path;},unregister(moduleId,path){assert.equal(moduleId,'av-v14-hotfix');const row=wrappers.get(path);if(User.prototype.getFlag===row?.wrapped)User.prototype.getFlag=row.original;wrappers.delete(path);}};
  const controls=Object.fromEntries(Object.entries({...high,imageQuality:'high',bumpMapping:true,sounds:true,showExtraDice:false,throwingForce:'medium',muteSoundSecretRolls:false,enableFlavorColorset:true,immersiveDarkness:false}).map(([k,v])=>[k,{value:v,checked:v,disabled:Object.hasOwn(quality,k)}]));
  const makeApp=()=>Object.assign(new context.DiceConfig(),{isUser:true,isActor:false,isWorldRole:false,isRoleMode:false,document:game.user,currentRole:'basic',_roleBuffers:{basic:{global:{colorset:'custom'}}},diceFactory:factory,diceScene:new context.DiceScene(),showcaseView:new context.ShowcaseView(),element:{querySelector:s=>controls[s.match(/^\[data-(.+)\]$/)?.[1]]??null,querySelectorAll:s=>[controls[s.match(/^\[name="(.+)"\]$/)?.[1]]].filter(Boolean)},_getRoleBuffer(role){return this._roleBuffers[role]??{global:{colorset:'custom'}};},_readFormAppearance(){return{global:{colorset:'custom'}};},_getRoleDieTypes:()=>null,_prepareRoleSwitcher:()=>null,_prepareTabs:()=>({}),async _buildAppearanceTabs(){return{navAppearance:{},displayHint:'',tabsAppearance:''};},async _writeRoleAppearance(){return[];},render(){this.renderPromise=this._prepareContext({});},changeTab(){}});
  return{runtime,game,registry,values,rules,callbacks,call,context,User,makeApp,factory,controls,updates,rendered,warnings,errors,wrappers,timers,get writes(){return writes;},set:v=>game.settings.set('example','selected',v),get:()=>game.settings.get('example','selected'),async flush(){let count=0;while(timers.length){assert.ok(++count<20,'unbounded repair');await timers.shift()();}}};
}

export async function optionalPatch(name){const url=new URL('../scripts/patches/'+name+'.mjs',import.meta.url);return fs.existsSync(url)?import(url.href):{};}
