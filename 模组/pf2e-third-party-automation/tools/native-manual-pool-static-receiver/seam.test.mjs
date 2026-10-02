import fs from 'node:fs';
import vm from 'node:vm';
import test from 'node:test';
import assert from 'node:assert/strict';
import {patchNativeManualPoolBatch} from '../native-manual-pool-batch/patch.mjs';

const sourcePath=process.env.PF2E_MANUAL_POOL_BATCH_SOURCE??process.env.PF2E_NATIVE_BUNDLE;
if(!sourcePath)throw Error('PF2E_MANUAL_POOL_BATCH_SOURCE is required');
const fixed=fs.readFileSync(sourcePath,'utf8');
let patcher;try{patcher=await import('./patch.mjs')}catch(error){if(error.code!=='ERR_MODULE_NOT_FOUND')throw error}
const region=(source,start,end)=>{const a=source.indexOf(start),b=source.indexOf(end,a);assert.ok(a>=0&&b>a);return source.slice(a,b)};
const deferred=()=>{let resolve;const promise=new Promise(r=>resolve=r);return {promise,resolve}};
const oldTests=fs.readFileSync(new URL('../native-manual-pool-batch/seam.test.mjs',import.meta.url),'utf8');
const plain=value=>JSON.parse(JSON.stringify(value));
export function receiverFixture({original=false,rules=[{value:5},{value:10}],options=[],outcome='success',holdWrapper=false}={}){
 const source=(!original&&patcher?patcher.patchStaticReceiver(Buffer.from(fixed)):patchNativeManualPoolBatch(Buffer.from(fixed))).bytes.toString('utf8');
 // The old native fixture executes the exact observer and button function.
 // Supply original native predicate/FlatModifier preparations to each clone.
 const fixtureSource=region(oldTests,'function fixture(source,','\n\ntest(');
 let context;const wrappedVM={...vm,createContext(value){context=vm.createContext(value);return context},runInContext(code,c){if(code.startsWith('const __nativeManualPoolBatch='))prepareContext(c,source);return vm.runInContext(code,c)}};
 const make=new Function('vm','assert','region','deferred','return ('+fixtureSource+')')(wrappedVM,assert,region,deferred);
 const f=make(source,{holdWrapper}),prepared=[],allRules=[];let constructCalls=0;
 function prepareContext(c,src){
  c.N=value=>!!value&&typeof value==='object'&&!Array.isArray(value);c.v=c.N;c.foundry={utils:{deepClone:structuredClone,getProperty:()=>{throw Error('injection-forbidden')}}};c._loc=value=>value;c.sluggify=value=>String(value);c.CONFIG.PF2E.damageTypes={};c.CONFIG.PF2E.abilities={};c.setHasElement=(set,value)=>set.has(value);c.objectHasKey=(o,k)=>Object.hasOwn(o,k);c.M=values=>[...new Set(values)];c.ErrorPF2e=message=>Error(message);c.recorder=()=>{constructCalls++;if(c.forbidConstruct)throw Error('construct-called-during-prediction')};
  vm.runInContext('const '+region(fixed,'Hn = class Predicate extends Array {','}, AutomaticBonusProgression$1 = class {')+'};',c);
  vm.runInContext('Math.clamp=(v,min,max)=>Math.min(Math.max(v,min),max);const AutomaticBonusProgression$1=class{'+region(fixed,'\tstatic isEnabled(e) {','\tstatic getStrikingDice(e) {')+region(fixed,'\tstatic suppressRuleElement(e, t) {','\tstatic getAttackPotency(e) {')+'};',c);
  const resolve=region(fixed,'\tresolveInjectedProperties(e, t = {}) {','\t#replaceFormulaData(e, t) {');
  vm.runInContext('const Y=class{constructor(raw,{parent,sourceIndex}){Object.assign(this,{ignored:false,invalid:false,predicate:new Hn(),type:"untyped",force:false,battleForm:false,fromEquipment:true,critical:null,removeAfterRoll:false,tags:[],hideIfDisabled:false},raw);this.predicate=new Hn(raw.predicate??[]);this.parent=parent;this.sourceIndex=sourceIndex}get item(){return this.parent}get actor(){return this.parent.actor}getReducedLabel(){return this.item.name}failValidation(){this.ignored=true}'+resolve+'#replaceFormulaData(){throw Error("dynamic-value")} };',c);
  vm.runInContext(region(fixed,'var mc = /* @__PURE__ */ new Set([','function createAttributeModifier('),c);
  const flat=source.slice(source.indexOf('FlatModifierRuleElement = class extends Y {'));
  const constructor=region(flat,'\tconstructor(e, t) {','\tstatic validateJoint(e) {');
  let preparation=region(flat,'\tbeforePrepareData() {','\tasync afterRoll(').replace('let construct = (n = {}) => {','let construct = (n = {}) => { recorder();');
  vm.runInContext('const FlatModifierRuleElement=class extends Y{'+constructor+'get selectors(){return this.selector}'+preparation+'};',c);
  vm.runInContext(region(fixed,'var HIGHER_BONUS =','var StatisticModifier =')+region(fixed,'function extractModifiers(','function extractDamageAlterations(')+region(fixed,'function extractDamageDice(','function processDamageCategoryStacking('),c);
  vm.runInContext('async function nativeReceiving({damage:e,token:t,item:n,rollOptions:r,skipIWR:i=true,outcome:c=null,final:l=false}){'+region(fixed,'\t\tlet f = typeof e == '+String.fromCharCode(96)+'number'+String.fromCharCode(96)+' ?', '\t\to.push(...v.map')+'return {amount:-x,flatTotal:b};}',c);
  c.game.pf2e.settings={variants:{abp:'noABP'}};c.game.pf2e.variantRules={AutomaticBonusProgression:vm.runInContext('AutomaticBonusProgression$1',c)};
  if(src.includes('const __nativeManualPoolStaticReceiver='))vm.runInContext(region(src,'const __nativeReceiverStacking=','const __nativeManualPoolBatch='),c);
 }
 function prepare(actor,raws){
  actor.items=new Map();actor.rules=[];actor.flags={pf2e:{}};actor.synthetics={modifiers:{},damageDice:{},modifierAdjustments:{}};
  raws.forEach((raw,index)=>{const data={key:'FlatModifier',selector:['healing-received'],...plain(raw)};const item={id:'I'+index,uuid:actor.uuid+'.Item.I'+index,name:'Static '+index,actor,_source:{system:{rules:[data]}},isOfType:type=>type==='effect'};actor.items.set(item.id,item);context.raw=plain(data);context.parent=item;const rule=vm.runInContext('new FlatModifierRuleElement(raw,{parent,sourceIndex:0})',context);actor.rules.push(rule);allRules.push(rule);rule.beforePrepareData()});prepared.push(actor);return actor;
 }
 const tokens=rules.map((raw,index)=>{const token=f.token('P'+index);const getClone=token.actor.getContextualClone;token.actor.getContextualClone=function(){return prepare(getClone.call(this),Array.isArray(raw)?raw:[raw])};return token});f.select(tokens);f.result.flags.pf2e.context.options=options;f.result.flags.pf2e.context.outcome=outcome;
 const originalAuthorize=f.authorize;f.authorize=event=>{const answer=originalAuthorize(event);if(event.phase==='select')for(const selection of answer.selections){const members=event.batch.candidates.filter(c=>selection.patientUUIDs.includes(c.patient.uuid));selection.selectedOrdinal=members.reduce((best,c)=>(c.receiver?.amount??0)>(best.receiver?.amount??0)?c:best).targetOrdinal}return answer};
 return {...f,context,tokens,prepared,allRules,parity(actor,params){context.modelActor=actor;context.modelParams=params;context.forbidConstruct=false;return vm.runInContext('nativeReceiving.call(modelActor,modelParams)',context)},constructCalls:()=>constructCalls,forbid(){context.forbidConstruct=true},pureModel(actor,params){context.modelActor=actor;context.modelParams=params;return vm.runInContext('__nativeManualPoolStaticReceiver.model(modelActor,modelParams)',context)}};
}

for(const values of [[5,10],[10,5]])test('actual registered constant receiving sources select the larger '+values.join('/'),async()=>{
 const f=receiverFixture({rules:values.map(value=>({value}))});f.forbid();f.subscribe(f.authorize);await f.run();assert.equal(f.nativeCalls.length,1);assert.equal(f.nativeCalls[0].params.token.id,'P'+values.indexOf(10));assert.equal(f.constructCalls(),0);const batch=f.events.find(e=>e.type==='batch-prepared').batch;assert.deepEqual(Array.from(batch.candidates,c=>c.receiver.amount),values.map(v=>10+v));
});

test('frozen empty reception component actually rejects a real constant source',async()=>{const f=receiverFixture({original:true,rules:[{value:5}]});f.subscribe(f.authorize);await assert.rejects(f.run(),/reception-unavailable/);assert.equal(f.constructCalls(),0);assert.equal(f.nativeCalls.length,0)});


test('the combined patcher rejects an altered base and retains the pinned batch component',()=>{
 assert.equal(patcher.patchStaticReceiver(Buffer.from(fixed)).batchSHA,'9c8e5f66313e49786b08826a0e1a60dd8605743c523808f7ca2aa941e325f64a');
 assert.throws(()=>patcher.patchStaticReceiver(Buffer.from(fixed+' ')),/source-mismatch/);
});
