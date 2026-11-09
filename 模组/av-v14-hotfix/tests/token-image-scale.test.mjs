import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {installTokenImageScalePatch,scaleBranchShape} from '../scripts/patches/token-image-scale.mjs';

const fixture=JSON.parse(fs.readFileSync(new URL('./fixtures/pf2e-8.7.0-token-size.json',import.meta.url),'utf8'));
const NATIVE=fixture.activeEffectApplyChange;
// #23641 as the bundle would minify it.
const FIXED=NATIVE.replace('if (!e.autoscale) return {};','if (t.phase === `initial` && !e.autoscale) return {};');

const getProperty=(object,path)=>path.split('.').reduce((node,key)=>node?.[key],object);
const setProperty=(object,path,value)=>{const keys=path.split('.'),last=keys.pop();keys.reduce((node,key)=>node[key]??={},object)[last]=value;};

// PF2e's ActiveEffectPF2e.applyChange (8.7.0) over a core class that records what reaches it.
function environment(method=NATIVE){
  const applied=[];
  const context=vm.createContext({
    game:{release:{version:'14.370'}},
    foundry:{utils:{getProperty,setProperty}},
    CONST:{TOKEN_DISPOSITIONS:{SECRET:-2}},
    M:value=>!!value&&typeof value==='object'&&!Array.isArray(value),
    applied
  });
  const ActiveEffect=vm.runInContext(`(class ActiveEffect {
    static applyChange(targetDoc, change, options) { applied.push({targetDoc, change, options}); return {[change.key]: change.value}; }
  })`,context);
  context.ActiveEffect=ActiveEffect;
  const ActiveEffectPF2e=vm.runInContext(`(class ActiveEffectPF2e extends ActiveEffect { ${method} })`,context);
  const registered=[];
  const runtime={
    game:{system:{id:'pf2e'}},
    foundry:{utils:{getProperty}},
    CONFIG:{ActiveEffect:{documentClass:ActiveEffectPF2e}},
    libWrapper:{register(moduleId,target,fn,type){
      registered.push({moduleId,target,type});
      assert.equal(target,'CONFIG.ActiveEffect.documentClass.applyChange');
      const original=ActiveEffectPF2e.applyChange;
      ActiveEffectPF2e.applyChange=function(...args){return Reflect.apply(fn,this,[original.bind(this),...args]);};
    }}
  };
  // Results come from the vm realm; spread them into plain objects for deepStrictEqual.
  return {runtime,ActiveEffectPF2e,applied,registered,apply:(...args)=>({...ActiveEffectPF2e.applyChange(...args)})};
}
const token=({autoscale=false,scaleX=1}={})=>({documentName:'Token',autoscale,linkToActorSize:true,_source:{texture:{scaleX,scaleY:1}}});
const scale=(phase,value=2,key='texture.scaleX')=>({key,value,type:'override',priority:50,phase});

test('8.7.0 drops a TokenImage scale on a token without autoscale',()=>{
  const {apply,applied}=environment();
  assert.deepEqual(apply(token(),scale('final'),{}),{});
  assert.deepEqual(applied,[]);
});

test('final-phase scale reaches the core class, mirrored like the native branch',()=>{
  const env=environment(),result=installTokenImageScalePatch({moduleId:'av-v14-hotfix',runtime:env.runtime});
  assert.equal(result.status,'installed');
  assert.deepEqual(env.registered,[{moduleId:'av-v14-hotfix',target:'CONFIG.ActiveEffect.documentClass.applyChange',type:'MIXED'}]);
  const options={replacementData:{},modifyTarget:true};
  assert.deepEqual(env.apply(token(),scale('final',1.5),options),{'texture.scaleX':1.5});
  assert.deepEqual(env.apply(token({scaleX:-1}),scale('final',1.5),options),{'texture.scaleX':-1.5});
  assert.deepEqual(env.apply(token(),scale('final',0.5,'texture.scaleY'),options),{'texture.scaleY':0.5});
  assert.equal(env.applied[0].options,options);
  assert.equal(env.applied[0].change.phase,'final');
  assert.deepEqual(result.stats,{applied:3});
});

test('the size-derived initial-phase scale stays gated, other changes stay native',()=>{
  const env=environment(),{stats}=installTokenImageScalePatch({runtime:env.runtime});
  const {apply,applied}=env;
  assert.deepEqual(apply(token(),scale('initial',0.8),{}),{});
  // autoscale on: the native branch already mirrors and applies.
  assert.deepEqual(apply(token({autoscale:true,scaleX:-1}),scale('final',2),{}),{'texture.scaleX':-2});
  assert.deepEqual(apply({...token(),linkToActorSize:false},{key:'width',value:3,phase:'final'},{}),{});
  const actor={documentName:'Actor'};
  assert.deepEqual(apply(actor,scale('final',2),{}),{'texture.scaleX':2});
  assert.equal(applied.length,2);
  assert.deepEqual(stats,{applied:0});
});

test('the scale branch shape is read through renamed parameters and locals',()=>{
  const {ActiveEffectPF2e}=environment();
  const source=Function.prototype.toString.call(ActiveEffectPF2e.applyChange);
  assert.equal(scaleBranchShape(source),'unfixed');
  const renamed=source.replace(/\be\b/g,'doc').replace(/\bt\b/g,'chg').replace(/\bn\b/g,'opts').replace(/\br\b/g,'range').replace(/\bi\b/g,'mode');
  assert.notEqual(renamed,source);
  assert.equal(scaleBranchShape(renamed),'unfixed');
  assert.equal(scaleBranchShape(FIXED.replace(/^static /,'')),'fixed');
  // The scale case written without its block, a different gate, or no mirroring is not this method.
  for(const altered of [
    source.replace('if (!e.autoscale) return {};','if (!e.autoscale && !e.linkToActorSize) return {};'),
    source.replace('n * Number(t.value)','Number(t.value)'),
    source.replace('case `texture.scaleY`: {','case `texture.scaleY`:\n\t\t\tcase `texture.alpha`: {'),
    source.replace('return super.applyChange(e, t, n);\n\t}','return {};\n\t}'),
    source.replace('if (e.documentName !== `Token`) return super.applyChange(e, t, n);','')
  ]){
    assert.notEqual(altered,source);
    assert.equal(scaleBranchShape(altered),null);
  }
});

test('a release carrying #23641 is left alone; an unknown shape is not wrapped',()=>{
  const fixed=environment(FIXED),reports=[];
  assert.deepEqual(installTokenImageScalePatch({runtime:fixed.runtime,report:entry=>reports.push(entry)}),
    {feature:'tokenImageScale',status:'skipped',reason:'upstream-fixed'});
  assert.deepEqual(fixed.registered,[]);
  assert.deepEqual(reports,[{feature:'tokenImageScale',status:'skipped',reason:'upstream-fixed'}]);
  // With the fix, the native method itself passes a final-phase scale on.
  assert.deepEqual(fixed.apply(token(),scale('final',1.5),{}),{'texture.scaleX':1.5});
  const altered=environment(NATIVE.replace('if (!e.autoscale) return {};','if (!e.autoscale) return { [t.key]: 1 };'));
  assert.equal(installTokenImageScalePatch({runtime:altered.runtime}).status,'unsupported-source');
  assert.deepEqual(altered.registered,[]);
});

test('install needs PF2e, libWrapper and the class\'s own method, once',()=>{
  const env=environment();
  assert.equal(installTokenImageScalePatch({runtime:{...env.runtime,game:{system:{id:'dnd5e'}}}}).status,'unsupported-system');
  assert.equal(installTokenImageScalePatch({runtime:{...env.runtime,libWrapper:undefined}}).status,'unsupported-runtime');
  // Already wrapped through an accessor by someone else.
  const wrapped=environment();
  Object.defineProperty(wrapped.ActiveEffectPF2e,'applyChange',{get:()=>()=>({}),configurable:true});
  assert.equal(installTokenImageScalePatch({runtime:wrapped.runtime}).status,'unsupported-source');
  assert.equal(installTokenImageScalePatch({runtime:env.runtime}).status,'installed');
  assert.equal(installTokenImageScalePatch({runtime:env.runtime}).status,'already-installed');
  assert.equal(env.registered.length,1);
});
