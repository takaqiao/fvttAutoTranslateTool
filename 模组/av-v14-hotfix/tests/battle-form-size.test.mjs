import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {installBattleFormSizePatch,queuedSizeChanges,sizeValues} from '../scripts/patches/battle-form-size.mjs';

const fixture=JSON.parse(fs.readFileSync(new URL('./fixtures/pf2e-8.7.0-token-size.json',import.meta.url),'utf8'));

// PF2e 8.7.0's ActorSizePF2e and ActorPF2e#applyActiveEffects. The core Actor stand-in rebuilds the token
// queue per phase as core 14.370 does; prepareData then runs rule elements' afterPrepareData, as
// ActorPF2e#prepareData does after super.prepareData(). The battle form stand-in sets the size the way
// BattleForm's #r() does through CreatureSize: a new size object, or `+1` resizing the existing one.
function environment(){
  const context=vm.createContext({CONST:{TOKEN_DISPOSITIONS:{FRIENDLY:1,NEUTRAL:0,HOSTILE:-1}}});
  vm.runInContext(`
const Dn=${fixture.sizes};
const ActorSizePF2e=(${fixture.actorSize});
class Actor {
  constructor({size, effects={}}) { this.ownSize=size; this.effects=effects; this.rules=[]; this.alliance='party'; this.system={traits:{}}; }
  applyActiveEffects(phase) { this.tokenActiveEffectChanges[phase]=(this.effects[phase]??[]).map(change=>({...change})); }
  prepareData() {
    this.tokenActiveEffectChanges={};
    this.system.traits.size=new ActorSizePF2e({value:this.ownSize});
    this.applyActiveEffects('initial');
    this.applyActiveEffects('final');
    for (const rule of this.rules) rule.afterPrepareData?.();
  }
}
class ActorPF2e extends Actor { ${fixture.actorApplyActiveEffects} }
class BattleForm {
  constructor(actor, size) { this.actor=actor; this.overrides={size}; }
  afterPrepareData() {
    const {size}=this.overrides;
    if (size === 1) this.actor.system.traits.size.increment();
    else if (size) this.actor.system.traits.size=new ActorSizePF2e({value:size});
  }
}
globalThis.api={ActorPF2e, BattleForm, ActorSizePF2e};`,context);
  const {ActorPF2e,BattleForm}=context.api,registered=[];
  const runtime={
    game:{system:{id:'pf2e'},pf2e:{RuleElements:{builtin:{BattleForm}}}},
    libWrapper:{register(moduleId,target,fn,type){
      registered.push({moduleId,target,type});
      const original=BattleForm.prototype.afterPrepareData;
      BattleForm.prototype.afterPrepareData=function(...args){return Reflect.apply(fn,this,[original.bind(this),...args]);};
    }}
  };
  const actor=(size,{form,effects}={})=>{
    const value=new ActorPF2e({size,effects});
    if(form!==undefined)value.rules.push(new BattleForm(value,form));
    value.prepareData();
    return value;
  };
  return {runtime,registered,actor,BattleForm,context};
}
// Queue values by key, from PF2e's own entries only.
const queued=actor=>{
  const found=queuedSizeChanges(actor);
  return found&&Object.fromEntries(Object.entries(found).map(([key,change])=>[key,change.value]));
};
const values=(width,height,depth,scale)=>({width,height,depth,'texture.scaleX':scale,'texture.scaleY':scale});

test('8.7.0 queues the token size from before the battle form',()=>{
  const env=environment();
  const formed=env.actor('med',{form:'lg'});
  assert.equal(formed.system.traits.size.value,'lg');
  assert.deepEqual(queued(formed),values(1,1,1,1));
});

test('the queue takes the battle form size',()=>{
  const env=environment(),result=installBattleFormSizePatch({moduleId:'av-v14-hotfix',runtime:env.runtime});
  assert.equal(result.status,'installed');
  assert.deepEqual(env.registered,[{moduleId:'av-v14-hotfix',target:'game.pf2e.RuleElements.builtin.BattleForm.prototype.afterPrepareData',type:'WRAPPER'}]);
  assert.deepEqual(queued(env.actor('med',{form:'lg'})),values(2,2,2,1));
  assert.deepEqual(queued(env.actor('sm',{form:'huge'})),values(3,3,3,1));
  assert.deepEqual(queued(env.actor('med',{form:'sm'})),values(1,1,1,0.8));
  assert.deepEqual(queued(env.actor('med',{form:'tiny'})),values(0.5,0.5,0.5,1));
  // Resized in place: the size object keeps its own depth.
  const grown=env.actor('med',{form:1});
  assert.equal(grown.system.traits.size.value,'lg');
  assert.deepEqual(queued(grown),values(2,2,1,1));
  assert.deepEqual(result.stats,{resized:5,unmatched:0});
  // Prepared again, the actor gets a new queue and it is resized again.
  const again=env.actor('med',{form:'lg'});
  again.prepareData();
  assert.deepEqual(queued(again),values(2,2,2,1));
  assert.deepEqual(result.stats,{resized:7,unmatched:0});
});

test('no size change, or a queue that is not PF2e\'s own, is left alone',()=>{
  const env=environment(),{stats}=installBattleFormSizePatch({runtime:env.runtime});
  assert.deepEqual(queued(env.actor('med',{form:null})),values(1,1,1,1));
  assert.deepEqual(queued(env.actor('lg',{form:'lg'})),values(2,2,2,1));
  assert.deepEqual(stats,{resized:0,unmatched:0});
  // An effect's own token.width entry sits beside PF2e's and keeps its value.
  const effect={key:'width',value:4,type:'override',priority:60,phase:'initial',effect:{id:'effect'}};
  const withEffect=env.actor('med',{form:'lg',effects:{initial:[effect]}});
  assert.deepEqual(queued(withEffect),values(2,2,2,1));
  assert.equal(withEffect.tokenActiveEffectChanges.initial.find(change=>'effect' in change).value,4);
  assert.deepEqual(stats,{resized:1,unmatched:0});
});

test('a queue edited by someone else, or missing, is counted and not changed',()=>{
  const env=environment(),{stats}=installBattleFormSizePatch({runtime:env.runtime});
  const actor=env.actor('med');
  actor.rules.push(new env.BattleForm(actor,'lg'));
  const prepare=actor.applyActiveEffects;
  actor.applyActiveEffects=function(phase){
    Reflect.apply(prepare,this,[phase]);
    if(phase==='final')this.tokenActiveEffectChanges.initial.find(change=>change.key==='width'&&!('effect' in change)).value=3;
  };
  actor.prepareData();
  assert.deepEqual(queued(actor),{...values(1,1,1,1),width:3});
  assert.deepEqual(stats,{resized:0,unmatched:1});
  // A release that queues the size after rule elements leaves nothing here to change.
  const later=env.actor('med');
  later.applyActiveEffects=function(phase){this.tokenActiveEffectChanges[phase]=[];};
  later.rules.push(new env.BattleForm(later,'lg'));
  later.prepareData();
  assert.deepEqual(later.tokenActiveEffectChanges.initial,[]);
  assert.deepEqual(stats,{resized:0,unmatched:2});
});

test('queue and size readers want exactly PF2e\'s entries',()=>{
  const env=environment(),actor=env.actor('med');
  assert.deepEqual(sizeValues(actor.system.traits.size),values(1,1,1,1));
  assert.equal(sizeValues({value:'med'}),null);
  const duplicate=env.actor('med');
  duplicate.tokenActiveEffectChanges.initial.push({...duplicate.tokenActiveEffectChanges.initial.find(change=>change.key==='height')});
  assert.equal(queuedSizeChanges(duplicate),null);
  const partial=env.actor('med');
  partial.tokenActiveEffectChanges.initial=partial.tokenActiveEffectChanges.initial.filter(change=>change.key!=='depth');
  assert.equal(queuedSizeChanges(partial),null);
});

test('install needs PF2e, libWrapper and BattleForm\'s own afterPrepareData, once',()=>{
  const env=environment();
  assert.equal(installBattleFormSizePatch({runtime:{...env.runtime,game:{system:{id:'sf2e'}}}}).status,'unsupported-system');
  assert.equal(installBattleFormSizePatch({runtime:{...env.runtime,libWrapper:undefined}}).status,'unsupported-runtime');
  assert.equal(installBattleFormSizePatch({runtime:{...env.runtime,game:{system:{id:'pf2e'},pf2e:{}}}}).status,'unsupported-runtime');
  const inherited=environment();
  inherited.runtime.game.pf2e.RuleElements.builtin.BattleForm=class extends inherited.BattleForm{};
  assert.equal(installBattleFormSizePatch({runtime:inherited.runtime}).status,'unsupported-source');
  assert.equal(installBattleFormSizePatch({runtime:env.runtime}).status,'installed');
  assert.equal(installBattleFormSizePatch({runtime:env.runtime}).status,'already-installed');
  assert.equal(env.registered.length,1);
});
