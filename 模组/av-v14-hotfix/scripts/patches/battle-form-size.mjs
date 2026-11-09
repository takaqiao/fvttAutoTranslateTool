// PF2e 8.7.0 issue #23652: a battle form no longer changes the token's size. ActorPF2e#applyActiveEffects
// queues the token's width, height, depth and scale in the actor's final phase, from the actor's size at
// that moment. BattleForm applies its size later, in afterPrepareData, so the queued values still describe
// the actor's own size. Tokens read the queue only after the actor has finished preparing.
//
// Right after a battle form's afterPrepareData, those five queued values are recomputed from the size it
// left. They are edited only if, just before, they were exactly what PF2e derives from the size then
// (dimensions from `tokenDimensions`, scale 0.8 for small, 1 otherwise): any other queue is not this one,
// and a release that queues them later, or applies the form's size earlier, leaves nothing to change.

const KEYS=['width','height','depth','texture.scaleX','texture.scaleY'];
const TARGET='game.pf2e.RuleElements.builtin.BattleForm.prototype.afterPrepareData';
const installed=new WeakSet();

// PF2e's own size entries in the token queue: one per key, without an effect behind them.
export function queuedSizeChanges(actor){
  const initial=actor?.tokenActiveEffectChanges?.initial;
  if(!Array.isArray(initial))return null;
  const found={};
  for(const change of initial){
    if(!KEYS.includes(change?.key)||change.phase!=='initial'||change.type!=='override'||change.priority!==50||'effect' in change)continue;
    if(found[change.key])return null;
    found[change.key]=change;
  }
  return KEYS.every(key=>found[key])?found:null;
}

export function sizeValues(size){
  const dimensions=size?.tokenDimensions;
  if(!dimensions||typeof size.value!=='string')return null;
  const scale=size.value==='sm'?0.8:1;
  return {width:dimensions.width,height:dimensions.height,depth:dimensions.depth,'texture.scaleX':scale,'texture.scaleY':scale};
}

const matches=(queued,values)=>KEYS.every(key=>queued[key].value===values[key]);

export function installBattleFormSizePatch({moduleId='av-v14-hotfix',runtime=globalThis,report=()=>{}}={}){
  const finish=(status,extra)=>{const result={feature:'battleFormSize',status,...extra};report(result);return result;};
  if(runtime.game?.system?.id!=='pf2e')return finish('unsupported-system');
  const BattleForm=runtime.game.pf2e?.RuleElements?.builtin?.BattleForm;
  if(typeof BattleForm!=='function'||typeof runtime.libWrapper?.register!=='function')return finish('unsupported-runtime');
  if(installed.has(BattleForm))return finish('already-installed');
  if(typeof Object.getOwnPropertyDescriptor(BattleForm.prototype,'afterPrepareData')?.value!=='function')return finish('unsupported-source');
  // `resized`: queues rewritten to the form's size. `unmatched`: the form changed the size but the queue
  // did not hold PF2e's values for the previous size, so it was left alone.
  const stats={resized:0,unmatched:0};
  runtime.libWrapper.register(moduleId,TARGET,function(wrapped,...args){
    const actor=this.actor;
    // The form may resize the existing size object in place, so read the values before it runs.
    const before=sizeValues(actor?.system?.traits?.size),queued=queuedSizeChanges(actor);
    const result=wrapped(...args);
    const after=sizeValues(actor?.system?.traits?.size);
    if(!before||!after||KEYS.every(key=>before[key]===after[key]))return result;
    if(!queued||!matches(queued,before)){stats.unmatched++;return result;}
    for(const key of KEYS)queued[key].value=after[key];
    stats.resized++;
    return result;
  },'WRAPPER');
  installed.add(BattleForm);
  return finish('installed',{stats});
}
