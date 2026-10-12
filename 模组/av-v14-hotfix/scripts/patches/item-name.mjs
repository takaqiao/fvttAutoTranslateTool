const enumerable=(object,key)=>Object.prototype.propertyIsEnumerable.call(object,key);
// The label key the original compares the stored name against: null when the base type is not listed, undefined
// when only the original's merged weapon map can tell (a weapon base named like an Object.prototype member).
function baseLabelKey(item,base,globals){
  if(item.isOfType('armor','shield')){
    const types=item.isOfType('armor')?globals.CONFIG.PF2E.baseArmorTypes:globals.CONFIG.PF2E.baseShieldTypes;
    const exists=base in types,key=types[base];
    return exists?key??'':null;
  }
  const weapon=globals.CONFIG.PF2E.baseWeaponTypes,shield=globals.CONFIG.PF2E.baseShieldTypes;
  if(enumerable(shield,base))return shield[base]??'';
  if(enumerable(weapon,base))return weapon[base]??'';
  return base in Object.prototype?undefined:null;
}

/** Wrapper around PF2e's generateItemName.
 * Only its existing no-op branches are moved before weapon-map allocation.
 * All actual generated names are still calculated by the original function.
 */
export function createItemNameFastPath(original,globals,stats={fast:0,delegated:0}){
  return function generateItemNameFastPath(...args){
    const item=args[0];
    const delegate=()=>{stats.delegated++;return original.apply(this,args);};
    if(typeof item?.isOfType!=='function'||!item.isOfType('armor','shield','weapon'))return delegate();
    const base=item.baseType??'';
    if(!base||item.isSpecific){stats.fast++;return item.name;}
    const key=baseLabelKey(item,base,globals);
    if(key===undefined)return delegate();
    if(key===null||item._source.name!==globals._loc(key)){stats.fast++;return item.name;}
    return delegate();
  };
}

const PROBE_NAME='av-v14-hotfix item-name probe';
const CLAIMS=['noBase','specific','customName'];
const GENERATED_SAMPLE_LIMIT=20;
const installed=new WeakSet();
// Foundry collections are Maps that iterate values; values() reads them and plain arrays alike.
const members=collection=>typeof collection?.values==='function'?collection.values():[];
function* worldItems(game){
  for(const actor of members(game.actors))yield* members(actor?.items);
  yield* members(game.items);
}
// A read-only view of an object with some properties replaced. Everything else is read from the real object, with
// it as receiver and its methods bound to it, because getters and methods may use private fields. The proxy target
// is an empty stand-in: Foundry defines _source as a read-only, non-configurable property, and a proxy may not
// report another value for such a property of its own target.
const probeView=(real,overrides)=>new Proxy(Object.create(Object.getPrototypeOf(real)),{
  get(_,key){
    if(Object.hasOwn(overrides,key))return overrides[key];
    const value=Reflect.get(real,key,real);
    return typeof value==='function'?value.bind(real):value;
  },
  has:(_,key)=>Reflect.has(real,key),
  // The original must not write to world items while it is probed; in strict mode a refused write throws.
  set:()=>false,defineProperty:()=>false,deleteProperty:()=>false
});
// Views of one item for checking the fast path's early returns: a baseline the original should generate a name for
// (listed base type, not specific, stored name equal to the base label) and one variant per claim that flips the
// single condition the fast path returns the name on. Null when the item has no listed base type.
function claimViews(item,globals){
  const base=item.baseType??'';
  if(!base)return null;
  const key=baseLabelKey(item,base,globals);
  if(key==null)return null;
  const label=globals._loc(key),sourceNamed=name=>probeView(item._source,{name});
  const baseline={name:PROBE_NAME,isSpecific:false,_source:sourceNamed(label)};
  const view=overrides=>probeView(item,{...baseline,...overrides});
  // The custom name extends the label: localized worlds store "<base label> <English name>".
  return {baseline:view({}),noBase:view({baseType:null}),specific:view({isSpecific:true}),customName:view({_source:sourceNamed(`${label} ${PROBE_NAME}`)})};
}

/** Installs the fast path once it is shown to take the same early returns as the running generateItemName.
 * PF2e stores generateItemName's result in item.name during data preparation, so on prepared items the two
 * functions return the same text even when their branches differ. Items are therefore read through views whose
 * name is a probe string: getting the probe back means the item's own name was returned.
 * An item without runes, material or grade gets its own name back whatever the early returns are, so world items
 * as they are prove little. Each claim is checked on an item the original does generate a name for, by flipping
 * the one condition that should make it return the name instead.
 */
export function installItemNameFastPath({runtime=globalThis,report=()=>{},limit=2000}={}){
  const finish=fields=>{const result={feature:'itemNames',...fields};report(result);return result;};
  const game=runtime.game;
  if(game?.system?.id!=='pf2e')return finish({status:'unsupported-system'});
  const target=game.pf2e?.system,original=target?.generateItemName;
  if(typeof original!=='function')return finish({status:'unsupported-runtime'});
  if(installed.has(original))return finish({status:'already-installed'});
  const probe=createItemNameFastPath(original,runtime),pending=new Set(CLAIMS),unverified={};
  let sampled=0,checked=0,generated=0;
  const mismatch=(item,error,claim)=>finish({status:'behavior-mismatch',checked,
    detail:{uuid:item.uuid??item.id,type:item.type,...(claim?{claim}:{}),...(error===undefined?{}:{error})}});
  const compare=view=>{
    let expected;
    try{expected=original.call(target,view);}catch(thrown){return {threw:true,error:String(thrown)};}
    try{return {same:probe.call(target,view)===expected};}catch(thrown){return {same:false,error:String(thrown)};}
  };
  for(const item of worldItems(game)){
    if(sampled>=limit)break;
    if(typeof item?.isOfType!=='function'||!item.isOfType('armor','shield','weapon'))continue;
    sampled++;
    // An item the original cannot name as it is says nothing about the fast path.
    const real=compare(probeView(item,{name:PROBE_NAME}));
    if(!real.threw){checked++;if(!real.same)return mismatch(item,real.error);}
    if(!pending.size||generated>=GENERATED_SAMPLE_LIMIT)continue;
    let views,name;
    // The lookup is the fast path's own, so its failure on a world item is one the fast path would hit too.
    try{views=claimViews(item,runtime);}catch(thrown){return mismatch(item,String(thrown));}
    if(!views)continue;
    try{name=original.call(target,views.baseline);}catch{continue;}
    if(name===PROBE_NAME)continue;
    generated++;
    for(const claim of pending){
      const flipped=compare(views[claim]);
      if(flipped.threw)unverified[claim]={item,error:flipped.error};
      else if(flipped.same)pending.delete(claim);
      else return mismatch(item,flipped.error,claim);
    }
  }
  if(!generated)return finish({status:'skipped',reason:'no-generated-name-sample',checked});
  // A claim is left only when the original threw on its variant of every sample tried: it no longer behaves as assumed there.
  const [claim]=pending;
  if(claim)return mismatch(unverified[claim].item,unverified[claim].error,claim);
  const stats={fast:0,delegated:0},fast=createItemNameFastPath(original,runtime,stats);
  target.generateItemName=fast;installed.add(fast);
  return finish({status:'installed',checked,verified:[...CLAIMS],stats});
}
