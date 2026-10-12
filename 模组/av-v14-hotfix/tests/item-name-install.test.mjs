import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {readFile} from 'node:fs/promises';
import {createItemNameFastPath,installItemNameFastPath} from '../scripts/patches/item-name.mjs';

const VERSIONS=['8.5.1','8.6.0','8.6.1'],prefix='PF2E.Item.Physical.GeneratedName.';
const sources=Object.fromEntries(await Promise.all(VERSIONS.map(async version=>
  [version,await readFile(new URL(`./fixtures/generate-item-name-${version}.js.txt`,import.meta.url),'utf8')])));
// Foundry collections are Maps whose default iterator yields values.
class Collection extends Map{[Symbol.iterator](){return this.values();}}
const collect=values=>new Collection(values.map((value,index)=>[index,value]));

// Runs an official function in strict mode, as PF2e's module does, with every release's closure names defined;
// edit lets a test alter its source first.
function world({version='8.6.1',edit,systemId='pf2e',actors=[],items=[]}={}){
  const runes={armor:{property:{fortification:{name:'Fortification'}},resilient:{1:{name:'Resilient'}}},
    weapon:{property:{flaming:{name:'Flaming'}},striking:{1:{name:'Striking'}}}},reinforcing={1:'Reinforcing'};
  const runtime={CONFIG:{PF2E:{baseArmorTypes:{plate:'Plate'},baseWeaponTypes:{sword:'Sword'},baseShieldTypes:{shield:'Shield'},
    preciousMaterials:{silver:'Silver'},grades:{commercial:'Commercial'}}},
    Jn:runes,Yn:runes,Xn:runes,Es:reinforcing,Ds:reinforcing,ks:reinforcing,o:Boolean,
    _loc:(key,values)=>values?`${key}:${values.base}`:key,AutomaticBonusProgression:{isEnabled:()=>false}};
  const source=edit?edit(sources[version]):sources[version];
  if(edit)assert.notEqual(source,sources[version],'the edit must change the official source');
  const original=vm.runInNewContext(`'use strict';(${source})`,runtime),target={generateItemName:original};
  // PF2e stores the generated name on every item while preparing its data, before ready.
  for(const value of [...actors.flat(),...items])value.name=original(value);
  runtime.game={system:{id:systemId},pf2e:{system:target},actors:collect(actors.map(owned=>({items:collect(owned)}))),items:collect(items)};
  return {runtime,original,target};
}
function item(type,baseType,name,{runes={potency:0,striking:0,property:[]},material=null,grade=null,...extra}={}){
  const value={type,baseType,name,isSpecific:false,isOfType(...types){return types.includes(this.type);},
    system:{runes,material:{type:material},grade},...extra};
  // Foundry defines _source as a read-only, non-configurable property holding a sealed object.
  return Object.defineProperty(value,'_source',{value:Object.seal({name}),writable:false,enumerable:false});
}
const potency={potency:1,striking:0,property:[]};
// The original generates a name for runed and shield as they are, and for custom and specific once they are
// read as base-named and not specific. The mundane ones get their own name back whatever the early returns are.
const samples=()=>({custom:item('weapon','sword','Heirloom blade',{runes:potency}),specific:item('weapon','sword','Sword',{runes:potency,isSpecific:true}),
  runed:item('weapon','sword','Sword',{runes:potency}),shield:item('shield','shield','Shield',{runes:{reinforcing:1}}),
  baseless:item('weapon',null,'Fist'),plain:item('armor','plate','Plate'),mundaneCustom:item('weapon','sword','Old blade'),
  mundaneSpecific:item('weapon','sword','Sword',{isSpecific:true}),feat:item('feat',null,'Feat')});
// Every outcome is reported once as the returned object; the module API clones it, so it must stay cloneable.
function install({runtime},options){
  const reports=[],result=installItemNameFastPath({runtime,report:value=>reports.push(value),...options});
  assert.equal(reports.length,1);assert.equal(reports[0],result);
  assert.deepEqual(structuredClone(result),result);
  return result;
}
const installedAfter=checked=>({feature:'itemNames',status:'installed',checked,verified:['noBase','specific','customName'],stats:{fast:0,delegated:0}});
const rejected=(w,checked,detail)=>{
  assert.deepEqual(install(w),{feature:'itemNames',status:'behavior-mismatch',checked,detail});
  assert.equal(w.target.generateItemName,w.original);
};

for(const version of VERSIONS){
  test(`official ${version}: every claim is verified and the fast path is installed with fresh counters`,()=>{
    const s=samples(),w=world({version,actors:[[s.custom,s.baseless,s.feat],[s.runed]],items:[s.specific,s.plain,s.shield]});
    assert.equal(s.runed.name,prefix+'Potency:Sword');assert.equal(s.shield.name,prefix+'Reinforcing:Shield');
    const result=install(w),fast=w.target.generateItemName;
    assert.deepEqual(result,installedAfter(6));assert.notEqual(fast,w.original);
    assert.equal(fast.call(w.target,s.custom),'Heirloom blade');assert.deepEqual(result.stats,{fast:1,delegated:0});
    assert.equal(fast.call(w.target,s.runed),prefix+'Potency:Sword');assert.deepEqual(result.stats,{fast:1,delegated:1});
  });
  test(`official ${version}: one item with runes, material or grade is enough, whatever its name or kind`,()=>{
    const s=samples(),generating={...s,silver:item('weapon','sword','Moon blade',{material:'silver'}),graded:item('armor','plate','Plate',{grade:'commercial'})};
    for(const key of ['custom','specific','runed','shield','silver','graded']){
      const w=world({version,actors:[[s.mundaneCustom,s.baseless]],items:[s.mundaneSpecific,generating[key],s.plain]});
      assert.deepEqual(install(w),installedAfter(5),key);
    }
  });
}

const earlyReturn='if (!a || !(a in t) || e.isSpecific || i !== _loc(t[a] ?? ``)) return e.name;';
// affected() builds a world item such an upstream would rename.
const upstreamChanges=[
  {change:'specific items get generated names',claim:'specific',edit:source=>source.replace(' || e.isSpecific',''),
    affected:()=>item('weapon','sword','Sword',{runes:potency,isSpecific:true}),generated:prefix+'Potency:Sword',stable:true},
  {change:'custom names are replaced by generated names',claim:'customName',edit:source=>source.replace(' || i !== _loc(t[a] ?? ``)',''),
    affected:()=>item('weapon','sword','Heirloom blade',{runes:potency}),generated:prefix+'Potency:Sword',stable:true},
  // Dropping only "!a ||" changes nothing, because an empty base is not in any map either.
  {change:'items without a base type get generated names',claim:'noBase',
    edit:source=>source.replace(earlyReturn,'if (a && (!(a in t) || e.isSpecific || i !== _loc(t[a] ?? ``))) return e.name;'),
    affected:()=>item('weapon',null,'Fist',{runes:potency}),generated:prefix+'Potency:Fist',stable:false}
];
for(const {change,claim,edit,affected,generated,stable} of upstreamChanges){
  test(`an upstream where ${change} is rejected by the ${claim} claim when no world item shows the difference`,()=>{
    const s=samples(),runed=Object.assign(s.runed,{uuid:'Item.runed'});
    const w=world({edit,actors:[[s.mundaneSpecific,s.mundaneCustom,s.baseless,s.plain]],items:[runed]});
    rejected(w,5,{uuid:'Item.runed',type:'weapon',claim});
    // These items agree as prepared and after the next data preparation resets their names.
    const direct=createItemNameFastPath(w.original,w.runtime);
    for(const value of [s.mundaneSpecific,s.mundaneCustom,s.baseless,s.plain,runed]){
      assert.equal(direct(value),w.original(value));
      value.name=value._source.name;assert.equal(direct(value),w.original(value));
    }
  });
  test(`an upstream where ${change} is rejected on a world item it renames, although that item already carries the new name`,()=>{
    const s=samples(),renamed=Object.assign(affected(),{id:'w1'}),w=world({edit,actors:[[s.plain,s.baseless]],items:[renamed,s.runed]});
    assert.equal(renamed.name,generated);
    rejected(w,3,{uuid:'w1',type:'weapon'});
  });
  // An upstream that feeds the current name back into the generated one gives itself away on prepared items.
  if(stable)test(`comparing outputs on the real prepared items cannot tell that ${change}`,()=>{
    const s=samples(),renamed=affected(),prepared=[s.plain,s.baseless,renamed,s.runed],w=world({edit,items:prepared});
    const direct=createItemNameFastPath(w.original,w.runtime);
    for(const value of prepared)assert.equal(direct(value),w.original(value));
    // The difference only shows when the next data preparation resets the name to its source.
    renamed.name=renamed._source.name;
    assert.equal(direct(renamed),renamed._source.name);assert.equal(w.original(renamed),generated);
  });
}

test('names stored as "<base label> <English name>" still yield a baseline, and a looser upstream comparison is caught',()=>{
  const translated=()=>[item('weapon','sword','Sword Longsword',{runes:potency}),item('armor','plate','Plate Full Plate'),
    item('shield','shield','Shield Steel Shield'),item('weapon',null,'Fist Fist')];
  assert.deepEqual(install(world({items:translated()})),installedAfter(4));
  // Only the custom-name variant, which extends the base label, tells a prefix comparison from an exact one here.
  const s=samples(),prefixMatch=source=>source.replace('i !== _loc(t[a] ?? ``)','!i.startsWith(_loc(t[a] ?? ``))');
  rejected(world({edit:prefixMatch,items:[s.baseless,Object.assign(s.custom,{uuid:'Item.custom'})]}),2,{uuid:'Item.custom',type:'weapon',claim:'customName'});
});

test('without an item the original generates a name for, nothing is verified and nothing is installed',()=>{
  const s=samples();
  for(const [stock,checked] of [[{},0],[{actors:[[s.feat]],items:[s.feat]},0],[{actors:[[s.plain,s.baseless]],items:[s.mundaneCustom,s.mundaneSpecific]},4]]){
    const w=world(stock);
    assert.deepEqual(install(w),{feature:'itemNames',status:'skipped',reason:'no-generated-name-sample',checked});
    assert.equal(w.target.generateItemName,w.original);
  }
});

test('an upstream that writes to the item is rejected and leaves world item names alone',()=>{
  const writes=source=>source.replace('?? ``)) return e.name;','?? ``)) return e.name = e.name;');
  // In the second world every item is custom-named, so the original throws on each of them as it is.
  for(const [keys,checked] of [[['plain','runed','mundaneSpecific','baseless'],2],[['custom','mundaneCustom'],0]]){
    const s=samples(),stock=keys.map(key=>Object.assign(s[key],{uuid:`Item.${key}`})),w=world({edit:writes,items:stock}),names=stock.map(value=>value.name);
    const result=install(w),{error,...detail}=result.detail,generating=keys.find(key=>['runed','custom'].includes(key));
    assert.equal(result.status,'behavior-mismatch');assert.equal(result.checked,checked);
    assert.deepEqual(detail,{uuid:`Item.${generating}`,type:'weapon',claim:'noBase'});assert.match(error,/^TypeError/);
    assert.equal(w.target.generateItemName,w.original);assert.deepEqual(stock.map(value=>value.name),names);
  }
});

const throwsFor=condition=>source=>source.replace('{',`{\n\tif (${condition}) throw new Error(\`unsupported\`);`);
test('a claim the original only ever throws on is reported instead of skipped',()=>{
  const s=samples(),w=world({edit:throwsFor('e.isSpecific'),items:[s.plain,Object.assign(s.custom,{uuid:'Item.custom'})]});
  rejected(w,2,{uuid:'Item.custom',type:'weapon',claim:'specific',error:'Error: unsupported'});
});

test('a claim that cannot be checked on one generated-name sample is checked on the next, up to twenty samples',()=>{
  const runedArmor=index=>item('armor','plate','Plate',{runes:{potency:1,resilient:0,property:[]},uuid:`Item.armor${index}`});
  const stocked=armors=>{
    const s=samples();
    return world({edit:throwsFor('e.isSpecific && e.type === `armor`'),
      items:[s.plain,s.mundaneCustom,...Array.from({length:armors},(_,index)=>runedArmor(index+1)),s.runed]});
  };
  assert.deepEqual(install(stocked(1)),installedAfter(4));
  assert.deepEqual(install(stocked(19)),installedAfter(22));
  rejected(stocked(20),23,{uuid:'Item.armor20',type:'armor',claim:'specific',error:'Error: unsupported'});
});

test('a sample the original cannot name is skipped and the rest are still checked',()=>{
  const s=samples(),w=world({items:[s.runed,s.custom,s.plain]});
  delete s.runed.system;assert.throws(()=>w.original(s.runed));
  assert.deepEqual(install(w),installedAfter(2));
});

test('a fast path that fails where the original succeeds is a mismatch',()=>{
  // The original settles an unlisted base type before it would read isSpecific; the fast path reads isSpecific first.
  const s=samples(),odd=Object.defineProperty(item('weapon','missing','Odd blade',{id:'w0'}),'isSpecific',{get(){throw new Error('no data');}});
  const own=world({items:[s.runed,odd]});
  // The official function keeps its own globals; a runtime without CONFIG fails the lookup a based item needs.
  const lookup=world({items:[s.baseless,Object.assign(s.custom,{id:'w1'})]});
  // A specific item never reaches the lookup as it is; building its baseline does.
  const baseline=world({items:[Object.assign(s.specific,{id:'w2'})]});
  for(const [w,runtime,checked,uuid,message] of [[own,own.runtime,2,'w0',/^Error: no data$/],
    [lookup,{game:lookup.runtime.game},2,'w1',/^TypeError/],[baseline,{game:baseline.runtime.game},1,'w2',/^TypeError/]]){
    const result=install({runtime}),{error,...detail}=result.detail;
    assert.equal(result.status,'behavior-mismatch');assert.equal(result.checked,checked);
    assert.deepEqual(detail,{uuid,type:'weapon'});assert.match(error,message);
    assert.equal(w.target.generateItemName,w.original);
  }
});

test('once every claim is verified no further baselines are built',()=>{
  const s=samples(),stock=[s.custom,s.specific,s.mundaneCustom],w=world({items:stock}),official=w.original;let calls=0;
  w.target.generateItemName=function(value){calls++;return official.call(this,value);};
  assert.deepEqual(install(w),installedAfter(3));
  // One call per item as it is, none of which the fast path delegates, plus the baseline and three variants of the first.
  assert.equal(calls,stock.length+4);
});

test('actor inventories are sampled before world items, and sampling stops at the limit',()=>{
  const seen=new Set();
  const tracked=(label,...args)=>Object.assign(item(...args),{label,isOfType(...types){seen.add(this.label);return types.includes(this.type);}});
  const stocked=()=>{
    const w=world({actors:[[tracked('feat','feat',null,'Feat'),tracked('first','weapon',null,'Fist')],[tracked('second','weapon','sword','Sword',{runes:potency})]],
      items:[tracked('third','weapon','sword','Heirloom blade'),tracked('fourth','armor','plate','Plate')]});
    seen.clear();return w;
  };
  assert.deepEqual(install(stocked(),{limit:2}),installedAfter(2));
  assert.deepEqual([...seen],['feat','first','second']);
  assert.deepEqual(install(stocked()),installedAfter(4));assert.deepEqual([...seen],['feat','first','second','third','fourth']);
  assert.deepEqual(install(stocked(),{limit:1}),{feature:'itemNames',status:'skipped',reason:'no-generated-name-sample',checked:1});
});

test('value-iterating collections, plain maps, arrays and missing collections are all read',()=>{
  const s=samples(),w=world({items:[s.custom,s.runed]}),game=w.runtime.game;
  for(const [actors,items,checked] of [
    [new Collection([['a',{items:new Collection([['i',s.custom]])}]]),new Collection([['w',s.runed]]),2],
    [new Map([['a',{items:new Map([['i',s.custom]])}]]),new Map([['w',s.runed]]),2],
    [[{items:[s.custom]},{},null],[null,s.runed],2],
    [undefined,[s.custom],1],[[{items:[s.custom]}],undefined,1]
  ]){
    Object.assign(game,{actors,items});w.target.generateItemName=w.original;
    assert.deepEqual(install(w),installedAfter(checked));
  }
});

test('getters and methods backed by private fields still run on the real item under the probe',()=>{
  class Weapon{
    #data;
    constructor(name,data){
      this.name=name;this.#data=data;this.system={runes:potency,material:{type:null},grade:null};
      Object.defineProperty(this,'_source',{value:Object.seal({name}),writable:false,enumerable:false});
    }
    get type(){return this.#data.type;}
    get baseType(){return this.#data.baseType;}
    get isSpecific(){return this.#data.isSpecific;}
    isOfType(...types){return types.includes(this.#data.type);}
  }
  const data={type:'weapon',baseType:'sword',isSpecific:false};
  const w=world({items:[new Weapon('Heirloom blade',data),new Weapon('Sword',data),new Weapon('Sword',{...data,isSpecific:true})]});
  assert.deepEqual(install(w),installedAfter(3));
});

test('a probed item still passes the checks upstream uses to tell documents from plain data',()=>{
  class Physical{}
  const s=samples(),stock=[s.custom,s.plain].map(value=>Object.setPrototypeOf(value,Physical.prototype)),w=world({items:stock}),official=w.original;
  w.target.generateItemName=function(value){
    if(!('_source' in value)||!(value instanceof Physical))throw new Error('not a document');
    return official.call(this,value);
  };
  assert.deepEqual(install(w),installedAfter(2));
});

test('the original is called as a method of game.pf2e.system by the probe and by the installed fast path',()=>{
  const s=samples(),w=world({items:[s.custom,s.runed]}),official=w.original;
  w.target.generateItemName=function(value){assert.equal(this,w.target);return official.call(this,value);};
  const result=install(w);assert.deepEqual(result,installedAfter(2));
  assert.equal(w.target.generateItemName(s.runed),prefix+'Potency:Sword');assert.deepEqual(result.stats,{fast:0,delegated:1});
});

test('other systems are left alone without reading the naming function',()=>{
  const w=world({systemId:'sf2e',items:[samples().custom]});let reads=0;
  Object.defineProperty(w.target,'generateItemName',{get(){reads++;return w.original;},set(){assert.fail('replaced');}});
  assert.deepEqual(install(w),{feature:'itemNames',status:'unsupported-system'});assert.equal(reads,0);
  assert.deepEqual(install({runtime:{}}),{feature:'itemNames',status:'unsupported-system'});
});

test('a runtime without a generateItemName function is unsupported',()=>{
  for(const pf2e of [undefined,{},{system:{}},{system:{generateItemName:'name'}}]){
    const w=world({items:[samples().custom]});w.runtime.game.pf2e=pf2e;
    assert.deepEqual(install(w),{feature:'itemNames',status:'unsupported-runtime'});
  }
});

test('a second install on the same system keeps the first fast path and its counters',()=>{
  const s=samples(),w=world({items:[s.custom,s.runed]}),first=install(w),fast=w.target.generateItemName;
  fast.call(w.target,s.custom);
  assert.deepEqual(install(w),{feature:'itemNames',status:'already-installed'});
  assert.equal(w.target.generateItemName,fast);assert.deepEqual(first.stats,{fast:1,delegated:0});
});
