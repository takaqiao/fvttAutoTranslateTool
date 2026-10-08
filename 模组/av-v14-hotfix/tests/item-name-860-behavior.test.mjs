import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {readFile} from 'node:fs/promises';
import {createItemNameFastPath} from '../scripts/patches/item-name.mjs';

const prior = await readFile(new URL('./fixtures/generate-item-name-8.5.1.js.txt', import.meta.url), 'utf8');
const source = await readFile(new URL('./fixtures/generate-item-name-8.6.0.js.txt', import.meta.url), 'utf8');
function environment() {
  let enumerations = 0;
  const map = value => new Proxy(value, {ownKeys(target){enumerations++;return Reflect.ownKeys(target);}});
  const runes = {armor:{property:{fortification:{name:'Fortification'}},resilient:{1:{name:'Resilient'}}},
    weapon:{property:{flaming:{name:'Flaming'}},striking:{1:{name:'Striking'}}}};
  const globals = {CONFIG:{PF2E:{baseArmorTypes:{plate:'Plate', 'hide-armor':'Hide'},
    baseWeaponTypes:map({sword:'Sword'}),baseShieldTypes:map({shield:'Shield', 'steel-shield':'Steel'}),
    preciousMaterials:{silver:'Silver'},grades:{commercial:'Commercial'}}},
    Jn:runes,Yn:runes,Es:{1:'Reinforcing'},Ds:{1:'Reinforcing'},o:Boolean,
    _loc:(key,values)=>values ? `${key}:${values.base}` : key,AutomaticBonusProgression:{isEnabled:()=>false}};
  const original = vm.runInNewContext(`(${source})`, globals), old = vm.runInNewContext(`(${prior})`, globals);
  return {original,old,fast:createItemNameFastPath(original,globals),enumerations:()=>enumerations};
}
function item(type,baseType,name,{runes={potency:0,striking:0,property:[]},material=null,grade=null,...extra}={}) {
  return {type,baseType,name,_source:{name},isSpecific:false,isOfType(...types){return types.includes(this.type);},
    system:{runes,material:{type:material},grade},...extra};
}

test('8.6.0 naming differs from 8.5.1 only in two compiler closure names',()=>{
  assert.equal(source.replaceAll('Yn.','Jn.').replaceAll('Ds[','Es['),prior);
});

test('native 8.6.0 still enumerates maps for no-op weapon names and the fast path avoids it',()=>{
  for(const value of [item('weapon',null,'Unarmed'),item('weapon','sword','Custom'),item('weapon','sword','Sword',{isSpecific:true})]){
    const f=environment();assert.equal(f.original(value),value.name);assert.equal(f.enumerations(),2);
    assert.equal(f.fast(value),value.name);assert.equal(f.enumerations(),2);
  }
});

test('8.6.0 original and patched naming preserve generated names and special-material bases',()=>{
  const f=environment(),prefix='PF2E.Item.Physical.GeneratedName.';
  const cases=[
    [item('weapon','sword','Sword'),'Sword'],
    [item('weapon','sword','Sword',{runes:{potency:1,striking:1,property:['flaming']}}),prefix+'PotencyFundamental2OneProperty:Sword'],
    [item('armor','plate','Plate',{runes:{potency:1,resilient:1,property:['fortification']}}),prefix+'PotencyFundamental2OneProperty:Plate'],
    [item('shield','shield','Shield',{runes:{reinforcing:1}}),prefix+'Reinforcing:Shield'],
    [item('weapon','sword','Sword',{material:'silver'}),prefix+'Material:Sword'],
    [item('weapon','sword','Sword',{grade:'commercial'}),prefix+'Grade:Sword'],
    [item('weapon','sword','Sword',{material:'silver',grade:'commercial'}),prefix+'GradeMaterial:Sword'],
    [item('armor','hide-armor','Hide',{material:'silver'}),prefix+'Material:TYPES.Item.armor'],
    [item('shield','steel-shield','Steel',{material:'silver'}),prefix+'Material:TYPES.Item.shield'],
    [item('weapon','missing','Unknown'),'Unknown'],[item('feat',null,'Feat'),'Feat']
  ];
  for(const [value,expected] of cases){
    assert.equal(f.old(value),expected);assert.equal(f.original(value),expected);assert.equal(f.fast(value),expected);
  }
});

test('8.6.0 invalid item arguments retain the original errors',()=>{
  const f=environment();
  for(const value of [null,undefined,{}]){assert.throws(()=>f.original(value));assert.throws(()=>f.fast(value));}
});
