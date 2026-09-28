import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {readFile} from 'node:fs/promises';

const source=await readFile(new URL('./generate-item-name-original.js.txt',import.meta.url),'utf8');
const {createItemNameFastPath}=process.env.PERF_LAB_BASELINE==='1'
  ? {createItemNameFastPath:original=>original}
  : await import('../scripts/patches/item-name.mjs');
function environment(){
  let enumerations=0;
  const weapons={sword:'SwordLabel',collision:'WeaponCollision'},shields={shield:'ShieldLabel',collision:'ShieldCollision'};
  const proxy=value=>new Proxy(value,{ownKeys(target){enumerations++;return Reflect.ownKeys(target);}});
  const globals={CONFIG:{PF2E:{baseArmorTypes:{plate:'PlateLabel'},baseWeaponTypes:proxy(weapons),baseShieldTypes:proxy(shields),preciousMaterials:{silver:'Silver'},grades:{commercial:'Commercial'}}},
    Kn:{armor:{property:{},resilient:{}},weapon:{property:{},striking:{}}},_s:{},o:Boolean,
    _loc:(key,values)=>values?`${key}:${values.base}`:key,
    AutomaticBonusProgression:{isEnabled:()=>false}};
  const original=vm.runInNewContext(`(${source})`,globals);
  return {original,fast:createItemNameFastPath(original,globals),weapons,shields,enumerations:()=>enumerations,reset:()=>{enumerations=0;}};
}
function item(type='weapon',baseType=null,name='Personal item',extra={}){
  return {type,baseType,name,_source:{name},isSpecific:false,isOfType(...types){return types.includes(this.type);},system:{runes:{potency:0,striking:0,property:[]},material:{type:null},grade:null},...extra};
}

// Catches retaining the full map-spread on the basic-unarmed/no-base fast path.
test('a weapon without a base returns its current name without enumerating type maps',()=>{
  const e=environment();assert.equal(e.fast(item()),'Personal item');assert.equal(e.enumerations(),0);
});
// Catches retaining full map-spread for specific and custom-named physical items.
test('specific and custom-named weapons do not enumerate type maps',()=>{
  const e=environment();assert.equal(e.fast(item('weapon','sword','Custom sword')),'Custom sword');
  assert.equal(e.fast(item('weapon','sword','SwordLabel',{isSpecific:true})),'SwordLabel');assert.equal(e.enumerations(),0);
});
test('all actual original naming branches preserve their returned text',()=>{
  const e=environment();
  const cases=[item(),item('weapon','sword','Custom sword'),item('weapon','missing','Unknown'),item('weapon','sword','SwordLabel'),
    item('weapon','shield','ShieldLabel'),item('armor','plate','PlateLabel'),item('shield','shield','ShieldLabel'),item('feat',null,'A feat'),
    item('weapon','collision','WeaponCollision'),item('weapon','collision','ShieldCollision'),
    item('weapon','sword','SwordLabel',{system:{runes:{potency:1,striking:0,property:[]},material:{type:null},grade:null}}),
    item('weapon','sword','SwordLabel',{system:{runes:{potency:0,striking:0,property:[]},material:{type:'silver'},grade:'commercial'}})];
  for(const fixture of cases)assert.equal(e.fast(fixture),e.original(fixture),JSON.stringify(fixture));
});
test('mapping edits and shield-over-weapon precedence are read anew on each call',()=>{
  const e=environment(),fixture=item('weapon','collision','ShieldCollision');
  assert.equal(e.fast(fixture),e.original(fixture));e.shields.collision='Changed';
  e.reset();assert.equal(e.fast(fixture),'ShieldCollision');assert.equal(e.enumerations(),0);
  delete e.shields.collision;assert.equal(e.fast(item('weapon','collision','WeaponCollision')),e.original(item('weapon','collision','WeaponCollision')));
});
test('merged weapon maps ignore inherited and non-enumerable entries just like object spread',()=>{
  const e=environment();Object.setPrototypeOf(e.weapons,{inherited:'Inherited'});Object.defineProperty(e.shields,'hidden',{value:'Hidden',enumerable:false});
  for(const fixture of [item('weapon','inherited','Inherited'),item('weapon','hidden','Hidden'),item('weapon','toString','Custom')])assert.equal(e.fast(fixture),e.original(fixture));
});
test('a present undefined mapping and an empty original name still use original naming rules',()=>{
  const e=environment();e.weapons.odd=undefined;
  const fixture=item('weapon','odd','');assert.equal(e.fast(fixture),e.original(fixture));
});
test('invalid arguments keep the original failure behavior',()=>{
  const e=environment();for(const fixture of [null,{},undefined]){assert.throws(()=>e.original(fixture));assert.throws(()=>e.fast(fixture));}
});
