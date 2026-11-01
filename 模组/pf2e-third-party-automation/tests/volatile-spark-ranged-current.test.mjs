import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';

assert(process.env.RANGED_COMBAT_CURRENT_ROOT,'RANGED_COMBAT_CURRENT_ROOT is required');
const root=process.env.RANGED_COMBAT_CURRENT_ROOT,manifest=JSON.parse(fs.readFileSync(path.join(root,'module.json'),'utf8'));
assert.equal(manifest.version,'8.0.7');
function fixture({loaded=true,cancel=false}={}){
 const callbacks=new Map(),checks=new Map(),wrappers=new Map(),calls=[];
 const actor={id:'pc'},ammunition={id:'ammo',name:'round',calculateTotalRemainingUses:()=>1,consume(count){calls.push(['consume',count])}};
 const weapon={actor,capacity:2,isCapacity:false,isRepeating:false,expend:1,loadedAmmunition:loaded?[ammunition]:[],selectedLoadedAmmunition:loaded?ammunition:null,remainingCapacity:1};
 const HookManager={register:(key,fn)=>callbacks.set(key,[...callbacks.get(key)??[],fn]),registerCheck:(key,fn)=>checks.set(key,fn),runCheck:(key,args)=>checks.get(key)(args),call:(key,args)=>callbacks.get(key)?.forEach(fn=>fn(args))};
 class Updates{static FAKE_ITEM_ID='FAKE';constructor(a){assert.equal(a,actor)}async commit(){calls.push(['commit'])}delete(){}}
 const context=vm.createContext({console,Set,Map,Hooks:{on(){},callAll(){}},libWrapper:{register:(_id,key,fn)=>wrappers.set(key,fn)},HookManager,Updates,
  isTargetHuntedPrey:()=>false,WeaponSystem:{transformWeapon:()=>weapon},AmmunitionSystem:{localize:key=>key,chooseLoadedAmmunition:async()=>loaded?ammunition:null},ClearJam:{checkJammed:()=>false},Configuration:{preventFiringWithoutLoading:()=>true},Util:{warn(){},getEffect:()=>null},Chat:{post(){}},CHAMBER_LOADED_EFFECT_ID:'chamber',LOADED_EFFECT_ID:'loaded'});
 const load=(file,name)=>{const source=fs.readFileSync(path.join(root,'scripts',file),'utf8').replace(/^import .*;\r?\n/gm,'').replace(`export class ${name}`,`globalThis.${name}=class ${name}`);vm.runInContext(source,context,{filename:file});return context[name]};
 load('ammunition-system/fire-weapon-check.js','FireWeaponCheck').initalise();load('ammunition-system/fire-weapon-processor.js','FireWeaponProcessor').initialise();
 const Core=load('core/core.js','Core');Core.buildAuxiliaryActions=()=>{};Core.initialise();
 const item={id:'gun',actor,type:'weapon'};
 const original=async opts=>wrappers.get('game.pf2e.Check.roll')(async()=>{calls.push(['native']);return cancel?null:{total:20}},null,{actor,item,type:'attack-roll',options:opts.options??new Set()});
 const strike=wrappers.get('CONFIG.PF2E.Actor.documentClasses.character.prototype.prepareStrike')(()=>({item,variants:[{roll:original}],damage(){},critical(){}}),{actor});
 return {strike,calls,context,load,weapon,ammunition,Updates};
}
test('current Ranged Combat native Strike consumes one round and suppresses system consumption',async()=>{
 const f=fixture(),params={options:new Set(['action:spellstrike'])};await f.strike.variants[0].roll(params);
 assert.equal(params.consumeAmmo,false);assert.deepEqual(f.calls,[['native'],['consume',1],['commit']]);
});
for(const mode of ['empty','cancelled'])test(`current Ranged Combat ${mode} native Strike does not consume ammunition`,async()=>{
 const f=fixture({loaded:mode!=='empty',cancel:mode==='cancelled'});assert.equal(await f.strike.variants[0].roll({options:new Set()}),undefined);assert.equal(f.calls.filter(c=>c[0]==='consume'||c[0]==='commit').length,0);
});
test('current Reload.reload loads one round and returns a completion boolean without committing itself',async()=>{
 const f=fixture(),Reload=f.load('ammunition-system/actions/reload.js','Reload');let popped=0,created=0;
 f.weapon.loadedAmmunition=[];f.weapon.createAmmunition=async ammo=>{created++;return ammo};
 Reload.chooseAmmunition=async()=>({pop(count){popped+=count;return {sourceId:'round'}}});Reload.postToChat=()=>{};
 assert.equal(await Reload.reload(f.weapon,new f.Updates(f.weapon.actor),{fullyReload:false}),true);assert.equal(popped,1);assert.equal(created,1);assert.equal(f.calls.length,0);
});
test('current Reload.reload cancelled ammunition choice returns false and performs no write',async()=>{
 const f=fixture(),Reload=f.load('ammunition-system/actions/reload.js','Reload');Reload.chooseAmmunition=async()=>null;
 assert.equal(await Reload.reload(f.weapon,new f.Updates(f.weapon.actor),{fullyReload:false}),false);assert.equal(f.calls.length,0);
});
