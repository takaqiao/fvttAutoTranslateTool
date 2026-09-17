import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';
import path from 'node:path';
let api={};try{api=await import('../scripts/metapower/damage.mjs')}catch(e){if(e.code!=='ERR_MODULE_NOT_FOUND')throw e}

// Native source remains outside the repository. This optional integration harness
// loads actual dice classes from the locally installed, licensed application.
// No paid descriptions or private actor fixtures are used or copied.
async function loadNativeDice(){
 const context=vm.createContext({console,structuredClone,deepClone:structuredClone,_loc:s=>s,ErrorPF2e:Error,tupleHasValue:(a,v)=>a.includes(v),N:v=>!!v&&typeof v==='object'&&!Array.isArray(v),v:x=>!!x&&typeof x==='object',o:x=>x!==null&&x!==undefined});
 vm.runInContext(`globalThis.foundry={dice:{terms:{}},utils:{deepClone},applications:{}};globalThis.CONFIG={Dice:{rolls:[],terms:{},termTypes:{},fulfillment:{methods:{},defaultMethod:'random'}},PF2E:{damageTypes:{electricity:'Electricity',fire:'Fire',cold:'Cold',bleed:'Bleed',vitality:'Vitality',void:'Void',untyped:'Untyped'},damageRollFlavors:{},materialDamageEffects:{silver:'Silver','cold-iron':'Cold iron'}}};Number.isNumeric=n=>typeof n==='number'&&Number.isFinite(n);Math.clamp=(n,a,b)=>Math.min(Math.max(n,a),b);`,context);
 for(const [file,name] of [['roll.mjs','Roll'],['terms/term.mjs','RollTerm'],['terms/numeric.mjs','NumericTerm'],['terms/operator.mjs','OperatorTerm'],['terms/dice.mjs','DiceTerm'],['terms/die.mjs','Die'],['terms/function.mjs','FunctionTerm'],['terms/pool.mjs','PoolTerm']]){
  const source=(await readFile(path.join(process.env.FVTT_NATIVE_APP,'client/dice',file),'utf8')).replace(/^import .*?;\r?\n/gm,'').replace('export default class '+name,'globalThis.'+name+'=class '+name);
  vm.runInContext(source,context,{filename:file});
  if(name!=='Roll')vm.runInContext(`foundry.dice.terms.${name}=${name};CONFIG.Dice.termTypes.${name}=${name};`,context);
 }
 const source=await readFile(process.env.PF2E_NATIVE_BUNDLE,'utf8');
 const slice=(start,end)=>{const a=source.indexOf(start),b=source.indexOf(end,a);assert.ok(a>=0&&b>a,'Native class boundaries must be reviewed when system changes');return source.slice(a,b)};
 vm.runInContext(slice('var tn = foundry.dice.terms','function nextDamageDieSize'),context);
 vm.runInContext(slice('function simplifyTerm','function processBaseDamage'),context);
 vm.runInContext(slice('var sn = foundry.dice.terms','var un = class PersistentDamageEditor'),context);
 await vm.runInContext('Promise.resolve()',context);
 vm.runInContext(`CONFIG.Dice.rolls=[Roll,cn,ln];CONFIG.Dice.terms.d=Die;for(const C of [nn,rn,IntermediateDie,InstancePool])CONFIG.Dice.termTypes[C.name]=C;globalThis.native={DamageRoll:cn,DamageInstance:ln,InstancePool,NumericTerm,Die,ArithmeticExpression:nn};`,context);
 return context.native;
}
const nativeEnabled=!!(process.env.FVTT_NATIVE_APP&&process.env.PF2E_NATIVE_BUNDLE);
const native=nativeEnabled?await loadNativeDice():null;
const nativeTest=(name,fn)=>test(name,{skip:!nativeEnabled},fn);
const json=x=>JSON.parse(JSON.stringify(x));
function makeRoll(specs,options={}){
 const instances=specs.map(({value=0,die,flavor='electricity',options:instanceOptions={},critical=false})=>{
  let term=native.NumericTerm.fromData({class:'NumericTerm',number:value,evaluated:true});
  if(die){const d=native.Die.fromData({class:'Die',number:1,faces:6,results:[{result:die,active:true}],evaluated:true,options:{}});term=native.ArithmeticExpression.fromData({class:'ArithmeticExpression',operator:'+',operands:[d.toJSON(),term.toJSON()],evaluated:true,options:{}})}
  if(critical)term=native.ArithmeticExpression.fromData({class:'ArithmeticExpression',operator:'*',operands:[{class:'NumericTerm',number:2,evaluated:true},term.toJSON()],evaluated:true,options:{crit:2}});
  const instance=native.DamageInstance.fromTerms([term],{flavor,...instanceOptions});
  instance.critRule=instanceOptions.critRule??null;
  return instance;
 });
 return native.DamageRoll.fromTerms([native.InstancePool.fromRolls(instances)],options);
}
test('damage API rejects an unevaluated or non-native roll before changing it',()=>{
 assert.equal(typeof api.convertSiphonRoll,'function');
 const roll={terms:[],_evaluated:false,options:{original:true}};
 assert.throws(()=>api.convertSiphonRoll(roll),/native|evaluated|DamageRoll/i);assert.deepEqual(roll,{terms:[],_evaluated:false,options:{original:true}});
});
nativeTest('native fixture reconstructs evaluated dice and demonstrates the per-instance rounding boundary',()=>{
 const roll=makeRoll([{die:4,value:3},{value:3,flavor:'cold'}]);
 assert.equal(roll.total,10);assert.equal(roll.alter(0.5,0).total,4);assert.equal(roll.dice[0].results[0].result,4);
});
nativeTest('conversion uses the same evaluated roll, keeps dice and modifiers, and removes persistent instances',()=>{
 assert.equal(typeof api.convertSiphonRoll,'function');
 const roll=makeRoll([{die:4,value:3},{value:3,flavor:'cold'},{value:9,flavor:'persistent,fire'}],{traits:['electricity','magical'],rollerId:'synthetic'});
 const originalInstances=roll.instances,originalData=originalInstances.map(i=>json(i.toJSON()));
 assert.equal(api.convertSiphonRoll(roll,{DamageRoll:native.DamageRoll}),roll);
 assert.equal(roll.total,10);assert.equal(roll.instances.length,1);assert.equal(roll.instances[0].type,'untyped');assert.equal(roll.instances[0].persistent,false);
 assert.deepEqual([...roll.instances[0].kinds],['damage']);assert.equal(roll.dice[0].results[0].result,4);assert.equal(roll.dice[0].number,1);
 assert.deepEqual(roll.options.traits,['electricity','magical']);assert.equal(roll.options.rollerId,'synthetic');
 assert.deepEqual(originalInstances.map(i=>json(i.toJSON())),originalData,'retained instance references must not be mutated');
 assert.equal(roll.alter(0.5,0).total,5,'odd original instances must be summed before scaling');
});
nativeTest('reconstruction, repeated conversion, message serialization and native alter retain untyped instances',()=>{
 const roll=makeRoll([{value:7},{value:3,flavor:'fire'}]);api.convertSiphonRoll(roll,{DamageRoll:native.DamageRoll});
 const first=JSON.stringify(roll.toJSON());api.convertSiphonRoll(roll,{DamageRoll:native.DamageRoll});assert.equal(JSON.stringify(roll.toJSON()),first);
 const restored=native.DamageRoll.fromData(roll.toJSON());assert.equal(restored.instances[0].type,'untyped');assert.equal(restored.total,10);
 assert.deepEqual([0,0.5,1,2].map(m=>restored.alter(m,0).alter(0.5,0).total),[0,2,5,10]);
});
nativeTest('vitality and void become damage-only untyped, while healing and persistent damage are removed',()=>{
 const roll=makeRoll([{value:7,flavor:'vitality,damage,healing'},{value:3,flavor:'void,damage'},{value:30,flavor:'vitality,healing'},{value:20,flavor:'bleed'}]);
 api.convertSiphonRoll(roll,{DamageRoll:native.DamageRoll});assert.equal(roll.total,10);assert.deepEqual([...roll.kinds],['damage']);assert.equal(roll.instances.length,1);assert.equal(roll.instances[0].type,'untyped');
});
nativeTest('all removed components produce evaluated zero without rerolling or minimum-one damage',()=>{
 const roll=makeRoll([{value:20,flavor:'persistent,fire'}]);api.convertSiphonRoll(roll,{DamageRoll:native.DamageRoll});
 assert.equal(roll.total,0);assert.equal(roll._evaluated,true);assert.equal(roll.instances[0].persistent,false);assert.equal(roll.instances[0].type,'untyped');assert.equal(roll.alter(0.5,0).total,0);
});
nativeTest('material partitions and bypass metadata stay scoped and critical arithmetic remains intact',()=>{
 const bypass={immunity:{ignore:[],downgrade:[],redirect:[]},resistance:{ignore:[{type:'electricity',max:Infinity}],redirect:[]}};
 const roll=makeRoll([{die:4,value:3,flavor:'electricity,silver',critical:true,options:{critRule:'double-damage'}},{value:3,flavor:'cold'}],{bypass});
 api.convertSiphonRoll(roll,{DamageRoll:native.DamageRoll});
 assert.equal(roll.total,17);assert.equal(roll.instances.length,2);assert.deepEqual([...roll.instances[0].materials],['silver']);assert.deepEqual([...roll.instances[1].materials],[]);
 assert.equal(roll.instances[0].critImmuneTotal,7);assert.equal(roll.options.bypass.resistance.ignore[0].type,'electricity');assert.equal(roll.options.bypass.resistance.ignore[0].max,Infinity);
 assert.equal(roll.instances[0].critRule,'double-damage');
});
nativeTest('precision and critical subtree flavors survive same-material consolidation',()=>{
 const roll=makeRoll([{die:4,value:3,critical:true,options:{critRule:'double-damage'}},{value:3,flavor:'cold',critical:true,options:{critRule:'double-damage'}}]);
 roll.instances[1].head.operands[1].options.flavor='precision';
 api.convertSiphonRoll(roll,{DamageRoll:native.DamageRoll});
 assert.equal(roll.total,20);assert.equal(roll.instances.length,1);assert.equal(roll.instances[0].critImmuneTotal,10);assert.equal(roll.dice[0].results[0].result,4);
 assert.ok(JSON.stringify(roll.toJSON()).includes('precision'));
});
nativeTest('odd total and clamped negative components preserve native final arithmetic',()=>{
 const odd=makeRoll([{value:3},{value:5,flavor:'cold'},{value:1,flavor:'fire'}]);api.convertSiphonRoll(odd,{DamageRoll:native.DamageRoll});
 assert.equal(odd.total,9);assert.equal(odd.instances.length,1);assert.equal(odd.alter(0.5,0).total,4);assert.equal(odd.alter(0.5,0).alter(0.5,0).total,2);
 const clamped=makeRoll([{value:-5},{value:7,flavor:'cold'}]);api.convertSiphonRoll(clamped,{DamageRoll:native.DamageRoll});
 assert.equal(clamped.total,7);assert.equal(clamped.instances.length,2);
});
nativeTest('different non-finite metadata never merges by JSON null coercion',()=>{
 const roll=makeRoll([{value:7,options:{limit:Infinity}},{value:3,options:{limit:null}}]);api.convertSiphonRoll(roll,{DamageRoll:native.DamageRoll});
 assert.equal(roll.instances.length,2);assert.equal(roll.instances[0].options.limit,Infinity);assert.equal(roll.instances[1].options.limit,null);
});
