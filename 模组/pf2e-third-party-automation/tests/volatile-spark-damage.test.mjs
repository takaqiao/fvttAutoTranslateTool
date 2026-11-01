import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
let api={};try{api=await import('../scripts/volatile-spark-damage.mjs')}catch(error){if(error.code!=='ERR_MODULE_NOT_FOUND')throw error}
const NS='pf2e-third-party-automation';
class Resistance{
 constructor({type,value,except=null}){Object.assign(this,{type,value,except,exceptions:[],label:type,applicationLabel:type,typeLabel:type});}
 test(options){return !options.has?.(this.except)&&[...options].some(o=>o===`damage:type:${this.type}`||this.type==='all-damage');}
 getDoubledValue(){return this.value;}
}
class Immunity extends Resistance{}
const source=fs.readFileSync(process.env.PF2E_NATIVE_861_BUNDLE,'utf8');
const begin=source.indexOf('function applyIWR('),end=source.indexOf('var ActorConditions',begin);
assert(begin>0&&end>begin);
const native=vm.createContext({Resistance,Immunity,game:{pf2e:{settings:{iwr:true}}},Wt:{CRITICAL_SUCCESS:3},o:x=>!!x,tupleHasValue:(xs,x)=>xs.includes(x),sn:{fromDamageType:type=>['piercing','bludgeoning','slashing'].includes(type)?'physical':'energy'}});
vm.runInContext(source.slice(begin,end),native);
const applyIWR=vm.runInContext('applyIWR',native);
function fixture({immunities=[],weaknesses=[],resistances=[['fire',10],['piercing',2],['bludgeoning',4]]}={}){
 const actor={uuid:'Actor.target',attributes:{immunities,weaknesses,resistances:resistances.map(([type,value])=>new Resistance({type,value}))},isAffectedBy:()=>true};
 const damage={total:20,options:{},instances:[{type:'fire',total:20,persistent:false,options:{},formalDescription:new Set(['damage','damage:type:fire','damage:category:energy']),critImmuneTotal:20,componentTotal:()=>0}]};
 const proof={activityMessageId:'use',targetUuid:'Scene.s.Token.t',spellUuid:'Actor.pc.Item.spell',weaponUuid:'Actor.pc.Item.gun',types:['fire'],mixedTypes:[]};
 const message={id:'damage',flags:{[NS]:{volatileSparkSpellDamage:proof,spellCombinationDamage:{activityMessageId:'use',targetUuid:proof.targetUuid,parts:[{itemUuid:proof.spellUuid}],attacks:[{weaponUuid:proof.weaponUuid}]}}}};
 const game={messages:new Map([[message.id,message]]),pf2e:{settings:{iwr:true}}};
 assert.equal(typeof api.createVolatileSparkDamage,'function');
 const handler=api.createVolatileSparkDamage({game,getRollContext:()=>({messageId:'damage',rollIndex:0})});
 const params={damage,token:{uuid:proof.targetUuid,actor},rollOptions:new Set()};
 const run=(next=p=>applyIWR(actor,p.damage,p.rollOptions))=>handler.applyDamage(actor,params,next);
 return {actor,damage,params,message,handler,run};
}
test('Volatile Spark uses native IWR with the lowest resistance and keeps fire weakness',()=>{
 const f=fixture({weaknesses:[new Resistance({type:'fire',value:5})]});assert.equal(f.run().finalDamage,23);assert.equal(f.damage.instances[0].type,'fire');assert.equal(f.actor.attributes.resistances[0].value,10);
});
test('original fire immunity remains and piercing weakness is not introduced',()=>{
 const f=fixture({immunities:[new Resistance({type:'fire',value:0})],weaknesses:[new Resistance({type:'piercing',value:12})]});assert.equal(f.run().finalDamage,0);
 const g=fixture({weaknesses:[new Resistance({type:'piercing',value:12})]});assert.equal(g.run().finalDamage,18);
});
test('physical immunity does not prevent selecting its lower resistance',()=>{
 const f=fixture({immunities:[new Resistance({type:'piercing',value:0})]});assert.equal(f.run().finalDamage,18);
});
test('a missing alternate resistance is zero; all-damage resistance remains applicable',()=>{
 assert.equal(fixture({resistances:[['fire',10]]}).run().finalDamage,20);
 assert.equal(fixture({resistances:[['fire',10],['all-damage',6]]}).run().finalDamage,14);
});
for(const changed of ['target','mixed-type','missing-proof','skip-iwr','final'])test(`Volatile Spark does not alter ${changed}`,()=>{
 const f=fixture();if(changed==='target')f.params.token.uuid='Scene.s.Token.other';if(changed==='mixed-type')f.message.flags[NS].volatileSparkSpellDamage.mixedTypes=['fire'];if(changed==='missing-proof')delete f.message.flags[NS].volatileSparkSpellDamage;if(changed==='skip-iwr')f.params.skipIWR=true;if(changed==='final')f.params.final=true;
 assert.equal(f.run().finalDamage,10);
});
test('unrelated damage in the same synchronous call does not inherit the resistance view',()=>{
 const f=fixture();f.run(params=>{assert.equal(applyIWR(f.actor,f.damage,new Set()).finalDamage,10);assert.equal(applyIWR(f.actor,f.damage,params.rollOptions).finalDamage,18)});
});
test('two outstanding native calls restore immediately and cannot restore each other out of order',async()=>{
 const f=fixture(),original=f.actor.attributes.resistances;let finishA,finishB;
 const a=f.run(params=>{assert.equal(applyIWR(f.actor,f.damage,params.rollOptions).finalDamage,18);return new Promise(resolve=>finishA=resolve)});
 assert.equal(f.actor.attributes.resistances,original);
 const b=f.run(params=>{assert.equal(applyIWR(f.actor,f.damage,params.rollOptions).finalDamage,18);return new Promise(resolve=>finishB=resolve)});
 assert.equal(f.actor.attributes.resistances,original);finishB();await b;finishA();await a;assert.equal(f.actor.attributes.resistances,original);
});
test('native 8.6.1 computes IWR before its first await',()=>{
 const begin=source.indexOf('async applyDamage('),end=source.indexOf('async undoDamage(',begin),method=source.slice(begin,end);
 assert(begin>0&&end>begin);assert(method.indexOf('applyIWR(')>0);assert(method.indexOf('applyIWR(')<method.indexOf('await '));
});
test('native exceptions, doubled resistance, and unrelated critical/precision resistances remain native',()=>{
 const f=fixture();f.actor.attributes.resistances[1].except='item:material:silver';f.params.rollOptions.add('item:material:silver');assert.equal(f.run().finalDamage,20);
 const g=fixture();g.actor.attributes.resistances[1].getDoubledValue=options=>options.has('item:trait:magical')?6:2;g.params.rollOptions.add('item:trait:magical');assert.equal(g.run().finalDamage,16);
 for(const type of ['precision','critical-hits']){
  const resistance=new Resistance({type,value:9});g.actor.attributes.resistances.push(resistance);
  g.run(params=>{const prepared=g.actor.attributes.resistances.at(-1);assert.equal(prepared.getDoubledValue(new Set([...params.rollOptions,'damage:type:fire'])),9);});
 }
});
test('a native throw restores the original resistance property descriptor',()=>{
 const f=fixture(),descriptor=Object.getOwnPropertyDescriptor(f.actor.attributes,'resistances');
 assert.throws(()=>f.run(()=>{throw Error('native failure')}),/native failure/);assert.deepEqual(Object.getOwnPropertyDescriptor(f.actor.attributes,'resistances'),descriptor);
});
test('native replacement of derived attributes retains the new resistance list',()=>{
 const f=fixture(),originalAttributes=f.actor.attributes,original=originalAttributes.resistances,replacement=[];
 f.run(()=>{f.actor.attributes={...originalAttributes,resistances:replacement};});
 assert.equal(originalAttributes.resistances,original);assert.equal(f.actor.attributes.resistances,replacement);
});
test('mixed same-type fire receives one native weakness and retains native resistance for explicit manual adjustment',()=>{
 const f=fixture({weaknesses:[new Resistance({type:'fire',value:5})]});f.damage.total=26;f.damage.instances[0].total=26;f.message.flags[NS].volatileSparkSpellDamage.mixedTypes=['fire'];
 const result=f.run();assert.equal(result.finalDamage,21);assert.equal(result.applications.filter(a=>a.category==='weakness').length,1);
});
test('merged firearm keeps its native concussive immunity redirection without changing spell immunity',()=>{
 const f=fixture({immunities:[new Immunity({type:'piercing',value:0}),new Immunity({type:'fire',value:0})]});
 f.damage.instances.unshift({...f.damage.instances[0],type:'piercing',total:10,critImmuneTotal:10,formalDescription:new Set(['damage','damage:type:piercing','damage:category:physical'])});f.damage.total=30;
 f.message.flags[NS].volatileSparkSpellDamage.weaponRedirects={immunity:[{from:'piercing',to:'bludgeoning'}],resistance:[{from:'piercing',to:'bludgeoning'}]};
 assert.equal(f.run().finalDamage,6);assert.equal(f.damage.options.bypass,undefined,'the native view must be removed synchronously');
});
test('damage provenance retains only native firearm redirections for exclusively weapon types',()=>{
 const bypass={immunity:{redirect:[{from:'piercing',to:'bludgeoning'},{from:'bludgeoning',to:'piercing'}]},resistance:{redirect:[{from:'piercing',to:'bludgeoning'}]}};
 const parts=[{item:{type:'weapon'},roll:{instances:[{type:'piercing'}],options:{bypass}}},{item:{type:'spell'},roll:{instances:[{type:'fire'}],options:{}}}];
 assert.deepEqual(api.volatileSparkDamageParts(parts).weaponRedirects,{immunity:[{from:'piercing',to:'bludgeoning'}],resistance:[{from:'piercing',to:'bludgeoning'}]});
 parts[1].roll.instances[0].type='piercing';assert.deepEqual(api.volatileSparkDamageParts(parts).weaponRedirects,{immunity:[],resistance:[]});
});
