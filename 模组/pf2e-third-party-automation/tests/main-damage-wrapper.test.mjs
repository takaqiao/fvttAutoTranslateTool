import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const source=fs.readFileSync(new URL('../scripts/main.mjs',import.meta.url),'utf8');
function wrapperFixture(){
 const start=source.indexOf("libWrapper.register(MODULE_ID,'CONFIG.Actor.documentClass.prototype.applyDamage'");
 const ending="\n },'WRAPPER');",end=source.indexOf(ending,start)+ending.length;assert.ok(start>0&&end>start);
 const original={damage:-9,rollOptions:new Set(['source'])},actor={};let wrapper,writes=0;
 const pass={applyDamage:(_actor,params,next)=>next(params)},native=async()=>{writes++;return actor};
 // The slice runs with these names only; a name that is not listed here throws.
 const context=vm.createContext({MODULE_ID:'pf2e-third-party-automation',libWrapper:{register(_id,path,fn){assert.equal(path,'CONFIG.Actor.documentClass.prototype.applyDamage');wrapper=fn}},
  activityResults:{applyNativeDamage:(_actor,params,native)=>native(params)},salubriousDamage:{async applyDamage(_actor,params,next){await Promise.resolve();return next({...params},()=>{})}},
  cycle:{getRollContext:()=>null,applyDamage:(_actor,next,params)=>next(params)},providers:[],
  runDamagePipeline:({params,apply})=>apply(params),reactionBudget:pass,combinations:pass,shieldAdapter:{...pass,withNativeFrame:(_actor,params,next)=>next(params)},shieldEvents:{wrapNativeDamage:(_actor,params,next)=>next(params)},destructiveBlock:{planFor:()=>null},glimpse:{wrapNativeDamage:(_actor,params,next)=>next(params)},scar:null,report:error=>assert.fail(error.message)});
 vm.runInContext(source.slice(start,end),context);
 return {run:()=>wrapper.call(actor,native,original),writes:()=>writes,actor};
}
test('an ordinary native damage call reaches the original leaf once and keeps its return value',async()=>{
 const f=wrapperFixture();assert.equal(await f.run(),f.actor);assert.equal(f.writes(),1);
});
test('main registers no wrapper on ChatMessage creation',()=>{
 const targets=[...source.matchAll(/libWrapper\.register\(MODULE_ID,(['`])(.*?)\1/g)].map(match=>match[2]);
 assert.equal(targets.length,source.split('libWrapper.register(').length-1);
 assert.ok(targets.includes('CONFIG.Actor.documentClass.prototype.applyDamage'));
 assert.deepEqual(targets.filter(target=>target.includes('ChatMessage')),[]);
});
