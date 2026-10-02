import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

function wrapperFixture({participating=true}={}){
 const source=fs.readFileSync(new URL('../../scripts/main.mjs',import.meta.url),'utf8'),start=source.indexOf("libWrapper.register(MODULE_ID,'CONFIG.Actor.documentClass.prototype.applyDamage'");
 const ending="\n },'WRAPPER');",end=source.indexOf(ending,start)+ending.length;assert.ok(start>0&&end>start);
 const original={damage:-9,rollOptions:new Set(['source'])},actor={},privateFrame={},order=[];let wrapper,writes=0,inside=true;
 const pass={applyDamage:(_actor,params,next)=>next(params)},native=async()=>{writes++;return actor};
 const context=vm.createContext({MODULE_ID:'pf2e-third-party-automation',libWrapper:{register(_id,path,fn){assert.equal(path,'CONFIG.Actor.documentClass.prototype.applyDamage');wrapper=fn}},
  exploration:{manualPoolApplication:{captureFrame(actual,params){assert.equal(actual,actor);assert.equal(params,original);return participating&&inside?privateFrame:null},async applyNativeDamage(actual,params,originalNative,frame){if(frame===privateFrame)throw Error('private-begin-unknown');return originalNative(params)}}},
  activityResults:{applyNativeDamage:(_actor,params,native)=>native(params)},salubriousDamage:{async applyDamage(actual,params,next){inside=false;await Promise.resolve();return next({...params},()=>{})}},
  disruptDamage:{applyDamage:(_actor,params,next)=>next(params,()=>{})},cycle:{getRollContext:()=>null,applyDamage:(_actor,next,params)=>next(params)},providers:[],
  runDamagePipeline:({params,apply})=>apply(params),reactionBudget:pass,shieldAdapter:{...pass,withNativeFrame:(_actor,params,next)=>next(params)},shieldEvents:{wrapNativeDamage:(_actor,params,next)=>next(params)},destructiveBlock:{planFor:()=>null},glimpse:{wrapNativeDamage:(_actor,params,next)=>next(params)},scar:null,report:error=>order.push(error)});
 vm.runInContext(source.slice(start,end),context);
 return {run:()=>wrapper.call(actor,native,original),writes:()=>writes,actor};
}
test('the unique main wrapper retains the synchronous batch frame before any awaited damage middleware',async()=>{
 const f=wrapperFixture();await assert.rejects(f.run(),/private-begin-unknown/);assert.equal(f.writes(),0);
});
test('an unregistered ordinary native damage call keeps the same original leaf and return value',async()=>{
 const f=wrapperFixture({participating:false});assert.equal(await f.run(),f.actor);assert.equal(f.writes(),1);
});
