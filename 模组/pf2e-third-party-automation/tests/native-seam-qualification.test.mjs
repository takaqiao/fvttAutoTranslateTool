import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import vm from 'node:vm';
import {buildNativeBridge,buildSharedManualPair} from '../tools/automatic-source-patches/native.mjs';
import {verifyNativeIWRBridge,isVerifiedNativeIWRBridge} from '../scripts/native-iwr-verification.mjs';
const hash=value=>createHash('sha256').update(value).digest('hex');
const base=readFileSync(process.env.PF2E_NATIVE_BUNDLE),toolbelt=readFileSync(process.env.TOOLBELT_MANUAL_SOURCE);
assert.equal(hash(base),'d63da8312831b84905e6866b1dd3f9d93e95c1012955b0177ad2ce8ccf246157');assert.equal(hash(toolbelt),'2946fa27eaf0963098f9f48ee48f777c5cf65166b56b043de9404f09813c240f');
function fixture(){
 const game={system:{id:'pf2e',version:'8.5.99'},pf2e:{},modules:new Map()},module={version:'3.99.0',active:true};game.modules.set('pf2e-toolbelt',module);
 const bridged=buildNativeBridge({source:Buffer.concat([base,Buffer.from('\n// changed outside seams\n')]),version:game.system.version});
 const sources=buildSharedManualPair({pf2eSource:bridged.buffer,toolbeltSource:Buffer.concat([toolbelt,Buffer.from('\n// changed outside seams\n')]),pf2eVersion:game.system.version,toolbeltVersion:module.version});
 const text=sources.pf2e.toString(),start=text.indexOf('\tasync applyDamage({')+1,end=text.indexOf('\n\tasync undoDamage(',start),applyDamage=Function('return ({'+text.slice(start,end)+'}).applyDamage')();
 const bridge=Object.freeze({...bridged.descriptor,applyDamage});
 const flat=class {beforePrepareData(){}},rule=class {resolveValue(){}resolveInjectedProperties(){}},predicate=class {test(){}};
 const hooks=[],context=vm.createContext({game,Hooks:{once:(event,fn)=>hooks.push(fn)},FlatModifierRuleElement:flat,Y:rule,Hn:predicate,AutomaticBonusProgression$1:{}});
 vm.runInContext(text.slice(text.indexOf('const __nativeReceiverStacking='),text.indexOf('async function applyDamageFromMessage('))+'\n__nativeManualPoolBatch.install();',context);
 vm.runInContext(sources.toolbelt.toString().split('/* end toolbelt manual pool */')[0],context);for(const hook of hooks)hook();
 return {game,module,bridge,sources,context,verify:()=>verifyNativeIWRBridge({game,source:new Uint8Array(sources.pf2e),getNativeBridge:()=>bridge,hash})};
}
test('a served seam build on a diagnostic new version receives private IWR proof',async()=>{
 const f=fixture(),proof=await f.verify();assert.equal(proof.ready,true,proof.reason);assert.equal(proof.patchedSHA256,hash(f.sources.pf2e));assert.equal(isVerifiedNativeIWRBridge(proof,f.game,f.bridge),true);
 assert.equal(isVerifiedNativeIWRBridge(Object.freeze({...proof}),f.game,f.bridge),false);
});
test('the actual served native method must agree with the retained bridge',async()=>{
 const f=fixture(),source=Buffer.from(f.sources.pf2e.toString().replace('delta: x - O - ie','delta: x - O'));
 const proof=await verifyNativeIWRBridge({game:f.game,source,getNativeBridge:()=>f.bridge,hash});assert.equal(proof.ready,false);
});
test('a claimed method contract cannot approve another live method',async()=>{
 const f=fixture(),other=Object.freeze({...f.bridge,applyDamage:async()=>{}});
 assert.equal((await verifyNativeIWRBridge({game:f.game,source:f.sources.pf2e,getNativeBridge:()=>other,hash})).ready,false);
});
test('a system document replacement invalidates an already issued native proof',async()=>{
 const f=fixture(),proof=await f.verify();assert.equal(proof.ready,true);f.game.system={...f.game.system};assert.equal(isVerifiedNativeIWRBridge(proof,f.game,f.bridge),false);
});
test('an unregistered native Toolbelt forward still invokes its own writer once',()=>{
 const f=fixture();let calls=0;f.context.patient={};f.context.master={isOwner:true};f.context.native=()=>{calls++;return Promise.resolve(f.context.master)};
 assert.equal(vm.runInContext('__toolbeltManualPool.forward(patient,master,{"system.attributes.hp.value":12},{},native,()=>{throw Error("wrong route")})',f.context),null);assert.equal(calls,1);
});
test('replacing the current native class cannot reuse an issued default bridge proof',async()=>{
 const f=fixture(),original=globalThis.CONFIG,Actor=class {};Actor.thirdPartyNativeIWRBridge=f.bridge;
 globalThis.CONFIG={Actor:{documentClass:Actor}};
 try{
  const proof=await verifyNativeIWRBridge({game:f.game,source:f.sources.pf2e,hash});assert.equal(proof.ready,true,proof.reason);
  const replacement=class {};replacement.thirdPartyNativeIWRBridge=f.bridge;globalThis.CONFIG.Actor.documentClass=replacement;
  assert.equal(isVerifiedNativeIWRBridge(proof,f.game,f.bridge),false);
 }finally{if(original===undefined)delete globalThis.CONFIG;else globalThis.CONFIG=original}
});
