import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import vm from 'node:vm';
import {buildNativeBridge,buildSharedManualPair} from '../tools/automatic-source-patches/native.mjs';
import {verifyNativeIWRBridge,isVerifiedNativeIWRBridge} from '../scripts/native-iwr-verification.mjs';
import {manualPoolBatchModel} from '../scripts/exploration/manual-pool-model.mjs';
import * as providers from '../scripts/exploration/manual-pool-provider.mjs';
const hash=value=>createHash('sha256').update(value).digest('hex');
const base=readFileSync(process.env.PF2E_NATIVE_BUNDLE),toolbelt=readFileSync(process.env.TOOLBELT_MANUAL_SOURCE);
assert.equal(hash(base),'d63da8312831b84905e6866b1dd3f9d93e95c1012955b0177ad2ce8ccf246157');assert.equal(hash(toolbelt),'2946fa27eaf0963098f9f48ee48f777c5cf65166b56b043de9404f09813c240f');
function fixture(transform=value=>value){
 const game={system:{id:'pf2e',version:'8.5.99'},pf2e:{},modules:new Map()},module={version:'3.99.0',active:true};game.modules.set('pf2e-toolbelt',module);
 const bridged=buildNativeBridge({source:Buffer.concat([base,Buffer.from('\n// changed outside seams\n')]),version:game.system.version});
 const sources=buildSharedManualPair({pf2eSource:bridged.buffer,toolbeltSource:Buffer.concat([toolbelt,Buffer.from('\n// changed outside seams\n')]),pf2eVersion:game.system.version,toolbeltVersion:module.version});
 sources.pf2e=transform(sources.pf2e);
 const text=sources.pf2e.toString(),start=text.indexOf('\tasync applyDamage({')+1,end=text.indexOf('\n\tasync undoDamage(',start),applyDamage=Function('return ({'+text.slice(start,end)+'}).applyDamage')();
 const bridge=Object.freeze({...bridged.descriptor,applyDamage});
 const flat=class {beforePrepareData(){}},rule=class {resolveValue(){}resolveInjectedProperties(){}},predicate=class {test(){}};
 const hooks=[],context=vm.createContext({game,Hooks:{once:(event,fn)=>hooks.push(fn)},FlatModifierRuleElement:flat,Y:rule,Hn:predicate,AutomaticBonusProgression$1:{}});
 vm.runInContext(text.slice(text.indexOf('const __nativeReceiverStacking='),text.indexOf('async function applyDamageFromMessage('))+'\n__nativeManualPoolBatch.install();',context);
 vm.runInContext(sources.toolbelt.toString().split('/* end toolbelt manual pool */')[0],context);for(const hook of hooks)hook();
 return {game,module,bridge,sources,context,verify:()=>verifyNativeIWRBridge({game,source:new Uint8Array(sources.pf2e),getNativeBridge:()=>bridge,hash}),qualify:()=>providers.verifyManualPoolProviders({game,pf2eSource:new Uint8Array(sources.pf2e),toolbeltSource:new Uint8Array(sources.toolbelt),hash})};
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
test('provider DTOs alone do not enable a new shared model',async()=>{
 const f=fixture();assert.equal(manualPoolBatchModel(f.game.pf2e.thirdPartyManualPoolBatch.descriptor),false);assert.equal(providers.isManualPoolProvider(f.module.api.explorationManualPool),false);
});
test('exact current installed observers qualify across version and unrelated changes',async()=>{
 const f=fixture(),proof=await f.qualify();assert.equal(proof.ready,true,proof.reason);
 assert.equal(manualPoolBatchModel(f.game.pf2e.thirdPartyManualPoolBatch.descriptor),true);assert.equal(providers.isManualPoolProvider(f.module.api.explorationManualPool),true);
 assert.equal(manualPoolBatchModel(Object.freeze({...f.game.pf2e.thirdPartyManualPoolBatch.descriptor})),false);
});
test('changed provider object identity invalidates private qualification',async()=>{
 const f=fixture();assert.equal((await f.qualify()).ready,true);const old=f.module;
 f.game.modules.set('pf2e-toolbelt',{...old});assert.equal(providers.isManualPoolProvider(old.api.explorationManualPool),false);assert.equal(manualPoolBatchModel(f.game.pf2e.thirdPartyManualPoolBatch.descriptor),false);
});
test('changed native observer code cannot qualify a self claimed descriptor',async()=>{
 const f=fixture();f.sources.pf2e=Buffer.from(f.sources.pf2e.toString().replace('attempted.add(key);if(participants.length!==1)','if(participants.length!==1)'));
 assert.equal((await f.qualify()).ready,false);assert.equal(manualPoolBatchModel(f.game.pf2e.thirdPartyManualPoolBatch.descriptor),false);
});
test('changed receiver transport source blocks shared qualification',async()=>{
 const f=fixture();f.sources.toolbelt=Buffer.from(f.sources.toolbelt.toString().replace('e(c,s)','e(c)'));
 assert.equal((await f.qualify()).ready,false);assert.equal(providers.isManualPoolProvider(f.module.api.explorationManualPool),false);
});
test('changing provider documents during hash awaits cannot issue qualification',async()=>{
 const f=fixture();let once=false;
 const proof=await providers.verifyManualPoolProviders({game:f.game,pf2eSource:f.sources.pf2e,toolbeltSource:f.sources.toolbelt,hash:async bytes=>{if(!once){once=true;f.game.system={...f.game.system}}return hash(bytes)}});
 assert.equal(proof.ready,false);assert.equal(manualPoolBatchModel(f.game.pf2e.thirdPartyManualPoolBatch.descriptor),false);
});
test('a failed fresh source recheck revokes a previously issued shared qualification',async()=>{
 const f=fixture();assert.equal((await f.qualify()).ready,true);
 f.sources.toolbelt=Buffer.from(f.sources.toolbelt.toString().replace('e(c,s)','e(c)'));
 assert.equal((await f.qualify()).ready,false);assert.equal(manualPoolBatchModel(f.game.pf2e.thirdPartyManualPoolBatch.descriptor),false);assert.equal(providers.isManualPoolProvider(f.module.api.explorationManualPool),false);
});
test('missing causal registration cannot qualify an otherwise complete observer',async()=>{
 const f=fixture();f.sources.pf2e=Buffer.from(f.sources.pf2e.toString().replace(' __nativeManualPoolStaticReceiver.register(construct,this,r,this.actor.synthetics.modifiers[r]);',''));
 assert.equal((await f.qualify()).ready,false);
});
test('a wrong static model declaration cannot qualify even with authentic observer code',async()=>{
 const f=fixture(source=>Buffer.from(source.toString().replace('"model":"numeric-static-reception.v1"','"model":"unknown"')));
 assert.equal((await f.qualify()).ready,false);assert.equal(manualPoolBatchModel(f.game.pf2e.thirdPartyManualPoolBatch.descriptor),false);
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
