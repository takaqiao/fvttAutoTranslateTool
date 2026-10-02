import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {patchStaticReceiver} from '../tools/native-manual-pool-static-receiver/patch.mjs';
import {verifyNativeIWRBridge,isVerifiedNativeIWRBridge} from '../scripts/native-iwr-verification.mjs';
import {createShieldDamageAdapter} from '../scripts/shield-damage-adapter.mjs';

const BASE_SHA='d63da8312831b84905e6866b1dd3f9d93e95c1012955b0177ad2ce8ccf246157';
const COMPOSITION_SHA='9be357c96dff3d0790edcb0d7889db98cfded0f41f34fd161233ea0bdd0dab11';
const METHOD_SHA='cd1d391b7d4c2c8f3b11b903c477a5e6e330343a94ba51dd5ddebbe610adcb5a';
const OLD_SHA='b4335fa31d1c7b36e100522e6fa3ced7c24075abb14b01a49f16a518e9fa4ec8';
const hash=value=>createHash('sha256').update(value).digest('hex');
function input(name,sha){
 assert.ok(process.env[name],`${name} is required`);
 const bytes=readFileSync(process.env[name]);assert.equal(hash(bytes),sha,name);return bytes;
}
const base=input('PF2E_NATIVE_BUNDLE',BASE_SHA),batchBase=input('PF2E_MANUAL_POOL_BATCH_SOURCE',BASE_SHA);
const old=input('PF2E_NATIVE_850_BUNDLE',OLD_SHA),combined=patchStaticReceiver(batchBase).bytes;
function method(bytes){
 const text=bytes.toString('utf8'),start=text.indexOf('\tasync applyDamage({ damage:')+1,end=text.indexOf('\n\tasync undoDamage(',start);
 assert.ok(start>0&&end>start,'the retained native method must be present');
 return Function('return ({'+text.slice(start,end)+'}).applyDamage')();
}
const baseMethod=method(base),combinedMethod=method(combined),oldMethod=method(old);
const originalSHA={
 '8.5.0':'2929cbcc2e0c27e1e1a921c05b00263e55e113ba3210fcde379af9d2aa61e43d',
 '8.5.1':'8fa38a2fcbf848ad967c75876ca33ebf5a46fb7d6a8cc90d8323c0fb5d471bc7'
};
function fixture(version='8.5.1',applyDamage=combinedMethod){
 const user={id:'gm',isGM:true},game={system:{id:'pf2e',version},user,users:{activeGM:user}};
 const bridge=Object.freeze({version,sourceSHA256:originalSHA[version],protocol:'pf2e-third-party-automation:iwr:2',applyDamage});
 const verify=source=>verifyNativeIWRBridge({game,source:new Uint8Array(source),getNativeBridge:()=>bridge,hash});
 return {game,bridge,verify};
}

test('the fixed shared composition preserves the original IWR method byte for byte',()=>{
 assert.equal(hash(combined),COMPOSITION_SHA);
 assert.equal(Function.prototype.toString.call(combinedMethod),Function.prototype.toString.call(baseMethod));
 assert.equal(hash(Function.prototype.toString.call(combinedMethod)),METHOD_SHA);
});

test('ready verification accepts the exact shared composition and records its actual source SHA',async()=>{
 const f=fixture(),proof=await f.verify(combined);
 assert.equal(proof.ready,true,proof.reason);assert.equal(proof.reason,'verified');
 assert.equal(proof.patchedSHA256,COMPOSITION_SHA);assert.equal(proof.applyDamageSHA256,METHOD_SHA);
 assert.equal(isVerifiedNativeIWRBridge(proof,f.game,f.bridge),true);
});

test('the shared composition retains native IWR observation and shield interception',async()=>{
 const f=fixture(),proof=await f.verify(combined),errors=[],observations=[];
 const adapter=createShieldDamageAdapter({game:f.game,nativeBridgeVerification:proof,getNativeBridge:()=>f.bridge,onError:error=>errors.push(error)});
 assert.equal(adapter.nativeBridgeAvailable(),true);
 assert.equal(adapter.nativeBridgeDiagnostic().verifiedSystemSHA256,COMPOSITION_SHA);
 const actor={uuid:'Actor.patient',type:'character',hitPoints:{value:25,max:25,temp:0},attributes:{hp:{}},heldShield:null,testUserPermission:user=>user===f.game.user};
 const token={uuid:'Scene.scene.Token.patient',id:'patient',parent:{id:'scene'},actor},item={uuid:'Actor.attacker.Item.weapon'};
 const params={damage:{total:15,instances:[]},token,item,rollOptions:new Set(['origin:trait:fiend'])};
 const result={finalDamage:7,applications:[{category:'resistance',type:'spirit damage from fiends',adjustment:-8,ignored:false}],persistent:[]};
 let intercepted=0;adapter.addNativeObserver(view=>observations.push(view));adapter.addNativeInterceptor(({incoming})=>{intercepted++;assert.equal(incoming,7)});
 assert.equal(await adapter.applyDamage(actor,params,p=>adapter.withNativeFrame(actor,p,async checked=>{
  await adapter.nativeDamageIWR(actor,checked,result,checked.rollOptions,{actorDamage:7,shieldDamage:0});return actor;
 })),actor);
 assert.equal(intercepted,1);assert.equal(observations.length,1);assert.equal(observations[0].nativeAmounts.actorDamage,7);assert.deepEqual(errors,[]);
});

test('the original 8.5.1 bridge remains supported with its own actual source SHA',async()=>{
 const f=fixture('8.5.1',baseMethod),proof=await f.verify(base);
 assert.equal(proof.ready,true,proof.reason);assert.equal(proof.patchedSHA256,BASE_SHA);
 assert.equal(proof.applyDamageSHA256,METHOD_SHA);assert.equal(isVerifiedNativeIWRBridge(proof,f.game,f.bridge),true);
});

test('the original 8.5.0 bridge retains exact source and retained-method verification',async()=>{
 const f=fixture('8.5.0',oldMethod),proof=await f.verify(old);
 assert.equal(proof.ready,true,proof.reason);assert.equal(proof.patchedSHA256,OLD_SHA);
 assert.equal(proof.applyDamageSHA256,'fdd09b9ff0acfda43025fb9972e98143a4afa645e97bd6c49b9c4263139952a0');
 assert.equal(isVerifiedNativeIWRBridge(proof,f.game,f.bridge),true);
});

test('one extra byte cannot qualify an otherwise matching shared composition',async()=>{
 const f=fixture(),proof=await f.verify(Buffer.concat([combined,Buffer.from(' ')]));
 assert.equal(proof.ready,false);assert.equal(proof.reason,'unknown-system-source');
 assert.equal(isVerifiedNativeIWRBridge(proof,f.game,f.bridge),false);
});

test('the plain batch composition is not implicitly approved by its retained IWR method',async()=>{
 const {patchNativeManualPoolBatch}=await import('../tools/native-manual-pool-batch/patch.mjs');
 const f=fixture(),proof=await f.verify(patchNativeManualPoolBatch(batchBase).bytes);
 assert.equal(proof.ready,false);assert.equal(proof.reason,'unknown-system-source');
});

test('a changed retained method cannot qualify even with the exact shared source',async()=>{
 const source=Function.prototype.toString.call(combinedMethod),changed=source.replace('actorDamage: x - O - ie','actorDamage: x - O');
 assert.notEqual(changed,source);const f=fixture('8.5.1',Function('return ({'+changed+'}).applyDamage')());
 const proof=await f.verify(combined);assert.equal(proof.ready,false);assert.equal(proof.reason,'native-method-sha-mismatch');
});

test('a bridge from the old profile cannot qualify with the shared 8.5.1 source',async()=>{
 const f=fixture('8.5.1',oldMethod),proof=await f.verify(combined);
 assert.equal(proof.ready,false);assert.equal(proof.reason,'native-method-sha-mismatch');
});

test('a proof for a base bridge cannot authorize a different composition bridge',async()=>{
 const f=fixture('8.5.1',baseMethod),proof=await f.verify(base),other=fixture();
 assert.equal(isVerifiedNativeIWRBridge(proof,f.game,other.bridge),false);
});

test('copied and caller-created composition proofs have no use-site authority',async()=>{
 const f=fixture(),proof=await f.verify(combined);assert.equal(proof.ready,true,proof.reason);
 assert.equal(isVerifiedNativeIWRBridge(Object.freeze({...proof}),f.game,f.bridge),false);
 assert.equal(isVerifiedNativeIWRBridge(Object.freeze({ready:true,bridge:f.bridge,patchedSHA256:COMPOSITION_SHA,applyDamageSHA256:METHOD_SHA}),f.game,f.bridge),false);
});

test('a changed game profile invalidates an issued composition proof',async()=>{
 const f=fixture(),proof=await f.verify(combined);assert.equal(proof.ready,true,proof.reason);
 f.game.system.version='8.5.0';assert.equal(isVerifiedNativeIWRBridge(proof,f.game,f.bridge),false);
});
