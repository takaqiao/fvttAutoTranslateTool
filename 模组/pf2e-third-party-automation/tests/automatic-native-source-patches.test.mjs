import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import * as builders from '../tools/automatic-source-patches/native.mjs';
import {patchStaticReceiver} from '../tools/native-manual-pool-static-receiver/patch.mjs';
import {patchToolbeltManualPool} from '../tools/toolbelt-manual-pool/patch.mjs';
import {verifyNativeIWRBridge} from '../scripts/native-iwr-verification.mjs';
import {automaticDescriptor,bridgeStatement} from '../scripts/native-source-shapes.mjs';
import {currentToolbelt} from './toolbelt-current-fixture.mjs';

const hash=value=>createHash('sha256').update(value).digest('hex');
function input(name,sha){assert.ok(process.env[name],`${name} is required`);const bytes=readFileSync(process.env[name]);assert.equal(hash(bytes),sha,name);return bytes}
const pf2e=input('PF2E_NATIVE_BUNDLE','d63da8312831b84905e6866b1dd3f9d93e95c1012955b0177ad2ce8ccf246157');
const toolbelt=input('TOOLBELT_MANUAL_SOURCE','2946fa27eaf0963098f9f48ee48f777c5cf65166b56b043de9404f09813c240f');
const old=input('PF2E_NATIVE_850_BUNDLE','b4335fa31d1c7b36e100522e6fa3ced7c24075abb14b01a49f16a518e9fa4ec8');
function unbridge(bytes){return Buffer.from(bytes.toString().replace(/\tstatic thirdPartyNativeIWRBridge[^\n]+\n/,'').replace(/\t\tconst thirdPartyIWR[^\n]+\n\t\tif\s*\(thirdPartyIWR[^\n]+\n/,''))}
function change(bytes){return Buffer.concat([bytes,Buffer.from('\n// unrelated upstream change\n')])}
const bridge=(source=change(unbridge(pf2e)),version='8.5.99')=>builders.buildNativeBridge({source,version});
const pair=(pf2eSource=bridge().buffer,toolbeltSource=change(toolbelt))=>builders.buildSharedManualPair({pf2eSource,toolbeltSource,pf2eVersion:'8.5.99',toolbeltVersion:'3.99.0'});
test('current authenticated socket aliases prepare a byte stable pair without a version allowlist',()=>{
 const first=pair(bridge().buffer,currentToolbelt()),again=pair(first.pf2e,first.toolbelt);
 assert.equal(again.alreadyPatched,true);assert.deepEqual(again.pf2e,first.pf2e);assert.deepEqual(again.toolbelt,first.toolbelt);
});
for(const [name,before,after]of [
 ['sender','e(c,s)','e(c)'],['type','o.__type__!==t','false'],['active GM','!game.user.isActiveGM','false'],
 ['await decode','await yA(o)','yA(o)'],['GM local sender','e(o,game.userId)','e(o,o.senderId)'],
 ['registration','game.socket.on(`module.${M.id}`,t)','game.socket.off(`module.${M.id}`,t)'],
 ['unregistration','game.socket.off(`module.${M.id}`,t)','game.socket.on(`module.${M.id}`,t)'],
 ['document decoder','return fromUuid(t)','return t']
])test(`current socket ${name} changes refuse both files`,()=>{
 const left=bridge().buffer,right=Buffer.from(currentToolbelt().toString().replace(before,after)),snapshots=[Buffer.from(left),Buffer.from(right)];
 assert.throws(()=>pair(left,right),/toolbelt-.*seam/);assert.deepEqual(left,snapshots[0]);assert.deepEqual(right,snapshots[1]);
});
test('a new version and unrelated bytes preserve the native damage method',()=>{
 const result=bridge();assert.equal(result.status,'patch');assert(Buffer.isBuffer(result.buffer));assert.equal(result.descriptor.version,'8.5.99');
 assert.equal(result.descriptor.sourceSHA256,hash(change(unbridge(pf2e))));
 const text=result.buffer.toString(),start=text.indexOf('\tasync applyDamage({')+1,end=text.indexOf('\n\tasync undoDamage(',start);
 assert.equal(hash(text.slice(start,end)),'cd1d391b7d4c2c8f3b11b903c477a5e6e330343a94ba51dd5ddebbe610adcb5a');
});
test('the older shield variable shape is selected from source rather than version',()=>{
 const result=bridge(change(unbridge(old)));assert.equal(result.status,'patch');assert.match(result.buffer.toString(),/actorDamage: x - D - ie, shieldDamage: ne/);
});
test('a local health delta change is rejected without mutating input',()=>{
 const input=Buffer.from(unbridge(pf2e).toString().replace('delta: x - O - ie','delta: x - O')),before=Buffer.from(input);
 assert.throws(()=>bridge(input),/native-.*seam/);assert.deepEqual(input,before);
});
test('duplicate native methods and partial bridge markers are rejected',()=>{
 assert.throws(()=>bridge(Buffer.concat([unbridge(pf2e),Buffer.from('\n\tasync applyDamage({ damage:')])),/native-.*seam/);
 assert.throws(()=>bridge(Buffer.concat([unbridge(pf2e),Buffer.from('\nconst thirdPartyIWR = 1;')])),/native-.*seam/);
});
test('the shared pair accepts unrelated bytes and diagnostic versions',()=>{
 const result=pair();assert(Buffer.isBuffer(result.pf2e));assert(Buffer.isBuffer(result.toolbelt));assert.equal(result.alreadyPatched,false);
 assert.match(result.pf2e.toString(),/numeric-static-reception.v1/);assert.match(result.toolbelt.toString(),/hpBaselineGuardVersion/);
});
test('repeated bridge and pair preparation is byte stable',()=>{
 const first=bridge(),again=bridge(first.buffer);assert.equal(again.status,'unchanged');assert.deepEqual(again.buffer,first.buffer);
 const firstPair=pair(),second=pair(firstPair.pf2e,firstPair.toolbelt);assert.equal(second.alreadyPatched,true);
 assert.deepEqual(second.pf2e,firstPair.pf2e);assert.deepEqual(second.toolbelt,firstPair.toolbelt);
});
test('legacy exact P3 and T1 require no rewriting',()=>{
 const legacyPF=patchStaticReceiver(pf2e).bytes,legacyTB=patchToolbeltManualPool(toolbelt).bytes;
 const result=builders.buildSharedManualPair({pf2eSource:legacyPF,toolbeltSource:legacyTB,pf2eVersion:'8.5.1',toolbeltVersion:'3.56.5'});
 assert.equal(result.alreadyPatched,true);assert.deepEqual(result.pf2e,legacyPF);assert.deepEqual(result.toolbelt,legacyTB);
 assert.equal(builders.buildNativeBridge({source:legacyPF,version:'8.5.1'}).status,'unchanged');
});
test('one invalid shared member refuses the pair without touching either input',()=>{
 const left=bridge().buffer,right=Buffer.from(toolbelt.toString().replace('this.isValidMaster(e)&&e.update(n)','e.update(n)')),snapshots=[Buffer.from(left),Buffer.from(right)];
 assert.throws(()=>pair(left,right),/toolbelt-.*seam/);assert.deepEqual(left,snapshots[0]);assert.deepEqual(right,snapshots[1]);
});
test('critical static receiver and transport changes cannot acquire a marker',()=>{
 assert.throws(()=>pair(Buffer.from(bridge().buffer.toString().replace('(this.actor.synthetics.modifiers[r] ??= []).push(construct);','(this.actor.synthetics.modifiers[r] ??= []).push(() => null);'))),/native-.*seam/);
 assert.throws(()=>pair(bridge().buffer,Buffer.from(toolbelt.toString().replace('e(c,s)','e(c)'))),/toolbelt-.*seam/);
});
test('partial and duplicate observer installations are refused',()=>{
 assert.throws(()=>pair(Buffer.concat([bridge().buffer,Buffer.from('\nconst __nativeManualPoolBatch=()=>{};')])),/native-.*seam/);
 assert.throws(()=>pair(bridge().buffer,Buffer.concat([toolbelt,Buffer.from('\nconst __toolbeltManualPool=()=>{};')])),/toolbelt-.*seam/);
});
test('an update of either clean partner is restored while the other remains valid',()=>{
 const original=pair();
 const toolUpdated=pair(original.pf2e,change(toolbelt));assert.equal(toolUpdated.alreadyPatched,false);
 const pfUpdated=pair(bridge().buffer,original.toolbelt);assert.equal(pfUpdated.alreadyPatched,false);
 assert.equal(pair(toolUpdated.pf2e,toolUpdated.toolbelt).alreadyPatched,true);
});
test('an installed marker cannot conceal a changed live transport receiver',()=>{
 const original=pair(),changed=Buffer.from(original.toolbelt.toString().replace('e(c,s)','e(c)'));
 assert.throws(()=>pair(original.pf2e,changed),/toolbelt-.*seam/);
});
test('manifest only version changes promote legacy descriptors into usable seam builds',()=>{
 const legacy=patchStaticReceiver(pf2e).bytes,bridgeResult=builders.buildNativeBridge({source:legacy,version:'8.6.0'});
 const result=builders.buildSharedManualPair({pf2eSource:bridgeResult.buffer,toolbeltSource:patchToolbeltManualPool(toolbelt).bytes,pf2eVersion:'8.6.0',toolbeltVersion:'3.57.0'});
 assert.equal(bridgeResult.status,'patch');assert.equal(result.alreadyPatched,false);assert.equal(result.descriptors.pf2e.providerVersion,'8.6.0');assert.equal(result.descriptors.toolbelt.providerVersion,'3.57.0');
});
test('changed installed descriptor semantics and stray static markers are rejected',()=>{
 const original=pair(),changed=Buffer.from(original.pf2e.toString().replace('"model":"numeric-static-reception.v1"','"model":"unknown"'));
 assert.throws(()=>pair(changed,original.toolbelt),/native-.*seam/);
 assert.throws(()=>pair(Buffer.concat([bridge().buffer,Buffer.from('\nconst __nativeManualPoolStaticReceiver=()=>{};')])),/native-.*seam/);
});
test('a leading UTF8 BOM remains unrelated source data',()=>{
 const input=Buffer.concat([Buffer.from([0xef,0xbb,0xbf]),unbridge(pf2e)]),result=bridge(input);
 assert.deepEqual(result.buffer.subarray(0,3),input.subarray(0,3));assert.equal(result.descriptor.sourceSHA256,hash(input));
});
test('a Toolbelt only version update keeps the recomposed native IWR usable',async()=>{
 const prior=patchStaticReceiver(pf2e).bytes,bridge=builders.buildNativeBridge({source:prior,version:'8.5.1'});
 const result=builders.buildSharedManualPair({pf2eSource:bridge.buffer,toolbeltSource:patchToolbeltManualPool(toolbelt).bytes,pf2eVersion:'8.5.1',toolbeltVersion:'3.57.0'});
 const text=result.pf2e.toString(),start=text.indexOf('\tasync applyDamage({')+1,end=text.indexOf('\n\tasync undoDamage(',start),applyDamage=Function('return ({'+text.slice(start,end)+'}).applyDamage')();
 const statement=bridgeStatement(text),descriptor=statement.includes('"sourceContract"')?automaticDescriptor(statement):{version:'8.5.1',sourceSHA256:'8fa38a2fcbf848ad967c75876ca33ebf5a46fb7d6a8cc90d8323c0fb5d471bc7',protocol:'pf2e-third-party-automation:iwr:2'};
 const retained=Object.freeze({...descriptor,applyDamage});
 const proof=await verifyNativeIWRBridge({game:{system:{id:'pf2e',version:'8.5.1'}},source:result.pf2e,getNativeBridge:()=>retained,hash});assert.equal(proof.ready,true,proof.reason);
});
test('paired preparation retains both leading BOMs and remains byte stable',()=>{
 const bom=Buffer.from([0xef,0xbb,0xbf]),left=bridge(Buffer.concat([bom,unbridge(pf2e)])).buffer,right=Buffer.concat([bom,toolbelt]);
 const result=pair(left,right);assert.deepEqual(result.pf2e.subarray(0,3),bom);assert.deepEqual(result.toolbelt.subarray(0,3),bom);
 const repeated=pair(result.pf2e,result.toolbelt);assert.equal(repeated.alreadyPatched,true);assert.deepEqual(repeated.pf2e,result.pf2e);assert.deepEqual(repeated.toolbelt,result.toolbelt);
});
test('an unknown legacy observer descriptor is rejected rather than relabeled',()=>{
 const legacyPF=Buffer.from(patchStaticReceiver(pf2e).bytes.toString().replace("model:'numeric-static-reception.v1'","model:'unknown'")),legacyTool=patchToolbeltManualPool(toolbelt).bytes;
 assert.throws(()=>builders.buildSharedManualPair({pf2eSource:legacyPF,toolbeltSource:legacyTool,pf2eVersion:'8.5.1',toolbeltVersion:'3.56.5'}),/native-.*seam/);
});
