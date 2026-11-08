import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {buildNativeBridge} from '../tools/automatic-source-patches/native.mjs';
import {verifyNativeIWRBridge,isVerifiedNativeIWRBridge} from '../scripts/native-iwr-verification.mjs';
import {NATIVE_DAMAGE_CONTRACT,NATIVE_IWR_BRIDGE_PROTOCOL} from '../scripts/native-iwr-profiles.mjs';
import {nativeMethod,automaticDescriptor,bridgeStatement,NATIVE_DAMAGE_REFERENCE} from '../scripts/native-source-shapes.mjs';
import {createShapeBinding} from '../scripts/source-shape-match.mjs';
import {native861Source,hash} from './native-iwr-861-fixture.mjs';
import {DAMAGE_CASES,assertCallback} from './native-damage-harness.mjs';

// Upstream files are read from the environment and never stored here. A case
// whose file is not supplied is skipped; the fixture cases always run.
function input(name,sha256){
 if(!process.env[name])return;
 const bytes=readFileSync(process.env[name]);assert.equal(hash(bytes),sha256,name);return bytes;
}
const pf2e870=input('PF2E_NATIVE_870_BUNDLE','c9a2f176c388b211ef62c4d39fa5061a3525feebad8ca7cf50a82d2e6e0c1da3');
// PF2e 8.6.1 as 0.9.41 installed it on the server, bridge descriptor pf2e-apply-damage:D:2.
const pf2e861Installed=input('PF2E_NATIVE_861_PATCHED','a720ae12536008dc6a20ab14fcc25b267b9697d6d1430153feb17ee97e7321af');
const methodOf=text=>Function('return ({'+nativeMethod(text)+'}).applyDamage')();
async function verify(bytes,version){
 const text=Buffer.from(bytes).toString(),retained=Object.freeze({...automaticDescriptor(bridgeStatement(text)),applyDamage:methodOf(text)}),game={system:{id:'pf2e',version}};
 return {game,retained,proof:await verifyNativeIWRBridge({game,source:new Uint8Array(bytes),getNativeBridge:()=>retained,hash})};
}
// Replace the one applyDamage of `text`; split and join keep `$` in the method literal.
const withMethod=(text,method)=>text.split(nativeMethod(text)).join(method);

test('PF2e 8.7.0 is patched in its own names and recognized afterwards',{skip:!pf2e870},()=>{
 const result=buildNativeBridge({source:pf2e870,version:'8.7.0'}),method=nativeMethod(result.buffer.toString());
 assert.equal(result.status,'patch');
 assert.deepEqual(result.descriptor,{version:'8.7.0',sourceSHA256:hash(pf2e870),protocol:NATIVE_IWR_BRIDGE_PROTOCOL,sourceContract:NATIVE_DAMAGE_CONTRACT,methodSHA256:hash(method)});
 // 8.7.0 renames RollNotePF2e and leaves the rest of the method as 8.6.1 has it.
 assert.equal(method,NATIVE_DAMAGE_REFERENCE.patched.replace('qi.notesToHTML','Ji.notesToHTML'));
 const again=buildNativeBridge({source:result.buffer,version:'8.7.0'});
 assert.equal(again.status,'unchanged');assert.deepEqual(again.buffer,result.buffer);assert.deepEqual(again.descriptor,result.descriptor);
});

test('PF2e 8.7.0 served bytes and retained method receive IWR proof',{skip:!pf2e870},async()=>{
 const output=buildNativeBridge({source:pf2e870,version:'8.7.0'}).buffer,{game,retained,proof}=await verify(output,'8.7.0');
 assert.equal(proof.ready,true,proof.reason);assert.equal(proof.patchedSHA256,hash(output));assert.equal(proof.applyDamageSHA256,retained.methodSHA256);
 assert.equal(isVerifiedNativeIWRBridge(proof,game,retained),true);
 const changed=Buffer.from(output.toString().replace('if (thirdPartyIWR && await thirdPartyIWR === true) x = D + ae;','if (thirdPartyIWR && await thirdPartyIWR === true) x = D;'));
 assert.notDeepEqual(changed,output);
 const refused=await verifyNativeIWRBridge({game,source:changed,getNativeBridge:()=>retained,hash});
 assert.equal(refused.ready,false);assert.equal(refused.reason,'native-source-seam-mismatch');
 const other=Object.freeze({...retained,applyDamage:methodOf(changed.toString())});
 const substituted=await verifyNativeIWRBridge({game,source:output,getNativeBridge:()=>other,hash});
 assert.equal(substituted.ready,false);assert.equal(substituted.reason,'native-method-sha-mismatch');
});

test('PF2e 8.7.0 without a bridge reports the bridge as missing',{skip:!pf2e870},async()=>{
 const result=await verifyNativeIWRBridge({game:{system:{id:'pf2e',version:'8.7.0'}},source:new Uint8Array(pf2e870),getNativeBridge:()=>undefined,hash});
 assert.equal(result.ready,false);assert.equal(result.reason,'missing-native-iwr-bridge');
});

for(const row of DAMAGE_CASES)for(const prevent of [false,true])test(`8.7.0 callback on ${row[0]} keeps native amounts when prevention is ${prevent}`,{skip:!pf2e870},async()=>{
 await assertCallback(nativeMethod(pf2e870.toString()),nativeMethod(buildNativeBridge({source:pf2e870,version:'8.7.0'}).buffer.toString()),row,prevent);
});

// The seven locals the bridge reads, each under a name no PF2e release has used for it.
const renaming=createShapeBinding({fixed:['ui']});renaming.match('x;f;r;D;ae;ne;oe','Qx;Qf;Qr;Qd;Qa;Qn;Qo');
const renamedMethod=renaming.render(NATIVE_DAMAGE_REFERENCE.original,{within:'class'});
const renamedSource=Buffer.from(withMethod(native861Source().toString(),renamedMethod));

test('locals renamed in an otherwise unchanged method are read under their new names',async()=>{
 assert.ok(renamedMethod.includes('Qx = (m ? Math.max : Math.min)(0, Qf.finalDamage - (y + b));'));
 const result=buildNativeBridge({source:renamedSource,version:'9.0.0'}),patched=nativeMethod(result.buffer.toString());
 assert.equal(result.status,'patch');
 assert.ok(patched.includes('?.nativeDamageIWR?.(this, arguments[0], Qf, Qr, { actorDamage: Qx - Qd - Qa, shieldDamage: Qn });\n\t\tif (thirdPartyIWR && await thirdPartyIWR === true) Qx = Qd + Qa;\n\t\tlet Qo = this.calculateHealthDelta({'));
 assert.equal(buildNativeBridge({source:result.buffer,version:'9.0.0'}).status,'unchanged');
 const {proof}=await verify(result.buffer,'9.0.0');assert.equal(proof.ready,true,proof.reason);
 for(const row of DAMAGE_CASES)for(const prevent of [false,true])await assertCallback(renamedMethod,patched,row,prevent);
});

for(const [name,before,after]of [
 ['a changed hardness rule','Qa = Qx > 0 ? Math.min(Qx - Qd, ie) : 0','Qa = Qx > 0 ? Math.max(Qx - Qd, ie) : 0'],
 ['two locals merged into one','delta: Qx - Qd - Qa','delta: Qx - Qd - Qd'],
 ['an extra statement before the health delta','\t\tlet Qo = this.calculateHealthDelta({','\t\tQx = 0;\n\t\tlet Qo = this.calculateHealthDelta({'],
])test(`a renamed method with ${name} is refused without touching the input`,()=>{
 const changed=Buffer.from(withMethod(renamedSource.toString(),renamedMethod.replace(before,after))),kept=Buffer.from(changed);
 assert.notDeepEqual(changed,renamedSource);
 assert.throws(()=>buildNativeBridge({source:changed,version:'9.0.0'}),/native-bridge-seam-method/);
 assert.deepEqual(changed,kept);
});

test('a changed PF2e 8.7.0 hardness rule is refused',{skip:!pf2e870},()=>{
 const changed=Buffer.from(pf2e870.toString().replace('ae = x > 0 ? Math.min(x - D, ie) : 0','ae = x > 0 ? Math.min(x, ie) : 0'));
 assert.notDeepEqual(changed,pf2e870);
 assert.throws(()=>buildNativeBridge({source:changed,version:'8.7.0'}),/native-bridge-seam-method/);
});

// Releases up to 0.9.51 named the 8.6.1 bytes, pf2e-apply-damage:D:2, in the descriptor.
const earlier=Buffer.from(buildNativeBridge({source:native861Source(),version:'8.6.1'}).buffer.toString().replace(`"sourceContract":"${NATIVE_DAMAGE_CONTRACT}"`,'"sourceContract":"pf2e-apply-damage:D:2"'));
for(const [label,installed]of [['the earlier bridge on the 8.6.1 seams',earlier],['the server capture',pf2e861Installed]])
test(`an 8.6.1 bridge installed under pf2e-apply-damage:D:2 stays as it is: ${label}`,{skip:!installed},async()=>{
 const result=buildNativeBridge({source:installed,version:'8.6.1'});
 assert.equal(result.status,'unchanged');assert.deepEqual(result.buffer,installed);assert.equal(result.descriptor.sourceContract,'pf2e-apply-damage:D:2');
 const {proof}=await verify(installed,'8.6.1');assert.equal(proof.ready,true,proof.reason);
});

test('an unknown contract id on an installed bridge is refused',()=>{
 assert.throws(()=>buildNativeBridge({source:Buffer.from(earlier.toString().replace('"pf2e-apply-damage:D:2"','"pf2e-apply-damage:D:9"')),version:'8.6.1'}),/native-bridge-seam-descriptor/);
});
