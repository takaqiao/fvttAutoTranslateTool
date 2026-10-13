import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import vm from 'node:vm';
import {spawnSync} from 'node:child_process';
import {buildNativeBridge,buildSharedManualPair} from '../tools/automatic-source-patches/native.mjs';
import {TOOL_SHAPE,toolSeams,observerRegion} from '../scripts/native-source-shapes.mjs';
import {native861Source,hash} from './native-iwr-861-fixture.mjs';
import {referenceToolbelt,edited,editedRegion,otherNames,renamed,renamedSeams} from './toolbelt-shape-fixture.mjs';

// Upstream files are read from the environment and never stored here. A case
// whose file is not supplied is skipped.
function input(name,sha256){
 if(!process.env[name])return;
 const bytes=fs.readFileSync(process.env[name]);assert.equal(hash(bytes),sha256,name);return bytes;
}
const toolbelt3590=input('TOOLBELT_3590_SOURCE','af823b33b8c01a24593fa607b0847544e18b27472f4871b642c08406628ede17');
const toolbelt3584=input('TOOLBELT_CURRENT_SOURCE','6acd53e7a5198ef8b9f7a35caae943b4582cabe61ff57bb76572038f61e549da');
// The pair TPA 0.9.41 installed on PF2e 8.6.1 and Toolbelt 3.58.4.
const toolbelt3584Patched=input('TOOLBELT_3584_PATCHED','ffc9ab831ef313b246e51cfdf1939365f338083152bb4b3a863c1ba5bb9c96c7');
const pf2ePatched=input('PF2E_NATIVE_861_PATCHED','a720ae12536008dc6a20ab14fcc25b267b9697d6d1430153feb17ee97e7321af');
const pf2eOriginal=input('PF2E_NATIVE_861_BUNDLE','a48f919b8b9a791868ff3eb4eb560b210855932847e8c861ee914e16e3448ed3');

const bridge=buildNativeBridge({source:native861Source(),version:'8.6.1'}).buffer;
const pair=(toolbeltSource,pf2eSource=bridge,toolbeltVersion='3.59.0')=>buildSharedManualPair({pf2eSource,toolbeltSource,pf2eVersion:'8.6.1',toolbeltVersion});
const observer=text=>observerRegion(text,'toolbelt').region+'/* end toolbelt manual pool */\n';
const restored=patched=>Buffer.from(toolSeams(edited(patched.toString(),observer(patched.toString()),'')).original);
const same=(actual,expected)=>assert.ok(Buffer.from(actual).equals(Buffer.from(expected)),'bytes differ');
const patchedText=(source,names=text=>text)=>edited(edited(source,names(TOOL_SHAPE.receive),names(TOOL_SHAPE.receivePatched)),names(TOOL_SHAPE.forward),names(TOOL_SHAPE.forwardPatched));

test('the reference regions are the Toolbelt 3.59.0 text',{skip:!toolbelt3590},()=>{
 const text=toolbelt3590.toString();
 for(const key of ['socket','transport','conversion','binding','validity','receive','forward'])assert.equal(text.split(TOOL_SHAPE[key]).length,2,key);
});

const builds=[['the reference bundle',referenceToolbelt,renamed(referenceToolbelt)]];
if(toolbelt3590)builds.push(['Toolbelt 3.59.0',toolbelt3590.toString(),renamedSeams(toolbelt3590.toString())]);
for(const [label,source,other]of builds){

test(`${label} is patched in its own names and restored byte for byte`,()=>{
 const seams=toolSeams(source);assert.equal(seams.installed,false);assert.equal(seams.original,source);
 assert.equal(seams.patched,patchedText(source));
 const again=toolSeams(seams.patched);assert.equal(again.installed,true);assert.equal(again.patched,seams.patched);assert.equal(again.original,source);
});

test(`${label} builds a shared pair once and is recognized afterwards`,()=>{
 const first=pair(Buffer.from(source)),again=pair(first.toolbelt,first.pf2e);
 assert.equal(first.alreadyPatched,false);assert.equal(again.alreadyPatched,true);same(again.pf2e,first.pf2e);same(again.toolbelt,first.toolbelt);
 assert.deepEqual(first.descriptors.toolbelt,{version:2,hpBaselineGuardVersion:1,providerId:'pf2e-toolbelt',providerVersion:'3.59.0',sourceSHA256:hash(source),sourceContract:'toolbelt-shared-manual:1'});
 same(restored(first.toolbelt),source);
});

for(const [name,key,before,after]of [
 ['another call on the master','receive','e.update(n)','e.delete(n)'],
 ['exchanged callback arguments','socket','e(c,s)','e(s,c)'],
 ['another document resolver','conversion','case"document":return fromUuid(t)','case"document":return fromUuidSync(t)'],
 ['one name split in two','receive','this.isValidMaster(e)&&e.update(n)','this.isValidMaster(e)&&q.update(n)'],
 ['two names merged into one','receive','e.update(n)','e.update(e)'],
 ['a removed statement','socket','ea();let s=UA(o)','let s=UA(o)'],
 ['another owner route','forward','m.isOwner?m.update(g)','m.isOwner?m.update(i)'],
 ['another actor test','validity','e instanceof Actor&&!e.pack','e instanceof Actor'],
 ['another master callback','binding','this.path("master"),this.#g.bind(this)','this.path("master"),this.#f.bind(this)'],
 ['another socket registration','transport','game.socket.on(`module.${A.id}`,t)','game.socket.once(`module.${A.id}`,t)'],
 ['another helper deciding who listens','socket','!Ne()||','!Zz()||'],
])test(`${label} with ${name} is refused and both inputs are left alone`,()=>{
 const changed=Buffer.from(editedRegion(source,key,before,after)),kept=[Buffer.from(bridge),Buffer.from(changed)];
 assert.throws(()=>pair(changed),/^Error: toolbelt-source-seam-/);
 same(bridge,kept[0]);same(changed,kept[1]);
});

test(`${label} with a changed installed seam is refused`,()=>{
 const patched=toolSeams(source).patched;
 for(const [before,after]of [['()=>this.isValidMaster(e)&&e.update(n)','()=>e.update(n)'],['.forward(e,m,g,r,','.forward(e,m,g,o,'],['if(x)await x','if(x)void x']])
  assert.throws(()=>toolSeams(edited(patched,before,after)),/^Error: toolbelt-source-seam-(receive|forward) \(token /,before);
 assert.throws(()=>toolSeams(edited(source,TOOL_SHAPE.receive,TOOL_SHAPE.receivePatched)),{message:'toolbelt-source-seam-partial'});
 assert.throws(()=>toolSeams(edited(source,TOOL_SHAPE.forward,TOOL_SHAPE.forwardPatched)),{message:'toolbelt-source-seam-partial'});
 // The observer without the seam it serves is not an installed pair either.
 assert.throws(()=>pair(Buffer.from(observer(pair(Buffer.from(source)).toolbelt.toString())+source)),{message:'toolbelt-source-seam-partial'});
});

test(`${label} under another consistent set of names is patched in those names`,()=>{
 assert.notEqual(other,source);
 const seams=toolSeams(other);assert.equal(seams.installed,false);
 assert.equal(seams.patched,patchedText(other,renamed));
 assert.ok(seams.patched.includes('#w1({master:$02,__explorationManualPool:x,...$03},$07){return __toolbeltManualPool.receive($02,$03,x,$07,()=>this.isValidMaster($02)&&$02.update($03))}'));
 assert.ok(seams.patched.includes('let x=__toolbeltManualPool.forward($02,$0s,$0u,$0a,()=>$0s.update($0u),$0t=>this.#w0.emit($0t));if(x)await x'));
 assert.equal(toolSeams(seams.patched).original,other);
 const built=pair(Buffer.from(other));assert.equal(built.alreadyPatched,false);same(restored(built.toolbelt),other);
});

}

test('a build that already uses x gives the pool binding another name',()=>{
 const other=renamed(referenceToolbelt,new Map([...otherNames,['e','x']])),patched=toolSeams(other).patched;
 assert.ok(patched.includes('#w1({master:x,__explorationManualPool:y,...$03},$07){return __toolbeltManualPool.receive(x,$03,y,$07,()=>this.isValidMaster(x)&&x.update($03))}'));
 assert.ok(patched.includes('let y=__toolbeltManualPool.forward(x,$0s,$0u,$0a,()=>$0s.update($0u),$0t=>this.#w0.emit($0t));if(y)await y'));
 assert.equal(toolSeams(patched).original,other);
});

test('the reference bundle and its patched forms parse',()=>{
 for(const source of [referenceToolbelt,renamed(referenceToolbelt)])for(const text of [source,toolSeams(source).patched,pair(Buffer.from(source)).toolbelt.toString()])new vm.Script(text);
});

for(const [name,reason,change]of [
 ['two registered tool classes','ambiguous',text=>text+text],
 ['a second declaration of the socket factory','ambiguous',text=>edited(text,'var Tool=','function _e(){}var Tool=')],
 ['no hook wrapper around the forward method','wrapper',text=>edited(text,'"_preUpdate",this.#h','"_preCreate",this.#h')],
 ['a static validity member','validity',text=>edited(text,'}isValidActor(e){','}static isValidActor(e){')],
 ['a generator in place of the receive method','receive',text=>edited(text,'}#g({master:','}*#g({master:')],
 ['an exported socket factory','socket',text=>edited(text,';function _e(t,e){',';export function _e(t,e){')],
 ['a socket factory that asks an unregistered helper','helper',text=>edited(text,'!Ne()||','!Zz()||')],
 ['the token test registered under another name','helper',text=>edited(text,'a(Ko,"isTokenObject")','a(Ko,"isToken")')],
])test(`a bundle with ${name} is refused`,()=>{
 assert.throws(()=>toolSeams(change(referenceToolbelt)),{message:'toolbelt-source-seam-'+reason});
});

test('the live PF2e 8.6.1 pair takes Toolbelt 3.59.0 without touching PF2e',{skip:!(pf2ePatched&&toolbelt3590)},t=>{
 const first=pair(toolbelt3590,pf2ePatched);
 assert.equal(first.alreadyPatched,false);same(first.pf2e,pf2ePatched);assert.notEqual(hash(first.toolbelt),hash(toolbelt3590));
 // The maintenance entry checks a candidate the same way before installing it.
 const directory=fs.mkdtempSync(path.join(os.tmpdir(),'toolbelt-shape-'));t.after(()=>fs.rmSync(directory,{recursive:true}));
 const candidate=path.join(directory,'main.js.tmp.mjs');fs.writeFileSync(candidate,first.toolbelt);
 const check=spawnSync(process.execPath,['--check',candidate],{encoding:'utf8'});assert.equal(check.status,0,check.stderr);
 const again=pair(first.toolbelt,first.pf2e);
 assert.equal(again.alreadyPatched,true);same(again.pf2e,first.pf2e);same(again.toolbelt,first.toolbelt);
 same(restored(first.toolbelt),toolbelt3590);
});

for(const [label,pf2e]of [['original',pf2eOriginal],['patched',pf2ePatched]])test(`${label} PF2e 8.6.1 and Toolbelt 3.58.4 still give the pair 0.9.41 installed`,{skip:!(pf2e&&toolbelt3584&&toolbelt3584Patched)},()=>{
 const built=pair(toolbelt3584,buildNativeBridge({source:pf2e,version:'8.6.1'}).buffer,'3.58.4');
 assert.equal(built.alreadyPatched,false);same(built.toolbelt,toolbelt3584Patched);
 if(pf2ePatched)same(built.pf2e,pf2ePatched);
});

test('the pair 0.9.41 installed is recognized and restored to Toolbelt 3.58.4',{skip:!(pf2ePatched&&toolbelt3584&&toolbelt3584Patched)},()=>{
 const again=pair(toolbelt3584Patched,pf2ePatched,'3.58.4');
 assert.equal(again.alreadyPatched,true);same(again.pf2e,pf2ePatched);same(again.toolbelt,toolbelt3584Patched);
 same(restored(toolbelt3584Patched),toolbelt3584);
});
