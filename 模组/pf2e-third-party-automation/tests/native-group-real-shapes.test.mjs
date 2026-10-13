import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {readFileSync} from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {maintainSourcePatches,buildGroup,groups} from '../tools/automatic-source-patches/maintain.mjs';
import {buildNativeBridge,buildSharedManualPair} from '../tools/automatic-source-patches/native.mjs';
import {automaticDescriptor,observerRegion} from '../scripts/native-source-shapes.mjs';
import {native861Source,hash} from './native-iwr-861-fixture.mjs';
import {referenceToolbelt} from './toolbelt-shape-fixture.mjs';

// Upstream files are read from the environment and never stored here. A case
// whose file is not supplied is skipped; the reference shapes always run.
function input(name,sha256){
 if(!process.env[name])return;
 const bytes=readFileSync(process.env[name]);assert.equal(hash(bytes),sha256,name);return bytes;
}
const targets={
 pf2e:{file:'Data/systems/pf2e/pf2e.mjs',manifest:'Data/systems/pf2e/system.json',id:'pf2e'},
 toolbelt:{file:'Data/modules/pf2e-toolbelt/scripts/main.js',manifest:'Data/modules/pf2e-toolbelt/module.json',id:'pf2e-toolbelt'}
};
const pf2eOriginal=native861Source(),pf2eBridged=buildNativeBridge({source:pf2eOriginal,version:'8.6.1'}).buffer;
const toolbeltOriginal=Buffer.from(referenceToolbelt);
// What the release before this one installed on those originals: the manual
// batch on top of the bridge, and the manual pool in Toolbelt.
const earlier=buildSharedManualPair({pf2eSource:pf2eBridged,toolbeltSource:toolbeltOriginal,pf2eVersion:'8.6.1',toolbeltVersion:'3.59.0'});
// The pair TPA 0.9.41 installed on PF2e 8.6.1 and Toolbelt 3.58.4, as captured from the server.
const pf2eInstalled=input('PF2E_NATIVE_861_PATCHED','a720ae12536008dc6a20ab14fcc25b267b9697d6d1430153feb17ee97e7321af');
const toolbeltInstalled=input('TOOLBELT_3584_PATCHED','ffc9ab831ef313b246e51cfdf1939365f338083152bb4b3a863c1ba5bb9c96c7');
const toolbelt3590=input('TOOLBELT_3590_SOURCE','af823b33b8c01a24593fa607b0847544e18b27472f4871b642c08406628ede17');
// A Toolbelt build in which the manual pool found no seam.
const toolbeltUnknown=Buffer.from('export const anotherBuild = true;\n');

const same=(actual,expected)=>assert.ok(Buffer.from(actual).equals(Buffer.from(expected)),'bytes differ');
async function installation(t,sources,versions={}){
 const dataPath=await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(),'native-group-')));
 t.after(()=>fs.rm(dataPath,{recursive:true,force:true}));
 for(const [role,target]of Object.entries(targets)){
  await fs.mkdir(path.dirname(path.join(dataPath,target.file)),{recursive:true});
  if(sources[role])await fs.writeFile(path.join(dataPath,target.file),sources[role]);
  await fs.writeFile(path.join(dataPath,target.manifest),JSON.stringify({id:target.id,version:versions[role]??(role==='pf2e'?'8.6.1':'3.59.0')}));
 }
 const journals=[],reports=[],file=role=>path.join(dataPath,targets[role].file),directory=path.join(dataPath,'Backups/pf2e-third-party-patches');
 // The pending record is unlinked once its group is installed, so it is read as it is published.
 const io=new Proxy(fs,{get(target,key){
  if(key!=='link')return target[key];
  return async(from,to)=>{if(path.basename(to)==='native.pending.json')journals.push(JSON.parse(await target.readFile(from,'utf8')));return target.link(from,to)};
 }});
 return {file,journals,reports,
  identity:async role=>{const {ino,size,mtimeMs,ctimeMs}=await fs.stat(file(role));return {ino,size,mtimeMs,ctimeMs}},
  backups:async()=>{try{return (await fs.readdir(directory)).sort()}catch(error){if(error.code==='ENOENT')return null;throw error}},
  backup:name=>fs.readFile(path.join(directory,name)),
  // No builders are passed: this is the startup path, down to the syntax check of the candidate.
  async run(){
   const result=await maintainSourcePatches({dataPath,io,assertIdle:async()=>({idle:true}),report:value=>reports.push(value)});
   return result.groups.find(group=>group.group==='native');
  }};
}
async function installsBridgeOnly(t,toolbelt,toolbeltVersion){
 const f=await installation(t,{pf2e:pf2eOriginal,toolbelt},{toolbelt:toolbeltVersion}),before=await f.identity('toolbelt');
 const entry={before:hash(pf2eOriginal),after:hash(pf2eBridged)};
 assert.deepEqual(await f.run(),{group:'native',status:'patched',files:[{file:targets.pf2e.file,...entry}]});
 same(await fs.readFile(f.file('pf2e')),pf2eBridged);assert.equal(pf2eBridged.includes('__nativeManualPool'),false);
 same(await fs.readFile(f.file('toolbelt')),toolbelt);assert.deepEqual(await f.identity('toolbelt'),before);
 assert.deepEqual(await f.backups(),[`pf2e-${entry.before}.bin`]);same(await f.backup(`pf2e-${entry.before}.bin`),pf2eOriginal);
 assert.equal(f.journals.length,1);assert.deepEqual(f.journals[0].entries.map(({role,before,after})=>({role,before,after})),[{role:'pf2e',...entry}]);
 assert.deepEqual(f.reports.find(report=>report.group==='native').files.map(row=>row.file),[targets.pf2e.file]);
 // The next start finds the bridge and writes nothing more.
 const installed=await f.identity('pf2e');
 assert.deepEqual(await f.run(),{group:'native',status:'unchanged'});
 assert.deepEqual(await f.identity('pf2e'),installed);assert.deepEqual(await f.identity('toolbelt'),before);
 assert.equal((await f.backups()).length,1);assert.equal(f.journals.length,1);
}

// The maintenance loop reads one output per member of the group. A missing one
// fails the group on every start, and the startup entry does not stop for that.
for(const [label,pf2e,toolbelt]of [
 ['bridge and manual patches installed, the reference shapes',earlier.pf2e,earlier.toolbelt],
 ['bridge and manual patches installed, the server capture',pf2eInstalled,toolbeltInstalled],
 ['original PF2e, the reference Toolbelt with its manual patch',pf2eOriginal,earlier.toolbelt],
 ['original PF2e, the server\'s Toolbelt with its manual patch',pf2eOriginal,toolbeltInstalled],
 ['original PF2e, the original reference Toolbelt',pf2eOriginal,toolbeltOriginal],
 ['original PF2e, original Toolbelt 3.59.0',pf2eOriginal,toolbelt3590],
 ['original PF2e, a Toolbelt build the manual seam refused',pf2eOriginal,toolbeltUnknown],
])test(`buildGroup returns one output for each member of the native group: ${label}`,{skip:!(pf2e&&toolbelt)},async()=>{
 // The bridge builder is the only one the group is given.
 const built=await buildGroup('native',{pf2e:{source:{bytes:pf2e},version:'8.6.1'},toolbelt:{source:{bytes:toolbelt},version:'3.59.0'}},{buildNativeBridge});
 assert.deepEqual(Object.keys(built).sort(),[...groups.native].sort());
 same(built.pf2e,pf2e===pf2eOriginal?pf2eBridged:pf2e);same(built.toolbelt,toolbelt);
});

for(const [label,pf2e,toolbelt,toolbeltVersion]of [['the reference shapes',earlier.pf2e,earlier.toolbelt,'3.59.0'],['the server capture',pf2eInstalled,toolbeltInstalled,'3.58.4']])
test(`a bridge and manual patches already installed are left as they are: ${label}`,{skip:!(pf2e&&toolbelt)},async t=>{
 assert.ok(pf2e.includes('thirdPartyNativeIWRBridge')&&pf2e.includes('__nativeManualPoolBatch')&&toolbelt.includes('__toolbeltManualPool'));
 const f=await installation(t,{pf2e,toolbelt},{toolbelt:toolbeltVersion}),before=[await f.identity('pf2e'),await f.identity('toolbelt')];
 assert.deepEqual(await f.run(),{group:'native',status:'unchanged'});
 same(await fs.readFile(f.file('pf2e')),pf2e);same(await fs.readFile(f.file('toolbelt')),toolbelt);
 assert.deepEqual([await f.identity('pf2e'),await f.identity('toolbelt')],before);
 assert.equal(await f.backups(),null);assert.deepEqual(f.journals,[]);
 assert.deepEqual(f.reports.find(report=>report.group==='native'),{component:'pf2e-third-party-source-patches',group:'native',status:'unchanged'});
});

for(const [label,toolbelt,toolbeltVersion]of [['the reference shapes',earlier.toolbelt,'3.59.0'],['the server capture',toolbeltInstalled,'3.58.4']])
test(`an updated PF2e gets the bridge alone beside a Toolbelt that keeps its manual patch: ${label}`,{skip:!toolbelt},async t=>{
 assert.ok(toolbelt.includes('__toolbeltManualPool'));
 await installsBridgeOnly(t,toolbelt,toolbeltVersion);
});

for(const [label,toolbelt]of [['the reference shapes',toolbeltOriginal],['Toolbelt 3.59.0',toolbelt3590]])
test(`original PF2e and Toolbelt change PF2e only: ${label}`,{skip:!toolbelt},async t=>{
 assert.equal(toolbelt.includes('__toolbeltManualPool'),false);
 await installsBridgeOnly(t,toolbelt,'3.59.0');
});

test('the bridge now installed on the 8.6.1 release is the source the server\'s manual patch was built on',{skip:!(process.env.PF2E_NATIVE_861_BUNDLE&&pf2eInstalled)},()=>{
 assert.equal(hash(pf2eBridged),automaticDescriptor(observerRegion(pf2eInstalled.toString(),'pf2e').statement).baseSourceSHA256);
});

test('a Toolbelt build the manual seam refused no longer holds back the bridge',async t=>{
 assert.throws(()=>buildSharedManualPair({pf2eSource:pf2eBridged,toolbeltSource:toolbeltUnknown,pf2eVersion:'8.6.1',toolbeltVersion:'9.9.9'}),/^Error: toolbelt-source-seam-/);
 await installsBridgeOnly(t,toolbeltUnknown,'9.9.9');
});

test('without the Toolbelt file the group is unavailable and PF2e is left alone',async t=>{
 const f=await installation(t,{pf2e:pf2eOriginal}),before=await f.identity('pf2e'),result=await f.run();
 assert.equal(result.status,'unavailable');assert.deepEqual(result.files,groups.native.map(role=>targets[role].file));
 same(await fs.readFile(f.file('pf2e')),pf2eOriginal);assert.deepEqual(await f.identity('pf2e'),before);
 assert.equal(await f.backups(),null);assert.deepEqual(f.journals,[]);
});
