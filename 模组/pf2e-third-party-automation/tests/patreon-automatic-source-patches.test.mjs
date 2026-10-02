import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import vm from 'node:vm';
import {buildPatreonSource} from '../tools/automatic-source-patches/patreon.mjs';
import {patchPatreon} from '../tools/patreon-time-completion/patch-patreon-v5.mjs';
import {patchPatreonManualImmunity} from '../tools/patreon-manual-immunity/patch.mjs';
import {createPatreonTimeCompletion} from '../scripts/exploration/patreon-time-completion.mjs';
import {createPatreonManualImmunity} from '../scripts/exploration/patreon-manual-immunity.mjs';

const hash=bytes=>createHash('sha256').update(bytes).digest('hex');
const sourcePath=process.env.PATREON_AUTOMATIC_SOURCE;
assert.ok(sourcePath,'PATREON_AUTOMATIC_SOURCE must name the original Patreon source');
const source=readFileSync(sourcePath),original=source.toString('utf8');
const sourceSHA256='89ded325b92fa6b03dcf9257337ae2d628b3e99c987fe22cef9fff4e1837f4e9';
assert.equal(hash(source),sourceSHA256,'original source fixture SHA256');
const seams={
 time:'8c32f064ac33ab4a0ce264c8d74c8acbf8621627f1a8ef6f111e9ecb606cfa92',
 manualImmunity:'bdc211e3b7e437e91f8d9ab2c8f54c6b189abd8789278989de3a5a5b84e566a0',
 createItem:'14162e56764de7549e40aa9dad55249c14fe072acf181045f288bd02bd9bd7df'
};
function region(start,end){
 const from=original.indexOf(start),to=original.indexOf(end,from);
 assert.ok(from>=0&&to>from,'original seam boundaries');
 return original.slice(from,to+end.length);
}
const originalRegions={
 time:region('Hooks.on("updateWorldTime",','r(Mn,"handleFastHealingTime");'),
 manualImmunity:region('async function Ma(','r(Ma,"treatWounds");'),
 createItem:region('async function p(','r(p,"addItemToActor");')
};
const timeOnly=patchPatreon(source).bytes;
const legacy=patchPatreonManualImmunity(source,{patchTime:patchPatreon}).bytes;
assert.equal(hash(timeOnly),'d29a3878f9521cc185eb295288fe660b4c41a86a0a07fcc966423e887b15de22','legacy time fixture');
assert.equal(hash(legacy),'6b664250f325838850a716c791b3a41df4265899d39377fb8b9b10ad083b10a0','legacy composed fixture');

const build=(bytes=source,versions={version:'3.3.0',pf2eVersion:'8.6.0'})=>buildPatreonSource({source:bytes,...versions});
function changed(before,after){
 assert.equal(original.split(before).length,2,'unique mutation fixture');
 return Buffer.from(original.replace(before,after));
}
function tamperedObserver(bytes,marker){
 const text=bytes.toString('utf8');
 assert.equal(text.split(marker).length,2,'one actual injected observer declaration');
 return Buffer.from(text.replace(marker,marker.replace('=','/* altered */=')));
}

test('original source receives one shared descriptor with independently pinned seams',()=>{
 for(const [name,part] of Object.entries(originalRegions))assert.equal(hash(part),seams[name],name+' original region');
 const result=build();
 assert.equal(result.status,'patch');
 assert.deepEqual(result.descriptor,{version:3,providerId:'patreon-v3',providerVersion:'3.3.0',pf2eVersion:'8.6.0',sourceSHA256,
  qualification:'patreon-original-seams.v1',seams,markedCommitOwnership:'private-prepare.v1'});
 assert.doesNotThrow(()=>new vm.Script(result.buffer.toString('utf8')));
});

test('unrelated source bytes and updated versions retain truthful source provenance',()=>{
 const updated=Buffer.concat([Buffer.from('// unrelated upstream comment\n'),source]);
 const result=build(updated,{version:'3.3.0',pf2eVersion:'8.6.0'});
 assert.equal(result.status,'patch');
 assert.equal(result.descriptor.sourceSHA256,hash(updated));
 assert.notEqual(result.descriptor.sourceSHA256,sourceSHA256);
 assert.equal(result.descriptor.providerVersion,'3.3.0');
 assert.equal(result.descriptor.pf2eVersion,'8.6.0');
 assert.deepEqual(result.descriptor.seams,seams);
 assert.ok(result.buffer.toString('utf8').startsWith('// unrelated upstream comment\n'));
 assert.doesNotThrow(()=>new vm.Script(result.buffer.toString('utf8')));
});

test('qualified output is unchanged on another startup despite diagnostic version changes',()=>{
 const first=build(),second=build(first.buffer,{version:'4.0.0',pf2eVersion:'9.0.0'});
 assert.equal(second.status,'unchanged');
 assert.deepEqual(second.buffer,first.buffer);
});

test('the exact legacy composed source remains byte-identical',()=>{
 const result=build(legacy,{version:'3.2.29',pf2eVersion:'8.5.1'});
 assert.equal(result.status,'unchanged');
 assert.deepEqual(result.buffer,legacy);
});

test('a legacy time-only installation is refused as an incomplete composition',()=>{
 assert.throws(()=>build(timeOnly),/patreon-/);
});

const alteredRegions=[
 ['time',()=>changed('.filter(l=>l.test())','.filter(l=>!l.test())')],
 ['manual immunity',()=>changed('async function Ma(a,e){','async function Ma(a,e){void 0;')],
 ['item creation',()=>changed('let i=await a.createEmbeddedDocuments("Item",[e]);','let i=await a.createEmbeddedDocuments("Item",[]);')]
];
for(const [name,alter] of alteredRegions)test('a changed '+name+' region is refused',()=>{
 assert.throws(()=>build(alter()),/patreon-/);
});

for(const [name,part] of Object.entries(originalRegions))test('duplicate '+name+' seams are refused',()=>{
 assert.throws(()=>build(Buffer.concat([source,Buffer.from('\n'+part)])),/patreon-/);
});

for(const marker of ['const __patreonTimeCompletion=','const __patreonManualImmunity=']){
 const name=marker.includes('TimeCompletion')?'time':'manual immunity';
 test('a partial '+name+' observer is refused',()=>{
  assert.throws(()=>build(Buffer.concat([Buffer.from(marker+'{};\n'),source])),/patreon-/);
 });
 test('tampered '+name+' observer output is refused',()=>{
  const result=build();
  assert.throws(()=>build(tamperedObserver(result.buffer,marker)),/patreon-/);
 });
}


function installGeneratedObservers(bytes,{version,pf2eVersion}){
 const callbacks=[],existingAPI={sibling:7},module={active:true,version,api:existingAPI};
 const user={id:'G'},game={user,users:{activeGM:user},time:{worldTime:100},system:{version:pf2eVersion},
  modules:new Map([['patreon-v3',module]])};
 const context=vm.createContext({game,Hooks:{once(name,fn){assert.equal(name,'init');callbacks.push(fn)}},
  crypto:{randomUUID:()=> 'I'},structuredClone,console});
 const text=bytes.toString('utf8');
 for(const name of ['TimeCompletion','ManualImmunity']){
  const start='const __patreon'+name+'=',end='Hooks.once("init",()=>__patreon'+name+'.install());';
  assert.equal(text.split(start).length,2,'one generated '+name+' observer');
  assert.equal(text.split(end).length,2,'one '+name+' initialization');
  const from=text.indexOf(start),to=text.indexOf(end,from)+end.length;
  vm.runInContext(text.slice(from,to),context);
 }
 assert.equal(callbacks.length,2);for(const initialize of callbacks)initialize();
 assert.equal(module.api,existingAPI);assert.equal(module.api.sibling,7);
 return {game,module};
}

async function assertGeneratedAdapters(result,versions){
 const {game,module}=installGeneratedObservers(result.buffer,versions);
 const timeAPI=module.api.explorationTimeCompletion,immunityAPI=module.api.explorationManualImmunity;
 for(const api of [timeAPI,immunityAPI]){
  assert.equal(api.descriptor.version,3);
  assert.equal(api.descriptor.sourceSHA256,sourceSHA256);
  assert.deepEqual(JSON.parse(JSON.stringify(api.descriptor.seams)),seams);
 }
 // Browser modules share a realm; normalize only the VM-to-host fixture boundary.
 const hostTime={descriptor:JSON.parse(JSON.stringify(timeAPI.descriptor)),subscribe:(...args)=>timeAPI.subscribe(...args)};
 let subscriptions=0,disposals=0;
 const hostImmunity={descriptor:JSON.parse(JSON.stringify(immunityAPI.descriptor)),subscribe(observe){
  subscriptions++;const dispose=immunityAPI.subscribe(observe);
  return ()=>{disposals++;dispose()};
 }};
 const hostGame={...game,modules:new Map([['patreon-v3',{...module,api:{...module.api,
  explorationTimeCompletion:hostTime,explorationManualImmunity:hostImmunity}}]])};
 const time=createPatreonTimeCompletion({game:hostGame,runtimeIdentity:()=>({userId:'G',clientNonce:'client'}),
  getDriverScope:()=>({leaseNonce:'lease'}),timeoutMs:1000});
 try{assert.deepEqual(await time.beforeAdvance({id:'C',sessionId:'S',from:100,to:700,gmId:'G'}),{status:'ready'})}
 finally{time.invalidate('test-finished')}
 const immunity=createPatreonManualImmunity({game:hostGame,fromUuid:async()=>null});
 const dispose=immunity.subscribe(()=>{});
 try{assert.equal(subscriptions,1)}finally{dispose()}
 assert.equal(disposals,1);
 assert.equal(module.api,game.modules.get('patreon-v3').api);assert.equal(module.api.sibling,7);
}

for(const [name,versions] of [
 ['Patreon',{version:'3.3.0',pf2eVersion:'8.5.1'}],
 ['PF2e',{version:'3.2.29',pf2eVersion:'8.6.0'}]
])test('changed '+name+' metadata rebuilds legacy observers for the actual adapters',async()=>{
 const result=build(legacy,versions);
 assert.equal(result.status,'patch');assert.equal(result.descriptor.version,3);
 assert.equal(result.descriptor.sourceSHA256,sourceSHA256);
 assert.equal(result.descriptor.providerVersion,versions.version);
 assert.equal(result.descriptor.pf2eVersion,versions.pf2eVersion);
 assert.doesNotThrow(()=>new vm.Script(result.buffer.toString('utf8')));
 await assertGeneratedAdapters(result,versions);
});

test('long diagnostic versions do not prevent an unchanged second startup',()=>{
 const versions={version:'Patreon-'+ 'x'.repeat(12000),pf2eVersion:'PF2e-'+ 'y'.repeat(12000)};
 const first=build(source,versions);
 assert.equal(first.descriptor.providerVersion,versions.version);assert.equal(first.descriptor.pf2eVersion,versions.pf2eVersion);
 const second=build(first.buffer,versions);
 assert.equal(second.status,'unchanged');assert.deepEqual(second.buffer,first.buffer);
});

test('unrelated bytes around qualified output leave its installed composition unchanged',()=>{
 const first=build(),prefix=Buffer.from('// unrelated upstream prefix\n'),suffix=Buffer.from('\n// unrelated upstream suffix\n');
 const updated=Buffer.concat([prefix,first.buffer,suffix]),result=build(updated,{version:'4.0.0',pf2eVersion:'9.0.0'});
 assert.equal(result.status,'unchanged');assert.deepEqual(result.buffer,updated);
 assert.equal(result.descriptor.version,3);assert.equal(result.descriptor.sourceSHA256,sourceSHA256);
 assert.deepEqual(result.descriptor.seams,seams);
 assert.doesNotThrow(()=>new vm.Script(result.buffer.toString('utf8')));
 assert.deepEqual(build(result.buffer).buffer,updated);
});

test('a changed original predicate in qualified time output is refused',()=>{
 const result=build(),text=result.buffer.toString('utf8');
 const before='.filter(l=>__patreonTimeCompletion.predicate(x,l,l.test()))';
 assert.equal(text.split(before).length,2,'one actual original time predicate');
 const altered=Buffer.from(text.replace(before,'.filter(l=>__patreonTimeCompletion.predicate(x,l,!l.test()))'));
 assert.throws(()=>build(altered),/patreon-time-/);
});

test('changed metadata upgrades a complete legacy composition with unrelated outer bytes',()=>{
 const prefix=Buffer.from('// unrelated legacy prefix\n'),updated=Buffer.concat([prefix,legacy]);
 const result=build(updated,{version:'3.3.0',pf2eVersion:'8.6.0'});
 assert.equal(result.status,'patch');assert.equal(result.descriptor.version,3);
 assert.equal(result.descriptor.sourceSHA256,hash(Buffer.concat([prefix,source])));
 assert.ok(result.buffer.toString('utf8').startsWith(prefix.toString('utf8')));
 assert.doesNotThrow(()=>new vm.Script(result.buffer.toString('utf8')));
});
