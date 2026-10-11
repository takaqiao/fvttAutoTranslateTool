import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {readFile} from 'node:fs/promises';
import * as naming from '../scripts/patches/item-name.mjs';
import {hashSource} from '../scripts/source-hash.mjs';

const entry=await readFile(new URL('../av-v14-hotfix.mjs',import.meta.url),'utf8');
const sources={
  '8.5.0':await readFile(new URL('./generate-item-name-original.js.txt',import.meta.url),'utf8'),
  '8.5.1':await readFile(new URL('./fixtures/generate-item-name-8.5.1.js.txt',import.meta.url),'utf8'),
  '8.6.0':await readFile(new URL('./fixtures/generate-item-name-8.6.0.js.txt',import.meta.url),'utf8')
};
async function boot(version,{sourceVersion=version,hash=hashSource,systemId='pf2e',duringHash,transformOriginal=value=>value}={}){
  const callbacks={},microtasks=[],settings=new Map(),self={},errors=[];
  const original=transformOriginal(vm.runInNewContext('('+sources[sourceVersion]+')')),target={};let reads=0,writes=0,current=original;
  Object.defineProperty(target,'generateItemName',{get(){reads++;return current;},set(value){writes++;current=value;}});
  const runtime={...naming,console:{info(){},error:(...args)=>errors.push(args)},structuredClone,
    Hooks:{once:(key,fn)=>callbacks[key]=fn},queueMicrotask:fn=>microtasks.push(fn),
    game:{version:'14.368',system:{id:systemId,version},pf2e:{system:target},modules:new Map([['av-v14-hotfix',self]]),
      settings:{register:(id,key,config)=>settings.set(key,config.default),get:(id,key)=>settings.get(key)}},
    libWrapper:{register(){}},
    hashSource:async value=>{const result=await hash(value);duringHash?.(runtime,target);return result;},
    registerBabeleIndex:()=>({status:()=>({state:'fixture'})})};
  for(const name of ['prepareSundryPatch','prepareTurnLifecyclePatch','registerLegacyCompat','installSundryPatch','installDurationPatch','installTimestampPatch','installChatDeleteCoalescing','installTokenizerChatPortraitPatch','installSoundStopPatch','registerDsnQualitySettings','installDsnQualityLocks','installTurnLifecyclePatch','installDsnChatRecovery'])runtime[name]=()=>{};
  vm.runInNewContext(entry.replace(/^import .*;\r?\n/gm,'').replaceAll('import.meta.url',JSON.stringify(new URL('../av-v14-hotfix.mjs',import.meta.url).href)),runtime);
  callbacks.init();await callbacks.setup();callbacks.ready();for(const fn of microtasks)await fn();
  return {state:self.api.status().patches.itemNames,reads,writes,current,original,errors};
}

for(const version of ['8.5.0','8.5.1','8.6.0'])test(`entry installs names for the audited ${version} release function`,async()=>{
 const r=await boot(version);assert.deepEqual(r.errors,[]);assert.equal(r.state.status,'installed');
 assert.equal(r.writes,1);assert.notEqual(r.current,r.original);
 const item={isOfType:()=>true,baseType:null,name:'custom item'};
 assert.equal(r.current(item),'custom item');
});
for(const [version,sourceVersion] of [['8.5.0','8.5.1'],['8.5.1','8.5.0']])test(`entry accepts audited ${sourceVersion} source with supported ${version} version`,async()=>{
 const r=await boot(version,{sourceVersion});assert.equal(r.state.status,'installed');assert.equal(r.writes,1);
});
test('other systems never read or replace the naming method',async()=>{
 for(const opts of [{version:'8.6.0',systemId:'sf2e'},{version:'9.0.0',systemId:'other'}]){
  const r=await boot(opts.version,{...opts,sourceVersion:'8.5.1'});
  assert.equal(r.state.status,'unsupported-system');assert.equal(r.reads,0);assert.equal(r.writes,0);
 }
});
test('a competing naming replacement during hashing is preserved',async()=>{
 const other=()=> 'another module';
 const r=await boot('8.5.1',{duringHash:(_runtime,target)=>{target.generateItemName=other;}});
 assert.equal(r.state.status,'source-changed-during-validation');assert.equal(r.current,other);assert.equal(r.writes,1);
});
test('a system version change during validation does not install a stale naming profile',async()=>{
 const r=await boot('8.5.1',{duringHash:runtime=>{runtime.game.system.version='8.5.2';}});
 assert.equal(r.state.status,'source-changed-during-validation');assert.equal(r.writes,0);
});

test('PF2e version labels do not replace the exact audited naming function contract',async()=>{
 for(const version of ['8.5.2','8.6.0','9.0.0']){
  const known=await boot(version,{sourceVersion:'8.6.0'});assert.equal(known.state.status,'installed');
  const foreign=await boot(version,{sourceVersion:'8.6.0',hash:()=> 'unknown'});
  assert.equal(foreign.state.status,'unsupported-source');assert.equal(foreign.writes,0);
 }
});

test('a replaced naming function cannot supply the audited source through its own toString',async()=>{
 const result=await boot('8.6.0',{transformOriginal:original=>{
  const foreign=()=> 'foreign name';foreign.toString=()=>original.toString();return foreign;
 }});
 assert.equal(result.state.status,'unsupported-source');assert.equal(result.writes,0);
});
