import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {readFile} from 'node:fs/promises';
import * as naming from '../scripts/patches/item-name.mjs';
import {hashSource} from '../scripts/source-hash.mjs';

const entry=await readFile(new URL('../av-v14-hotfix.mjs',import.meta.url),'utf8');
const sources={
  '8.5.0':await readFile(new URL('./generate-item-name-original.js.txt',import.meta.url),'utf8'),
  '8.5.1':await readFile(new URL('./fixtures/generate-item-name-8.5.1.js.txt',import.meta.url),'utf8')
};
async function boot(version,{sourceVersion=version,hash=hashSource,systemId='pf2e',duringHash}={}){
  const callbacks={},microtasks=[],settings=new Map(),self={},errors=[];
  const original=vm.runInNewContext('('+sources[sourceVersion]+')'),target={};let reads=0,writes=0,current=original;
  Object.defineProperty(target,'generateItemName',{get(){reads++;return current;},set(value){writes++;current=value;}});
  const runtime={...naming,console:{info(){},error:(...args)=>errors.push(args)},structuredClone,
    Hooks:{once:(key,fn)=>callbacks[key]=fn},queueMicrotask:fn=>microtasks.push(fn),
    game:{version:'14.368',system:{id:systemId,version},pf2e:{system:target},modules:new Map([['av-v14-hotfix',self]]),
      settings:{register:(id,key,config)=>settings.set(key,config.default),get:(id,key)=>settings.get(key)}},
    libWrapper:{register(){}},captureGridNative:()=>({}),
    hashSource:async value=>{const result=await hash(value);duringHash?.(runtime,target);return result;},
    registerBabeleIndex:()=>({status:()=>({state:'fixture'})})};
  for(const name of ['registerLegacyCompat','installGrid','installSundryPatch','installDurationPatch','installTimestampPatch','installChatDeleteCoalescing','installTokenizerChatPortraitPatch','installSoundStopPatch','registerDsnQualitySettings','installDsnQualityLocks','installBbmmHardLocks','installTurnLifecyclePatch','installDsnChatRecovery'])runtime[name]=()=>{};
  vm.runInNewContext(entry.replace(/^import .*;\r?\n/gm,''),runtime);
  callbacks.init();await callbacks.setup();callbacks.ready();for(const fn of microtasks)await fn();
  return {state:self.api.status().patches.itemNames,reads,writes,current,original,errors};
}

for(const version of ['8.5.0','8.5.1'])test(`entry installs names only for the exact ${version} release function`,async()=>{
 const r=await boot(version);assert.deepEqual(r.errors,[]);assert.equal(r.state.status,'installed');
 assert.equal(r.writes,1);assert.notEqual(r.current,r.original);
 const item={isOfType:()=>true,baseType:null,name:'custom item'};
 assert.equal(r.current(item),'custom item');
});
for(const [version,sourceVersion] of [['8.5.0','8.5.1'],['8.5.1','8.5.0']])test(`entry rejects ${sourceVersion} source with ${version} version`,async()=>{
 const r=await boot(version,{sourceVersion});assert.equal(r.state.status,'unsupported-source');assert.equal(r.writes,0);
});
test('unknown PF2e and other systems never read or replace the naming method',async()=>{
 for(const opts of [{version:'8.5.2'},{version:'8.5.1',systemId:'sf2e'}]){
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
