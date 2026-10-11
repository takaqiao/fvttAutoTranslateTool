import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import fs from 'node:fs';
const source=fs.readFileSync(new URL('../av-v14-hotfix.mjs',import.meta.url),'utf8').replace(/^import .*;\r?\n/gm,'').replaceAll('import.meta.url',JSON.stringify(new URL('../av-v14-hotfix.mjs',import.meta.url).href));
async function boot(enabled){
  const hooks={},tasks=[],settings=new Map(),self={},calls=[];let phase='load';
  const runtime={console:{info(){},error(){}},structuredClone,queueMicrotask:fn=>tasks.push(fn),Hooks:{once:(name,fn)=>hooks[name]=fn},hashSource:async()=>'',
    game:{version:'14.368',release:{generation:14},modules:new Map([['av-v14-hotfix',self]]),settings:{register:(id,key,value)=>settings.set(key,value),get:(id,key)=>enabled.includes(key)}},
    registerDsnQualitySettings:({report})=>{calls.push(['register',phase]);report?.({feature:'dsnQualityLocks',status:'registered'});},
    installDsnQualityLocks:({report})=>{calls.push(['dsn',phase]);report({feature:'dsnQualityLocks',status:'installed',restore(){}});},
    installBbmmHardLocks:({report})=>{calls.push(['bbmm',phase]);report({feature:'bbmmLocks',status:'installed',restore(){}});}};
  vm.runInNewContext(source,runtime);phase='init';hooks.init();phase='setup';await hooks.setup();phase='ready';hooks.ready();for(const task of tasks)await task();
  return{calls,settings,state:self.api.status()};
}
test('quality bridge registers before setup; retired BBMM has no switch or installer',async()=>{
 const e=await boot(['bbmmLocks','dsnQualityLocks']);assert.deepEqual(e.calls,[['register','init'],['dsn','setup']]);
 const cfg=e.settings.get('dsnQualityLocks');assert.equal(cfg.default,true);assert.equal(cfg.scope,'world');assert.equal(cfg.requiresReload,true);
 assert.equal(e.settings.has('bbmmLocks'),false);assert.equal(e.state.patches.bbmmLocks.status,'retired');
 assert.equal(e.state.patches.dsnQualityLocks.status,'installed');assert.doesNotThrow(()=>structuredClone(e.state));
});
for(const enabled of [[],['bbmmLocks'],['dsnQualityLocks']])test('quality bridge honors its remaining switch '+JSON.stringify(enabled),async()=>{
 const e=await boot(enabled);assert.deepEqual(e.calls.filter(([k])=>k!=='register').map(([k])=>k),enabled.includes('dsnQualityLocks')?['dsn']:[]);
 if(!enabled.includes('dsnQualityLocks'))assert.ok(!e.calls.some(([k])=>k==='register'));
});
