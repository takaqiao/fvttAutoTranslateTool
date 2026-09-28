import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
const source=fs.readFileSync(new URL('../av-v14-hotfix.mjs',import.meta.url),'utf8').replace(/^import .*;\r?\n/gm,'');
async function boot(disabled=[]){
 const hooks={},tasks=[],settings=new Map(),calls=[],phases=[],self={};let phase='load';
 const names=['registerLegacyCompat','installGrid','installSundryPatch','installPatreonPatch','installDurationPatch','installTimestampPatch','installTokenizerChatPortraitPatch','installSoundStopPatch','installWayfinderFogPatch','installBbmmHardLocks','registerDsnQualitySettings','installDsnQualityLocks','installTurnLifecyclePatch','installDsnChatRecovery'];
 const g={console:{info(){},error(...args){throw Error(args.join(' '));}},structuredClone,queueMicrotask:fn=>tasks.push(fn),
   Hooks:{once:(name,fn)=>hooks[name]=fn},captureGridNative:()=>({}),hashSource:async()=>'',ITEM_NAME_HASHES:{},libWrapper:{register(){}},
   installChatDeleteCoalescing:()=>true,registerBabeleIndex:()=>({status:()=>({state:'inactive'})}),
   game:{version:'14.368',release:{generation:14},system:{id:'pf2e',version:'8.5.1'},modules:new Map([['av-v14-hotfix',self]]),settings:{register:(id,key,def)=>settings.set(key,def),get:(id,key)=>!disabled.includes(key)}}};
 for(const name of names)g[name]=()=>{calls.push(name);phases.push({name,phase});};
 vm.runInNewContext(source,g);phase='init';hooks.init();phase='setup';const setup=hooks.setup(),setupSyncCalls=[...calls];
 await setup;phase='ready';hooks.ready();phase='ready-async';for(const task of tasks)await task();
 return {calls,phases,setupSyncCalls,settings,status:self.api.status()};
}
test('current entry retires old Patreon/Wayfinder installers and exposes new fixes',async()=>{
 const f=await boot();
 assert(!f.calls.includes('installPatreonPatch'));assert(!f.calls.includes('installWayfinderFogPatch'));
 assert(!f.settings.has('patreon'));assert(!f.settings.has('wayfinderFog'));
 assert(f.calls.includes('installTurnLifecyclePatch'));assert(f.calls.includes('installDsnChatRecovery'));
 assert.equal(f.status.patches.patreon.status,'retired');assert.equal(f.status.patches.wayfinderFog.status,'retired');
 assert.doesNotThrow(()=>structuredClone(f.status));
});
test('new fixes honor their world switches',async()=>{
 const f=await boot(['turnLifecycle','dsnChat']);
 assert(!f.calls.includes('installTurnLifecyclePatch'));assert(!f.calls.includes('installDsnChatRecovery'));
 assert.equal(f.status.patches.turnLifecycle.status,'disabled');assert.equal(f.status.patches.dsnChat.status,'disabled');
});

test('DsN model recovery registers synchronously in setup before asynchronous ready work',async()=>{
 const enabled=await boot(),calls=enabled.phases.filter(call=>call.name==='installDsnChatRecovery');
 assert(enabled.setupSyncCalls.includes('installDsnChatRecovery'));assert.equal(calls[0].phase,'setup');
 assert(calls.some(call=>call.phase==='ready-async'));
 const disabled=await boot(['dsnChat']);
 assert(!disabled.setupSyncCalls.includes('installDsnChatRecovery'));
 assert(!disabled.phases.some(call=>call.name==='installDsnChatRecovery'));
 assert.equal(disabled.status.patches.dsnChat.status,'disabled');
});
