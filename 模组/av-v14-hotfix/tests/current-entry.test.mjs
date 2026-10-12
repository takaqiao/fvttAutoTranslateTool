import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
const source=fs.readFileSync(new URL('../av-v14-hotfix.mjs',import.meta.url),'utf8').replace(/^import .*;\r?\n/gm,'').replaceAll('import.meta.url',JSON.stringify(new URL('../av-v14-hotfix.mjs',import.meta.url).href));
const names=['prepareSundryPatch','prepareTurnLifecyclePatch','registerLegacyCompat','installGrid','installSundryPatch','installPatreonPatch','installDurationPatch','installTimestampPatch','installItemNameFastPath','installChatDeleteCoalescing','installTokenizerChatPortraitPatch','installSoundStopPatch','installWayfinderFogPatch','installBbmmHardLocks','registerDsnQualitySettings','installDsnQualityLocks','installTurnLifecyclePatch','installDsnChatRecovery','registerBabeleIndex'];
async function boot(disabled=[],generation=14,{globals={},modules=[],manual=false,throwOnError=true}={}){
 const hooks={},tasks=[],settings=new Map(),calls=[],phases=[],self={},logs=[];let phase='load';
 const log=level=>(...args)=>{logs.push({level,text:String(args[0]),args});if(level==='error'&&throwOnError)throw Error(args.join(' '));};
 const g={console:{info:log('info'),warn:log('warn'),error:log('error')},structuredClone,queueMicrotask:fn=>tasks.push(fn),
   Hooks:{once:(name,fn)=>hooks[name]=fn},libWrapper:{register(){}},
   game:{version:`${generation}.369`,release:{generation},system:{id:'pf2e',version:'8.5.1'},modules:new Map([['av-v14-hotfix',self]]),settings:{register:(id,key,def)=>settings.set(key,def),get:(id,key)=>!disabled.includes(key)}}};
 for(const name of names)g[name]=()=>{calls.push(name);phases.push({name,phase});};
 for(const id of modules)g.game.modules.set(id,{active:true,version:'fixture'});
 Object.assign(g,globals);
 vm.runInNewContext(source,g);phase='init';hooks.init();phase='setup';const setup=hooks.setup(),setupSyncCalls=[...calls];
 await setup;phase='ready';hooks.ready();phase='ready-async';
 const f={calls,phases,setupSyncCalls,settings,logs,tasks,api:self.api,get status(){return self.api.status();},
   lines:level=>logs.filter(entry=>entry.level===level).map(entry=>entry.text)};
 if(!manual)for(const task of tasks)await task();
 return f;
}
const reporting=(feature,result)=>({report})=>report({feature,...result});
test('diagnostic version matches the package manifest',async()=>{
 const f=await boot(),manifest=JSON.parse(fs.readFileSync(new URL('../module.json',import.meta.url),'utf8'));
 assert.equal(f.status.version,manifest.version);
});
test('retired index, duration and adventure-setting repairs keep a reason but no switch, installer or module file',async()=>{
 const f=await boot();
 for(const key of ['babele','duration'])assert.equal(f.settings.has(key),false,key);
 for(const key of ['babele','duration','legacyAdventureSettings']){
  assert.equal(f.status.patches[key].status,'retired',key);assert.equal(typeof f.status.patches[key].detail,'string',key);
 }
 assert(!f.calls.includes('installDurationPatch'));assert(!f.calls.includes('registerBabeleIndex'));
 assert.equal(Object.hasOwn(f.status,'babele'),false);
 for(const file of ['../scripts/patches/babele.mjs','../scripts/patches/duration.mjs'])assert.equal(fs.existsSync(new URL(file,import.meta.url)),false,file);
 assert.deepEqual([...f.settings.keys()].sort(),['chat','dsnChat','dsnQualityLocks','itemNames','legacy','soundStop','sundry','timestamps','tokenizerChat','turnLifecycle']);
});
test('every imported module file exists and exports the names the entry uses',async()=>{
 const entry=fs.readFileSync(new URL('../av-v14-hotfix.mjs',import.meta.url),'utf8');
 const imports=[...entry.matchAll(/^import \{([^}]+)\} from '([^']+)';$/gm)];
 assert.equal(imports.length,entry.split('\n').filter(line=>line.startsWith('import ')).length);
 for(const [,list,path] of imports){
  const module=await import(new URL(path,new URL('../av-v14-hotfix.mjs',import.meta.url)).href);
  for(const name of list.split(',').map(value=>value.trim()))assert.equal(typeof module[name],'function',`${path} ${name}`);
 }
});
test('current entry retires old Patreon/Wayfinder installers and exposes new fixes',async()=>{
 const f=await boot();
 assert(!f.calls.includes('installPatreonPatch'));assert(!f.calls.includes('installWayfinderFogPatch'));
 assert(!f.settings.has('patreon'));assert(!f.settings.has('wayfinderFog'));
 assert(f.calls.includes('installTurnLifecyclePatch'));assert(f.calls.includes('installDsnChatRecovery'));
 assert.equal(f.status.patches.patreon.status,'retired');assert.equal(f.status.patches.wayfinderFog.status,'retired');
 assert.doesNotThrow(()=>structuredClone(f.status));
});

test('retired Grid and BBMM repairs neither register switches nor run installers',async()=>{
 const f=await boot();
 assert(!f.calls.includes('installGrid'));assert(!f.calls.includes('installBbmmHardLocks'));
 assert(!f.settings.has('grid'));assert(!f.settings.has('bbmmLocks'));
 assert.equal(f.status.patches.grid.status,'retired');assert.equal(f.status.patches.bbmmLocks.status,'retired');
 assert(f.calls.includes('registerDsnQualitySettings'));assert(f.calls.includes('installDsnQualityLocks'));
});
test('new fixes honor their world switches',async()=>{
 const f=await boot(['turnLifecycle','dsnChat','timestamps','itemNames','chat']);
 for(const name of ['installTurnLifecyclePatch','installDsnChatRecovery','installTimestampPatch','installItemNameFastPath','installChatDeleteCoalescing'])assert(!f.calls.includes(name),name);
 for(const key of ['turnLifecycle','dsnChat','timestamps','itemNames','chat'])assert.equal(f.status.patches[key].status,'disabled',key);
 assert.deepEqual(f.lines('warn'),[]);
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
test('the timestamp wrapper registers synchronously in setup, before the awaited legacy step and only once',async()=>{
 const f=await boot(),calls=f.phases.filter(call=>call.name==='installTimestampPatch');
 assert.deepEqual(calls.map(call=>call.phase),['setup']);
 assert(f.setupSyncCalls.includes('installTimestampPatch'));
 assert(f.setupSyncCalls.indexOf('installTimestampPatch')<f.setupSyncCalls.indexOf('registerLegacyCompat'));
});
test('ready installs run in order, each exactly once, after setup',async()=>{
 const f=await boot(),ready=f.phases.filter(call=>call.phase==='ready-async').map(call=>call.name);
 assert.deepEqual(ready,['installItemNameFastPath','prepareSundryPatch','installSundryPatch','prepareTurnLifecyclePatch','installTurnLifecyclePatch','installDsnChatRecovery','installChatDeleteCoalescing']);
});

test('later core generations reach independently guarded installers and ready diagnostics',async()=>{
 const f=await boot([],15);
 assert(f.calls.includes('registerLegacyCompat'));assert(f.calls.includes('installSoundStopPatch'));
 assert(f.calls.includes('installDsnChatRecovery'));assert(f.calls.includes('installSundryPatch'));
 assert.equal(f.status.patches.runtime.status,'ready');assert.equal(typeof f.status.patches.runtime.elapsedMs,'number');
});

test('status keys follow the world switches rather than the internal feature names',async()=>{
 const f=await boot([],14,{globals:{
  registerLegacyCompat:reporting('legacy-compat',{moduleId:'av-v14-hotfix',status:'installed',chatStatus:'installed',sceneStatus:'installed'}),
  installTurnLifecyclePatch:reporting('turn-lifecycle',{status:'installed',parts:{reaction:{status:'installed'}}})}});
 assert.equal(f.status.patches.legacy.status,'installed');assert.equal(f.status.patches.legacy.feature,'legacy');
 assert.equal(f.status.patches.turnLifecycle.status,'installed');
 assert.equal(Object.hasOwn(f.status.patches,'legacy-compat'),false);assert.equal(Object.hasOwn(f.status.patches,'turn-lifecycle'),false);
 assert(f.lines('info').includes('av-v14-hotfix | legacy: installed'));assert(f.lines('info').includes('av-v14-hotfix | turnLifecycle: installed'));
});

test('a repair that should be running and is not warns once and is listed again after startup',async()=>{
 let again;
 const f=await boot([],14,{globals:{
  installSundryPatch:({report})=>{again=()=>report({feature:'sundry',status:'skipped',reason:'source-contract-mismatch'});again();},
  installTokenizerChatPortraitPatch:reporting('tokenizerChat',{status:'unsupported-runtime'}),
  installItemNameFastPath:reporting('itemNames',{status:'behavior-mismatch',checked:3,detail:{uuid:'Item.a',type:'weapon'}})}});
 const warnings=f.lines('warn');
 assert.deepEqual(warnings.slice(0,-1).sort(),['av-v14-hotfix | itemNames is not active: behavior-mismatch','av-v14-hotfix | sundry is not active: skipped (source-contract-mismatch)','av-v14-hotfix | tokenizerChat is not active: unsupported-runtime']);
 assert.match(warnings.at(-1),/^av-v14-hotfix \| 3 repair\(s\) not active after startup: /);
 for(const text of ['tokenizerChat: unsupported-runtime','itemNames: behavior-mismatch','sundry: skipped (source-contract-mismatch)'])assert(warnings.at(-1).includes(text),text);
 again();assert.equal(f.lines('warn').length,warnings.length);
});
test('expected idle states stay quiet',async()=>{
 const f=await boot([],14,{globals:{
  installSundryPatch:reporting('sundry',{status:'skipped',reason:'feature-disabled'}),
  installItemNameFastPath:reporting('itemNames',{status:'skipped',reason:'no-generated-name-sample',checked:4}),
  installDsnChatRecovery:reporting('dsnChat',{status:'waiting-dsn'}),
  installDsnQualityLocks:reporting('dsnQualityLocks',{status:'inactive-bbmm'}),
  installTokenizerChatPortraitPatch:reporting('tokenizerChat',{status:'unsupported-system'}),
  installTimestampPatch:reporting('timestamps',{status:'already-installed'}),
  installTurnLifecyclePatch:reporting('turn-lifecycle',{status:'skipped',reason:'no-supported-consumers',parts:{
   reaction:{status:'skipped',reason:'module-inactive'},sustain:{status:'skipped',reason:'module-inactive'},summons:{status:'skipped',reason:'feature-disabled'}}})}});
 assert.deepEqual(f.lines('warn'),[]);
 const info=f.lines('info');
 assert(info.includes('av-v14-hotfix | sundry: skipped (feature-disabled)'));assert(info.includes('av-v14-hotfix | dsnChat: waiting-dsn'));
 assert(info.includes('av-v14-hotfix | turnLifecycle: skipped (no-supported-consumers, reaction module-inactive, sustain module-inactive, summons feature-disabled)'));
});
test('an installed repair whose parts are still waiting says so without warning',async()=>{
 const f=await boot([],14,{globals:{
  installDsnChatRecovery:reporting('dsnChat',{status:'installed',queueStatus:'waiting-dsn',queueCompletionStatus:'waiting-dsn',modelStatus:'installed'}),
  installTurnLifecyclePatch:reporting('turn-lifecycle',{status:'installed',parts:{reaction:{status:'skipped',reason:'module-inactive'},sustain:{status:'installed'},summons:{status:'skipped',reason:'feature-disabled'}}})}});
 assert.deepEqual(f.lines('warn'),[]);
 assert(f.lines('info').includes('av-v14-hotfix | dsnChat: installed (queueStatus waiting-dsn, queueCompletionStatus waiting-dsn)'));
 assert(f.lines('info').includes('av-v14-hotfix | turnLifecycle: installed (reaction module-inactive, summons feature-disabled)'));
});
test('partial failures inside an installed repair are named',async()=>{
 const f=await boot([],14,{globals:{
  registerLegacyCompat:reporting('legacy-compat',{status:'installed',chatStatus:'installed',sceneStatus:'unsupported-schema'}),
  installDsnChatRecovery:reporting('dsnChat',{status:'installed',queueStatus:'unsupported-queue',queueCompletionStatus:'unsupported-queue',modelStatus:'waiting-dsn'}),
  installTurnLifecyclePatch:reporting('turn-lifecycle',{status:'installed',parts:{
   reaction:{status:'skipped',reason:'callback-fingerprint-mismatch',hook:'pf2e.startTurn',matches:0},sustain:{status:'installed'},summons:{status:'skipped',reason:'feature-disabled'}}})}});
 const warnings=f.lines('warn');
 assert(warnings.includes('av-v14-hotfix | legacy is not active: sceneStatus unsupported-schema'));
 assert(warnings.includes('av-v14-hotfix | dsnChat is not active: queueStatus unsupported-queue; queueCompletionStatus unsupported-queue'));
 assert(warnings.includes('av-v14-hotfix | turnLifecycle is not active: reaction skipped (callback-fingerprint-mismatch)'));
});
test('counters stay live in status and never cause another log line; a changed outcome does',async()=>{
 const stats={fast:0,delegated:0};let report;
 const f=await boot([],14,{globals:{installSundryPatch:options=>{report=options.report;report({feature:'sundry',status:'installed',stats,hooks:[{hook:'refreshToken',id:1}],restore(){}});}}});
 assert.deepEqual(f.status.patches.sundry.stats,{fast:0,delegated:0});assert.equal(Object.hasOwn(f.status.patches.sundry,'restore'),false);
 stats.fast=7;stats.delegated=2;
 assert.deepEqual(f.status.patches.sundry.stats,{fast:7,delegated:2});
 const before=f.logs.length;
 report({feature:'sundry',status:'installed',stats,hooks:[{hook:'refreshToken',id:9}]});
 assert.equal(f.logs.length,before);
 report({feature:'sundry',status:'skipped',reason:'core-hooks-fingerprint-mismatch'});
 assert.equal(f.lines('warn').at(-1),'av-v14-hotfix | sundry is not active: skipped (core-hooks-fingerprint-mismatch)');
 assert.doesNotThrow(()=>structuredClone(f.status));
});
test('status names the ready step in progress and how long it has been waiting',async()=>{
 let release;const gate=new Promise(resolve=>{release=resolve;});
 const f=await boot([],14,{manual:true,globals:{prepareSundryPatch:()=>gate}});
 assert.equal(f.status.patches.runtime,undefined);
 const running=f.tasks[0]();
 for(let i=0;i<10;i++)await null;
 const runtime=f.status.patches.runtime;
 assert.equal(runtime.status,'initializing');assert.equal(runtime.step,'sundry');
 assert.equal(typeof runtime.since,'number');assert(runtime.waitedMs>=0);
 release();await running;
 assert.equal(f.status.patches.runtime.status,'ready');assert.equal(Object.hasOwn(f.status.patches.runtime,'waitedMs'),false);
});
test('a throwing installer is recorded as failed, warned about and does not stop later steps',async()=>{
 const f=await boot([],14,{throwOnError:false,globals:{installSundryPatch:()=>{throw new TypeError('boom');}}});
 assert.equal(f.status.patches.sundry.status,'failed');assert.equal(f.status.patches.sundry.detail,'TypeError: boom');
 assert(f.lines('warn').includes('av-v14-hotfix | sundry is not active: failed'));assert.equal(f.lines('error').length,1);
 assert(f.calls.includes('installChatDeleteCoalescing'));assert.equal(f.status.patches.runtime.status,'ready');
});
