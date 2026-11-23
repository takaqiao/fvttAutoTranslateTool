import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import * as api from '../scripts/dual-strike-critical-compat.mjs';
const MARKER='pf2e-third-party-automation:dual-strike-critical-specialization',GRAPH='6JdDvJY9YKdxoi2o',NODE='oYYR4W7IaJpI1OpI',SETTING='trigger-engine.pf2e-trigger-triggers';
const captured='C:/Users/Taka/Desktop/fvtt/work/bob2-tpa-gap-research-20261101';
const graph=JSON.parse(fs.readFileSync(`${captured}/snapshot/modules/pf2e-trigger-trove/triggers/base-triggers.json`,'utf8')).find(graph=>graph.id===GRAPH);
const pf=fs.readFileSync(process.env.PF2E_NATIVE_861_BUNDLE??`${captured}/snapshot/system/pf2e.mjs`,'utf8');
const Predicate=Function('M',`let Wn=${pf.match(/Wn = (class Predicate[\s\S]*?), AutomaticBonusProgression\$1 =/)[1]};return Wn;`)(x=>!!x&&typeof x==='object'&&!Array.isArray(x));
const predicate=source=>JSON.parse(source.nodes.find(node=>node.id===NODE).inputs.predicate.value);
function fixture({world='bob-2',patreon=true,gm=true,disabled=false,sources=[],enabled=[GRAPH],graphs=[graph]}={}){
 const user={id:gm?'gm':'player'},activeGM={id:'gm'},writes=[];
 const data=new Map([[SETTING,{enabled:structuredClone(enabled),disabled:disabled?[GRAPH]:[],folders:[{name:'My folder'}],sources:structuredClone(sources)}],['pf2e-third-party-automation.configurationBackups',[]],['patreon-v3.rulesV3',{uYv7S8tdeXXpmd8a:{isActive:patreon,handlerRules:[{triggerType:'damage-roll',value:'criticalSpecializationSword'}]}}]]);
 const game={world:{id:world},system:{id:'pf2e'},user,users:{activeGM},modules:new Map(['patreon-v3','trigger-engine','pf2e-trigger-trove'].map(id=>[id,{active:true}])),settings:{get:(module,key)=>data.get(`${module}.${key}`),set:async(module,key,value)=>{data.set(`${module}.${key}`,structuredClone(value));writes.push({key:`${module}.${key}`,value:structuredClone(value)});}}};
 const initialize=async()=>api.createDualStrikeCriticalCompatibility?.({game,loadGraphs:async()=>structuredClone(graphs)}).initialize();
 return {game,data,writes,initialize};
}
test('the saved same-id world override excludes only marked Double Slice specialization',async()=>{
 const f=fixture(),before=structuredClone(f.data.get(SETTING)),original=structuredClone(graph);await f.initialize();
 const after=f.data.get(SETTING);assert.equal(after.sources.length,1);
 const options=['check:outcome:critical-success','critical-specialization','item:group:sword'];
 assert.equal(Predicate.test(predicate(after.sources[0]),options),true);
 assert.equal(Predicate.test(predicate(after.sources[0]),[...options,MARKER]),false);
 const restored=structuredClone(after.sources[0]);restored.nodes.find(node=>node.id===NODE).inputs.predicate.value=graph.nodes.find(node=>node.id===NODE).inputs.predicate.value;
 assert.deepEqual(restored,original);assert.deepEqual(graph,original);
 assert.deepEqual({...after,sources:[]},before);
 assert.deepEqual(f.writes.map(write=>write.key),['pf2e-third-party-automation.configurationBackups',SETTING]);
 const backup=f.writes[0].value[0];assert.deepEqual(backup.changes[0].before,before);
 await f.initialize();assert.equal(f.writes.length,2);
});
test('an existing same-id override keeps its customization and other graphs',async()=>{
 const custom=structuredClone(graph);custom.name='My sword rule';custom.nodes.find(node=>node.type==='create-condition').inputs.name={value:'My effect'};
 const other={id:'OtherGraph',nodes:[]},f=fixture({sources:[custom,other]});await f.initialize();
 const after=f.data.get(SETTING).sources;assert.equal(after.length,2);assert.equal(after[0].name,'My sword rule');assert.deepEqual(after[1],other);
 assert.equal(Predicate.test(predicate(after[0]),['check:outcome:critical-success','critical-specialization',MARKER]),false);
});
test('a changed custom predicate is left untouched and reported',async()=>{
 const custom=structuredClone(graph);custom.nodes.find(node=>node.id===NODE).inputs.predicate.value='["my-custom-predicate"]';const f=fixture({sources:[custom]});
 await assert.rejects(f.initialize(),/专精|predicate/);assert.equal(f.writes.length,0);assert.deepEqual(f.data.get(SETTING).sources,[custom]);
});
for(const options of [{world:'-'},{patreon:false},{gm:false},{disabled:true}])test(`critical specialization compatibility leaves unrelated settings alone: ${JSON.stringify(options)}`,async()=>{
 const f=fixture(options);await f.initialize();assert.equal(f.writes.length,0);
});
test('restoring sword metadata waits for the Trove exclusion when both consumers are enabled',async()=>{
 const f=fixture();assert.equal(api.dualStrikeSwordAutomationReady?.(f.game)??false,false);
 await f.initialize();assert.equal(api.dualStrikeSwordAutomationReady?.(f.game)??false,true);
 f.data.get('patreon-v3.rulesV3').uYv7S8tdeXXpmd8a.isActive=false;assert.equal(api.dualStrikeSwordAutomationReady?.(f.game)??false,false);
});
test('the current Engine prepares one graph and one event after applying the same-id override',async()=>{
 const source=fs.readFileSync(`${captured}/snapshot/modules/trigger-engine/scripts/main.js`,'utf8');
 const section=(start,end)=>{const a=source.indexOf(start),b=source.indexOf(end,a+start.length);assert(a>=0&&b>a,start);return source.slice(a,b);};
 const methods=section('getTriggersSetting(){if(!this.isSettingApplication','async _executeEvent(')+section('async prepare(){let e=performance.now(),r=this.getTriggersSetting()','addFile(e)');
 const f=fixture();
 const NativeApplication=Function('game','KS','c','Zw','Vw','I',`return class {
  #g={};#y={};#p=[];#s;
  isSettingApplication=true;moduleId='trigger-engine';settingKey='pf2e-trigger-triggers';
  moduleSources=[];events=new Map([['damage-taken-event',true]]);
  createTrigger(source){return {data:{...source,toObject:()=>structuredClone(source)}};}
  getMenuApplication(){}inspect(){return {graphs:Object.values(this.#g),events:this.#y['damage-taken-event']};}
  ${methods}
 }`)(f.game,{deepClone:structuredClone},{isIncludedIn:(value,list)=>list.includes(value),values:Object.values,pipe:(value,...fns)=>fns.reduce((result,fn)=>fn(result),value),sortBy:()=>values=>values,prop:key=>value=>value[key]},source=>!!source?.id,()=>true,{group(){},debug(){},groupEnd(){}});
 const application=new NativeApplication();application.moduleSources=[structuredClone(graph)];await application.prepare();
 assert.equal(application.inspect().graphs.length,1);assert.equal(application.inspect().events.length,1);
 assert.equal(Predicate.test(predicate(application.inspect().graphs[0]),['check:outcome:critical-success','critical-specialization',MARKER]),true);
 await f.initialize();await application.prepare();
 const prepared=application.inspect();assert.equal(prepared.graphs.length,1);assert.equal(prepared.events.length,1);
 assert.equal(Predicate.test(predicate(prepared.graphs[0]),['check:outcome:critical-success','critical-specialization']),true);
 assert.equal(Predicate.test(predicate(prepared.graphs[0]),['check:outcome:critical-success','critical-specialization',MARKER]),false);
});

// Trigger Trove 2.5.0 (installed 11-23) rebuilt this graph under a new id and added an axe
// branch; trigger-engine 1.38.0 is the engine installed beside it.
const NEW='NzgcT7S1lbrIAVRb',work='C:/Users/Taka/Desktop/fvtt/work';
const troveGraphs=JSON.parse(fs.readFileSync(process.env.TROVE_250_BASE_TRIGGERS??`${work}/main-compat-20261123/live/modules/pf2e-trigger-trove/triggers/base-triggers.json`,'utf8'));
const current=troveGraphs.find(source=>source.id===NEW);
const excluded=JSON.stringify(['check:outcome:critical-success','critical-specialization',{not:MARKER}]);
const withPredicate=(source,value)=>{const copy=structuredClone(source);copy.nodes.find(node=>node.id===NODE).inputs.predicate.value=value;return copy;};
const tpaOldCopy=withPredicate(graph,excluded),tpaNewCopy=withPredicate(current,excluded);
const others=[{id:'OtherGraphA',nodes:[]},{id:'OtherGraphB',nodes:[]}];
test('Trove 2.5.0 ships the new graph with the same predicate node and an axe branch',()=>{
 assert.equal(troveGraphs.some(source=>source.id===GRAPH),false);
 assert.equal(current.nodes.length,29);assert.equal(graph.nodes.length,18);
 assert.deepEqual(predicate(current),predicate(graph));
 assert.ok(JSON.stringify(current).includes('CriticalSpecialization.axe'));
});
test('the TPA copy of the old graph becomes a copy of the new graph, and the new id takes the old id\'s place',()=>{
 const setting={enabled:['first',GRAPH,'last'],disabled:[],folders:{a:{name:'A'}},sources:[others[0],tpaOldCopy,others[1]]};
 const result=api.buildDualStrikeCriticalCompatibility(structuredClone(setting),{graph:current});
 assert.deepEqual(result,{...setting,enabled:['first',NEW,'last'],sources:[others[0],tpaNewCopy,others[1]]});
 assert.equal(Predicate.test(predicate(result.sources[1]),['check:outcome:critical-success','critical-specialization','item:group:axe']),true);
 assert.equal(Predicate.test(predicate(result.sources[1]),['check:outcome:critical-success','critical-specialization',MARKER]),false);
 assert.deepEqual(api.buildDualStrikeCriticalCompatibility(structuredClone(result),{graph:current}),result,'a migrated setting comes back unchanged');
});
test('the old id enabled without a world copy is replaced by the new id and a TPA copy',()=>{
 const result=api.buildDualStrikeCriticalCompatibility({enabled:[GRAPH],disabled:[],folders:{},sources:[]},{graph:current});
 assert.deepEqual(result,{enabled:[NEW],disabled:[],folders:{},sources:[tpaNewCopy]});
});
test('an old id listed beside the new one is dropped rather than duplicated',()=>{
 const result=api.buildDualStrikeCriticalCompatibility({enabled:[NEW,GRAPH],disabled:[],folders:{},sources:[tpaOldCopy,tpaNewCopy]},{graph:current});
 assert.deepEqual(result,{enabled:[NEW],disabled:[],folders:{},sources:[tpaNewCopy]});
});
for(const [name,value]of [['base','["check:outcome:critical-success","critical-specialization"]'],['custom','["my-custom-predicate"]']])test(`an old-id copy with a ${name} predicate is not TPA's and is left untouched`,async()=>{
 const stale=withPredicate(graph,value),f=fixture({sources:[stale],graphs:troveGraphs});
 assert.throws(()=>api.buildDualStrikeCriticalCompatibility(f.data.get(SETTING),{graph:current}),/专精/);
 await assert.rejects(f.initialize(),/专精/);assert.equal(f.writes.length,0);assert.deepEqual(f.data.get(SETTING).sources,[stale]);
});
test('while Trove still ships the old graph, the old id stays current',()=>{
 const setting={enabled:[GRAPH],disabled:[],folders:{},sources:[tpaOldCopy]};
 assert.deepEqual(api.buildDualStrikeCriticalCompatibility(structuredClone(setting),{graph}),setting);
});
test('a graph under an unknown id is refused',()=>{
 assert.throws(()=>api.buildDualStrikeCriticalCompatibility({enabled:[],sources:[]},{graph:{...current,id:'SomethingElse0000'}}),/重击专精/);
 assert.throws(()=>api.buildDualStrikeCriticalCompatibility({enabled:[],sources:[]},{graph:undefined}),/重击专精/);
});
test('initialize migrates the saved setting from the Trove file once',async()=>{
 const f=fixture({sources:[tpaOldCopy,others[0]],graphs:troveGraphs}),before=structuredClone(f.data.get(SETTING));
 await f.initialize();
 assert.deepEqual(f.data.get(SETTING),{...before,enabled:[NEW],sources:[tpaNewCopy,others[0]]});
 assert.deepEqual(f.writes.map(write=>write.key),['pf2e-third-party-automation.configurationBackups',SETTING]);
 assert.deepEqual(f.writes[0].value[0].changes[0].before,before);
 await f.initialize();assert.equal(f.writes.length,2);
});
test('sword readiness follows whichever Critical Specialization graph can run',()=>{
 const ready=setting=>{const f=fixture();f.data.set(SETTING,setting);return api.dualStrikeSwordAutomationReady(f.game);};
 assert.equal(ready({enabled:[GRAPH],disabled:[],sources:[tpaOldCopy]}),true,'unmigrated: the old copy carries the exclusion');
 assert.equal(ready({enabled:[NEW],disabled:[],sources:[]}),false,'the new module graph runs without the exclusion');
 assert.equal(ready({enabled:[GRAPH,NEW],disabled:[],sources:[tpaOldCopy]}),false);
 assert.equal(ready({enabled:[NEW],disabled:[],sources:[tpaNewCopy]}),true);
 assert.equal(ready({enabled:[],disabled:[],sources:[]}),true);
});
test('trigger-engine 1.38.0 runs one excluded graph with the axe branch after the migration',async()=>{
 const source=fs.readFileSync(process.env.TRIGGER_ENGINE_1380_SOURCE??`${work}/main-compat-20261108/live/modules/trigger-engine/scripts/main.js`,'utf8');
 const section=(start,end)=>{const a=source.indexOf(start),b=source.indexOf(end,a+start.length);assert(a>=0&&b>a,start);return source.slice(a,b);};
 const methods=section('getTriggersSetting(){if(!this.isSettingApplication','async _executeEvent(')+section('async prepare(){let e=performance.now(),r=this.getTriggersSetting()','addFile(e)');
 const f=fixture({sources:[tpaOldCopy],graphs:troveGraphs});
 const Application=Function('game','WS','c','Vw','Bw','I','mr',`return class {
  #a;#g={};#y={};#p=[];
  isSettingApplication=true;moduleId='trigger-engine';settingKey='pf2e-trigger-triggers';applicationKey='trigger-engine';
  moduleSources=[];events=new Map([['damage-taken-event',true]]);
  createTrigger(source){return {data:{...source,toObject:()=>structuredClone(source)}};}
  getMenuApplication(){}inspect(){return {graphs:Object.values(this.#g),events:this.#y['damage-taken-event']??[]};}
  ${methods}
 }`)(f.game,{deepClone:structuredClone},{isIncludedIn:(value,list)=>list.includes(value),values:Object.values,keys:Object.keys,isArray:Array.isArray,pipe:(value,...fns)=>fns.reduce((result,fn)=>fn(result),value),sortBy:()=>values=>values,prop:key=>value=>value[key]},source=>!!source?.id,()=>true,{group(){},debug(){},groupEnd(){}},()=>false);
 const application=new Application();application.moduleSources=structuredClone(troveGraphs);
 const crit=()=>application.inspect().graphs.filter(prepared=>[GRAPH,NEW].includes(prepared.id));
 await application.prepare();
 assert.deepEqual(crit().map(prepared=>[prepared.id,prepared.nodes.length]),[[GRAPH,18]],'before: only the old world copy runs');
 await f.initialize();await application.prepare();
 assert.deepEqual(crit().map(prepared=>[prepared.id,prepared.nodes.length]),[[NEW,29]]);
 assert.equal(application.inspect().events.filter(event=>[GRAPH,NEW].includes(event.data.id)).length,1);
 assert.equal(Predicate.test(predicate(crit()[0]),['check:outcome:critical-success','critical-specialization','item:group:axe']),true);
 assert.equal(Predicate.test(predicate(crit()[0]),['check:outcome:critical-success','critical-specialization',MARKER]),false);
});
