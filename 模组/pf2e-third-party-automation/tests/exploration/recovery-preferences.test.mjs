import {test} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createRecoveryPanel} from '../../scripts/exploration/panel.mjs';
import * as schema from '../../scripts/exploration/schema.mjs';

const P='Actor.P',OTHER='Actor.OTHER';
const preferences=()=>({version:1,targetIntentsByActor:{[P]:{mode:'percent',value:25},[OTHER]:{mode:'absolute',value:70}},requireNoWounded:false,failureStop:{enabled:false,limit:3}});
const deferred=()=>{let resolve;return {promise:new Promise(r=>resolve=r),resolve}};

function panelFixture(t,{policy={recovery:preferences()},snapshotGate}={}){
 const previous=globalThis.foundry;t.after(()=>globalThis.foundry=previous);
 globalThis.foundry={applications:{api:{ApplicationV2:class{render(){return this}}}}};
 const actors=[{actorUUID:P,name:'P',hp:{value:1,max:41},focus:{value:0,max:0},pool:{poolUUID:P,ready:true},wardCapacity:1}],calls=[],saved=[];
 const game={user:{isGM:true,getFlag:()=>policy,setFlag:async(_module,_key,value)=>{saved.push(value);policy=value}},messages:new Map()};
 const app=createRecoveryPanel({game,coordinator:{},getSessionId:()=>null,capabilities:{snapshot:async()=>{await snapshotGate?.();return actors}},start:async config=>calls.push(config)}).open([P]);
 const fields={budget:{value:'120'},activityBudget:{value:'100'},risky:{checked:false},focus:{checked:true},extension:{checked:false},assurance:{checked:false},rank:{value:'trained'},manualFirstRound:{checked:false},activityFirstRound:{checked:false}},target={dataset:{pool:P},value:'11'};
 const content={querySelector:selector=>fields[selector.match(/name=(\w+)/)[1]],querySelectorAll:()=>[target]};
 return {app,actors,calls,saved,content,target,policy};
}

test('the existing pool input previews saved intentions and an unchanged value preserves them',async t=>{
 const f=panelFixture(t);assert.match(await f.app._renderHTML(await f.app._prepareContext()),/data-pool="Actor.P"[^>]*value="11"/);
 await f.app.act('start',f.content);
 assert.deepEqual(f.calls[0].recovery.targetIntentsByActor,{[P]:{mode:'percent',value:25}});
 assert.deepEqual(f.saved[0].recovery.targetIntentsByActor[OTHER],{mode:'absolute',value:70});assert.equal(f.saved[0].requireFullFocus,true);
 f.actors[0].hp.max=81;assert.match(await f.app._renderHTML(await f.app._prepareContext()),/data-pool="Actor.P"[^>]*value="21"/);
 f.target.value='21';await f.app.act('start',f.content);assert.equal(f.calls[1].recovery.targetIntentsByActor[P].value,25);
});

test('editing the existing pool target saves an absolute intention without deleting other patients',async t=>{
 const f=panelFixture(t);await f.app._renderHTML(await f.app._prepareContext());f.target.value='10';await f.app.act('start',f.content);
 assert.deepEqual(f.calls[0].recovery.targetIntentsByActor,{[P]:{mode:'absolute',value:10}});
 assert.deepEqual(f.saved[0].recovery.targetIntentsByActor[OTHER],{mode:'absolute',value:70});
});

test('an unchanged rendered target keeps its intention when max changes before clicking start',async t=>{
 const f=panelFixture(t);await f.app._renderHTML(await f.app._prepareContext());f.actors[0].hp.max=81;await f.app.act('start',f.content);
 assert.deepEqual(f.calls[0].recovery.targetIntentsByActor[P],{mode:'percent',value:25});
});

test('a refresh during the start snapshot cannot reinterpret the captured pool input as an edit',async t=>{
 const entered=deferred(),release=deferred();let snapshots=0;
 const f=panelFixture(t,{snapshotGate:async()=>{if(++snapshots===2){entered.resolve();await release.promise}}});
 await f.app._renderHTML(await f.app._prepareContext());const pending=f.app.act('start',f.content);await entered.promise;
 f.actors[0].hp.max=81;await f.app._renderHTML(await f.app._prepareContext());release.resolve();await pending;
 assert.deepEqual(f.calls[0].recovery.targetIntentsByActor[P],{mode:'percent',value:25});
});

test('pool edits update every selected patient in that pool while preserving unselected preferences',async t=>{
 const f=panelFixture(t);f.app.actorUUIDs.push('Actor.B');f.actors.push({...f.actors[0],actorUUID:'Actor.B',name:'B'});
 await f.app._renderHTML(await f.app._prepareContext());f.target.value='9';await f.app.act('start',f.content);
 assert.deepEqual(f.calls[0].recovery.targetIntentsByActor,{[P]:{mode:'absolute',value:9},'Actor.B':{mode:'absolute',value:9}});
 assert.equal(f.saved[0].recovery.targetIntentsByActor[OTHER].value,70);
});

test('bad saved preferences fall back to max with a visible reason and never execute a getter',async t=>{
 let reads=0;const recovery=Object.defineProperty({},'targetIntentsByActor',{enumerable:true,get(){reads++;throw Error('getter-executed')}}),f=panelFixture(t,{policy:{recovery}});
 const html=await f.app._renderHTML(await f.app._prepareContext());assert.match(html,/invalid-recovery-preferences/);assert.match(html,/data-pool="Actor.P"[^>]*value="41"/);
 f.target.value='41';await f.app.act('start',f.content);assert.equal(reads,0);assert.deepEqual(f.calls[0].recovery.targetIntentsByActor[P],{mode:'max',value:null});
});

test('an empty pool field is rejected before saving preferences or creating work',async t=>{
 const f=panelFixture(t);f.target.value='';await assert.rejects(f.app.act('start',f.content),/invalid-recovery/);assert.equal(f.calls.length,0);assert.equal(f.saved.length,0);
});

test('saving a selection merges other patient preferences changed during the snapshot await',async t=>{
 const entered=deferred(),release=deferred(),f=panelFixture(t,{snapshotGate:async()=>{entered.resolve();await release.promise}}),pending=f.app.act('start',f.content);
 await entered.promise;f.policy.recovery.targetIntentsByActor[OTHER].value=90;f.policy.recovery.targetIntentsByActor[P].value=80;release.resolve();await pending;
 assert.equal(f.saved[0].recovery.targetIntentsByActor[OTHER].value,90);assert.equal(f.calls[0].recovery.targetIntentsByActor[P].value,25);
});

function publicStart(refreshStorage,captured){
 const source=fs.readFileSync(new URL('../../scripts/exploration/runtime.mjs',import.meta.url),'utf8'),begin=source.indexOf('const start=async config=>'),end=source.indexOf('\n async function refreshStorage',begin),names=Object.keys(schema);
 assert.ok(begin>0&&end>begin);
 return new Function('isActiveGM','game','refreshStorage','coordinator','ledger',...names,'let lastSessionId=null;'+source.slice(begin,end)+';return start')(()=>true,{user:{setFlag:async()=>{}},system:{version:'8.5.1'}},refreshStorage,{start:async config=>{captured.push(config);return {id:'S'}}},{getSession:async()=>null},...names.map(name=>schema[name]));
}

test('public runtime captures detached recovery intentions before its storage await',async()=>{
 const release=deferred(),captured=[],start=publicStart(()=>release.promise,captured),config={actorUUIDs:[P],recovery:{...preferences(),targetIntentsByActor:{[P]:{mode:'percent',value:25}}}};
 const pending=start(config);config.recovery.targetIntentsByActor[P].value=90;release.resolve({state:'ready'});await pending;assert.equal(captured[0].recovery.targetIntentsByActor[P].value,25);
});

test('public runtime rejects recovery accessors before cloning or reading storage',async()=>{
 let reads=0,storageReads=0;const start=publicStart(async()=>{storageReads++;return {state:'ready'}},[]),config=Object.defineProperty({actorUUIDs:[P]},'recovery',{enumerable:true,get(){reads++;throw Error('getter-executed')}});
 await assert.rejects(start(config),/invalid-recovery-preferences/);assert.equal(reads,0);assert.equal(storageReads,0);
});

test('public runtime rejects a foreign patient intention before reading storage',async()=>{
 let storageReads=0;const start=publicStart(async()=>{storageReads++;return {state:'ready'}},[]);
 await assert.rejects(start({actorUUIDs:[P],recovery:preferences()}),/invalid-recovery-preferences/);assert.equal(storageReads,0);
});
