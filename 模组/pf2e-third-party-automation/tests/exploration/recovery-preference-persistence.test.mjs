import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {join} from 'node:path';
import {pathToFileURL} from 'node:url';
import {createRecoveryPanel} from '../../scripts/exploration/panel.mjs';
import * as recovery from '../../scripts/exploration/recovery-goals.mjs';
import {captureRecoveryPreferences} from '../../scripts/exploration/schema.mjs';

const MODULE='pf2e-third-party-automation',P='Actor.6s11bLaPM8yOMuac',OTHER='Actor.H9UrldsdnP24n6MQ',TOKEN='Scene.S.Token.T.Actor.A';
const preferences=()=>({version:1,targetIntentsByActor:{[P]:{mode:'absolute',value:80},[OTHER]:{mode:'percent',value:30},[TOKEN]:{mode:'max',value:null}},requireNoWounded:true,failureStop:{enabled:true,limit:2}});
let nativeField;
async function objectField(){
 nativeField??=(async()=>{
  const root=process.env.FOUNDRY_NATIVE_APP_ROOT??'C:/Program Files/Foundry Virtual Tabletop/resources/app';
  for(const [file,sha] of [['common/data/fields.mjs','b8a7eafb6063e201cc34137da26b081385ac731a4e52c4a7bf80b7c5ed6fb6c2'],['common/utils/helpers.mjs','25e75a70ab539c6878108ab4369416bde6edb7aef8a629f91675cd6a2d11b006'],['common/data/operators.mjs','0ee26ded57750615b6cb4033c8aa90c444224b41df940caba508dd036d5f3121']]){
   assert.equal(createHash('sha256').update(await readFile(join(root,file))).digest('hex'),sha,`Foundry 14.368 source changed: ${file}`);
  }
  return (await import(pathToFileURL(join(root,'common/data/fields.mjs')).href)).ObjectField;
 })();
 return new (await nativeField)();
}
async function panelFixture(t,stored,{snapshotGate}={}){
 const field=await objectField(),previous=globalThis.foundry;t.after(()=>globalThis.foundry=previous);
 globalThis.foundry={applications:{api:{ApplicationV2:class{render(){return this}}}}};
 const source={[MODULE]:{siblingFlag:{keep:'module'},explorationPolicy:{customPolicy:{keep:'policy'},requireFullFocus:false,recovery:stored}},otherModule:{keep:'namespace'}};
 const state={source:structuredClone(source),diff:{},model:null,failure:{fields:{}}},saved=[],calls=[];
 const game={user:{isGM:true,getFlag:(_module,key)=>state.source[MODULE][key],setFlag:async(_module,key,value)=>{
  saved.push(structuredClone(value));const changes=field.clean({[key]:structuredClone(value)},{partial:true});field._updateDiff(MODULE,changes,{},state);
 }},messages:new Map()};
 const actors=[{actorUUID:P,name:'Patient P',hp:{value:1,max:60},focus:{value:0,max:0},pool:{poolUUID:P,ready:true},wardCapacity:1}];
 const app=createRecoveryPanel({game,coordinator:{},getSessionId:()=>null,capabilities:{snapshot:async()=>{await snapshotGate?.();return actors}},start:async config=>calls.push(config)}).open([P]);
 const mode={dataset:{targetMode:'',actor:P},value:'absolute'},target={value:'80'},fields={budget:{value:'120'},activityBudget:{value:'100'},risky:{checked:false},focus:{checked:true},extension:{checked:false},assurance:{checked:false},rank:{value:'trained'},manualFirstRound:{checked:false},activityFirstRound:{checked:false},wounded:{checked:true},failureStop:{checked:true},failureLimit:{value:'2'}};
 const content={querySelector:selector=>selector.includes('data-target-value')?target:fields[selector.match(/\[name=(\w+)\]/)?.[1]],querySelectorAll:selector=>selector==='[data-target-mode]'?[mode]:[]};
 return {field,state,source,saved,calls,app,mode,target,content};
}

test('pinned Foundry flag cleaning reproduces dotted UUID expansion in the old recovery object',async()=>{
 const field=await objectField(),cleaned=field.clean({explorationPolicy:{recovery:preferences()}},{partial:true});
 assert.deepEqual(cleaned.explorationPolicy.recovery.targetIntentsByActor.Actor,{'6s11bLaPM8yOMuac':{mode:'absolute',value:80},H9UrldsdnP24n6MQ:{mode:'percent',value:30}});
 assert.deepEqual(cleaned.explorationPolicy.recovery.targetIntentsByActor.Scene.S.Token.T.Actor.A,{mode:'max',value:null});
 assert.throws(()=>recovery.normalizeRecoveryPreferences(cleaned.explorationPolicy.recovery),/invalid-recovery-preferences/);
});

test('panel preferences survive native cleaning and readback while preserving other targets and flags',async t=>{
 const input=preferences(),f=await panelFixture(t,JSON.stringify(input));
 const before=await f.app._renderHTML(await f.app._prepareContext());assert.match(before,/data-target-value[^>]*value="80"/);assert.doesNotMatch(before,/invalid-recovery-preferences/);
 await f.app.act('start',f.content);
 assert.equal(typeof f.saved[0].recovery,'string');assert.deepEqual(JSON.parse(f.state.source[MODULE].explorationPolicy.recovery),input);
 assert.deepEqual(f.calls[0].recovery,{...input,targetIntentsByActor:{[P]:input.targetIntentsByActor[P]}});assert.deepEqual(f.calls[0].goalsByPool,[{poolUUID:P,targetHP:60}]);
 assert.deepEqual(f.state.source[MODULE].siblingFlag,f.source[MODULE].siblingFlag);assert.deepEqual(f.state.source.otherModule,f.source.otherModule);assert.deepEqual(f.state.source[MODULE].explorationPolicy.customPolicy,f.source[MODULE].explorationPolicy.customPolicy);assert.equal(f.state.source[MODULE].explorationPolicy.requireFullFocus,true);
 const after=await f.app._renderHTML(await f.app._prepareContext());assert.match(after,/data-target-value[^>]*value="80"/);assert.doesNotMatch(after,/invalid-recovery-preferences/);
});

test('a new saved string replaces the corrupt old object without guessing lost UUID intentions',async t=>{
 const field=await objectField(),corrupt=field.clean({recovery:preferences()},{partial:true}).recovery,f=await panelFixture(t,corrupt);
 const before=await f.app._renderHTML(await f.app._prepareContext());assert.match(before,/invalid-recovery-preferences/);assert.match(before,/data-pool="Actor.6s11bLaPM8yOMuac"[^>]*value="60"/);
 assert.equal(f.saved.length,0);assert.equal(f.calls.length,0);
 f.mode.value='percent';f.target.value='30';await f.app.act('start',f.content);
 assert.equal(typeof f.state.source[MODULE].explorationPolicy.recovery,'string');
 assert.deepEqual(JSON.parse(f.state.source[MODULE].explorationPolicy.recovery),{version:1,targetIntentsByActor:{[P]:{mode:'percent',value:30}},requireNoWounded:true,failureStop:{enabled:true,limit:2}});
 assert.deepEqual(f.state.source[MODULE].siblingFlag,f.source[MODULE].siblingFlag);assert.deepEqual(f.state.source[MODULE].explorationPolicy.customPolicy,f.source[MODULE].explorationPolicy.customPolicy);assert.deepEqual(f.state.source.otherModule,f.source.otherModule);
 assert.doesNotMatch(await f.app._renderHTML(await f.app._prepareContext()),/invalid-recovery-preferences/);
});

test('storage encoding normalizes before serialization and decoding returns detached preferences',()=>{
 assert.equal(typeof recovery.encodeRecoveryPreferences,'function');assert.equal(typeof recovery.decodeRecoveryPreferences,'function');
 const input={targetIntentsByActor:{[P]:{mode:'absolute',value:80}}},encoded=recovery.encodeRecoveryPreferences(input),decoded=recovery.decodeRecoveryPreferences(encoded);
 assert.equal(typeof encoded,'string');assert.deepEqual(decoded,recovery.normalizeRecoveryPreferences(input));decoded.targetIntentsByActor[P].value=10;assert.equal(input.targetIntentsByActor[P].value,80);
 assert.deepEqual(recovery.decodeRecoveryPreferences(input),recovery.normalizeRecoveryPreferences(input));assert.deepEqual(recovery.decodeRecoveryPreferences(),recovery.normalizeRecoveryPreferences());
 assert.throws(()=>captureRecoveryPreferences({actorUUIDs:[P],recovery:encoded}),/invalid-recovery-preferences/);
});

test('stored strings reject malformed JSON, primitive values, unknown fields and expanded UUID maps',()=>{
 assert.equal(typeof recovery.decodeRecoveryPreferences,'function');
 for(const input of ['', '{', 'null', '[]', 'false', '0', '"text"', '{"version":2}', '{"extra":true}', '{"targetIntentsByActor":{"Actor":{"P":{"mode":"max","value":null}}}}'])assert.throws(()=>recovery.decodeRecoveryPreferences(input),/invalid-recovery-preferences/);
 assert.deepEqual(recovery.decodeRecoveryPreferences(' '.repeat(65534)+'{}'),recovery.normalizeRecoveryPreferences());
 assert.throws(()=>recovery.decodeRecoveryPreferences(' '.repeat(65536)+'{}'),/invalid-recovery-preferences/);
});

test('storage encoding never executes accessors or toJSON and rejects values JSON would silently alter',()=>{
 assert.equal(typeof recovery.encodeRecoveryPreferences,'function');let reads=0;
 const input=Object.defineProperty({},'targetIntentsByActor',{enumerable:true,get(){reads++;return {}}});
 for(const value of [input,{toJSON(){reads++;return {}}},{targetIntentsByActor:{[P]:{mode:'percent',value:NaN}}},{targetIntentsByActor:{[P]:{mode:'percent',value:Infinity}}}])assert.throws(()=>recovery.encodeRecoveryPreferences(value),/invalid-recovery-preferences/);
 assert.equal(reads,0);
 const large={targetIntentsByActor:Object.fromEntries(Array.from({length:1500},(_,i)=>[`Actor.Patient${i}`,{mode:'absolute',value:80}]))};assert.throws(()=>recovery.encodeRecoveryPreferences(large),/invalid-recovery-preferences/);
});

test('the panel rejects a saved recovery accessor without executing it',async t=>{
 const f=await panelFixture(t,JSON.stringify(preferences()));let reads=0;
 Object.defineProperty(f.state.source[MODULE].explorationPolicy,'recovery',{enumerable:true,get(){reads++;throw Error('getter-executed')}});
 const html=await f.app._renderHTML(await f.app._prepareContext());assert.match(html,/invalid-recovery-preferences/);assert.equal(reads,0);
});

test('malformed strings and inherited recovery keep a visible fallback without saving on read',async t=>{
 const f=await panelFixture(t,'null');
 for(const stored of ['null','{',' '.repeat(65536)+'{}']){
  f.state.source[MODULE].explorationPolicy.recovery=stored;const html=await f.app._renderHTML(await f.app._prepareContext());
  assert.match(html,/invalid-recovery-preferences/);assert.match(html,/data-pool="Actor.6s11bLaPM8yOMuac"[^>]*value="60"/);assert.equal(f.saved.length,0);assert.equal(f.calls.length,0);
 }
 delete f.state.source[MODULE].explorationPolicy.recovery;let reads=0;
 Object.setPrototypeOf(f.state.source[MODULE].explorationPolicy,Object.defineProperty({},'recovery',{get(){reads++;return JSON.stringify(preferences())}}));
 assert.match(await f.app._renderHTML(await f.app._prepareContext()),/invalid-recovery-preferences/);assert.equal(reads,0);assert.equal(f.saved.length,0);
});

test('native storage merges the latest encoded unselected target after awaiting the start snapshot',async t=>{
 let entered,release;const ready=new Promise(r=>entered=r),gate=new Promise(r=>release=r),f=await panelFixture(t,JSON.stringify(preferences()),{snapshotGate:async()=>{entered();await gate}});
 const pending=f.app.act('start',f.content);await ready;
 const latest=preferences();latest.targetIntentsByActor[OTHER].value=90;latest.targetIntentsByActor[P].value=85;f.state.source[MODULE].explorationPolicy.recovery=JSON.stringify(latest);release();await pending;
 const stored=JSON.parse(f.state.source[MODULE].explorationPolicy.recovery);assert.equal(stored.targetIntentsByActor[OTHER].value,90);assert.equal(stored.targetIntentsByActor[P].value,80);assert.equal(f.calls[0].recovery.targetIntentsByActor[P].value,80);
});
