import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {harness,optionalPatch} from './bbmm-harness.mjs';
const {bbmmCompatibility=()=> 'missing-reader',readBbmmHardRules=()=>({})}=await optionalPatch('bbmm-rules');
const fixture=JSON.parse(fs.readFileSync(new URL('./fixtures/bbmm-rules-native.json',import.meta.url)));

test('current native BBMM writer produces rules accepted by the independent reader',async()=>{
 const e=harness();
 const context=vm.createContext({game:e.game,BBMM_ID:'bbmm',BBMM_SYNC_CH:'module.bbmm',setTimeout:()=>{}});
 vm.runInContext(fixture.writeLocks+';globalThis.writeLocks=_lc_writeLockChanges;',context);
 e.registry.set('example.selected',{scope:'client',requiresReload:true});
 await context.writeLocks({toAdd:[{namespace:'example',key:'selected',lockType:'locked',value:true}]});
 assert.equal(bbmmCompatibility(e.runtime),null);
 assert.equal(readBbmmHardRules(e.runtime)['example.selected'].value,true);
 assert.equal(e.callbacks.has('clientSettingChanged'),false);assert.equal(e.warnings.length,0);
});

for(const kind of ['missing','wrong-type','wrong-scope','unavailable'])test(`BBMM ${kind} registry cannot enforce stale saved rules`,()=>{
 const e=harness();
 if(kind==='missing')e.registry.delete('bbmm.userSettingSync');
 if(kind==='wrong-type')e.registry.get('bbmm.userSettingSync').type=Array;
 if(kind==='wrong-scope')e.registry.get('bbmm.enableUserSettingSync').scope='client';
 if(kind==='unavailable')e.game.settings.get=()=>{throw Error('setting not registered');};
 assert.deepEqual(readBbmmHardRules(e.runtime),{});
});

test('only available client/user hard rules with matching identifiers and a value are read',()=>{
 const e=harness(),id='example.selected';
 for(const row of [null,[],{}, {namespace:'wrong',key:'selected',value:false},
   {namespace:'example',key:'other',value:false}, {namespace:'example',key:'selected',value:false,soft:'false'},
   {namespace:'example',key:'selected',value:false,soft:true}]){
  e.rules[id]=row;assert.equal(readBbmmHardRules(e.runtime)[id],undefined);
 }
 e.rules[id]={namespace:'example',key:'selected',value:false};
 assert.equal(readBbmmHardRules(e.runtime)[id].value,false);
 e.registry.delete(id);assert.equal(readBbmmHardRules(e.runtime)[id],undefined);
});

test('current account, GM exemption, disabled sync and malformed stores preserve saved data',()=>{
 for(const options of [{gm:true},{sync:false},{bbmmActive:false}]){
  const e=harness(options),before=structuredClone(e.rules);assert.deepEqual(readBbmmHardRules(e.runtime),{});assert.deepEqual(e.rules,before);
 }
 const e=harness();assert.deepEqual(readBbmmHardRules(e.runtime,new e.User('other',false)),{});
 for(const value of [null,[],false,'rules']){e.values.set('bbmm.userSettingSync',value);assert.deepEqual(readBbmmHardRules(e.runtime),{});}
});

test('matching BBMM rules remain readable on later core generations',()=>{
 const e=harness({generation:15});
 assert.equal(readBbmmHardRules(e.runtime)['example.selected']?.value,false);
});

test('future BBMM labels retain rules and registry checks',()=>{
 const e=harness({bbmmVersion:'2.0.0'});
 assert.equal(readBbmmHardRules(e.runtime)['example.selected']?.value,false);
 e.registry.get('bbmm.userSettingSync').type=Array;
 assert.deepEqual(readBbmmHardRules(e.runtime),{});
});
