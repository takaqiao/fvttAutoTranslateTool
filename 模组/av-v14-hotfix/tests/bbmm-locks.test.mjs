import test from 'node:test';
import assert from 'node:assert/strict';
import {harness,optionalPatch,plain} from './bbmm-harness.mjs';
const patch=await optionalPatch('bbmm-locks');
function setup(opts){const e=harness(opts);e.install=()=>patch.installBbmmHardLocks?.({runtime:e.runtime});e.result=e.install();return e;}
test('native v14 client setter restores a changed hard setting',async()=>{const e=setup();await e.set(true);await e.flush();assert.equal(e.get(),false);assert.equal(e.writes,2);});
for(const existing of [false,true])test('native user-setting '+(existing?'update':'create')+' is repaired',async()=>{const e=setup({scope:'user'});if(existing)e.values.set('example.selected',false);await e.set(true);await e.flush();assert.equal(e.get(),false);});
for(const[name,opts]of Object.entries({GM:{gm:true},soft:{soft:true},disabled:{sync:false},world:{scope:'world'},inactive:{bbmmActive:false},unknownBBMM:{bbmmVersion:'1.5.0'},unknownCore:{generation:15}}))test(name+' retains its own value',async()=>{const e=setup(opts);await e.set(true);await e.flush();assert.equal(e.get(),true);assert.equal(e.writes,1);});
for(const change of ['unlock','disable','GM','newTarget'])test('queued repair rereads '+change,async()=>{const e=setup();await e.set(true);if(change==='unlock')delete e.rules['example.selected'];if(change==='disable')e.values.set('bbmm.enableUserSettingSync',false);if(change==='GM')e.game.user.isGM=true;if(change==='newTarget')e.rules['example.selected'].value=true;await e.flush();assert.equal(e.get(),true);assert.equal(e.writes,1);});
test('object locks keep their data and repeated events coalesce',async()=>{const target={illumination:40,other:{yes:1}};const e=setup({target});await e.set({illumination:100});for(let i=0;i<20;i++)e.runtime.Hooks.callAll('clientSettingChanged','example.selected');await e.flush();assert.deepEqual(plain(e.get()),target);assert.deepEqual(e.rules['example.selected'].value,target);assert.equal(e.writes,2);});
test('other users and unregistered keys do not trigger local writes',async()=>{const e=setup({scope:'user'});e.values.set('example.selected',true);e.runtime.Hooks.callAll('updateSetting',{key:'example.selected',user:'other'});e.runtime.Hooks.callAll('clientSettingChanged','unknown.value');await e.flush();assert.equal(e.writes,0);});
test('repeat install is idempotent and restore cancels queued repair',async()=>{const e=setup();assert.equal(e.install()?.status,'already-installed');await e.set(true);e.result.restore();await e.flush();assert.equal(e.get(),true);assert.equal(e.writes,1);});
test('a rejected write reports once without retrying forever; later events can retry',async()=>{const e=setup();await e.set(true);const set=e.game.settings.set;e.game.settings.set=async()=>{throw Error('offline');};await e.flush();assert.equal(e.errors.length,1);e.game.settings.set=set;e.runtime.Hooks.callAll('clientSettingChanged','example.selected');await e.flush();assert.equal(e.get(),false);});

test('a second edit during a restoring write is repaired', async () => {
  const e = setup();
  let edited = false;
  e.runtime.Hooks.on('clientSettingChanged', (id, value) => {
    if (id === 'example.selected' && value === false && !edited) {
      edited = true;
      void e.set(true);
    }
  });
  await e.set(true);
  await e.flush();
  assert.equal(e.get(), false);
  assert.equal(e.writes, 4);
});
