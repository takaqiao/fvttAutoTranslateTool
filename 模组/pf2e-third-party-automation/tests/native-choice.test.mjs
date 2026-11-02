import test from 'node:test';
import assert from 'node:assert/strict';
import * as native from '../scripts/native-context.mjs';

test('Strike choices describe the prepared usage without changing the weapon name',()=>{
 assert.equal(typeof native.nativeStrikeChoiceLabel,'function');
 const weapon={name:'组合武器 & Test',isMelee:false,isRanged:true};
 assert.equal(native.nativeStrikeChoiceLabel({item:weapon}),'远程 · 组合武器 & Test');
 assert.equal(native.nativeStrikeChoiceLabel({item:{...weapon,isMelee:true,isRanged:false,altUsageType:'melee'}}),'近战 · 组合武器 & Test');
 assert.equal(native.nativeStrikeChoiceLabel({item:{...weapon,isThrown:true,altUsageType:'thrown'}}),'远程（投掷） · 组合武器 & Test');
 assert.equal(native.nativeStrikeChoiceLabel({item:{name:'远程名称不决定模式',isMelee:true}}),'近战 · 远程名称不决定模式');
 assert.equal(native.nativeStrikeChoiceLabel({item:{name:'未知模式',altUsageType:'melee'}}),'未知模式');
});

test('native choice labels stay literal text and callbacks preserve the selected value',async()=>{
 const previous=globalThis.foundry;let config;
 globalThis.foundry={applications:{api:{DialogV2:{wait:async options=>{config=options;return options.buttons[1].callback();}}}}};
 try{
  const choices=[{value:'weapon:',label:'远程 · Test & <Weapon>'},{value:'weapon:melee',label:'近战 · Test & <Weapon>'}];
  assert.equal(await native.showNativeChoice({title:'选择攻击模式',choices}),'weapon:melee');
  assert.deepEqual(config.buttons.map(button=>button.label),choices.map(choice=>choice.label));
  assert.deepEqual(config.buttons.map(button=>button.callback()),['weapon:','weapon:melee']);
  assert.equal(config.rejectClose,false);
 }finally{globalThis.foundry=previous;}
});

test('closing a native choice keeps cancellation and a single option needs no window',async()=>{
 const previous=globalThis.foundry;let opened=0;
 globalThis.foundry={applications:{api:{DialogV2:{wait:async()=>{opened++;return null;}}}}};
 try{
  assert.equal(await native.showNativeChoice({choices:[{value:'one',label:'近战 · One'}]}),'one');
  assert.equal(opened,0);
  assert.equal(await native.showNativeChoice({choices:[{value:'one',label:'近战 · One'},{value:'two',label:'远程 · Two'}]}),null);
  assert.equal(opened,1);
 }finally{globalThis.foundry=previous;}
});
