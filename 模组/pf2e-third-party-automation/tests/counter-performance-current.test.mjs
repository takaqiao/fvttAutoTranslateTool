import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {counterPerformanceOutcome} from '../scripts/counter-performance.mjs';

const outcomes=['criticalFailure','failure','success','criticalSuccess'];
const read=name=>{assert.ok(process.env[name],`${name} must identify the captured current dependency`);return readFileSync(process.env[name],'utf8');};

test('replacement outcomes match the captured PF2e 8.6.1 degree evaluator, including natural dice and save adjustments',()=>{
 const source=read('BOB2_CURRENT_PF2E_SOURCE'),start=source.indexOf('class DegreeOfSuccess {'),end=source.indexOf('\n}, ',start);
 assert.ok(start>=0&&end>start,'current native DegreeOfSuccess class');
 const math=Object.create(Math);math.clamp=(value,min,max)=>Math.min(max,Math.max(min,value));
 const NativeDegree=new Function('Roll','foundry','Math','Gt','Ut','sluggify','_loc',`return (${source.slice(start,end+2)});`)(class {},{},math,outcomes,{INCREASE:1,LOWER:-1},value=>value,value=>value);
 const adjustments=[{}, {all:{label:'incapacitation',amount:1}}, {success:{label:'resolve',amount:'criticalSuccess'}}, {failure:{label:'failure improvement',amount:1}}, {all:{label:'penalty',amount:-1}}, {criticalFailure:{label:'improvement',amount:2}}];
 for(const total of [5,12,21,22,25,32,36])for(const die of [1,10,20])for(const adjustment of adjustments){
  const native=new NativeDegree({dieValue:die,modifier:total-die},22,adjustment);
  assert.deepEqual(counterPerformanceOutcome(total,die,22,adjustment),{success:outcomes[native.value],unadjustedOutcome:outcomes[native.unadjusted]},JSON.stringify({total,die,adjustment}));
 }
});

test('captured Toolbelt 3.59.2 reads effective totals and outcomes while retaining original roll and applied marks',()=>{
 const source=read('BOB2_CURRENT_TOOLBELT_SOURCE'),start=source.indexOf('var lt=class{'),end=source.indexOf(';var Lv;',start);
 assert.ok(start>=0&&end>start,'current Toolbelt TargetHelper class');
 const TargetHelper=new Function('game','a',`return (${source.slice(start+7,end)});`)({user:{isGM:true}},(object,name)=>Object.defineProperty(object,'name',{value:name,configurable:true}));
 const original=JSON.stringify({class:'CheckRoll',total:17,options:{type:'saving-throw'},terms:[{class:'Die',faces:20,total:10}]}),data={type:'damage',targets:[],splashTargets:[],applied:{target:{0:true}},saveVariants:{null:{dc:22,basic:true,statistic:'will',saves:{target:{value:25,success:'success',roll:original,die:10}}}}};
 const helper=new TargetHelper(data);
 assert.equal(helper.targetSave('target').value,25);assert.equal(helper.targetSave('target').success,'success');
 assert.equal(helper.targetSave('target').roll,original);assert.equal(helper.targetSave('target').die,10);
 assert.deepEqual(helper.targetApplied('target'),{0:true});assert.equal(helper.saveVariant.dc,22);assert.equal(helper.isBasic,true);
});
