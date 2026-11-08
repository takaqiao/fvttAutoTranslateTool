import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {counterPerformanceOutcome} from '../scripts/counter-performance.mjs';

const outcomes=['criticalFailure','failure','success','criticalSuccess'];
const read=name=>{assert.ok(process.env[name],`${name} must identify the captured current dependency`);return readFileSync(process.env[name],'utf8');};
const optional=name=>process.env[name]?readFileSync(process.env[name],'utf8'):null;
const single=(text,pattern,label)=>{const found=[...text.matchAll(pattern)];assert.equal(found.length,1,label);return found[0];};
// The captured current pair, and PF2e 8.7.0 and Toolbelt 3.61.3 when they are supplied.
const pf2eBuilds=[['captured',read('BOB2_CURRENT_PF2E_SOURCE')],['8.7.0',optional('PF2E_NATIVE_870_BUNDLE')]];
const toolbeltBuilds=[['captured',read('BOB2_CURRENT_TOOLBELT_SOURCE')],['3.61.3',optional('TOOLBELT_3613_SOURCE')]];

// The class reads two bundle constants under minified names: the outcome list
// it indexes with its value, and the adjustment amounts it compares with.
function nativeDegree(source){
 const start=source.indexOf('class DegreeOfSuccess {'),end=source.indexOf('\n}, ',start);
 assert.ok(start>=0&&end>start,'native DegreeOfSuccess class');
 const text=source.slice(start,end+2);
 const list=single(text,/this\.key = ([\w$]+)\[this\.value\]/g,'outcome list')[1],amounts=single(text,/i !== ([\w$]+)\.INCREASE\)/g,'adjustment amounts')[1];
 const math=Object.create(Math);math.clamp=(value,min,max)=>Math.min(max,Math.max(min,value));
 return new Function('Roll','foundry','Math',list,amounts,'sluggify','_loc',`return (${text});`)(class {},{},math,outcomes,{INCREASE:1,LOWER:-1},value=>value,value=>value);
}
for(const [label,source]of pf2eBuilds)test(`replacement outcomes match the ${label} PF2e degree evaluator, including natural dice and save adjustments`,{skip:!source},()=>{
 const NativeDegree=nativeDegree(source);
 const adjustments=[{}, {all:{label:'incapacitation',amount:1}}, {success:{label:'resolve',amount:'criticalSuccess'}}, {failure:{label:'failure improvement',amount:1}}, {all:{label:'penalty',amount:-1}}, {criticalFailure:{label:'improvement',amount:2}}];
 for(const total of [5,12,21,22,25,32,36])for(const die of [1,10,20])for(const adjustment of adjustments){
  const native=new NativeDegree({dieValue:die,modifier:total-die},22,adjustment);
  assert.deepEqual(counterPerformanceOutcome(total,die,22,adjustment),{success:outcomes[native.value],unadjustedOutcome:outcomes[native.unadjusted]},JSON.stringify({total,die,adjustment}));
 }
});

// esbuild opens the class with its name registration; the next top-level var ends it.
function targetHelper(source){
 const registration=single(source,/var [\w$]+=class\{static\{([\w$]+)\(this,"TargetHelper"\)\}/g,'TargetHelper class');
 const start=registration.index+registration[0].indexOf('class{'),end=source.indexOf(';var ',start);
 assert.ok(end>start,'end of the TargetHelper class');
 return new Function('game',registration[1],`return (${source.slice(start,end)});`)({user:{isGM:true}},(object,name)=>Object.defineProperty(object,'name',{value:name,configurable:true}));
}
for(const [label,source]of toolbeltBuilds)test(`${label} Toolbelt reads effective totals and outcomes while retaining original roll and applied marks`,{skip:!source},()=>{
 const TargetHelper=targetHelper(source);
 const original=JSON.stringify({class:'CheckRoll',total:17,options:{type:'saving-throw'},terms:[{class:'Die',faces:20,total:10}]}),data={type:'damage',targets:[],splashTargets:[],applied:{target:{0:true}},saveVariants:{null:{dc:22,basic:true,statistic:'will',saves:{target:{value:25,success:'success',roll:original,die:10}}}}};
 const helper=new TargetHelper(data);
 assert.equal(helper.targetSave('target').value,25);assert.equal(helper.targetSave('target').success,'success');
 assert.equal(helper.targetSave('target').roll,original);assert.equal(helper.targetSave('target').die,10);
 assert.deepEqual(helper.targetApplied('target'),{0:true});assert.equal(helper.saveVariant.dc,22);assert.equal(helper.isBasic,true);
});
