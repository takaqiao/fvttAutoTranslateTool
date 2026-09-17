import test from 'node:test';
import assert from 'node:assert/strict';
import {preserveDamagePartForMerge} from '../scripts/native-damage-components.mjs';
import * as api from '../scripts/native-damage-components.mjs';

const NS='pf2e-third-party-automation',types={fire:'Fire',slashing:'Slashing',bludgeoning:'Bludgeoning'};
const part=(instances,ignore=[])=>({instances:instances.map(i=>typeof i==='string'?{type:i,total:10,persistent:false}:i),options:{bypass:{immunity:{ignore:[],downgrade:[],redirect:[]},resistance:{ignore,redirect:[]}}}});
const ignore=(type='fire',max=Infinity)=>({type,max});
function merge(parts){assert.equal(typeof api.preserveMergedDamageBypass,'function');const result=part(parts.flatMap(p=>p.instances));assert.equal(api.preserveMergedDamageBypass(result,parts,types),result);return result;}
function alter(original,{multiplier=2,addend=0}={}){assert.equal(typeof api.preserveDamageBypassOnAlter,'function');const result={instances:original.instances,options:{}};assert.equal(api.preserveDamageBypassOnAlter(original,result,{multiplier,addend,damageTypes:types}),result);return result;}
const ignored=r=>r.options.bypass?.resistance.ignore??[];
test('merge protection only fills empty native component flavor and keeps the evaluated roll',()=>{
 const roll={instances:[{head:{options:{}}},{head:{options:{flavor:'precision'}}},{head:{options:{flavor:'splash'}}},{head:{options:{flavor:'persistent,fire'}}}]};
 assert.equal(preserveDamagePartForMerge(roll),roll);assert.deepEqual(roll.instances.map(i=>i.head.options.flavor),['damage','precision','splash','persistent,fire']);
});

test('unique fire contributor keeps its native ignore without granting it to slashing',()=>{
 const fire=part(['fire'],[ignore()]),blade=part(['slashing']);const result=merge([fire,blade]);
 assert.deepEqual(ignored(result),[{type:'fire',max:Infinity}]);assert.notEqual(ignored(result)[0],ignored(fire)[0]);
});
test('all contributors of a type must carry the exact same ignore entry',()=>{
 assert.deepEqual(ignored(merge([part(['fire'],[ignore()]),part(['fire'],[ignore()])])),[ignore()]);
 assert.deepEqual(ignored(merge([part(['fire'],[ignore()]),part(['fire'])])),[]);
 assert.deepEqual(ignored(merge([part(['fire'],[ignore('fire',5)]),part(['fire'],[ignore('fire',10)])])),[]);
});
test('unrolled persistent fire counts even though its current total is zero',()=>{
 const pending=part([{type:'fire',persistent:true,total:0}]);
 assert.deepEqual(ignored(merge([part(['fire'],[ignore()]),pending])),[]);
 pending.options.bypass.resistance.ignore=[ignore()];assert.deepEqual(ignored(merge([part(['fire'],[ignore()]),pending])),[ignore()]);
});
test('broad physical, custom and all-damage ignore entries never become literal type proof',()=>{
 assert.deepEqual(ignored(merge([part(['slashing'],[ignore('physical'),ignore('custom'),ignore('all-damage')]),part(['bludgeoning'])])),[]);
});
test('an entry for a type with no contributing instance is not copied',()=>{
 assert.deepEqual(ignored(merge([part(['slashing'],[ignore()])])),[]);
});
test('a missing roll or instance list prevents incomplete contributor proof',()=>{
 const result=part(['fire']);api.preserveMergedDamageBypass(result,[part(['fire'],[ignore()]),{}],types);assert.deepEqual(ignored(result),[]);
});
test('same item identity is not used to collapse distinct roll contributions',()=>{
 const a=part(['fire'],[ignore()]),b=part(['fire']);a.itemUuid=b.itemUuid='Actor.a.Item.fist';assert.deepEqual(ignored(merge([a,b])),[]);
});
test('safe proof roundtrips through JSON without losing the original unbounded max',()=>{
 const original=merge([part(['fire'],[ignore()]),part(['slashing'])]);
 const loaded=JSON.parse(JSON.stringify(original));assert.equal(ignored(loaded)[0].max,null,'Native JSON uses null, proof must retain Infinity independently');
 const doubled=alter(loaded);assert.deepEqual(ignored(doubled),[ignore()]);assert.deepEqual(ignored(alter(doubled,{multiplier:0.5})),[ignore()]);
});
test('alter never copies arbitrary unproven native bypass or unrelated options',()=>{
 const raw=part(['fire'],[ignore()]);raw.options.degreeOfSuccess=3;assert.deepEqual(ignored(alter(raw)),[]);
 const safe=merge([raw]);safe.options.other='x';const changed=alter(safe);assert.equal(changed.options.other,undefined);assert.equal(changed.options.degreeOfSuccess,undefined);
});
test('manual addend invalidates the first type only and cannot silently keep fire proof',()=>{
 const source=merge([part(['fire','slashing'],[ignore(),ignore('slashing',5)])]);
 const changed=alter(source,{addend:1});assert.deepEqual(ignored(changed),[ignore('slashing',5)]);
 assert.deepEqual(ignored(alter(source,{multiplier:-1})),[]);
});
test('finite max and Infinity never become equal by JSON serialization during intersection',()=>{
 assert.deepEqual(ignored(merge([part(['fire'],[ignore()]),part(['fire'],[ignore('fire',null)])])),[]);
 assert.deepEqual(ignored(merge([part(['fire'],[ignore('fire',0)]),part(['fire'],[ignore('fire',0)])])),[ignore('fire',0)]);
});
test('malformed module proof cannot restore an unknown version or categorical bypass',()=>{
 const safe=merge([part(['fire'],[ignore()])]),loaded=JSON.parse(JSON.stringify(safe));
 loaded.options[NS].safeDamageBypass.version=999;assert.deepEqual(ignored(alter(loaded)),[]);
});
test('null proof entries cannot throw during an otherwise native damage alter',()=>{
 const safe=merge([part(['fire'],[ignore()])]);safe.options[NS].safeDamageBypass.ignore=[null];assert.deepEqual(ignored(alter(safe)),[]);
});
