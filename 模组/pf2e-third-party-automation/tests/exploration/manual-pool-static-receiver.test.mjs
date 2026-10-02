import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {receiverFixture} from '../../tools/native-manual-pool-static-receiver/seam.test.mjs';

const sourcePrefix='pf2e-third-party-automation:source:MESSAGE123456789';
const sourceMarker=sourcePrefix+':0';
const receiptMarker='pf2e-third-party-automation:manual-pool-native:abcdefab-abcd-4abc-8abc-abcdefabcdef';
const receiptPattern='pf2e-third-party-automation:manual-pool-native:[^:]*([0-9])[^:]*|never';
const godless=[{or:['action:battle-medicine','action:treat-wounds']}];
for(const [name,predicate]of [
 ['explicit null is not an empty predicate',null],
 ['exact source negation',[{not:sourceMarker}]],
 ['source numeric eq',[{eq:[sourcePrefix,0]}]],
 ['unknown private receipt regex',[{not:{gt:[receiptPattern,-1]}}]],
 ['regex without module name',[{gt:['.*([0-9]).*|never',-1]}]],
 ['plain action string',['action:treat-wounds']],
 ['parent options',['parent:feat:godless-healing']],
 ['partial Godless OR',[{or:['action:treat-wounds']}]],
 ['other boolean composition',[{and:['action:treat-wounds']}]],
 ['extra Godless branch',[{or:['action:battle-medicine','action:treat-wounds','other']}]]
])test('only the exact supported receiver predicate qualifies: '+name,async()=>{
 const f=receiverFixture({rules:[{value:5,predicate}],options:['action:treat-wounds']});f.forbid();f.subscribe(f.authorize);
 await assert.rejects(f.run(),/static-receiver-unavailable/);assert.equal(f.constructCalls(),0);assert.equal(f.nativeCalls.length,0);
});
for(const predicate of [[],godless])test('qualified predicate is unchanged by actual leaf marker additions '+JSON.stringify(predicate),async()=>{
 const f=receiverFixture({rules:[{value:5,predicate}],options:['action:treat-wounds']});f.forbid();f.subscribe(f.authorize);await f.run();
 const params=f.nativeCalls[0].params,model=f.pureModel(f.prepared[0],params);assert.equal(model.flatTotal,5);assert.equal(f.constructCalls(),0);
 const actual=await f.parity(f.prepared[0],{...params,rollOptions:new Set([...params.rollOptions,sourceMarker,receiptMarker])});
 assert.equal(actual.flatTotal,5);assert.equal(actual.amount,model.amount);assert.equal(model.isCurrent(),true);
});
test('unsupported predicate remains usable on the ordinary unregistered native path',async()=>{
 const f=receiverFixture({rules:[{value:5,predicate:[{not:{gt:[receiptPattern,-1]}}]}]});f.subscribe(()=>({status:'unregistered'}));await f.run();
 assert.equal(f.nativeCalls.length,1);assert.equal(f.frames[0].callFrame,null);
 const original=await f.parity(f.prepared[0],f.nativeCalls[0].params);assert.equal(original.flatTotal,5);
 const tagged=await f.parity(f.prepared[0],{...f.nativeCalls[0].params,rollOptions:new Set([receiptMarker])});assert.equal(tagged.flatTotal,0);
});

const cases=[
 ['untyped constants',[{value:5},{value:10}],[],25],
 ['typed bonus and penalty',[{type:'status',value:5},{type:'status',value:10},{type:'status',value:-2},{type:'status',value:-4}],[],16],
 ['duplicate slug is not deduplicated',[{slug:'same',value:5},{slug:'same',value:10}],[],25],
 ['Godless Healing predicate',[{value:5,predicate:[{or:['action:battle-medicine','action:treat-wounds']}]}],['action:treat-wounds'],15],
 ['Godless predicate false',[{value:5,predicate:[{or:['action:battle-medicine','action:treat-wounds']}]}],[],10],
 ['native critical normalization',[{value:5,critical:true}],[],15],
 ['native clamp',[{value:12,min:2,max:5}],[],15],
 ['negative clamp',[{value:-50}],[],-0]
];
for(const [name,rules,options,expected]of cases)test('model matches fixed native receiving path: '+name,async()=>{
 const f=receiverFixture({rules:[rules],options});f.forbid();f.subscribe(f.authorize);await f.run();const actor=f.prepared[0],params=f.nativeCalls[0].params;
 const before=JSON.stringify(f.allRules.map(rule=>({source:rule.item._source,value:rule.value,predicate:[...rule.predicate],ignored:rule.ignored})));
 const prediction=f.pureModel(actor,params);assert.equal(prediction.amount,expected);assert.equal(f.constructCalls(),0);
 assert.equal(JSON.stringify(f.allRules.map(rule=>({source:rule.item._source,value:rule.value,predicate:[...rule.predicate],ignored:rule.ignored}))),before);
 const actual=await f.parity(actor,params);assert.equal(actual.amount,prediction.amount);assert.equal(actual.flatTotal,prediction.flatTotal);assert.equal(f.constructCalls(),rules.length);
});

for(const [name,rule]of [['formula',{value:'@actor.level'}],['number string',{value:'5'}],['inject',{value:5,predicate:['self:{actor|id}']}],['battle form',{value:5,battleForm:true}],['item ABP',{value:5,type:'item'}],['ability',{value:5,type:'ability',ability:'wis'}],['dynamic object',{value:{brackets:[]}}],['inverted clamp',{value:5,min:10,max:0}],['unknown raw field',{value:5,execute:true}]])test('unqualified original source is rejected without invoking construct: '+name,async()=>{
 const f=receiverFixture({rules:[rule]});f.forbid();f.subscribe(f.authorize);await assert.rejects(f.run(),/static-receiver-unavailable/);assert.equal(f.constructCalls(),0);assert.equal(f.nativeCalls.length,0);
});

for(const kind of ['callback','dice','adjustment','duplicate','source edit','item flags','ignored','rule removed','item replaced','array replaced','resolve method','predicate method','ABP method'])test('selection await rechecks '+kind,async()=>{
 const f=receiverFixture({rules:[{value:5}]});f.forbid();f.subscribe(event=>{
  const answer=f.authorize(event);if(event.phase==='select'){
   const actor=f.prepared[0],rule=f.allRules[0];
   if(kind==='callback')actor.synthetics.modifiers['healing-received'].push(()=>{throw Error('must-not-call')});
   if(kind==='dice')actor.synthetics.damageDice['healing-received']=[()=>{throw Error('must-not-call')}];
   if(kind==='adjustment')actor.synthetics.modifierAdjustments['healing-received']=[{test(){throw Error('must-not-call')}}];
   if(kind==='duplicate')actor.synthetics.modifiers['healing-received'].push(actor.synthetics.modifiers['healing-received'][0]);
   if(kind==='source edit')rule.item._source.system.rules[0].value=99;
   if(kind==='item flags')rule.item._source.flags={tampered:true};
   if(kind==='ignored')rule.ignored=true;
   if(kind==='rule removed')actor.rules=[];
   if(kind==='item replaced')actor.items.set(rule.item.id,{...rule.item});
   if(kind==='array replaced')actor.synthetics.modifiers['healing-received']=[...actor.synthetics.modifiers['healing-received']];
   if(kind==='resolve method')rule.resolveValue=()=>5;
   if(kind==='predicate method')vm.runInContext('Hn.prototype.test=()=>true',f.context);
   if(kind==='ABP method')vm.runInContext('AutomaticBonusProgression$1.suppressRuleElement=()=>false',f.context);
  }return answer;
 });await assert.rejects(f.run(),/evidence-changed/);assert.equal(f.constructCalls(),0);assert.equal(f.nativeCalls.length,0);
});

test('model snapshots are deeply frozen and do not expose a mutator',async()=>{
 const f=receiverFixture({rules:[{value:5,predicate:[{or:['action:battle-medicine','action:treat-wounds']}]}],options:['action:treat-wounds']});f.subscribe(f.authorize);await f.run();
 const entry=f.events.find(e=>e.type==='batch-prepared').batch.candidates[0].receiver.entries[0];assert.equal(Object.isFrozen(entry.predicate[0].or),true);
 assert.deepEqual(Object.keys(f.api).sort(),['currentCall','descriptor','subscribe']);assert.equal(f.api.descriptor.staticReceiverModelVersion,1);assert.equal(f.api.descriptor.receiverPredicateModelVersion,1);
});

test('same UUID and copied synthetics are not the contextual source identity',async()=>{
 const f=receiverFixture({rules:[{value:5}]});f.subscribe(f.authorize);await f.run();const actor=f.prepared[0],params=f.nativeCalls[0].params;
 assert.throws(()=>f.pureModel({...actor},params),/static-receiver/);
 const model=f.pureModel(actor,params);params.rollOptions.add('changed');assert.equal(model.isCurrent(),false);
});

test('a later native leaf rejects changed sources but normal HP does not stale its model',async()=>{
 const f=receiverFixture({rules:[{value:5}],holdWrapper:true});f.subscribe(f.authorize);const work=f.run();await new Promise(setImmediate);const frame=f.frames[0].callFrame;
 f.prepared[0].system={attributes:{hp:{value:20}}};assert.equal(frame.isCurrent(),true);f.allRules[0].value=7;assert.equal(frame.isCurrent(),false);f.wrapperGate.resolve();await assert.rejects(work,/stale-wrapper/);assert.equal(f.nativeCalls.length,0);
});

test('finite inputs that overflow the receiving sum never authorize a native call',async()=>{
 const f=receiverFixture({rules:[[{value:Number.MAX_VALUE},{value:Number.MAX_VALUE}]]});f.forbid();f.subscribe(f.authorize);await assert.rejects(f.run(),/static-receiver-unavailable/);assert.equal(f.nativeCalls.length,0);assert.equal(f.constructCalls(),0);
});

test('ordinary unregistered healing keeps its original calls, including unsupported receivers',async()=>{
 const f=receiverFixture({rules:[{value:'@actor.level'},{value:5,type:'item'}]});f.subscribe(()=>({status:'unregistered'}));await f.run();assert.equal(f.nativeCalls.length,2);assert.equal(f.frames.every(frame=>!frame.callFrame),true);
});

test('a GM selection must choose the actual maximum, not an arbitrary claimant',async()=>{
 const f=receiverFixture({rules:[{value:5},{value:10}]});f.subscribe(event=>{const result=f.authorize(event);if(event.phase==='select')result.selections[0].selectedOrdinal=0;return result});await assert.rejects(f.run(),/selection-mismatch/);assert.equal(f.nativeCalls.length,0);
});
