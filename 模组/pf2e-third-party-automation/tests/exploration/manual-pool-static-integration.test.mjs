import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createManualPoolApplication} from '../../scripts/exploration/manual-pool-application.mjs';

const sourceTag='pf2e-third-party-automation:source:R:0';
const descriptor={version:1,providerId:'pf2e',providerVersion:'8.5.1',protocol:'pf2e-third-party-automation:manual-pool-batch:1',model:'numeric-empty-reception.v1',baseSourceSHA256:'d63da8312831b84905e6866b1dd3f9d93e95c1012955b0177ad2ce8ccf246157'};
const staticDescriptor={...descriptor,model:'numeric-static-reception.v1',staticReceiverModelVersion:1,receiverPredicateModelVersion:1};

function fixture(model=descriptor,beforeClaim=async()=>{},sourceOptions=['action:treat-wounds','target:Actor.P']){
 const patient={id:'P',uuid:'Actor.P'},actor={id:'P',uuid:'Actor.P'},token={actor:patient},roll={},message={id:'R',rolls:[roll]};
 const game={user:{id:'O',active:true},messages:new Map([['R',message]])};
 const original={damage:-10,token,item:null,skipIWR:true,rollOptions:new Set(sourceOptions),outcome:'success',shieldBlockRequest:undefined};
 const frame={contextualActor:actor,patient,token,message,roll,rollIndex:0,item:null,grant:{permitNonce:'original-permit'},isCurrent:()=>true,
  paramsSnapshot:{damage:-10,skipIWR:true,rollOptions:[...sourceOptions],outcome:'success',shieldBlockRequest:undefined}};
 const provider={descriptor:model,currentCall:(actual,params)=>actual===actor&&params===original?frame:null};
 let claims=0,nativeCalls=0,application;
 const completion={async withApplication(grant,actual,operation){assert.equal(grant,frame.grant);assert.equal(actual,patient);claims++;await beforeClaim();return {result:await operation()}}};
 application=createManualPoolApplication({game,completion,getProvider:()=>provider});
 const captured=application.captureFrame(actor,original);
 const native=async params=>{
  nativeCalls++;
  const data={speaker:{actor:'P'},flags:{pf2e:{context:{type:'damage-taken',options:[...params.rollOptions]},appliedDamage:{uuid:'Actor.P',isHealing:true,isReverted:false}}}};
  await application.observeCreate(async input=>{const receipt={...input,id:'receipt',author:'O'};game.messages.set(receipt.id,receipt);return receipt},data);
  return actor;
 };
 const params={...original,rollOptions:new Set([...original.rollOptions,sourceTag])};
 return {application,actor,frame,captured,params,native,receipt:()=>game.messages.get('receipt'),claims:()=>claims,nativeCalls:()=>nativeCalls,
  run:()=>application.applyNativeDamage(actor,params,native,captured)};
}

test('the qualified constant receiver captures the original frame before any application',()=>{
 const f=fixture(staticDescriptor);assert.equal(f.captured,f.frame);assert.equal(f.claims(),0);assert.equal(f.nativeCalls(),0);
});

for(const field of ['staticReceiverModelVersion','receiverPredicateModelVersion'])test(`an unqualified ${field} cannot enroll a constant receiver frame`,()=>{
 const f=fixture({...staticDescriptor,[field]:2});assert.equal(f.captured,null);
});

for(const model of [descriptor,staticDescriptor])test(`${model.model} permits only the original options and same-result markers`,async()=>{
 const f=fixture(model);assert.equal(await f.run(),f.actor);assert.equal(f.claims(),1);assert.equal(f.nativeCalls(),1);
 const options=f.frame.paramsSnapshot.rollOptions;assert.deepEqual(options,['action:treat-wounds','target:Actor.P']);
});

const changes={
 'removed action option':options=>options.delete('action:treat-wounds'),
 'changed target option':options=>{options.delete('target:Actor.P');options.add('target:Actor.Other')},
 'missing source marker':options=>options.delete(sourceTag),
 'another source marker':options=>options.add('pf2e-third-party-automation:source:OTHER:0'),
 'another roll index':options=>{options.delete(sourceTag);options.add('pf2e-third-party-automation:source:R:1')},
 'preexisting receipt marker':options=>options.add('pf2e-third-party-automation:manual-pool-native:old-call'),
 'unobserved option':options=>options.add('action:battle-medicine')
};
for(const [name,change]of Object.entries(changes))test(`a ${name} is rejected before claiming or entering the original native leaf`,async()=>{
 const f=fixture();change(f.params.rollOptions);await assert.rejects(f.run(),/manual-pool-native-params-changed/);assert.equal(f.claims(),0);assert.equal(f.nativeCalls(),0);
});

test('options changed while the original claim is pending cannot reach the native leaf',async()=>{
 let release,started;
 const gate=new Promise(resolve=>{release=resolve}),entered=new Promise(resolve=>{started=resolve});
 const f=fixture(descriptor,async()=>{started();await gate}),pending=f.run();pending.catch(()=>{});
 await entered;f.params.rollOptions.add('unobserved-late-option');release();await assert.rejects(pending,/manual-pool-native-params-changed/);assert.equal(f.nativeCalls(),0);
});

// The native button inherits the result card's already-recorded source marker.
// Keep the observed 76-option shape without retaining private fixture IDs.
const markedOptions=['action:treat-wounds',...Array.from({length:74},(_,i)=>`fixture:option:${i}`),sourceTag];
for(const model of [descriptor,staticDescriptor])test(`${model.model} completes the original receipt once with an inherited same-result source`,async()=>{
 const f=fixture(model,async()=>{},markedOptions);
 assert.equal(await f.run(),f.actor);assert.equal(f.claims(),1);assert.equal(f.nativeCalls(),1);
 assert.deepEqual(f.frame.paramsSnapshot.rollOptions,markedOptions);
 const options=f.receipt().flags.pf2e.context.options;
 assert.equal(options.filter(option=>option===sourceTag).length,1);
 assert.equal(options.filter(option=>option.startsWith('pf2e-third-party-automation:manual-pool-native:')).length,1);
 assert.equal(options.length,77);
 await assert.rejects(f.run(),/manual-pool-original-frame-required/);assert.equal(f.nativeCalls(),1);
});

const invalidOriginal={
 'different result':['pf2e-third-party-automation:source:OTHER:0'],
 'different index':['pf2e-third-party-automation:source:R:1'],
 'multiple sources':[sourceTag,'pf2e-third-party-automation:source:OTHER:0'],
 'duplicate source':[sourceTag,sourceTag],
 'private receipt':[sourceTag,'pf2e-third-party-automation:manual-pool-native:prior'],
 'duplicate ordinary option':['action:treat-wounds',sourceTag]
};
for(const [name,options]of Object.entries(invalidOriginal))test(`an inherited ${name} cannot enter a claim or native application`,async()=>{
 const f=fixture(descriptor,async()=>{},['action:treat-wounds',...options]);
 await assert.rejects(f.run(),/manual-pool-native-params-changed/);assert.equal(f.claims(),0);assert.equal(f.nativeCalls(),0);
});

for(const [name,change]of Object.entries(changes))test(`an inherited source still rejects ${name} after the claim await`,async()=>{
 const f=fixture(descriptor,async()=>change(f.params.rollOptions),['action:treat-wounds','target:Actor.P',sourceTag]);
 await assert.rejects(f.run(),/manual-pool-native-params-changed/);assert.equal(f.claims(),1);assert.equal(f.nativeCalls(),0);
});

test('a duplicate actual source after the claim await cannot reach native',async()=>{
 const f=fixture(descriptor,async()=>{f.params.rollOptions=[...f.params.rollOptions,sourceTag]},markedOptions);
 await assert.rejects(f.run(),/manual-pool-native-params-changed/);assert.equal(f.claims(),1);assert.equal(f.nativeCalls(),0);
});
