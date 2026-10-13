import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createPatreonTimeCompletion} from '../scripts/exploration/patreon-time-completion.mjs';

const baseSourceSHA256='89ded325b92fa6b03dcf9257337ae2d628b3e99c987fe22cef9fff4e1837f4e9';
const pf2eSourceSHA256='d63da8312831b84905e6866b1dd3f9d93e95c1012955b0177ad2ce8ccf246157';
const checkpoint={id:'C',sessionId:'S',from:100,to:700,gmId:'G'};

function qualifiedDescriptor(){
 return {version:3,providerId:'patreon-v3',providerVersion:'3.3.0',pf2eVersion:'8.6.0',sourceSHA256:'b'.repeat(64),
  qualification:'patreon-original-seams.v1',seams:{
   time:'8c32f064ac33ab4a0ce264c8d74c8acbf8621627f1a8ef6f111e9ecb606cfa92',
   manualImmunity:'bdc211e3b7e437e91f8d9ab2c8f54c6b189abd8789278989de3a5a5b84e566a0',
   createItem:'14162e56764de7549e40aa9dad55249c14fe072acf181045f288bd02bd9bd7df'
  },markedCommitOwnership:'private-prepare.v1'};
}

function legacyDescriptor(version){
 return {version,providerId:'patreon-v3',providerVersion:'3.2.29',baseSourceSHA256,pf2eSourceSHA256,
  ...version===2?{markedCommitOwnership:'private-prepare.v1'}:{}};
}

function providerFixture(descriptor,apiName,{providerVersion=descriptor.providerVersion,pf2eVersion=descriptor.pf2eVersion??'8.5.1'}={}){
 const observers=new Set();let subscriptions=0,disposals=0,gate;
 const provider={descriptor,subscribe(observe,options={}){
  subscriptions++;observers.add(observe);gate=options.authorizeMarkedCommit;
  return ()=>{if(observers.delete(observe))disposals++};
 }};
 const game={user:{id:'G'},users:{activeGM:{id:'G'}},time:{worldTime:100},system:{id:'pf2e',version:pf2eVersion},
  modules:new Map([['patreon-v3',{active:true,version:providerVersion,api:{[apiName]:provider}}]])};
 return {game,observers,subscriptions:()=>subscriptions,disposals:()=>disposals,gate:invocation=>gate?.(invocation)};
}

function timeFixture(descriptor,versions){
 const f=providerFixture(descriptor,'explorationTimeCompletion',versions);
 f.adapter=createPatreonTimeCompletion({game:f.game,runtimeIdentity:()=>({userId:'G',clientNonce:'client'}),
  getDriverScope:()=>({leaseNonce:'lease'}),timeoutMs:1000});
 f.invocation={invocationId:'I',worldTime:700,delta:600,userId:'G',handlerUserId:'G',activeGMId:'G',
  options:{pf2eThirdPartyAutomation:{exploration:{sessionId:'S',checkpointId:'C',expectedFrom:100,expectedTo:700,gmId:'G'}}}};
 return f;
}

async function assertTimeReady(descriptor,versions){
 const f=timeFixture(descriptor,versions);
 try{
  assert.deepEqual(await f.adapter.beforeAdvance(checkpoint),{status:'ready'});
  assert.equal(f.observers.size,1);
 }finally{f.adapter.invalidate('test-finished')}
 assert.equal(f.observers.size,0);
 assert.equal(f.disposals(),1);
}

test('source-qualified time prepares the real checkpoint adapter',async()=>{
 await assertTimeReady(qualifiedDescriptor());
});

test('source-qualified time treats live module and PF2e versions as diagnostics',async()=>{
 await assertTimeReady(qualifiedDescriptor(),{providerVersion:'4.0.0',pf2eVersion:'9.0.0'});
});

test('the exact legacy time descriptor still prepares a checkpoint',async()=>{
 await assertTimeReady(legacyDescriptor(2));
});

test('source qualification retains exact marked invocation and one authorization',async()=>{
 const f=timeFixture(qualifiedDescriptor());
 try{
  assert.deepEqual(await f.adapter.beforeAdvance(checkpoint),{status:'ready'});
  const foreign=structuredClone(f.invocation);foreign.options.pf2eThirdPartyAutomation.exploration.checkpointId='other';
  assert.equal(f.gate(foreign),false);
  assert.equal(f.gate(f.invocation),true);
  assert.equal(f.gate({...f.invocation,invocationId:'second'}),false);
 }finally{f.adapter.invalidate('test-finished')}
 assert.equal(f.observers.size,0);
});

const invalidDescriptors=[
 ['a different qualification',descriptor=>{descriptor.qualification='unverified'}],
 ['a missing source hash',descriptor=>{delete descriptor.sourceSHA256}],
 ['a malformed source hash',descriptor=>{descriptor.sourceSHA256='not-a-sha256'}],
 ['a different provider',descriptor=>{descriptor.providerId='other'}],
 ['a different marked-commit owner',descriptor=>{descriptor.markedCommitOwnership='other'}],
 ...['time','manualImmunity','createItem'].flatMap(seam=>[
  ['a changed '+seam+' region',descriptor=>{descriptor.seams[seam]='a'.repeat(64)}],
  ['a missing '+seam+' region',descriptor=>{delete descriptor.seams[seam]}]
 ])
];

for(const [name,change] of invalidDescriptors){
 test('time refuses '+name+' before subscribing',async()=>{
  const descriptor=qualifiedDescriptor();change(descriptor);const f=timeFixture(descriptor);
  try{
   assert.equal((await f.adapter.beforeAdvance(checkpoint)).status,'blocked');
   assert.equal(f.subscriptions(),0);
  }finally{f.adapter.invalidate('test-finished')}
 });
}
