import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createPatreonTimeCompletion} from '../scripts/exploration/patreon-time-completion.mjs';
import {createPatreonManualImmunity} from '../scripts/exploration/patreon-manual-immunity.mjs';
import {createManualPoolSources} from '../scripts/exploration/manual-pool-source.mjs';
import {MODULE_ID} from '../scripts/exploration/schema.mjs';

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
 const game={user:{id:'G'},users:{activeGM:{id:'G'}},time:{worldTime:100},system:{version:pf2eVersion},
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

function immunityFixture(descriptor,versions){
 const f=providerFixture(descriptor,'explorationManualImmunity',versions);
 f.adapter=createPatreonManualImmunity({game:f.game,fromUuid:async()=>null});
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

function assertImmunitySubscribed(descriptor,versions){
 const f=immunityFixture(descriptor,versions),dispose=f.adapter.subscribe(()=>{});
 try{
  assert.equal(f.subscriptions(),1);
  assert.equal(f.observers.size,1);
 }finally{dispose()}
 assert.equal(f.observers.size,0);
 assert.equal(f.disposals(),1);
}

test('source-qualified time prepares the real checkpoint adapter',async()=>{
 await assertTimeReady(qualifiedDescriptor());
});

test('source-qualified manual immunity registers one provider observation',()=>{
 assertImmunitySubscribed(qualifiedDescriptor());
});

test('source-qualified time treats live module and PF2e versions as diagnostics',async()=>{
 await assertTimeReady(qualifiedDescriptor(),{providerVersion:'4.0.0',pf2eVersion:'9.0.0'});
});

test('source-qualified manual immunity treats live module and PF2e versions as diagnostics',()=>{
 assertImmunitySubscribed(qualifiedDescriptor(),{providerVersion:'4.0.0',pf2eVersion:'9.0.0'});
});

test('the exact legacy time descriptor still prepares a checkpoint',async()=>{
 await assertTimeReady(legacyDescriptor(2));
});

test('the exact legacy manual-immunity descriptor still subscribes',()=>{
 assertImmunitySubscribed(legacyDescriptor(1));
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
 test('manual immunity refuses '+name+' before subscribing',()=>{
  const descriptor=qualifiedDescriptor();change(descriptor);const f=immunityFixture(descriptor);
  const dispose=f.adapter.subscribe(()=>{});
  try{assert.equal(f.subscriptions(),0)}finally{dispose()}
 });
}


function manualPoolFixture(){
 const user={id:'G',active:true,isGM:true},users=new Map([['G',user]]);users.activeGM=user;
 const actors=new Map(['H','P','M'].map(id=>[id,{id,uuid:'Actor.'+id,testUserPermission:owner=>owner===user}]));
 const healer=actors.get('H'),patient=actors.get('P'),messages=new Map(),hooks=new Map();
 const enrolled=[],recorded=[],resolved=[],updated=[];
 const variant={use(){}},action={slug:'treat-wounds',use(){},toActionVariant:()=>variant};
 const game={user,users,actors,messages,time:{worldTime:100},system:{version:'8.5.1'},
  pf2e:{actions:new Map([['treat-wounds',action]])},
  modules:new Map([['patreon-v3',{active:true,version:'3.3.0',api:{explorationManualImmunity:{descriptor:qualifiedDescriptor()}}}]])};
 const session={id:'S',manual:true,status:'recording',startedAt:100,actorUUIDs:['Actor.H','Actor.P','Actor.M']};
 function message(id,isCheckRoll,flags){
  return {id,isCheckRoll,isReroll:false,author:user,speaker:{actor:'H'},rolls:[{_evaluated:true,total:isCheckRoll?24:9}],flags,
   toObject(){return {id:this.id,isCheckRoll:this.isCheckRoll,author:this.author.id,speaker:structuredClone(this.speaker),
    rolls:structuredClone(this.rolls),flags:structuredClone(this.flags)}},
   async update(changes){
    updated.push({id:this.id,changes:structuredClone(changes)});
    for(const [path,value] of Object.entries(changes)){
     const keys=path.split('.');let at=this;for(const key of keys.slice(0,-1))at=at[key]??={};
     at[keys.at(-1)]=structuredClone(value);
    }
    return this;
   }};
 }
 const check=message('C',true,{pf2e:{context:{type:'skill-check',origin:{actor:healer.uuid},target:{actor:patient.uuid},
  options:['action:treat-wounds','exploration-manual:U']}},
  [MODULE_ID]:{explorationManualNative:{useId:'U',tag:'exploration-manual:U',patientUUID:patient.uuid,riskySurgery:false,startedAt:100}}});
 const result=message('D',false,{pf2e:{origin:{messageId:'C'}}});messages.set('C',check);messages.set('D',result);
 const batch={descriptor:{version:1,providerId:'pf2e',providerVersion:'8.5.1',
  protocol:'pf2e-third-party-automation:manual-pool-batch:1',model:'numeric-empty-reception.v1',baseSourceSHA256:pf2eSourceSHA256}};
 const sources=createManualPoolSources({game,Hooks:{on:(name,fn)=>{hooks.set(name,fn);return name},off:name=>hooks.delete(name)},
  ledger:{async recordManualPoolSource(source){recorded.push(structuredClone(source))}},
  fromUuid:async uuid=>{resolved.push(uuid);return [...actors.values()].find(actor=>actor.uuid===uuid)??null},
  hpPools:{discover:()=>({ready:true,poolUUID:'Actor.M',memberUUIDs:['Actor.M','Actor.P']})},
  clientNonce:'client',getSession:async()=>session,isIssuer:()=>true,onEnroll:async source=>enrolled.push(structuredClone(source)),
  getBatchProvider:()=>batch,onError:error=>{throw error}});
 const scope={slug:'treat-wounds',actors:[healer],user,action,variant,params:{target:patient}};
 return {sources,scope,check,result,healer,enrolled,recorded,resolved,updated,
  saveOriginalBinding(){
   check.flags[MODULE_ID].explorationManualNative.patreonImmunity={invocationId:'I',messageId:'C',useId:'U',
    tag:'exploration-manual:U',actorUUID:'Actor.H',patientUUID:'Actor.P',sourceUserId:'G',startedAt:100,
    recordingSessionId:'S',targetSnapshot:{type:'check-context',actorUUID:'Actor.P',tokenUUID:'Scene.S.Token.T'}};
  }};
}

async function originalPoolUse(f){
 f.sources.start();
 const handle=await f.sources.beginNative(f.scope,{useId:'U',tag:'exploration-manual:U'});
 assert.ok(handle);f.resolved.length=0;
 await f.sources.nativeCheck(handle,{actor:f.healer,message:f.check});
 await f.sources.nativeResult(f.result);
 return handle;
}

test('a qualified updated Patreon source waits for original patient metadata before publication',async()=>{
 const f=manualPoolFixture();
 try{
  await originalPoolUse(f);
  assert.deepEqual(f.enrolled,[]);
  assert.deepEqual(f.recorded,[]);
  assert.deepEqual(f.resolved,[]);
  assert.deepEqual(f.updated,[]);
 }finally{f.sources.stop()}
});

test('saving the original patient binding releases only the observed check and result once',async()=>{
 const f=manualPoolFixture();
 try{
  const handle=await originalPoolUse(f);
  assert.equal(f.recorded.length,0);
  f.saveOriginalBinding();
  await f.sources.nativeCheck(handle,{actor:f.healer,message:f.check});
  await f.sources.nativeResult(f.result);
  assert.equal(f.enrolled.length,1);assert.equal(f.recorded.length,1);
  assert.equal(f.recorded[0].checkId,'C');assert.equal(f.recorded[0].resultId,'D');
  assert.equal(f.recorded[0].patientUUID,'Actor.P');assert.equal(f.recorded[0].useId,'U');
  assert.equal(f.updated.filter(write=>write.id==='D').length,1);
 }finally{f.sources.stop()}
});
