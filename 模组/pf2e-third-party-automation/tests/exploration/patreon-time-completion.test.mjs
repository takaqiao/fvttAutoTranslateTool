import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createPatreonTimeCompletion} from '../../scripts/exploration/patreon-time-completion.mjs';
import {createTimeEffects} from '../../scripts/exploration/time-effects.mjs';
import {createClock} from '../../scripts/exploration/clock.mjs';

const baseSourceSHA256='89ded325b92fa6b03dcf9257337ae2d628b3e99c987fe22cef9fff4e1837f4e9';
const pf2eSourceSHA256='d63da8312831b84905e6866b1dd3f9d93e95c1012955b0177ad2ce8ccf246157';
const checkpoint={id:'C',sessionId:'S',from:100,to:700,gmId:'G'};
const nativeCallback=JSON.parse(readFileSync(new URL('./fixtures/patreon-world-time-native-options.json',import.meta.url),'utf8'));
function deferred(){let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b});return {promise,resolve,reject}}
// Version 2 is an offline boundary contract, not the prepared dependency v4.
function fixture({version=2,timeoutMs=1000,preparedCheckpoint=checkpoint,options}={}){
 const descriptor={version,providerId:'patreon-v3',providerVersion:'3.2.29',baseSourceSHA256,pf2eSourceSHA256,...version===2?{markedCommitOwnership:'private-prepare.v1'}:{}};
 const {gmId,from,to,id,sessionId}=preparedCheckpoint;
 let observer,gate,identity={userId:gmId,clientNonce:'client'},scope={leaseNonce:'lease'};
 const api={descriptor,subscribe(fn,options){observer=fn;gate=options?.authorizeMarkedCommit;return ()=>{observer=undefined;gate=undefined}}};
 const module={active:true,api:{explorationTimeCompletion:api}},game={user:{id:gmId},users:{activeGM:{id:gmId}},modules:new Map([['patreon-v3',module]]),time:{worldTime:from}};
 const adapter=createPatreonTimeCompletion({game,runtimeIdentity:()=>identity,getDriverScope:()=>scope,timeoutMs});
 const invocation={invocationId:'I',worldTime:to,delta:to-from,userId:gmId,handlerUserId:gmId,activeGMId:gmId,
  options:options??{pf2eThirdPartyAutomation:{exploration:{sessionId,checkpointId:id,expectedFrom:from,expectedTo:to,gmId}}}};
 const proof={descriptor,invocation,members:[],rules:[],rolls:[],writes:[],errors:[]};
 return {adapter,game,module,descriptor,invocation,proof,checkpoint:preparedCheckpoint,gate:(inv=invocation)=>gate?.(inv),emit:(terminalPromise,inv=invocation)=>observer?.({descriptor,invocation:inv,terminalPromise}),setScope:value=>scope=value,setIdentity:value=>identity=value};
}
function nativeFixture(){
 const marker=nativeCallback.options.pf2eThirdPartyAutomation.exploration;
 const preparedCheckpoint={id:marker.checkpointId,sessionId:marker.sessionId,from:marker.expectedFrom,to:marker.expectedTo,gmId:marker.gmId};
 const f=fixture({preparedCheckpoint,options:structuredClone(nativeCallback.options)});
 Object.assign(f.invocation,{worldTime:nativeCallback.time,delta:nativeCallback.dt,userId:nativeCallback.userId});
 return f;
}

test('actual Foundry 14.368 world-time callback authorizes and retains complete terminal options',async()=>{
 const f=nativeFixture();assert.equal((await f.adapter.beforeAdvance(f.checkpoint)).status,'ready');
 assert.equal(f.gate(),true);assert.equal(f.gate({...f.invocation,invocationId:'I2'}),false);
 f.emit(Promise.resolve(f.proof));assert.deepEqual(await f.adapter.settle(f.checkpoint),{status:'ready',proof:[f.proof]});
});

test('native modifiedTime accepts valid finite millisecond timestamps',async()=>{
 for(const modifiedTime of [0,1790841529358,8640000000000000]){
  const f=nativeFixture();f.invocation.options.modifiedTime=modifiedTime;await f.adapter.beforeAdvance(f.checkpoint);
  assert.equal(f.gate(),true);f.emit(Promise.resolve(f.proof));assert.equal((await f.adapter.settle(f.checkpoint)).status,'ready');
 }
});

test('gate rejects incomplete, changed and foreign native option fields without consuming ownership',async()=>{
 const changes=[
  ...['action','documentName','modifiedTime','diff','recursive','render','parent'].map(key=>options=>{delete options[key]}),
  options=>{options.action='create'},options=>{options.documentName='Actor'},
  ...['diff','recursive','render'].map(key=>options=>{options[key]=false}),options=>{options.parent={id:'foreign'}},
  ...[NaN,Infinity,-Infinity,-1,1.5,'1790841529357',null,undefined,8640000000000001,Number.MAX_SAFE_INTEGER+1]
   .map(value=>options=>{options.modifiedTime=value}),
  options=>{options.foreign=true},options=>{options.foreign=undefined},
  options=>{delete options.pf2eThirdPartyAutomation},
  options=>{options.pf2eThirdPartyAutomation.foreign=true},options=>{options.pf2eThirdPartyAutomation.foreign=undefined},
  options=>{options.pf2eThirdPartyAutomation.exploration.foreign=true},options=>{options.pf2eThirdPartyAutomation.exploration.foreign=undefined},
  ...['sessionId','checkpointId','expectedFrom','expectedTo','gmId'].flatMap(key=>[
   options=>{delete options.pf2eThirdPartyAutomation.exploration[key]},
   options=>{options.pf2eThirdPartyAutomation.exploration[key]=key.startsWith('expected')?-1:'foreign'}
  ])
 ];
 for(const change of changes){
  const f=nativeFixture();await f.adapter.beforeAdvance(f.checkpoint);const invocation=structuredClone(f.invocation);change(invocation.options);
  assert.equal(f.gate(invocation),false);assert.equal(f.gate(),true);
  f.emit(Promise.resolve(f.proof));assert.equal((await f.adapter.settle(f.checkpoint)).status,'ready');
 }
});

test('marker-only options reject foreign undefined fields at every marker level',async()=>{
 for(const level of ['options','marker','exploration']){
  const f=fixture();await f.adapter.beforeAdvance(checkpoint);const invocation=structuredClone(f.invocation);
  const target=level==='options'?invocation.options:level==='marker'?invocation.options.pf2eThirdPartyAutomation:invocation.options.pf2eThirdPartyAutomation.exploration;
  target.foreign=undefined;assert.equal(f.gate(invocation),false);assert.equal(f.gate(),true);
  f.emit(Promise.resolve(f.proof));assert.equal((await f.adapter.settle(checkpoint)).status,'ready');
 }
});

test('native observation must match the full authorized invocation snapshot',async()=>{
 const f=nativeFixture();await f.adapter.beforeAdvance(f.checkpoint);const authorized=structuredClone(f.invocation);assert.equal(f.gate(),true);
 f.invocation.options.modifiedTime++;f.emit(Promise.resolve({...f.proof,invocation:f.invocation}));
 let settled=false;const result=f.adapter.settle(f.checkpoint).then(value=>{settled=true;return value});await Promise.resolve();assert.equal(settled,false);
 const proof={...f.proof,invocation:authorized};f.emit(Promise.resolve(proof),authorized);assert.deepEqual(await result,{status:'ready',proof:[proof]});
});

test('native terminal proof rejects changed, missing and foreign callback options',async()=>{
 const changes=[
  invocation=>{invocation.options.modifiedTime++},
  ...['action','documentName','modifiedTime','diff','recursive','render','parent'].map(key=>invocation=>{delete invocation.options[key]}),
  invocation=>{invocation.options.diff=false},invocation=>{invocation.options.foreign=undefined},
  invocation=>{invocation.options.pf2eThirdPartyAutomation.exploration.gmId='foreign'},
  invocation=>{invocation.options.pf2eThirdPartyAutomation.exploration.foreign=undefined},invocation=>{invocation.foreign=undefined}
 ];
 for(const change of changes){
  const f=nativeFixture();await f.adapter.beforeAdvance(f.checkpoint);assert.equal(f.gate(),true);
  const invocation=structuredClone(f.invocation);change(invocation);
  f.emit(Promise.resolve({...f.proof,invocation}));assert.equal((await f.adapter.settle(f.checkpoint)).reason,'patreon-completion-unproven');
 }
});

test('actual v4 descriptor is blocked before time when a passing Patreon rule needs completion',async()=>{
 const f=fixture({version:1});
 const effects=createTimeEffects({capabilities:{activePassiveRules:async()=>[{providerId:'pf2e-patreon',passing:true}]},completionAdapters:[f.adapter]});
 assert.deepEqual(await effects.beforeAdvance(checkpoint),{status:'blocked',reason:'patreon-marked-ownership-unavailable'});
 assert.equal(f.gate(),undefined);
});
test('active provider without passing rules retains ordinary time behavior',async()=>{
 const f=fixture({version:1});
 const effects=createTimeEffects({capabilities:{activePassiveRules:async()=>[]},completionAdapters:[f.adapter]});
 assert.equal((await effects.beforeAdvance(checkpoint)).status,'ready');
 assert.deepEqual(await effects.settle(checkpoint),{status:'ready',proof:[]});
});
test('exact v2 provider prepares ownership with zero rules and waits original decrement writes',async()=>{
 const f=fixture(),native=deferred();
 const effects=createTimeEffects({capabilities:{activePassiveRules:async()=>[]},completionAdapters:[f.adapter]});
 assert.equal((await effects.beforeAdvance(checkpoint)).status,'ready');assert.equal(f.gate(),true);
 f.emit(native.promise);let settled=false;const result=effects.settle(checkpoint).then(value=>{settled=true;return value});
 await Promise.resolve();assert.equal(settled,false);
 const proof={...f.proof,writes:[{sequence:1,kind:'effect-start',documentUUID:'Actor.Other.Item.E',data:{'system.start.value':700},status:'fulfilled'},
  {sequence:2,kind:'effect-decrease',documentUUID:'Actor.Other.Item.E',data:null,status:'fulfilled'}]};
 native.resolve(proof);assert.deepEqual(await result,{status:'ready',proof:[proof]});
});
test('inactive provider leaves ordinary time untouched',async()=>{
 const f=fixture();f.module.active=false;
 const effects=createTimeEffects({capabilities:{activePassiveRules:async()=>[]},completionAdapters:[f.adapter]});
 assert.equal((await effects.beforeAdvance(checkpoint)).status,'ready');
 assert.deepEqual(await effects.settle(checkpoint),{status:'ready',proof:[]});
});
test('future gated contract waits the same terminal Promise and preserves no-op proof',async()=>{
 const f=fixture(),native=deferred();assert.equal((await f.adapter.beforeAdvance(checkpoint)).status,'ready');
 assert.equal(f.gate(),true);f.emit(native.promise);let settled=false;
 const result=f.adapter.settle(checkpoint).then(value=>{settled=true;return value});await Promise.resolve();assert.equal(settled,false);
 native.resolve(f.proof);assert.deepEqual(await result,{status:'ready',proof:[f.proof]});
 assert.equal(f.gate(),undefined);
});
for(const field of ['worldTime','delta','userId','handlerUserId','activeGMId','invocationId','options'])test(`gate rejects wrong ${field} and cannot authorize a second invocation`,async()=>{
 const f=fixture();await f.adapter.beforeAdvance(checkpoint);const wrong=structuredClone(f.invocation);
 wrong[field]=field==='options'?{foreign:true}:field==='invocationId'?'':typeof wrong[field]==='number'?0:'Other';
 assert.equal(f.gate(wrong),false);assert.equal(f.gate(),true);assert.equal(f.gate({...f.invocation,invocationId:'I2'}),false);
 f.emit(Promise.resolve(f.proof));assert.equal((await f.adapter.settle(checkpoint)).status,'ready');
});
test('unclaimed observation cannot prove completion or consume a later gate',async()=>{
 const f=fixture({timeoutMs:10});await f.adapter.beforeAdvance(checkpoint);f.emit(Promise.resolve(f.proof));
 assert.equal((await f.adapter.settle(checkpoint)).status,'uncertain');assert.equal(f.gate(),undefined);
});
test('duplicate exact observation makes completion uncertain',async()=>{
 const f=fixture(),native=deferred();await f.adapter.beforeAdvance(checkpoint);f.gate();f.emit(native.promise);f.emit(native.promise);
 native.resolve(f.proof);assert.equal((await f.adapter.settle(checkpoint)).status,'uncertain');
});
test('rejection, undefined terminal and unproven writes never become completion',async()=>{
 for(const kind of ['reject','undefined','unproven']){
  const f=fixture();await f.adapter.beforeAdvance(checkpoint);f.gate();
  if(kind==='reject')f.emit(Promise.reject(Error('native-write-failed')));
  else if(kind==='undefined')f.emit(undefined);
  else f.emit(Promise.resolve({...f.proof,writes:[{status:'unproven'}]}));
  assert.equal((await f.adapter.settle(checkpoint)).status,'uncertain');
 }
});
test('timeout stays uncertain after late terminal and cannot authorize again',async()=>{
 const f=fixture({timeoutMs:10}),native=deferred();await f.adapter.beforeAdvance(checkpoint);f.gate();f.emit(native.promise);
 assert.equal((await f.adapter.settle(checkpoint)).reason,'patreon-completion-timeout');native.resolve(f.proof);await Promise.resolve();
 assert.equal((await f.adapter.settle(checkpoint)).status,'uncertain');assert.equal((await f.adapter.beforeAdvance(checkpoint)).status,'blocked');
});
test('lost private lease, rotated identity and changed API reject before native',async()=>{
 for(const kind of ['lease','identity','api']){
  const f=fixture();await f.adapter.beforeAdvance(checkpoint);
  if(kind==='lease')f.setScope(null);if(kind==='identity')f.setIdentity({userId:'G',clientNonce:'new'});if(kind==='api')f.module.api.explorationTimeCompletion={};
  assert.equal(f.gate(),false);assert.equal((await f.adapter.settle(checkpoint)).status,'uncertain');
 }
});
test('invalidate disposes private observation and permits a fresh legitimate checkpoint',async()=>{
 const f=fixture();await f.adapter.beforeAdvance(checkpoint);f.adapter.invalidate('disconnect');assert.equal(f.gate(),undefined);
 assert.equal((await f.adapter.settle(checkpoint)).status,'uncertain');
 f.setIdentity({userId:'G',clientNonce:'new'});assert.equal((await f.adapter.beforeAdvance({...checkpoint,id:'C2'})).status,'ready');f.adapter.invalidate('end');
});
test('clock progress metadata does not change the prepared checkpoint identity',async()=>{
 const f=fixture();await f.adapter.beforeAdvance(checkpoint);f.gate();f.emit(Promise.resolve(f.proof));
 assert.equal((await f.adapter.settle({...checkpoint,state:'nativeResolved',evidence:[{kind:'world-time-hook'}]})).status,'ready');
});
test('settled terminal cannot outlive its captured private lease',async()=>{
 const f=fixture();await f.adapter.beforeAdvance(checkpoint);f.gate();f.emit(Promise.resolve(f.proof));await Promise.resolve();
 f.setScope(null);assert.equal((await f.adapter.settle(checkpoint)).status,'uncertain');
});
test('new private lease can prepare after a retired uncertain operation without replaying it',async()=>{
 const f=fixture({timeoutMs:10});await f.adapter.beforeAdvance(checkpoint);await f.adapter.settle(checkpoint);
 f.setScope({leaseNonce:'new-lease'});
 assert.equal((await f.adapter.beforeAdvance({...checkpoint,id:'C2',sessionId:'S2'})).status,'ready');f.adapter.invalidate('end');
});
test('v4 qualification reaches the real clock but issues neither time nor a ledger claim',async()=>{
 const f=fixture({version:1});let advances=0,claims=0;
 f.game.time.advance=async()=>{advances++};
 const timeEffects=createTimeEffects({capabilities:{activePassiveRules:async()=>[{providerId:'pf2e-patreon',passing:true}]},completionAdapters:[f.adapter]});
 const clock=createClock({game:f.game,Hooks:{},ledger:{getClockCommit:async()=>null,upsertClockCommit:async()=>{claims++}},timeEffects,isAuthority:()=>true});
 assert.equal((await clock.advanceTo(checkpoint)).reason,'patreon-marked-ownership-unavailable');
 assert.equal(advances,0);assert.equal(claims,0);
});
test('descriptor source mismatch blocks and a mismatched terminal proof stays uncertain',async()=>{
 const wrong=fixture();wrong.descriptor.baseSourceSHA256='0'.repeat(64);
 assert.equal((await wrong.adapter.beforeAdvance(checkpoint)).status,'blocked');
 const f=fixture();await f.adapter.beforeAdvance(checkpoint);f.gate();
 f.emit(Promise.resolve({...f.proof,invocation:{...f.invocation,delta:6}}));
 assert.equal((await f.adapter.settle(checkpoint)).reason,'patreon-completion-unproven');
});
test('native option key order is structural while foreign fields are rejected',async()=>{
 const f=fixture();await f.adapter.beforeAdvance(checkpoint);
 const invocation=structuredClone(f.invocation);
 invocation.options.pf2eThirdPartyAutomation.exploration={gmId:'G',expectedTo:700,expectedFrom:100,checkpointId:'C',sessionId:'S'};
 assert.equal(f.gate({...invocation,foreign:true}),false);assert.equal(f.gate({...invocation,foreign:undefined}),false);assert.equal(f.gate(invocation),true);
 f.emit(Promise.resolve({...f.proof,invocation}),invocation);assert.equal((await f.adapter.settle(checkpoint)).status,'ready');
});
