import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {createShieldDamageAdapter} from '../scripts/shield-damage-adapter.mjs';
import {verifyNativeIWRBridge} from '../scripts/native-iwr-verification.mjs';
import {NATIVE_IWR_PROFILES} from '../scripts/native-iwr-profiles.mjs';

const nativePath=process.env.PF2E_NATIVE_BUNDLE;
const profile=NATIVE_IWR_PROFILES['8.5.1'];
const hash=value=>createHash('sha256').update(value).digest('hex');
const source=nativePath?readFileSync(nativePath):null;
let nativeMethod;
if(source){
 assert.equal(hash(source),profile.patchedSHA256);
 const text=source.toString('utf8'),start=text.indexOf('\tasync applyDamage({ damage:')+1,end=text.indexOf('\n\tasync undoDamage(',start);
 assert(start>0&&end>start);
 nativeMethod=Function('return ({'+text.slice(start,end)+'}).applyDamage')();
 assert.equal(hash(Function.prototype.toString.call(nativeMethod)),profile.applyDamageSHA256);
}
const enabled={skip:!nativePath};
async function fixture({verified=true}={}){
 const user={id:'gm',isGM:true},game={system:{id:'pf2e',version:'8.5.1'},user,users:{activeGM:user}};
 const bridge=Object.freeze({version:'8.5.1',sourceSHA256:profile.originalSHA256,protocol:profile.protocol,applyDamage:nativeMethod});
 const proof=await verifyNativeIWRBridge({game,source:new Uint8Array(source),getNativeBridge:()=>bridge,hash});
 assert.equal(proof.ready,true,'Tests use actual issued source/method verification, not caller-created proof');
 const actor={uuid:'Actor.target',type:'character',hitPoints:{value:25,max:25,temp:0},attributes:{hp:{}},heldShield:null,testUserPermission:u=>u===user};
 const token={uuid:'Scene.scene.Token.target',id:'target',parent:{id:'scene'},actor},item={uuid:'Actor.fiend.Item.attack'};
 const params={damage:{total:15,instances:[]},token,item,rollOptions:new Set(['origin:trait:fiend'])},errors=[];
 const adapter=createShieldDamageAdapter({game,nativeBridgeVerification:verified?proof:null,getNativeBridge:()=>bridge,onError:e=>errors.push(e)});
 const invoke=(result,amounts={actorDamage:result.finalDamage,shieldDamage:0},dispatch)=>adapter.applyDamage(actor,params,p=>adapter.withNativeFrame(actor,p,async checked=>{
  if(dispatch)return dispatch(checked);
  await adapter.nativeDamageIWR(actor,checked,result,checked.rollOptions,amounts);return actor;
 }));
 return {game,actor,token,item,params,adapter,invoke,errors};
}
const result=(damage=0)=>({finalDamage:damage,applications:[{category:'resistance',type:'spirit damage from fiends',adjustment:-15,ignored:false}],persistent:[]});

test('trusted native frame observes complete zero absorption without an interceptor',enabled,async()=>{
 const f=await fixture(),seen=[];assert.equal(typeof f.adapter.addNativeObserver,'function');
 f.adapter.addNativeObserver(view=>seen.push(view));
 const raw=result();assert.equal(await f.invoke(raw),f.actor);assert.equal(seen.length,1);
 assert.deepEqual(seen[0],{actorUuid:f.actor.uuid,tokenUuid:f.token.uuid,itemUuid:f.item.uuid,total:15,rollOptions:['origin:trait:fiend'],iwr:raw,nativeAmounts:{actorDamage:0,shieldDamage:0}});
 assert.notEqual(seen[0].iwr,raw);assert.equal(f.errors.length,0);
});
test('observation is deeply immutable and conserves original native data',enabled,async()=>{
 const f=await fixture(),raw=result(),before=structuredClone(raw);let seen;
 f.adapter.addNativeObserver(view=>{seen=view;assert.throws(()=>view.iwr.applications[0].adjustment=-999,TypeError);assert.throws(()=>view.rollOptions.push('forged'),TypeError);assert.throws(()=>view.nativeAmounts.actorDamage=99,TypeError)});
 await f.invoke(raw);assert.deepEqual(raw,before);assert.deepEqual([...f.params.rollOptions],['origin:trait:fiend']);assert(Object.isFrozen(seen));assert.equal(f.errors.length,0);
});
test('snapshot precedes a genuine adapter prevention and retains native hardness amounts',enabled,async()=>{
 const f=await fixture(),raw=result(7),seen=[];raw.applications.push({category:'reduction',type:'Hardness',adjustment:-2});
 f.adapter.addNativeObserver(view=>seen.push(view));
 f.adapter.addNativeInterceptor(({prevent})=>prevent({marker:'pf2e-third-party-automation:transcendent-deflection:probe',flagKey:'transcendentDeflection',proof:{nonce:'probe',actorUuid:f.actor.uuid,weaponUuid:'Actor.target.Item.weapon',claimKey:'claim'}}));
 await f.invoke(raw,{actorDamage:5,shieldDamage:0});assert.equal(raw.finalDamage,0);assert.equal(seen.length,1);assert.equal(seen[0].iwr.finalDamage,7);assert.equal(seen[0].nativeAmounts.actorDamage,5);assert.equal(seen[0].iwr.applications[1].adjustment,-2);
});
test('public, copied and stale params cannot publish observations',enabled,async()=>{
 const f=await fixture(),seen=[];f.adapter.addNativeObserver(view=>seen.push(view));
 const call=p=>f.adapter.nativeDamageIWR(f.actor,p,result(),p.rollOptions,{actorDamage:0,shieldDamage:0});
 assert.equal(call(f.params),undefined);
 await f.invoke(result(),undefined,async p=>{assert.equal(call({...p}),undefined);await call(p);return f.actor});
 assert.equal(call(f.params),undefined);assert.equal(seen.length,1);
});
test('a second native bridge claim is rejected without a second observation',enabled,async()=>{
 const f=await fixture(),seen=[];f.adapter.addNativeObserver(view=>seen.push(view));
 await assert.rejects(f.invoke(result(),undefined,async p=>{
  await f.adapter.nativeDamageIWR(f.actor,p,result(),p.rollOptions,{actorDamage:0,shieldDamage:0});
  await f.adapter.nativeDamageIWR(f.actor,p,result(),p.rollOptions,{actorDamage:0,shieldDamage:0});return f.actor;
 }),/重复|重入/);assert.equal(seen.length,1);
});
test('observer matching and unsubscribe leave unrelated native execution available',enabled,async()=>{
 const f=await fixture();let calls=0,passedParams;
 const off=f.adapter.addNativeObserver(()=>calls++,{matches:(actor,params)=>{assert.equal(actor,f.actor);passedParams=params;return false}});
 await f.invoke(result());assert.equal(passedParams,f.params);assert.equal(calls,0);off();off();
 await f.invoke(result());assert.equal(calls,0);assert.equal(f.errors.length,0);
});
test('unverified bridge leaves ordinary damage alone and issues no observation',enabled,async()=>{
 const f=await fixture({verified:false});let calls=0;f.adapter.addNativeObserver(()=>calls++);
 assert.equal(await f.invoke(result()),f.actor);assert.equal(calls,0);assert.equal(f.errors.length,0);
});
test('observer and matching failures are reported without preventing native execution',enabled,async()=>{
 const f=await fixture();let calls=0;
 f.adapter.addNativeObserver(()=>{throw Error('observer failure')});
 f.adapter.addNativeObserver(()=>assert.fail('Rejected matcher must not run'),{matches:()=>{throw Error('match failure')}});
 f.adapter.addNativeObserver(()=>calls++);
 assert.equal(await f.invoke(result()),f.actor);assert.equal(calls,1);assert.deepEqual(f.errors.map(e=>e.message).sort(),['match failure','observer failure']);
});
test('returned promises do not delay native damage and their rejection is handled',enabled,async()=>{
 const f=await fixture();let reject;
 f.adapter.addNativeObserver(()=>new Promise((_resolve,r)=>{reject=r}));
 assert.equal(await f.invoke(result()),f.actor);reject(Error('late observer failure'));
 await new Promise(resolve=>setImmediate(resolve));assert.deepEqual(f.errors.map(e=>e.message),['late observer failure']);
});
test('pending persistent type/formula are copied without leaking native instances',enabled,async()=>{
 const f=await fixture(),raw=result(),instance={type:'fire',head:{expression:'1d6'}},seen=[];raw.persistent=[instance];
 f.adapter.addNativeObserver(view=>seen.push(view));await f.invoke(raw);
 assert.deepEqual(seen[0].iwr.persistent,[{type:'fire',formula:'1d6'}]);assert.notEqual(seen[0].iwr.persistent[0],instance);assert(Object.isFrozen(seen[0].iwr.persistent[0]));
});
test('disposing an enrolled observer before the native callback suppresses its receipt',enabled,async()=>{
 const f=await fixture();let calls=0;const off=f.adapter.addNativeObserver(()=>calls++);
 await f.invoke(result(),undefined,async p=>{off();await f.adapter.nativeDamageIWR(f.actor,p,result(),p.rollOptions,{actorDamage:0,shieldDamage:0});return f.actor});
 assert.equal(calls,0);assert.equal(f.errors.length,0);
});
test('a changed verified system profile invalidates observations already enrolled',enabled,async()=>{
 const f=await fixture();let calls=0;f.adapter.addNativeObserver(()=>calls++);
 assert.equal(await f.invoke(result(),undefined,async p=>{
  f.game.system.version='8.5.0';await f.adapter.nativeDamageIWR(f.actor,p,result(),p.rollOptions,{actorDamage:0,shieldDamage:0});return f.actor;
 }),f.actor);assert.equal(calls,0);assert.equal(f.errors.length,1);
});
test('uncloneable observation data cannot change or stop the native result',enabled,async()=>{
 const f=await fixture(),raw=result();raw.applications[0].unknownCallback=()=>{};let calls=0;f.adapter.addNativeObserver(()=>calls++);
 assert.equal(await f.invoke(raw),f.actor);assert.equal(calls,0);assert.equal(raw.finalDamage,0);assert.equal(typeof raw.applications[0].unknownCallback,'function');assert.equal(f.errors.length,1);
});
test('the pre-existing interceptor path still works with no observers',enabled,async()=>{
 const f=await fixture(),raw=result(7);let calls=0;
 f.adapter.addNativeInterceptor(({incoming})=>{calls++;assert.equal(incoming,7)});
 assert.equal(await f.invoke(raw),f.actor);assert.equal(calls,1);assert.equal(raw.finalDamage,7);assert.equal(f.errors.length,0);
});
test('async matcher results cannot enroll and rejected matches are handled',enabled,async()=>{
 const f=await fixture();let calls=0;
 f.adapter.addNativeObserver(()=>calls++,{matches:()=>Promise.resolve(true)});
 f.adapter.addNativeObserver(()=>calls++,{matches:()=>Promise.reject(Error('async match failure'))});
 assert.equal(await f.invoke(result()),f.actor);await new Promise(resolve=>setImmediate(resolve));
 assert.equal(calls,0);assert.deepEqual(f.errors.map(e=>e.message),['async match failure']);
});
