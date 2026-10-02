import fs from 'node:fs';
import vm from 'node:vm';
import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';

const sourcePath=process.env.PF2E_MANUAL_POOL_BATCH_SOURCE;
const fixedSource=sourcePath?fs.readFileSync(sourcePath,'utf8'):null;
let patcher;
try{patcher=await import('./patch.mjs')}catch(error){if(error.code!=='ERR_MODULE_NOT_FOUND')throw error}
const candidate=process.env.NATIVE_BATCH_TEST_ORIGINAL==='1'||!patcher?fixedSource:fixedSource&&patcher.patchNativeManualPoolBatch(Buffer.from(fixedSource)).bytes.toString('utf8');
const sourceTest={skip:!fixedSource};
const turn=()=>new Promise(resolve=>setImmediate(resolve));
function deferred(){let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b});return {promise,resolve,reject}}
function region(source,start,end){const from=source.indexOf(start),to=source.indexOf(end,from);assert.ok(from>=0&&to>from);return source.slice(from,to)}
function fixture(source,{holdWrapper=false,actorIsTarget=false,deferPf2eAPI=false}={}){
 const hooks=[],nativeCalls=[],frames=[],events=[],clones=[],wrapperGate=deferred();
 let selected=[],ordinal=0;
 class DamageRoll{constructor(total){this.total=total;this._evaluated=true}alter(t,n){return {total:this.total*t+n}}toJSON(){return {total:this.total,_evaluated:this._evaluated}}}
 const game={pf2e:deferPf2eAPI?undefined:{existing:7},time:{worldTime:100},user:{id:'OWNER',active:true,getActiveTokens:()=>selected},messages:new Map()};
 Object.defineProperty(game.user,'targets',{get(){throw Error('gm-targets-must-not-be-read')}});
 const context=vm.createContext({game,crypto:{randomUUID:()=>`BATCH-${++ordinal}`},Hooks:{once:(name,fn)=>{if(name==='init')hooks.push(fn)}},jm:{onInit(){game.pf2e??={existing:7}}},setTimeout,clearTimeout,
  ui:{chat:{element:{}},notifications:{error(){}}},cn:DamageRoll,CONFIG:{PF2E:{chatDamageButtonShieldToggle:false}},
  htmlQuery:()=>({dataset:{actorIsTarget:actorIsTarget?'true':''}}),gt:(values,key)=>{const seen=new Set();return values.filter(value=>{const id=key(value);if(seen.has(id))return false;seen.add(id);return true})},
  extractEphemeralEffects:async()=>[],toggleOffShieldBlock:()=>{},shiftAdjustDamage:()=>{},ErrorPF2e:message=>Error(message)
 });
 if(source.includes('const __nativeManualPoolBatch='))vm.runInContext(region(source,'const __nativeManualPoolBatch=','async function applyDamageFromMessage('),context);
 vm.runInContext(region(source,'async function applyDamageFromMessage(','async function shiftAdjustDamage('),context);
 const initSuffix=source.slice(source.indexOf('jm.onInit();')).match(/^jm\.onInit\(\);(?:__nativeManualPoolBatch\.install\(\);)?/)[0];hooks.push(()=>vm.runInContext(initSuffix,context));for(const hook of hooks)hook();
 const api=game.pf2e.thirdPartyManualPoolBatch;
 function token(id,poolUUID='Actor.M'){
  const actor={id,uuid:`Actor.${id}`,poolUUID,alliance:null,ruleNonce:0,synthetics:{damageDice:{},modifiers:{},modifierAdjustments:{}},
   toObject(){return {_id:id,ruleNonce:this.ruleNonce,poolUUID:this.poolUUID}},getSelfRollOptions:prefix=>[`${prefix}:actor:${id}`],
   getContextualClone(){
    const actorSource=this.toObject();
    const contextualActor={uuid:this.uuid,synthetics:this.synthetics,getSelfRollOptions:()=>[`self:actor:${id}`],toObject:()=>actorSource,
     applyDamage(params){
      const callFrame=api?.currentCall(this,params);frames.push({callFrame,actor:this,params,patient:actor,wrongActor:api?.currentCall({uuid:this.uuid},params),copiedParams:api?.currentCall(this,{...params})});
      return (async()=>{if(holdWrapper)await wrapperGate.promise;if(callFrame&&callFrame.isCurrent?.()!==true)throw Error('native-batch-stale-wrapper');nativeCalls.push({actor:this,params});return this})();
     }};clones.push(contextualActor);return contextualActor;
   }};
  return {id,uuid:`Scene.S.Token.${id}`,actor,flags:{pf2e:{}}};
 }
 function message(id='R1',total=10){const result={id,uuid:`ChatMessage.${id}`,rolls:[new DamageRoll(total)],actor:null,item:null,
  flags:{pf2e:{context:{options:['source:healing'],outcome:'success'}}},toObject(){return {_id:this.id,flags:this.flags,rolls:this.rolls.map(roll=>roll.toJSON())}}};game.messages.set(id,result);return result}
 const result=message();
 const f={game,api,events,nativeCalls,frames,clones,token,message,result,recording:true,wrapperGate,select:tokens=>{selected=tokens},run:(message=result,input={})=>context.applyDamageFromMessage({message,multiplier:-1,...input})};
 f.authorize=(event)=>{
  if(event.phase==='admit')return {status:'participating',sourceBinding:{sourceType:'native-action',useId:'USE',resultId:event.batch.message.id,rollIndex:event.batch.rollIndex,stage:'healing',effectId:`${event.batch.message.id}:${event.batch.rollIndex}`},isCurrent:()=>f.recording};
  const groups=new Map();for(const member of event.batch.candidates){const poolUUID=member.patient.poolUUID;let group=groups.get(poolUUID);if(!group)groups.set(poolUUID,group=[]);group.push(member)}
  return {status:'selected',batchId:event.batch.batchId,sourceDigest:'a'.repeat(64),selections:[...groups].map(([poolUUID,members])=>({poolUUID,effectKey:`${event.batch.message.id}:${event.batch.rollIndex}`,patientUUIDs:[...new Set(members.map(member=>member.patient.uuid))],selectedOrdinal:members[0].targetOrdinal,grant:{permitNonce:`PERMIT-${poolUUID}`}}))};
 };
 f.subscribe=(authorizeBatch=f.authorize,options={})=>api?.subscribe(event=>events.push(event),{authorizeBatch,...options});return f;
}

test('fixed batch source supports one equal-tie native call for a shared pool',sourceTest,async()=>{
 const f=fixture(candidate);f.subscribe();f.select([f.token('A'),f.token('B')]);await f.run();
 assert.deepEqual(f.nativeCalls.map(call=>call.params.token.id),['A']);assert.equal(f.api.existing,undefined);assert.equal(f.game.pf2e.existing,7);
});
test('reverse target order changes only the equal-tie winner',sourceTest,async()=>{
 const f=fixture(candidate);f.subscribe();f.select([f.token('B'),f.token('A')]);await f.run();assert.deepEqual(f.nativeCalls.map(call=>call.params.token.id),['B']);
});
test('selection settles before the first original call without waiting for a later patient call',sourceTest,async()=>{
 const f=fixture(candidate),gate=deferred();let selected=false;f.subscribe(event=>event.phase==='admit'?f.authorize(event):gate.promise);f.select([f.token('A'),f.token('B')]);
 const work=f.run();await turn();assert.equal(f.nativeCalls.length,0);selected=true;
 const selectionEvent=f.events.find(event=>event.type==='batch-prepared');assert.ok(selectionEvent,'all candidates captured before any native call');
 gate.resolve(f.authorize({phase:'select',batch:selectionEvent.batch}));await work;assert.equal(selected,true);assert.equal(f.nativeCalls.length,1);
});
test('the actual captured token objects survive a later UI selection change',sourceTest,async()=>{
 const f=fixture(candidate),gate=deferred();f.subscribe(event=>event.phase==='admit'?f.authorize(event):gate.promise);
 const a=f.token('A'),b=f.token('B');f.select([a,b]);const work=f.run();await turn();f.select([f.token('DECOY')]);
 gate.resolve(f.authorize({phase:'select',batch:f.events.find(event=>event.type==='batch-prepared').batch}));await work;assert.equal(f.nativeCalls[0].params.token,a);
});
test('ordinary unregistered buttons keep all original calls and preparation order',sourceTest,async()=>{
 const before=fixture(fixedSource),after=fixture(candidate);after.api?.subscribe(()=>{throw Error('observer-failure')},{authorizeBatch:()=>({status:'unregistered'})});
 before.select([before.token('A'),before.token('B')]);after.select([after.token('A'),after.token('B')]);await before.run();await after.run();
 assert.deepEqual(after.nativeCalls.map(call=>[call.params.token.id,call.params.damage]),before.nativeCalls.map(call=>[call.params.token.id,call.params.damage]));assert.equal(after.frames.every(frame=>!frame.callFrame),true);
});
test('independent result and roll identities each retain one application',sourceTest,async()=>{
 const f=fixture(candidate);f.subscribe();const a=f.token('A'),b=f.token('B');f.select([a,b]);await f.run();const another=f.message('R2',7);await f.run(another);
 assert.deepEqual(f.nativeCalls.map(call=>call.params.damage),[-10,-7]);assert.equal(f.nativeCalls.reduce((total,call)=>total-call.params.damage,0),17);
});
test('separate roll indexes keep independent real source frames',sourceTest,async()=>{
 const f=fixture(candidate);f.subscribe();f.result.rolls.push(new f.result.rolls[0].constructor(7));f.select([f.token('A'),f.token('B')]);await f.run();await f.run(f.result,{rollIndex:1});assert.deepEqual(f.nativeCalls.map(call=>call.params.damage),[-10,-7]);
});
test('all independent pools are preflighted before the original sequence writes',sourceTest,async()=>{
 const f=fixture(candidate);f.subscribe();f.select([f.token('A','Actor.M'),f.token('B','Actor.N'),f.token('C','Actor.M')]);await f.run();assert.deepEqual(f.nativeCalls.map(call=>call.params.token.id),['A','B']);
 const denied=fixture(candidate);denied.subscribe();const first=denied.token('A','Actor.M'),second=denied.token('B','Actor.N');second.actor.synthetics.modifiers['healing-received']=[()=>{}];denied.select([first,second]);await assert.rejects(denied.run(),/reception-unavailable/);assert.equal(denied.nativeCalls.length,0);
});
test('a completed frame no longer qualifies and no batch notification is an HP terminal',sourceTest,async()=>{
 const f=fixture(candidate);f.subscribe();f.select([f.token('A'),f.token('B')]);await f.run();assert.equal(f.frames[0].callFrame.isCurrent(),false);assert.equal(f.events.filter(event=>event.type==='member-linked').length,1);assert.equal(f.events.some(event=>'terminalPromise'in event||'terminal'in event||'receipt'in event),false);
});
test('a shared single target is eligible without any second call',sourceTest,async()=>{
 const f=fixture(candidate);f.subscribe();f.select([f.token('A')]);await f.run();assert.equal(f.nativeCalls.length,1);assert.ok(f.frames[0].callFrame);
});
test('the source frame preserves the exact private grant object instead of a JSON clone',sourceTest,async()=>{
 const f=fixture(candidate),grant={permitNonce:'PRIVATE-GRANT'};f.subscribe(event=>{const answer=f.authorize(event);if(event.phase==='select')answer.selections[0].grant=grant;return answer});f.select([f.token('A')]);await f.run();assert.equal(f.frames[0].callFrame.grant,grant);
});
test('the current call qualifies only the exact original actor and params in its synchronous stack',sourceTest,async()=>{
 const f=fixture(candidate,{holdWrapper:true});f.subscribe();f.select([f.token('A')]);const work=f.run();await turn();const captured=f.frames[0];assert.ok(captured?.callFrame);
 assert.equal(f.api.currentCall(captured.actor,captured.params),null,'synchronous source stack already returned');
 assert.equal(captured.wrongActor,null);assert.equal(captured.copiedParams,null);
 assert.equal(f.api.currentCall(captured.actor,{...captured.params}),null);assert.equal(f.api.currentCall({uuid:captured.actor.uuid},captured.params),null);
 f.wrapperGate.resolve();await work;assert.equal(f.nativeCalls.length,1);assert.ok(Object.isFrozen(captured.callFrame));
});
for(const selector of ['damageDice','modifiers','modifierAdjustments'])test(`an unknown healing ${selector} entry is excluded without executing it`,sourceTest,async()=>{
 const f=fixture(candidate);f.subscribe();const target=f.token('A');let calls=0;target.actor.synthetics[selector]['healing-received']=[()=>{calls++;throw Error('must-not-pre-execute')}];f.select([target]);
 await assert.rejects(f.run(),/native-manual-batch-reception-unavailable/);assert.equal(calls,0);assert.equal(f.nativeCalls.length,0);
});
test('an inherited reception callback cannot bypass the empty-selector model',sourceTest,async()=>{
 const f=fixture(candidate);f.subscribe();const target=f.token('A');Object.setPrototypeOf(target.actor.synthetics.modifiers,{'healing-received':[()=>{}]});f.select([target]);await assert.rejects(f.run(),/reception-unavailable/);assert.equal(f.nativeCalls.length,0);
});
test('an asynchronous admission rejection is handled and cannot become an ordinary retry',sourceTest,async()=>{
 const f=fixture(candidate),dispose=f.subscribe(()=>Promise.reject(Error('async-admission-rejected')));f.select([f.token('A')]);await assert.rejects(f.run(),/synchronous-admission-required/);await turn();dispose?.();await assert.rejects(f.run(),/source-already-attempted/);assert.equal(f.nativeCalls.length,0);
});
test('a denied or unknown selection never invokes the original method',sourceTest,async()=>{
 const f=fixture(candidate);f.subscribe(event=>event.phase==='admit'?f.authorize(event):{status:'unknown'});f.select([f.token('A')]);await assert.rejects(f.run(),/native-manual-batch-selection/);assert.equal(f.nativeCalls.length,0);
});
test('a selection timeout closes the source and a late result does not run it',sourceTest,async()=>{
 const f=fixture(candidate),gate=deferred();f.subscribe(event=>event.phase==='admit'?f.authorize(event):gate.promise,{timeoutMs:10});f.select([f.token('A')]);
 await assert.rejects(f.run(),/native-manual-batch-selection-unknown/);const batch=f.events.find(event=>event.type==='batch-prepared')?.batch;assert.ok(batch);gate.resolve(f.authorize({phase:'select',batch}));await turn();assert.equal(f.nativeCalls.length,0);
 await assert.rejects(f.run(),/native-manual-batch-source-already-attempted/);
});
for(const [name,change] of [
 ['Stop',f=>{f.recording=false}],['result replacement',f=>{f.result.rolls[0]=new f.result.rolls[0].constructor(10)}],
 ['source options change',f=>{f.result.flags.pf2e.context.options.push('changed')}],['actor change',f=>{f.frames.length=0;f.target.actor.ruleNonce++}]
])test(`${name} during selection rejects before any native call`,sourceTest,async()=>{
 const f=fixture(candidate),gate=deferred();f.subscribe(event=>event.phase==='admit'?f.authorize(event):gate.promise);f.target=f.token('A');f.select([f.target]);const work=f.run();await turn();const batch=f.events.find(event=>event.type==='batch-prepared')?.batch;assert.ok(batch);change(f);gate.resolve(f.authorize({phase:'select',batch}));await assert.rejects(work,/native-manual-batch-evidence-changed/);assert.equal(f.nativeCalls.length,0);
});
test('an already participating result cannot fall back to an ordinary unregistered call',sourceTest,async()=>{
 const f=fixture(candidate);const dispose=f.subscribe();f.select([f.token('A')]);await f.run();dispose?.();await assert.rejects(f.run(),/native-manual-batch-source-already-attempted/);assert.equal(f.nativeCalls.length,1);
});
test('multiple participating gates cannot select an executor',sourceTest,async()=>{
 const f=fixture(candidate);f.subscribe();f.api?.subscribe(()=>{},{authorizeBatch:f.authorize});f.select([f.token('A')]);await assert.rejects(f.run(),/native-manual-batch-gate-ambiguous/);assert.equal(f.nativeCalls.length,0);
});
test('a participating answer followed by another gate failure leaves a source tombstone',sourceTest,async()=>{
 const f=fixture(candidate),dispose=f.subscribe(),other=f.api?.subscribe(()=>{},{authorizeBatch:()=>{throw Error('second-gate-failure')}});f.select([f.token('A')]);
 await assert.rejects(f.run(),/second-gate-failure/);dispose?.();other?.();await assert.rejects(f.run(),/native-manual-batch-source-already-attempted/);assert.equal(f.nativeCalls.length,0);
});
test('changing the original candidate method during selection cannot lend a frame to its replacement',sourceTest,async()=>{
 const f=fixture(candidate),gate=deferred();f.subscribe(event=>event.phase==='admit'?f.authorize(event):gate.promise);f.select([f.token('A')]);const work=f.run();await turn();
 const batch=f.events.find(event=>event.type==='batch-prepared').batch;batch.candidates[0].contextualActor.applyDamage=async()=>{};gate.resolve(f.authorize({phase:'select',batch}));await assert.rejects(work,/native-manual-batch-evidence-changed/);assert.equal(f.nativeCalls.length,0);
});
test('a captured frame can reject Stop at a later asynchronous wrapper leaf',sourceTest,async()=>{
 const f=fixture(candidate,{holdWrapper:true});f.subscribe();f.select([f.token('A')]);const work=f.run();await turn();assert.ok(f.frames[0]?.callFrame);assert.equal(f.frames[0].callFrame.isCurrent?.(),true);f.recording=false;f.wrapperGate.resolve();await assert.rejects(work,/native-batch-stale-wrapper/);assert.equal(f.nativeCalls.length,0);
});
test('replacing the original roll options object at an asynchronous leaf invalidates the captured frame',sourceTest,async()=>{
 const f=fixture(candidate,{holdWrapper:true});f.subscribe();f.select([f.token('A')]);const work=f.run();await turn();const captured=f.frames[0];assert.equal(captured.callFrame.isCurrent(),true);captured.params.rollOptions=new Set(captured.params.rollOptions);f.wrapperGate.resolve();await assert.rejects(work,/native-batch-stale-wrapper/);assert.equal(f.nativeCalls.length,0);
});
test('a selected member cannot be an extra or later equal-tie target',sourceTest,async()=>{
 for(const selectedOrdinal of [1,8]){const f=fixture(candidate);f.subscribe(event=>{const result=f.authorize(event);if(event.phase==='select')result.selections[0].selectedOrdinal=selectedOrdinal;return result});f.select([f.token('A'),f.token('B')]);await assert.rejects(f.run(),/native-manual-batch-selection/);assert.equal(f.nativeCalls.length,0)}
});
test('actorIsTarget captures the message token and the modifier prompt has no frame',sourceTest,async()=>{
 const f=fixture(candidate,{actorIsTarget:true});f.subscribe();f.result.token=f.token('A');f.select([f.token('DECOY')]);await f.run();assert.equal(f.nativeCalls[0].params.token,f.result.token);
 const prompt=fixture(candidate);prompt.subscribe();prompt.select([prompt.token('A')]);await prompt.run(prompt.result,{promptModifier:true});assert.equal(prompt.nativeCalls.length,0);assert.equal(prompt.events.length,0);
});
test('the public source surface has no arbitrary execute or terminal writer',sourceTest,()=>{
 const f=fixture(candidate);assert.ok(f.api);assert.deepEqual(Object.keys(f.api).sort(),['currentCall','descriptor','subscribe']);assert.ok(Object.isFrozen(f.api));assert.ok(Object.isFrozen(f.api.descriptor));
});
test('the observer installs after the fixed PF2e init actually creates its API',sourceTest,async()=>{
 const f=fixture(candidate,{deferPf2eAPI:true});assert.ok(f.api);f.subscribe();f.select([f.token('A')]);await f.run();assert.equal(f.nativeCalls.length,1);assert.equal(f.game.pf2e.existing,7);
});
test('foreign source bytes cannot publish a fixed-source candidate',()=>{
 assert.ok(patcher,'patcher exists');assert.throws(()=>patcher.patchNativeManualPoolBatch(Buffer.from('async function applyDamageFromMessage() {}')),/native-manual-batch-source-sha-mismatch/);
});
test('the full source candidate stays outside the complete checkout',()=>{
 assert.ok(patcher,'patcher exists');const output=fileURLToPath(new URL('../../../../private-batch',import.meta.url));assert.throws(()=>patcher.prepareNativeManualPoolBatch('not-read',output),/native-manual-batch-output-in-workspace/);
});

test('a Workbench result without outcome keeps the original optional native parameter',sourceTest,async()=>{
 const f=fixture(candidate);delete f.result.flags.pf2e.context.outcome;f.subscribe();f.select([f.token('A')]);await f.run();
 assert.equal(f.nativeCalls.length,1);const {params,callFrame}=f.frames[0];assert.equal(Object.hasOwn(params,'outcome'),true);assert.equal(params.outcome,undefined);assert.equal(Object.hasOwn(callFrame.paramsSnapshot,'outcome'),false);assert.equal(callFrame.paramsSnapshot.shieldBlockRequest,false);
});

test('an explicit null outcome remains present in the native parameter and snapshot',sourceTest,async()=>{
 const f=fixture(candidate);f.result.flags.pf2e.context.outcome=null;f.subscribe();f.select([f.token('A')]);await f.run();
 assert.equal(f.nativeCalls.length,1);assert.equal(Object.hasOwn(f.frames[0].callFrame.paramsSnapshot,'outcome'),true);assert.equal(f.frames[0].callFrame.paramsSnapshot.outcome,null);assert.equal(f.frames[0].params.outcome,null);
});

for(const outcome of [null,'success'])test(`undefined outcome changing to ${outcome} at the asynchronous native leaf is rejected`,sourceTest,async()=>{
 const f=fixture(candidate,{holdWrapper:true});delete f.result.flags.pf2e.context.outcome;f.subscribe();f.select([f.token('A')]);const work=f.run();work.catch(()=>{});await turn();
 const captured=f.frames[0];assert.ok(captured?.callFrame);assert.equal(captured.callFrame.isCurrent(),true);captured.params.outcome=outcome;f.wrapperGate.resolve();await assert.rejects(work,/native-batch-stale-wrapper/);assert.equal(f.nativeCalls.length,0);
});

test('optional outcome handling does not permit undefined source binding fields',sourceTest,async()=>{
 const f=fixture(candidate);delete f.result.flags.pf2e.context.outcome;f.subscribe(event=>{const answer=f.authorize(event);if(event.phase==='admit')answer.sourceBinding.invalid=undefined;return answer});f.select([f.token('A')]);await assert.rejects(f.run(),/native-manual-batch-invalid-json/);assert.equal(f.nativeCalls.length,0);
});
