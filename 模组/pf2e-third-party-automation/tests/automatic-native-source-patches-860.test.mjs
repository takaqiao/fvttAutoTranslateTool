import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {readFileSync} from 'node:fs';
import {buildNativeBridge,buildSharedManualPair} from '../tools/automatic-source-patches/native.mjs';
import {verifyNativeIWRBridge,isVerifiedNativeIWRBridge} from '../scripts/native-iwr-verification.mjs';
import {nativeMethod,automaticDescriptor,bridgeStatement} from '../scripts/native-source-shapes.mjs';
import {currentToolbelt} from './toolbelt-current-fixture.mjs';
import {native860Source,hash} from './native-iwr-860-fixture.mjs';
import {receiverContext} from './native-iwr-860-receiver-fixture.mjs';

const source=native860Source();
assert.ok(process.env.TOOLBELT_MANUAL_SOURCE,'TOOLBELT_MANUAL_SOURCE is required');
const oldToolbelt=readFileSync(process.env.TOOLBELT_MANUAL_SOURCE);
assert.equal(hash(oldToolbelt),'2946fa27eaf0963098f9f48ee48f777c5cf65166b56b043de9404f09813c240f');
const toolbelt=currentToolbelt(oldToolbelt);
const bridge=(input=source,version='8.6.0')=>buildNativeBridge({source:input,version});
const pair=(pf2eSource=bridge().buffer,toolbeltSource=toolbelt)=>buildSharedManualPair({pf2eSource,toolbeltSource,pf2eVersion:'8.6.0',toolbeltVersion:'3.57.1'});

test('audited 8.6.0 damage source builds independently of the manifest version',()=>{
 const result=bridge(source,'unlisted-release');
 assert.equal(result.status,'patch');assert.equal(result.descriptor.version,'unlisted-release');
 assert.equal(result.descriptor.sourceSHA256,hash(source));
 assert.equal(hash(nativeMethod(result.buffer.toString())),'e1d69d11d7a4202673bd432a0f9780d082c9e8f83941910f95da6bc0db955a00');
 assert.deepEqual(bridge(result.buffer).buffer,result.buffer);
});

test('8.6.0 and current Toolbelt build a byte stable shared manual pair',()=>{
 const result=pair(),again=pair(result.pf2e,result.toolbelt);
 assert.equal(again.alreadyPatched,true);assert.deepEqual(again.pf2e,result.pf2e);assert.deepEqual(again.toolbelt,result.toolbelt);
 const updated=pair(bridge().buffer,result.toolbelt);assert.equal(updated.alreadyPatched,false);
 assert.deepEqual(pair(updated.pf2e,updated.toolbelt).pf2e,updated.pf2e);
});

test('8.6.0 receiver captures its actual RuleElement and Predicate bindings',()=>{
 const context=receiverContext(pair().pf2e.toString());
 vm.runInContext(`
  const actor={uuid:'Actor.patient',items:new Map(),rules:[],flags:{pf2e:{}},synthetics:{modifiers:{},damageDice:{},modifierAdjustments:{}}};
  const raw={key:'FlatModifier',selector:['healing-received'],value:5,type:'status'};
  const item={id:'bonus',uuid:'Actor.patient.Item.bonus',name:'Healing bonus',actor,_source:{system:{rules:[raw]}},isOfType:type=>type==='effect'};
  actor.items.set(item.id,item);
  const rule=new FlatModifierRuleElement(raw,{parent:item,sourceIndex:0});actor.rules.push(rule);rule.beforePrepareData();
  const params={damage:-10,rollOptions:new Set(),outcome:'success'};
  globalThis.model=__nativeManualPoolStaticReceiver.model(actor,params);
 `,context);
 assert.equal(context.model.amount,15);assert.equal(context.model.flatTotal,5);assert.equal(context.model.isCurrent(),true);
 vm.runInContext('Un.prototype.test=()=>true;',context);
 assert.equal(context.model.isCurrent(),false);
});

test('a patched 8.6.0 observer cannot substitute another native binding',()=>{
 const result=pair(),changed=Buffer.from(result.pf2e.toString().replace('resolveValue:q.prototype.resolveValue','resolveValue:Y.prototype.resolveValue'));
 assert.notDeepEqual(changed,result.pf2e);
 assert.throws(()=>pair(changed,result.toolbelt),/native-shared-seam-observer/);
});

test('8.6.0 served source and retained native function receive identity bound IWR proof',async()=>{
 const result=pair(),text=result.pf2e.toString(),applyDamage=Function('return ({'+nativeMethod(text)+'}).applyDamage')();
 const retained=Object.freeze({...automaticDescriptor(bridgeStatement(text)),applyDamage}),game={system:{id:'pf2e',version:'8.6.0'}};
 const proof=await verifyNativeIWRBridge({game,source:result.pf2e,getNativeBridge:()=>retained,hash});
 assert.equal(proof.ready,true,proof.reason);assert.equal(proof.patchedSHA256,hash(result.pf2e));
 assert.equal(isVerifiedNativeIWRBridge(proof,game,retained),true);
 assert.equal(isVerifiedNativeIWRBridge({...proof},game,retained),false);
 assert.equal(isVerifiedNativeIWRBridge(proof,game,Object.freeze({...retained})),false);
 const changed=Buffer.from(text.replace('actorDamage: x - O - ie','actorDamage: x - O'));
 const refused=await verifyNativeIWRBridge({game,source:changed,getNativeBridge:()=>retained,hash});
 assert.equal(refused.ready,false);assert.equal(refused.reason,'native-source-seam-mismatch');
 game.system.version='different';assert.equal(isVerifiedNativeIWRBridge(proof,game,retained),false);
});

for(const [name,before,after]of [
 ['health delta','delta: x - O - ie','delta: x - O'],
 ['IWR result','finalDamage: Math.trunc(e)','finalDamage: Math.ceil(e)'],
 ['note binding','notes: Gi.notesToHTML(s)','notes: UnknownRollNotes.notesToHTML(s)'],
])test(`unknown 8.6.0 ${name} source is refused`,()=>{
 const changed=Buffer.from(source.toString().replace(before,after));assert.notDeepEqual(changed,source);
 assert.throws(()=>bridge(changed),/native-bridge-seam-method/);
});

for(const [name,before,after]of [
 ['batch roll identity','s instanceof ln','s instanceof Object'],
 ['receiver modifier','\t\t\t\tlet o = new Y({\n\t\t\t\t\tslug: t,','\t\t\t\tlet o = new UnknownModifier({\n\t\t\t\t\tslug: t,'],
 ['API initialization','Nm.onInit();','UnknownAPI.onInit();'],
])test(`unknown 8.6.0 ${name} refuses both shared inputs`,()=>{
 const input=bridge().buffer,changed=Buffer.from(input.toString().replace(before,after)),snapshots=[Buffer.from(changed),Buffer.from(toolbelt)];
 assert.notDeepEqual(changed,input);assert.throws(()=>pair(changed),/native-.*seam/);
 assert.deepEqual(changed,snapshots[0]);assert.deepEqual(toolbelt,snapshots[1]);
});

for(const prevent of [false,true])test(`8.6.0 native callback preserves shield damage when prevention is ${prevent}`,async()=>{
 const calls=[],stop=Error('health-delta-captured'),params={damage:15,token:{name:'Target'},shieldBlockRequest:true};
 const context=vm.createContext({game:{modules:new Map([['pf2e-third-party-automation',{api:{async nativeDamageIWR(...args){calls.push(args);return prevent}}}]])},extractNotes:()=>[],applyStackingRules:()=>0,_loc:value=>value});
 const applyDamage=vm.runInContext('({'+nativeMethod(bridge().buffer.toString())+'}).applyDamage',context);
 let delta;
 const actor={hitPoints:{value:30,max:30,temp:0},heldShield:{id:'shield',_source:{system:{hp:{value:20}}}},hardness:3,synthetics:{rollNotes:{}},
  attributes:{hp:{},shield:{itemId:'shield',hardness:5,raised:true,hp:{value:20}}},isOfType:type=>type==='character',
  calculateHealthDelta(value){delta=value.delta;throw stop}};
 await assert.rejects(applyDamage.call(actor,params),error=>error===stop);
 assert.equal(calls.length,1);assert.equal(calls[0][0],actor);assert.equal(calls[0][1],params);
 assert.deepEqual({...calls[0][4]},{actorDamage:7,shieldDamage:10});assert.equal(delta,prevent?0:7);
});
