import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {readFileSync} from 'node:fs';
import {buildNativeBridge,buildSharedManualPair} from '../tools/automatic-source-patches/native.mjs';
import {verifyNativeIWRBridge,isVerifiedNativeIWRBridge} from '../scripts/native-iwr-verification.mjs';
import {NATIVE_IWR_PROFILES,NATIVE_DAMAGE_SHAPES,NATIVE_SHARED_SHAPES} from '../scripts/native-iwr-profiles.mjs';
import {sourceText,nativeMethod,batchRegion,flatRegion,stackingRegion,observerRegion,automaticDescriptor,bridgeStatement} from '../scripts/native-source-shapes.mjs';
import {native860} from './native-iwr-860-fixture.mjs';
import {native861,native861Source,hash} from './native-iwr-861-fixture.mjs';
import {receiverContext} from './native-iwr-861-receiver-fixture.mjs';

const source=native861Source();
assert.ok(process.env.TOOLBELT_CURRENT_SOURCE,'TOOLBELT_CURRENT_SOURCE is required');
const toolbelt=readFileSync(process.env.TOOLBELT_CURRENT_SOURCE);
assert.equal(hash(toolbelt),'6acd53e7a5198ef8b9f7a35caae943b4582cabe61ff57bb76572038f61e549da');
const bridge=(input=source,version='8.6.1')=>buildNativeBridge({source:input,version});
const pair=(pf2eSource=bridge().buffer,toolbeltSource=toolbelt)=>buildSharedManualPair({pf2eSource,toolbeltSource,pf2eVersion:'8.6.1',toolbeltVersion:'3.58.4'});
const damageShape=NATIVE_DAMAGE_SHAPES.find(row=>row.id==='pf2e-apply-damage:D:2');
const sharedShape=NATIVE_SHARED_SHAPES.find(row=>row.batchOriginal===native861.regionSHA256.batchRegion);

test('the 8.6.1 fixture holds the release bundle seams verbatim',{skip:!process.env.PF2E_NATIVE_861_BUNDLE},()=>{
 const text=sourceText(source);assert.equal(native861.sourceSHA256,hash(source));
 for(const [key,extract]of Object.entries({nativeMethod,batchRegion,flatRegion,stackingRegion})){assert.equal(extract(text),native861.regions[key],key);assert.equal(hash(extract(text)),native861.regionSHA256[key],key)}
 for(const [key,region]of Object.entries(native861.receiverRegions)){assert.equal(text.split(region).length,2,key);assert.equal(hash(region),native861.receiverRegionSHA256[key],key)}
 assert.equal(text.split(native861.init).length,2);
});

test('registered 8.6.1 input hashes and bindings are those of the release seams',()=>{
 const text=sourceText(source);
 assert.equal(damageShape.originalSHA256,hash(nativeMethod(text)));
 assert.equal(sharedShape.batchOriginal,hash(batchRegion(text)));assert.equal(sharedShape.flatOriginal,hash(flatRegion(text)));assert.equal(sharedShape.stacking,hash(stackingRegion(text)));
 assert.equal(sharedShape.init,native861.init);assert.equal(sharedShape.receiverRule,native861.receiverBindings.ruleElement);assert.equal(sharedShape.receiverPredicate,native861.receiverBindings.predicate);
});

test('every damage shape composes its callback from its own locals',()=>{
 const rows=Object.fromEntries(NATIVE_DAMAGE_SHAPES.map(row=>[row.id,row]));assert.equal(Object.keys(rows).length,NATIVE_DAMAGE_SHAPES.length);
 for(const id of ['pf2e-apply-damage:O:2','pf2e-apply-damage:O:1'])assert.equal(rows[id].callbackSignature,NATIVE_IWR_PROFILES['8.5.1'].callbackSignature);
 assert.equal(rows['pf2e-apply-damage:D:1'].callbackSignature,NATIVE_IWR_PROFILES['8.5.0'].callbackSignature);
 assert.equal(damageShape.callbackSignature,'?.nativeDamageIWR?.(this, arguments[0], f, r, { actorDamage: x - D - ae, shieldDamage: ne })');
});

test('audited 8.6.1 damage source builds independently of the manifest version',()=>{
 const result=bridge(source,'unlisted-release');
 assert.equal(result.status,'patch');assert.equal(result.descriptor.version,'unlisted-release');
 assert.equal(result.descriptor.sourceSHA256,hash(source));assert.equal(result.descriptor.sourceContract,'pf2e-apply-damage:D:2');
 assert.equal(hash(nativeMethod(result.buffer.toString())),'4235d69229778f923d73edfac89725d0afebe14e7f68e378fe0114b0ef71ad94');
 const again=bridge(result.buffer);assert.equal(again.status,'unchanged');assert.deepEqual(again.buffer,result.buffer);
});

test('8.6.1 and Toolbelt 3.58.4 build a byte stable shared manual pair',()=>{
 const result=pair(),again=pair(result.pf2e,result.toolbelt),text=result.pf2e.toString();
 assert.equal(result.alreadyPatched,false);assert.equal(again.alreadyPatched,true);assert.deepEqual(again.pf2e,result.pf2e);assert.deepEqual(again.toolbelt,result.toolbelt);
 assert.equal(hash(batchRegion(text)),sharedShape.batchPatched);assert.equal(hash(observerRegion(text,'pf2e').normalized),sharedShape.nativeObserver);
 assert.equal(text.split(sharedShape.init+'__nativeManualPoolBatch.install();').length,2);
 const updated=pair(bridge().buffer,result.toolbelt);assert.equal(updated.alreadyPatched,false);
 assert.deepEqual(updated.toolbelt,result.toolbelt);assert.deepEqual(pair(updated.pf2e,updated.toolbelt).pf2e,updated.pf2e);
});

test('8.6.1 receiver captures its actual RuleElement and Predicate bindings',()=>{
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
 vm.runInContext('Wn.prototype.test=()=>true;',context);
 assert.equal(context.model.isCurrent(),false);
});

test('the reflowed 8.6.1 stacking loop totals and enables modifiers as 8.6.0 does',()=>{
 const load=region=>vm.runInNewContext(region+';applyStackingRules'),before=load(native860.regions.stackingRegion),after=load(native861.regions.stackingRegion);
 const record=(modifier,type,extra={})=>({modifier,type,force:false,ignored:false,enabled:false,...extra});
 for(const records of [
  [],
  [record(2,'status'),record(1,'status'),record(-1,'status'),record(-3,'status'),record(4,'untyped'),record(-2,'untyped')],
  [record(1,'circumstance'),record(1,'circumstance'),record(3,'item'),record(-2,'item'),record(5,'potency',{ignored:true})],
  [record(3,'ability'),record(4,'ability'),record(2,'ability',{force:true}),record(1,'proficiency')],
  [record(-1,'circumstance'),record(-1,'circumstance'),record(0,'status'),record(2,'status',{ignored:true}),record(6,'ability',{ignored:true})],
 ]){
  const a=structuredClone(records),b=structuredClone(records);
  assert.equal(after(b),before(a));assert.deepEqual(b,a);
 }
});

test('a patched 8.6.1 observer cannot substitute another native binding',()=>{
 const result=pair(),changed=Buffer.from(result.pf2e.toString().replace('resolveValue:q.prototype.resolveValue','resolveValue:Y.prototype.resolveValue'));
 assert.notDeepEqual(changed,result.pf2e);
 assert.throws(()=>pair(changed,result.toolbelt),/native-shared-seam-observer/);
});

test('8.6.1 served source and retained native function receive identity bound IWR proof',async()=>{
 const result=pair(),text=result.pf2e.toString(),applyDamage=Function('return ({'+nativeMethod(text)+'}).applyDamage')();
 const retained=Object.freeze({...automaticDescriptor(bridgeStatement(text)),applyDamage}),game={system:{id:'pf2e',version:'8.6.1'}};
 const proof=await verifyNativeIWRBridge({game,source:result.pf2e,getNativeBridge:()=>retained,hash});
 assert.equal(proof.ready,true,proof.reason);assert.equal(proof.patchedSHA256,hash(result.pf2e));assert.equal(proof.applyDamageSHA256,damageShape.applyDamageSHA256);
 assert.equal(isVerifiedNativeIWRBridge(proof,game,retained),true);
 assert.equal(isVerifiedNativeIWRBridge({...proof},game,retained),false);
 assert.equal(isVerifiedNativeIWRBridge(proof,game,Object.freeze({...retained})),false);
 const changed=Buffer.from(text.replace('actorDamage: x - D - ae','actorDamage: x - D'));assert.notDeepEqual(changed,result.pf2e);
 const refused=await verifyNativeIWRBridge({game,source:changed,getNativeBridge:()=>retained,hash});
 assert.equal(refused.ready,false);assert.equal(refused.reason,'native-source-seam-mismatch');
 game.system.version='different';assert.equal(isVerifiedNativeIWRBridge(proof,game,retained),false);
});

test('an 8.6.1 bridge retained under the 8.6.0 contract id is not verified',async()=>{
 const text=bridge().buffer.toString(),applyDamage=Function('return ({'+nativeMethod(text)+'}).applyDamage')();
 const retained=Object.freeze({...automaticDescriptor(bridgeStatement(text)),sourceContract:'pf2e-apply-damage:O:2',applyDamage}),game={system:{id:'pf2e',version:'8.6.1'}};
 const refused=await verifyNativeIWRBridge({game,source:Buffer.from(text),getNativeBridge:()=>retained,hash});
 assert.equal(refused.ready,false);assert.equal(refused.reason,'native-bridge-invalid');
});

for(const [name,before,after]of [
 ['health delta','delta: x - D - ae','delta: x - D'],
 ['applied hardness','ae = x > 0 ? Math.min(x - D, ie) : 0','ae = x > 0 ? Math.min(x, ie) : 0'],
 ['IWR result','finalDamage: Math.trunc(e)','finalDamage: Math.ceil(e)'],
 ['note binding','notes: qi.notesToHTML(s)','notes: UnknownRollNotes.notesToHTML(s)'],
])test(`unknown 8.6.1 ${name} source is refused`,()=>{
 const changed=Buffer.from(source.toString().replace(before,after));assert.notDeepEqual(changed,source);
 assert.throws(()=>bridge(changed),/native-bridge-seam-method/);
});

for(const [name,before,after]of [
 ['batch roll identity','s instanceof un','s instanceof Object'],
 ['troop iterator','_t(a, (e) => e.flags.pf2e.troop?.id ?? e)','gt(a, (e) => e.flags.pf2e.troop?.id ?? e)'],
 ['receiver modifier','\t\t\t\tlet o = new Y({\n\t\t\t\t\tslug: t,','\t\t\t\tlet o = new UnknownModifier({\n\t\t\t\t\tslug: t,'],
 ['stacking rule','i.modifier < 0 ? t += applyStacking(r, i, LOWER_PENALTY)','i.modifier <= 0 ? t += applyStacking(r, i, LOWER_PENALTY)'],
 ['stacking loop layout',native861.regions.stackingRegion,native860.regions.stackingRegion],
 ['API initialization','Fm.onInit();','UnknownAPI.onInit();'],
])test(`unknown 8.6.1 ${name} refuses both shared inputs`,()=>{
 const input=bridge().buffer,changed=Buffer.from(input.toString().replace(before,after)),snapshots=[Buffer.from(changed),Buffer.from(toolbelt)];
 assert.notDeepEqual(changed,input);assert.throws(()=>pair(changed),/native-.*seam/);
 assert.deepEqual(changed,snapshots[0]);assert.deepEqual(toolbelt,snapshots[1]);
});

// Runs applyDamage up to calculateHealthDelta and reports the delta it was about to apply.
function healthDelta(methodSource,prevent){
 const calls=[],stop=Error('health-delta-captured');
 const context=vm.createContext({game:{modules:new Map([['pf2e-third-party-automation',{api:{async nativeDamageIWR(...args){calls.push(args);return prevent}}}]])},extractNotes:()=>[],applyStackingRules:()=>0,_loc:value=>value});
 const applyDamage=vm.runInContext('({'+methodSource+'}).applyDamage',context);
 return async({damage,hardness,block})=>{
  let delta;
  const params={damage,token:{name:'Target'},shieldBlockRequest:block,rollOptions:new Set(['target:fixture'])};
  const actor={hitPoints:{value:30,max:30,temp:0},heldShield:{id:'shield',_source:{system:{hp:{value:20}}}},hardness,synthetics:{rollNotes:{}},
   attributes:{hp:{},shield:{itemId:'shield',hardness:5,raised:true,hp:{value:20}}},isOfType:type=>type==='character',
   calculateHealthDelta(value){delta=value.delta;throw stop}};
  await assert.rejects(applyDamage.call(actor,params),error=>error===stop);
  return {delta,calls,actor,params};
 };
}
// The shield has hardness 5. In the middle two rows the actor's hardness exceeds
// what is left of the damage, so the applied hardness is smaller than the
// hardness value and the two must not be interchanged.
for(const [name,input,nativeDelta,shieldDamage]of [
 ['a blocked hit above both hardness values',{damage:15,hardness:3,block:true},7,10],
 ['hardness 5 against 3 damage left after a block',{damage:8,hardness:5,block:true},0,3],
 ['hardness 5 against an unblocked hit of 3',{damage:3,hardness:5,block:false},0,0],
 ['an unblocked hit above hardness',{damage:20,hardness:5,block:false},15,0],
])for(const prevent of [false,true])test(`8.6.1 callback on ${name} keeps native amounts when prevention is ${prevent}`,async()=>{
 const native=await healthDelta(nativeMethod(source.toString()))(input);
 assert.equal(native.delta,nativeDelta);assert.equal(native.calls.length,0);
 const patched=await healthDelta(nativeMethod(bridge().buffer.toString()),prevent)(input),[actor,params,result,rollOptions,amounts]=patched.calls[0];
 assert.equal(patched.calls.length,1);assert.equal(actor,patched.actor);assert.equal(params,patched.params);
 assert.equal(result.finalDamage,input.damage);assert.equal(rollOptions,patched.params.rollOptions);
 assert.deepEqual({...amounts},{actorDamage:native.delta,shieldDamage});
 assert.equal(patched.delta,prevent?0:native.delta);
});
