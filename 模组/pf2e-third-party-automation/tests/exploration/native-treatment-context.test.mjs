import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createNativeTreatment} from '../../scripts/exploration/native-treatment.mjs';
const defaultBundle='C:/Users/Taka/Desktop/fvtt/output/exploration-recovery-20260930/qa/runtime/Data/systems/pf2e/pf2e.mjs';
const source=await readFile(process.env.PF2E_NATIVE_BUNDLE??defaultBundle,'utf8');
const method=source.slice(source.indexOf('\tasync applyDamage({ damage:'),source.indexOf('\tasync undoDamage(e)'));assert.ok(method.startsWith('\tasync applyDamage'));
const nativeApply=Function('game','extractDamageDice','extractModifiers','applyStackingRules','extractNotes','signedInteger','applyIWR',`return ({${method}}).applyDamage`)(
 {modules:new Map()},()=>[],s=>s.modifiers??[],mods=>mods.reduce((n,m)=>n+m.modifier,0),()=>[],n=>String(n),(_a,roll,options)=>({finalDamage:options.has('self:trait:dwarf')?roll.total-2:roll.total,applications:[],persistent:[]}));
async function fixture(stage='healing',medicBonus=0){
 const hooks=new Map(),messages=new Map();let seq=0,delta,received,cloneArgs,ephemeralTest;
 const Hooks={on:(event,fn)=>{hooks.set(++seq,{event,fn});return seq},off:(_event,id)=>hooks.delete(id)};
 const roll={total:18,_evaluated:true,materials:new Set(),toJSON:()=>({formula:stage==='healing'?'{18[healing]}':'{1d8[slashing]}',total:18,evaluated:true})};
 const message={id:'D',rolls:[roll],flags:{pf2e:{context:{options:['self:trait:elf','target:trait:obsolete-patient','action:treat-wounds','skip-handling-message']}}}};messages.set(message.id,message);
 const healer={uuid:'Actor.H',alliance:'party',getRollOptions:()=>['source-domain-option'],synthetics:{ephemeralEffects:{'damage-received':{target:[async opts=>{ephemeralTest=opts.test;return {type:'effect',system:{rules:[]}}}]}}}};
 const patient={uuid:'Actor.P',id:'P',alliance:'party',getSelfRollOptions:(prefix='self')=>[`${prefix}:trait:dwarf`],getActiveTokens:()=>[{document:{uuid:'Scene.S.Token.P'}}]};
 patient.getContextualClone=(origin,effects)=>{cloneArgs={origin,effects};return {...patient,hitPoints:{value:1,max:50},heldShield:null,hardness:0,synthetics:{damageDice:{},modifiers:[{label:'Patient condition',modifier:5,critical:null,predicate:{test:opts=>opts.has('self:trait:dwarf')&&opts.has('origin:trait:elf')}}]},isOfType:()=>false,calculateHealthDelta:({delta:amount})=>{delta=amount;throw sentinel}}};
 const sentinel=new Error('stop-before-real-HP-write');
 const game={time:{worldTime:600},user:{id:'GM'},messages};
 const native=createNativeTreatment({game,Hooks,fromUuid:async uuid=>uuid===healer.uuid?healer:patient,ownerOperations:{isActivityContext:()=>true},damageGuard:{authorizeExploration:async()=>()=>{}},hpPools:{withNativeApplication:async(a,p,op)=>({result:await op()})},apply:async request=>{
  received=request;
  // Keep the real native receiver calculation; supply prepared data on the
  // current broken branch too, so RED fails on the amount, not a missing mock.
  const actor=request.recipient.hitPoints?request.recipient:{...patient,...patient.getContextualClone([],[])};
  try{await nativeApply.call(actor,request.params)}catch(error){if(error!==sentinel)throw error}
  const receipt={id:'R',author:{id:'GM'},speaker:{actor:'P'},flags:{pf2e:{appliedDamage:null,context:{type:'damage-taken',options:[request.source,request.application]}}}};messages.set('R',receipt);for(const h of hooks.values())if(h.event==='createChatMessage')h.fn(receipt);
 }});
 await native.applySavedResult({id:'A',actorUUID:healer.uuid,endsAt:600},{message,patient,stage,medicBonus,outcome:stage==='healing'?'success':'criticalFailure'},{validate(){}});
 return {delta,received,cloneArgs,ephemeralTest};
}
test('native healing evaluates the patient self condition with the original healer origin',async()=>{
 const f=await fixture();assert.equal(f.delta,-23);
 assert.ok(f.received.params.rollOptions.has('origin:trait:elf'));assert.ok(f.received.params.rollOptions.has('origin:ally'));
 assert.equal(f.received.params.rollOptions.has('self:trait:elf'),false);assert.equal(f.received.params.rollOptions.has('target:trait:obsolete-patient'),false);
 assert.ok(f.received.params.rollOptions.has('pf2e-third-party-automation:source:D:0'));
 assert.ok(f.received.params.rollOptions.has('pf2e-third-party-automation:exploration-apply:A:D:Actor.P'));
});
test('Medic contextual rules retain patient preparation and healer origin',async()=>{
 const f=await fixture('healing',5);assert.equal(f.delta,-23);assert.deepEqual(f.cloneArgs.origin,['origin:trait:elf']);assert.equal(f.cloneArgs.effects.length,1);
 assert.deepEqual(f.cloneArgs.effects[0].system.rules.map(r=>r.value),[-5,5]);
});
test('surgery preserves the DamageRoll and native patient IWR plus ephemeral target context',async()=>{
 const f=await fixture('surgery');assert.equal(f.delta,16);assert.equal(typeof f.received.params.damage,'object');assert.equal(f.received.params.skipIWR,false);
 assert.ok(f.ephemeralTest.includes('target:trait:dwarf'));assert.ok(f.ephemeralTest.includes('source-domain-option'));assert.equal(f.ephemeralTest.includes('target:trait:obsolete-patient'),false);
 assert.equal(f.cloneArgs.effects[0].system.context.origin.actor,'Actor.H');assert.equal(f.cloneArgs.effects[0].system.context.target.actor,'Actor.P');
});
