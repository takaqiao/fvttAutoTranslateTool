import {test} from 'node:test';import assert from 'node:assert/strict';
import {classifyResult,createNativeTreatment,medicStackingEffect} from '../../scripts/exploration/native-treatment.mjs';
import {createSalubriousCheckScope} from '../../scripts/salubrious-kiss-check-scope.mjs';
import {readFile} from 'node:fs/promises';
test('verified native context without action field enters once and copied nonce cannot enter',async()=>{
  const fixture=JSON.parse(await readFile(new URL('./fixtures/native-treatment-contract.json',import.meta.url)));
  const ctx={validate:()=>{}},healer={},activity={id:'A1',options:{skill:'medicine',rank:'trained'}};
  const scope=createSalubriousCheckScope({game:{},isExplorationContext:c=>c===ctx});
  const context={...fixture.checkContext,actor:healer,options:new Set([...fixture.checkContext.options,'exploration-activity:A1'])};
  await scope.runExploration({ctx,healer,activity},()=>scope.interceptCheck(async(c,actual)=>{assert.equal(actual.skipDialog,true);return 'rolled'},{},context));
  await assert.rejects(scope.runExploration({ctx:{...ctx},healer,activity},async()=>{}),/context/);
});
test('Medic baked bonus becomes a typed native recipient modifier',()=>{
  const effect=medicStackingEffect({medicBonus:5,sourceActorUUID:'Actor.H'});
  assert.deepEqual(effect.system.rules.map(r=>[r.type,r.value]),[['untyped',-5],['circumstance',5]]);
});
const dice=(formula,total)=>({_evaluated:true,total,toJSON:()=>({formula,total,evaluated:true}),instances:[{type:formula.includes('slashing')?'slashing':'untyped',kinds:new Set(formula.includes('healing')?['healing']:['damage'])}]});
test('source stages keep Risky Surgery failure damage and separate critical failure',()=>{
  assert.equal(classifyResult(dice('{1d8[slashing]}',5),'failure'),'surgery');
  assert.equal(classifyResult(dice('{(1d8)}',3),'criticalFailure'),'failure-damage');
  assert.throws(()=>classifyResult(dice('{(1d8)}',3),'success'),/stage/);
});
function fixture(outcome='criticalSuccess',risky=true){
  const hooks=new Map();let seq=0;const Hooks={on:(name,fn)=>{const id=++seq;hooks.set(id,{name,fn});return id},off:(name,id)=>hooks.delete(id)};
  const fire=(name,...args)=>{for(const h of hooks.values())if(h.name===name)h.fn(...args)};
  const messages=new Map(),healer={uuid:'Actor.H',id:'H',items:[{type:'feat',slug:'risky-surgery'}]},patient={uuid:'Actor.P',id:'P',isOwner:true,createEmbeddedDocuments:async(type,list)=>{assert.equal(list[0].system.duration.value,50);return [{uuid:'Actor.P.Item.Immunity'}]}};
  let release;const deferred=new Promise(r=>release=r);let called=0;const stages=[];
  const game={user:{id:'G'},time:{worldTime:600},messages,pf2e:{actions:{get:()=>({use:async options=>{
    called++;const marker=options.rollOptions[0];const check={id:'C',actor:healer,author:{id:'G'},flags:{pf2e:{context:{action:'treat-wounds',options:[marker],outcome},modifiers:risky?[{slug:'risky-surgery',enabled:true}]:[]}},rolls:[{total:20}]};messages.set('C',check);
    deferred.then(()=>{let i=0;for(const roll of [risky?dice('{1d8[slashing]}',5):null,outcome==='failure'?null:dice(outcome==='criticalFailure'?'{(1d8)}':'{(4d8)[healing]}',outcome==='criticalFailure'?3:19)].filter(Boolean)){
      const m={id:`D${++i}`,actor:healer,author:{id:'G'},flags:{pf2e:{origin:{messageId:'C'},context:{options:[marker],outcome}}},rolls:[roll]};fire('preCreateChatMessage',m,m);messages.set(m.id,m);fire('createChatMessage',m);
    }});return [{actor:healer,message:check,outcome,roll:check.rolls[0]}];
  }})}}};
  game.pf2e.DamageRoll=class {constructor(formula){this.formula=formula}async evaluate(){Object.assign(this,dice(this.formula,5));return this}};
  const native=createNativeTreatment({game,Hooks,createMessage:async data=>{const m={...data,id:'CompatCut',actor:healer,rolls:[dice('{1d8[slashing]}',5)]};fire('preCreateChatMessage',m,m);messages.set(m.id,m);fire('createChatMessage',m);return m},fromUuid:async id=>id.startsWith('Compendium.')?{toObject:()=>({type:'effect',system:{}})}:id===healer.uuid?healer:patient,ownerOperations:{isActivityContext:(ctx,id)=>ctx.id===id},checkScope:{runExploration:async(s,op)=>op()},damageGuard:{authorizeExploration:async()=>()=>{}},hpPools:{withNativeApplication:async(a,p,op)=>({result:await op()})},apply:async request=>{
    stages.push(request.stage);const m={id:`R${stages.length}`,author:{id:'G'},speaker:{actor:'P'},flags:{pf2e:{context:{type:'damage-taken',options:[request.source,request.application]},appliedDamage:null}}};messages.set(m.id,m);fire('createChatMessage',m);return patient
  },timeoutMs:150});
  const activity={id:'A',actorUUID:'Actor.H',patientUUIDs:['Actor.P'],startedAt:0,endsAt:600,options:{skill:'medicine',rank:'trained',riskySurgery:risky}};
  return {native,activity,release,stages,get called(){return called},game};
}
test('native use return waits delayed persistent results; surgery is applied first',async()=>{
  const f=fixture();let done=false;const result=f.native.run(f.activity,{id:'A'}).then(r=>{done=true;return r});await new Promise(r=>setImmediate(r));assert.equal(done,false);f.release();
  const output=await result;assert.equal(output.status,'confirmed');assert.deepEqual(f.stages,['surgery','healing']);assert.equal(output.rolledHealing,19);assert.equal(f.called,1);
});
test('Risky failure still applies injury and critical failure applies both damages',async()=>{
  for(const outcome of ['failure','criticalFailure']){const f=fixture(outcome);const p=f.native.run(f.activity,{id:'A'});f.release();assert.equal((await p).status,'confirmed');assert.deepEqual(f.stages,outcome==='failure'?['surgery']:['surgery','failure-damage'])}
});
test('copied public context and unknown child outputs never authorize native dice',async()=>{
  const f=fixture();await assert.rejects(f.native.run(f.activity,{id:'forged'}),/context/);assert.equal(f.called,0);
});
test('suppressed circumstance modifier still requires exactly one genuine surgical cut',async()=>{
  const f=fixture('success',false);f.activity.options.riskySurgery=true;
  const p=f.native.run(f.activity,{id:'A'});f.release();const result=await p;
  assert.deepEqual(f.stages,['surgery','healing']);assert.equal(result.proof.resultIds.includes('CompatCut'),true);
});
test('fifty minute extension applies original rolled amount once without new check or cut',async()=>{
  const f=fixture();const p=f.native.run(f.activity,{id:'A'});f.release();const original={...f.activity,...await p,state:'confirmed'};
  f.game.time.worldTime=3600;
  const extension={...f.activity,id:'E',startedAt:600,endsAt:3600};const result=await f.native.extend(extension,original,{id:'E'});
  assert.equal(result.status,'confirmed');assert.equal(f.called,1);assert.deepEqual(f.stages,['surgery','healing','healing']);
  await assert.rejects(f.native.extend(extension,original,{id:'E'}),/already/);
});
test('actual PF2e stacking function prevents Medic and Robust Health duplicate circumstance bonus',async()=>{
  const source=await readFile(process.env.PF2E_NATIVE_BUNDLE??'C:/Users/Taka/Desktop/fvtt/tmp/fortress-gap-audit-20260925/code/systems/pf2e/pf2e.mjs','utf8');
  const start=source.indexOf('var HIGHER_BONUS ='),end=source.indexOf('var StatisticModifier =',start);assert.ok(start>0&&end>start);
  const nativeStack=Function(`${source.slice(start,end)};return applyStackingRules`)();
  for(const [robust,expected] of [[9,28],[3,24]]){
    const rules=medicStackingEffect({medicBonus:5,sourceActorUUID:'Actor.H'}).system.rules;
    const mods=[...rules.map(r=>({type:r.type,modifier:r.value,ignored:false})),{type:'circumstance',modifier:robust,ignored:false}];
    assert.equal(24+nativeStack(mods),expected);
  }
});
