import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createHpPools} from '../../scripts/exploration/hp-pool.mjs';

const MODULE='pf2e-third-party-automation';
function fixture({shared=false}={}) {
  const patient={id:'Patient',uuid:'Actor.Patient',system:{attributes:{hp:{value:36,max:36,temp:0}}}};
  patient._source={system:{attributes:{hp:{value:36,max:0,temp:0}}}};
  const master={id:'Master',uuid:'Actor.Master',isOwner:true,system:structuredClone(patient.system)};
  if(shared)patient.modules={'pf2e-toolbelt':{shareData:{data:{health:true}}}};
  const source=`${MODULE}:source:Result:0`,application=`${MODULE}:exploration-apply:Activity:Result:${patient.uuid}`;
  const receipt={id:'Receipt',author:'GM',speaker:{actor:patient.id},flags:{pf2e:{
    appliedDamage:{uuid:patient.uuid,isHealing:true,shield:null,persistent:[],updates:[]},
    context:{type:'damage-taken',domains:['healing-received'],options:[source,application]}
  }}};
  const game={user:{id:'GM'},messages:new Map([['Receipt',receipt]]),modules:new Map([['pf2e-toolbelt',{active:shared}]]),settings:{get:()=>shared},
    toolbelt:{api:{shareData:{getMasterInMemory:()=>master,getSlavesInMemory:()=>[patient]}}}};
  let middleware;
  const pools=createHpPools({game,actorUpdateEvents:{addActorUpdateMiddleware:fn=>{middleware=fn}}});
  const invoke=(actor,wrapped,changes={'system.attributes.hp.value':36})=>middleware.call(actor,wrapped,changes,{});
  const run=(saved=undefined)=>pools.withNativeApplication({id:'Activity',hpPoolUUIDs:[shared?master.uuid:patient.uuid]},patient,async()=>{
    await invoke(patient,async()=>{
      if(shared)await invoke(master,async()=>saved);
      return saved;
    });
    return {receipt};
  });
  return {patient,master,receipt,game,pools,invoke,run};
}

test('full HP native healing may return undefined after its empty update diff while saving a healing receipt',async()=>{
  const f=fixture(),result=await f.run();
  assert.deepEqual(result.poolReceipt,{activityId:'Activity',actorUUID:f.patient.uuid,noChange:true,receiptId:f.receipt.id});
});

test('the exact original update promise must settle before an empty healing receipt can confirm no change',async()=>{
  const f=fixture();let finish,settled=false;
  const pending=new Promise(resolve=>finish=resolve);
  const application=f.pools.withNativeApplication({id:'Activity'},f.patient,async()=>{
    f.invoke(f.patient,()=>pending);
    return {receipt:f.receipt};
  }).then(value=>{settled=true;return value});
  await new Promise(resolve=>setImmediate(resolve));assert.equal(settled,false);
  finish(undefined);assert.equal((await application).poolReceipt.noChange,true);
});

test('the original persisted receipt source cannot change while the exact update awaits',async()=>{
  const f=fixture();let finish;const pending=new Promise(resolve=>finish=resolve);
  const application=f.pools.withNativeApplication({id:'Activity'},f.patient,async()=>{
    f.invoke(f.patient,()=>pending);return {receipt:f.receipt};
  });
  application.catch(()=>{});await new Promise(resolve=>setImmediate(resolve));
  f.receipt.flags.other={changed:true};finish(undefined);
  await assert.rejects(application,/native-hp-forward-unconfirmed/);
});

for(const [name,mutate] of [
  ['unpersisted receipt',f=>f.game.messages.clear()],
  ['different persisted receipt instance',f=>f.game.messages.set('Receipt',structuredClone(f.receipt))],
  ['wrong author',f=>f.receipt.author='Other'],
  ['wrong speaker',f=>f.receipt.speaker.actor='Other'],
  ['wrong context type',f=>f.receipt.flags.pf2e.context.type='damage-roll'],
  ['wrong received domain',f=>f.receipt.flags.pf2e.context.domains=['damage-received']],
  ['missing received domain',f=>delete f.receipt.flags.pf2e.context.domains],
  ['wrong application patient',f=>f.receipt.flags.pf2e.context.options[1]=`${MODULE}:exploration-apply:Activity:Result:Actor.Other`],
  ['wrong application activity',f=>f.receipt.flags.pf2e.context.options[1]=`${MODULE}:exploration-apply:Other:Result:Actor.Patient`],
  ['wrong source result',f=>f.receipt.flags.pf2e.context.options[0]=`${MODULE}:source:Other:0`],
  ['duplicate application marker',f=>f.receipt.flags.pf2e.context.options.push(f.receipt.flags.pf2e.context.options[1])],
  ['missing applied damage',f=>delete f.receipt.flags.pf2e.appliedDamage],
  ['null applied damage with an unconfirmed update',f=>f.receipt.flags.pf2e.appliedDamage=null],
  ['wrong native actor UUID',f=>f.receipt.flags.pf2e.appliedDamage.uuid='Actor.Other'],
  ['non-healing native receipt',f=>f.receipt.flags.pf2e.appliedDamage.isHealing=false],
  ['shield operation',f=>f.receipt.flags.pf2e.appliedDamage.shield={id:'Shield',damage:1}],
  ['persistent operation',f=>f.receipt.flags.pf2e.appliedDamage.persistent=['Condition']],
  ['missing persistent list',f=>delete f.receipt.flags.pf2e.appliedDamage.persistent],
  ['nonempty native HP delta',f=>f.receipt.flags.pf2e.appliedDamage.updates=[{path:'system.attributes.hp.value',value:-1}]],
  ['malformed native updates',f=>f.receipt.flags.pf2e.appliedDamage.updates={}],
  ['missing native updates',f=>delete f.receipt.flags.pf2e.appliedDamage.updates],
  ['reverted native receipt',f=>f.receipt.flags.pf2e.appliedDamage.isReverted=true],
  ['patient no longer at full HP',f=>f.patient.system.attributes.hp.value=35],
  ['patient maximum differs from full HP',f=>f.patient.system.attributes.hp.max=40],
  ['raw HP differs from the requested prepared HP',f=>f.patient._source.system.attributes.hp.value=35],
  ['raw HP source unavailable',f=>delete f.patient._source]
])test(`${name} keeps the native update unconfirmed`,async()=>{
  const f=fixture();mutate(f);await assert.rejects(f.run(),/native-hp-forward-unconfirmed/);
});

test('a shared pool still requires its actual saved master, even with the full HP native receipt',async()=>{
  const f=fixture({shared:true});await assert.rejects(f.run(),/native-hp-forward-unconfirmed/);
});

test('an empty native receipt without an observed patient update cannot prove this boundary',async()=>{
  const f=fixture();await assert.rejects(f.pools.withNativeApplication({id:'Activity'},f.patient,async()=>({receipt:f.receipt})),/native-hp-forward-unconfirmed/);
});

test('an unchanged receipt cannot confirm a requested HP change that lacks the saved actor',async()=>{
  const f=fixture();await assert.rejects(f.pools.withNativeApplication({id:'Activity'},f.patient,async()=>{
    await f.invoke(f.patient,async()=>undefined,{'system.attributes.hp.value':35});return {receipt:f.receipt};
  }),/native-hp-forward-unconfirmed/);
});

for(const [name,key,value] of [['value','value',35],['maximum','max',40],['temporary HP','temp',1]])
test(`prepared ${name} changing while the original update awaits prevents no-change confirmation`,async()=>{
  const f=fixture();await assert.rejects(f.pools.withNativeApplication({id:'Activity'},f.patient,async()=>{
    await f.invoke(f.patient,async()=>{await Promise.resolve();f.patient.system.attributes.hp[key]=value;return undefined});
    return {receipt:f.receipt};
  }),/native-hp-forward-unconfirmed/);
});

test('raw HP changing while the original update awaits prevents no-change confirmation',async()=>{
  const f=fixture();await assert.rejects(f.pools.withNativeApplication({id:'Activity'},f.patient,async()=>{
    await f.invoke(f.patient,async()=>{await Promise.resolve();f.patient._source.system.attributes.hp.temp=1;return undefined});
    return {receipt:f.receipt};
  }),/native-hp-forward-unconfirmed/);
});

test('a request that also changes temporary HP cannot use the full HP fallback',async()=>{
  const f=fixture();await assert.rejects(f.pools.withNativeApplication({id:'Activity'},f.patient,async()=>{
    await f.invoke(f.patient,async()=>undefined,{'system.attributes.hp.value':36,'system.attributes.hp.temp':0});return {receipt:f.receipt};
  }),/native-hp-forward-unconfirmed/);
});

test('a wrong fulfilled actor or rejected original update cannot fall back to the no-change receipt',async()=>{
  const f=fixture();await assert.rejects(f.run({uuid:'Actor.Other'}),/native-hp-forward-unconfirmed/);
  await assert.rejects(f.pools.withNativeApplication({id:'Activity'},f.patient,async()=>{
    await f.invoke(f.patient,async()=>{throw Error('original-update-rejected')});return {receipt:f.receipt};
  }),/original-update-rejected/);
});

test('a genuine saved direct actor still uses the original HP receipt path',async()=>{
  const f=fixture(),result=await f.run(f.patient);assert.equal(result.poolReceipt.noChange,undefined);
  assert.deepEqual(result.poolReceipt.fields,{'system.attributes.hp.value':36});
});
