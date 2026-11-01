import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createComputedDocuments} from '../../scripts/exploration/computed-documents.mjs';
import {TREAT_WOUNDS_IMMUNITY} from '../../scripts/salubrious-kiss-rules.mjs';
import {canonicalJSON} from '../../scripts/exploration/revision-codec.mjs';

function fixture() {
 const calls={batches:0,updates:0,items:0,wounded:0},docs=new Map();
 const set=(object,path,value)=>{const keys=path.split('.'),last=keys.pop();let target=object;for(const key of keys)target=target[key]??={};target[last]=structuredClone(value)};
 function actor(id,hp=10,focus=0) {
  const a={id,uuid:`Actor.${id}`,name:id,type:'character',flags:{},system:{attributes:{hp:{value:hp,max:50,temp:0}},resources:{focus:{value:focus,max:2}}},items:[],wounded:false,isOwner:true,
   hasCondition:slug=>slug==='wounded'&&a.wounded,
   update:async(data,options)=>{calls.updates++;for(const [key,value]of Object.entries(data))if(key!=='_id')set(a,key,value);return a},
   createEmbeddedDocuments:async(type,rows)=>{calls.items++;const saved=rows.map(row=>({...structuredClone(row),uuid:`${a.uuid}.Item.${row._id}`}));a.items.push(...saved);return saved},
   decreaseCondition:async()=>{calls.wounded++;a.wounded=false}
  };docs.set(a.uuid,a);return a;
 }
 const h=actor('H',50,1),p=actor('P',10),s=actor('S',10);
 const game={user:{id:'G'},time:{worldTime:600}};
 const fromUuid=async uuid=>uuid===TREAT_WOUNDS_IMMUNITY?{toObject:()=>({name:'Immune',type:'effect',system:{duration:{}},flags:{}})}:docs.get(uuid);
 const Actor={updateDocuments:async rows=>{calls.batches++;return Promise.all(rows.map(row=>docs.get(`Actor.${row._id}`).update(row)))}};
 const adapter=createComputedDocuments({game,fromUuid,Actor});
 const snapshot=(a,pool=a.uuid)=>({actorUUID:a.uuid,name:a.name,hp:structuredClone((docs.get(pool)??a).system.attributes.hp),focus:structuredClone(a.system.resources.focus),wounded:a.wounded,pool:{poolUUID:pool,ready:true}});
 return {calls,docs,h,p,s,game,adapter,snapshot};
}

test('computed documents combine HP and focus updates, write shared master once, and do not use chat APIs',async()=>{
 const f=fixture(),before=[f.snapshot(f.h),f.snapshot(f.p),f.snapshot(f.s,f.p.uuid)],after=structuredClone(before);
 after[0].focus.value=0;after[1].hp.value=30;after[2].hp.value=30;
 const plan=await f.adapter.prepare({id:'B',to:600,beforeActors:before,afterActors:after,events:[]});
 const receipt=await f.adapter.applyActors(plan);
 assert.equal(f.h.system.resources.focus.value,0);assert.equal(f.p.system.attributes.hp.value,30);assert.equal(f.s.system.attributes.hp.value,10);
 assert.equal(f.calls.batches,1);assert.equal(f.calls.updates,2);assert.equal(receipt.actors.length,2);
 await assert.rejects(f.adapter.applyActors(plan),/already-applied/);
 assert.equal(f.calls.updates,2);
});

test('computed document preflight refuses changed HP before writing any actor',async()=>{
 const f=fixture(),before=[f.snapshot(f.p)],after=structuredClone(before);after[0].hp.value=30;
 const plan=await f.adapter.prepare({id:'B',to:600,beforeActors:before,afterActors:after,events:[]});
 f.p.system.attributes.hp.value=11;
 await assert.rejects(f.adapter.applyActors(plan),/changed/);assert.equal(f.calls.updates,0);
});

test('only the last unexpired immunity is created and successful treatment clears wounded',async()=>{
 const f=fixture();f.p.wounded=true;const before=[f.snapshot(f.p)],after=structuredClone(before);after[0].hp.value=40;after[0].wounded=false;
 const plan=await f.adapter.prepare({id:'B',to:600,beforeActors:before,afterActors:after,events:[
  {providerId:'treat-wounds',patientUUID:f.p.uuid,startedAt:0,endsAt:600,cooldownExpiresAt:600},
  {providerId:'treat-wounds',actorUUID:f.h.uuid,patientUUID:f.p.uuid,startedAt:0,endsAt:600,cooldownExpiresAt:3600}
 ]});
 await f.adapter.applyActors(plan);const receipt=await f.adapter.applyItems(plan);
 assert.equal(f.p.items.length,1);assert.equal(f.p.items[0].system.duration.value,50);
 assert.equal(f.p.items[0].flags['pf2e-third-party-automation'].salubriousKiss.expiresAt,3600);
 assert.equal(f.p.wounded,false);assert.equal(f.calls.wounded,1);assert.equal(receipt.items.length,1);
 await assert.rejects(f.adapter.applyItems(plan),/already-applied/);assert.equal(f.p.items.length,1);
});

test('damage stages preserve loss before healing and carry damageTaken to the native Actor update',async()=>{
 const f=fixture(),before=[f.snapshot(f.p)],after=structuredClone(before);after[0].hp.value=20;
 const observed=[];const update=f.p.update;f.p.update=async(data,options)=>{observed.push({hp:data['system.attributes.hp.value'],damageTaken:options?.damageTaken});return update(data,options)};
 const plan=await f.adapter.prepare({id:'B',to:600,beforeActors:before,afterActors:after,events:[{patientUUID:f.p.uuid,poolUUID:f.p.uuid,hpStages:[{before:10,after:5,damageTaken:5},{before:5,after:20,damageTaken:0}]}]});
 await f.adapter.applyActors(plan);
 assert.deepEqual(observed.filter(row=>row.hp!==undefined),[{hp:5,damageTaken:5},{hp:20,damageTaken:0}]);
});

test('unknown document acknowledgement can be read back but never automatically applied again',async()=>{
 const f=fixture(),before=[f.snapshot(f.p)],after=structuredClone(before);after[0].hp.value=30;
 const plan=await f.adapter.prepare({id:'B',to:600,beforeActors:before,afterActors:after,events:[]});
 const update=f.p.update;f.p.update=async(...args)=>{await update(...args);throw Error('connection-lost')};
 await assert.rejects(f.adapter.applyActors(plan),/connection-lost/);
 const readback=await f.adapter.inspect(plan);assert.equal(readback.actors[0].state,'confirmed');
 await assert.rejects(f.adapter.applyActors(plan),/already-applied/);assert.equal(f.calls.updates,1);
});

test('batch marker survives Foundry expansion of dotted keys in nested flags',async()=>{
 const f=fixture(),before=[f.snapshot(f.p)],after=structuredClone(before);after[0].hp.value=30;
 const update=f.p.update;
 f.p.update=async changes=>{
  const copy=structuredClone(changes),receipt=copy['flags.pf2e-third-party-automation.computedRecoveryBatch'];
  if(receipt?.after){const expanded={};for(const [path,value]of Object.entries(receipt.after)){let object=expanded;const keys=path.split('.'),last=keys.pop();for(const key of keys)object=object[key]??={};object[last]=value}receipt.after=expanded}
  return update(copy);
 };
 const plan=await f.adapter.prepare({id:'B',to:600,beforeActors:before,afterActors:after,events:[]});
 await f.adapter.applyActors(plan);assert.equal((await f.adapter.inspect(plan)).actors[0].state,'confirmed');
});

test('persisted batch readback ignores JSON key ordering after ledger reconstruction',async()=>{
 const f=fixture(),before=[f.snapshot(f.p)],after=structuredClone(before);after[0].hp.value=30;
 const plan=await f.adapter.prepare({id:'B',to:600,beforeActors:before,afterActors:after,events:[]});await f.adapter.applyActors(plan);
 const restored=JSON.parse(canonicalJSON(plan));assert.equal((await f.adapter.inspect(restored)).actors[0].state,'confirmed');
});

test('lost acknowledgement during damage has an unknown readback and never repeats the stage',async()=>{
 const f=fixture(),before=[f.snapshot(f.p)],after=structuredClone(before);after[0].hp.value=20;
 const plan=await f.adapter.prepare({id:'B',to:600,beforeActors:before,afterActors:after,events:[{patientUUID:f.p.uuid,poolUUID:f.p.uuid,hpStages:[{before:10,after:5,damageTaken:5},{before:5,after:20,damageTaken:-15}]}]});
 const update=f.p.update;f.p.update=async(...args)=>{await update(...args);throw Error('lost-stage-ack')};
 await assert.rejects(f.adapter.applyActors(plan),/lost-stage-ack/);
 const result=await f.adapter.inspect(plan);assert.equal(result.actors[0].state,'unknown');assert.equal(result.actors[0].stage,0);
 await assert.rejects(f.adapter.applyActors(plan),/already-applied/);assert.equal(f.calls.updates,1);
});

test('malformed persisted marker is unknown rather than an inspection failure',async()=>{
 const f=fixture(),before=[f.snapshot(f.p)],after=structuredClone(before);after[0].hp.value=30;
 const plan=await f.adapter.prepare({id:'B',to:600,beforeActors:before,afterActors:after,events:[]});
 f.p.flags['pf2e-third-party-automation']={computedRecoveryBatch:{id:'B',fields:'not JSON'}};
 assert.equal((await f.adapter.inspect(plan)).actors[0].state,'unknown');
});

for(const scenario of [{type:'character',mode:'both',damage:10,expected:true},{type:'character',mode:'both',damage:4,expected:false},{type:'npc',mode:'npcsOnly',damage:4,expected:true}]){
 test(`zero HP preserves native defeated automation for ${scenario.type}, ${scenario.damage} damage`,async()=>{
  const f=fixture();f.p.type=scenario.type;f.p.system.attributes.hp={value:3,max:5,temp:0};
  Object.defineProperty(f.p,'isDead',{get:()=>f.p.system.attributes.hp.value===0});
  f.game.settings={get:()=>scenario.mode};let defeatedCalls=0;
  const combatant={uuid:'Combat.C.Combatant.P',isDefeated:false,toggleDefeated:async options=>{assert.equal(options.overlayIcon,true);defeatedCalls++;combatant.isDefeated=true}};
  f.docs.set(combatant.uuid,combatant);f.p.getActiveTokens=()=>[{combatant}];
  const before=[f.snapshot(f.p)],after=structuredClone(before);after[0].hp.value=0;
  const plan=await f.adapter.prepare({id:'B',to:600,beforeActors:before,afterActors:after,events:[{patientUUID:f.p.uuid,poolUUID:f.p.uuid,hpStages:[{before:3,after:0,damageTaken:scenario.damage}]}]});
  await f.adapter.applyActors(plan);assert.equal(defeatedCalls,Number(scenario.expected));assert.equal((await f.adapter.inspect(plan)).actors[0].state,'confirmed');
 });
}

test('unknown defeated acknowledgement leaves the actor stage unknown and cannot replay',async()=>{
 const f=fixture();f.p.type='npc';f.game.settings={get:()=> 'npcsOnly'};let calls=0;
 Object.defineProperty(f.p,'isDead',{get:()=>f.p.system.attributes.hp.value===0});
 const combatant={uuid:'Combat.C.Combatant.P',isDefeated:false,toggleDefeated:async()=>{calls++;combatant.isDefeated=true;throw Error('lost-defeated-ack')}};
 f.docs.set(combatant.uuid,combatant);f.p.getActiveTokens=()=>[{combatant}];
 const before=[f.snapshot(f.p)],after=structuredClone(before);after[0].hp.value=0;
 const plan=await f.adapter.prepare({id:'B',to:600,beforeActors:before,afterActors:after,events:[{patientUUID:f.p.uuid,poolUUID:f.p.uuid,hpStages:[{before:10,after:0,damageTaken:10}]}]});
 await assert.rejects(f.adapter.applyActors(plan),/lost-defeated-ack/);assert.equal((await f.adapter.inspect(plan)).actors[0].state,'unknown');
 await assert.rejects(f.adapter.applyActors(plan),/already-applied/);assert.equal(calls,1);
});
