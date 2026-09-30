import {test} from 'node:test';
import assert from 'node:assert/strict';
import {wardCapacity,cooldown,earliestTreatmentStart,createCapabilities} from '../../scripts/exploration/capabilities.mjs';
import {readFile} from 'node:fs/promises';
test('current roster prerequisites and existing cooldown remain explicit',async()=>{
  const f=JSON.parse(await readFile(new URL('./fixtures/roster-capabilities.json',import.meta.url)));
  assert.equal(wardCapacity(f.unqualifiedWard),1);
  assert.equal(earliestTreatmentStart({now:f.existingImmunity.now,existingExpiresAt:f.existingImmunity.expiresAt})-f.existingImmunity.now,183);
});
test('Ward Medic uses Medicine, patient cooldown starts at treatment start',()=>{
  assert.deepEqual([1,2,3,4].map(medicineRank=>wardCapacity({wardMedic:true,medicineRank})),[1,2,4,8]);
  assert.deepEqual(cooldown({startedAt:0,finishedAt:600,continualRecovery:false}),{expiresAt:3600,remainingSeconds:3000});
  assert.deepEqual(cooldown({startedAt:0,finishedAt:600,continualRecovery:true}),{expiresAt:600,remainingSeconds:0});
  assert.equal(earliestTreatmentStart({now:67994,existingExpiresAt:68177}),68177);
});
test('legacy compendium source UUID still enforces a real patient immunity',async()=>{
 const actor={uuid:'Actor.P',system:{attributes:{hp:{value:10,max:20}}},items:[{type:'effect',uuid:'Actor.P.Item.I',sourceId:'Compendium.pf2e.feat-effects.Lb4q2bBAgxamtix5',isExpired:false,system:{start:{value:1000},duration:{value:1,unit:'hours'}}}]};
 const c=createCapabilities({game:{time:{worldTime:4417}},fromUuid:async()=>actor,hpPools:{discover:()=>({poolUUID:'Actor.P',ready:true})}});assert.equal((await c.discover(actor.uuid)).cooldownExpiresAt,4600);
});
test('prepared Medic grant and Assurance belonging to another skill',async()=>{
  const actor={uuid:'Actor.A',name:'A',system:{attributes:{hp:{value:9,max:40}},resources:{focus:{value:0,max:2}}},items:[
    {type:'feat',uuid:'Actor.A.Item.M',slug:'medic',system:{slug:'medic'}},
    {type:'feat',slug:'assurance',system:{slug:'assurance'},flags:{pf2e:{rulesSelections:{assurance:'occultism'}}}},
    {type:'feat',slug:'ward-medic',system:{slug:'ward-medic'}}
  ],getStatistic:s=>({rank:s==='medicine'?2:3,mod:10})};
  const c=createCapabilities({game:{time:{worldTime:600},actors:{party:{members:[actor]}},modules:new Map([['patreon-v3',{active:true}]]),settings:{get:()=>true}},fromUuid:async()=>actor,hpPools:{discover:a=>({poolUUID:a.uuid,ready:true})}});
  const result=await c.discover(actor.uuid);assert.equal(result.medicine.rank,2);assert.equal(result.wardCapacity,2);assert.equal(result.assuranceSkills.includes('medicine'),false);assert.deepEqual(result.assuranceSkills,['occultism']);
  actor.rules=[{key:'FastHealing',test:()=>true}];assert.equal((await c.activePassiveRules())[0].passing,true);
  assert.equal(result.cooldownExpiresAt,0,'current world time is not a real patient immunity');
});
test('passive time readiness follows enabled Patreon and all parties, including an unselected member',async()=>{
 const actor={uuid:'Actor.A',rules:[{key:'FastHealing',test:()=>true}]};let enabled=false;
 const actors=[{type:'party',members:[actor]},{type:'party',members:[actor]}];actors.party={members:[]};
 const c=createCapabilities({game:{actors,modules:new Map([['patreon-v3',{active:true}]]),settings:{get:()=>enabled}},fromUuid:async()=>null,hpPools:{}});
 assert.deepEqual(await c.activePassiveRules(),[]);enabled=true;
 assert.deepEqual((await c.activePassiveRules()).map(r=>r.actorUUID),['Actor.A']);
});
