import {test} from 'node:test';import assert from 'node:assert/strict';import {chooseNext,expectedHealingPerMinute} from '../../scripts/exploration/policy.mjs';
const p=(h,t,duration=600)=>({providerId:'treat-wounds',actorUUID:h,patientUUIDs:[t],hpPoolUUIDs:[t],durationSeconds:duration,earliestStart:0,actorExclusive:true,patientTreatmentExclusive:true,expectedNetHealing:9,options:{}});
test('two healers select distinct patients and pools concurrently',()=>{const proposals=['H1','H2'].flatMap(h=>['P1','P2','P3'].map(t=>p(h,t)));const next=chooseNext({snapshot:{},proposals,now:0,session:{budgetEndsAt:7200}});assert.equal(next.activities.length,2);assert.equal(new Set(next.activities.flatMap(a=>a.patientUUIDs)).size,2);assert.equal(next.checkpointAt,600)});
test('pool conflict, future cooldown, long active activity and budget constrain choices',()=>{
 assert.equal(chooseNext({snapshot:{},proposals:[{...p('H1','P1'),hpPoolUUIDs:['POOL']},{...p('H2','P2'),hpPoolUUIDs:['POOL']}],now:0,session:{budgetEndsAt:7200}}).activities.length,1);
 assert.equal(chooseNext({snapshot:{},proposals:[{...p('H','P'),earliestStart:3600}],now:600,session:{budgetEndsAt:7200}}).checkpointAt,3600);
 assert.equal(chooseNext({snapshot:{},proposals:[p('H','P')],now:7100,session:{budgetEndsAt:7200}}).reason,'budget');
});
test('expectation enumerates faces without consuming randomness',()=>{let faces=0;assert.equal(expectedHealingPerMinute({outcomeForFace:f=>{faces++;return f>10?2:1},meanForOutcome:d=>d===2?18:0,expectedDamage:4.5,durationSeconds:600}),0.45);assert.equal(faces,20)});
