import {test} from 'node:test';import assert from 'node:assert/strict';import {chooseNext,expectedHealingPerMinute,recoveryProposals} from '../../scripts/exploration/policy.mjs';
const p=(h,t,duration=600)=>({providerId:'treat-wounds',actorUUID:h,patientUUIDs:[t],hpPoolUUIDs:[t],durationSeconds:duration,earliestStart:0,actorExclusive:true,patientTreatmentExclusive:true,expectedNetHealing:9,options:{}});
test('two healers select distinct patients and pools concurrently',()=>{const proposals=['H1','H2'].flatMap(h=>['P1','P2','P3'].map(t=>p(h,t)));const next=chooseNext({snapshot:{},proposals,now:0,session:{budgetEndsAt:7200}});assert.equal(next.activities.length,2);assert.equal(new Set(next.activities.flatMap(a=>a.patientUUIDs)).size,2);assert.equal(next.checkpointAt,600)});
test('pool conflict, future cooldown, long active activity and budget constrain choices',()=>{
 assert.equal(chooseNext({snapshot:{},proposals:[{...p('H1','P1'),hpPoolUUIDs:['POOL']},{...p('H2','P2'),hpPoolUUIDs:['POOL']}],now:0,session:{budgetEndsAt:7200}}).activities.length,1);
 assert.equal(chooseNext({snapshot:{},proposals:[{...p('H','P'),earliestStart:3600}],now:600,session:{budgetEndsAt:7200}}).checkpointAt,3600);
 assert.equal(chooseNext({snapshot:{},proposals:[p('H','P')],now:7100,session:{budgetEndsAt:7200}}).reason,'budget');
});
test('expectation enumerates faces without consuming randomness',()=>{let faces=0;assert.equal(expectedHealingPerMinute({outcomeForFace:f=>{faces++;return f>10?2:1},meanForOutcome:d=>d===2?18:0,expectedDamage:4.5,durationSeconds:600}),0.45);assert.equal(faces,20)});
test('owned Natural Medicine permits a Nature-trained healer without Medicine',()=>{const h={actorUUID:'H',medicine:{rank:0},nature:{rank:2},slugs:['natural-medicine','risky-surgery'],assuranceSkills:[],items:[],focus:{value:0,max:0},pool:{poolUUID:'H',ready:true},hp:{value:10,max:10}},patient={...h,actorUUID:'P',slugs:[],nature:{rank:0},pool:{poolUUID:'P',ready:true},hp:{value:1,max:10},modeOfBeing:'living'};const p=recoveryProposals({actors:[h,patient],activities:[],session:{goalsByPool:[{poolUUID:'P',targetHP:10}],riskySurgery:true},now:0,providerIds:['treat-wounds']});assert.equal(p.length,1);assert.equal(p[0].options.skill,'nature');assert.equal(p[0].options.riskySurgery,false)});
test('a confirmed checkpoint treatment keeps its start-based cooldown even when the effect is no longer listed',()=>{
 const healer={actorUUID:'H',medicine:{rank:1},slugs:[],assuranceSkills:[],items:[],focus:{value:0,max:0},pool:{poolUUID:'H',ready:true},hp:{value:20,max:20}},patient={...healer,actorUUID:'P',medicine:{rank:0},modeOfBeing:'living',pool:{poolUUID:'P',ready:true},hp:{value:1,max:20}};
 const activity={providerId:'manual',state:'confirmed',startedAt:0,patientUUIDs:['P'],temporalSource:{type:'checkpoint-reservation'},options:{continualRecovery:false}};
 const proposals=recoveryProposals({actors:[healer,patient],activities:[activity],session:{goalsByPool:[{poolUUID:'P',targetHP:20}]},now:600,providerIds:['treat-wounds']});assert.equal(proposals[0].earliestStart,3600);
 activity.options.continualRecovery=true;assert.equal(recoveryProposals({actors:[healer,patient],activities:[activity],session:{goalsByPool:[{poolUUID:'P',targetHP:20}]},now:600,providerIds:['treat-wounds']})[0].earliestStart,600);
});

function woundedFixture(){
 const healer={actorUUID:'Actor.H',medicine:{rank:2},nature:{rank:2},slugs:['natural-medicine','risky-surgery'],riskySurgery:true,assuranceSkills:['medicine','nature'],items:[{uuid:'Actor.H.Item.LOH',sourceId:'Compendium.pf2e.spells-srd.Item.zNN9212H2FGfM7VS'}],focus:{value:0,max:1},pool:{poolUUID:'Actor.H',ready:true},hp:{value:20,max:20},threePecks:true,wardCapacity:2};
 const patient={...healer,actorUUID:'Actor.P',medicine:{rank:0},nature:{rank:0},slugs:[],items:[],threePecks:false,assuranceSkills:[],modeOfBeing:'living',pool:{poolUUID:'Actor.P',ready:true},focus:{value:1,max:1},wounded:true,vitalityHealingReady:true};
 const session={goalsByPool:[{poolUUID:'Actor.H',targetHP:20},{poolUUID:'Actor.P',targetHP:20}],recoveryGoals:{requireNoWounded:true},treatmentRank:'expert',useAssurance:true,riskySurgery:true};
 return {healer,patient,session};
}

test('full HP wounded recovery admits Medicine and Nature TW with the saved options and zero HP deficit',()=>{
 const {healer,patient,session}=woundedFixture();healer.focus.value=1;const proposals=recoveryProposals({actors:[healer,patient],activities:[],session,now:0,providerIds:['treat-wounds','focus-healing','refocus']});
 assert.equal(proposals.length,2);assert.ok(proposals.every(p=>p.providerId==='treat-wounds'&&p.patientUUIDs[0]==='Actor.P'&&p.expectedNetHealing===0&&p.options.rank==='expert'&&p.options.assurance===true));
 assert.deepEqual(proposals.map(p=>[p.options.skill,p.options.riskySurgery]),[['medicine',true],['nature',false]]);
 session.recoveryGoals.requireNoWounded=false;assert.deepEqual(recoveryProposals({actors:[healer,patient],activities:[],session,now:0,providerIds:['treat-wounds','focus-healing','refocus']}),[]);
});

test('wounded-only TW keeps both native immunity and the saved start-based cooldown',()=>{
 const {healer,patient,session}=woundedFixture();patient.cooldownExpiresAt=2400;
 const activity={providerId:'treat-wounds',state:'confirmed',startedAt:0,patientUUIDs:['Actor.P'],options:{continualRecovery:false}};
 const proposals=()=>recoveryProposals({actors:[healer,patient],activities:[activity],session,now:600,providerIds:['treat-wounds']});
 assert.ok(proposals().length);assert.ok(proposals().every(p=>p.earliestStart===3600));activity.options.continualRecovery=true;assert.ok(proposals().every(p=>p.earliestStart===2400));
 patient.cooldownExpiresAt=null;assert.ok(proposals().every(p=>p.earliestStart===600));
});

test('Ward leaves two full HP wounded patients in one shared pool as independent single-patient TW candidates',()=>{
 const {healer,patient,session}=woundedFixture(),other={...patient,actorUUID:'Actor.Q',pool:{poolUUID:'Actor.P',ready:true}};
 const proposals=recoveryProposals({actors:[healer,patient,other],activities:[],session,now:0,providerIds:['treat-wounds']});
 assert.ok(proposals.length);assert.ok(proposals.every(p=>p.patientUUIDs.length===1));assert.deepEqual(new Set(proposals.flatMap(p=>p.patientUUIDs)),new Set(['Actor.P','Actor.Q']));
 const next=chooseNext({snapshot:{},proposals,session:{...session,budgetEndsAt:7200},now:0});assert.equal(next.activities.length,1);assert.equal(next.checkpointAt,600);
});
