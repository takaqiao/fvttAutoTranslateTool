import {test} from 'node:test';import assert from 'node:assert/strict';
import {createManualEvents,workbenchFacts,observeWorkbenchCommand} from '../../scripts/exploration/manual-events.mjs';
test('manual observations never roll/apply, deduplicate sources and leave incomplete failure',async()=>{
 const records=[];const recorder=createManualEvents({game:{time:{worldTime:100}},ledger:{insertActivity:async a=>records.push(a)},isAuthority:()=>true,sessionId:()=> 'S'});
 const event={id:'M',actorUUID:'Actor.H',patientUUIDs:['Actor.P'],kind:'treatment',source:{type:'workbench',hashVerified:true},checkIds:['C'],resultIds:['M'],durationSeconds:600,sourceDegree:1,rolledHealing:undefined};
 await recorder.observe(event);await recorder.observe(event);assert.equal(records.length,1);assert.equal(records[0].state,'awaiting-evidence');assert.equal(records[0].options.rolledHealing,null);
 await recorder.observe({id:'Q',kind:'medicine-check',actorUUID:'Actor.H',patientUUIDs:[]});assert.equal(records.length,2);assert.equal(records[1].state,'awaiting-evidence');
});
test('Workbench source degree and actual critical healing remain separate',()=>{
 const facts=workbenchFacts({flags:{treat_wounds_battle_medicine:{id:'T',healerId:'H',dos:2,healing:19}},rolls:[{formula:'(4d8)[healing]',total:19}]});assert.equal(facts.sourceDegree,2);assert.equal(facts.effectiveOutcome,'criticalSuccess');assert.equal(facts.rolledHealing,19);
});
test('verified Workbench instrumentation binds each callback target, unknown source is rejected',()=>{
 const command='const rollTreatWounds = async ({ target, bmtw, skillUsed, isRiskySurgery }) => {\n const dc = {};\n};';
 assert.throws(()=>observeWorkbenchCommand(command,'unknown'),/unverified/);
 const observed=observeWorkbenchCommand(command,'b3bac907654522fda62b80da182fc253f7147493a465a82081219fb2a0f1308f');assert.match(observed,/explorationManualTarget/);assert.match(observed,/target, bmtw, skillUsed/);assert.match(observed,/skillUsed = explorationScope.skill/);
});
