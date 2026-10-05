import {test} from 'node:test';import assert from 'node:assert/strict';
import {createManualEvents,workbenchFacts,observeWorkbenchCommand,registerWorkbenchObservation} from '../../scripts/exploration/manual-events.mjs';
import {readFile} from 'node:fs/promises';
import {recordingLedger} from './manual-evidence-fixture.mjs';
test('manual observations never roll/apply, deduplicate sources and leave incomplete failure',async()=>{
 const records=[];const recorder=createManualEvents({game:{time:{worldTime:100}},ledger:recordingLedger(records),isAuthority:()=>true,sessionId:()=> 'S'});
 const event={id:'M',actorUUID:'Actor.H',patientUUIDs:['Actor.P'],kind:'treatment',source:{type:'workbench',hashVerified:true},checkIds:['C'],resultIds:['M'],durationSeconds:600,sourceDegree:1,rolledHealing:undefined};
 await recorder.observe(event);await recorder.observe(event);assert.equal(records.length,1);assert.equal(records[0].state,'awaiting-evidence');assert.equal(records[0].options.rolledHealing,null);
 await recorder.observe({id:'Q',kind:'medicine-check',actorUUID:'Actor.H',patientUUIDs:[]});assert.equal(records.length,2);assert.equal(records[1].state,'awaiting-evidence');
});
test('future Workbench pack reads get their own verified lexical observer',async()=>{
 const command=await readFile(process.env.FVTT_WORKBENCH_MACRO??'C:/Users/Taka/Desktop/fvtt/output/automation-native-20260930/treat-wounds-actual-command.txt','utf8');
 const make=()=>({name:'XDY DO_NOT_IMPORT Treat Wounds and Battle Medicine',command,clone:changes=>({...make(),...changes}),execute:async function(scope){return typeof scope.explorationManualTarget==='function'}});
 const pack={getDocuments:async()=>[make()],getDocument:async()=>make()},game={packs:new Map([['xdy-pf2e-workbench.asymonous-benefactor-macros-internal',pack]])};
 const cleanup=await registerWorkbenchObservation({game,recorder:{}});const macro=await pack.getDocument('M');assert.equal(await macro.execute({}),true);assert.equal(macro.command,command);cleanup();
});
test('public Workbench forwarding clone carries observation through owner fallback',async()=>{
 const command=await readFile(new URL('./fixtures/workbench-treat-wounds-forwarding.txt',import.meta.url),'utf8');
 const make=()=>({name:'Treat Wounds and Battle Medicine',command,clone:changes=>({...make(),...changes}),execute:async function(scope){return typeof scope.explorationExecuteImmunity==='function'}});const pack={getDocuments:async()=>[make()],getDocument:async()=>make()};
 await registerWorkbenchObservation({game:{packs:new Map([['xdy-pf2e-workbench.asymonous-benefactor-macros-internal',pack]])},recorder:{}});const m=await pack.getDocument('M');assert.equal(await m.execute({}),true);assert.equal(m.command,command);
});
test('Workbench source degree and actual critical healing remain separate',()=>{
 const facts=workbenchFacts({flags:{treat_wounds_battle_medicine:{id:'T',healerId:'H',dos:2,healing:19}},rolls:[{formula:'(4d8)[healing]',total:19}]});assert.equal(facts.sourceDegree,2);assert.equal(facts.effectiveOutcome,'criticalSuccess');assert.equal(facts.rolledHealing,19);
});
test('actual DamageRoll display formula does not omit the saved healing kind',()=>{const facts=workbenchFacts({flags:{treat_wounds_battle_medicine:{id:'T',healerId:'H',dos:2,healing:23}},rolls:[{formula:'4d8 + 5',toJSON:()=>({formula:'{(4d8 + 5)[healing]}'}),total:23}]});assert.equal(facts.effectiveOutcome,'criticalSuccess')});
test('native action checks use context origin UUID and speaker ID, not context actor UUID',async()=>{let middleware;const records=[],message={id:'C',speaker:{actor:'H'},flags:{pf2e:{context:{actor:'H',origin:{actor:'Actor.H'},options:[]}}}};const recorder=createManualEvents({game:{time:{worldTime:0},messages:new Map([['C',message]])},Hooks:{on:()=>1,off(){}},ledger:recordingLedger(records),nativeActions:{addMiddleware:fn=>{middleware=fn;return ()=>{}}},isAuthority:()=>true,sessionId:()=> 'S'});recorder.start();const scope={slug:'treat-wounds',params:{target:{uuid:'Actor.P'}},tagRollOption:tag=>message.flags.pf2e.context.options.push(tag)};await middleware(scope,async()=>[{actor:{uuid:'Actor.H',id:'H',items:[]},message}]);assert.equal(records.length,1);assert.deepEqual(records[0].patientUUIDs,['Actor.P'])});
test('verified Workbench instrumentation binds each callback target, unknown source is rejected',()=>{
 const command='const rollTreatWounds = async ({ target, bmtw, skillUsed, isRiskySurgery }) => {\n const dc = {};\n};';
 assert.throws(()=>observeWorkbenchCommand(command,'unknown'),/unverified/);
 const observed=observeWorkbenchCommand(command,'b3bac907654522fda62b80da182fc253f7147493a465a82081219fb2a0f1308f');assert.match(observed,/explorationManualTarget/);assert.match(observed,/target, bmtw, skillUsed/);assert.match(observed,/skillUsed = explorationScope.skill/);
});
