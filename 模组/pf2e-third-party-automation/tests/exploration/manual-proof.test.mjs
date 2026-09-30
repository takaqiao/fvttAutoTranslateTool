import {test} from 'node:test';import assert from 'node:assert/strict';
import {createManualEvents,WORKBENCH_SOURCE_SHA} from '../../scripts/exploration/manual-events.mjs';
import {createLedger} from '../../scripts/exploration/ledger.mjs';
function fixture(){let data={sessions:{S:{id:'S',status:'recording',activityIds:[]}},activities:{},clocks:{}};const handlers=new Map(),messages=new Map(),Hooks={on:(n,f)=>{handlers.set(n,f);return n},off(){}};const game={user:{id:'G'},time:{worldTime:0},messages};const ledger=createLedger({read:async()=>data,write:async s=>{data=s},isAuthority:()=>true});const r=createManualEvents({game,Hooks,ledger,isAuthority:()=>true,sessionId:()=> 'S'});r.start();return {r,game,ledger,handlers,fire:async m=>{messages.set(m.id,m);handlers.get('createChatMessage')(m);for(let i=0;i<5;i++)await new Promise(res=>setImmediate(res))}}}
test('a source-bound Workbench failure without surgery requires immunity but no nonexistent HP application',async()=>{
 const f=fixture();const m={id:'W',rolls:[],flags:{'pf2e-third-party-automation':{explorationManual:{lexicalSource:true,sourceSHA:WORKBENCH_SOURCE_SHA,useId:'U',actorUUID:'Actor.H',patientUUID:'Actor.P',kind:'treatment',checkIds:['C'],stageIds:[],riskySurgery:false}},treat_wounds_battle_medicine:{id:'T',healerId:'H',dos:1}}};await f.fire(m);assert.deepEqual((await f.ledger.getActivity('manual:W')).options.missing,['native-immunity-receipt']);
 m.id='Risky';m.flags['pf2e-third-party-automation'].explorationManual.riskySurgery=true;m.flags['pf2e-third-party-automation'].explorationManual.stageIds=['Cut'];await f.fire(m);assert.ok((await f.ledger.getActivity('manual:Risky')).options.missing.includes('native-application-receipt'));
});
test('a manual slave receipt cannot confirm the unawaited shared master write',async()=>{
 const records=[],r=createManualEvents({game:{time:{worldTime:0}},ledger:{insertActivity:async a=>records.push(a)},isAuthority:()=>true,sessionId:()=> 'S',fromUuid:async()=>({uuid:'Actor.P'}),hpPools:{discover:()=>({poolUUID:'Actor.Master',ready:true})}});
 await r.observe({id:'W',kind:'treatment',actorUUID:'Actor.H',patientUUIDs:['Actor.P'],missing:['native-immunity-receipt']});assert.ok(records[0].options.missing.includes('shared-hp-completion-unavailable'));
});
test('native child damage messages attach to one treatment and preserve known cooldown',async()=>{
 const f=fixture(),native={tag:'exploration-manual:N',useId:'N',patientUUID:'Actor.P',continualRecovery:true};
 const root={id:'C',isCheckRoll:true,speaker:{actor:'H'},author:{id:'G'},rolls:[{_evaluated:true}],flags:{'pf2e-third-party-automation':{explorationManualNative:native},pf2e:{context:{type:'skill-check',origin:{actor:'Actor.H'},options:[native.tag],outcome:'success'}}}};await f.fire(root);
 const child={...root,id:'D',isCheckRoll:false,flags:structuredClone(root.flags),rolls:[{_evaluated:true,toJSON:()=>({formula:'{(2d8)[healing]}'})}],update:async function(changes){this.flags.pf2e.context.options=changes['flags.pf2e.context.options'];return this}};child.flags.pf2e.origin={messageId:'C'};await f.fire(child);
 const d=await f.ledger.snapshot('S');assert.equal(d.activities.length,1);assert.equal(d.activities[0].treatmentImmunitySeconds,600);assert.deepEqual(d.activities[0].proof.resultIds,['D']);assert.equal(d.activities[0].state,'awaiting-evidence');assert.ok(child.flags.pf2e.context.options.includes('pf2e-third-party-automation:source:D:0'));
});
test('the active GM observes a player-created source-bound immunity across clients',async()=>{
 const f=fixture(),meta={lexicalSource:true,sourceSHA:WORKBENCH_SOURCE_SHA,useId:'U',actorUUID:'Actor.H',patientUUID:'Actor.P',kind:'treatment'};const m={id:'W',flags:{'pf2e-third-party-automation':{explorationManual:meta},treat_wounds_battle_medicine:{id:'T',healerId:'H',dos:2}}};f.game.messages.set('W',m);await f.r.observe({id:'W',actorUUID:'Actor.H',patientUUIDs:['Actor.P'],kind:'treatment',useId:'U',checkIds:['C'],resultIds:['W'],source:{type:'workbench'},missing:['native-immunity-receipt']});
 const item={uuid:'Actor.P.Item.I',actor:{uuid:'Actor.P'},type:'effect',sourceId:'Compendium.pf2e.feat-effects.Item.Lb4q2bBAgxamtix5',flags:{'pf2e-third-party-automation':{explorationManualImmunity:{messageId:'W',useId:'U',patientUUID:'Actor.P',kind:'treatment',sourceSHA:'aa3aa174524021b06e38f9128fd29196ac5f5da863bd818068a9b2fa0e699d20',creatorId:'P'}}}};
 f.handlers.get('createItem')?.(item,{},'P');for(let i=0;i<5;i++)await new Promise(res=>setImmediate(res));assert.equal((await f.ledger.getActivity('manual:W')).state,'confirmed');
});
test('Workbench proof waits for source-bound native application and immunity; unbound receipt is ignored',async()=>{
 const f=fixture();await f.r.observe({id:'W',actorUUID:'Actor.H',patientUUIDs:['Actor.P'],kind:'treatment',durationSeconds:600,useId:'U',checkIds:['C'],resultIds:['D','W'],source:{type:'workbench',sourceSHA:WORKBENCH_SOURCE_SHA,lexicalSource:true},effectiveOutcome:'success',missing:['native-application-receipt','native-immunity-receipt']});
 const source={id:'D',rolls:[{_evaluated:true}],flags:{'pf2e-third-party-automation':{explorationManual:{lexicalSource:true,sourceSHA:WORKBENCH_SOURCE_SHA,useId:'U',actorUUID:'Actor.H',patientUUID:'Actor.P',kind:'treatment'}}}};f.game.messages.set('D',source);
 const receipt={id:'R',speaker:{actor:'P'},author:{id:'G'},flags:{pf2e:{context:{type:'damage-taken',options:[]},appliedDamage:{uuid:'Actor.P',isHealing:true,isReverted:false}}}};await f.fire(receipt);assert.deepEqual((await f.ledger.getActivity('manual:W')).proof.receiptIds,[]);
 receipt.flags.pf2e.context.options=['pf2e-third-party-automation:source:D:0'];await f.fire(receipt);const a=await f.ledger.getActivity('manual:W');assert.deepEqual(a.proof.receiptIds,['R']);assert.equal(a.state,'awaiting-evidence');
});
test('source-pinned immunity observation awaits the original native create and completes only its selected Workbench treatment',async()=>{
 const f=fixture(),meta={lexicalSource:true,sourceSHA:WORKBENCH_SOURCE_SHA,useId:'U',actorUUID:'Actor.H',patientUUID:'Actor.P',kind:'treatment'};
 const m={id:'W',flags:{'pf2e-third-party-automation':{explorationManual:meta},treat_wounds_battle_medicine:{id:'T',healerId:'H',dos:2}},rolls:[]};f.game.messages.set(m.id,m);
 await f.r.observe({id:'W',actorUUID:'Actor.H',patientUUIDs:['Actor.P'],kind:'treatment',useId:'U',durationSeconds:600,checkIds:['C'],resultIds:['W'],receiptIds:['R'],source:{type:'workbench',sourceSHA:WORKBENCH_SOURCE_SHA,lexicalSource:true},missing:['native-immunity-receipt']});
 let calls=0;const actor={uuid:'Actor.P',createEmbeddedDocuments:async(type,data)=>{calls++;assert.equal(type,'Item');assert.equal(data[0].flags['pf2e-third-party-automation'].explorationManualImmunity.messageId,'W');return [{uuid:'Actor.P.Item.I'}]}};
 const observer=f.r.bindImmunity({message:m,token:{id:'T',actor},kind:'treatment',sourceSHA:'aa3aa174524021b06e38f9128fd29196ac5f5da863bd818068a9b2fa0e699d20'});
 await observer.createEmbeddedDocuments('Item',[{system:{},flags:{core:{sourceId:'Compendium.pf2e.feat-effects.Lb4q2bBAgxamtix5'}}}]);assert.equal(calls,1);const a=await f.ledger.getActivity('manual:W');assert.equal(a.state,'confirmed');assert.deepEqual(a.proof.immunityIds,['Actor.P.Item.I']);
});
