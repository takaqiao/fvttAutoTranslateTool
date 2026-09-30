import {test} from 'node:test';import assert from 'node:assert/strict';
import {createManualEvents,WORKBENCH_SOURCE_SHA} from '../../scripts/exploration/manual-events.mjs';
import {createLedger} from '../../scripts/exploration/ledger.mjs';
import {manualEvidenceFixture,flush} from './manual-evidence-fixture.mjs';
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
 const f=manualEvidenceFixture(),r=f.createRecorder();r.start();await r.observe(f.event);await f.fire(f.receipt());
 const item=f.immunity();f.patient.items.set(item.id,item);f.handlers.get('createItem')(item,{},'PUSER');await flush();assert.equal((await f.ledger.getActivity('manual:W')).state,'confirmed');r.stop();
});
test('Workbench proof waits for source-bound native application and immunity; unbound receipt is ignored',async()=>{
 const f=manualEvidenceFixture(),r=f.createRecorder();r.start();await r.observe({...f.event,resultIds:['D','W']});
 const source={...f.result,id:'D'};f.messages.set('D',source);
 const receipt=f.receipt('R','G');receipt.flags.pf2e.context.options=[];await f.fire(receipt);assert.deepEqual((await f.ledger.getActivity('manual:W')).proof.receiptIds,[]);
 receipt.flags.pf2e.context.options=['pf2e-third-party-automation:source:D:0'];await f.fire(receipt);const a=await f.ledger.getActivity('manual:W');assert.deepEqual(a.proof.receiptIds,['R']);assert.equal(a.state,'awaiting-evidence');r.stop();
});
test('source-pinned immunity observation awaits the original native create and completes only its selected Workbench treatment',async()=>{
 const f=manualEvidenceFixture(),r=f.createRecorder();r.start();await r.observe(f.event);await f.fire(f.receipt());
 let calls=0;f.patient.createEmbeddedDocuments=async(type,data)=>{calls++;assert.equal(type,'Item');assert.equal(data[0].flags['pf2e-third-party-automation'].explorationManualImmunity.messageId,'W');return [{...f.immunity(),flags:data[0].flags}]};
 const observer=r.bindImmunity({message:f.result,token:{id:'T',actor:f.patient},kind:'treatment',sourceSHA:'aa3aa174524021b06e38f9128fd29196ac5f5da863bd818068a9b2fa0e699d20'});
 await observer.createEmbeddedDocuments('Item',[{system:{},flags:{core:{sourceId:'Compendium.pf2e.feat-effects.Lb4q2bBAgxamtix5'}}}]);assert.equal(calls,1);const a=await f.ledger.getActivity('manual:W');assert.equal(a.state,'confirmed');assert.deepEqual(a.proof.immunityIds,['Actor.P.Item.I']);r.stop();
});
