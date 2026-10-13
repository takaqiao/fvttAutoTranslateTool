// Manual recording and the first-round manual window are retired, but worlds still hold what they saved.
// Those records must stay readable, stoppable and closable, and nothing may add to them or resume them.
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {authorityFixture} from './authority-fixture.mjs';
import {createCoordinator} from '../../scripts/exploration/coordinator.mjs';
import {createRecoveryPanel,formatSessionResult} from '../../scripts/exploration/panel.mjs';
import {chooseNext} from '../../scripts/exploration/policy.mjs';
import {resolveRecoveryGoals} from '../../scripts/exploration/recovery-goals.mjs';

const ROOT='JournalEntry.ROOT000000000001',H='Actor.HEALER',P='Actor.PATIENT';
const actors=[H,P].map((actorUUID,i)=>({actorUUID,name:i?'Patient':'Healer',medicine:{rank:i?0:1},slugs:[],items:[],assuranceSkills:[],pool:{poolUUID:actorUUID,ready:true},hp:{value:i?1:20,max:20},focus:{value:0,max:0},wardCapacity:1,modeOfBeing:'living'}));
// Every field an earlier version's start saved, including assumptions and waitForManualFirstRound, which nothing reads now.
function savedSession(id,startedAt,patch={}){
 const {goalsByPool,recoveryGoals}=resolveRecoveryGoals(actors);
 return {id,actorUUIDs:[H,P],activityIds:[],allowFiniteResources:false,assumptions:['different-actors-may-overlap'],budgetSeconds:7200,budgetEndsAt:startedAt+7200,startedAt,cursorAt:startedAt,extendTreatment:false,goalsByPool,recoveryGoals,manual:false,maxActivities:100,nativeOwnerByActor:{},protocol:{version:1,rootUUID:ROOT,epoch:'epoch'},requireFullFocus:false,riskySurgery:false,treatmentRank:'trained',useAssurance:false,waitForActivityFirstRound:false,waitForManualFirstRound:false,status:'paused',stopReason:'user-stopped',driver:{userId:'G',clientNonce:'earlier-tab',leaseNonce:`lease-${id}`},...patch};
}
const manualSession=(id,startedAt,patch={})=>{const {driver,...session}=savedSession(id,startedAt,{manual:true,manualPoolIssuer:{userId:'G',clientNonce:'earlier-tab'},...patch});return session};
const treatment=(id,sessionId,startedAt)=>({id,sessionId,providerId:'treat-wounds',actorUUID:H,patientUUIDs:[P],hpPoolUUIDs:[P],groupId:id,startedAt,endsAt:startedAt+600,durationSeconds:600,options:{},state:'confirmed',status:'confirmed',source:{type:'coordinator'},proof:{useId:id,checkIds:[`check-${id}`],receiptIds:[`receipt-${id}`]}});
const clockRow=(id,sessionId,from,patch={})=>({id,sessionId,gmId:'G',from,to:from+600,state:'confirmed',evidence:[],...patch});

// One world's ledger as an earlier version left it: four paused automatic sessions, the last of them with an
// unconfirmed time effect, then a stopped manual session whose cursor is exactly the current world time.
function stoppedWorld(){
 const seed={sessions:{},activities:{},clocks:{}};
 for(const [i,[id,stopReason]] of [['A','blocked'],['B','encounter-started'],['C','world-time-conflict'],['D','encounter-started']].entries()){
  const startedAt=i*1800;seed.sessions[id]=savedSession(id,startedAt,{cursorAt:startedAt+600,stopReason,activityIds:[`treat-${id}`]});
  seed.activities[`treat-${id}`]=treatment(`treat-${id}`,id,startedAt);seed.clocks[`clock-${id}`]=clockRow(`clock-${id}`,id,startedAt);
 }
 seed.clocks.unsettled=clockRow('unsettled','D',6000,{state:'uncertain',reason:'time-effects-unconfirmed'});
 seed.sessions.M=manualSession('M',9000);
 return {seed,worldTime:9000};
}
async function world(t,{seed,worldTime}){
 const previous=globalThis.foundry;t.after(()=>{globalThis.foundry=previous});let note='核对过了';
 globalThis.foundry={applications:{api:{ApplicationV2:class{render(){return this}},DialogV2:{wait:async()=>note}}}};
 const server=await authorityFixture(seed),ledger=server.client('this-tab'),pauses=[];let selected;
 const game={user:{id:'G',isGM:true,active:true,getFlag:()=>({})},time:{worldTime},paused:false,messages:new Map()};
 const capabilities={snapshot:async uuids=>structuredClone(actors.filter(actor=>uuids.includes(actor.actorUUID)))};
 const coordinator=createCoordinator({game,ledger,clock:{stop(){}},capabilities,providers:[],policy:chooseNext,isAuthority:()=>true,now:()=>game.time.worldTime,pauseGame:async()=>{pauses.push(game.time.worldTime);game.paused=true},ownerOperations:{}});
 const app=createRecoveryPanel({game,coordinator,capabilities,start:()=>assert.fail('nothing here starts a recovery'),getSessionId:()=>selected}).open([H,P]);
 const render=async id=>{selected=id;return app._renderHTML(await app._prepareContext())};
 const act=(id,action)=>{selected=id;return app.act(action,null)};
 return {server,ledger,coordinator,game,pauses,render,act};
}
const mainActions=html=>[...html.match(/<div class="recovery-actions recovery-main-actions">([\s\S]*?)<\/div>/)[1].matchAll(/data-recovery="([^"]+)"/g)].map(match=>match[1]);
const allActions=html=>[...html.matchAll(/data-recovery="([^"]+)"/g)].map(match=>match[1]);
const manualRow=(sessionId,at)=>({id:'late',sessionId,providerId:'manual',actorUUID:H,patientUUIDs:[P],hpPoolUUIDs:[P],startedAt:at,endsAt:at+600,state:'awaiting-evidence',source:{manual:true,type:'user-record'}});

test('the panel renders every session an earlier version stopped, and offers each a reviewed close',async t=>{
 const f=await world(t,stoppedWorld());
 for(const id of ['A','B','C','D','M']){const actions=mainActions(await f.render(id));assert.ok(actions.includes('review'),id);assert.ok(!actions.includes('start'),id)}
 const unsettled=await f.render('D');assert.match(unsettled,/<summary>首轮登记<\/summary>/);assert.doesNotMatch(unsettled,/首轮手动/);
 // What a blocked start throws has a Chinese reading wherever a reason is shown.
 assert.match(formatSessionResult({durationSeconds:0,remainingHP:0,status:'unresolved-evidence-no-replay'}),/旧会话还有未确认的结果或时间；先在那场会话里处理，处理不了可换新账本/);
assert.ok(!mainActions(unsettled).includes('resume'));assert.match(unsettled,/时间相关效果尚未完成确认/);assert.doesNotMatch(unsettled,/time-effects-unconfirmed/);
 assert.match(await f.render('M'),/已确认经过 0 分钟/);assert.equal(f.game.paused,false);assert.deepEqual(f.pauses,[]);
});

test('a stopped manual session cannot be resumed although its cursor equals the world time and nothing is unresolved',async t=>{
 const f=await world(t,stoppedWorld()),pages=f.server.raw.pages.length;
 assert.deepEqual(mainActions(await f.render('M')),['reconcile','review']);
 await assert.rejects(f.act('M','resume'),/paused-automatic-session-required/);await assert.rejects(f.coordinator.resume('M',{autoRun:false}),/paused-automatic-session-required/);
 await assert.rejects(f.ledger.resumeSession('M',{cursorAt:9000}),/paused-automatic-session-required/);await assert.rejects(f.ledger.takeoverSession('M'),/automatic-session-required/);
 assert.equal(f.server.raw.pages.length,pages);assert.equal(f.game.paused,false);assert.deepEqual(f.pauses,[]);assert.equal(f.coordinator.executionScope('M'),undefined);
 // Only the manual flag separates it from a session that would be offered a resume.
 const twin=stoppedWorld();twin.seed.sessions.M.manual=false;assert.ok(mainActions(await (await world(t,twin)).render('M')).includes('resume'));
});

test('a stopped manual session takes no new activity and closes after review with its saved fields intact',async t=>{
 const setup=stoppedWorld(),f=await world(t,setup);
 await assert.rejects(f.ledger.insertActivity(manualRow('M',9000)),/manual-recording-unavailable/);
 // Nor a row of any other kind, written past the coordinator.
 await assert.rejects(f.ledger.insertActivity({id:'direct',sessionId:'M',providerId:'treat-wounds',actorUUID:H,patientUUIDs:[P],hpPoolUUIDs:[P],startedAt:9000,endsAt:9600,state:'planned',source:{type:'coordinator'}}),/manual-recording-unavailable/);
 await assert.rejects(f.coordinator.addActivity('M',{providerId:'treat-wounds',actorUUID:H,patientUUIDs:[P],hpPoolUUIDs:[P],startedAt:9000,endsAt:9600,options:{}}),/session-driver-required/);
 await f.act('M','review');
 const closed=await f.ledger.getSession('M'),{status,stopReason,review,...kept}=closed,{status:oldStatus,stopReason:oldReason,...saved}=setup.seed.sessions.M;
 assert.equal(status,'closed');assert.equal(stopReason,'gm-reviewed-closed');assert.equal(review.note,'核对过了');assert.equal(review.disposition,'closed-without-replay');
 assert.deepEqual(kept,saved);assert.deepEqual(kept.manualPoolIssuer,{userId:'G',clientNonce:'earlier-tab'});assert.deepEqual((await f.ledger.all()).activities,setup.seed.activities);
 assert.deepEqual(mainActions(await f.render('M')),['start']);
});

// A world that was upgraded in the middle of a manual recording.
function recordingWorld(){
 const session=manualSession('R',0,{status:'recording',stopReason:null,activityIds:['seen']});
 const seen={id:'seen',sessionId:'R',providerId:'manual',actorUUID:H,patientUUIDs:[P],hpPoolUUIDs:[P],startedAt:0,endsAt:600,durationSeconds:600,order:0,state:'awaiting-evidence',options:{label:'Treat Wounds',missing:['native-immunity-receipt']},proof:{useId:'seen',checkIds:['check'],resultIds:['result'],receiptIds:[],immunityIds:[]},source:{manual:true,type:'user-record',messageId:'result'}};
 return {seed:{sessions:{R:session},activities:{seen},clocks:{}},worldTime:600};
}

test('a recording session left running shows only stop, takes no new activity, and can be stopped and closed',async t=>{
 const setup=recordingWorld(),f=await world(t,setup),html=await f.render('R');
 assert.deepEqual(mainActions(html),['stop']);assert.ok(!allActions(html).some(action=>['activity','record','openActivityCheckpoint','takeover'].includes(action)));assert.match(html,/Treat Wounds/);
 await assert.rejects(f.ledger.insertActivity(manualRow('R',600)),/manual-recording-unavailable/);await assert.rejects(f.ledger.insertActivity({id:'direct',sessionId:'R',providerId:'treat-wounds',actorUUID:H,patientUUIDs:[P],hpPoolUUIDs:[P],startedAt:600,endsAt:1200,state:'planned',source:{type:'coordinator'}}),/manual-recording-unavailable/);
 await f.act('R','stop');
 const stopped=await f.ledger.getSession('R');assert.equal(stopped.status,'paused');assert.equal(stopped.stopReason,'user-stopped');assert.equal(stopped.driver,undefined);assert.deepEqual(stopped.activityIds,['seen']);
 assert.deepEqual(await f.ledger.getActivity('seen'),setup.seed.activities.seen);assert.equal(f.game.paused,false);assert.deepEqual(f.pauses,[]);
 assert.deepEqual(mainActions(await f.render('R')),['reconcile','review']);await f.act('R','review');
 assert.equal((await f.ledger.getSession('R')).status,'closed');
 const reviewed=await f.ledger.getActivity('seen');assert.equal(reviewed.state,'awaiting-evidence');assert.equal(reviewed.review.note,'核对过了');
});

test('an encounter start stops a recording session that an earlier version left running',async t=>{
 const f=await world(t,recordingWorld());
 assert.equal((await f.coordinator.stopActive('R','encounter-started')).status,'paused');assert.equal((await f.ledger.getSession('R')).stopReason,'encounter-started');
 assert.equal((await f.coordinator.stopActive('R','user-stopped')).stopReason,'encounter-started');
});

test('the ledger refuses every field the retired features used to write',async()=>{
 const server=await authorityFixture(),ledger=server.client('this-tab'),config={actorUUIDs:[H,P],startedAt:0,cursorAt:0,budgetEndsAt:7200,goalsByPool:[{poolUUID:P,targetHP:20}]};
 const session=await ledger.createSession({id:'S',...config}),options={leaseNonce:session.driver.leaseNonce},row={id:'new',sessionId:'S',providerId:'treat-wounds',actorUUID:H,patientUUIDs:[P],hpPoolUUIDs:[P],startedAt:0,endsAt:600,state:'planned',source:{type:'coordinator'}};
 await assert.rejects(ledger.insertActivity({...row,proof:{poolApplications:{}}},options),/manual-pool-claim-required/);
 await assert.rejects(ledger.insertActivity({...row,proof:{manualPoolSource:{}}},options),/manual-pool-source-required/);
 await assert.rejects(ledger.insertActivity(manualRow('S',0),options),/manual-recording-unavailable/);
 await assert.rejects(ledger.createSession({id:'T',...config,manualCheckpoint:{phase:'open'}}),/manual-checkpoint-open-required/);
 await assert.rejects(ledger.createSession({id:'T',...config,manualPoolIssuer:{userId:'G',clientNonce:'this-tab'}}),/manual-pool-issuer-required/);
 for(const field of ['manual','manualCheckpoint','manualPoolIssuer'])await assert.rejects(ledger.updateSession('S',{[field]:true},options),/immutable-session/);
 const state=await ledger.all();assert.deepEqual(state.activities,{});assert.deepEqual(Object.keys(state.sessions),['S']);
});

// Earlier versions could leave a shared-pool claim (on a manual recording) or a first-round reservation (in an
// automatic session) unsettled. Nothing writes them now; one that is still there blocks a new or resumed recovery
// for the same members, as it did then. Reviewing its session closes the session and leaves the row unsettled, so
// the block stays; replacing the ledger is the way past it.
test('an unsettled pool claim or checkpoint reservation from an earlier version still blocks recovery for its members',async()=>{
 const config={actorUUIDs:[H,P],startedAt:9000,cursorAt:9000,budgetEndsAt:16200,goalsByPool:[{poolUUID:P,targetHP:20}]};
 const row=(id,sessionId,patch)=>({id,sessionId,providerId:'manual',actorUUID:H,patientUUIDs:[P],hpPoolUUIDs:[P],startedAt:0,endsAt:600,durationSeconds:600,order:0,state:'awaiting-evidence',options:{missing:[]},proof:{useId:id,checkIds:[],resultIds:[],receiptIds:[],immunityIds:[]},source:{manual:true,type:'workbench'},...patch});
 const claim=state=>row('claimed','M',{state:'confirmed',proof:{useId:'claimed',checkIds:['check'],resultIds:['result'],receiptIds:[],immunityIds:[],poolApplications:{one:{state}}}});
 const reservation=temporalSource=>row('reserved','A',temporalSource?{temporalSource}:{});
 const seeded=async activity=>{
  const sessions={A:savedSession('A',0,{cursorAt:600}),M:manualSession('M',1800)};sessions[activity.sessionId].activityIds=[activity.id];
  return (await authorityFixture({sessions,activities:{[activity.id]:activity},clocks:{}})).client('this-tab');
 };
 for(const activity of [claim('pending'),reservation({type:'checkpoint-reservation'})]){
  await assert.rejects((await seeded(activity)).createSession({id:'NEW',...config}),/unresolved-evidence-no-replay/,activity.id);
  await assert.rejects((await seeded(activity)).resumeSession('A',{cursorAt:600}),/unresolved-evidence-no-replay/,activity.id);
 }
 // The same rows once settled, or without the reservation, do not block.
 for(const activity of [claim('settled'),reservation()]){
  assert.equal((await (await seeded(activity)).createSession({id:'NEW',...config})).status,'running',activity.id);
  assert.equal((await (await seeded(activity)).resumeSession('A',{cursorAt:600})).status,'running',activity.id);
 }
});
