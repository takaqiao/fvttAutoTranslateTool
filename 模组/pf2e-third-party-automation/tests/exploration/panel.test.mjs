import {test} from 'node:test';import assert from 'node:assert/strict';import {formatSessionResult,escapeHTML,recoveryDefaults,sessionTimeSummary,recoveryBudgetPolicy} from '../../scripts/exploration/panel.mjs';
import * as panelUI from '../../scripts/exploration/panel.mjs';
test('earliest under assumptions is distinct from elapsed; deficit is shown',()=>{const text=formatSessionResult({durationSeconds:1800,certainty:'earliest-under-assumptions',remainingHP:12,status:'stopped'});assert.match(text,/最早|假设/);assert.match(text,/30/);assert.match(text,/12/)});
test('defaults are max HP and no finite resources; actor names escape HTML',()=>{const d=recoveryDefaults([{actorUUID:'A',hp:{value:3,max:42},pool:{poolUUID:'A'}}]);assert.equal(d.goalsByPool[0].targetHP,42);assert.equal(d.allowFiniteResources,false);assert.equal(d.riskySurgery,false);assert.equal(escapeHTML('<b>"A" & B</b>'),'&lt;b&gt;&quot;A&quot; &amp; B&lt;/b&gt;')});
test('stopped manual sessions remain reconstructed rather than claiming observed zero time',()=>{const s=sessionTimeSummary({session:{manual:true,status:'paused',startedAt:0,cursorAt:0,assumptions:['different-actors-may-overlap']},activities:[{id:'A',actorUUID:'H',durationSeconds:600,order:0}]});assert.equal(s.certainty,'earliest-under-assumptions');assert.equal(s.durationSeconds,600)});
test('user budget is configurable beyond the default two hours and rejects invalid requests',()=>{assert.deepEqual(recoveryBudgetPolicy({minutes:150,activities:8}),{budgetSeconds:9000,maxActivities:8});for(const minutes of [NaN,0,-1,Infinity])assert.throws(()=>recoveryBudgetPolicy({minutes,activities:8}),/预算/);assert.throws(()=>recoveryBudgetPolicy({minutes:30,activities:2.5}),/次数/)});

const message=id=>({id,visible:true,isContentVisible:true,update:()=>assert.fail('evidence navigation must not update messages')});
function historyFixture(){return {session:{manual:true,status:'paused',startedAt:-600,cursorAt:-600,assumptions:['different-actors-may-overlap'],review:{note:'已核对 <实际 HP> & 时间'}},actors:[{actorUUID:'Actor.H',name:'治疗者 <H>'},{actorUUID:'Actor.P',name:'患者 & P'}],clocks:[{id:'CLOCK',state:'uncertain',from:-600,to:0,reason:'clock-event-unconfirmed'}],activities:[{id:'A',providerId:'manual',actorUUID:'Actor.H',patientUUIDs:['Actor.P'],state:'awaiting-evidence',startedAt:-600,endsAt:0,durationSeconds:600,order:0,options:{label:'治疗 <一次>',missing:['native-immunity-receipt','native-immunity-receipt','<未知证据>']},proof:{checkIds:['C'],resultIds:['D'],receiptIds:['R','R','refocus-nonce'],immunityIds:['Actor.P.Item.I']},source:{manual:true,messageId:'D'}}]}}
test('history names patients and exposes deduplicated missing evidence with escaped review notes',()=>{
 assert.equal(typeof panelUI.renderRecoveryHistory,'function');const data=historyFixture(),before=structuredClone(data),html=panelUI.renderRecoveryHistory(data,{messages:new Map([['C',message('C')],['D',message('D')],['R',message('R')]])});
 assert.match(html,/治疗者 &lt;H&gt;.*→.*患者 &amp; P/);assert.match(html,/治疗 &lt;一次&gt;/);assert.equal((html.match(/缺少原生免疫回执/g)??[]).length,1);assert.match(html,/&lt;未知证据&gt;/);assert.match(html,/已核对 &lt;实际 HP&gt; &amp; 时间/);assert.doesNotMatch(html,/<未知证据>|<实际 HP>|<H>/);assert.deepEqual(data,before);
});
test('history source controls require saved messages visible to the current user; receipt nonces stay plain',()=>{
 assert.equal(typeof panelUI.renderRecoveryHistory,'function');const data=historyFixture(),messages=new Map([['C',message('C')],['D',{...message('D'),isContentVisible:false}],['R',{...message('R'),visible:false}]]),html=panelUI.renderRecoveryHistory(data,{messages});
 assert.deepEqual([...html.matchAll(/data-recovery-message="([^"]+)"/g)].map(m=>m[1]),['C']);assert.match(html,/refocus-nonce/);assert.match(html,/Actor\.P\.Item\.I/);assert.equal((html.match(/<code>R<\/code>/g)??[]).length,1);
});
test('proven group timeline gaps are attached to each affected history member',()=>{
 assert.equal(typeof panelUI.renderRecoveryHistory,'function');const data=historyFixture();data.activities=['A','B'].map(id=>({...data.activities[0],id,groupId:'G',groupProof:'lexical-proof',observedStart:-1200,observedEnd:-600,options:{},proof:{}}));const html=panelUI.renderRecoveryHistory(data);
 assert.equal((html.match(/记录时间早于可开始时间/g)??[]).length,2);
});
test('paused clock context shows its interval relative to a negative session epoch and retains the reason',()=>{
 assert.equal(typeof panelUI.renderRecoveryHistory,'function');const html=panelUI.renderRecoveryHistory(historyFixture());assert.match(html,/第 0 → 10 分钟/);assert.match(html,/时间提交.*结果未知.*clock-event-unconfirmed/);assert.match(html,/尚未确认/);assert.match(html,/治疗者 &lt;H&gt;/);
});
function chatFixture({loaded=['NEW'],onRender,onBatch}={}){
 const messages=new Map(['OLD','MID','NEW'].map(id=>[id,message(id)]));messages.contents=[...messages.values()];const cards=new Map(),calls=[];
 const add=id=>cards.set(id,{dataset:{messageId:id},scrollIntoView:options=>calls.push(['scroll',id,options])});loaded.forEach(add);
 const chat={rendered:true,element:{querySelectorAll:selector=>{assert.equal(selector,'.chat-log .message[data-message-id]');return [...cards.values()]}},activate:()=>calls.push(['activate']),render:async options=>{calls.push(['render',options]);chat.rendered=true;await onRender?.(messages)},renderBatch:async size=>{calls.push(['batch',size]);await onBatch?.(messages);add('OLD');add('MID')},postOne:()=>assert.fail('must not repost a saved message')};
 return {game:{messages,time:{worldTime:123},actors:new Map()},chat,calls,messages};
}
test('evidence navigation activates and scrolls the existing native card without applying or reposting it',async()=>{
 assert.equal(typeof panelUI.showRecoveryMessage,'function');const f=chatFixture({loaded:['OLD','NEW']});assert.equal(await panelUI.showRecoveryMessage('OLD',f),true);assert.deepEqual(f.calls,[['activate'],['scroll','OLD',{block:'center',behavior:'auto'}]]);assert.equal(f.game.time.worldTime,123);assert.equal(f.messages.size,3);
});
test('evidence navigation loads the older native batch through the target before scrolling',async()=>{
 assert.equal(typeof panelUI.showRecoveryMessage,'function');const f=chatFixture();assert.equal(await panelUI.showRecoveryMessage('OLD',f),true);assert.deepEqual(f.calls,[['activate'],['batch',3],['scroll','OLD',{block:'center',behavior:'auto'}]]);
});
test('hidden, deleted and nonchat receipt IDs cannot open native chat evidence',async()=>{
 assert.equal(typeof panelUI.showRecoveryMessage,'function');for(const mode of ['hidden','content-hidden','deleted','nonce']){const f=chatFixture();if(mode==='hidden')f.messages.get('OLD').visible=false;if(mode==='content-hidden')f.messages.get('OLD').isContentVisible=false;if(mode==='deleted')f.messages.delete('OLD');assert.equal(await panelUI.showRecoveryMessage(mode==='nonce'?'refocus-nonce':'OLD',f),false);assert.deepEqual(f.calls,[])}
});
test('evidence navigation revalidates message deletion after rendering and privacy after batch loading',async()=>{
 assert.equal(typeof panelUI.showRecoveryMessage,'function');const first=chatFixture({onRender:messages=>messages.delete('OLD')});first.chat.rendered=false;assert.equal(await panelUI.showRecoveryMessage('OLD',first),false);assert.deepEqual(first.calls,[['render',{force:true}]]);
 const second=chatFixture({onBatch:messages=>{messages.get('OLD').isContentVisible=false}});assert.equal(await panelUI.showRecoveryMessage('OLD',second),false);assert.deepEqual(second.calls,[['activate'],['batch',3]]);
});
test('the rendered panel routes a saved source click to native chat without changing recovery facts',async t=>{
 const previousFoundry=globalThis.foundry,previousUI=globalThis.ui;t.after(()=>{globalThis.foundry=previousFoundry;globalThis.ui=previousUI});globalThis.foundry={applications:{api:{ApplicationV2:class{render(){return this}}}}};
 const f=chatFixture({loaded:['C']}),data=historyFixture();f.messages.set('C',message('C'));data.session.goalsByPool=[];data.actors=data.actors.map(a=>({...a,hp:{value:7,max:20},pool:{poolUUID:a.actorUUID,ready:true},focus:{value:1,max:3},wardCapacity:1}));const before=structuredClone(data);globalThis.ui={chat:f.chat,notifications:{warn:()=>assert.fail('visible source should open')}};
 const game={...f.game,user:{isGM:true,getFlag:()=>({}),setFlag:()=>assert.fail('source clicks must not save policy')}},unexpected=()=>assert.fail('source clicks must not execute a recovery action'),coordinator={snapshot:async()=>data,start:unexpected,stop:unexpected,resume:unexpected,reconcile:unexpected,review:unexpected};
 const api=panelUI.createRecoveryPanel({game,coordinator,capabilities:{snapshot:async()=>data.actors},start:unexpected,record:unexpected,getSessionId:()=> 'S',onError:error=>assert.fail(error.message)}),app=api.open(data.actors.map(a=>a.actorUUID)),html=await app._renderHTML(await app._prepareContext());assert.match(html,/data-recovery-message="C"/);assert.match(html,/recovery-review/);
 let click;app._replaceHTML(html,{addEventListener:(event,handler)=>{assert.equal(event,'click');click=handler}});click({target:{closest:selector=>selector==='[data-recovery-message]'?{dataset:{recoveryMessage:'C'}}:null},preventDefault(){}});await new Promise(resolve=>setImmediate(resolve));
 assert.deepEqual(f.calls,[['activate'],['scroll','C',{block:'center',behavior:'auto'}]]);assert.deepEqual(data,before);assert.equal(game.time.worldTime,123);
});
test('hostile source IDs are escaped in controls and selected by exact dataset equality',async()=>{
 const id='C" onclick="bad<',data=historyFixture();data.activities[0].proof={checkIds:[id]};data.activities[0].source={};const html=panelUI.renderRecoveryHistory(data,{messages:new Map([[id,message(id)]])});assert.match(html,/data-recovery-message="C&quot; onclick=&quot;bad&lt;"/);assert.doesNotMatch(html,/onclick="bad</);
 const f=chatFixture({loaded:[id]});f.messages.set(id,message(id));assert.equal(await panelUI.showRecoveryMessage(id,f),true);assert.equal(f.calls[1][1],id);
});
test('a public manual declaration explains its missing source in Chinese without inventing a chat link',()=>{
 const data=historyFixture();data.activities[0].options={label:'登记搜索',missing:['manual-source-requires-review']};data.activities[0].proof={resultIds:['declaration-id']};data.activities[0].source={type:'user-record',unverified:true,messageId:'declaration-id'};
 const html=panelUI.renderRecoveryHistory(data);assert.match(html,/人工登记的行动来源需 GM 核对/);assert.doesNotMatch(html,/manual-source-requires-review|data-recovery-message=|骰点结果/);
});
test('a saved uncertain clock explains the actual missing world-time source in Chinese',()=>{
 const data=historyFixture();data.clocks[0].reason='world-time-source-unconfirmed';const html=panelUI.renderRecoveryHistory(data);assert.match(html,/世界时间来源回执尚未确认/);assert.doesNotMatch(html,/world-time-source-unconfirmed/);
});

function checkpointPanelFixture(t,{phase='open',status='running'}={}){
 const previous=globalThis.foundry;t.after(()=>{globalThis.foundry=previous});globalThis.foundry={applications:{api:{ApplicationV2:class{render(){return this}}}}};
 const actors=[{actorUUID:'Actor.H',name:'H',hp:{value:20,max:20},focus:{value:1,max:1},pool:{poolUUID:'Actor.H',ready:true},wardCapacity:1}],binding={id:'C',sessionId:'S',rootUUID:'JournalEntry.ROOT',epoch:'E',observationNonce:'N',from:0,to:600},calls=[];
 const data={session:{id:'S',status,manual:false,startedAt:0,cursorAt:0,goalsByPool:[{poolUUID:'Actor.H',targetHP:20}],manualCheckpoint:{...binding,phase}},actors,activities:[],clocks:[]};
 const game={user:{isGM:true,getFlag:()=>({}),setFlag:async()=>{}},messages:new Map()},coordinator={snapshot:async()=>data,closeManualCheckpoint:async value=>calls.push(['close',value])};
 const api=panelUI.createRecoveryPanel({game,coordinator,capabilities:{snapshot:async()=>actors},start:async config=>calls.push(['start',config]),getSessionId:()=> 'S'}),app=api.open(['Actor.H']);return {app,data,binding,calls};
}
test('an open manual checkpoint shows its waiting instruction and continue action only while open',async t=>{
 const f=checkpointPanelFixture(t),context=await f.app._prepareContext(),html=await f.app._renderHTML(context);assert.match(html,/name="manualFirstRound"/);assert.match(html,/首轮等待手动治疗/);assert.match(html,/data-recovery="closeManualCheckpoint"/);assert.match(html,/继续本轮/);assert.match(html,/Workbench/);
 f.data.session.manualCheckpoint.phase='advancing';assert.doesNotMatch(await f.app._renderHTML(await f.app._prepareContext()),/data-recovery="closeManualCheckpoint"/);
});
test('continue routes only the current immutable checkpoint binding through the panel action',async t=>{
 const f=checkpointPanelFixture(t);await f.app.act('closeManualCheckpoint',null);assert.deepEqual(f.calls,[['close',f.binding]]);
});
test('the first-round waiting choice reaches start while ordinary starts remain the default',async t=>{
 const f=checkpointPanelFixture(t,{status:'complete'}),values={budget:{value:'10'},activityBudget:{value:'2'},risky:{checked:false},focus:{checked:true},extension:{checked:false},assurance:{checked:false},rank:{value:'trained'},manualFirstRound:{checked:true}},content={querySelector:selector=>values[selector.match(/name=(\w+)/)[1]],querySelectorAll:()=>[{dataset:{pool:'Actor.H'},value:'20'}]};
 await f.app.act('start',content);assert.equal(f.calls[0][1].waitForManualFirstRound,true);values.manualFirstRound.checked=false;await f.app.act('start',content);assert.equal(f.calls[1][1].waitForManualFirstRound,false);
});
