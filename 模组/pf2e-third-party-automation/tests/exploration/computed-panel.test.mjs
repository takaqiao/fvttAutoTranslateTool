import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createRecoveryPanel} from '../../scripts/exploration/panel.mjs';

function fixture(t,{policy={},data=null}={}){
 const previous=globalThis.foundry;t.after(()=>globalThis.foundry=previous);
 globalThis.foundry={applications:{api:{ApplicationV2:class{render(){return this}}}}};
 const actors=[{actorUUID:'Actor.P',name:'患者 <P>',hp:{value:3,max:40},focus:{value:0,max:2},pool:{poolUUID:'Actor.P',ready:true},wardCapacity:1}],started=[],saved=[],resumed=[],errors=[];
 const game={user:{isGM:true,getFlag:()=>policy,setFlag:async(_module,_key,value)=>{saved.push(value);policy=value}},messages:new Map(),time:{worldTime:0}};
 const coordinator={snapshot:async()=>data,resume:async id=>resumed.push(id),openActivityCheckpoint:()=>assert.fail('must not open a recovery window'),closeActivityCheckpoint:()=>assert.fail('must not continue a native recovery')};
 const capabilities={snapshot:async()=>actors};
 const api=createRecoveryPanel({game,coordinator,capabilities,start:async config=>started.push(config),getSessionId:()=>data?.session.id??null,onError:error=>errors.push(error)}),app=api.open(['Actor.P']);
 const fields={recoveryStrategy:{value:'success-first'},budget:{value:'120'},activityBudget:{value:'100'},risky:{checked:false},focus:{checked:true},extension:{checked:false},rank:{value:'auto'},wounded:{checked:true},failureStop:{checked:true},failureLimit:{value:'2'}};
 const modes=[{dataset:{targetMode:'',actor:'Actor.P'},value:'percent'}],listeners=new Map();
 const content={addEventListener:(type,handler)=>listeners.set(type,handler),querySelectorAll:selector=>selector==='[data-target-mode]'?modes:[],querySelector:selector=>selector.includes('data-target-value')?{value:'75'}:fields[selector.match(/\[name=(\w+)\]/)?.[1]]};
 const render=async()=>{const html=await app._renderHTML(await app._prepareContext());app._replaceHTML(html,content);return html};
 const click=async action=>{listeners.get('click')({target:{closest:selector=>selector==='[data-recovery]'?{dataset:{recovery:action}}:null}});await new Promise(resolve=>setImmediate(resolve))};
 return {app,actors,started,saved,resumed,errors,fields,content,render,click,capabilities,game,coordinator};
}
const session=(extra={})=>({id:'S',status:'paused',startedAt:0,cursorAt:0,goalsByPool:[{poolUUID:'Actor.P',targetHP:40}],...extra});
const snapshot=extra=>({session:session(extra),actors:[{actorUUID:'Actor.P',pool:{poolUUID:'Actor.P'},hp:{value:3}}],activities:[],clocks:[]});

test('the start view exposes the saved strategy before advanced controls and removes native roll controls',async t=>{
 const f=fixture(t,{policy:{recoveryStrategy:'time-first',useAssurance:true}}),html=await f.render();
 assert.match(html,/<select name="recoveryStrategy">[\s\S]*?<option value="time-first" selected>/);
 assert.ok(html.indexOf('name="recoveryStrategy"')<html.indexOf('class="recovery-advanced"'));
 assert.match(html,/data-recovery="start">计算恢复并推进时间/);
 assert.match(html,/<option value="auto" selected>/);
 assert.doesNotMatch(html,/name="(?:assurance|activityFirstRound)"|掷骰恢复并推进时间/);
});

test('clicking start passes the selected strategy and saves it with target and stop preferences',async t=>{
 const f=fixture(t);await f.render();f.fields.recoveryStrategy.value='time-first';await f.click('start');
 assert.deepEqual(f.errors,[]);assert.equal(f.started.length,1);assert.equal(f.started[0].recoveryStrategy,'time-first');
 assert.deepEqual(f.started[0].goalsByPool,[{poolUUID:'Actor.P',targetHP:30}]);
 assert.equal(f.started[0].requireFullFocus,true);assert.equal(f.started[0].treatmentRank,'auto');
 assert.equal(f.saved[0].recoveryStrategy,'time-first');assert.equal(JSON.parse(f.saved[0].recovery).failureStop.limit,2);
 assert.equal(Object.hasOwn(f.saved[0],'useAssurance'),false);assert.equal(Object.hasOwn(f.started[0],'waitForActivityFirstRound'),false);
 assert.match(await f.render(),/<option value="time-first" selected>/);
});

test('the default strategy is success-first and a submitted strategy cannot change during the actor read',async t=>{
 const f=fixture(t);assert.match(await f.render(),/<option value="success-first" selected>/);
 let release,entered;const gate=new Promise(resolve=>release=resolve),reading=new Promise(resolve=>entered=resolve);
 f.capabilities.snapshot=async()=>{entered();await gate;return f.actors};
 const pending=f.app.act('start',f.content);await reading;f.fields.recoveryStrategy.value='time-first';release();await pending;
 assert.equal(f.started[0].recoveryStrategy,'success-first');assert.equal(f.saved[0].recoveryStrategy,'success-first');
});

test('an invalid strategy cannot start recovery or overwrite preferences',async t=>{
 const f=fixture(t);f.fields.recoveryStrategy.value='unknown';
 await assert.rejects(f.app.act('start',f.content),/恢复策略/);assert.equal(f.started.length,0);assert.equal(f.saved.length,0);
});

test('legacy sessions retain review actions but cannot resume even through a stale click handler',async t=>{
 const f=fixture(t,{data:snapshot()}),html=await f.render();
 assert.match(html,/data-recovery="reconcile"/);assert.match(html,/data-recovery="review"/);assert.match(html,/旧版.*会话/);
 assert.doesNotMatch(html,/data-recovery="resume"/);await f.click('resume');
 assert.equal(f.resumed.length,0);assert.match(f.errors[0].message,/核对.*关闭/);
});

test('the computed resume control follows the current batch checkpoint availability',async t=>{
 const data=snapshot({executionMode:'computed-v1'}),f=fixture(t,{data});data.canResumeComputed=false;
 assert.doesNotMatch(await f.render(),/data-recovery="resume"/);await assert.rejects(f.app.act('resume',f.content),/未确认|继续/);
 assert.equal(f.resumed.length,0);data.canResumeComputed=true;assert.match(await f.render(),/data-recovery="resume"/);
 await f.click('resume');assert.deepEqual(f.resumed,['S']);
});

test('neither legacy nor computed recovery opens or seals a registration window',async t=>{
 for(const executionMode of [undefined,'computed-v1']){
  const data=snapshot({status:'running',executionMode,activityCheckpoint:{phase:'open',from:0,registrations:{}}}),f=fixture(t,{data});
  assert.doesNotMatch(await f.render(),/data-recovery="(?:openActivityCheckpoint|closeActivityCheckpoint|activity)"/);
  for(const action of ['openActivityCheckpoint','closeActivityCheckpoint','activity'])await assert.rejects(f.app.act(action,f.content),/登记/);
 }
});

test('computed recovery renders one confirmed summary and keeps sampled process details collapsed',async t=>{
 const data=snapshot({executionMode:'computed-v1',status:'complete',cursorAt:1200,computedRecovery:{summary:{elapsedSeconds:1200,stopReason:'goals-met',activityCount:3,actors:[{actorUUID:'Actor.P',name:'患者 <P>',hpBefore:3,hpAfter:40,hpMax:40,targetHP:40,focusBefore:0,focusAfter:2,focusMax:2,woundedBefore:1,woundedAfter:0,status:'goals-met'}],events:[{providerId:'treat-wounds',actorUUID:'Actor.P',patientUUIDs:['Actor.P'],startedAt:0,endsAt:600,results:[{degree:2,healing:17}]}]}}}),f=fixture(t,{data}),html=await f.render();
 assert.match(html,/已确认经过 20 分钟/);assert.match(html,/恢复目标完成/);assert.match(html,/3 → 40 \/ 40/);assert.match(html,/0 → 2 \/ 2/);assert.match(html,/1 → 0/);
 assert.match(html,/患者 &lt;P&gt;/);assert.doesNotMatch(html,/患者 <P>/);
 assert.match(html,/<details class="recovery-computed-events"><summary>过程详情（3 项行动）/);
 assert.match(html,/治疗伤势/);assert.equal((html.match(/class="recovery-time"/g)??[]).length,1);
 assert.doesNotMatch(html,/recovery-member-summary|recovery-history-details|data-recovery-message/);
});

test('computed treatment details report actual check and dice results from a saved event',async t=>{
 const data=snapshot({executionMode:'computed-v1',status:'complete',cursorAt:600,computedRecovery:{summary:{elapsedSeconds:600,stopReason:'goals-met',activityCount:1,actors:[],events:[{actorUUID:'Actor.P',patientUUID:'Actor.P',providerId:'treat-wounds',startedAt:0,endsAt:600,check:{face:14,total:21,dc:15,assurance:false},outcome:2,riskyDice:[3],healingDice:[6,5],healing:11,damage:3,hpBefore:3,hpAfter:11}]}}}),f=fixture(t,{data}),html=await f.render();
 assert.match(html,/检定 21 \/ DC 15/);assert.match(html,/d20：14/);assert.match(html,/治疗骰：6、5/);assert.match(html,/手术骰：3/);assert.match(html,/成功；治疗 11 HP；伤害 3 HP；HP 3 → 11/);
});

test('unconfirmed computed batches and clocks remain visible without presenting proposed HP as applied',async t=>{
 const data=snapshot({executionMode:'computed-v1',stopReason:'unresolved-computed-batch',computedRecovery:{batches:[{id:'B<1>',state:'uncertain',from:0,to:600,reason:'computed-actor-changed',payload:{afterActors:[{hp:{value:999}}]}}]}});
 data.clocks=[{id:'C<1>',state:'uncertain',from:0,to:600,reason:'world-time-source-unconfirmed'}];
 const f=fixture(t,{data}),html=await f.render();assert.match(html,/计算批次仍有未确认结果/);assert.match(html,/B&lt;1&gt;/);assert.match(html,/C&lt;1&gt;/);assert.match(html,/角色状态已改变/);assert.match(html,/第 0 → 10 分钟/);
 assert.doesNotMatch(html,/999|data-recovery="resume"/);
});

test('an advanced time boundary without a confirmed batch is shown for review and cannot resume',async t=>{
 const data=snapshot({executionMode:'computed-v1',cursorAt:600,computedRecovery:{batches:[],pendingBoundary:{from:0,to:600}}}),f=fixture(t,{data});
 data.canResumeComputed=true;f.game.time.worldTime=600;
 const html=await f.render();assert.match(html,/恢复计算.*第 0 → 10 分钟.*待核对/);assert.match(html,/本段不自动重抽/);
 assert.doesNotMatch(html,/data-recovery="resume"/);await assert.rejects(f.app.act('resume',f.content),/未确认|不能继续/);assert.equal(f.resumed.length,0);
});

test('a pending stable calculation shows only its start time and blocks resume before a batch exists',async t=>{
 const data=snapshot({executionMode:'computed-v1',cursorAt:600,computedRecovery:{batches:[],pendingCalculation:{from:600}}}),f=fixture(t,{data});
 data.canResumeComputed=true;f.game.time.worldTime=600;
 const html=await f.render();assert.match(html,/第 10 分钟起/);assert.match(html,/恢复计算已开始，结果尚未确认，不自动重抽/);
 assert.doesNotMatch(html,/NaN|undefined|data-recovery="resume"/);await assert.rejects(f.app.act('resume',f.content),/未确认|不能继续/);assert.equal(f.resumed.length,0);
 data.session.computedRecovery.batches.push({id:'B',state:'uncertain',from:600,to:1200});
 const withBatch=await f.render();assert.match(withBatch,/计算批次.*第 10 → 20 分钟/);assert.doesNotMatch(withBatch,/恢复计算已开始/);
});

test('a closed computed review retains the GM note and rule diagnostic alongside the summary',async t=>{
 const data=snapshot({executionMode:'computed-v1',status:'closed',stopReason:'closed-without-replay',computedRecovery:{review:{disposition:'closed-without-replay',note:'已核对 <HP>，保留未知结果'},batches:[],simulationState:{diagnostics:['computed-treatment-model-unverified']},summary:{elapsedSeconds:0,activityCount:0,stopReason:'unreachable',actors:[],events:[]}}}),f=fixture(t,{data}),html=await f.render();
 assert.match(html,/GM 已核对结束，旧结果不重放/);assert.match(html,/治疗规则模型尚未核验/);assert.match(html,/已核对 &lt;HP&gt;，保留未知结果/);
});

test('compact summaries and simulation checkpoints both retain distinct rule diagnostics',async t=>{
 const data=snapshot({executionMode:'computed-v1',computedRecovery:{batches:[],simulationState:{diagnostics:['refocus-recovery-unadapted','healing-received-modifier-unverified']},summary:{elapsedSeconds:0,activityCount:0,stopReason:'unreachable',actors:[],diagnostics:['native-healing-model-unverified','healing-received-modifier-unverified']}}}),f=fixture(t,{data}),html=await f.render();
 assert.match(html,/Magic Hands/);assert.match(html,/特殊再聚能规则尚未适配/);assert.equal((html.match(/患者的受疗修正尚未适配/g)??[]).length,1);
 assert.doesNotMatch(html,/native-healing-model-unverified|healing-received-modifier-unverified/);
});

test('a compact summary reads sampled events from confirmed batches and excludes uncertain ones',async t=>{
 const event={actorUUID:'Actor.P',patientUUID:'Actor.P',providerId:'treat-wounds',startedAt:0,endsAt:600,check:{face:14,total:21,dc:15,assurance:false},outcome:2,healingDice:[6,5],healing:11};
 const data=snapshot({executionMode:'computed-v1',status:'paused',computedRecovery:{batches:[{id:'B1',state:'confirmed',from:0,to:600,events:[event]},{id:'B2',state:'uncertain',from:600,to:1200,events:[{...event,healing:999}]}],summary:{elapsedSeconds:600,activityCount:1,actors:[],diagnostics:[]}}}),f=fixture(t,{data}),html=await f.render();
 assert.match(html,/检定 21 \/ DC 15/);assert.match(html,/治疗骰：6、5/);assert.doesNotMatch(html,/999/);assert.equal((html.match(/检定 21/g)??[]).length,1);
});

test('the computed close dialog explains that review keeps unknown records and allows a fresh session',async t=>{
 const data=snapshot({executionMode:'computed-v1'}),f=fixture(t,{data}),reviews=[];let dialog;
 globalThis.foundry.applications.api.DialogV2={wait:async options=>{dialog=options;return '已核对'}};
 f.coordinator.review=async(id,options)=>reviews.push({id,...options});
 await f.app.act('review',f.content);
 assert.match(dialog.content,/未知记录.*保留/);assert.match(dialog.content,/关闭后可开始新的计算恢复/);assert.doesNotMatch(dialog.content,/仍会占用/);
 assert.equal(reviews[0].note,'已核对');assert.equal(reviews[0].id,'S');
});

test('reconcile displays current actor and effect readbacks without changing the unknown batch',async t=>{
 const data=snapshot({executionMode:'computed-v1',computedRecovery:{batches:[{id:'B',state:'uncertain',from:0,to:600}]}}),f=fixture(t,{data});
 f.coordinator.reconcile=async()=>({...structuredClone(data),computedReadbacks:[{batchId:'B',actors:[{actorUUID:'Actor.P',state:'confirmed'}],items:[{actorUUID:'Actor.P',state:'unknown'}]}]});
 await f.render();await f.click('reconcile');const html=await f.render();assert.deepEqual(f.errors,[]);
 assert.match(html,/<details class="recovery-readback"><summary>本次只读核对/);assert.match(html,/角色状态.*已应用/);assert.match(html,/效果.*尚无法确认/);assert.match(html,/未确认批次仍然保留/);
 assert.equal(data.session.computedRecovery.batches[0].state,'uncertain');assert.equal(f.saved.length,0);assert.equal(f.started.length,0);
});

test('switching the displayed session clears a previous computed readback',async t=>{
 const data=snapshot({executionMode:'computed-v1'}),f=fixture(t,{data});
 f.coordinator.reconcile=async()=>({...structuredClone(data),computedReadbacks:[{batchId:'OLD-BATCH',actors:[{actorUUID:'Actor.P',state:'confirmed'}],items:[]}]});
 await f.render();await f.click('reconcile');assert.match(await f.render(),/OLD-BATCH/);
 data.session.id='OTHER';assert.doesNotMatch(await f.render(),/OLD-BATCH|recovery-readback/);
 data.session.id='S';assert.doesNotMatch(await f.render(),/OLD-BATCH|recovery-readback/);
});
