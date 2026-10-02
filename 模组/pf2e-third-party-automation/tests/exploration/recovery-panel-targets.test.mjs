import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createRecoveryPanel} from '../../scripts/exploration/panel.mjs';

const P='Actor.P',B='Actor.B',OTHER='Actor.OTHER';
const preferences=()=>({version:1,targetIntentsByActor:{[P]:{mode:'percent',value:25},[OTHER]:{mode:'absolute',value:70}},requireNoWounded:false,failureStop:{enabled:false,limit:3}});
const deferred=()=>{let resolve;return {promise:new Promise(r=>resolve=r),resolve}};
function fixture(t,{policy={recovery:preferences()},data=null,snapshotGate}={}){
 const previous=globalThis.foundry;t.after(()=>globalThis.foundry=previous);
 globalThis.foundry={applications:{api:{ApplicationV2:class{render(){return this}}}}};
 const actors=[{actorUUID:P,name:'Patient P',hp:{value:1,max:41},focus:{value:0,max:0},pool:{poolUUID:P,ready:true},wardCapacity:1}],calls=[],saved=[],resumed=[];
 const game={user:{isGM:true,getFlag:()=>policy,setFlag:async(_module,_key,value)=>{saved.push(value);policy=value}},messages:new Map()};
 const api=createRecoveryPanel({game,coordinator:{snapshot:async()=>data,resume:async id=>resumed.push(id)},getSessionId:()=>data?.session.id??null,capabilities:{snapshot:async uuids=>{await snapshotGate?.();return actors.filter(a=>uuids.includes(a.actorUUID))}},start:async config=>calls.push(config)}),app=api.open([P]);
 const modes=[{dataset:{targetMode:'',actor:P},value:'percent'}],values=new Map([[P,{value:'25'}]]);
 const fields={budget:{value:'120'},activityBudget:{value:'100'},risky:{checked:false},focus:{checked:true},extension:{checked:false},assurance:{checked:false},rank:{value:'trained'},manualFirstRound:{checked:false},activityFirstRound:{checked:false},wounded:{checked:false},failureStop:{checked:false},failureLimit:{value:'3'}};
 const outputs=new Map([[P,{value:'11',textContent:'11'}]]),explanations=new Map([[P,{textContent:''}]]),error={textContent:''};
 const content={querySelector:selector=>{
  const name=selector.match(/\[name=(\w+)\]/)?.[1];if(name)return fields[name];
  const actor=selector.match(/data-actor="([^"]+)"/)?.[1];if(actor)return selector.includes('data-target-value')?values.get(actor):explanations.get(actor);
  const pool=selector.match(/data-pool="([^"]+)"/)?.[1];if(pool)return outputs.get(pool);
  if(selector==='[data-target-error]')return error;
 },querySelectorAll:selector=>selector==='[data-target-mode]'?modes:selector==='[data-pool]'?[...outputs].map(([pool,out])=>({...out,dataset:{pool}})):[]};
 return {api,app,actors,calls,saved,resumed,modes,values,fields,content,outputs,explanations,error};
}

test('panel renders selected patients separately with intentions and a merged shared target',async t=>{
 const policy={recovery:preferences()};policy.recovery.targetIntentsByActor[P]={mode:'percent',value:75};policy.recovery.targetIntentsByActor[B]={mode:'absolute',value:20};
 const f=fixture(t,{policy});f.app.actorUUIDs.push(B);f.actors[0].hp.max=80;f.actors.push({...f.actors[0],actorUUID:B,name:'Patient B'});
 const html=await f.app._renderHTML(await f.app._prepareContext());
 assert.match(html,/data-target-mode[^>]*data-actor="Actor.P"/);assert.match(html,/data-target-mode[^>]*data-actor="Actor.B"/);
 assert.match(html,/data-target-value[^>]*data-actor="Actor.P"[^>]*value="75"/);
 assert.match(html,/data-target-value[^>]*data-actor="Actor.B"[^>]*value="20"/);
 assert.match(html,/data-pool="Actor.P"[^>]*value="60"/);
 assert.match(html,/name="wounded"/);assert.match(html,/name="failureStop"/);assert.match(html,/name="failureLimit"[^>]*value="3"/);
 assert.equal(f.calls.length,0);
});

test('submitted mode and stop controls persist selected intent and preserve other patients',async t=>{
 const f=fixture(t);f.fields.wounded.checked=true;f.fields.failureStop.checked=true;f.fields.failureLimit.value='2';
 await f.app.act('start',f.content);
 assert.deepEqual(f.calls[0].recovery,{version:1,targetIntentsByActor:{[P]:{mode:'percent',value:25}},requireNoWounded:true,failureStop:{enabled:true,limit:2}});
 assert.deepEqual(f.calls[0].goalsByPool,[{poolUUID:P,targetHP:11}]);
 assert.equal(f.saved[0].recovery.targetIntentsByActor[OTHER].value,70);assert.equal(f.saved[0].recovery.requireNoWounded,true);assert.equal(f.saved[0].recovery.failureStop.limit,2);
});

test('panel captures intention and stop controls before an asynchronous snapshot',async t=>{
 const entered=deferred(),release=deferred(),f=fixture(t,{snapshotGate:async()=>{entered.resolve();await release.promise}});
 f.fields.wounded.checked=true;f.fields.failureStop.checked=true;f.fields.failureLimit.value='2';
 const pending=f.app.act('start',f.content);await entered.promise;
 f.modes[0].value='absolute';f.values.get(P).value='40';f.fields.wounded.checked=false;f.fields.failureLimit.value='5';release.resolve();await pending;
 assert.deepEqual(f.calls[0].recovery.targetIntentsByActor[P],{mode:'percent',value:25});assert.equal(f.calls[0].recovery.requireNoWounded,true);assert.equal(f.calls[0].recovery.failureStop.limit,2);
});

test('empty and out-of-range target or failure inputs create no work or saved preference',async t=>{
 for(const edit of [f=>f.values.get(P).value='',f=>f.values.get(P).value='101',f=>{f.modes[0].value='absolute';f.values.get(P).value='1.5'},f=>f.fields.failureLimit.value='0',f=>f.fields.failureLimit.value='',f=>f.modes.push({dataset:{targetMode:'',actor:OTHER},value:'max'})]){
  const f=fixture(t);edit(f);await assert.rejects(f.app.act('start',f.content),/invalid-recovery/);assert.equal(f.calls.length,0);assert.equal(f.saved.length,0);
 }
});

test('absolute clamping preserves the input and explains its current preview',async t=>{
 const f=fixture(t);f.actors[0].hp.max=30;f.modes[0].value='absolute';f.values.get(P).value='40';
 f.app.targetActors=f.actors;f.app.updateTargetPreview(f.content);
 assert.equal(f.outputs.get(P).value,'30');assert.match(f.explanations.get(P).textContent,/40.*30/);
 await f.app.act('start',f.content);assert.equal(f.saved[0].recovery.targetIntentsByActor[P].value,40);assert.equal(f.calls[0].goalsByPool[0].targetHP,30);
});

test('mode changes and typed values update the preview without saving or starting work',async t=>{
 const f=fixture(t),listeners=new Map(),input=f.values.get(P),mode=f.modes[0];
 await f.app._renderHTML(await f.app._prepareContext());
 f.content.addEventListener=(event,handler)=>listeners.set(event,handler);
 input.removeAttribute=name=>delete input[name];input.dataset={targetValue:'',actor:P};input.closest=()=>input;mode.closest=()=>mode;
 f.app._replaceHTML('',f.content);assert.deepEqual([...listeners.keys()],['click','input','change']);
 mode.value='max';listeners.get('change')({target:mode});assert.equal(input.hidden,true);assert.equal(input.disabled,true);assert.equal(f.outputs.get(P).value,'41');
 mode.value='absolute';input.value='';listeners.get('change')({target:mode});assert.equal(input.hidden,false);assert.equal(input.disabled,false);assert.equal(input.step,'1');assert.equal(input.value,'41');
 input.value='1.5';listeners.get('input')({target:input});assert.match(f.error.textContent,/非负整数/);
 input.value='20';listeners.get('input')({target:input});assert.equal(f.outputs.get(P).value,'20');assert.equal(f.error.textContent,'');
 assert.equal(f.saved.length,0);assert.equal(f.calls.length,0);
});

test('a paused session displays its frozen target separately from the next-start intention',async t=>{
 const recoveryGoals={version:1,patientTargets:[{patientUUID:P,poolUUID:P,intent:{mode:'percent',value:25},basisMaxHP:41,targetHP:11}],requireNoWounded:false,failureStop:{enabled:false,limit:3}};
 const session={id:'S',manual:false,status:'paused',stopReason:'user-stopped',startedAt:0,cursorAt:0,goalsByPool:[{poolUUID:P,targetHP:11}],recoveryGoals};
 const f=fixture(t,{data:{session,actors:[{pool:{poolUUID:P},hp:{value:1}}],activities:[],clocks:[]}});f.actors[0].hp.max=81;
 const html=await f.app._renderHTML(await f.app._prepareContext());
 assert.match(html,/会话目标[^<]*11/);assert.match(html,/启动时最大 HP[^<]*41/);assert.match(html,/data-pool="Actor.P"[^>]*value="21"/);
 await f.app.act('resume',f.content);assert.deepEqual(f.resumed,['S']);assert.equal(f.calls.length,0);assert.equal(f.saved.length,0);
});

test('changing the current pool mapping cannot hide or replace frozen patient targets',async t=>{
 const oldPool='Actor.OLD',newPool='Actor.NEW',recoveryGoals={version:1,patientTargets:[{patientUUID:P,poolUUID:oldPool,intent:{mode:'percent',value:25},basisMaxHP:40,targetHP:10},{patientUUID:B,poolUUID:newPool,intent:{mode:'absolute',value:18},basisMaxHP:20,targetHP:18}],requireNoWounded:false,failureStop:{enabled:false,limit:3}};
 const session={id:'S',manual:false,status:'paused',stopReason:'user-stopped',startedAt:0,cursorAt:0,goalsByPool:[{poolUUID:oldPool,targetHP:10},{poolUUID:newPool,targetHP:18}],recoveryGoals};
 const f=fixture(t,{data:{session,actors:[{actorUUID:P,pool:{poolUUID:newPool},hp:{value:1}}],activities:[],clocks:[]}});f.actors[0].pool.poolUUID=newPool;f.actors[0].hp.max=20;
 const html=await f.app._renderHTML(await f.app._prepareContext());
 assert.match(html,/会话目标 10 HP；启动时最大 HP 40/);assert.doesNotMatch(html,/共享执行目标 18/);assert.match(html,/data-pool="Actor.NEW"[^>]*value="5"/);
 session.goalsByPool.pop();assert.match(await f.app._renderHTML(await f.app._prepareContext()),/会话目标 10 HP；启动时最大 HP 40/);
 await f.app.act('resume',f.content);assert.deepEqual(f.resumed,['S']);assert.equal(f.saved.length,0);
});

test('the same panel reuses a new selection and fixes that selection before awaiting its snapshot',async t=>{
 const policy={recovery:preferences()};policy.recovery.targetIntentsByActor[B]={mode:'absolute',value:18};
 const entered=deferred(),release=deferred();let gated=false;
 const f=fixture(t,{policy,snapshotGate:async()=>{if(gated){entered.resolve();await release.promise}}});f.actors.push({...f.actors[0],actorUUID:B,name:'Patient B',hp:{value:1,max:30},pool:{poolUUID:B,ready:true}});
 await f.app.act('start',f.content);assert.equal(f.saved[0].recovery.targetIntentsByActor[B].value,18);
 assert.equal(f.api.open([B]),f.app);const html=await f.app._renderHTML(await f.app._prepareContext());assert.match(html,/data-target-value[^>]*data-actor="Actor.B"[^>]*value="18"/);assert.doesNotMatch(html,/data-target-mode[^>]*data-actor="Actor.P"/);
 f.modes[0].dataset.actor=B;f.modes[0].value='absolute';f.values.set(B,{value:'18'});gated=true;
 const pending=f.app.act('start',f.content);await entered.promise;f.api.open([P]);f.modes[0].dataset.actor=P;f.values.get(B).value='29';release.resolve();await pending;
 assert.deepEqual(f.calls[1].actorUUIDs,[B]);assert.deepEqual(f.calls[1].recovery.targetIntentsByActor,{[B]:{mode:'absolute',value:18}});assert.deepEqual(f.calls[1].goalsByPool,[{poolUUID:B,targetHP:18}]);
 assert.deepEqual(f.saved[1].recovery.targetIntentsByActor[P],{mode:'percent',value:25});assert.equal(f.saved[1].recovery.targetIntentsByActor[OTHER].value,70);
});

test('failure pause names each patient and keeps HP and wounded gaps visible',async t=>{
 const session={id:'S',manual:false,status:'paused',stopReason:'consecutive-treatment-failures',startedAt:0,cursorAt:1200,goalsByPool:[{poolUUID:P,targetHP:11}]};
 const data={session,actors:[{actorUUID:P,name:'Patient <P>',pool:{poolUUID:P},hp:{value:11}},{actorUUID:B,name:'Patient B',pool:{poolUUID:P},hp:{value:11}}],activities:[],clocks:[],treatmentFailures:{remaining:[{patientUUID:P,streak:2,limit:2,currentHP:11,targetHP:11,hpGap:0,woundedGap:true},{patientUUID:B,streak:3,limit:2,currentHP:5,targetHP:11,hpGap:6,woundedGap:false}]}};
 const f=fixture(t,{data}),html=await f.app._renderHTML(await f.app._prepareContext());
 assert.match(html,/达到连续治疗失败上限/);assert.match(html,/Patient &lt;P&gt;：连续失败 2 \/ 2；HP 11 \/ 11（尚缺 0 HP）；wounded 尚未清除/);
 assert.match(html,/Patient B：连续失败 3 \/ 2；HP 5 \/ 11（尚缺 6 HP）/);assert.equal((html.match(/wounded 尚未清除/g)??[]).length,1);
 assert.doesNotMatch(html,/consecutive-treatment-failures|Patient <P>/);assert.equal(f.calls.length,0);assert.equal(f.saved.length,0);
});
