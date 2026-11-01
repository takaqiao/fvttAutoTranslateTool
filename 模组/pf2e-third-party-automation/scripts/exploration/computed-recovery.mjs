import {captureRecoveryPreferences} from './schema.mjs';
import {resolveRecoveryGoals} from './recovery-goals.mjs';
import {createRecoverySimulation} from './computed-plan.mjs';

export const computedMode='computed-v1';
const unresolved=session=>!!session?.computedRecovery?.pendingBoundary||!!session?.computedRecovery?.pendingCalculation||session?.computedRecovery?.batches.some(batch=>batch.state!=='confirmed');
const compactActors=(actors,{stamps=true}={})=>actors.map(({actorUUID,name,hp,focus,wounded,pool,executionStamp})=>({actorUUID,name,hp,focus,wounded,pool,...stamps&&executionStamp?{executionStamp}:{}}));
const copy=value=>structuredClone(value);

function validateGoalBasis(session,actors){
 for(const initial of session.computedInitialActors){
  const actor=actors.find(row=>row.actorUUID===initial.actorUUID);
  if(!actor?.pool?.ready||actor.pool.poolUUID!==initial.pool.poolUUID)throw Error('recovery-pool-changed');
  if(actor.hp.max!==initial.hp.max)throw Error('recovery-max-changed');
 }
}

export function computedSummary(session,state,actors,reason) {
 const initial=new Map(session.computedInitialActors.map(actor=>[actor.actorUUID,actor]));
 return {elapsedSeconds:state.cursorAt-session.startedAt,activityCount:state.activityCount??state.history?.length??0,stopReason:reason,
  actors:actors.map(actor=>{const before=initial.get(actor.actorUUID),targetHP=session.goalsByPool.find(goal=>goal.poolUUID===actor.pool.poolUUID)?.targetHP??actor.hp.max;
   return {actorUUID:actor.actorUUID,name:actor.name,hpBefore:before.hp.value,hpAfter:actor.hp.value,hpMax:actor.hp.max,targetHP,focusBefore:before.focus.value,focusAfter:actor.focus.value,focusMax:actor.focus.max,woundedBefore:before.wounded,woundedAfter:actor.wounded,status:actor.hp.value<=0?'needs-review':actor.hp.value>=targetHP&&(!session.recoveryGoals?.requireNoWounded||!actor.wounded)&&(!session.requireFullFocus||actor.focus.value>=actor.focus.max)?'goals-met':'remaining'};}),diagnostics:copy(state.diagnostics??[])};
}

/** The driver owns only computed sessions; legacy providers are never invoked. */
export function createComputedRecovery({game,ledger,clock,capabilities,documents,prepareActors=capabilities.snapshot,pauseGame,isAuthority,onChange=()=>{},simulationFactory=createRecoverySimulation,timeBoundary=async()=>({segmented:true,until:Infinity})}) {
 const scopes=new Map(),runners=new Map(),stops=new Map();
 const check=()=>{if(!isAuthority()||game.user.active===false)throw Error('active-gm-required')};
 const scopeFor=id=>scopes.get(id);
 const guard=(id,scope)=>{check();if(scopeFor(id)!==scope)throw Error('computed-driver-changed');if(game.combat?.started)throw Error('encounter-started');return true};
 async function owned(id,scope,{running=true}={}){guard(id,scope);const session=await ledger.getSession(id);guard(id,scope);if(!ledger.ownsSession(session,scope))throw Error('session-driver-required');if(running&&session.status!=='running')throw Error('recovery-session-stopped');return session}
 function computed(session){if(session?.executionMode!==computedMode)throw Error('legacy-recovery-view-only');return session}
 const scopeOptions=scope=>({leaseNonce:scope?.leaseNonce});
 function accept(session){const scope={leaseNonce:session.driver?.leaseNonce};if(!ledger.ownsSession(session,scope))throw Error('session-driver-required');scopes.set(session.id,scope);return scope}
 async function pause(id,reason,scope=scopeFor(id)){
  check();const session=await ledger.getSession(id);if(!session||!['running','paused'].includes(session.status))return session;
  const saved=await ledger.updateSession(id,{status:'paused',stopReason:reason},scope?scopeOptions(scope):{});scopes.delete(id);onChange(id);return saved;
 }
 async function start(config){
  check();if(config.manual)throw Error('manual-recording-unavailable');if(config.waitForManualFirstRound||config.waitForActivityFirstRound)throw Error('computed-manual-checkpoint-unavailable');
  if('computedRecovery' in config||'recoveryGoals' in config)throw Error('invalid-computed-start');
  const recovery=captureRecoveryPreferences(config),strategy=config.recoveryStrategy??'success-first';
  if(!['success-first','time-first'].includes(strategy))throw Error('invalid-recovery-strategy');
  const uuids=[...new Set(config.actorUUIDs??[])];if(!uuids.length)throw Error('invalid-recovery-actors');
  const budget=config.budgetSeconds??7200,max=config.maxActivities??100;if(!Number.isFinite(budget)||budget<=0||!Number.isInteger(max)||max<1||max>100)throw Error('invalid-recovery-budget');
  await pauseGame();check();const actors=await prepareActors(uuids);check();const resolved=resolveRecoveryGoals(actors,recovery);
  const goals=config.goalsByPool&&recovery===undefined?copy(config.goalsByPool):resolved.goalsByPool;
  if(goals.some(goal=>!actors.some(actor=>actor.pool.poolUUID===goal.poolUUID&&Number.isFinite(goal.targetHP)&&goal.targetHP>=0&&goal.targetHP<=actor.hp.max)))throw Error('invalid-recovery-goals');
  const at=game.time.worldTime;
  const session=await ledger.createSession({id:config.id??crypto.randomUUID(),executionMode:computedMode,actorUUIDs:uuids,startedAt:at,cursorAt:at,budgetEndsAt:at+budget,maxActivities:max,goalsByPool:goals,
   ...config.goalsByPool&&recovery===undefined?{}:{recoveryGoals:resolved.recoveryGoals},computedInitialActors:compactActors(actors,{stamps:false}),recoveryStrategy:strategy,treatmentRank:config.treatmentRank??'auto',riskySurgery:config.riskySurgery===true,requireFullFocus:config.requireFullFocus===true,extendTreatment:config.extendTreatment===true,status:'running',assumptions:['different-actors-may-overlap']},
   {guard:()=>{check();return game.time.worldTime===at&&!game.combat?.started}});
  accept(session);onChange(session.id);if(config.autoRun!==false)void drive(session.id).catch(error=>onChange({error:error.message}));return session;
 }
 async function advance(session,to,scope){
  if(to===game.time.worldTime)return session;
  const receipt=await clock.advanceTo({id:crypto.randomUUID(),sessionId:session.id,from:session.cursorAt,to},{...scopeOptions(scope),guard:()=>guard(session.id,scope)});
  if(receipt.status!=='confirmed')throw Error(receipt.reason??'computed-clock-unconfirmed');
  const saved=await ledger.updateSession(session.id,{cursorAt:to},scopeOptions(scope));guard(session.id,scope);if(game.time.worldTime!==to)throw Error('external-world-time-change');return saved;
 }
 async function commit(session,batch,simulation,scope,{advanceTime=false}={}){
  const id=crypto.randomUUID(),payload={id,from:batch.from,to:batch.to,beforeActors:compactActors(batch.beforeActors),afterActors:compactActors(batch.afterActors),events:batch.events};
  payload.documents=await documents.prepare(payload);
  await ledger.claimComputedBatch(session.id,payload,{...scopeOptions(scope),guard:()=>guard(session.id,scope)});
  const receipts={};
  try{
   if(advanceTime)session=await advance(session,batch.to,scope);
   await owned(session.id,scope);const options={guard:()=>guard(session.id,scope)};
   receipts.actors=await documents.applyActors(payload.documents,options);
   await owned(session.id,scope);
   receipts.items=await documents.applyItems(payload.documents,options);
   const state=simulation.snapshot(),summary=computedSummary(session,state,batch.afterActors,state.stopReason);
   await ledger.transitionComputedBatch(session.id,id,{expected:['claimed'],state:'confirmed',receipts,summary,simulationState:simulation.checkpoint()},{...scopeOptions(scope),...options});
  }catch(error){
   if(isAuthority())await ledger.transitionComputedBatch(session.id,id,{expected:['claimed'],state:'uncertain',receipts,reason:error.message},scopeOptions(scope)).catch(()=>{});
   throw error;
  }
 }
 async function run(id,scope){
  let session=computed(await owned(id,scope)),actors=await prepareActors(session.actorUUIDs);guard(id,scope);
  validateGoalBasis(session,actors);
  const simulation=simulationFactory({actors,session,state:session.computedRecovery.simulationState});
  while(true){
   session=await owned(id,scope);if(stops.has(id))return pause(id,stops.get(id),scope);
   if(game.time.worldTime!==session.cursorAt)throw Error('external-world-time-change');
   const boundary=await timeBoundary(session.actorUUIDs),next=simulation.next({until:boundary.until});guard(id,scope);
   if(next.done){
    const reason=next.reason??'blocked',status=reason==='goals-met'?'complete':'paused';
    const state=simulation.snapshot(),summary=computedSummary(session,state,state.actors,reason);
    await ledger.finishComputedSession(id,{status,stopReason:reason,summary,simulationState:simulation.checkpoint()},{...scopeOptions(scope),guard:()=>guard(id,scope)});scopes.delete(id);onChange(id);return;
   }
   if(boundary.segmented){
    session=await advance(session,next.to,scope);actors=await prepareActors(session.actorUUIDs);guard(id,scope);validateGoalBasis(session,actors);
    const batch=simulation.settle({actors});await commit(session,batch,simulation,scope);
   }else{
    await ledger.beginComputedCalculation(id,{...scopeOptions(scope),guard:()=>guard(id,scope)});guard(id,scope);
    const from=session.cursorAt,beforeActors=copy(actors),events=[];let batch=simulation.settle(),count=0;events.push(...batch.events);
    while(true){const upcoming=simulation.next();if(upcoming.done)break;batch=simulation.settle();events.push(...batch.events);if(++count%8===0){await new Promise(resolve=>setTimeout(resolve,0));guard(id,scope);if(stops.has(id))break}}
    const state=simulation.snapshot();batch={from,to:state.cursorAt,beforeActors,afterActors:state.actors,events};
    await commit(session,batch,simulation,scope,{advanceTime:true});actors=state.actors;
   }
   onChange(id);
  }
 }
 async function drive(id){
  if(runners.has(id))return runners.get(id);const scope=scopeFor(id);if(!scope)return;
  const pending=run(id,scope).catch(async error=>{if(isAuthority())await pause(id,error.message,scope).catch(()=>{});return {status:'paused',reason:error.message}}).finally(()=>{runners.delete(id);stops.delete(id)});
  runners.set(id,pending);return pending;
 }
 async function stop(id,reason='user-stopped'){check();const session=computed(await ledger.getSession(id));if(!['running','paused'].includes(session.status))return session;if(runners.has(id)){stops.set(id,reason);return session}return pause(id,reason)}
 async function stopActive(id,reason){check();const session=computed(await ledger.getSession(id));return session.status==='running'?stop(id,reason):session}
 async function resume(id,{autoRun=true}={}){
  check();const session=computed(await ledger.getSession(id));if(session.status!=='paused')throw Error('paused-automatic-session-required');if(unresolved(session))throw Error('unresolved-computed-batch');if(game.time.worldTime!==session.cursorAt)throw Error('external-world-time-change');
  await pauseGame();check();if(game.time.worldTime!==session.cursorAt)throw Error('external-world-time-change');
  validateGoalBasis(session,await prepareActors(session.actorUUIDs));check();if(game.time.worldTime!==session.cursorAt)throw Error('external-world-time-change');
  const saved=await ledger.resumeSession(id,{cursorAt:session.cursorAt});accept(saved);stops.delete(id);onChange(id);if(autoRun)void drive(id);return saved;
 }
 async function takeover(id){check();computed(await ledger.getSession(id));scopes.delete(id);stops.set(id,'explicit-driver-takeover');clock.stop?.('explicit-driver-takeover',{sessionId:id});const result=await ledger.takeoverSession(id);onChange(id);return result}
 function describeSnapshot(data,actors){
  const session=copy(data.session),records=new Map(data.activities.filter(activity=>activity.kind==='computed-recovery-batch').map(activity=>[activity.id,activity]));
  if(session){
   session.computedRecovery.batches=session.computedRecovery.batches.map(index=>{
    const batch=index.kind==='computed-recovery-batch'?index:records.get(index.id);
    if(!batch)throw Error('computed-batch-unavailable');return copy(batch);
   });
   if(session.computedRecovery.summary)session.computedRecovery.summary.events=session.computedRecovery.batches.filter(batch=>batch.state==='confirmed').flatMap(batch=>batch.events??[]);
  }
  return {...data,session,activities:data.activities.filter(activity=>activity.kind!=='computed-recovery-batch'),actors,worldTime:game.time.worldTime,canAcceptCurrentTime:false,canResumeComputed:session?.status==='paused'&&!unresolved(session)&&game.time.worldTime===session.cursorAt};
 }
 async function snapshot(id){const data=await ledger.snapshot(id),actors=data.session?await capabilities.snapshot(data.session.actorUUIDs):[];return describeSnapshot(data,actors)}
 async function reconcile(id){check();const session=computed(await ledger.getSession(id));const readbacks=[];for(const index of session.computedRecovery.batches.filter(batch=>batch.state!=='confirmed')){const batch=await ledger.getComputedBatch(id,index.id);readbacks.push({batchId:batch.id,...await documents.inspect(batch.documents)})}return {...await snapshot(id),computedReadbacks:readbacks}}
 async function review(id,{note,userId=game.user.id}={}){check();if(runners.has(id))throw Error('session-in-flight');if(!note?.trim())throw Error('review-note-required');computed(await ledger.getSession(id));const result=await ledger.reviewComputedSession(id,{note,userId,at:game.time.worldTime});onChange(id);return result}
 return {start,drive,stop,stopActive,resume,takeover,snapshot,describeSnapshot,reconcile,review,executionScope:scopeFor,invalidate:reason=>{for(const id of scopes.keys()){stops.set(id,reason);scopes.delete(id)}}};
}
