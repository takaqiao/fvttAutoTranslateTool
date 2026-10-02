import {recoveryProposals,checkpointDeclaration,scheduleCheckpointActivities} from './policy.mjs';
import {extensionPatients} from './treatment.mjs';
import {clone,normalizeNativeOwnerMap,checkpointBinding,sameCheckpoint,manualSourceIntent,activityCheckpointBinding,sameActivityCheckpoint} from './schema.mjs';
import {WORKBENCH_SOURCE_SHA} from './manual-events.mjs';
export function createCoordinator({ledger,capabilities,providers,clock,policy,isAuthority,now=()=>globalThis.game.time.worldTime,ownerOperations,onChange=()=>{},game=globalThis.game,fromUuid=globalThis.fromUuid,getHpPool,manualEvents}){
 const byId=new Map(providers.map(p=>[p.id,p])),locks=new Map(),contexts=new Map(),runners=new Map(),leases=new Map(),generations=new Map(),checkpointRequests=new Map();
 const atomic=ledger.atomic===true;
 const check=()=>{if(!isAuthority()||game?.user?.active===false)throw Error('active-gm-required')};
 const scopeOptions=scope=>atomic?{leaseNonce:scope?.leaseNonce}:{};
 function cancelCheckpointRequest(id,reason){const request=checkpointRequests.get(id);if(request){checkpointRequests.delete(id);request.reject(Error(reason))}}
 const invalidate=(id,reason='session-driver-changed')=>{cancelCheckpointRequest(id,reason);leases.delete(id);generations.set(id,(generations.get(id)??0)+1)};
 function currentScope(id,scope){check();if(atomic&&(!scope||leases.get(id)!==scope))throw Error('session-driver-required')}
 async function owned(id,scope,{running=true}={}){
  currentScope(id,scope);const session=await ledger.getSession(id);currentScope(id,scope);
  if(atomic&&!ledger.ownsSession(session,scope))throw Error('session-driver-required');
  if(running&&session?.status!=='running')throw Error('recovery-session-stopped');return session;
 }
 async function nativeOwnerEligibility(session,activities){
  const map=session.nativeOwnerByActor??{},requirements=[];
  for(const activity of [...Object.keys(map).map(actorUUID=>({actorUUID})),...activities??[]]){
   const ownerId=Object.hasOwn(map,activity.actorUUID)?map[activity.actorUUID]:undefined;if(ownerId===undefined)continue;
   if(ownerId!==game?.user?.id&&activity.options?.threePecks)throw Error('remote-three-pecks-unsupported');
   if(activity.patientUUIDs?.length){
    const patients=await capabilities.snapshot(activity.patientUUIDs),actualPools=new Set();
    for(const uuid of activity.patientUUIDs){const patient=patients.find(p=>p.actorUUID===uuid);if(!patient?.pool?.ready)throw Error('native-owner-pool-unavailable');actualPools.add(patient.pool.poolUUID)}
    if(actualPools.size!==new Set(activity.hpPoolUUIDs).size||[...actualPools].some(uuid=>!activity.hpPoolUUIDs.includes(uuid)))throw Error('native-owner-pool-changed');
   }
   requirements.push({ownerId,patientUUIDs:activity.patientUUIDs??[],hpPoolUUIDs:activity.hpPoolUUIDs??[],uuids:[...new Set([activity.actorUUID,...activity.patientUUIDs??[],...activity.hpPoolUUIDs??[]])]});
  }
  const documents=new Map();
  for(const uuid of new Set(requirements.flatMap(r=>r.uuids))){if(typeof fromUuid!=='function')throw Error('native-owner-document-unavailable');documents.set(uuid,await fromUuid(uuid))}
  // Return a synchronous check so later ledger/context awaits cannot hide a permission change.
  const validate=()=>{check();for(const {ownerId,uuids,patientUUIDs,hpPoolUUIDs} of requirements){
   const user=game?.users?.get(ownerId);if(!user?.active)throw Error('native-owner-unavailable');
   for(const uuid of uuids){const actor=documents.get(uuid);if(!actor||actor.uuid!==uuid||actor.testUserPermission?.(user,'OWNER')!==true)throw Error('native-owner-permission-required');
    if(game?.actors&&uuid===`Actor.${actor.id}`&&game.actors.get(actor.id)!==actor)throw Error('native-owner-document-unavailable');
   }
   if(patientUUIDs.length){
    const currentPools=new Set();
    for(const uuid of patientUUIDs){const pool=getHpPool?.(documents.get(uuid));if(!pool?.ready)throw Error('native-owner-pool-unavailable');currentPools.add(pool.poolUUID)}
    if(currentPools.size!==new Set(hpPoolUUIDs).size||[...currentPools].some(uuid=>!hpPoolUUIDs.includes(uuid)))throw Error('native-owner-pool-changed');
   }
  }};
  validate();return validate;
 }
 function accept(id,session,generation){
  check();if(!atomic)return;
  if((generations.get(id)??0)!==generation)return;
  const scope=Object.freeze({leaseNonce:session.driver?.leaseNonce});
  if(!scope.leaseNonce||!ledger.ownsSession(session,scope))throw Error('session-driver-required');leases.set(id,scope);
 }
 async function snapshot(id){const data=await ledger.snapshot(id);return {...data,...checkpointRequests.has(id)?{activityCheckpointRequested:true}:{},actors:data.session?await capabilities.snapshot(data.session.actorUUIDs):[]}}
 async function pause(id,reason,scope,explicit=false){
  check();if(!explicit)await owned(id,scope);
  if(!explicit)currentScope(id,scope);else invalidate(id,reason);
  if(explicit)clock.stop?.(reason,{sessionId:id});const options=explicit?{}:{...scopeOptions(scope),expectedStatus:'running'};
  const result=await ledger.updateSession(id,{status:'paused',stopReason:reason},options);
  if(!explicit)clock.stop?.(reason,{sessionId:id,...scopeOptions(scope)});
  if(!explicit){currentScope(id,scope);await owned(id,scope,{running:false})}
  const data=await ledger.snapshot(id);check();if(!explicit)currentScope(id,scope);
  const stoppedActivities=new Set(result.activityIds);
  for(const a of data.activities.filter(a=>(!atomic||stoppedActivities.has(a.id))&&['planned','started','completing'].includes(a.state))){
   await ownerOperations.cancelActivity?.(a);if(a.state==='completing')continue;
   await byId.get(a.providerId)?.cancel?.(a,contexts.get(a.id));check();if(!explicit)currentScope(id,scope);
   try{await ledger.transitionActivity(a.id,{expected:['planned','started'],patch:{state:'cancelled',reason},...options});contexts.delete(a.id)}catch(error){if(error.message!=='state-conflict')throw error}
  }
  if(!explicit&&leases.get(id)===scope)invalidate(id,reason);onChange(id);return result;
 }
 const stop=(id,reason='user-stopped')=>pause(id,reason,undefined,true);
 async function addActivity(id,input,scope=leases.get(id)){
  input=clone(input);const session=await owned(id,scope),map=session.nativeOwnerByActor??{},ownerId=Object.hasOwn(map,input.actorUUID)?map[input.actorUUID]:undefined;
  if(session.manualCheckpoint&&session.manualCheckpoint.phase!=='settled'&&(session.manualCheckpoint.phase!=='sealed'||input.providerId!=='refocus'||input.options?.threePecks||input.patientUUIDs?.length||input.startedAt!==session.manualCheckpoint.from||input.endsAt!==session.manualCheckpoint.to))throw Error('manual-checkpoint-activity-unavailable');
  const proposal={...input,id:input.id??crypto.randomUUID(),sessionId:id,state:'planned',source:{type:'coordinator',...ownerId===undefined?{}:{ownerId}}};
  const eligible=await nativeOwnerEligibility(session,[proposal]);await owned(id,scope);eligible();
  const activity=await ledger.insertActivity(proposal,scopeOptions(scope));
  await owned(id,scope);const provider=byId.get(activity.providerId);if(!provider)throw Error('unknown-provider');
  const ctx=await ownerOperations.createActivityContext(activity);contexts.set(activity.id,ctx);
  try{
   await owned(id,scope);eligible();const begun=await provider.begin(activity,ctx);await owned(id,scope);
   const saved=await ledger.transitionActivity(activity.id,{expected:['planned'],patch:{state:begun.status==='started'?'started':'blocked',reason:begun.reason},...scopeOptions(scope)});
   await owned(id,scope);if(saved.state!=='started')contexts.delete(saved.id);return saved;
  }catch(error){await provider.cancel?.(activity,ctx);contexts.delete(activity.id);throw error}
 }
 async function openManualCheckpoint(id){
  const scope=leases.get(id),session=await owned(id,scope);if(!atomic)throw Error('atomic-manual-checkpoint-required');
  if(locks.has(id)||runners.has(id))throw Error('session-in-flight');
  if(session.manualCheckpoint){if(session.manualCheckpoint.phase!=='open')throw Error('manual-checkpoint-closed');return checkpointBinding(session.manualCheckpoint)}
  const data=await ledger.snapshot(id);await owned(id,scope);
  if(now()!==session.cursorAt||data.activities.length||data.clocks.length||session.cursorAt+600>session.budgetEndsAt)throw Error('manual-checkpoint-unavailable');
  const c={id:crypto.randomUUID(),sessionId:id,rootUUID:session.protocol.rootUUID,epoch:session.protocol.epoch,observationNonce:crypto.randomUUID(),from:session.cursorAt,to:session.cursorAt+600,phase:'open'};
  await ledger.updateSession(id,{manualCheckpoint:c},{...scopeOptions(scope),expectedStatus:'running'});await owned(id,scope);onChange(id);return checkpointBinding(c);
 }
 const declarationBinding=checkpoint=>activityCheckpointBinding(Object.fromEntries(['id','sessionId','rootUUID','epoch','observationNonce','from'].map(key=>[key,checkpoint[key]])));
 async function declarationEligibility(session,scope,at){
  const entries=[];
  for(const row of Object.values(session.activityCheckpoint?.registrations??{})){
   if(['completed','cancelled','interrupted'].includes(row.status))continue;
   entries.push({row,actor:await fromUuid?.(row.declaration.actorUUID)});await owned(session.id,scope);
  }
  const guard=currentSession=>{currentScope(session.id,scope);if(now()!==at)return false;if(currentSession&&(!sameActivityCheckpoint(currentSession.activityCheckpoint,declarationBinding(session.activityCheckpoint))||currentSession.activityCheckpoint.phase!==session.activityCheckpoint.phase||entries.some(({row})=>!currentSession.actorUUIDs.includes(row.declaration.actorUUID))))return false;for(const {row,actor} of entries){const user=game?.users?.get(row.source.userId);if(!user?.active||actor?.uuid!==row.declaration.actorUUID||actor.testUserPermission?.(user,'OWNER')!==true||game?.actors&&game.actors.get(actor.id)!==actor)return false}return true};
  if(!guard())throw Error('activity-checkpoint-changed');return guard;
 }
 async function openActivityCheckpointAtBoundary(id,scope,runner,request){
  if(!atomic)throw Error('atomic-activity-checkpoint-required');await owned(id,scope);
  if(locks.has(id)||runners.has(id)&&runners.get(id)!==runner)throw Error('session-in-flight');const lock={scope};locks.set(id,lock);let submitted=false;
  const current=()=>{currentScope(id,scope);if(request&&checkpointRequests.get(id)!==request)throw Error('activity-checkpoint-request-cancelled');if(game?.combat?.started)throw Error('encounter-started')};
  try{
   const session=await owned(id,scope);current();if(now()!==session.cursorAt)throw Error('external-world-time-change');
   if(session.activityCheckpoint?.phase==='open')return declarationBinding(session.activityCheckpoint);
   if(session.activityCheckpoint?.phase==='sealed')throw Error('activity-checkpoint-already-opened');
   const binding={id:crypto.randomUUID(),sessionId:id,rootUUID:session.protocol.rootUUID,epoch:session.protocol.epoch,observationNonce:crypto.randomUUID(),from:session.cursorAt};
   submitted=true;await ledger.openActivityCheckpoint(binding,{...scopeOptions(scope),guard:()=>{current();return locks.get(id)===lock&&now()===binding.from}});await owned(id,scope);current();onChange(id);return binding;
  }catch(error){if(submitted)invalidate(id,error.message);throw error}finally{if(locks.get(id)===lock)locks.delete(id)}
 }
 async function openActivityCheckpoint(id){
  const scope=leases.get(id);currentScope(id,scope);if(!atomic)throw Error('atomic-activity-checkpoint-required');
  const runner=runners.get(id);
  if(runner){
   if(runner.scope!==scope)throw Error('session-driver-required');
   const prior=checkpointRequests.get(id);if(prior)return prior.promise;
   let resolve,reject;const promise=new Promise((yes,no)=>{resolve=yes;reject=no});promise.catch(()=>{});
   checkpointRequests.set(id,{scope,runner,promise,resolve,reject});onChange(id);return promise;
  }
  return openActivityCheckpointAtBoundary(id,scope);
 }
 async function consumeCheckpointRequest(id,runner,result){
  const request=checkpointRequests.get(id);if(!request)return false;
  if(request.runner!==runner)return false;
  if(request.scope!==runner.scope||!['running','waiting-activities'].includes(result.status)){cancelCheckpointRequest(id,result.reason??'activity-checkpoint-unavailable');return false}
  try{const binding=await openActivityCheckpointAtBoundary(id,runner.scope,runner,request);if(checkpointRequests.get(id)!==request)return false;checkpointRequests.delete(id);request.resolve(binding);return true}
  catch(error){cancelCheckpointRequest(id,error.message);throw error}
 }
 async function closeActivityCheckpoint(input,{autoRun=true}={}){
  const binding=activityCheckpointBinding(input),id=binding.sessionId,scope=leases.get(id);await owned(id,scope);
  if(locks.has(id)||runners.has(id))throw Error('session-in-flight');const lock={scope};locks.set(id,lock);let result,submitted=false;
  try{
   const session=await owned(id,scope);
   if(!sameActivityCheckpoint(session.activityCheckpoint,binding)||session.activityCheckpoint.phase!=='open')throw Error('activity-checkpoint-closed');
   const data=await ledger.snapshot(id);await owned(id,scope);
   scheduleCheckpointActivities({registrations:session.activityCheckpoint.registrations,activities:data.activities,session,from:binding.from});
   const guard=await declarationEligibility(session,scope,binding.from);await owned(id,scope);
   submitted=true;result=await ledger.sealActivityCheckpoint(binding,{...scopeOptions(scope),registrations:session.activityCheckpoint.registrations,guard});await owned(id,scope);onChange(id);
  }catch(error){if(submitted)invalidate(id);throw error}finally{if(locks.get(id)===lock)locks.delete(id)}
  if(autoRun)void drive(id).catch(error=>onChange({error:error.message}));return result;
 }
 async function advanceDeclarations(session,scope){
  if(session.activityCheckpoint?.phase!=='sealed')return;
  const at=now(),guard=await declarationEligibility(session,scope,at);await owned(session.id,scope);
  await ledger.advanceActivityCheckpoint(declarationBinding(session.activityCheckpoint),{...scopeOptions(scope),at,guard});await owned(session.id,scope);
 }
 async function manualCheckpointOptions(binding){
  const scope=leases.get(binding?.sessionId),session=await owned(binding?.sessionId,scope);
  if(!sameCheckpoint(session.manualCheckpoint,binding)||!['open','sealed','advancing'].includes(session.manualCheckpoint.phase))throw Error('manual-checkpoint-mismatch');
  return {...scopeOptions(scope),checkpointBinding:clone(binding)};
 }
 async function reserveManualSource(binding,intent,callerId){
  intent=manualSourceIntent(intent);const scope=leases.get(binding?.sessionId),session=await owned(binding?.sessionId,scope),c=session.manualCheckpoint;
  if(!sameCheckpoint(c,binding)||c.phase!=='open')throw Error('manual-checkpoint-closed');
  if(now()!==c.from||locks.has(session.id))throw Error('manual-checkpoint-unavailable');
  const user=game?.users?.get(callerId),actor=await fromUuid?.(intent.actorUUID),patient=await fromUuid?.(intent.patientUUID),actors=await capabilities.snapshot([intent.actorUUID,intent.patientUUID]);await owned(session.id,scope);
  const healer=actors.find(a=>a.actorUUID===intent.actorUUID),target=actors.find(a=>a.actorUUID===intent.patientUUID),pool=getHpPool?.(patient)??target?.pool;
  if(!user?.active||!actor||actor.uuid!==intent.actorUUID||actor.testUserPermission?.(user,'OWNER')!==true||!session.actorUUIDs.includes(intent.actorUUID)||!session.actorUUIDs.includes(intent.patientUUID))throw Error('manual-actor-owner-required');
  if(healer?.isDead||healer?.unconscious||target?.isDead||!patient||patient.uuid!==intent.patientUUID||!pool?.ready||pool.poolUUID!==patient.uuid||target?.pool?.poolUUID!==patient.uuid||now()!==c.from)throw Error('manual-checkpoint-patient-unavailable');
  const id=crypto.randomUUID(),a={id,sessionId:session.id,providerId:'manual',actorUUID:intent.actorUUID,patientUUIDs:[intent.patientUUID],hpPoolUUIDs:[intent.patientUUID],state:'awaiting-evidence',startedAt:c.from,endsAt:c.to,durationSeconds:600,kind:'treatment',order:0,treatmentImmunitySeconds:healer.continualRecovery?600:3600,checkpointBinding:clone(binding),temporalSource:{type:'checkpoint-reservation'},source:{type:'workbench',sourceSHA:WORKBENCH_SOURCE_SHA,lexicalSource:true,manual:true,reservationId:id,useId:intent.useId,userId:callerId},options:{continualRecovery:!!healer.continualRecovery,riskySurgery:!!intent.riskySurgery,missing:['native-application-receipt','native-immunity-receipt','checkpoint-time-confirmation']},proof:{useId:intent.useId,checkIds:[],resultIds:[],receiptIds:[],immunityIds:[]}};
  await ledger.insertActivity(a,{...scopeOptions(scope),checkpointBinding:binding});await owned(session.id,scope);onChange(session.id);return {reservationId:id,activityId:id,checkpointBinding:clone(binding)};
 }
 async function closeManualCheckpoint(binding,{autoRun=true}={}){
  const id=binding?.sessionId,scope=leases.get(id),session=await owned(id,scope),c=session.manualCheckpoint;
  if(!sameCheckpoint(c,binding)||c.phase!=='open')throw Error('manual-checkpoint-closed');
  if(locks.has(id)||runners.has(id))throw Error('session-in-flight');const lock={scope};locks.set(id,lock);let result;
  try{
   if(now()!==c.from)return await pause(id,'external-world-time-change',scope);
   const flushed=await manualEvents?.flushCheckpoint(binding);await owned(id,scope);
   if(flushed?.status!=='ready'){onChange(id);return {status:'awaiting-evidence',missing:flushed?.missing??['native-manual-source-unavailable']}}
   await ledger.updateSession(id,{manualCheckpoint:{...c,phase:'sealed'}},scopeOptions(scope));await owned(id,scope);
   const data=await snapshot(id);await owned(id,scope);
   const proposals=recoveryProposals({actors:data.actors,activities:data.activities,session,now:c.from,providerIds:[...byId.keys()]}).filter(p=>p.providerId==='refocus'&&p.durationSeconds===600&&!p.options?.threePecks&&!p.patientUUIDs.length&&p.earliestStart<=c.from);
   const next=policy({snapshot:data,proposals,session,now:c.from});
   for(const proposal of next.activities.slice(0,session.maxActivities)){const added=await addActivity(id,proposal,scope);if(added.state==='blocked')return await pause(id,added.reason,scope)}
   await ledger.updateSession(id,{manualCheckpoint:{...c,phase:'advancing'}},scopeOptions(scope));await owned(id,scope);
   const pending=await ledger.snapshot(id),eligible=await nativeOwnerEligibility(pending.session,pending.activities.filter(a=>a.state==='started'));await owned(id,scope);eligible();
   const sealedEvidence=await manualEvents.flushCheckpoint(binding);await owned(id,scope);if(sealedEvidence.status!=='ready')return await pause(id,'unresolved-manual-evidence',scope);
   const receipt=await clock.advanceTo({id:c.id,sessionId:id,from:c.from,to:c.to},scopeOptions(scope));await owned(id,scope);if(receipt.status!=='confirmed')return await pause(id,receipt.reason??'clock-unconfirmed',scope);
   await ledger.updateSession(id,{cursorAt:now()},scopeOptions(scope));await owned(id,scope);
   const current=await ledger.snapshot(id);await owned(id,scope);
   for(const a of current.activities.filter(a=>!a.source.manual&&a.state==='started'&&a.endsAt===c.to)){
    const claimed=await ledger.transitionActivity(a.id,{expected:['started'],patch:{state:'completing'},...scopeOptions(scope)});await owned(id,scope);
    let completion;try{completion=await byId.get(a.providerId).complete(claimed,contexts.get(a.id));await owned(id,scope)}catch(error){await owned(id,scope);completion={status:'uncertain',reason:error.message,...error.proof?{proof:error.proof}:{}}}
    await ledger.transitionActivity(a.id,{expected:['completing'],patch:{...completion,state:completion.status==='confirmed'?'confirmed':completion.status==='blocked'?'blocked':'uncertain'},...scopeOptions(scope)});contexts.delete(a.id);await owned(id,scope);
    if(completion.status!=='confirmed')return await pause(id,completion.reason??'native-result-unconfirmed',scope);
   }
   const finalEvidence=await manualEvents.flushCheckpoint(binding,{afterAdvance:true});await owned(id,scope);if(finalEvidence.status!=='ready')return await pause(id,'unresolved-manual-evidence',scope);
   for(const activityId of finalEvidence.activityIds){const a=await ledger.getActivity(activityId);await owned(id,scope);await ledger.transitionActivity(a.id,{expected:['awaiting-evidence'],patch:{state:'confirmed',options:{...a.options,missing:a.options.missing.filter(m=>m!=='checkpoint-time-confirmation')}},...scopeOptions(scope),checkpointBinding:binding});await owned(id,scope)}
   await ledger.updateSession(id,{manualCheckpoint:{...c,phase:'settled'}},scopeOptions(scope));await owned(id,scope);onChange(id);result={status:'running'};
  }catch(error){try{await owned(id,scope);await pause(id,error.message,scope)}catch{}throw error}
  finally{if(locks.get(id)===lock)locks.delete(id)}
  if(autoRun)void drive(id).catch(error=>onChange({error:error.message}));return result;
 }
 async function step(id){check();const scope=leases.get(id);if(atomic&&!scope)return {status:'observing'};if(locks.has(id)&&locks.get(id).scope===scope)return {status:'busy'};const lock={scope};locks.set(id,lock);
  try{
   const initial=await owned(id,scope,{running:false});if(initial?.status!=='running')return {status:initial?.status??'missing'};
   let data=await snapshot(id),s=data.session;await owned(id,scope);
   if(now()!==s.cursorAt)return await pause(id,'external-world-time-change',scope);
   if(s.activityCheckpoint?.phase==='open')return {status:'waiting-activities',checkpointBinding:declarationBinding(s.activityCheckpoint)};
   if(s.manualCheckpoint?.phase==='open')return {status:'waiting-manual',checkpointBinding:checkpointBinding(s.manualCheckpoint)};
   if(data.activities.some(a=>['uncertain','awaiting-evidence','completing'].includes(a.state))||data.clocks.some(c=>c.state!=='confirmed'))return await pause(id,'unresolved-evidence',scope);
   if(now()!==s.cursorAt)return await pause(id,'external-world-time-change',scope);
   if(s.activityCheckpoint?.phase==='sealed'){await advanceDeclarations(s,scope);data=await snapshot(id);s=data.session;await owned(id,scope)}
   const active=data.activities.filter(a=>a.state==='started'),pendingDeclarations=data.activities.filter(a=>checkpointDeclaration(a)&&['planned','started'].includes(a.state));
   const goalsMet=s.goalsByPool.every(g=>data.actors.some(a=>a.pool.poolUUID===g.poolUUID&&a.hp.value>=g.targetHP));
   if(!active.length&&!pendingDeclarations.length&&goalsMet&&(!s.requireFullFocus||data.actors.every(a=>a.focus.value>=a.focus.max))){await ledger.updateSession(id,{status:'complete',stopReason:'goals-met'},scopeOptions(scope));if(leases.get(id)===scope)invalidate(id);onChange(id);return {status:'complete'}}
   const activityCount=data.activities.filter(a=>!a.source.manual).length;
   if((activityCount>=s.maxActivities&&!active.length&&!pendingDeclarations.length)||now()>=s.budgetEndsAt)return await pause(id,'budget',scope);
   const proposals=activityCount>=s.maxActivities?[]:recoveryProposals({actors:data.actors,activities:data.activities,session:s,now:now(),providerIds:[...byId.keys()]});
   const next=policy({snapshot:data,proposals,session:s,now:now()});
   for(const proposal of next.activities.slice(0,s.maxActivities-data.activities.filter(a=>!a.source.manual).length)){await owned(id,scope);const added=await addActivity(id,{...proposal,source:{type:'coordinator'},groupId:undefined},scope);if(added.state==='blocked')return await pause(id,added.reason,scope)}
   if(next.checkpointAt===null)return await pause(id,data.actors.some(a=>a.refocusUnsupported?.length&&(a.threePecks||s.requireFullFocus&&a.focus.value<a.focus.max))?'refocus-recovery-unadapted':next.reason,scope);
   const pending=await ledger.snapshot(id),eligible=await nativeOwnerEligibility(pending.session,pending.activities.filter(a=>a.state==='started'));
   const declared=pending.session.activityCheckpoint?.phase==='sealed'?await declarationEligibility(pending.session,scope,now()):null;await owned(id,scope);eligible();if(declared&&!declared())throw Error('activity-checkpoint-changed');
   const receipt=await clock.advanceTo({id:crypto.randomUUID(),sessionId:id,from:now(),to:next.checkpointAt},{...scopeOptions(scope),guard:session=>{currentScope(id,scope);eligible();return declared?declared(session):true}});await owned(id,scope);if(receipt.status!=='confirmed')return await pause(id,receipt.reason??'clock-unconfirmed',scope);
   await ledger.updateSession(id,{cursorAt:now()},scopeOptions(scope));await owned(id,scope);data=await snapshot(id);await owned(id,scope);
   for(const a of data.activities.filter(a=>!checkpointDeclaration(a)&&a.state==='started'&&a.endsAt<=now())){
    await owned(id,scope);const claimed=await ledger.transitionActivity(a.id,{expected:['started'],patch:{state:'completing'},...scopeOptions(scope)});await owned(id,scope);
    let result;try{result=await byId.get(a.providerId).complete(claimed,contexts.get(a.id));await owned(id,scope)}catch(error){await owned(id,scope);result={status:'uncertain',reason:error.message,...error.proof?{proof:error.proof}:{}}}
    await owned(id,scope);await ledger.transitionActivity(a.id,{expected:['completing'],patch:{...result,state:result.status==='confirmed'?'confirmed':result.status==='blocked'?'blocked':'uncertain'},...scopeOptions(scope)});contexts.delete(a.id);await owned(id,scope);
    if(result.status!=='confirmed')return await pause(id,result.reason??'native-result-unconfirmed',scope);
    const freshLedger=await ledger.snapshot(id);await owned(id,scope);const remainingSlots=s.maxActivities-freshLedger.activities.filter(row=>!row.source.manual).length;
    if(remainingSlots>0&&s.extendTreatment&&a.providerId==='treat-wounds'&&!a.options.extensionOf&&extensionPatients({...a,...result}).length&&a.startedAt+3600<=s.budgetEndsAt){
     const eligible=new Set(extensionPatients({...a,...result}).map(r=>r.patientUUID??a.patientUUIDs[0])),fresh=(await capabilities.snapshot(a.patientUUIDs)).filter(p=>eligible.has(p.actorUUID)&&p.hp.value<(s.goalsByPool.find(g=>g.poolUUID===p.pool.poolUUID)?.targetHP??0));await owned(id,scope);if(fresh.length)await addActivity(id,{providerId:a.providerId,actorUUID:a.actorUUID,patientUUIDs:fresh.map(p=>p.actorUUID),hpPoolUUIDs:[...new Set(fresh.map(p=>p.pool.poolUUID))],startedAt:now(),endsAt:a.startedAt+3600,options:{...a.options,extensionOf:a.id},source:{type:'coordinator'}},scope);
    }
   }
   if(s.activityCheckpoint?.phase==='sealed')await advanceDeclarations(await ledger.getSession(id),scope);
   onChange(id);return {status:'running'};
  }catch(error){
   if(isAuthority())try{await owned(id,scope);await pause(id,error.message,scope);return {status:'paused',reason:error.message}}catch{}
   return {status:atomic?'observing':'paused',reason:error.message};
  }
  finally{if(locks.get(id)===lock)locks.delete(id)}
 }
 async function drive(id){const scope=leases.get(id);if(atomic&&!scope)return;if(runners.has(id)&&runners.get(id).scope===scope)return;const runner={scope};runners.set(id,runner);try{while(isAuthority()){await owned(id,scope);const result=await step(id);if(await consumeCheckpointRequest(id,runner,result)||result.status!=='running')break;await Promise.resolve()}}finally{if(runners.get(id)===runner){runners.delete(id);cancelCheckpointRequest(id,'activity-checkpoint-driver-ended')}}}
 async function start(config){check();const nativeOwnerByActor=normalizeNativeOwnerMap(config.nativeOwnerByActor,config.actorUUIDs,{manual:config.manual===true});config={...clone(config),nativeOwnerByActor};const actors=await capabilities.snapshot(config.actorUUIDs);check();const at=now();const goals=config.goalsByPool??[...new Map(actors.map(a=>[a.pool.poolUUID,{poolUUID:a.pool.poolUUID,targetHP:a.hp.max}])).values()];
  if(config.waitForManualFirstRound&&!config.manual&&(!atomic||(config.budgetSeconds??7200)<600))throw Error('manual-checkpoint-unavailable');
  if(config.waitForActivityFirstRound&&!config.manual&&(!atomic||config.waitForManualFirstRound))throw Error('activity-checkpoint-start-choice-conflict');
  if(!config.manual){const all=await ledger.all();check();const ids=new Set(config.actorUUIDs),pools=new Set(actors.map(a=>a.pool.poolUUID)),reviewed=row=>!atomic&&row.review&&all.sessions[row.sessionId]?.status==='closed';if(Object.values(all.sessions).some(s=>s.status==='running'))throw Error('recovery-session-already-running');if(Object.values(all.clocks).some(c=>c.state!=='confirmed'&&!reviewed(c))||Object.values(all.activities).some(a=>(!a.source?.manual||a.temporalSource?.type==='checkpoint-reservation')&&!reviewed(a)&&['uncertain','awaiting-evidence','completing','started'].includes(a.state)&&(ids.has(a.actorUUID)||a.patientUUIDs.some(u=>ids.has(u))||a.hpPoolUUIDs.some(u=>pools.has(u)))))throw Error('unresolved-evidence-no-replay')}
  if(!actors.length||goals.some(g=>!actors.some(a=>a.pool.poolUUID===g.poolUUID&&Number.isFinite(g.targetHP)&&g.targetHP>=0&&g.targetHP<=a.hp.max)))throw Error('invalid-recovery-goals');
  const eligible=await nativeOwnerEligibility(config);eligible();
  const id=config.id??crypto.randomUUID(),generation=generations.get(id)??0;
  const s=await ledger.createSession({...config,id,actorUUIDs:[...new Set(config.actorUUIDs)],startedAt:at,cursorAt:at,budgetEndsAt:at+Math.max(0,config.budgetSeconds??7200),maxActivities:Math.min(100,Math.max(1,config.maxActivities??100)),goalsByPool:goals,status:config.manual?'recording':'running',assumptions:config.assumptions??['different-actors-may-overlap']});
  if(!config.manual)accept(id,s,generation);
  if(config.waitForManualFirstRound&&!config.manual){await openManualCheckpoint(id);Object.assign(s,await ledger.getSession(id))}
  if(config.waitForActivityFirstRound&&!config.manual){await openActivityCheckpoint(id);Object.assign(s,await ledger.getSession(id))}
  if(config.autoRun!==false&&!config.manual)void drive(s.id).catch(error=>onChange({error:error.message}));onChange(s.id);return s;
 }
 async function recover(id){check();if(atomic||runners.has(id)||locks.has(id))return ledger.getSession(id);const data=await ledger.snapshot(id);if(!data.session||data.session.status!=='running')return data.session;
  await ledger.updateSession(id,{status:'paused',stopReason:'client-context-lost'});for(const a of data.activities){if(a.state==='planned')await ledger.transitionActivity(a.id,{expected:['planned'],patch:{state:'cancelled',reason:'client-context-lost'}});else if(['started','completing'].includes(a.state))await ledger.transitionActivity(a.id,{expected:[a.state],patch:{state:'uncertain',reason:'client-context-lost'}})}onChange(id);return ledger.getSession(id);
 }
 async function restore(preferred){check();const all=await ledger.all(),sessions=Object.values(all.sessions).filter(s=>s.status!=='closed').sort((a,b)=>b.startedAt-a.startedAt),chosen=sessions.find(s=>s.id===preferred),pending=sessions.find(s=>s.startedAt>=(chosen?.startedAt??-Infinity)&&(Object.values(all.activities).some(a=>a.sessionId===s.id&&(!a.source.manual||a.temporalSource?.type==='checkpoint-reservation'||checkpointDeclaration(a))&&['started','completing','uncertain','awaiting-evidence'].includes(a.state))||Object.values(all.clocks).some(c=>c.sessionId===s.id&&c.state!=='confirmed'))),active=sessions.find(s=>s.status==='running')??pending??sessions.find(s=>s.status==='recording'),s=active??chosen??sessions[0];return s?recover(s.id):null}
 async function reconcile(id){check();const data=await ledger.snapshot(id);if(!data.session||data.session.status!=='paused')throw Error('paused-session-required');
  for(const c of data.clocks.filter(c=>c.state!=='confirmed')){await clock.reconcile(c);check()}
  for(const a of data.activities.filter(a=>!a.source.manual&&['uncertain','completing','started'].includes(a.state))){const result=await ownerOperations.reconcile?.(a);check();if(result?.status==='confirmed')await ledger.transitionActivity(a.id,{expected:[a.state],patch:{...result,state:'confirmed',reconciled:true},reconcile:true})}
  const fresh=await ledger.snapshot(id),confirmed=fresh.clocks.filter(c=>c.state==='confirmed');if(confirmed.length)await ledger.updateSession(id,{cursorAt:Math.max(data.session.cursorAt,...confirmed.map(c=>c.to))},{reconcile:true});onChange(id);return snapshot(id);
 }
 async function review(id,{note,userId}={}){check();if(locks.has(id)||runners.has(id))throw Error('session-in-flight');if(typeof note!=='string'||!note.trim())throw Error('review-note-required');const data=await ledger.snapshot(id);if(data.session?.status!=='paused')throw Error('paused-session-required');const review={note:note.trim().slice(0,2000),userId,at:now(),disposition:'closed-without-replay'};
  for(const a of data.activities.filter(a=>['uncertain','awaiting-evidence','completing','started'].includes(a.state)))await ledger.transitionActivity(a.id,{expected:[a.state],patch:{review}});for(const c of data.clocks.filter(c=>c.state!=='confirmed'))await ledger.transitionClockCommit(c.id,{expected:[c.state],patch:{review}});await ledger.updateSession(id,{status:'closed',stopReason:'gm-reviewed-closed',review});onChange(id);return ledger.getSession(id);
 }
 async function resume(id,{autoRun=true}={}){check();const data=await snapshot(id);check();if(data.session?.status!=='paused'||data.session.manual)throw Error('paused-automatic-session-required');if(data.activities.some(a=>['uncertain','awaiting-evidence','completing','started'].includes(a.state))||data.clocks.some(c=>c.state!=='confirmed'))throw Error('unresolved-evidence-no-replay');if(now()!==data.session.cursorAt)throw Error('external-world-time-change');
  const eligible=await nativeOwnerEligibility(data.session);eligible();
  invalidate(id);const generation=generations.get(id),session=atomic?await ledger.resumeSession(id,{cursorAt:now()}):await ledger.updateSession(id,{status:'running',stopReason:null});accept(id,session,generation);
  if(now()!==session.cursorAt){await pause(id,'external-world-time-change',leases.get(id));throw Error('external-world-time-change')}
  if(autoRun)void drive(id).catch(error=>onChange({error:error.message}));return session;
 }
 async function takeover(id){check();invalidate(id,'explicit-driver-takeover');clock.stop?.('explicit-driver-takeover',{sessionId:id});const session=await ledger.takeoverSession(id);onChange(id);return session}
 const executionScope=id=>leases.has(id)?{leaseNonce:leases.get(id).leaseNonce}:undefined;
 return {start,step,stop,resume,recover,restore,reconcile,review,addActivity,snapshot,takeover,executionScope,openManualCheckpoint,closeManualCheckpoint,reserveManualSource,manualCheckpointOptions,openActivityCheckpoint,closeActivityCheckpoint,invalidate:reason=>{for(const id of new Set([...leases.keys(),...checkpointRequests.keys()]))invalidate(id,reason)}};
}
