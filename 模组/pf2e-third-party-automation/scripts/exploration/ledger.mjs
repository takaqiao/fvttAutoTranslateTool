import {clone,emptyLedger,createActivity,validateSession,validateClock,id,finite,sameCheckpoint,activityCheckpointBinding,sameActivityCheckpoint,manualPoolRequest,MANUAL_POOL_OPERATION} from './schema.mjs';
import {normalizeCheckpointActivity} from './manual-time.mjs';
import {canonicalJSON} from './revision-codec.mjs';
import {checkpointDeclaration,scheduleCheckpointActivities} from './policy.mjs';
const transitions={
  planned:['started','blocked','cancelled'],started:['completing','awaiting-evidence','blocked','uncertain','cancelled'],
  completing:['awaiting-evidence','confirmed','blocked','uncertain'],
  'awaiting-evidence':['confirmed','uncertain'],confirmed:[],blocked:[],uncertain:['confirmed'],cancelled:[]
};
export function createLedger({read,write,transact,isAuthority,identity}) {
  const atomic=typeof transact==='function';
  if(atomic&&typeof identity!=='function')throw Error('runtime-identity-required');
  let tail=Promise.resolve();
  const check=()=>{if(!isAuthority())throw Error('active-gm-required')};
  const localIdentity=()=>{const value=clone(identity());id(value.userId,'runtime-user');id(value.clientNonce,'runtime-client');return value};
  function mutate(fn,validateCommit) {
    const caller=atomic?localIdentity():null;
    const current=()=>{check();if(atomic&&JSON.stringify(localIdentity())!==JSON.stringify(caller))throw Error('runtime-identity-changed')};
    const result=tail.then(async()=>{
      current();
      const validate=()=>{current();validateCommit?.();current();return true};
      if(atomic){const value=await transact((state,context)=>{current();const value=fn(state,context,caller);current();return value},validateCommit?{validateCommit:validate}:undefined);current();return clone(value)}
      const state=clone(await read()??emptyLedger());current();const value=fn(state);validate();await write(state);current();return clone(value);
    });
    tail=result.catch(()=>{});return result;
  }
  const get=async(collection,key)=>clone((await read())[collection]?.[key]??null);
  const automatic=session=>session&&session.manual!==true;
  const protocol=context=>({version:1,rootUUID:id(context.rootUUID,'protocol-root'),epoch:id(context.epoch,'protocol-epoch')});
  function noOpenActivityCheckpoint(session){if(session?.activityCheckpoint?.phase==='open')throw Error('activity-checkpoint-open')}
  function checkpointGuard(guard){
    if(typeof guard!=='function')throw Error('synchronous-checkpoint-guard-required');
    if(guard()!==true)throw Error('activity-checkpoint-changed');
  }
  function interruptDeclarations(state,session,reason){
    const checkpoint=session.activityCheckpoint;if(!checkpoint||['settled','interrupted'].includes(checkpoint.phase))return;
    checkpoint.phase='interrupted';
    for(const row of Object.values(checkpoint.registrations)){
      if(['completed','cancelled','interrupted'].includes(row.status))continue;
      const activity=state.activities[row.activityId];if(!activity)continue;
      row.status=activity.state==='started'?'interrupted':'cancelled';
      activity.state='cancelled';activity.reason=reason;activity.elapsedSeconds=Math.max(0,Math.min(session.cursorAt,activity.endsAt)-activity.startedAt);
    }
  }
  function declarationWindow(state,binding,options,caller,context){
    if(!atomic)throw Error('atomic-activity-checkpoint-required');
    const session=driver(state,binding.sessionId,options,caller,true);
    if(!sameActivityCheckpoint(session.activityCheckpoint,binding)||session.protocol.rootUUID!==binding.rootUUID||session.protocol.epoch!==binding.epoch||context.rootUUID!==binding.rootUUID||context.epoch!==binding.epoch)throw Error('activity-checkpoint-mismatch');
    if(session.activityCheckpoint.phase!=='sealed')throw Error('activity-checkpoint-closed');return session;
  }
  function advanceDeclarations(state,session){
    const checkpoint=session.activityCheckpoint,at=session.cursorAt;
    for(const row of Object.values(checkpoint.registrations)){
      const activity=state.activities[row.activityId];
      if(!session.actorUUIDs.includes(activity.actorUUID))throw Error('session-actor-required');
      if(activity.state==='planned'&&activity.startedAt<=at){
        if(activity.startedAt!==at||activity.dependsOn.some(id=>state.activities[id]?.state!=='confirmed'))throw Error('checkpoint-dependency-incomplete');
        activity.state='started';row.status='started';
      }
      if(activity.state==='started'&&activity.endsAt<=at){
        let covered=activity.startedAt;
        for(const clock of Object.values(state.clocks).filter(c=>c.sessionId===session.id&&c.state==='confirmed').sort((a,b)=>a.from-b.from))if(clock.from<=covered&&clock.to>covered)covered=clock.to;
        if(covered<activity.endsAt)throw Error('confirmed-clock-required');
        activity.state='confirmed';row.status='completed';activity.elapsedSeconds=activity.durationSeconds;
      }
    }
    if(Object.values(checkpoint.registrations).every(row=>row.status==='completed'))checkpoint.phase='settled';
    return checkpoint;
  }
  function checkpointSession(state,binding,options,caller,context){
    if(!atomic)throw Error('atomic-activity-checkpoint-required');
    const session=driver(state,binding.sessionId,options,caller,true);
    if(!automatic(session)||session.protocol?.rootUUID!==binding.rootUUID||session.protocol?.epoch!==binding.epoch||context.rootUUID!==binding.rootUUID||context.epoch!==binding.epoch||session.cursorAt!==binding.from)throw Error('activity-checkpoint-mismatch');
    if(session.manualCheckpoint&&!['settled','interrupted'].includes(session.manualCheckpoint.phase))throw Error('manual-checkpoint-active');
    if(Object.values(state.clocks).some(c=>c.state!=='confirmed')||Object.values(state.activities).some(a=>a.sessionId===session.id&&(unresolvedPool(a)||a.executor&&a.executor.state!=='settled'||['completing','uncertain','awaiting-evidence'].includes(a.state))))throw Error('unresolved-evidence-no-replay');
    return session;
  }
  function enrollCheckpointActivity(inputBinding,input,{authenticatedCaller,leaseNonce,guard}={}){
    const binding=activityCheckpointBinding(inputBinding),event=normalizeCheckpointActivity(input),userId=id(authenticatedCaller,'authenticated-caller');
    if(!sameActivityCheckpoint(binding,event.checkpointBinding))throw Error('activity-checkpoint-mismatch');
    checkpointGuard(guard);
    const {registrationId,checkpointBinding:ignored,...declaration}=event;
    return mutate((state,context,caller)=>{
      checkpointGuard(guard);const session=checkpointSession(state,binding,{leaseNonce},caller,context),checkpoint=session.activityCheckpoint;
      if(!sameActivityCheckpoint(checkpoint,binding)||checkpoint.phase!=='open')throw Error('activity-checkpoint-closed');
      if(!session.actorUUIDs.includes(declaration.actorUUID))throw Error('session-actor-required');
      const prior=checkpoint.registrations[registrationId];
      if(prior){
        if(prior.source.userId!==userId||canonicalJSON(prior.declaration)!==canonicalJSON(declaration))throw Error('checkpoint-registration-conflict');
        return prior;
      }
      if(Object.keys(checkpoint.registrations).length>=128)throw Error('checkpoint-registration-limit');
      for(const dependency of declaration.dependsOn){const activity=state.activities[dependency];if(!activity||activity.sessionId!==session.id||!session.activityIds.includes(dependency)||!['planned','started','confirmed'].includes(activity.state)||activity.executor&&activity.executor.state!=='settled'||unresolvedPool(activity))throw Error('invalid-manual-dependency')}
      const registration={registrationId,registrationOrder:Object.keys(checkpoint.registrations).length,checkpointBinding:binding,declaration,source:{type:'user-record',userId,unverified:true},temporalSource:{type:'checkpoint-declaration',registeredAt:binding.from},status:'registered'};
      checkpoint.registrations[registrationId]=registration;return registration;
    },()=>checkpointGuard(guard));
  }
  async function lookupCheckpointActivity(inputBinding,registrationId,{authenticatedCaller,actorUUID,guard}={}){
    const binding=activityCheckpointBinding(inputBinding),key=id(registrationId,'registration'),userId=id(authenticatedCaller,'authenticated-caller'),actor=id(actorUUID,'actor');
    check();checkpointGuard(guard);const state=await read();check();checkpointGuard(guard);
    const session=state.sessions[binding.sessionId];
    const checkpoint=[session?.activityCheckpoint,...session?.activityCheckpointHistory??[]].find(window=>sameActivityCheckpoint(window,binding));
    if(!checkpoint)throw Error('activity-checkpoint-mismatch');
    const saved=checkpoint.registrations[key];if(!saved)return null;
    if(saved.source.userId!==userId||saved.declaration.actorUUID!==actor)throw Error('checkpoint-registration-owner-required');
    return clone(saved);
  }
  function driver(state,sessionId,options,caller,running=false){
    const session=state.sessions[sessionId];if(!session)throw Error('missing-session');
    if(!atomic||!automatic(session))return session;
    if(session.driver?.userId!==caller.userId||session.driver?.clientNonce!==caller.clientNonce||session.driver?.leaseNonce!==options?.leaseNonce)throw Error('session-driver-required');
    if(running&&session.status!=='running')throw Error('recovery-session-stopped');
    return session;
  }
  const boundManual=a=>a.source?.manual&&a.temporalSource?.type==='checkpoint-reservation';
  const unresolvedPool=a=>Object.values(a.proof?.poolApplications??{}).some(claim=>claim.state!=='settled');
  const unresolvedActivity=a=>unresolvedPool(a)||(!a.source?.manual||boundManual(a)||checkpointDeclaration(a))&&(a.executor&&a.executor.state!=='settled'||['planned','started','completing','uncertain','awaiting-evidence'].includes(a.state));
  function manualCheckpoint(state,activity,options,caller,context,{open=false}={}){
    const session=driver(state,activity.sessionId,options,caller,true),c=session.manualCheckpoint;
    if(!sameCheckpoint(c,options?.checkpointBinding)||!sameCheckpoint(c,activity.checkpointBinding)||c.rootUUID!==session.protocol?.rootUUID||c.epoch!==session.protocol?.epoch||c.rootUUID!==context.rootUUID||c.epoch!==context.epoch||open&&c.phase!=='open'||!['open','sealed','advancing'].includes(c.phase))throw Error('manual-checkpoint-mismatch');
    return c;
  }
  const exactClock=c=>c.nativeResolved===true&&c.effectsSettled===true&&c.evidence.some(r=>{
    const e=r.options?.pf2eThirdPartyAutomation?.exploration;
    return e?.checkpointId===c.id&&e.sessionId===c.sessionId&&e.gmId===c.gmId&&e.expectedFrom===c.from&&e.expectedTo===c.to&&r.userId===c.gmId&&r.worldTime===c.to&&r.dt===c.to-c.from;
  });
  function recoveryAvailable(state,session,exceptSession){
    if(Object.values(state.sessions).some(s=>s.id!==exceptSession&&s.status==='running'&&!s.manual))throw Error('recovery-session-already-running');
    const actors=new Set(session.actorUUIDs??[]),pools=new Set((session.goalsByPool??[]).map(g=>g.poolUUID));
    if(Object.values(state.clocks).some(c=>c.state!=='confirmed')||Object.values(state.activities).some(a=>unresolvedActivity(a)&&(actors.has(a.actorUUID)||a.patientUUIDs.some(u=>actors.has(u))||a.hpPoolUUIDs.some(u=>pools.has(u)))))throw Error('unresolved-evidence-no-replay');
  }
  function transition(collection,key,{expected,patch:inputPatch,...options},preparePatch) {
    if(options.evidenceGuard!==undefined&&typeof options.evidenceGuard!=='function')throw Error('synchronous-evidence-guard-required');
    let validateCommit;
    return mutate((s,context,caller)=>{
      const old=s[collection][key];if(!old||!expected.includes(old.state))throw Error('state-conflict');
      const patch=preparePatch?preparePatch(old):inputPatch;
      if(collection==='activities'&&checkpointDeclaration(old)&&Object.keys(patch).some(k=>k!=='review'))throw Error('activity-checkpoint-lifecycle-required');
      if(collection==='activities'&&patch.proof&&canonicalJSON(patch.proof.manualPoolSource??null)!==canonicalJSON(old.proof?.manualPoolSource??null))throw Error('manual-pool-source-required');
      if(collection==='activities'&&patch.proof&&canonicalJSON(patch.proof.poolApplications??null)!==canonicalJSON(old.proof?.poolApplications??null))throw Error('manual-pool-claim-required');
      if(collection==='activities'&&Object.keys(old.proof?.poolApplications??{}).length){
        if(patch.proof&&(patch.proof.useId!==old.proof.useId||['checkIds','resultIds'].some(field=>!Array.isArray(patch.proof[field])||old.proof[field].some(id=>!patch.proof[field].includes(id)))))throw Error('manual-pool-source-immutable');
        if(patch.state==='confirmed'&&unresolvedPool(old))throw Error('manual-pool-completion-required');
      }
      const legal=collection==='clocks'?{started:['confirmed','uncertain'],confirmed:[],uncertain:['confirmed']}:transitions;
      if(patch.state&&!legal[old.state]?.includes(patch.state))throw Error('illegal-transition');
      const immutable=collection==='clocks'?['id','sessionId','from','to','gmId','claim','source']:['id','sessionId','providerId','actorUUID','patientUUIDs','hpPoolUUIDs','startedAt','endsAt','source','groupId','executor','executionResult','checkpointBinding','temporalSource'];
      if(immutable.some(k=>k in patch&&JSON.stringify(patch[k])!==JSON.stringify(old[k])))throw Error('immutable-provenance');
      if(atomic&&collection==='activities'&&boundManual(old)&&!Object.keys(patch).every(k=>k==='review')){
        const c=manualCheckpoint(s,old,options,caller,context);
        if(['kind','durationSeconds','treatmentImmunitySeconds','order'].some(key=>key in patch&&patch[key]!==old[key])||patch.proof&&patch.proof.useId!==old.proof.useId||patch.options&&['continualRecovery','riskySurgery'].some(key=>patch.options[key]!==old.options[key])||old.options.sourceMessageId&&patch.options&&patch.options.sourceMessageId!==old.options.sourceMessageId)throw Error('immutable-provenance');
        if(patch.proof&&['checkIds','resultIds','receiptIds','immunityIds'].some(key=>!Array.isArray(patch.proof[key])||old.proof[key].some(id=>!patch.proof[key].includes(id))))throw Error('manual-evidence-regression');
        if(patch.executor!==undefined||patch.executionResult!==undefined)throw Error('manual-native-execution-forbidden');
        if(old.proof.checkpointImmunity&&patch.proof&&canonicalJSON(patch.proof.checkpointImmunity)!==canonicalJSON(old.proof.checkpointImmunity))throw Error('immutable-manual-seal');
        if(patch.state==='confirmed'&&(c.phase!=='advancing'||!s.clocks[c.id]||s.clocks[c.id].state!=='confirmed'||!old.proof.checkpointImmunity||patch.options?.missing?.length!==0))throw Error('checkpoint-completion-required');
      }
      if(atomic&&(collection==='clocks'||!old.source?.manual)){
        const reviewOnly=Object.keys(patch).every(k=>k==='review');
        if(collection==='clocks'){
          if(!options.reconcile&&!reviewOnly)driver(s,old.sessionId,options,caller);
          if(patch.state==='confirmed'&&!exactClock(options.reconcile?old:{...old,...patch}))throw Error('clock-completion-required');
          if(options.reconcile&&Object.keys(patch).some(k=>!['state','reason','reconciled','review'].includes(k)))throw Error('saved-clock-proof-required');
        }
        if(collection==='activities'&&['started','completing'].includes(patch.state)){const session=driver(s,old.sessionId,options,caller,true);noOpenActivityCheckpoint(session)}
        if(collection==='activities'&&old.state==='completing'&&!options.reconcile&&!reviewOnly)driver(s,old.sessionId,options,caller);
        if(collection==='activities'&&patch.state==='confirmed'&&old.executor?.state!=='settled')throw Error('native-completion-required');
        if(collection==='activities'&&old.executor?.state==='settled'&&Object.entries(old.executionResult??{}).some(([key,value])=>key in patch&&canonicalJSON(patch[key])!==canonicalJSON(value)))throw Error('native-result-conflict');
      }
      const next={...old,...clone(patch)};
      validateCommit=()=>{
        if(options.expectedSessionStatus!==undefined&&s.sessions[old.sessionId]?.status!==options.expectedSessionStatus)throw Error('session-state-conflict');
        if(options.evidenceGuard){const valid=options.evidenceGuard(next);if(valid&&typeof valid.then==='function')throw Error('synchronous-evidence-guard-required');if(valid!==true)throw Error('manual-evidence-changed')}
      };
      validateCommit();s[collection][key]=next;return next;
    },options.evidenceGuard||options.expectedSessionStatus!==undefined?()=>validateCommit():undefined);
  }
  function appendManualEvidence(key,{activity,proof,resolveOptions}) {
    const expected=clone(activity),delta=clone(proof),ids=['checkIds','resultIds','receiptIds','immunityIds'];
    if(typeof resolveOptions!=='function'||!delta||Object.keys(delta).some(field=>!['useId','nativeImmunity',...ids].includes(field)))throw Error('invalid-ordinary-evidence');
    if(ids.some(field=>delta[field]!==undefined&&(!Array.isArray(delta[field])||delta[field].some(id=>typeof id!=='string'||!id))))throw Error('invalid-ordinary-evidence');
    const same=(a,b)=>canonicalJSON(a)===canonicalJSON(b);
    const identity=current=>{
      if(!current.source?.manual||boundManual(current)||!['native-action','workbench'].includes(current.source.type)||!['treatment','battle-medicine'].includes(current.kind)
        ||['id','sessionId','actorUUID','patientUUIDs','hpPoolUUIDs','source','kind','startedAt','endsAt'].some(field=>!same(current[field]??null,expected[field]??null))
        ||current.proof.useId!==expected.proof.useId||delta.useId!==undefined&&delta.useId!==current.proof.useId)throw Error('ordinary-evidence-source-changed');
    };
    const optionsFor=current=>{
      const value=resolveOptions(clone(current));
      if(!value||typeof value.then==='function'||!Array.isArray(value.missing)||value.missing.some(reason=>typeof reason!=='string'))throw Error('invalid-ordinary-evidence');
      return clone(value);
    };
    return transition('activities',key,{expected:['awaiting-evidence'],expectedSessionStatus:'recording',evidenceGuard:current=>{
      identity(current);return same(optionsFor(current),current.options);
    }},current=>{
      identity(current);
      // Pool source and claim records belong to their own protocol. Merge only
      // ordinary evidence into the latest transaction state; never copy them back.
      const merged={...current.proof};
      for(const field of ids)if(delta[field])merged[field]=[...new Set([...current.proof[field],...delta[field]])];
      if(delta.nativeImmunity!==undefined){
        if(current.source.type!=='native-action'||merged.nativeImmunity&&!same(merged.nativeImmunity,delta.nativeImmunity))throw Error('ordinary-immunity-conflict');
        merged.nativeImmunity=delta.nativeImmunity;
      }
      const next={...current,proof:merged},options=optionsFor(next);
      return {proof:merged,options,...options.missing.length===0?{state:'confirmed'}:{}};
    });
  }
  function manualPoolMutation(input,evidenceGuard,change){
    if(!atomic||typeof evidenceGuard!=='function')throw Error('manual-pool-evidence-guard-required');
    const captured=clone(input),request=manualPoolRequest(captured.request);let validateCommit;
    return mutate(s=>{
      const a=s.activities[request.activityId],session=s.sessions[request.sessionId];
      const claim=Object.values(a?.proof?.poolApplications??{}).find(c=>canonicalJSON(c.request)===canonicalJSON(request));
      if(!claim||a.sessionId!==session?.id||session.manual!==true||session.status!=='recording')throw Error('manual-pool-recording-required');
      const {request:original,...permit}=claim;
      if(canonicalJSON(permit)!==canonicalJSON(captured.permit))throw Error('manual-pool-permit-mismatch');
      validateCommit=()=>{if(session.status!=='recording')throw Error('manual-pool-recording-required');const valid=evidenceGuard();if(valid&&typeof valid.then==='function')throw Error('synchronous-evidence-guard-required');if(valid!==true)throw Error('manual-pool-evidence-changed')};
      validateCommit();return change(s,claim,captured);
    },()=>validateCommit());
  }
  return {
    recordManualPoolSource:(input,{evidenceGuard}={})=>{
      if(!atomic||typeof evidenceGuard!=='function')throw Error('manual-pool-source-guard-required');
      const source=clone(input);let validateCommit;
      const fields=['version','sessionId','activityId','actorUUID','patientUUID','sourceType','useId','checkId','resultId','rollIndex','worldTime','sourceUserId','sourceClientNonce','sourceNonce','provider','documentsDigest'];
      if(source.version!==1||Object.keys(source).length!==fields.length||Object.keys(source).some(key=>!fields.includes(key))||!['native-action','workbench'].includes(source.sourceType)||source.rollIndex!==0||!/^[a-f0-9]{64}$/.test(source.documentsDigest??''))throw Error('invalid-manual-pool-source');
      for(const key of fields.filter(key=>!['version','rollIndex','worldTime','provider','documentsDigest'].includes(key)))id(source[key],key);finite(source.worldTime,'source-time');
      return mutate((s,_context,caller)=>{
        const session=s.sessions[source.sessionId],a=s.activities[source.activityId];
        if(session?.manual!==true||session.status!=='recording'||session.manualPoolIssuer?.userId!==caller.userId||session.manualPoolIssuer.clientNonce!==caller.clientNonce||!a||boundManual(a)||a.state!=='awaiting-evidence'||a.source.type!==source.sourceType||a.actorUUID!==source.actorUUID||a.proof.useId!==source.useId||a.patientUUIDs.length!==1||a.patientUUIDs[0]!==source.patientUUID||!a.proof.checkIds.includes(source.checkId)||!a.proof.resultIds.includes(source.resultId)||a.startedAt!==source.worldTime)throw Error('manual-pool-source-mismatch');
        if(a.proof.manualPoolSource&&canonicalJSON(a.proof.manualPoolSource)!==canonicalJSON(source))throw Error('manual-pool-source-immutable');
        validateCommit=()=>{const valid=evidenceGuard();if(valid?.then||valid!==true||session.status!=='recording')throw Error('manual-pool-source-changed')};validateCommit();
        a.proof.manualPoolSource=source;return source;
      },()=>validateCommit());
    },
    beginManualPoolApplication:(input,{evidenceGuard}={})=>{
      if(Object.keys(input).some(key=>!['request','permit'].includes(key)))throw Error('invalid-manual-pool-begin');
      const applicationNonce=crypto.randomUUID();
      return manualPoolMutation(input,evidenceGuard,(s,claim)=>{
        if(claim.state!=='granted')throw Error('manual-pool-application-already-issued');
        if(Object.values(s.activities).some(a=>Object.values(a.proof?.poolApplications??{}).some(c=>c.poolUUID===claim.poolUUID&&c.state==='applying')))throw Error('manual-pool-write-busy');
        claim.state='applying';claim.applicationNonce=applicationNonce;const {request,...permit}=claim;return permit;
      });
    },
    recordManualPoolTerminal:(input,{evidenceGuard}={})=>manualPoolMutation(input,evidenceGuard,(_s,claim,captured)=>{
      if(claim.state!=='applying'||Object.keys(captured).some(key=>!['request','permit','receiptId','noChange','master'].includes(key)))throw Error('invalid-manual-pool-terminal');
      id(captured.receiptId,'manual-pool-receipt');
      const m=captured.master,b=m?.binding;
      if(typeof captured.noChange!=='boolean'||(captured.noChange?m!==null:!m||m.terminal!=='fulfilled'||m.poolUUID!==claim.poolUUID||![claim.ownerUserId,claim.issuerUserId].includes(m.writerUserId)||b?.permitNonce!==claim.permitNonce||b.applicationNonce!==claim.applicationNonce||b.ownerUserId!==claim.ownerUserId||b.poolUUID!==claim.poolUUID||b.patientUUID!==claim.selectedPatientUUID))throw Error('invalid-manual-pool-terminal');
      const unchanged=m&&Object.hasOwn(m,'updateOutcome');
      const masterKeys=unchanged?'before,binding,fields,poolUUID,terminal,updateOutcome,writerUserId':'before,binding,fields,poolUUID,terminal,writerUserId';
      if(m&&(Object.keys(m).sort().join(',')!==masterKeys||(unchanged&&m.updateOutcome!=='unchanged')||Object.keys(b).sort().join(',')!=='applicationNonce,ownerUserId,patientUUID,permitNonce,poolUUID'||!m.before||Array.isArray(m.before)||Object.keys(m.before).some(key=>!['value','max','temp','sp'].includes(key))))throw Error('invalid-manual-pool-terminal');
      if(m&&(!m.fields||!Object.keys(m.fields).length||Object.entries(m.fields).some(([key,value])=>!['system.attributes.hp.value','system.attributes.hp.sp.value','system.attributes.hp.temp'].includes(key)||!Number.isFinite(value)||value<0)))throw Error('invalid-manual-pool-terminal');
      if(unchanged)for(const [key,value] of Object.entries(m.fields)){
        let prior=m.before;
        for(const part of key.slice('system.attributes.hp.'.length).split('.')){
          if(!prior||typeof prior!=='object'||!Object.hasOwn(prior,part))throw Error('invalid-manual-pool-terminal');
          prior=prior[part];
        }
        if(prior!==value)throw Error('invalid-manual-pool-terminal');
      }
      claim.state='settled';claim.terminal={receiptId:captured.receiptId,noChange:captured.noChange,master:clone(m)};return clone(claim.terminal);
    }),
    claimManualPoolApplication:(input,{evidenceGuard}={})=>{
      if(!atomic)throw Error('atomic-manual-pool-required');
      if(typeof evidenceGuard!=='function')throw Error('manual-pool-evidence-guard-required');
      const request=manualPoolRequest(input.request),captured=clone(input);
      if(Object.keys(captured).some(key=>!['request','effectId','selectedPatientUUID','sourceDigest','ownerUserId','permitNonce','worldTime'].includes(key)))throw Error('invalid-manual-pool-claim');
      for(const key of ['effectId','selectedPatientUUID','ownerUserId','permitNonce'])id(captured[key],`manual-pool-${key}`);
      if(!/^[a-f0-9]{64}$/.test(captured.sourceDigest??'')||!request.patientUUIDs.includes(captured.selectedPatientUUID))throw Error('invalid-manual-pool-claim');
      finite(captured.worldTime,'manual-pool-time');
      const effectKey=canonicalJSON([request.sourceType,request.useId,captured.effectId,request.rollIndex,request.stage]);let validateCommit;
      return mutate((s,context,caller)=>{
        const a=s.activities[request.activityId],session=s.sessions[request.sessionId];
        if(session?.manual!==true||session.status!=='recording'||!a||a.sessionId!==session.id||a.providerId!=='manual'||a.kind!=='treatment'||a.source?.manual!==true||a.source.type!==request.sourceType||a.state!=='awaiting-evidence'||a.options?.riskySurgery||boundManual(a))throw Error('manual-pool-recording-required');
        if(a.actorUUID!==request.actorUUID||a.proof.useId!==request.useId||!a.proof.checkIds.includes(request.checkId)||!a.proof.resultIds.includes(request.resultId)||!a.hpPoolUUIDs.includes(request.poolUUID)||request.patientUUIDs.some(uuid=>!a.patientUUIDs.includes(uuid))||[a.actorUUID,...request.patientUUIDs].some(uuid=>!session.actorUUIDs.includes(uuid))||a.startedAt!==captured.worldTime||session.startedAt>captured.worldTime)throw Error('manual-pool-source-mismatch');
        for(const row of Object.values(s.activities))for(const claim of Object.values(row.proof?.poolApplications??{}))if(claim.poolUUID===request.poolUUID&&claim.effectKey===effectKey)throw Error('manual-pool-already-claimed');
        const grant={operationId:MANUAL_POOL_OPERATION,...protocol(context),revision:context.revision,sessionId:a.sessionId,activityId:a.id,actorUUID:a.actorUUID,
          effectKey,poolUUID:request.poolUUID,patientUUIDs:request.patientUUIDs,selectedPatientUUID:captured.selectedPatientUUID,sourceDigest:captured.sourceDigest,
          ownerUserId:captured.ownerUserId,ownerClientNonce:request.ownerClientNonce,attemptNonce:request.attemptNonce,permitNonce:captured.permitNonce,issuerUserId:caller.userId,state:'granted'};
        validateCommit=()=>{if(session.status!=='recording')throw Error('manual-pool-recording-required');const valid=evidenceGuard();if(valid&&typeof valid.then==='function')throw Error('synchronous-evidence-guard-required');if(valid!==true)throw Error('manual-pool-evidence-changed')};
        validateCommit();a.proof.poolApplications={...a.proof.poolApplications,[effectKey]:{...grant,request:clone(request)}};return grant;
      },()=>validateCommit());
    },
    lookupManualPoolProof:async(input,ownerUserId)=>{
      check();const request=manualPoolRequest(input),s=await read();check();
      const a=s.activities[request.activityId];if(a?.sessionId!==request.sessionId)return null;
      const claim=Object.values(a.proof?.poolApplications??{}).find(c=>c.ownerUserId===ownerUserId&&canonicalJSON(c.request)===canonicalJSON(request));
      return claim?{status:claim.state==='settled'?'settled':'reserved',effectKey:claim.effectKey,poolUUID:claim.poolUUID,sourceDigest:claim.sourceDigest,...claim.state==='settled'?{receiptId:claim.terminal.receiptId,noChange:claim.terminal.noChange}:{}}:null;
    },
    quarantineLegacySessions:()=>mutate(s=>{
      if(!atomic)throw Error('atomic-migration-required');
      const sessions=Object.values(s.sessions).filter(session=>automatic(session)&&session.status==='running'&&!session.protocol&&!session.driver);
      const quarantinedSessionIds=sessions.map(session=>session.id),ids=new Set(quarantinedSessionIds);
      for(const session of sessions){session.status='paused';session.stopReason='legacy-migration-quarantine'}
      for(const a of Object.values(s.activities).filter(activity=>ids.has(activity.sessionId)&&!activity.source?.manual)){
        if(a.state==='planned'&&!a.executor){a.state='cancelled';a.reason='legacy-migration-quarantine'}
        else if(['started','completing'].includes(a.state)){a.state='uncertain';a.reason='legacy-migration-quarantine'}
      }
      for(const c of Object.values(s.clocks).filter(clock=>ids.has(clock.sessionId)&&clock.state==='started')){c.state='uncertain';c.reason='legacy-migration-quarantine'}
      return {quarantinedSessionIds};
    }),
    createSession:async input=>{if('activityCheckpointHistory' in input)throw Error('activity-checkpoint-open-required');if('activityCheckpoint' in input)throw Error('activity-checkpoint-open-required');if('manualCheckpoint' in input)throw Error('manual-checkpoint-open-required');if('manualPoolIssuer' in input)throw Error('manual-pool-issuer-required');const captured=validateSession(input),leaseNonce=atomic?crypto.randomUUID():null;return mutate((s,context,caller)=>{const v=clone(captured);if(s.sessions[v.id])throw Error('duplicate-session');if(atomic){v.protocol=protocol(context);if(v.manual===true)v.manualPoolIssuer={...caller};if(automatic(v)){if(v.status!=='running')throw Error('invalid-initial-session');recoveryAvailable(s,v);v.driver={...caller,leaseNonce}}}s.sessions[v.id]=v;return v})},
    openActivityCheckpoint:(input,{leaseNonce,guard}={})=>{
      const binding=activityCheckpointBinding(input);checkpointGuard(guard);
      return mutate((state,context,caller)=>{
        checkpointGuard(guard);const session=checkpointSession(state,binding,{leaseNonce},caller,context);
        if(session.activityCheckpoint){
          if(!['settled','interrupted'].includes(session.activityCheckpoint.phase))throw Error('activity-checkpoint-already-opened');
          if([session.activityCheckpoint,...session.activityCheckpointHistory??[]].some(old=>old.id===binding.id||old.observationNonce===binding.observationNonce))throw Error('activity-checkpoint-binding-reused');
          (session.activityCheckpointHistory??=[]).push(clone(session.activityCheckpoint));
        }
        session.activityCheckpoint={...binding,phase:'open',registrations:{}};return session.activityCheckpoint;
      },()=>checkpointGuard(guard));
    },
    enrollCheckpointActivity,lookupCheckpointActivity,
    sealActivityCheckpoint:(input,{leaseNonce,registrations,guard}={})=>{
      const binding=activityCheckpointBinding(input),expected=clone(registrations);checkpointGuard(guard);
      return mutate((state,context,caller)=>{
        checkpointGuard(guard);const session=checkpointSession(state,binding,{leaseNonce},caller,context),checkpoint=session.activityCheckpoint;
        if(!sameActivityCheckpoint(checkpoint,binding)||checkpoint.phase!=='open')throw Error('activity-checkpoint-closed');
        if(canonicalJSON(checkpoint.registrations)!==canonicalJSON(expected))throw Error('activity-checkpoint-registrations-changed');
        const activities=Object.values(state.activities).filter(a=>a.sessionId===session.id),plan=scheduleCheckpointActivities({registrations:checkpoint.registrations,activities,session,from:binding.from});
        for(const item of plan){
          const row=checkpoint.registrations[item.registrationId],d=row.declaration;
          if(!session.actorUUIDs.includes(d.actorUUID))throw Error('session-actor-required');
          const activityId=JSON.stringify(['checkpoint-declaration',binding.id,item.registrationId]);if(state.activities[activityId])throw Error('duplicate-activity');
          const activity=createActivity({...d,id:activityId,sessionId:session.id,providerId:'manual',patientUUIDs:[],hpPoolUUIDs:[],state:'planned',startedAt:item.startedAt,endsAt:item.endsAt,kind:'activity',source:{...row.source,manual:true},temporalSource:{...row.temporalSource},checkpointBinding:binding,registrationId:item.registrationId});
          state.activities[activityId]=activity;session.activityIds.push(activityId);row.activityId=activityId;row.status='planned';
        }
        checkpoint.phase='sealed';return advanceDeclarations(state,session);
      },()=>checkpointGuard(guard));
    },
    advanceActivityCheckpoint:(input,{leaseNonce,at,guard}={})=>{
      const binding=activityCheckpointBinding(input);finite(at,'checkpoint-time');checkpointGuard(guard);
      return mutate((state,context,caller)=>{
        checkpointGuard(guard);const session=declarationWindow(state,binding,{leaseNonce},caller,context);
        if(session.cursorAt!==at||Object.values(state.clocks).some(c=>c.state!=='confirmed'))throw Error('unresolved-clock');
        return advanceDeclarations(state,session);
      },()=>checkpointGuard(guard));
    },
    getSession:key=>get('sessions',key),getActivity:key=>get('activities',key),getClockCommit:key=>get('clocks',key),
    updateSession:(key,patch,options={})=>mutate((s,context,caller)=>{
      const v=s.sessions[key];if(!v)throw Error('missing-session');if(['id','startedAt','activityIds','driver','protocol','manual','manualPoolIssuer','nativeOwnerByActor','activityCheckpoint','activityCheckpointHistory'].some(k=>k in patch))throw Error('immutable-session');
      if(options.expectedStatus!==undefined&&v.status!==options.expectedStatus)throw Error('session-state-conflict');
      if('manualCheckpoint' in patch){
        if(v.activityCheckpoint&&!['settled','interrupted'].includes(v.activityCheckpoint.phase))throw Error('activity-checkpoint-active');
        if(!atomic||!automatic(v))throw Error('atomic-manual-checkpoint-required');driver(s,key,options,caller,true);
        const next=patch.manualCheckpoint,old=v.manualCheckpoint;
        if(!next||Reflect.ownKeys(next).some(field=>!['id','sessionId','rootUUID','epoch','observationNonce','from','to','phase'].includes(field))||next.sessionId!==key||next.rootUUID!==v.protocol?.rootUUID||next.epoch!==v.protocol?.epoch||next.rootUUID!==context.rootUUID||next.epoch!==context.epoch)throw Error('manual-checkpoint-mismatch');
        for(const field of ['id','observationNonce'])id(next[field],field);finite(next.from,'checkpoint-from');finite(next.to,'checkpoint-to');
        if(!old){if(next.phase!=='open'||next.from!==v.cursorAt||next.to!==next.from+600||next.to>v.budgetEndsAt||v.activityIds.length||Object.values(s.clocks).some(c=>c.sessionId===key))throw Error('manual-checkpoint-unavailable')}
        else if(!sameCheckpoint(old,Object.fromEntries(['id','sessionId','rootUUID','epoch','observationNonce','from','to'].map(field=>[field,next[field]])))||({open:'sealed',sealed:'advancing',advancing:'settled'})[old.phase]!==next.phase)throw Error('manual-checkpoint-phase-conflict');
        if(next.phase==='sealed'){const rows=Object.values(s.activities).filter(a=>a.sessionId===key&&boundManual(a)&&a.checkpointBinding.id===next.id);if(rows.length!==1||rows.some(a=>a.state!=='awaiting-evidence'||!a.proof.checkpointImmunity||a.options.missing.some(m=>m!=='checkpoint-time-confirmation')))throw Error('manual-checkpoint-evidence-required')}
        if(next.phase==='settled'&&(!s.clocks[next.id]||s.clocks[next.id].state!=='confirmed'||Object.values(s.activities).some(a=>a.sessionId===key&&boundManual(a)&&a.checkpointBinding.id===next.id&&a.state!=='confirmed')))throw Error('checkpoint-completion-required');
      }
      if(atomic&&automatic(v)){
        if(patch.status==='complete'){noOpenActivityCheckpoint(v);if(v.activityCheckpoint?.phase==='sealed')throw Error('activity-checkpoint-pending')}
        if(patch.status==='recording')throw Error('invalid-session-mode');
        if(patch.status==='running'&&v.status!=='running')throw Error('explicit-resume-required');
        if('leaseNonce' in options||patch.status==='complete'||'cursorAt' in patch&&!options.reconcile)driver(s,key,options,caller);
        if('cursorAt' in patch){finite(patch.cursorAt,'cursor');if(patch.cursorAt<v.cursorAt)throw Error('session-cursor-regression');if(patch.cursorAt!==v.cursorAt&&!Object.values(s.clocks).some(c=>c.sessionId===key&&c.state==='confirmed'&&c.to===patch.cursorAt))throw Error('confirmed-clock-required')}
      }
      Object.assign(v,clone(patch));if(['paused','closed'].includes(v.status))interruptDeclarations(s,v,v.stopReason??v.status);if(['paused','closed'].includes(v.status)&&v.manualCheckpoint&&v.manualCheckpoint.phase!=='settled')v.manualCheckpoint.phase='interrupted';if(['paused','closed'].includes(v.status)&&v.activityCheckpoint&&v.activityCheckpoint.phase!=='settled')v.activityCheckpoint.phase='interrupted';return v;
    }),
    insertActivity:(input,options={})=>mutate((s,context,caller)=>{
      const a=createActivity(input);if(checkpointDeclaration(a))throw Error('activity-checkpoint-seal-required');if(s.activities[a.id])throw Error('duplicate-activity');const session=s.sessions[a.sessionId];if(!session)throw Error('missing-session');
      if(Object.hasOwn(a.proof,'poolApplications'))throw Error('manual-pool-claim-required');
      if(Object.hasOwn(a.proof,'manualPoolSource'))throw Error('manual-pool-source-required');
      if(atomic&&a.source.manual&&automatic(session)){
        if(!boundManual(a))throw Error('manual-checkpoint-reservation-required');const c=manualCheckpoint(s,a,options,caller,context,{open:true});
        if(a.providerId!=='manual'||a.kind!=='treatment'||a.source.type!=='workbench'||a.source.reservationId!==a.id||a.source.useId!==a.proof.useId||a.state!=='awaiting-evidence'||a.executor!==undefined||a.executionResult!==undefined||a.startedAt!==c.from||a.endsAt!==c.to||a.durationSeconds!==600||a.patientUUIDs.length!==1||a.hpPoolUUIDs.length!==1||a.hpPoolUUIDs[0]!==a.patientUUIDs[0]||['checkIds','resultIds','receiptIds','immunityIds'].some(field=>a.proof[field].length)||!a.options.missing.includes('checkpoint-time-confirmation'))throw Error('invalid-manual-reservation');
        if(!session.actorUUIDs.includes(a.actorUUID)||!session.actorUUIDs.includes(a.patientUUIDs[0]))throw Error('session-actor-required');
        if(Object.values(s.activities).some(row=>boundManual(row)&&(row.proof.useId===a.proof.useId||row.sessionId===a.sessionId&&row.checkpointBinding.id===c.id)))throw Error('manual-source-already-reserved');
      }
      if(atomic&&!a.source.manual){
        driver(s,a.sessionId,options,caller,true);
        noOpenActivityCheckpoint(session);
        if(session.manualCheckpoint&&session.manualCheckpoint.phase!=='settled'&&(session.manualCheckpoint.phase!=='sealed'||a.providerId!=='refocus'||a.options.threePecks||a.patientUUIDs.length||a.startedAt!==session.manualCheckpoint.from||a.endsAt!==session.manualCheckpoint.to))throw Error('manual-checkpoint-activity-unavailable');
        if(a.state!=='planned'||a.executor!==undefined||a.executionResult!==undefined)throw Error('initial-activity-claim-required');
        if(Object.hasOwn(session,'nativeOwnerByActor')&&a.source.ownerId!==(Object.hasOwn(session.nativeOwnerByActor,a.actorUUID)?session.nativeOwnerByActor[a.actorUUID]:undefined))throw Error('native-owner-selection-mismatch');
        if(!(session.actorUUIDs??[]).includes(a.actorUUID)||a.patientUUIDs.some(u=>!session.actorUUIDs.includes(u)))throw Error('session-actor-required');
        if(Object.values(s.activities).some(row=>(!row.source?.manual||boundManual(row)||checkpointDeclaration(row))&&row.actorUUID===a.actorUUID&&!['cancelled','blocked'].includes(row.state)&&row.startedAt<a.endsAt&&a.startedAt<row.endsAt))throw Error('actor-activity-overlap');
      }
      s.activities[a.id]=a;session.activityIds.push(a.id);return a;
    }),
    transitionActivity:(key,options)=>transition('activities',key,options),
    appendManualEvidence,
    upsertClockCommit:(input,options={})=>{let session;const guard=options.guard===undefined?undefined:()=>checkpointGuard(()=>options.guard(session));return mutate((s,context,caller)=>{
      const c=validateClock(input);if(s.clocks[c.id])throw Error('duplicate-clock');if(!s.sessions[c.sessionId])throw Error('missing-session');
      session=s.sessions[c.sessionId];guard?.();
      if(atomic){noOpenActivityCheckpoint(s.sessions[c.sessionId]);const boundary=Object.values(s.activities).filter(a=>a.sessionId===c.sessionId&&checkpointDeclaration(a)&&['planned','started'].includes(a.state)).flatMap(a=>[a.startedAt,a.endsAt]).filter(at=>at>c.from);if(boundary.some(at=>c.to>at))throw Error('activity-checkpoint-time-boundary')}
      if(atomic){const session=driver(s,c.sessionId,options,caller,true);if(c.state!=='started'||c.evidence.length||['nativeIssued','nativeResolved','effectsSettled','claim','source'].some(k=>k in input))throw Error('initial-clock-claim-required');if(c.gmId!==caller.userId)throw Error('clock-driver-mismatch');if(c.from!==session.cursorAt)throw Error('world-time-conflict');if(session.manualCheckpoint&&session.manualCheckpoint.phase!=='settled'&&(session.manualCheckpoint.phase!=='advancing'||c.id!==session.manualCheckpoint.id||c.from!==session.manualCheckpoint.from||c.to!==session.manualCheckpoint.to))throw Error('manual-checkpoint-clock-required');if(Object.values(s.clocks).some(row=>row.state!=='confirmed'))throw Error('unresolved-clock');c.claim={...protocol(context),revision:context.revision,...session.driver}}
      s.clocks[c.id]=c;return c;
    },guard)},
    transitionClockCommit:(key,options)=>transition('clocks',key,options),
    claimExecution:(key,input)=>mutate((s,context,caller)=>{
      if(!atomic)throw Error('atomic-execution-required');const a=s.activities[key];if(!a||a.state!=='completing')throw Error('state-conflict');
      if(a.source?.manual)throw Error('manual-native-execution-forbidden');
      const session=driver(s,a.sessionId,input,caller,true);if(a.executor)throw Error('activity-already-executed');
      if(input.operationId!==(a.options?.extensionOf?'treatment-extension':a.providerId))throw Error('native-operation-mismatch');
      if(input.ownerUserId!==(a.source?.ownerId??session.driver.userId))throw Error('native-owner-mismatch');
      for(const field of ['ownerUserId','ownerClientNonce','attemptNonce','permitNonce'])id(input[field],field);
      a.executor={protocol:'pf2e-third-party-automation.exploration-owner.v1',rootUUID:context.rootUUID,epoch:context.epoch,revision:context.revision,sessionId:a.sessionId,activityId:a.id,operationId:input.operationId,actorUUID:a.actorUUID,ownerUserId:input.ownerUserId,ownerClientNonce:input.ownerClientNonce,attemptNonce:input.attemptNonce,permitNonce:input.permitNonce,leaseNonce:input.leaseNonce,state:'granted'};
      for(const field of ['offerId','requestId','commandDigest'])if(input[field]!==undefined)a.executor[field]=id(input[field],field);
      return a.executor;
    }),
    recordExecutionResult:(key,{permit,result})=>mutate(s=>{
      if(!atomic)throw Error('atomic-execution-required');const a=s.activities[key],executor=a?.executor;
      if(!executor||['protocol','rootUUID','epoch','revision','sessionId','activityId','operationId','actorUUID','ownerUserId','ownerClientNonce','attemptNonce','permitNonce','leaseNonce','offerId','requestId','commandDigest'].some(field=>executor[field]!==permit?.[field]))throw Error('native-permit-mismatch');
      if(!['completing','uncertain'].includes(a.state)||!['confirmed','blocked','uncertain'].includes(result?.status))throw Error('invalid-native-result');
      const fields=['status','reason','proof','sourceDegree','effectiveOutcome','rolledHealing','medicBonus','expiresAt','resourceReceiptIds','patientUUID','results','focusBefore','focusAfter','treatment'];
      if(Object.keys(result).some(field=>!fields.includes(field)))throw Error('immutable-provenance');
      const checkOnlyFailure=a.providerId==='treat-wounds'&&!a.options?.extensionOf&&!a.options?.riskySurgery&&result.effectiveOutcome==='failure'&&result.proof?.checkIds?.length>0&&result.proof?.resultIds?.length===0;
      if(result.status==='confirmed'&&(!result.proof||result.proof.useId!==(a.options?.extensionOf??a.id)||!Array.isArray(result.proof.receiptIds)||!result.proof.receiptIds.length&&!checkOnlyFailure))throw Error('native-completion-required');
      if(executor.state==='settled'&&canonicalJSON(a.executionResult)!==canonicalJSON(result))throw Error('native-result-conflict');
      a.executionResult=clone(result);Object.assign(a,clone(result));a.executor={...executor,state:result.status==='confirmed'?'settled':'uncertain'};return a;
    }),
    resumeSession:(key,{cursorAt})=>{const leaseNonce=crypto.randomUUID();return mutate((s,context,caller)=>{
      if(!atomic)throw Error('atomic-resume-required');const session=s.sessions[key];if(!session||session.status!=='paused'||session.manual)throw Error('paused-automatic-session-required');
      if(cursorAt!==session.cursorAt)throw Error('external-world-time-change');recoveryAvailable(s,session,key);
      session.status='running';session.stopReason=null;session.driver={...caller,leaseNonce};session.protocol=protocol(context);return session;
    })},
    takeoverSession:key=>mutate(s=>{
      if(!atomic)throw Error('atomic-takeover-required');const session=s.sessions[key];if(!session||!['running','paused'].includes(session.status)||session.manual)throw Error('automatic-session-required');
      session.status='paused';session.stopReason='explicit-driver-takeover';
      if(session.manualCheckpoint&&session.manualCheckpoint.phase!=='settled')session.manualCheckpoint.phase='interrupted';
      interruptDeclarations(s,session,'explicit-driver-takeover');
      for(const a of Object.values(s.activities).filter(row=>row.sessionId===key&&(!row.source?.manual||boundManual(row)))){
        if(a.state==='planned'&&!a.executor){a.state='cancelled';a.reason='explicit-driver-takeover'}
        else if(['started','completing'].includes(a.state)){a.state='uncertain';a.reason='explicit-driver-takeover'}
      }
      for(const c of Object.values(s.clocks).filter(row=>row.sessionId===key&&row.state==='started')){c.state='uncertain';c.reason='explicit-driver-takeover'}
      return session;
    }),
    ownsSession:(session,options={})=>!atomic||session?.driver?.userId===localIdentity().userId&&session?.driver?.clientNonce===localIdentity().clientNonce&&session?.driver?.leaseNonce===options.leaseNonce,
    atomic,
    all:async()=>clone(await read()),
    snapshot:async key=>{const s=await read();return clone({session:s.sessions[key]??null,activities:Object.values(s.activities).filter(a=>a.sessionId===key),clocks:Object.values(s.clocks).filter(c=>c.sessionId===key)})}
  };
}
