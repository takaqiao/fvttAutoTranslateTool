import {clone,emptyLedger,createActivity,validateSession,validateClock,id,finite,sameCheckpoint} from './schema.mjs';
import {canonicalJSON} from './revision-codec.mjs';
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
  function driver(state,sessionId,options,caller,running=false){
    const session=state.sessions[sessionId];if(!session)throw Error('missing-session');
    if(!atomic||!automatic(session))return session;
    if(session.driver?.userId!==caller.userId||session.driver?.clientNonce!==caller.clientNonce||session.driver?.leaseNonce!==options?.leaseNonce)throw Error('session-driver-required');
    if(running&&session.status!=='running')throw Error('recovery-session-stopped');
    return session;
  }
  const boundManual=a=>a.source?.manual&&a.temporalSource?.type==='checkpoint-reservation';
  const unresolvedActivity=a=>(!a.source?.manual||boundManual(a))&&(a.executor&&a.executor.state!=='settled'||['planned','started','completing','uncertain','awaiting-evidence'].includes(a.state));
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
  function transition(collection,key,{expected,patch,...options}) {
    if(options.evidenceGuard!==undefined&&typeof options.evidenceGuard!=='function')throw Error('synchronous-evidence-guard-required');
    let validateCommit;
    return mutate((s,context,caller)=>{
      const old=s[collection][key];if(!old||!expected.includes(old.state))throw Error('state-conflict');
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
        if(collection==='activities'&&['started','completing'].includes(patch.state))driver(s,old.sessionId,options,caller,true);
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
  return {
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
    createSession:async input=>{if('manualCheckpoint' in input)throw Error('manual-checkpoint-open-required');const captured=validateSession(input),leaseNonce=atomic?crypto.randomUUID():null;return mutate((s,context,caller)=>{const v=clone(captured);if(s.sessions[v.id])throw Error('duplicate-session');if(atomic){v.protocol=protocol(context);if(automatic(v)){if(v.status!=='running')throw Error('invalid-initial-session');recoveryAvailable(s,v);v.driver={...caller,leaseNonce}}}s.sessions[v.id]=v;return v})},
    getSession:key=>get('sessions',key),getActivity:key=>get('activities',key),getClockCommit:key=>get('clocks',key),
    updateSession:(key,patch,options={})=>mutate((s,context,caller)=>{
      const v=s.sessions[key];if(!v)throw Error('missing-session');if(['id','startedAt','activityIds','driver','protocol','manual','nativeOwnerByActor'].some(k=>k in patch))throw Error('immutable-session');
      if(options.expectedStatus!==undefined&&v.status!==options.expectedStatus)throw Error('session-state-conflict');
      if('manualCheckpoint' in patch){
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
        if(patch.status==='recording')throw Error('invalid-session-mode');
        if(patch.status==='running'&&v.status!=='running')throw Error('explicit-resume-required');
        if('leaseNonce' in options||patch.status==='complete'||'cursorAt' in patch&&!options.reconcile)driver(s,key,options,caller);
        if('cursorAt' in patch){finite(patch.cursorAt,'cursor');if(patch.cursorAt<v.cursorAt)throw Error('session-cursor-regression');if(patch.cursorAt!==v.cursorAt&&!Object.values(s.clocks).some(c=>c.sessionId===key&&c.state==='confirmed'&&c.to===patch.cursorAt))throw Error('confirmed-clock-required')}
      }
      Object.assign(v,clone(patch));if(['paused','closed'].includes(v.status)&&v.manualCheckpoint&&v.manualCheckpoint.phase!=='settled')v.manualCheckpoint.phase='interrupted';return v;
    }),
    insertActivity:(input,options={})=>mutate((s,context,caller)=>{
      const a=createActivity(input);if(s.activities[a.id])throw Error('duplicate-activity');const session=s.sessions[a.sessionId];if(!session)throw Error('missing-session');
      if(atomic&&a.source.manual&&automatic(session)){
        if(!boundManual(a))throw Error('manual-checkpoint-reservation-required');const c=manualCheckpoint(s,a,options,caller,context,{open:true});
        if(a.providerId!=='manual'||a.kind!=='treatment'||a.source.type!=='workbench'||a.source.reservationId!==a.id||a.source.useId!==a.proof.useId||a.state!=='awaiting-evidence'||a.executor!==undefined||a.executionResult!==undefined||a.startedAt!==c.from||a.endsAt!==c.to||a.durationSeconds!==600||a.patientUUIDs.length!==1||a.hpPoolUUIDs.length!==1||a.hpPoolUUIDs[0]!==a.patientUUIDs[0]||['checkIds','resultIds','receiptIds','immunityIds'].some(field=>a.proof[field].length)||!a.options.missing.includes('checkpoint-time-confirmation'))throw Error('invalid-manual-reservation');
        if(!session.actorUUIDs.includes(a.actorUUID)||!session.actorUUIDs.includes(a.patientUUIDs[0]))throw Error('session-actor-required');
        if(Object.values(s.activities).some(row=>boundManual(row)&&(row.proof.useId===a.proof.useId||row.sessionId===a.sessionId&&row.checkpointBinding.id===c.id)))throw Error('manual-source-already-reserved');
      }
      if(atomic&&!a.source.manual){
        driver(s,a.sessionId,options,caller,true);
        if(session.manualCheckpoint&&session.manualCheckpoint.phase!=='settled'&&(session.manualCheckpoint.phase!=='sealed'||a.providerId!=='refocus'||a.options.threePecks||a.patientUUIDs.length||a.startedAt!==session.manualCheckpoint.from||a.endsAt!==session.manualCheckpoint.to))throw Error('manual-checkpoint-activity-unavailable');
        if(a.state!=='planned'||a.executor!==undefined||a.executionResult!==undefined)throw Error('initial-activity-claim-required');
        if(Object.hasOwn(session,'nativeOwnerByActor')&&a.source.ownerId!==(Object.hasOwn(session.nativeOwnerByActor,a.actorUUID)?session.nativeOwnerByActor[a.actorUUID]:undefined))throw Error('native-owner-selection-mismatch');
        if(!(session.actorUUIDs??[]).includes(a.actorUUID)||a.patientUUIDs.some(u=>!session.actorUUIDs.includes(u)))throw Error('session-actor-required');
        if(Object.values(s.activities).some(row=>(!row.source?.manual||boundManual(row))&&row.actorUUID===a.actorUUID&&!['cancelled','blocked'].includes(row.state)&&row.startedAt<a.endsAt&&a.startedAt<row.endsAt))throw Error('actor-activity-overlap');
      }
      s.activities[a.id]=a;session.activityIds.push(a.id);return a;
    }),
    transitionActivity:(key,options)=>transition('activities',key,options),
    upsertClockCommit:(input,options={})=>mutate((s,context,caller)=>{
      const c=validateClock(input);if(s.clocks[c.id])throw Error('duplicate-clock');if(!s.sessions[c.sessionId])throw Error('missing-session');
      if(atomic){const session=driver(s,c.sessionId,options,caller,true);if(c.state!=='started'||c.evidence.length||['nativeIssued','nativeResolved','effectsSettled','claim','source'].some(k=>k in input))throw Error('initial-clock-claim-required');if(c.gmId!==caller.userId)throw Error('clock-driver-mismatch');if(c.from!==session.cursorAt)throw Error('world-time-conflict');if(session.manualCheckpoint&&session.manualCheckpoint.phase!=='settled'&&(session.manualCheckpoint.phase!=='advancing'||c.id!==session.manualCheckpoint.id||c.from!==session.manualCheckpoint.from||c.to!==session.manualCheckpoint.to))throw Error('manual-checkpoint-clock-required');if(Object.values(s.clocks).some(row=>row.state!=='confirmed'))throw Error('unresolved-clock');c.claim={...protocol(context),revision:context.revision,...session.driver}}
      s.clocks[c.id]=c;return c;
    }),
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
