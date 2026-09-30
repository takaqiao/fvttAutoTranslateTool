import {recoveryProposals} from './policy.mjs';
import {extensionPatients} from './treatment.mjs';
export function createCoordinator({ledger,capabilities,providers,clock,policy,isAuthority,now=()=>globalThis.game.time.worldTime,ownerOperations,onChange=()=>{}}){
 const byId=new Map(providers.map(p=>[p.id,p])),locks=new Set(),contexts=new Map(),runners=new Map();
 const check=()=>{if(!isAuthority())throw Error('active-gm-required')};
 async function snapshot(id){const data=await ledger.snapshot(id);return {...data,actors:data.session?await capabilities.snapshot(data.session.actorUUIDs):[]}}
 async function stop(id,reason='user-stopped'){check();const result=await ledger.updateSession(id,{status:'paused',stopReason:reason});const data=await ledger.snapshot(id);check();for(const a of data.activities.filter(a=>['planned','started','completing'].includes(a.state))){ownerOperations.cancelActivity?.(a);if(a.state==='completing')continue;await byId.get(a.providerId)?.cancel?.(a,contexts.get(a.id));check();try{await ledger.transitionActivity(a.id,{expected:['planned','started'],patch:{state:'cancelled',reason}});contexts.delete(a.id)}catch(error){if(error.message!=='state-conflict')throw error}}onChange(id);return result}
 async function addActivity(id,input){check();const activity=await ledger.insertActivity({...input,id:input.id??crypto.randomUUID(),sessionId:id,state:'planned',source:input.source??{type:'coordinator'}});const provider=byId.get(activity.providerId);if(!provider)throw Error('unknown-provider');
  const ctx=await ownerOperations.createActivityContext(activity);check();const begun=await provider.begin(activity,ctx);check();
  if((await ledger.getSession(id)).status!=='running'){await provider.cancel?.(activity,ctx);return ledger.transitionActivity(activity.id,{expected:['planned'],patch:{state:'cancelled',reason:'session-stopped'}})}
  const saved=await ledger.transitionActivity(activity.id,{expected:['planned'],patch:{state:begun.status==='started'?'started':'blocked',reason:begun.reason}});if(saved.state==='started')contexts.set(saved.id,ctx);return saved;
 }
 async function step(id){check();if(locks.has(id))return {status:'busy'};locks.add(id);
  try{
   let data=await snapshot(id),s=data.session;check();if(s.status!=='running')return {status:s.status};
   if(data.activities.some(a=>['uncertain','awaiting-evidence','completing'].includes(a.state))||data.clocks.some(c=>c.state!=='confirmed'))return stop(id,'unresolved-evidence');
   if(now()!==s.cursorAt)return stop(id,'external-world-time-change');
   const active=data.activities.filter(a=>a.state==='started');
   const goalsMet=s.goalsByPool.every(g=>data.actors.some(a=>a.pool.poolUUID===g.poolUUID&&a.hp.value>=g.targetHP));
   if(!active.length&&goalsMet&&(!s.requireFullFocus||data.actors.every(a=>a.focus.value>=a.focus.max))){await ledger.updateSession(id,{status:'complete',stopReason:'goals-met'});onChange(id);return {status:'complete'}}
   const activityCount=data.activities.filter(a=>!a.source.manual).length;
   if((activityCount>=s.maxActivities&&!active.length)||now()>=s.budgetEndsAt)return stop(id,'budget');
   const proposals=activityCount>=s.maxActivities?[]:recoveryProposals({actors:data.actors,activities:data.activities,session:s,now:now(),providerIds:[...byId.keys()]});
   const next=policy({snapshot:data,proposals,session:s,now:now()});
   for(const proposal of next.activities.slice(0,s.maxActivities-data.activities.filter(a=>!a.source.manual).length)){check();if((await ledger.getSession(id)).status!=='running')return {status:'paused'};const added=await addActivity(id,{...proposal,source:{type:'coordinator'},groupId:undefined});if(added.state==='blocked')return stop(id,added.reason)}
   if(next.checkpointAt===null)return stop(id,data.actors.some(a=>a.refocusUnsupported?.length&&(a.threePecks||s.requireFullFocus&&a.focus.value<a.focus.max))?'refocus-recovery-unadapted':next.reason);
   if((await ledger.getSession(id)).status!=='running')return {status:'paused'};
   const receipt=await clock.advanceTo({id:crypto.randomUUID(),sessionId:id,from:now(),to:next.checkpointAt});check();if(receipt.status!=='confirmed')return stop(id,receipt.reason??'clock-unconfirmed');
   await ledger.updateSession(id,{cursorAt:now()});data=await snapshot(id);check();
   for(const a of data.activities.filter(a=>a.state==='started'&&a.endsAt<=now())){
    check();if((await ledger.getSession(id)).status!=='running')return {status:'paused'};const claimed=await ledger.transitionActivity(a.id,{expected:['started'],patch:{state:'completing'}});
    let result;try{result=await byId.get(a.providerId).complete(claimed,contexts.get(a.id));check()}catch(error){result={status:'uncertain',reason:error.message,...error.proof?{proof:error.proof}:{}}}
    check();await ledger.transitionActivity(a.id,{expected:['completing'],patch:{...result,state:result.status==='confirmed'?'confirmed':result.status==='blocked'?'blocked':'uncertain'}});contexts.delete(a.id);
    if(result.status!=='confirmed')return stop(id,result.reason??'native-result-unconfirmed');
    const remainingSlots=s.maxActivities-(await ledger.snapshot(id)).activities.filter(row=>!row.source.manual).length;
    if(remainingSlots>0&&s.extendTreatment&&a.providerId==='treat-wounds'&&!a.options.extensionOf&&extensionPatients({...a,...result}).length&&a.startedAt+3600<=s.budgetEndsAt){
     const eligible=new Set(extensionPatients({...a,...result}).map(r=>r.patientUUID??a.patientUUIDs[0])),fresh=(await capabilities.snapshot(a.patientUUIDs)).filter(p=>eligible.has(p.actorUUID)&&p.hp.value<(s.goalsByPool.find(g=>g.poolUUID===p.pool.poolUUID)?.targetHP??0));check();if(fresh.length)await addActivity(id,{providerId:a.providerId,actorUUID:a.actorUUID,patientUUIDs:fresh.map(p=>p.actorUUID),hpPoolUUIDs:[...new Set(fresh.map(p=>p.pool.poolUUID))],startedAt:now(),endsAt:a.startedAt+3600,options:{...a.options,extensionOf:a.id},source:{type:'coordinator'}});
    }
   }
   onChange(id);return {status:'running'};
  }catch(error){if(isAuthority())await stop(id,error.message);return {status:'paused',reason:error.message}}
  finally{locks.delete(id)}
 }
 async function drive(id){if(runners.has(id))return;runners.set(id,true);try{while(isAuthority()&&(await ledger.getSession(id))?.status==='running'){await step(id);await Promise.resolve()}}finally{runners.delete(id)}}
 async function start(config){check();const actors=await capabilities.snapshot(config.actorUUIDs);check();const at=now();const goals=config.goalsByPool??[...new Map(actors.map(a=>[a.pool.poolUUID,{poolUUID:a.pool.poolUUID,targetHP:a.hp.max}])).values()];
  if(!config.manual){const all=await ledger.all();check();const ids=new Set(config.actorUUIDs),pools=new Set(actors.map(a=>a.pool.poolUUID)),reviewed=row=>row.review&&all.sessions[row.sessionId]?.status==='closed';if(Object.values(all.sessions).some(s=>s.status==='running'))throw Error('recovery-session-already-running');if(Object.values(all.clocks).some(c=>c.state!=='confirmed'&&!reviewed(c))||Object.values(all.activities).some(a=>!a.source?.manual&&!reviewed(a)&&['uncertain','awaiting-evidence','completing','started'].includes(a.state)&&(ids.has(a.actorUUID)||a.patientUUIDs.some(u=>ids.has(u))||a.hpPoolUUIDs.some(u=>pools.has(u)))))throw Error('unresolved-evidence-no-replay')}
  if(!actors.length||goals.some(g=>!actors.some(a=>a.pool.poolUUID===g.poolUUID&&Number.isFinite(g.targetHP)&&g.targetHP>=0&&g.targetHP<=a.hp.max)))throw Error('invalid-recovery-goals');
  const s=await ledger.createSession({...config,id:config.id??crypto.randomUUID(),actorUUIDs:[...new Set(config.actorUUIDs)],startedAt:at,cursorAt:at,budgetEndsAt:at+Math.max(0,config.budgetSeconds??7200),maxActivities:Math.min(100,Math.max(1,config.maxActivities??100)),goalsByPool:goals,status:config.manual?'recording':'running',assumptions:config.assumptions??['different-actors-may-overlap']});
  if(config.autoRun!==false&&!config.manual)void drive(s.id).catch(error=>onChange({error:error.message}));onChange(s.id);return s;
 }
 async function recover(id){check();if(runners.has(id)||locks.has(id))return ledger.getSession(id);const data=await ledger.snapshot(id);if(!data.session||data.session.status!=='running')return data.session;
  await ledger.updateSession(id,{status:'paused',stopReason:'client-context-lost'});for(const a of data.activities){if(a.state==='planned')await ledger.transitionActivity(a.id,{expected:['planned'],patch:{state:'cancelled',reason:'client-context-lost'}});else if(['started','completing'].includes(a.state))await ledger.transitionActivity(a.id,{expected:[a.state],patch:{state:'uncertain',reason:'client-context-lost'}})}onChange(id);return ledger.getSession(id);
 }
 async function restore(preferred){check();const all=await ledger.all(),sessions=Object.values(all.sessions).filter(s=>s.status!=='closed').sort((a,b)=>b.startedAt-a.startedAt),chosen=sessions.find(s=>s.id===preferred),pending=sessions.find(s=>s.startedAt>=(chosen?.startedAt??-Infinity)&&(Object.values(all.activities).some(a=>a.sessionId===s.id&&!a.source.manual&&['started','completing','uncertain','awaiting-evidence'].includes(a.state))||Object.values(all.clocks).some(c=>c.sessionId===s.id&&c.state!=='confirmed'))),active=sessions.find(s=>s.status==='running')??pending??sessions.find(s=>s.status==='recording'),s=active??chosen??sessions[0];return s?recover(s.id):null}
 async function reconcile(id){check();const data=await ledger.snapshot(id);if(!data.session||data.session.status!=='paused')throw Error('paused-session-required');
  for(const c of data.clocks.filter(c=>c.state!=='confirmed')){await clock.reconcile(c);check()}
  for(const a of data.activities.filter(a=>!a.source.manual&&['uncertain','completing','started'].includes(a.state))){const result=await ownerOperations.reconcile?.(a);check();if(result?.status==='confirmed')await ledger.transitionActivity(a.id,{expected:[a.state],patch:{...result,state:'confirmed',reconciled:true}})}
  const fresh=await ledger.snapshot(id),confirmed=fresh.clocks.filter(c=>c.state==='confirmed');if(confirmed.length)await ledger.updateSession(id,{cursorAt:Math.max(data.session.cursorAt,...confirmed.map(c=>c.to))});onChange(id);return snapshot(id);
 }
 async function review(id,{note,userId}={}){check();if(locks.has(id)||runners.has(id))throw Error('session-in-flight');if(typeof note!=='string'||!note.trim())throw Error('review-note-required');const data=await ledger.snapshot(id);if(data.session?.status!=='paused')throw Error('paused-session-required');const review={note:note.trim().slice(0,2000),userId,at:now(),disposition:'closed-without-replay'};
  for(const a of data.activities.filter(a=>['uncertain','awaiting-evidence','completing','started'].includes(a.state)))await ledger.transitionActivity(a.id,{expected:[a.state],patch:{review}});for(const c of data.clocks.filter(c=>c.state!=='confirmed'))await ledger.transitionClockCommit(c.id,{expected:[c.state],patch:{review}});await ledger.updateSession(id,{status:'closed',stopReason:'gm-reviewed-closed',review});onChange(id);return ledger.getSession(id);
 }
 async function resume(id,{autoRun=true}={}){check();const data=await snapshot(id);if(data.session?.status!=='paused'||data.session.manual)throw Error('paused-automatic-session-required');if(data.activities.some(a=>['uncertain','awaiting-evidence','completing','started'].includes(a.state))||data.clocks.some(c=>c.state!=='confirmed'))throw Error('unresolved-evidence-no-replay');if(now()!==data.session.cursorAt)throw Error('external-world-time-change');await ledger.updateSession(id,{status:'running',stopReason:null});if(autoRun)void drive(id).catch(error=>onChange({error:error.message}));return ledger.getSession(id)}
 return {start,step,stop,resume,recover,restore,reconcile,review,addActivity,snapshot};
}
