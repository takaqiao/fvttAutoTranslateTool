import {recoveryProposals} from './policy.mjs';
export function createCoordinator({ledger,capabilities,providers,clock,policy,isAuthority,now=()=>globalThis.game.time.worldTime,ownerOperations,onChange=()=>{}}){
 const byId=new Map(providers.map(p=>[p.id,p])),locks=new Set(),contexts=new Map(),runners=new Map();
 const check=()=>{if(!isAuthority())throw Error('active-gm-required')};
 async function snapshot(id){const data=await ledger.snapshot(id);return {...data,actors:data.session?await capabilities.snapshot(data.session.actorUUIDs):[]}}
 async function stop(id,reason='user-stopped'){check();const result=await ledger.updateSession(id,{status:'paused',stopReason:reason});onChange(id);return result}
 async function addActivity(id,input){check();const activity=await ledger.insertActivity({...input,id:input.id??crypto.randomUUID(),sessionId:id,state:'planned',source:input.source??{type:'coordinator'}});const provider=byId.get(activity.providerId);if(!provider)throw Error('unknown-provider');
  const ctx=await ownerOperations.createActivityContext(activity);check();const begun=await provider.begin(activity,ctx);check();
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
   if(data.activities.filter(a=>!a.source.manual).length>=s.maxActivities||now()>=s.budgetEndsAt)return stop(id,'budget');
   const proposals=recoveryProposals({actors:data.actors,activities:data.activities,session:s,now:now(),providerIds:[...byId.keys()]});
   const next=policy({snapshot:data,proposals,session:s,now:now()});
   for(const proposal of next.activities.slice(0,s.maxActivities-data.activities.filter(a=>!a.source.manual).length)){check();if((await ledger.getSession(id)).status!=='running')return {status:'paused'};const added=await addActivity(id,{...proposal,source:{type:'coordinator'},groupId:undefined});if(added.state==='blocked')return stop(id,added.reason)}
   if(next.checkpointAt===null)return stop(id,next.reason);
   if((await ledger.getSession(id)).status!=='running')return {status:'paused'};
   const receipt=await clock.advanceTo({id:crypto.randomUUID(),sessionId:id,from:now(),to:next.checkpointAt});check();if(receipt.status!=='confirmed')return stop(id,receipt.reason??'clock-unconfirmed');
   await ledger.updateSession(id,{cursorAt:now()});data=await snapshot(id);check();
   for(const a of data.activities.filter(a=>a.state==='started'&&a.endsAt<=now())){
    check();if((await ledger.getSession(id)).status!=='running')return {status:'paused'};const claimed=await ledger.transitionActivity(a.id,{expected:['started'],patch:{state:'completing'}});
    let result;try{result=await byId.get(a.providerId).complete(claimed,contexts.get(a.id));check()}catch(error){result={status:'uncertain',reason:error.message}}
    check();await ledger.transitionActivity(a.id,{expected:['completing'],patch:{...result,state:result.status==='confirmed'?'confirmed':'uncertain'}});contexts.delete(a.id);
    if(result.status!=='confirmed')return stop(id,result.reason??'native-result-unconfirmed');
    if(s.extendTreatment&&a.providerId==='treat-wounds'&&!a.options.extensionOf&&['success','criticalSuccess'].includes(result.effectiveOutcome)&&a.startedAt+3600<=s.budgetEndsAt){
     const fresh=await capabilities.snapshot(a.patientUUIDs);check();if(fresh.some(p=>p.hp.value<(s.goalsByPool.find(g=>g.poolUUID===p.pool.poolUUID)?.targetHP??0)))await addActivity(id,{providerId:a.providerId,actorUUID:a.actorUUID,patientUUIDs:a.patientUUIDs,hpPoolUUIDs:a.hpPoolUUIDs,startedAt:now(),endsAt:a.startedAt+3600,options:{...a.options,extensionOf:a.id},source:{type:'coordinator'}});
    }
   }
   onChange(id);return {status:'running'};
  }catch(error){if(isAuthority())await stop(id,error.message);return {status:'paused',reason:error.message}}
  finally{locks.delete(id)}
 }
 async function drive(id){if(runners.has(id))return;runners.set(id,true);try{while(isAuthority()&&(await ledger.getSession(id))?.status==='running'){await step(id);await Promise.resolve()}}finally{runners.delete(id)}}
 async function start(config){check();const actors=await capabilities.snapshot(config.actorUUIDs);check();const at=now();const goals=config.goalsByPool??[...new Map(actors.map(a=>[a.pool.poolUUID,{poolUUID:a.pool.poolUUID,targetHP:a.hp.max}])).values()];
  if(!actors.length||goals.some(g=>!actors.some(a=>a.pool.poolUUID===g.poolUUID&&Number.isFinite(g.targetHP)&&g.targetHP>=0&&g.targetHP<=a.hp.max)))throw Error('invalid-recovery-goals');
  const s=await ledger.createSession({...config,id:config.id??crypto.randomUUID(),actorUUIDs:[...new Set(config.actorUUIDs)],startedAt:at,cursorAt:at,budgetEndsAt:at+Math.min(7200,Math.max(0,config.budgetSeconds??7200)),maxActivities:Math.min(100,Math.max(1,config.maxActivities??100)),goalsByPool:goals,status:config.manual?'recording':'running',assumptions:config.assumptions??['different-actors-may-overlap']});
  if(config.autoRun!==false&&!config.manual)void drive(s.id).catch(error=>onChange({error:error.message}));onChange(s.id);return s;
 }
 async function resume(id){check();const data=await snapshot(id);if(data.activities.some(a=>['uncertain','awaiting-evidence','completing','started'].includes(a.state))||data.clocks.some(c=>c.state!=='confirmed'))throw Error('unresolved-evidence-no-replay');await ledger.updateSession(id,{status:'running',stopReason:null,cursorAt:now()});void drive(id);return ledger.getSession(id)}
 return {start,step,stop,resume,addActivity,snapshot};
}
