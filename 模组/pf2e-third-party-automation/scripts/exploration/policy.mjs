import {selectTreatmentRank} from './efficiency.mjs';
import {treatablePatient} from './capabilities.mjs';
import {treatmentFailureState} from './treatment-streaks.mjs';
import {patientRecoveryNeed} from './recovery-goals.mjs';
export const checkpointDeclaration=a=>a.temporalSource?.type==='checkpoint-declaration';
export function scheduleCheckpointActivities({registrations,activities,session,from}){
 const reservations=activities.filter(a=>['planned','started'].includes(a.state)),planned=[];
 const rows=Object.values(registrations).map((row,index)=>({...row,index:row.registrationOrder??index})).sort((a,b)=>(a.declaration.order??a.index)-(b.declaration.order??b.index)||a.index-b.index);
 for(const row of rows){
  const d=row.declaration;let start=Math.max(from,d.notBefore??from);
  for(const id of d.dependsOn){const dependency=activities.find(a=>a.id===id);if(!dependency||dependency.sessionId!==session.id||!['planned','started','confirmed'].includes(dependency.state))throw Error('invalid-manual-dependency');start=Math.max(start,dependency.endsAt)}
  for(const prior of planned.filter(a=>a.actorUUID===d.actorUUID))start=Math.max(start,prior.endsAt);
  for(const reservation of reservations.filter(a=>a.actorUUID===d.actorUUID).sort((a,b)=>a.startedAt-b.startedAt)){
   if(start<reservation.endsAt&&(start+d.durationSeconds>reservation.startedAt||d.durationSeconds===0&&start>=reservation.startedAt))start=reservation.endsAt;
  }
  const end=start+d.durationSeconds;if(!Number.isFinite(end)||end>session.budgetEndsAt)throw Error('activity-checkpoint-budget');
  planned.push({registrationId:row.registrationId,actorUUID:d.actorUUID,startedAt:start,endsAt:end});
 }
 return planned;
}
export function expectedHealingPerMinute({outcomeForFace,meanForOutcome,expectedDamage=0,durationSeconds,assuranceOutcome}){
 const mean=assuranceOutcome!==undefined?meanForOutcome(assuranceOutcome):Array.from({length:20},(_,i)=>meanForOutcome(outcomeForFace(i+1))).reduce((a,b)=>a+b,0)/20;
 return (mean-expectedDamage)/(durationSeconds/60);
}
export function chooseNext({snapshot,proposals,session,now}){
 const declarations=(snapshot.activities??[]).filter(a=>checkpointDeclaration(a)&&['planned','started'].includes(a.state));
  const active=(snapshot.activities??[]).filter(a=>a.state==='started'||a.temporalSource?.type==='checkpoint-reservation'&&a.checkpointBinding?.id===session.manualCheckpoint?.id&&['awaiting-evidence','uncertain'].includes(a.state)),actors=new Set(active.map(a=>a.actorUUID)),patients=new Set(active.filter(a=>a.providerId==='treat-wounds'||a.options?.threePecks||a.temporalSource?.type==='checkpoint-reservation').flatMap(a=>a.patientUUIDs)),pools=new Set(active.flatMap(a=>a.hpPoolUUIDs));
 const available=proposals.filter(p=>!actors.has(p.actorUUID)&&!(p.patientTreatmentExclusive&&p.patientUUIDs.some(u=>patients.has(u)))&&!p.hpPoolUUIDs.some(u=>pools.has(u))&&!declarations.some(a=>a.actorUUID===p.actorUUID&&Math.max(now,p.earliestStart)<a.endsAt&&a.startedAt<Math.max(now,p.earliestStart)+p.durationSeconds));
 const ranked=available.filter(p=>p.earliestStart<=now&&now+p.durationSeconds<=session.budgetEndsAt).sort((a,b)=>((b.expectedNetHealing??0)/b.durationSeconds)-((a.expectedNetHealing??0)/a.durationSeconds)||(a.resourceCost?.focus??0)-(b.resourceCost?.focus??0)||(a.expectedDamage??0)-(b.expectedDamage??0)||a.actorUUID.localeCompare(b.actorUUID)||a.patientUUIDs.join().localeCompare(b.patientUUIDs.join()));
 const activities=[];for(const p of ranked){if(actors.has(p.actorUUID)||p.hpPoolUUIDs.some(u=>pools.has(u))||p.patientTreatmentExclusive&&p.patientUUIDs.some(u=>patients.has(u)))continue;activities.push({...p,startedAt:now,endsAt:now+p.durationSeconds});actors.add(p.actorUUID);p.hpPoolUUIDs.forEach(u=>pools.add(u));if(p.patientTreatmentExclusive)p.patientUUIDs.forEach(u=>patients.add(u))}
 const checkpoints=[...declarations.flatMap(a=>[a.startedAt,a.endsAt]),...active.map(a=>a.endsAt),...activities.map(a=>a.endsAt),...available.filter(p=>p.earliestStart>now&&p.earliestStart+p.durationSeconds<=session.budgetEndsAt).map(p=>p.earliestStart)].filter(t=>t>now&&t<=session.budgetEndsAt);
 return {activities,checkpointAt:checkpoints.length?Math.min(...checkpoints):null,reason:checkpoints.length?'ready':available.some(p=>now+p.durationSeconds>session.budgetEndsAt)?'budget':'blocked'};
}
export function recoveryProposals({actors,activities,session,now,providerIds}){
 const failures=treatmentFailureState(activities,session.recoveryGoals?.failureStop,{sessionId:session.id,activityIds:session.activityIds}),blocked=new Set(failures.blockedPatientUUIDs);
 const patients=actors.filter(p=>!blocked.has(p.actorUUID)&&p.pool?.ready&&!p.isDead),targets=patients.filter(p=>patientRecoveryNeed(p,session).hp),treatmentTargets=patients.filter(p=>{const need=patientRecoveryNeed(p,session);return need.hp||need.wounded});
 const deficit=p=>Math.max(0,(session.goalsByPool.find(g=>g.poolUUID===p.pool.poolUUID)?.targetHP??p.hp.value)-p.hp.value);
 const ready=p=>Math.max(now,p.cooldownExpiresAt??now,...activities.filter(a=>a.state==='confirmed'&&(a.providerId==='treat-wounds'||a.options?.threePecks||a.temporalSource?.type==='checkpoint-reservation')&&a.patientUUIDs.includes(p.actorUUID)&&!a.options?.extensionOf).map(a=>a.startedAt+(a.options.continualRecovery?600:3600)));
 const proposals=[];
 for(const h of actors.filter(a=>!a.isDead&&!a.unconscious)){
  const options={skill:'medicine',rank:session.treatmentRank??'trained',assurance:session.useAssurance===true&&h.assuranceSkills.includes('medicine'),riskySurgery:session.riskySurgery===true&&h.riskySurgery,continualRecovery:h.continualRecovery};
  const base={actorUUID:h.actorUUID,earliestStart:now,actorExclusive:true,patientTreatmentExclusive:true,resourceCost:{},options};
  if(providerIds.includes('treat-wounds')&&(h.medicine?.rank??0)>=1&&!h.unsupported?.length){
   const eligible=treatmentTargets.filter(p=>treatablePatient(p,h)).sort((a,b)=>deficit(b)-deficit(a)||a.actorUUID.localeCompare(b.actorUUID));
   for(const patient of eligible){const estimate=selectTreatmentRank(h,[patient],{...options,treatmentRank:session.treatmentRank},deficit);if(estimate)proposals.push({...base,providerId:'treat-wounds',patientUUIDs:[patient.actorUUID],hpPoolUUIDs:[patient.pool.poolUUID],durationSeconds:600,earliestStart:ready(patient),options:{...options,rank:estimate.rank,estimate:estimate.estimate,...estimate.estimateReason?{estimateReason:estimate.estimateReason}:{},...estimate.estimateSource?{estimateSource:estimate.estimateSource}:{}},expectedNetHealing:estimate.expectedNetHealing,expectedDamage:estimate.expectedHPDamage??0})}
   const group=[...new Map(eligible.filter(p=>ready(p)===now).map(p=>[p.pool.poolUUID,p])).values()].slice(0,h.wardCapacity??1);
   if(group.length>1){const estimate=selectTreatmentRank(h,group,{...options,treatmentRank:session.treatmentRank},deficit);if(estimate)proposals.push({...base,providerId:'treat-wounds',patientUUIDs:group.map(p=>p.actorUUID),hpPoolUUIDs:group.map(p=>p.pool.poolUUID),durationSeconds:600,options:{...options,rank:estimate.rank,estimate:estimate.estimate,...estimate.estimateReason?{estimateReason:estimate.estimateReason}:{},...estimate.estimateSource?{estimateSource:estimate.estimateSource}:{}},expectedNetHealing:estimate.expectedNetHealing,expectedDamage:estimate.expectedHPDamage??0})}
  }
  if(providerIds.includes('treat-wounds')&&h.slugs.includes('natural-medicine')&&(h.nature?.rank??0)>=1&&!h.unsupported?.length){
   const eligible=treatmentTargets.filter(p=>p.modeOfBeing==='living').sort((a,b)=>deficit(b)-deficit(a)||a.actorUUID.localeCompare(b.actorUUID));
   const natureOptions={skill:'nature',riskySurgery:false,continualRecovery:h.continualRecovery,assurance:session.useAssurance===true&&h.assuranceSkills.includes('nature')};
   for(const patient of eligible){const estimate=selectTreatmentRank(h,[patient],{...natureOptions,treatmentRank:session.treatmentRank},deficit);if(estimate)proposals.push({...base,providerId:'treat-wounds',patientUUIDs:[patient.actorUUID],hpPoolUUIDs:[patient.pool.poolUUID],durationSeconds:600,earliestStart:ready(patient),options:{...natureOptions,rank:estimate.rank,estimate:estimate.estimate,...estimate.estimateReason?{estimateReason:estimate.estimateReason}:{},...estimate.estimateSource?{estimateSource:estimate.estimateSource}:{}},expectedNetHealing:estimate.expectedNetHealing,expectedDamage:estimate.expectedHPDamage??0})}
   const group=[],pools=new Set();
   for(const patient of eligible){
    if(ready(patient)!==now||pools.has(patient.pool.poolUUID))continue;
    group.push(patient);pools.add(patient.pool.poolUUID);
    if(group.length>=(h.wardCapacity??1))break;
   }
   if(group.length>1){const estimate=selectTreatmentRank(h,group,{...natureOptions,treatmentRank:session.treatmentRank},deficit);if(estimate)proposals.push({...base,providerId:'treat-wounds',patientUUIDs:group.map(p=>p.actorUUID),hpPoolUUIDs:group.map(p=>p.pool.poolUUID),durationSeconds:600,options:{...natureOptions,rank:estimate.rank,estimate:estimate.estimate,...estimate.estimateReason?{estimateReason:estimate.estimateReason}:{},...estimate.estimateSource?{estimateSource:estimate.estimateSource}:{}},expectedNetHealing:estimate.expectedNetHealing,expectedDamage:estimate.expectedHPDamage??0})}
  }
  const spell=h.items?.find(i=>i.sourceId==='Compendium.pf2e.spells-srd.Item.zNN9212H2FGfM7VS');
  if(spell&&providerIds.includes('focus-healing')&&h.focus.value>0)for(const patient of targets.filter(p=>p.vitalityHealingReady===true))proposals.push({...base,providerId:'focus-healing',patientUUIDs:[patient.actorUUID],hpPoolUUIDs:[patient.pool.poolUUID],durationSeconds:6,expectedNetHealing:Math.min(deficit(patient),6*Math.max(1,Math.ceil((h.level??1)/2))),patientTreatmentExclusive:false,resourceCost:{focus:1},options:{itemUUID:spell.uuid}});
  if(providerIds.includes('refocus')&&!h.refocusUnsupported?.length&&(h.focus.value<h.focus.max&&(spell&&targets.length||session.requireFullFocus))){proposals.push({...base,providerId:'refocus',patientUUIDs:[],hpPoolUUIDs:[],durationSeconds:600,patientTreatmentExclusive:false,expectedNetHealing:0,options:{}})}
  if(providerIds.includes('refocus')&&h.threePecks&&!h.refocusUnsupported?.length&&!h.unsupported?.length)for(const patient of targets.filter(p=>p.modeOfBeing==='living'))proposals.push({...base,providerId:'refocus',patientUUIDs:[patient.actorUUID],hpPoolUUIDs:[patient.pool.poolUUID],durationSeconds:600,earliestStart:ready(patient),expectedNetHealing:Math.min(deficit(patient),9),options:{...options,threePecks:true,skill:(h.occultism?.rank??0)>=(h.medicine?.rank??0)?'occultism':'medicine'}});
 }
 return proposals;
}
