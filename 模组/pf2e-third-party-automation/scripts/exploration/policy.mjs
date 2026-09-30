export function expectedHealingPerMinute({outcomeForFace,meanForOutcome,expectedDamage=0,durationSeconds,assuranceOutcome}){
 const mean=assuranceOutcome!==undefined?meanForOutcome(assuranceOutcome):Array.from({length:20},(_,i)=>meanForOutcome(outcomeForFace(i+1))).reduce((a,b)=>a+b,0)/20;
 return (mean-expectedDamage)/(durationSeconds/60);
}
export function chooseNext({snapshot,proposals,session,now}){
 const active=(snapshot.activities??[]).filter(a=>a.state==='started'),actors=new Set(active.map(a=>a.actorUUID)),patients=new Set(active.filter(a=>a.providerId==='treat-wounds'||a.options?.threePecks).flatMap(a=>a.patientUUIDs)),pools=new Set(active.flatMap(a=>a.hpPoolUUIDs));
 const available=proposals.filter(p=>!actors.has(p.actorUUID)&&!(p.patientTreatmentExclusive&&p.patientUUIDs.some(u=>patients.has(u)))&&!p.hpPoolUUIDs.some(u=>pools.has(u)));
 const ranked=available.filter(p=>p.earliestStart<=now&&now+p.durationSeconds<=session.budgetEndsAt).sort((a,b)=>((b.expectedNetHealing??0)/b.durationSeconds)-((a.expectedNetHealing??0)/a.durationSeconds)||(a.resourceCost?.focus??0)-(b.resourceCost?.focus??0)||(a.expectedDamage??0)-(b.expectedDamage??0)||a.actorUUID.localeCompare(b.actorUUID)||a.patientUUIDs.join().localeCompare(b.patientUUIDs.join()));
 const activities=[];for(const p of ranked){if(actors.has(p.actorUUID)||p.hpPoolUUIDs.some(u=>pools.has(u))||p.patientTreatmentExclusive&&p.patientUUIDs.some(u=>patients.has(u)))continue;activities.push({...p,startedAt:now,endsAt:now+p.durationSeconds});actors.add(p.actorUUID);p.hpPoolUUIDs.forEach(u=>pools.add(u));if(p.patientTreatmentExclusive)p.patientUUIDs.forEach(u=>patients.add(u))}
 const checkpoints=[...active.map(a=>a.endsAt),...activities.map(a=>a.endsAt),...available.filter(p=>p.earliestStart>now&&p.earliestStart+p.durationSeconds<=session.budgetEndsAt).map(p=>p.earliestStart)].filter(t=>t>now&&t<=session.budgetEndsAt);
 return {activities,checkpointAt:checkpoints.length?Math.min(...checkpoints):null,reason:checkpoints.length?'ready':available.some(p=>now+p.durationSeconds>session.budgetEndsAt)?'budget':'blocked'};
}
export function recoveryProposals({actors,activities,session,now,providerIds}){
 const targets=actors.filter(p=>p.pool?.ready&&!p.isDead&&session.goalsByPool.some(g=>g.poolUUID===p.pool.poolUUID&&p.hp.value<g.targetHP));
 const deficit=p=>Math.max(0,(session.goalsByPool.find(g=>g.poolUUID===p.pool.poolUUID)?.targetHP??p.hp.max)-p.hp.value);
 const ready=p=>Math.max(now,p.cooldownExpiresAt??now,...activities.filter(a=>a.state==='confirmed'&&(a.providerId==='treat-wounds'||a.options?.threePecks)&&a.patientUUIDs.includes(p.actorUUID)&&!a.options?.extensionOf).map(a=>a.startedAt+(a.options.continualRecovery?600:3600)));
 const proposals=[];
 for(const h of actors.filter(a=>!a.isDead&&!a.unconscious)){
  const options={skill:'medicine',rank:session.treatmentRank??'trained',assurance:session.useAssurance===true&&h.assuranceSkills.includes('medicine'),riskySurgery:session.riskySurgery===true&&h.riskySurgery,continualRecovery:h.continualRecovery};
  const base={actorUUID:h.actorUUID,earliestStart:now,actorExclusive:true,patientTreatmentExclusive:true,resourceCost:{},options};
  if(providerIds.includes('treat-wounds')&&(h.medicine?.rank??0)>=1&&!h.unsupported?.length){
   const eligible=targets.filter(p=>p.modeOfBeing==='living'||h.slugs.includes('stitch-flesh')).sort((a,b)=>deficit(b)-deficit(a)||a.actorUUID.localeCompare(b.actorUUID));
   for(const patient of eligible)proposals.push({...base,providerId:'treat-wounds',patientUUIDs:[patient.actorUUID],hpPoolUUIDs:[patient.pool.poolUUID],durationSeconds:600,earliestStart:ready(patient),expectedNetHealing:Math.min(deficit(patient),9),expectedDamage:options.riskySurgery?4.5:0});
   const group=[...new Map(eligible.filter(p=>ready(p)===now).map(p=>[p.pool.poolUUID,p])).values()].slice(0,h.wardCapacity??1);
   if(group.length>1)proposals.push({...base,providerId:'treat-wounds',patientUUIDs:group.map(p=>p.actorUUID),hpPoolUUIDs:group.map(p=>p.pool.poolUUID),durationSeconds:600,expectedNetHealing:group.reduce((n,p)=>n+Math.min(deficit(p),9),0),expectedDamage:options.riskySurgery?group.length*4.5:0});
  }
  const spell=h.items?.find(i=>i.sourceId==='Compendium.pf2e.spells-srd.Item.zNN9212H2FGfM7VS');
  if(spell&&providerIds.includes('focus-healing')&&h.focus.value>0)for(const patient of targets.filter(p=>p.modeOfBeing==='living'))proposals.push({...base,providerId:'focus-healing',patientUUIDs:[patient.actorUUID],hpPoolUUIDs:[patient.pool.poolUUID],durationSeconds:6,expectedNetHealing:Math.min(deficit(patient),6*Math.max(1,Math.ceil((h.level??1)/2))),patientTreatmentExclusive:false,resourceCost:{focus:1},options:{itemUUID:spell.uuid}});
  if(providerIds.includes('refocus')&&(h.focus.value<h.focus.max&&(spell&&targets.length||session.requireFullFocus))){proposals.push({...base,providerId:'refocus',patientUUIDs:[],hpPoolUUIDs:[],durationSeconds:600,patientTreatmentExclusive:false,expectedNetHealing:0,options:{}})}
  if(providerIds.includes('refocus')&&h.threePecks&&!h.unsupported?.length)for(const patient of targets.filter(p=>p.modeOfBeing==='living'))proposals.push({...base,providerId:'refocus',patientUUIDs:[patient.actorUUID],hpPoolUUIDs:[patient.pool.poolUUID],durationSeconds:600,earliestStart:ready(patient),expectedNetHealing:Math.min(deficit(patient),9),options:{...options,threePecks:true,skill:(h.occultism?.rank??0)>=(h.medicine?.rank??0)?'occultism':'medicine'}});
 }
 return proposals;
}
