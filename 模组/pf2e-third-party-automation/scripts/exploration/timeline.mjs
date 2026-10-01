/** Constraint reconstruction only. This module cannot change the world clock. */
export function reconstructEarliest({startedAt,activities,assumptions=[]}) {
 if(!Number.isFinite(startedAt))throw Error('invalid-session-start');
 const missing=[],nodes=[],groups=new Map(),aliases=new Map();
 for(const [index,a]of activities.entries()){
  for(const reason of a.options?.missing??[])missing.push({id:a.id,reason});
  if(['uncertain','blocked','cancelled'].includes(a.state))missing.push({id:a.id,reason:`activity-${a.state}`});
  if(!a.id||!a.actorUUID||!Number.isFinite(a.durationSeconds??a.endsAt-a.startedAt)){missing.push({id:a.id,reason:'missing-activity-source'});continue}
  const duration=a.durationSeconds??a.endsAt-a.startedAt;
  let invalid;
  if(duration<0||duration===0&&a.kind!=='activity')invalid='invalid-duration';
  else if(a.order!==undefined&&(!Number.isSafeInteger(a.order)||a.order<0))invalid='invalid-actor-order';
  else if(a.notBefore!==undefined&&!Number.isFinite(a.notBefore))invalid='invalid-not-before';
  else if((a.observedStart!==undefined||a.observedEnd!==undefined)&&(!Number.isFinite(a.observedStart)||!Number.isFinite(a.observedEnd)||a.observedEnd<a.observedStart))invalid='invalid-observed-interval';
  else if(a.dependsOn!==undefined&&(!Array.isArray(a.dependsOn)||[...a.dependsOn].some(id=>typeof id!=='string'||!id.trim())))invalid='invalid-dependency';
  else if(aliases.has(a.id))invalid='duplicate-activity-source';
  if(invalid){missing.push({id:a.id,reason:invalid});continue}
  const key=a.groupProof&&a.groupId?`${a.groupId}:${a.groupProof}`:a.id;
  if(a.groupId&&a.groupId!==a.id&&!a.groupProof)missing.push({id:a.id,reason:'unproven-group'});
  let n=groups.get(key);
  if(n){if(n.actorUUID!==a.actorUUID||n.durationSeconds!==(a.durationSeconds??a.endsAt-a.startedAt)){missing.push({id:a.id,reason:'inconsistent-group'});continue}n.patientUUIDs=[...new Set([...n.patientUUIDs,...a.patientUUIDs??[]])];n.ids.push(a.id);}
  else{n={...a,id:key,ids:[a.id],index,patientUUIDs:[...a.patientUUIDs??[]],durationSeconds:a.durationSeconds??a.endsAt-a.startedAt,deps:new Set(a.dependsOn??[])};groups.set(key,n);nodes.push(n)}aliases.set(a.id,key);
 }
 const byActor=new Map(),byPatient=new Map();
 for(const n of nodes){const list=byActor.get(n.actorUUID)??[];list.push(n);byActor.set(n.actorUUID,list);if(n.kind==='treatment'||n.providerId==='treat-wounds'||n.options?.threePecks)for(const p of n.patientUUIDs){const l=byPatient.get(p)??[];l.push(n);byPatient.set(p,l)}}
 for(const list of byActor.values()){
  if(list.some(n=>!Number.isFinite(n.order)))missing.push({id:list[0].id,reason:'missing-actor-order'});
  list.sort((a,b)=>(a.order??a.index)-(b.order??b.index)||a.index-b.index);
  for(let i=1;i<list.length;i++)list[i].deps.add(list[i-1].id);
 }
 // Patient history follows the supplied evidence sequence; it is never optimized by permutation.
 for(const list of byPatient.values()){list.sort((a,b)=>(a.patientOrder??a.index)-(b.patientOrder??b.index));for(let i=1;i<list.length;i++)list[i].deps.add(list[i-1].id)}
 for(const n of nodes)n.deps=new Set([...n.deps].map(id=>aliases.get(id)??id));
 const pending=new Map(nodes.map(n=>[n.id,n])),done=new Map(),patientReady=new Map(),actorFree=new Map(),scheduled=[];let contradictory=false;
 while(pending.size){let progress=false;
  for(const [id,n]of pending){if([...n.deps].some(d=>!done.has(d)))continue;
   let earliest=Math.max(startedAt,n.notBefore??startedAt,actorFree.get(n.actorUUID)??startedAt,...n.patientUUIDs.map(p=>patientReady.get(p)??startedAt),...[...n.deps].map(d=>done.get(d).endsAt));
   // Across actors concurrency must be explicitly accepted or directly observed.
   if(!assumptions.includes('different-actors-may-overlap')&&!Number.isFinite(n.observedStart))earliest=Math.max(earliest,...scheduled.map(x=>x.endsAt));
   const start=Number.isFinite(n.observedStart)?n.observedStart:earliest,end=Number.isFinite(n.observedEnd)?n.observedEnd:start+n.durationSeconds;
   if(!Number.isFinite(earliest)||!Number.isFinite(start+n.durationSeconds)||!Number.isFinite(end)){missing.push({id,reason:'invalid-computed-time'});pending.delete(id);progress=true;continue}
   if(start<earliest||end<start+n.durationSeconds){contradictory=true;missing.push({id,reason:'observed-before-ready',earliest,observedStart:start})}
   const row={id,activityIds:n.ids,actorUUID:n.actorUUID,patientUUIDs:n.patientUUIDs,startedAt:start,endsAt:end,...Object.fromEntries(['source','temporalSource','durationSource'].filter(key=>n[key]!==undefined).map(key=>[key,structuredClone(n[key])]))};scheduled.push(row);done.set(id,row);pending.delete(id);progress=true;actorFree.set(n.actorUUID,end);
   const immune=n.treatmentImmunitySeconds??n.options?.treatmentImmunitySeconds;
   if(Number.isFinite(immune))for(const p of n.patientUUIDs)patientReady.set(p,Math.max(patientReady.get(p)??startedAt,start+immune));
  }
  if(!progress){for(const n of pending.values())missing.push({id:n.id,reason:[...n.deps].some(d=>!groups.has(d))?'missing-dependency':'dependency-cycle'});break}
 }
 const endsAt=Math.max(startedAt,...scheduled.map(a=>a.endsAt));
 const observed=nodes.length>0&&nodes.every(a=>Number.isFinite(a.observedStart)&&Number.isFinite(a.observedEnd)&&a.source?.type!=='user-record'&&a.temporalSource?.type!=='user-declared');
 return {endsAt,durationSeconds:endsAt-startedAt,certainty:contradictory?'contradictory':missing.length?'incomplete':observed?'observed':'earliest-under-assumptions',scheduled,missing,assumptions:[...assumptions]};
}
