const ranks=['trained','expert','master','legendary'];
const clone=value=>structuredClone(value);
const poolOf=actor=>actor.pool.poolUUID;
const meanHealing=model=>model.constant+model.dice.reduce((sum,die)=>sum+die.number*(die.faces+1)/2,0);
const able=actor=>!actor.isDead&&!actor.unconscious&&actor.canAct!==false&&actor.hp.value>0&&!actor.needsReview;
const treatmentKey=model=>JSON.stringify([model.skill,model.rank,model.assurance,model.riskySurgery]);
const receivedBonus=(patient,medicBonus=0)=>Math.max(medicBonus,patient?.computed?.patient?.robustHealth??0)-medicBonus;
const treatmentPatientType=(patient,healer,skill,threePecks=false)=>patient.modeOfBeing==='living'||!threePecks&&skill==='medicine'&&patient.modeOfBeing==='undead'&&healer.slugs?.includes('stitch-flesh');

function jsonData(value){
 if(Array.isArray(value))return value.map(jsonData);
 if(value&&typeof value==='object')return Object.fromEntries(Object.entries(value).filter(([,v])=>v!==undefined).map(([k,v])=>[k,jsonData(v)]));
 if(typeof value==='number'&&!Number.isFinite(value)||typeof value==='function')throw Error('computed-state-not-serializable');
 return value;
}
function actorData(actors){
 const fields=['actorUUID','name','systemVersion','type','level','isDead','unconscious','wounded','canAct','modeOfBeing','slugs','hp','focus','pool','computed','executionStamp','needsReview'];
 const rows=actors.map(actor=>jsonData({...Object.fromEntries(fields.filter(key=>actor[key]!==undefined).map(key=>[key,actor[key]])),cooldownUnknown:actor.cooldownUnknown===true||actor.cooldownExpiresAt!=null&&!Number.isFinite(actor.cooldownExpiresAt),cooldownExpiresAt:Number.isFinite(actor.cooldownExpiresAt)?actor.cooldownExpiresAt:null}));
 const seen=new Set(),pools=new Map();
 for(const actor of rows){
  if(!actor.actorUUID||seen.has(actor.actorUUID)||!actor.pool?.ready||!actor.pool.poolUUID||!Number.isFinite(actor.hp?.value)||!Number.isFinite(actor.hp?.max)||actor.hp.value<0||actor.hp.value>actor.hp.max)throw Error('invalid-computed-actors');
  seen.add(actor.actorUUID);const prior=pools.get(poolOf(actor));
  if(prior&&['value','max','temp'].some(key=>prior.hp[key]!==actor.hp[key]))throw Error('inconsistent-patient-pool');
  pools.set(poolOf(actor),actor);
 }
 return rows;
}
function validTreatment(model){
 return model?.ready===true&&ranks.includes(model.rank)&&[15,20,30,40][ranks.indexOf(model.rank)]===model.dc&&Number.isFinite(model.modifier)&&
  typeof model.assurance==='boolean'&&typeof model.riskySurgery==='boolean'&&Array.isArray(model.outcomes)&&model.outcomes.length===(model.assurance?1:20)&&model.outcomes.every(n=>Number.isInteger(n)&&n>=0&&n<=3)&&
  Number.isFinite(model.bonus)&&model.bonus>=0&&model.durationSeconds===600&&[600,3600].includes(model.cooldownSeconds)&&Number.isInteger(model.capacity)&&model.capacity>=1&&model.capacity<=8;
}
function expectedTreatment(model,patient){
 const bonus=model.bonus+receivedBonus(patient,model.medicBonus);
 return model.outcomes.reduce((sum,outcome)=>sum+(outcome===0?-4.5:outcome===2?9+bonus:outcome===3?18+bonus:0),0)/model.outcomes.length-(model.riskySurgery?4.5:0);
}

/** Planning never samples. Settle only after the caller confirms the clock boundary. */
export function createRecoverySimulation({actors,session,random=Math.random,state}){
 const strategy=session.recoveryStrategy??session.strategy??'success-first';
 if(!['success-first','time-first'].includes(strategy)||!Number.isFinite(session.startedAt)||!Number.isFinite(session.budgetEndsAt)||session.budgetEndsAt<session.startedAt)throw Error('invalid-computed-session');
 let data=state?jsonData(clone(state)):{version:1,sessionId:session.id,cursorAt:session.startedAt,initialActors:actorData(actors),actors:actorData(actors),active:[],pending:null,history:[],failures:{},activityCount:0,stopReason:null,diagnostics:[]};
 if(data.version!==1||data.sessionId!==session.id||data.cursorAt<session.startedAt||data.cursorAt>session.budgetEndsAt)throw Error('invalid-computed-state');
 data.initialActors??=actorData(session.computedInitialActors??actors);
 const find=id=>data.actors.find(actor=>actor.actorUUID===id);
 const target=actor=>Math.min(actor.hp.max,session.goalsByPool?.find(goal=>goal.poolUUID===poolOf(actor))?.targetHP??actor.hp.value);
 const deficit=actor=>Math.max(0,target(actor)-actor.hp.value);
 const need=actor=>deficit(actor)>0||session.recoveryGoals?.requireNoWounded===true&&actor.wounded===true;
 const focusNeed=actor=>session.requireFullFocus===true&&actor.focus.value<actor.focus.max;
 const failureBlocked=actor=>session.recoveryGoals?.failureStop?.enabled===true&&(data.failures[actor.actorUUID]??0)>=session.recoveryGoals.failureStop.limit;
 const diagnose=reason=>{if(reason&&!data.diagnostics.includes(reason))data.diagnostics.push(reason)};
 const done=reason=>{data.stopReason=reason;return {done:true,reason}};
 const deadline=actor=>actor.cooldownUnknown?Infinity:Math.max(data.cursorAt,actor.cooldownExpiresAt??data.cursorAt);
 const currentModel=(actor,activity)=>activity.options.threePecks?actor.computed?.threePecks?.treatments?.find(model=>treatmentKey(model)===treatmentKey(activity.model)):actor.computed?.treatments?.find(model=>treatmentKey(model)===treatmentKey(activity.model));
 function refresh(fresh){
  const incoming=actorData(fresh);if(incoming.length!==data.actors.length||incoming.some(actor=>!find(actor.actorUUID)||poolOf(find(actor.actorUUID))!==poolOf(actor)))throw Error('computed-actor-domain-changed');
  for(const actor of incoming){const prior=find(actor.actorUUID);actor.cooldownExpiresAt=Math.max(actor.cooldownExpiresAt??0,prior.cooldownExpiresAt??0)||null;actor.cooldownUnknown||=prior.cooldownUnknown;actor.needsReview||=prior.needsReview===true}
  data.actors=incoming;
 }
 if(state&&actors){refresh(actors);if(data.stopReason!=='needs-review')data.stopReason=null}

 function patientReady(patient,model,vitality=false){
  const rules=patient.computed?.patient;
  if(patient.needsReview||patient.isDead||patient.hp.value<=0||failureBlocked(patient))return false;
  if(rules?.healingReady!==true){diagnose(rules?.healingReason??'patient-healing-model-unverified');return false}
  if(vitality&&(patient.modeOfBeing!=='living'||rules.vitalityReady!==true)){diagnose(rules.vitalityReason??'patient-vitality-unavailable');return false}
  if(model&&(model.riskySurgery||model.outcomes.includes(0))&&rules.damageReady!==true){diagnose(rules.damageReason??'patient-damage-model-unverified');return false}
  return true;
 }
 function proposals(){
  const result=[],patients=data.actors.filter(need),now=data.cursorAt;
  for(const healer of data.actors.filter(able)){
   const rules=healer.computed;
   if(rules?.version!==1){diagnose('computed-model-unavailable');continue}
   if(!rules.treatments?.length)for(const reason of rules.treatmentReasons??[])if(!['assurance-selection-unavailable','risky-selection-unavailable'].includes(reason))diagnose(reason);
   const addTreatment=(model,threePecks=false)=>{
    if(!validTreatment(model)){diagnose('computed-treatment-model-unverified');return}
    if(model.riskySurgery&&session.riskySurgery!==true)return;
    if(session.treatmentRank&&session.treatmentRank!=='auto'&&model.rank!==session.treatmentRank)return;
    // A deterministic failure cannot improve recovery, even as a last resort.
    if(!model.outcomes.some(outcome=>outcome>=2))return;
    const eligible=patients.filter(patient=>patientReady(patient,model,threePecks)&&treatmentPatientType(patient,healer,model.skill,threePecks))
     .sort((a,b)=>deficit(b)-deficit(a)||a.actorUUID.localeCompare(b.actorUUID));
    const groups=eligible.map(patient=>[patient]);
    const distinct=new Map();for(const patient of eligible.filter(patient=>deadline(patient)<=now))if(!distinct.has(poolOf(patient)))distinct.set(poolOf(patient),patient);
    const group=[...distinct.values()].slice(0,threePecks?1:model.capacity);if(group.length>1)groups.push(group);
    for(const group of groups){
     const start=Math.max(now,...group.map(deadline));if(!Number.isFinite(start)){diagnose('treatment-immunity-expiry-unknown');continue}
     result.push({providerId:threePecks?'refocus':'treat-wounds',actorUUID:healer.actorUUID,patientUUIDs:group.map(patient=>patient.actorUUID),hpPoolUUIDs:group.map(poolOf),earliestStart:start,durationSeconds:600,model,
      certain:model.outcomes.every(outcome=>outcome>=2),expectedHealing:Math.min(...group.map(patient=>expectedTreatment(model,patient))),options:{skill:model.skill,rank:model.rank,assurance:model.assurance,riskySurgery:model.riskySurgery,continualRecovery:model.cooldownSeconds===600,...threePecks?{threePecks:true}:{}}});
    }
   };
   for(const model of rules.treatments??[])addTreatment(model);
   if(rules.threePecks?.ready&&rules.refocus?.ready)for(const model of rules.threePecks.treatments)addTreatment(model,true);
   const focus=rules.focusHealing;
   if(focus?.ready&&healer.focus.value>=1)for(const patient of patients.filter(patient=>deficit(patient)>0&&patientReady(patient,null,true))){
    result.push({providerId:'focus-healing',actorUUID:healer.actorUUID,patientUUIDs:[patient.actorUUID],hpPoolUUIDs:[poolOf(patient)],earliestStart:now,durationSeconds:focus.durationSeconds,model:focus,certain:true,expectedHealing:meanHealing(focus.healing),options:{itemUUID:focus.itemUUID}});
   }
   const focusTargets=focus?.ready&&patients.some(patient=>deficit(patient)>0&&patientReady(patient,null,true));
   if(rules.refocus?.ready&&healer.focus.value<healer.focus.max&&(focusNeed(healer)||focusTargets)&&(!focusTargets||healer.focus.value===0)){
    result.push({providerId:'refocus',actorUUID:healer.actorUUID,patientUUIDs:[],hpPoolUUIDs:[],earliestStart:now,durationSeconds:600,model:rules.refocus,certain:true,expectedHealing:0,options:{}});
   }else if(focusNeed(healer)&&!rules.refocus?.ready)diagnose(rules.refocus?.reason??'refocus-recovery-unadapted');
   if(healer.wounded&&session.recoveryGoals?.requireNoWounded===true&&healer.hp.value===healer.hp.max){
    result.push({providerId:'rest',actorUUID:healer.actorUUID,patientUUIDs:[healer.actorUUID],hpPoolUUIDs:[],earliestStart:now,durationSeconds:600,model:{},certain:true,expectedHealing:0,options:{}});
   }
   if(session.extendTreatment){
    const groups=new Map();
    for(const event of data.history.filter(event=>event.actorUUID===healer.actorUUID&&event.providerId==='treat-wounds'&&!event.options.extensionOf&&event.endsAt===now&&event.outcome>=2)){
     const patient=find(event.patientUUID);if(!need(patient)||!patientReady(patient)||!treatmentPatientType(patient,healer,event.options.skill??'medicine'))continue;
     if(!groups.has(event.activityId))groups.set(event.activityId,[]);groups.get(event.activityId).push(event);
    }
    for(const [extensionOf,events] of groups){
     const model={healingByPatient:Object.fromEntries(events.map(event=>[event.patientUUID,event.baseHealing??event.healing])),medicBonusByPatient:Object.fromEntries(events.map(event=>[event.patientUUID,event.medicBonus??0]))};
     result.push({providerId:'treat-wounds',actorUUID:healer.actorUUID,patientUUIDs:events.map(event=>event.patientUUID),hpPoolUUIDs:events.map(event=>event.poolUUID),earliestStart:now,durationSeconds:3000,model,certain:true,expectedHealing:Math.min(...events.map(event=>model.healingByPatient[event.patientUUID]+receivedBonus(find(event.patientUUID),event.medicBonus))),options:{extensionOf,skill:events[0].options.skill??'medicine'}});
    }
   }
  }
  return result;
 }
 function completionEstimate(proposal){
  const healer=find(proposal.actorUUID),remaining=Math.max(0,...proposal.patientUUIDs.map(id=>deficit(find(id))));
  if(proposal.providerId==='rest')return 600;
  if(proposal.options.extensionOf)return 3000;
  if(!proposal.patientUUIDs.length){
   const focus=healer.computed.focusHealing;
   if(focus?.ready&&data.actors.some(actor=>deficit(actor)>0))return 600+Math.ceil(Math.max(...data.actors.map(deficit))/meanHealing(focus.healing))*(600+focus.durationSeconds)-600;
   return 600*Math.max(1,healer.focus.max-healer.focus.value);
  }
  if(proposal.expectedHealing<=0)return Infinity;
  const attempts=Math.min(1000,Math.max(1,Math.ceil(remaining/proposal.expectedHealing)));
  if(proposal.providerId==='focus-healing'){
   const refocus=Math.max(0,attempts-healer.focus.value)+(session.requireFullFocus?Math.min(attempts,healer.focus.value)+Math.max(0,healer.focus.max-healer.focus.value):0);
   return attempts*proposal.durationSeconds+refocus*600;
  }
  const wait=Math.max(0,proposal.earliestStart-data.cursorAt),healingTime=wait+600+(attempts-1)*proposal.model.cooldownSeconds;
  return Math.max(healingTime,session.requireFullFocus?600*Math.max(0,healer.focus.max-healer.focus.value-(proposal.options.threePecks?attempts:0)):0);
 }
 function order(left,right){
  if(strategy==='success-first'&&left.certain!==right.certain)return left.certain?-1:1;
  // Estimate time to the requested targets, including repeat immunity and
  // Focus recovery. This is a bounded local heuristic, not a global optimum.
  const time=completionEstimate(left)-completionEstimate(right);
  if(time)return time;
  return right.patientUUIDs.length-left.patientUUIDs.length||right.expectedHealing-left.expectedHealing||
   Number(left.model.riskySurgery===true)-Number(right.model.riskySurgery===true)||left.actorUUID.localeCompare(right.actorUUID)||left.patientUUIDs.join().localeCompare(right.patientUUIDs.join());
 }
 function next({until=Infinity}={}){
  if(data.pending)return clone(data.pending);
  if(data.stopReason)return {done:true,reason:data.stopReason};
  if(until<=data.cursorAt)throw Error('invalid-computed-checkpoint');
  if(data.actors.some(actor=>actor.needsReview))return done('needs-review');
  if(!data.active.length&&!data.actors.some(actor=>need(actor)||focusNeed(actor)))return done('goals-met');
  if(data.cursorAt>=session.budgetEndsAt)return done('budget');
  if(!data.active.length&&data.activityCount>=(session.maxActivities??10000))return done('activity-limit');
  const candidates=proposals(),busyActors=new Set(data.active.map(a=>a.actorUUID)),busyPools=new Set(data.active.flatMap(a=>a.hpPoolUUIDs)),busyPatients=new Set(data.active.filter(a=>a.providerId==='treat-wounds'||a.options.threePecks).flatMap(a=>a.patientUUIDs));
  const free=proposal=>!busyActors.has(proposal.actorUUID)&&!proposal.hpPoolUUIDs.some(pool=>busyPools.has(pool))&&!(proposal.providerId==='treat-wounds'||proposal.options.threePecks?proposal.patientUUIDs.some(id=>busyPatients.has(id)):false);
  const activities=[];
  for(const proposal of candidates.filter(p=>p.earliestStart<=data.cursorAt&&data.cursorAt+p.durationSeconds<=session.budgetEndsAt).sort(order)){
   if(data.activityCount>=(session.maxActivities??10000))break;
   if(!free(proposal))continue;
   const activity={...clone(proposal),id:`${session.id}:computed:${++data.activityCount}`,sessionId:session.id,startedAt:data.cursorAt,endsAt:data.cursorAt+proposal.durationSeconds};
   activities.push(activity);data.active.push(activity);busyActors.add(activity.actorUUID);activity.hpPoolUUIDs.forEach(pool=>busyPools.add(pool));
   if(activity.providerId==='treat-wounds'||activity.options.threePecks)activity.patientUUIDs.forEach(id=>busyPatients.add(id));
  }
  const checkpoints=[...data.active.map(a=>a.endsAt),...candidates.filter(p=>free(p)&&p.earliestStart>data.cursorAt&&p.earliestStart+p.durationSeconds<=session.budgetEndsAt).map(p=>p.earliestStart)].filter(time=>time>data.cursorAt);
  if(!checkpoints.length){
   if(candidates.some(p=>p.earliestStart+p.durationSeconds>session.budgetEndsAt))return done('budget');
   if(data.actors.some(actor=>need(actor)&&failureBlocked(actor)))return done('failure-stop');
   return done('unreachable');
  }
  data.pending={from:data.cursorAt,to:Math.min(...checkpoints,until,session.budgetEndsAt),activities:clone(activities)};
  return clone(data.pending);
 }
 function rollDie(faces){const sample=random();if(!Number.isFinite(sample)||sample<0||sample>=1)throw Error('invalid-computed-random');return Math.floor(sample*faces)+1}
 function rollDice(number,faces){return Array.from({length:number},()=>rollDie(faces))}
 function updateHP(patient,event,amount,kind){
  const before=patient.hp.value,after=kind==='damage'?Math.max(0,before-amount):Math.min(patient.hp.max,before+amount);
  event.changes.push({kind,amount,before,after});event.hpStages.push({before,after,damageTaken:kind==='damage'?amount:-amount});
  for(const member of data.actors.filter(actor=>poolOf(actor)===poolOf(patient))){member.hp.value=after;if(kind==='damage'&&after===0){member.needsReview=true;member.canAct=false}}
  if(after<patient.hp.max)for(const rest of data.active.filter(activity=>activity.providerId==='rest'&&poolOf(find(activity.actorUUID))===poolOf(patient)))rest.restInterrupted=true;
  if(kind==='damage'&&after===0){event.needsReview=true;event.reason='hp-zero-needs-review'}
 }
 function baseEvent(activity,patient){return {activityId:activity.id,providerId:activity.providerId,actorUUID:activity.actorUUID,startedAt:activity.startedAt,endsAt:activity.endsAt,options:clone(activity.options),...patient?{patientUUID:patient.actorUUID,poolUUID:poolOf(patient),hpBefore:patient.hp.value,hpAfter:patient.hp.value,tempBefore:patient.hp.temp??0,tempAfter:patient.hp.temp??0,woundedBefore:patient.wounded===true,woundedAfter:patient.wounded===true}: {},healing:0,damage:0,changes:[],hpStages:[]}}
 function settleTreatment(activity,healer,patient){
  const event=baseEvent(activity,patient),model=currentModel(healer,activity);
  if(patient.cooldownUnknown||patient.cooldownExpiresAt!=null&&patient.cooldownExpiresAt>activity.startedAt){event.cancelled=true;event.reason='computed-patient-immunity-changed';diagnose(event.reason);return event}
  if(!validTreatment(model)||!able(healer)||activity.patientUUIDs.length>model.capacity||model.cooldownSeconds!==(activity.options.continualRecovery?600:3600)||
   activity.options.threePecks&&(!healer.computed?.threePecks?.ready||!healer.computed?.refocus?.ready)||
   !patientReady(patient,model,activity.options.threePecks)||!treatmentPatientType(patient,healer,model.skill,activity.options.threePecks)){
   event.cancelled=true;event.reason='computed-treatment-context-changed';diagnose(event.reason);return event;
  }
  if(model.riskySurgery){event.riskyDice=rollDice(1,8);event.riskyDamage=event.riskyDice[0];event.damage+=event.riskyDamage;updateHP(patient,event,event.riskyDamage,'damage')}
  if(!event.needsReview){
   const face=model.assurance?10:rollDie(20),outcome=model.outcomes[model.assurance?0:face-1];
   event.check={face,total:face+model.modifier,dc:model.dc,assurance:model.assurance};event.outcome=outcome;
   if(outcome===0){event.damageDice=rollDice(1,8);event.checkDamage=event.damageDice[0];event.damage+=event.checkDamage;updateHP(patient,event,event.checkDamage,'damage')}
   else if(outcome>=2){
    event.healingDice=rollDice(outcome===3?4:2,8);event.baseHealing=event.healingDice.reduce((sum,n)=>sum+n,0)+model.bonus;event.medicBonus=model.medicBonus??0;event.healing=event.baseHealing+receivedBonus(patient,event.medicBonus);
    updateHP(patient,event,event.healing,'healing');patient.wounded=false;
   }
   data.failures[patient.actorUUID]=outcome<2?(data.failures[patient.actorUUID]??0)+1:0;
  }
  event.cooldownExpiresAt=activity.startedAt+model.cooldownSeconds;patient.cooldownExpiresAt=Math.max(patient.cooldownExpiresAt??0,event.cooldownExpiresAt);
  event.hpAfter=patient.hp.value;event.woundedAfter=patient.wounded===true;
  return event;
 }
 function settle({actors:fresh}={}){
  if(!data.pending)throw Error('computed-no-pending-boundary');
  const boundary=data.pending;
  if(fresh){
   refresh(fresh);
  }
  for(const activity of data.active)if(activity.providerId==='rest'&&find(activity.actorUUID).hp.value<find(activity.actorUUID).hp.max)activity.restInterrupted=true;
  const beforeActors=clone(data.actors),events=[];data.cursorAt=boundary.to;
  for(const activity of data.active.filter(a=>a.endsAt<=boundary.to).sort((a,b)=>a.endsAt-b.endsAt||a.id.localeCompare(b.id))){
   const healer=find(activity.actorUUID);
   if(!able(healer)){events.push({...baseEvent(activity),cancelled:true,reason:'actor-cannot-act'});continue}
   if(activity.providerId==='treat-wounds'||activity.options.threePecks){
    for(const id of activity.patientUUIDs){
     const patient=find(id);
     if(activity.options.extensionOf){
      const event=baseEvent(activity,patient);
      if(patientReady(patient)&&treatmentPatientType(patient,healer,activity.options.skill??'medicine')){event.baseHealing=activity.model.healingByPatient[id];event.medicBonus=activity.model.medicBonusByPatient?.[id]??0;event.healing=event.baseHealing+receivedBonus(patient,event.medicBonus);updateHP(patient,event,event.healing,'healing');patient.wounded=false;event.woundedAfter=false;event.hpAfter=patient.hp.value}
      else{event.cancelled=true;event.reason='computed-treatment-context-changed'}
      events.push(event);
     }else events.push(settleTreatment(activity,healer,patient));
    }
   }
   if(activity.providerId==='focus-healing'){
    const patient=find(activity.patientUUIDs[0]),model=healer.computed?.focusHealing,event=baseEvent(activity,patient);
    if(!model?.ready||model.itemUUID!==activity.options.itemUUID||healer.focus.value<1||!patientReady(patient,null,true)){event.cancelled=true;event.reason='focus-healing-context-changed'}
    else{
     event.focusBefore=healer.focus.value;healer.focus.value--;event.focusAfter=healer.focus.value;
     event.healingDice=model.healing.dice.flatMap(die=>rollDice(die.number,die.faces));event.healing=model.healing.constant+event.healingDice.reduce((sum,n)=>sum+n,0);updateHP(patient,event,event.healing,'healing');event.hpAfter=patient.hp.value;
     if(model.effects?.length)event.effects=model.effects.filter(effect=>!effect.otherOnly||patient.actorUUID!==healer.actorUUID).map(effect=>({sourceUUID:effect.sourceUUID,expiresAt:activity.endsAt+effect.durationSeconds}));
    }
    events.push(event);
   }
   if(activity.providerId==='refocus'&&able(healer)){
    const event=baseEvent(activity);
    if(healer.computed?.refocus?.ready&&(!activity.options.threePecks||healer.computed?.threePecks?.ready)){event.focusBefore=healer.focus.value;healer.focus.value=Math.min(healer.focus.max,healer.focus.value+1);event.focusAfter=healer.focus.value}
    else{event.cancelled=true;event.reason=healer.computed?.refocus?.reason??'refocus-recovery-unadapted'}
    events.push(event);
   }
   if(activity.providerId==='rest'){
    const event=baseEvent(activity,healer);if(!activity.restInterrupted&&healer.hp.value===healer.hp.max){healer.wounded=false;event.woundedAfter=false}else{event.cancelled=true;event.reason='rest-full-hp-changed'}events.push(event);
   }
  }
  data.active=data.active.filter(activity=>activity.endsAt>boundary.to);data.history.push(...events);data.pending=null;
  if(events.some(event=>event.needsReview))data.stopReason='needs-review';
  return {from:boundary.from,to:boundary.to,beforeActors,afterActors:clone(data.actors),events:clone(events)};
 }
 const snapshot=()=>jsonData(clone(data));
 function checkpoint(){
  const progressFields=['actorUUID','hp','focus','wounded','pool','cooldownExpiresAt','cooldownUnknown','needsReview'];
  const activityProgress=activity=>({id:activity.id,sessionId:activity.sessionId,providerId:activity.providerId,actorUUID:activity.actorUUID,patientUUIDs:activity.patientUUIDs,hpPoolUUIDs:activity.hpPoolUUIDs,startedAt:activity.startedAt,endsAt:activity.endsAt,options:activity.options,...activity.restInterrupted?{restInterrupted:true}:{},model:activity.options.extensionOf?activity.model:Object.fromEntries(['skill','rank','assurance','riskySurgery'].filter(key=>activity.model[key]!==undefined).map(key=>[key,activity.model[key]]))});
  const history=session.extendTreatment?data.history.filter(event=>event.providerId==='treat-wounds'&&!event.options.extensionOf&&event.endsAt===data.cursorAt&&event.outcome>=2)
   .map(event=>Object.fromEntries(['activityId','providerId','actorUUID','patientUUID','poolUUID','endsAt','outcome','healing','baseHealing','medicBonus','options'].map(key=>[key,event[key]]))):[];
  return jsonData(clone({version:data.version,sessionId:data.sessionId,cursorAt:data.cursorAt,actors:data.actors.map(actor=>Object.fromEntries(progressFields.filter(key=>actor[key]!==undefined).map(key=>[key,actor[key]]))),active:data.active.map(activityProgress),pending:data.pending?{from:data.pending.from,to:data.pending.to,activities:data.pending.activities.map(activityProgress)}:null,history,failures:data.failures,activityCount:data.activityCount,stopReason:data.stopReason,diagnostics:data.diagnostics}));
 }
 function run(){while(!next().done)settle();return snapshot()}
 return {next,settle,snapshot,checkpoint,run};
}
