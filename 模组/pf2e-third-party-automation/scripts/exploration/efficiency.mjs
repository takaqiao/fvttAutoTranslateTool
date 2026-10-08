const ranks=['trained','expert','master','legendary'];
const dcs=[15,20,30,40],bonuses=[0,10,30,50];
const distributions=new Map([[0,new Map([[0,1]])]]);
function diceDistribution(dice){
 if(!distributions.has(dice)){
  const next=new Map();
  for(const [sum,p]of diceDistribution(dice-1))for(let face=1;face<=8;face++)next.set(sum+face,(next.get(sum+face)??0)+p/8);
  distributions.set(dice,next);
 }
 return distributions.get(dice);
}
function uniquePatients(patients){
 const pools=new Map();
 for(const patient of patients){const key=patient.pool?.poolUUID??patient.actorUUID??patient;if(!pools.has(key))pools.set(key,patient)}
 return [...pools.values()];
}
function placeholder(patients,deficit){return patients.reduce((sum,p)=>{const missing=deficit(p);return sum+(Number.isFinite(missing)?Math.min(9,Math.max(0,missing)):0)},0)}
function fallback(rank,patients,deficit,reason,fixed){return {rank,estimate:fixed?'fixed-selected-dc':'fixed-dc-unverified-context',estimateReason:reason,expectedNetHealing:placeholder(patients,deficit),expectedHPDamage:null}}
function legacyCases(modifier,dc){
 return Array.from({length:20},(_,index)=>{
  const face=index+1,total=face+modifier;
  let outcome=total>=dc+10?3:total>=dc?2:total<=dc-10?0:1;
  outcome=Math.max(0,Math.min(3,outcome+(face===20?1:face===1?-1:0)));
  return {weight:0.05,outcome};
 });
}
function preparedSelection(context,healer,skill,skillRank,options){
 if(!Array.isArray(context.selections))return {reason:'malformed-prepared-selections'};
 const matches=context.selections.filter(s=>s?.riskySurgery===(options.riskySurgery===true)&&s?.assurance===(options.assurance===true));
 if(matches.length!==1)return {reason:'prepared-selection-unavailable'};
 const selection=matches[0];
 if(selection.ready!==true)return {reason:typeof selection.reason==='string'&&selection.reason.trim()?selection.reason:'prepared-selection-unavailable'};
 if(typeof healer.systemVersion!=='string'||!healer.systemVersion||selection.sourceVersion!==healer.systemVersion||selection.source?.actorUUID!==healer.actorUUID||typeof healer.actorUUID!=='string'||selection.source?.skill!==skill||!Array.isArray(selection.source.ruleSources)||Array.from(selection.source.ruleSources).some(id=>typeof id!=='string'||!id.trim())||new Set(selection.source.ruleSources).size!==selection.source.ruleSources.length)return {reason:'unverified-prepared-source'};
 if(options.assurance&&!healer.assuranceSkills?.includes(skill))return {reason:'skill-assurance-unavailable'};
 const rows=selection.outcomesByRank,count=options.assurance?1:20,weight=1/count;
 if(!Array.isArray(rows)||rows.length!==skillRank)return {reason:'malformed-prepared-outcomes'};
 const ordered=[];
 for(let i=0;i<skillRank;i++){
  const matching=rows.filter(row=>row?.rank===ranks[i]);
  if(matching.length!==1)return {reason:'malformed-prepared-outcomes'};
  const row=matching[0];
  if(row.dc!==dcs[i]||!Array.isArray(row.cases)||row.cases.length!==count||Array.from(row.cases).some(c=>!c||c.weight!==weight||!Number.isInteger(c.outcome)||c.outcome<0||c.outcome>3))return {reason:'malformed-prepared-outcomes'};
  ordered.push(row);
 }
 return {selection,rows:ordered};
}
function patientModelReason(healer,patients,rows,options,deficit){
 if(patients.some(p=>p.healingExpectationReady!==true))return 'healing-model-unverified';
 if(patients.some(p=>!Number.isFinite(p.hp?.value)||!Number.isFinite(p.hp?.max)||p.hp.value<0||p.hp.value>p.hp.max||!Number.isFinite(deficit(p))||deficit(p)<0))return 'patient-hp-model-unverified';
 const byPool=new Map();
 for(const patient of patients){
  const key=patient.pool?.poolUUID??patient.actorUUID??patient,previous=byPool.get(key);
  if(previous&&(['value','max','temp'].some(field=>previous.hp[field]!==patient.hp[field])||deficit(previous)!==deficit(patient)))return 'inconsistent-patient-pool';
  byPool.set(key,patient);
 }
 if(patients.some(p=>p.actorUUID===healer.actorUUID||p.pool?.poolUUID&&p.pool.poolUUID===healer.pool?.poolUUID))return 'self-treatment-context-unverified';
 const risky=options.riskySurgery===true,damage=risky||rows.some(row=>row.cases.some(c=>c.outcome===0));
 if(damage&&patients.some(p=>p.damageExpectationReady!==true||p.hp.temp!==0))return 'damage-model-unverified';
 if(risky){
  if(options.skill!=='medicine'||healer.riskySurgery!==true||!healer.slugs?.includes('risky-surgery'))return 'risky-surgery-source-unverified';
  if(!healer.pool?.ready||typeof healer.pool.poolUUID!=='string'||patients.some(p=>!p.pool?.ready||typeof p.pool.poolUUID!=='string'||p.pool.poolUUID===healer.pool.poolUUID||p.actorUUID===healer.actorUUID))return 'action-capacity-after-damage-unverified';
  if(patients.some(p=>p.hp.value<17))return 'action-capacity-after-damage-unverified';
 }
 return null;
}
function expectedResult(patient,cases,bonus,risky,deficit){
 const initial=patient.hp.value,target=Math.min(patient.hp.max,initial+deficit(patient));
 let net=0,damage=0;
 for(const c of cases)for(const [cut,cutWeight]of diceDistribution(risky?1:0)){
  const afterCut=Math.max(0,initial-cut),cutDamage=initial-afterCut;
  const dice=c.outcome===0?1:c.outcome===2?2:c.outcome===3?4:0;
  for(const [rolled,rollWeight]of diceDistribution(dice)){
   const after=c.outcome===0?Math.max(0,afterCut-rolled):c.outcome>=2?Math.min(patient.hp.max,afterCut+rolled+bonus):afterCut;
   const weight=c.weight*cutWeight*rollWeight;
   net+=(Math.min(target,after)-Math.min(target,initial))*weight;
   damage+=(cutDamage+(c.outcome===0?afterCut-after:0))*weight;
  }
 }
 return {net,damage};
}
/** Consume prepared outcomes only: no native roll, rule mutation, or future sampling. */
export function selectTreatmentRank(healer,patients,options,deficit){
 const skill=options.skill??'medicine',skillRank=healer[skill]?.rank??0,requested=options.treatmentRank??'trained',fixed=requested!=='auto';
 if(!Number.isInteger(skillRank)||skillRank<1||skillRank>4||fixed&&(ranks.indexOf(requested)<0||ranks.indexOf(requested)>=skillRank))return null;
 options={...options,skill};
 const distinct=uniquePatients(patients),rank=fixed?requested:'trained',context=healer.treatmentEstimate?.[skill];
 const fail=reason=>fallback(rank,distinct,deficit,reason,fixed);
 let rows,selection,legacy=false;
 if(context&&Object.hasOwn(context,'selections')){
  const prepared=preparedSelection(context,healer,skill,skillRank,options);
  if(prepared.reason)return fail(prepared.reason);
  ({rows,selection}=prepared);
  const reason=patientModelReason(healer,patients,rows,options,deficit);if(reason)return fail(reason);
 }else{
  if(!context?.ready||!Number.isFinite(context.modifier)||options.riskySurgery||options.assurance||patients.some(p=>!p.healingExpectationReady))return fail(context?.reason??'unverified-native-context');
  legacy=true;rows=ranks.slice(0,skillRank).map((rank,i)=>({rank,dc:dcs[i],cases:legacyCases(context.modifier,dcs[i])}));
 }
 let best;
 for(let i=0;i<skillRank;i++){
  if(fixed&&ranks[i]!==requested)continue;
  const bonus=bonuses[i]+(healer.slugs?.includes('medic-dedication')?[0,5,10,15][i]:0);
  let net=0,damage=0;
  for(const patient of distinct){
   if(legacy){
    // Retain the pre-prepared unconditional estimate contract for old snapshots.
    for(const c of rows[i].cases){
     if(c.outcome===0){net-=4.5*c.weight;damage+=4.5*c.weight}
     else if(c.outcome>=2)for(const [healing,p]of diceDistribution(c.outcome===3?4:2))net+=Math.min(deficit(patient),healing+bonus)*p*c.weight;
    }
   }else{const result=expectedResult(patient,rows[i].cases,bonus,options.riskySurgery===true,deficit);net+=result.net;damage+=result.damage}
  }
  if(!best||net>best.expectedNetHealing+1e-8)best={rank:ranks[i],estimate:fixed?'fixed-selected-dc':legacy?'verified-unconditional-native-context':'verified-prepared-native-context',expectedNetHealing:net,expectedHPDamage:damage,...selection?{estimateSource:{...structuredClone(selection.source),sourceVersion:selection.sourceVersion,riskySurgery:selection.riskySurgery,assurance:selection.assurance}}:{}};
 }
 return best;
}
