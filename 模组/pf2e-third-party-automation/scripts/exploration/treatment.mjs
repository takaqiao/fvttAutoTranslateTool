import {cooldown,treatablePatient} from './capabilities.mjs';
export function extensionPatients(original,selected=original.patientUUIDs){return (original.results??[original]).filter(result=>['success','criticalSuccess'].includes(result.effectiveOutcome)&&selected.includes(result.patientUUID??original.patientUUIDs[0]))}
export function extension({startedAt,checkpointAt,outcome,rolledHealing}) {
  if(checkpointAt!==startedAt+600)throw Error('invalid-extension-checkpoint');
  if(!['success','criticalSuccess'].includes(outcome))return null;
  if(!Number.isFinite(rolledHealing)||rolledHealing<0)throw Error('missing-rolled-healing');
  return {endsAt:startedAt+3600,additionalHealing:rolledHealing};
}
export function completionState({checkId,expectedResults,persistedResultIds,applicationReceiptIds,expectedApplications}) {
  return checkId&&new Set(persistedResultIds).size===expectedResults&&new Set(applicationReceiptIds).size===expectedApplications?'confirmed':'awaiting-evidence';
}
export function createTreatmentProvider({capabilities,nativeTreatment,ledger,ownerOperations}) {
  const reserved=new Map();
  async function begin(activity) {
    const healer=await capabilities.discover(activity.actorUUID);
    if(activity.options.estimateSource&&(typeof healer.systemVersion!=='string'||!healer.systemVersion||activity.options.estimateSource.sourceVersion!==healer.systemVersion))return {status:'blocked',reason:'treatment-estimate-runtime-changed'};
    if(activity.options.extensionOf){const original=await ledger.getActivity(activity.options.extensionOf);if(original?.state!=='confirmed'||extensionPatients(original,activity.patientUUIDs).length!==activity.patientUUIDs.length)return {status:'blocked',reason:'extension-source-unconfirmed'};const patients=await capabilities.snapshot(activity.patientUUIDs);if(healer.isDead||healer.unconscious||patients.some(p=>p.isDead||!p.pool.ready||!treatablePatient(p,healer)))return {status:'blocked',reason:'patient-immune-or-token-or-pool-unavailable'};reserved.set(activity.id,{extension:original});return {status:'started'}}
    const skill=activity.options.skill??'medicine',rank=healer[skill]?.rank??0;
    if(rank<1||({trained:1,expert:2,master:3,legendary:4}[activity.options.rank??'trained']??5)>rank)return {status:'blocked',reason:'skill-or-dc-unqualified'};
    if(skill==='nature'&&!healer.slugs.includes('natural-medicine')||skill==='occultism')return {status:'blocked',reason:'substitute-skill-not-a-standalone-treatment'};
    if(activity.options.assurance&&!healer.assuranceSkills.includes(skill))return {status:'blocked',reason:'assurance-skill-unqualified'};
    if(activity.options.riskySurgery&&!healer.riskySurgery)return {status:'blocked',reason:'risky-surgery-unqualified'};
    if(activity.patientUUIDs.length>healer.wardCapacity)return {status:'blocked',reason:'ward-capacity'};
    const patients=await capabilities.snapshot(activity.patientUUIDs);
    if(healer.isDead||healer.unconscious||patients.some(p=>p.isDead||p.hasActiveToken===false||!p.pool.ready||p.cooldownExpiresAt!=null&&p.cooldownExpiresAt>activity.startedAt||!treatablePatient(p,healer)))return {status:'blocked',reason:'patient-immune-or-token-or-pool-unavailable'};
    reserved.set(activity.id,{healer,patients});return {status:'started'};
  }
  async function complete(activity,ctx) {
    if(!reserved.has(activity.id))return {status:'uncertain',reason:'activity-begin-context-lost'};
    const qualified=await begin(activity);if(qualified.status!=='started')return {status:'blocked',reason:qualified.reason};
    if(activity.options.extensionOf)return ownerOperations.runActivityWithOwner(activity,'treatment-extension');
    try{return await ownerOperations.runActivityWithOwner(activity,'treat-wounds')}finally{reserved.delete(activity.id)}
  }
  return {id:'treat-wounds',describe:capabilities.discover,begin,complete,
    cancel:async activity=>{reserved.delete(activity.id)},propose:async()=>[],observe:async()=>null,reconcile:nativeTreatment.reconcile,
    immunityDeadline:(activity)=>cooldown({startedAt:activity.startedAt,finishedAt:activity.endsAt,continualRecovery:activity.options.continualRecovery}).expiresAt};
}
