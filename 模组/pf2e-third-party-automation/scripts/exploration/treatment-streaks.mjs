const outcomeDegrees=new Map([['criticalFailure',0],['failure',1],['success',2],['criticalSuccess',3]]);
const proofLists=['checkIds','resultIds','receiptIds','immunityIds'];

function dataRecord(input){
  return !!input&&typeof input==='object'&&[Object.prototype,null].includes(Object.getPrototypeOf(input))&&Reflect.ownKeys(input).every(key=>{
    const field=Object.getOwnPropertyDescriptor(input,key);
    return typeof key==='string'&&field.enumerable&&Object.hasOwn(field,'value');
  });
}
const field=(input,key)=>Object.getOwnPropertyDescriptor(input??{},key)?.value;
function dataArray(input){
  if(!Array.isArray(input)||Object.getPrototypeOf(input)!==Array.prototype||Reflect.ownKeys(input).length!==input.length+1)return null;
  const entries=[];
  for(let index=0;index<input.length;index++){
    const item=Object.getOwnPropertyDescriptor(input,String(index));if(!item?.enumerable||!Object.hasOwn(item,'value'))return null;
    entries.push(item.value);
  }
  return entries;
}
const validId=value=>typeof value==='string'&&value.length>0&&value.length<=128&&value.trim()===value&&!/[\u0000-\u001f\u007f]/.test(value)&&!['__proto__','constructor','prototype'].includes(value);
const validPatient=value=>typeof value==='string'&&value.length<=128&&/^(?:Actor\.[A-Za-z0-9_-]+|Scene\.[A-Za-z0-9_-]+\.Token\.[A-Za-z0-9_-]+\.Actor\.[A-Za-z0-9_-]+)$/.test(value);
function ids(input){
  const values=dataArray(input);return values&&values.every(validId)&&new Set(values).size===values.length?values:null;
}
function normalizeFailureStop(input){
  if(input===undefined)return {enabled:false,limit:3};
  const label='invalid-treatment-failure-stop';
  if(!dataRecord(input)||Object.keys(input).some(key=>!['enabled','limit'].includes(key)))throw Error(label);
  const enabled=Object.hasOwn(input,'enabled')?field(input,'enabled'):false,limit=Object.hasOwn(input,'limit')?field(input,'limit'):3;
  if(typeof enabled!=='boolean'||!Number.isSafeInteger(limit)||limit<1||limit>100)throw Error(label);
  return {enabled,limit};
}
function sessionOrder(input){
  const label='invalid-treatment-failure-context';
  if(!dataRecord(input)||Object.keys(input).some(key=>!['sessionId','activityIds'].includes(key))||!validId(field(input,'sessionId')))throw Error(label);
  const activityIds=ids(field(input,'activityIds'));if(!activityIds)throw Error(label);
  return {sessionId:field(input,'sessionId'),ordinal:new Map(activityIds.map((id,index)=>[id,index]))};
}
function nativeProof(input,useId){
  if(!dataRecord(input)||field(input,'useId')!==useId)return null;
  const proof={useId};
  for(const key of proofLists){proof[key]=ids(field(input,key));if(!proof[key])return null}
  return proof;
}
const sameIds=(left,right)=>left.length===right.length&&new Set(left).size===left.length&&new Set(right).size===right.length&&left.every(id=>right.includes(id));
const sameProof=(left,right)=>!!left&&!!right&&left.useId===right.useId&&proofLists.every(key=>sameIds(left[key],right[key]));

function nativeFacts(activity,ordinal){
  if(!dataRecord(activity)||field(activity,'state')!=='confirmed'||field(activity,'providerId')!=='treat-wounds')return [];
  const options=field(activity,'options')??{};
  if(!dataRecord(options)||Object.hasOwn(options,'extensionOf'))return [];
  const id=field(activity,'id'),actorUUID=field(activity,'actorUUID'),executor=field(activity,'executor'),result=field(activity,'executionResult');
  if(!validId(id)||!validPatient(actorUUID)||!dataRecord(executor)||field(executor,'state')!=='settled'||field(executor,'protocol')!=='pf2e-third-party-automation.exploration-owner.v1'||field(executor,'operationId')!=='treat-wounds'||field(executor,'activityId')!==id||field(executor,'actorUUID')!==actorUUID||field(executor,'sessionId')!==field(activity,'sessionId')||!dataRecord(result)||field(result,'status')!=='confirmed')return [];
  const startedAt=field(activity,'startedAt'),patients=ids(field(activity,'patientUUIDs')),proof=nativeProof(field(result,'proof'),id);
  if(!Number.isFinite(startedAt)||!patients?.length||!patients.every(validPatient)||!proof||proof.checkIds.length!==patients.length||!sameProof(proof,nativeProof(field(activity,'proof'),id)))return [];
  if(!ordinal.has(id))throw Error('invalid-treatment-failure-context');
  const summaries=Object.hasOwn(result,'results')?dataArray(field(result,'results')):patients.length===1?[result]:null;
  if(!summaries||summaries.length!==patients.length)return [];
  const seenPatients=new Set(),merged=Object.fromEntries(proofLists.map(key=>[key,[]])),facts=[];
  for(const summary of summaries){
    if(!dataRecord(summary)||field(summary,'status')!=='confirmed')return [];
    const patientUUID=field(summary,'patientUUID'),outcome=field(summary,'effectiveOutcome'),patientProof=nativeProof(field(summary,'proof'),id);
    if(!patients.includes(patientUUID)||seenPatients.has(patientUUID)||!outcomeDegrees.has(outcome)||!patientProof||patientProof.checkIds.length!==1)return [];
    seenPatients.add(patientUUID);for(const key of proofLists)merged[key].push(...patientProof[key]);
    facts.push({patientUUID,useId:id,checkId:patientProof.checkIds[0],outcome,startedAt,ordinal:ordinal.get(id),actorUUID});
  }
  if(proofLists.some(key=>!sameIds(merged[key],proof[key])))return [];
  return facts;
}

/** Enabled derivation requires the session's saved activityIds, never completion arrival order. */
export function treatmentFailureState(activities,failureStop=undefined,context=undefined){
  const stop=normalizeFailureStop(failureStop),streakByPatient={};
  if(!stop.enabled)return {streakByPatient,blockedPatientUUIDs:[]};
  const {sessionId,ordinal}=sessionOrder(context),rows=dataArray(activities);
  if(!rows)throw Error('invalid-treatment-failure-context');
  const unique=new Map(),patientByCheck=new Map();
  for(const activity of rows){
    if(field(activity,'sessionId')!==sessionId)throw Error('mixed-treatment-sessions');
    for(const fact of nativeFacts(activity,ordinal)){
      const checkKey=JSON.stringify([sessionId,fact.useId,fact.checkId]);
      if(patientByCheck.has(checkKey)&&patientByCheck.get(checkKey)!==fact.patientUUID)throw Error('conflicting-treatment-check');
      patientByCheck.set(checkKey,fact.patientUUID);
      const key=JSON.stringify([sessionId,fact.useId,fact.checkId,fact.patientUUID]),old=unique.get(key);
      if(old&&(old.outcome!==fact.outcome||old.startedAt!==fact.startedAt||old.ordinal!==fact.ordinal||old.actorUUID!==fact.actorUUID))throw Error('conflicting-treatment-check');
      unique.set(key,fact);
    }
  }
  const ordered=[...unique.values()].sort((left,right)=>left.startedAt-right.startedAt||left.ordinal-right.ordinal||left.patientUUID.localeCompare(right.patientUUID));
  for(const fact of ordered)streakByPatient[fact.patientUUID]=outcomeDegrees.get(fact.outcome)<2?(streakByPatient[fact.patientUUID]??0)+1:0;
  return {streakByPatient,blockedPatientUUIDs:Object.keys(streakByPatient).filter(patientUUID=>streakByPatient[patientUUID]>=stop.limit).sort()};
}
