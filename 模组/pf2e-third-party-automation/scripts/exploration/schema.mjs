import {normalizeRecoveryPreferences,validateRecoveryGoals} from './recovery-goals.mjs';
export const MODULE_ID='pf2e-third-party-automation';
export const emptyLedger=()=>({sessions:{},activities:{},clocks:{}});
export const clone=value=>structuredClone(value);
export function captureRecoveryPreferences(config){
  const field=Object.getOwnPropertyDescriptor(config,'recovery');
  if(!field){if('recovery' in config)throw Error('invalid-recovery-preferences');return undefined}
  if(!field.enumerable||!Object.hasOwn(field,'value'))throw Error('invalid-recovery-preferences');
  const preferences=normalizeRecoveryPreferences(field.value);
  if(config.actorUUIDs!==undefined&&(!Array.isArray(config.actorUUIDs)||Object.keys(preferences.targetIntentsByActor).some(uuid=>!config.actorUUIDs.includes(uuid))))throw Error('invalid-recovery-preferences');
  return preferences;
}
export function finite(value,label) {if(!Number.isFinite(value))throw Error(`invalid-${label}`);return value}
export function id(value,label='id') {if(typeof value!=='string'||!value.trim()||['__proto__','constructor','prototype'].includes(value))throw Error(`invalid-${label}`);return value}
export const MANUAL_POOL_OPERATION='manual-pool-application';
export function manualPoolRequest(input){
  const fields=['sessionId','activityId','actorUUID','sourceType','useId','checkId','resultId','rollIndex','stage','poolUUID','patientUUIDs','batchId','ownerClientNonce','attemptNonce'];
  if(!input||typeof input!=='object'||![Object.prototype,null].includes(Object.getPrototypeOf(input))||Reflect.ownKeys(input).length!==fields.length)throw Error('invalid-manual-pool-request');
  const result={};
  for(const key of fields){const d=Object.getOwnPropertyDescriptor(input,key);if(!d?.enumerable||!Object.hasOwn(d,'value'))throw Error('invalid-manual-pool-request');result[key]=d.value}
  for(const key of fields.filter(k=>!['patientUUIDs','rollIndex'].includes(k))){id(result[key],`manual-pool-${key}`);if(result[key].length>128||result[key].trim()!==result[key]||/[\u0000-\u001f\u007f]/.test(result[key]))throw Error('invalid-manual-pool-request')}
  if(!['native-action','workbench'].includes(result.sourceType)||result.stage!=='healing'||!Number.isSafeInteger(result.rollIndex)||result.rollIndex<0||result.rollIndex>31)throw Error('invalid-manual-pool-source');
  const patients=result.patientUUIDs;
  if(!Array.isArray(patients)||Object.getPrototypeOf(patients)!==Array.prototype||patients.length<1||patients.length>8||Reflect.ownKeys(patients).length!==patients.length+1)throw Error('invalid-manual-pool-patients');
  result.patientUUIDs=Array.from({length:patients.length},(_,i)=>{const d=Object.getOwnPropertyDescriptor(patients,String(i));if(!d||!Object.hasOwn(d,'value'))throw Error('invalid-manual-pool-patients');return id(d.value,'manual-pool-patient')});
  if(new Set(result.patientUUIDs).size!==patients.length)throw Error('invalid-manual-pool-patients');return result;
}
const checkpointFields=['id','sessionId','rootUUID','epoch','observationNonce','from','to'];
const activityCheckpointFields=['id','sessionId','rootUUID','epoch','observationNonce','from'];
export function activityCheckpointBinding(input){
  if(!input||typeof input!=='object'||![Object.prototype,null].includes(Object.getPrototypeOf(input))||Reflect.ownKeys(input).length!==activityCheckpointFields.length)throw Error('invalid-activity-checkpoint');
  const value={};
  for(const key of activityCheckpointFields){const field=Object.getOwnPropertyDescriptor(input,key);if(!field?.enumerable||!Object.hasOwn(field,'value'))throw Error('invalid-activity-checkpoint');value[key]=key==='from'?finite(field.value,'checkpoint-from'):id(field.value,key)}
  return value;
}
export function sameActivityCheckpoint(expected,binding){
  const value=activityCheckpointBinding(binding);return !!expected&&activityCheckpointFields.every(key=>value[key]===expected[key]);
}
export const checkpointBinding=checkpoint=>Object.fromEntries(checkpointFields.map(key=>[key,checkpoint[key]]));
export function sameCheckpoint(expected,binding){
  return !!expected&&!!binding&&typeof binding==='object'&&[Object.prototype,null].includes(Object.getPrototypeOf(binding))&&Reflect.ownKeys(binding).length===checkpointFields.length&&checkpointFields.every(key=>{
    const field=Object.getOwnPropertyDescriptor(binding,key);return field&&Object.hasOwn(field,'value')&&field.value===expected[key];
  });
}
export function manualSourceIntent(input){
  const fields=['sourceType','kind','useId','actorUUID','patientUUID','riskySurgery'];
  if(!input||typeof input!=='object'||![Object.prototype,null].includes(Object.getPrototypeOf(input))||Reflect.ownKeys(input).some(key=>!fields.includes(key)))throw Error('invalid-manual-source-intent');
  const value={};for(const key of Reflect.ownKeys(input)){const field=Object.getOwnPropertyDescriptor(input,key);if(!Object.hasOwn(field,'value'))throw Error('invalid-manual-source-intent');value[key]=field.value}
  if(value.sourceType!=='workbench'||value.kind!=='treatment'||value.riskySurgery!==undefined&&typeof value.riskySurgery!=='boolean')throw Error('manual-checkpoint-source-unavailable');
  for(const key of ['useId','actorUUID','patientUUID'])id(value[key],key);return value;
}
export function normalizeNativeOwnerMap(value,actorUUIDs,{manual=false}={}) {
  if(value===undefined)return {};
  if(value===null||typeof value!=='object'||![Object.prototype,null].includes(Object.getPrototypeOf(value)))throw Error('invalid-native-owner-map');
  const selected=new Set(actorUUIDs??[]),result={};
  for(const key of Reflect.ownKeys(value)){
    id(key,'native-owner-actor');const descriptor=Object.getOwnPropertyDescriptor(value,key);
    if(!descriptor.enumerable||!Object.hasOwn(descriptor,'value')||!selected.has(key))throw Error('invalid-native-owner-map');
    result[key]=id(descriptor.value,'native-owner-user');
  }
  if(manual&&Object.keys(result).length)throw Error('manual-native-owner-map');
  return result;
}
export const activityStates=['planned','started','completing','awaiting-evidence','confirmed','blocked','uncertain','cancelled'];
export function createActivity(input) {
  const a=clone(input);
  for(const key of ['id','sessionId','providerId','actorUUID'])id(a[key],key);
  finite(a.startedAt,'start');finite(a.endsAt,'end');if(a.endsAt<a.startedAt)throw Error('negative-duration');
  if(!activityStates.includes(a.state))throw Error('invalid-state');
  for(const key of ['patientUUIDs','hpPoolUUIDs']){if(!Array.isArray(a[key]))throw Error(`invalid-${key}`);a[key]=[...new Set(a[key].map(v=>id(v,key)))];}
  a.groupId??=a.id;a.options??={};a.source??={};
  a.proof={useId:null,checkIds:[],resultIds:[],receiptIds:[],immunityIds:[],...a.proof};
  for(const key of ['checkIds','resultIds','receiptIds','immunityIds'])if(!Array.isArray(a.proof[key]))throw Error('invalid-proof');
  return a;
}
export function validateSession(input) {
  const nativeOwnerByActor=normalizeNativeOwnerMap(input.nativeOwnerByActor,input.actorUUIDs,{manual:input.manual===true});
  let recoveryGoals;
  if('recoveryGoals' in input){const field=Object.getOwnPropertyDescriptor(input,'recoveryGoals');if(!field?.enumerable||!Object.hasOwn(field,'value'))throw Error('invalid-recovery-goals');recoveryGoals=validateRecoveryGoals(field.value,input.actorUUIDs,input.goalsByPool)}
  const s={...clone(input),nativeOwnerByActor,...recoveryGoals?{recoveryGoals}:{}};id(s.id);finite(s.startedAt,'start');finite(s.budgetEndsAt,'budget');
  if(s.budgetEndsAt<s.startedAt)throw Error('negative-duration');
  s.activityIds??=[];s.goalsByPool??=[];s.assumptions??=[];s.status??='running';s.stopReason??=null;
  return s;
}
export function validateClock(input) {
  const c=clone(input);id(c.id);id(c.sessionId);id(c.gmId);finite(c.from,'from');finite(c.to,'to');
  if(c.to<c.from)throw Error('negative-duration');if(!['started','confirmed','uncertain'].includes(c.state))throw Error('invalid-state');c.evidence??=[];return c;
}
