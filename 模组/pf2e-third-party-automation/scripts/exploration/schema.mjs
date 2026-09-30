export const MODULE_ID='pf2e-third-party-automation';
export const emptyLedger=()=>({sessions:{},activities:{},clocks:{}});
export const clone=value=>structuredClone(value);
export function finite(value,label) {if(!Number.isFinite(value))throw Error(`invalid-${label}`);return value}
export function id(value,label='id') {if(typeof value!=='string'||!value.trim()||['__proto__','constructor','prototype'].includes(value))throw Error(`invalid-${label}`);return value}
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
  const s=clone(input);id(s.id);finite(s.startedAt,'start');finite(s.budgetEndsAt,'budget');
  if(s.budgetEndsAt<s.startedAt)throw Error('negative-duration');
  s.activityIds??=[];s.goalsByPool??=[];s.assumptions??=[];s.status??='running';s.stopReason??=null;
  return s;
}
export function validateClock(input) {
  const c=clone(input);id(c.id);id(c.sessionId);id(c.gmId);finite(c.from,'from');finite(c.to,'to');
  if(c.to<c.from)throw Error('negative-duration');if(!['started','confirmed','uncertain'].includes(c.state))throw Error('invalid-state');c.evidence??=[];return c;
}
