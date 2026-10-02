const preferenceFields=['version','targetIntentsByActor','requireNoWounded','failureStop'];
const goalFields=['version','patientTargets','requireNoWounded','failureStop'];
const patientFields=['patientUUID','poolUUID','intent','basisMaxHP','targetHP'];
const maximumStoredRecoveryLength=65536;

function plainObject(input,label){
  if(!input||typeof input!=='object'||![Object.prototype,null].includes(Object.getPrototypeOf(input)))throw Error(label);
  return input;
}
function fields(input,allowed,label,required=[]){
  plainObject(input,label);const result={};
  for(const key of Reflect.ownKeys(input)){
    const field=Object.getOwnPropertyDescriptor(input,key);
    if(!allowed.includes(key)||!field.enumerable||!Object.hasOwn(field,'value'))throw Error(label);
    result[key]=field.value;
  }
  if(required.some(key=>!Object.hasOwn(result,key)))throw Error(label);
  return result;
}
function dataArray(input,label){
  if(!Array.isArray(input)||Object.getPrototypeOf(input)!==Array.prototype||Reflect.ownKeys(input).length!==input.length+1)throw Error(label);
  return Array.from({length:input.length},(_,index)=>{
    const field=Object.getOwnPropertyDescriptor(input,String(index));
    if(!field?.enumerable||!Object.hasOwn(field,'value'))throw Error(label);
    return field.value;
  });
}
function uuid(input,label){
  if(typeof input!=='string'||input.length>128||!/^(?:Actor\.[A-Za-z0-9_-]+|Scene\.[A-Za-z0-9_-]+\.Token\.[A-Za-z0-9_-]+\.Actor\.[A-Za-z0-9_-]+)$/.test(input))throw Error(label);
  return input;
}
function hpValue(input,label){
  if(!Number.isSafeInteger(input)||input<0)throw Error(label);
  return input;
}
function targetIntent(input,label){
  const intent=fields(input,['mode','value'],label,['mode','value']);
  if(intent.mode==='max'){if(intent.value!==null)throw Error(label)}
  else if(intent.mode==='percent'){if(!Number.isFinite(intent.value)||intent.value<0||intent.value>100)throw Error(label)}
  else if(intent.mode==='absolute')hpValue(intent.value,label);
  else throw Error(label);
  return intent;
}
function completionFlags(input,label,complete=false){
  const requireNoWounded=Object.hasOwn(input,'requireNoWounded')?input.requireNoWounded:false;
  if(typeof requireNoWounded!=='boolean')throw Error(label);
  const stop=Object.hasOwn(input,'failureStop')?fields(input.failureStop,['enabled','limit'],label,complete?['enabled','limit']:[]):{};
  const enabled=Object.hasOwn(stop,'enabled')?stop.enabled:false,limit=Object.hasOwn(stop,'limit')?stop.limit:3;
  if(typeof enabled!=='boolean'||!Number.isSafeInteger(limit)||limit<1||limit>100)throw Error(label);
  return {requireNoWounded,failureStop:{enabled,limit}};
}
function resolvedTarget(intent,max){
  if(intent.mode==='max')return max;
  if(intent.mode==='absolute')return Math.min(intent.value,max);
  // Floating multiplication can lose an HP at whole targets near the safe-integer limit.
  const [coefficient,exponent='0']=String(intent.value).split('e'),[integer,fraction='']=coefficient.split('.');
  const numerator=BigInt(max)*BigInt(integer+fraction),divisor=10n**BigInt(fraction.length-Number(exponent)+2);
  return Number((numerator+divisor-1n)/divisor);
}
function poolGoals(patientTargets){
  const goals=new Map();
  for(const target of patientTargets)goals.set(target.poolUUID,Math.max(goals.get(target.poolUUID)??0,target.targetHP));
  return [...goals].map(([poolUUID,targetHP])=>({poolUUID,targetHP}));
}

export function normalizeRecoveryPreferences(input=undefined){
  const label='invalid-recovery-preferences',value=fields(input===undefined?{}:input,preferenceFields,label);
  if(Object.hasOwn(value,'version')&&value.version!==1)throw Error(label);
  const source=Object.hasOwn(value,'targetIntentsByActor')?plainObject(value.targetIntentsByActor,label):{},targetIntentsByActor={};
  for(const key of Reflect.ownKeys(source)){
    uuid(key,label);const field=Object.getOwnPropertyDescriptor(source,key);
    if(!field.enumerable||!Object.hasOwn(field,'value'))throw Error(label);
    targetIntentsByActor[key]=targetIntent(field.value,label);
  }
  return {version:1,targetIntentsByActor,...completionFlags(value,label)};
}

// Foundry expands dotted UUID keys inside flag objects; a JSON scalar preserves them.
export function encodeRecoveryPreferences(input){
  const encoded=JSON.stringify(normalizeRecoveryPreferences(input));
  if(encoded.length>maximumStoredRecoveryLength)throw Error('invalid-recovery-preferences');
  return encoded;
}
export function decodeRecoveryPreferences(input=undefined){
  if(typeof input==='string'){
    if(input.length>maximumStoredRecoveryLength)throw Error('invalid-recovery-preferences');
    try{input=JSON.parse(input)}catch{throw Error('invalid-recovery-preferences')}
  }
  return normalizeRecoveryPreferences(input);
}

function snapshotField(input,key,label){
  plainObject(input,label);const field=Object.getOwnPropertyDescriptor(input,key);
  if(!field?.enumerable||!Object.hasOwn(field,'value'))throw Error(label);
  return field.value;
}
export function resolveRecoveryGoals(actors,preferences=undefined){
  const label='invalid-recovery-actors',patients=new Set(),maxByPool=new Map(),normalized=normalizeRecoveryPreferences(preferences);
  const patientTargets=dataArray(actors,label).map(actor=>{
    const patientUUID=uuid(snapshotField(actor,'actorUUID',label),label),pool=snapshotField(actor,'pool',label),hp=snapshotField(actor,'hp',label);
    const poolUUID=uuid(snapshotField(pool,'poolUUID',label),label),basisMaxHP=hpValue(snapshotField(hp,'max',label),label);
    if(patients.has(patientUUID)||snapshotField(pool,'ready',label)!==true||maxByPool.has(poolUUID)&&maxByPool.get(poolUUID)!==basisMaxHP)throw Error(label);
    patients.add(patientUUID);maxByPool.set(poolUUID,basisMaxHP);
    const intent=normalized.targetIntentsByActor[patientUUID]??{mode:'max',value:null};
    return {patientUUID,poolUUID,intent,basisMaxHP,targetHP:resolvedTarget(intent,basisMaxHP)};
  });
  if(Object.keys(normalized.targetIntentsByActor).some(patientUUID=>!patients.has(patientUUID)))throw Error('invalid-recovery-preferences');
  return {goalsByPool:poolGoals(patientTargets),recoveryGoals:{version:1,patientTargets,requireNoWounded:normalized.requireNoWounded,failureStop:normalized.failureStop}};
}

export function patientRecoveryNeed(patient,session){
  return {hp:session.goalsByPool.some(goal=>goal.poolUUID===patient.pool?.poolUUID&&patient.hp.value<goal.targetHP),wounded:session.recoveryGoals?.requireNoWounded===true&&patient.wounded===true};
}

export function validateRecoveryGoals(input,actorUUIDs,goalsByPool){
  const label='invalid-recovery-goals',value=fields(input,goalFields,label,goalFields);
  if(value.version!==1)throw Error(label);
  const selected=dataArray(actorUUIDs,label).map(actorUUID=>uuid(actorUUID,label)),patients=new Set(selected),seen=new Set(),maxByPool=new Map();
  if(patients.size!==selected.length)throw Error(label);
  const patientTargets=dataArray(value.patientTargets,label).map(input=>{
    const target=fields(input,patientFields,label,patientFields),patientUUID=uuid(target.patientUUID,label),poolUUID=uuid(target.poolUUID,label);
    const intent=targetIntent(target.intent,label),basisMaxHP=hpValue(target.basisMaxHP,label),targetHP=hpValue(target.targetHP,label);
    if(!patients.has(patientUUID)||seen.has(patientUUID)||targetHP!==resolvedTarget(intent,basisMaxHP)||maxByPool.has(poolUUID)&&maxByPool.get(poolUUID)!==basisMaxHP)throw Error(label);
    seen.add(patientUUID);maxByPool.set(poolUUID,basisMaxHP);
    return {patientUUID,poolUUID,intent,basisMaxHP,targetHP};
  });
  if(seen.size!==patients.size)throw Error(label);
  const expected=new Map(poolGoals(patientTargets).map(goal=>[goal.poolUUID,goal.targetHP])),goals=dataArray(goalsByPool,label);
  if(goals.length!==expected.size)throw Error(label);
  for(const input of goals){
    const goal=fields(input,['poolUUID','targetHP'],label,['poolUUID','targetHP']),poolUUID=uuid(goal.poolUUID,label),targetHP=hpValue(goal.targetHP,label);
    if(!expected.has(poolUUID)||expected.get(poolUUID)!==targetHP)throw Error(label);
    expected.delete(poolUUID);
  }
  return {version:1,patientTargets,...completionFlags(value,label,true)};
}
