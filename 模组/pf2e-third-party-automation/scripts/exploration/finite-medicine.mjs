import {canonicalItemSource} from './source-ids.mjs';

export const SOURCES=Object.freeze({
  battleMedicine:'Compendium.pf2e.feats-srd.Item.wYerMk6F1RZb0Fwt',
  medic:'Compendium.pf2e.feats-srd.Item.MJg24e9fJd7OASvF',
  robust:'Compendium.pf2e.feats-srd.Item.yTLGclKtWVFZLKIz',
  godless:'Compendium.pf2e.feats-srd.Item.tnzZvaJ97t0N9g6y',
  immunity:'Compendium.pf2e.feat-effects.Item.2XEYQNZTCGpdkyR6',
  tools:Object.freeze(['Compendium.pf2e.equipment-srd.Item.s1vB3HdXjMigYAnY','Compendium.pf2e.equipment-srd.Item.SGkOHFyBbzWdBk8D'])
});
const moduleId='pf2e-third-party-automation',ranks=['trained','expert','master','legendary'];
const baselineFields=['id','actorUUID','itemUUID','source','nativeCounter','remaining','period','checkedAt','reviewedBy','previousBaselineId','renewal'];
const claimFields=['id','baselineId','actorUUID','itemUUID','period','state','activityId','permitNonce','claimedAt','checkId','usedAt'];
const field=(input,key)=>Object.getOwnPropertyDescriptor(input??{},key)?.value;
const validTime=value=>Number.isFinite(value)&&Math.abs(value)<=Number.MAX_SAFE_INTEGER;
const validId=value=>typeof value==='string'&&value.length>0&&value.length<=128&&value.trim()===value&&!/[\u0000-\u001f\u007f]/.test(value)&&!['__proto__','prototype','constructor'].includes(value);
const validActor=value=>typeof value==='string'&&value.length<=128&&/^(?:Actor\.[A-Za-z0-9_-]+|Scene\.[A-Za-z0-9_-]+\.Token\.[A-Za-z0-9_-]+\.Actor\.[A-Za-z0-9_-]+)$/.test(value);
const validActorItem=(value,actorUUID)=>validId(value)&&value.startsWith(`${actorUUID}.Item.`)&&/^[A-Za-z0-9_-]+$/.test(value.slice(`${actorUUID}.Item.`.length));
function dataRecord(input){
  return !!input&&typeof input==='object'&&[Object.prototype,null].includes(Object.getPrototypeOf(input))&&Reflect.ownKeys(input).every(key=>{
    const d=Object.getOwnPropertyDescriptor(input,key);return typeof key==='string'&&validId(key)&&d.enumerable&&Object.hasOwn(d,'value');
  });
}
function fields(input,allowed,label,required=[]){
  if(!dataRecord(input)||Reflect.ownKeys(input).some(key=>!allowed.includes(key))||required.some(key=>!Object.hasOwn(input,key)))throw Error(label);
  return Object.fromEntries(Object.keys(input).map(key=>[key,field(input,key)]));
}
function dataArray(input,label){
  if(!Array.isArray(input)||Object.getPrototypeOf(input)!==Array.prototype||Reflect.ownKeys(input).length!==input.length+1)throw Error(label);
  return Array.from({length:input.length},(_,index)=>{
    const d=Object.getOwnPropertyDescriptor(input,String(index));if(!d?.enumerable||!Object.hasOwn(d,'value'))throw Error(label);return d.value;
  });
}
function normalizeBaseline(input,actorUUID,label){
  const b=fields(input,baselineFields,label,baselineFields);
  if(!validId(b.id)||b.actorUUID!==actorUUID||!validActorItem(b.itemUUID,actorUUID)||
    b.source!=='gm-reviewed'||b.nativeCounter!==false||![0,1].includes(b.remaining)||!['daily','hourly'].includes(b.period)||!validTime(b.checkedAt)||!validId(b.reviewedBy))throw Error(label);
  if(b.renewal==='initial'){if(b.previousBaselineId!==null)throw Error(label)}
  else if(!validId(b.previousBaselineId)||b.previousBaselineId===b.id||b.renewal!==(b.period==='daily'?'new-preparation':'hour-window'))throw Error(label);
  return b;
}
export function normalizeFiniteMedicine(input=undefined,actorUUIDs=[]){
  const label='invalid-finite-medicine',actors=dataArray(actorUUIDs,label),selected=new Set(actors);
  if(actors.some(actor=>!validActor(actor))||actors.length!==selected.size)throw Error(label);
  const value=fields(input===undefined?{}:input,['version','secondsPerUse','battleMedicine','medicBypass'],label);
  if(Object.hasOwn(value,'version')&&value.version!==1)throw Error(label);
  const secondsPerUse=Object.hasOwn(value,'secondsPerUse')?value.secondsPerUse:6;
  if(!validTime(secondsPerUse)||secondsPerUse<0)throw Error(label);
  const map=(input,normalize)=>{
    if(!dataRecord(input))throw Error(label);
    return Object.fromEntries(Object.keys(input).map(key=>{if(!selected.has(key))throw Error(label);return [key,normalize(field(input,key),key)]}));
  };
  const group=(input,medic)=>{
    const value=fields(input,medic?['enabled','maxUsesByActor','baselineByActor']:['enabled','maxUsesByActor','rankByActor'],label);
    const enabled=Object.hasOwn(value,'enabled')?value.enabled:false;if(typeof enabled!=='boolean')throw Error(label);
    const maxUsesByActor=map(Object.hasOwn(value,'maxUsesByActor')?value.maxUsesByActor:{},count=>{
      if(!Number.isSafeInteger(count)||count<(medic?0:1)||count>(medic?1:100))throw Error(label);return count;
    });
    if(medic)return {enabled,maxUsesByActor,baselineByActor:map(Object.hasOwn(value,'baselineByActor')?value.baselineByActor:{},(b,actor)=>normalizeBaseline(b,actor,label))};
    return {enabled,maxUsesByActor,rankByActor:map(Object.hasOwn(value,'rankByActor')?value.rankByActor:{},rank=>{if(!ranks.includes(rank))throw Error(label);return rank})};
  };
  return {version:1,secondsPerUse,battleMedicine:group(Object.hasOwn(value,'battleMedicine')?value.battleMedicine:{},false),medicBypass:group(Object.hasOwn(value,'medicBypass')?value.medicBypass:{},true)};
}

export function battleMedicineDuration({robust,godless}){
  if(typeof robust!=='boolean'||typeof godless!=='boolean')throw Error('invalid-battle-medicine');
  return robust||godless?3600:86400;
}
export function battleMedicineOutcome({degree,rank,medic}){
  if(!Number.isInteger(degree)||degree<0||degree>3||!ranks.includes(rank)||typeof medic!=='boolean')throw Error('invalid-battle-medicine');
  const tier=ranks.indexOf(rank),medicBonus=medic?[0,5,10,15][tier]:0,bonus=[0,10,30,50][tier]+medicBonus;
  return {outcome:['criticalFailure','failure','success','criticalSuccess'][degree],
    formula:degree===0?'1d8':degree===1?null:`${degree===3?'4d8':'2d8'}${bonus?`+${bonus}`:''}`,medicBonus};
}
function immunityExpiry(effect,now){
  if(Object.hasOwn(effect,'expiresAt'))return validTime(effect.expiresAt)?effect.expiresAt:null;
  const start=effect.system?.start?.value,duration=effect.system?.duration;
  if(start!==undefined||duration!==undefined){
    const multiplier={rounds:6,minutes:60,hours:3600,days:86400}[duration?.unit];
    return validTime(start)&&validTime(duration?.value)&&duration.value>=0&&multiplier&&validTime(start+duration.value*multiplier)?start+duration.value*multiplier:null;
  }
  const remaining=effect.remainingDuration?.remaining;
  return validTime(remaining)&&remaining>=0&&validTime(now+remaining)?now+remaining:null;
}
export function battleMedicineImmunity({effects,patientUUID,healerUUID,now}){
  const clear={status:'clear',expiresAt:null,effectIds:[]};
  if(!validActor(patientUUID)||!validActor(healerUUID)||!validTime(now)||!Array.isArray(effects))return {...clear,status:'uncertain'};
  let uncertain=false,expiresAt=null;const effectIds=[];
  for(const effect of effects){
    const sources=[effect?.sourceId,effect?._stats?.compendiumSource,effect?.flags?.core?.sourceId].filter(value=>value!==undefined&&value!==null).map(canonicalItemSource);
    if(!sources.includes(SOURCES.immunity))continue;
    if(effect.isExpired===true||effect.remainingDuration?.expired===true||effect.isSuppressed===true||effect.system?.suppressed===true)continue;
    const targets=[effect.patientUUID,effect.parent?.uuid].filter(value=>value!==undefined&&value!==null);
    if(targets.some(value=>!validActor(value))||new Set(targets).size>1){uncertain=true;continue}
    if(targets.length&&targets[0]!==patientUUID)continue;
    const healers=[effect.flags?.[moduleId]?.healerUuid,effect.system?.context?.origin?.actor].filter(value=>value!==undefined&&value!==null);
    if(!healers.length||healers.some(value=>!validActor(value))||new Set(healers).size!==1||sources.some(value=>value!==SOURCES.immunity)){uncertain=true;continue}
    const originItem=effect.system?.context?.origin?.item;
    if(originItem!==undefined&&originItem!==null&&!validActorItem(originItem,healers[0])){uncertain=true;continue}
    if(healers[0]!==healerUUID)continue;
    const end=immunityExpiry(effect,now),id=effect.uuid??effect.id??effect._id;
    if(end===null||!validId(id)){uncertain=true;continue}
    if(end<=now)continue;
    expiresAt=Math.max(expiresAt??end,end);effectIds.push(id);
  }
  return {status:uncertain?'uncertain':expiresAt===null?'clear':'immune',expiresAt,effectIds:[...new Set(effectIds)].sort()};
}

const sameProtocol=(left,right)=>dataRecord(left)&&dataRecord(right)&&left.version===1&&right.version===1&&validId(left.rootUUID)&&validId(left.epoch)&&left.rootUUID===right.rootUUID&&left.epoch===right.epoch;
const sameBaseline=(left,right)=>baselineFields.every(key=>left[key]===right[key]);
const availability=(state,claimIds=[])=>({state,remaining:state==='available'?1:0,claimIds:[...new Set(claimIds)].sort()});
export function medicAvailability({sessions,activities},{actorUUID,itemUUID,baselineId,now}){
  if(!validActor(actorUUID)||!validId(itemUUID)||!validId(baselineId)||!validTime(now)||!dataRecord(sessions)||!dataRecord(activities))return availability('uncertain');
  const reviews=[];
  try{
    for(const session of Object.values(sessions)){
      if(!dataRecord(session))throw Error('invalid-medic-state');
      const finite=field(session,'finiteMedicine');if(finite===undefined)continue;
      if(!dataRecord(finite)||!dataRecord(field(finite,'medicBypass'))||!dataRecord(field(field(finite,'medicBypass'),'baselineByActor')))throw Error('invalid-medic-state');
      const review=field(field(field(finite,'medicBypass'),'baselineByActor'),actorUUID);
      if(review!==undefined)reviews.push({baseline:normalizeBaseline(review,actorUUID,'invalid-medic-state'),protocol:field(session,'protocol')});
    }
  }catch{return availability('uncertain')}
  const found=reviews.filter(row=>row.baseline.id===baselineId&&row.baseline.itemUUID===itemUUID);
  if(!found.length)return availability('unreviewed');
  const baseline=found[0].baseline,protocol=found[0].protocol;
  if(!sameProtocol(protocol,protocol)||found.some(row=>!sameProtocol(row.protocol,protocol)||!sameBaseline(row.baseline,baseline))||baseline.checkedAt>now)return availability('uncertain');
  const scoped=reviews.filter(row=>sameProtocol(row.protocol,protocol)),unique=new Map();
  for(const row of scoped){
    const old=unique.get(row.baseline.id);if(old&&!sameBaseline(old,row.baseline))return availability('uncertain');unique.set(row.baseline.id,row.baseline);
  }
  const related=new Set([baseline.id]);let predecessor=baseline;
  while(predecessor.previousBaselineId!==null){
    const previous=unique.get(predecessor.previousBaselineId);
    if(!previous||related.has(previous.id)||previous.checkedAt>predecessor.checkedAt||previous.itemUUID!==itemUUID)return availability('uncertain');
    related.add(previous.id);predecessor=previous;
  }
  let superseded=false;
  for(const review of unique.values())if(review.previousBaselineId===baseline.id){
    if(superseded||review.itemUUID!==itemUUID||review.checkedAt<baseline.checkedAt||review.checkedAt>now)return availability('uncertain');superseded=true;
  }
  let uncertain=false,spent=baseline.remaining===0||superseded;const claimIds=[],seen=new Map();
  for(const session of Object.values(sessions)){
    if(!sameProtocol(field(session,'protocol'),protocol))continue;
    const events=field(session,'finiteMedicineReviewChecks');if(events===undefined)continue;
    try{
      for(const event of dataArray(events,'invalid-medic-state')){
        const e=fields(event,['actorUUID','checkId','observedAt'],'invalid-medic-state',['actorUUID','checkId','observedAt']);
        if(!validActor(e.actorUUID)||!validId(e.checkId)||!validTime(e.observedAt))throw Error('invalid-medic-state');
        // Protected events are appended after review even when world time stays in the same tick.
        if(e.actorUUID===actorUUID&&e.observedAt>=baseline.checkedAt)uncertain=true;
      }
    }catch{uncertain=true}
  }
  for(const activity of Object.values(activities)){
    if(!dataRecord(activity)){uncertain=true;continue}
    if(!sameProtocol(field(sessions,field(activity,'sessionId'))?.protocol,protocol))continue;
    const proof=field(activity,'proof');if(!dataRecord(proof)){uncertain=true;continue}
    const input=field(proof,'medicBypass');if(input===undefined)continue;
    if(!dataRecord(input)){uncertain=true;continue}
    if(field(input,'actorUUID')!==actorUUID||!related.has(field(input,'baselineId')))continue;
    const id=field(input,'id');if(validId(id))claimIds.push(id);
    try{
      const c=fields(input,claimFields,'invalid-medic-state',claimFields),parent=unique.get(c.baselineId);
      if(!validId(c.id)||c.itemUUID!==itemUUID||c.period!==parent.period||c.activityId!==field(activity,'id')||field(activity,'actorUUID')!==actorUUID||field(activity,'providerId')!=='battle-medicine'||
        !validTime(c.claimedAt)||c.claimedAt<parent.checkedAt||c.claimedAt>now||!['reserved','claimed','used','released','uncertain'].includes(c.state)||
        c.permitNonce!==null&&!validId(c.permitNonce)||c.checkId!==null&&!validId(c.checkId)||c.usedAt!==null&&!validTime(c.usedAt))throw Error('invalid-medic-state');
      const previous=seen.get(c.id);if(previous&&claimFields.some(key=>previous[key]!==c[key]))throw Error('invalid-medic-state');seen.set(c.id,c);
      if(c.state==='released'){
        if(c.permitNonce!==null||c.checkId!==null||c.usedAt!==null||field(activity,'executor')!==undefined||!['planned','cancelled'].includes(field(activity,'state')))throw Error('invalid-medic-state');
        claimIds.pop();continue;
      }
      if(c.state==='used'){
        if(!validId(c.permitNonce)||!validId(c.checkId)||!validTime(c.usedAt)||c.usedAt<parent.checkedAt||c.usedAt>now)throw Error('invalid-medic-state');
        if(c.baselineId===baseline.id)spent=true;
      }else uncertain=true;
    }catch{uncertain=true}
  }
  return availability(uncertain?'uncertain':spent?'spent':'available',claimIds);
}
