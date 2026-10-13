import {canonicalJSON} from './revision-codec.mjs';

// No script calls this. It stays exported for tests/exploration/manual-pool-batch.test.mjs:
// tools/native-manual-pool-static-receiver/patch.mjs pins that test file's hash and refuses to load when it changes.
export function deduplicatePoolEffects(effects) {
  const selected=new Map();
  for(const e of effects){const key=JSON.stringify([e.poolUUID,e.effectId]);if(!selected.has(key)||selected.get(key).amount<e.amount)selected.set(key,e)}
  return [...selected.values()];
}
const health=a=>a?.modules?.['pf2e-toolbelt']?.shareData?.data?.health===true;
const changed=(changes,path)=>Object.hasOwn(changes??{},path)?changes[path]:path.split('.').reduce((v,p)=>v?.[p],changes);
const hpFields=changes=>Object.fromEntries(['system.attributes.hp.value','system.attributes.hp.sp.value','system.attributes.hp.temp'].map(k=>[k,k in changes?changes[k]:k.split('.').reduce((v,p)=>v?.[p],changes)]).filter(([,v])=>v!==undefined));
export function createHpPools({game,actorUpdateEvents}) {
  let active=null,forwarding=null;
  function discover(actor) {
    const api=game.toolbelt?.api?.shareData;
    let enabled=false;try{enabled=game.modules?.get('pf2e-toolbelt')?.active===true&&game.settings.get('pf2e-toolbelt','shareData.enabled')===true}catch{}
    if(!enabled)return {poolUUID:actor.uuid,memberUUIDs:[actor.uuid],provider:'native',ready:true};
    if(!api?.getMasterInMemory||!api?.getSlavesInMemory)return {poolUUID:actor.uuid,memberUUIDs:[actor.uuid],provider:'pf2e-toolbelt',ready:false,reason:'share-data-api-unavailable'};
    const master=health(actor)?api.getMasterInMemory(actor):null;
    if(health(actor)&&!master)return {poolUUID:actor.uuid,memberUUIDs:[actor.uuid],provider:'pf2e-toolbelt',ready:false,reason:'share-data-master-unavailable'};
    const root=master||actor;
    const members=[root,...api.getSlavesInMemory(root,false).filter(health)];
    return {poolUUID:root.uuid,memberUUIDs:[...new Set(members.map(a=>a.uuid))],provider:members.length>1?'pf2e-toolbelt':'native',ready:true};
  }
  function assertCurrentPool(scope,changes) {
    const current=discover(scope.patient);
    if(!current.ready||current.poolUUID!==scope.pool.poolUUID)throw Error('hp-pool-domain-changed');
    if(current.poolUUID!==scope.patient.uuid&&!game.toolbelt.api.shareData.getMasterInMemory(scope.patient)?.isOwner)throw Error('shared-hp-master-owner-required');
    // Toolbelt selects its forwarding target from incoming flags before the
    // prepared sharing graph changes. An HP application cannot change that link.
    const data=changed(changes,'flags.pf2e-toolbelt.shareData');
    const incoming=data?.['==data']??data?.data;
    const incomingMaster=incoming?.master??changed(changes,'flags.pf2e-toolbelt.shareData.data.master');
    const incomingHealth=incoming?.health??changed(changes,'flags.pf2e-toolbelt.shareData.data.health');
    if(incomingMaster!==undefined&&incomingMaster!==scope.masterId)throw Error('hp-pool-domain-changed');
    if(incomingHealth!==undefined&&incomingHealth!==(scope.pool.poolUUID!==scope.patient.uuid))throw Error('hp-pool-domain-changed');
  }
  function rejectForward(scope,error) {
    scope.forwardError=error;
    const rejected=Promise.reject(error);rejected.catch(()=>{});return rejected;
  }
  const nativeReceiptSource=receipt=>canonicalJSON(typeof receipt.toObject==='function'?receipt.toObject(true):{speaker:receipt.speaker,flags:receipt.flags});
  function directHealingNoChange(scope,before,rawBefore,result,receiptBefore) {
    const {patient,pool,fields}=scope,receipt=result?.receipt,pf=receipt?.flags?.pf2e,c=pf?.context,d=pf?.appliedDamage;
    if(pool.poolUUID!==patient.uuid||!scope.patientCall||Object.keys(fields??{}).length!==1||
      fields['system.attributes.hp.value']!==before.value||!Number.isFinite(before.max)||before.max<=0||before.value!==before.max||
      !rawBefore||rawBefore.value!==before.value||canonicalJSON(patient._source?.system?.attributes?.hp)!==canonicalJSON(rawBefore))return false;
    const hp=patient.system?.attributes?.hp??{},current=Object.fromEntries(Object.keys(before).map(key=>[key,hp[key]]));
    if(canonicalJSON(current)!==canonicalJSON(before)||!receipt||game.messages?.get(receipt.id)!==receipt||nativeReceiptSource(receipt)!==receiptBefore||
      (receipt.author?.id??receipt.author??receipt.user?.id??receipt.user)!==game.user?.id||receipt.speaker?.actor!==patient.id||
      c?.type!=='damage-taken'||!Array.isArray(c.domains)||c.domains.length!==1||c.domains[0]!=='healing-received'||!Array.isArray(c.options))return false;
    const prefix=`pf2e-third-party-automation:exploration-apply:${scope.activityId}:`,suffix=`:${patient.uuid}`;
    const applications=c.options.filter(option=>typeof option==='string'&&option.startsWith(prefix));
    if(applications.length!==1||!applications[0].endsWith(suffix))return false;
    const resultId=applications[0].slice(prefix.length,-suffix.length);
    if(!/^[A-Za-z0-9]+$/.test(resultId)||!c.options.includes(`pf2e-third-party-automation:source:${resultId}:0`))return false;
    return !!d&&Object.keys(d).length===5&&d.uuid===patient.uuid&&d.isHealing===true&&d.shield===null&&
      Array.isArray(d.persistent)&&d.persistent.length===0&&Array.isArray(d.updates)&&d.updates.length===0;
  }
  const unregister=actorUpdateEvents?.addActorUpdateMiddleware(function(wrapped,changes={},options={}){
    const scope=active,fields=hpFields(changes);
    if(forwarding&&Object.keys(fields).length){
      const current=discover(forwarding.patient);
      if(this.uuid!==forwarding.patient.uuid&&this.uuid===current.poolUUID&&this.uuid!==forwarding.pool.poolUUID)return rejectForward(forwarding,Error('hp-pool-domain-changed'));
    }
    if(forwarding&&this.uuid===forwarding.pool.poolUUID&&Object.keys(fields).length){
      try{assertCurrentPool(forwarding)}catch(error){return rejectForward(forwarding,error)}
      if(JSON.stringify(fields)!==JSON.stringify(forwarding.fields)||forwarding.masterPromise)throw Error('ambiguous-share-data-forward');
      const promise=wrapped(changes,options);forwarding.masterPromise=Promise.resolve(promise);forwarding.masterPromise.catch(()=>{});return promise;
    }
    if(!scope||this.uuid!==scope.patient.uuid||!Object.keys(fields).length)return wrapped(changes,options);
    assertCurrentPool(scope,changes);
    if(scope.patientCall)throw Error('duplicate-patient-hp-write');scope.patientCall=true;
    scope.fields=structuredClone(fields);
    if(scope.pool.poolUUID===this.uuid){const p=wrapped(changes,options);scope.masterPromise=Promise.resolve(p);return p}
    const previous=forwarding;forwarding=scope;
    try{return wrapped(changes,options)}finally{forwarding=previous}
  });
  async function withNativeApplication(activity,patient,operation) {
    if(active)throw Error('hp-application-busy');const pool=discover(patient);if(!pool.ready)throw Error(pool.reason);
    if(Object.hasOwn(activity,'hpPoolUUIDs')&&(!Array.isArray(activity.hpPoolUUIDs)||!activity.hpPoolUUIDs.includes(pool.poolUUID)))throw Error('hp-pool-domain-changed');
    const shared=pool.poolUUID!==patient.uuid;
    const master=shared?game.toolbelt.api.shareData.getMasterInMemory(patient):patient;
    if(shared&&!master.isOwner)throw Error('shared-hp-master-owner-required');
    if(!actorUpdateEvents)throw Error('hp-update-observer-unavailable');
    const hp=master.system?.attributes?.hp??{},before=Object.fromEntries(['value','max','temp','sp'].filter(k=>hp[k]!==undefined).map(k=>[k,structuredClone(hp[k])]));
    const rawBefore=!shared&&master._source?.system?.attributes?.hp?structuredClone(master._source.system.attributes.hp):null;
    const scope={activityId:activity.id,patient,pool,masterId:shared?master.id:undefined,patientCall:false,masterPromise:null};active=scope;
    // Toolbelt forwards synchronously inside async _preUpdate, after update()
    // has returned its Promise. Bound only this exact original patient boundary;
    // keep no broad master watcher alive while awaiting the application.
    const originalPreUpdate=patient._preUpdate;
    const preUpdate=typeof originalPreUpdate==='function'?function(changes,...args){
      const fields=hpFields(changes);if(this!==patient||!scope.patientCall||JSON.stringify(fields)!==JSON.stringify(scope.fields))return originalPreUpdate.call(this,changes,...args);
      assertCurrentPool(scope,changes);
      const previous=forwarding;forwarding=scope;try{return originalPreUpdate.call(this,changes,...args)}finally{forwarding=previous}
    }:null;
    if(shared&&preUpdate)patient._preUpdate=preUpdate;
    try{
      const result=await operation(scope);
      if(scope.forwardError)throw scope.forwardError;
      if(!scope.masterPromise){
        const receipt=result?.receipt,c=receipt?.flags?.pf2e?.context;
        if(receipt&&game.messages?.get(receipt.id)===receipt&&c?.type==='damage-taken'&&receipt.flags.pf2e.appliedDamage===null&&receipt.speaker?.actor===patient.id&&c.options?.some(o=>o.startsWith(`pf2e-third-party-automation:exploration-apply:${activity.id}:`)||o===`pf2e-third-party-automation:salubrious-apply:${activity.id}`))return {result,poolReceipt:{activityId:activity.id,actorUUID:pool.poolUUID,noChange:true,receiptId:receipt.id}};
        throw Error('native-hp-forward-unconfirmed');
      }
      const receiptBefore=!shared&&result?.receipt?nativeReceiptSource(result.receipt):null;
      const saved=await scope.masterPromise;
      // Foundry filters an empty HP diff and returns undefined. PF2e still
      // saves a healing receipt with an empty undo delta at full health.
      if(saved===undefined&&directHealingNoChange(scope,before,rawBefore,result,receiptBefore)){
        assertCurrentPool(scope);
        return {result,poolReceipt:{activityId:activity.id,actorUUID:pool.poolUUID,noChange:true,receiptId:result.receipt.id}};
      }
      if(saved?.uuid!==pool.poolUUID)throw Error('native-hp-forward-unconfirmed');
      const after={...before};for(const [path,value] of Object.entries(scope.fields)){if(path==='system.attributes.hp.value')after.value=value;if(path==='system.attributes.hp.temp')after.temp=value;if(path==='system.attributes.hp.sp.value')after.sp={...before.sp,value}}
      return {result,poolReceipt:{activityId:activity.id,actorUUID:pool.poolUUID,patientUUID:patient.uuid,before,after,fields:scope.fields,provider:pool.provider}};
    }finally{if(shared&&preUpdate&&patient._preUpdate===preUpdate)patient._preUpdate=originalPreUpdate;active=null}
  }
  return {discover,withNativeApplication,dispose:()=>unregister?.()};
}
