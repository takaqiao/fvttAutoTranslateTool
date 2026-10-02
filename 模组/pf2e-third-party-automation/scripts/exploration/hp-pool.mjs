import {canonicalJSON} from './revision-codec.mjs';

export function deduplicatePoolEffects(effects) {
  const selected=new Map();
  for(const e of effects){const key=JSON.stringify([e.poolUUID,e.effectId]);if(!selected.has(key)||selected.get(key).amount<e.amount)selected.set(key,e)}
  return [...selected.values()];
}
const health=a=>a?.modules?.['pf2e-toolbelt']?.shareData?.data?.health===true;
const changed=(changes,path)=>Object.hasOwn(changes??{},path)?changes[path]:path.split('.').reduce((v,p)=>v?.[p],changes);
const hpFields=changes=>Object.fromEntries(['system.attributes.hp.value','system.attributes.hp.sp.value','system.attributes.hp.temp'].map(k=>[k,k in changes?changes[k]:k.split('.').reduce((v,p)=>v?.[p],changes)]).filter(([,v])=>v!==undefined));
export function manualHpBaseline(actor){
  const hp=actor?.hitPoints??actor?.system?.attributes?.hp,sp=actor?.system?.attributes?.hp?.sp;
  if(!hp||['value','max','temp'].some(key=>!Number.isFinite(hp[key])||hp[key]<0)||sp!==undefined&&(!sp||['value','max'].some(key=>!Number.isFinite(sp[key])||sp[key]<0)))throw Error('manual-pool-hp-inputs-unavailable');
  return {value:hp.value,max:hp.max,temp:hp.temp,...sp===undefined?{}:{sp:{value:sp.value,max:sp.max}}};
}
export function assertManualHpBaseline(actor,baseline){
  if(canonicalJSON(manualHpBaseline(actor))!==canonicalJSON(baseline))throw Error('manual-pool-hp-baseline-changed');
}
export function createHpPools({game,actorUpdateEvents}) {
  let active=null,forwarding=null,manual=null;
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
    if(manual&&this.uuid===manual.patient.uuid&&Object.keys(fields).length){
      if(this!==manual.updateActor)throw Error('manual-pool-update-instance-mismatch');
      const pending=manual;pending.checkWrite(changes);if(pending.fields)throw Error('duplicate-patient-hp-write');pending.fields=structuredClone(fields);
      const promise=wrapped(changes,options);if(pending.direct)pending.observeMaster(promise);return promise;
    }
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
    if(active||manual)throw Error('hp-application-busy');const pool=discover(patient);if(!pool.ready)throw Error(pool.reason);
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
  async function withManualApplication({permit,provider,validate,request,prepareForward,remoteCompletion,updateActor},patient,operation){
    updateActor??=patient;
    if(active||manual)throw Error('hp-application-busy');
    if(!actorUpdateEvents||typeof operation!=='function'||typeof validate!=='function'||provider?.descriptor?.sourceSHA256!=='2946fa27eaf0963098f9f48ee48f777c5cf65166b56b043de9404f09813c240f'||provider.descriptor.version!==1||provider.descriptor.hpBaselineGuardVersion!==1||typeof provider.subscribe!=='function')throw Error('manual-pool-provider-unavailable');
    const pool=discover(patient),master=pool.poolUUID===patient.uuid?patient:game.toolbelt?.api?.shareData?.getMasterInMemory(patient);
    if(!pool.ready||!master||pool.poolUUID!==permit.poolUUID||patient.uuid!==permit.selectedPatientUUID||pool.provider!=='pf2e-toolbelt')throw Error('manual-pool-domain-mismatch');
    const direct=pool.poolUUID===patient.uuid;
    const binding={permitNonce:permit.permitNonce,applicationNonce:permit.applicationNonce,ownerUserId:permit.ownerUserId,patientUUID:patient.uuid,poolUUID:pool.poolUUID};
    if(updateActor?.uuid!==patient.uuid)throw Error('manual-pool-update-instance-mismatch');
    const scope={patient,updateActor,pool,masterId:master.id,direct,fields:null,options:null,forwarded:false,masterPromise:null};
    const check=changes=>{
      const token=patient.token?.document??patient.token;
      if(token?token.actor!==patient||token.parent?.tokens?.get(token.id)!==token:game.actors?.get(patient.id)!==patient)throw Error('manual-pool-patient-instance-changed');
      const current=discover(patient);if(validate()!==true||!current.ready||current.poolUUID!==pool.poolUUID||(!direct&&game.toolbelt.api.shareData.getMasterInMemory(patient)!==master)||(direct&&!master.isOwner))throw Error('manual-pool-evidence-changed');
      const incoming=changed(changes,'flags.pf2e-toolbelt.shareData');
      if(incoming!==undefined||changed(changes,'flags.pf2e-toolbelt.shareData.data.master')!==undefined||changed(changes,'flags.pf2e-toolbelt.shareData.data.health')!==undefined)throw Error('hp-pool-domain-changed');
    };
    scope.check=check;check();
    const baseline=manualHpBaseline(master);
    // The contextual clone can outlive a claim await. Compare all native HP
    // inputs until the original master write starts, never after its success.
    const checkWrite=changes=>{
      check(changes);
      if(!scope.writeStarted)for(const actor of [master,patient,updateActor])assertManualHpBaseline(actor,baseline);
    };
    scope.checkWrite=checkWrite;checkWrite();
    const hp=master.system?.attributes?.hp??{},before=Object.fromEntries(['value','max','temp','sp'].filter(k=>hp[k]!==undefined).map(k=>[k,structuredClone(hp[k])]));
    scope.observeMaster=promise=>{
      scope.forwarded=true;const writer=game.user;
      scope.masterPromise=(async()=>{if(!promise||typeof promise.then!=='function')throw Error('manual-pool-native-promise-required');const saved=await promise;check();if(saved!==master||game.user!==writer)throw Error('manual-pool-forward-unconfirmed');return {binding,poolUUID:pool.poolUUID,writerUserId:writer.id,fields:scope.fields,before,terminal:'fulfilled'}})();scope.masterPromise.catch(()=>{});
    };
    manual=scope;
    const dispose=provider.subscribe(event=>{
      if(event.phase==='prepare'&&event.patient===patient){
        return (async()=>{checkWrite();if(!scope.fields||scope.forwarded||event.options!==scope.options||event.master!==master||canonicalJSON(event.fields)!==canonicalJSON(scope.fields))throw Error('manual-pool-forward-mismatch');scope.forwarded=true;if(!master.isOwner){if(typeof prepareForward!=='function')throw Error('manual-pool-forward-authorization-required');await prepareForward(scope.fields,baseline);checkWrite()}return {binding,validate:()=>{check();return true},beforeWrite:()=>{checkWrite();scope.writeStarted=true;return true}}})();
      }
      if(event.phase==='write'&&event.binding?.applicationNonce===binding.applicationNonce){
        if(!scope.forwarded||scope.masterPromise||event.master!==master||canonicalJSON(event.fields)!==canonicalJSON(scope.fields)||canonicalJSON(event.binding)!==canonicalJSON(binding))throw Error('manual-pool-terminal-mismatch');
        scope.masterPromise=event.terminalPromise;
      }
    });
    const original=patient._preUpdate;
    if(typeof original!=='function'){manual=null;dispose();throw Error('manual-pool-native-preupdate-required')}
    const wrapped=async function(changes,options,...args){
      if(this!==patient||!manual?.fields)return original.call(this,changes,options,...args);
      checkWrite(changes);if(canonicalJSON(hpFields(changes))!==canonicalJSON(scope.fields))throw Error('manual-pool-forward-mismatch');scope.options=options;
      try{
        const result=await original.call(this,changes,options,...args);
        // A directly treated master has no Toolbelt forwarding callback. Core
        // resolves its canonical pre-update before dispatching the saved delta.
        if(direct){checkWrite(changes);scope.writeStarted=true}
        return result;
      }finally{scope.options=null}
    };
    patient._preUpdate=wrapped;
    try{
      checkWrite();const originalPromise=operation();if(!originalPromise||typeof originalPromise.then!=='function')throw Error('manual-pool-native-promise-required');const result=await originalPromise;check();
      const receipt=result?.receipt;
      const receiptCurrent=()=>{const pf=receipt?.flags?.pf2e;return !!receipt&&game.messages?.get(receipt.id)===receipt&&receipt.speaker?.actor===patient.id&&pf?.context?.type==='damage-taken'&&pf.context.options?.includes(`pf2e-third-party-automation:source:${request.resultId}:${request.rollIndex}`)&&!pf.appliedDamage?.isReverted};
      if(!receiptCurrent())throw Error('manual-pool-receipt-unavailable');const pf=receipt.flags.pf2e,receiptSource=()=>canonicalJSON(typeof receipt.toObject==='function'?receipt.toObject(true):{speaker:receipt.speaker,flags:receipt.flags}),savedReceiptSource=receiptSource();
      if(!scope.forwarded){checkWrite();if(scope.fields||pf.appliedDamage!==null)throw Error('manual-pool-forward-unconfirmed');return {result,poolReceipt:{actorUUID:pool.poolUUID,patientUUID:patient.uuid,receiptId:receipt.id,noChange:true}}}
      if(pf.appliedDamage?.uuid!==patient.uuid||pf.appliedDamage.isHealing!==true)throw Error('manual-pool-receipt-unavailable');
      const terminal=await (scope.masterPromise??remoteCompletion?.(binding));check();
      if(!receiptCurrent()||receiptSource()!==savedReceiptSource||receipt.flags.pf2e.appliedDamage?.uuid!==patient.uuid||receipt.flags.pf2e.appliedDamage.isHealing!==true)throw Error('manual-pool-receipt-unavailable');
      if(!terminal||terminal.terminal!=='fulfilled'||terminal.poolUUID!==pool.poolUUID||canonicalJSON(terminal.binding)!==canonicalJSON(binding)||canonicalJSON(terminal.fields)!==canonicalJSON(scope.fields))throw Error('manual-pool-forward-unconfirmed');
      return {result,poolReceipt:{actorUUID:pool.poolUUID,patientUUID:patient.uuid,receiptId:receipt.id,noChange:false,master:terminal}};
    }finally{if(patient._preUpdate===wrapped)patient._preUpdate=original;dispose();manual=null}
  }
  return {discover,withNativeApplication,withManualApplication,dispose:()=>unregister?.()};
}
