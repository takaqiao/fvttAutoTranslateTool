import {MODULE_ID} from './schema.mjs';
import {canonicalJSON} from './revision-codec.mjs';
import {canonicalItemSource} from './source-ids.mjs';

const descriptor=Object.freeze({version:1,providerId:'patreon-v3',providerVersion:'3.2.29',
 baseSourceSHA256:'89ded325b92fa6b03dcf9257337ae2d628b3e99c987fe22cef9fff4e1837f4e9',
 pf2eSourceSHA256:'d63da8312831b84905e6866b1dd3f9d93e95c1012955b0177ad2ce8ccf246157'});
const immunitySource='Compendium.pf2e.feat-effects.Item.Lb4q2bBAgxamtix5';
const author=message=>message?.author?.id??message?.author??message?.user?.id??message?.user;
function same(a,b){try{return canonicalJSON(a)===canonicalJSON(b)}catch{return false}}
const copy=value=>JSON.parse(canonicalJSON(value));

/** Observe only the fixed provider's original create Promise. Item flags alone are not completion. */
export function createPatreonManualImmunity({game,fromUuid}){
 const provider=()=>game.modules?.get('patreon-v3');
 const api=()=>provider()?.api?.explorationManualImmunity;
 const available=()=>provider()?.active===true&&provider().version==='3.2.29'&&game.system?.version==='8.5.1'&&same(api()?.descriptor,descriptor);
 function root(binding){
  if(!binding||!['invocationId','messageId','useId','tag','actorUUID','patientUUID','sourceUserId'].every(key=>typeof binding[key]==='string'&&binding[key])
   ||binding.tag!==`exploration-manual:${binding.useId}`||!Number.isFinite(binding.startedAt))return null;
  const message=game.messages?.get(binding.messageId),c=message?.flags?.pf2e?.context,meta=message?.flags?.[MODULE_ID]?.explorationManualNative;
  const snapshot=binding.targetSnapshot,token=snapshot?.actorUUID===binding.patientUUID&&typeof snapshot.tokenUUID==='string'&&snapshot.tokenUUID.startsWith('Scene.')&&snapshot.tokenUUID.includes('.Token.');
  const target=snapshot?token&&(snapshot.type==='patreon-single-target'?c?.target==null:snapshot.type==='check-context'&&c?.target?.actor===binding.patientUUID&&(!c.target.token||c.target.token===snapshot.tokenUUID)):c?.target?.actor===binding.patientUUID;
  if(!available()||message?.isCheckRoll!==true||message.isReroll||message.rolls?.[0]?._evaluated!==true||c?.type!=='skill-check'
   ||!Array.isArray(c.options)||!c.options.includes('action:treat-wounds')||!c.options.includes(binding.tag)||c.origin?.actor!==binding.actorUUID||!target
   ||message.speaker?.actor!==binding.actorUUID.replace(/^Actor\./,'')||author(message)!==binding.sourceUserId
   ||meta?.useId!==binding.useId||meta.tag!==binding.tag||meta.patientUUID!==binding.patientUUID||meta.startedAt!==binding.startedAt
   ||binding.recordingSessionId&&binding.recordingSessionId!==meta.recordingSessionId
   ||meta.riskySurgery!==false||c.options.includes('risky-surgery')||!Array.isArray(message.flags.pf2e.modifiers)||message.flags.pf2e.modifiers.some(m=>m.slug==='risky-surgery'&&m.enabled)
   ||!same(meta.patreonImmunity,binding))return null;
  return message;
 }
 async function evidence(proof,activity){
  if(!proof||!same(proof.descriptor,descriptor)||activity?.source?.type!=='native-action'||activity.kind!=='treatment'
   ||activity.patientUUIDs?.length!==1||activity.temporalSource?.type==='checkpoint-reservation')return null;
  const binding=proof.binding,message=root(binding);if(!message||activity.proof.useId!==binding.useId||activity.actorUUID!==binding.actorUUID
   ||binding.recordingSessionId&&activity.sessionId!==binding.recordingSessionId
   ||activity.patientUUIDs[0]!==binding.patientUUID||!activity.proof.checkIds.includes(binding.messageId)||activity.startedAt!==binding.startedAt
   ||game.time.worldTime!==binding.startedAt)return null;
  const healer=await fromUuid(binding.actorUUID),patient=await fromUuid(binding.patientUUID),item=await fromUuid(proof.itemUUID),targetToken=binding.targetSnapshot&&await fromUuid(binding.targetSnapshot.tokenUUID);
  // Keep only resolved documents. The ledger can recheck them synchronously
  // after its own reads and revision digest preparation have finished.
  function isCurrent(current=activity){
   if(current?.source?.type!=='native-action'||current.kind!=='treatment'||current.temporalSource?.type==='checkpoint-reservation'
    ||current.proof.useId!==binding.useId||current.actorUUID!==binding.actorUUID||current.patientUUIDs?.length!==1||current.patientUUIDs[0]!==binding.patientUUID
    ||binding.recordingSessionId&&current.sessionId!==binding.recordingSessionId||!current.proof.checkIds.includes(binding.messageId)||current.startedAt!==binding.startedAt||root(binding)!==message)return false;
   const sourceUser=game.users?.get(binding.sourceUserId),creator=game.users?.get(proof.creatorId),duration=item?.system?.duration,start=item?.system?.start?.value;
   const seconds=duration?.value*({minutes:60,hours:3600}[duration?.unit]??NaN),expected=message.flags[MODULE_ID].explorationManualNative.continualRecovery?600:3600;
   const source=canonicalItemSource(item?.sourceId??item?._stats?.compendiumSource??item?.flags?.core?.sourceId);
   const items=Array.from(patient?.items?.values?.()??patient?.items??[]).filter(i=>{
   const mark=i.flags?.[MODULE_ID]?.explorationManualPatreonImmunity,source=canonicalItemSource(i.sourceId??i._stats?.compendiumSource??i.flags?.core?.sourceId);
   return mark?.useId===binding.useId&&mark.messageId===binding.messageId
    ||source===immunitySource&&i.system?.start?.value===binding.startedAt&&i.system?.context?.origin?.actor===binding.actorUUID;
   });
   if(healer?.uuid!==binding.actorUUID||patient?.uuid!==binding.patientUUID||!sourceUser?.active||healer.testUserPermission?.(sourceUser,'OWNER')!==true
   ||binding.targetSnapshot&&(targetToken?.uuid!==binding.targetSnapshot.tokenUUID||targetToken.actor!==patient||targetToken.parent?.tokens?.get(targetToken.id)!==targetToken)
   ||!creator?.active||!creator.isGM&&patient.testUserPermission?.(creator,'OWNER')!==true||items.length!==1||items[0]!==item
   ||item?.type!=='effect'||item.actor!==patient||patient.items.get(item.id)!==item||source!==immunitySource
   ||!same(item.flags?.[MODULE_ID]?.explorationManualPatreonImmunity,{...binding,creatorId:proof.creatorId})
   ||start!==binding.startedAt||start!==proof.start||!same(duration,proof.duration)||duration?.expiry!=='turn-start'||duration.sustained!==false
   ||seconds!==expected||proof.expiresAt!==start+seconds||game.time.worldTime!==binding.startedAt)return false;
   return true;
  }
  return isCurrent()?{item,proof:copy(proof),isCurrent}:null;
 }
 function noApplication(activity){
  if(activity.source?.type!=='native-action'||activity.kind!=='treatment'||activity.temporalSource?.type==='checkpoint-reservation')return false;
  const proof=activity.proof.nativeImmunity,message=proof&&root(proof.binding);
  return !!message&&activity.options.effectiveOutcome==='failure'&&message.flags.pf2e.context.outcome==='failure'
   &&Array.isArray(message.flags.pf2e.modifiers)&&activity.proof.resultIds.length===0&&activity.proof.receiptIds.length===0
   &&!Array.from(game.messages?.contents??game.messages?.values?.()??[]).some(child=>child.flags?.pf2e?.origin?.messageId===message.id);
 }
 function subscribe(observe){
  const providerAPI=api();if(!available()||typeof providerAPI.subscribe!=='function')return ()=>{};
  let active=true;
  const dispose=providerAPI.subscribe(event=>{
   if(!active||api()!==providerAPI||!available()||!same(event?.descriptor,descriptor)||!root(event.binding)||typeof event.terminalPromise?.then!=='function')return;
   event.terminalPromise.then(proof=>{
    if(!active||api()!==providerAPI||!available()||!same(proof?.descriptor,descriptor)||!same(event.binding,proof.binding)||!root(proof.binding))return;
    try{observe(copy(proof))}catch{}
   },()=>{});
  });
  return ()=>{active=false;dispose?.()};
 }
 return {subscribe,evidence,noApplication};
}
