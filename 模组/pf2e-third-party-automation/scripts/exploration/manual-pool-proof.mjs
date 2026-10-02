import {MODULE_ID} from './schema.mjs';
import {canonicalJSON} from './revision-codec.mjs';

const author=message=>message?.author?.id??message?.author??message?.user?.id??message?.user;
const copy=value=>JSON.parse(canonicalJSON(value));
const same=(a,b)=>canonicalJSON(a)===canonicalJSON(b);
const sourceOf=document=>canonicalJSON(document.toObject(true));
const actorCurrent=(game,actor)=>{const token=actor?.token?.document??actor?.token;return !!actor&&(token?token.actor===actor&&token.parent?.tokens?.get(token.id)===token:game.actors?.get(actor.id)===actor)};
async function digest(value){return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(value))),byte=>byte.toString(16).padStart(2,'0')).join('')}

/** Only an already sealed GM claim can remove the shared-completion gap. No
 * lookup, document marker, or master-only observer becomes an application. */
export function createManualPoolProof({game,fromUuid,hpPools,resolveSource}){
 return {async evidence(activity){
  const source=activity.proof?.manualPoolSource,claims=Object.values(activity.proof?.poolApplications??{});
  if(!source||claims.length!==1||claims[0].state!=='settled'||activity.patientUUIDs.length!==1)return null;
  const claim=copy(claims[0]),terminal=claim.terminal,request=claim.request,patientUUID=activity.patientUUIDs[0];
  if(!terminal||request?.activityId!==activity.id||request.sessionId!==activity.sessionId||request.actorUUID!==activity.actorUUID||request.useId!==activity.proof.useId||request.sourceType!==activity.source.type||request.patientUUIDs.length!==1||request.patientUUIDs[0]!==patientUUID||claim.selectedPatientUUID!==patientUUID||!activity.hpPoolUUIDs.includes(claim.poolUUID)||!activity.proof.receiptIds.includes(terminal.receiptId)||!activity.proof.checkIds.includes(request.checkId)||activity.proof.resultIds.length!==1||activity.proof.resultIds[0]!==request.resultId)return null;
  const receipt=game.messages.get(terminal.receiptId),check=game.messages.get(request.checkId),result=game.messages.get(request.resultId);
  if(!receipt||!check||!result||[receipt,check,result].some(document=>typeof document.toObject!=='function'))return null;
  const before={receipt:sourceOf(receipt),check:sourceOf(check),result:sourceOf(result),source:copy(source)},owner=game.users.get(claim.ownerUserId),writer=game.users.get(terminal.master?.writerUserId);
  const live=await resolveSource(copy(request),{callerId:claim.ownerUserId,readOnly:true}),patient=await fromUuid(patientUUID),master=await fromUuid(claim.poolUUID);
  if(!live)return null;
  const expectedDigest=await digest(canonicalJSON({binding:live.binding,documents:canonicalJSON([check.toObject(true),result.toObject(true)])}));
  const validate=current=>{
   if(!current||!same(current.proof.manualPoolSource,before.source)||Object.values(current.proof.poolApplications??{}).length!==1||!same(Object.values(current.proof.poolApplications)[0],claim)||current.proof.receiptIds.length!==1||current.proof.receiptIds[0]!==receipt.id||live.isCurrent()!==true||claim.sourceDigest!==expectedDigest||game.messages.get(check.id)!==check||game.messages.get(result.id)!==result||game.messages.get(receipt.id)!==receipt||sourceOf(check)!==before.check||sourceOf(result)!==before.result||sourceOf(receipt)!==before.receipt)return false;
   if(!owner?.active||game.users.get(owner.id)!==owner||patient?.uuid!==patientUUID||!actorCurrent(game,patient)||!actorCurrent(game,master)||patient.testUserPermission?.(owner,'OWNER')!==true||master?.uuid!==claim.poolUUID)return false;
   const pool=hpPools.discover(patient);if(!pool.ready||pool.poolUUID!==claim.poolUUID||!pool.memberUUIDs.includes(patientUUID))return false;
   const pf=receipt.flags?.pf2e,tag=`${MODULE_ID}:source:${request.resultId}:${request.rollIndex}`,options=pf?.context?.options??[];
   if(author(receipt)!==owner.id||receipt.speaker?.actor!==patient.id||pf?.context?.type!=='damage-taken'||options.filter(option=>option.startsWith(`${MODULE_ID}:source:`)).length!==1||!options.includes(tag)||pf.appliedDamage?.isReverted)return false;
   const matches=Array.from(game.messages.values()).filter(document=>document.flags?.pf2e?.context?.type==='damage-taken'&&document.speaker?.actor===patient.id&&document.flags.pf2e.context.options?.includes(tag)&&!document.flags.pf2e.appliedDamage?.isReverted);
   if(matches.length!==1||matches[0]!==receipt)return false;
   if(terminal.noChange)return pf.appliedDamage===null&&terminal.master===null;
   const m=terminal.master,b=m?.binding;
   return pf.appliedDamage?.uuid===patientUUID&&pf.appliedDamage.isHealing===true&&m?.terminal==='fulfilled'&&m.poolUUID===claim.poolUUID&&writer?.active===true&&master.testUserPermission?.(writer,'OWNER')===true&&b?.permitNonce===claim.permitNonce&&b.applicationNonce===claim.applicationNonce&&b.ownerUserId===owner.id&&b.patientUUID===patientUUID&&b.poolUUID===claim.poolUUID&&m.fields&&Object.keys(m.fields).length>0;
  };
  return validate(activity)?{poolUUID:claim.poolUUID,receiptId:receipt.id,noChange:terminal.noChange,isCurrent:validate}:null;
 }};
}
