import {MODULE_ID,clone} from './schema.mjs';
import {canonicalJSON} from './revision-codec.mjs';
import {classifyResult} from './native-treatment.mjs';
import {extensionPatients} from './treatment.mjs';
import {validateSalubriousCard} from '../salubrious-kiss-executor.mjs';
import {currentToken,claimOf,marker} from '../salubrious-kiss-context.mjs';
import {salubriousFeat} from '../salubrious-kiss-rules.mjs';

const resultFields=['status','reason','proof','sourceDegree','effectiveOutcome','rolledHealing','medicBonus','expiresAt','resourceReceiptIds','patientUUID','results','focusBefore','focusAfter','treatment'];
const proofFields=['useId','checkIds','resultIds','receiptIds','immunityIds','poolReceipts'];
export const permitFields=['protocol','rootUUID','epoch','revision','sessionId','activityId','operationId','actorUUID','ownerUserId','ownerClientNonce','attemptNonce','permitNonce','leaseNonce','offerId','requestId','commandDigest'];
export const samePermit=(left,right)=>permitFields.every(key=>left?.[key]===right?.[key]);
export function projectPermit(value){return Object.fromEntries(permitFields.map(key=>[key,value[key]]))}
export async function commandDigest(command){const bytes=new TextEncoder().encode(canonicalJSON(command));return [...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(v=>v.toString(16).padStart(2,'0')).join('')}
const pick=(value,fields)=>Object.fromEntries(fields.filter(key=>Object.hasOwn(value??{},key)&&value[key]!==undefined).map(key=>[key,clone(value[key])]));
export function projectResult(result){
 if(!['confirmed','uncertain','blocked'].includes(result?.status))throw Error('invalid-native-result');
 const saved=pick(result,resultFields.filter(key=>!['proof','results','treatment'].includes(key)));
 if(result.proof)saved.proof=pick(result.proof,proofFields);
 if(result.results)saved.results=result.results.map(projectResult);
 if(result.treatment)saved.treatment=projectResult(result.treatment);
 return saved;
}
export function projectCommand(activity,operationId,original=null){
 if(operationId!==(activity.options?.extensionOf?'treatment-extension':activity.providerId))throw Error('native-operation-mismatch');
 const parameters=pick(activity,['actorUUID','patientUUIDs','hpPoolUUIDs','startedAt','endsAt']);
 parameters.options={};
 if(['treat-wounds','treatment-extension'].includes(operationId)){
  const o=activity.options??{};
  if(!['medicine','nature'].includes(o.skill??'medicine')||!['trained','expert','master','legendary'].includes(o.rank??'trained'))throw Error('invalid-treatment-command');
  parameters.options={skill:o.skill??'medicine',rank:o.rank??'trained'};
  for(const key of ['assurance','riskySurgery','continualRecovery'])if(key in o){if(typeof o[key]!=='boolean')throw Error('invalid-treatment-command');parameters.options[key]=o[key]}
 }else if(operationId==='focus-healing'){
  if(typeof activity.options?.itemUUID!=='string'||activity.patientUUIDs.length!==1)throw Error('invalid-focus-command');parameters.options.itemUUID=activity.options.itemUUID;
 }else if(operationId==='refocus'){
  if(activity.options?.threePecks){parameters.options.threePecks=true;parameters.options.rank=activity.options.rank??'trained'}
 }else throw Error('unknown-native-operation');
 const command={parameters};
 if(operationId==='treatment-extension'){
  if(original?.id!==activity.options.extensionOf||original.state!=='confirmed'||original.actorUUID!==activity.actorUUID||original.endsAt!==activity.startedAt||activity.endsAt!==original.startedAt+3600)throw Error('extension-origin-unconfirmed');
  const results=extensionPatients(original,activity.patientUUIDs);
  if(results.length!==activity.patientUUIDs.length)throw Error('extension-source-unconfirmed');
  parameters.options.extensionOf=original.id;
  command.extensionOriginal={...pick(original,['id','startedAt','endsAt','state','actorUUID']),patientUUIDs:[...activity.patientUUIDs],proof:{useId:original.id,checkIds:[...new Set(results.flatMap(r=>r.proof?.checkIds??[]))],resultIds:[...new Set(results.flatMap(r=>r.proof?.resultIds??[]))]},results:results.map(result=>({...pick(result,['patientUUID','effectiveOutcome','rolledHealing','medicBonus']),proof:pick(result.proof,['useId','checkIds','resultIds'])}))};
 }
 return command;
}
export function nativeParameters(command,permit){
 const p=clone(command.parameters);
 if(p?.actorUUID!==permit.actorUUID||!Number.isFinite(p.startedAt)||!Number.isFinite(p.endsAt)||p.endsAt<p.startedAt||['patientUUIDs','hpPoolUUIDs'].some(key=>!Array.isArray(p[key])||!p[key].length&&!(permit.operationId==='refocus'&&!p.options?.threePecks)||new Set(p[key]).size!==p[key].length||p[key].some(id=>typeof id!=='string'||!id.startsWith('Actor.')&&!id.startsWith('Scene.'))))throw Error('invalid-native-command');
 const activity={id:permit.activityId,sessionId:permit.sessionId,providerId:permit.operationId==='treatment-extension'?'treat-wounds':permit.operationId,state:'completing',source:{type:'coordinator'},...p};
 if(canonicalJSON(projectCommand(activity,permit.operationId,command.extensionOriginal))!==canonicalJSON(command))throw Error('invalid-native-command');
 return activity;
}

export function validateExtensionOriginal({game,original,actorUUID,patientUUIDs}){
 if(original?.state!=='confirmed'||original.actorUUID!==actorUUID||original.proof?.useId!==original.id)throw Error('extension-source-unconfirmed');
 const allowed=new Set(patientUUIDs),results=extensionPatients(original,patientUUIDs);
 if(results.length!==allowed.size||new Set(results.map(r=>r.patientUUID??original.patientUUIDs[0])).size!==allowed.size)throw Error('extension-patient-mismatch');
 for(const result of results){
  const patientUUID=result.patientUUID??original.patientUUIDs[0],checks=result.proof?.checkIds?.map(id=>game.messages.get(id)),cards=result.proof?.resultIds?.map(id=>game.messages.get(id));
  if(!checks?.length||!cards?.length||!Number.isFinite(result.rolledHealing)||!Number.isFinite(result.medicBonus??0))throw Error('extension-source-unconfirmed');
  const valid=message=>message&&message.speaker?.actor===actorUUID.split('.').at(-1)&&game.users.get(author(message))?.active&&message.flags?.[MODULE_ID]?.exploration?.activityId===original.id&&message.flags[MODULE_ID].exploration.patientUUID===patientUUID&&message.flags?.pf2e?.context?.options?.includes(`exploration-activity:${original.id}`);
  if(checks.some(m=>!valid(m)||m.flags.pf2e.context.outcome!==result.effectiveOutcome))throw Error('extension-check-unconfirmed');
  const healing=cards.filter(m=>valid(m)&&checks.some(c=>c.id===m.flags.pf2e.origin?.messageId&&author(c)===author(m))&&classifyResult(m.rolls?.[0],result.effectiveOutcome)==='healing');
  if(healing.length!==1||healing[0].rolls[0].total!==result.rolledHealing)throw Error('original-healing-roll-unconfirmed');
 }
 return true;
}

const author=message=>message.author?.id??message.author??message.user?.id??message.user;
/** Re-read persisted native sources. A saved owner flag alone cannot confirm HP. */
export async function validateNativeResult({game,fromUuid,activity,permit,result,extensionOriginal}){
 if(result?.status!=='confirmed')return projectResult(result);
 const actor=await fromUuid(activity.actorUUID),user=game.users.get(permit.ownerUserId),proof=result.proof;
 if(!actor||!user||!actor.testUserPermission(user,'OWNER')||!proof||proof.useId!==(activity.options?.extensionOf??activity.id))throw Error('saved-owner-proof-mismatch');
 for(const key of ['checkIds','resultIds','receiptIds','immunityIds'])if(!Array.isArray(proof[key])||new Set(proof[key]).size!==proof[key].length)throw Error('invalid-native-proof');
 if(permit.operationId==='refocus'){
  const intent=actor.flags?.[MODULE_ID]?.avRefocusIntent;
  if(intent?.nonce!==activity.id||intent.actorUuid!==actor.uuid||intent.userId!==permit.ownerUserId||intent.startedAt!==activity.startedAt||intent.before!==result.focusBefore||intent.after!==result.focusAfter)throw Error('saved-refocus-intent-unavailable');
  if(proof.receiptIds[0]!==activity.id||!Number.isFinite(result.focusBefore)||!Number.isFinite(result.focusAfter)||result.focusAfter<result.focusBefore||result.focusAfter>actor.system?.resources?.focus?.max)throw Error('saved-refocus-proof-mismatch');
  if(!activity.options?.threePecks){if(proof.checkIds.length||proof.resultIds.length||proof.receiptIds.length!==1||result.focusAfter!==Math.min(actor.system.resources.focus.max,result.focusBefore+1))throw Error('saved-refocus-proof-mismatch');return projectResult(result)}
  const treatment=result.treatment,claim=claimOf(actor,activity.id),token=claim&&await fromUuid(claim.tokenUuid),target=claim&&await fromUuid(claim.targetUuid);
  if(!treatment||claim?.state!=='done'||claim.actorUuid!==actor.uuid||claim.userId!==permit.ownerUserId||claim.startedAt!==activity.startedAt||activity.patientUUIDs.length!==1||claim.targetActorUuid!==activity.patientUUIDs[0]||salubriousFeat(actor)?.uuid!==claim.itemUuid||!currentToken(token,game)||token.actor!==actor||!currentToken(target,game)||target.actor.uuid!==claim.targetActorUuid)throw Error('saved-three-pecks-proof-unavailable');
  const check=game.messages.get(claim.result?.checkId),degree=validateSalubriousCard({game,message:check,claim}),damage=claim.result.damageId?game.messages.get(claim.result.damageId):null;
  if(degree!==claim.result.degree||treatment.sourceDegree!==degree||treatment.effectiveOutcome!==['criticalFailure','failure','success','criticalSuccess'][degree]||canonicalJSON(proof.checkIds)!==canonicalJSON([check.id])||canonicalJSON(proof.resultIds)!==canonicalJSON(damage?[damage.id]:[])||degree===1&&damage||degree!==1&&!damage)throw Error('saved-three-pecks-result-mismatch');
  if(damage){validateSalubriousCard({game,message:damage,claim,damage:true});if(damage.flags.pf2e.origin.messageId!==check.id)throw Error('saved-three-pecks-result-mismatch')}
  const receipts=proof.receiptIds.slice(1);if(receipts.length!==(damage?1:0)||damage&&receipts[0]!==claim.receipt?.messageId)throw Error('saved-three-pecks-receipt-mismatch');
  if(damage){const receipt=game.messages.get(receipts[0]),context=receipt?.flags?.pf2e?.context,applied=receipt?.flags?.pf2e?.appliedDamage;
   if(!receipt||author(receipt)!==permit.ownerUserId||receipt.speaker?.actor!==target.actor.id||`Scene.${receipt.speaker?.scene}.Token.${receipt.speaker?.token}`!==target.uuid||context?.type!=='damage-taken'||!context.options?.includes(marker('apply',claim))||!context.options.includes(`${MODULE_ID}:source:${damage.id}:0`)||applied&&(applied.uuid!==target.actor.uuid||applied.isReverted))throw Error('saved-three-pecks-application-unavailable')}
  if(canonicalJSON(proof.immunityIds)!==canonicalJSON(claim.immunityIds??[]))throw Error('saved-three-pecks-immunity-mismatch');
  for(const uuid of proof.immunityIds){const item=await fromUuid(uuid);if(item?.actor?.uuid!==target.actor.uuid||item.flags?.[MODULE_ID]?.salubriousKiss?.nonce!==activity.id||item.flags[MODULE_ID].salubriousKiss.kind!=='immunity')throw Error('saved-three-pecks-immunity-unavailable')}
  if((proof.poolReceipts??[]).some(p=>!activity.hpPoolUUIDs.includes(p.actorUUID)||p.activityId!==undefined&&p.activityId!==activity.id||p.noChange&&!receipts.includes(p.receiptId)))throw Error('saved-pool-proof-mismatch');
  return projectResult(result);
 }
 const sourceId=activity.options?.extensionOf??activity.id;
 const patients=new Map();for(const uuid of activity.patientUUIDs){const patient=await fromUuid(uuid);if(!patient||!patient.testUserPermission(user,'OWNER'))throw Error('saved-native-patient-unavailable');patients.set(uuid,patient)}
 const checks=proof.checkIds.map(id=>game.messages.get(id)),results=proof.resultIds.map(id=>game.messages.get(id));
 const sourced=message=>message&&(permit.operationId==='treatment-extension'?game.users.get(author(message))?.active:author(message)===permit.ownerUserId)&&message.speaker?.actor===actor.id&&message.flags?.[MODULE_ID]?.exploration?.activityId===sourceId&&patients.has(message.flags[MODULE_ID].exploration.patientUUID)&&message.flags?.pf2e?.context?.options?.includes(`exploration-activity:${sourceId}`);
 if(permit.operationId==='focus-healing'){
  const commit=actor.flags?.[MODULE_ID]?.explorationFocusCommits?.[activity.id],card=results[0],damage=results[1],item=await fromUuid(activity.options.itemUUID);
  const paid=actor.flags?.[MODULE_ID]?.nativeCasts?.find(r=>r.id===commit?.castNonce),cast=card?.flags?.[MODULE_ID]?.nativeCast;
  if(checks.length||results.length!==2||item?.actor?.uuid!==actor.uuid||actor.items?.get(item.id)!==item||!commit||commit.activityId!==activity.id||commit.itemUuid!==item.uuid||commit.cost!==1||commit.after!==commit.before-1||!paid||paid.state!=='used'||paid.messageId!==card?.id||paid.userId!==permit.ownerUserId||paid.actorUuid!==actor.uuid||paid.itemUuid!==item.uuid||paid.focusPoints!==1||cast?.id!==commit.castNonce||cast.actorUuid!==actor.uuid||cast.itemUuid!==item.uuid||cast.userId!==permit.ownerUserId||card.flags[MODULE_ID].explorationFocus?.activityId!==activity.id||author(card)!==permit.ownerUserId||card.speaker?.actor!==actor.id||!sourced(damage)||damage.flags.pf2e.origin?.messageId!==card.id||damage.flags.pf2e.origin?.uuid!==item.uuid||result.resourceReceiptIds?.length!==1||result.resourceReceiptIds[0]!==commit.castNonce)throw Error('saved-focus-payment-unconfirmed');
 }else{
  if(checks.length!==activity.patientUUIDs.length||checks.some(message=>!sourced(message)||!['criticalFailure','failure','success','criticalSuccess'].includes(message.flags.pf2e.context.outcome)))throw Error('saved-native-check-unavailable');
  if(new Set(checks.map(m=>m.flags[MODULE_ID].exploration.patientUUID)).size!==patients.size)throw Error('saved-native-patient-mismatch');
  for(const message of results)if(!sourced(message)||!checks.some(check=>check.id===message.flags.pf2e.origin?.messageId&&check.flags[MODULE_ID].exploration.patientUUID===message.flags[MODULE_ID].exploration.patientUUID))throw Error('saved-native-result-unavailable');
  const summaries=result.results??[result];
  if(permit.operationId==='treat-wounds')for(const check of checks){
   const stages=results.filter(m=>m.flags.pf2e.origin.messageId===check.id).map(m=>classifyResult(m.rolls?.[0],check.flags.pf2e.context.outcome));
   const risky=check.flags.pf2e.modifiers?.some(m=>m.slug==='risky-surgery'&&m.enabled)||activity.options?.riskySurgery;
   const expected=[...risky?['surgery']:[],...check.flags.pf2e.context.outcome==='failure'?[]:[check.flags.pf2e.context.outcome==='criticalFailure'?'failure-damage':'healing']];
   if(canonicalJSON([...stages].sort())!==canonicalJSON(expected.sort()))throw Error('saved-native-stages-mismatch');
   const summary=summaries.find(r=>(r.patientUUID??activity.patientUUIDs[0])===check.flags[MODULE_ID].exploration.patientUUID);
   if(summary?.effectiveOutcome!==check.flags.pf2e.context.outcome)throw Error('saved-native-outcome-mismatch');
  }
  if(permit.operationId==='treatment-extension')validateExtensionOriginal({game,original:extensionOriginal,actorUUID:activity.actorUUID,patientUUIDs:activity.patientUUIDs});
 }
 const receipts=proof.receiptIds.filter(id=>!(permit.operationId==='refocus'&&id===activity.id)).map(id=>game.messages.get(id));
 const expected=permit.operationId==='treatment-extension'?results.filter(m=>/\[healing\]/.test(m.rolls?.[0]?.toJSON?.().formula??'')):permit.operationId==='focus-healing'?results.slice(1):results;
 if(receipts.length!==expected.length)throw Error('saved-native-receipt-count-mismatch');
 for(const message of expected){
  const patientUUID=message.flags[MODULE_ID].exploration.patientUUID,patient=patients.get(patientUUID),application=`${MODULE_ID}:exploration-apply:${activity.id}:${message.id}:${patientUUID}`;
  const matches=receipts.filter(receipt=>receipt&&author(receipt)===permit.ownerUserId&&receipt.speaker?.actor===patient.id&&receipt.flags?.pf2e?.context?.type==='damage-taken'&&receipt.flags.pf2e.context.options?.includes(application)&&receipt.flags.pf2e.context.options.includes(`${MODULE_ID}:source:${message.id}:0`)&&(!receipt.flags.pf2e.appliedDamage||receipt.flags.pf2e.appliedDamage.uuid===patientUUID&&!receipt.flags.pf2e.appliedDamage.isReverted));
  if(matches.length!==1)throw Error('saved-native-application-unavailable');
 }
 for(const uuid of proof.immunityIds){const item=await fromUuid(uuid);if(!patients.has(item?.actor?.uuid??item?.parent?.uuid)||item.flags?.[MODULE_ID]?.exploration?.activityId!==activity.id||item.flags[MODULE_ID].exploration.kind!=='immunity')throw Error('saved-native-immunity-unavailable')}
 if((proof.poolReceipts??[]).some(p=>p.activityId!==activity.id||!activity.hpPoolUUIDs.includes(p.actorUUID)||p.noChange&&!proof.receiptIds.includes(p.receiptId)||p.patientUUID&&!patients.has(p.patientUUID)))throw Error('saved-pool-proof-mismatch');
 return projectResult(result);
}
