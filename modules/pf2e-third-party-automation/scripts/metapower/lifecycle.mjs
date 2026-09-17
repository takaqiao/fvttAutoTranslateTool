import {SerialActions} from '../runtime.mjs';
import {metapowerKind,powerProfile,sourceUuid,buildChannelSnapshot} from './rules.mjs';
export const MODULE_ID='pf2e-third-party-automation';
const copy=value=>structuredClone(value);
export const turnIdentity=game=>{const c=game.combat;return c?`${c.id}:${c.round}:${c.turn}:${c.combatant?.id??''}`:null};
export const ledgerState=actor=>copy(actor.flags?.[MODULE_ID]?.metapower??{version:1,sequence:0,armed:null,pending:null,receipts:{}});
export const chargedEffect=actor=>Array.from(actor?.items?.values?.()??actor?.items??[]).find(i=>['Compendium.battlezoo-eldamon-pf2e.conditions.Item.Bi2aHykg6CZrQCnR','Compendium.battlezoo-eldamon-pf2e.conditions.Bi2aHykg6CZrQCnR'].includes(sourceUuid(i))&&i.system.badge?.value>0);

/** Prepared roll options are evaluated by PF2e; raw ChoiceSet selections are not
 * evidence that a suppressed or inactive preparation slot is available. */
export function validatePowerAdmission(item,{kind,selection={}}={}){
 const profile=powerProfile(item);if(!profile)return null;
 const options=item.actor.getRollOptions?.(['all'])??[];
 if(!options.some(o=>/^active-power-(one|two|three|four|reactive(?:-two)?|refresh(?:-two)?):/.test(o)&&o.endsWith(`:${profile.id}`)))throw Error('Power is not currently prepared.');
 if(item.system.frequency&&!(item.system.frequency.value>0))throw Error('Native power frequency is depleted.');
 if(profile.reaction&&selection.triggerConfirmed!==true)throw Error('The actual reaction trigger must be confirmed before using this power.');
 if(profile.id==='reactive-chain'&&(!Number.isFinite(selection.triggerDamage)||selection.triggerDamage<=0||selection.eligibleTargetConfirmed!==true))throw Error('Reactive Chain requires actual triggering electricity damage and confirmed legal target eligibility.');
 if(kind==='siphoning'&&selection.discharge&&profile.id==='reactive-chain')throw Error('Siphoning removes this discharge branch’s only benefit; choose the ordinary branch.');
 return profile;
}

/** Short GM mutation queue. No native invocation or remote dialog runs under it.
 * A durable lease prevents another client from overtaking an unfinished use;
 * ambiguous native completion is archived, never refunded or automatically retried. */
export function createMetapowerLedger({game,fromUuid,queue=new SerialActions()}){
 const mutate=(payload,user,fn)=>queue.run(payload.actorUuid,async()=>{
  if(game.user?.id!==game.users.activeGM?.id)throw Error('Only the active GM may mutate metapower state.');
  const actor=await fromUuid(payload.actorUuid);
  if(!actor||!user||!actor.testUserPermission(user,'OWNER'))throw Error('Actor owner permission is required.');
  const state=ledgerState(actor);
  if(state.armed?.turn!==turnIdentity(game))state.armed=null;
  const result=await fn(actor,state);
  if(game.user?.id!==game.users.activeGM?.id)throw Error('Active GM changed during admission; retry reconciliation.');
  await actor.update({[`flags.${MODULE_ID}.metapower`]:state});return copy(result);
 });
 const bound=(state,payload,user)=>{
  const r=state.receipts[payload.nonce];if(!r||r.userId!==user.id)throw Error('Invocation binding is invalid.');return r;
 };
 return {
  begin:(payload,user)=>mutate(payload,user,async(actor,state)=>{
   if(typeof payload.nonce!=='string'||!payload.nonce||payload.nonce.length>100)throw Error('Invalid invocation nonce.');
   const old=state.receipts[payload.nonce];
   if(old){if(old.userId!==user.id||old.itemUuid!==(payload.itemUuid??null))throw Error('Invocation source binding mismatch.');return old;}
   if(state.pending)throw Error('Another native action is in progress; finish it before using the next action.');
   const item=payload.itemUuid?await fromUuid(payload.itemUuid):null;
   if(payload.itemUuid&&(!item||item.actor!==actor||actor.items.get(item.id)!==item))throw Error('Owned embedded item identity is required.');
   const kind=metapowerKind(item),profile=validatePowerAdmission(item,{kind:state.armed?.kind,selection:payload.selection});
   const built=state.armed&&profile?buildChannelSnapshot({kind:state.armed.kind,item,selection:payload.selection,policy:{dischargeNonDamage:'remove',dischargeRange:'retain',dischargeSaveDowngrade:'retain',highVoltage:'convert'}}):null;
   const snapshot=built?{...built,...(profile.id==='reactive-chain'?{triggerDamage:payload.selection.triggerDamage}:{})}:null;
   const charge=snapshot?.dischargeCost?chargedEffect(actor):null;
   if(snapshot?.dischargeCost&&!charge)throw Error('The selected discharge branch requires Charged.');
   const r={nonce:payload.nonce,sequence:++state.sequence,actorUuid:actor.uuid,itemUuid:item?.uuid??null,sourceUuid:sourceUuid(item),userId:user.id,turn:turnIdentity(game),activationNonce:state.armed?.nonce??null,kind,snapshot,selection:copy(payload.selection??{}),status:'reserved',messageUuid:null};
   r.entry=payload.entry??'item';
   if(charge)r.charge={itemUuid:charge.uuid,before:charge.system.badge.value,after:charge.system.badge.value-1};
   state.pending=r.nonce;state.receipts[r.nonce]=r;return r;
  }),
  start:(payload,user)=>mutate(payload,user,(_actor,state)=>{const r=bound(state,payload,user);if(r.status!=='reserved')throw Error('Invocation has already started or finished.');r.status='started';return r}),
  finish:(payload,user)=>mutate(payload,user,async(actor,state)=>{
   const r=bound(state,payload,user);
   if(!['reserved','started'].includes(r.status)){
    if(r.status!==payload.status||r.messageUuid!==(payload.messageUuid??null))throw Error('Original card binding does not match the completed invocation.');return r;
   }
   if(!['committed','cancelled','uncertain'].includes(payload.status))throw Error('Invalid native completion status.');
   if(payload.status==='cancelled'&&r.status==='started'&&!(r.entry==='native-check'&&payload.confirmation==='native-check-no-result'))throw Error('Started native actions need verified cancellation evidence; no refund is assumed.');
   if(payload.messageUuid){
    const m=await fromUuid(payload.messageUuid),proof=m?.flags?.[MODULE_ID]?.metapowerUse;
    if(!m||proof?.nonce!==r.nonce||proof.actorUuid!==actor.uuid||proof.itemUuid!==r.itemUuid||m.speaker?.actor!==actor.id||(m.author?.id??m.user?.id??m.user)!==user.id||(m.flags?.pf2e?.origin?.uuid??null)!==r.itemUuid)throw Error('Original native card binding is invalid.');
   }else if(payload.status==='committed'&&r.itemUuid)throw Error('Original native card is required to commit item use.');
   if(payload.status==='committed'&&r.charge){
    const charge=await fromUuid(r.charge.itemUuid);
    if(charge?.flags?.[MODULE_ID]?.payment?.nonce!==r.nonce){
     if(!charge||charge.system.badge.value!==r.charge.before)throw Error('Charged changed during native use; reconcile this original card before continuing.');
     // Counter and payment proof share one embedded document update. A retry or
     // GM handover sees the proof and cannot charge a second time.
     await charge.update({'system.badge.value':r.charge.after,[`flags.${MODULE_ID}.payment`]:{nonce:r.nonce,after:r.charge.after}});
    }
   }
   r.status=payload.status;r.messageUuid=payload.messageUuid??null;state.pending=null;
   if(r.status!=='cancelled'){
    if(state.armed?.nonce===r.activationNonce)state.armed=null;
    if(r.status==='committed'&&r.kind&&r.turn===turnIdentity(game))state.armed={nonce:r.nonce,kind:r.kind,itemUuid:r.itemUuid,sourceUuid:r.sourceUuid,sequence:r.sequence,turn:r.turn,messageUuid:r.messageUuid};
   }
   return r;
  }),
  clear:(payload,user)=>mutate(payload,user,(_actor,state)=>{if(!payload.activationNonce||state.armed?.nonce===payload.activationNonce)state.armed=null;return state.armed}),
  expire:(payload,user)=>mutate(payload,user,(_actor,state)=>state.armed),
 };
}
