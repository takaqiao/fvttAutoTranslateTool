import {SerialActions} from '../runtime.mjs';
import {metapowerKind,powerProfile,sourceUuid,buildChannelSnapshot} from './rules.mjs';
export const MODULE_ID='pf2e-third-party-automation';
const copy=value=>structuredClone(value);
export const turnIdentity=(game,actor)=>{
 const encounters=game.combats&&actor?Array.from(game.combats.values()).filter(c=>c.started&&Array.from(c.combatants?.values?.()??c.turns??[]).some(t=>t.actor?.uuid===actor.uuid)):null;
 if(encounters?.length>1)throw Error('Actor belongs to multiple started encounters; resolve the encounter before using a metapower.');
 const c=encounters?encounters[0]:game.combat;return c?`${c.id}:${c.round}:${c.turn}:${c.combatant?.id??''}`:null};
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
export function createMetapowerLedger({game,fromUuid,queue=new SerialActions(),validateSelection=async()=>{}}){
 const mutate=(payload,user,fn)=>queue.run(payload.actorUuid,async()=>{
  if(game.user?.id!==game.users.activeGM?.id)throw Error('Only the active GM may mutate metapower state.');
  const actor=await fromUuid(payload.actorUuid);
  if(!actor||!user||!actor.testUserPermission(user,'OWNER'))throw Error('Actor owner permission is required.');
  const state=ledgerState(actor);
  if(state.armed?.turn!==turnIdentity(game,actor))state.armed=null;
  const result=await fn(actor,state);
  // Keep source/channel, uncertain and live proofs indefinitely. Ordinary
  // completed actions have no downstream card consumer; their client high-water
  // marks reject replay after the bounded detail archive has been pruned.
  const ordinary=Object.values(state.receipts).filter(r=>r.clientId&&!r.kind&&!r.powerId&&!r.snapshot&&r.nonce!==state.pending&&(!r.delivery||r.delivery.status==='done')&&['committed','cancelled'].includes(r.status)).sort((a,b)=>b.sequence-a.sequence);
  for(const r of ordinary.slice(64))delete state.receipts[r.nonce];
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
   let clientKey;
   if(payload.clientId){
    if(typeof payload.clientId!=='string'||payload.clientId.length>100||!Number.isSafeInteger(payload.clientSequence)||payload.clientSequence<1)throw Error('Invalid client sequence.');
    clientKey=`${user.id}:${payload.clientId}`;state.clients??={};
    if(payload.clientSequence<=(state.clients[clientKey]??0))throw Error('Archived invocation replay or out-of-order client sequence.');
   }
   if(state.pending)throw Error('Another native action is in progress; finish it before using the next action.');
   if(Object.values(state.receipts).some(r=>r.delivery&&r.delivery.status!=='done'))throw Error('Committed native action follow-up is awaiting GM recovery before the next action.');
   const item=payload.itemUuid?await fromUuid(payload.itemUuid):null;
   if(payload.itemUuid&&(!item||item.actor!==actor||actor.items.get(item.id)!==item))throw Error('Owned embedded item identity is required.');
   const kind=metapowerKind(item),profile=validatePowerAdmission(item,{kind:state.armed?.kind,selection:payload.selection});
   if(profile)await validateSelection({actor,item,selection:payload.selection??{},kind:state.armed?.kind??'normal',user});
   const built=profile?buildChannelSnapshot({kind:state.armed?.kind??'normal',item,selection:payload.selection,policy:{dischargeNonDamage:'remove',dischargeArea:'retain',dischargeRange:'retain',dischargeSaveDowngrade:'retain',highVoltage:'convert'}}):null;
   const snapshot=built?{...built,...(profile.id==='reactive-chain'?{triggerDamage:payload.selection.triggerDamage}:{})}:null;
   const charge=snapshot?.dischargeCost?chargedEffect(actor):null;
   if(snapshot?.dischargeCost&&!charge)throw Error('The selected discharge branch requires Charged.');
   const r={nonce:payload.nonce,sequence:++state.sequence,actorUuid:actor.uuid,itemUuid:item?.uuid??null,sourceUuid:sourceUuid(item),userId:user.id,turn:turnIdentity(game,actor),activationNonce:state.armed?.nonce??null,kind,snapshot,selection:copy(payload.selection??{}),status:'reserved',messageUuid:null};
   r.entry=payload.entry??'item';
   r.powerId=profile?.id??null;
   if(clientKey){r.clientId=payload.clientId;r.clientSequence=payload.clientSequence;state.clients[clientKey]=payload.clientSequence;}
   if(charge)r.charge={itemUuid:charge.uuid,before:charge.system.badge.value,after:charge.system.badge.value-1};
   state.pending=r.nonce;state.receipts[r.nonce]=r;return r;
  }),
  start:(payload,user)=>mutate(payload,user,(actor,state)=>{const r=bound(state,payload,user);if(r.status!=='reserved')throw Error('Invocation has already started or finished.');if(r.turn!==turnIdentity(game,actor))throw Error('Turn changed before native execution; repeat this action on the current turn.');r.status='started';return r}),
  finish:(payload,user)=>mutate(payload,user,async(actor,state)=>{
   const r=bound(state,payload,user);
   if(!['reserved','started'].includes(r.status)){
    if(r.status!==payload.status||r.messageUuid!==(payload.messageUuid??null))throw Error('Original card binding does not match the completed invocation.');return r;
   }
   if(!['committed','cancelled','uncertain'].includes(payload.status))throw Error('Invalid native completion status.');
   if(payload.status==='cancelled'&&r.status==='started'&&!(r.entry==='native-check'&&payload.confirmation==='native-check-no-result'))throw Error('Started native actions need verified cancellation evidence; no refund is assumed.');
   if(payload.messageUuid){
    const m=await fromUuid(payload.messageUuid),proof=m?.flags?.[MODULE_ID]?.metapowerUse;
    const originMatches=m?.flags?.pf2e?.origin?.uuid===r.itemUuid||m?.flags?.pf2e?.context?.type==='self-effect'&&actor.items.get(m.flags.pf2e.context.item)?.uuid===r.itemUuid;
    if(!m||proof?.nonce!==r.nonce||proof.actorUuid!==actor.uuid||proof.itemUuid!==r.itemUuid||m.speaker?.actor!==actor.id||(m.author?.id??m.user?.id??m.user)!==user.id||!originMatches)throw Error('Original native card binding is invalid.');
   }else if(payload.status==='committed'&&r.itemUuid)throw Error('Original native card is required to commit item use.');
   if(payload.status==='committed'&&r.charge){
    const charge=await fromUuid(r.charge.itemUuid);
    if(charge?.flags?.[MODULE_ID]?.payment?.nonce!==r.nonce){
     if(r.paymentStarted)throw Error('Discharge payment completion is uncertain; the GM must reconcile the original card without retrying payment.');
     if(!charge||charge.system.badge.value!==r.charge.before)throw Error('Charged changed during native use; reconcile this original card before continuing.');
     // Native PF2e deletes a counter effect at zero, including its GrantItem
     // children. Persist intent before that deletion so a lost response never
     // permits a second payment or silently assumes an unrelated deletion paid.
     r.paymentStarted=true;r.messageUuid=payload.messageUuid;
     await actor.update({[`flags.${MODULE_ID}.metapower`]:state});
     // Counter and payment proof share one embedded document update. A retry or
     // GM handover sees the proof and cannot charge a second time.
     await charge.update({'system.badge.value':r.charge.after,[`flags.${MODULE_ID}.payment`]:{nonce:r.nonce,after:r.charge.after}});
    }
    r.paymentPaid=true;
   }
   r.status=payload.status;r.messageUuid=payload.messageUuid??r.messageUuid??null;state.pending=null;
   if(r.status==='committed'&&r.messageUuid)r.delivery={status:'pending',attempts:0};
   if(r.status!=='cancelled'){
    if(state.armed?.nonce===r.activationNonce)state.armed=null;
    if(r.status==='committed'&&r.kind&&r.turn===turnIdentity(game,actor))state.armed={nonce:r.nonce,kind:r.kind,itemUuid:r.itemUuid,sourceUuid:r.sourceUuid,sequence:r.sequence,turn:r.turn,messageUuid:r.messageUuid};
   }
   return r;
  }),
  clear:(payload,user)=>mutate(payload,user,(_actor,state)=>{if(!payload.activationNonce||state.armed?.nonce===payload.activationNonce)state.armed=null;return state.armed}),
  reconcile:(payload,user)=>mutate(payload,user,(_actor,state)=>{
   if(user.id!==game.users.activeGM?.id)throw Error('Only the active GM can reconcile an abandoned native invocation.');
   const r=state.receipts[payload.nonce];if(!r||state.pending!==r.nonce||!['reserved','started'].includes(r.status)||payload.confirmation!=='archive-uncertain')throw Error('The original pending invocation and explicit uncertain resolution are required.');
   r.status='uncertain';r.reconciledBy=user.id;state.pending=null;if(state.armed?.nonce===r.activationNonce)state.armed=null;return r;
  }),
  delivery:(payload,user)=>mutate(payload,user,(_actor,state)=>{
   if(user.id!==game.users.activeGM?.id)throw Error('Only the active GM can deliver committed native follow-up.');
   const r=state.receipts[payload.nonce];if(r?.status!=='committed'||!r.messageUuid||!r.delivery)throw Error('Original committed channel delivery is unavailable.');
   if(r.delivery.status==='done')return r;
   if(!['started','pending','done'].includes(payload.status))throw Error('Invalid channel delivery status.');
   r.delivery={...r.delivery,status:payload.status,...(payload.status==='started'?{attempts:r.delivery.attempts+1}:{}),...(payload.confirmation==='gm-manual-effects-settled'?{manuallySettled:true,resolvedBy:user.id}:{}),error:payload.status==='pending'?String(payload.error??'Interrupted native follow-up').slice(0,500):null};return r;
  }),
  expire:(payload,user)=>mutate(payload,user,(_actor,state)=>state.armed),
 };
}
