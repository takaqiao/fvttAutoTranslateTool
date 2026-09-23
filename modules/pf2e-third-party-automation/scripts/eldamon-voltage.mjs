import {SerialActions} from './runtime.mjs';
import {sourceUuid} from './metapower/rules.mjs';

export const VOLTAGE_MODULE_ID='pf2e-third-party-automation';
export const HIGH_VOLTAGE_SOURCE='Compendium.battlezoo-eldamon-pf2e.powers.Item.9bElF2uVf5FCJtb9';
export const ELEMENTAL_POWERS_SOURCE='Compendium.battlezoo-eldamon-pf2e.eldamon-features.Item.naawsnBug9EOpzfN';
const ID=VOLTAGE_MODULE_ID,copy=value=>structuredClone(value),values=c=>Array.from(c?.values?.()??c??[]);
export const VOLTAGE_APPLY_PREFIX=`${ID}:voltage-apply:`;
export const voltageRollOption=a=>`${ID}:voltage:${a.messageUuid.split('.').at(-1)}:${a.nonce}`;
const author=m=>m?.author?.id??m?.user?.id??m?.user;
const outcomes=new Set(['criticalSuccess','success','failure','criticalFailure']);
const evidence=m=>JSON.stringify({author:author(m),speaker:m.speaker,pf:m.flags?.pf2e,proof:m.flags?.[ID]?.voltageDamage,rolls:m.rolls?.map(r=>r.toJSON?.()??{formula:r.formula,total:r.total,evaluated:r._evaluated,options:r.options})});
export function voltageReceiptMatches(message,activation){
 const application=activation.native?.application,pf=message?.flags?.pf2e,options=pf?.context?.options??[];
 return !!application&&author(message)===application.userId&&message?.speaker?.actor===activation.trigger.targetActorUuid.split('.').at(-1)&&tokenUuid(message.speaker)===activation.trigger.targetUuid&&
  pf?.context?.type==='damage-taken'&&options.filter(o=>typeof o==='string'&&o.startsWith(VOLTAGE_APPLY_PREFIX)).length===1&&options.includes(VOLTAGE_APPLY_PREFIX+application.id)&&
  options.includes(voltageRollOption(activation))&&pf.origin?.uuid===activation.itemUuid&&(!pf.appliedDamage||pf.appliedDamage.uuid===activation.trigger.targetActorUuid&&!pf.appliedDamage.isHealing&&!pf.appliedDamage.isReverted);
}
export const voltageState=actor=>copy(actor?.flags?.[ID]?.voltage??{version:1,activeNonce:null,activations:{},refreshes:{}});
const prepared=(actor,slug)=>actor.getRollOptions?.(['all']).some(o=>/^active-power-(one|two|three|four|reactive(?:-two)?|refresh(?:-two)?):/.test(o)&&o.endsWith(':'+slug));
const ownedItem=(actor,item)=>item?.actor===actor&&actor.items.get(item.id)===item;
const tokenUuid=s=>s?.scene&&s?.token?`Scene.${s.scene}.Token.${s.token}`:null;
export const currentVoltageToken=t=>!!t?.actor&&t.parent?.tokens?.get(t.id)===t;

/** Book p.70: refresh active/reactive powers, never preparation options or other
 * resources. The package represents these as one-use PT10M frequencies. */
export function voltageRefreshUpdates(actor){
 return values(actor.items).flatMap(item=>{
  const source=sourceUuid(item),frequency=item.frequency??item.system?.frequency;
  if(!ownedItem(actor,item)||!source?.startsWith('Compendium.battlezoo-eldamon-pf2e.powers.Item.')||source===HIGH_VOLTAGE_SOURCE||
   !prepared(actor,item.system.slug)||item.system.traits?.value?.includes('refresh')||frequency?.per!=='PT10M'||
   !Number.isInteger(frequency.max)||frequency.max<1||!Number.isInteger(frequency.value)||frequency.value<0||frequency.value>=frequency.max)return [];
  return [{itemUuid:item.uuid,before:frequency.value,after:frequency.max}];
 });
}
export function voltageAttackEligibility({outcome,melee,adjacent,unarmed,metal}){
 return ['success','criticalSuccess'].includes(outcome)&&melee===true&&(adjacent===true||unarmed===true||metal===true);
}
function timing(game,actor,origin){
 const candidates=(game.combats?values(game.combats):[game.combat]).filter(c=>c?.started&&c.turns?.some(t=>t.actor?.uuid===actor.uuid&&(!t.token||t.token.uuid===origin?.uuid)));
 if(candidates.length!==1)return null;
 const c=candidates[0],index=c.turns.findIndex(t=>t.actor?.uuid===actor.uuid&&(!t.token||t.token.uuid===origin?.uuid));
 if(!c?.started||index<0||!Number.isInteger(c.round)||!Number.isInteger(c.turn))return null;
 return {combatId:c.id,combatantId:c.turns[index].id,round:c.round+(index<=c.turn?1:0)};
}
function expired(game,activation){
 const t=activation.expires,c=t&&(game.combats?.get?game.combats.get(t.combatId):game.combat);if(!t||c?.id!==t.combatId||!c.started)return true;
 const index=c.turns.findIndex(x=>x.id===t.combatantId);
 return index<0||c.round>t.round||(c.round===t.round&&c.turn>=index);
}
const knownMetals=new Set(['adamantine','cold-iron','silver','dawnsilver','orichalcum','mithral','steel','iron']);

/** Short active-GM transactions only. Native saves/damage run AFTER claim resolves
 * so metapower settlement and native action observers cannot reenter this queue. */
export function createVoltageLedger({game,fromUuid,onRefresh=async()=>{},queue=new SerialActions()}){
 const gm=()=>{if(game.user?.id!==game.users.activeGM?.id)throw Error('Only the active GM may coordinate High Voltage.');};
 const owner=(actor,user)=>{if(!user||game.users.get(user.id)!==user||!actor?.testUserPermission(user,'OWNER'))throw Error('Actor owner permission is required.');};
 const save=async(actor,state)=>{gm();await actor.update({[`flags.${ID}.voltage`]:state});};
 const mutate=(payload,user,fn)=>queue.run(payload.actorUuid,async()=>{gm();const actor=await fromUuid(payload.actorUuid);owner(actor,user);return fn(actor,voltageState(actor));});
 async function original(actor,payload,user,source=HIGH_VOLTAGE_SOURCE){
  if(!user||game.users.get(user.id)!==user)throw Error('Original High Voltage card author is unavailable.');
  const receipt=actor.flags?.[ID]?.metapower?.receipts?.[payload.nonce],message=await fromUuid(payload.messageUuid);
  if(!receipt||receipt.status!=='committed'||receipt.actorUuid!==actor.uuid||receipt.messageUuid!==message?.uuid||message?.uuid!==payload.messageUuid||game.messages.get(message?.id)!==message||
   receipt.userId!==user.id||receipt.sourceUuid!==source||message.flags?.[ID]?.metapowerUse?.nonce!==receipt.nonce||message.flags[ID].metapowerUse.actorUuid!==actor.uuid||
   message.flags[ID].metapowerUse.itemUuid!==receipt.itemUuid||message.flags?.pf2e?.origin?.uuid!==receipt.itemUuid||message.speaker?.actor!==actor.id||(message.author?.id??message.user?.id??message.user)!==user.id)throw Error('Original High Voltage card binding is invalid.');
  const item=await fromUuid(receipt.itemUuid);if(!ownedItem(actor,item)||sourceUuid(item)!==source)throw Error('Original owned source is missing.');
  return {receipt,message,item};
 }
 async function refresh(actor,state,nonce){
  let r=state.refreshes[nonce];
  if(!r){r=state.refreshes[nonce]={status:'started',updates:voltageRefreshUpdates(actor)};await save(actor,state);}
  if(r.status!=='done'){
  for(const update of r.updates){
   const item=await fromUuid(update.itemUuid);if(!ownedItem(actor,item))throw Error('Refresh source item changed; reconcile the original activity.');
   if(item.flags?.[ID]?.voltageRefresh?.nonce===nonce)continue;
   if((item.frequency??item.system.frequency)?.value!==update.before)throw Error('Power frequency changed during Refresh; reconcile the original activity.');
   gm();await item.update({'system.frequency.value':update.after,[`flags.${ID}.voltageRefresh`]:{nonce,after:update.after}});
  }
   r.status='done';await save(actor,state);
  }
  // Lifecycle cleanup is independently idempotent under this nonce. A failed
  // cleanup may be retried, but already committed resource writes never repeat.
  if(!r.effectsDone){await onRefresh({actor,nonce});r.effectsDone=true;await save(actor,state);}
  return r;
 }
 async function closeIfStale(actor,state,activation){
  if(activation?.status!=='armed')return false;
  const item=await fromUuid(activation.itemUuid),origin=await fromUuid(activation.originUuid);
  if(expired(game,activation)||!ownedItem(actor,item)||sourceUuid(item)!==HIGH_VOLTAGE_SOURCE||!currentVoltageToken(origin)||origin.actor!==actor){
   activation.status='expired';if(state.activeNonce===activation.nonce)state.activeNonce=null;await save(actor,state);return true;
  }
  return false;
 }
 async function attackTarget(actor,state,activation,payload){
  const message=await fromUuid(payload.attackUuid),c=message?.flags?.pf2e?.context;
  if(game.messages.get(message?.id)!==message||message?.isCheckRoll!==true||message.rolls?.[0]?._evaluated!==true||c?.type!=='attack-roll'||message.timestamp<activation.channelTimestamp||
   c.target?.actor!==activation.actorUuid||c.target?.token!==activation.originUuid)return null;
  const target=await fromUuid(tokenUuid(message.speaker)),origin=await fromUuid(activation.originUuid),item=message.item;
  if(!currentVoltageToken(target)||target.parent!==origin?.parent||target.actor.uuid===activation.actorUuid||message.speaker.actor!==target.actor.id||item?.actor?.uuid!==target.actor.uuid)return null;
  const options=c.options??[],tagPrefix=`${ID}:voltage-attack:${activation.nonce}:`,tag=options.find(o=>o.startsWith(tagPrefix));
  activation.attacks??={};
  const confirmedKeptReroll=c.isReroll&&!tag&&payload.nativeInteraction===true&&payload.confirmed===true&&options.includes('check:reroll');
  if(c.isReroll&&!confirmedKeptReroll){
   const prior=activation.attacks[tag?.slice(tagPrefix.length)];
   // PF2e deletes the original card before publishing the kept reroll, copying
   // context options. Previously recorded history survives deletion and reload.
   if(!prior||!options.includes('check:reroll')||game.messages.get(prior.latestMessageId)&&prior.latestMessageId!==message.id||prior.actorUuid!==target.actor.uuid||prior.targetUuid!==activation.originUuid||prior.itemUuid!==(item.uuid??null))return null;
   prior.latestMessageId=message.id;prior.outcome=c.outcome;
  }else{
   // A new owner confirmation may first see PF2e's kept native reroll. Its
   // current evaluated card still proves source, outcome and target above.
   if(tag&&tag!==tagPrefix+message.id)return null;
   activation.attacks[message.id]={latestMessageId:message.id,actorUuid:target.actor.uuid,targetUuid:activation.originUuid,itemUuid:item.uuid??null,outcome:c.outcome};
   if(!tag)await message.update?.({'flags.pf2e.context.options':[...options,tagPrefix+message.id]});
  }
  await save(actor,state);
  const melee=options.includes('item:melee')||options.includes('melee')||item.isMelee===true;
  const adjacent=origin.object?.distanceTo?.(target.object)<=5;
  const unarmed=item.system?.category==='unarmed'||item.category==='unarmed'||item.system?.traits?.value?.includes('unarmed');
  const metal=knownMetals.has(item.system?.material?.type)||payload.kind==='metal-hit'&&payload.confirmed===true;
  if(!voltageAttackEligibility({outcome:c.outcome,melee,adjacent,unarmed,metal}))return null;
  return target;
 }
 async function nativeContext(actor,a,payload){
  if(!a||a.messageUuid!==payload.messageUuid||a.status!=='claimed')throw Error('High Voltage claim is already consumed or not awaiting a native result.');
  if(a.native?.version!==1)throw Error('Legacy High Voltage claim requires GM reconciliation; it cannot replay.');
  const originalUser=game.users.get(a.userId),{item}=await original(actor,payload,originalUser);
  const [origin,target]=await Promise.all([fromUuid(a.originUuid),fromUuid(a.trigger?.targetUuid)]);
  if(!currentVoltageToken(origin)||origin.actor!==actor||!currentVoltageToken(target)||target.parent!==origin.parent||target.actor.uuid!==a.trigger.targetActorUuid)throw Error('High Voltage source or bound target changed.');
  return {item,origin,target};
 }
 const nativeMutation=(payload,user,fn)=>queue.run(payload.actorUuid,async()=>{
  gm();if(!user||game.users.get(user.id)!==user)throw Error('Current user permission is required.');
  const actor=await fromUuid(payload.actorUuid),state=voltageState(actor),a=state.activations[payload.nonce],ctx=await nativeContext(actor,a,payload);
  await fn(actor,a,ctx);await save(actor,state);return copy(a);
 });
 function phase(a,wanted){if(a.native.phase!==wanted)throw Error('High Voltage native step is already consumed or not awaiting this result.');}
 function operation(payload){if(typeof payload.operationId!=='string'||! /^[A-Za-z0-9_-]{1,80}$/.test(payload.operationId))throw Error('A bounded native operation ID is required.');return payload.operationId;}
 function sameOperation(record,payload,user){if(record?.id!==payload.operationId||record.userId!==user.id)throw Error('High Voltage native operation binding is invalid.');}
 function saveMatches(m,a,target,user){
  const c=m?.flags?.pf2e?.context;
  return !!m&&game.messages.get(m.id)===m&&m.isCheckRoll&&m.rolls?.[0]?._evaluated===true&&!!user&&author(m)===user.id&&m.speaker?.actor===target.actor.id&&tokenUuid(m.speaker)===target.uuid&&
   m.flags?.pf2e?.origin?.uuid===a.itemUuid&&c?.type==='saving-throw'&&outcomes.has(c.outcome)&&c.dc?.value===a.dc&&c.options?.includes(voltageRollOption(a));
 }
 async function boundSave(a,{target,allowReroll=false}={}){
  const m=await fromUuid(a.native.save?.messageUuid);
  if(m&&game.messages.get(m.id)===m){if(evidence(m)!==a.native.save.evidence)throw Error('High Voltage native save source changed.');return m}
  if(allowReroll){
   // Only the explicit damage click may adopt a kept reroll before publication.
   // Editing a live original, ambiguous copies or later rerolls never reprice
   // a published damage card or authorize another application.
   const kept=values(game.messages).filter(message=>{const c=message.flags?.pf2e?.context,user=game.users.get(author(message));return c?.isReroll&&c.options?.includes('check:reroll')&&saveMatches(message,a,target,user)&&target.actor.testUserPermission(user,'OWNER')});
   if(kept.length===1){Object.assign(a.native.save,{priorMessageUuid:a.native.save.messageUuid,messageUuid:kept[0].uuid,userId:author(kept[0]),outcome:kept[0].flags.pf2e.context.outcome,evidence:evidence(kept[0])});return kept[0]}
  }
  throw Error('High Voltage native save source changed; reconcile its kept native result.');
 }
 async function boundDamage(a){
  await boundSave(a);const m=await fromUuid(a.native.damage?.messageUuid);
  if(!m||game.messages.get(m.id)!==m||evidence(m)!==a.native.damage.evidence)throw Error('High Voltage native damage source changed.');return m;
 }
 return {
  refreshOutsideEncounter:(payload,user)=>mutate(payload,user,async(actor,state)=>{
   if(user!==game.users.activeGM)throw Error('Automatic Refresh requires the active GM.');
   if(!values(actor.items).some(item=>sourceUuid(item)===ELEMENTAL_POWERS_SOURCE))throw Error('Elemental Powers is required for automatic Refresh.');
   const encounters=new Set([...values(game.combats),game.combat].filter(Boolean));
   if([...encounters].some(combat=>combat.started&&values(combat.combatants??combat.turns).some(c=>c.actor?.uuid===actor.uuid)))throw Error('Automatic outside-encounter Refresh cannot run during an active encounter.');
   if(typeof payload.nonce!=='string'||!payload.nonce||payload.nonce.length>160)throw Error('Automatic Refresh requires a bounded event nonce.');
   return copy(await refresh(actor,state,`outside:${payload.nonce}`));
  }),
  channel:(payload,user)=>mutate(payload,user,async(actor,state)=>{
   const {receipt,message,item}=await original(actor,payload,user),old=state.activations[receipt.nonce];
   if(old){if(old.messageUuid!==message.uuid)throw Error('Original activation binding mismatch.');if(!old.refreshSuppressed)await refresh(actor,state,receipt.nonce);return copy(old);}
   if(!prepared(actor,item.system.slug))throw Error('High Voltage is not currently prepared.');
   const origin=await fromUuid(tokenUuid(message.speaker));if(!currentVoltageToken(origin)||origin.actor!==actor)throw Error('Original source token is required for High Voltage.');
   const level=receipt.snapshot?.level??actor.level;if(!Number.isInteger(level)||level<1)throw Error('Invalid High Voltage level.');
   if(state.activeNonce&&state.activations[state.activeNonce]?.status==='armed')state.activations[state.activeNonce].status='replaced';
   const snapshot=copy(receipt.snapshot??null),refreshSuppressed=snapshot?.kind==='siphoning'&&snapshot?.siphon?.applies===true;
   if(snapshot&&(snapshot.itemUuid!==item.uuid||snapshot.actorUuid!==actor.uuid||snapshot.powerSourceUuid!==HIGH_VOLTAGE_SOURCE))throw Error('Channel snapshot source binding is invalid.');
   const activation={nonce:receipt.nonce,actorUuid:actor.uuid,itemUuid:item.uuid,sourceUuid:HIGH_VOLTAGE_SOURCE,messageUuid:message.uuid,originUuid:origin.uuid,userId:user.id,
    speaker:copy(message.speaker),channelTimestamp:message.timestamp,level,dc:actor.getStatistic?.('eldamon')?.dc?.value??null,traits:[...(item.system.traits?.value??[])],snapshot,refreshSuppressed,expires:timing(game,actor,origin),status:'armed'};
   if(!activation.expires){activation.status='expired';activation.expiryReason='no-encounter';}
   state.activations[receipt.nonce]=activation;state.activeNonce=activation.status==='armed'?receipt.nonce:null;await save(actor,state);
   if(!refreshSuppressed)await refresh(actor,state,receipt.nonce);return copy(activation);
  }),
  refreshActivity:(payload,user)=>mutate(payload,user,async(actor,state)=>{
   const {message}=await original(actor,payload,user,ELEMENTAL_POWERS_SOURCE);
   if(message.flags?.[ID]?.voltageRefreshActivity?.actions!==2)throw Error('The original two-action Refresh activity is required.');
   return copy(await refresh(actor,state,payload.nonce));
  }),
  claim:(payload,user)=>mutate(payload,user,async(actor,state)=>{
   const a=state.activations[payload.nonce];if(!a||a.messageUuid!==payload.messageUuid)throw Error('Original activation card binding is invalid.');
   if(a.status!=='armed'||await closeIfStale(actor,state,a))return null;
   await original(actor,payload,game.users.get(a.userId));
   let target;
   if(['attack','metal-hit'].includes(payload.kind)){
    if(payload.kind==='attack'&&payload.confirmed!==true&&user!==game.users.activeGM)throw Error('Confirm the native hit on the original High Voltage card.');
    if(payload.kind==='metal-hit'&&payload.confirmed!==true)throw Error('Confirm that the native melee hit used a metal weapon.');
    target=await attackTarget(actor,state,a,payload);if(!target)return null;
   }
   else if(payload.kind==='touch'){
    if(payload.confirmed!==true)throw Error('Confirm that the other creature actually touched the source.');
    target=await fromUuid(payload.targetUuid);
    const origin=await fromUuid(a.originUuid);
    if(!currentVoltageToken(target)||target.parent!==origin?.parent||target.actor.uuid===actor.uuid)throw Error('A different current creature in the source scene is required.');
   }else throw Error('Unsupported High Voltage trigger.');
   if(await closeIfStale(actor,state,a))return null;
   a.status='claimed';a.trigger={kind:payload.kind,attackUuid:payload.attackUuid??null,targetUuid:target.uuid,targetActorUuid:target.actor.uuid,userId:user.id};state.activeNonce=null;
   if(payload.nativeInteraction===true)a.native={version:1,phase:'awaiting-save'};
   await save(actor,state);return copy(a);
  }),
  beginSave:(payload,user)=>nativeMutation(payload,user,async(_actor,a,{target})=>{
   owner(target.actor,user);phase(a,'awaiting-save');if(!Number.isFinite(a.dc))throw Error('The original Eldamon power DC is unavailable.');
   a.native.phase='rolling-save';a.native.save={id:operation(payload),userId:user.id};
  }),
  finishSave:(payload,user)=>nativeMutation(payload,user,async(_actor,a,{target})=>{
   owner(target.actor,user);phase(a,'rolling-save');sameOperation(a.native.save,payload,user);
   const m=await fromUuid(payload.saveUuid),c=m?.flags?.pf2e?.context;
   if(!saveMatches(m,a,target,user))throw Error('Original native Reflex result binding is invalid.');
   Object.assign(a.native.save,{messageUuid:m.uuid,outcome:c.outcome,evidence:evidence(m)});a.native.phase='awaiting-damage';
  }),
  beginRoll:(payload,user)=>nativeMutation(payload,user,async(actor,a,{target})=>{
   owner(actor,user);phase(a,'awaiting-damage');await boundSave(a,{target,allowReroll:true});a.native.phase='rolling-damage';a.native.damage={id:operation(payload),userId:user.id};
  }),
  finishRoll:(payload,user)=>nativeMutation(payload,user,async(actor,a)=>{
   owner(actor,user);phase(a,'rolling-damage');sameOperation(a.native.damage,payload,user);await boundSave(a);
   const m=await fromUuid(payload.damageUuid),c=m?.flags?.pf2e?.context,p=m?.flags?.[ID]?.voltageDamage,r=m?.rolls?.[0],targets=m?.flags?.['pf2e-toolbelt']?.targetHelper?.targets;
   if(game.messages.get(m?.id)!==m||!m?.isDamageRoll||m.rolls?.length!==1||r?._evaluated!==true||!Number.isFinite(r.total)||r.total<0||author(m)!==user.id||m.speaker?.actor!==actor.id||tokenUuid(m.speaker)!==a.originUuid||
    m.flags?.pf2e?.origin?.uuid!==a.itemUuid||c?.type!=='damage-roll'||c.outcome!==a.native.save.outcome||c.target?.token!==a.trigger.targetUuid||c.target?.actor!==a.trigger.targetActorUuid||
    !c.options?.includes(voltageRollOption(a))||p?.actorUuid!==a.actorUuid||p?.nonce!==a.nonce||p?.messageUuid!==a.messageUuid||p?.targetUuid!==a.trigger.targetUuid||p?.operationId!==a.native.damage.id||
    targets?.length!==1||targets[0]!==a.trigger.targetUuid||m.flags?.['pf2e-toolbelt']?.targetHelper?.saveVariants)throw Error('Native High Voltage damage card binding is invalid.');
   Object.assign(a.native.damage,{messageUuid:m.uuid,evidence:evidence(m)});a.native.phase='awaiting-application';
  }),
  beginDamage:(payload,user)=>nativeMutation(payload,user,async(_actor,a,{target})=>{
   owner(target.actor,user);phase(a,'awaiting-application');const message=await boundDamage(a);
   if(payload.damageUuid!==message.uuid||payload.targetUuid!==target.uuid||payload.targetActorUuid!==target.actor.uuid||payload.rollIndex!==0)throw Error('High Voltage damage is bound to its original target and native card.');
   a.native.phase='applying';a.native.application={id:operation(payload),userId:user.id};
  }),
  finishDamage:(payload,user)=>nativeMutation(payload,user,async(_actor,a,{target})=>{
   owner(target.actor,user);phase(a,'applying');sameOperation(a.native.application,payload,user);await boundDamage(a);
   const message=await fromUuid(payload.receiptUuid);
   if(game.messages.get(message?.id)!==message||!voltageReceiptMatches(message,a))throw Error('The authentic native High Voltage damage receipt is required.');
   a.status='done';a.native.phase='done';a.result={saveUuid:a.native.save.messageUuid,damageUuid:a.native.damage.messageUuid,receiptUuid:message.uuid,outcome:a.native.save.outcome};
  }),
  abortNative:(payload,user)=>nativeMutation(payload,user,async(_actor,a)=>{
   const record=a.native.phase==='rolling-save'?a.native.save:a.native.phase==='rolling-damage'?a.native.damage:a.native.phase==='applying'?a.native.application:null;
   sameOperation(record,payload,user);if(!['cancelled','uncertain'].includes(payload.status))throw Error('Invalid native cancellation status.');
   a.status=payload.status;a.native.phase=payload.status;
  }),
  settle:(payload,user)=>mutate(payload,user,async(actor,state)=>{
   if(user!==game.users.activeGM)throw Error('Only the active GM can settle High Voltage.');
   const a=state.activations[payload.nonce];if(!a||a.messageUuid!==payload.messageUuid||a.status!=='claimed')throw Error('High Voltage claim is not awaiting settlement.');
   if(a.native?.version===1)throw Error('Native High Voltage requires its authenticated damage receipt for settlement.');
   if(!['done','cancelled','uncertain'].includes(payload.status))throw Error('Invalid High Voltage result.');
   a.status=payload.status;a.result=copy(payload.result??null);await save(actor,state);return copy(a);
  }),
  expire:(payload,user)=>mutate(payload,user,async(actor,state)=>{for(const a of Object.values(state.activations))await closeIfStale(actor,state,a);return state.activeNonce;})
 };
}
