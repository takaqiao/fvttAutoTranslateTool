import {SerialActions} from './runtime.mjs';
import {sourceUuid} from './metapower/rules.mjs';

export const VOLTAGE_MODULE_ID='pf2e-third-party-automation';
export const HIGH_VOLTAGE_SOURCE='Compendium.battlezoo-eldamon-pf2e.powers.Item.9bElF2uVf5FCJtb9';
export const ELEMENTAL_POWERS_SOURCE='Compendium.battlezoo-eldamon-pf2e.eldamon-features.Item.naawsnBug9EOpzfN';
const ID=VOLTAGE_MODULE_ID,copy=value=>structuredClone(value),values=c=>Array.from(c?.values?.()??c??[]);
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
  if(c.isReroll){
   const prior=activation.attacks[tag?.slice(tagPrefix.length)];
   // PF2e deletes the original card before publishing the kept reroll, copying
   // context options. The persisted history survives that deletion and reload.
   if(!prior||!options.includes('check:reroll')||game.messages.get(prior.latestMessageId)&&prior.latestMessageId!==message.id||prior.actorUuid!==target.actor.uuid||prior.targetUuid!==activation.originUuid||prior.itemUuid!==(item.uuid??null))return null;
   prior.latestMessageId=message.id;prior.outcome=c.outcome;
  }else{
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
   let target;
   if(['attack','metal-hit'].includes(payload.kind)){
    if(payload.kind==='attack'&&user!==game.users.activeGM)throw Error('Native attack observation requires the active GM.');
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
   await save(actor,state);return copy(a);
  }),
  settle:(payload,user)=>mutate(payload,user,async(actor,state)=>{
   if(user!==game.users.activeGM)throw Error('Only the active GM can settle High Voltage.');
   const a=state.activations[payload.nonce];if(!a||a.messageUuid!==payload.messageUuid||a.status!=='claimed')throw Error('High Voltage claim is not awaiting settlement.');
   if(!['done','cancelled','uncertain'].includes(payload.status))throw Error('Invalid High Voltage result.');
   a.status=payload.status;a.result=copy(payload.result??null);await save(actor,state);return copy(a);
  }),
  expire:(payload,user)=>mutate(payload,user,async(actor,state)=>{for(const a of Object.values(state.activations))await closeIfStale(actor,state,a);return state.activeNonce;})
 };
}
