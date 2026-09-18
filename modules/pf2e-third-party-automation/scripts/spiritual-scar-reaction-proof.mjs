import {MODULE_ID as M} from './rules.mjs';
import {SPIRITUAL_SCAR_SOURCE} from './spiritual-scar-native.mjs';
import {isActualUseMessage} from './usage-events.mjs';
const values=c=>Array.from(c?.values?.()??c??[]),paid=new Set(['reserved','paid','native','followup','done','uncertain']);
const validId=value=>typeof value==='string'&&/^[A-Za-z0-9_-]{1,100}$/.test(value);
export const spiritualScarClaims=document=>document?.flags?.[M]?.spiritualScarClaims??[];
export function paidSpiritualScarClaims(actor,combatant,combat){
 return spiritualScarClaims(combatant).filter(c=>paid.has(c.status)&&validId(c.nonce)&&c.claimKey===`scar:${c.nonce}`&&c.actorUuid===actor.uuid&&c.tokenUuid===combatant.token?.uuid&&c.combatId===combat.id&&c.combatantId===combatant.id&&typeof c.epoch==='string'&&c.epoch.length>0)
  .map(c=>({...c,cost:1,slug:'spiritual-scar',checkId:c.messageId}));
}
export function findSpiritualScarClaim(game,nonce){
 if(!validId(nonce))return null;
 const matches=values(game.combats).flatMap(combat=>values(combat.turns).flatMap(combatant=>spiritualScarClaims(combatant).filter(c=>c.nonce===nonce).map(claim=>({combat,combatant,claim}))));
 return matches.length===1?matches[0]:null;
}
/** Accounting proof only. The live provider and Use ledger authorize effects;
 * this connects an original daily Use to its already reserved reaction slot. */
export function provenSpiritualScarReactionCard(message,actor,game){
 const proof=message?.flags?.[M]?.spiritualScarInput,bound=findSpiritualScarClaim(game,proof?.nonce);
 if(!bound)return null;
 const {combat,combatant}=bound,claim=paidSpiritualScarClaims(actor,combatant,combat).find(c=>c.nonce===proof.nonce);
 const origin=message.flags?.pf2e?.origin,user=game.users?.get(message.author?.id??message.user?.id??message.user);
 if(!claim||game.messages?.get(message.id)!==message||message.rolls?.length||message.speaker?.actor!==actor.id||claim.userId!==user?.id||actor.testUserPermission?.(user,'OWNER')!==true||claim.messageId&&claim.messageId!==message.id||origin?.actor!==actor.uuid||origin.uuid!==claim.itemUuid||origin.type!=='action'||`Scene.${message.speaker?.scene}.Token.${message.speaker?.token}`!==claim.tokenUuid||!isActualUseMessage(message))return null;
 const item=actor.items?.get(claim.itemUuid.split('.').at(-1)),source=item?.sourceId??item?._stats?.compendiumSource??item?.flags?.core?.sourceId;
 if(item?.actor!==actor||item.uuid!==claim.itemUuid||source!==SPIRITUAL_SCAR_SOURCE||item.system?.actionType?.value!=='reaction')return null;
 const use=item.flags?.[M]?.spiritualScarUse?.operations?.[claim.nonce];
 if(!use||!['paid','ready','consumed','uncertain'].includes(use.status)||use.actorUuid!==actor.uuid||use.itemUuid!==item.uuid||use.userId!==claim.userId||use.nonce!==claim.nonce||!validId(use.paymentNonce)||use.paymentNonce!==proof.paymentNonce||!validId(message.flags[M].usageInput?.frequencyReceiptId)||use.messageId&&use.messageId!==message.id||use.frequencyReceiptId&&use.frequencyReceiptId!==message.flags[M].usageInput.frequencyReceiptId)return null;
 const whisper=message.whisper??[];
 if(!Array.isArray(whisper)||!Array.isArray(use.privacy?.whisper)||whisper.some(id=>typeof id!=='string')||use.privacy.blind!==(message.blind===true)||JSON.stringify([...new Set(whisper)].sort())!==JSON.stringify([...new Set(use.privacy.whisper)].sort()))return null;
 return {...claim,checkId:message.id};
}
