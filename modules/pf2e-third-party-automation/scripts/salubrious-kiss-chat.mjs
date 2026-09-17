import {assertSalubriousCardPrivacy} from './salubrious-privacy.mjs';
const NS='pf2e-third-party-automation';
const READ_ONLY='pf2e-third-party-salubrious-readonly';
const id=value=>typeof value==='string'&&/^[A-Za-z0-9_-]+$/.test(value);
const tokenUuid=value=>typeof value==='string'&&/^Scene\.[A-Za-z0-9_-]+\.Token\.[A-Za-z0-9_-]+$/.test(value);
const actorUuid=value=>typeof value==='string'&&/^(?:Actor\.[A-Za-z0-9_-]+|Scene\.[A-Za-z0-9_-]+\.Token\.[A-Za-z0-9_-]+\.Actor\.[A-Za-z0-9_-]+)$/.test(value);
const applicationLabels=new Set(['Full','Half','Double','Triple','Healing'].map(n=>`PF2E.DamageButton.${n}Context`));
const decoratedPredicates=new WeakSet();

/** Presentation classification, NOT permission to apply damage. The executor's
 * persisted exact proof and both nonce markers identify these automatic cards.
 * No live actor/claim lookup: deleted sources, reconnects and uncertain results
 * must not expose manual Apply again. The private damage guard remains required. */
function automaticDamageCard(message){
 const p=message?.flags?.[NS]?.salubriousKiss;
 if(p?.kind!=='damage'||message.isDamageRoll!==true)return false;
 const pf=message.flags.pf2e,c=pf?.context,r=message.rolls?.[0],s=message.speaker;
 const author=message.author?.id??message.author??message._source?.author??message.user?.id??message.user;
 if(!id(p.nonce)||p.nonce.length>80||!/^Actor\.[A-Za-z0-9_-]+$/.test(p.actorUuid??'')||!id(p.userId)||!tokenUuid(p.tokenUuid)||!tokenUuid(p.targetUuid)||!actorUuid(p.targetActorUuid)||!Number.isFinite(p.startedAt)||p.skill!=='occultism'||!Number.isInteger(p.tier)||p.tier<1||p.tier>4||[15,20,30,40][p.tier-1]!==p.dc)return false;
 const itemPrefix=`${p.actorUuid}.Item.`;
 if(typeof p.itemUuid!=='string'||!p.itemUuid.startsWith(itemPrefix)||!id(p.itemUuid.slice(itemPrefix.length)))return false;
 if(author!==p.userId||s?.actor!==p.actorUuid.slice(6)||`Scene.${s?.scene}.Token.${s?.token}`!==p.tokenUuid||!p.targetUuid.startsWith(`Scene.${s.scene}.Token.`))return false;
 if(pf?.origin?.type!=='feat'||pf.origin.actor!==p.actorUuid||pf.origin.uuid!==p.itemUuid||!id(pf.origin.messageId)||pf.suppressDamageButtons!==true||c?.origin?.actor!==p.actorUuid||c.origin.token!==p.tokenUuid||c.type!=='skill-check'||c.action!=='treat-wounds'||c.dc?.value!==p.dc||!Array.isArray(c.domains)||!c.domains.includes('occultism'))return false;
 if(!['criticalFailure','success','criticalSuccess'].includes(c.outcome)||!Array.isArray(c.options))return false;
 try{assertSalubriousCardPrivacy({message,claim:p})}catch{return false}
 if(!['action:treat-wounds','skip-handling-message',`${NS}:salubrious-check:${p.nonce}`,`${NS}:salubrious-damage:${p.nonce}`].every(option=>c.options.includes(option)))return false;
 return message.rolls.length===1&&Number.isFinite(r?.total)&&(r._evaluated===true||r.evaluated===true);
}

/** Called synchronously from the existing renderChatMessageHTML hook. A root
 * class also covers Toolbelt's later async clones without observing the DOM.
 * All automatic states are read-only from publication onward; this deliberately
 * does not promise that a currently uncertain card has finished applying. */
export function renderSalubriousCard(message,html){
 if(!html?.matches?.('.message[data-message-id]')||html.dataset.messageId!==message?.id)return false;
 const readonly=automaticDamageCard(message);
 html.classList.toggle(READ_ONLY,readonly);
 return readonly;
}

/** getChatMessageContextOptions(app, entries), Foundry 14's supported hook.
 * Entries are created once, so evaluate the exact target message each time the
 * menu opens. Preserve native callbacks, entry identities and original visible
 * semantics for every ordinary card; do not alter check rerolls/inspection. */
export function filterSalubriousDamageContext(game,entries){
 if(!Array.isArray(entries))return;
 for(const entry of entries){
  if(!applicationLabels.has(entry.label)||decoratedPredicates.has(entry.visible))continue;
  const nativeVisible=entry.visible;
  const visible=function(element,...args){
   const messageId=element?.dataset?.messageId;
   if(messageId&&automaticDamageCard(game.messages.get(messageId)))return false;
   return typeof nativeVisible==='function'?Reflect.apply(nativeVisible,this,[element,...args]):nativeVisible===undefined?true:nativeVisible;
  };
  decoratedPredicates.add(visible);entry.visible=visible;
 }
}
