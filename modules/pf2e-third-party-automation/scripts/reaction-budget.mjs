import {MODULE_ID} from './rules.mjs';
import {SerialActions} from './runtime.mjs';
import {isActiveGM,isUnappliedDamageError} from './native-context.mjs';

const AAT='pf2e-auto-action-tracker',values=c=>Array.from(c?.values?.()??c??[]),own=d=>d?.flags?.[MODULE_ID]??{};
const shieldPrefix=`${MODULE_ID}:native-shield:`,authorId=m=>m.author?.id??m.user?.id??m.user;
const shieldReady=actor=>{const s=actor?.attributes?.shield;return !!(actor?.hitPoints&&s?.itemId&&s.raised&&!s.broken&&!s.destroyed);};
const textOnly=html=>String(html??'').replace(/<[^>]*>/g,'').replace(/&(?:amp|lt|gt|quot|#39|#x27);/g,e=>({'&amp;':'&','&lt;':'<','&gt;':'>','&quot;':'"','&#39;':"'",'&#x27;':"'"}[e])).replace(/\s+/g,' ').trim();
const regexEscape=s=>s.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');
// PF2e 8.5 keeps no structured block flag when hardness absorbs all damage.
// Read only its exact damage-taken statement element, including nested actor span.
function damageStatement(content){
 if(!/^\s*<section\s+class=["']damage-taken["'][^>]*>/.test(content??''))return null;
 const start=/<span\s+class=["']statements["'][^>]*>/.exec(content);if(!start)return null;
 const offset=start.index+start[0].length,tags=/<\/?span\b[^>]*>/g;tags.lastIndex=offset;let depth=1,tag;
 while((tag=tags.exec(content))){depth+=tag[0].startsWith('</')?-1:1;if(depth===0)return textOnly(content.slice(offset,tag.index));}return null;
}
function nativeDamageStatement(message,token,game,keys){
 const statement=damageStatement(message.content);if(!statement)return false;
 for(const key of keys){
  const path=`PF2E.Actor.ApplyDamage.${key}`,template=game.i18n?.localize?.(path);if(!template||template===path)continue;
  const placeholders={actor:regexEscape(textOnly(String(token.name??'').replace(/[<>]/g,''))),absorbedDamage:'[1-9][0-9]*(?:[.,][0-9]+)?',hpDamage:'[0-9]+(?:[.,][0-9]+)?'};
  const pattern=textOnly(template).split(/(\{[^}]+\})/).map(part=>part.startsWith('{')?(placeholders[part.slice(1,-1)]??'(?!)'):regexEscape(part)).join('');
  // Additional native statements describe damage to the shield/death. They do
  // not establish the block; only the complete localized primary statement does.
  if(new RegExp(`^${pattern}(?:$|\\s)`).test(statement))return true;
 }
 return false;
}
/** null means unproven, rather than a confirmed non-block. No budget mutation. */
export function classifyNativeShieldBlock(message,{token,shieldId,nativeBlockNonce=null,game}){
 if(message?.flags?.pf2e?.context?.type!=='damage-taken')return null;
 const opts=message.flags.pf2e.context.options??[],custom=opts.filter(o=>typeof o==='string'&&o.startsWith(`${MODULE_ID}:destructive-block:`));
 if(custom.length){
  const proof=own(message).shieldBlock;
  if(custom.length!==1||proof?.kind!=='destructive-block'||proof.uncertain||(proof.nativeBlockNonce??null)!==nativeBlockNonce||proof.shieldId!==shieldId||custom[0]!==`${MODULE_ID}:destructive-block:${proof.nonce}`||!Number.isFinite(proof.incoming)||proof.incoming<0||proof.blocked!==(proof.incoming>0))return null;
  return proof.blocked;
 }
 const shield=message.flags.pf2e.appliedDamage?.shield;
 // A structured ID outranks prose: a shield changed while waiting cannot be
 // identified as the original shield solely by the localized absorption text.
 if(shield&&shield.id!==shieldId)return null;
 if(shield?.id===shieldId&&Number(shield.damage)>0||nativeDamageStatement(message,token,game,['DamagedForNShield','ShieldAbsorbsAll']))return true;
 if(!shield&&nativeDamageStatement(message,token,game,['TakesNoDamage','DamagedForN']))return false;
 return null;
}
const queues=new WeakMap(),restricted={
 'combat-reflexes':['reactive-strike'],'tactical-reflexes':['reactive-strike'],
 'quick-shield-block':['shield-block'],
 'divine-reflexes':['retributive-strike','glimpse-of-redemption','liberating-step','iron-command','selfish-shield','destructive-vengeance'],
 'esoteric-reflexes':['implements-interruption','amulets-abeyance','bells-disruption'],
};
export function reactionEpoch(actor,game){const c=game.combat,index=c?.turns?.findIndex(t=>t.actor?.uuid===actor.uuid)??-1;return c?.started&&index>=0?`${c.id}:${c.round-(index>c.turn?1:0)}`:null;}
const combatantFor=(actor,game)=>game.combat?.turns?.find(t=>t.actor?.uuid===actor.uuid);
const disruptPaidStates=new Set(['claimed','attack-rolled','damage-rolled','applying','done','uncertain']);
const disruptIdentity=(proof,actor)=>typeof proof?.nonce==='string'&&proof.nonce.length>0&&proof.claimKey===`disrupt:${proof.nonce}`&&proof.actorUuid===actor?.uuid;
const paidDisruptClaims=actor=>(own(actor).disruptPrey?.reactions??[]).filter(r=>disruptPaidStates.has(r.state)&&disruptIdentity(r,actor));
const paidDeflectionClaims=actor=>(own(actor).transcendentDeflection?.reactions??[]).filter(r=>['claimed','prevented','done','uncertain'].includes(r.state)&&typeof r.nonce==='string'&&r.nonce&&r.claimKey===`deflect:${r.nonce}`&&r.actorUuid===actor.uuid).map(r=>({...r,cost:1,slug:'transcendent-deflection'}));
const paidCastClaims=actor=>(own(actor).nativeCasts??[]).filter(r=>['paid','used','disrupted','uncertain'].includes(r.state)&&r.actorUuid===actor.uuid&&r.nativeCastScope?.castNonce===r.id&&r.sourceAction?.type==='reaction'&&r.sourceAction.value===1&&typeof r.sourceAction.epoch==='string').map(r=>({...r,epoch:r.sourceAction.epoch,cost:1,slug:'cast-a-spell',claimKey:`cast:${r.id}`,checkId:r.messageId}));
function provenCastClaim(message,actor,game){
 const proof=own(message).nativeCast??own(message).disruptPreySourceStopped,stopped=!!own(message).disruptPreySourceStopped;
 if(!message?.id||game.messages?.get?.(message.id)!==message||!proof||message.rolls?.length||message.speaker?.actor!==actor.id||proof.actorUuid!==actor.uuid||proof.userId!==authorId(message))return null;
 const matches=paidCastClaims(actor).filter(r=>r.id===proof.id&&r.itemUuid===proof.itemUuid&&r.userId===proof.userId&&(!r.messageId||r.messageId===message.id)&&(!r.nativeCastScope.tokenUuid||r.nativeCastScope.tokenUuid===`Scene.${message.speaker.scene}.Token.${message.speaker.token}`)&&(stopped||message.flags?.pf2e?.origin?.uuid===r.itemUuid&&(!message.flags.pf2e.origin.actor||message.flags.pf2e.origin.actor===actor.uuid)));
 return matches.length===1?matches[0]:null;
}
// A delayed native/RPC card belongs to the epoch of its durable payment. Only
// an exact live attack and its paid source claim can establish that epoch; a
// marker's epoch, a log claimKey alone, or a similar weapon cannot do so.
function provenDisruptClaim(message,actor,game,claims){
 const pf=message?.flags?.pf2e,context=pf?.context,proof=own(message).disruptPreyReaction;
 const author=message?.author??game.users?.get?.(authorId(message??{}));
 if(!message?.id||game.messages?.get?.(message.id)!==message||!author||!actor.testUserPermission?.(author,'OWNER')||message.actor?.uuid!==actor.uuid||message.speaker?.actor!==actor.id||context?.type!=='attack-roll'||context.action!=='strike'||context.isReroll||!message.rolls?.length||!context.options?.includes('action:reaction')||!context.options.includes('action:disrupt-prey')||!disruptIdentity(proof,actor))return null;
 const matches=(claims??paidDisruptClaims(actor)).filter(claim=>/^[A-Za-z0-9_-]{1,80}$/.test(claim.nonce)&&typeof claim.epoch==='string'&&claim.epoch.length>0&&claim.nonce===proof.nonce&&claim.claimKey===proof.claimKey&&claim.userId===author.id&&(!claim.checkId||claim.checkId===message.id)&&(!proof.weaponKey||proof.weaponKey===claim.weaponKey)&&typeof claim.itemUuid==='string'&&pf.origin?.actor===actor.uuid&&pf.origin.uuid===claim.itemUuid&&`Scene.${message.speaker.scene}.Token.${message.speaker.token}`===claim.tokenUuid&&typeof claim.targetActorUuid==='string'&&context.target?.actor===claim.targetActorUuid&&typeof claim.targetUuid==='string'&&context.target?.token===claim.targetUuid);
 return matches.length===1?matches[0]:null;
}
const sameClaim=(entry,claim)=>!!(claim.checkId&&entry.msgId===claim.checkId||claim.claimKey&&entry.claimKey===claim.claimKey);
/** Only the availability recheck and durable claim write belong in this lock. */
export function withReactionReservation(actor,game,fn){let queue=queues.get(game);if(!queue){queue=new SerialActions();queues.set(game,queue);}return queue.run(actor.uuid,fn);}

export function genericReactionAvailable(actor,game,{pending=[]}={}){
 const current=reactionEpoch(actor,game);if(!current)return false;
 const combatant=combatantFor(actor,game),ledger=own(combatant).reactionBudget;
 let entries=game.modules?.get(AAT)?.active?(combatant.getFlag?.(AAT,'log')??combatant.flags?.[AAT]?.log??[]).filter(e=>e.type==='reaction'):[...(ledger?.epoch===current?ledger.entries??[]:[])];
 const c=game.combat,index=c.turns.findIndex(t=>t.actor?.uuid===actor.uuid),hadOwnTurn=c.round>1||index<=c.turn;
 const slots=(hadOwnTurn?Object.entries(restricted):[]).filter(([slug])=>values(actor.items).some(i=>slug==='quick-shield-block'&&(i.sourceId??i._stats?.compendiumSource)?(i.sourceId??i._stats?.compendiumSource)==='Compendium.pf2e.feats-srd.Item.pRqcm5P2ZFihSpVI':(i.slug??i.system?.slug)===slug)).map(([,allowed])=>({allowed})).sort((a,b)=>a.allowed.length-b.allowed.length);
 for(let n=0;n<(actor.system?.resources?.reactions?.max||1);n++)slots.push({allowed:null});
 const fear=(own(actor).fear?.reactions??[]).map(r=>({...r,claimKey:r.id?`battle:${r.id}`:null,slug:'demoralize'}));
 const checks=(own(actor).reactionChecks?.reactions??[]).filter(r=>['claimed','used'].includes(r.state)&&['clock','squawk','eat'].includes(r.kind)).map(r=>({...r,claimKey:r.nonce?`check:${r.nonce}`:null,slug:{clock:'turn-back-the-clock',squawk:'squawk',eat:'eat-fortune'}[r.kind]}));
 const paidDisrupt=paidDisruptClaims(actor),disrupt=paidDisrupt.filter(r=>r.epoch===current).map(r=>({...r,slug:'disrupt-prey',cost:1}));
 const casts=paidCastClaims(actor);
 // AAT only stores a message ID. Its create hook can finish before the provider
 // writes that ID back to the pending claim: derive identity from that exact
 // live native card, never from a recent-message or weapon-name search.
 if(paidDisrupt.length)entries=entries.flatMap(entry=>{
  const claim=provenDisruptClaim(game.messages?.get?.(entry.msgId),actor,game,paidDisrupt);
  return !claim?[entry]:claim.epoch!==current?[]:[{...entry,claimKey:claim.claimKey}];
 });
 if(casts.length)entries=entries.flatMap(entry=>{const claim=provenCastClaim(game.messages?.get?.(entry.msgId),actor,game);return !claim?[entry]:claim.epoch!==current?[]:[{...entry,claimKey:claim.claimKey,slug:claim.slug}];});
 // AAT may label a reaction Strike by its weapon or generic action. The exact
 // paid claim identifies its slot eligibility without modifying AAT's log.
 for(const claim of disrupt)for(let i=0;i<entries.length;i++)if(sameClaim(entries[i],claim))entries[i]={...entries[i],slug:claim.slug,cost:1};
 for(const claim of [...fear,...checks,...disrupt,...casts,...paidDeflectionClaims(actor),...pending])if(claim.epoch===current&&!entries.some(e=>sameClaim(e,claim)))entries.push({type:'reaction',cost:claim.cost??1,slug:claim.slug??'demoralize',msgId:claim.checkId,claimKey:claim.claimKey});
 for(const entry of entries)for(let n=0;n<Math.max(1,Number(entry.cost)||0);n++){const slot=slots.find(s=>!s.spent&&(!s.allowed||entry.msgId==='System'||s.allowed.includes(entry.slug)));if(slot)slot.spent=true;}
 return slots.some(s=>!s.spent&&!s.allowed);
}

function reactionData(message,item,actor){
 const context=message.flags?.pf2e?.context??{},proof=own(message).reactionChecks;
 if(context.isReroll||proof?.kind==='check-reaction-result'||['damage-roll','damage-taken'].includes(context.type))return null;
 const options=context.options??[],rolled=message.rolls?.length>0;
 const marker=options.includes('action:reaction')||options.includes('trait:reaction')||context.type==='reaction';
 const glyph=/class=["'][^"']*\baction-glyph\b[^"']*["'][^>]*>\s*R\s*</.test(message.flavor??'');
 const itemReaction=!rolled&&(!context.type||['self-effect','spell-cast'].includes(context.type))&&item?.actor&&(['reaction'].includes(item.system?.actionType?.value)||item.system?.time?.value==='reaction');
 if(!(rolled&&(marker||glyph)||itemReaction))return null;
 const disrupt=own(message).disruptPreyReaction;
 if(context.type==='attack-roll'&&rolled&&options.includes('action:reaction')&&options.includes('action:disrupt-prey')&&disruptIdentity(disrupt,actor))return {type:'reaction',msgId:message.id,cost:1,slug:'disrupt-prey',claimKey:disrupt.claimKey};
 const action=options.find(o=>/^action:[a-z][a-z-]*$/.test(o)&&!['action:reaction','action:free'].includes(o))?.slice(7);
 const battle=options.find(o=>o.startsWith(`${MODULE_ID}:battle-cry:`))?.slice(`${MODULE_ID}:battle-cry:`.length);
 const claimKey=battle?`battle:${battle}`:proof?.kind==='reaction-use'&&proof.nonce?`check:${proof.nonce}`:null;
 return {type:'reaction',msgId:message.id,cost:1,slug:action??item?.slug??item?.system?.slug??'reaction',...(claimKey?{claimKey}:{})};
}

export function createReactionBudget({game,fromUuid=globalThis.fromUuid,onError=console.error}={}){
 const scopes=new Map();let socket;
 const owner=(actor,user)=>{if(!isActiveGM(game)||!user||!actor?.testUserPermission?.(user,'OWNER'))throw Error('盾牌格挡回执需要当前主GM验证角色所有者。');};
 const persist=(combatant,data)=>{if(!isActiveGM(game))throw Error('主GM已改变，不能写入格挡回执。');return combatant.update({[`flags.${MODULE_ID}.reactionBudget`]:data});};
 const resolveShield=async(payload,user)=>{const actor=await fromUuid(payload.actorUuid),token=await fromUuid(payload.tokenUuid);owner(actor,user);if(token?.documentName!=='Token'||token.actor?.uuid!==actor.uuid)throw Error('盾牌格挡Token来源不匹配。');return {actor,token};};
 async function beginShield(payload,user){
  if(typeof payload?.nonce!=='string'||!/^[A-Za-z0-9-]{8,80}$/.test(payload.nonce))throw Error('盾牌格挡认领编号无效。');
  const {actor}=await resolveShield(payload,user);
  return withReactionReservation(actor,game,async()=>{
   owner(actor,user);if(game.modules?.get(AAT)?.active)return null;
   if(reactionEpoch(actor,game)!==payload.epoch)throw Error('盾牌格挡请求期间回合已改变，尚未应用伤害；请按当前回合重新操作。');
   if(!shieldReady(actor)||actor.attributes.shield.itemId!==payload.shieldId)return null;
   const combatant=combatantFor(actor,game),previous=own(combatant).reactionBudget,entries=previous?.epoch===payload.epoch?[...previous.entries??[]]:[];
   if(entries.some(e=>e.shield?.nonce===payload.nonce))throw Error('本次盾牌格挡已认领，不会重复执行。');
   const index=entries.findIndex(e=>e.slug==='shield-block'&&!e.shield),source=index<0?null:entries[index];
   const receipt={...payload,userId:user.id,state:'pending',sourceCardId:source?.msgId??null};
   const entry={...source,type:'reaction',cost:1,slug:'shield-block',msgId:source?.msgId??`shield:${payload.nonce}`,shield:receipt};
   if(index<0)entries.push(entry);else entries[index]=entry;
   await persist(combatant,{epoch:payload.epoch,entries});owner(actor,user);return receipt;
  });
 }
 async function finishShield(payload,user){
  const {actor,token}=await resolveShield(payload,user);
  return withReactionReservation(actor,game,async()=>{
   owner(actor,user);if(reactionEpoch(actor,game)!==payload.epoch)return false;
   const combatant=combatantFor(actor,game),ledger=own(combatant).reactionBudget,index=ledger?.entries?.findIndex(e=>e.shield?.nonce===payload.nonce)??-1;
   if(ledger?.epoch!==payload.epoch||index<0)throw Error('盾牌格挡认领回执不存在。');
   const entries=[...ledger.entries],entry=entries[index],claim=entry.shield;if(claim.userId!==user.id||claim.actorUuid!==actor.uuid||claim.tokenUuid!==token.uuid)throw Error('盾牌格挡认领所有者不匹配。');
   if(payload.messageId){
    const message=game.messages.get(payload.messageId),pf=message?.flags?.pf2e;
    if(!message||authorId(message)!==user.id||message.speaker?.actor!==actor.id||`Scene.${message.speaker?.scene}.Token.${message.speaker?.token}`!==token.uuid||pf?.context?.type!=='damage-taken'||!pf.context.options?.includes(shieldPrefix+payload.nonce)||message.content!==payload.content||typeof payload.content!=='string'||payload.content.length>100000)throw Error('原生盾牌格挡回执已改变或不匹配。');
    if(claim.state==='used'){if(claim.messageId!==message.id)throw Error('同一格挡已绑定另一张伤害回执。');return true;}
   }
   if(payload.messageId&&payload.blocked===true){
    entries[index]={...entry,shield:{...claim,state:'used',messageId:payload.messageId}};
   }else if(payload.messageId&&payload.blocked===false||payload.enteredNative===false){
    if(claim.state!=='pending')return false;
    if(claim.sourceCardId){const restored={...entry};delete restored.shield;entries[index]=restored;}else entries.splice(index,1);
   }else throw Error('未获得明确原生格挡结果，保留认领，不会返还反应。');
   await persist(combatant,{epoch:ledger.epoch,entries});return payload.blocked===true;
  });
 }
 async function shieldRpc(method,payload){
  if(isActiveGM(game))return method==='begin'?beginShield(payload,game.user):finishShield(payload,game.user);
  if(!socket||!game.users.activeGM)throw Error('原生格挡反应记录需要在线主GM。');
  const result=await socket.executeAsUser(`reaction-budget:shield-${method}`,game.users.activeGM.id,payload);if(!result?.ok)throw Error(result?.error??'格挡反应回执失败。');return result.value;
 }
 function captureShield(message,creator){
  const opts=message.flags?.pf2e?.context?.options??[],markers=opts.filter(o=>typeof o==='string'&&o.startsWith(shieldPrefix));if(markers.length!==1)return;
  const scope=scopes.get(markers[0].slice(shieldPrefix.length));if(!scope||scope.messageId||creator!==game.user.id||authorId(message)!==game.user.id||message.actor?.uuid!==scope.actor.uuid||message.flags.pf2e.context.type!=='damage-taken'||`Scene.${message.speaker?.scene}.Token.${message.speaker?.token}`!==scope.token.uuid)return;
  const blocked=classifyNativeShieldBlock(message,{token:scope.token,shieldId:scope.receipt.shieldId,nativeBlockNonce:scope.receipt.nonce,game});
  if(blocked!==null){scope.messageId=message.id;scope.content=message.content;scope.blocked=blocked;}
 }
 async function applyDamage(actor,params,apply){
  const damage=typeof params.damage==='number'?params.damage:params.damage?.total,token=params.token?.document??params.token;
  if(game.modules?.get(AAT)?.active||!params.shieldBlockRequest||params.final||!Number.isFinite(damage)||damage<=0||!shieldReady(actor)||!reactionEpoch(actor,game)||token?.actor?.uuid!==actor.uuid)return apply(params);
  if(!actor.testUserPermission?.(game.user,'OWNER'))throw Error('无权使用这个角色的盾牌格挡。');
  const payload={nonce:globalThis.foundry?.utils?.randomID?.(24)??globalThis.crypto.randomUUID(),actorUuid:actor.uuid,tokenUuid:token.uuid,shieldId:actor.attributes.shield.itemId,epoch:reactionEpoch(actor,game)},receipt=await shieldRpc('begin',payload);
  if(!receipt)return apply(params);
  const scope={actor,token,receipt};scopes.set(payload.nonce,scope);let completed=false;
  try{const result=await apply({...params,rollOptions:new Set([...params.rollOptions??[],shieldPrefix+payload.nonce])});completed=true;
   try{await shieldRpc('finish',{...payload,messageId:scope.messageId??null,content:scope.content??null,blocked:scope.blocked});}catch(error){try{onError(error)}catch{/* Native damage already completed; preserve its result. */}}
   return result;
  }
  catch(error){
   // A thrown native call may already have applied damage. Only a captured real
   // block can commit it; absence of a card is not proof of non-execution.
   if(!completed&&scope.messageId)try{await shieldRpc('finish',{...payload,messageId:scope.messageId,content:scope.content,blocked:scope.blocked});}catch(e){try{onError(e)}catch{/* Preserve native failure. */}}
   else if(!completed&&!scope.messageId&&isUnappliedDamageError(error))try{
    // The final adapter explicitly proved it never called native damage. Restore
    // only this pending reservation, retaining a previously posted manual card.
    await shieldRpc('finish',{...payload,enteredNative:false});
   }catch(e){try{onError(e)}catch{/* Preserve the original pre-native failure. */}}
   throw error;
  }finally{scopes.delete(payload.nonce);}
 }
 async function record(message,creator){
  if(!isActiveGM(game)||game.modules?.get(AAT)?.active||!message?.id||game.messages.get(message.id)!==message)return false;
  const author=message.author??game.users.get(message.user?.id??message.user);
  if(!author||creator&&creator!==author.id&&creator!==game.users.activeGM?.id)return false;
  const actorId=message.speaker?.actor,actor=message.actor??(typeof actorId==='string'?await fromUuid?.(`Actor.${actorId}`):null);if(!actor?.testUserPermission?.(author,'OWNER'))return false;
  // The viewed encounter can differ on each client. Bind native card payment
  // to its actor/token's unique actual encounter and recheck it after awaits.
  const encounter=()=>{
   const token=message.speaker?.scene&&message.speaker?.token?`Scene.${message.speaker.scene}.Token.${message.speaker.token}`:null;
   const matches=values(game.combats??(game.combat?[game.combat]:[])).filter(c=>c.started&&c.turns?.some(t=>t.actor?.uuid===actor.uuid&&(!token||t.token?.uuid===token)));
   return matches.length===1?matches[0]:null;
  };
  const combat=encounter();if(!combat)return false;
  const bounded={combat,modules:game.modules,messages:game.messages,users:game.users};
  const receipt=own(message).reactionBudget,epoch=reactionEpoch(actor,bounded),current=()=>encounter()===combat&&epoch===reactionEpoch(actor,bounded);
  if(!epoch||receipt?.epoch&&receipt.epoch!==epoch&&!own(message).disruptPreyReaction&&!provenCastClaim(message,actor,game))return false;
  // Queue before resolving the item: a competing automatic reaction must wait
  // for this already-posted native reaction to be persisted first.
  return withReactionReservation(actor,game,async()=>{
   if(!isActiveGM(game)||!current())return false;
   const itemUuid=message.flags?.pf2e?.origin?.uuid,item=message.item??(typeof itemUuid==='string'?await fromUuid?.(itemUuid):null);
   if(!isActiveGM(game)||!current()||game.messages.get(message.id)!==message||!actor.testUserPermission?.(author,'OWNER'))return false;
   if(item?.actor&&item.actor.uuid!==actor.uuid)return false;
   const cast=provenCastClaim(message,actor,game),entry=cast?{type:'reaction',cost:1,slug:cast.slug,msgId:message.id,claimKey:cast.claimKey}:reactionData(message,item,actor);if(!entry)return false;
   const paid=cast??provenDisruptClaim(message,actor,game);
   if(paid&&paid.epoch!==epoch){
    // Keep the current combatant ledger untouched, even when it already has a
    // different reaction. Stamp only this verified card for duplicate delivery.
    if(receipt?.epoch===paid.epoch)return false;
    if(!isActiveGM(game))return false;
    await message.update({[`flags.${MODULE_ID}.reactionBudget`]:{epoch:paid.epoch,actorUuid:actor.uuid,combatantId:combatantFor(actor,bounded)?.id}});
    return true;
   }
   const combatant=combatantFor(actor,bounded),previous=own(combatant).reactionBudget,entries=previous?.epoch===epoch?[...previous.entries??[]]:[],index=entries.findIndex(e=>e.msgId===entry.msgId);
   if(index>=0){const updated={...entries[index],...entry};if(JSON.stringify(entries[index])===JSON.stringify(updated))return false;entries[index]=updated;}else entries.push(entry);
   if(!isActiveGM(game)||!current())return false;
   await combatant.update({[`flags.${MODULE_ID}.reactionBudget`]:{epoch,entries}});
   if(!receipt&&isActiveGM(game))await message.update({[`flags.${MODULE_ID}.reactionBudget`]:{epoch,actorUuid:actor.uuid,combatantId:combatant.id}});
   return true;
  });
 }
  function register({Hooks,socket:socketApi}={}){
   socket=socketApi;
   if(socket)for(const method of ['begin','finish'])socket.register(`reaction-budget:shield-${method}`,async function(payload){try{return {ok:true,value:await (method==='begin'?beginShield:finishShield)(payload,game.users.get(this.socketdata.userId))}}catch(error){return {ok:false,error:error.message}}});
   const onCreate=(message,_options,userId)=>{captureShield(message,userId);record(message,userId).catch(onError);};
  const onUpdate=(message,changes,_options,userId)=>{if(changes.flags?.[MODULE_ID]?.reactionChecks||Object.keys(changes).some(k=>k.startsWith(`flags.${MODULE_ID}.reactionChecks`)))record(message,userId).catch(onError);};
  const ids=[['createChatMessage',Hooks.on('createChatMessage',onCreate)],['updateChatMessage',Hooks.on('updateChatMessage',onUpdate)]];
  return()=>{for(const[name,id]of ids)Hooks.off(name,id);};
 }
  return {record,register,applyDamage,available:actor=>genericReactionAvailable(actor,game)};
}
export function registerReactionBudget({Hooks,...options}){return createReactionBudget(options).register({Hooks});}
