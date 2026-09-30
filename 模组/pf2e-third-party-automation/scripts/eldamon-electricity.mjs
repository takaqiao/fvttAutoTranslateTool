import {SerialActions} from './runtime.mjs';
import {sourceUuid} from './metapower/rules.mjs';
import {genericReactionAvailable} from './reaction-budget.mjs';
import {reactionPermitted} from './reaction-restriction.mjs';
import {publicTargetName} from './native-context.mjs';
import {isActualUseMessage} from './usage-events.mjs';

export const ELECTRICITY_MODULE_ID='pf2e-third-party-automation';
export const ELECTRICITY_SOURCES=Object.freeze({
 charged:'Compendium.battlezoo-eldamon-pf2e.conditions.Item.Bi2aHykg6CZrQCnR',
 shocked:'Compendium.battlezoo-eldamon-pf2e.conditions.Item.1fZbuJEbVmE3J4XL',
 shell:'Compendium.battlezoo-eldamon-pf2e.eldamon-features.Item.FBc2TxqWSRomT9fC',
 surge:'Compendium.battlezoo-eldamon-pf2e.powers.Item.veFrnrxYjlqca13w',
 anvil:'Compendium.battlezoo-eldamon-pf2e.powers.Item.hQOa1yaP9C6wajNn',
 static:'Compendium.battlezoo-eldamon-pf2e.powers.Item.KWQgx7RMeY3RKW6J',
 shot:'Compendium.battlezoo-eldamon-pf2e.powers.Item.QIYppaP0zcGvb5Bd',
 chain:'Compendium.battlezoo-eldamon-pf2e.powers.Item.fzV5Ly3a9nEsfcAJ',
});
export const ELECTRICITY_BASIC_SOURCES=Object.freeze({
 element:'Compendium.battlezoo-eldamon-pf2e.eldamon-features.Item.9KtNlRXeuxZSoVaI',
 shield:'Compendium.battlezoo-eldamon-pf2e.actions.Item.g8lH9enxTY6Cpx9V',
 shieldEffect:'Compendium.battlezoo-eldamon-pf2e.effects.Item.OJMStIdZzBU4L4N6',
 manipulation:'Compendium.battlezoo-eldamon-pf2e.eldamon-features.Item.h4D0hXhSHqmyBEQN',
});
const ID=ELECTRICITY_MODULE_ID,S=ELECTRICITY_SOURCES,copy=x=>structuredClone(x),values=c=>Array.from(c?.values?.()??c??[]);
export const electricityBasicSettlementEnabled=game=>game?.world?.id==='ujx5r8oipw7ercdr'&&game.system?.id==='pf2e'&&game.system.version==='8.5.1'&&game.modules?.get('battlezoo-eldamon-pf2e')?.active===true;
export const electricityBasicCardType=message=>!!message&&!message.isRoll&&!message.isCheckRoll&&!message.isDamageRoll&&!['attack-roll','damage-roll','saving-throw','skill-check','damage-taken'].includes(message.flags?.pf2e?.context?.type);
export function electricityBasicAction(item){
 const kind=['shield','manipulation'].find(key=>sourceUuid(item)===ELECTRICITY_BASIC_SOURCES[key]);
 if(!kind)return null;
 if(!values(item?.actor?.items).some(i=>sourceUuid(i)===ELECTRICITY_BASIC_SOURCES.element))return null;
 return kind;
}
export const ELECTRICITY_APPLY_PREFIX=`${ID}:electricity-apply:`;
export const ELECTRICITY_SOURCE_PREFIX=`${ID}:electricity-source:`;
const author=m=>m?.author?.id??m?.user?.id??m?.user;
const visibleTo=(message,user)=>!!message&&!!user&&(user.isGM===true||!message.blind&&(!message.whisper?.length||message.whisper.includes(user.id)||author(message)===user.id));
const tokenUuid=s=>s?.scene&&s?.token?`Scene.${s.scene}.Token.${s.token}`:null;
const normalized=s=>s?.replace(/(\.conditions)\.(?!Item\.)/,'$1.Item.');
export const electricityEffects=(actor,source)=>values(actor?.items).filter(i=>normalized(sourceUuid(i))===source);
export const electricityState=actor=>copy(actor?.flags?.[ID]?.electricity??{version:1,damage:{},operations:{},pendingShocks:{}});
const liveToken=t=>t?.documentName==='Token'&&t.actor&&t.parent?.tokens?.get(t.id)===t;
const ownCharge=i=>i?.flags?.[ID]?.electricityCharge?.ownPower===true;
const shell=actor=>values(actor.items).some(i=>sourceUuid(i)===S.shell);
const siphon=r=>r?.snapshot?.siphon?.applies===true;
const frame=combat=>combat?.started?`${combat.id}:${combat.round}:${combat.turn}`:null;
/** Viewed combat is UI state. Only one actual started membership can provide
 * initiative timing; absent or ambiguous membership has no automated clock. */
export function electricityEncounter(game,actorUuid,tokenUuid=null){
 const combats=game.combats?values(game.combats):game.combat?[game.combat]:[];
 const candidates=combats.filter(c=>c.started&&values(c.combatants??c.turns).some(t=>t.actor?.uuid===actorUuid&&(!tokenUuid||t.token?.uuid===tokenUuid)));
 return candidates.length===1?candidates[0]:null;
}

export function classifyElectricityDamage(roll){
 const instances=roll?.instances;if(!Array.isArray(instances))return 'none';
 const direct=instances.filter(i=>!i.persistent&&Number(i.total)>0);
 if(!direct.some(i=>i.type==='electricity'))return 'none';
 return direct.every(i=>i.type==='electricity')?'pure':'mixed';
}
export function electricityRemovalPlan({charged=0,ownPowerCharge=false,shell=false}={}){
 if(charged>0&&!(shell&&ownPowerCharge))return {charged:Math.max(0,charged-1),removeShocked:false};
 return {charged,removeShocked:true};
}
export function sourceTurnExpiry(combat,actorUuid,{rounds=1,phase='end',tokenUuid=null}={}){
 const index=combat?.turns?.findIndex(c=>c.actor?.uuid===actorUuid&&(!tokenUuid||c.token?.uuid===tokenUuid))??-1;
 if(!combat?.started||index<0)return null;
 return {combatId:combat.id,combatantId:combat.turns[index].id,round:combat.round+(index<=combat.turn?rounds:rounds-1),phase};
}
export function expiryReached(expiry,combat,combatant,phase){
 return !!expiry&&combat?.id===expiry.combatId&&combatant?.id===expiry.combatantId&&
  (combat.round>expiry.round||combat.round===expiry.round&&phase===expiry.phase);
}
export function chainEligibility({triggerDamage,enemy,shocked,hitBySameEffect,reactionAvailable,discharge=false,siphoning=false}){
 return Number.isFinite(triggerDamage)&&triggerDamage>0&&enemy===true&&hitBySameEffect===false&&reactionAvailable===true&&(shocked===true||discharge===true&&!siphoning);
}
/** This declaration concerns the shield's real AC contribution, not guessed
 * arithmetic. Native already establishes a failed melee attack and its target. */
export function electricityShieldAttack(message,{actor,sourceTokenUuid,item=message?.item}={}){
 const pf=message?.flags?.pf2e,c=pf?.context,target=c?.target,roll=message?.rolls?.[0];
 return message?.isCheckRoll===true&&roll?._evaluated===true&&Number.isFinite(roll.total)&&c?.type==='attack-roll'&&c.outcome==='failure'&&
  target?.actor===actor?.uuid&&target?.token===sourceTokenUuid&&(item?.isMelee===true||(c.options??[]).some(o=>['item:melee','attack:melee','melee'].includes(o)));
}
export function electricityShieldActivity(actor,game){
 const state=actor?.flags?.[ID]?.electricity,record=state?.basicActions?.[state?.activeShield];if(record?.kind!=='shield'||record.status!=='armed')return null;
 const effect=actor.items?.get(record.effectId);if(!effect||sourceUuid(effect)!==ELECTRICITY_BASIC_SOURCES.shieldEffect||effect.isExpired===true||
  effect.system?.context?.origin?.actor!==actor.uuid||effect.system?.context?.origin?.item!==record.itemUuid)return null;
 const expiry=record.expires;if(expiry){const combat=game.combats?.get?.(expiry.combatId)??(game.combat?.id===expiry.combatId?game.combat:null),index=combat?.turns?.findIndex(t=>t.id===expiry.combatantId)??-1;
  if(!combat?.started||index<0||combat.round>expiry.round||combat.round===expiry.round&&combat.turn>=index)return null;}
 return record;
}
export function receiptMatches(message,record){
 const pf=message?.flags?.pf2e,options=pf?.context?.options??[];
 return !!message?.id&&author(message)===record.userId&&message.speaker?.actor===record.actorUuid.split('.').at(-1)&&tokenUuid(message.speaker)===record.tokenUuid&&
  pf?.context?.type==='damage-taken'&&options.filter(o=>typeof o==='string'&&o.startsWith(ELECTRICITY_APPLY_PREFIX)).length===1&&options.includes(ELECTRICITY_APPLY_PREFIX+record.nonce)&&
  (pf.origin?.uuid??null)===record.sourceItemUuid&&(!pf.appliedDamage||pf.appliedDamage.uuid===record.actorUuid&&!pf.appliedDamage.isHealing&&!pf.appliedDamage.isReverted);
}
export function receiptElectricityAmount(message,record){
 if(record.kind!=='pure'||!receiptMatches(message,record))return null;
 const fact=message.flags?.[ID]?.electricityApplied;
 return fact?.nonce===record.nonce&&Number.isFinite(fact.amount)&&fact.amount>=0?fact.amount:null;
}
const fingerprint=m=>JSON.stringify({author:author(m),speaker:m.speaker,pf:m.flags?.pf2e,electricity:m.flags?.[ID]?.electricityApplied,shieldBlock:m.flags?.[ID]?.shieldBlock});
// Toolbelt may attach the original template's targets in preCreate, after the
// DamageRoll publisher ran. The persisted native card is the final authority.
const sourceTargets=m=>m.flags?.['pf2e-toolbelt']?.targetHelper?.targets??m.flags?.[ID]?.electricitySource?.targetUuids??[];
/** DSN 6.3.1 ThrowPipeline stamps Die-only role options; DiceNotation attaches
 * the animation throw index to that Die's direct results. Walk only native
 * Roll/term children, never arbitrary options or flag metadata. Type, flavor,
 * formula, totals, result states and every other option remain evidence. */
export function electricityRollWitness(rollJSON){
 const roll=copy(rollJSON);
 const cleanRoll=value=>{
  if(!value||typeof value.class!=='string'||!value.class.endsWith('Roll')&&value.class!=='DamageInstance')return value;
  if(Array.isArray(value.terms))value.terms=value.terms.map(cleanTerm);return value;
 };
 const cleanTerm=value=>{
  if(!value||typeof value.class!=='string')return value;
  if(value.class==='Die'){
   if(value.options&&typeof value.options==='object'&&!Array.isArray(value.options)){delete value.options.dsnRole;delete value.options.dsnRoleManaged;}
   if(Array.isArray(value.results))for(const result of value.results)if(result&&typeof result==='object'&&!Array.isArray(result))delete result.indexThrow;
  }
  for(const field of ['terms','operands'])if(Array.isArray(value[field]))value[field]=value[field].map(cleanTerm);
  if(Array.isArray(value.rolls))value.rolls=value.rolls.map(cleanRoll);
  if(value.term&&typeof value.term==='object')value.term=cleanTerm(value.term);
  return value;
 };
 return cleanRoll(roll);
}
const sourceWitness=data=>({...data,...(Array.isArray(data?.rolls)?{rolls:data.rolls.map(electricityRollWitness)}:{})});
const sourceFingerprint=m=>JSON.stringify(sourceWitness({pf:m.flags?.pf2e,source:m.flags?.[ID]?.electricitySource,rolls:m.rolls?.map(r=>r.toJSON?.()??{options:r.options,instances:r.instances})}));
const sameSourceFingerprint=(message,saved)=>{
 // Existing records retain their raw proof. Normalize only its dice display
 // witness at comparison time, so pre-fix receipts need no migration or replay.
 try{return sourceFingerprint(message)===JSON.stringify(sourceWitness(JSON.parse(saved)));}catch{return false;}
};

/** Active-GM ledger. Document effects retain their published rules and GrantItem
 * links. An interrupted mutation is never inferred from unrelated HP changes. */
export function createElectricityLedger({game,reactionRestriction,fromUuid,receiptMessages=()=>[],queue=new SerialActions(),reactionAvailable=genericReactionAvailable}={}){
 // The provider supplies the live hook-maintained receipt projection. Without
 // it, native receipt settlement fails closed; ordinary actions never scan chat.
 const gm=()=>{if(game.user?.id!==game.users.activeGM?.id)throw Error('电元素结算需要当前主 GM。');};
 const owner=(actor,user)=>{gm();if(!user||game.users.get(user.id)!==user||!actor?.testUserPermission?.(user,'OWNER'))throw Error('需要当前角色的拥有者权限。');};
 const save=async(actor,state,options={})=>{
  gm();const path=`flags.${ID}.electricity`,update={[path]:state};
  // Native updates recursively merge: omission alone cannot consume or expire
  // a pending effect. Delete only removed entries inside this provider's map.
  for(const key of Object.keys(actor.flags?.[ID]?.electricity?.pendingShocks??{}))if(!Object.hasOwn(state.pendingShocks,key))update[`${path}.pendingShocks.-=${key}`]=null;
  await actor.update(update,options);
 };
 const mutate=(actor,fn)=>queue.run(actor.uuid,async()=>{gm();return fn(electricityState(actor));});
 async function original(payload,user){
  const actor=await fromUuid(payload.actorUuid);owner(actor,user);
  const r=actor.flags?.[ID]?.metapower?.receipts?.[payload.nonce],m=await fromUuid(payload.messageUuid),item=await fromUuid(r?.itemUuid);
  if(!r||r.status!=='committed'||r.messageUuid!==m?.uuid||game.messages.get(m?.id)!==m||r.actorUuid!==actor.uuid||r.userId!==user.id||author(m)!==user.id||m.speaker?.actor!==actor.id||
   m.flags?.[ID]?.metapowerUse?.nonce!==r.nonce||m.flags?.pf2e?.origin?.uuid!==item?.uuid||item?.actor!==actor||actor.items.get(item.id)!==item||sourceUuid(item)!==r.sourceUuid)throw Error('需要原本已完成支付的电元素威能使用记录。');
  return {actor,receipt:r,message:m,item};
 }
 async function effectSource(source,flags,context){
  const template=await fromUuid(source);if(!template||template.type!=='effect')throw Error('无法取得原始电元素效果，请检查爱达梦规则包。');
  const data=template.toObject();delete data._id;data._stats={...data._stats,compendiumSource:source};
  data.flags={...data.flags,[ID]:{...data.flags?.[ID],...flags}};data.system.context=context;return data;
 }
 async function charge(actor,state,key,{gain=false,clear=false,context=null}={}){
  let op=state.operations[key];if(op?.status==='done')return;
  const effects=electricityEffects(actor,S.charged).filter(i=>i.system.badge?.value>0);if(effects.length>1)throw Error('存在多个蓄电效果，请由 GM 核对。');
  let item=effects[0];
  // The embedded effect mutation refreshes its UI; these two writes only journal it.
  if(!op){
   const before=item?.system.badge?.value??0,after=clear?0:gain?Math.min(3,before+1):Math.max(0,before-1);
   op=state.operations[key]={status:'started',itemId:item?.id??null,before,after,gain};await save(actor,state,{render:false});
  }
  const proof=item?.flags?.[ID]?.electricityCharge;
  if(proof?.operation!==key){
   if(op.itemId&&(!item||item.id!==op.itemId||item.system.badge.value!==op.before))throw Error('蓄电变更被中断，请由 GM 核对原操作。');
   if(!op.itemId&&item)throw Error('记录蓄电增加时，原蓄电效果发生了变化。请由 GM 核对。');
   if(op.after>0&&!item){
    const data=await effectSource(S.charged,{electricityCharge:{operation:key,ownPower:true}},context);data.system.badge.value=op.after;
    gm();await actor.createEmbeddedDocuments('Item',[data]);
   }else if(item&&op.before!==op.after){
    gm();await item.update({'system.badge.value':op.after,[`flags.${ID}.electricityCharge`]:{operation:key,ownPower:gain?(op.before===0||ownCharge(item)):ownCharge(item)}});
   }
  }
  op.status='done';await save(actor,state,{render:false});
 }
 async function remove(actor,state,key,{onlyIds=null}={}){
  if(state.operations[key]?.status==='done')return;
  const charged=electricityEffects(actor,S.charged).find(i=>i.system.badge?.value>0),plan=electricityRemovalPlan({charged:charged?.system.badge?.value??0,ownPowerCharge:ownCharge(charged),shell:shell(actor)});
  if(charged&&plan.charged<charged.system.badge.value){await charge(actor,state,key);return;}
  const ids=electricityEffects(actor,S.shocked).filter(i=>!i.flags?.pf2e?.grantedBy?.id&&(!onlyIds||onlyIds.includes(i.id))).map(i=>i.id);
  state.operations[key]={status:'started',deleteIds:ids};await save(actor,state);
  if(ids.length){gm();await actor.deleteEmbeddedDocuments('Item',ids);}
  state.operations[key].status='done';await save(actor,state);
 }
 async function applyShock(actor,state,pending){
  const key=`shock:${pending.key}`;if(state.operations[key]?.status==='done')return;
  const existing=electricityEffects(actor,S.shocked).find(i=>pending.identity?i.flags?.[ID]?.electricityShock?.identity===pending.identity:i.flags?.[ID]?.electricityShock?.key===pending.key);
  if(pending.identity){
   const operation=state.operations[key],proof=existing?.flags?.[ID]?.electricityShock;
   if(operation?.status==='started'&&proof?.key!==pending.key&&(!existing||existing.id!==operation.beforeId||(proof?.key??null)!==operation.beforeKey))throw Error('原带电效果的结算结果不确定或已被后续活动替代，请由 GM 核对；不会重新施加。');
   if(!operation){state.operations[key]={status:'started',beforeId:existing?.id??null,beforeKey:proof?.key??null};await save(actor,state,{render:false});}
  }
  const duration=pending.nativeDuration??{value:-1,unit:'unlimited',expiry:null,sustained:false};
  if(!existing){
   const data=await effectSource(S.shocked,{electricityShock:copy(pending)},pending.context);
   // The published gate protects Shocked granted by one's own Charged parent.
   // A separately inflicted hostile copy still penalizes electricity saves.
   if(pending.sourceActorUuid!==actor.uuid)for(const rule of data.system.rules??[]){
    if(rule.key==='FlatModifier'&&rule.value===-2&&rule.selector?.length===2&&rule.selector.includes('fortitude')&&rule.selector.includes('reflex')&&
     JSON.stringify(rule.predicate)===JSON.stringify([{and:['electricity',{nor:['resistant-shell']}]}]))rule.predicate=['electricity'];
   }
   // The original unlimited effect must not expire on the recipient's initiative.
   data.system.duration=copy(duration);
   if(pending.identity)data.system.start=copy(pending.nativeStart??{value:game.time?.worldTime??0,initiative:null});
   gm();await actor.createEmbeddedDocuments('Item',[data]);
  }else if(pending.identity&&existing.flags[ID].electricityShock.key!==pending.key){
   gm();await existing.update({[`flags.${ID}.electricityShock`]:copy(pending),'system.context':pending.context,'system.duration':copy(duration),'system.start':copy(pending.nativeStart??{value:game.time?.worldTime??0,initiative:null})});
  }
  state.operations[key]={status:'done'};delete state.pendingShocks[pending.key];await save(actor,state);
 }
 async function validateSource(record){
  const message=await fromUuid(record.sourceMessageUuid),roll=message?.rolls?.[record.rollIndex];
  if(game.messages.get(message?.id)!==message||message.flags?.pf2e?.context?.type!=='damage-roll'||roll?.options?.[ID]?.electricitySource?.nonce!==record.sourceNonce||
   classifyElectricityDamage(roll)!==record.kind||message.flags?.[ID]?.electricitySource?.effectKey!==record.effectKey||record.sourceFingerprint&&!sameSourceFingerprint(message,record.sourceFingerprint)||Object.hasOwn(record,'sourceTargets')&&JSON.stringify(sourceTargets(message))!==JSON.stringify(record.sourceTargets))throw Error('原电击伤害来源已改变或已失效。');
  return message;
 }
 async function verified(record){
  try{await validateSource(record);}catch{return false;}const m=await fromUuid(record.receiptUuid);
  return !!m&&game.messages.get(m.id)===m&&receiptMatches(m,record)&&fingerprint(m)===record.fingerprint;
 }
 async function chainFacts({actor,selection,kind,user},record){
  const evidence=selection.electricityEvidence,source=await fromUuid(record.tokenUuid),target=await fromUuid(evidence?.targetUuid),caster=await fromUuid(evidence?.sourceTokenUuid);
  const sourceCombat=electricityEncounter(game,source?.actor?.uuid,source?.uuid),casterCombat=electricityEncounter(game,actor.uuid,caster?.uuid);
  if(!liveToken(source)||!liveToken(target)||!liveToken(caster)||caster.actor.uuid!==actor.uuid||source.parent!==target.parent||target.parent!==caster.parent||source.uuid===target.uuid||
   !sourceCombat||casterCombat?.id!==sourceCombat.id||selection.targetUuids?.length!==1||selection.targetUuids[0]!==target.uuid||record.frame!==frame(sourceCombat)||record.status!=='confirmed'||record.receiptUuid!==evidence.receiptUuid||record.effectKey!==evidence.effectKey||!await verified(record))return false;
  const all=values(source.parent.tokens).flatMap(t=>Object.values(electricityState(t.actor).damage));
  const sourceMessage=await fromUuid(record.sourceMessageUuid),receiptMessage=await fromUuid(record.receiptUuid);
  if(!visibleTo(sourceMessage,user)||!visibleTo(receiptMessage,user))return false;
  const manifest=Object.hasOwn(record,'sourceTargets')?sourceTargets(sourceMessage):sourceMessage.flags?.[ID]?.electricitySource?.targetUuids??[];
  // Reserve original targets until every observed application to that target
  // has an unchanged authentic zero receipt. Pending/mixed/positive evidence
  // never releases the reservation, including a later sibling application.
  const applications=all.filter(r=>r.effectKey===record.effectKey&&r.tokenUuid===target.uuid);
  const hit=applications.length?!(await Promise.all(applications.map(async r=>r.status==='confirmed'&&r.electricityAmount===0&&await verified(r)))).every(Boolean):manifest.includes(target.uuid);
  return chainEligibility({triggerDamage:record.electricityAmount,enemy:!!actor.alliance&&!!target.actor.alliance&&actor.alliance!==target.actor.alliance,
   shocked:electricityEffects(target.actor,S.shocked).length>0,hitBySameEffect:hit,reactionAvailable:reactionPermitted(actor,reactionRestriction)&&reactionAvailable(actor,{combat:casterCombat,modules:game.modules,messages:game.messages,users:game.users},{reactionRestriction}),discharge:selection.discharge,siphoning:kind==='siphoning'});
 }
 async function basicOriginal(payload,user,{actualUse=false}={}){
  if(!electricityBasicSettlementEnabled(game))throw Error('当前世界、系统或 Eldamon 模组不支持此结算入口。');
  const actor=await fromUuid(payload.actorUuid);owner(actor,user);
  const item=await fromUuid(payload.itemUuid),message=await fromUuid(payload.messageUuid),kind=electricityBasicAction(item),pf=message?.flags?.pf2e;
  const originMatches=pf?.origin?.uuid===item?.uuid||pf?.context?.type==='self-effect'&&pf.context.item===item?.id;
  const originalUser=game.users.get(author(message));
  const sourceTokenUuid=tokenUuid(message?.speaker),sourceToken=sourceTokenUuid?await fromUuid(sourceTokenUuid):null;
  if(!kind||item.actor!==actor||actor.items.get(item.id)!==item||game.messages.get(message?.id)!==message||!electricityBasicCardType(message)||!originMatches||
   message.speaker?.actor!==actor.id||sourceTokenUuid&&(!liveToken(sourceToken)||sourceToken.actor.uuid!==actor.uuid)||!originalUser||!actor.testUserPermission(originalUser,'OWNER')||actualUse&&author(message)!==user.id||!isActualUseMessage(message))throw Error('需要原操作者实际使用的元素动作卡。');
  return {actor,item,message,kind,key:`use-${message.id}`};
 }
 async function settleBasic({actor,item,message,kind},target,keySuffix='',binding){
  const sourceToken=tokenUuid(message.speaker),expires=binding.shockExpires;
  const pending={key:`basic-${message.id}${keySuffix}`,identity:`basic:${actor.uuid}:${kind}`,sourceActorUuid:actor.uuid,sourceMessageUuid:message.uuid,expires,
   ...!expires&&kind==='manipulation'?{nativeDuration:{value:2,unit:'rounds',expiry:'turn-start',sustained:false}}:{},
   nativeStart:{value:binding.worldTime,initiative:null},context:{origin:{actor:actor.uuid,token:sourceToken,item:item.uuid,spellcasting:null,rollOptions:[]},target:{actor:target.actor.uuid,token:target.uuid},roll:null}};
  await mutate(target.actor,state=>applyShock(target.actor,state,pending));return {targetUuid:target.uuid,kind,manualExpiry:!expires&&kind==='shield'};
 }
 return {
  async basicUse(payload,user){
   const context=await basicOriginal(payload,user,{actualUse:true}),{actor,message,kind,key}=context;
   const admitted=await mutate(actor,async state=>{
    state.basicActions??={};const old=state.basicActions[key];if(old)return copy(old);
    const sourceToken=tokenUuid(message.speaker),combat=electricityEncounter(game,actor.uuid,sourceToken);
    const shieldEffect=kind==='shield'?electricityEffects(actor,ELECTRICITY_BASIC_SOURCES.shieldEffect).findLast(effect=>!effect.isExpired&&effect.system?.context?.origin?.actor===actor.uuid&&effect.system?.context?.origin?.item===context.item.uuid):null;
    if(kind==='shield'&&!shieldEffect)throw Error('原生元素护盾效果尚未启用；请使用角色拥有的元素护盾动作。');
    const targets=message.flags?.[ID]?.usageInput?.targetUuids??[];
    if(kind==='manipulation'&&targets.length>1)throw Error('元素操控施加带电时，请只锁定一个目标。');
    const target=targets.length===1?await fromUuid(targets[0]):null;if(kind==='manipulation'&&targets.length===1&&!liveToken(target))throw Error('本次元素操控捕获的目标已失效。');
    const record={key,kind,status:kind==='shield'?'armed':target?'started':'done',messageUuid:message.uuid,itemUuid:context.item.uuid,userId:user.id,sourceTokenUuid:sourceToken,createdAt:message.timestamp,
     targetUuid:kind==='manipulation'?target?.uuid??null:null,targetActorUuid:kind==='manipulation'?target?.actor.uuid??null:null,effectId:shieldEffect?.id??null,worldTime:game.time?.worldTime??0,
     shockExpires:kind==='manipulation'?sourceTurnExpiry(combat,actor.uuid,{rounds:2,phase:'start',tokenUuid:sourceToken}):null,expires:kind==='shield'?sourceTurnExpiry(combat,actor.uuid,{phase:'start',tokenUuid:sourceToken}):null};
    if(kind==='shield'){const previous=state.basicActions[state.activeShield];if(previous?.status==='armed')previous.status='replaced';state.activeShield=key;}
    state.basicActions[key]=record;await save(actor,state,{render:kind==='shield'});return copy(record);
   });
   if(kind==='shield')return {status:'done',result:'元素护盾已启用；实际触发时可从攻击卡或角色动作结算。'};
   if(admitted.status==='done')return {status:'done',result:admitted.targetUuid?'本次元素操控已结算。':'已使用元素操控；没有锁定带电目标，其余操控效果由 GM 裁定。'};
   const target=await fromUuid(admitted.targetUuid);if(!liveToken(target)||target.actor.uuid!==admitted.targetActorUuid)throw Error('已记录的元素操控目标已失效，请由 GM 核对本次活动。');
   await settleBasic(context,target,'',admitted);await mutate(actor,async state=>{state.basicActions[key].status='done';await save(actor,state,{render:false})});
   return {status:'done',result:'已对本次锁定目标施加两轮带电。'};
  },
  /** Explicit native-card settlement. Position and the shield trigger are player/
   * GM facts; only the selected card, owner and recipient are resolved here. */
  async confirmedAction(payload,user){
   let target,context,triggerKey,attackBinding;
   try{
    context=await basicOriginal(payload,user);target=await fromUuid(payload.targetUuid);
    if(payload.confirmed!==true||!/^[A-Za-z0-9_-]{8,100}$/.test(payload.nonce??''))throw Error('请声明本次实际元素触发。');
    if(context.kind==='shield'){
     const shield=electricityShieldActivity(context.actor,game);if(shield?.key!==context.key)throw Error('原元素护盾已到期或已被新的护盾替代。');
     if(payload.triggerMessageUuid){const attack=await fromUuid(payload.triggerMessageUuid),weapon=attack?.item??await fromUuid(attack?.flags?.pf2e?.origin?.uuid);
      if(game.messages.get(attack?.id)!==attack||!Number.isFinite(shield.createdAt)||!Number.isFinite(attack.timestamp)||attack.timestamp<shield.createdAt||
       !electricityShieldAttack(attack,{actor:context.actor,sourceTokenUuid:shield.sourceTokenUuid,item:weapon}))throw Error('需要此护盾启用后、以本角色为目标的实际近战未命中卡。');
      target=await fromUuid(tokenUuid(attack.speaker));triggerKey=`attack-${attack.id}`;
      const attackUser=game.users.get(author(attack));if(!liveToken(target)||!attackUser||!target.actor.testUserPermission(attackUser,'OWNER')||weapon?.actor?.uuid!==target.actor.uuid||attack.flags.pf2e.origin?.uuid!==weapon?.uuid)throw Error('原近战攻击的操作者、武器或场景来源已改变。');
      if(!visibleTo(attack,user))throw Error('此原近战检定对当前操作者不可见，请由 GM 结算。');
      const prefix=`${ID}:electricity-shield:${context.key}:`,options=attack.flags.pf2e.context.options??[],tags=options.filter(option=>typeof option==='string'&&option.startsWith(prefix));
      if(tags.length>1)throw Error('原护盾攻击的重掷来源不明确，请由 GM 核对。');
      const originalId=tags[0]?.slice(prefix.length)??attack.id,prior=shield.triggers?.[`attack-${originalId}`],reroll=attack.flags.pf2e.context.isReroll===true;
      if(tags.length&&originalId!==attack.id&&(!reroll||!options.includes('check:reroll')||!prior||game.messages.get(prior.latestMessageId)&&prior.latestMessageId!==attack.id||prior.attackActorUuid!==target.actor.uuid||prior.attackItemUuid!==weapon.uuid||prior.targetUuid!==target.uuid))throw Error('需要原攻击删除后保存的同一原生重掷卡。');
      triggerKey=`attack-${originalId}`;attackBinding={attackActorUuid:target.actor.uuid,attackItemUuid:weapon.uuid,latestMessageId:attack.id};
      if(!tags.length){if(typeof attack.update!=='function')throw Error('原生攻击卡不能保存护盾来源，请由 GM 核对。');gm();await attack.update({'flags.pf2e.context.options':[...options,prefix+originalId]});}
     }else triggerKey=`declared-${payload.nonce}`;
     if(!liveToken(target)||target.actor.uuid===context.actor.uuid||!context.actor.alliance||context.actor.alliance===target.actor.alliance)throw Error('请选择实际因元素护盾而未命中的敌人。');
    }
   }catch(error){
    // Only validation is inside this boundary. A mutation/transport failure
    // below must keep its original nonce because an effect may already exist.
    error.electricityNotApplied=true;throw error;
   }
   if(context.kind==='manipulation')return this.basicUse(payload,game.users.get(author(context.message)));
   const admitted=await mutate(context.actor,async state=>{const record=state.basicActions[context.key];record.triggers??={};const old=record.triggers[triggerKey];if(old){if(attackBinding&&old.latestMessageId!==attackBinding.latestMessageId){Object.assign(old,attackBinding);await save(context.actor,state,{render:false})}return copy(old);}
    const entry={targetUuid:target.uuid,targetActorUuid:target.actor.uuid,...attackBinding,status:'started',worldTime:game.time?.worldTime??0,shockExpires:sourceTurnExpiry(electricityEncounter(game,context.actor.uuid,record.sourceTokenUuid),context.actor.uuid,{tokenUuid:record.sourceTokenUuid})};record.triggers[triggerKey]=entry;await save(context.actor,state,{render:false});return copy(entry)});
   if(admitted.status==='done')return {kind:'shield',targetUuid:admitted.targetUuid,replayed:true};
   target=await fromUuid(admitted.targetUuid);if(!liveToken(target)||target.actor.uuid!==admitted.targetActorUuid)throw Error('原护盾触发目标已失效，请由 GM 核对。');
   const result=await settleBasic(context,target,`-${triggerKey}`,admitted);
   await mutate(context.actor,async state=>{state.basicActions[context.key].triggers[triggerKey].status='done';await save(context.actor,state,{render:false})});return result;
  },
  async channel(payload,user){const {actor,receipt:r,message,item}=await original(payload,user);
   if(siphon(r)||r.selection?.discharge||![S.surge,S.anvil,S.static,S.shot].includes(r.sourceUuid))return;
   return mutate(actor,state=>charge(actor,state,`channel:${r.nonce}`,{gain:true,context:{origin:{actor:actor.uuid,token:tokenUuid(message.speaker),item:item.uuid,spellcasting:null,rollOptions:[]},target:null,roll:null}}));
  },
  async beginDamage(payload,user){
   const actor=await fromUuid(payload.actorUuid),token=await fromUuid(payload.tokenUuid);owner(actor,user);
   if(!liveToken(token)||token.actor.uuid!==actor.uuid||!/^[A-Za-z0-9_-]{8,100}$/.test(payload.nonce??'')||!['pure','mixed'].includes(payload.kind))throw Error('本次原生电击伤害应用记录无效。');
   const source=await validateSource(payload);
   return mutate(actor,async state=>{if(state.damage[payload.nonce])throw Error('本次电击伤害应用已记录，不能重复提交。');
    const r={...copy(payload),sourceFingerprint:sourceFingerprint(source),sourceTargets:copy(sourceTargets(source)),userId:user.id,status:'pending',frame:frame(electricityEncounter(game,actor.uuid,token.uuid))};state.damage[r.nonce]=r;await save(actor,state);return copy(r);});
  },
  async finishDamage(payload,user){
   const actor=await fromUuid(payload.actorUuid);owner(actor,user);
   return mutate(actor,async state=>{
    const r=state.damage[payload.nonce];if(!r||r.userId!==user.id)throw Error('本次电击伤害应用与原操作者不符。');
    if(r.status==='confirmed'&&r.lifecycleDone)return copy(r);
    const m=await fromUuid(payload.receiptUuid);await validateSource(r);
    const matches=receiptMessages(r.nonce).filter(x=>game.messages.get(x?.id)===x&&receiptMatches(x,r));
    if(game.messages.get(m?.id)!==m||!receiptMatches(m,r)||matches.length!==1||matches[0]!==m)throw Error('需要唯一且可核验的原生电击伤害应用回执。');
    r.receiptUuid=m.uuid;r.fingerprint=fingerprint(m);r.electricityAmount=r.attribution?.amount??receiptElectricityAmount(m,r);r.status=r.electricityAmount===null?'needs-attribution':'confirmed';
    await save(actor,state);
    if(r.electricityAmount>0)await remove(actor,state,`damage:${r.nonce}`);
    for(const pending of Object.values(state.pendingShocks))if(pending.effectKey===r.effectKey)await applyShock(actor,state,pending);
    r.lifecycleDone=true;await save(actor,state);
    return copy(r);
   });
  },
  async confirmMixed(payload,user){
   gm();if(user!==game.users.activeGM)throw Error('混合伤害的电击分量只能由当前主 GM 确认。');
   const actor=await fromUuid(payload.actorUuid);
   return mutate(actor,async state=>{const r=state.damage[payload.nonce];
    if(!r||r.kind!=='mixed'||r.status!=='needs-attribution'||!await verified(r)||payload.receiptUuid!==r.receiptUuid||payload.confirmed!==true||!Number.isFinite(payload.amount)||payload.amount<0)throw Error('请为这张原混合伤害回执确认实际电击分量。');
    const native=(await fromUuid(r.receiptUuid)).flags?.[ID]?.electricityApplied;
    if(native?.nonce!==r.nonce||!Number.isFinite(native.amount)||payload.amount>native.amount)throw Error('电击分量不能超过本次原生实际伤害总量。');
    r.electricityAmount=payload.amount;r.status='confirmed';r.lifecycleDone=false;r.attribution={userId:user.id,amount:payload.amount};await save(actor,state);
    if(r.electricityAmount>0)await remove(actor,state,`damage:${r.nonce}`);r.lifecycleDone=true;await save(actor,state);return copy(r);
   });
  },
  async check(message){
   gm();const c=message?.flags?.pf2e?.context;if(game.messages.get(message?.id)!==message||!message.isCheckRoll||!message.rolls?.[0]?._evaluated)return;
   const markers=(c?.options??[]).filter(o=>o.startsWith(`${ID}:metapower:`));if(markers.length!==1)return;
   const [cardId,nonce]=markers[0].slice(`${ID}:metapower:`.length).split(':'),card=game.messages.get(cardId),sourceActor=await fromUuid(card?.flags?.[ID]?.metapowerUse?.actorUuid);
   const r=sourceActor?.flags?.[ID]?.metapower?.receipts?.[nonce];
   if(!r||r.status!=='committed'||r.messageUuid!==card?.uuid||message.flags?.pf2e?.origin?.uuid!==r.itemUuid||siphon(r))return;
   const anvil=r.sourceUuid===S.anvil&&c.type==='saving-throw',staticShock=r.sourceUuid===S.static&&c.type==='attack-roll';if(!anvil&&!staticShock)return;
   const target=await fromUuid(anvil?tokenUuid(message.speaker):c.target?.token),source=await fromUuid(tokenUuid(card.speaker));
   if(!liveToken(target)||!liveToken(source)||target.parent!==source.parent||staticShock&&!r.selection?.targetUuids?.includes(target.uuid))return;
   if(anvil&&(!sourceActor.alliance||!target.actor.alliance||sourceActor.alliance===target.actor.alliance))return;
   const outcome=c.outcome,qualifies=anvil?['failure','criticalFailure'].includes(outcome):['criticalSuccess','success','failure'].includes(outcome);
   const effectKey=`channel:${sourceActor.uuid}:${nonce}`;
   // Foundry expands dots in nested update keys. Keep UUIDs as values and use
   // a reversible flat key for pending entries and their shock operations.
   const key=encodeURIComponent(`${effectKey}:${target.uuid}`).replaceAll('.','%2E');
   return mutate(target.actor,async state=>{
    if(!qualifies){delete state.pendingShocks[key];await save(target.actor,state);return;}
    const combat=electricityEncounter(game,sourceActor.uuid,source.uuid),expires=sourceTurnExpiry(combat,sourceActor.uuid,{rounds:staticShock&&outcome==='criticalSuccess'?2:1,tokenUuid:source.uuid});
    if(!expires)return; // No invented initiative clock outside an encounter.
    const pending={key,effectKey,sourceActorUuid:sourceActor.uuid,checkUuid:message.uuid,expires,context:{origin:{actor:sourceActor.uuid,token:source.uuid,item:r.itemUuid,spellcasting:null,rollOptions:[]},target:{actor:target.actor.uuid,token:target.uuid},roll:{total:message.rolls[0].total,degreeOfSuccess:message.rolls[0].degreeOfSuccess}}};
    state.pendingShocks[key]=pending;await save(target.actor,state);
    if(Object.values(state.damage).some(d=>d.effectKey===effectKey&&d.receiptUuid))await applyShock(target.actor,state,pending);
   });
  },
  async validateSelection({actor,item,selection,kind,user}){
   if(sourceUuid(item)!==S.chain)return;owner(actor,user);
   const evidence=selection.electricityEvidence,recipient=await fromUuid(evidence?.actorUuid),r=electricityState(recipient).damage[evidence?.nonce];
   if(!r||!await chainFacts({actor,selection,kind,user},r)||selection.triggerDamage!==r.electricityAmount)throw Error('反应电链需要当前可核验的原电击回执和符合条件的目标。');
  },
  async candidates(payload,user){
   const actor=await fromUuid(payload.actorUuid);owner(actor,user);const caster=await fromUuid(payload.sourceTokenUuid);if(!liveToken(caster)||caster.actor.uuid!==actor.uuid)return [];
   const bound=payload.receiptUuid?await fromUuid(payload.receiptUuid):null,source=bound?await fromUuid(tokenUuid(bound.speaker)):null;
   if(payload.receiptUuid&&(!liveToken(source)||source.parent!==caster.parent||game.messages.get(bound?.id)!==bound))return [];
   const results=[];for(const token of source?[source]:values(caster.parent.tokens))for(const r of Object.values(electricityState(token.actor).damage)){
    if(payload.receiptUuid&&r.receiptUuid!==payload.receiptUuid)continue;
    const evidence={actorUuid:token.actor.uuid,nonce:r.nonce,receiptUuid:r.receiptUuid,effectKey:r.effectKey,targetUuid:payload.selection?.targetUuids?.[0],sourceTokenUuid:caster.uuid};
    const selection={...payload.selection,electricityEvidence:evidence};if(await chainFacts({actor,selection,kind:payload.kind,user},r))results.push({evidence,amount:r.electricityAmount,label:`${publicTargetName(token,{game,user})}：${r.electricityAmount}`});
   }return results;
  },
  async expire({combat,combatant,phase,ended=false,actors=[]}){
   gm();for(const actor of actors)await mutate(actor,async state=>{
    const expired=electricityEffects(actor,S.shocked).filter(i=>{const e=i.flags?.[ID]?.electricityShock?.expires;return ended?e?.combatId===combat.id:expiryReached(e,combat,combatant,phase);});
    const pending=Object.entries(state.pendingShocks).filter(([,p])=>ended?p.expires?.combatId===combat.id:expiryReached(p.expires,combat,combatant,phase));
    const chargeKey=`encounter:${combat.id}:charge`,operation=state.operations[chargeKey];
    const clearCharge=()=>ended&&values(combat.combatants).some(c=>c.actor?.uuid===actor.uuid)&&operation?.status!=='done'&&
     (electricityEffects(actor,S.charged).some(i=>i.system.badge?.value>0)||operation?.status==='started'&&operation.after===0&&operation.gain===false);
    // Turn hooks inspect all actors, including ordinary recipients of hostile
    // Shocked. Only actual work may create state; retain exact interrupted
    // encounter cleanup so its existing charge reconciliation still runs.
    if(!expired.length&&!pending.length&&!clearCharge())return;
    for(const item of expired){const key=`expiry:${item.id}`;await remove(actor,state,key,{onlyIds:[item.id]});if(actor.items.get(item.id)&&!item.flags?.pf2e?.grantedBy?.id)await actor.deleteEmbeddedDocuments('Item',[item.id]);}
    for(const [key]of pending)delete state.pendingShocks[key];
    if(clearCharge())await charge(actor,state,chargeKey,{clear:true});await save(actor,state);
   });
  },
  async refresh({actor,nonce}){gm();if(values(game.combats).some(c=>c.started&&values(c.combatants).some(x=>x.actor?.uuid===actor.uuid)))return;
   return mutate(actor,state=>charge(actor,state,`refresh:${nonce}`,{clear:true}));
  },
  async interact(payload,user){const actor=await fromUuid(payload.actorUuid);owner(actor,user);if(payload.confirmed!==true||!payload.nonce)throw Error('请声明已经完成本次交互动作。');return mutate(actor,state=>remove(actor,state,`interact:${payload.nonce}`));},
 };
}
