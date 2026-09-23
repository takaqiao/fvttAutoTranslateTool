import {MODULE_ID} from './rules.mjs';
import {withDamageMessageTarget} from './damage-message-targets.mjs';

const RANGED='pf2e-ranged-combat';
const SOURCE=Object.freeze({
 ranger:'Compendium.pf2e.feats-srd.Item.1JnERVwnPtX620f2',
 bear:'Compendium.pf2e-animal-companions.AC-Ancestries-and-Class.Item.eBgMfYf0PVbsGOYp',
 support:'Compendium.pf2e-animal-companions.AC-Support.Item.AvDlo1mgxXd7ZA8W',
 effect:'Compendium.pf2e-animal-companions.AC-effects.Item.zAvLdUNaJrKOMfQY',
 link:'Compendium.pf2e-ranged-combat.feats.Item.bmDVg2hU3CSAZGJ8',
 hunt:'Compendium.pf2e.actionspf2e.Item.JYi4MnsdFu618hPm',
 flurry:'Compendium.pf2e.classfeatures.Item.6v4Rj7wWfOH1882r',
});
// Verified campaign identity, additionally checked against the actual rules sources.
const KNOWN_PAIR={master:'mzV1KHpy21LSQkoo',companion:'sYTUVpyrPKhGdTX5'};
const values=c=>Array.from(c?.values?.()??c??[]);
const sourceOf=i=>i?.sourceId??i?._stats?.compendiumSource??i?.flags?.core?.sourceId;
const has=(a,s)=>values(a?.items).some(i=>sourceOf(i)===s);
const ownFlags=i=>i?.flags?.[MODULE_ID]??{};
const supports=a=>values(a.items).filter(i=>i.type==='effect'&&ownFlags(i).kind==='bear-support');
const authority=g=>g.user?.isGM===true&&g.users?.activeGM?.id===g.user.id;
const documentOf=t=>t?.document??t;

/** Reuse RangedCombat's explicit association; never infer ownership or match names. */
function linkedMaster(game,companion){
 const links=values(companion.items).filter(i=>sourceOf(i)===SOURCE.link);
 if(links.length!==1)return null;
 const flags=links[0].flags?.[RANGED],master=game.actors.get(flags?.['master-id']);
 return master&&has(master,SOURCE.ranger)&&master.flags?.[RANGED]?.animalCompanionId===companion.id
  &&flags['master-signature']===master.signature?master:null;
}

function makeTiming(game,master){
 const combat=game.combat;
 if(!combat?.started)return {combatId:null,expiresAt:game.time.worldTime+6};
 const turns=values(combat.turns),index=turns.findIndex(c=>c.actor?.uuid===master.uuid);
 if(index<0||!Number.isInteger(combat.turn)||!Number.isInteger(combat.round))throw Error('主人须在当前遭遇中，以追踪熊支援至其下一回合开始。');
 return {combatId:combat.id,combatantId:turns[index].id,endRound:combat.round+(index<=combat.turn?1:0),expiresAt:null};
}

function isExpired(flags,game,combat=game.combat){
 if(!flags.combatId)return game.time.worldTime>=flags.expiresAt;
 if(combat?.id!==flags.combatId||!combat.started)return true;
 const index=values(combat.turns).findIndex(c=>c.id===flags.combatantId);
 return index<0||combat.round>flags.endRound||(combat.round===flags.endRound&&combat.turn>=index);
}

function inReach(companion,bearToken,target){
 const origin=bearToken?.object,other=target?.object;
 if(!origin||!other||bearToken.parent?.id!==target.parent?.id||companion.isDead||companion.canAttack===false)return false;
 const reach=companion.getReach?.({action:'attack'})??companion.system?.attributes?.reach?.base;
 if(!Number.isFinite(reach)||reach<=0||typeof origin.distanceTo!=='function')return false;
 const distance=origin.distanceTo(other,{reach});
 if(!Number.isFinite(distance)||distance>reach)return false;
 // Native token distance includes occupied spaces/elevation; walls still block the attack.
 if(typeof origin.checkCollision!=='function')return false;
 return !origin.checkCollision(other.center,{origin:origin.center,type:'move',mode:'any'});
}

export function createCompanionAutomation({game,fromUuid=globalThis.fromUuid,wrapStrike,onError=()=>{}}={}){
 const queues=new Map();let registered=false,socket=null;
 const serial=(key,fn)=>{const task=(queues.get(key)??Promise.resolve()).catch(()=>{}).then(fn);queues.set(key,task);task.finally(()=>{if(queues.get(key)===task)queues.delete(key)}).catch(()=>{});return task};
 const resolveAction=item=>['action','feat'].includes(item?.type)&&sourceOf(item)===SOURCE.support?'companion:bear-support':null;

 async function maintain(actor){
  if(!authority(game))return {status:'not-authority'};
  if(![KNOWN_PAIR.master,KNOWN_PAIR.companion].includes(actor?.id))return {status:'unrelated'};
  if(!game.modules.get(RANGED)?.active)return {status:'dependency-disabled'};
  return serial('companion-repair',async()=>{
   const master=game.actors.get(KNOWN_PAIR.master),companion=game.actors.get(KNOWN_PAIR.companion);
   if(!master||!companion||!has(master,SOURCE.ranger)||!has(companion,SOURCE.bear)||!has(companion,SOURCE.support)||!has(master,SOURCE.flurry))return {status:'source-mismatch'};
   const linked=master.flags?.[RANGED]?.animalCompanionId;
   const links=values(companion.items).filter(i=>sourceOf(i)===SOURCE.link);
   if((linked&&linked!==companion.id)||links.length>1||links.some(i=>i.flags?.[RANGED]?.['master-id']!==master.id))return {status:'conflict'};
   const expected={'master-id':master.id,'master-signature':master.signature,'hunters-edge':'flurry'};
   if(!master.signature)return {status:'missing-signature'};
   const repairs=[master,companion].map(a=>({actor:a,items:values(a.items).filter(i=>sourceOf(i)===SOURCE.hunt&&Array.isArray(i.system.rules)&&i.system.rules.some(r=>r.key==='RollOption'&&r.option==='hunted-prey'&&r.toggleable===true&&r.value===true))}));
   const linkChanged=!links.length||Object.entries(expected).some(([k,v])=>links[0].flags?.[RANGED]?.[k]!==v);
   if(linked===companion.id&&!linkChanged&&!repairs.some(r=>r.items.length))return {status:'unchanged'};
   for(const repair of repairs){
    if(!ownFlags(repair.actor).companionRepairV1)await repair.actor.update({[`flags.${MODULE_ID}.companionRepairV1`]:{worldTime:game.time.worldTime,previousAssociation:repair.actor.flags?.[RANGED]??null,linkItems:repair.actor===companion?links.map(i=>i.toObject()):[],huntRules:repair.items.map(i=>({id:i.id,rules:i.system.rules}))}});
   }
   if(!links.length){
    const template=await fromUuid(SOURCE.link);if(!template)throw Error('未找到RangedCombat原生伙伴关联条目。');
    const data=template.toObject();delete data._id;
    data._stats={...data._stats,compendiumSource:SOURCE.link};
    data.flags={...data.flags,core:{...data.flags?.core,sourceId:SOURCE.link},[RANGED]:expected};
    await companion.createEmbeddedDocuments('Item',[data]);
   }else if(linkChanged)await links[0].update({[`flags.${RANGED}`]:expected});
   if(linked!==companion.id)await master.update({[`flags.${RANGED}.animalCompanionId`]:companion.id});
   // RangedCombat supplies this option per target. A legacy permanent true value
   // would incorrectly retain Flurry when the target is not the master's prey.
   for(const repair of repairs)for(const item of repair.items){const rules=structuredClone(item.system.rules);for(const r of rules)if(r.key==='RollOption'&&r.option==='hunted-prey'&&r.toggleable===true&&r.value===true)r.value=false;await item.update({'system.rules':rules});}
   return {status:'repaired',masterUuid:master.uuid,companionUuid:companion.uuid};
  });
 }

 async function executeUsage({actor,item,message,user,action}){
  if(!authority(game))throw Error('熊支援必须由当前主 GM 结算。');
  if(action!=='companion:bear-support'||resolveAction(item)!==action||item.actor?.uuid!==actor?.uuid||!actor.testUserPermission(user,'OWNER')||!has(actor,SOURCE.bear))throw Error('无权使用此熊伙伴的支援。');
  return serial(actor.uuid,async()=>{
   const master=linkedMaster(game,actor);if(!master)throw Error('熊伙伴与游侠的原生关联尚未建立或存在冲突。');
   if(actor.isDead||actor.canAttack===false)throw Error('熊伙伴当前无法提供支援。');
   if(!message?.id)throw Error('熊支援需要原始使用消息。');
   const existing=supports(actor);
   if(existing.some(i=>ownFlags(i).sourceMessageId===message.id))return '本次熊支援已启用。';
   const timing=makeTiming(game,master);
   const tokenId=message.speaker?.token,sceneId=message.speaker?.scene;
   const candidates=values(actor.getActiveTokens?.()).map(documentOf);
   const token=(tokenId&&sceneId?game.scenes?.get(sceneId)?.tokens?.get(tokenId):null)??(candidates.length===1?candidates[0]:null);
   if(!token||token.actor?.uuid!==actor.uuid)throw Error('请从场景中的熊伙伴使用支援，以确定其位置和触及。');
   const data={name:'熊支援',type:'effect',img:item.img??'icons/svg/pawprint.svg',system:{slug:'third-party-bear-support',description:{value:'主人每次打击命中熊触及内的目标时，由熊造成支援伤害；持续至主人下一回合开始。支援期间伙伴其余动作只能是为就位进行的基本移动。'},duration:{value:-1,unit:'unlimited',expiry:null,sustained:false},start:{value:game.time.worldTime,initiative:null},tokenIcon:{show:true},rules:[]},flags:{[MODULE_ID]:{kind:'bear-support',sourceMessageId:message.id,startedAt:message.timestamp??Date.now(),sourceItemUuid:item.uuid,companionTokenUuid:token.uuid,masterUuid:master.uuid,processed:[],...timing}}};
   if(existing.length){await existing[0].update(data);if(existing.length>1)await actor.deleteEmbeddedDocuments('Item',existing.slice(1).map(i=>i.id));}
   else await actor.createEmbeddedDocuments('Item',[data]);
   return '熊支援已启用，主人命中熊触及内目标时自动掷支援伤害，至主人下一回合开始结束。';
  });
 }

 async function handleAttack(message){
  if(!authority(game)||!message?.id||ownFlags(message).usageGenerated||!message.isCheckRoll)return;
  const context=message.flags?.pf2e?.context,origin=message.flags?.pf2e?.origin;
  if(context?.type!=='attack-roll'||!['success','criticalSuccess'].includes(context.outcome)||!['weapon','melee'].includes(origin?.type??message.item?.type))return;
  // Weapon skill manoeuvres and spell attacks do not trigger a Strike benefit.
  const options=context.options??[];
  if(options.some(o=>/^action:(?:grapple|shove|trip|disarm|reposition|escape)$/.test(o)))return;
  const master=game.actors.get(message.speaker?.actor),user=game.users.get(message.author?.id??message.user?.id??message.user);
  if(!master||origin?.actor!==master.uuid||!master.testUserPermission(user,'OWNER')||!context.target?.token)return;
  const companion=game.actors.get(master.flags?.[RANGED]?.animalCompanionId);
  if(!companion||linkedMaster(game,companion)?.uuid!==master.uuid)return;
  return serial(companion.uuid,async()=>{
   const effect=supports(companion)[0];if(!effect)return;
   const flags=ownFlags(effect);
   if(isExpired(flags,game)){await companion.deleteEmbeddedDocuments('Item',[effect.id]);return;}
   if(flags.masterUuid!==master.uuid||flags.processed?.includes(message.id)||!Number.isFinite(message.timestamp)||message.timestamp<flags.startedAt)return;
   const [target,bearToken]=await Promise.all([fromUuid(context.target.token),fromUuid(flags.companionTokenUuid)]);
   if(!target?.actor||target.actor.uuid!==context.target.actor||bearToken?.actor?.uuid!==companion.uuid||!inReach(companion,bearToken,target))return;
   const supportItem=values(companion.items).find(i=>sourceOf(i)===SOURCE.support);
   if(!supportItem)return;
   const DamageRoll=globalThis.CONFIG?.Dice?.rolls?.find(c=>c.name==='DamageRoll');
   if(!DamageRoll)throw Error('未找到PF2e原生DamageRoll，熊支援伤害尚未掷出。');
   const dice=supportItem.system.traits?.otherTags?.includes('support-benefit:bear')?2:1;
   // Claim before emitting: failures after message creation must never replay damage.
   await effect.update({[`flags.${MODULE_ID}.processed`]:[...(flags.processed??[]),message.id]});
   const roll=await new DamageRoll(`${dice}d8[slashing]`).evaluate();
   return roll.toMessage(withDamageMessageTarget({speaker:globalThis.ChatMessage.getSpeaker({actor:companion,token:bearToken}),flavor:'熊支援',whisper:message.whisper??[],blind:message.blind??false,flags:{[MODULE_ID]:{usageGenerated:true,kind:'bear-support-damage',supportMessageId:flags.sourceMessageId,attackMessageId:message.id},pf2e:{origin:{uuid:supportItem.uuid,type:'action',actor:companion.uuid},context:{type:'damage-roll',domains:['damage'],options:['origin:action:slug:bear-support-benefit'],target:{actor:target.actor.uuid,token:target.uuid}}}}},target.uuid));
  });
 }

 async function expire(changedCombat=null,deleted=false){
  if(!authority(game))return;
  for(const actor of values(game.actors))await serial(actor.uuid,async()=>{
   const expired=supports(actor).filter(i=>{
    const flags=ownFlags(i);
    if(changedCombat&&flags.combatId!==changedCombat.id)return false;
    if(changedCombat&&deleted)return true;
    const combat=changedCombat??game.combats?.get(flags.combatId)??(game.combat?.id===flags.combatId?game.combat:null);
    return isExpired(flags,game,combat);
   });
   if(expired.length)await actor.deleteEmbeddedDocuments('Item',expired.map(i=>i.id));
  });
 }

 const sourceReference=params=>{
  const option=values(params.rollOptions).find(o=>typeof o==='string'&&o.startsWith(`${MODULE_ID}:source:`));
  if(!option)return null;const [messageId,index]=option.slice(`${MODULE_ID}:source:`.length).split(':');
  const rollIndex=Number(index);return Number.isInteger(rollIndex)&&rollIndex>=0?{messageId,rollIndex}:null;
 };
 const attackIds=message=>values(message?.flags?.pf2e?.context?.options).filter(o=>typeof o==='string'&&o.startsWith(`${MODULE_ID}:bear-attack:`)).map(o=>o.slice(`${MODULE_ID}:bear-attack:`.length));

 async function claimApplications(payload,user){
  if(!authority(game))throw Error('熊支援需要当前主 GM。');
  const parent=game.messages.get(payload.messageId),target=await fromUuid(payload.tokenUuid);
  if(!parent?.isDamageRoll||!parent.rolls?.[payload.rollIndex]||!target?.actor||target.actor.uuid!==payload.actorUuid||!target.actor.testUserPermission(user,'OWNER'))return null;
  if(parent.flags.pf2e.context?.target?.token!==target.uuid||parent.flags.pf2e.context?.target?.actor!==target.actor.uuid)return null;
  const ids=[...new Set(attackIds(parent))];if(!ids.length)return null;
  const receipts=[];
  for(const id of ids){
   const attack=game.messages.get(id);
   const origin=attack?.flags.pf2e.origin;
   const dualReference=ownFlags(parent).dualStrike?.attacks?.find(ref=>ref.messageId===id&&ref.actorUuid===origin?.actor&&ref.weaponUuid===origin?.uuid);
   if(!attack?.isCheckRoll||origin?.actor!==parent.flags.pf2e.origin?.actor||(!dualReference&&origin?.uuid!==parent.flags.pf2e.origin?.uuid)||attack.flags.pf2e.context?.target?.token!==target.uuid)continue;
   // A fast automatic damage roll may arrive while the hit hook is still creating
   // its card. Waiting on the same companion queue resolves that race exactly.
   await handleAttack(attack);
   const cards=values(game.messages).filter(m=>ownFlags(m).kind==='bear-support-damage'&&ownFlags(m).attackMessageId===id&&m.flags.pf2e.context?.target?.token===target.uuid);
   if(cards.length!==1)continue;
   await serial(`support-card:${cards[0].id}`,async()=>{
    const card=cards[0],state=ownFlags(card).autoApplication;
    if(state&&state.status!=='pending')return;
    const receipt={nonce:globalThis.crypto.randomUUID(),cardId:card.id,messageId:parent.id,rollIndex:payload.rollIndex,tokenUuid:target.uuid,actorUuid:target.actor.uuid,userId:user.id};
    await card.update({[`flags.${MODULE_ID}.autoApplication`]:{...receipt,status:'claimed'}});
    receipts.push(receipt);
   });
  }
  return receipts.length?{claims:receipts}:null;
 }

 async function completeApplications({receipt,result},user){
  if(!authority(game))throw Error('熊支援需要当前主 GM。');
  for(const entry of receipt?.claims??[])await serial(`support-card:${entry.cardId}`,async()=>{
   const card=game.messages.get(entry.cardId),claim=ownFlags(card).autoApplication;
   if(!claim||claim.status!=='claimed'||claim.nonce!==entry.nonce||claim.userId!==user?.id||['cardId','messageId','rollIndex','tokenUuid','actorUuid'].some(k=>claim[k]!==entry[k]))return;
   const target=await fromUuid(claim.tokenUuid);
   if(!target?.actor||target.actor.uuid!==claim.actorUuid||!target.actor.testUserPermission(user,'OWNER'))return;
   if(!result?.applied){await card.update({[`flags.${MODULE_ID}.autoApplication.status`]:result?.uncertain?'uncertain':'pending'});return;}
   const supportItem=await fromUuid(card.flags.pf2e.origin.uuid);
   if(!supportItem||sourceOf(supportItem)!==SOURCE.support||!card.isDamageRoll||!card.rolls?.[0])throw Error('熊支援原生伤害来源已失效。');
   await card.update({[`flags.${MODULE_ID}.autoApplication.status`]:'applying'});
   const options=new Set([...card.flags.pf2e.context.options,`${MODULE_ID}:source:${card.id}:0`,`${MODULE_ID}:bear-auto:${claim.nonce}`]);
   try{
    const contextual=target.actor.getContextualClone([...options]);
    await contextual.applyDamage({damage:card.rolls[0].alter(1,0),token:target,item:supportItem,rollOptions:options,skipIWR:false,shieldBlockRequest:false,breakdown:['熊支援（独立来源；撤销请使用本条原生撤销）']});
    await card.update({[`flags.${MODULE_ID}.autoApplication.status`]:'done'});
   }catch(error){await card.update({[`flags.${MODULE_ID}.autoApplication.status`]:'uncertain'});throw error;}
  });
 }

 async function rpc(method,payload){
  if(authority(game))return method==='companion-support-claim'?claimApplications(payload,game.user):completeApplications(payload,game.user);
  if(!socket||!game.users.activeGM)throw Error('熊支援自动伤害需要在线主 GM。');
  const result=await socket.executeAsUser(method,game.users.activeGM.id,payload);
  if(!result?.ok)throw Error(result?.error??'熊支援自动伤害未完成。');return result.value;
 }

 async function beforeDamage(actor,params){
  const ref=sourceReference(params);if(!ref)return {params};
  const message=game.messages.get(ref.messageId);if(!message?.isDamageRoll)return {params};
  if(ownFlags(message).kind==='bear-support-damage'){
   const claim=ownFlags(message).autoApplication;
   const internal=authority(game)&&claim?.status==='applying'&&values(params.rollOptions).includes(`${MODULE_ID}:bear-auto:${claim.nonce}`)&&params.token?.uuid===claim.tokenUuid;
   if(!internal)throw Error('熊支援会随主人原伤害自动结算，无需另行应用支援伤害。');
   return {params};
  }
  if(params.final||typeof params.damage==='number'||!params.token?.uuid||!attackIds(message).length)return {params};
  const receipt=await rpc('companion-support-claim',{...ref,actorUuid:actor.uuid,tokenUuid:params.token.uuid});
  return receipt?{params,receipt}:{params};
 }
 async function afterDamage(receipt,result){if(receipt)await rpc('companion-support-complete',{receipt,result});}

 function register({Hooks,libWrapper,socket:socketApi,onError:report=onError}={}){
  socket=socketApi;
  if(registered)return ()=>{};registered=true;const registrations=[];
  if(socket)for(const[method,fn]of [['companion-support-claim',claimApplications],['companion-support-complete',completeApplications]])socket.register(method,async function(payload){
   try{return {ok:true,value:await fn(payload,game.users.get(this.socketdata.userId))};}catch(error){return {ok:false,error:error.message};}
  });
  const strikePath='CONFIG.PF2E.Actor.documentClasses.character.prototype.prepareStrike';
  if(libWrapper)libWrapper.register(MODULE_ID,strikePath,function(wrapped,...args){
   const strike=wrapped(...args),firstRoll=strike?.variants?.[0]?.roll;
   for(const method of ['damage','critical']){
    const native=strike?.[method];if(typeof native!=='function')continue;
    strike[method]=async function(params={}){
     // Native PF2e chat buttons and Workbench pass this exact context object.
     // No search by most recent weapon, name, actor, or target is permitted.
     const attack=params.checkContext&&values(game.messages).find(m=>m.isCheckRoll&&m.flags.pf2e.context===params.checkContext);
     const options=new Set(params.options??[]);
     if(attack)options.add(`${MODULE_ID}:bear-attack:${attack.id}`);
     return native.call(this,{...params,options});
    };
   }
   const result=wrapStrike?.(strike,this)??strike;
   if(typeof firstRoll==='function'&&typeof result?.variants?.[0]?.roll==='function')for(const alias of ['attack','roll'])if(result[alias]===firstRoll)result[alias]=result.variants[0].roll;
   return result;
  },'WRAPPER');
  const on=(name,fn)=>registrations.push([name,Hooks.on(name,(...args)=>Promise.resolve().then(()=>fn(...args)).catch(report))]);
  on('createChatMessage',handleAttack);
  on('updateChatMessage',message=>handleAttack(message));
  on('updateCombat',(combat,changes={})=>('round'in changes||'turn'in changes)?expire(combat):undefined);
  on('deleteCombat',combat=>expire(combat,true));
  on('updateWorldTime',()=>expire());
  on('renderChatMessageHTML',(message,html)=>{
   if(ownFlags(message).kind!=='bear-support-damage')return;
   const root=html?.[0]??html;
   for(const element of root?.querySelectorAll?.('.damage-application,.damage-buttons,[data-action="applyDamage"]')??[])element.hidden=true;
  });
  if(libWrapper)game.actors.get(KNOWN_PAIR.master)?.reset?.();
  return ()=>{for(const[name,id]of registrations)Hooks.off(name,id);if(libWrapper)libWrapper.unregister(MODULE_ID,strikePath);registered=false;};
 }
 return {resolveAction,executeUsage,register,maintain,beforeDamage,afterDamage};
}
