import {MODULE_ID} from './rules.mjs';

const prefix=`${MODULE_ID}:source:`,unsupported=unsupportedReason=>({verified:false,unsupportedReason});
const list=x=>Array.isArray(x)?x:[],tokenDocument=x=>x?.document??x;
const serial=x=>JSON.stringify(x,(_key,value)=>value&&typeof value==='object'&&!Array.isArray(value)?Object.fromEntries(Object.entries(value).sort(([a],[b])=>a<b?-1:a>b?1:0)):value);
const digest=async value=>Array.from(new Uint8Array(await globalThis.crypto.subtle.digest('SHA-256',new TextEncoder().encode(value))),byte=>byte.toString(16).padStart(2,'0')).join('');
const physicalWeapon=item=>item?.type==='weapon'&&item.category!=='unarmed'&&item.system?.category!=='unarmed'&&!list(item.system?.traits?.value).some(t=>['unarmed','free-hand'].includes(t));
const held=item=>item?.system?.equipped?.carryType==='held'&&Number.isInteger(item.system.equipped.handsHeld)&&item.system.equipped.handsHeld>0;
const uuidOfSpeaker=s=>typeof s?.scene==='string'&&typeof s?.token==='string'?`Scene.${s.scene}.Token.${s.token}`:null;
const flagsOf=source=>source?.flags?.pf2e;
function targetHelperEvidence(value){
 if(!value||typeof value!=='object'||Array.isArray(value))return value;
 // Toolbelt 3.56.2 writes applied[tokenId][rollIndex] after native damage and
 // re-encodes the full TargetsData schema, adding previously absent defaults.
 // Ignore only that bookkeeping; keep every target/source field (and unknown
 // fields) so retargeting or changing a merged part still invalidates proof.
 const {applied,...stable}=value;
 const defaults={area:null,author:null,expended:0,item:null,isRegen:false,options:[],private:false,saveVariants:{},splashIndex:-1,traits:[],splashTargets:[],targets:[]};
 for(const [key,fallback] of Object.entries(defaults))if(stable[key]===undefined)stable[key]=fallback;
 if(stable.saveVariants&&typeof stable.saveVariants==='object'&&!Array.isArray(stable.saveVariants)){
  stable.saveVariants=Object.fromEntries(Object.entries(stable.saveVariants).map(([id,variant])=>[id,variant&&typeof variant==='object'&&!Array.isArray(variant)?{...variant,basic:variant.basic===undefined?false:variant.basic,saves:variant.saves===undefined?{}:variant.saves}:variant]));
 }
 return stable;
}
function sourceEvidence(source){
 const f=flagsOf(source);
 return {speaker:source.speaker,origin:f?.origin,strike:f?.strike,context:f?.context,target:f?.target,targetHelper:targetHelperEvidence(source.flags?.['pf2e-toolbelt']?.targetHelper)};
}
function targetMatches(source,actor,token){
 const f=flagsOf(source);
 for(const target of [f?.context?.target,f?.target]){
  if(target==null)continue;
  if(typeof target!=='object'||target.actor!=null&&target.actor!==actor.uuid||target.token!=null&&target.token!==token.uuid)return false;
 }
 const targets=source.flags?.['pf2e-toolbelt']?.targetHelper?.targets;
 return !Array.isArray(targets)||!targets.length||targets.includes(token.uuid);
}
function itemOptions(source){
 const f=flagsOf(source),context=list(f?.context?.options).filter(x=>typeof x==='string'),origin=list(f?.origin?.rollOptions).filter(x=>typeof x==='string').map(x=>x.replace(/^origin:item:/,'item:'));
 return new Set([...context,...origin]);
}
function usableRoll(roll){
 return roll&&Number.isFinite(roll.total)&&roll.total>0&&Array.isArray(roll.instances)&&roll.instances.some(i=>!i.persistent&&Number(i.total)>0)&&!roll.options?.splashOnly&&!roll.options?.evaluatePersistent;
}
function rollEvidence(roll){
 const value=roll.toJSON?.();return typeof value==='string'?JSON.parse(value):value;
}
function partRollEvidence(source){
 return list(source.rolls).map(roll=>typeof roll==='string'?JSON.parse(roll):roll);
}
function hasPrimaryPartRoll(source){
 const roll=partRollEvidence(source)[0];
 return roll?.class==='DamageRoll'&&roll.evaluated===true&&Number.isFinite(roll.total)&&roll.total>0&&!roll.options?.splashOnly&&!roll.options?.evaluatePersistent;
}
async function bindPart({source,actor,token,fromUuid,checkHeld}){
 const f=flagsOf(source),o=f?.origin,s=f?.strike,c=f?.context;
 if(!o||!s||s.damaging!==true||c?.type!=='damage-roll'||c.sourceType!=='attack'||!['weapon','melee'].includes(o.type))return unsupported('not-native-strike-damage');
 if(!targetMatches(source,actor,token))return unsupported('source-target-mismatch');
 const attackerToken=await fromUuid(uuidOfSpeaker(source.speaker)),attacker=attackerToken?.actor;
 if(attackerToken?.documentName!=='Token'||attackerToken.parent?.id!==token.parent?.id||!attacker||attacker.id!==source.speaker.actor||attacker.uuid!==o.actor||attacker.uuid!==s.actor||c.actor!=null&&c.actor!==attacker.id||c.token!=null&&c.token!==attackerToken.id)return unsupported('attacker-provenance-mismatch');
 const attackItem=await fromUuid(o.uuid);
 if(!attackItem||attackItem.uuid!==o.uuid||attackItem.actor?.uuid!==attacker.uuid||attackItem.type!==o.type)return unsupported('attack-item-mismatch');
 const options=itemOptions(source),ids=[...options].filter(x=>x.startsWith('item:id:')),types=[...options].filter(x=>x.startsWith('item:type:'));
 if(ids.length!==1||ids[0]!==`item:id:${attackItem.id}`||types.some(t=>t!==`item:type:${attackItem.type}`)||!options.has('item:melee')||options.has('item:ranged')||options.has('item:thrown-melee')||options.has('item:category:unarmed')||options.has('item:trait:unarmed')||options.has('item:trait:free-hand'))return unsupported('not-held-melee-usage');
 let weapon;
 if(attackItem.type==='weapon'){
  if(![null,undefined,'melee'].includes(s.altUsage))return unsupported('not-held-melee-usage');
  const usage=s.altUsage==='melee'?list(attackItem.getAltUsages?.()).find(i=>i.id===attackItem.id&&i.altUsageType==='melee'):attackItem;
  if(!usage?.isMelee||usage.isRanged||!physicalWeapon(usage)||!options.has('item:equipped')||![...options].some(x=>/^item:hands-held:[1-9]\d*$/.test(x)))return unsupported('not-held-melee-usage');
  weapon=attackItem;
 }else{
  if(attacker.type!=='npc'||attackItem.system?.action!=='strike'||!attackItem.isMelee||attackItem.isRanged||list(attackItem.system?.traits?.value).some(t=>['unarmed','free-hand'].includes(t))||s.altUsage!=null)return unsupported('not-npc-melee-strike');
  const linkedId=attackItem.flags?.pf2e?.linkedWeapon;
  weapon=typeof linkedId==='string'?attacker.items?.get?.(linkedId):null;
  if(!weapon||attackItem.linkedWeapon?.uuid!==weapon.uuid||weapon.id!==linkedId||weapon.actor?.uuid!==attacker.uuid||(await fromUuid(weapon.uuid))?.uuid!==weapon.uuid)return unsupported('npc-weapon-link-unavailable');
 }
 if(!physicalWeapon(weapon)||checkHeld&&!held(weapon))return unsupported('weapon-not-held-or-disarmable');
 return {verified:true,attackerActorUuid:attacker.uuid,attackerTokenUuid:attackerToken.uuid,attackItemUuid:attackItem.uuid,weaponUuid:weapon.uuid};
}
async function inspect({game,fromUuid,actor,token,message,rollIndex,checkHeld,paramsItem,checkParamsItem}){
 if(!message||game.messages?.get?.(message.id)!==message||!message.isDamageRoll)return unsupported('damage-message-unavailable');
 if(!Number.isInteger(rollIndex)||rollIndex<0||rollIndex>=message.rolls?.length)return unsupported('damage-roll-unavailable');
 // In this native version every additional roll is splash, including merged
 // cards. The merged primary roll cannot be indexed into its contributing parts.
 if(rollIndex!==0||!usableRoll(message.rolls[rollIndex]))return unsupported('not-primary-strike-damage');
 if(token?.documentName!=='Token'||token.actor?.uuid!==actor?.uuid||!token.parent?.id)return unsupported('recipient-token-mismatch');
 if(!targetMatches(message,actor,token))return unsupported('source-target-mismatch');
 const merge=message.flags?.['pf2e-toolbelt']?.betterChat?.mergeDamage;
 let parts;
 if(merge){
  if(merge.merged!==true||!Array.isArray(merge.data)||merge.data.length<2||merge.data.some(p=>!p?.source||typeof p.source!=='object'))return unsupported('merged-source-incomplete');
  parts=merge.data.map(p=>p.source);
  if(parts.some(p=>!hasPrimaryPartRoll(p)))return unsupported('merged-roll-source-incomplete');
 }else parts=[message];
 const inputs=()=>({author:message.author?.id??message.user?.id??message.user,speaker:message.speaker,top:sourceEvidence(message),parts:parts.map(source=>({source:sourceEvidence(source),rolls:merge?partRollEvidence(source):undefined})),roll:rollEvidence(message.rolls[rollIndex])});
 const capturedInputs=serial(inputs()),bindings=[];
 for(const source of parts){
  const result=await bindPart({source,actor,token,fromUuid,checkHeld});if(!result.verified)return result;bindings.push(result);
 }
 const first=bindings[0];
 if(bindings.some(b=>b.attackerActorUuid!==first.attackerActorUuid||b.attackerTokenUuid!==first.attackerTokenUuid||b.weaponUuid!==first.weaponUuid||b.attackItemUuid!==first.attackItemUuid))return unsupported('multiple-strike-sources');
 if(uuidOfSpeaker(message.speaker)!==first.attackerTokenUuid||message.speaker.actor!==(await fromUuid(first.attackerActorUuid))?.id)return unsupported('attacker-provenance-mismatch');
 if(checkParamsItem){
  // Native merged cards can have no item. An injected merge can retain the one
  // real item, but it must agree with every independently validated part.
  if(!paramsItem&&!merge||paramsItem&&(paramsItem.uuid!==first.attackItemUuid||paramsItem.actor?.uuid!==first.attackerActorUuid))return unsupported('application-item-mismatch');
 }
 const roll=rollEvidence(message.rolls[rollIndex]);if(!roll||typeof roll!=='object')return unsupported('damage-roll-evidence-unavailable');
 const evidence=await digest(serial({inputs:JSON.parse(capturedInputs),bindings}));
 if(game.messages?.get?.(message.id)!==message||capturedInputs!==serial(inputs()))return unsupported('source-evidence-changed');
 const sourceSnapshot={schema:1,damageMessageId:message.id,rollIndex,actorUuid:actor.uuid,tokenUuid:token.uuid,attackerActorUuid:first.attackerActorUuid,attackerTokenUuid:first.attackerTokenUuid,attackItemUuid:first.attackItemUuid,weaponUuid:first.weaponUuid,partCount:parts.length,
  evidence};
 return {verified:true,damageMessageId:message.id,rollIndex,actorUuid:actor.uuid,tokenUuid:token.uuid,attackerActorUuid:first.attackerActorUuid,attackerTokenUuid:first.attackerTokenUuid,attackItemUuid:first.attackItemUuid,weaponUuid:first.weaponUuid,sourceSnapshot};
}

/** Caller must obtain the source marker from cycle.getRollContext, not accept a
 * caller-supplied marker as authentication. This function only reads documents.
 * A source match is not proof that Shield Block actually completed. */
export async function resolveShieldBlockSource({game,fromUuid=globalThis.fromUuid,actor,token,params}={}){
 try{
  const markers=Array.from(params?.rollOptions??[]).filter(x=>typeof x==='string'&&x.startsWith(prefix));
  if(markers.length!==1)return unsupported('damage-source-marker-unavailable');
  const match=/^([^:]+):(0|[1-9]\d*)$/.exec(markers[0].slice(prefix.length));if(!match)return unsupported('damage-source-marker-invalid');
  if(!usableRoll(params?.damage))return unsupported('application-not-damage-roll');
  const recipient=tokenDocument(token??params.token),rollIndex=Number(match[2]),message=game.messages?.get?.(match[1]);
  return await inspect({game,fromUuid,actor,token:recipient,message,rollIndex,checkHeld:true,paramsItem:params.item,checkParamsItem:true});
 }catch{return unsupported('source-document-unavailable');}
}

/** Rebuild identities from the exact live source and compare the captured
 * evidence. The GM must additionally authenticate the actual block nonce and
 * recipient/owner; this serializable snapshot is not a signed capability.
 * Current holding/ability checks belong to the consuming reaction provider. */
export async function validateShieldBlockSource({game,fromUuid=globalThis.fromUuid,snapshot}={}){
 try{
  if(snapshot?.schema!==1||typeof snapshot.evidence!=='string'||!/^[a-f0-9]{64}$/.test(snapshot.evidence))return unsupported('source-snapshot-invalid');
  const token=await fromUuid(snapshot.tokenUuid),actor=token?.actor;
  if(!actor||actor.uuid!==snapshot.actorUuid)return unsupported('recipient-token-mismatch');
  const result=await inspect({game,fromUuid,actor,token,message:game.messages?.get?.(snapshot.damageMessageId),rollIndex:snapshot.rollIndex,checkHeld:false,checkParamsItem:false});
  if(!result.verified)return result;
  if(serial(result.sourceSnapshot)!==serial(snapshot))return unsupported('source-evidence-changed');
  return result;
 }catch{return unsupported('source-document-unavailable');}
}
