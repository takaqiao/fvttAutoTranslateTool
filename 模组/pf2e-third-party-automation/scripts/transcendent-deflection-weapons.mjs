import {MODULE_ID} from './rules.mjs';
import {SerialActions} from './runtime.mjs';
import {isActiveGM} from './native-context.mjs';
import {withReactionReservation} from './reaction-budget.mjs';
import {deflectionBroken,eligibleDeflectionWeapon,getDeflectionSwapWeapons} from './transcendent-deflection-rules.mjs';
const own=a=>a?.flags?.[MODULE_ID]?.transcendentDeflection??{},random=()=>globalThis.foundry?.utils?.randomID?.(16)??crypto.randomUUID().replaceAll('-','').slice(0,16);
const claims=a=>own(a).reactions??[];
const methodBrand=Symbol('deflection-weapon-guard');
export function createDeflectionWeapons({game}={}){
 const queue=new SerialActions();
 const gm=()=>{if(!isActiveGM(game))throw Error('靖涛定风剑主GM已改变。')};
 function check(actor,weapon,claim){
  gm();const current=claims(actor).find(c=>c.nonce===claim?.nonce);
  if(!current||current.claimKey!==`deflect:${claim.nonce}`||current.actorUuid!==actor?.uuid||current.weaponUuid!==weapon?.uuid||claim.actorUuid!==current.actorUuid||claim.weaponUuid!==current.weaponUuid||claim.claimKey!==current.claimKey||actor.items?.get(weapon.id)!==weapon||weapon.actor!==actor)throw Error('靖涛定风剑缺少确切已支付武器认领。');
  return current;
 }
 async function save(actor,nonce,changes){
  await withReactionReservation(actor,game,async()=>{gm();const state=structuredClone(own(actor)),claim=state.reactions?.find(c=>c.nonce===nonce);if(!claim)throw Error('武器认领已不存在。');Object.assign(claim,changes);await actor.update({[`flags.${MODULE_ID}.transcendentDeflection`]:state});gm()});
 }
 async function breakWeapon({actor,weapon,claim}){
  return queue.run(actor.uuid,async()=>{
   const current=check(actor,weapon,claim),broken=deflectionBroken(weapon);
   if(current.weaponState==='broken'&&broken?.nonce===current.nonce)return weapon;
   if(current.weaponState||current.state!=='claimed')throw Error('武器结算不确定，不会重复拆分或破损。');
   if(!eligibleDeflectionWeapon(weapon)&&!eligibleDeflectionWeapon(weapon,{melee:actor.system?.actions?.flatMap(s=>[s,...s.altUsages??[]]).find(s=>s.item?.uuid===weapon.uuid&&s.item.isMelee)?.item}))throw Error('选择的武器已经不再合格。');
   const quantity=weapon.system.quantity;if(!Number.isInteger(quantity)||quantity<1)throw Error('武器数量无效。');
   const hp=weapon.system.hp,max=hp?.max??0,bt=Math.floor(max/2);
   if(max>0&&(bt<1||hp.value<=bt))throw Error('武器无法变为破损而保留实体。');
   const spare=quantity>1?weapon.toObject():null,spareId=spare?random():null;
   if(spare){spare._id=spareId;spare.system.quantity=quantity-1;spare.system.equipped={...spare.system.equipped,carryType:'worn',handsHeld:0};spare.system.containerId=null;delete spare.flags?.[MODULE_ID]?.transcendentDeflection;}
   // Reserve before every physical mutation. A rejected/unknown write never
   // proves it did not occur, so a restart cannot split this stack again.
   await save(actor,current.nonce,{weaponState:'breaking',weaponPlan:{quantity,spareId,hp:structuredClone(hp??{})}});check(actor,weapon,claim);
   await weapon.update({'system.quantity':1,...(max>0?{'system.hp.value':bt}:{}),[`flags.${MODULE_ID}.transcendentDeflection.broken`]:{nonce:current.nonce,claimKey:current.claimKey,actorUuid:actor.uuid,weaponUuid:weapon.uuid,virtualHP:max===0}});check(actor,weapon,claim);
   if(spare){const created=await actor.createEmbeddedDocuments('Item',[spare],{keepId:true});check(actor,weapon,claim);if(created?.length!==1||created[0].id!==spareId)throw Error('备用武器拆分结果不确定。');}
   await save(actor,current.nonce,{weaponState:'broken'});return weapon;
  });
 }
 async function swapWeapon({actor,weapon,claim,replacement}){
  return queue.run(actor.uuid,async()=>{
   const current=check(actor,weapon,claim);
   if(current.swap?.state==='done'&&current.swap.weaponUuid===replacement?.uuid)return replacement;
   if(current.state!=='done'||current.weaponState!=='broken'||deflectionBroken(weapon)?.nonce!==current.nonce||current.swap)throw Error('换持缺少已完成的确切防伤回执，或结果不确定。');
   if(!getDeflectionSwapWeapons(actor).includes(replacement)||replacement===weapon)throw Error('只能换持随身合格的另一把武器。');
   const hands=replacement.system.usage?.hands??1,released=weapon.system.equipped?.carryType==='held'?weapon.system.equipped.handsHeld:0,free=actor.handsReallyFree??actor.system?.attributes?.handsFree;
   if(!Number.isInteger(hands)||hands<1||Number.isFinite(free)&&free+released<hands)throw Error('没有足够的手换持这把武器。');
   await save(actor,current.nonce,{swap:{state:'started',weaponUuid:replacement.uuid}});check(actor,weapon,claim);
   await actor.changeCarryType(weapon,{carryType:'worn',handsHeld:0});check(actor,weapon,claim);
   if(!getDeflectionSwapWeapons(actor).includes(replacement))throw Error('换持期间备用武器已改变。');
   await actor.changeCarryType(replacement,{carryType:'held',handsHeld:hands});check(actor,weapon,claim);
   await save(actor,current.nonce,{swap:{state:'done',weaponUuid:replacement.uuid}});return replacement;
  });
 }
 function wrapStrike(strike,actor){
  const seen=new Set(),visit=s=>{
   if(!s||seen.has(s))return;seen.add(s);
   const weapon=actor.items?.get(s.item?.id);
   if(weapon?.uuid===s.item?.uuid&&weapon.type==='weapon'){
    const guard=()=>{if(deflectionBroken(actor.items?.get(weapon.id)))throw Error('这把武器已破损；请正常修理后使用。')};
    if(deflectionBroken(weapon)){s.ready=false;s.canAttack=false;}
    const wrap=(object,key,formula=false)=>{const native=object?.[key];if(typeof native!=='function'||native[methodBrand])return;const wrapped=async function(params={},...rest){if(!(formula&&params.getFormula===true))guard();return native.call(this,params,...rest)};wrapped[methodBrand]=true;object[key]=wrapped};
    for(const variant of s.variants??[])wrap(variant,'roll');for(const key of ['attack','roll','damage','critical'])wrap(s,key,['damage','critical'].includes(key));
   }
   for(const alternate of s.altUsages??[])visit(alternate);
  };visit(strike);return strike;
 }
 return {breakWeapon,swapWeapon,wrapStrike};
}
