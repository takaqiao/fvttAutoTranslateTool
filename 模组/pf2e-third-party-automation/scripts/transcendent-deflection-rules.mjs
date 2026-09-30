import {MODULE_ID} from './rules.mjs';
import {getSourceId} from './native-context.mjs';
import {isCurrentDisruptToken} from './disrupt-prey-rules.mjs';

export const TRANSCENDENT_DEFLECTION_SOURCE='Compendium.pf2e.feats-srd.Item.WglDQlEtHnZ2eFhW';
const values=c=>Array.from(c?.values?.()??c??[]);
export const deflectionFeat=actor=>values(actor?.items).find(item=>item.type==='feat'&&getSourceId(item)===TRANSCENDENT_DEFLECTION_SOURCE)??null;
export const deflectionBroken=item=>item?.flags?.[MODULE_ID]?.transcendentDeflection?.broken??null;
function eligibleDeflectionMelee(item,{held=true,melee=item}={}){
 const traits=melee?.system?.traits?.value??[],usage=melee?.system?.usage,physical=item?.actor?.items?.get?.(item.id);
 return !!(physical===item&&item.type==='weapon'&&item.uuid&&melee?.isMelee===true&&melee.isRanged!==true&&
  item.system?.category!=='unarmed'&&!traits.some(t=>['unarmed','free-hand'].includes(t))&&
  !item.isBroken&&!item.isDestroyed&&!deflectionBroken(item)&&
  (usage?.hands===1||usage?.value==='held-in-one-hand'||traits.includes('agile')||traits.includes('finesse'))&&
  (!held||item.system?.equipped?.carryType==='held'&&item.system.equipped.handsHeld>0));
}
// A replacement is wielded, not sacrificed. Artifact destruction restrictions
// therefore constrain the reacting weapon only, never the ordinary Swap choice.
export const eligibleDeflectionWeapon=(item,options={})=>eligibleDeflectionMelee(item,options)&&!(options.melee??item)?.system?.traits?.value?.includes('artifact');
export function getTranscendentDeflectionOptions({game,actor,token,attacker,victim}){
 if(!deflectionFeat(actor)||actor.canAct===false||actor.isDead||actor.hasCondition?.('unconscious')||
  ![token,attacker,victim].every(t=>isCurrentDisruptToken(t,game))||token.actor!==actor||
  token.parent!==attacker.parent||token.parent!==victim.parent||!actor.isEnemyOf?.(attacker.actor)||
  actor.uuid!==victim.actor.uuid&&!actor.isAllyOf?.(victim.actor))return [];
 const result=[],seen=new Set();
 const visit=strike=>{
  if(!strike||seen.has(strike))return;seen.add(strike);
  const item=strike.item,weapon=actor.items.get(item?.id);
  if(strike.type==='strike'&&strike.ready===true&&strike.canAttack!==false&&item?.actor?.uuid===actor.uuid&&
   weapon?.uuid===item.uuid&&eligibleDeflectionWeapon(weapon,{melee:item})){
   const reach=actor.getReach?.({action:'attack',weapon:item}),distance=token.object.distanceTo?.(attacker.object,{reach}),point=attacker.getCenterPoint?.();
   const clear=!!point&&typeof token.object.checkCollision==='function'&&token.object.checkCollision(point,{type:'move',mode:'any'})===false;
   if(clear&&Number.isFinite(reach)&&Number.isFinite(distance)&&distance>=0&&distance<=reach&&!result.some(r=>r.weaponUuid===weapon.uuid))
    result.push({weaponUuid:weapon.uuid,weapon,strike,usage:item.altUsageType??null,reach,distance});
  }
  for(const alt of values(strike.altUsages))visit(alt);
 };
 for(const strike of values(actor.system?.actions))visit(strike);
 return result;
}
export const getDeflectionSwapWeapons=actor=>values(actor?.items).filter(item=>eligibleDeflectionMelee(item,{held:false})&&item.system?.equipped?.carryType==='worn'&&!item.system?.containerId);
