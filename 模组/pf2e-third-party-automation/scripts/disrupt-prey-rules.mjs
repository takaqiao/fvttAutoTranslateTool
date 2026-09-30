import {getSourceId} from './native-context.mjs';

export const DISRUPT_PREY_SOURCE='Compendium.pf2e.feats-srd.Item.qav9ec9cR4lFcz3C';
const values=c=>Array.from(c?.values?.()??c??[]);

/** Require the live scene document and its actual canvas object, never a UUID-shaped copy. */
export function isCurrentDisruptToken(token,game){
 return !!(token?.documentName==='Token'&&token.id&&token.uuid&&token.actor?.uuid&&
  game?.scenes?.get(token.parent?.id)===token.parent&&token.parent?.tokens?.get(token.id)===token&&
  token.object?.document===token);
}

export const hasDisruptPrey=actor=>values(actor?.items).some(item=>item.type==='feat'&&getSourceId(item)===DISRUPT_PREY_SOURCE);

export function isDisruptPreyTarget(actor,target,game){
 return !!(actor?.uuid&&isCurrentDisruptToken(target,game)&&target.actor.uuid!==actor.uuid&&
  actor.synthetics?.tokenMarks?.get(target.uuid)?.includes('hunted-prey'));
}

/**
 * Native held/ready and reach candidates only. The coordinator separately proves
 * the event, walls, owner and reaction budget. Keep prepared Strike identity so
 * existing native roll wrappers and generated unarmed/alternate usages survive.
 */
export function getDisruptPreyReadyMeleeStrikes(actor){
 if(!hasDisruptPrey(actor)||actor.canAct===false||actor.isDead||actor.hasCondition?.('unconscious'))return [];
 const candidates=[],seen=new Set(),visit=strike=>{
  if(!strike||seen.has(strike))return;seen.add(strike);
  const item=strike.item;
  if(strike.type==='strike'&&strike.ready===true&&strike.canAttack!==false&&item?.actor?.uuid===actor.uuid&&
   item.id&&item.uuid&&item.isMelee===true&&item.isRanged!==true&&typeof strike.variants?.[0]?.roll==='function'){
   const reach=actor.getReach?.({action:'attack',weapon:item});
   if(Number.isFinite(reach)&&reach>=0){
    const usage=item.altUsageType??null,key=`${item.uuid}#${usage??'base'}`;
    if(!candidates.some(c=>c.key===key))candidates.push({key,itemUuid:item.uuid,usage,strike,reach});
   }
  }
  for(const alternate of values(strike.altUsages))visit(alternate);
 };
 for(const strike of values(actor.system?.actions))visit(strike);
 return candidates;
}

export function getDisruptPreyMeleeOptions({actor,token,target,game}){
 if(!isCurrentDisruptToken(token,game)||token.actor.uuid!==actor?.uuid||
  token.parent!==target?.parent||!isDisruptPreyTarget(actor,target,game))return [];
 return getDisruptPreyReadyMeleeStrikes(actor).flatMap(candidate=>{
  const distance=token.object.distanceTo?.(target.object,{reach:candidate.reach});
  return Number.isFinite(distance)&&distance>=0&&distance<=candidate.reach?[{...candidate,distance}]:[];
 });
}
