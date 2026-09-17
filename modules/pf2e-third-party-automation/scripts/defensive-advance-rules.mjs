import {shieldEncounter} from './reaction-budget.mjs';
import {isCurrentDisruptToken} from './disrupt-prey-rules.mjs';

export const advancePosition=token=>({x:token.x,y:token.y,elevation:token.elevation??0});
export const sameAdvancePosition=(a,b)=>!!a&&!!b&&['x','y','elevation'].every(k=>(a[k]??0)===(b[k]??0));
const values=c=>Array.from(c?.values?.()??c??[]);

export function defensiveAdvanceContext({game,actor,token}){
 if(actor?.type!=='character'||actor.canAct===false||actor.isDead||actor.hasCondition?.('unconscious')||!isCurrentDisruptToken(token,game)||token.actor!==actor)throw Error('列盾突进的角色或原Token现在不能行动。');
 const encounter=shieldEncounter(actor,token,game);
 if(encounter&&encounter.index!==encounter.combat.turn)throw Error('列盾突进需要角色自己的回合。');
 const speed=actor.movement?.speeds?.land?.value??actor.system?.movement?.speeds?.land?.value;
 if(!Number.isFinite(speed)||speed<=0||actor.hasCondition?.('immobilized')||actor.hasCondition?.('restrained'))throw Error('无法确认可用的原生陆地速度；请手工完成后续活动。');
 return {turn:encounter?`${encounter.combat.id}:${encounter.combat.round}:${encounter.combat.turn}`:null,speed};
}

/** Only the chosen native melee Strike needs reach. An enemy's reach is irrelevant. */
export function defensiveAdvanceMeleeChoices({game,actor,token,target}){
 if(!isCurrentDisruptToken(token,game)||!isCurrentDisruptToken(target,game)||token.parent!==target.parent||target.actor===actor||!actor.alliance||!target.actor.alliance||actor.alliance===target.actor.alliance||target.actor.isDead)return [];
 const center=target.getCenterPoint?.();
 if(!center||typeof token.object.checkCollision!=='function'||token.object.checkCollision(center,{type:'move',mode:'any'})!==false)return [];
 const result=[],seen=new Set(),visit=strike=>{
  if(!strike||seen.has(strike))return;seen.add(strike);
  const item=strike.item;
  if(strike.type==='strike'&&strike.ready===true&&strike.canAttack!==false&&item?.actor?.uuid===actor.uuid&&item.isMelee===true&&item.isRanged!==true&&typeof strike.variants?.[0]?.roll==='function'){
   const reach=actor.getReach?.({action:'attack',weapon:item}),distance=token.object.distanceTo?.(target.object,{reach});
   if(Number.isFinite(reach)&&reach>=0&&Number.isFinite(distance)&&distance>=0&&distance<=reach){const usage=item.altUsageType??null,key=`${item.uuid}#${usage??'base'}`;if(!result.some(r=>r.key===key))result.push({key,strike,itemUuid:item.uuid,usage,reach});}
  }
  for(const alt of values(strike.altUsages))visit(alt);
 };
 for(const strike of values(actor.system?.actions))visit(strike);
 return result;
}

/** Accept the server's moveToken payload, never an updateToken coordinate change. */
export function defensiveAdvanceMovementProof({token,movement,operation,user,receipt}){
 if(!receipt?.planId||operation?._movement?.[token.id]!==movement||user?.id!==receipt.userId||!(movement.id===receipt.planId||movement.chain?.[0]===receipt.planId)||receipt.movementIds.includes(movement.id))return null;
 const cost=movement.passed?.cost,path=movement.passed?.waypoints;
 if(!Array.isArray(path)||!path.length||path.some(w=>w.action!=='walk')||!Number.isFinite(cost)||cost<=0||receipt.movementCost+cost>receipt.speed||movement.constrained||!sameAdvancePosition(movement.origin,receipt.lastPosition)||!sameAdvancePosition(movement.destination,advancePosition(token)))throw Error('本次原生步行路径、速度或位置不匹配；不继续打击。');
 return {movementIds:[...receipt.movementIds,movement.id],movementCost:receipt.movementCost+cost,lastPosition:advancePosition(token)};
}
