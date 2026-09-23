import {shieldEncounter} from './reaction-budget.mjs';
import {isCurrentDisruptToken} from './disrupt-prey-rules.mjs';

const values=c=>Array.from(c?.values?.()??c??[]);

export function defensiveAdvanceContext({game,actor,token}){
 if(actor?.type!=='character'||actor.canAct===false||actor.isDead||actor.hasCondition?.('unconscious')||!isCurrentDisruptToken(token,game)||token.actor!==actor)throw Error('列盾突进的角色或原Token现在不能行动。');
 const encounter=shieldEncounter(actor,token,game);
 if(encounter&&encounter.index!==encounter.combat.turn)throw Error('列盾突进需要角色自己的回合。');
 return {turn:encounter?`${encounter.combat.id}:${encounter.combat.round}:${encounter.combat.turn}`:null};
}

/** Keep the prepared native melee choices; the GM adjudicates movement and reach. */
export function defensiveAdvanceMeleeChoices({game,actor,token,target}){
 if(!isCurrentDisruptToken(token,game)||!isCurrentDisruptToken(target,game)||token.parent!==target.parent||target.actor===actor||!actor.alliance||!target.actor.alliance||actor.alliance===target.actor.alliance||target.actor.isDead)return [];
 const result=[],seen=new Set(),visit=strike=>{
  if(!strike||seen.has(strike))return;seen.add(strike);
  const item=strike.item;
  if(strike.type==='strike'&&strike.ready===true&&strike.canAttack!==false&&item?.actor?.uuid===actor.uuid&&item.isMelee===true&&item.isRanged!==true&&typeof strike.variants?.[0]?.roll==='function'){
   const usage=item.altUsageType??null,key=`${item.uuid}#${usage??'base'}`;if(!result.some(r=>r.key===key))result.push({key,strike,itemUuid:item.uuid,usage});
  }
  for(const alt of values(strike.altUsages))visit(alt);
 };
 for(const strike of values(actor.system?.actions))visit(strike);
 return result;
}
