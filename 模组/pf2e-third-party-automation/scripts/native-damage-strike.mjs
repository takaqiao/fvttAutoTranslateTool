/** PF2e 8.5.1 only writes flags.pf2e.strike for characters and NPCs.
 * A hazard's native melee Strike instead binds to its actual owned, prepared
 * Strike item. This is not a fallback for missing creature provenance. */
export function hasNativeDamageStrike(actor,item,pf){
 if(pf.strike?.damaging===true&&pf.strike.actor===actor.uuid)return true;
 if(pf.strike!=null||pf.context?.sourceType!=='attack'||actor.type!=='hazard'||item.type!=='melee'||item.actor!==actor
   ||item.system?.action!=='strike'||actor.items?.get?.(item.id)!==item)return false;
 return Array.isArray(actor.system?.actions)&&actor.system.actions.filter(strike=>strike.type==='strike'&&strike.item===item).length===1;
}
