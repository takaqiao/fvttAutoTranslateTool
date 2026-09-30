import {getSourceId} from './native-context.mjs';

const TUMBLE_BEHIND='Compendium.patreon-v3.effects.Item.Vh5E1Qgp34sTKfVs';
const TUMBLE_OPTION='self:effect:off-guard-tumble-behind';
const OFF_GUARD='target:condition:off-guard';
const outcomes=new Set(['criticalFailure','failure','success','criticalSuccess']);
const values=collection=>Array.from(collection?.values?.()??collection??[]);
const revision=item=>JSON.stringify(item.toObject?.()??item._source??{system:item.system,flags:item.flags});

/** An own activity can delay damage until after its next attack. Consume only
 * the recognized one-attack effect that existed when this Strike began, while
 * retaining the witnessed condition for native damage predicates. This is not
 * a global check hook: ordinary Strike buttons keep their existing workflow. */
export function createNextStrikeEffectFrame({actor,strike,target}){
 const item=strike?.item,valid=item?.type==='weapon'&&!!item.id&&!!item.uuid&&item.actor?.uuid===actor.uuid;
 const weaponUuid=item?.uuid,targetUuid=target?.uuid;
 const effects=valid?values(actor.items).filter(effect=>effect.type==='effect'&&effect.id&&!effect.isExpired&&getSourceId(effect)===TUMBLE_BEHIND).map(effect=>({effect,revision:revision(effect)})):[];
 let captured=false,offGuard=false,consumption;
 function capture(message){
  if(captured||!valid)return false;
  const flags=message?.flags?.pf2e,context=flags?.context;
  if(flags?.origin?.uuid!==weaponUuid||context?.type!=='attack-roll'||!outcomes.has(context.outcome))return false;
  captured=true;
  offGuard=!!targetUuid&&context.target?.token===targetUuid&&context.options?.includes(OFF_GUARD)===true;
  return true;
 }
 function consume(){
  if(!captured)return Promise.resolve(false);
  return consumption??=(async()=>{
   // Do not delete a replacement or refreshed effect granted after this roll.
   const live=values(actor.items),ids=effects.filter(entry=>live.includes(entry.effect)&&getSourceId(entry.effect)===TUMBLE_BEHIND&&revision(entry.effect)===entry.revision).map(entry=>entry.effect.id);
   if(!ids.length)return false;
   try{await actor.deleteEmbeddedDocuments('Item',ids)}catch(error){
    // Patreon can finish its miss cleanup concurrently with this settlement.
    if(values(actor.items).some(item=>ids.includes(item.id)))throw error;
   }
   return true;
  })();
 }
 function damageOptions(options=[]){
  const result=new Set(options);
  // A delayed damage card must not ask Patreon to consume a newly gained use.
  if(captured&&effects.length)result.delete(TUMBLE_OPTION);
  if(offGuard)result.add(OFF_GUARD);
  return result;
 }
 return {capture,consume,damageOptions};
}
