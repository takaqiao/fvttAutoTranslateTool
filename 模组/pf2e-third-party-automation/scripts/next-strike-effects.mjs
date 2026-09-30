import {getSourceId} from './native-context.mjs';
import {createWeaponSurgeSnapshot,isWeaponSurgeFor,validWeaponSurgeSnapshot,weaponSurgeSourceData,weaponSurgeTransientItems,prepareWeaponSurgeDamageSnapshotItems} from './weapon-surge-snapshot.mjs';

const TUMBLE_BEHIND='Compendium.patreon-v3.effects.Item.Vh5E1Qgp34sTKfVs';
const TUMBLE_OPTION='self:effect:off-guard-tumble-behind';
const OFF_GUARD='target:condition:off-guard';
const outcomes=new Set(['criticalFailure','failure','success','criticalSuccess']);
const values=collection=>Array.from(collection?.values?.()??collection??[]);
const revision=item=>JSON.stringify(item.toObject?.()??item._source??{system:item.system,flags:item.flags});

/** A native Strike or own activity can delay damage until after its next attack.
 * Consume only the recognized one-attack effect that existed for this invocation,
 * retaining its native damage rules and witnessed target condition. */
export function createNextStrikeEffectFrame({actor,strike,target,consumeTumble=true}){
 const item=strike?.item,valid=item?.type==='weapon'&&!!item.id&&!!item.uuid&&item.actor?.uuid===actor.uuid;
 const weaponUuid=item?.uuid,targetUuid=target?.uuid;
 const effects=valid?values(actor.items).filter(effect=>effect.type==='effect'&&effect.id&&!effect.isExpired&&(consumeTumble&&getSourceId(effect)===TUMBLE_BEHIND||isWeaponSurgeFor(effect,item.id))).map(effect=>({effect,revision:revision(effect)})):[];
 const surge=effects.filter(entry=>isWeaponSurgeFor(entry.effect,item.id));
 let record=valid?createWeaponSurgeSnapshot(actor,item,surge.map(entry=>entry.effect)):null,captured=false,offGuard=false,consumption,reroll=false;
 function capture(message,{confirmedNativeStrike=false}={}){
  if(captured||!valid)return false;
  const flags=message?.flags?.pf2e,context=flags?.context;
  if(flags?.origin?.uuid!==weaponUuid||context?.type!=='attack-roll'||!outcomes.has(context.outcome)&&!(confirmedNativeStrike&&context.outcome==null))return false;
  const saved=context.weaponSurgeSnapshot;
  if(saved!=null&&!validWeaponSurgeSnapshot(saved,actor,item))return false;
  captured=true;
  if(validWeaponSurgeSnapshot(saved,actor,item))record=structuredClone(saved);
  reroll=context.isReroll===true;
  offGuard=!!targetUuid&&context.target?.token===targetUuid&&context.options?.includes(OFF_GUARD)===true;
  return true;
 }
 function consume(){
  if(!captured)return Promise.resolve(false);
  return consumption??=(async()=>{
   // Do not delete a replacement or refreshed effect granted after this roll.
   const live=values(actor.items),ids=effects.filter(entry=>!reroll&&live.includes(entry.effect)&&revision(entry.effect)===entry.revision&&(getSourceId(entry.effect)===TUMBLE_BEHIND||isWeaponSurgeFor(entry.effect,item.id)&&record.effects.some(data=>JSON.stringify(data)===JSON.stringify(weaponSurgeSourceData(entry.effect))))).map(entry=>entry.effect.id);
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
  if(captured&&effects.some(entry=>getSourceId(entry.effect)===TUMBLE_BEHIND))result.delete(TUMBLE_OPTION);
  if(captured)result.delete('self:effect:spell-effect-weapon-surge');
  if(offGuard)result.add(OFF_GUARD);
  return result;
 }
 function snapshot(){return record?structuredClone(record):null}
 function transientItems(){return captured&&record?weaponSurgeTransientItems(record):[]}
 function damage(current){
  if(!captured||!valid||current.item?.actor?.uuid!==actor.uuid||current.item?.uuid!==weaponUuid)return current;
  const owner=current.item.actor;
  if(!record.effects.length&&!values(owner.items).some(effect=>isWeaponSurgeFor(effect,item.id)))return current;
  const copy=owner.clone({items:prepareWeaponSurgeDamageSnapshotItems(owner,transientItems())},{keepId:true});
  return values(copy.system.actions).flatMap(action=>[action,...action.altUsages??[]]).find(action=>action.item?.id===item.id&&(action.item.altUsageType??'')===(current.item.altUsageType??''))??current;
 }
 return {capture,consume,damageOptions,snapshot,transientItems,damage};
}
