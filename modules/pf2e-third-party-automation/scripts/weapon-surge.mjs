import {createNextStrikeEffectFrame} from './next-strike-effects.mjs';
import {isEatFortuneProbe} from './eat-fortune.mjs';
import {MODULE_ID} from './rules.mjs';
import {WEAPON_SURGE_OPTION,validWeaponSurgeSnapshot} from './weapon-surge-snapshot.mjs';
export {prepareWeaponSurgeDamageSnapshotItems} from './weapon-surge-snapshot.mjs';

const values=collection=>Array.from(collection?.values?.()??collection??[]);
const marked=actor=>actor?._source?.items?.some(item=>item.flags?.[MODULE_ID]?.weaponSurgeSnapshot);
const serialKey=Symbol('weaponSurgeNativeGate');
/** Observe the actual native Strike and awaited Check.roll callback. No chat
 * hook, inferred latest attack, geometry or second confirmation is involved. */
export function createWeaponSurgeAutomation({game}={}){
 const open=new Map(),gates=new Map(),wrappedVariants=new WeakSet(),wrappedDamage=new WeakSet();
 function acquire(actor){
  const wait=gates.get(actor.uuid)??Promise.resolve();let finish,released=false;
  const pending=new Promise(resolve=>{finish=resolve});gates.set(actor.uuid,pending);
  return {wait,release(){if(released)return;released=true;finish();if(gates.get(actor.uuid)===pending)gates.delete(actor.uuid)}};
 }
 function wrapStrike(strike,actor){
  if(strike?.item?.type!=='weapon'||strike.item.actor?.uuid!==actor?.uuid||marked(actor))return strike;
  for(const [index,variant]of (strike.variants??[]).entries()){
   const native=variant.roll;if(typeof native!=='function'||wrappedVariants.has(native))continue;
   const wrapped=async function(params={}){
    if(isEatFortuneProbe(params))return native.call(this,params);
    const carried=params[serialKey],gate=carried??acquire(actor);
    if(!carried)await gate.wait;
    try{
     // A prior accepted Strike can reset prepared statistics when its effect is
     // removed. Rebind after waiting, while retaining the exact usage/MAP entry.
     const current=values(actor.system?.actions).flatMap(action=>[action,...action.altUsages??[]]).find(action=>action.item?.uuid===strike.item.uuid&&(action.item.altUsageType??'')===(strike.item.altUsageType??''));
     if(!carried&&current&&current!==strike&&typeof current.variants?.[index]?.roll==='function'){
      wrapStrike(current,actor);return await current.variants[index].roll({...params,[serialKey]:gate});
     }
     const {[serialKey]:_gate,...nativeParams}=params;params=nativeParams;
     const target=params.target?.document??params.target??values(game?.user?.targets)[0]?.document??values(game?.user?.targets)[0];
     const frame=createNextStrikeEffectFrame({actor,strike,target,consumeTumble:false}),record=frame.snapshot(),marker=record?WEAPON_SURGE_OPTION+record.nonce:null;
     if(!marker)return native.call(this,params);
     const options=new Set([...params.options??[],marker]);open.set(marker,{actor,strike,record});let settlement;
     try{return await native.call(this,{...params,options,extraRollOptions:[...params.extraRollOptions??[],marker],callback:(roll,outcome,message)=>settlement??=(async()=>{
      // The Check middleware normally puts this in the card before publication.
      // createMessage:false also provides an unsaved native document here.
      if(message&&!message.flags?.pf2e?.context?.weaponSurgeSnapshot&&message.updateSource)message.updateSource({'flags.pf2e.context.weaponSurgeSnapshot':record});
      frame.capture(message,{confirmedNativeStrike:roll?._evaluated===true&&Number.isFinite(roll.total)});
      // Release once the actual check has settled, before calling activity code:
      // a callback may await another Strike by this same actor.
      try{await frame.consume()}finally{gate.release()}
      return params.callback?.(roll,outcome,message);
     })()});}finally{open.delete(marker)}
    }finally{gate.release()}
   };
   wrappedVariants.add(wrapped);variant.roll=wrapped;
  }
  for(const method of ['damage','critical']){
   const native=strike[method];if(typeof native!=='function'||wrappedDamage.has(native))continue;
   const wrapped=async function(params={}){
    const context=params.checkContext,record=context?.weaponSurgeSnapshot;
    if(record==null)return native.call(this,params);
    if(!validWeaponSurgeSnapshot(record,actor,strike.item))throw Error('本次激发武器快照无效。');
    const targetUuid=context.target?.token,target=targetUuid?{uuid:targetUuid}:null,frame=createNextStrikeEffectFrame({actor,strike,target,consumeTumble:false});
    const ownCard=context.options?.includes(WEAPON_SURGE_OPTION+record.nonce)===true;
    if(!frame.capture({flags:{pf2e:{origin:{uuid:strike.item.uuid},context}}},{confirmedNativeStrike:ownCard}))throw Error('本次激发武器攻击快照尚未确认。');
    const prepared=frame.damage(strike),options=frame.damageOptions(params.options??[]);
    return prepared===strike?native.call(this,{...params,options}):prepared[method]({...params,options});
   };
   wrappedDamage.add(wrapped);strike[method]=wrapped;
  }
  return strike;
 }
 function interceptCheck(native,check,context={},...args){
  if(context.type!=='attack-roll')return native(check,context,...args);
  const entry=values(context.options).map(option=>open.get(option)).find(Boolean);
  if(!entry||context.origin?.actor?.uuid!==entry.actor.uuid||context.origin?.item?.uuid!==entry.strike.item.uuid)return native(check,context,...args);
  return native(check,{...context,weaponSurgeSnapshot:structuredClone(entry.record)},...args);
 }
 const register=()=>()=>open.clear();
 return {wrapStrike,interceptCheck,register};
}
