const MODULE='pf2e-reaction',VERSION='1.4.3';
const SHA256='4a81322796ce1c6ed545edc09e1aa3a96a9c8a96dfd034403bf657068ed7036c';
const unavailable=()=>Error('当前 Reaction Checker 的格挡资源接口无法可靠核验；本次未应用伤害，请使用手工结算流程。');
const defaultHash=async source=>Array.from(new Uint8Array(await globalThis.crypto.subtle.digest('SHA-256',new TextEncoder().encode(source))),b=>b.toString(16).padStart(2,'0')).join('');
const defaultFetch=async()=>{
 const route=globalThis.foundry?.utils?.getRoute?.('modules/pf2e-reaction/pf2e-reaction.js')??'/modules/pf2e-reaction/pf2e-reaction.js';
 const response=await fetch(route,{cache:'no-store'});if(!response.ok)throw unavailable();return response.text();
};

/** The audited module has no public resource API. Only return patches for the
 * exact native Shield Block claim: its ledger and resource share one update. */
export function createShieldReactionResources({game,fetchSource=defaultFetch,hashSource=defaultHash}={}){
 let verified,verification;const snapshots=new WeakSet();
 const current=module=>game.modules?.get(MODULE)===module&&module?.active===true&&module.version===VERSION;
 async function snapshot(combatant){
  const module=game.modules?.get(MODULE);
  if(!module?.active){const result={active:false};snapshots.add(result);return result;}
  if(!current(module))throw unavailable();
  if(verified!==module){
   verification??=Promise.resolve().then(async()=>{if(await hashSource(await fetchSource())!==SHA256)throw unavailable();return module;}).catch(error=>{verification=null;throw error});
   verified=await verification;if(verified!==module||!current(module))throw unavailable();
  }
  const flags=combatant.flags?.[MODULE]??{},state=flags.state,quick=flags['quick-shield-block']??0;
  if(typeof state!=='boolean'||!Number.isSafeInteger(quick)||quick<0)throw unavailable();
  const result={active:true,module,combatant,state,quick};snapshots.add(result);return result;
 }
 function available(value,slot){
  if(!snapshots.has(value))throw unavailable();
  if(!value.active)return true;if(!current(value.module))throw unavailable();
  return slot==='generic'?value.state:slot==='quick-shield-block'&&value.quick>0;
 }
 function reserve(value,slot,{prepaid=false}={}){
  if(!snapshots.has(value))throw unavailable();
  if(!value.active)return {proof:null,changes:{}};
  const live=value.combatant.flags?.[MODULE]??{};
  if(live.state!==value.state||(live['quick-shield-block']??0)!==value.quick)throw Error('Reaction Checker 反应资源在验证期间已改变，本次未应用伤害。');
  const canUse=available(value,slot),key=slot==='generic'?'state':slot==='quick-shield-block'?'quick-shield-block':null;
  if(!key||!canUse&&!prepaid)throw Error('本次盾牌格挡没有可用反应；伤害尚未应用。');
  const before=key==='state'?value.state:value.quick,after=canUse?(key==='state'?false:before-1):before;
  return {proof:{version:VERSION,sourceSHA256:SHA256,key,before,after,consumed:canUse},changes:canUse?{[`flags.${MODULE}.${key}`]:after}:{}};
 }
 async function release(combatant,proof,{stillSpent=false}={}){
  if(!proof?.consumed||stillSpent)return {};
  if(proof.version!==VERSION||proof.sourceSHA256!==SHA256||!['state','quick-shield-block'].includes(proof.key))throw unavailable();
  const value=await snapshot(combatant);
  if(!value.active)throw unavailable();
  const actual=proof.key==='state'?value.state:value.quick;
  // A manual edit or another module's turn refresh owns the newer value.
  return actual===proof.after?{[`flags.${MODULE}.${proof.key}`]:proof.before}:{};
 }
 return {snapshot,available,reserve,release};
}
