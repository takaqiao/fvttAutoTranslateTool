import {isActiveGM} from './native-context.mjs';
import {SerialActions} from './runtime.mjs';

const MODULE='pf2e-reaction',VERSION='1.4.3',SLUG='glimpse-of-redemption';
const SHA256='4a81322796ce1c6ed545edc09e1aa3a96a9c8a96dfd034403bf657068ed7036c';
const values=collection=>Array.from(collection?.values?.()??collection??[]);
const defaultHash=async source=>Array.from(new Uint8Array(await globalThis.crypto.subtle.digest('SHA-256',new TextEncoder().encode(source))),b=>b.toString(16).padStart(2,'0')).join('');
const defaultFetch=async()=>{
 const path=globalThis.foundry?.utils?.getRoute?.('modules/pf2e-reaction/pf2e-reaction.js')??'/modules/pf2e-reaction/pf2e-reaction.js';
 const response=await fetch(path,{cache:'no-store'});if(!response.ok)throw Error('无法核验 Reaction Checker 战斗提醒缓存接口。');return response.text();
};
const holder=actor=>values(actor?.items??actor?.itemTypes?.action).some(item=>item?.type==='action'&&(item.slug??item.system?.slug)===SLUG);

/** Reaction 1.4.3 caches enabled builtins at combat start. Restoring its setting
 * also needs this one cache member; no reaction payment/state belongs here. */
export function createGlimpseReactionCache({game,fetchSource=defaultFetch,hashSource=defaultHash}={}){
 let verified;const queue=new SerialActions();
 const current=module=>module?.active===true&&module.version===VERSION&&game.modules?.get(MODULE)===module;
 const ready=()=>!!verified&&current(verified);
 async function initialize(){
  verified=undefined;const module=game.modules?.get(MODULE);if(!current(module))return false;
  if(await hashSource(await fetchSource())!==SHA256||!current(module))return false;
  verified=module;return true;
 }
 const enabled=()=>{const setting=game.settings.get(MODULE,'builtinReactionsEnabled');return Array.isArray(setting)&&setting.includes(SLUG)};
 const eligible=combat=>ready()&&isActiveGM(game)&&enabled()&&game.combats?.get(combat.id)===combat&&combat.started===true&&values(combat.turns).some(c=>holder(c.actor));
 const read=combat=>combat.getFlag(MODULE,'availableReactions');
 const restore=()=>queue.run('restore',async()=>{
  if(!ready()||!isActiveGM(game)||!enabled())return [];
  const restored=[];
  for(const combat of values(game.combats)){
   if(!eligible(combat))continue;
   const cached=read(combat);if(!Array.isArray(cached)||cached.includes(SLUG))continue;
   const before=[...cached],signature=JSON.stringify(before),next=[...before,SLUG];
   // No await between the live comparison and native update. If another native
   // writer changes the cache, its updateCombat event schedules a new pass.
   const live=read(combat);
   if(!Array.isArray(live)||JSON.stringify(live)!==signature||!eligible(combat))continue;
   await combat.update({[`flags.${MODULE}.availableReactions`]:next});
   restored.push(combat.id);
  }
  return restored;
 });
 return {initialize,ready,restore};
}
