import {MODULE_ID} from './rules.mjs';

const MODULE='pf2e-reaction',SLUG='glimpse-of-redemption';
const PATH='CONFIG.Combat.documentClass.prototype.getFlag';
const values=collection=>Array.from(collection?.values?.()??collection??[]);
const holder=actor=>values(actor?.items??actor?.itemTypes?.action).some(item=>item?.type==='action'&&(item.slug??item.system?.slug)===SLUG);

/** Reaction 1.4.3 caches enabled builtins at combat start. Supply one missing
 * member on reads only: independent array writes can overwrite native additions. */
export function createGlimpseReactionCache({game}={}){
 let verified,installed,verification,generation=0;
 const current=module=>module?.active===true&&game.modules?.get(MODULE)===module;
 const ready=()=>!!installed&&!!verified&&current(verified);
 const enabled=()=>{const setting=game.settings.get(MODULE,'builtinReactionsEnabled');return Array.isArray(setting)&&setting.includes(SLUG)};
 const eligible=combat=>ready()&&enabled()&&game.combats?.get(combat?.id)===combat&&combat?.started===true&&values(combat.turns).some(c=>holder(c.actor));
 function read(wrapped,...args){
  const native=wrapped(...args);
  if(args[0]!==MODULE||args[1]!=='availableReactions'||!Array.isArray(native)||native.includes(SLUG))return native;
  try{return eligible(this)?[...native,SLUG]:native;}catch{return native;}
 }
 async function initialize({libWrapper=globalThis.libWrapper}={}){
  if(ready())return true;
  if(verification)return verification;
  verified=undefined;const module=game.modules?.get(MODULE),epoch=generation;
  if(!current(module)||typeof libWrapper?.register!=='function')return false;
  verification=(async()=>{
   if(!current(module)||generation!==epoch)return false;
   if(!installed){libWrapper.register(MODULE_ID,PATH,read,'WRAPPER');installed=libWrapper;}
   verified=module;return ready();
  })();
  try{return await verification;}finally{verification=undefined;}
 }
 function unregister(){generation++;verified=undefined;if(installed){installed.unregister(MODULE_ID,PATH);installed=undefined;}}
 return {initialize,ready,unregister};
}
