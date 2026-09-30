import {MODULE_ID} from './rules.mjs';
import {GLIMPSE_SOURCES,glimpseSourceId} from './glimpse-source.mjs';
export const GLIMPSE_REACTION_REASON='救赎瞥视改由已验证的原伤害流程询问和付款，避免重复提示。';
const SLUG='glimpse-of-redemption',SETTING='pf2e-reaction.builtinReactionsEnabled';
const values=collection=>Array.from(collection?.values?.()??collection??[]);
/** Callers include world and synthetic scene actors. A broad holder test keeps
 * unsupported native reminders, even if the provider needs stricter sources. */
export function canSuppressGlimpseReminder(actors,provider){
 try{
  const holders=values(actors).filter(actor=>values(actor?.items??actor?.itemTypes?.action).some(item=>item.type==='action'&&(item.slug===SLUG||item.system?.slug===SLUG||glimpseSourceId(item)===GLIMPSE_SOURCES.glimpse)));
  return holders.length>0&&holders.every(actor=>provider?.handlesActor?.(actor)===true);
 }catch{return false;}
}
/** Restore only this integration's still-owned member, not an old whole array. */
export function glimpseReactionSetting(value,game,ready){
 if(!Array.isArray(value))return value;
 if(ready)return value.filter(v=>v!==SLUG);
 if(value.includes(SLUG))return value;
 const history=game.settings.get(MODULE_ID,'configurationBackups')??[];
 let removal;
 for(const entry of Array.isArray(history)?history:[]){
  if(entry?.setting!==SETTING||!Array.isArray(entry.changes))continue;
  for(const change of entry.changes){
   if(change?.path!=='builtinReactionsEnabled'||!Array.isArray(change.before)||!Array.isArray(change.after))continue;
   const before=change.before.includes(SLUG),after=change.after.includes(SLUG);
   if(before!==after)removal=before&&!after&&change.reason===GLIMPSE_REACTION_REASON?change:undefined;
  }
 }
 if(!removal)return value;
 const restored=[...value];restored.splice(Math.min(removal.before.indexOf(SLUG),restored.length),0,SLUG);return restored;
}
