import {MODULE_ID} from './rules.mjs';
export const GLIMPSE_REACTION_REASON='救赎瞥视改由已验证的原伤害流程询问和付款，避免重复提示。';
/** Restore only the exact setting this integration previously changed. */
export function glimpseReactionSetting(value,game,ready){
 if(!Array.isArray(value))return value;
 const slug='glimpse-of-redemption';
 if(ready)return value.filter(v=>v!==slug);
 const history=game.settings.get(MODULE_ID,'configurationBackups')??[];
 const change=history.flatMap(h=>h.setting==='pf2e-reaction.builtinReactionsEnabled'?h.changes??[]:[]).filter(c=>c.path==='builtinReactionsEnabled'&&c.reason===GLIMPSE_REACTION_REASON).at(-1);
 return Array.isArray(change?.before)&&change.before.includes(slug)&&Array.isArray(change.after)&&!change.after.includes(slug)&&JSON.stringify(value)===JSON.stringify(change.after)?structuredClone(change.before):value;
}
