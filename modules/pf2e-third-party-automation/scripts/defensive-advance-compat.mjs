import {USE_ACTION_OPTION} from './usage-events.mjs';

export const DEFENSIVE_ADVANCE_SOURCE='Compendium.pf2e.feats-srd.Item.117d4Me9nAn1GMry';
export const RAISED_SHIELD_SOURCE='Compendium.pf2e.equipment-effects.Item.2YgXoHvJfrDHucMr';
const GROUP='YmDi9KpCSdq3ATRP';
const unraised={not:'self:shield:raised'};

/** Keep Patreon's native effect handler; narrow only its verified action predicate. */
export function buildDefensiveAdvancePatreonRepairs(original){
 const rules=structuredClone(original),changes=[],group=rules?.[GROUP];
 if(group?.uuid!==GROUP||group.source?.length!==1||group.source[0]!==DEFENSIVE_ADVANCE_SOURCE)return {rules,changes};
 for(const [index,rule] of (group.baseRules??[]).entries()){
  if(rule.type!=='base'||rule.triggerType!=='postInfo'||rule.target!=='SelfEffect'||rule.value!==RAISED_SHIELD_SOURCE||!Array.isArray(rule.predicate)||!rule.predicate.includes('origin:item:defensive-advance'))continue;
  const before=structuredClone(rule.predicate);
  if(!rule.predicate.includes(USE_ACTION_OPTION))rule.predicate.push(USE_ACTION_OPTION);
  if(!rule.predicate.some(p=>JSON.stringify(p)===JSON.stringify(unraised)))rule.predicate.push({...unraised});
  if(JSON.stringify(before)!==JSON.stringify(rule.predicate))changes.push({path:`${GROUP}.baseRules.${index}.predicate`,before,after:structuredClone(rule.predicate),reason:'列盾突进仅从真实Use举盾，已举盾时复用原效果，展示卡不再添加盾效果。'});
 }
 return {rules,changes};
}
