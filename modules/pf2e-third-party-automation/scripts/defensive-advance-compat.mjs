import {USE_ACTION_OPTION} from './usage-events.mjs';

export const DEFENSIVE_ADVANCE_SOURCE='Compendium.pf2e.feats-srd.Item.117d4Me9nAn1GMry';
export const RAISED_SHIELD_SOURCE='Compendium.pf2e.equipment-effects.Item.2YgXoHvJfrDHucMr';
const GROUP='YmDi9KpCSdq3ATRP';
const unraised={not:'self:shield:raised'};

/** Capture before configuration maintenance at ready. Saved settings cannot
 * replace the Patreon rule cache compiled when this particular client started. */
export function defensiveAdvanceStartupCompatibility({game,rules}){
 const module=game.modules?.get('patreon-v3'),group=rules?.[GROUP];
 const unavailable=reason=>Object.freeze({status:'unavailable',reason});
 if(module?.active!==true||module.version!=='3.2.28')return unavailable('需要已核验的 Patreon 3.2.28；请使用手工后续流程。');
 if(group?.isActive!==true||group.uuid!==GROUP||group.source?.length!==1||group.source[0]!==DEFENSIVE_ADVANCE_SOURCE)return unavailable('列盾突进准确上游规则未启用或来源不匹配。');
 const candidates=(group.baseRules??[]).filter(r=>r.target==='SelfEffect'&&r.value===RAISED_SHIELD_SOURCE);
 if(candidates.length!==1||candidates[0].type!=='base'||candidates[0].triggerType!=='postInfo'||!Array.isArray(candidates[0].predicate)||!candidates[0].predicate.includes('origin:item:defensive-advance'))return unavailable('无法唯一确认原生举盾执行器。');
 const predicate=candidates[0].predicate;
 const ready=predicate.includes(USE_ACTION_OPTION)&&predicate.some(p=>JSON.stringify(p)===JSON.stringify(unraised));
 return Object.freeze({status:ready?'ready':'requires-reload',reason:ready?null:'本客户端启动时尚未编译真实Use举盾门禁；保存修复后请整页刷新。'});
}

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
