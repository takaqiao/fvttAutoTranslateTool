import {sourceId,values} from '../salubrious-kiss-rules.mjs';
import {canonicalItemSource} from './source-ids.mjs';
import {treatmentOutcomeRows} from './treatment-context.mjs';

const riskySource='Compendium.pf2e.feats-srd.Item.bkZgWFSFV4cAf5Ot';
const assuranceSource='Compendium.pf2e.feats-srd.Item.W6Gl9ePmItfDHji0';
const riskyRules=[
 {key:'RollOption',domain:'medicine',option:'risky-surgery',toggleable:true},
 {key:'FlatModifier',selector:'medicine',type:'circumstance',value:2,predicate:['action:treat-wounds','risky-surgery']},
 {key:'Note',selector:'medicine',predicate:['action:treat-wounds','risky-surgery'],text:'PF2E.SpecificRule.Feat.RiskySurgery.Note',title:'{item|name}'},
 {key:'AdjustDegreeOfSuccess',selector:'medicine',predicate:['risky-surgery','action:treat-wounds'],adjustment:{success:'one-degree-better'}}
];
const assuranceSelector='{item|flags.system.rulesSelections.assurance}';
function canonical(value){return Array.isArray(value)?value.map(canonical):value&&typeof value==='object'?Object.fromEntries(Object.keys(value).sort().map(key=>[key,canonical(value[key])])):value}
const same=(a,b)=>JSON.stringify(canonical(a))===JSON.stringify(canonical(b));
const fail=reason=>{throw Error(reason)};
const itemSource=item=>canonicalItemSource(sourceId(item));
function originalRules(item){
 // PF2e adds schema defaults and normalizes selectors in the prepared rules.
 // Qualify the stored definition, while keeping ignored prepared rules blocked.
 if(values(item.system?.rules).some(rule=>rule.ignored)||values(item.rules).some(rule=>rule.ignored))return null;
 if(typeof item.toObject==='function')return item.toObject(true)?.system?.rules;
 if(item._source!==undefined)return item._source?.system?.rules;
 return item.system?.rules;
}
function validRisky(item){
 const original=originalRules(item);
 if(itemSource(item)!==riskySource||!Array.isArray(original))return false;
 const rules=structuredClone(original);
 if(rules[0]?.value!==undefined){if(typeof rules[0].value!=='boolean')return false;delete rules[0].value}
 return same(rules,riskyRules);
}
function assuranceSkill(item){return item.flags?.pf2e?.rulesSelections?.assurance??item.flags?.system?.rulesSelections?.assurance}
function validAssurance(item){
 const skill=assuranceSkill(item);
 return itemSource(item)===assuranceSource&&typeof skill==='string'&&same(originalRules(item),[
  {key:'ChoiceSet',choices:{config:'skills',predicate:[{gte:['skill:{choice|value}:rank',1]}]},flag:'assurance',prompt:'PF2E.SpecificRule.Prompt.Skill',selection:skill},
  {key:'SubstituteRoll',label:'PF2E.SpecificRule.SubstituteRoll.Assurance',selector:assuranceSelector,slug:'assurance',value:10},
  {key:'AdjustModifier',selector:assuranceSelector,predicate:['substitute:assurance',{not:'bonus:type:proficiency'}],suppress:true}
 ]);
}
function testPredicate(predicate,options,game){
 if(predicate===undefined)return true;
 if(typeof game.pf2e.Predicate!=='function'||!(predicate instanceof game.pf2e.Predicate))fail('native-predicate-unverified');
 return predicate.test(options);
}
function knownSuppress(adjustment){
 const keys=Object.keys(adjustment).filter(key=>key!=='applications').sort();
 return same(keys,['getDamageType','getNewValue','slug','suppress','test'])&&adjustment.slug===null&&adjustment.suppress===true&&['test','getNewValue','getDamageType'].every(key=>typeof adjustment[key]==='function');
}
function projection({game,actor,skill,slugs,riskySurgery,assurance,source}){
 if(game.system?.version!=='8.5.1'||typeof game.pf2e?.CheckModifier!=='function')fail('native-estimate-version-unverified');
 if(slugs.some(slug=>['magic-hands','mortal-healing'].includes(slug)))fail('native-healing-model-unverified');
 const items=values(actor.items).filter(item=>!item.isSuppressed&&!item.system?.suppressed);
 const riskyItems=items.filter(item=>itemSource(item)===riskySource||item.slug==='risky-surgery'||item.system?.slug==='risky-surgery');
 if(riskyItems.some(item=>!validRisky(item))||riskyItems.length>1)fail('risky-source-unverified');
 if(riskySurgery&&(skill!=='medicine'||riskyItems.length!==1))fail('risky-selection-unavailable');
 const assuranceItems=items.filter(item=>itemSource(item)===assuranceSource||item.slug==='assurance'||item.system?.slug==='assurance');
 if(assuranceItems.some(item=>!validAssurance(item)))fail('assurance-source-unverified');
 const matching=assuranceItems.filter(item=>assuranceSkill(item)===skill);
 if(matching.length>1||assurance&&matching.length!==1)fail('assurance-selection-unavailable');

 // The first check getter can construct StatisticCheck and run adjustments.
 // Establish source safety before reading that getter or a statistic's mod.
 const synthetics=actor.synthetics??{},rules=values(actor.rules).filter(rule=>!rule.ignored);
 if(rules.filter(rule=>rule.key==='AdjustModifier'&&matching.includes(rule.item)).length!==matching.length)fail('native-assurance-rule-unverified');
 for(const [domain,entries] of Object.entries(synthetics.modifierAdjustments??{})){
  const expected=assuranceItems.filter(item=>assuranceSkill(item)===domain);
  if(!Array.isArray(entries)||entries.length!==expected.length||entries.some(entry=>!knownSuppress(entry)))fail('native-modifier-adjustment-unverified');
 }
 for(const rule of rules){
  if(rule.key==='AdjustModifier'&&!assuranceItems.includes(rule.item))fail('native-adjustment-source-unverified');
  if(typeof rule.beforeRoll==='function'&&rule.key!=='RollOption'&&!(rule.key==='ActiveEffectLike'&&rule.phase!=='beforeRoll'))fail('native-before-roll-unverified');
 }

 // Read the already prepared statistic. Constructing another Statistic can run
 // modifier adjustment closures before we have established their safety.
 const stat=actor.getStatistic?.(skill),check=stat?.check,domains=check?.domains;
 if(!Number.isInteger(stat?.rank)||stat.rank<1||stat.rank>4||!Array.isArray(domains)||!domains.includes(skill)||!Array.isArray(check.modifiers))fail('native-statistic-unverified');
 const extra=[...actor.getRollOptions(['all','skill-check','medicine']),'action:treat-wounds',`check:statistic:${skill}`,'check:type:skill',...['exploration','healing','manipulate'].flatMap(trait=>[trait,`item:trait:${trait}`])];
 if(stat.base?.slug)extra.push(`check:statistic:base:${stat.base.slug}`);
 const options=check.createRollOptions({origin:actor,extraRollOptions:extra});
 if(!(options instanceof Set)||options.has('fortune')||options.has('misfortune'))fail('native-roll-options-unverified');
 if(skill!=='medicine'&&options.has('risky-surgery'))fail('native-nature-risky-context-unverified');
 for(const rule of rules){
  if(rule.key==='AdjustModifier'&&!assuranceItems.includes(rule.item))fail('native-adjustment-source-unverified');
  if(typeof rule.beforeRoll!=='function')continue;
  if(rule.key==='ActiveEffectLike'&&rule.phase!=='beforeRoll')continue;
  if(rule.key==='RollOption'&&typeof rule.domain==='string'&&!rule.domain.includes('{')&&!domains.includes(rule.domain))continue;
  if(rule.key==='RollOption'&&rule.domain==='medicine'&&rule.option==='risky-surgery'&&typeof rule.value==='boolean'&&!rule.suboptions?.length&&riskyItems.includes(rule.item))continue;
  fail('native-before-roll-unverified');
 }
 if(riskySurgery)options.add('risky-surgery');else if(skill==='medicine')options.delete('risky-surgery');
 options.delete('substitute:assurance');
 if(assurance)options.add('substitute:assurance');

 if(values(synthetics.ephemeralEffects?.['damage-received']?.target).length)fail('native-patient-context-unverified');
 const adjustments=[...new Set(domains.flatMap(domain=>synthetics.modifierAdjustments?.[domain]??[]))];
 if(adjustments.length!==matching.length||adjustments.some(adjustment=>!knownSuppress(adjustment)))fail('native-modifier-adjustment-unverified');
 const modifiers=check.modifiers.map(modifier=>{
  if(typeof modifier.clone!=='function'||!Number.isFinite(modifier.modifier)||!Array.isArray(modifier.adjustments)||modifier.adjustments.length!==adjustments.length||modifier.adjustments.some(adjustment=>!knownSuppress(adjustment)))fail('native-modifier-unverified');
  testPredicate(modifier.predicate,options,game);
  if(modifier.ignored&&modifier.predicate.length===0)fail('native-modifier-suppression-unverified');
  const copy=modifier.clone();
  // These are only the exact canonical Assurance suppression adjustments.
  // Their prepared functions contain closures over the original rule object.
  copy.adjustments=[];
  return copy;
 });
 if(riskySurgery&&!modifiers.some(modifier=>modifier.slug==='risky-surgery'&&modifier.type==='circumstance'&&modifier.modifier===2&&riskyItems.includes(modifier.rule?.item)))fail('native-risky-modifier-unverified');
 const substitutions=domains.flatMap(domain=>synthetics.rollSubstitutions?.[domain]??[]).filter(entry=>testPredicate(entry.predicate,options,game));
 if(substitutions.some(entry=>entry.required)||!assurance&&substitutions.some(entry=>entry.selected))fail('native-substitution-conflict');
 const actualAssurance=substitutions.filter(entry=>entry.slug==='assurance');
 if(assurance&&(actualAssurance.length!==1||actualAssurance[0].value!==10||actualAssurance[0].effectType!=='fortune'))fail('native-assurance-substitution-unverified');
 if(domains.flatMap(domain=>synthetics.rollTwice?.[domain]??[]).some(entry=>testPredicate(entry.predicate,options,game)))fail('native-roll-twice-unverified');
 const selectedModifiers=assurance?modifiers.filter(modifier=>modifier.type==='proficiency'):modifiers;
 const prepared=new game.pf2e.CheckModifier(skill,{modifiers:selectedModifiers},[],options);
 if(!Number.isFinite(prepared.totalModifier)||assurance&&prepared.modifiers.filter(modifier=>modifier.enabled&&modifier.type==='proficiency').length!==1)fail('native-prepared-modifier-unverified');
 const degreeAdjustments=domains.flatMap(domain=>synthetics.degreeOfSuccessAdjustments?.[domain]??[]);
 for(const entry of degreeAdjustments)if(entry.predicate!==undefined&&!(entry.predicate instanceof game.pf2e.Predicate))fail('native-degree-predicate-unverified');
 if(riskySurgery&&!degreeAdjustments.some(entry=>same(entry.predicate?.toObject?.(),['risky-surgery','action:treat-wounds'])&&entry.adjustments?.success?.amount===1))fail('native-risky-outcome-unverified');
 if(assurance)options.add('fortune');
 const outcomesByRank=treatmentOutcomeRows({rank:stat.rank,modifier:prepared.totalModifier,assurance,options,adjustments:degreeAdjustments});
 return {riskySurgery,assurance,ready:true,reason:null,sourceVersion:'8.5.1',source,modifier:prepared.totalModifier,domains:[...domains],options:[...options],outcomesByRank};
}

/** Prepare only source-verified, side-effect-free variants of the native check. */
export function preparedTreatmentSelections({game,actor,skill,slugs=[]}){
 const source={actorUUID:actor.uuid,skill,ruleSources:[...new Set(values(actor.rules).filter(rule=>!rule.ignored).map(rule=>rule.item?.uuid).filter(uuid=>typeof uuid==='string'))]};
 return [false,true].flatMap(riskySurgery=>[false,true].map(assurance=>{
  try{return projection({game,actor,skill,slugs,riskySurgery,assurance,source})}
  catch(error){return {riskySurgery,assurance,ready:false,reason:error.message,sourceVersion:'8.5.1',source}}
 }));
}

export function simpleTreatmentDamageModel({actor,poolUUID}){
 const attributes=actor?.attributes,hp=attributes?.hp;
 return poolUUID===actor?.uuid&&actor.hardness===0&&hp?.temp===0&&(hp.sp===undefined||hp.sp?.max===0)&&
  ['immunities','resistances','weaknesses'].every(key=>Array.isArray(attributes?.[key])&&attributes[key].length===0);
}
