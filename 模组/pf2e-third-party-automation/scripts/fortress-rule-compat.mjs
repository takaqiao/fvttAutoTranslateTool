import {MODULE_ID} from './rules.mjs';
import {getSourceId,isActiveGM} from './native-context.mjs';
import {SerialActions} from './runtime.mjs';
import {FORTRESS_RULE_COMPAT_WORLDS,BOB2_AUTOMATION_WORLDS} from './world-gates.mjs';

const ALBATROSS='FTGjyYBZJ4JCpTZw',HANDS='fzUfZcstmQZGGwcJ';
const OVERTURE='Compendium.pf2e.spell-effects.Item.ITErgFRfydm1xmnW';
const values=collection=>Array.from(collection?.values?.()??collection??[]);
const copy=value=>structuredClone(value),equal=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
const sourceGroup=(group,id,source)=>group?.uuid===id&&group.source?.length===1&&group.source[0]===source;
const includesAll=(predicate,terms)=>Array.isArray(predicate)&&terms.every(term=>predicate.includes(term));

export const fortressRuleCompatibilityEnabled=game=>FORTRESS_RULE_COMPAT_WORLDS.has(game?.world?.id)&&game.system?.id==='pf2e';

/** Configuration maintenance owns the backup/write. Repair only the captured
 * upstream branches; custom predicates and enabled/disabled state survive. */
export function buildFortressPatreonRepairs(original,{game}={}){
 const rules=copy(original),changes=[],patreon=game?.modules?.get('patreon-v3');
 if(!fortressRuleCompatibilityEnabled(game)||BOB2_AUTOMATION_WORLDS.has(game?.world?.id)||patreon?.active!==true)return {rules,changes};
 const change=(object,key,after,path,reason)=>{const before=copy(object[key]);object[key]=copy(after);changes.push({path,before,after:copy(after),reason})};
 const albatross=rules?.[ALBATROSS];
 if(sourceGroup(albatross,ALBATROSS,'Compendium.pf2e.spells-srd.Item.93SjFTGJUTTmAt6j')&&Array.isArray(albatross.complexRules)){
  const candidates=albatross.complexRules.map((rule,index)=>({rule,index})).filter(({rule})=>{
   const value=rule.values?.[0];
   return rule.type==='complex'&&rule.triggerType==='saving-throw'&&rule.target==='SelfEffect'&&includesAll(rule.predicate,['origin:item:albatross-curse','outcome:criticalFailure'])&&rule.values?.length===1&&equal(value?.conditions,[])&&equal(value?.effects,['Compendium.patreon-v3.effects.Item.xTHKw0VHuqu6SI93'])&&value.duration?.unit==='hours'&&Object.keys(value.duration).length===1;
  });
  if(candidates.length===1){
   const {rule,index}=candidates[0];
   change(rule.values[0],'effects',['Compendium.pf2e.spell-effects.Item.jSvpjSGnIVBAuFDu'],`${ALBATROSS}.complexRules.${index}.values.0.effects`,'信天翁诅咒大失败使用一小时内下一次意志取低效果，避免误授予缓慢。');
  }
 }
 const hands=rules?.[HANDS];
 if(sourceGroup(hands,HANDS,'Compendium.pf2e.spells-srd.Item.zNN9212H2FGfM7VS')&&Array.isArray(hands.baseRules)){
  const candidates=hands.baseRules.map((rule,index)=>({rule,index})).filter(({rule})=>rule.type==='base'&&rule.triggerType==='saving-throw'&&rule.target==='TargetEffect'&&rule.value==='Compendium.pf2e.spell-effects.Item.JhihziXQuoteftdd'&&includesAll(rule.predicate,['outcome:criticalFailure','item:slug:lay-on-hands','self:trait:undead'])&&rule.predicate.filter(p=>typeof p==='string'&&p.startsWith('outcome:')).length===1);
  if(candidates.length===1){
   const {rule,index}=candidates[0],path=`${HANDS}.baseRules.${index}`;
   change(rule,'value','Compendium.pf2e.spell-effects.Item.lyLMiauxIVUM3oF1',`${path}.value`,'圣疗使用现行统一AC效果；原生受术者self:mode:undead把+2调整为-2。');
   change(rule,'target','SelfEffect',`${path}.target`,'圣疗豁免后的减值给予实际进行豁免的不死生物，不读取当前选中目标。');
   change(rule,'predicate',rule.predicate.map(p=>p==='outcome:criticalFailure'?{or:['outcome:failure','outcome:criticalFailure']}:p),`${path}.predicate`,'圣疗不死分支的普通失败和大失败均施加一轮AC减值。');
  }
 }
 return {rules,changes};
}

function upliftingRepair(item,{aidOnly=false}={}){
 if(item?.type!=='effect'||getSourceId(item)!==OVERTURE)return null;
 const before=item._source?.system?.rules??item.system?.rules;
 if(!Array.isArray(before))return null;
 const candidates=before.map((rule,index)=>({rule,index})).filter(({rule})=>rule.key==='AdjustDegreeOfSuccess'&&rule.selector==='performance'&&(equal(rule.adjustment,{success:'one-degree-better'})||aidOnly&&equal(rule.adjustment,{failure:'one-degree-better'}))&&equal(rule.predicate,[{lte:['skill:performance:rank',3]}]));
 if(candidates.length!==1)return null;
 const rules=copy(before);rules[candidates[0].index].adjustment={failure:'one-degree-better'};
 if(aidOnly){
  rules[candidates[0].index].predicate.push('action:aid');
  for(const rule of rules)if(rule.key==='AdjustDegreeOfSuccess'&&rule.selector==='performance'&&equal(rule.adjustment,{all:'to-critical-success'})&&equal(rule.predicate,[{gte:['skill:performance:rank',4]}]))rule.predicate.push('action:aid');
 }
 const prior=item.flags?.[MODULE_ID]?.fortressRuleRepair;
 return {'system.rules':rules,[`flags.${MODULE_ID}.fortressRuleRepair`]:{version:1,before:{'system.rules':copy(before),...copy(prior?.before??{})}}};
}

function bolsterMindRepair(item){
 if(item?.type!=='effect'||getSourceId(item)!=='Compendium.pf2e.feat-effects.Item.52owimBcCQNBuNfM')return null;
 const before=item._source?.system?.rules??item.system?.rules;if(!Array.isArray(before))return null;
 const rules=copy(before);
 for(const rule of rules){
  if(rule.key==='Note'&&rule.text==='PF2E.SpecificRule.BastionOfBlasphemies.Background.BloodedByTheDead.BolsterTheMind.Note'&&equal(rule.selector,['saving-throw'])&&equal(rule.predicate,['blooded-by-the-dead:bolster-the-fury','item:trait:incapacitation']))rule.predicate=['blooded-by-the-dead:bolster-the-mind','item:trait:incapacitation','item:trait:mental'];
  if(rule.key==='FlatModifier'&&rule.value===2&&rule.type===undefined&&equal(rule.selector,['will'])&&equal(rule.predicate,['blooded-by-the-dead:bolster-the-mind','item:slug:dominate','origin:trait:vampire']))rule.type='circumstance';
 }
 if(equal(before,rules))return null;
 const prior=item.flags?.[MODULE_ID]?.fortressRuleRepair;
 return {'system.rules':rules,[`flags.${MODULE_ID}.fortressRuleRepair`]:{version:2,before:{'system.rules':copy(before),...copy(prior?.before??{})}}};
}

/** A creating client repairs its own new effect synchronously. Existing owned
 * effects are repaired only through explicit active-GM actor maintenance. */
export function createFortressRuleCompatibility({game}){
 const queue=new SerialActions();
 const eligible=actor=>fortressRuleCompatibilityEnabled(game)&&actor&&!actor.flags?.[MODULE_ID]?.autoRepairDisabled;
 const patchItem=item=>upliftingRepair(item,{aidOnly:BOB2_AUTOMATION_WORLDS.has(game.world.id)})??(BOB2_AUTOMATION_WORLDS.has(game.world.id)?bolsterMindRepair(item):null);
 return {
  maintain:async actor=>{
   if(!eligible(actor)||!isActiveGM(game))return;
   return queue.run(actor.uuid,async()=>{
    if(!eligible(actor)||!isActiveGM(game))return;
    const updates=values(actor.items).flatMap(item=>{const patch=patchItem(item);return patch?[{_id:item.id??item._id,...patch}]:[]});
    if(updates.length)await actor.updateEmbeddedDocuments('Item',updates);
   });
  },
  register:({Hooks})=>{
   if(!fortressRuleCompatibilityEnabled(game))return()=>{};
   const id=Hooks.on('preCreateItem',(item,_data,_options,userId)=>{
    if(userId!==game.user?.id||!eligible(item.actor))return;
    const patch=patchItem(item);if(patch)item.updateSource(patch);
   });
   return()=>Hooks.off('preCreateItem',id);
  },
 };
}
