import {MODULE_ID,hasSource} from './rules.mjs';
export const AUTOMATIC_KNOWLEDGE_SOURCE='Compendium.pf2e.feats-srd.Item.H3I2X0f7v4EzwxuN';
export const ASSURANCE_SOURCE='Compendium.pf2e.feats-srd.Item.W6Gl9ePmItfDHji0';
const values=x=>Array.from(x?.values?.()??x??[]);
const recallSkills=new Set(['arcana','crafting','medicine','nature','occultism','religion','society']);
export function assuranceSkill(item){return item?.flags?.system?.rulesSelections?.assurance??item?.flags?.pf2e?.rulesSelections?.assurance??item?.system?.rules?.find(r=>r.key==='ChoiceSet'&&r.flag==='assurance')?.selection??null;}
export function hasSkillAssurance(actor,skill){return values(actor?.items).some(item=>hasSource(item,ASSURANCE_SOURCE)&&assuranceSkill(item)===skill);}
/** Automatic Knowledge has no ChoiceSet in PF2e's native feat. When more than
 * one eligible Assurance skill exists, configure the feat's permanent choice
 * once. This never asks a player which skill identifies the targeted creature. */
export function automaticKnowledgeChoices(actor,item){
 if(item?.actor?.uuid!==actor?.uuid||!hasSource(item,AUTOMATIC_KNOWLEDGE_SOURCE))throw Error('耳熟能详专长来源不匹配。');
 const choices=[...new Set(values(actor.items).filter(i=>hasSource(i,ASSURANCE_SOURCE)).map(assuranceSkill))].filter(slug=>{const stat=actor.skills?.[slug];return stat?.rank>=2&&(recallSkills.has(slug)||stat.lore);}).map(slug=>({value:slug,label:actor.skills[slug].label??slug}));
 const selected=item.flags?.[MODULE_ID]?.knowledge?.automaticSkill??item.flags?.system?.rulesSelections?.automaticKnowledge??item.flags?.pf2e?.rulesSelections?.automaticKnowledge;
 if(selected&&!choices.some(c=>c.value===selected))throw Error('耳熟能详的固定技能必须为专家以上且具有相应驾轻就熟。');
 return {choices,statistic:selected??(choices.length===1?choices[0].value:null)};
}
export function automaticKnowledgeRound(game){const combat=game.combat;if(!combat?.started||!Number.isInteger(combat.round))throw Error('耳熟能详需要已开始的遭遇以记录每轮一次。');return `${combat.uuid??combat.id}:${combat.round}`;}
