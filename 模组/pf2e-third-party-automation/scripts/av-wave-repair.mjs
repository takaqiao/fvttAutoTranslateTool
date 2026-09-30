import {MODULE_ID} from './rules.mjs';
export const WAVE_SOURCE='Compendium.pf2e.classfeatures.Item.5tSR0WzMPFn5s3Xs';
export const WAVE_SPELL_SLUGS=Object.freeze(['arctic-rift','blazing-bolt','breathe-fire','falling-stars','fireball','frostbite','frozen-fog','howling-blizzard','ice-storm','ignition','volcanic-eruption']);
export const WAVE_SPELL_SOURCES=Object.freeze(['y6rAdMK6EFlV6U0t','ZxHC7V7HtjUsB8zH','sxQZ6yqTn0czJxVd','kHyjQbibRGPNCixx','xxWhyl81w3ckslAU','nOVSmPZsCm1C1sI3','O7ZEqWjwdKyo2CUv','C2GYCH3TtUFqPfdX','jrBa9deU2ULFWvSl','IxhGEKl63R4QBvkj','6DfLZBl8wKIV03Iq'].map(id=>'Compendium.pf2e.spells-srd.Item.'+id));
const source=item=>item?.sourceId??item?._stats?.compendiumSource??item?.flags?.core?.sourceId;
export const getAvWaveFeature=actor=>Array.from(actor?.items?.values?.()??actor?.items??[]).find(i=>source(i)===WAVE_SOURCE&&!i.isSuppressed);
export const isAvWaveSpell=item=>item?.type==='spell'&&WAVE_SPELL_SOURCES.includes(source(item))&&!!getAvWaveFeature(item.actor);
// PF2e returns null for a valid toggle whose source already has that value.
// Verify the prepared result, including rejection of contradictory selections.
export const hasAvWaveEnergy=(actor,energy)=>{
 const selections=(actor.getRollOptions?.(['all'])??[]).filter(option=>/^conservation-of-energy:(?:fire|cold|none)$/.test(option));
 return selections.length===1&&selections[0]==='conservation-of-energy:'+energy;
};
const ORIGINAL=['arctic-rift','breathe-fire','blazing-bolt','falling-star','fireball','frostbite','frozen-fog','ice-storm','ignition','volcanic-eruption'].map(s=>'item:slug:'+s);
const clone=value=>globalThis.foundry?.utils?.deepClone?.(value)??structuredClone(value);
/** Only the verified PF2e 8.5 source typo/omission; custom lists are preserved. */
export function buildAvWaveRepair(item){
 const sourceId=source(item);
 if(sourceId!==WAVE_SOURCE)return null;
 const rules=clone(item.system.rules),changes=[],prior=item.flags?.[MODULE_ID]?.avWaveRepair;
 for(const[index,rule]of rules.entries()){
  const list=rule.predicate?.length===1?rule.predicate[0]?.or:null;
  const original=Array.isArray(list)&&list.length===ORIGINAL.length&&ORIGINAL.every(s=>list.includes(s));
  const intermediate=prior?.sourceId===WAVE_SOURCE&&Array.isArray(list)&&list.length===WAVE_SPELL_SLUGS.length&&WAVE_SPELL_SLUGS.every(s=>list.includes('item:slug:'+s));
  if(rule.key!=='DamageAlteration'||rule.mode!=='override'||rule.property!=='damage-type'||rule.slug!=='base'||JSON.stringify(rule.selectors)!=='["spell-damage"]'||rule.value!=='{item|flags.system.rulesSelections.conservationOfEnergy}'||(!original&&!intermediate))continue;
  const before=clone(rule.predicate);rule.predicate=[{or:WAVE_SPELL_SLUGS.map(s=>'item:slug:'+s)},{or:['conservation-of-energy:fire','conservation-of-energy:cold']}];changes.push({index,path:'predicate',before,after:clone(rule.predicate)});
 }
 const recognized=changes.length||rules.some(r=>r.key==='DamageAlteration'&&JSON.stringify(r.predicate)===JSON.stringify([{or:WAVE_SPELL_SLUGS.map(s=>'item:slug:'+s)},{or:['conservation-of-energy:fire','conservation-of-energy:cold']}])&&r.value==='{item|flags.system.rulesSelections.conservationOfEnergy}');
 if(!recognized)return null;
 for(const[index,rule]of rules.entries())if(rule.key==='RollOption'&&rule.option==='conservation-of-energy'&&rule.alwaysActive&&rule.placement==='spellcasting'&&Array.isArray(rule.suboptions)&&rule.suboptions.length===2&&['fire','cold'].every(s=>rule.suboptions.some(o=>o.value===s))){
  const before=clone(rule.suboptions);rule.suboptions=[{label:'中性（再聚能）',value:'none'},...rule.suboptions];changes.push({index,path:'suboptions',before,after:clone(rule.suboptions)});
 }
 if(!changes.length)return null;
 return {'system.rules':rules,[`flags.${MODULE_ID}.avWaveRepair`]:{...prior,sourceId,reason:'PF2e 8.5 granted spell UUIDs and legal Refocus neutrality',changes:[...(prior?.changes??[]),...changes]}};
}
