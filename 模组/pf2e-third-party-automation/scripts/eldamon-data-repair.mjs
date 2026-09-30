import {MODULE_ID} from './rules.mjs';
import {SerialActions} from './runtime.mjs';

const prefix='Compendium.battlezoo-eldamon-pf2e.';
const sources=Object.freeze({
 powers:prefix+'eldamon-features.Item.naawsnBug9EOpzfN',electricity:prefix+'eldamon-features.Item.9KtNlRXeuxZSoVaI',
 resistance:prefix+'feats.Item.5pbln7VjWjAIxLZT',surge:prefix+'powers.Item.veFrnrxYjlqca13w',
 voltage:prefix+'powers.Item.9bElF2uVf5FCJtb9',siphon:prefix+'actions.Item.4w72ljp4eLBqeZB2',widen:prefix+'feats.Item.3YasBiZw3N96rdUW',
 wraps:'Compendium.pf2e.equipment-srd.Item.FNDq4NFSN0g2HKWO',
});
const sourceOf=item=>item?.sourceId??item?._stats?.compendiumSource??item?.flags?.core?.sourceId;
const includes=(value,target)=>Array.isArray(value)?value.includes(target):value===target;
const copy=value=>structuredClone(value);

/** Narrow, source-identified data omissions; never replace a customized rule. */
export function buildEldamonDataRepairs(actor){
 if(actor?.type!=='character'||actor.flags?.[MODULE_ID]?.autoRepairDisabled)return [];
 const items=Array.from(actor.items??[]),has=key=>items.some(item=>sourceOf(item)===sources[key]),updates=[];
 for(const item of items){
  const source=sourceOf(item),system=item.system??{},patch={},before={};
  const change=(path,value,old)=>{patch[path]=copy(value);before[path]=copy(old)};
  const rules=Array.isArray(system.rules)?system.rules:[];
  if(source===sources.wraps&&has('powers')&&!rules.some(rule=>rule.key==='FlatModifier'&&includes(rule.selector,'eldamon-power-attack')&&rule.type==='item')){
   change('system.rules',[...rules,{key:'FlatModifier',selector:'eldamon-power-attack',value:'@item.system.runes.potency',type:'item'}],rules);
  }
  if(source===sources.resistance&&has('electricity')&&!rules.some(rule=>rule.key==='Resistance'&&includes(rule.type,'electricity'))){
   change('system.rules',[...rules,{key:'Resistance',type:'electricity',value:'floor(@actor.level / 2)',predicate:['feature:electricity-element']}],rules);
  }
  if(source===sources.surge&&typeof system.description?.value==='string'){
   const old=system.description.value;
   const corrected=old.replace(/@Damage\[\(1\+@actor\.level\)d4\[electricity\]\]\{d8(s?)\}/g,'@Damage[(1+@actor.level)d8[electricity]]{d8$1}');
   if(corrected!==old)change('system.description.value',corrected,old);
  }
  const addTraits=source===sources.voltage?['refresh']:[sources.siphon,sources.widen].includes(source)?['elemental-avatar','metapower']:[];
  const traits=Array.isArray(system.traits?.value)?system.traits.value:[];
  if(addTraits.some(trait=>!traits.includes(trait)))change('system.traits.value',[...new Set([...traits,...addTraits])],traits);
  if(!Object.keys(patch).length)continue;
  const previous=item.flags?.[MODULE_ID]?.eldamonDataRepair;
  patch[`flags.${MODULE_ID}.eldamonDataRepair`]={version:1,before:{...before,...copy(previous?.before??{})}};
  updates.push({_id:item.id??item._id,...patch});
 }
 return updates;
}

export function createEldamonDataRepair({game}){
 const queue=new SerialActions();
 return {maintain:async actor=>{
  if(!actor||game.user!==game.users.activeGM)return;
  return queue.run(actor.uuid,async()=>{
   if(game.user!==game.users.activeGM)return;
   const updates=buildEldamonDataRepairs(actor);
   if(updates.length)await actor.updateEmbeddedDocuments('Item',updates);
  });
 }};
}
