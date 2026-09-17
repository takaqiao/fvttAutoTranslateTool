import {MODULE_ID} from './rules.mjs';
import {getSourceId} from './native-context.mjs';

export const ELEMENTAL_MEDICINE_SOURCE='Compendium.pf2e.feats-srd.Item.NMbOfl885pn0Yzu9';
export const ELEMENTAL_MEDICINE_EFFECT='Compendium.pf2e.feat-effects.Item.ZqEOsqnGFLI2ob9m';
export const ELEMENTAL_MEDICINE_DAILY=`${MODULE_ID}-elemental-medicine`;
export const ELEMENTAL_MEDICINE_ELEMENTS=Object.freeze({
 earth:{element:'earth',resistance:'void',weakness:'vitality'},fire:{element:'fire',damageType:'fire',resistance:'vitality',weakness:'void'},
 metal:{element:'metal',resistance:'vitality',weakness:'void'},water:{element:'water',resistance:'void',weakness:'vitality'},wood:{element:'wood',resistance:'void',weakness:'vitality'},
});
const values=c=>Array.from(c?.values?.()??c??[]);
const levels=[13,14,15,16,18,19,20,22,23,24,26,27,28,30,31,32,34,35,36,38,39,40,42,44,46,48,50];
const skills=['crafting','medicine','herbalism-lore'];
const labels={crafting:'手艺 Crafting',medicine:'医疗 Medicine','herbalism-lore':'草药学识 Herbalism Lore'};
export const elementalMedicineFeat=actor=>values(actor?.items).find(i=>i.type==='feat'&&getSourceId(i)===ELEMENTAL_MEDICINE_SOURCE&&!i.isSuppressed&&!i.system?.suppressed);
export const hasElementalMedicine=actor=>actor?.type==='character'&&!!elementalMedicineFeat(actor);
/** Exact PF2e 8.5 standard DC table (bundle 16766), without invented rarity adjustment. */
export function elementalMedicineDC(level,{pwol=false,adjustment=0}={}){
 if(!Number.isInteger(level)||level< -1||level>25||!Number.isInteger(adjustment)||Math.abs(adjustment)>50)throw Error('患者等级或GM提供的DC调整无效。');
 return levels[level+1]-(pwol?Math.max(level,0):0)+adjustment;
}
export function elementalMedicineSkills(actor){return skills.flatMap(slug=>{
 const statistic=actor?.getStatistic?.(slug)??actor?.skills?.[slug];return statistic?[{slug,label:labels[slug],statistic}]:[];
});}
export function validateElementalMedicinePatients(actor,rows,patients){
 if(!hasElementalMedicine(actor))throw Error('角色没有可用的五气养生专长。');
 if(Object.entries(rows??{}).some(([key,value])=>/^patient\d+$/.test(key)&&Number(key.slice(7))>6&&value))throw Error('一次每日准备最多诊治六名患者。');
 const allowed=new Set(elementalMedicineSkills(actor).map(s=>s.slug)),result=[];
 for(let index=1;index<=6;index++){
  const uuid=rows?.[`patient${index}`];if(uuid==null||uuid==='')continue;
  const patient=patients.find(a=>a.uuid===uuid),skill=rows?.[`skill${index}`];
  if(!patient||!['character','npc','familiar'].includes(patient.type)||!allowed.has(skill))throw Error('患者或其所选原生技能已不可用。');
  if(result.some(r=>r.patientUuid===uuid))throw Error('同一次每日准备不能重复诊治同一患者。');
  result.push({patientUuid:uuid,skill});
 }return result;
}
/** Native save roll options prove both the exact item and its actual origin. */
export function elementalMedicineBinding(item){
 const actor=item?.actor;
 if(!item?.id||!actor?.uuid||typeof actor.signature!=='string'||!actor.signature||item.uuid!==`${actor.uuid}.Item.${item.id}`||typeof item.getRollOptions!=='function'||!item.getRollOptions('item').includes(`item:id:${item.id}`))throw Error('需要准确的现有内嵌病症豁免来源；不能以通用疾病标签代替。');
 return {itemUuid:item.uuid,itemId:item.id,originUuid:actor.uuid,originSignature:actor.signature,predicate:[`item:id:${item.id}`,`origin:signature:${actor.signature}`]};
}
export function buildElementalMedicineEffect({source,degree,skill,element,binding,actor,item,patientUuid,nonce,checkId,worldTime,effectId}={}){
 if(degree===1)return null;
 if(![0,2,3].includes(degree)||source?.type!=='effect'||!skills.includes(skill)||!Object.hasOwn(ELEMENTAL_MEDICINE_ELEMENTS,element)||!actor?.uuid||!patientUuid||!nonce||!checkId||!Number.isFinite(worldTime)||!effectId||!binding?.itemId||!binding.originSignature||JSON.stringify(binding.predicate)!==JSON.stringify([`item:id:${binding.itemId}`,`origin:signature:${binding.originSignature}`]))throw Error('五气养生的实际检定、元素或精确病症来源不完整。');
 const data=structuredClone(source),bonus={0:-1,2:1,3:2}[degree];data._id=effectId;delete data.folder;delete data.sort;delete data.ownership;
 data.name='五气养生';data.system.description={value:'<p>五气养生的药效持续24小时，仅适用于本次诊治的病症豁免。再次每日准备需要重新诊断；同时只能有一种药效。</p>',gm:''};
 data.system.duration={value:24,unit:'hours',expiry:null,sustained:false};data.system.start={value:worldTime,initiative:null};data.system.badge=null;
 data.system.context={origin:{actor:actor.uuid,item:item?.uuid??`${actor.uuid}.Item.${item?.id}`,token:null},target:{actor:patientUuid,token:null},roll:null};
 const modifier={key:'FlatModifier',selector:'saving-throw',slug:'prepare-elemental-medicine',label:'五气养生',type:'circumstance',value:bonus,predicate:[...binding.predicate]};
 const extra=(data.system.rules??[]).filter(r=>r.key!=='ChoiceSet'&&!(r.key==='FlatModifier'&&r.selector==='saving-throw'));
 data.system.rules=[modifier,...extra];
 data.flags??={};data.flags.system={...data.flags.system,rulesSelections:{...data.flags.system?.rulesSelections,value:bonus,skill,prepareElementalMedicine:structuredClone(ELEMENTAL_MEDICINE_ELEMENTS[element])}};
 data.flags[MODULE_ID]={elementalMedicine:{kind:'medicine',nonce,checkId,patientUuid,sourceActorUuid:actor.uuid,afflictionUuid:binding.itemUuid}};
 data._stats={...data._stats,compendiumSource:ELEMENTAL_MEDICINE_EFFECT};return data;
}
export function isActiveElementalMedicine(effect,time){
 if(effect?.type!=='effect'||getSourceId(effect)!==ELEMENTAL_MEDICINE_EFFECT||effect.isExpired===true)return false;
 const start=effect.system?.start?.value,duration=effect.system?.duration,multiplier={rounds:6,minutes:60,hours:3600,days:86400}[duration?.unit];
 return Number.isFinite(start)&&Number.isFinite(duration?.value)&&Number.isFinite(multiplier)&&time<start+duration.value*multiplier;
}
