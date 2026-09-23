import {getSourceId} from './native-context.mjs';
export const MEDIC_SOURCES=Object.freeze({treatCondition:'Compendium.pf2e.feats-srd.Item.rfnEcjxIFqwlJwJT',visitation:'Compendium.pf2e.feats-srd.Item.1fBHZpM3Z3MQtzvi',legendaryMedic:'Compendium.pf2e.feats-srd.Item.Kk4AMZtpQnLEgN0b',battleMedicine:'Compendium.pf2e.feats-srd.Item.wYerMk6F1RZb0Fwt',toolkit:'Compendium.pf2e.equipment-srd.Item.s1vB3HdXjMigYAnY'});
export const values=c=>Array.from(c?.values?.()??c??[]);
export const medicFeat=(actor,key)=>values(actor?.items).find(i=>i.type==='feat'&&getSourceId(i)===MEDIC_SOURCES[key]);
export function medicAction(item){return item?.type==='feat'?({[MEDIC_SOURCES.treatCondition]:'medic:treat-condition',[MEDIC_SOURCES.visitation]:'medic:doctors-visitation'}[getSourceId(item)]??null):null;}
export function visitationBranches(actor){return [{value:'battle-medicine',label:'行走 → 战地医疗（1动作）',cost:1},{value:'treat-poison',label:'行走 → 治疗毒素（1动作）',cost:1},{value:'administer-first-aid',label:'行走 → 急救（2动作）',cost:2},...(medicFeat(actor,'treatCondition')?[{value:'treat-condition',label:'行走 → 处理状态（2动作）',cost:2}]:[])].filter(b=>b.value!=='battle-medicine'||medicFeat(actor,'battleMedicine'));}
export const conditionValue=c=>c?.value??c?.system?.value?.value;
export const conditionBound=c=>Boolean(c?.isLocked||c?.inMemoryOnly||c?.system?.references?.parent?.id||c?.flags?.pf2e?.grantedBy);
export function usableToolkit(actor){return values(actor?.items).some(i=>i.type==='equipment'&&[MEDIC_SOURCES.toolkit,'Compendium.pf2e.equipment-srd.Item.SGkOHFyBbzWdBk8D'].includes(getSourceId(i))&&(i.system?.quantity??0)>0&&!i.system?.containerId&&(i.system?.equipped?.carryType==='held'||i.system?.equipped?.carryType==='worn'&&(actor.handsFree??actor.system?.attributes?.handsFree??0)>0));}
export function validateTreatment({actor,condition,facts}){
 if(!condition||condition.type!=='condition'||!['clumsy','enfeebled','sickened'].includes(condition.slug)||!Number.isInteger(conditionValue(condition))||conditionValue(condition)<1)throw Error('没有所选的可处理状态。');
 if(conditionBound(condition))throw Error('此状态由父效果授予或锁定；当前不能安全单独修改，未删除父效果。');
 if(!usableToolkit(actor))throw Error('需要持握医疗工具包，或穿戴工具包且有空手。');
 if(!Number.isInteger(facts?.dc)||facts.dc<1||typeof facts.restricted!=='boolean'||typeof facts.continuous!=='boolean')throw Error('需要GM补齐真实来源DC、神器/20级以上及持续情境事实。');
 if(facts.continuous)throw Error('该状态持续来源的情境仍存在：处理状态无效。');
 if(facts.restricted&&!medicFeat(actor,'legendaryMedic'))throw Error('神器或20级以上来源需要传奇医师专长。');
 return {dc:facts.dc+(facts.restricted?10:0)};
}
export function treatmentValue(value,degree){if(!Number.isInteger(value)||value<1||!Number.isInteger(degree)||degree<0||degree>3)throw Error('无效的原生成功度或状态值。');return Math.max(0,value+[1,0,-1,-2][degree]);}
export function conditionSnapshot(condition){return JSON.stringify({id:condition.id,slug:condition.slug,value:conditionValue(condition),source:getSourceId(condition),system:condition._source?.system??condition.system,flags:condition._source?.flags??condition.flags,modified:condition._stats?.modifiedTime});}
