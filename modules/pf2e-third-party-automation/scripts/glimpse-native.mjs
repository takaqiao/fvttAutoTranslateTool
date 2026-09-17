import {MODULE_ID} from './rules.mjs';
export const glimpseMarker=nonce=>`${MODULE_ID}:glimpse:${nonce}`;
export function validGlimpseTemplate(template){const r=template?.system?.rules;return template?.type==='effect'&&r?.length===1&&Object.keys(r[0]).sort().join(',')==='key,type,value'&&r[0].key==='Resistance'&&r[0].type==='all-damage'&&r[0].value==='@item.origin.level+2'}
/** Prepare the published rule through PF2e; the rule-element predicate itself
 * is not retained by the resulting Resistance, so guard the native instance. */
export function compileGlimpseResistance({actor,template,champion,ability,nonce,options=[]}){
 if(!validGlimpseTemplate(template)||champion?.level!==5||typeof actor.getContextualClone!=='function')throw Error('救赎瞥视原生抗力模板或等级无法验证。');
 const effect=structuredClone(template);
 // Contextual clones embed their supplied source data immediately: unlike
 // createEmbeddedDocuments, this path does not assign new document IDs.
 do{effect._id=globalThis.foundry?.utils?.randomID?.(16)??crypto.randomUUID().replaceAll('-','').slice(0,16)}while(actor.items?.has?.(effect._id));
 effect.system.context={origin:{actor:champion.uuid,item:ability.uuid},target:null,roll:null};
 const clone=actor.getContextualClone([...new Set([...options,glimpseMarker(nonce)])],[effect]);
 const resistance=clone.attributes?.resistances?.find(r=>r.type==='all-damage'&&!(r.exceptions?.length)&&!(r.doubleVs?.length)&&r.value>=champion.level+2&&typeof r.test==='function'&&typeof r.getDoubledValue==='function');
 if(!resistance||actor.attributes?.resistances?.includes(resistance))throw Error('无法生成独立的救赎瞥视原生抗力。');
 const nativeTest=resistance.test.bind(resistance);
 resistance.test=options=>new Set(options??[]).has(glimpseMarker(nonce))&&nativeTest(options);
 return resistance;
}
export async function withGlimpseResistance(actor,resistance,native){
 const original=actor.attributes?.resistances;if(!Array.isArray(original))throw Error('原生抗力列表不可用。');
 original.push(resistance);
 try{return await native()}finally{for(const list of new Set([original,actor.attributes?.resistances])){if(!Array.isArray(list))continue;const i=list.indexOf(resistance);if(i>=0)list.splice(i,1)}}
}
export function repentParams(params,nonce){return {...params,damage:0,final:true,shieldBlockRequest:false,rollOptions:new Set([...params.rollOptions??[],glimpseMarker(nonce)])}}
