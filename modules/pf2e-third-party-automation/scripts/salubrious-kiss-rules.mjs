import {MODULE_ID} from './rules.mjs';
export const SALUBRIOUS_SOURCE='Compendium.pf2e.feats-srd.Item.Qg5M34t95rtT0sOp';
export const TREAT_WOUNDS_SOURCE='Compendium.pf2e.actionspf2e.Item.1kGNdIIhuglAjIp9';
export const TREAT_WOUNDS_IMMUNITY='Compendium.pf2e.feat-effects.Item.Lb4q2bBAgxamtix5';
export const sourceId=item=>item?.sourceId??item?._stats?.compendiumSource??item?.flags?.core?.sourceId;
export const values=collection=>Array.from(collection?.values?.()??collection??[]);
export const kissState=doc=>doc?.flags?.[MODULE_ID]?.salubriousKiss??{};
export const salubriousFeat=actor=>values(actor?.items).find(i=>i.type==='feat'&&sourceId(i)===SALUBRIOUS_SOURCE&&!i.isSuppressed);
const unsupported=new Set(['risky-surgery','mortal-healing','magic-hands','medic-dedication','continual-recovery','ward-medic']);
export function treatmentTiers(actor,{rankPolicy='selected-skill'}={}){
 if(!salubriousFeat(actor))throw Error('没有当前仙露三吻专长。');
 if(!['selected-skill','medicine-high-dc'].includes(rankPolicy))throw Error('Unknown treatment rank policy');
 const rank=actor.skills?.occultism?.rank;if(!Number.isInteger(rank)||rank<1||rank>4||actor.skills.occultism.proficient===false)throw Error('Occultism 必须至少受训。');
 const extra=values(actor.items).find(i=>i.type==='feat'&&unsupported.has(i.slug??i.system?.slug));if(extra)throw Error('此医疗变体尚未接入隔离草稿：'+(extra.slug??extra.system.slug));
 // PF2e treat() uses the selected skill rank. The stricter optional table
 // ruling changes only high DC eligibility, never the actual Occultism roll.
 const maximum=rankPolicy==='selected-skill'?rank:Math.min(rank,Math.max(1,actor.skills?.medicine?.rank??0));
 return [15,20,30,40].slice(0,maximum).map((dc,index)=>({tier:index+1,dc,bonus:[0,10,30,50][index]}));
}
export function treatmentOutcome({degree,tier}){
 if(!Number.isInteger(degree)||degree<0||degree>3||!Number.isInteger(tier)||tier<1||tier>4)throw Error('Invalid native treatment degree/tier');
 const bonus=[0,10,30,50][tier-1],formula=degree===0?'{1d8}':degree===1?null:`{(${degree===3?'4d8':'2d8'}${bonus?'+'+bonus:''})[healing]}`;
 return {degree,kind:degree===0?'damage':degree===1?'none':'healing',formula,removeWounded:degree>=2};
}
export function treatmentImmune(actor,now){
 return values(actor?.items).some(item=>{
  if(sourceId(item)!==TREAT_WOUNDS_IMMUNITY&&!(item.type==='effect'&&kissState(item).kind==='immunity'))return false;
  if(typeof item.remainingDuration?.expired==='boolean')return !item.remainingDuration.expired;
  if(Number.isFinite(kissState(item).expiresAt))return kissState(item).expiresAt>now;
  // Unknown duration is not permission to repeat a patient's treatment.
  return item.isExpired!==true;
 });
}
export function treatmentImmunityData(source,claim,now){
 if(source?.type!=='effect'||!Number.isFinite(claim.startedAt)||!Number.isFinite(now)||now<claim.startedAt||now>=claim.startedAt+3600)throw Error('Unproven or expired treatment activity time');
 const data=structuredClone(source);delete data._id;
 data._stats={...data._stats,compendiumSource:TREAT_WOUNDS_IMMUNITY};
 data.system.duration={value:(claim.startedAt+3600-now)/60,unit:'minutes',expiry:'turn-start',sustained:false};
 // EffectPF2e._preCreate writes current time again; remaining duration above
 // preserves start+1h even if the external exploration clock already advanced.
 data.system.start={value:now,initiative:null};
 data.system.context={origin:{actor:claim.actorUuid,token:claim.tokenUuid,item:claim.itemUuid,rollOptions:[]},target:{actor:claim.targetActorUuid,token:claim.targetUuid},roll:null};
 data.flags={...data.flags,[MODULE_ID]:{...data.flags?.[MODULE_ID],salubriousKiss:{kind:'immunity',nonce:claim.nonce,startedAt:claim.startedAt,expiresAt:claim.startedAt+3600}}};
 return data;
}
