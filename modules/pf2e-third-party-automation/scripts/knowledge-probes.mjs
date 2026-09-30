import {MODULE_ID} from './rules.mjs';

// The option is only a locator. Authority comes from this client's live Map,
// actor/statistic binding and the single callback, never from a document flag.
const liveProbes=new Map();
let serial=0;
const escape=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
export async function withKnowledgeProbe({actor,statistic,globals=globalThis},native){
 const nonce=globalThis.crypto?.randomUUID?.()??`${Date.now()}-${++serial}-${Math.random()}`;
 const marker=`${MODULE_ID}:knowledge-probe:${nonce}`,receipt={captured:false};
 liveProbes.set(marker,{actorUuid:actor.uuid,statistic,globals,receipt});
 try{return {result:await native(marker),receipt};}finally{liveProbes.delete(marker);}
}

/** Called at the OUTERMOST native Check.roll middleware boundary. */
export async function interceptKnowledgeProbe(native,check,context,event,callback){
 const options=new Set(context?.options??[]),marker=[...options].find(option=>liveProbes.has(option)),entry=liveProbes.get(marker);
 if(!entry||entry.receipt.captured||context.type!=='skill-check'||context.actor?.uuid!==entry.actorUuid||check.slug!==entry.statistic||!options.has('action:recall-knowledge')||!options.has(`action:recall-knowledge:${entry.statistic}`))return native(check,context,event,callback);
 options.delete(marker);
 // This is the same predicate/stacking calculation performed by PF2e Check.roll.
 // StatisticCheck has already resolved native context and called beforeRoll.
 check.calculateTotal(options);
 if(!Number.isFinite(check.totalModifier))throw Error('原生回忆知识加值不可验证。');
 const serializeParticipant=participant=>participant?.actor?{actor:participant.actor.uuid,token:participant.token?.uuid}:null;
 const data={content:'',speaker:{actor:context.actor.id,token:context.token?.id,scene:context.token?.parent?.id},flags:{pf2e:{context:{type:context.type,actor:context.actor.id,token:context.token?.id??null,origin:serializeParticipant(context.origin),target:serializeParticipant(context.target),domains:context.domains??[],options:[...options].sort(),traits:context.traits??[],title:context.title,dc:null,createMessage:false,messageMode:'blind',rollTwice:context.rollTwice??false,substitutions:context.substitutions??[]},modifierName:check.slug,modifiers:check.modifiers.map(modifier=>modifier.toObject?.()??{slug:modifier.slug,label:modifier.label,modifier:modifier.modifier,enabled:modifier.enabled})}},flavor:`<div class="tags modifiers">${check.modifiers.filter(modifier=>modifier.enabled).map(modifier=>`<span class="tag" data-slug="${escape(modifier.slug)}">${escape(modifier.label)} ${modifier.modifier<0?'':'+'}${modifier.modifier}</span>`).join('')}</div>`};
 const Messages=entry.globals.CONFIG?.ChatMessage?.documentClass??entry.globals.ChatMessage;
 const message=new Messages(data),roll={total:10+check.totalModifier,options:{totalModifier:check.totalModifier}};
 Object.assign(entry.receipt,{captured:true,native,check,context:{...context,options},actor:context.actor,domains:context.domains??[],rollOptions:options,message,roll});
 await callback?.(roll,undefined,message);
 // Native StatisticCheck skips its afterRoll loop on null. No extra die, document
 // creation, consumable use or downstream provider middleware is invoked.
 return null;
}

/** Run real rule hooks once, after saving a claim on the one real Workbench card. */
export async function consumeKnowledgePrimary({message,candidate,receipt,roll}){
 if(!receipt?.captured||!candidate)return;
 if(message.flags?.[MODULE_ID]?.workbenchRecall?.probeUse)return;
 const claim={status:'claimed',statistic:candidate.statistic,targetUuid:candidate.targetUuid};
 await message.update({[`flags.${MODULE_ID}.workbenchRecall.probeUse`]:claim});
 // This is the actual primary CheckRoll, with native fortune/substitution dice
 // and methods intact. Its Check continuation mutates the same native context.
 for(const rule of receipt.actor.rules?.filter(rule=>!rule.ignored)??[])await rule.afterRoll?.({roll,check:receipt.check,context:receipt.primaryContext,domains:receipt.domains,rollOptions:receipt.primaryContext.options});
 await message.update({[`flags.${MODULE_ID}.workbenchRecall.probeUse`]:{...claim,status:'done'}});
}
