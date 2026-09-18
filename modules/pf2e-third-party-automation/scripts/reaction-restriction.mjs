/** Synchronous eligibility only. Never write or mask the underlying resources:
 * completed reactions and exact refunds still use their original receipts. */
export function reactionRestrictionStatus(actor,reactionRestriction){
 if(reactionRestriction==null)return 'clear';
 try{const status=reactionRestriction(actor)?.status;return status==='clear'||status==='restricted'?status:'manual'}
 catch{return 'manual'}
}
export const reactionPermitted=(actor,reactionRestriction)=>reactionRestrictionStatus(actor,reactionRestriction)==='clear';
export function requireReactionPermitted(actor,reactionRestriction){
 const status=reactionRestrictionStatus(actor,reactionRestriction);
 if(status==='restricted')throw Error('当前来源禁止使用反应；本模块未开始新的反应。');
 if(status!=='clear')throw Error('反应限制状态待 GM 核对；本模块未开始新的反应，请手工结算。');
}
