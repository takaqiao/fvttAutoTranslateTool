import {MODULE_ID} from './rules.mjs';

const escape=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
/** Called only by the live source adapter, never registered as a public socket operation. */
export function createDisruptPreySourceCost({game,createMessage=data=>globalThis.ChatMessage.create(data)}={}){
 const pending=new Map();
 function stopped(context,result){
  const {actor,item,user,token,castNonce}=context??{},key=`${actor?.uuid}:${castNonce}`;
  if(pending.has(key))return pending.get(key);
  const task=Promise.resolve().then(async()=>{
   const receipt=actor?.flags?.[MODULE_ID]?.nativeCasts?.find(r=>r.id===castNonce);
   if(!user||game.user!==user||game.users.get(user.id)!==user||!actor?.testUserPermission?.(user,'OWNER')||item?.type!=='spell'||item.actor!==actor||token?.actor!==actor||token.uuid!==context.tokenUuid||!receipt||receipt.actorUuid!==actor.uuid||receipt.itemUuid!==item.uuid||receipt.userId!==user.id||receipt.nativeCastScope?.castNonce!==castNonce||receipt.nativeCastScope.tokenUuid!==token.uuid||!['paid','used','disrupted','uncertain'].includes(receipt.state))throw Error('施法中止回执与本次原生支付不符。');
   const action=receipt.sourceAction,cost=action?.type==='reaction'?'反应已使用':action?.type==='action'?`${action.value} 个动作已使用`:action?.type==='free'?'本次自由动作已使用':'本次施法已中止';
   const detail=result?.status==='uncertain'?'反应结算结果待确认。':'本次施法未产生效果。';
   // No original item/origin/actualCast: this is a terminal receipt, not another
   // cast. Patreon must not apply the interrupted spell's effects from it.
   return createMessage({user:user.id,speaker:{actor:actor.id,scene:token.parent.id,token:token.id,alias:actor.name},content:`<p><strong>${escape(item.name)}：施法中止</strong></p><p>${cost}，已支付的法术资源不返还。${detail}</p>`,flags:{pf2e:{context:{options:['skip-handling-message']}},[MODULE_ID]:{usageGenerated:true,disruptPreySourceStopped:{id:castNonce,actorUuid:actor.uuid,itemUuid:item.uuid,userId:user.id,sourceAction:structuredClone(action??null),status:result?.status??'stopped'}}}});
  });
  // Keep failed publication too: a lost create acknowledgement is not proof
  // that no card exists. Replaying a still-live continuation cannot create two.
  pending.set(key,task);return task;
 }
 return {stopped};
}
