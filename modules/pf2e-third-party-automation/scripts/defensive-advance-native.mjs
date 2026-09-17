import {MODULE_ID} from './rules.mjs';
import {advancePosition,sameAdvancePosition} from './defensive-advance-rules.mjs';

export const defensiveAdvanceStrikeMarker=nonce=>`${MODULE_ID}:defensive-advance:${nonce}`;

/** This runs on the original operator, without creating another Stride action. */
export async function runDefensiveAdvanceMovement({game,token,receipt,validate,bindPlan,confirm}){
 let plan;
 try{
  validate();
  plan=await token.object.planMovement({allowedActions:['walk'],maxCost:receipt.speed,preventDrop:true});
  if(!plan)return null;
  validate();if(!sameAdvancePosition(advancePosition(token),receipt.origin))throw Error('选路时原Token已经移动；未启动此计划。');
  await bindPlan(plan.id);validate();
  const finished=token.movement.finished;
  if(!finished?.then||token.movement.id!==plan.id)throw Error('原生移动计划已被替换。');
  if(!await token.startMovement(plan.id)||await finished!==true)return null;
  validate();return await confirm();
 }finally{
  // Do not cancel someone else's replacement or a completed movement.
  if(plan?.id&&token.movement?.id===plan.id&&token.movement.state==='planned'&&token.movement.user?.id===game.user.id&&token.parent?.tokens?.get(token.id)===token)token.stopMovement();
 }
}

export function defensiveAdvanceStrikeProof(message,{game,actor,token,target,receipt,option}){
 const pf=message?.flags?.pf2e,c=pf?.context,mark=message?.flags?.[MODULE_ID]?.defensiveAdvanceStrike;
 if(!message?.id||game.messages.get(message.id)!==message||!message.isCheckRoll||message.rolls?.length!==1||!Number.isFinite(message.rolls[0].total)||(message.author?.id??message.user?.id)!==receipt.userId||message.speaker?.actor!==actor.id||message.speaker.scene!==token.parent.id||message.speaker.token!==token.id||pf.origin?.actor!==actor.uuid||pf.origin.uuid!==option.itemUuid||c?.type!=='attack-roll'||c.action!=='strike'||c.mapIncreases!==receipt.map||c.target?.actor!==target.actor.uuid||c.target.token!==target.uuid||c.origin?.token!==token.uuid||!c.options?.includes(defensiveAdvanceStrikeMarker(receipt.nonce))||!c.options.includes('action:free')||c.isReroll||!['criticalFailure','failure','success','criticalSuccess'].includes(c.outcome)||mark?.nonce!==receipt.nonce||mark.messageId!==receipt.messageId)throw Error('列盾突进原生Strike回执不匹配；不会重掷。');
 return message.id;
}

/** Keep native Strike publication and all native callbacks. This scoped hook
 * labels only the included Strike, never a new activity or extra action fee. */
export async function rollDefensiveAdvanceStrike({game,Hooks,actor,token,target,receipt,option,validate}){
 const current=validate();if(current?.key!==option.key||actor.getActiveTokens?.(false,true)?.[0]?.uuid!==token.uuid)throw Error('原生Strike将使用不同Token或武器；未掷骰。');
 let card=null,hookError=null;
 const marker=defensiveAdvanceStrikeMarker(receipt.nonce);
 const id=Hooks.on('preCreateChatMessage',message=>{
  if(!message.flags?.pf2e?.context?.options?.includes(marker))return;
  try{
   validate();const c=message.flags.pf2e.context;
   if(c.type!=='attack-roll'||c.action!=='strike'||c.mapIncreases!==receipt.map||message.flags.pf2e.origin?.uuid!==option.itemUuid||c.target?.token!==target.uuid)throw Error('内含Strike的原生草稿来源不符。');
   const flavor=String(message.flavor??'').replace(/(<span\b[^>]*\bclass=["'][^"']*\baction-glyph\b[^"']*["'][^>]*>)[\s\S]*?(<\/span>)/,(_m,a,b)=>a+'F'+b);
   message.updateSource({flavor,[`flags.${MODULE_ID}.defensiveAdvanceStrike`]:{nonce:receipt.nonce,messageId:receipt.messageId,activityCost:2,included:true}});
  }catch(error){hookError=error;return false;}
 });
 try{
  const roll=await current.strike.variants[receipt.map].roll({target:target.object,altUsage:current.usage??undefined,options:new Set(['action:defensive-advance','action:free',marker]),createMessage:true,callback:async(_roll,_outcome,message)=>{if(card&&card!==message)throw Error('同一列盾突进出现多个Strike结果。');card=message;},event:{ctrlKey:false,metaKey:false,shiftKey:false}});
  if(hookError)throw hookError;
  if(!roll&&!card)return null;
  return defensiveAdvanceStrikeProof(card,{game,actor,token,target,receipt,option:current});
 }finally{Hooks.off('preCreateChatMessage',id);}
}
