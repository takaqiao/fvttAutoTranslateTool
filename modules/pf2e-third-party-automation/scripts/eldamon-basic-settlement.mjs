import {electricityBasicAction,electricityBasicSettlementEnabled as enabled,electricityBasicCardType} from './eldamon-electricity.mjs';

const values=c=>Array.from(c?.values?.()??c??[]);
const escape=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const random=()=>globalThis.foundry.utils.randomID(24);
async function nativeConfirm({kind,targetName,retry}){
 const text=retry?'上次未收到完成回执；本次重试相同目标与结算，不会作为新的触发重复施加。':kind==='shield'?'确认本次为电元素护盾，此敌人的近战攻击仅因护盾的 AC 加值而未命中。':'确认本次已使用电元素操控，并且此目标满足邻接条件。';
 return globalThis.foundry.applications.api.DialogV2.confirm({window:{title:kind==='shield'?'元素护盾：施加带电':'元素操控：施加带电'},content:`<p>目标：<strong>${escape(targetName)}</strong></p><p>${text}</p>`});
}

/** Explicit settlement on the native card. Targeting and trigger confirmation
 * happen on the invoking client; the existing GM electricity ledger owns the
 * document mutation and turn expiry. No movement/visibility listener is added. */
export function createEldamonBasicSettlement({game,apply,confirm=nativeConfirm,notify=message=>globalThis.ui.notifications.info(message),onError=console.error,random:nonce=random}={}){
 const pending=new Set(),unresolved=new Map();
 function cardContext(message){
  if(!enabled(game)||!message?.id||game.messages.get(message.id)!==message||!electricityBasicCardType(message))return null;
  const pf=message.flags?.pf2e,type=pf?.context?.type;
  const speaker=message.speaker,actor=game.scenes?.get(speaker?.scene)?.tokens?.get(speaker?.token)?.actor??game.actors.get(speaker?.actor);
  if(!actor?.testUserPermission?.(game.user,'OWNER'))return null;
  const prefix=`${actor.uuid}.Item.`,origin=pf?.origin?.uuid;
  const id=typeof origin==='string'&&origin.startsWith(prefix)?origin.slice(prefix.length):type==='self-effect'?pf.context.item:null;
  const item=id?actor.items.get(id):null,kind=electricityBasicAction(item);
  return kind?{actor,item,kind}:null;
 }
 async function settleFromCard(message){
  const context=cardContext(message);if(!context||pending.has(message.id))return null;
  for(const id of unresolved.keys())if(!game.messages.get(id))unresolved.delete(id);
  let attempt=unresolved.get(message.id);const retry=!!attempt;
  if(!attempt){
   const targets=values(game.user.targets).map(t=>t.document??t);
   if(targets.length!==1||!targets[0].actor||!targets[0].uuid)throw Error('请先用 T 锁定一个带电目标，再点击结算。');
   const target=targets[0];attempt={targetName:target.name??target.actor.name,payload:{actorUuid:context.actor.uuid,itemUuid:context.item.uuid,messageUuid:message.uuid,targetUuid:target.uuid,nonce:nonce(),confirmed:true}};
  }
  pending.add(message.id);
  try{
   if(!await confirm({kind:context.kind,targetName:attempt.targetName,retry}))return null;
   unresolved.set(message.id,attempt);
   const result=await apply(attempt.payload);
   unresolved.delete(message.id);
   notify(retry?'已确认上次结算结果，未重复施加带电。':result?.manualExpiry?'已施加带电；当前没有明确的战斗回合，请在护盾规定的到期时点手动移除。':'已施加带电，并记录持续时间。');
   return result;
  }catch(error){if(error?.electricityNotApplied===true)unresolved.delete(message.id);throw error}
  finally{pending.delete(message.id)}
 }
 function render(message,html){
  const context=cardContext(message);if(!context)return;
  const root=html?.[0]??html;if(!root?.querySelector||root.querySelector('[data-eldamon-basic-settlement]'))return;
  const button=document.createElement('button');button.type='button';button.dataset.eldamonBasicSettlement=context.kind;
  button.textContent=context.kind==='shield'?'确认护盾触发：施加带电':'对 T 目标施加两轮带电';
  button.addEventListener('click',async event=>{event.preventDefault();button.disabled=true;try{await settleFromCard(message)}catch(error){onError(error)}finally{button.disabled=false}});
  (root.querySelector('.message-content')??root).append(button);
 }
 return {cardContext,settleFromCard,register:({Hooks})=>{if(enabled(game))Hooks.on('renderChatMessageHTML',render)}};
}
