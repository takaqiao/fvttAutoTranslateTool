import {ELECTRICITY_MODULE_ID as ID,electricityBasicAction,electricityBasicSettlementEnabled as enabled,electricityBasicCardType,electricityShieldActivity,electricityShieldAttack} from './eldamon-electricity.mjs';
import {isActualUseMessage} from './usage-events.mjs';
import {ensureNativeUseControls} from './metapower/entrances.mjs';
import {showNativeChoice} from './native-context.mjs';

const values=c=>Array.from(c?.values?.()??c??[]);
const random=()=>globalThis.foundry.utils.randomID(24);

/** Explicit settlement on the native card. The card click declares the trigger
 * happen on the invoking client; the existing GM electricity ledger owns the
 * document mutation and turn expiry. No movement/visibility listener is added. */
export function createEldamonBasicSettlement({game,apply,basicUse,fromUuid=globalThis.fromUuid,selectChoice=showNativeChoice,notify=message=>globalThis.ui.notifications.info(message),onError=console.error,random:nonce=random}={}){
 const pending=new Set(),unresolved=new Map(),attackKeys=new WeakMap();let attacks;
 function forgetAttack(message){const key=attackKeys.get(message),bucket=attacks?.get(key);if(bucket){bucket.delete(message);if(!bucket.size)attacks.delete(key);}attackKeys.delete(message);}
 function rememberAttack(message){if(!attacks)return;forgetAttack(message);const key=message.flags?.pf2e?.context?.target?.actor;if(!key||message.flags.pf2e.context.type!=='attack-roll')return;const bucket=attacks.get(key)??new Set();bucket.add(message);attacks.set(key,bucket);attackKeys.set(message,key);}
 async function recordedAttack(context,target){
  if(!attacks){attacks=new Map();for(const message of values(game.messages))rememberAttack(message);}
  const shield=electricityShieldActivity(context.actor,game);if(!shield)return null;
  const candidates=[];for(const message of attacks.get(context.actor.uuid)??[]){if(game.messages.get(message.id)!==message||message.isContentVisible===false||message.timestamp<shield.createdAt||`Scene.${message.speaker?.scene}.Token.${message.speaker?.token}`!==target.uuid)continue;
   if(electricityShieldAttack(message,{actor:context.actor,sourceTokenUuid:shield.sourceTokenUuid,item:message.item??await fromUuid(message.flags?.pf2e?.origin?.uuid)}))candidates.push(message);}
  if(candidates.length<2)return candidates[0]??null;
  const choice=await selectChoice({title:'选择本次实际触发护盾的近战攻击',choices:candidates.map(message=>({value:message.id,label:`近战未命中 · ${new Date(message.timestamp).toLocaleString()}`}))});
  return choice===null||choice===undefined?false:candidates.find(message=>message.id===choice)??false;
 }
 function cardContext(message){
  if(!enabled(game)||!message?.id||game.messages.get(message.id)!==message||!electricityBasicCardType(message)||!isActualUseMessage(message))return null;
  const pf=message.flags?.pf2e,type=pf?.context?.type;
  const speaker=message.speaker,actor=game.scenes?.get(speaker?.scene)?.tokens?.get(speaker?.token)?.actor??game.actors.get(speaker?.actor);
  if(!actor?.testUserPermission?.(game.user,'OWNER'))return null;
  const prefix=`${actor.uuid}.Item.`,origin=pf?.origin?.uuid;
  const id=typeof origin==='string'&&origin.startsWith(prefix)?origin.slice(prefix.length):type==='self-effect'?pf.context.item:null;
  const item=id?actor.items.get(id):null,kind=electricityBasicAction(item);
  return kind?{actor,item,kind}:null;
 }
 const resolveAction=item=>enabled(game)&&electricityBasicAction(item)?`electricity:${electricityBasicAction(item)}`:null;
 async function executeUsage({actor,item,message,user}){
  if(!resolveAction(item)||item.actor!==actor||game.messages.get(message?.id)!==message||!isActualUseMessage(message))throw Error('需要实际使用元素动作后生成的原生卡。');
  return basicUse({actorUuid:actor.uuid,itemUuid:item.uuid,messageUuid:message.uuid},user);
 }
 async function declareShieldTrigger(actor,{message=null}={}){
  if(!enabled(game)||!actor?.testUserPermission(game.user,'OWNER'))throw Error('需要当前角色的拥有者权限。');
  const shield=electricityShieldActivity(actor,game),card=shield?.messageUuid?await fromUuid(shield.messageUuid):null;
  if(!shield||!cardContext(card))throw Error('没有当前可触发的原元素护盾。');
  if(message&&!electricityShieldAttack(message,{actor,sourceTokenUuid:shield.sourceTokenUuid,item:message.item??await fromUuid(message.flags?.pf2e?.origin?.uuid)}))throw Error('需要以本角色为目标的实际近战未命中卡。');
  return settleFromCard(card,{triggerMessage:message});
 }
 async function settleFromCard(message,{triggerMessage=null}={}){
  const context=cardContext(message);if(!context||pending.has(message.id))return null;
  pending.add(message.id);
  try{
  for(const id of unresolved.keys())if(!game.messages.get(id))unresolved.delete(id);
  let attempt=unresolved.get(message.id);const retry=!!attempt;
  if(!attempt){
   const targets=triggerMessage?[await fromUuid(`Scene.${triggerMessage.speaker.scene}.Token.${triggerMessage.speaker.token}`)]:context.kind==='manipulation'?
    await Promise.all((message.flags?.[ID]?.usageInput?.targetUuids??[]).map(fromUuid)):values(game.user.targets).map(t=>t.document??t);
   if(targets.length!==1||!targets[0]?.actor||!targets[0]?.uuid)throw Error('请先用 T 锁定一个实际触发护盾的敌人，再结算。');
   const target=targets[0];if(context.kind==='shield'&&!triggerMessage){triggerMessage=await recordedAttack(context,target);if(triggerMessage===false)return null;}
   attempt={payload:{actorUuid:context.actor.uuid,itemUuid:context.item.uuid,messageUuid:message.uuid,targetUuid:target.uuid,nonce:nonce(),confirmed:true,...(triggerMessage?{triggerMessageUuid:triggerMessage.uuid}:{})}};
  }
   unresolved.set(message.id,attempt);
   const result=await apply(attempt.payload);
   unresolved.delete(message.id);
   notify(retry||result?.replayed?'已确认上次结算结果，未重复施加带电。':result?.manualExpiry?'已施加带电；当前没有明确的战斗回合，请在护盾规定的到期时点手动移除。':'已施加带电，并记录持续时间。');
   return result;
  }catch(error){if(error?.electricityNotApplied===true)unresolved.delete(message.id);throw error}
  finally{pending.delete(message.id)}
 }
 function render(message,html){
  const context=cardContext(message);if(!context)return;
  const root=html?.[0]??html;if(!root?.querySelector||root.querySelector('[data-eldamon-basic-settlement]'))return;
  const button=document.createElement('button');button.type='button';button.dataset.eldamonBasicSettlement=context.kind;
  if(context.kind!=='shield'||!electricityShieldActivity(context.actor,game))return;
  button.textContent='护盾令敌方近战攻击未命中：施加带电';
  button.addEventListener('click',async event=>{event.preventDefault();button.disabled=true;try{await settleFromCard(message)}catch(error){onError(error)}finally{button.disabled=false}});
  (root.querySelector('.message-content')??root).append(button);
 }
 function renderAttack(message,html){
  const target=message.flags?.pf2e?.context?.target,actor=game.scenes?.get(target?.token?.split('.')[1])?.tokens?.get(target?.token?.split('.')[3])?.actor??game.actors.get(target?.actor?.split('.').at(-1));
  const shield=actor&&electricityShieldActivity(actor,game),root=html?.[0]??html;
  if(message.isContentVisible===false||!actor?.testUserPermission(game.user,'OWNER')||!shield||message.timestamp<shield.createdAt||!electricityShieldAttack(message,{actor,sourceTokenUuid:shield.sourceTokenUuid})||!root?.querySelector||root.querySelector('[data-eldamon-shield-trigger]'))return;
  const button=root.ownerDocument.createElement('button');button.type='button';button.dataset.eldamonShieldTrigger='';button.textContent='护盾令此次攻击未命中：施加带电';
  button.addEventListener('click',event=>{event.preventDefault();button.disabled=true;declareShieldTrigger(actor,{message}).catch(onError).finally(()=>{button.disabled=false})});(root.querySelector('.message-content')??root).append(button);
 }
 function renderActor(app,html){
  const root=html?.[0]??html,actor=app.actor;if(!root?.querySelector||!actor?.testUserPermission(game.user,'OWNER'))return;
  ensureNativeUseControls(root,actor,item=>!!resolveAction(item));
  if(!electricityShieldActivity(actor,game)||root.querySelector('[data-eldamon-shield-trigger]'))return;
  const button=root.ownerDocument.createElement('button');button.type='button';button.dataset.eldamonShieldTrigger='';button.textContent='结算实际护盾未命中触发';
  button.title='仅当敌方近战攻击因元素护盾的 AC 加值而未命中时使用；请用 T 锁定该敌人。';
  button.addEventListener('click',event=>{event.preventDefault();button.disabled=true;declareShieldTrigger(actor).catch(onError).finally(()=>{button.disabled=false})});(root.querySelector('.tab.actions[data-tab="actions"],section[data-tab="actions"]')??root).append(button);
 }
 return {cardContext,settleFromCard,declareShieldTrigger,resolveAction,requiresActualUse:item=>!!resolveAction(item),executeUsage,register:({Hooks})=>{if(!enabled(game))return;Hooks.on('renderChatMessageHTML',(message,html)=>{render(message,html);renderAttack(message,html)});Hooks.on('createChatMessage',rememberAttack);Hooks.on('updateChatMessage',rememberAttack);Hooks.on('deleteChatMessage',message=>{unresolved.delete(message.id);forgetAttack(message)});for(const name of ['renderActorSheetPF2e','renderCharacterSheetPF2e','renderActorSheetV2'])Hooks.on(name,renderActor)}};
}
