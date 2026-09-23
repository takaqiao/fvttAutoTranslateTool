import {MODULE_ID} from './rules.mjs';
import {SerialActions} from './runtime.mjs';
import {getSourceId,isActiveGM} from './native-context.mjs';
import {isActualUseMessage} from './usage-events.mjs';
import {DEFENSIVE_ADVANCE_SOURCE} from './defensive-advance-compat.mjs';
import {defensiveAdvanceContext,defensiveAdvanceMeleeChoices} from './defensive-advance-rules.mjs';
import {rollDefensiveAdvanceStrike,defensiveAdvanceStrikeProof} from './defensive-advance-native.mjs';

const values=c=>Array.from(c?.values?.()??c??[]),own=m=>m?.flags?.[MODULE_ID]?.defensiveAdvance,input=m=>m?.flags?.[MODULE_ID]?.defensiveAdvanceInput;
const author=m=>m?.author?.id??m?.user?.id??m?.user;
const nonce=()=>globalThis.foundry?.utils?.randomID?.(16)??globalThis.crypto.randomUUID();
const shieldReady=a=>{const s=a.attributes?.shield;return !!(s?.itemId&&s.raised&&!s.broken&&!s.destroyed)};

/** The original two-action flourish owns the native shield and included Strike.
 * The operator moves on the map; a reload/GM migration never replays this Use. */
export function createDefensiveAdvance({game,fromUuid=globalThis.fromUuid,choose,startupCompatibility}={}){
 const startup=Object.freeze({...startupCompatibility}),queue=new SerialActions(),ownerEntered=new Set();let socket,Hooks;
 const resolveAction=item=>game.world?.id==='ujx5r8oipw7ercdr'&&item?.type==='feat'&&getSourceId(item)===DEFENSIVE_ADVANCE_SOURCE?'defensive-advance':null;
 function compatibility(){
  if(startup.status!=='ready')throw Error(startup.reason??'列盾突进启动配置尚未就绪；请修复配置后整页刷新，再使用手工后续流程。');
  const p=game.modules?.get('patreon-v3');if(!p?.active||p.version!=='3.2.28')throw Error('列盾突进的 Patreon 执行器已变化。');
  if(game.modules?.get('pf2e-auto-action-tracker')?.active)throw Error('当前列盾突进尚未核验 Auto Action Tracker 的内含动作计费；请手工继续。');
 }
 function gm(){if(!isActiveGM(game)||!game.user.isGM)throw Error('列盾突进需要当前主GM。');compatibility();}
 const captureUsage=item=>resolveAction(item)?{defensiveAdvanceInput:{nonce:nonce(),startup:startup.status}}:null;
 function captureCard(message){
  if(!isActualUseMessage(message)||message.isRoll||message.rolls?.length)return;
  const actor=message.actor??game.actors.get(message.speaker?.actor),token=game.scenes.get(message.speaker?.scene)?.tokens.get(message.speaker?.token);
  if(!actor||token?.actor!==actor)return;
  let turn;try{turn=defensiveAdvanceContext({game,actor,token}).turn}catch{return}
  if(message.flags?.pf2e?.origin?.rollOptions?.includes('origin:item:trait:flourish'))message.updateSource({[`flags.${MODULE_ID}.defensiveAdvanceObservedTurn`]:turn});
  const item=message.item??actor.items.get(message.flags?.pf2e?.origin?.uuid?.split('.').at(-1));if(!resolveAction(item))return;
  message.updateSource({[`flags.${MODULE_ID}.defensiveAdvanceInput`]:{...input(message),nonce:input(message)?.nonce??nonce(),startup:startup.status,turn,tokenUuid:token.uuid}});
 }
 async function context(message,user){
  const item=await fromUuid(message?.flags?.pf2e?.origin?.uuid),actor=item?.actor;
  const token=await fromUuid(`Scene.${message?.speaker?.scene}.Token.${message?.speaker?.token}`);
  const ctx={actor,item,message,user,token};validate(ctx);return ctx;
 }
 function validate(ctx){
  compatibility();const {actor,item,message,user,token}=ctx,r=own(message),i=input(message);
  if(!user?.active||game.users.get(user.id)!==user||!actor?.testUserPermission?.(user,'OWNER')||actor.items.get(item?.id)!==item||!resolveAction(item))throw Error('列盾突进的原操作者或专长来源已改变。');
  if(!message?.id||game.messages.get(message.id)!==message||author(message)!==user.id||message.speaker?.actor!==actor.id||message.flags?.pf2e?.origin?.uuid!==item.uuid||!isActualUseMessage(message)||message.isRoll||message.rolls?.length||typeof i?.nonce!=='string'||!i.nonce||i.startup!=='ready'||i.tokenUuid!==token?.uuid)throw Error('需要列盾突进原始真实Use及启动就绪回执。');
  const current=defensiveAdvanceContext({game,actor,token});if(current.turn!==i.turn||r&&r.turn!==current.turn)throw Error('列盾突进的实际回合已改变，后续活动过期。');
  if(r&&(r.nonce!==i.nonce||r.userId!==user.id||r.actorUuid!==actor.uuid||r.itemUuid!==item.uuid||r.tokenUuid!==token.uuid||r.gmId!==game.users.activeGM?.id))throw Error('列盾突进回执或主GM已改变；不会重放。');
  return current;
 }
 async function save(message,changes){gm();if(game.messages.get(message.id)!==message)throw Error('原始列盾突进卡已不存在。');await message.update({[`flags.${MODULE_ID}.defensiveAdvance`]:{...own(message),...changes}});gm();}
 async function ownerContext(payload,sender){
  if(sender?.id!==game.users.activeGM?.id)throw Error('只有当前主GM可以请求原操作者的列盾突进。');
  const message=game.messages.get(payload.messageId),r=own(message);if(!r||r.nonce!==payload.nonce||r.status!=='striking'||r.userId!==game.user.id)throw Error('列盾突进原操作者认领不匹配。');
  const ctx=await context(message,game.user);if(ownerEntered.has(r.nonce))throw Error('本阶段已经进入；不会重放。');ownerEntered.add(r.nonce);return ctx;
 }
 async function originalTarget(ctx){
  const source=ctx.message.flags?.[MODULE_ID]?.usageInput?.targetUuids;if(!Array.isArray(source)||source.some(uuid=>typeof uuid!=='string'))return null;
  const uuids=[...new Set(source)];if(uuids.length!==1||!/^Scene\.[A-Za-z0-9_-]+\.Token\.[A-Za-z0-9_-]+$/.test(uuids[0]))return null;
  const target=await fromUuid(uuids[0]);return target?.uuid===uuids[0]&&defensiveAdvanceMeleeChoices({...ctx,game,target}).length?target:null;
 }
 function targetContext(ctx){
  validate(ctx);const r=own(ctx.message),target=ctx.token.parent.tokens.get(r.targetId),options=defensiveAdvanceMeleeChoices({...ctx,game,target});
  if(target?.uuid!==r.targetUuid||target?.actor?.uuid!==r.targetActorUuid||!options.length)throw Error('所选近战Strike或原卡目标已改变。');
  return {target,options};
 }
 function selected(ctx){const {target,options}=targetContext(ctx),r=own(ctx.message),option=options.find(o=>o.key===r.weaponKey);if(!option||!Number.isInteger(r.map)||r.map<0||r.map>2||typeof option.strike.variants?.[r.map]?.roll!=='function')throw Error('所选近战Strike、原卡目标或MAP已改变。');return {target,option};}
 async function ownerStrike(payload,sender){const ctx=await ownerContext(payload,sender),{target,option}=selected(ctx);return rollDefensiveAdvanceStrike({...ctx,game,Hooks,target,option,receipt:own(ctx.message),validate:()=>selected(ctx).option});}
 async function callOwner(ctx){const payload={messageId:ctx.message.id,nonce:own(ctx.message).nonce};gm();if(ctx.user.id===game.user.id)return ownerStrike(payload,game.user);if(!socket)throw Error('缺少原操作者客户端通讯。');const r=await socket.executeAsUser('defensive-advance:strike',ctx.user.id,payload);if(!r?.ok)throw Error(r?.error??'原操作者结果尚未确认。');return r.value;}
 async function pick(ctx,title,choices){const value=await choose({...ctx,title,choices});validate(ctx);if(value==null)return null;if(!choices.some(c=>c.value===value))throw Error('无效的列盾突进选择。');return value;}
 async function executeUsage({actor,item,message,user,action}){
  gm();if(action!=='defensive-advance'||resolveAction(item)!==action)throw Error('不是列盾突进的准确入口。');
  const ctx=await context(message,user);if(ctx.actor!==actor||ctx.item!==item)throw Error('原卡角色不匹配。');
  const entered=await queue.run(actor.uuid,async()=>{
   gm();validate(ctx);if(own(message))return false;
   if(item.system.actionType?.value!=='action'||item.system.actions?.value!==2||!item.system.traits?.value?.includes('flourish'))throw Error('原始双动作华丽专长已改变。');
   const current=validate(ctx),uses=actor.flags?.[MODULE_ID]?.defensiveAdvanceUses??[];
   if(uses.some(r=>r.nonce===input(message).nonce))throw Error('此Use nonce已属于另一张原卡。');
   if(current.turn&&(uses.some(r=>r.turn===current.turn)||values(game.messages).some(m=>m.id!==message.id&&m.speaker?.actor===actor.id&&m.flags?.[MODULE_ID]?.defensiveAdvanceObservedTurn===current.turn&&isActualUseMessage(m)&&m.flags?.pf2e?.origin?.rollOptions?.includes('origin:item:trait:flourish'))))throw Error('实际本回合已承诺华丽动作。');
   const target=await originalTarget(ctx);gm();validate(ctx);
   const r={nonce:input(message).nonce,messageId:message.id,actorUuid:actor.uuid,itemUuid:item.uuid,tokenUuid:ctx.token.uuid,userId:user.id,gmId:game.user.id,turn:current.turn,cost:2,flourish:true,status:'awaiting-movement',targetId:target?.id??null,targetUuid:target?.uuid??null,targetActorUuid:target?.actor?.uuid??null};
   await actor.update({[`flags.${MODULE_ID}.defensiveAdvanceUses`]:[...uses.slice(-63),{nonce:r.nonce,messageId:message.id,turn:r.turn,cost:2}]});await save(message,r);return true;
  });
  if(!entered)return own(message)?.result??'本次列盾突进已开始或完成；不会重放。';
  try{
   if(!own(message).targetId){await save(message,{status:'done',result:'本次原始Use卡未绑定唯一可用近战目标，已结束后续。请在原始Use前用T选中一个敌方目标；已消耗动作不回退。'});return own(message).result;}
   const continuation=await pick(ctx,'列盾突进：完成地图移动后继续',[{value:'continue',label:'地图移动后继续内含打击'},{value:'decline',label:'结束此活动'}]);
   if(continuation!=='continue'){await save(message,{status:'cancelled',result:'已结束列盾突进后续；原始活动费用与已有原生效果不回退。'});return own(message).result;}
   // Patreon owns the shield effect. The operator's continuation replaces polling.
   if(!shieldReady(actor))throw Error('原生举盾尚未确认；不创建替代效果，请手工继续。');
   await save(message,{status:'ready-to-strike'});
   const {options}=targetContext(ctx);
   const weaponKey=await pick(ctx,'选择原生近战Strike',options.map(o=>({value:o.key,label:o.strike.label??o.strike.item.name})));
   if(!weaponKey){await save(message,{status:'cancelled',result:'已取消内含Strike，已完成动作不回退。'});return own(message).result;}
   const map=await pick(ctx,'列盾突进：选择当前MAP',[{value:'0',label:'本回合尚未攻击（MAP 0）'},{value:'1',label:'已攻击一次（MAP 1）'},{value:'2',label:'已攻击两次或更多（MAP 2）'}]);
   if(map===null){await save(message,{status:'cancelled',result:'已取消内含Strike。'});return own(message).result;}
   await save(message,{status:'striking',weaponKey,map:Number(map)});selected(ctx);
   const checkId=await callOwner(ctx);validate(ctx);
   if(checkId===null){await save(message,{status:'cancelled',result:'已取消原生Strike，已完成动作不回退。'});return own(message).result;}
   const selection=selected(ctx);defensiveAdvanceStrikeProof(game.messages.get(checkId),{...ctx,...selection,game,receipt:own(message)});
   await save(message,{status:'done',checkId,result:'列盾突进后续已完成：原生举盾与一次内含近战Strike；地图移动由操作者完成。'});return own(message).result;
  }catch(error){if(isActiveGM(game)&&own(message)?.gmId===game.user.id&&game.messages.get(message.id)===message)await save(message,{status:'uncertain',result:`${error.message} 已完成动作不回退，未知结果不会重放。`});throw error;}
 }
 function register({Hooks:hooks,socket:api}={}){
  Hooks=hooks;socket=api;
  const id=Hooks.on('preCreateChatMessage',captureCard);
  socket?.register('defensive-advance:strike',async function(payload){try{return {ok:true,value:await ownerStrike(payload,game.users.get(this.socketdata.userId))}}catch(error){return {ok:false,error:error.message}}});
  return()=>{Hooks.off('preCreateChatMessage',id);};
 }
 return {resolveAction,requiresActualUse:item=>!!resolveAction(item),captureUsage,executeUsage,register,get diagnostic(){return startup}};
}
