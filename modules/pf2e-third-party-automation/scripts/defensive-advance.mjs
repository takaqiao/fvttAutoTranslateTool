import {MODULE_ID} from './rules.mjs';
import {SerialActions} from './runtime.mjs';
import {getSourceId,isActiveGM} from './native-context.mjs';
import {isActualUseMessage} from './usage-events.mjs';
import {DEFENSIVE_ADVANCE_SOURCE} from './defensive-advance-compat.mjs';
import {advancePosition,sameAdvancePosition,defensiveAdvanceContext,defensiveAdvanceMeleeChoices,defensiveAdvanceMovementProof} from './defensive-advance-rules.mjs';
import {runDefensiveAdvanceMovement,rollDefensiveAdvanceStrike,defensiveAdvanceStrikeProof} from './defensive-advance-native.mjs';

const values=c=>Array.from(c?.values?.()??c??[]),own=m=>m?.flags?.[MODULE_ID]?.defensiveAdvance,input=m=>m?.flags?.[MODULE_ID]?.defensiveAdvanceInput;
const author=m=>m?.author?.id??m?.user?.id??m?.user;
const nonce=()=>globalThis.foundry?.utils?.randomID?.(16)??globalThis.crypto.randomUUID();
const shieldReady=a=>{const s=a.attributes?.shield;return !!(s?.itemId&&s.raised&&!s.broken&&!s.destroyed)};

/** The original two-action flourish owns one native Stride and optional Strike.
 * A reload/GM migration never replays an unfinished or uncertain transaction. */
export function createDefensiveAdvance({game,fromUuid=globalThis.fromUuid,choose,startupCompatibility,onError=console.error}={}){
 const startup=Object.freeze({...startupCompatibility}),queue=new SerialActions(),pending=new Map(),finished=new Map(),ownerEntered=new Set();let socket,Hooks;
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
  message.updateSource({[`flags.${MODULE_ID}.defensiveAdvanceInput`]:{...input(message),nonce:input(message)?.nonce??nonce(),startup:startup.status,turn,tokenUuid:token.uuid,origin:advancePosition(token)}});
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
  if(!r&&!sameAdvancePosition(advancePosition(token),i.origin))throw Error('原始Use之后Token已经移动；不能借用新的起点。');
  if(r&&(r.nonce!==i.nonce||r.userId!==user.id||r.actorUuid!==actor.uuid||r.itemUuid!==item.uuid||r.tokenUuid!==token.uuid||r.gmId!==game.users.activeGM?.id))throw Error('列盾突进回执或主GM已改变；不会重放。');
  return current;
 }
 async function save(message,changes){gm();if(game.messages.get(message.id)!==message)throw Error('原始列盾突进卡已不存在。');await message.update({[`flags.${MODULE_ID}.defensiveAdvance`]:{...own(message),...changes}});gm();}
 async function asGM(name,payload,handler){if(isActiveGM(game))return handler(payload,game.user);if(!socket||!game.users.activeGM)throw Error('没有主GM通讯，列盾突进不会重放。');const r=await socket.executeAsUser(name,game.users.activeGM.id,payload);if(!r?.ok)throw Error(r?.error??'主GM通讯未确认。');return r.value;}
 async function ownerContext(payload,sender,status){
  if(sender?.id!==game.users.activeGM?.id)throw Error('只有当前主GM可以请求原操作者的列盾突进。');
  const message=game.messages.get(payload.messageId),r=own(message);if(!r||r.nonce!==payload.nonce||r.status!==status||r.userId!==game.user.id)throw Error('列盾突进原操作者认领不匹配。');
  const ctx=await context(message,game.user),key=`${r.nonce}:${status}`;if(ownerEntered.has(key))throw Error('本阶段已经进入；不会重放。');ownerEntered.add(key);return ctx;
 }
 async function bindPlan(payload,user){gm();return queue.run(payload.messageId,async()=>{
  gm();const m=game.messages.get(payload.messageId),ctx=await context(m,user),r=own(m),movement=ctx.token.movement,current=validate(ctx);
  if(r?.status!=='planning'||r.nonce!==payload.nonce||movement?.id!==payload.planId||movement.state!=='planned'||movement.user?.id!==user.id||!sameAdvancePosition(movement.origin,r.origin)||!sameAdvancePosition(advancePosition(ctx.token),r.origin)||!Number.isFinite(movement.pending?.cost)||movement.pending.cost<=0||movement.pending.cost>Math.min(r.speed,current.speed)||!movement.pending.waypoints?.length||movement.pending.waypoints.some(w=>w.action!=='walk'))throw Error('原生列盾突进计划无法与本Use绑定。');
  await save(m,{status:'moving',planId:movement.id});return true;
 });}
 async function recordMovement(token,movement,operation,user){
  if(!isActiveGM(game))return;const id=pending.get(token.uuid);if(!id)return;
  return queue.run(id,async()=>{
   gm();const m=game.messages.get(id),r=own(m);if(r?.status!=='moving')return;
   const ctx=await context(m,game.users.get(r.userId)),current=validate(ctx);
   const proof=defensiveAdvanceMovementProof({token,movement,operation,user,receipt:{...r,speed:Math.min(r.speed,current.speed)}});if(!proof)return;
   if(!movement.finished?.then)throw Error('缺少服务器原生移动完成承诺。');
   finished.set(r.nonce,{id:movement.id,promise:movement.finished});await save(m,proof);
  }).catch(async error=>{if(isActiveGM(game)&&game.messages.get(id))await save(game.messages.get(id),{status:'uncertain',result:error.message});throw error;});
 }
 async function confirmMovement(payload,user){gm();return queue.run(payload.messageId,async()=>{
  const m=game.messages.get(payload.messageId),ctx=await context(m,user),r=own(m),proof=finished.get(r?.nonce);validate(ctx);
  if(r?.nonce!==payload.nonce||r.status!=='moving'||!r.movementIds?.length||!proof||proof.id!==r.movementIds.at(-1)||ctx.token.movement?.id!==proof.id||ctx.token.movement.state!=='completed'||await proof.promise!==true)throw Error('没有本计划实际完成的服务器移动回执；不会继续Strike。');
  validate(ctx);if(!sameAdvancePosition(advancePosition(ctx.token),r.lastPosition))throw Error('完成后Token位置已改变。');await save(m,{status:'moved'});return true;
 });}
 async function ownerMove(payload,sender){
  const ctx=await ownerContext(payload,sender,'planning'),r=own(ctx.message);
  return runDefensiveAdvanceMovement({game,token:ctx.token,receipt:r,validate:()=>{validate(ctx);if(!['planning','moving','moved'].includes(own(ctx.message)?.status))throw Error('列盾突进移动阶段已失效。');},bindPlan:planId=>asGM('defensive-advance:plan',{...payload,planId},bindPlan),confirm:()=>asGM('defensive-advance:moved',payload,confirmMovement)});
 }
 function selected(ctx){validate(ctx);const r=own(ctx.message),target=game.scenes.get(ctx.token.parent.id)?.tokens.get(r.targetId),option=defensiveAdvanceMeleeChoices({...ctx,game,target}).find(o=>o.key===r.weaponKey);if(!option||!Number.isInteger(r.map)||typeof option.strike.variants?.[r.map]?.roll!=='function'||!sameAdvancePosition(advancePosition(ctx.token),r.lastPosition)||!shieldReady(ctx.actor))throw Error('所选近战Strike、目标触及、举盾或完成位置已改变。');return {target,option};}
 async function ownerStrike(payload,sender){const ctx=await ownerContext(payload,sender,'striking'),{target,option}=selected(ctx);return rollDefensiveAdvanceStrike({...ctx,game,Hooks,target,option,receipt:own(ctx.message),validate:()=>selected(ctx).option});}
 async function callOwner(kind,ctx){const payload={messageId:ctx.message.id,nonce:own(ctx.message).nonce},handler=kind==='move'?ownerMove:ownerStrike;gm();if(ctx.user.id===game.user.id)return handler(payload,game.user);if(!socket)throw Error('缺少原操作者客户端通讯。');const r=await socket.executeAsUser(`defensive-advance:${kind}`,ctx.user.id,payload);if(!r?.ok)throw Error(r?.error??'原操作者结果尚未确认。');return r.value;}
 async function pick(ctx,title,choices){const value=await choose({...ctx,title,choices});validate(ctx);if(value==null)return null;if(!choices.some(c=>c.value===value))throw Error('无效的列盾突进选择。');return value;}
 async function executeUsage({actor,item,message,user,action}){
  gm();if(action!=='defensive-advance'||resolveAction(item)!==action)throw Error('不是列盾突进的准确入口。');
  const ctx=await context(message,user);if(ctx.actor!==actor||ctx.item!==item)throw Error('原卡角色不匹配。');
  const entered=await queue.run(actor.uuid,async()=>{
   validate(ctx);if(own(message))return false;
   if(item.system.actionType?.value!=='action'||item.system.actions?.value!==2||!item.system.traits?.value?.includes('flourish'))throw Error('原始双动作华丽专长已改变。');
   const current=validate(ctx),uses=actor.flags?.[MODULE_ID]?.defensiveAdvanceUses??[];
   if(uses.some(r=>r.nonce===input(message).nonce))throw Error('此Use nonce已属于另一张原卡。');
   if(current.turn&&(uses.some(r=>r.turn===current.turn)||values(game.messages).some(m=>m.id!==message.id&&m.speaker?.actor===actor.id&&m.flags?.[MODULE_ID]?.defensiveAdvanceObservedTurn===current.turn&&isActualUseMessage(m)&&m.flags?.pf2e?.origin?.rollOptions?.includes('origin:item:trait:flourish'))))throw Error('实际本回合已承诺华丽动作。');
   const r={nonce:input(message).nonce,messageId:message.id,actorUuid:actor.uuid,itemUuid:item.uuid,tokenUuid:ctx.token.uuid,userId:user.id,gmId:game.user.id,turn:current.turn,speed:current.speed,cost:2,flourish:true,status:'planning',origin:advancePosition(ctx.token),lastPosition:advancePosition(ctx.token),movementIds:[],movementCost:0};
   await actor.update({[`flags.${MODULE_ID}.defensiveAdvanceUses`]:[...uses.slice(-63),{nonce:r.nonce,messageId:message.id,turn:r.turn,cost:2}]});await save(message,r);pending.set(ctx.token.uuid,message.id);return true;
  });
  if(!entered)return own(message)?.result??'本次列盾突进已开始或完成；不会重放。';
  try{
   // Patreon owns Raise a Shield. Allow its original postInfo executor to finish.
   for(let i=0;!shieldReady(actor)&&i<20;i++){await new Promise(resolve=>setTimeout(resolve,100));validate(ctx);}
   if(!shieldReady(actor))throw Error('原生举盾尚未确认；不创建替代效果，请手工继续。');
   if(await callOwner('move',ctx)!==true){await save(message,{status:'cancelled',result:'已取消后续移动；已完成举盾与双动作华丽不回退。'});return own(message).result;}
   validate(ctx);
   const targets=values(ctx.token.parent.tokens).filter(target=>defensiveAdvanceMeleeChoices({...ctx,game,target}).length);
   if(!targets.length){await save(message,{status:'done',result:'举盾与Stride已完成；没有可及的敌人可作内含近战Strike。'});return own(message).result;}
   const targetId=await pick(ctx,'列盾突进：选择内含近战Strike的敌人',[...targets.map(t=>({value:t.id,label:t.name??t.actor.name??t.id})),{value:'decline',label:'结束活动，不作Strike'}]);
   if(!targetId||targetId==='decline'){await save(message,{status:'done',result:'举盾与Stride已完成，已放弃内含Strike。'});return own(message).result;}
   const target=ctx.token.parent.tokens.get(targetId),options=defensiveAdvanceMeleeChoices({...ctx,game,target});
   const weaponKey=await pick(ctx,'选择原生近战Strike',options.map(o=>({value:o.key,label:o.strike.label??o.strike.item.name})));
   if(!weaponKey){await save(message,{status:'cancelled',result:'已取消内含Strike，已完成动作不回退。'});return own(message).result;}
   const map=await pick(ctx,'列盾突进：选择当前MAP',[{value:'0',label:'本回合尚未攻击（MAP 0）'},{value:'1',label:'已攻击一次（MAP 1）'},{value:'2',label:'已攻击两次或更多（MAP 2）'}]);
   if(map===null){await save(message,{status:'cancelled'});return '已取消内含Strike。';}
   await save(message,{status:'striking',targetId,weaponKey,map:Number(map)});selected(ctx);
   const checkId=await callOwner('strike',ctx);validate(ctx);
   if(checkId===null){await save(message,{status:'cancelled',result:'已取消原生Strike，已完成动作不回退。'});return own(message).result;}
   const selection=selected(ctx);defensiveAdvanceStrikeProof(game.messages.get(checkId),{...ctx,...selection,game,receipt:own(message)});
   await save(message,{status:'done',checkId,result:'列盾突进已完成：举盾、Stride与一次原生近战Strike。'});return own(message).result;
  }catch(error){if(isActiveGM(game)&&own(message)?.gmId===game.user.id)await save(message,{status:'uncertain',result:`${error.message} 已完成动作不回退，未知结果不会重放。`});throw error;}
  finally{if(pending.get(ctx.token.uuid)===message.id)pending.delete(ctx.token.uuid);finished.delete(input(message).nonce);}
 }
 function register({Hooks:hooks,socket:api}={}){
  Hooks=hooks;socket=api;
  const ids=[['preCreateChatMessage',Hooks.on('preCreateChatMessage',captureCard)],['moveToken',Hooks.on('moveToken',(...args)=>recordMovement(...args).catch(onError))]];
  for(const[name,handler]of [['plan',bindPlan],['moved',confirmMovement],['move',ownerMove],['strike',ownerStrike]])socket?.register(`defensive-advance:${name}`,async function(payload){try{return {ok:true,value:await handler(payload,game.users.get(this.socketdata.userId))}}catch(error){return {ok:false,error:error.message}}});
  return()=>{for(const[name,id]of ids)Hooks.off(name,id);};
 }
 return {resolveAction,requiresActualUse:item=>!!resolveAction(item),captureUsage,executeUsage,register,get diagnostic(){return startup}};
}
