import {MODULE_ID} from './rules.mjs';
import {SerialActions} from './runtime.mjs';
import {isActiveGM} from './native-context.mjs';
import {captureWorkbenchRecall,finalizeWorkbenchRecall,WORKBENCH_RECALL_UUID} from './knowledge-workbench.mjs';
const values=x=>Array.from(x?.values?.()??x??[]),doc=t=>t?.document??t;
const publicUUID='Compendium.xdy-pf2e-workbench.asymonous-benefactor-macros.Macro.es70r3Bq0bxZSCuk';
const source=i=>i?.sourceId??i?._stats?.compendiumSource??i?.flags?.core?.sourceId;
const random=()=>globalThis.foundry?.utils?.randomID?.()??globalThis.crypto.randomUUID();
/** Original owner runs the installed macro. GM only receives a persisted card id,
 * verifies that card, and settles its automatic primary result. No roll totals/DC
 * cross the public RPC response. */
export function createWorkbenchRecallController({game,fromUuid=globalThis.fromUuid,globals=globalThis,onResolved=()=>{},onError=console.error}={}){
 const queue=new SerialActions(),requests=new Map();let socket,registered=false;
 function activeGM(){if(!isActiveGM(game))throw Error('回忆知识的主 GM 已交接。');}
 async function settle(messageId,requester){
  if(!requester||game.users.get(requester.id)!==requester)throw Error('回忆知识原操作者不可验证。');
  activeGM();return queue.run(`settle:${messageId}`,async()=>{
   activeGM();const message=game.messages.get(messageId);if(!message||message.author?.id!==requester.id||game.users.get(requester.id)!==requester)throw Error('回忆知识原生卡与原操作者不匹配。');
   const state=message.flags?.[MODULE_ID]?.workbenchRecall,token=await fromUuid(state?.tokenUuid);
   if(token?.documentName!=='Token'||token.actor?.uuid!==message.actor?.uuid||message.speaker?.token!==token.id||message.speaker?.scene!==token.parent?.id)throw Error('回忆知识原生 Token 不匹配。');
    await finalizeWorkbenchRecall({game,message,fromUuid});if(!message.flags?.[MODULE_ID]?.workbenchRecall?.resolved){await onResolved(message);await message.update({[`flags.${MODULE_ID}.workbenchRecall.resolved`]:true});}return {messageId:message.id};
  });
 }
 async function local(input){
  const user=game.user;if(!user||game.users.get(user.id)!==user)throw Error('原操作者已离线。');
  const sourceCard=input.origin?.messageId?game.messages.get(input.origin.messageId):null;
  const state=sourceCard?.flags?.[MODULE_ID]?.knowledge?.recall,previous=state?.workbenchOperation;
  if(sourceCard&&(sourceCard.author?.id!==user.id||sourceCard.actor?.uuid!==input.actor.uuid))throw Error('回忆知识原生动作卡不属于原操作者。');
  if(previous?.requestId!==undefined&&previous.requestId!==input.requestId)throw Error('本次原生动作已经绑定另一次回忆知识。');
  if(previous?.status==='rolling')throw Error('原生回忆知识已经开始，结果尚不明确；不会再次投骰。');
  let messageId=previous?.messageId;
  if(!messageId){
   if(sourceCard)await sourceCard.update({[`flags.${MODULE_ID}.knowledge.recall`]:{...state,workbenchOperation:{requestId:input.requestId,status:'rolling'}}});
   const capture=await captureWorkbenchRecall({...input,game,user,fromUuid,globals});messageId=capture.message.id;
   if(sourceCard)await sourceCard.update({[`flags.${MODULE_ID}.knowledge.recall`]:{...sourceCard.flags[MODULE_ID].knowledge.recall,workbenchOperation:{requestId:input.requestId,status:'done',messageId}}});
  }
  if(isActiveGM(game))await settle(messageId,user);
  else {const gm=game.users.activeGM;if(!socket||!gm?.active)throw Error('原生秘骰已保存，需要在线 GM 处理知识联动；不会再次投骰。');const response=await socket.executeAsUser('knowledge-rk-finalize',gm.id,{messageId});if(!response?.ok)throw Error(response?.error??'回忆知识结果尚未结算；不会再次投骰。');}
  return {messageId};
 }
 async function run({actor,token,user=game.user,targetUuids=null,origin=null,requestId=random(),statistic=null,assurance=false,dc=null}={}){
  const input={actor,token:doc(token),targetUuids:targetUuids??values(user.targets).map(t=>doc(t).uuid),origin,requestId,statistic,assurance,dc};
  if(!user||game.users.get(user.id)!==user||!actor?.testUserPermission?.(user,'OWNER'))throw Error('回忆知识需要原操作者权限。');
  if(user.id===game.user.id)return queue.run(`actor:${actor.uuid}`,()=>local(input));
  activeGM();if(!user.active||!socket)throw Error('原操作者不在线，未代为进行回忆知识。');
  if(!origin?.messageId)throw Error('附带回忆知识缺少原生动作卡。');
  const payload={actorUuid:actor.uuid,tokenUuid:doc(token)?.uuid,targetUuids:input.targetUuids,requestId,origin,statistic,assurance,dc};
  const reply=await socket.executeAsUser('knowledge-rk-run',user.id,payload);activeGM();if(!reply?.ok)throw Error(reply?.error??'回忆知识回复不明确；不会重新投骰。');
  // Slow native document synchronization may make the card temporarily absent.
  // The owner also sent the finalize RPC. Never poll/replay the macro here.
  return reply.value;
 }
 function actorFor(input={}){return input.actor??values(input.actors)[0]??globals.canvas?.tokens?.controlled?.[0]?.actor??game.user.character;}
 function tokenFor(actor,input={}){const supplied=doc(input.token);if(supplied?.actor?.uuid===actor?.uuid)return supplied;const controlled=values(globals.canvas?.tokens?.controlled).map(doc).filter(t=>t.actor?.uuid===actor?.uuid),active=values(actor?.getActiveTokens?.(true,true)).map(doc);return controlled[0]??(active.length===1?active[0]:null);}
 async function action(input={}){const actor=actorFor(input),token=tokenFor(actor,input);if(!actor||!token)throw Error('请选中进行回忆知识的角色 Token。');
  // A plain UI statistic is a player guess. Only an explicit ability constraint
  // changes the automatic primary skill or applies Assurance.
  const constraint=input.knowledgeConstraint??{};
  const result=await run({actor,token,targetUuids:input.target?[doc(input.target).uuid]:null,origin:{rollOptions:input.rollOptions??[]},...constraint});
  const message=game.messages.get(result.messageId);return [{message}];
 }
 function register({Hooks,libWrapper,socket:socketApi}={}){
  if(registered)return()=>{};registered=true;socket=socketApi;const cleanup=[],listeners=new Map(),hooks=[];
  const on=(name,fn)=>{if(Hooks)hooks.push([name,Hooks.on(name,fn)]);};
  if(socket){socket.register('knowledge-rk-finalize',async function(payload){try{if(!registered)throw Error('回忆知识入口已关闭。');return {ok:true,value:await settle(payload?.messageId,game.users.get(this.socketdata.userId))};}catch(e){return {ok:false,error:e.message};}});
   socket.register('knowledge-rk-run',async function(payload){try{if(!registered)throw Error('回忆知识入口已关闭。');
    if(this.socketdata.userId!==game.users.activeGM?.id)throw Error('只有主 GM 可派发原生动作附带的回忆知识。');
    if(!payload?.requestId||!payload.origin?.messageId)throw Error('缺少本次原生动作来源。');
    if(requests.has(payload.requestId))return await requests.get(payload.requestId);
    const operation=queue.run(`actor:${payload.actorUuid}`,async()=>{const actor=await fromUuid(payload.actorUuid),token=await fromUuid(payload.tokenUuid),original=game.messages.get(payload.origin.messageId),state=original?.flags?.[MODULE_ID]?.knowledge?.recall;
     if(original?.author?.id!==game.user.id||original.actor?.uuid!==actor?.uuid||state?.userId!==game.user.id||state.actorUuid!==actor.uuid||JSON.stringify(state.targetUuids??[state.targetUuid])!==JSON.stringify(payload.targetUuids)||token?.actor?.uuid!==actor.uuid)throw Error('附带回忆知识不属于当前原操作者的本次动作。');
     return {ok:true,value:await local({actor,token,targetUuids:payload.targetUuids,origin:payload.origin,requestId:payload.requestId,statistic:payload.statistic,assurance:payload.assurance,dc:payload.dc})};});requests.set(payload.requestId,operation);try{return await operation;}finally{if(requests.get(payload.requestId)===operation)requests.delete(payload.requestId);}
   }catch(e){return {ok:false,error:e.message};}});
  }
  const native=game.pf2e?.actions?.get?.('recall-knowledge');
  if(native?.use){const original=native.use,wrapped=input=>action(input);native.use=wrapped;cleanup.push(()=>{if(native.use===wrapped)native.use=original;});
   // Explicit native variants otherwise bypass the action instance's use().
   const variant=native.getDefaultVariant?.(),prototype=variant&&Object.getPrototypeOf(variant),use=prototype?.use;if(use){const bridge=function(input){return this.slug==='recall-knowledge'?action(input):use.call(this,input);};prototype.use=bridge;cleanup.push(()=>{if(prototype.use===bridge)prototype.use=use;});}}
   // Installed HUD apiExpose freezes actions and locks the parent property.
   // Its real statistic-action DOM boundary is captured below before the HUD
   // handler opens a skill chooser; writing to its public API would abort ready.
  const macroPath='CONFIG.Macro.documentClass.prototype.execute';
  if(libWrapper){libWrapper.register('pf2e-third-party-automation',macroPath,function(wrapped,scope={}){
    if(![WORKBENCH_RECALL_UUID,publicUUID].includes(this.uuid)&&![WORKBENCH_RECALL_UUID,publicUUID].includes(source(this)))return wrapped(scope);
    if(scope.ChatMessage&&scope.ChatMessage!==globals.ChatMessage)return wrapped(scope); // Adapter passes scoped native capture classes.
    return action(scope);
   },'MIXED');cleanup.push(()=>libWrapper.unregister('pf2e-third-party-automation',macroPath));
   // Embedded native RK items use the shared usage-events hotbar wrapper. Their
   // actual-use source card then dispatches knowledge:recall to this controller.
   // libWrapper permits only one registration per module/path.
   }
  function release(app){const entry=listeners.get(app);if(entry){for(const type of ['click','contextmenu'])entry.root.removeEventListener(type,entry.listener,true);listeners.delete(app);}}
  function render(app,html){release(app);const root=html?.addEventListener?html:html?.[0]??app.element,actor=app.actor??app.document?.actor??app.parent?.actor;if(!root?.addEventListener||!actor)return;
   const listener=event=>{const button=event.target?.closest?.('[data-action="roll-statistic-action"][data-key="recall-knowledge"],[data-action="recall-knowledge"],[data-pf2-action="recall-knowledge"]');if(!button||!root.contains(button))return;event.preventDefault();event.stopImmediatePropagation();Promise.resolve(action({actor,event})).catch(onError);};for(const type of ['click','contextmenu'])root.addEventListener(type,listener,true);listeners.set(app,{root,listener});}
  for(const name of ['renderApplication','renderApplicationV2','renderActorSheetPF2e','renderCharacterSheetPF2e','renderActorSheetV2'])on(name,render);
  for(const name of ['closeApplication','closeApplicationV2','closeActorSheetPF2e'])on(name,release);
  const recover=message=>{const state=message?.flags?.[MODULE_ID]?.workbenchRecall;if(isActiveGM(game)&&state?.schema===1&&!state.resolved&&['pending','done'].includes(state.status))Promise.resolve(settle(message.id,message.author)).catch(onError);};
  on('createChatMessage',recover);on('updateChatMessage',recover);on('updateUser',()=>{if(isActiveGM(game))for(const message of values(game.messages))recover(message);});
  if(isActiveGM(game))for(const message of values(game.messages))recover(message);
  return()=>{for(const [name,id]of hooks)Hooks.off(name,id);for(const app of listeners.keys())release(app);for(const fn of cleanup.reverse())fn();registered=false;};
 }
 return {run,action,register,settle};
}
