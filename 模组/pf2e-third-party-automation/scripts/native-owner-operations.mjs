import {MODULE_ID as ID} from './rules.mjs';
import {isActiveGM} from './native-context.mjs';
import {prepareWeaponSurgeDamageSnapshotItems} from './weapon-surge.mjs';
import {nativeRollEvent} from './manual-native-roll.mjs';

const values=collection=>Array.from(collection?.values?.()??collection??[]);
const clone=data=>structuredClone(data);
const author=message=>message.author?.id??message.author??message.user?.id??message.user;
const operations=message=>message.flags?.[ID]?.nativeOwnerOperations??{};
const Message=()=>globalThis.CONFIG?.ChatMessage?.documentClass??globalThis.ChatMessage;
const sessions=new WeakMap();

/** Check.roll awaits this resolver. Payment follows the user's final native
 * acceptance, and a closed dialog leaves resources intact. */
export async function beforeNativeRoll({Hooks,marker,showDialog,commit,native,assertLive,dialogKind='check',signal}){
 let commitTask,error;const wrapped=new WeakSet(),dialogs=new Set(),listeners=[];
 const validate=()=>{
  if(error)throw error;
  if(signal?.aborted)throw signal.reason instanceof Error?signal.reason:Error('Native execution aborted');
  assertLive?.();
 };
 const once=async()=>{validate();await (commitTask??=Promise.resolve().then(()=>{validate();return commit()}));validate();};
 if(!showDialog){await once();validate();return native(once);}
 if(!Hooks?.on)throw Error('缺少原生检定窗口接口，尚未支付。');
 let abort;const aborted=new Promise((_resolve,reject)=>{abort=reject;});
 const fail=caught=>{error??=caught;for(const dialog of dialogs)dialog.cancel();abort(error);};
 const checkLive=()=>{try{validate()}catch(caught){fail(caught)}};
 signal?.addEventListener('abort',checkLive,{once:true});
 if(assertLive)for(const event of ['updateUser','userConnected','updateChatMessage','deleteChatMessage','updateActor','deleteActor'])listeners.push([event,Hooks.on(event,checkLive)]);
 const dialogHook=dialogKind==='damage'?'renderDamageModifierDialog':'renderCheckModifiersDialog';
 const id=Hooks.on(dialogHook,app=>{
  if(!values(app.context?.options).includes(marker)||wrapped.has(app)||typeof app.resolve!=='function')return;
  wrapped.add(app);const resolve=app.resolve.bind(app);let submitted=false,settled=false,closed=false;
  const settle=value=>{
   if(settled)return;
   if(value)try{validate()}catch(caught){fail(caught);return;}
   settled=true;return resolve(value);
  };
  const dialog={cancel:()=>{settle(false);if(!closed){closed=true;void Promise.resolve(app.close?.()).catch(()=>{})}}};dialogs.add(dialog);
  app.resolve=accepted=>{if(submitted||settled)return;submitted=true;if(!accepted)return settle(false);return once().then(()=>settle(true),caught=>{fail(caught);return settle(false)});};
  checkLive();
 });
 checkLive();const nativeTask=Promise.resolve().then(()=>{validate();return native(once);});
 // Preparation can finish after the caller aborts. Keep only this exact-marker
 // render guard until the native promise settles, so a late window also closes.
 nativeTask.then(()=>Hooks.off(dialogHook,id),()=>Hooks.off(dialogHook,id));
 try{const result=await Promise.race([nativeTask,aborted]);if(error)throw error;return result;}finally{signal?.removeEventListener('abort',checkLive);for(const[event,id]of listeners)Hooks.off(event,id);}
}

/** Native UI belongs to the original owner. Only IDs and evaluated native data
 * cross the socket; the GM verifies the saved operation, never a reply alone. */
export function createNativeOwnerOperations({game,fromUuid=globalThis.fromUuid,scope,syncTimeoutMs=15000}={}){
 if(!/^[a-z][a-z0-9-]{0,40}$/.test(scope))throw Error('原生执行入口无效。');
 if(!sessions.has(game))sessions.set(game,new Map());const pending=sessions.get(game),committers=new Map();let socket,Hooks;
 const authority=user=>{if(!user||user.id!==game.users.activeGM?.id||user.isGM===false||user.active===false)throw Error('原生执行只能由当前主GM请求。');};
 const waitFor=read=>{
  const current=read();if(current)return Promise.resolve(current);
  if(!Hooks?.on)throw Error('原生执行文档尚未同步。');
  return new Promise((resolve,reject)=>{
   const ids=[];let done=false;const finish=(error,value)=>{if(done)return;done=true;clearTimeout(timer);for(const[event,id]of ids)Hooks.off(event,id);error?reject(error):resolve(value);};
   const check=()=>{try{const value=read();if(value)finish(null,value);}catch(error){finish(error);}};
   const timer=setTimeout(()=>finish(Error('原生执行同步超时；请由GM查看原卡，不能重复投骰。')),syncTimeoutMs);
   for(const event of ['createChatMessage','updateChatMessage'])ids.push([event,Hooks.on(event,check)]);check();
  });
 };
 async function operationReply(message,nonce,invoke,{readDone=state=>state?.status==='done',assertLive=()=>authority(game.user)}={}){
  const registrations=[];let complete;
  const saved=new Promise(resolve=>{complete=resolve;});
  const inspect=()=>{
   try{assertLive();}catch(error){complete({error});return;}
   const state=operations(message)[nonce];
   if(readDone(state))complete({persisted:true});
   else if(state?.status==='uncertain')complete({error:Error('原生执行结果尚未确认；不能重复投骰。')});
  };
  if(Hooks?.on)for(const event of ['updateChatMessage','updateUser','userConnected'])registrations.push([event,Hooks.on(event,inspect)]);
  const reply=Promise.resolve().then(invoke).then(response=>({response}),error=>({error}));
  // A saved native result is authoritative even if socketlib's pending reply
  // never settles. Native dialog time is not constrained by a timeout here.
  inspect();
  try{return await Promise.race(Hooks?.on?[saved,reply]:[reply]);}finally{for(const[event,id]of registrations)Hooks.off(event,id);}
 }
 function live(message,operation,requester,actor){
  authority(requester);
  if(game.messages.get(message.id)!==message||operations(message)[operation.nonce]?.nonce!==operation.nonce||operation.scope!==scope||operation.requesterId!==requester.id||operation.userId!==game.user.id||!actor?.testUserPermission(game.user,'OWNER')||actor.uuid!==operation.actorUuid||game.user.active===false)throw Error('原操作者、角色或本次执行身份已改变。');
 }
 async function executeNative({message,operation,actor,guard}){
  const request=operation.request,target=await fromUuid(request.targetUuid);guard();
  const originalGuard=guard;
  guard=()=>{originalGuard();if(!target?.actor||!target.object||target.uuid!==request.targetUuid||target.actor.uuid!==request.targetActorUuid)throw Error('原生操作的原目标已改变。');};guard();
  const showDialog=true;
  const options=new Set(request.options??[]),event=nativeRollEvent(game,request.type==='attack'?'check':'damage');
  const marker=`${ID}:native-operation:${operation.nonce}`;options.add(marker);
  const selectedPrivacy=context=>{
   const messageMode=context?.messageMode??game.settings?.get?.('core','messageMode')??'public',Class=Message();
   if(typeof Class?.applyMode!=='function'){if(messageMode!=='public')throw Error('缺少原生秘骰可见性接口，不能公开伤害。');return {messageMode,blind:false,whisper:[]};}
   const data=Class.applyMode({author:game.user.id},messageMode);return {messageMode,blind:data.blind===true,whisper:[...data.whisper??[]]};
  };
  let roller=actor;
  if(request.transientItems?.length){
   if(request.transientItems.length>20||request.transientItems.some(item=>item.type!=='effect'))throw Error('本次原生效果快照无效。');
   roller=actor.clone({items:prepareWeaponSurgeDamageSnapshotItems(actor,request.transientItems)},{keepId:true});
  }
  if(request.type==='spell-damage'){
   const original=actor.items.get(request.spellId);if(original?.type!=='spell')throw Error('原法术不存在。');
   const spell=original.loadVariant?.({castRank:request.rank,overlayIds:request.overlayIds??[]})??original;
   const native=await spell.getDamage({target,skipDialog:!showDialog});guard();
   if(!native)return {status:'cancelled'};
   const roll=await native.template.damage.roll.evaluate();guard();
   return {status:'rolled',roll:roll.toJSON(),privacy:selectedPrivacy(native.context),context:{options:[...native.context.options??[]],domains:[...native.context.domains??[]],traits:[...native.context.traits??[]],messageMode:native.context.messageMode}};
  }
  const strike=values(roller.system.actions).flatMap(strike=>[strike,...strike.altUsages??[]]).find(strike=>strike.type==='strike'&&strike.item?.id===request.weaponId&&(strike.item.altUsageType??'')===(request.altUsageType??''));
  if(!strike||strike.ready===false||!Number.isInteger(request.map)||request.map<0||request.map>2)throw Error('原生武器或多重攻击档位已失效。');
  if(request.type==='damage'){
   let context;const hook=Hooks?.on?.('renderDamageModifierDialog',app=>{if(values(app.context?.options).includes(marker))context=app.context;});
   try{const roll=await beforeNativeRoll({Hooks,marker,showDialog,dialogKind:'damage',commit:async()=>{},assertLive:guard,native:()=>strike[request.critical?'critical':'damage']({target:target.object,checkContext:request.checkContext,mapIncreases:request.map,options,event,createMessage:false})});guard();
    return roll?{status:'rolled',roll:roll.toJSON(),privacy:selectedPrivacy(context)}:{status:'cancelled'};
   }finally{if(hook!==undefined)Hooks.off('renderDamageModifierDialog',hook);}
  }
  if(request.type!=='attack')throw Error('不支持的原生执行类型。');
  let created;
  const rollNative=async commit=>strike.variants[request.map].roll({target:target.object,options,event,createMessage:false,callback:async(_roll,_outcome,raw)=>{
   // A native caller can explicitly suppress its dialog. Its evaluated callback
   // is the last safe payment boundary before publishing the actual check.
   if(commit)await commit();
   guard();const data=raw.toObject();delete data._id;data.author=game.user.id;
   data.flags={...data.flags,'xdy-pf2e-workbench':{...data.flags?.['xdy-pf2e-workbench'],noAutoDamageRoll:true},[ID]:{...data.flags?.[ID],...request.flags,usageGenerated:true,nativeOwnerOperation:{activityMessageId:message.id,nonce:operation.nonce,scope}}};
   created=await Message().create(data);guard();
  }});
  const commit=async()=>{
   guard();if(operations(message)[operation.nonce]?.committed)return;
   if(isActiveGM(game))await commitOperation({messageId:message.id,nonce:operation.nonce},game.user);
   else{
    if(!socket?.executeAsUser)throw Error('原生支付缺少主GM连接。');
    const {error}=await operationReply(message,operation.nonce,()=>socket.executeAsUser(`native-owner:${scope}:commit`,operation.requesterId,{messageId:message.id,nonce:operation.nonce}),{readDone:state=>state?.committed===true,assertLive:guard});
    if(error&&!operations(message)[operation.nonce]?.committed)throw error;
   }
   await waitFor(()=>operations(message)[operation.nonce]?.committed===true);guard();
  };
  const roll=await beforeNativeRoll({Hooks,marker,showDialog,commit:operation.requiresCommit?commit:async()=>{},native:operation.requiresCommit?rollNative:()=>rollNative(),assertLive:guard});guard();
  if(!roll&&!created)return {status:'cancelled'};
  if(!created)throw Error('原生攻击结果未确认；不能重复投骰。');
  return {status:'rolled',messageId:created.id};
 }
 async function ownerExecute(payload,requester){
  authority(requester);
  const message=await waitFor(()=>game.messages.get(payload?.messageId));
  const operation=await waitFor(()=>operations(message)[payload?.nonce]);
  const actor=await fromUuid(operation.actorUuid),guard=()=>live(message,operation,requester,actor);guard();
  const key=`${scope}:${message.id}:${operation.nonce}`;
  if(pending.has(key))return pending.get(key);
  const prior=operations(message)[operation.nonce];if(prior.status==='done')return clone(prior.result);
  if(prior.status!=='requested')throw Error('此原生执行已开始，结果尚未确认；不能重试。');
  const task=(async()=>{
   const save=(status,result)=>message.update({[`flags.${ID}.nativeOwnerOperations.${operation.nonce}`]:{...operations(message)[operation.nonce],status,...result?{result}:{} }});
   await save('started');guard();
   try{const result=await executeNative({message,operation,actor,guard});guard();await save('done',result);return result;}
   catch(error){await save('uncertain').catch(()=>{});throw error;}
  })();pending.set(key,task);task.finally(()=>{if(pending.get(key)===task)pending.delete(key);}).catch(()=>{});return task;
 }
 async function commitOperation(payload,requester){
  authority(game.user);const message=game.messages.get(payload?.messageId),operation=operations(message??{})[payload?.nonce],entry=committers.get(payload?.nonce);
  if(!operation||operation.scope!==scope||!['requested','started'].includes(operation.status)||operation.requesterId!==game.user.id||operation.userId!==requester?.id||!entry||entry.message!==message||!entry.actor.testUserPermission(requester,'OWNER'))throw Error('本次原生支付身份或状态不符。');
  if(operation.committed)return true;
  if(entry.pending)return entry.pending;
  return entry.pending=(async()=>{await entry.commit();authority(game.user);await message.update({[`flags.${ID}.nativeOwnerOperations.${operation.nonce}`]:{...operations(message)[operation.nonce],committed:true}});return true;})();
 }
 async function run({actor,message,user},request,_localNative,beforeRoll){
  authority(game.user);
  if(!user||user.active===false||!actor.testUserPermission(user,'OWNER')||game.messages.get(message.id)!==message||author(message)!==user.id)throw Error('原操作者离线、连接或权限无效；尚未代投。');
  if(user.id!==game.user.id&&!socket?.executeAsUser)throw Error('缺少原操作者连接；尚未在GM端代投。');
  const target=await fromUuid(request.targetUuid);authority(game.user);
  if(!target?.actor?.uuid)throw Error('原生操作的原目标不存在。');
  request={...request,targetActorUuid:target.actor.uuid};
  const nonce=globalThis.foundry?.utils?.randomID?.()??globalThis.crypto.randomUUID();
  const operation={nonce,scope,actorUuid:actor.uuid,userId:user.id,requesterId:game.user.id,status:'requested',requiresCommit:typeof beforeRoll==='function',request:clone(request)};
  if(operation.requiresCommit)committers.set(nonce,{actor,message,commit:beforeRoll});
  try{
  await message.update({[`flags.${ID}.nativeOwnerOperations.${nonce}`]:operation});authority(game.user);
  const {response,error}=await operationReply(message,nonce,()=>user.id===game.user.id?ownerExecute({messageId:message.id,nonce},game.user):socket.executeAsUser(`native-owner:${scope}`,user.id,{messageId:message.id,nonce}),{assertLive:()=>{
   authority(game.user);
   if(game.messages.get(message.id)!==message||author(message)!==user.id||target.actor?.uuid!==request.targetActorUuid)throw Error('本次原生操作来源或目标已改变；不能重复执行。');
   // A completed saved operation remains valid after its owner goes offline.
   // An unresolved request must release the GM queue when that owner leaves.
   if(operations(message)[nonce]?.status!=='done'&&(user.active===false||!actor.testUserPermission(user,'OWNER')))throw Error('原操作者已离线或失去角色权限；本次结果尚未确认，请GM查看原卡，不能重复执行。');
  }});
  authority(game.user);
  let saved=operations(message)[nonce];
  if(saved?.status!=='done'&&response&&!error)saved=await waitFor(()=>operations(message)[nonce]?.status==='done'&&operations(message)[nonce]);
  if(saved?.status==='done'){
   if(response&&JSON.stringify(response)!==JSON.stringify(saved.result))throw Error('原生执行回复与原卡记录不符。');
   const result=clone(saved.result);
   if(request.type==='attack'&&result.status==='rolled'){
    const check=await waitFor(()=>game.messages.get(result.messageId)),proof=check.flags?.[ID]?.nativeOwnerOperation,context=check.flags?.pf2e?.context;
    if(author(check)!==user.id||proof?.nonce!==nonce||proof.activityMessageId!==message.id||context?.type!=='attack-roll'||context.target?.token!==request.targetUuid||context.target?.actor!==request.targetActorUuid||target.actor?.uuid!==request.targetActorUuid||check.flags?.pf2e?.origin?.actor!==actor.uuid)throw Error('原生攻击消息与本次操作者或目标不符。');
   }
   return result;
  }
  throw error??Error('原生执行尚未确认；不能自动重复。');
  }finally{committers.delete(nonce);}
 }
 function register({socket:providedSocket,Hooks:providedHooks}={}){
  socket=providedSocket;Hooks=providedHooks;
  socket?.register?.(`native-owner:${scope}`,function(payload){return ownerExecute(payload,game.users.get(this.socketdata?.userId));});
  socket?.register?.(`native-owner:${scope}:commit`,function(payload){return commitOperation(payload,game.users.get(this.socketdata?.userId));});
 }
 return {run,register};
}

export function nativeTransientItems(strike,actor){
 const ids=new Set(values(actor.items).map(item=>item.id));
 return clone((strike.item.actor?._source?.items??[]).filter(item=>!ids.has(item._id)));
}
