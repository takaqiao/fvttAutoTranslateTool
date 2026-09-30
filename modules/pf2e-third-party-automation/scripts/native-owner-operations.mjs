import {MODULE_ID as ID} from './rules.mjs';
import {isActiveGM} from './native-context.mjs';

const values=collection=>Array.from(collection?.values?.()??collection??[]);
const clone=data=>structuredClone(data);
const author=message=>message.author?.id??message.author??message.user?.id??message.user;
const operations=message=>message.flags?.[ID]?.nativeOwnerOperations??{};
const Message=()=>globalThis.CONFIG?.ChatMessage?.documentClass??globalThis.ChatMessage;
const sessions=new WeakMap();

/** Check.roll awaits this resolver. Payment follows the user's final native
 * acceptance, and a closed dialog leaves resources intact. */
export async function beforeNativeRoll({Hooks,marker,showDialog,commit,native}){
 let committed=false,error;const wrapped=new WeakSet();
 const once=async()=>{if(!committed){await commit();committed=true;}};
 if(!showDialog){await once();return native(once);}
 if(!Hooks?.on)throw Error('缺少原生检定窗口接口，尚未支付。');
 const id=Hooks.on('renderCheckModifiersDialog',app=>{
  if(!values(app.context?.options).includes(marker)||wrapped.has(app)||typeof app.resolve!=='function')return;
  wrapped.add(app);const resolve=app.resolve.bind(app);let submitted=false;
  app.resolve=accepted=>{if(submitted)return;submitted=true;if(!accepted)return resolve(false);return once().then(()=>resolve(true),caught=>{error=caught;return resolve(false);});};
 });
 try{const result=await native(once);if(error)throw error;return result;}finally{Hooks.off('renderCheckModifiersDialog',id);}
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
 function live(message,operation,requester,actor){
  authority(requester);
  if(game.messages.get(message.id)!==message||operations(message)[operation.nonce]?.nonce!==operation.nonce||operation.scope!==scope||operation.requesterId!==requester.id||operation.userId!==game.user.id||!actor?.testUserPermission(game.user,'OWNER')||actor.uuid!==operation.actorUuid||game.user.active===false)throw Error('原操作者、角色或本次执行身份已改变。');
 }
 async function executeNative({message,operation,actor,guard}){
  const request=operation.request,target=await fromUuid(request.targetUuid);guard();
  if(!target?.actor||!target.object||target.uuid!==request.targetUuid)throw Error('原生操作的原目标已改变。');
  const options=new Set(request.options??[]),event={ctrlKey:false,metaKey:false,shiftKey:game.user.settings?.[request.type==='attack'?'showCheckDialogs':'showDamageDialogs']??true};
  const marker=`${ID}:native-operation:${operation.nonce}`;options.add(marker);
  let roller=actor;
  if(request.transientItems?.length){
   if(request.transientItems.length>20||request.transientItems.some(item=>item.type!=='effect'))throw Error('本次原生效果快照无效。');
   const ids=new Set(values(actor.items).map(item=>item.id));
   roller=actor.clone({items:[...clone(actor._source.items),...clone(request.transientItems).filter(item=>!ids.has(item._id))]},{keepId:true});
  }
  if(request.type==='spell-damage'){
   const original=actor.items.get(request.spellId);if(original?.type!=='spell')throw Error('原法术不存在。');
   const spell=original.loadVariant?.({castRank:request.rank,overlayIds:request.overlayIds??[]})??original;
   const native=await spell.getDamage({target,skipDialog:!event.shiftKey});guard();
   if(!native)return {status:'cancelled'};
   const roll=await native.template.damage.roll.evaluate();guard();
   return {status:'rolled',roll:roll.toJSON(),context:{options:[...native.context.options??[]],domains:[...native.context.domains??[]],traits:[...native.context.traits??[]]}};
  }
  const strike=values(roller.system.actions).flatMap(strike=>[strike,...strike.altUsages??[]]).find(strike=>strike.type==='strike'&&strike.item?.id===request.weaponId&&(strike.item.altUsageType??'')===(request.altUsageType??''));
  if(!strike||strike.ready===false||!Number.isInteger(request.map)||request.map<0||request.map>2)throw Error('原生武器或多重攻击档位已失效。');
  if(request.type==='damage'){
   const roll=await strike[request.critical?'critical':'damage']({target:target.object,checkContext:request.checkContext,mapIncreases:request.map,options,event,createMessage:false});guard();
   return roll?{status:'rolled',roll:roll.toJSON()}:{status:'cancelled'};
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
    await socket.executeAsUser(`native-owner:${scope}:commit`,operation.requesterId,{messageId:message.id,nonce:operation.nonce});
   }
   await waitFor(()=>operations(message)[operation.nonce]?.committed===true);guard();
  };
  const roll=operation.requiresCommit?await beforeNativeRoll({Hooks,marker,showDialog:event.shiftKey,commit,native:rollNative}):await rollNative();guard();
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
  const nonce=globalThis.foundry?.utils?.randomID?.()??globalThis.crypto.randomUUID();
  const operation={nonce,scope,actorUuid:actor.uuid,userId:user.id,requesterId:game.user.id,status:'requested',requiresCommit:typeof beforeRoll==='function',request:clone(request)};
  if(operation.requiresCommit)committers.set(nonce,{actor,message,commit:beforeRoll});
  try{
  await message.update({[`flags.${ID}.nativeOwnerOperations.${nonce}`]:operation});authority(game.user);
  let response,error;
  try{response=user.id===game.user.id?await ownerExecute({messageId:message.id,nonce},game.user):await socket.executeAsUser(`native-owner:${scope}`,user.id,{messageId:message.id,nonce});}catch(caught){error=caught;}
  authority(game.user);
  let saved=operations(message)[nonce];
  if(saved?.status!=='done'&&response&&!error)saved=await waitFor(()=>operations(message)[nonce]?.status==='done'&&operations(message)[nonce]);
  if(saved?.status==='done'){
   if(response&&JSON.stringify(response)!==JSON.stringify(saved.result))throw Error('原生执行回复与原卡记录不符。');
   const result=clone(saved.result);
   if(request.type==='attack'&&result.status==='rolled'){
    const check=await waitFor(()=>game.messages.get(result.messageId)),proof=check.flags?.[ID]?.nativeOwnerOperation,context=check.flags?.pf2e?.context;
    if(author(check)!==user.id||proof?.nonce!==nonce||proof.activityMessageId!==message.id||context?.type!=='attack-roll'||context.target?.token!==request.targetUuid||check.flags?.pf2e?.origin?.actor!==actor.uuid)throw Error('原生攻击消息与本次操作者或目标不符。');
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
