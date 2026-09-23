import {MODULE_ID} from './rules.mjs';

const sessions=new WeakMap();
const activity=message=>message?.flags?.[MODULE_ID]?.medic;
const execution=message=>message?.flags?.[MODULE_ID]?.medicExecution;
const sameResult=(a,b)=>a?.status===b?.status&&a?.checkId===b?.checkId&&a?.resultId===b?.resultId;

/** Only the original author runs native UI. The GM receives saved message IDs, never a remote roll total. */
export function createMedicOwner({game,fromUuid,context,execute,syncTimeoutMs=15000}={}){
 if(!sessions.has(game))sessions.set(game,new Map());
 const pending=sessions.get(game);let socket,Hooks;
 const authority=user=>{if(!user?.active||!user.isGM||game.users.get(user.id)!==user||game.users.activeGM?.id!==user.id)throw Error('医疗原生执行只能由当前主GM请求。');};
 function waitFor(read,events){
  const current=read();if(current)return Promise.resolve(current);
  if(!Hooks?.on)throw Error('准确医疗文档尚未同步。');
  return new Promise((resolve,reject)=>{
   let finished=false;const ids=[];
   const finish=(error,value)=>{if(finished)return;finished=true;clearTimeout(timer);for(const [name,id]of ids)Hooks.off(name,id);error?reject(error):resolve(value);};
   const check=()=>{try{const value=read();if(value)finish(null,value);}catch(error){finish(error);}};
   const timer=setTimeout(()=>finish(Error('准确医疗文档同步超时；不能重复执行。')),syncTimeoutMs);
   for(const event of events)ids.push([event,Hooks.on(event,check)]);check();
  });
 }
 const message=id=>{if(typeof id!=='string'||!id)throw Error('缺少准确医疗消息ID。');return waitFor(()=>game.messages.get(id),['createChatMessage']);};
 async function ownerExecute(payload,requester){
  authority(requester);
  const card=await message(payload?.messageId);
  const state=await waitFor(()=>{authority(requester);const state=activity(card);return state?.nativeKind?state:null;},['updateChatMessage']);
  if(state.nonce!==payload.nonce||state.userId!==game.user.id)throw Error('本客户端不是原医疗操作者。');
  const ctx=await context(card),healer=await fromUuid(state.healerUuid),target=await fromUuid(state.targetUuid);
  if(healer?.actor!==ctx.actor||!target?.actor)throw Error('原医疗Token不存在。');
  const guard=()=>{
   authority(requester);const current=activity(card);
   if(game.messages.get(card.id)!==card||game.user!==ctx.user||!ctx.user.active||!ctx.actor.testUserPermission(ctx.user,'OWNER')||ctx.actor.items.get(ctx.item.id)!==ctx.item||ctx.actor.flags?.[MODULE_ID]?.medicUses?.[state.nonce]!==card.id||current?.nonce!==state.nonce||current.nativeKind!==state.nativeKind||current.healerUuid!==healer.uuid||current.targetUuid!==target.uuid||current.userId!==ctx.user.id||!['rolling','treatment'].includes(current.status))throw Error('原医疗执行身份或状态已变化，不能重复执行。');
  };
  guard();const key=`${card.id}:${state.nonce}`;
  if(pending.has(key))return pending.get(key);
  const operation=(async()=>{
   const prior=execution(card);if(prior)throw Error('此医疗原生执行已开始或结果未确认，不能重掷。');
   const proof={nonce:state.nonce,kind:state.nativeKind,userId:ctx.user.id,healerUuid:healer.uuid,targetUuid:target.uuid};
   const save=async(status,result)=>{await card.update({[`flags.${MODULE_ID}.medicExecution`]:{...proof,status,...result?{result}:{}}});};
   await save('started');guard();
   try{
    const result=await execute({...ctx,healer,target,state,dc:payload.dc,validate:guard});guard();
    if(!['cancelled','delegated'].includes(result?.status)||result.status==='delegated'&&typeof result.checkId!=='string')throw Error('原生医疗没有准确结果消息。');
    await save('done',result);return result;
   }catch(error){await save('uncertain').catch(()=>{});throw error;}
  })();
  pending.set(key,operation);operation.finally(()=>{if(pending.get(key)===operation)pending.delete(key);}).catch(()=>{});return operation;
 }
 async function run(ctx,{dc}={}){
  authority(game.user);const state=activity(ctx.message),payload={messageId:ctx.message.id,nonce:state.nonce,...dc===undefined?{}:{dc}};
  let result;
  if(ctx.user.id===game.user.id)result=await ownerExecute(payload,game.user);
  else{
   if(!socket?.executeAsUser)throw Error('缺少原医疗操作者连接；未在GM客户端代投。');
   const response=await socket.executeAsUser('medic:execute',ctx.user.id,payload);
   if(!response?.ok)throw Error(response?.error??'未收到操作者的医疗执行结果。');result=response.value;
  }
  authority(game.user);
  const saved=await waitFor(()=>{const value=execution(ctx.message);return value?.status==='done'?value:null;},['updateChatMessage']);
  if(saved.nonce!==state.nonce||saved.kind!==state.nativeKind||saved.userId!==ctx.user.id||saved.healerUuid!==state.healerUuid||saved.targetUuid!==state.targetUuid||!sameResult(saved.result,result))throw Error('操作者的医疗结果与原卡记录不符。');
  return {...result,check:result.checkId?await message(result.checkId):null,resultMessage:result.resultId?await message(result.resultId):null};
 }
 function register({socket:providedSocket,Hooks:providedHooks}={}){
  socket=providedSocket;Hooks=providedHooks;
  socket?.register?.('medic:execute',async function(payload){try{return {ok:true,value:await ownerExecute(payload,game.users.get(this.socketdata?.userId))};}catch(error){return {ok:false,error:error.message};}});
 }
 return {run,register};
}
