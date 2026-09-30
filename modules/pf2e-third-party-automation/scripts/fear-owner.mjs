import {MODULE_ID as ID} from './rules.mjs';
import {getSourceId} from './native-context.mjs';

const sessions=new WeakMap(),author=message=>message?.author?.id??message?.user?.id??message?.user;
const operation=(message,kind)=>message?.flags?.[ID]?.fearNative?.[kind];
/** The owner uses native UI; the GM settles only verified saved check messages.
 * Secret DCs travel in the authenticated request, never the public activity. */
export function createFearOwner({game,context,execute,syncTimeoutMs=15000}={}){
 if(!sessions.has(game))sessions.set(game,new Map());const pending=sessions.get(game);let socket,Hooks;
 const authority=user=>{if(!user?.isGM||user.active===false||game.users.get(user.id)!==user||game.users.activeGM?.id!==user.id)throw Error('恐惧能力原生执行只能由当前主GM请求。');};
 function waitFor(read){
  const value=read();if(value)return Promise.resolve(value);if(!Hooks?.on)throw Error('恐惧能力的原生文档尚未同步。');
  return new Promise((resolve,reject)=>{
   const ids=[];let done=false;const finish=(error,value)=>{if(done)return;done=true;clearTimeout(timer);for(const[name,id]of ids)Hooks.off(name,id);error?reject(error):resolve(value);};
   const check=()=>{try{const value=read();if(value)finish(null,value);}catch(error){finish(error);}};
   const timer=setTimeout(()=>finish(Error('恐惧能力原生文档同步超时；不能重复投骰。')),syncTimeoutMs);
   for(const name of ['createChatMessage','updateChatMessage'])ids.push([name,Hooks.on(name,check)]);check();
  });
 }
 async function ownerExecute(payload,requester){
  authority(requester);if(!['knowledge','battleCry'].includes(payload?.kind))throw Error('无效的恐惧能力原生入口。');
  const message=await waitFor(()=>game.messages.get(payload.messageId));
  const saved=await waitFor(()=>operation(message,payload.kind));
  if(saved.nonce!==payload.nonce||saved.userId!==game.user.id||author(message)!==game.user.id)throw Error('本客户端不是本次原操作者。');
  const ctx=await context(message,game.user,payload.kind);
  const guard=()=>{
   authority(requester);const current=operation(message,payload.kind);
   if(game.messages.get(message.id)!==message||!game.user.active||!ctx.actor.testUserPermission(game.user,'OWNER')||ctx.actor.items.get(ctx.item.id)!==ctx.item||getSourceId(ctx.item)!==saved.itemSourceId||current?.nonce!==saved.nonce||current.userId!==game.user.id||current.actorUuid!==ctx.actor.uuid||current.itemUuid!==ctx.item.uuid||current.targetUuid!==ctx.target.uuid||ctx.target.actor?.uuid!==saved.targetActorUuid||ctx.target.parent?.tokens.get(ctx.target.id)!==ctx.target)throw Error('本次恐惧能力的操作者、来源或目标已改变。');
  };
  guard();const key=`${message.id}:${payload.kind}:${saved.nonce}`;if(pending.has(key))return pending.get(key);
  if(saved.status==='done')return saved.result;if(saved.status!=='requested')throw Error('原生恐惧检定已经开始，不能重复投骰。');
  const task=(async()=>{
   const save=(status,result)=>message.update({[`flags.${ID}.fearNative.${payload.kind}`]:{...saved,status,...result?{result}:{}}});
   await save('started');guard();
   try{
    const marker=`${ID}:fear:${saved.nonce}`,check=await execute({...ctx,payload,marker,validate:guard});guard();
    if(check){
     if(game.messages.get(check.id)!==check||author(check)!==game.user.id||check.speaker?.actor!==ctx.actor.id||check.flags?.pf2e?.context?.type!=='skill-check'||!check.flags.pf2e.context.options?.includes(marker))throw Error('缺少本次原操作者的原生检定卡。');
     await check.update({[`flags.${ID}.fearNativeCheck`]:{messageId:message.id,nonce:saved.nonce,kind:payload.kind,userId:game.user.id}});guard();
    }
    const result=check?{status:'rolled',checkId:check.id}:{status:'cancelled'};await save('done',result);return result;
   }catch(error){await save('uncertain').catch(()=>{});throw error;}
  })();pending.set(key,task);task.finally(()=>{if(pending.get(key)===task)pending.delete(key);}).catch(()=>{});return task;
 }
 async function run(ctx,kind,data={}){
  authority(game.user);const {actor,item,message,user,target}=ctx;
  if(!user?.active||!actor.testUserPermission(user,'OWNER')||author(message)!==user.id||game.messages.get(message.id)!==message)throw Error('原恐惧能力操作者或消息已改变。');
  if(user.id!==game.user.id&&!socket?.executeAsUser)throw Error('缺少原操作者连接，未在GM端代投。');
  if(operation(message,kind))throw Error('本次恐惧能力原生执行已经开始，不能重放。');
  const nonce=globalThis.foundry?.utils?.randomID?.()??globalThis.crypto.randomUUID(),saved={nonce,kind,userId:user.id,actorUuid:actor.uuid,itemUuid:item.uuid,itemSourceId:getSourceId(item),targetUuid:target.uuid,targetActorUuid:target.actor.uuid,status:'requested'};
  await message.update({[`flags.${ID}.fearNative.${kind}`]:saved});authority(game.user);
  const payload={messageId:message.id,kind,nonce,...data};
  let response,error;try{response=user.id===game.user.id?{ok:true,value:await ownerExecute(payload,game.user)}:await socket.executeAsUser('fear:execute',user.id,payload);}catch(caught){error=caught;}
  authority(game.user);let final=operation(message,kind);
  if(final?.status!=='done'){
   if(!response?.ok)throw error??Error(response?.error??'原操作者结果尚未确认。');
   final=await waitFor(()=>operation(message,kind)?.status==='done'&&operation(message,kind));
  }
  if(final.nonce!==nonce||response?.ok&&JSON.stringify(response.value)!==JSON.stringify(final.result))throw Error('恐惧能力执行回复与原卡记录不符。');
  if(final.result.status==='cancelled')return {status:'cancelled'};
  const check=await waitFor(()=>game.messages.get(final.result.checkId)),pf=check.flags?.pf2e?.context,proof=check.flags?.[ID]?.fearNativeCheck;
  if(author(check)!==user.id||check.speaker?.actor!==actor.id||check.rolls?.length!==1||!Number.isFinite(check.rolls[0].total)||pf?.type!=='skill-check'||pf.isReroll||pf.target?.token!==target.uuid||pf.target?.actor!==saved.targetActorUuid||target.actor?.uuid!==saved.targetActorUuid||target.parent?.tokens.get(target.id)!==target||!pf.options?.includes(`${ID}:fear:${nonce}`)||proof?.nonce!==nonce||proof.messageId!==message.id||proof.kind!==kind||proof.userId!==user.id)throw Error('恐惧能力检定卡与本次操作者、来源或目标不符。');
  return {status:'rolled',check};
 }
 function register({socket:api,Hooks:hooks}={}){socket=api;Hooks=hooks;socket?.register?.('fear:execute',async function(payload){try{return {ok:true,value:await ownerExecute(payload,game.users.get(this.socketdata?.userId))}}catch(error){return {ok:false,error:error.message}}});}
 return {run,register};
}
