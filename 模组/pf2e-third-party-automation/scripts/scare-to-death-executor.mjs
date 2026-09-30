import {MODULE_ID} from './rules.mjs';
import {SerialActions} from './runtime.mjs';
import {getSourceId,isActiveGM} from './native-context.mjs';
import {SCARE_SOURCE,SCARE_TRAITS,SCARE_OUTCOMES,scareState,assertScareTokens} from './scare-to-death-rules.mjs';

const identity=['usageId','actorUuid','originUuid','targetUuid','targetActorUuid','itemUuid','userId','saveUserId','penalty','nonce'];
const same=(a,b)=>a&&b&&identity.every(k=>a[k]===b[k]);
const marker=(claim,stage)=>`${MODULE_ID}:scare:${claim.nonce}:${stage}`;
const fail=reason=>Error(`肝胆俱裂执行未确认：${reason}；不会自动重掷。`);
const validId=id=>typeof id==='string'&&id.length>0&&id.length<257;
const ownExecutions=actor=>actor.flags?.[MODULE_ID]?.scareExecutions??[];

export function validateScareCard({game,message,claim,stage}){
 const save=stage==='fortitude',pf=message?.flags?.pf2e,c=pf?.context,roll=message?.rolls?.[0],roller=save?claim.targetActorUuid:claim.actorUuid,token=save?claim.targetUuid:claim.originUuid;
 const degree=SCARE_OUTCOMES.indexOf(c?.outcome),options=c?.options??[];
 if(!['intimidation','fortitude'].includes(stage)||game.messages.get(message?.id)!==message||!message.isCheckRoll||message.rolls?.length!==1||
  (message.author?.id??message.user?.id??message.user)!==(save?claim.saveUserId:claim.userId)||message.actor?.uuid!==roller||
  message.speaker?.actor!==roller.split('.').at(-1)||`Scene.${message.speaker?.scene}.Token.${message.speaker?.token}`!==token||
  pf?.origin?.actor!==claim.actorUuid||pf.origin.uuid!==claim.itemUuid||c?.origin?.actor!==claim.actorUuid||c.origin.token!==claim.originUuid||
  c?.target?.actor!==claim.targetActorUuid||c.target.token!==claim.targetUuid||c.type!==(save?'saving-throw':'skill-check')||c.action!=='scare-to-death'||
  c.dc?.slug!==(save?'intimidation':'will')||!Number.isFinite(c.dc?.value)||!options.includes(marker(claim,stage))||!options.includes('item:trait:incapacitation')||
  roll?._evaluated!==true||roll.toJSON?.()?.evaluated!==true||!Number.isFinite(roll.total)||degree<0||roll.options?.degreeOfSuccess!==degree)throw fail('原生卡的作者、来源、目标或最终成功度不一致');
 return degree;
}

export function createScareExecutor({game,fromUuid=globalThis.fromUuid,Hooks=globalThis.Hooks,timeoutMs=60000}={}){
 let socket;const live=new Map(),writes=new SerialActions();
 const requesterGM=requester=>{if(!requester?.active||!requester.isGM||game.users.get(requester.id)!==requester||game.users.activeGM!==requester)throw fail('请求者不是当前主GM');};
 const requireGM=()=>{if(!isActiveGM(game))throw fail('不是当前主GM');requesterGM(game.user);};
 function waitFor(read,hook,match,label){
  const first=read();if(first)return Promise.resolve(first);
  if(!Hooks?.on||!Hooks.off)throw fail(label+'尚未同步');
  return new Promise((resolve,reject)=>{let id,timer,done=false;
   const finish=(error,value)=>{if(done)return;done=true;clearTimeout(timer);if(id!==undefined)Hooks.off(hook,id);error?reject(error):resolve(value);};
   const check=()=>{try{const value=read();if(value)finish(null,value);}catch(error){finish(error);}};
   id=Hooks.on(hook,(...args)=>{if(match(...args))check();});timer=setTimeout(()=>finish(fail(label+'同步超时')),timeoutMs);check();
  });
 }
 async function deadline(promise){let timer;try{return await Promise.race([promise,new Promise((_resolve,reject)=>{timer=setTimeout(()=>reject(fail('拥有者通讯超时，原检定可能仍在完成')),timeoutMs);})]);}finally{clearTimeout(timer);}}
 const findMessage=id=>{if(!validId(id))throw fail('没有准确消息ID');return waitFor(()=>game.messages.get(id),'createChatMessage',m=>m?.id===id,'原生消息');};
 function claimFrom(message,payload){
  const claim=scareState(message).claim;if(!claim)return null;
  if(!identity.filter(k=>k!=='penalty').every(k=>validId(claim[k]))||claim.usageId!==message.id||claim.nonce!==payload.nonce||![0,-4].includes(claim.penalty))throw fail('本次使用的持久认领身份无效');
  return claim;
 }
 async function contextFor(claim,stage){
  const [actor,origin,target,item]=await Promise.all([fromUuid(claim.actorUuid),fromUuid(claim.originUuid),fromUuid(claim.targetUuid),fromUuid(claim.itemUuid)]);
  if(!actor||origin?.actor!==actor||target?.actor?.uuid!==claim.targetActorUuid||item?.actor!==actor||getSourceId(item)!==SCARE_SOURCE||actor.items.get(item.id)!==item)throw fail('来源角色、Token或真实专长已改变');
  assertScareTokens(origin,target);
  const roller=stage==='fortitude'?target.actor:actor,user=game.users.get(stage==='fortitude'?claim.saveUserId:claim.userId);
  if(!user?.active||!roller.testUserPermission?.(user,'OWNER'))throw fail('原拥有者已离线或无权操作');
  return {actor,origin,target,item,roller,user,claim:structuredClone(claim)};
 }
 function stageReady(claim,stage){
  if(claim.state!==stage+'-ready')throw fail('检定已经开始或不处于可执行阶段');
  if(stage==='fortitude'&&!validId(claim.checkId))throw fail('没有确切大成功威吓卡');
 }
 async function ownerRoll(payload,requester){
  requesterGM(requester);
  if(!validId(payload?.usageId)||!/^[A-Za-z0-9_-]{1,80}$/.test(payload?.nonce??'')||!['intimidation','fortitude'].includes(payload?.stage))throw fail('无效检定请求');
  const usage=await findMessage(payload.usageId),read=()=>{requesterGM(requester);return claimFrom(usage,payload);};
  let claim=await waitFor(read,'updateChatMessage',m=>m?.id===usage.id,'本次使用认领');
  // Socket messages and document updates can reach the owner in either order.
  if(payload.stage==='fortitude'&&['intimidation-ready','intimidation-done'].includes(claim.state))claim=await waitFor(()=>{const current=read();return current&&!['intimidation-ready','intimidation-done'].includes(current.state)?current:null;},'updateChatMessage',m=>m?.id===usage.id,'强韧阶段');
  const context=await contextFor(claim,payload.stage);requesterGM(requester);
  if(game.user!==context.user||game.users.get(game.user.id)!==game.user)throw fail('本客户端不是该阶段的原拥有者');
  const key=claim.usageId+':'+payload.stage,fingerprint=JSON.stringify(identity.map(k=>claim[k])),existing=live.get(key);
  if(existing){if(existing.fingerprint!==fingerprint)throw fail('重复请求改变了认领');return existing.promise;}
  const promise=(async()=>{
   const {roller}=context;
   const previous=ownExecutions(roller).find(r=>r.usageId===usage.id&&r.stage===payload.stage);
   if(previous){
    if(previous.fingerprint!==fingerprint||previous.status!=='done'||!previous.messageId)throw fail('已有开始记录，不能重掷');
    const card=await findMessage(previous.messageId);validateScareCard({game,message:card,claim,stage:payload.stage});return {messageId:card.id};
   }
   stageReady(read(),payload.stage);
   if(payload.stage==='fortitude'){
    const card=await findMessage(claim.checkId);if(validateScareCard({game,message:card,claim,stage:'intimidation'})!==3)throw fail('威吓并非大成功');
   }
   const write=async(status,messageId)=>writes.run(roller.uuid,async()=>{
    requesterGM(requester);const records=structuredClone(ownExecutions(roller)),index=records.findIndex(r=>r.usageId===usage.id&&r.stage===payload.stage);
    if(status==='started'&&index>=0||status!=='started'&&(index<0||records[index].status!=='started'||records[index].fingerprint!==fingerprint))throw fail('执行记录已变');
    const record={usageId:usage.id,stage:payload.stage,fingerprint,status,...messageId?{messageId}:{}};
    if(index>=0)records[index]=record;else records.push(record);
    await roller.update({[`flags.${MODULE_ID}.scareExecutions`]:records});
    const saved=ownExecutions(roller).find(r=>r.usageId===usage.id&&r.stage===payload.stage);if(saved?.fingerprint!==fingerprint||saved.status!==status||messageId&&saved.messageId!==messageId)throw fail('执行记录未持久保存');
   });
   await write('started');
   try{
    requesterGM(requester);if(!same(read(),claim))throw fail('投骰前认领已变');stageReady(read(),payload.stage);await contextFor(claim,payload.stage);requesterGM(requester);
    const save=payload.stage==='fortitude',statistic=roller.getStatistic?.(save?'fortitude':'intimidation');if(!statistic?.roll)throw fail('缺少原生Statistic');
    const traits=[...new Set([...context.item.system.traits.value,...SCARE_TRAITS])],modifiers=[];
    if(!save&&claim.penalty){if(!game.pf2e.Modifier)throw fail('缺少原生情境修正');modifiers.push(new game.pf2e.Modifier({slug:'scare-to-death-language',label:'肝胆俱裂：无法听见或理解本次语言',modifier:claim.penalty,type:'circumstance'}));}
    let card;
    await statistic.roll({token:save?context.target:context.origin,...save?{origin:context.actor}:{target:context.target.actor},item:context.item,action:'scare-to-death',dc:{slug:save?'intimidation':'will'},traits,modifiers,
     extraRollOptions:['action:scare-to-death',marker(claim,payload.stage),...SCARE_TRAITS.map(t=>`item:trait:${t}`)],skipDialog:true,createMessage:true,callback:(_roll,_outcome,message)=>{card=message;}});
    requesterGM(requester);if(!same(read(),claim))throw fail('原生结果返回时认领已变');
    validateScareCard({game,message:card,claim,stage:payload.stage});await write('done',card.id);return {messageId:card.id};
   }catch(error){await write('uncertain').catch(()=>{});throw error;}
  })();live.set(key,{fingerprint,promise});promise.finally(()=>{if(live.get(key)?.promise===promise)live.delete(key);}).catch(()=>{});return promise;
 }
 async function roll(claim,stage){
  requireGM();const usage=await findMessage(claim.usageId),persisted=claimFrom(usage,{nonce:claim.nonce});if(!same(persisted,claim))throw fail('GM认领未持久保存');stageReady(persisted,stage);
  const context=await contextFor(claim,stage),payload={usageId:claim.usageId,nonce:claim.nonce,stage};let result;
  if(context.user===game.user)result=await ownerRoll(payload,game.user);
  else{if(!socket?.executeAsUser)throw fail('缺少原拥有者通讯');const response=await deadline(socket.executeAsUser('scare-to-death:roll',context.user.id,payload));requireGM();if(!response?.ok)throw fail(response?.error??'拥有者没有确认结果');result=response.value;}
  const card=await findMessage(result?.messageId);requireGM();if(!same(claimFrom(usage,{nonce:claim.nonce}),claim))throw fail('GM收到结果时认领已变');validateScareCard({game,message:card,claim,stage});return card;
 }
 function register({socket:api}={}){
  if(socket){if(socket!==api)throw fail('通讯重复注册');return;}socket=api;
  socket?.register('scare-to-death:roll',async function(payload){try{return {ok:true,value:await ownerRoll(payload,game.users.get(this.socketdata?.userId))};}catch(error){return {ok:false,error:String(error.message??error)};}});
 }
 return {ownerRoll,roll,register};
}
