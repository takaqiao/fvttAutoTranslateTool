import {MODULE_ID} from './rules.mjs';
import {isActiveGM} from './native-context.mjs';
import {getDisruptPreyMeleeOptions,isCurrentDisruptToken} from './disrupt-prey-rules.mjs';
import {rollDisruptPreyAttack,rollDisruptPreyDamage} from './disrupt-prey-strike.mjs';
import {prepareNativeTargetDamage} from './native-target-damage.mjs';

const identity=['nonce','eventId','claimKey','actorUuid','actorId','tokenUuid','targetUuid','targetTokenUuid','targetActorUuid','targetActorId','weaponKey','itemUuid','userId','epoch','map'];
const same=(a,b)=>!!a&&!!b&&identity.every(k=>a[k]===b[k]);
const own=actor=>actor?.flags?.[MODULE_ID]?.disruptPrey??{};
const recorded=(actor,nonce)=>own(actor).reactions?.find(r=>r.nonce===nonce);
const author=message=>message?.author?.id??message?.user?.id??message?.user;
const options=message=>message?.flags?.pf2e?.context?.options??[];
const outcomes=['criticalFailure','failure','success','criticalSuccess'];
const fail=reason=>Error(`扰乱狩猎执行未确认：${reason}；不会自动重掷或重复伤害。`);
const bounded=value=>typeof value==='string'&&value.length>0&&value.length<=256;

/** Socket transport never conveys an executable weapon or damage object. The
 * owner reloads the GM's claim and persists a separate start tombstone before
 * native dice. Reconnection recovery belongs to the coordinator, proof only. */
export function createDisruptPreyExecutor({game,fromUuid=globalThis.fromUuid,Hooks=globalThis.Hooks,authorizeDamage,
 rollAttack=rollDisruptPreyAttack,rollDamage=rollDisruptPreyDamage,prepareDamage=prepareNativeTargetDamage,timeoutMs=60000}={}){
 let socket;const pipelines=new Map(),ownerStages=new Map(),actorWrites=new Map();
 const gm=()=>{if(!isActiveGM(game)||game.user!==game.users.get(game.users.activeGM.id)||!game.user.isGM)throw fail('当前客户端不是主GM');};
 const requesterGM=requester=>{if(!requester?.active||!requester.isGM||game.users.get(requester.id)!==requester||game.users.activeGM!==requester)throw fail('请求者不是当前主GM');};
 const deadline=async(promise,label)=>{
  let timer;try{return await Promise.race([Promise.resolve(promise),new Promise((_resolve,reject)=>{timer=setTimeout(()=>reject(fail(label+'超时，原调用可能仍在完成')),timeoutMs);})]);}finally{clearTimeout(timer);}
 };
 function waitFor(read,hook,match,label){
  const first=read();if(first)return Promise.resolve(first);
  if(!Hooks?.on||!Hooks.off)throw fail('缺少准确文档同步观察接口');
  return new Promise((resolve,reject)=>{
   let id,timer,done=false;
   const finish=(error,value)=>{if(done)return;done=true;clearTimeout(timer);if(id!==undefined)Hooks.off(hook,id);error?reject(error):resolve(value);};
   const check=()=>{try{const value=read();if(value)finish(null,value);}catch(error){finish(error);}};
   id=Hooks.on(hook,(...args)=>{if(match(...args))check();});timer=setTimeout(()=>finish(fail(label+'同步超时')),timeoutMs);check();
  });
 }
 const messageById=id=>{
  if(!bounded(id))throw fail('远端没有返回准确消息ID');
  return waitFor(()=>game.messages.get(id),'createChatMessage',message=>message?.id===id,'确切原生消息 '+id);
 };
 function basicClaim(actor,nonce,claimKey){
  const claim=recorded(actor,nonce);
  if(!claim)return null;
  if(claim.claimKey!==claimKey||claim.actorUuid!==actor.uuid||claim.actorId!==actor.id||!identity.filter(k=>k!=='map').every(k=>bounded(claim[k]))||!Number.isInteger(claim.map)||claim.map<0||claim.map>2||claim.targetUuid!==claim.targetTokenUuid)throw fail('持久认领身份不一致');
  const event=own(actor).events?.find(e=>e.eventId===claim.eventId);
  if(!event||event.nonce!==nonce||event.actorUuid!==claim.actorUuid||event.tokenUuid!==claim.tokenUuid||event.sourceActorUuid!==claim.targetActorUuid||event.sourceTokenUuid!==claim.targetUuid||event.userId!==claim.userId||event.weaponKey!==claim.weaponKey||event.map!==claim.map)throw fail('持久事件与认领不一致');
  return {claim,event};
 }
 async function contextFor(actor,record){
  const claim=record.claim,token=await fromUuid(claim.tokenUuid),target=await fromUuid(claim.targetUuid),user=game.users.get(claim.userId);
  if(!same(recorded(actor,claim.nonce),claim)||!isCurrentDisruptToken(token,game)||!isCurrentDisruptToken(target,game)||token.actor!==actor||target.actor.uuid!==claim.targetActorUuid||target.actor.id!==claim.targetActorId||token.parent!==target.parent||!user?.active||!actor.testUserPermission?.(user,'OWNER'))throw fail('真实拥有者、角色或Token来源已变');
  const option=getDisruptPreyMeleeOptions({game,actor,token,target}).find(o=>o.key===claim.weaponKey);
  if(!option||option.itemUuid!==claim.itemUuid||typeof option.strike.variants?.[claim.map]?.roll!=='function')throw fail('准确武器、持握、猎物或触及已变');
  return {actor,token,target,user,option:{...option,map:claim.map},claim:structuredClone(claim)};
 }
 const stageReady=(record,stage,checkId)=>{
  if(record.event.status!=='choosing'||record.event.stage!==record.claim.state)throw fail('持久事件不在可执行阶段');
  if(stage==='attack'){
   if(record.claim.state!=='claimed'||record.claim.checkId||record.claim.damageMessageId)throw fail('攻击已经开始或认领不处于初始阶段');
  }else if(record.claim.state!=='attack-rolled'||!bounded(checkId)||record.claim.checkId!==checkId||![2,3].includes(record.claim.degree)||record.claim.damageMessageId)throw fail('伤害没有确切已命中的攻击阶段');
 };
 function cardProof(message,claim,stage){
  const pf=message?.flags?.pf2e,c=pf?.context,o=options(message),damage=stage==='damage',kind=damage?'damage':'attack',r=message?.rolls?.[0];
  if(!message?.id||game.messages.get(message.id)!==message||author(message)!==claim.userId||message.actor?.uuid!==claim.actorUuid||message.speaker?.actor!==claim.actorId||message.speaker?.token!==claim.tokenUuid.split('.').at(-1)||`Scene.${message.speaker?.scene}.Token.${message.speaker?.token}`!==claim.tokenUuid||pf?.origin?.actor!==claim.actorUuid||pf.origin.uuid!==claim.itemUuid||c?.type!==kind+'-roll'||c.target?.actor!==claim.targetActorUuid||c.target.token!==claim.targetUuid||c.mapIncreases!==claim.map||!o.includes(`${MODULE_ID}:disrupt-${kind}:${claim.nonce}`)||!o.includes('item:melee')||o.includes('item:ranged')||r?._evaluated!==true||!Number.isFinite(r.total)||r.toJSON?.()?.evaluated!==true)throw fail('原生卡作者、骰子或确切来源不符');
  const degree=outcomes.indexOf(c.outcome);if(degree<0||r.options?.degreeOfSuccess!==degree)throw fail('原生成功度不一致');
  if(damage){
   if(!message.isDamageRoll||![2,3].includes(degree)||degree!==claim.degree||!o.includes(`${MODULE_ID}:bear-attack:${claim.checkId}`)||o.includes('action:reaction')||o.includes('trait:reaction')||c.sourceType!=='attack'||pf.strike?.actor!==claim.actorUuid||pf.strike.damaging!==true)throw fail('伤害卡不是该命中攻击的原生伤害');
  }else{
   const proof=message.flags?.[MODULE_ID]?.disruptPreyReaction;
   if(!message.isCheckRoll||message.rolls.length!==1||c.action!=='strike'||!o.includes('action:reaction')||!o.includes('action:disrupt-prey')||proof?.nonce!==claim.nonce||proof.claimKey!==claim.claimKey||proof.actorUuid!==claim.actorUuid||proof.weaponKey!==claim.weaponKey)throw fail('攻击卡缺少本次反应证明');
  }
  return degree;
 }
 function serialWrite(actor,operation){
  const previous=actorWrites.get(actor.uuid)??Promise.resolve(),next=previous.catch(()=>{}).then(operation);actorWrites.set(actor.uuid,next);
  next.finally(()=>{if(actorWrites.get(actor.uuid)===next)actorWrites.delete(actor.uuid);}).catch(()=>{});return next;
 }
 async function writeAttempt(context,stage,status,messageId){
  return serialWrite(context.actor,async()=>{
   const {actor,claim}=context,state=structuredClone(actor.flags?.[MODULE_ID]?.disruptPreyExecutions??{records:[]});state.records??=[];
   let attempt=state.records.find(r=>r.nonce===claim.nonce&&r.stage===stage);
   if(status==='started'){
    if(attempt)throw fail('该拥有者投骰阶段已有持久执行记录，须仅恢复已有卡');
    const current=basicClaim(actor,claim.nonce,claim.claimKey);if(!current||!same(current.claim,claim))throw fail('保存开始记录前认领已变');stageReady(current,stage,claim.checkId);
    attempt={...Object.fromEntries(identity.map(k=>[k,claim[k]])),stage,status};state.records.push(attempt);
   }else{
    if(!attempt||!same(attempt,claim)||attempt.status!=='started')throw fail('投骰开始记录已变');Object.assign(attempt,{status,...(messageId?{messageId}:{})});
   }
   await actor.update({[`flags.${MODULE_ID}.disruptPreyExecutions`]:state});
   const saved=actor.flags?.[MODULE_ID]?.disruptPreyExecutions?.records?.find(r=>r.nonce===claim.nonce&&r.stage===stage);
   if(!saved||!same(saved,claim)||saved.status!==status||messageId&&saved.messageId!==messageId)throw fail('拥有者执行记录未持久保存');
  });
 }
 async function ownerRoll(payload,requester){
  requesterGM(requester);
  if(!bounded(payload?.actorUuid)||!/^[A-Za-z0-9_-]{1,80}$/.test(payload?.nonce??'')||!bounded(payload?.claimKey)||!['attack','damage'].includes(payload?.stage))throw fail('无效拥有者请求');
  const actor=await fromUuid(payload.actorUuid);requesterGM(requester);
  if(!actor||actor.uuid!==payload.actorUuid||game.users.get(game.user?.id)!==game.user||!game.user.active||!actor.testUserPermission?.(game.user,'OWNER'))throw fail('当前客户端不是确切拥有者');
  const read=()=>{
   requesterGM(requester);const record=basicClaim(actor,payload.nonce,payload.claimKey);if(!record)return null;
   if(record.claim.userId!==game.user.id)throw fail('当前客户端不是认领中的拥有者');
   return record;
  };
  let record=await waitFor(read,'updateActor',updated=>updated?.uuid===actor.uuid,'拥有者认领');
  const key=actor.uuid+':'+payload.nonce+':'+payload.stage,fingerprint=JSON.stringify({identity:Object.fromEntries(identity.map(k=>[k,record.claim[k]])),checkId:payload.checkId??null}),existing=ownerStages.get(key);
  if(existing){if(existing.fingerprint!==fingerprint)throw fail('重复请求改变了阶段身份');return existing.promise;}
  const promise=deadline((async()=>{
   // A GM's update can arrive just after its damage RPC. Wait for that exact
   // persisted attack ID; other terminal/stale states never authorize a roll.
   if(payload.stage==='damage'&&record.claim.state==='claimed')record=await waitFor(()=>{
    const current=read();if(!current)return null;if(current.claim.state==='claimed')return null;return current;
   },'updateActor',updated=>updated?.uuid===actor.uuid,'命中攻击阶段');
   stageReady(record,payload.stage,payload.checkId);const context=await contextFor(actor,record);requesterGM(requester);
   if(context.user!==game.user)throw fail('拥有者在执行前已变');
   await writeAttempt(context,payload.stage,'started');requesterGM(requester);
   const current=basicClaim(actor,payload.nonce,payload.claimKey);if(!current||!same(current.claim,context.claim))throw fail('原生投骰前认领已变');stageReady(current,payload.stage,payload.checkId);
   try{
    const attack=payload.stage==='damage'?await messageById(context.claim.checkId):null;if(attack)cardProof(attack,context.claim,'attack');
    requesterGM(requester);
    const fresh=basicClaim(actor,payload.nonce,payload.claimKey);
    if(!fresh||!same(fresh.claim,context.claim)||game.user!==context.user||game.users.get(game.user.id)!==game.user||!game.user.active||!actor.testUserPermission?.(game.user,'OWNER'))throw fail('开始原生投骰前拥有者或认领已变');
    stageReady(fresh,payload.stage,payload.checkId);
    const message=payload.stage==='attack'?await rollAttack({...context,game}):await rollDamage({...context,game,attack,Hooks});
    cardProof(message,context.claim,payload.stage);await writeAttempt(context,payload.stage,'done',message.id);requesterGM(requester);
    if(!same(recorded(actor,payload.nonce),context.claim))throw fail('原生卡完成时认领身份已变');return {messageId:message.id};
   }catch(error){
    const attempt=actor.flags?.[MODULE_ID]?.disruptPreyExecutions?.records?.find(r=>r.nonce===payload.nonce&&r.stage===payload.stage);
    if(attempt?.status==='started')await writeAttempt(context,payload.stage,'uncertain').catch(()=>{});throw error;
   }
  })(),'拥有者'+payload.stage);ownerStages.set(key,{fingerprint,promise});return promise;
 }
 function persistent(context,state){
  gm();const record=basicClaim(context.actor,context.claim.nonce,context.claim.claimKey);
  if(!record||!same(record.claim,context.claim)||record.claim.state!==state)throw fail('GM回调未保存预期阶段 '+state);
  if(record.event.stage!==state||record.event.status!==(state==='done'?'done':'choosing'))throw fail('GM事件阶段不同步');
  Object.assign(context.claim,structuredClone(record.claim));return record.claim;
 }
 async function remote(context,stage){
  gm();const gmUser=game.user,payload={actorUuid:context.actor.uuid,nonce:context.claim.nonce,claimKey:context.claim.claimKey,stage,...(stage==='damage'?{checkId:context.claim.checkId}:{})};
  let value;if(context.user===game.user)value=await ownerRoll(payload,gmUser);
  else{
   if(!socket?.executeAsUser)throw fail('缺少拥有者原生投骰通讯');
   const response=await deadline(socket.executeAsUser('disrupt-prey:roll',context.user.id,payload),'拥有者RPC');gm();
   if(!response?.ok)throw fail(response?.error??'拥有者没有确认原生卡');value=response.value;
  }
  gm();const message=await messageById(value?.messageId);gm();cardProof(message,context.claim,stage);return message;
 }
 function receiptProof(message,claim){
  const c=message?.flags?.pf2e?.context,o=options(message),applied=message?.flags?.pf2e?.appliedDamage;
  if(!message?.id||game.messages.get(message.id)!==message||author(message)!==claim.applicationUserId||message.speaker?.actor!==claim.targetActorId||`Scene.${message.speaker?.scene}.Token.${message.speaker?.token}`!==claim.targetUuid||c?.type!=='damage-taken'||!o.includes(`${MODULE_ID}:disrupt-apply:${claim.nonce}`)||!o.includes(`${MODULE_ID}:source:${claim.damageMessageId}:0`)||o.some(v=>v.startsWith(`${MODULE_ID}:source:`)&&v!==`${MODULE_ID}:source:${claim.damageMessageId}:0`)||applied?.uuid&&applied.uuid!==claim.targetActorUuid||applied?.isReverted)throw fail('实际伤害回执的目标、作者或来源不符');
 }
 async function performStrike(args){
  gm();const {actor,token,target,user,claim}=args;
  if(!claim||!bounded(actor?.uuid)||claim.actorUuid!==actor.uuid)throw fail('缺少确切执行认领');
  const key=actor.uuid+':'+claim.nonce,fingerprint=JSON.stringify(Object.fromEntries(identity.map(k=>[k,claim[k]]))),existing=pipelines.get(key);
  if(existing){if(existing.fingerprint!==fingerprint)throw fail('重复执行改变了认领身份');return existing.promise;}
  const promise=(async()=>{
   if(await fromUuid(actor.uuid)!==actor)throw fail('执行者不是当前角色文档');gm();
   const record=basicClaim(actor,claim.nonce,claim.claimKey);if(!record||!same(record.claim,claim))throw fail('执行认领未持久保存');stageReady(record,'attack');
   const context=await contextFor(actor,record);gm();
   if(token!==context.token||target!==context.target||user!==context.user||args.option?.key!==context.claim.weaponKey||args.option.map!==context.claim.map)throw fail('执行参数不等于当前认领');
   for(const name of ['onAttack','onDamage','onApplying','onApplied'])if(typeof args[name]!=='function')throw fail('缺少持久阶段回调 '+name);
   if(typeof authorizeDamage!=='function')throw fail('缺少实际伤害的一次性许可守卫');
   const attack=await remote(context,'attack'),degree=cardProof(attack,context.claim,'attack');gm();await args.onAttack({messageId:attack.id});
   persistent(context,degree<2?'done':'attack-rolled');if(context.claim.checkId!==attack.id||context.claim.degree!==degree)throw fail('攻击回调绑定了不同检定');
   if(degree<2)return {checkId:attack.id,degree};
   const damage=await remote(context,'damage');gm();await args.onDamage({messageId:damage.id});persistent(context,'damage-rolled');if(context.claim.damageMessageId!==damage.id)throw fail('伤害回调绑定了不同卡');
   const prepared=await prepareDamage({game,message:damage,target:context.target,applicationOption:`${MODULE_ID}:disrupt-apply:${context.claim.nonce}`});persistent(context,'damage-rolled');
   await args.onApplying();persistent(context,'applying');if(context.claim.applicationUserId!==game.user.id)throw fail('实际应用GM未保存');
   const receiptCards=new Set(),applicationOption=`${MODULE_ID}:disrupt-apply:${context.claim.nonce}`;let revoke;
   if(!Hooks?.on||!Hooks.off)throw fail('缺少准确原生应用回执观察接口');
   const hook=Hooks.on('createChatMessage',message=>{if(options(message).includes(applicationOption))receiptCards.add(message);});
   try{
    revoke=await authorizeDamage({reactor:actor,actor:prepared.actor,message:damage,target:context.target,claim:context.claim,params:prepared.params});persistent(context,'applying');
    if(typeof revoke!=='function')throw fail('一次性伤害许可未返回撤销句柄');
    await deadline(prepared.actor.applyDamage(prepared.params),'原生伤害应用');gm();
    if(receiptCards.size===0)await waitFor(()=>receiptCards.size>0,'createChatMessage',message=>options(message).includes(applicationOption),'原生伤害回执');
    if(receiptCards.size!==1)throw fail('原生应用没有唯一回执');const [receipt]=receiptCards;receiptProof(receipt,context.claim);gm();
    await args.onApplied({messageId:receipt.id});persistent(context,'done');if(context.claim.receiptId!==receipt.id)throw fail('完成回调绑定了不同应用回执');
    return {checkId:attack.id,degree,damageMessageId:damage.id,receiptId:receipt.id};
   }finally{Hooks.off('createChatMessage',hook);if(typeof revoke==='function')revoke();}
  })();pipelines.set(key,{fingerprint,promise});return promise;
 }
 function register({socket:api}={}){
  if(socket&&socket!==api)throw fail('拥有者通讯已经注册');if(socket)return;socket=api;
  socket?.register('disrupt-prey:roll',async function(payload){
   try{return {ok:true,value:await ownerRoll(payload,game.users.get(this.socketdata?.userId))};}
   catch(error){return {ok:false,error:String(error.message??error)};}
  });
 }
 return {performStrike,ownerRoll,register};
}
