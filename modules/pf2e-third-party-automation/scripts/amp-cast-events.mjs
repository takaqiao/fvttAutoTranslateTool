import {MODULE_ID} from './rules.mjs';
import {SerialActions} from './runtime.mjs';
import {reactionEpoch} from './reaction-budget.mjs';

const instances=new WeakMap();
const source=item=>item?.sourceId??item?._stats?.compendiumSource??item?.flags?.core?.sourceId??null;
const values=collection=>Array.from(collection?.values?.()??collection??[]);
const id=()=>globalThis.foundry?.utils?.randomID?.(32)??globalThis.crypto.randomUUID();
const gm=game=>!!game.user?.id&&game.user.id===game.users.activeGM?.id;
const author=message=>message?.author?.id??message?.user?.id??message?.user;
const copy=data=>globalThis.foundry?.utils?.deepClone?.(data)??structuredClone(data);
const asError=error=>error instanceof Error?error:Error(String(error));

/** One resource queue per game, shared by all feature providers. */
export function getNativeCastEvents(options={}){
 let instance=instances.get(options.game);
 if(!instance){instance=createNativeCastEvents(options);instances.set(options.game,instance);}
 return instance;
}

export function createNativeCastEvents({game,fromUuid=globalThis.fromUuid,messageTimeoutMs=15000}={}){
 const castMiddlewares=new Set(),actorUpdateMiddlewares=new Set();
 const invocationAdapters=new Map(),enrollments=new Map(),slotRequests=new Map();
 const queue=new SerialActions(),localCasts=new SerialActions(),matchers=new Set(),activityMatchers=new Set(),actorMatchers=new Set(),consumePolicies=new Set(),paidCastPolicies=new Set(),captures=new Map(),scopes=new Map(),messageInvocations=new WeakMap();
 // A local capability, never serialized or accepted from a socket payload.
 const nativeCapability=Object.freeze({});let socket,installed=false,focusCall=null,slotCall=null,slotHookAvailable=false;
 const focusRequests=new Map();
 const matches=item=>[...matchers].some(match=>match(item));
 const managed=actor=>[...actorMatchers].some(match=>match(actor))||values(actor?.items).some(matches);
 const captureData=item=>Object.fromEntries([...captures].map(([key,capture])=>[key,capture(item)]).filter(([,value])=>value!==undefined));
 const sameInput=(a,b)=>['actorUuid','itemUuid','sourceId','entryUuid','rank','slotId','focusPoints','overlayIds'].every(key=>JSON.stringify(a[key]??null)===JSON.stringify(b[key]??null));
 const ledger=actor=>actor.flags?.[MODULE_ID]?.nativeCasts??[];
 const save=(actor,receipts)=>{if(!gm(game))throw Error('主GM已切换，停止旧客户端的施法资源操作。');return actor.update({[`flags.${MODULE_ID}.nativeCasts`]:receipts});};
 const owner=(actor,user)=>{if(!gm(game)||!user||!actor?.testUserPermission?.(user,'OWNER'))throw Error('施法资源必须由主GM验证角色所有者后结算。');};
 const equal=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
 function invocationGM(invocation){if(!invocation||!gm(game)||invocation.gmId!==game.user.id)throw Error('原生施法调用的主GM已改变。');}
 function liveEnrollment(scope){
  const item=scope.item,actor=item.actor,original=item.original??item;
  if(enrollments.get(scope.castNonce)!==scope||scope.closed||scope.user!==game.user||scope.user.active!==true||game.users.activeGM?.id!==scope.invocation.gmId||game.actors?.get(actor.id)!==actor||actor.items.get(original.id)!==original||actor.items.get(scope.entry.id)!==scope.entry||!actor.testUserPermission(scope.user,'OWNER')||!sameInput(scope.input,input(item,scope.options))||scope.options.messageMode!=='public')throw Error('本次原生施法的调用能力已失效。');
  const token=sourceToken(actor);if(token.tokenUuid!==scope.tokenContext.tokenUuid||token.token!==scope.tokenContext.token)throw Error('本次施法的准确来源Token已改变。');
 }
 function invocationProof(payload,sender){
  const scope=enrollments.get(payload?.id);
  if(!scope||sender!==game.users.activeGM?.id||sender!==scope.invocation.gmId||payload.userId!==scope.user.id||!sameInput(scope.input,payload)||payload.messageMode!==scope.input.messageMode||!equal(scope.invocation,payload.invocation)||payload.nativeCastScope?.castNonce!==scope.castNonce||payload.nativeCastScope?.tokenUuid!==scope.tokenContext.tokenUuid)return false;
  if(Object.hasOwn(payload,'completedWorldTime')&&(scope.invocation.data.captureCompletionTime!==true||!Number.isFinite(scope.completedWorldTime)||payload.completedWorldTime!==scope.completedWorldTime))return false;
  try{liveEnrollment(scope);return true;}catch{return false;}
 }
 async function verifyInvocation(payload,user){
  invocationGM(payload.invocation);
  if(user?.active!==true||!invocationAdapters.has(payload.invocation.kind))throw Error('原生施法适配或操作者无效。');
  const request={...copy(payload),userId:user.id};
  const proof=user.id===game.user.id?invocationProof(request,game.user.id):await socket?.executeAsUser('native-cast-invocation-proof',user.id,request);
  invocationGM(payload.invocation);if(proof!==true)throw Error('原操作者没有本次真实施法调用，未授权支付。');
 }
 function input(item,{rank=item.rank,slotId}={}){
  return {actorUuid:item.actor.uuid,itemUuid:item.uuid,sourceId:source(item),entryUuid:item.spellcasting?.uuid??item.actor.items.get(item.system.location?.value)?.uuid,
   rank,slotId:slotId??null,focusPoints:item.system.cast?.focusPoints??0,overlayIds:values(item.appliedOverlays)};
 }
 async function resolve(payload,user){
  const actor=await fromUuid(payload?.actorUuid);owner(actor,user);
  let item=await fromUuid(payload.itemUuid);const entry=await fromUuid(payload.entryUuid);
  if(item?.type!=='spell'||item.actor?.uuid!==actor.uuid||source(item)!==payload.sourceId||entry?.type!=='spellcastingEntry'||entry.actor?.uuid!==actor.uuid||item.system.location?.value!==entry.id)throw Error('施法者、法术来源或施法条目不匹配。');
  if(!matches(item)&&!managed(actor)&&!invocationAdapters.has(payload.invocation?.kind))throw Error('此法术不属于已启用的施法资源适配。');
  if(!Number.isInteger(payload.rank)||payload.rank<1||payload.rank>10||payload.slotId!==null&&(!Number.isInteger(payload.slotId)||payload.slotId<0))throw Error('实际施法环级或法术位编号无效。');
  if(payload.invocation&&(payload.messageMode!=='public'||payload.invocation.data?.messageMode!=='public'))throw Error('本次施法调用没有明确的公开消息模式。');
  if(payload.overlayIds?.length||item.rank!==payload.rank)item=item.loadVariant?.({castRank:payload.rank,overlayIds:payload.overlayIds??[]})??item;
  if((item.system.cast?.focusPoints??0)!==payload.focusPoints)throw Error('施法增幅配置已改变，请按当前配置重新使用法术。');
  return {actor,item,entry};
 }
 async function pay(payload,user,{activity=null}={}){
  const actor=await fromUuid(payload?.actorUuid);owner(actor,user);
  return queue.run(actor.uuid,async()=>{
   const {item,entry}=await resolve(payload,user);
   if(payload.invocation)await verifyInvocation(payload,user);
   const existing=ledger(actor).find(r=>r.id===payload.id);
   if(existing){
    if(existing.itemUuid!==item.uuid||existing.userId!==user.id||!sameInput(existing,payload)||JSON.stringify(existing.activity??null)!==JSON.stringify(activity)||JSON.stringify(existing.nativeCastScope??null)!==JSON.stringify(payload.nativeCastScope??null)||!equal(existing.invocation??null,payload.invocation??null))throw Error('支付回执参数不匹配。');
    if(!['paid','used'].includes(existing.state))throw Error('原生支付结果未能确认，不会重复扣款。');
    return copy(existing);
   }
   const adapter=payload.invocation?invocationAdapters.get(payload.invocation.kind):null;
   if(adapter){
    const context={actor,item,entry,user,payload:copy(payload),castNonce:payload.id,invocation:copy(payload.invocation)};
    if(await adapter.validate(context)!==true)throw Error('本次施法适配未通过主GM验证。');invocationGM(payload.invocation);owner(actor,user);
   }
   if(payload.nativeCastScope){
    const scope=payload.nativeCastScope,token=scope.tokenUuid?await fromUuid(scope.tokenUuid):null;
    if(scope.castNonce!==payload.id||typeof scope.castNonce!=='string'||scope.tokenUuid!==null&&(!token||token.actor?.uuid!==actor.uuid||token.documentName!=='Token'))throw Error('原生施法的来源Token或调用标识不匹配。');
   }
   const time=item.system.time?.value,sourceAction=payload.nativeCastScope?{type:time==='reaction'?'reaction':time==='free'?'free':/^[123]$/.test(time??'')?'action':'other',value:time==='reaction'?1:/^[123]$/.test(time??'')?Number(time):null,epoch:reactionEpoch(actor,game)}:null;
   const receipt={...copy(payload),activity:copy(activity),userId:user.id,state:'claiming',createdAt:game.time?.worldTime??0,...(sourceAction?{sourceAction}:{})};
   // Uncertain or not-yet-bound native payments must survive any number of
   // later casts. Used receipts can be bounded because their cards hold proof;
   // a missing old proof is rejected, never treated as a fresh payment.
   const rejectedNative=r=>r.state==='rejected'&&!/^(chat|activity):/.test(r.id);
   const claimed=await save(actor,[...ledger(actor).filter(r=>r.state!=='used'&&!rejectedNative(r)),...ledger(actor).filter(rejectedNative).slice(-99),...ledger(actor).filter(r=>r.state==='used').slice(-199),receipt]);
   if(payload.invocation){invocationGM(payload.invocation);if(claimed!==actor||!equal(ledger(actor).find(r=>r.id===receipt.id),receipt))throw Error('本次原生施法认领未能持久保存，未开始消费。');}
   let enteredNative=false;
   try{
    // PF2e's own consume covers focus, prepared, spontaneous and innate usage.
    // Its cast method explicitly exempts at-will spells; preserve that contract.
    let focusRequest,slotRequest;
    const context={actor,item,entry,user,payload:copy(payload),castNonce:payload.id,invocation:copy(payload.invocation??null),expectFocusCommit:({before,cost,changes})=>{
     if(enteredNative||focusRequest||!Number.isInteger(before)||!Number.isInteger(cost)||cost<1||before<cost||cost!==payload.focusPoints||typeof changes!=='function')throw Error('原生聚能提交配置无效或重复。');
     focusRequest={actor,item,entry,before,cost,changes,castNonce:payload.id,captured:false};
    },expectSlotCommit:({before,cost=1,changes})=>{
     if(!payload.invocation||enteredNative||slotRequest||!slotHookAvailable||entry.isSpontaneous!==true||!Number.isInteger(payload.rank)||payload.rank<1||payload.rank>3||payload.slotId!==null||payload.focusPoints!==0||item.atWill||item.isCantrip||!Number.isInteger(before)||before<1||cost!==1||typeof changes!=='function')throw Error('原生法术位提交配置无效或重复。');
     slotRequest={actor,item,entry,before,cost,rank:payload.rank,changes,castNonce:payload.id,userId:user.id,gmId:payload.invocation.gmId,captured:false,witness:false,returned:false};
    }};
    const native=async()=>{owner(actor,user);if(payload.invocation){invocationGM(payload.invocation);if(user.active!==true||actor.items.get((item.original??item).id)!==(item.original??item)||actor.items.get(entry.id)!==entry||source(item)!==payload.sourceId||!slotRequest)throw Error('本次原生施法的来源或法术位提交见证已失效。');}if(enteredNative)throw Error('同一施法只能调用一次原生资源消费，拒绝重复支付。');enteredNative=true;return item.atWill||await entry.consume(item,payload.rank,payload.slotId??undefined,nativeCapability);};
    const consume=[...consumePolicies,...(adapter?[adapter.consumePolicy]:[])].reduceRight((next,policy)=>()=>policy(context,next),native);
    // The request is exposed only while this exact native entry is invoked.
    // A policy cannot forge the private capability used by its consume call.
    const invoke=async()=>{focusRequests.set(actor.uuid,()=>focusRequest);slotRequests.set(actor.uuid,()=>slotRequest);try{return await consume();}finally{focusRequests.delete(actor.uuid);slotRequests.delete(actor.uuid);}};
    const paid=await invoke();
    if(paid&&!enteredNative)throw Error('施法支付策略未执行原生资源消费。');
    if(paid&&focusRequest&&!focusRequest.captured)throw Error('原生聚能写入未能绑定本次同步消费；支付不确定，不会推断或重试。');
    if(payload.invocation)invocationGM(payload.invocation);
    if(paid&&slotRequest){
     if(!slotRequest.captured||!slotRequest.witness||!slotRequest.returned||!equal(entry.flags?.[MODULE_ID]?.nativeSlotCommit,slotRequest.proof)||entry.system.slots[`slot${slotRequest.rank}`].value!==slotRequest.before-slotRequest.cost)throw Error('原生法术位写入未获得同步调用、实际更新和持久回执的共同证明。');
     receipt.slotCommit=copy(slotRequest.proof);
    }
    receipt.state=paid?(matches(item)||activity||payload.nativeCastScope?'paid':'used'):'rejected';
    await save(actor,ledger(actor).map(r=>r.id===receipt.id?receipt:r));
    owner(actor,user);
    if(!paid)throw Error('法术资源不足，无法支付本次施法。');
    return copy(receipt);
   }catch(error){
    if(receipt.state==='claiming'&&gm(game)){receipt.state=enteredNative?'uncertain':'rejected';await save(actor,ledger(actor).map(r=>r.id===receipt.id?receipt:r));}
    if(!enteredNative)error.code='NATIVE_CAST_REJECTED';
    throw error;
   }
  });
 }
 async function remotePay(payload){
  if(gm(game))return pay(payload,game.user);
  if(!socket||!game.users.activeGM)throw Error('自动施法支付需要在线主GM。');
  const response=await socket.executeAsUser('native-cast-pay',game.users.activeGM.id,payload);
  if(!response?.ok){const error=Error(response?.error??'原生施法支付失败。');error.code=response?.code;throw error;}return response.value;
 }
 function sourceToken(actor){
  const sceneId=game.scenes?.current?.id??globalThis.canvas?.scene?.id;
  const candidates=actor.isToken?[actor.token]:values(actor.getActiveTokens?.(true,true));
  const tokens=[...new Map(candidates.map(token=>token?.document??token).filter(token=>token?.uuid&&token.actor?.uuid===actor.uuid&&(!sceneId||token.parent?.id===sceneId)).map(token=>[token.uuid,token])).values()];
  return tokens.length===1?{token:tokens[0],tokenUuid:tokens[0].uuid}:{token:null,tokenUuid:null,unsupportedReason:tokens.length?'ambiguous-source-token':'no-source-token'};
 }
 async function finishPaidCast(payload,user){
  const actor=await fromUuid(payload?.input?.actorUuid);owner(actor,user);
  return queue.run(actor.uuid,async()=>{
   owner(actor,user);
   const receipt=ledger(actor).find(r=>r.id===payload.id);
   if(receipt?.invocation)invocationGM(receipt.invocation);
   if(!receipt||receipt.userId!==user.id||!sameInput(receipt,payload.input)||receipt.nativeCastScope?.castNonce!==receipt.id||receipt.activity||receipt.messageId||!['disrupted','uncertain','used'].includes(payload.state))throw Error('无法确认本次原生施法中断的支付来源。');
   if(receipt.state===payload.state)return copy(receipt);
   if(!['paid','used'].includes(receipt.state))throw Error('原生施法已结束或支付结果不确定，不会重复结算。');
   const terminal={...receipt,state:payload.state,castOutcome:{reason:String(payload.reason??'').slice(0,500),eventId:typeof payload.eventId==='string'?payload.eventId.slice(0,200):null}};
   // No spell card replaces an interrupted cast's proof. Keep these receipts
   // until a verified durable terminal message can make pruning safe.
   await save(actor,ledger(actor).map(r=>r.id===receipt.id?terminal:r));
   return copy(terminal);
  });
 }
 async function remoteFinishPaidCast(scope,state,detail={}){
  const payload={id:scope.receipt.id,input:copy(scope.input),state,reason:detail.reason,eventId:detail.eventId};
  if(gm(game))return finishPaidCast(payload,scope.user);
  if(!socket||!game.users.activeGM)throw Error('原生施法中断结算需要在线主GM。');
  const response=await socket.executeAsUser('native-cast-outcome',game.users.activeGM.id,payload);
  if(!response?.ok)throw Error(response?.error??'原生施法中断结算未能确认。');return response.value;
 }
 async function bindInvocation(payload,user){
  const actor=await fromUuid(payload.input?.actorUuid);owner(actor,user);
  return queue.run(actor.uuid,async()=>{
   owner(actor,user);const prior=ledger(actor).find(r=>r.id===payload.id);
   if(!prior?.invocation||!prior.slotCommit||prior.userId!==user.id||!sameInput(prior,payload.input)||!equal(prior.invocation,payload.invocation)||!['paid','used'].includes(prior.state))throw Error('本次调用没有已证实的法术位支付，不能绑定消息。');
   const timed=prior.invocation.data.captureCompletionTime===true;
   const timeProof=timed?{completedWorldTime:payload.completedWorldTime}:{};
   if(timed&&(!Number.isFinite(payload.completedWorldTime)||payload.completedWorldTime!==game.time?.worldTime||prior.state==='used'&&prior.completedWorldTime!==payload.completedWorldTime))throw Error('本次施法的完成时间未能确认。');
   await verifyInvocation({...prior,...timeProof},user);owner(actor,user);
   if(timed&&payload.completedWorldTime!==game.time?.worldTime)throw Error('确认施法完成时间期间游戏时钟已改变，请核对持续时长。');
   const message=await fromUuid(`ChatMessage.${payload.messageId}`),proof=message?.flags?.[MODULE_ID]?.nativeCast;
   invocationGM(prior.invocation);
   if(!message?.id||game.messages.get(message.id)!==message||author(message)!==user.id||message.rolls?.length||message.isRoll||message.blind!==false||!Array.isArray(message.whisper)||message.whisper.length||proof?.id!==prior.id||proof.actorUuid!==actor.uuid||proof.itemUuid!==prior.itemUuid||proof.userId!==user.id||message.flags?.pf2e?.origin?.uuid!==prior.itemUuid||message.flags.pf2e.origin.actor!==actor.uuid||!sameInput(message.flags?.[MODULE_ID]?.nativeCastInput??{},prior)||message.flags[MODULE_ID].nativeCastInput.messageMode!==prior.messageMode||prior.messageId&&prior.messageId!==message.id)throw Error('原生施法消息与本次调用的支付来源不一致。');
   const bound={...prior,...timeProof,state:'used',messageId:message.id};
   await save(actor,ledger(actor).map(r=>r.id===prior.id?bound:r));invocationGM(prior.invocation);
   const persisted=ledger(actor).find(r=>r.id===prior.id);
   if(!equal(persisted,bound))throw Error('原生施法原卡绑定未持久保存。');return copy(persisted);
  });
 }
 async function completeInvocation(scope,nativeResult){
  liveEnrollment(scope);
  if(!scope.receipt?.slotCommit||!scope.finalMessage||scope.error||scope.messageError)throw Error('本次施法没有完整的原生付款和原卡结果。');
  const timeProof={};
  if(scope.invocation.data.captureCompletionTime===true){
   const value=game.time?.worldTime;
   if(!Number.isFinite(value))throw Error('本次施法的原生完成时间不可用，请手工核对持续时长。');
   scope.completedWorldTime=value;timeProof.completedWorldTime=value;
  }
  const payload={id:scope.castNonce,input:copy(scope.input),invocation:copy(scope.invocation),messageId:scope.finalMessage.id,...timeProof};
  const response=gm(game)?{ok:true,value:await bindInvocation(payload,scope.user)}:await socket?.executeAsUser('native-cast-invocation-bind',scope.invocation.gmId,payload);
  liveEnrollment(scope);if(!response?.ok)throw Error(response?.error??'本次施法原卡绑定未获主GM确认。');scope.receipt=response.value;
  return {status:'completed',castNonce:scope.castNonce,input:copy(scope.input),receipt:copy(scope.receipt),message:scope.finalMessage,nativeResult,...timeProof};
 }
 async function uncertainCast(scope,error){
  const failure=asError(error);scope.error=failure;
  try{scope.receipt=await remoteFinishPaidCast(scope,'uncertain',{reason:failure.message});}catch(settleError){if(Object.isExtensible(failure))failure.cause??=settleError;}
  return failure;
 }
 async function applyPaidCastPolicies(scope){
  // This function runs only from our live cast/consume wrappers, never from
  // actualCast chat options or a provider-supplied payment proof.
  scope.policyPending=true;
  try{
   if(scope.error)throw scope.error;
   let tokenContext=scope.tokenContext;
   const sceneId=game.scenes?.current?.id??globalThis.canvas?.scene?.id;
   if(tokenContext.token&&(await fromUuid(tokenContext.tokenUuid)!==tokenContext.token||tokenContext.token.actor?.uuid!==scope.input.actorUuid||sceneId&&tokenContext.token.parent?.id!==sceneId))tokenContext={token:null,tokenUuid:null,unsupportedReason:'source-token-changed'};
   const context={actor:scope.item.actor,item:scope.item,entry:scope.entry,user:scope.user,castNonce:scope.castNonce,input:copy(scope.input),targetUuids:[...scope.targets],receipt:copy(scope.receipt),...tokenContext};
   for(const policy of scope.policies){
    const decision=await policy(context);
    if(scope.error)throw scope.error;
    if(decision?.disrupted===true){
     scope.receipt=await remoteFinishPaidCast(scope,'disrupted',decision);
     scope.terminal={disrupted:true,castNonce:scope.castNonce,input:copy(scope.input),receipt:copy(scope.receipt),reason:decision.reason??null,eventId:decision.eventId??null};
     return false;
    }
   }
   if(!matches(scope.item)&&!scope.invocation)scope.receipt=await remoteFinishPaidCast(scope,'used');
   return true;
  }catch(error){
   await uncertainCast(scope,error);
   // Native cast uses this boolean to suppress its body. Surface the error
   // through our outer cast promise, including Toolbelt's detached native call.
   return false;
  }finally{scope.policyPending=false;}
 }
 async function consumeForCast(scope,request){
  try{
   const receipt=await remotePay({...request,id:scope?.castNonce??id(),...(scope?.policies.length||scope?.invocation?{nativeCastScope:{castNonce:scope.castNonce,tokenUuid:scope.tokenContext.tokenUuid}}:{}),...(scope?.invocation?{invocation:copy(scope.invocation)}:{})});
   if(scope)scope.receipt=receipt;
   if(scope?.invocation)liveEnrollment(scope);
   if(scope?.error&&!scope.policies.length)return false;
   return scope?.policies.length?applyPaidCastPolicies(scope):true;
  }catch(error){if(error?.code==='NATIVE_CAST_REJECTED'||/资源不足|无法支付/.test(error?.message)){globalThis.ui?.notifications?.warn?.(error.message);return false;}if(scope){scope.error=asError(error);return false;}throw error;}
 }
 function captureUsage(item,{options={}}={}){
  const current=options.actualCast&&scopes.get(item.uuid),scope=current&&(!current.invocation||current.item===item)&&!current.policyPending&&!current.terminal&&!current.error?current:null;
  if(!matches(item)&&!scope)return null;
  if(scope?.invocation)liveEnrollment(scope);
  if(scope?.receipt&&options.actualCast===true)messageInvocations.set(options,{scope,item,actor:item.actor,nonce:scope.castNonce});
  const captured=scope?.captured??captureData(item);
  return {...copy(captured),nativeCastInput:copy(scope?.input??input(item,{rank:options.data?.castRank??item.rank})),
   ...(scope?{usageInput:{targetUuids:[...scope.targets]},...(scope.receipt?{nativeCast:{id:scope.receipt.id,actorUuid:item.actor.uuid,itemUuid:item.uuid,userId:scope.receipt.userId}}:{})}:{})};
 }
 /** The shared toMessage wrapper reports only its final created document or
  * failure. Its captured nonce, not item UUID alone, binds this late signal.
  * true lets that wrapper suppress a detached rejection; the cast throws once. */
 function captureMessageOutcome(item,options={},outcome={}){
  const {message,error,castNonce}=outcome,failed=Object.hasOwn(outcome,'error'),invocation=messageInvocations.get(options);
  if(!invocation||invocation.item!==item||invocation.actor!==item.actor||invocation.nonce!==castNonce)return false;
  const {scope}=invocation;
  // A detached native body may reject after the watchdog has retired its cast.
  // Consume only that exact captured invocation's failure, without consulting
  // or settling any newer same-item scope. Weak keys retain no completed task.
  if(scopes.get(item.uuid)!==scope||scope.terminal||scope.error){messageInvocations.delete(options);return failed;}
  if(!scope.receipt||scope.policyPending||options.actualCast!==true)return false;
  messageInvocations.delete(options);
  if(failed)scope.messageError??=asError(error);
  else{
   const proof=message?.flags?.[MODULE_ID]?.nativeCast,pf=message?.flags?.pf2e;
   if(!message?.id||game.messages.get(message.id)!==message||message.rolls?.length||author(message)!==scope.user.id||proof?.id!==castNonce||proof.actorUuid!==scope.input.actorUuid||proof.itemUuid!==scope.input.itemUuid||proof.userId!==scope.user.id||pf?.origin?.uuid!==scope.input.itemUuid||pf.origin.actor&&pf.origin.actor!==scope.input.actorUuid||scope.finalMessage&&scope.finalMessage!==message)scope.messageError??=Error('原生施法最终消息的来源或唯一性无法确认。');
   else scope.finalMessage=message;
  }
  scope.resolveCapture();return true;
 }
 function trackConsume(scope,task){
  scope.consumeTask=task;task.then(scope.resolveConsume,scope.resolveConsume);return task;
 }
 async function finishMessage(scope,nativeTask){
  let timer;
  try{
   // Start only after consume and paid policies finish. A normal reaction
   // decision is never timed out by this final-message watchdog.
   const limit=Number.isFinite(messageTimeoutMs)&&messageTimeoutMs>0?messageTimeoutMs:15000;
   const [result]=await Promise.race([Promise.all([nativeTask,scope.capturePromise]),new Promise((_resolve,reject)=>{timer=setTimeout(()=>reject(Error('原生施法最终消息等待超时，支付结果保留且不会重试。')),limit);})]);
   if(scope.messageError)throw scope.messageError;return result;
  }finally{clearTimeout(timer);}
 }
 async function activitySource(actor,message,user){
  if(author(message)!==user.id||game.messages.get(message?.id)!==message||message.isRoll||message.rolls?.length)throw Error('支付需要原作者的真实活动消息。');
  const pf=message.flags?.pf2e??{},activity=pf.origin?.uuid?await fromUuid(pf.origin.uuid):actor.items.get(pf.context?.item);
  if(!activity||!['feat','action'].includes(activity.type)||activity.actor?.uuid!==actor.uuid||![...activityMatchers].some(match=>match(activity)))throw Error('此活动没有已注册的原生施法流程。');
  return {messageId:message.id,itemUuid:activity.uuid,sourceId:source(activity)};
 }
 /** A GM-side activity workflow pays first, then publishes one canonical spell card. */
 async function payForActivity({actor,item,message,user,rank=item.rank,slotId=null}){
  owner(actor,user);const activity=await activitySource(actor,message,user);
  if(item?.type!=='spell'||item.actor?.uuid!==actor.uuid)throw Error('组合活动法术不属于该角色。');
  const request=input(item,{rank,slotId}),receiptId='activity:'+message.id,proof=message.flags?.[MODULE_ID]?.activityPayment;
  if(proof&&(!sameInput(proof,request)||proof.id!==receiptId||!ledger(actor).some(r=>r.id===receiptId)))throw Error('组合活动原生支付回执已失效或参数不匹配，不会再次扣费。');
  const receipt=await pay({...request,id:receiptId},user,{activity});
  owner(actor,user);
  if(!proof)await message.update({[`flags.${MODULE_ID}.activityPayment`]:{...copy(request),id:receiptId}});
  return {id:receipt.id,receipt,flags:{nativeCast:{id:receipt.id,actorUuid:actor.uuid,itemUuid:item.uuid,userId:user.id,activityMessageId:message.id},nativeCastInput:copy(request)}};
 }
 async function finishActivityWithoutSpell({actor,message,user}){
  owner(actor,user);const activity=await activitySource(actor,message,user);
  return queue.run(actor.uuid,async()=>{
   owner(actor,user);
   const receipt=ledger(actor).find(r=>r.id==='activity:'+message.id);
   if(!receipt||receipt.userId!==user.id||JSON.stringify(receipt.activity)!==JSON.stringify(activity)||!['paid','used'].includes(receipt.state))throw Error('无法确认活动的原生支付回执。');
   if(receipt.messageId&&receipt.messageId!==message.id)throw Error('此活动已生成另一张法术消息。');
   if(receipt.state!=='used')await save(actor,ledger(actor).map(r=>r.id===receipt.id?{...r,state:'used',messageId:message.id}:r));
  });
 }
 async function ensurePaid({actor,item,message,user}){
  owner(actor,user);
  const proof=message.flags?.[MODULE_ID]?.nativeCast;
  if(!matches(item)&&!proof?.activityMessageId||item.actor?.uuid!==actor.uuid||author(message)!==user.id||game.messages.get(message?.id)!==message||message.flags?.pf2e?.origin?.uuid!==item.uuid)throw Error('支付需要原作者的真实法术消息。');
  // A receipt is consumed by a message, not a provider: one spell can grant
  // several legal class benefits without paying several times.
  const request=message.flags?.[MODULE_ID]?.nativeCastInput??input(item,{rank:message.flags?.pf2e?.origin?.castRank??item.rank});
  if(request.actorUuid!==actor.uuid||request.itemUuid!==item.uuid||request.sourceId!==source(item))throw Error('施法消息参数与角色法术不匹配。');
  const receiptId=proof?.id??'chat:'+message.id;
  if(proof){
   const prior=ledger(actor).find(r=>r.id===receiptId);
   if(!prior||prior.userId!==user.id||!sameInput(prior,request)||!['paid','used'].includes(prior.state))throw Error('原生施法支付参数不匹配或结果未能确认。');
   if(prior.activity||proof.activityMessageId){
    if(prior.activity?.messageId!==proof.activityMessageId)throw Error('组合活动支付回执不匹配。');
    const activity=await activitySource(actor,game.messages.get(proof.activityMessageId),user);
    if(JSON.stringify(prior.activity)!==JSON.stringify(activity))throw Error('组合活动来源不匹配。');
   }
  }else{
   await pay({...request,id:receiptId},user);
   owner(actor,user);
   await message.update({[`flags.${MODULE_ID}.nativeCast`]:{id:receiptId,actorUuid:actor.uuid,itemUuid:item.uuid,userId:user.id},[`flags.${MODULE_ID}.nativeCastInput`]:copy(request)});
  }
  return queue.run(actor.uuid,async()=>{
   owner(actor,user);
   const receipt=ledger(actor).find(r=>r.id===receiptId);
   if(!receipt||receipt.userId!==user.id||receipt.itemUuid!==item.uuid||!['paid','used'].includes(receipt.state))throw Error('原生施法支付结果未能确认。');
   if(receipt.messageId&&receipt.messageId!==message.id)throw Error('这笔施法支付已用于另一张消息。');
   if(receipt.state!=='used'||!receipt.messageId)await save(actor,ledger(actor).map(r=>r.id===receiptId?{...r,state:'used',messageId:message.id}:r));
   return copy({...receipt,state:'used',messageId:message.id});
  });
 }
 function register({libWrapper=globalThis.libWrapper,socket:socketApi,Hooks=globalThis.Hooks}={}){
  if(installed)return ()=>{};if(!libWrapper)throw Error('施法资源适配需要 libWrapper。');installed=true;socket=socketApi;
  socket?.register('native-cast-pay',async function(payload){try{return {ok:true,value:await pay(payload,game.users.get(this.socketdata.userId))};}catch(error){return {ok:false,error:error.message,code:error.code};}});
  socket?.register('native-cast-outcome',async function(payload){try{return {ok:true,value:await finishPaidCast(payload,game.users.get(this.socketdata.userId))};}catch(error){return {ok:false,error:error.message};}});
  socket?.register('native-cast-invocation-proof',function(payload){return invocationProof(payload,this.socketdata.userId)});
  socket?.register('native-cast-invocation-bind',async function(payload){try{return {ok:true,value:await bindInvocation(payload,game.users.get(this.socketdata.userId))};}catch(error){return {ok:false,error:error.message};}});
  slotHookAvailable=typeof Hooks?.on==='function';
  const slotHook=Hooks?.on?.('updateItem',(entry,changes,options,userId)=>{
   const request=slotRequests.get(entry.actor?.uuid)?.();
   if(!request?.captured||entry!==request.entry||userId!==request.gmId)return;
   const marker=options?.[MODULE_ID]?.nativeSlotCommit,delta=changes[`flags.${MODULE_ID}.nativeSlotCommit`]??changes.flags?.[MODULE_ID]?.nativeSlotCommit;
   const after=changes[`system.slots.slot${request.rank}.value`]??changes.system?.slots?.[`slot${request.rank}`]?.value;
   // Foundry omits unchanged marker fields on later updates. Require this new
   // nonce in the actual slot update, its exact proof subset, and the complete
   // options and persisted marker; options alone never witness payment.
   const exactDelta=delta&&typeof delta==='object'&&!Array.isArray(delta)&&Object.hasOwn(delta,'castNonce')&&delta.castNonce===request.proof.castNonce&&Object.entries(delta).every(([key,value])=>Object.hasOwn(request.proof,key)&&equal(value,request.proof[key]));
   if(exactDelta&&equal(marker,request.proof)&&equal(entry.flags?.[MODULE_ID]?.nativeSlotCommit,request.proof)&&after===request.proof.after)request.witness=true;
  });
  const publicationHook=Hooks?.on?.('preCreateChatMessage',message=>{
   const own=message.flags?.[MODULE_ID],scope=enrollments.get(own?.nativeCast?.id);
   if(!scope)return;
   try{liveEnrollment(scope);if(!sameInput(own.nativeCastInput??{},scope.input)||message.flags?.pf2e?.origin?.uuid!==scope.item.uuid)throw Error('本次原生施法卡来源已改变。');}
   catch(error){scope.messageError=asError(error);return false;}
  });
  const paths=[];const wrap=(path,fn)=>{libWrapper.register(MODULE_ID,path,fn,'MIXED');paths.push(path);};
  wrap('CONFIG.Actor.documentClass.prototype.update',function(wrapped,changes={},options={}){
   const actor=this;
   const original=(changes,options)=>{
   const scope=focusCall,after=changes['system.resources.focus.value']??changes.system?.resources?.focus?.value;
   if(!scope||actor!==scope.actor||after===undefined)return wrapped(changes,options);
   if(scope.captured||actor.system.resources?.focus?.value!==scope.before||after!==scope.before-scope.cost)throw Error('原生聚能写入的来源、次数或金额已改变。');
   const proof={castNonce:scope.castNonce,itemUuid:scope.item.uuid,entryUuid:scope.entry.uuid,before:scope.before,after,cost:scope.cost};
   const extra=scope.changes(Object.freeze(proof));
   if(!extra||typeof extra!=='object'||Object.keys(extra).some(key=>!key.startsWith(`flags.${MODULE_ID}.`)||Object.hasOwn(changes,key)))throw Error('原生聚能提交扩展只能附加自身回执。');
   scope.captured=true;
   return wrapped({...changes,...extra},options);
   };
   // Share the one libWrapper registration with native Refocus observers.
   // No async trampoline: an observer that defers loses the exact call binding.
   const chain=[...actorUpdateMiddlewares];
   const invoke=(index,changes,options)=>index===chain.length?original(changes,options):chain[index].call(actor,(nextChanges,nextOptions)=>invoke(index+1,nextChanges,nextOptions),changes,options);
   return invoke(0,changes,options);
  });
  wrap('CONFIG.PF2E.Item.documentClasses.spellcastingEntry.prototype.update',function(wrapped,changes={},options={}){
   const request=slotCall,path=request?`system.slots.slot${request.rank}.value`:null;
   const after=request?(changes[path]??changes.system?.slots?.[`slot${request.rank}`]?.value):undefined;
   if(!request||this!==request.entry||after===undefined)return wrapped(changes,options);
   if(request.captured||game.user.id!==request.gmId||game.users.activeGM?.id!==request.gmId||this.system.slots[`slot${request.rank}`].value!==request.before||after!==request.before-request.cost)throw Error('原生法术位写入的来源、次数或金额已改变。');
   const proof={castNonce:request.castNonce,itemUuid:request.item.uuid,entryUuid:this.uuid,rank:request.rank,before:request.before,after,cost:request.cost,userId:request.userId,gmId:request.gmId};
   const extra=request.changes(Object.freeze(proof)),markerPath=`flags.${MODULE_ID}.nativeSlotCommit`;
   if(!extra||typeof extra!=='object'||Object.keys(extra).some(key=>!key.startsWith(`flags.${MODULE_ID}.`)||key===markerPath||Object.hasOwn(changes,key))||Object.hasOwn(changes,markerPath)||changes.flags?.[MODULE_ID]?.nativeSlotCommit)throw Error('原生法术位提交扩展只能附加自身回执。');
   request.captured=true;request.proof=proof;
   const result=wrapped({...changes,...extra,[markerPath]:proof},{...options,[MODULE_ID]:{...options[MODULE_ID],nativeSlotCommit:proof}});
   return Promise.resolve(result).then(doc=>{request.returned=doc===request.entry;return doc;});
  });
  wrap('CONFIG.PF2E.Item.documentClasses.spellcastingEntry.prototype.cast',async function(wrapped,item,options={}){
   let enrollment=null,invocationOpen=true;
   const nativeEntry=async()=>{
   if(!matches(item)&&!managed(item.actor)&&!enrollment)return wrapped(item,options);
   const policies=options.consume!==false&&options.message!==false?[...paidCastPolicies]:[];
   const scope={item,entry:this,options,user:game.user,castNonce:id(),policies,tokenContext:policies.length||enrollment?sourceToken(item.actor):{},input:{...input(item,options),...(enrollment?{messageMode:options.messageMode}:{})},targets:[...new Set(values(game.user.targets).map(t=>t.document?.uuid??t.uuid).filter(Boolean))],captured:captureData(item),...(enrollment?{invocation:enrollment}:{})};
   scope.capturePromise=new Promise(resolve=>{scope.resolveCapture=resolve;});
   scope.consumeSignal=new Promise(resolve=>{scope.resolveConsume=resolve;});
   return localCasts.run(item.actor.uuid,async()=>{
    if(!sameInput(scope.input,input(item,options))||JSON.stringify(scope.captured)!==JSON.stringify(captureData(item)))throw Error('排队期间施法配置已改变，请重新使用法术。');
    scopes.set(item.uuid,scope);
    try{
     if(enrollment){enrollments.set(scope.castNonce,scope);liveEnrollment(scope);if(scope.invocation.data.sourceTokenUuid!==scope.tokenContext.tokenUuid)throw Error('原生施法调用的来源Token与冻结分配不一致。');}
     // PF2e intentionally skips consume for at-will spells. Confirm that native
     // exemption through the same payment coordinator before running policies.
     if(policies.length&&item.atWill){
      trackConsume(scope,consumeForCast(scope,scope.input));
      if(!await scope.consumeTask){if(scope.error)throw scope.error;return scope.terminal;}
     }
     const nativeTask=Promise.resolve().then(()=>wrapped(item,options));
     // Native PF2e awaits toMessage, while Toolbelt detaches it. Observe both
     // forms without waiting forever inside wrapped before arming the watchdog.
     const first=await Promise.race([nativeTask.then(result=>({result}),error=>({error})),scope.consumeSignal.then(()=>({consumed:true}))]);
     if(first.error){
      // An outer Toolbelt/module wrapper can throw after detaching the native
      // consume. Retain the scope until that payment settles, and make its
      // already-running consume return false instead of emitting a success card.
      if(scope.consumeTask){scope.error??=asError(first.error);await scope.consumeTask;}
      throw first.error;
     }
     // Toolbelt 3.56's actionable cast wrapper starts native cast but omits its
     // return/await. Keep this exact scope until the already-started native
     // consume and its actualCast capture finish; no time/nearest-card guessing.
     if(scope.consumeTask){const paid=await scope.consumeTask;if(scope.error)throw scope.error;if(scope.terminal)return enrollment?{status:'disrupted',castNonce:scope.castNonce,input:copy(scope.input),receipt:copy(scope.receipt),message:null,nativeResult:undefined}:scope.terminal;if(paid&&options.message!==false&&(matches(item)||scope.policies.length||enrollment)){const result=await finishMessage(scope,nativeTask);return enrollment?await completeInvocation(scope,result):result;}}
     if(scope.error)throw scope.error;
     if(enrollment)throw Error('本次施法没有成功消费并生成准确原卡。');
     return await nativeTask;
    }catch(error){
     if((scope.policies.length||enrollment)&&scope.receipt&&!scope.terminal&&!scope.error)throw await uncertainCast(scope,error);
     throw asError(error);
    }finally{if(scopes.get(item.uuid)===scope)scopes.delete(item.uuid);if(enrollment){scope.closed=true;enrollments.delete(scope.castNonce);}}
   });
   };
   const chain=[...castMiddlewares],invoke=index=>{
    if(index===chain.length)return nativeEntry();let continued=false;
    const next=()=>{if(enrollment&&continued)throw Error('本次原生施法调用只能继续一次。');continued=true;return invoke(index+1)};
    next.withOutcome=async({kind,data}={})=>{
     if(!invocationOpen||continued||enrollment||!invocationAdapters.has(kind)||options.consume===false||options.message===false||!game.users?.activeGM?.id)throw Error('原生施法调用适配无效、重复或不是实际施法。');
     const serialized=JSON.stringify(data);if(!serialized||serialized.length>16000||typeof data!=='object'||data===null||Array.isArray(data))throw Error('原生施法调用参数无效。');
     if(data.messageMode!=='public'||(options.messageMode??game.settings?.get('core','messageMode'))!=='public')throw Error('本次施法调用只支持已选择的公开消息模式。');
     options={...options,messageMode:'public'};
     const original=item.original??item,Spell=globalThis.CONFIG?.PF2E?.Item?.documentClasses?.spell,rank=options.rank??item.rank;
     if(typeof Spell!=='function'||!(item instanceof Spell)||!(original instanceof Spell)||original.actor.items.get(original.id)!==original||!Number.isInteger(rank)||rank<1||rank>3||values(item.appliedOverlays).length)throw Error('本次调用需要准确的原生法术或无覆盖升环实例。');
     // PF8.5.1's native factory performs late preparation and binds .original.
     // With no requested rank its same-rank path still creates a fresh instance.
     const privateItem=original.loadVariant({castRank:rank})??original.loadVariant({});
     if(!(privateItem instanceof Spell)||privateItem===original||privateItem===item||privateItem.original!==original||privateItem.actor!==item.actor||privateItem.uuid!==item.uuid||privateItem.rank!==rank||privateItem.spellcasting!==this||values(privateItem.appliedOverlays).length||source(privateItem)!==source(original))throw Error('原生法术未能提供独立的本次调用实例。');
     item=privateItem;
     enrollment={kind,data:JSON.parse(serialized),gmId:game.users.activeGM.id};
     // A spontaneous button has no prepared-slot index; no other route changes.
     if(this.isSpontaneous===true&&(options.slotId===undefined||Number.isNaN(options.slotId)))options={...options,slotId:null};
     return next();
    };
    return chain[index]({item,options,entry:this},next);
   };
   try{return await invoke(0);}finally{invocationOpen=false;}
  });
  wrap('CONFIG.PF2E.Item.documentClasses.spellcastingEntry.prototype.consume',async function(wrapped,item,rank,slotId,capability){
   if(capability===nativeCapability){
    const request=focusRequests.get(item.actor?.uuid)?.(),slotRequest=slotRequests.get(item.actor?.uuid)?.(),prior=focusCall,priorSlot=slotCall;
    if(request&&(request.actor!==item.actor||request.item!==item||request.entry!==this))throw Error('原生聚能消费对象已改变。');
    // PF2e 8.5.1 calls actor.update synchronously before its first await.
    // End the binding immediately when wrapped returns, never after its promise
    // settles: deferred/foreign updates cannot borrow a nearby payment scope.
    focusCall=request??null;
    if(slotRequest&&(slotRequest.item!==item||slotRequest.entry!==this||slotRequest.rank!==rank))throw Error('原生法术位消费对象已改变。');
    slotCall=slotRequest??null;
    try{return wrapped(item,rank,slotId);}finally{focusCall=prior;slotCall=priorSlot;}
   }
   const current=scopes.get(item.uuid),scope=current?.item===item&&current.entry===this?current:null;
   if(!matches(item)&&!managed(item.actor)&&!scope?.invocation)return wrapped(item,rank,slotId);
   if((current?.policies.length||current?.invocation)&&(!scope||!sameInput(current.input,input(item,{rank,slotId})))){current.error=Error('原生施法对象、来源或参数在实际消费前发生变化，无法安全执行反应。');return false;}
   if(scope?.consumeTask)return scope.consumeTask;
   const task=consumeForCast(scope,scope?.input??input(item,{rank,slotId}));
   if(scope)trackConsume(scope,task);return task;
  });
  return ()=>{for(const path of paths)libWrapper.unregister(MODULE_ID,path);if(slotHook!==undefined)Hooks?.off?.('updateItem',slotHook);if(publicationHook!==undefined)Hooks?.off?.('preCreateChatMessage',publicationHook);slotHookAvailable=false;installed=false;};
 }
 // Paid policies run on the initiating client outside the resource queue.
 // Return {disrupted:true,reason?,eventId?} to stop this live cast after payment;
 // undefined admits it. Explicit consume:false/message:false and activity/chat
 // payment flows are not native paid-cast events. Policies must validate any
 // unsupportedReason before using a source token for positional reactions.
 function addInvocationAdapter(kind,adapter){if(typeof kind!=='string'||!/^[-a-z0-9]{1,80}$/.test(kind)||invocationAdapters.has(kind)||typeof adapter?.validate!=='function'||typeof adapter?.consumePolicy!=='function')throw Error('原生施法调用适配注册无效或重复。');invocationAdapters.set(kind,Object.freeze({...adapter}));}
 return {addInvocationAdapter,addActorUpdateMiddleware:middleware=>{actorUpdateMiddlewares.add(middleware);return()=>actorUpdateMiddlewares.delete(middleware)},withActorResourceLock:(actor,operation)=>queue.run(actor.uuid,operation),addCastMiddleware:middleware=>castMiddlewares.add(middleware),addMatcher:matcher=>matchers.add(matcher),addActivityMatcher:matcher=>activityMatchers.add(matcher),addActorMatcher:matcher=>actorMatchers.add(matcher),addConsumePolicy:policy=>consumePolicies.add(policy),addPaidCastPolicy:policy=>paidCastPolicies.add(policy),addCapture:(key,capture)=>captures.set(key,capture),captureUsage,captureMessageOutcome,ensurePaid,payForActivity,finishActivityWithoutSpell,register};
}
