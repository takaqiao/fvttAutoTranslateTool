import {MODULE_ID as ID} from './rules.mjs';
import {isActiveGM,getSourceId} from './native-context.mjs';
import {prepareWeaponSurgeDamageSnapshotItems} from './weapon-surge.mjs';
import {nativeRollEvent,manualDamageRoll as rollManualDamage,manualDamagePrivacy as damagePrivacy,confirmManualFlatCheck,restrictNativePrivacy} from './manual-native-roll.mjs';

const values=collection=>Array.from(collection?.values?.()??collection??[]);
const clone=data=>structuredClone(data);
const author=message=>message.author?.id??message.author??message.user?.id??message.user;
const operations=message=>message.flags?.[ID]?.nativeOwnerOperations??{};
const Message=()=>globalThis.CONFIG?.ChatMessage?.documentClass??globalThis.ChatMessage;
const sessions=new WeakMap();
const invocations=new WeakMap();
const typedTypes=new Set(['check','flat','d20','formula-damage']);
const damageTypes=new Set(['damage','spell-damage','formula-damage']);
const privateTypes=new Set([...typedTypes,...damageTypes]);
const publicRequest=request=>Object.fromEntries(['type','statistic','itemUuid','sourceUuid','tokenUuid','targetUuid','targetActorUuid','checkKind','weaponId','altUsageType','map','critical','spellId','rank','overlayIds'].filter(key=>request[key]!==undefined).map(key=>[key,request[key]]));

/** Only a locally authenticated native execution can establish its target. */
export function getNativeOwnerInvocation(game,input){
 const options=input?.options??input?.extraRollOptions??input;
 return values(options).map(option=>invocations.get(game)?.get(option)).find(Boolean)??null;
}

const pinnedTarget=token=>new Proxy(Object.create(Object.getPrototypeOf(token.actor)),{
 get:(_target,key)=>{if(key==='getActiveTokens')return()=>[token];const value=Reflect.get(token.actor,key,token.actor);return typeof value==='function'?value.bind(token.actor):value;},
 has:(_target,key)=>key in token.actor,
});

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
 if(assertLive)for(const event of ['updateUser','userConnected','updateChatMessage','deleteChatMessage','updateActor','deleteActor','updateItem','deleteItem','updateToken','deleteToken'])listeners.push([event,Hooks.on(event,checkLive)]);
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
export function createNativeOwnerOperations({game,fromUuid=globalThis.fromUuid,scope,syncTimeoutMs=15000,manualDamageRoll=rollManualDamage,manualDamagePrivacy=damagePrivacy,confirmFlatCheck=confirmManualFlatCheck}={}){
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
 async function saveDamageResult({message,operation,actor,request,roll,privacy,context,guard}){
  const whisper=values(game.users).filter(user=>user.isGM).map(user=>user.id);if(!whisper.length)throw Error('没有GM可保存原生伤害凭据。');
  const proof=await Message().create({author:game.user.id,speaker:{actor:actor.id},blind:true,whisper,rolls:[],content:'<p>原生伤害投骰凭据；最终伤害由原活动继续结算。</p>',flags:{[ID]:{nativeOwnerDamageResult:{activityMessageId:message.id,nonce:operation.nonce,scope,actorUuid:actor.uuid,type:request.type,request:publicRequest(request),roll:roll.toJSON(),privacy,...context?{context}:{}}}}});guard();
  if(!proof?.id||game.messages.get(proof.id)!==proof)throw Error('原生伤害凭据没有保存，不能自动重投。');return {status:'rolled',messageId:proof.id};
 }
 async function executeNative({message,operation,actor,guard,request=operation.request}){
  const target=request.targetUuid?await fromUuid(request.targetUuid):null;guard();
  const originalGuard=guard;
  guard=()=>{originalGuard();if(request.targetUuid&&(!target?.actor||!target.object||target.uuid!==request.targetUuid||target.actor.uuid!==request.targetActorUuid))throw Error('原生操作的原目标已改变。');};guard();
  if(typedTypes.has(request.type))return executeTyped({message,operation,actor,target,request,guard});
  const showDialog=true;
  const options=new Set(request.options??[]),event=nativeRollEvent(game,request.type==='attack'?'check':'damage');
  const marker=`${ID}:native-operation:${operation.nonce}`;options.add(marker);
  const selectedPrivacy=context=>{
   const messageMode=context?.messageMode??game.settings?.get?.('core','messageMode')??'public',Class=Message();
   if(typeof Class?.applyMode!=='function'){if(messageMode!=='public')throw Error('缺少原生秘骰可见性接口，不能公开伤害。');return {messageMode,blind:false,whisper:[]};}
   const data=Class.applyMode({author:game.user.id},messageMode);return restrictNativePrivacy({messageMode,blind:data.blind===true,whisper:[...data.whisper??[]]},request.minimumPrivacy,game.user.id);
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
   return saveDamageResult({message,operation,actor,request,roll,privacy:selectedPrivacy(native.context),context:{options:[...native.context.options??[]],domains:[...native.context.domains??[]],traits:[...native.context.traits??[]],messageMode:native.context.messageMode},guard});
  }
  const strike=values(roller.system.actions).flatMap(strike=>[strike,...strike.altUsages??[]]).find(strike=>strike.type==='strike'&&strike.item?.id===request.weaponId&&(strike.item.altUsageType??'')===(request.altUsageType??''));
  if(!strike||strike.ready===false||!Number.isInteger(request.map)||request.map<0||request.map>2)throw Error('原生武器或多重攻击档位已失效。');
  if(request.type==='damage'){
   let context;const hook=Hooks?.on?.('renderDamageModifierDialog',app=>{if(values(app.context?.options).includes(marker))context=app.context;});
   try{const roll=await beforeNativeRoll({Hooks,marker,showDialog,dialogKind:'damage',commit:async()=>{},assertLive:guard,native:()=>strike[request.critical?'critical':'damage']({target:target.object,checkContext:request.checkContext,mapIncreases:request.map,options,event,createMessage:false})});guard();
    return roll?saveDamageResult({message,operation,actor,request,roll,privacy:selectedPrivacy(context),context:{options:values(context?.options),domains:values(context?.domains),traits:values(context?.traits),messageMode:context?.messageMode},guard}):{status:'cancelled'};
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
 async function executeTyped({message,operation,actor,target,request,guard}){
  const marker=`${ID}:native-operation:${operation.nonce}`,options=new Set(values(request.options)),token=request.tokenUuid?await fromUuid(request.tokenUuid):null,item=request.itemUuid?await fromUuid(request.itemUuid):null;
  options.add(marker);
  const originalGuard=guard;
  guard=()=>{originalGuard();if(request.tokenUuid&&(token?.actor!==actor||token.uuid!==request.tokenUuid||token.parent?.tokens&&token.parent.tokens.get(token.id)!==token))throw Error('原生检定的来源Token已改变。');if(request.itemUuid&&(!item||item.actor!==actor||actor.items.get(item.id)!==item||request.sourceUuid&&getSourceId(item)!==request.sourceUuid||message.flags?.pf2e?.origin?.uuid&&message.flags.pf2e.origin.uuid!==item.uuid))throw Error('原生检定的能力来源已改变。');};guard();
  if(!invocations.has(game))invocations.set(game,new Map());
  const local=invocations.get(game);local.set(marker,Object.freeze({actorUuid:actor.uuid,tokenUuid:token?.uuid??null,targetUuid:target?.uuid??null,user:game.user,usageId:message.id}));
  const commit=async()=>{
   guard();if(!operation.requiresCommit||operations(message)[operation.nonce]?.committed)return;
   if(isActiveGM(game))await commitOperation({messageId:message.id,nonce:operation.nonce},game.user);
   else{const {error}=await operationReply(message,operation.nonce,()=>socket.executeAsUser(`native-owner:${scope}:commit`,operation.requesterId,{messageId:message.id,nonce:operation.nonce}),{readDone:state=>state?.committed===true,assertLive:guard});if(error&&!operations(message)[operation.nonce]?.committed)throw error;}
   await waitFor(()=>operations(message)[operation.nonce]?.committed===true);guard();
  };
  try{
   if(request.type==='formula-damage'){
    if(operation.requiresCommit)throw Error('公式伤害不能在未确认窗口前支付新的费用。');
    const DamageRoll=globalThis.CONFIG?.Dice?.rolls?.find(Class=>Class.name==='DamageRoll');if(!DamageRoll||typeof request.formula!=='string'||!request.formula.trim())throw Error('原生伤害公式无效。');
    const roll=await manualDamageRoll({game,roll:new DamageRoll(request.formula),Hooks,messageMode:request.messageMode,assertLive:guard});guard();if(!roll)return {status:'cancelled'};
    const privacy=manualDamagePrivacy(roll,request.minimumPrivacy);if(!privacy)throw Error('原生伤害的接收者未确认，尚未发布。');
    return saveDamageResult({message,operation,actor,request,roll,privacy,guard});
   }
   let created;
   const callback=async(_roll,_outcome,raw)=>{
    await commit();guard();if(created)throw Error('同一原生检定出现多个结果，不能重复结算。');
    const data=raw.toObject();delete data._id;data.author=game.user.id;
    data.flags={...data.flags,[ID]:{...data.flags?.[ID],usageGenerated:true,nativeOwnerOperation:{activityMessageId:message.id,nonce:operation.nonce,scope,type:request.type,targetUuid:target?.uuid??null,targetActorUuid:target?.actor.uuid??null,...request.statistic?{statistic:request.statistic}:{}}}};
    created=await Message().create(data);guard();
   };
   const parameters={token:token??undefined,item:item??undefined,...target?{target:pinnedTarget(target)}:{target:null},dc:request.dc,action:request.action,label:request.label,title:request.title,traits:request.traits,extraRollOptions:[...options],skipDialog:false,event:null,messageMode:request.messageMode,createMessage:false,callback};
   let roll;
   if(request.type==='check'){
    let statistic=actor.getStatistic?.(request.statistic)??actor.skills?.[request.statistic]??actor.saves?.[request.statistic];
    if(request.checkDomains){if(!Array.isArray(request.checkDomains)||request.checkDomains.some(domain=>typeof domain!=='string')||typeof statistic?.clone!=='function')throw Error('原生检定领域无效。');statistic=statistic.clone({check:{domains:request.checkDomains}});}
    const native=request.checkMethod==='roll'?statistic:statistic?.check??statistic;if(typeof native?.roll!=='function')throw Error('原生技能或豁免统计不可用。');
    // A value-only DC cannot inherit the owner's unrelated selected creature.
    if(!target&&parameters.dc){parameters.dc={...parameters.dc};delete parameters.dc.slug;delete parameters.dc.statistic;}
    roll=await beforeNativeRoll({Hooks,marker,showDialog:true,commit,assertLive:guard,native:()=>native.roll(parameters)});
   }else{
    if(!game.pf2e?.Check?.roll||!game.pf2e?.CheckModifier)throw Error('原生检定接口不可用。');
    if(request.type==='flat'){if(!await confirmFlatCheck({label:request.label??'平检',dc:request.dc?.visible===false?undefined:request.dc?.value}))return {status:'cancelled'};guard();await commit();}
    const type=request.type==='flat'?'flat-check':'check',native=()=>game.pf2e.Check.roll(new game.pf2e.CheckModifier(request.action??request.type,{modifiers:[]}),{actor,token,item,type,title:request.title??request.label,domains:[type],options,dc:request.dc,skipDialog:false,createMessage:false,messageMode:request.messageMode,rollTwice:false,substitutions:[]},null,callback);
    roll=request.type==='flat'?await native():await beforeNativeRoll({Hooks,marker,showDialog:true,commit,assertLive:guard,native});
   }
   guard();if(!roll&&!created)return {status:'cancelled'};if(!created)throw Error('原生检定结果未保存，不能自动重投。');return {status:'rolled',messageId:created.id};
  }finally{local.delete(marker);}
 }
 async function ownerExecute(payload,requester){
  authority(requester);
  const message=await waitFor(()=>game.messages.get(payload?.messageId));
  const operation=await waitFor(()=>operations(message)[payload?.nonce]);
  let request=operation.request;
  if(privateTypes.has(request?.type)){
   const binding={actorUuid:operation.actorUuid,userId:operation.userId,requesterId:operation.requesterId};
   if(JSON.stringify(binding)!==JSON.stringify(payload?.binding)||JSON.stringify(publicRequest(payload?.privateRequest??{}))!==JSON.stringify(request))throw Error('原生检定的认证请求与原活动不符。');
   request=clone(payload.privateRequest);
  }
  const actor=await fromUuid(operation.actorUuid),guard=()=>live(message,operation,requester,actor);guard();
  const key=`${scope}:${message.id}:${operation.nonce}`;
  if(pending.has(key))return pending.get(key);
  const prior=operations(message)[operation.nonce];if(prior.status==='done')return clone(prior.result);
  if(prior.status!=='requested')throw Error('此原生执行已开始，结果尚未确认；不能重试。');
  const task=(async()=>{
   const save=(status,result)=>message.update({[`flags.${ID}.nativeOwnerOperations.${operation.nonce}`]:{...operations(message)[operation.nonce],status,...result?{result}:{} }});
   await save('started');guard();
   try{const result=await executeNative({message,operation,actor,guard,request});guard();await save('done',result);return result;}
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
  const typed=typedTypes.has(request.type),target=request.targetUuid?await fromUuid(request.targetUuid):null;authority(game.user);
  if((!typed||request.targetUuid)&&!target?.actor?.uuid)throw Error('原生操作的原目标不存在。');
  request={...request,...target?{targetActorUuid:target.actor.uuid}:{}};
  const nonce=globalThis.foundry?.utils?.randomID?.()??globalThis.crypto.randomUUID();
  const privateRequest=privateTypes.has(request.type);
  const operation={nonce,scope,actorUuid:actor.uuid,userId:user.id,requesterId:game.user.id,status:'requested',requiresCommit:typeof beforeRoll==='function',request:clone(privateRequest?publicRequest(request):request)};
  const payload={messageId:message.id,nonce,...privateRequest?{binding:{actorUuid:actor.uuid,userId:user.id,requesterId:game.user.id},privateRequest:clone(request)}:{}};
  if(operation.requiresCommit)committers.set(nonce,{actor,message,commit:beforeRoll});
  try{
  await message.update({[`flags.${ID}.nativeOwnerOperations.${nonce}`]:operation});authority(game.user);
  const {response,error}=await operationReply(message,nonce,()=>user.id===game.user.id?ownerExecute(payload,game.user):socket.executeAsUser(`native-owner:${scope}`,user.id,payload),{assertLive:()=>{
   authority(game.user);
   if(game.messages.get(message.id)!==message||author(message)!==user.id||target&&target.actor?.uuid!==request.targetActorUuid)throw Error('本次原生操作来源或目标已改变；不能重复执行。');
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
   if(damageTypes.has(request.type)&&result.status==='rolled'){
    const receipt=await waitFor(()=>game.messages.get(result.messageId)),proof=receipt.flags?.[ID]?.nativeOwnerDamageResult,whisper=values(receipt.whisper);
    if(author(receipt)!==user.id||receipt.speaker?.actor!==actor.id||receipt.blind!==true||!whisper.length||whisper.some(id=>!game.users.get(id)?.isGM)||receipt.rolls?.length||receipt.flags?.pf2e?.context||proof?.nonce!==nonce||proof.scope!==scope||proof.activityMessageId!==message.id||proof.actorUuid!==actor.uuid||proof.type!==request.type||JSON.stringify(proof.request)!==JSON.stringify(publicRequest(request)))throw Error('原生伤害凭据与本次操作者、来源或受众不符。');
    if(!Number.isFinite(proof.roll?.total)||typeof proof.roll.formula!=='string'||!proof.privacy||!Array.isArray(proof.privacy.whisper))throw Error('保存的原生伤害结果无效。');
    const privacy={...clone(proof.privacy),messageMode:proof.privacy.messageMode==='self'&&user.id!==game.user.id?'gm':proof.privacy.messageMode};
    if(request.type==='formula-damage'){
     const DamageRoll=globalThis.CONFIG?.Dice?.rolls?.find(Class=>Class.name==='DamageRoll'),roll=DamageRoll?.fromJSON(JSON.stringify(proof.roll));if(!roll?._evaluated||!Number.isFinite(roll.total))throw Error('保存的原生伤害结果无效。');return {...result,roll,privacy};
    }
    return {...result,roll:clone(proof.roll),privacy,context:clone(proof.context)};
   }
   if(typed&&result.status==='rolled'){
    const check=await waitFor(()=>game.messages.get(result.messageId));
    const proof=check.flags?.[ID]?.nativeOwnerOperation,context=check.flags?.pf2e?.context,roll=check.rolls?.[0],expected=request.type==='flat'?'flat-check':request.type==='d20'?'check':request.checkKind??(['fortitude','reflex','will'].includes(request.statistic)?'saving-throw':request.statistic==='perception'?'perception-check':'skill-check');
    const badDC=Number.isFinite(request.dc?.value)&&(context?.dc?.value!==request.dc.value||['criticalFailure','failure','success','criticalSuccess'].indexOf(context?.outcome)!==roll?.options?.degreeOfSuccess);
    const badOrigin=request.itemUuid&&check.flags?.pf2e?.origin?.uuid&&check.flags.pf2e.origin.uuid!==request.itemUuid;
    // Numeric-DC skill checks deliberately use native self context, even with a
    // selected target. Bind that target in our authenticated proof, not by adding
    // an opposed-statistic field that would change native modifiers or rules.
    const numericSkill=request.type==='check'&&expected==='skill-check'&&Number.isFinite(request.dc?.value)&&!request.dc.slug&&!Object.hasOwn(request.dc,'statistic');
    const badTarget=target&&(proof?.targetUuid!==target.uuid||proof.targetActorUuid!==request.targetActorUuid||(context?.target?context.target.token!==target.uuid||context.target.actor!==request.targetActorUuid:!numericSkill));
    if(author(check)!==user.id||check.speaker?.actor!==actor.id||check.rolls?.length!==1||roll?._evaluated!==true||!Number.isFinite(roll.total)||proof?.nonce!==nonce||proof.scope!==scope||proof.activityMessageId!==message.id||proof.type!==request.type||context?.type!==expected||badDC||badOrigin||request.type==='check'&&!values(context?.options).includes(`check:statistic:${request.statistic}`)||!values(context?.options).includes(`${ID}:native-operation:${nonce}`)||badTarget)throw Error('原生检定卡与本次操作者、来源或目标不符。');
    return {...result,check};
   }
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
 return {run,register,getInvocation:input=>getNativeOwnerInvocation(game,input)};
}

export function nativeTransientItems(strike,actor){
 const ids=new Set(values(actor.items).map(item=>item.id));
 return clone((strike.item.actor?._source?.items??[]).filter(item=>!ids.has(item._id)));
}
