import {MODULE_ID as ID} from './rules.mjs';
import {isActiveGM,getSourceId} from './native-context.mjs';
import {beforeNativeRoll} from './native-owner-operations.mjs';
import {restrictNativePrivacy} from './manual-native-roll.mjs';

const values=c=>Array.from(c?.values?.()??c??[]),author=m=>m?.author?.id??m?.user?.id??m?.user;
const states=m=>m?.flags?.[ID]?.nativeTargetSaves??{},copy=x=>structuredClone(x);
const outcomes=['criticalFailure','failure','success','criticalSuccess'];
const sameBinding=(one,two)=>!!one&&!!two&&Object.entries(two).every(([key,value])=>one[key]===value);
const failure=()=>Error('原生目标豁免的来源、拥有者或主GM已改变；结果未确认，不能自动重掷。');
const owns=(actor,user)=>!!user?.active&&!user.isGM&&actor.testUserPermission?.(user,'OWNER')===true;

export async function selectNativeTargetSaveOwner({game,actor,casterUser,choose}){
 const owners=values(game.users).filter(user=>owns(actor,user));
 const bound=owners.filter(user=>(user.character?.uuid??user.character)===actor.uuid||user.character?.id===actor.id||user.character===actor.id);
 const candidates=bound.length?bound:owns(actor,casterUser)?[casterUser]:owners;
 if(!candidates.length)throw Error('目标角色的玩家不在线；尚未在GM端代投豁免。');
 if(candidates.length===1)return candidates[0];
 if(typeof choose!=='function')throw Error('目标角色有多名在线拥有者，需要明确选择本次投骰者。');
 const id=await choose({actor,user:game.user,title:'目标豁免 · 选择在线玩家',choices:candidates.map(user=>({value:user.id,label:user.name??user.id}))});
 const user=candidates.find(user=>user.id===id);if(!user)throw Error('已取消选择目标豁免玩家。');return user;
}

/** Target saves have their own authority binding. The source card's author and
 * caster item stay intact while the target's player operates the native save. */
export function createNativeTargetSaves({game,fromUuid=globalThis.fromUuid,choose,scope,syncTimeoutMs=15000}={}){
 if(!['spell-combination','spiritual-scar'].includes(scope))throw Error('未知的目标豁免来源。');
 const requests=new Map(),ownerPending=new Map();let Hooks,socket;
 const gm=()=>{if(!isActiveGM(game)||!game.user.isGM||game.user.active===false)throw failure();};
 const leader=user=>{if(!user?.isGM||user.active===false||game.users.get(user.id)!==user||game.users.activeGM?.id!==user.id)throw failure();};
 const save=(message,nonce,state)=>message.update({[`flags.${ID}.nativeTargetSaves.${nonce}`]:state});
 const sourceValid=(binding,source,item)=>{
  if(scope==='spell-combination')return source.flags?.[ID]?.usageGenerated===true&&!!source.flags[ID].spellCombination?.activityMessageId&&source.flags?.pf2e?.origin?.uuid===item.uuid&&source.flags.pf2e.origin.actor===binding.sourceActorUuid;
  const proof=source.flags?.[ID]?.spiritualScarFollowup;
  return source.flags?.pf2e?.context?.type==='damage-taken'&&proof?.status==='rolling'&&proof.actorUuid===binding.sourceActorUuid&&proof.itemUuid===item.uuid&&proof.targetActorUuid===binding.targetActorUuid&&proof.targetTokenUuid===binding.targetUuid&&proof.damageMessageId===source.id;
 };
 async function resolve(binding){
  const source=game.messages.get(binding.sourceMessageId),[actor,item,target]=await Promise.all([binding.sourceActorUuid,binding.sourceItemUuid,binding.targetUuid].map(fromUuid));
  if(!source||author(source)!==binding.sourceAuthorId||!actor||!item||item.actor!==actor||actor.items.get(item.id)!==item||!target?.actor||target.actor.type!=='character'||target.actor.uuid!==binding.targetActorUuid||target.parent?.tokens?.get(target.id)!==target||!sourceValid(binding,source,item))throw failure();
  return {source,actor,item,target,targetActor:target.actor};
 }
 function live(binding,docs,{completed=false}={}){
  const {source,actor,item,target,targetActor}=docs,user=game.users.get(binding.saveUserId);
  leader(game.users.get(binding.requesterId));
  if(game.messages.get(source.id)!==source||author(source)!==binding.sourceAuthorId||states(source)[binding.nonce]&&!sameBinding(states(source)[binding.nonce].binding,binding)||[actor,targetActor].some(doc=>!doc.isToken&&game.actors?.get(doc.id)!==doc)||actor.items.get(item.id)!==item||item.actor!==actor||target.parent.tokens.get(target.id)!==target||target.actor!==targetActor||targetActor.uuid!==binding.targetActorUuid||!sourceValid(binding,source,item)||!completed&&!owns(targetActor,user))throw failure();
 }
 function checkProof(binding,request,card){
  const proof=card?.flags?.[ID]?.nativeTargetSave,c=card?.flags?.pf2e?.context,r=card?.rolls?.[0],degree=outcomes.indexOf(c?.outcome),marker=`${ID}:target-save:${binding.nonce}`;
  if(game.messages.get(card?.id)!==card||author(card)!==binding.saveUserId||card.speaker?.actor!==binding.targetActorUuid.split('.').at(-1)||card.rolls?.length!==1||!r?._evaluated||!Number.isFinite(r.total)||degree<0||r.options?.degreeOfSuccess!==degree||c?.type!=='saving-throw'||c.action!==request.action||!values(c.options).includes(marker)||card.flags.pf2e.origin?.uuid!==binding.sourceItemUuid||card.flags.pf2e.origin.actor!==binding.sourceActorUuid||Object.entries(binding).some(([key,value])=>proof?.[key]!==value)||Number.isFinite(request.dc?.value)&&c.dc?.value!==request.dc.value||c.target&&(c.target.actor!==binding.targetActorUuid||c.target.token!==binding.targetUuid))throw Error('原生目标豁免卡与本次来源、玩家或成功度不符。');
  if(!values(c.options).includes(`check:statistic:${request.statistic}`)||scope==='spell-combination'&&(card.flags.pf2e.origin.castRank!==request.rank||JSON.stringify(card.flags.pf2e.origin.variant?.overlays??[])!==JSON.stringify(request.overlayIds??[])))throw Error('原生目标豁免的统计、法术环阶或变体不符。');
  const narrowed=restrictNativePrivacy({messageMode:c.messageMode??'public',blind:card.blind===true,whisper:[...card.whisper??[]]},request.minimumPrivacy,binding.saveUserId);
  if(narrowed.blind!==card.blind||JSON.stringify(narrowed.whisper)!==JSON.stringify(card.whisper??[]))throw Error('原生目标豁免的秘密受众不符。');
  return card;
 }
 function observe(read,invoke,assertLive){
  const initial=read();if(initial)return Promise.resolve(initial);
  if(!Hooks?.on)throw Error('缺少目标豁免回执监听接口。');
  const listeners=[];let finish;
  const saved=new Promise(resolve=>finish=resolve),inspect=()=>{try{const value=read();if(value){finish({value});return;}assertLive();}catch(error){finish({error});}};
  for(const event of ['createChatMessage','updateChatMessage','deleteChatMessage','updateUser','userConnected','updateActor','deleteActor','updateToken','deleteToken','deleteItem'])listeners.push([event,Hooks.on(event,inspect)]);
  const reply=Promise.resolve().then(invoke).then(value=>({value}),error=>({error}));inspect();
  return Promise.race([saved,reply]).then(result=>{if(result.error)throw result.error;return result.value;}).finally(()=>{for(const[event,id]of listeners)Hooks.off(event,id);});
 }
 async function waitFor(read){
  if(read())return read();if(!Hooks?.on)throw failure();
  return new Promise((resolve,reject)=>{
   const listeners=[];let timer;const finish=(error,value)=>{clearTimeout(timer);for(const[e,id]of listeners)Hooks.off(e,id);error?reject(error):resolve(value);},inspect=()=>{const value=read();if(value)finish(null,value);};
   for(const event of ['createChatMessage','updateChatMessage'])listeners.push([event,Hooks.on(event,inspect)]);
   timer=setTimeout(()=>finish(Error('目标豁免凭据尚未同步，不能重掷。')),syncTimeoutMs);inspect();
  });
 }
 async function stage({messageId,nonce,status,messageIdResult},user){
  gm();const entry=requests.get(nonce),operation=states(game.messages.get(messageId))[nonce];
  if(!entry||entry.docs.source.id!==messageId||!sameBinding(operation?.binding,entry.binding)||operation.binding.saveUserId!==user?.id||operation.binding.requesterId!==game.user.id||operation.scope!==scope)throw failure();
  live(entry.binding,entry.docs,{completed:status==='done'});
  if(status==='done'){
   const card=checkProof(entry.binding,entry.request,game.messages.get(messageIdResult));
   if(operation.status==='done'){if(operation.messageId!==card.id)throw failure();return {status:'rolled',messageId:card.id};}
   if(operation.status!=='started')throw failure();
   await save(entry.docs.source,nonce,{...operation,status:'done',messageId:card.id});return {status:'rolled',messageId:card.id};
  }
  if(status==='started'&&operation.status==='requested'){await save(entry.docs.source,nonce,{...operation,status:'started'});return {status:'started'};}
  if(status==='cancelled'&&operation.status==='started'){await save(entry.docs.source,nonce,{...operation,status:'done',cancelled:true});return {status:'cancelled'};}
  if(status==='uncertain'&&operation.status!=='done'){await save(entry.docs.source,nonce,{...operation,status:'uncertain'});return {status:'uncertain'};}
  if(status==='started'&&operation.status==='started')return {status:'started'};
  throw failure();
 }
 async function ownerExecute(payload,requester){
  leader(requester);const source=await waitFor(()=>game.messages.get(payload?.messageId));
  const operation=await waitFor(()=>states(source)[payload?.nonce]);
  const binding=operation.binding,request=payload.request;
  if(operation.scope!==scope||!sameBinding(binding,payload.binding)||binding?.requesterId!==requester.id||binding.saveUserId!==game.user.id||payload.nonce!==binding.nonce||!request||!['fortitude','reflex','will'].includes(request.statistic))throw failure();
  const docs=await resolve(binding);live(binding,docs);
  const key=`${source.id}:${binding.nonce}`;if(ownerPending.has(key))return ownerPending.get(key);
  if(operation.status==='done')return operation.cancelled?{status:'cancelled'}:{status:'rolled',messageId:operation.messageId};
  if(operation.status!=='requested')throw Error('该目标豁免已经开始；不能再次投骰。');
  const task=(async()=>{
   const guard=()=>{live(binding,docs);if(game.user.id!==binding.saveUserId||!['started','done'].includes(states(source)[binding.nonce]?.status))throw failure();};
   const send=(status,id)=>socket.executeAsUser(`native-target-save:${scope}:stage`,requester.id,{messageId:source.id,nonce:binding.nonce,status,messageIdResult:id});
   await observe(()=>['started','done'].includes(states(source)[binding.nonce]?.status)&&{status:'started'},()=>send('started'),()=>live(binding,docs));guard();
   let item=docs.item;
   if(scope==='spell-combination'){
    if(item.type!=='spell'||request.rank!==source.flags.pf2e.origin.castRank)throw failure();
    item=item.loadVariant?.({castRank:request.rank,overlayIds:request.overlayIds??[]})??item;
    if(item.actor!==docs.actor||item.uuid!==docs.item.uuid||item.rank!==request.rank||JSON.stringify([...item.appliedOverlays?.values?.()??[]])!==JSON.stringify(request.overlayIds??[]))throw Error('原法术环阶或变体无法重建；尚未投骰。');
   }
   const marker=`${ID}:target-save:${binding.nonce}`,options=[...request.options??[],marker];
   const initialMode=request.messageMode??(request.minimumPrivacy?.blind?'blind':request.minimumPrivacy?.whisper?.length?'gm':undefined);
   let roller=docs.target.actor;
   if(request.adjustment){
    if(request.adjustment!=='one-degree-worse'||getSourceId(docs.item)!=='Compendium.pf2e.spells-srd.Item.r7ihOgKv19eJQnik')throw failure();
    roller=roller.clone({items:[...copy(roller._source.items),{_id:globalThis.foundry.utils.randomID(),name:item.name,type:'effect',system:{duration:{value:-1,unit:'unlimited'},rules:[{key:'AdjustDegreeOfSuccess',selector:'saving-throw',predicate:[marker],adjustment:{all:'one-degree-worse'}}]}}]},{keepId:true});
   }
   const statistic=roller.getStatistic?.(request.statistic),native=statistic?.check??statistic;if(typeof native?.roll!=='function')throw failure();
   let card,created=0,publicationError;
   const hook=Hooks.on('preCreateChatMessage',(message,_data,_options,userId)=>{
    const pf=message.flags?.pf2e,c=pf?.context,marked=values(c?.options).includes(marker);
    if(!marked&&!(c?.action===request.action&&pf?.origin?.uuid===docs.item.uuid&&message.speaker?.actor===docs.targetActor.id&&author(message)===game.user.id))return;
    try{
     guard();
     if(!marked||userId!==game.user.id||author(message)!==game.user.id||message.speaker?.actor!==docs.target.actor.id||c.type!=='saving-throw'||message.flags.pf2e.origin?.actor!==docs.actor.uuid||message.flags.pf2e.origin.uuid!==docs.item.uuid||created++)throw failure();
     const privacy=restrictNativePrivacy({messageMode:c.messageMode??initialMode??'public',blind:message.blind===true,whisper:[...message.whisper??[]]},request.minimumPrivacy,game.user.id);
     message.updateSource({blind:privacy.blind,whisper:privacy.whisper,'flags.pf2e.context.messageMode':privacy.messageMode,[`flags.${ID}.usageGenerated`]:true,[`flags.${ID}.nativeTargetSave`]:copy(binding)});
    }catch(error){publicationError=error;return false;}
   });
   try{
    const roll=await beforeNativeRoll({Hooks,marker,showDialog:true,commit:async()=>{},assertLive:guard,native:()=>native.roll({origin:docs.actor,item,token:docs.target,dc:request.dc,action:request.action,traits:request.traits,extraRollOptions:options,messageMode:initialMode,skipDialog:false,event:null,createMessage:true,callback:(_roll,_outcome,message)=>{if(card)throw failure();card=message;}})});
    guard();if(publicationError)throw publicationError;
    if(!roll&&!card)return await send('cancelled');
    if(!card||created!==1)throw Error('原生目标豁免没有唯一保存结果，不能重掷。');
    checkProof(binding,request,card);return await observe(()=>states(source)[binding.nonce]?.status==='done'&&{status:'rolled',messageId:states(source)[binding.nonce].messageId},()=>send('done',card.id),guard);
   }catch(error){await send('uncertain').catch(()=>{});throw error;}
   finally{Hooks.off('preCreateChatMessage',hook);}
  })();ownerPending.set(key,task);task.finally(()=>{if(ownerPending.get(key)===task)ownerPending.delete(key);}).catch(()=>{});return task;
 }
 async function run({sourceActor,sourceItem,sourceMessage,target,casterUser},request){
  gm();if(!Hooks||!socket?.executeAsUser||target?.actor?.type!=='character')throw failure();
  const existing=Object.values(states(sourceMessage)).filter(state=>state.scope===scope&&state.binding?.sourceActorUuid===sourceActor.uuid&&state.binding.sourceItemUuid===sourceItem.uuid&&state.binding.targetUuid===target.uuid);
  if(existing.length){
   if(existing.length!==1||existing[0].status!=='done')throw Error('该目标豁免已有未确认的执行，不能再次投骰。');
   const previous=existing[0],docs=await resolve(previous.binding);live(previous.binding,docs,{completed:true});
   return previous.cancelled?{status:'cancelled'}:{status:'rolled',messageId:previous.messageId,check:checkProof(previous.binding,request,game.messages.get(previous.messageId))};
  }
  const originalActivity=scope==='spell-combination'?game.messages.get(sourceMessage.flags?.[ID]?.spellCombination?.activityMessageId):null;
  const user=await selectNativeTargetSaveOwner({game,actor:target.actor,casterUser:casterUser??game.users.get(author(originalActivity??sourceMessage)),choose});gm();
  const nonce=globalThis.foundry?.utils?.randomID?.()??globalThis.crypto.randomUUID(),binding={nonce,scope,requesterId:game.user.id,saveUserId:user.id,sourceMessageId:sourceMessage.id,sourceAuthorId:author(sourceMessage),sourceActorUuid:sourceActor.uuid,sourceItemUuid:sourceItem.uuid,targetUuid:target.uuid,targetActorUuid:target.actor.uuid};
  if(!binding.sourceAuthorId)throw failure();const docs=await resolve(binding);gm();live(binding,docs);
  const privateRequest=copy(request),entry={binding,docs,request:privateRequest};requests.set(nonce,entry);
  const operation={scope,binding,status:'requested'};await save(sourceMessage,nonce,operation);gm();
  const observer=Hooks.on('createChatMessage',card=>{
   if(card.flags?.[ID]?.nativeTargetSave?.nonce!==nonce)return;
   void stage({messageId:sourceMessage.id,nonce,status:'done',messageIdResult:card.id},user).catch(()=>{});
  });
  try{
   await observe(()=>{const state=states(sourceMessage)[nonce];return state?.status==='done'?state.cancelled?{status:'cancelled'}:{status:'rolled',messageId:state.messageId}:state?.status==='uncertain'?{status:'uncertain'}:null;},()=>socket.executeAsUser(`native-target-save:${scope}`,user.id,{messageId:sourceMessage.id,nonce,binding:copy(binding),request:privateRequest}),()=>{gm();live(binding,docs);});gm();
   const state=states(sourceMessage)[nonce];if(state?.status!=='done')throw failure();live(binding,docs,{completed:true});
   if(state.cancelled)return {status:'cancelled'};
   const card=await waitFor(()=>game.messages.get(state.messageId));return {status:'rolled',messageId:card.id,check:checkProof(binding,privateRequest,card)};
  }catch(error){if(isActiveGM(game)&&states(sourceMessage)[nonce]?.status!=='done')await save(sourceMessage,nonce,{...states(sourceMessage)[nonce],status:'uncertain'}).catch(()=>{});throw error;}
  finally{Hooks.off('createChatMessage',observer);requests.delete(nonce);}
 }
 function register({Hooks:hooks,socket:api}){
  Hooks=hooks;socket=api;
  socket.register(`native-target-save:${scope}`,function(payload){return ownerExecute(payload,game.users.get(this.socketdata?.userId));});
  socket.register(`native-target-save:${scope}:stage`,function(payload){return stage(payload,game.users.get(this.socketdata?.userId));});
 }
 return {run,register};
}
