import {MODULE_ID,hasSource} from './rules.mjs';
import {SerialActions,requireOwner} from './runtime.mjs';
import {isActiveGM,resolveMessageTargets} from './native-context.mjs';
import {getNativeCastEvents} from './amp-cast-events.mjs';

export const CONSUME_THRALL_SOURCE='Compendium.pf2e.actionspf2e.Item.HW8FAK8Gp9GBrFUo';
const values=value=>Array.from(value?.values?.()??value?.contents??value??[]);
const tokenDocument=token=>token?.document??token;
const state=actor=>actor?.flags?.[MODULE_ID]??{};
const focus=actor=>actor?.system?.resources?.focus;
const hasConsume=actor=>values(actor?.items).some(item=>hasSource(item,CONSUME_THRALL_SOURCE));
const restricted=actor=>Math.min(Math.max(0,Number(focus(actor)?.value)||0),Math.max(0,Number(state(actor).graveFocus?.value)||0));
const graveSpell=item=>item?.type==='spell'&&new Set(item.traits??item.system?.traits?.value??[]).has('necromancer');
const changedFocus=changes=>changes['system.resources.focus.value']??changes.system?.resources?.focus?.value;

/** Match the Summons Assistant's explicit owner and thrall marker, never actor names. */
export function ownedLivingThrall(token,actor){
 const thrall=token?.actor;
 return !!thrall&&thrall.uuid!==actor?.uuid&&thrall.flags?.['pf2e-summons-assistant']?.summoner?.uuid===actor?.uuid
  &&(thrall.rollOptions?.all?.['self:trait:thrall']===true||thrall.flags?.pf2e?.rollOptions?.all?.['self:trait:thrall']===true)
  &&Number(thrall.system?.attributes?.hp?.value)>0&&thrall.isDead!==true;
}

/** Uses native distance without the special ten-foot melee-reach diagonal rule. */
function inRange(source,target){
 if(!source?.object||!target?.object||source.parent?.id!==target.parent?.id)return false;
 const distance=source.object.distanceTo?.(target.object);
 return Number.isFinite(distance)&&distance>=0&&distance<=30;
}

function captureTokenActor(token,game){
 const actor=token?.actor,baseActor=token?.baseActor,actorId=token?.actorId,actorLink=token?.actorLink,synthetic=actor?.isToken===true;
 return {actor,current:()=>{
  if(!actor||!token||token.actor!==actor||token.baseActor!==baseActor||token.actorId!==actorId||token.actorLink!==actorLink||(actor.isToken===true)!==synthetic)return false;
  if(synthetic)return token.documentName==='Token'&&actor.token===token&&actor.parent===token&&actorLink===false&&!!baseActor&&actorId===baseActor.id&&actor.id===baseActor.id&&game.actors.get(actorId)===baseActor;
  return baseActor===undefined||baseActor===actor&&actorId===actor.id&&actorLink===true&&game.actors.get(actorId)===baseActor;
 }};
}

export function createThrallAutomation({game,fromUuid=globalThis.fromUuid,choose,onError=console.error,castEvents=getNativeCastEvents({game,fromUuid})}={}){
 const queue=new SerialActions(),consumptions=new Map();
 const resolveAction=item=>hasSource(item,CONSUME_THRALL_SOURCE)?'thrall:consume':undefined;
 const gm=()=>{if(!isActiveGM(game))throw Error('吞噬奴仆必须由当前主GM结算。');};
 const updateState=(actor,key,value)=>{gm();return actor.update({[`flags.${MODULE_ID}.${key}`]:value});};
 const pick=async(actor,user,title,choices)=>{
  if(!choices.length)throw Error('吞噬奴仆需要30尺内存活且属于自己的奴仆。');
  const answer=choices.length===1?choices[0].value:await choose?.({actor,user,title,choices});
  if(answer==null)throw Error('已取消吞噬奴仆。');
  if(!choices.some(choice=>choice.value===answer))throw Error('吞噬奴仆目标选择无效。');
  return answer;
 };
 const validReceipt=(receipt,item,user)=>!!receipt&&receipt.itemUuid===item.uuid&&receipt.userId===user.id&&receipt.before===1&&receipt.after===0;
 const remaining=item=>item.system.frequency?.value??item.system.frequency?.max??0;
 async function sourceFor(actor,message,user){
  const {scene,token}=message.speaker??{};
  if(scene&&token){const source=await fromUuid(`Scene.${scene}.Token.${token}`);if(source?.actor?.uuid!==actor.uuid)throw Error('吞噬奴仆的使用者Token已改变。');return source;}
  const tokens=values(actor.getActiveTokens?.(true,true)).map(tokenDocument).filter(t=>t?.actor?.uuid===actor.uuid);
  if(!tokens.length)throw Error('吞噬奴仆需要场景中的使用者Token。');
  const uuid=await pick(actor,user,'吞噬奴仆：选择使用者Token',tokens.map(t=>({value:t.uuid,label:t.name??actor.name})));
  return tokens.find(t=>t.uuid===uuid);
 }
 async function executeUsage(context){
  const {actor,item,message,user,action,frequencyReceipt}=context;gm();requireOwner(actor,user);const originalGM=game.user;
  if(item.actor?.uuid!==actor.uuid||resolveAction(item)!==action||!hasConsume(actor))throw Error('吞噬奴仆来源与使用事件不匹配。');
  if(game.messages?.get?.(message?.id)!==message||(message.author?.id??message.user?.id??message.user)!==user.id)throw Error('吞噬奴仆需要原作者的真实使用消息。');
   const origin=structuredClone(message.flags?.pf2e?.origin??{}),speaker=structuredClone(message.speaker??{}),inputProof=structuredClone(message.flags?.[MODULE_ID]?.usageInput??{}),pool={...focus(actor)},frequency={max:item.system.frequency?.max,per:item.system.frequency?.per};
   const sourceProof=[item.sourceId,item._stats?.compendiumSource,item.flags?.core?.sourceId],receiptProof=frequencyReceipt?structuredClone(frequencyReceipt):null;
   const sourceCandidates=speaker.scene&&speaker.token?[game.scenes.get(speaker.scene)?.tokens.get(speaker.token)]:values(actor.getActiveTokens?.(true,true)).map(tokenDocument);
   const sourceActors=new Map(sourceCandidates.filter(Boolean).map(token=>[token,token.actor]));
   const targetActors=new Map(sourceCandidates.flatMap(token=>values(token?.parent?.tokens)).map(token=>[token,token.actor]));
   const sourceBindings=new Map(sourceCandidates.filter(Boolean).map(token=>[token,captureTokenActor(token,game)])),targetBindings=new Map([...targetActors.keys()].map(token=>[token,captureTokenActor(token,game)]));
   return queue.run(actor.uuid,async()=>{
    const assertBase=()=>{
     gm();requireOwner(actor,user);
     if(game.user!==originalGM||game.users.activeGM!==originalGM||game.users.get(originalGM.id)!==originalGM)throw Error('吞噬奴仆的原始主GM文档已改变。');
     const input=message.flags?.[MODULE_ID]?.usageInput,stored=message.flags?.[MODULE_ID]?.usage?.frequencyReceipt;
     if(game.users.get(user.id)!==user||!actor.isToken&&game.actors.get(actor.id)!==actor||actor.canAct!==true||actor.isDead===true||actor.items.get(item.id)!==item||item.actor!==actor||resolveAction(item)!==action||item.suppressed===true||item.isSuppressed||item.system?.suppressed||game.messages.get(message.id)!==message||(message.author?.id??message.user?.id??message.user)!==user.id||message.flags?.pf2e?.origin?.uuid!==item.uuid||message.flags.pf2e.origin.actor!==actor.uuid||JSON.stringify(message.flags.pf2e.origin)!==JSON.stringify(origin)||JSON.stringify(message.speaker)!==JSON.stringify(speaker)||input?.actualUse!==true||frequency.max!==1||frequency.per!=='day'||item.system.frequency?.max!==frequency.max||item.system.frequency?.per!==frequency.per)throw Error('吞噬奴仆的原始来源、权限或每日资源已改变。');
     if(JSON.stringify(input)!==JSON.stringify(inputProof)||[item.sourceId,item._stats?.compendiumSource,item.flags?.core?.sourceId].some((value,index)=>value!==sourceProof[index]))throw Error('吞噬奴仆的原始目标记录或能力来源已改变。');
     if(receiptProof&&(!receiptProof.id||input.frequencyReceiptId!==receiptProof.id||JSON.stringify(stored)!==JSON.stringify(receiptProof)))throw Error('吞噬奴仆的原始付款回执已改变。');
    };
    assertBase();
   const receipt=validReceipt(frequencyReceipt,item,user);
   if(frequencyReceipt&&!receipt)throw Error('吞噬奴仆每日次数回执无效。');
    const prior=state(actor).thrallUse;
    if(prior?.id===message.id)throw Error('这条吞噬奴仆消息已经结算。');
    if(prior&&['claimed','deleting','uncertain'].includes(prior.status))throw Error('上次吞噬奴仆的结果尚未确认，不会再次销毁奴仆。');
    let stage='validating',paid=receipt,transaction,nativeEntered=false;
    const samePending=phase=>{const pending=state(actor).thrallUse;return !!pending&&!!transaction&&Object.keys(pending).length===Object.keys(transaction).length&&Object.keys(transaction).every(key=>pending[key]===(key==='status'?phase:transaction[key]));};
   try{
    if(!receipt&&remaining(item)<1)throw Error('吞噬奴仆每日可用次数已用完。');
    if(Number(focus(actor)?.value)!==0||Number(focus(actor)?.max)<1)throw Error('吞噬奴仆要求聚能点为0，且存在可用聚能池。');
     const source=await sourceFor(actor,message,user);assertBase();
     const scene=source?.parent,sourceActor=source?.actor;
     const assertSource=()=>{
      assertBase();
      if(sourceActors.get(source)!==actor||sourceActor!==actor||!sourceBindings.get(source)?.current()||!scene||game.scenes.get(scene.id)!==scene||scene.tokens.get(source.id)!==source)throw Error('吞噬奴仆的原始使用者Token已改变。');
     };
     assertSource();const targets=await resolveMessageTargets(message,{fromUuid});assertSource();
    const specified=message.flags?.[MODULE_ID]?.usageInput?.targetUuids??[];
    if(specified.length&&!targets.length)throw Error('吞噬奴仆选中的目标已不存在。');
    const candidates=(targets.length?targets:values(source.parent?.tokens)).filter(t=>ownedLivingThrall(t,actor)&&inRange(source,t));
    if(targets.length&&candidates.length!==targets.length)throw Error('吞噬奴仆选中的目标必须是30尺内属于自己的存活奴仆。');
    const uuid=await pick(actor,user,'吞噬奴仆：选择摧毁的奴仆',candidates.map(t=>({value:t.uuid,label:t.name??t.actor.name})));
     assertSource();
    const target=await fromUuid(uuid);
     const selected=candidates.find(candidate=>candidate.uuid===uuid),targetActor=targetActors.get(selected);
     const assertLive=(expectedFrequency,phase=null,deleted=false)=>{
      assertSource();
      if(phase&&!samePending(phase))throw Error('吞噬奴仆的待结算回执已改变。');
      if(focus(actor)?.value!==0||focus(actor)?.max!==pool.max||!Number.isInteger(pool.max)||pool.max<1||remaining(item)!==expectedFrequency)throw Error('吞噬奴仆的聚能池或每日次数已经改变。');
      if(!deleted&&(target!==selected||target?.actor!==targetActor||!targetBindings.get(selected)?.current()||target?.parent!==scene||scene.tokens.get(target.id)!==target||target.uuid!==uuid||!ownedLivingThrall(target,actor)||!inRange(source,target)))throw Error('所选奴仆已改变或不在30尺内。');
     };
     assertLive(receipt?0:1);
     transaction={id:message.id,userId:user.id,sourceTokenUuid:source.uuid,targetUuid:target.uuid,targetActorUuid:targetActor.uuid,itemUuid:item.uuid,frequencyReceiptId:receiptProof?.id??null,focusMax:pool.max,status:'claimed',createdAt:game.time?.worldTime??0};
    await updateState(actor,'thrallUse',transaction);stage='claimed';
     assertLive(paid?0:1,'claimed');
     if(!paid){await item.update({'system.frequency.value':0},{[MODULE_ID]:{usageInternal:true}});paid=true;assertLive(0,'claimed');}
    transaction.status='deleting';await updateState(actor,'thrallUse',transaction);stage='deleting';
    // Native Token deletion preserves Summons Assistant's destruction hooks.
     assertLive(0,'deleting');
     nativeEntered=true;
    await target.delete();
     const surviving=await fromUuid(target.uuid);assertLive(0,'deleting',true);
     if(surviving)throw Error('奴仆销毁尚未确认，不会发放聚能点。');
    stage='granting';
    await grant(actor,transaction);
     assertBase();if(!samePending('complete')||focus(actor)?.value!==1||focus(actor)?.max!==pool.max)throw Error('奴仆已销毁，但原始聚能发放回执尚未确认；不会重试。');
    stage='complete';return `已摧毁${target.name??'所选奴仆'}并恢复1点仅用于坟墓法术的聚能点；已消耗每日次数。`;
   }catch(error){
    if(!isActiveGM(game))throw error;
     try{assertBase();}catch{throw error;}
     if(stage==='deleting'&&samePending('deleting'))await updateState(actor,'thrallUse',{...transaction,status:nativeEntered?'uncertain':'rejected'}).catch(onError);
    else if(!['complete','granting'].includes(stage)){
      if(stage==='claimed'&&!samePending('claimed'))throw error;
      if(paid&&remaining(item)===0)await item.update({'system.frequency.value':1},{[MODULE_ID]:{usageInternal:true}});
      assertBase();
      if(stage==='claimed'&&samePending('claimed'))await updateState(actor,'thrallUse',{...transaction,status:'rejected'});
    }
    throw error;
   }
  });
 }

 async function grant(actor,transaction){
  gm();
  await actor.update({'system.resources.focus.value':1,[`flags.${MODULE_ID}.graveFocus`]:{value:1,sourceMessageId:transaction.id},[`flags.${MODULE_ID}.thrallUse`]:{...transaction,status:'complete'}});
 }

 async function maintain(actor){
  if(!isActiveGM(game)||!actor?.flags?.[MODULE_ID]?.thrallUse)return;
  const originalGM=game.user;
  return queue.run(actor.uuid,async()=>{
   gm();
   const transaction=state(actor).thrallUse;
    if(!transaction?.sourceTokenUuid||!['claimed','deleting','uncertain'].includes(transaction.status))return;
    const snapshot=JSON.stringify(transaction),item=values(actor.items).find(i=>i.uuid===transaction.itemUuid),message=game.messages.get(transaction.id),user=game.users.get(transaction.userId);
    const sourceParts=/^Scene\.([^.]+)\.Token\.([^.]+)$/.exec(transaction.sourceTokenUuid),registeredSource=sourceParts&&game.scenes.get(sourceParts[1])?.tokens.get(sourceParts[2]),sourceBinding=captureTokenActor(registeredSource,game);
    if(sourceBinding.actor!==actor||!sourceBinding.current())return;
    const source=await fromUuid(transaction.sourceTokenUuid),scene=source?.parent;
    const current=()=>{
     gm();
     if(game.user!==originalGM||game.users.activeGM!==originalGM||game.users.get(originalGM.id)!==originalGM)return false;
     if(source!==registeredSource||!sourceBinding.current())return false;
     const receipt=message?.flags?.[MODULE_ID]?.usage?.frequencyReceipt,input=message?.flags?.[MODULE_ID]?.usageInput;
     return JSON.stringify(state(actor).thrallUse)===snapshot&&!!user&&game.users.get(user.id)===user&&actor.testUserPermission(user,'OWNER')&&actor.canAct===true&&actor.isDead!==true&&(actor.isToken||game.actors.get(actor.id)===actor)&&item?.actor===actor&&actor.items.get(item.id)===item&&hasSource(item,CONSUME_THRALL_SOURCE)&&item.suppressed!==true&&!item.isSuppressed&&!item.system?.suppressed&&item.system.frequency?.max===1&&item.system.frequency?.per==='day'&&game.messages.get(message?.id)===message&&(message?.author?.id??message?.user?.id??message?.user)===user.id&&message.flags?.pf2e?.origin?.uuid===item.uuid&&message.flags.pf2e.origin.actor===actor.uuid&&input?.actualUse===true&&(!transaction.frequencyReceiptId||input.frequencyReceiptId===transaction.frequencyReceiptId&&validReceipt(receipt,item,user)&&receipt.id===transaction.frequencyReceiptId)&&source?.actor===actor&&!!scene&&game.scenes.get(scene.id)===scene&&scene.tokens.get(source.id)===source&&focus(actor)?.value===0&&focus(actor)?.max===transaction.focusMax;
    };
    if(!current())return;
   if(transaction?.status==='claimed'){
    // The native delete call is entered only after the deleting phase commits.
     if(![0,1].includes(remaining(item)))return;
     if(remaining(item)===0){await item.update({'system.frequency.value':1},{[MODULE_ID]:{usageInternal:true}});if(!current())return;}
    await updateState(actor,'thrallUse',{...transaction,status:'rejected'});return;
   }
    if(remaining(item)!==0||!/^Scene\.[^.]+\.Token\.[^.]+$/.test(transaction.targetUuid??''))return;
    const target=await fromUuid(transaction.targetUuid);if(!current()||remaining(item)!==0||target)return;
   // Final focus/provenance/status were one document update. A committed grant
   // therefore cannot still have deleting/uncertain status after reconnect.
   await grant(actor,transaction);
  });
 }

 async function applyFocusPolicy({actor,item,entry},next){
  const cost=Number(item?.system?.cast?.focusPoints)||0,reserved=restricted(actor);
  if(!reserved||cost<=0||item?.atWill||entry?.isRitual)return next();
  gm();
  const before=Number(focus(actor)?.value)||0;
  if(!graveSpell(item)&&before-reserved<cost){
   throw Error('吞噬奴仆恢复的聚能点只能用于坟墓法术。');
  }
  const pending={before,after:before-cost,reservedAfter:graveSpell(item)?Math.max(0,reserved-cost):reserved,committed:false};
  if(consumptions.has(actor.uuid))throw Error('聚能消耗发生重入，不会重复结算。');
  consumptions.set(actor.uuid,pending);
  try{
   const paid=await next();
   if(paid&&(!pending.committed||Number(focus(actor)?.value)!==pending.after||state(actor).graveFocus?.value!==pending.reservedAfter))throw Error('原生聚能扣款未能确认，不会将本次施法标为已支付。');
   return paid;
  }finally{if(consumptions.get(actor.uuid)===pending)consumptions.delete(actor.uuid);}
 }

 function register({Hooks,...dependencies}={}){
  const registrations=[],on=(name,callback)=>registrations.push([name,Hooks.on(name,callback)]);
  on('preUpdateActor',(actor,changes)=>{
   const next=changedFocus(changes);if(!Number.isFinite(next))return;
   const pending=consumptions.get(actor.uuid);
   if(pending&&Number(focus(actor)?.value)===pending.before&&next===pending.after){
    changes[`flags.${MODULE_ID}.graveFocus.value`]=pending.reservedAfter;pending.committed=true;
   }else if(next===0&&state(actor).graveFocus?.value>0)changes[`flags.${MODULE_ID}.graveFocus.value`]=0;
  });
  on('pf2e.restForTheNight',actor=>{
   // PF2e emits this hook only on the client performing the completed rest.
   if(!actor?.testUserPermission?.(game.user,'OWNER')||!state(actor).graveFocus&&!state(actor).thrallUse)return;
   queue.run(actor.uuid,()=>{
    const transaction=state(actor).thrallUse;
    return actor.update({[`flags.${MODULE_ID}.graveFocus.value`]:0,...transaction&&['claimed','deleting','uncertain'].includes(transaction.status)?{[`flags.${MODULE_ID}.thrallUse`]:{...transaction,status:'rested'}}:{}});
   }).catch(onError);
  });
  on('deleteToken',token=>{
   if(!isActiveGM(game))return;
   const ownerUuid=token.actor?.flags?.['pf2e-summons-assistant']?.summoner?.uuid;if(!ownerUuid)return;
   Promise.resolve(fromUuid(ownerUuid)).then(actor=>{
    if(state(actor).thrallUse?.targetUuid===token.uuid)return maintain(actor);
   }).catch(onError);
  });
  const removeCast=castEvents.register(dependencies);
  return()=>{for(const[name,id]of registrations)Hooks.off(name,id);removeCast?.();};
 }
 // A managed actor's ordinary spell cards need no capture waiter. All of its
 // resource consumption still crosses the existing authenticated GM queue.
 castEvents.addActorMatcher(actor=>hasConsume(actor)||restricted(actor)>0);
 castEvents.addConsumePolicy(applyFocusPolicy);
 return {resolveAction,executeUsage,register,maintain,applyFocusPolicy,tracksFrequency:item=>hasSource(item,CONSUME_THRALL_SOURCE)};
}
