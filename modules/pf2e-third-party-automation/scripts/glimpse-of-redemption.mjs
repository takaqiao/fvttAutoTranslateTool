import {MODULE_ID as M} from './rules.mjs';
import {SerialActions} from './runtime.mjs';
import {isActiveGM,showNativeChoice,validateNativeChoices,markUnappliedDamageError} from './native-context.mjs';
import {withReactionReservation,genericReactionAvailable} from './reaction-budget.mjs';
import {createShieldReactionResources} from './shield-reaction-resources.mjs';
import {requireReactionPermitted} from './reaction-restriction.mjs';
import {GLIMPSE_SOURCES as S,glimpseSourceId,glimpseEncounter,glimpseCandidates,glimpseClaims,findGlimpseClaim,resolveGlimpseSource,validateGlimpseSource} from './glimpse-source.mjs';
import {compileGlimpseResistance,withGlimpseResistance,repentParams,glimpseMarker} from './glimpse-native.mjs';
import {glimpseWorld} from './glimpse-compat.mjs';
import {glimpseExpiryFor} from './glimpse-expiry.mjs';
const values=c=>Array.from(c?.values?.()??c??[]),author=m=>m?.author?.id??m?.user?.id??m?.user,random=()=>globalThis.foundry?.utils?.randomID?.(24)??crypto.randomUUID(),same=(a,b)=>JSON.stringify(a)===JSON.stringify(b),brand=Symbol('native-glimpse-scope');
const keyOf=s=>`${s.damageMessageId}:${s.rollIndex}:${s.tokenUuid}`;
/** Awaitable, source-bound reaction. All resource mutation is elected-GM work;
 * only a private live source-client scope may enter the original native call. */
export function createGlimpseProvider({game,reactionRestriction,fromUuid=globalThis.fromUuid,getRollContext,compat,choose,publishUse,reactionResources=createShieldReactionResources({game,reactionRestriction}),show=showNativeChoice,onError=console.error,onUnsupported=()=>globalThis.ui?.notifications?.warn?.('合并伤害按原生流程结算；本次救赎瞥视请手动处理，自动化未消耗反应。')}={}){
 const live=new Map(),plans=new WeakSet(),queue=new SerialActions(),pendingUses=new Map();let socket,installation;
 const ready=()=>glimpseWorld(game)&&compat?.ready()&&!game.modules?.get('pf2e-auto-action-tracker')?.active&&game.pf2e?.settings?.iwr!==false;
 const handlesActor=actor=>ready()&&actor?.type==='character'&&actor.level===5&&[S.glimpse,S.aura].every(source=>values(actor.items).some(i=>glimpseSourceId(i)===source))&&!values(actor.items).some(i=>glimpseSourceId(i)===S.weight);
 const owner=(actor,user)=>!!user?.active&&actor?.testUserPermission?.(user,'OWNER')===true;
 const gm=()=>{if(!isActiveGM(game)||!ready())throw Error('救赎瞥视主GM、世界或已验证兼容层已改变。');};
 const preferred=actor=>{const users=values(game.users).filter(u=>owner(actor,u));return users.find(u=>!u.isGM&&(u.character?.uuid===actor.uuid||u.character?.id===actor.id))??users.find(u=>!u.isGM)??users.find(u=>u.id===game.users.activeGM?.id)};
 const boundedGame=combat=>({combat,modules:game.modules,messages:game.messages,users:game.users});
 const allClaims=()=>values(game.combats).flatMap(c=>values(c.turns).flatMap(t=>glimpseClaims(t)));
 async function verified(snapshot){const c=await validateGlimpseSource({game,fromUuid,snapshot});if(!c.verified)throw Error(`救赎瞥视来源已改变：${c.unsupportedReason}`);return c}
 function scopeProof(payload,user){
  const p=live.get(payload?.scopeId);if(!p||p.userId!==user?.id||p.leader!==game.users.activeGM?.id||!same(p.snapshot,payload.snapshot)||!owner(p.actor,user)||JSON.stringify(p.damage.toJSON())!==p.damageEvidence)throw Error('没有这个拥有者等待中的原生救赎瞥视调用，或待处理伤害已改变。');
  if(payload.phase==='receipt')return {scopeId:p.scopeId,snapshot:p.snapshot,nonce:p.claim?.nonce,nativeEntered:p.nativeEntered===true,nativeReturned:p.nativeReturned===true,messageId:p.messageId??null,multiple:!!p.multiple};
  if(p.state!=='waiting')throw Error('本次原生伤害范围已关闭。');return {scopeId:p.scopeId,snapshot:p.snapshot};
 }
 async function authenticate(payload,user,phase){
  gm();const context=await verified(payload?.snapshot);gm();if(!owner(context.actor,user))throw Error('伤害申请者不是盟友拥有者。');
  const args={scopeId:payload.scopeId,snapshot:payload.snapshot,phase};
  const result=user.id===game.user.id?scopeProof(args,user):await socket?.executeAsUser('glimpse:scope',user.id,args),proof=user.id===game.user.id?result:result?.ok?result.value:null;gm();
  if(!proof||proof.scopeId!==payload.scopeId||!same(proof.snapshot,payload.snapshot))throw Error('原生伤害范围无法验证。');return {...context,proof};
 }
 function currentOption(context,expected){
  const option=glimpseCandidates(context,game).find(o=>o.token.uuid===expected.tokenUuid&&o.actor.uuid===expected.actorUuid&&o.ability.uuid===expected.itemUuid);
  if(!option||option.combat.id!==expected.combatId||option.combatant.id!==expected.combatantId||option.epoch!==expected.epoch||option.combat.round!==expected.round||option.combat.turn!==expected.turn||!owner(option.actor,game.users.get(expected.userId)))throw Error('救赎瞥视灵光、拥有者、来源或实际回合已改变。');return option;
 }
 async function choice(context){
  gm();if(!owner(context.actor,context.user))throw Error('能力选择者不再拥有实际角色。');
  let selected;
  if(choose)selected=await choose(context);
  else if(context.user.id===game.user.id)selected=await show(context);
  else{const r=await socket?.executeAsUser('glimpse:choice',context.user.id,{actorUuid:context.actor.uuid,title:context.title,choices:context.choices});if(!r?.ok)throw Error(r?.error??'敌方拥有者选择未完成。');selected=r.value}
  gm();if(selected!=null&&!context.choices.some(c=>c.value===selected))throw Error('收到未经证明的救赎瞥视选择。');return selected;
 }
 async function updateClaim(nonce,change){
  gm();const initial=findGlimpseClaim(game,nonce);if(!initial)throw Error('缺少救赎瞥视付款回执。');
  return withReactionReservation(initial.combatant.actor,game,async()=>{gm();const bound=findGlimpseClaim(game,nonce);if(!bound)throw Error('付款回执已改变。');const claims=structuredClone(glimpseClaims(bound.combatant)),claim=claims.find(c=>c.nonce===nonce);Object.assign(claim,change);await bound.combatant.update({[`flags.${M}.glimpseClaims`]:claims});gm();return claim});
 }
 async function publishLocal({claim,token,ability}){
  if(!owner(ability.actor,game.user)||game.user.id!==claim.userId||glimpseSourceId(ability)!==S.glimpse)throw Error('原卡拥有者或能力来源不匹配。');
  const message=await ability.toMessage(null,{create:false,actualUse:false});if(!message)throw Error('原生救赎瞥视卡未生成。');
  // The proof is first stamped after the claim has its messageId. Foundry
  // suppresses no-change updates, so pre-stamping would hide the paid-card
  // transition from updateChatMessage accounting.
  message.updateSource({speaker:{actor:ability.actor.id,scene:token.parent.id,token:token.id},[`flags.${M}.usageGenerated`]:true});
  return globalThis.ChatMessage.create(message.toObject());
 }
 async function card(option,claim,existing){
  let message=existing;
  if(!message){
   if(publishUse)message=await publishUse({...option,claim});
   else if(claim.userId===game.user.id)message=await publishLocal({...option,claim});
   else{const result=await socket?.executeAsUser('glimpse:card',claim.userId,{nonce:claim.nonce});if(!result?.ok)throw Error(result?.error??'原生反应卡的结果不确定。');message=await exactMessage(result.value)}
  }
  gm();const origin=message?.flags?.pf2e?.origin;
  if(game.messages.get(message?.id)!==message||author(message)!==claim.userId||message.rolls?.length||message.speaker?.actor!==option.actor.id||origin?.actor!==option.actor.uuid||origin.uuid!==option.ability.uuid||`Scene.${message.speaker?.scene}.Token.${message.speaker?.token}`!==option.token.uuid)throw Error('原生救赎瞥视卡不属于本次付款。');
  await updateClaim(claim.nonce,{messageId:message.id});await message.update({[`flags.${M}.glimpseUse`]:{nonce:claim.nonce,claimKey:claim.claimKey}});return message;
 }
 async function request(payload,user){
  const initial=await authenticate(payload,user);
  return queue.run(initial.token.uuid,async()=>{
   let context=await authenticate(payload,user);const key=keyOf(context.snapshot);
   if(allClaims().some(c=>c.sourceKey===key&&c.status!=='refunded'))throw Error('本次原伤害已处理或不确定，不能重复执行。');
   for(const initialOption of glimpseCandidates(context,game)){
    const decisionUser=preferred(initialOption.actor);if(!decisionUser)throw Error('神卫没有在线拥有者。');
    const expected={actorUuid:initialOption.actor.uuid,tokenUuid:initialOption.token.uuid,itemUuid:initialOption.ability.uuid,userId:decisionUser.id,combatId:initialOption.combat.id,combatantId:initialOption.combatant.id,epoch:initialOption.epoch,round:initialOption.combat.round,turn:initialOption.combat.turn};
    if(!genericReactionAvailable(initialOption.actor,boundedGame(initialOption.combat),{reactionRestriction}))continue;
    let resolveUse;const manual=new Promise(resolve=>resolveUse=resolve),pending={expected,payload,user,resolve:resolveUse};pendingUses.set(payload.scopeId,pending);
    let answer;try{answer=await Promise.race([choice({actor:initialOption.actor,user:decisionUser,title:'救赎瞥视：是否为本次盟友伤害使用反应？',choices:[{value:'use',label:'使用救赎瞥视'},{value:'decline',label:'不使用'}]}).then(value=>({value})),manual])}finally{pendingUses.delete(payload.scopeId)}
    if(answer.value==null||answer.value==='decline')continue;
    context=await authenticate(payload,user);let option=currentOption(context,expected);
    const enemyUser=preferred(context.attacker.actor);if(!enemyUser)throw Error('敌方没有在线拥有者，不能决定忏悔或抗拒。');
    const mindless=new Set(context.attacker.actor.system?.traits?.value??[]).has('mindless');
    const decision=mindless?'resist':await choice({actor:context.attacker.actor,user:enemyUser,title:'救赎瞥视：敌方选择本次结果',choices:[{value:'repent',label:'忏悔：盟友不受本次伤害'},{value:'resist',label:'抗拒：盟友获得抗力，之后自身衰弱 2'}]});
    if(!['repent','resist'].includes(decision))throw Error('敌方尚未完成忏悔／抗拒选择，伤害未应用。');
    context=await authenticate(payload,user);option=currentOption(context,expected);
    const nowMindless=new Set(context.attacker.actor.system?.traits?.value??[]).has('mindless');
    if(!owner(context.attacker.actor,enemyUser)||mindless&&!nowMindless||!mindless&&nowMindless&&decision==='repent')throw Error('敌方决定的资格已改变。');
    const nonce=random(),claim={...expected,nonce,claimKey:`glimpse:${nonce}`,decision,enemyUserId:enemyUser.id,mindless,sourceKey:key,source:context.snapshot,sourceUserId:user.id,scopeId:payload.scopeId,status:'paid'};
    await withReactionReservation(option.actor,game,async()=>{
     gm();const current=await authenticate(payload,user);option=currentOption(current,expected);
     requireReactionPermitted(option.actor,reactionRestriction);
     if(!genericReactionAvailable(option.actor,boundedGame(option.combat),{reactionRestriction}))throw Error('本次通用反应已消耗。');
     const snapshot=await reactionResources.snapshot(option.combatant);gm();const actual=await authenticate(payload,user);currentOption(actual,expected);claim.expiry=glimpseExpiryFor(actual.attacker,game);
     requireReactionPermitted(option.actor,reactionRestriction);
     if(!reactionResources.available(snapshot,'generic'))throw Error('Reaction Checker 通用反应已消耗。');
     const resource=reactionResources.reserve(snapshot,'generic');claim.resource=resource.proof;
     const previous=option.combatant.flags?.[M]?.reactionBudget,entries=previous?.epoch===claim.epoch?[...previous.entries??[]]:[];
     entries.push({type:'reaction',cost:1,slug:'glimpse-of-redemption',claimKey:claim.claimKey});
     await option.combatant.update({[`flags.${M}.reactionBudget`]:{epoch:claim.epoch,entries},[`flags.${M}.glimpseClaims`]:[...structuredClone(glimpseClaims(option.combatant)),claim],...resource.changes});gm();
    });
    try{currentOption(await authenticate(payload,user),expected);await card(option,claim,answer.message);currentOption(await authenticate(payload,user),expected)}catch(error){await updateClaim(nonce,{status:'uncertain'}).catch(onError);throw error}
    return {status:'paid',nonce};
   }
   return {status:'passed'};
  });
 }
 async function beginNative(payload,user){
  const context=await authenticate(payload,user),bound=findGlimpseClaim(game,payload.nonce),claim=bound?.claim;
  if(!claim||claim.status!=='paid'||claim.scopeId!==payload.scopeId||claim.sourceUserId!==user.id||!same(claim.source,payload.snapshot))throw Error('本次原生伤害付款不能重复进入。');currentOption(context,claim);
  await updateClaim(claim.nonce,{status:'native'});return true;
 }
 async function exactMessage(id){
  if(game.messages.get(id))return game.messages.get(id);if(!installation)throw Error('原生消息回执不存在。');
  return new Promise((resolve,reject)=>{let hook;const timer=setTimeout(()=>{installation.Hooks.off('createChatMessage',hook);reject(Error('原生消息同步超时，不能重试。'))},15000);hook=installation.Hooks.on('createChatMessage',m=>{if(m.id!==id)return;clearTimeout(timer);installation.Hooks.off('createChatMessage',hook);resolve(m)})});
 }
 async function exactClaim(nonce){
  const current=findGlimpseClaim(game,nonce)?.claim;if(current)return current;if(!installation)throw Error('救赎瞥视付款尚未同步。');
  return new Promise((resolve,reject)=>{let hook;const done=()=>{const claim=findGlimpseClaim(game,nonce)?.claim;if(!claim)return;clearTimeout(timer);installation.Hooks.off('updateCombatant',hook);resolve(claim)},timer=setTimeout(()=>{installation.Hooks.off('updateCombatant',hook);reject(Error('救赎瞥视付款同步超时；不能重复支付。'))},15000);hook=installation.Hooks.on('updateCombatant',done);done()});
 }
 function receiptMatches(message,context,claim,user){
  const pf=message?.flags?.pf2e,proof=message?.flags?.[M]?.glimpseDamage;
  return game.messages.get(message?.id)===message&&author(message)===user.id&&message.speaker?.actor===context.actor.id&&`Scene.${message.speaker?.scene}.Token.${message.speaker?.token}`===context.token.uuid&&pf?.context?.type==='damage-taken'&&pf.context.options?.includes(glimpseMarker(claim.nonce))&&pf.origin?.actor===context.attacker.actor.uuid&&pf.origin.uuid===context.item.uuid&&proof?.nonce===claim.nonce&&proof.scopeId===claim.scopeId&&proof.sourceKey===claim.sourceKey&&proof.decision===claim.decision&&!pf.appliedDamage?.isReverted&&pf.appliedDamage?.isHealing!==true&&(claim.decision!=='repent'||pf.appliedDamage==null);
 }
 async function complete(payload,user){
  const context=await authenticate(payload,user,'receipt'),claim=findGlimpseClaim(game,payload.nonce)?.claim;
  if(!claim||claim.scopeId!==payload.scopeId||claim.sourceUserId!==user.id||!same(claim.source,payload.snapshot))throw Error('本次完成回执没有确切付款。');
  if(claim.status==='done')return true;
  const p=context.proof;
  if(!payload.applied&&!p.nativeEntered&&claim.status==='paid'){
   const bound=findGlimpseClaim(game,claim.nonce);
   await withReactionReservation(bound.combatant.actor,game,async()=>{
    gm();const token=await fromUuid(claim.tokenUuid),now=glimpseEncounter(token,game),current=findGlimpseClaim(game,claim.nonce)?.claim;
    if(now.combatant!==bound.combatant||now.epoch!==claim.epoch||current?.status!=='paid')throw Error('未进入原生，但付款回合已改变；不能自动退回其他回合的反应。');
    const ledger=bound.combatant.flags?.[M]?.reactionBudget;if(ledger?.epoch!==claim.epoch)throw Error('原付款账本已改变，不能自动退款。');
    const entries=ledger.entries.filter(e=>e.claimKey!==claim.claimKey),stillSpent=entries.some(e=>e.type==='reaction');
    const refund=await reactionResources.release(bound.combatant,claim.resource,{stillSpent});gm();
    if(glimpseEncounter(token,game).epoch!==claim.epoch||bound.combatant.flags?.[M]?.reactionBudget!==ledger)throw Error('退款验证期间账本或回合已改变。');
    const claims=structuredClone(glimpseClaims(bound.combatant));Object.assign(claims.find(c=>c.nonce===claim.nonce),{status:'refunded'});
    await bound.combatant.update({[`flags.${M}.reactionBudget`]:{...ledger,entries},[`flags.${M}.glimpseClaims`]:claims,...refund});
   });return true;
  }
  if(!payload.applied||!p.nativeEntered||!p.nativeReturned||p.nonce!==claim.nonce||p.multiple||!p.messageId){await updateClaim(claim.nonce,{status:'uncertain'});throw Error('缺少唯一真实原生伤害回执；保持已付反应，不重试。');}
  const message=await exactMessage(p.messageId);gm();
  if(!receiptMatches(message,context,claim,user)){await updateClaim(claim.nonce,{status:'uncertain'});throw Error('原生伤害回执来源不一致。');}
  if(claim.status!=='native')throw Error('本次后续已认领或不确定，不能重复创建条件。');
  if(claim.decision==='resist'){
   await updateClaim(claim.nonce,{status:'followup',receiptId:message.id});
   try{
    const result=await compat.apply({nonce:claim.nonce,enemy:context.attacker,expiry:claim.expiry,authorize:async()=>{gm();const current=findGlimpseClaim(game,claim.nonce)?.claim;return current?.status==='followup'&&current.receiptId===message.id&&receiptMatches(message,context,current,user)}});
    await updateClaim(claim.nonce,{status:'done',receiptId:message.id,...result});
   }catch(error){await updateClaim(claim.nonce,{status:'uncertain'}).catch(onError);throw error}
  }else await updateClaim(claim.nonce,{status:'done',receiptId:message.id});return true;
 }
 async function rpc(method,payload){
  if(isActiveGM(game))return ({request,complete,native:beginNative})[method](payload,game.user);
  const leader=game.users.activeGM?.id;if(!socket||!leader)throw Error('救赎瞥视需要在线主GM。');const r=await socket.executeAsUser(`glimpse:${method}`,leader,payload);
  if(game.users.activeGM?.id!==leader||!r?.ok)throw Error(r?.error??'救赎瞥视结算不确定。');return r.value;
 }
 function potential(params){const token=params.token?.document??params.token;return values(token?.parent?.tokens).some(t=>t.actor?.uuid!==token.actor?.uuid&&handlesActor(t.actor)&&t.auras?.get('champions-aura')?.containsToken?.(token)===true)}
 async function beforeDamage(actor,params){
  if(!ready()||!potential(params)||params.final||params.skipIWR||typeof params.damage==='number'&&params.damage<=0||params.damage?.total<=0)return {params};
  const binding=await resolveGlimpseSource({game,fromUuid,actor,params,source:getRollContext?.(params.damage)});
  // A supported upstream damage operation can contain multiple attacks. Skip
  // only our unbound automation before payment; never block its native damage
  // or treat the combined roll as one authenticated reaction source.
  if(!binding.verified&&binding.unsupportedReason==='combined-attacks-unsupported'){onUnsupported(binding);return {params};}
  if(!binding.verified)throw markUnappliedDamageError(Error(`救赎瞥视无法证明原伤害来源：${binding.unsupportedReason}。请明确来源后结算。`));
  const p={scopeId:random(),snapshot:structuredClone(binding.snapshot),actor,damage:params.damage,damageEvidence:JSON.stringify(params.damage.toJSON()),outcome:params.outcome,token:binding.token,item:binding.item,userId:game.user.id,leader:game.users.activeGM?.id,state:'waiting'};plans.add(p);live.set(p.scopeId,p);
  try{
   const result=await rpc('request',{scopeId:p.scopeId,snapshot:p.snapshot});
   if(result.status!=='paid'){live.delete(p.scopeId);plans.delete(p);return {params}}
   const claim=await exactClaim(result.nonce);if(!claim||claim.status!=='paid'||claim.scopeId!==p.scopeId||claim.sourceUserId!==p.userId||!same(claim.source,p.snapshot))throw Error('本次救赎瞥视付款未同步，不能继续原生伤害。');
   p.claim=claim;const actual=claim.decision==='repent'?repentParams(params,claim.nonce):{...params,rollOptions:new Set([...params.rollOptions??[],glimpseMarker(claim.nonce)])};actual[brand]=p;
   return {params:actual,receipt:p};
  }catch(error){live.delete(p.scopeId);plans.delete(p);throw markUnappliedDamageError(error)}
 }
 async function wrapNativeDamage(actor,params,native){
  const p=params?.[brand];if(!plans.has(p))return native(params);
  try{
   if(p.state!=='waiting'||p.leader!==game.users.activeGM?.id||game.user.id!==p.userId||actor.uuid!==p.actor.uuid||(params.token?.document??params.token)!==p.token||params.item?.uuid!==p.item.uuid||params.outcome!==p.outcome||!new Set(params.rollOptions??[]).has(glimpseMarker(p.claim.nonce))||JSON.stringify(p.damage.toJSON())!==p.damageEvidence||p.claim.decision==='resist'&&(params.damage!==p.damage||params.final||params.skipIWR)||p.claim.decision==='repent'&&(params.damage!==0||params.final!==true||params.shieldBlockRequest!==false))throw Error('最终原生伤害范围已改变。');
   const context=await verified(p.snapshot),option=currentOption(context,p.claim);let resistance;
   if(game.pf2e?.settings?.iwr===false)throw Error('原生IWR已关闭，不能应用救赎瞥视抗力。');
   if(p.claim.decision==='resist')resistance=compileGlimpseResistance({actor,template:compat.template(),champion:option.actor,ability:option.ability,nonce:p.claim.nonce,options:params.rollOptions});
   await rpc('native',{scopeId:p.scopeId,snapshot:p.snapshot,nonce:p.claim.nonce});currentOption(await verified(p.snapshot),p.claim);
   if(p.leader!==game.users.activeGM?.id)throw Error('原生调用前主GM已改变。');
   p.state='native';p.nativeEntered=true;
   const result=resistance?await withGlimpseResistance(actor,resistance,()=>native(params)):await native(params);p.nativeReturned=true;return result;
  }catch(error){if(!p.nativeEntered)throw markUnappliedDamageError(error);throw error}
 }
 function decorate(message,_data,_options,userId){
  if(userId!==game.user.id||author(message)!==game.user.id||message.flags?.pf2e?.context?.type!=='damage-taken')return;
  const opts=new Set(message.flags.pf2e.context.options??[]),matches=[...live.values()].filter(p=>p.state==='native'&&p.nativeEntered&&p.claim&&opts.has(glimpseMarker(p.claim.nonce))&&message.speaker?.actor===p.actor.id&&`Scene.${message.speaker?.scene}.Token.${message.speaker?.token}`===p.token.uuid);
  if(matches.length!==1)return;const p=matches[0],receiptNonce=random();message.updateSource({[`flags.${M}.glimpseDamage`]:{nonce:p.claim.nonce,scopeId:p.scopeId,sourceKey:keyOf(p.snapshot),decision:p.claim.decision,receiptNonce}});p.creations??=new Set();p.creations.add(receiptNonce);
 }
 function capture(message,_options,userId){const proof=message.flags?.[M]?.glimpseDamage,p=live.get(proof?.scopeId);if(!p||p.state!=='native'||!p.nativeEntered||userId!==p.userId||author(message)!==p.userId||!p.creations?.delete(proof.receiptNonce))return;if(p.messageId&&p.messageId!==message.id)p.multiple=true;else p.messageId=message.id;}
 async function afterDamage(p,{applied}){if(!plans.has(p))return;try{await rpc('complete',{scopeId:p.scopeId,snapshot:p.snapshot,nonce:p.claim.nonce,applied})}finally{live.delete(p.scopeId);plans.delete(p)}}
 const resolveAction=item=>handlesActor(item?.actor)&&glimpseSourceId(item)===S.glimpse?'glimpse:use':undefined;
 async function executeUsage({actor,item,message,user}){
  gm();if(!resolveAction(item)||item.actor!==actor||!owner(actor,user)||game.messages.get(message?.id)!==message||author(message)!==user.id||message.flags?.[M]?.usageInput?.actualUse!==true||message.flags?.pf2e?.origin?.uuid!==item.uuid||message.rolls?.length)throw Error('需要确切原生救赎瞥视 Use 卡。');
  const matches=[...pendingUses.values()].filter(p=>p.expected.actorUuid===actor.uuid&&p.expected.userId===user.id&&p.expected.itemUuid===item.uuid);
  if(matches.length!==1)throw Error('没有唯一等待中的原伤害；请在该伤害应用时使用救赎瞥视。');
  const p=matches[0];currentOption(await authenticate(p.payload,p.user),p.expected);if(pendingUses.get(p.payload.scopeId)!==p)throw Error('该原伤害的选择范围已关闭。');p.resolve({value:'use',message});return '已关联本次原伤害，正在等待敌方选择。';
 }
 function register({Hooks,socket:api}={}){
  if(installation)return;socket=api;const ids=[['preCreateChatMessage',Hooks.on('preCreateChatMessage',decorate)],['createChatMessage',Hooks.on('createChatMessage',capture)]];installation={Hooks,ids};
  const handle=(name,fn)=>socket?.register(`glimpse:${name}`,async function(payload){try{return {ok:true,value:await fn(payload,game.users.get(this.socketdata.userId))}}catch(error){return {ok:false,error:error.message}}});
  for(const [name,fn]of [['request',request],['native',beginNative],['complete',complete]])handle(name,fn);
  handle('scope',(payload,user)=>{if(user?.id!==game.users.activeGM?.id)throw Error('只有主GM可验证真实原生范围。');return scopeProof(payload,game.user)});
  handle('choice',async(payload,user)=>{const actor=await fromUuid(payload.actorUuid);if(user?.id!==game.users.activeGM?.id||!owner(actor,game.user))throw Error('选择请求者或实际角色拥有者无效。');return show({title:payload.title,choices:validateNativeChoices(payload.choices)})});
  handle('card',async(payload,user)=>{if(user?.id!==game.users.activeGM?.id)throw Error('仅当前主GM可请求原卡。');const claim=await exactClaim(payload.nonce);if(claim?.status!=='paid'||claim.userId!==game.user.id)throw Error('原始反应卡缺少本次付款。');const token=await fromUuid(claim.tokenUuid),ability=await fromUuid(claim.itemUuid);return (await publishLocal({claim,token,ability})).id});
 }
 function unregister(){if(installation)for(const [name,id]of installation.ids)installation.Hooks.off(name,id);installation=null;live.clear();pendingUses.clear()}
 return {ready,handlesActor,beforeDamage,afterDamage,wrapNativeDamage,register,unregister,resolveAction,requiresActualUse:item=>!!resolveAction(item),executeUsage};
}
