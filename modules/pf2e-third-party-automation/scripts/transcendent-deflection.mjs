import {MODULE_ID} from './rules.mjs';
import {SerialActions} from './runtime.mjs';
import {isActiveGM} from './native-context.mjs';
import {genericReactionAvailable,reactionEpoch,withReactionReservation} from './reaction-budget.mjs';
import {deflectionFeat,deflectionBroken,getTranscendentDeflectionOptions,getDeflectionSwapWeapons} from './transcendent-deflection-rules.mjs';
import {isCurrentDisruptToken} from './disrupt-prey-rules.mjs';
import {resolveDeflectionSource,validateDeflectionSource} from './transcendent-deflection-source.mjs';
import {createDeflectionWeapons} from './transcendent-deflection-weapons.mjs';
import {reactionPermitted} from './reaction-restriction.mjs';
const values=c=>Array.from(c?.values?.()??c??[]),own=a=>a?.flags?.[MODULE_ID]?.transcendentDeflection??{},author=m=>m?.author?.id??m?.user?.id??m?.user;
const random=()=>globalThis.foundry?.utils?.randomID?.(24)??crypto.randomUUID(),brand=Symbol('native-deflection-plan');
const sourceKey=s=>`${s.damageMessageId}:${s.rollIndex}:${s.tokenUuid}`;
const claimFor=(a,nonce)=>own(a).reactions?.find(c=>c.nonce===nonce);
const application=(a,key)=>own(a).applications?.find(c=>c.sourceKey===key);
const marker=nonce=>`${MODULE_ID}:transcendent-deflection:${nonce}`;
const same=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
const validShield=s=>s===null||!!(s&&typeof s.id==='string'&&s.id&&Number.isFinite(s.damage)&&s.damage>0&&Object.keys(s).length===2);
const sameShield=(a,b)=>validShield(a)&&validShield(b)&&(a===null?b===null:b!==null&&a.id===b.id&&a.damage===b.damage);

/** Provider for the one native damage pipeline. The source-client scope proves
 * a real waiting post-IWR invocation; a UUID, roll option or RPC alone cannot. */
export function createTranscendentDeflection({game,reactionRestriction,fromUuid=globalThis.fromUuid,choose,getRollContext,nativeBridgeAvailable=()=>false,weapons=createDeflectionWeapons({game}),onError=console.error}={}){
 const plans=new WeakSet(),live=new Map(),queue=new SerialActions();let socket,installation;
 const gm=()=>{if(!isActiveGM(game))throw Error('靖涛定风剑主GM已改变。')};
 const owner=(actor,user)=>!!(user?.active&&actor?.testUserPermission?.(user,'OWNER'));
 const selectedOwner=actor=>{const users=values(game.users).filter(u=>owner(actor,u));return users.find(u=>!u.isGM&&(u.character?.uuid===actor.uuid||u.character?.id===actor.id))??users.find(u=>!u.isGM)??users.find(u=>u.id===game.users.activeGM?.id)};
 const now=()=>{const n=game.time?.worldTime;if(!Number.isFinite(n))throw Error('缺少原生世界时间，不能判定十分钟冷却。');return n};
 const encounter=()=>game.combat?.started?game.combat.id:null;
 const canPay=actor=>reactionPermitted(actor,reactionRestriction)&&deflectionFeat(actor)&&(deflectionFeat(actor).system?.frequency?.value??1)>0&&now()>=(own(actor).availableAt??-Infinity)&&(!encounter()||genericReactionAvailable(actor,game));
 async function save(actor,change){await withReactionReservation(actor,game,async()=>{gm();const state=structuredClone(own(actor));state.reactions??=[];state.applications??=[];change(state);await actor.update({[`flags.${MODULE_ID}.transcendentDeflection`]:state});gm()})}
 async function verified(snapshot){const result=await validateDeflectionSource({game,fromUuid,snapshot});gm();if(!result.verified)throw Error(`攻击伤害来源已改变：${result.unsupportedReason}`);return result}
 function scopeProof(payload,user){
  const scope=live.get(payload?.scopeId);
  if(!scope||scope.state!=='waiting'||scope.userId!==user?.id||scope.leader!==game.users.activeGM?.id||!same(scope.snapshot,payload.snapshot)||!owner(scope.actor,user))throw Error('没有这个拥有者仍在等待的真实原生伤害调用。');
  return {scopeId:scope.scopeId,snapshot:scope.snapshot,incoming:scope.incoming,persistent:scope.persistent,shield:scope.shield};
 }
 async function authenticate(payload,user){
  gm();const context=await verified(payload?.snapshot);if(!owner(context.actor,user))throw Error('伤害申请者不是实际受害者的拥有者。');
  const proof=user.id===game.user.id?scopeProof(payload,user):await socket?.executeAsUser('deflection:scope',user.id,{scopeId:payload.scopeId,snapshot:payload.snapshot});gm();
  const actual=user.id===game.user.id?proof:proof?.ok?proof.value:null;
  if(!actual||actual.scopeId!==payload.scopeId||!same(actual.snapshot,payload.snapshot)||!Number.isFinite(actual.incoming)||actual.incoming<0||!Array.isArray(actual.persistent)||!validShield(actual.shield))throw Error('原生IWR调用范围验证失败。');
  return {...context,incoming:actual.incoming,persistent:actual.persistent,shield:actual.shield};
 }
 function candidates(context){
  const result=[];
  for(const token of values(context.token.parent.tokens)){
   const actor=token.actor,user=selectedOwner(actor);if(!user||!canPay(actor))continue;
   const options=getTranscendentDeflectionOptions({game,actor,token,attacker:context.attacker,victim:context.token});
   for(const option of options)result.push({actor,token,user,...option,value:`${token.uuid}::${option.weaponUuid}`});
  }return result;
 }
 function assertPaidChoice(option,context,claim,{broken=false}={}){
  const {actor,token,user}=option,weapon=actor.items.get(option.weapon.id);
  if(!owner(actor,user)||!isCurrentDisruptToken(token,game)||token.actor!==actor||token.parent!==context.token.parent||reactionEpoch(actor,game)!==claim.epoch||encounter()!==claim.encounter||!deflectionFeat(actor)||actor.canAct===false||actor.isDead||actor.hasCondition?.('unconscious')||!actor.isEnemyOf?.(context.attacker.actor)||actor.uuid!==context.actor.uuid&&!actor.isAllyOf?.(context.actor))throw Error('支付期间反应拥有者、触及或角色状态已改变。');
  if(!broken){if(!getTranscendentDeflectionOptions({game,actor,token,attacker:context.attacker,victim:context.token}).some(o=>o.weaponUuid===claim.weaponUuid))throw Error('支付期间武器或触及已改变。');return;}
  if(weapon?.uuid!==claim.weaponUuid||deflectionBroken(weapon)?.nonce!==claim.nonce||weapon.system.equipped?.carryType!=='held'||!(weapon.system.equipped.handsHeld>0))throw Error('破损后的确切武器持握已改变。');
  const strike=values(actor.system?.actions).flatMap(s=>[s,...values(s.altUsages)]).find(s=>s.item?.uuid===weapon.uuid&&(s.item.altUsageType??null)===option.usage),item=strike?.item;
  const reach=item&&actor.getReach?.({action:'attack',weapon:item}),distance=token.object.distanceTo?.(context.attacker.object,{reach}),point=context.attacker.getCenterPoint?.();
  if(!item?.isMelee||item.isRanged||!Number.isFinite(reach)||!Number.isFinite(distance)||distance<0||distance>reach||!point||token.object.checkCollision?.(point,{type:'move',mode:'any'})!==false)throw Error('破损武器期间攻击者位置或反应触及已改变。');
 }
 async function request(payload,user){
  gm();const first=await authenticate(payload,user),key=sourceKey(first.snapshot);
  return queue.run(first.actor.uuid,async()=>{
   const context=await authenticate(payload,user);
   if(application(context.actor,key))throw Error('本次攻击伤害已处理或结果不确定，不会重复执行。');
   if(context.incoming===0&&!context.persistent.length)return {status:'no-damage'};
   const initial=candidates(context);if(!initial.length)return {status:'ineligible'};
   await save(context.actor,state=>state.applications.push({sourceKey:key,scopeId:payload.scopeId,sourceUserId:user.id,snapshot:context.snapshot,status:'choosing'}));
   for(const actor of [...new Set(initial.map(c=>c.actor))]){
    let current=await authenticate(payload,user),options=candidates(current).filter(c=>c.actor.uuid===actor.uuid);
    if(!options.length)continue;
    const chosenUser=options[0].user,epoch=reactionEpoch(actor,game),encounterId=encounter(),selected=await choose({actor,user:chosenUser,title:'靖涛定风剑：选择反应武器',choices:[...options.map(c=>({value:c.value,label:`${c.token.name??actor.name??actor.id}：${c.weapon.name}`})),{value:'decline',label:'不使用此反应'}]});gm();
    if(selected==null||selected==='decline')continue;
    current=await authenticate(payload,user);const option=candidates(current).find(c=>c.actor.uuid===actor.uuid&&c.value===selected);
    if(!option||option.user.id!==chosenUser.id||reactionEpoch(actor,game)!==epoch||encounter()!==encounterId)throw Error('选择期间武器、触及、拥有者或反应回合已改变。');
    const nonce=random(),claim={nonce,claimKey:`deflect:${nonce}`,actorUuid:actor.uuid,tokenUuid:option.token.uuid,weaponUuid:option.weaponUuid,usage:option.usage,userId:chosenUser.id,epoch,encounter:encounterId,sourceKey:key,sourceUserId:user.id,scopeId:payload.scopeId,source:current.snapshot,state:'claimed',usedAt:now(),incoming:current.incoming,persistentPrevented:current.persistent.length,shield:structuredClone(current.shield)};
    await withReactionReservation(actor,game,async()=>{
     gm();if(!canPay(actor)||reactionEpoch(actor,game)!==epoch)throw Error('本次反应或十分钟频次已经消耗。');
     const state=structuredClone(own(actor));state.reactions??=[];state.reactions.push(claim);state.availableAt=claim.usedAt+600;
     await actor.update({[`flags.${MODULE_ID}.transcendentDeflection`]:state});gm();
    });
    await save(current.actor,state=>Object.assign(state.applications.find(a=>a.sourceKey===key),{status:'claimed',reactorUuid:actor.uuid,nonce}));
    assertPaidChoice(option,await authenticate(payload,user),claim);await weapons.breakWeapon({actor,weapon:option.weapon,claim});gm();assertPaidChoice(option,await authenticate(payload,user),claim,{broken:true});
    await save(actor,state=>Object.assign(state.reactions.find(c=>c.nonce===nonce),{state:'prevented'}));
    return {status:'prevented',proof:{nonce,actorUuid:actor.uuid,weaponUuid:option.weaponUuid,claimKey:claim.claimKey,sourceDamageMessageId:current.snapshot.damageMessageId,sourceRollIndex:current.snapshot.rollIndex,targetTokenUuid:current.token.uuid}};
   }
   await save(context.actor,state=>Object.assign(state.applications.find(a=>a.sourceKey===key),{status:'passed'}));return {status:'passed'};
  });
 }
 async function rpc(method,payload){
  if(isActiveGM(game))return method==='request'?request(payload,game.user):complete(payload,game.user);
  const leader=game.users.activeGM?.id;if(!socket||!leader)throw Error('靖涛定风剑需要在线主GM。');
  const result=await socket.executeAsUser(`deflection:${method}`,leader,payload);
  if(game.users.activeGM?.id!==leader||!result?.ok)throw Error(result?.error??'反应结算不确定。');return result.value;
 }
 async function beforeDamage(actor,params){
  if(!nativeBridgeAvailable()||!values((params.token?.document??params.token)?.parent?.tokens).some(t=>deflectionFeat(t.actor)))return {params};
  const source=getRollContext?.(params.damage),binding=await resolveDeflectionSource({game,fromUuid,actor,params,source});if(!binding.verified)return {params};
  const plan={scopeId:random(),actor,source:snapshotCopy(binding.snapshot),damage:params.damage,token:params.token?.document??params.token,userId:game.user.id,leader:game.users.activeGM?.id,state:'prepared'};plans.add(plan);
  return {params:{...params,[brand]:plan},receipt:plan};
 }
 async function interceptNative({actor,params,incoming,persistent,shield,prevent}){
  const plan=params?.[brand];if(!plans.has(plan))return;
  if(plan.state!=='prepared'||actor.uuid!==plan.actor.uuid||params.damage!==plan.damage||(params.token?.document??params.token)?.uuid!==plan.token.uuid||game.user.id!==plan.userId||game.users.activeGM?.id!==plan.leader)throw Error('本次原生防伤范围已改变或重复进入。');
  if(!Number.isFinite(incoming)||incoming<0||!Array.isArray(persistent)||!validShield(shield))throw Error('缺少本次原生IWR及硬度后结果。');
  if(incoming===0&&!persistent.length){plan.state='no-damage';return;}
  Object.assign(plan,{state:'waiting',snapshot:plan.source,incoming,persistent:structuredClone(persistent),shield:structuredClone(shield)});live.set(plan.scopeId,plan);
  try{
   const result=await rpc('request',{scopeId:plan.scopeId,snapshot:plan.source});
   if(result.status==='prevented'){
    const context=await validateDeflectionSource({game,fromUuid,snapshot:plan.source}),reactor=await fromUuid(result.proof?.actorUuid),claim=claimFor(reactor,result.proof?.nonce),token=claim&&await fromUuid(claim.tokenUuid),weapon=reactor?.items?.get(claim?.weaponUuid?.split('.').at(-1));
    if(!context.verified||!claim||claim.state!=='prevented'||claim.weaponState!=='broken'||claim.claimKey!==result.proof.claimKey||claim.weaponUuid!==result.proof.weaponUuid||claim.sourceKey!==sourceKey(plan.source)||claim.scopeId!==plan.scopeId||claim.sourceUserId!==plan.userId||!same(claim.source,plan.source)||!sameShield(claim.shield,plan.shield))throw Error('最终原生防伤授权或来源已改变。');
    assertPaidChoice({actor:reactor,token,user:game.users.get(claim.userId),weapon,usage:claim.usage},context,claim,{broken:true});
    if(plan.leader!==game.users.activeGM?.id||game.user.id!==plan.userId)throw Error('防伤前拥有者或主GM已改变。');
    plan.proof=result.proof;prevent({marker:marker(result.proof.nonce),flagKey:'transcendentDeflection',proof:result.proof});plan.state='prevented';
   }else plan.state=result.status;
  }catch(error){plan.state='uncertain';throw error}
  finally{live.delete(plan.scopeId)}
 }
 function capture(message,_options,userId){
  const proof=message?.flags?.[MODULE_ID]?.transcendentDeflection;if(proof?.kind!=='damage-prevented')return;
  // Pending completed-native plans remain indexed separately from live scopes;
  // no scan of recent chat messages is used to infer an application.
  const plan=receipts.get(proof.nonce);if(!plan||author(message)!==plan.userId||userId!==plan.userId)return;
  if(plan.messageId&&plan.messageId!==message.id){plan.multiple=true;return;}plan.messageId=message.id;
 }
 const receipts=new Map();
 // Capture the real card after interceptNative returned and before afterDamage.
 const intercepted=async context=>{await interceptNative(context);const p=context.params?.[brand];if(plans.has(p)&&p.proof)receipts.set(p.proof.nonce,p)};
 async function exactMessage(id){
  if(game.messages.get(id))return game.messages.get(id);
  if(!installation?.Hooks)throw Error('缺少真实防伤回执。');
  return new Promise((resolve,reject)=>{let hook;const timer=setTimeout(()=>{installation.Hooks.off('createChatMessage',hook);reject(Error('真实防伤回执同步超时；不重复执行。'))},15000);hook=installation.Hooks.on('createChatMessage',m=>{if(m.id!==id)return;clearTimeout(timer);installation.Hooks.off('createChatMessage',hook);resolve(m)})});
 }
 async function complete(payload,user){
  gm();const context=await verified(payload.snapshot),key=sourceKey(context.snapshot);if(!owner(context.actor,user))throw Error('防伤回执拥有者无效。');
  const app=application(context.actor,key);if(!app||app.scopeId!==payload.scopeId||app.sourceUserId!==user.id)throw Error('没有本次防伤认领。');
  if(app.status==='done')return true;
  if(app.status==='passed'){if(!payload.applied)throw Error('原生伤害完成状态不确定。');await save(context.actor,s=>Object.assign(s.applications.find(a=>a.sourceKey===key),{status:'done'}));return true;}
  const actor=await fromUuid(app.reactorUuid);gm();const claim=claimFor(actor,app.nonce);
  if(!claim||claim.sourceKey!==key||claim.scopeId!==payload.scopeId||claim.sourceUserId!==user.id)throw Error('防伤武器认领已改变。');
  if(!payload.messageId||payload.multiple)throw Error('没有唯一真实防伤回执；保持反应已支付。');
  const message=await exactMessage(payload.messageId);gm();const pf=message.flags?.pf2e,proof=message.flags?.[MODULE_ID]?.transcendentDeflection;
  if(game.messages.get(message.id)!==message||author(message)!==user.id||message.speaker?.actor!==context.actor.id||`Scene.${message.speaker?.scene}.Token.${message.speaker?.token}`!==context.token.uuid||pf?.context?.type!=='damage-taken'||!pf.context.options?.includes(marker(claim.nonce))||proof?.kind!=='damage-prevented'||proof.nonce!==claim.nonce||proof.claimKey!==claim.claimKey||proof.actorUuid!==actor.uuid||proof.weaponUuid!==claim.weaponUuid||proof.sourceDamageMessageId!==context.snapshot.damageMessageId||proof.sourceRollIndex!==context.snapshot.rollIndex||proof.targetTokenUuid!==context.token.uuid||!Number.isFinite(proof.incoming)||proof.incoming<0||!Number.isInteger(proof.persistentPrevented)||proof.persistentPrevented<0||pf.appliedDamage?.isReverted)throw Error('原生防伤回执来源不一致。');
  if(proof.incoming!==claim.incoming||proof.persistentPrevented!==claim.persistentPrevented||!sameShield(proof.shield,claim.shield))throw Error('原生防伤回执与本次IWR及硬度后结果不一致。');
  const applied=pf.appliedDamage;
  if(claim.shield===null?applied!=null:!applied||applied.uuid!==context.actor.uuid||applied.isHealing!==false||!Array.isArray(applied.updates)||applied.updates.length||!Array.isArray(applied.persistent)||applied.persistent.length||!sameShield(applied.shield,claim.shield))throw Error('原生防伤回执仍含角色伤害或盾牌结算不一致。');
  await save(actor,s=>Object.assign(s.reactions.find(c=>c.nonce===claim.nonce),{state:'done',receiptId:message.id}));
  await save(context.actor,s=>Object.assign(s.applications.find(a=>a.sourceKey===key),{status:'done',receiptId:message.id}));
  const weapon=actor.items.get(claim.weaponUuid.split('.').at(-1)),options=getDeflectionSwapWeapons(actor),userChoice=game.users.get(claim.userId);
  if(options.length&&owner(actor,userChoice)){
   const selected=await choose({actor,user:userChoice,title:'靖涛定风剑：是否立即换持武器？',choices:[...options.map(w=>({value:w.uuid,label:w.name})),{value:'decline',label:'不换持'}]});gm();
   if(selected&&selected!=='decline'){const replacement=getDeflectionSwapWeapons(actor).find(w=>w.uuid===selected);if(!replacement)throw Error('换持武器已经改变。');await weapons.swapWeapon({actor,weapon,claim,replacement})}
  }return true;
 }
 async function afterDamage(plan,{applied,uncertain}){
  if(!plans.has(plan))return;
  try{if(['passed','prevented','uncertain'].includes(plan.state))await rpc('complete',{scopeId:plan.scopeId,snapshot:plan.source,applied,uncertain,messageId:plan.messageId,multiple:plan.multiple})}
  finally{if(plan.proof)receipts.delete(plan.proof.nonce);plans.delete(plan);live.delete(plan.scopeId)}
 }
 function register({Hooks,socket:api}={}){
  if(installation)return unregister;socket=api;const id=Hooks.on('createChatMessage',capture);installation={Hooks,id};
  for(const [name,fn]of [['request',request],['complete',complete]])socket?.register(`deflection:${name}`,async function(payload){try{return {ok:true,value:await fn(payload,game.users.get(this.socketdata.userId))}}catch(error){return {ok:false,error:error.message}}});
  socket?.register('deflection:scope',function(payload){try{if(this.socketdata.userId!==game.users.activeGM?.id)throw Error('只有当前主GM可验证原生范围。');return {ok:true,value:scopeProof(payload,game.user)}}catch(error){return {ok:false,error:error.message}}});return unregister;
 }
 function unregister(){if(installation)installation.Hooks.off('createChatMessage',installation.id);installation=null;live.clear();receipts.clear()}
 const hasNativePlan=(actor,params)=>{const plan=params?.[brand];return plans.has(plan)&&plan.actor?.uuid===actor?.uuid&&plan.damage===params.damage&&plan.token.uuid===(params.token?.document??params.token)?.uuid};
 return {beforeDamage,hasNativePlan,interceptNative:intercepted,afterDamage,register,unregister,wrapStrike:weapons.wrapStrike};
}
const snapshotCopy=structuredClone;
