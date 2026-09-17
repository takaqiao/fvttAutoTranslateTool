import {MODULE_ID,hasSource} from './rules.mjs';
import {SerialActions} from './runtime.mjs';
import {reactionEpoch} from './reaction-budget.mjs';
import {markUnappliedDamageError} from './native-context.mjs';
import {NATIVE_IWR_PROFILES,NATIVE_IWR_BRIDGE_PROTOCOL} from './native-iwr-profiles.mjs';
import {isVerifiedNativeIWRBridge} from './native-iwr-verification.mjs';

export const DESTRUCTIVE_BLOCK_SOURCE='Compendium.pf2e.feats-srd.Item.dY0jhJOEj6DHc0ud';
const INDESTRUCTIBLE_SHIELD='Compendium.pf2e.equipment-srd.Item.SUbYk6B1iPoGyyjh';
const prefix=`${MODULE_ID}:destructive-block:`,nativePrefix=`${MODULE_ID}:native-shield:`;
export {NATIVE_IWR_BRIDGE_PROTOCOL};
const finite=value=>Number.isFinite(value)&&value>=0;
export function destructiveBlockAmounts({incoming,shieldHardness,shieldHP,actorHardness=0}){
 if(![incoming,shieldHardness,shieldHP,actorHardness].every(finite))throw Error('破坏性格挡的伤害、硬度或盾牌生命值无效。');
 const absorbedDamage=Math.min(incoming,shieldHardness*2),actorHardnessReduction=Math.min(incoming-absorbedDamage,actorHardness);
 return {incoming,absorbedDamage,actorHardnessReduction,actorDamage:incoming-absorbedDamage-actorHardnessReduction,shieldDamage:Math.min(shieldHP,Math.max(0,incoming*2-shieldHardness))};
}

/** Match native PF2e 8.5 actor hardness, including the attack's adamantine grade. */
export function effectiveActorHardness(hardness,params){
 const item=params.item,adamantine=typeof params.damage!=='number'&&params.damage?.materials?.has?.('adamantine');
 const weapon=item?.isOfType?.('weapon')??item?.type==='weapon';
 const grade=weapon&&item.system?.material?.type==='adamantine'?item.system.material.grade??'standard':'standard';
 const threshold={low:0,standard:10,high:13}[grade];
 return params.final?0:adamantine&&threshold>=hardness?Math.floor(hardness/2):hardness;
}

function renderBlockContent({data,calculation,canUndo,shieldState}){
 const doc=new DOMParser().parseFromString(data.content,'text/html'),section=doc.querySelector('section.damage-taken');
 if(!section)throw Error('原生伤害卡结构不匹配，无法显示破坏性格挡结果。');
 const detail=doc.createElement('span');detail.className='destructive-block-result';
 const condition=shieldState.destroyed?'（已摧毁）':shieldState.broken?'（已破损）':'';
 detail.textContent=` 破坏性格挡抵挡 ${calculation.absorbedDamage} 点伤害；盾牌受到 ${calculation.shieldDamage} 点伤害${condition}。`;
 const statements=section.querySelector('.statements');if(statements)statements.append(detail);else section.prepend(detail);
 if(calculation.actorHardnessReduction>0){
  const reduction=doc.createElement('span');reduction.textContent=` 自身硬度另抵挡 ${calculation.actorHardnessReduction} 点伤害。`;detail.append(reduction);
 }
 if(canUndo&&!section.querySelector('[data-action="revertDamage"]')){
  const button=doc.createElement('button');button.type='button';button.className='revert-damage';button.dataset.action='revertDamage';button.dataset.tooltip='PF2E.RevertDamage.ButtonTooltip';button.dataset.tooltipDirection='UP';
  const icon=doc.createElement('i');icon.className='fa-solid fa-rotate-left';button.append(icon);section.append(button);
 }
 return doc.body.innerHTML;
}

/** Keep native IWR, HP, persistent damage and undo; vary only this block's deltas. */
export function createShieldDamageAdapter({game,createMessageMiddleware,nativeBridgeVerification,onError=console.error,renderContent=renderBlockContent,getNativeBridge=()=>globalThis.CONFIG?.Actor?.documentClass?.thirdPartyNativeIWRBridge}={}){
 const queue=new SerialActions(),actors=new WeakMap(),nonces=new Map(),frames=new WeakMap(),nativeFrames=new WeakMap(),preventions=new Map(),interceptors=new Map();let registered=false;
 const report=error=>{try{onError(error)}catch{/* Preserve an already-applied native result. */}};
 function nativeBridgeAvailable(){
  try{return isVerifiedNativeIWRBridge(nativeBridgeVerification,game,getNativeBridge());}catch{return false;}
 }
 function nativeBridgeDiagnostic(){
  const ready=nativeBridgeAvailable(),version=game.system?.version??null,profile=Object.hasOwn(NATIVE_IWR_PROFILES,version)?NATIVE_IWR_PROFILES[version]:null;
  let bridge;try{bridge=getNativeBridge();}catch{/* Availability failures must not block ordinary damage. */}
  const field=key=>bridge?Object.getOwnPropertyDescriptor(bridge,key)?.value??null:null;
  const reason=ready?null:game.system?.id!=='pf2e'||!profile?'unsupported-system-version':nativeBridgeVerification?.ready===false?nativeBridgeVerification.reason:nativeBridgeVerification?.ready===true&&nativeBridgeVerification.bridge?'native-iwr-proof-invalidated':'native-iwr-verification-required';
  return Object.freeze({ready,status:ready?'ready':'unsupported',reason,systemVersion:version,bridgeVersion:field('version'),protocol:field('protocol'),expectedSourceSHA256:profile?.originalSHA256??null,expectedPatchedSHA256:profile?.patchedSHA256??null,verifiedSystemSHA256:ready?nativeBridgeVerification.patchedSHA256:null,verifiedApplyDamageSHA256:ready?nativeBridgeVerification.applyDamageSHA256:null,requiresVerifiedSystemBridge:!ready,unavailableFeatures:Object.freeze(ready?[]:['transcendent-deflection'])});
 }
 const addNativeInterceptor=(callback,{matches=()=>true}={})=>{if(typeof callback!=='function'||typeof matches!=='function')throw Error('原生伤害拦截器及匹配器必须是函数。');interceptors.set(callback,matches);return()=>interceptors.delete(callback);};
 const healthState=actor=>{const hp=actor?.hitPoints,sp=actor?.attributes?.hp?.sp,shield=actor?.heldShield;return JSON.stringify([hp?.value,hp?.max,hp?.temp,sp?.value,sp?.max,shield?.id,shield?._source?.system?.hp?.value]);};
 function checkFrame(frame,actor,params){
  if(!frame.open||frames.get(frame.actor)!==frame||actor!==frame.actor||params!==frame.nativeParams||game.user!==frame.user||frame.user.isGM&&game.users?.activeGM?.id!==frame.user.id||!frame.actor.testUserPermission?.(frame.user,'OWNER'))throw Error('原生伤害角色、操作者或私有上下文已失效。');
  if(['damage','token','item'].some(key=>params[key]!==frame.source[key])||['final','skipIWR','shieldBlockRequest'].some(key=>!!params[key]!==!!frame.source[key])||(typeof params.damage==='number'?params.damage:params.damage?.total)!==frame.total)throw Error('原生伤害骰子或来源上下文发生变化。');
  const live=(params.token?.document??params.token)?.actor;
  if(live?.uuid!==actor.uuid||!live.testUserPermission?.(frame.user,'OWNER'))throw Error('真实目标的角色或操作者权限已改变。');
  if(frame.health&&(healthState(actor)!==frame.health.actor||healthState(live)!==frame.health.live))throw Error('等待原生伤害反应期间生命值或盾牌已改变，不能继续旧上下文。');
 }
 /** Called by main immediately around its one final, already-bound native call.
  * This method is not published in the module API. */
 async function withNativeFrame(actor,params,native){
  const frame=frames.get(actor);if(!frame)return native(params);
  if(frame.entered)throw markUnappliedDamageError(Error('同一次原生伤害不能重复或重入。'));
  frame.entered=true;frame.nativeParams=params;
  frame.health={actor:healthState(actor),live:healthState((params.token?.document??params.token)?.actor)};
  try{checkFrame(frame,actor,params);}catch(error){throw markUnappliedDamageError(error);}
  nativeFrames.set(params,frame);
  try{
   const result=await native(params);
   if(frame.failure)throw frame.failure;
   if(!frame.claimed)throw Error('原生伤害未经过已声明的 IWR 桥，结果无法确认，不能重新应用。');
   return result;
  }finally{nativeFrames.delete(params);}
 }
 /** Exact patched native callback. A nonparticipating public call is a sync no-op. */
 function nativeDamageIWR(actor,params,result,rollOptions,nativeAmounts){
  const frame=nativeFrames.get(params);if(!frame)return;
  const task=(async()=>{
   if(frame.claimed)throw Error('本次原生 IWR 结果已经接管，不能重复或重入。');frame.claimed=true;
   checkFrame(frame,actor,params);
   if(!Number.isFinite(result?.finalDamage)||!Array.isArray(result.applications)||!Array.isArray(result.persistent)||!(rollOptions instanceof Set)||params.rollOptions!=null&&params.rollOptions!==rollOptions||!Number.isFinite(nativeAmounts?.actorDamage)||!finite(nativeAmounts.shieldDamage))throw Error('原生 IWR、硬度后伤害结果或选项结构不匹配。');
   // The ordinary delta is already native post-hardness. Destructive Block
   // suppresses those native reductions and owns the one calculation below;
   // calculateHealthDelta must later reuse it, including retained shield loss.
   const scope=actors.get(actor);
   if(scope?.nativePhase)scope.calculation=destructiveBlockAmounts({incoming:Math.max(0,nativeAmounts.actorDamage),shieldHardness:scope.shieldHardness,shieldHP:scope.shieldHP,actorHardness:scope.actorHardness});
   frame.nativeShield=nativeAmounts.shieldDamage>0?Object.freeze({id:actor.attributes?.shield?.itemId??'',damage:nativeAmounts.shieldDamage}):null;
   const shield=scope?.calculation?.shieldDamage>0?Object.freeze({id:scope.shield.id,damage:scope.calculation.shieldDamage}):frame.nativeShield;
   const incoming=scope?.calculation?.actorDamage??Math.max(0,nativeAmounts.actorDamage),persistent=Object.freeze(result.persistent.map(instance=>{
    if(typeof instance?.type!=='string'||typeof instance.head?.expression!=='string')throw Error('原生持续伤害结果结构不匹配。');
    return Object.freeze({type:instance.type,formula:instance.head.expression});
   }));
   if(incoming<=0&&!persistent.length)return;
   const view={...params,rollOptions:new Set(rollOptions)};let accepting=true;
   const prevent=({marker,flagKey,proof}={})=>{
    if(!accepting||!frame.open)throw Error('本次防伤许可已经结束或失效。');checkFrame(frame,actor,params);
    if(frame.prevention)throw Error('本次伤害只能防止一次，不能重复使用许可。');
    if(flagKey!=='transcendentDeflection'||!/^[A-Za-z0-9_-]{1,80}$/.test(proof?.nonce??'')||marker!==`${MODULE_ID}:transcendent-deflection:${proof.nonce}`||typeof proof.actorUuid!=='string'||typeof proof.weaponUuid!=='string'||typeof proof.claimKey!=='string'||preventions.has(marker))throw Error('本次防伤证明或唯一标记不匹配。');
    const saved=structuredClone(proof);frame.prevention={marker,flagKey,proof:{...saved,kind:'damage-prevented',incoming,persistentPrevented:persistent.length,shield},captured:false};
    preventions.set(marker,frame);result.finalDamage=0;result.persistent=[];rollOptions.add(marker);
    if(scope?.calculation)scope.calculation={...scope.calculation,actorDamage:0};
   };
   try{for(const interceptor of frame.interceptors){await interceptor({actor,params:view,incoming,persistent,shield,prevent});checkFrame(frame,actor,params);if(frame.prevention)break;}}
   finally{accepting=false;}
   return !!frame.prevention;
  })();
  return task.catch(error=>{frame.failure=markUnappliedDamageError(error);throw frame.failure;});
 }
 function preparePreventionMessage(data){
  const context=data?.flags?.pf2e?.context;if(context?.type!=='damage-taken')return data;
  const matches=(context.options??[]).map(option=>preventions.get(option)).filter(Boolean);if(!matches.length)return data;
  if(matches.length!==1)throw Error('原生防伤消息含有多个执行证明。');
  const frame=matches[0],record=frame.prevention,token=frame.params.token?.document??frame.params.token;
  const applied=data.flags.pf2e.appliedDamage,shield=frame.nativeShield;
  const validApplied=shield?applied?.uuid===frame.actor.uuid&&applied.isHealing===false&&!applied.isReverted&&Array.isArray(applied.updates)&&applied.updates.length===0&&Array.isArray(applied.persistent)&&applied.persistent.length===0&&applied.shield?.id===shield.id&&applied.shield.damage===shield.damage:!applied;
  if(!frame.open||record.captured||data.speaker?.actor!==frame.actor.id||data.speaker?.scene!==token.parent?.id||data.speaker?.token!==token.id||!validApplied)throw Error('原生防伤消息与本次角色零伤害及盾牌回执不匹配。');
  record.captured=true;
  return {...data,flags:{...data.flags,[MODULE_ID]:{...data.flags?.[MODULE_ID],[record.flagKey]:record.proof}}};
 }
 function refreshCloneHealth(actor,params){
  const token=params.token?.document??params.token,live=token?.actor;
  if(actor===live||live?.uuid!==actor.uuid||actor.type!=='character')return;
  const raw=live._source?.system?.attributes?.hp;if(!raw)return;
  const original=actor._source?.system?.attributes?.hp,changes={};
  for(const [key,value]of [['value',raw.value],['temp',raw.temp],['sp.value',raw.sp?.value]]){
   const old=key==='sp.value'?original?.sp?.value:original?.[key];
   if(typeof value==='number'&&Number.isFinite(value)&&old!==value)changes[`system.attributes.hp.${key}`]=value;
  }
  // getContextualClone keeps its actor ID and temporary IWR effects. Refresh
  // only mutable health resources on that private clone, never the live source.
  if(Object.keys(changes).length)actor.updateSource(changes);
 }
 function validate(actor,params,plan){
  if(actor?.type!=='character'||!actor.testUserPermission?.(game.user,'OWNER'))throw Error('无权结算这个角色的破坏性格挡。');
  const shield=actor.heldShield,prepared=actor.attributes?.shield,token=params.token?.document??params.token;
  if(!plan||typeof plan.nonce!=='string'||!/^[A-Za-z0-9-]{8,80}$/.test(plan.nonce)||!hasSource(actor.items.get(plan.featId),DESTRUCTIVE_BLOCK_SOURCE))throw Error('破坏性格挡专长或本次认领不匹配。');
  if(Object.hasOwn(plan,'epoch')&&plan.epoch!==reactionEpoch(actor,game))throw Error('等待伤害结算期间回合已经改变，本次伤害尚未应用。');
  if(!finite(typeof params.damage==='number'?params.damage:params.damage?.total)||!params.shieldBlockRequest||params.final||!shield||shield.id!==plan.shieldId||prepared?.itemId!==shield.id||!prepared.raised||prepared.broken||prepared.destroyed||hasSource(shield,INDESTRUCTIBLE_SHIELD)||token?.actor?.uuid!==actor.uuid)throw Error('破坏性格挡的盾牌或伤害来源已改变。');
  const shieldHP=shield._source?.system?.hp?.value,shieldHardness=prepared.hardness,hardness=actor.hardness;
  if(!finite(shieldHP)||shieldHP===0||!finite(shieldHardness)||!finite(hardness))throw Error('这次格挡没有可损坏的实体盾牌。');
  const live=token.actor,liveShield=live.heldShield,livePrepared=live.attributes?.shield;
  if(!live.testUserPermission?.(game.user,'OWNER')||!hasSource(live.items.get(plan.featId),DESTRUCTIVE_BLOCK_SOURCE)||liveShield?.id!==shield.id||livePrepared?.itemId!==shield.id||!livePrepared.raised||livePrepared.broken||livePrepared.destroyed||liveShield._source?.system?.hp?.value!==shieldHP||hasSource(liveShield,INDESTRUCTIBLE_SHIELD))throw Error('真实角色的盾牌或专长已改变，本次伤害尚未应用。');
  return {actor,shield,token,plan,shieldHP,shieldHardness,brokenThreshold:prepared.brokenThreshold??shield.system.hp.max/2,actorHardness:effectiveActorHardness(hardness,params),nativePhase:true,calculation:null,captured:false};
 }
 async function applyDamage(actor,params,native,plan){
  if(frames.get(actor)?.params===params)throw markUnappliedDamageError(Error('同一次伤害参数不能重入当前队列。'));
  return queue.run(actor.uuid,async()=>{
   let scope;
   try{refreshCloneHealth(actor,params);if(plan){scope=validate(actor,params,plan);if(actors.has(actor)||nonces.has(plan.nonce))throw Error('破坏性格挡已经在结算。');}}
   catch(error){throw markUnappliedDamageError(error);}
   const actual=plan?{...params,shieldBlockRequest:false,rollOptions:new Set([...params.rollOptions??[],prefix+plan.nonce])}:params;
   let matched=[];
   try{if(interceptors.size&&actor.hitPoints&&nativeBridgeAvailable())matched=[...interceptors].filter(([,matches])=>matches(actor,actual)===true).map(([callback])=>callback);}
   catch(error){throw markUnappliedDamageError(error);}
   const frame=matched.length?{actor,params:actual,interceptors:matched,source:Object.fromEntries(['damage','token','item','final','skipIWR','shieldBlockRequest'].map(key=>[key,actual[key]])),user:game.user,total:typeof actual.damage==='number'?actual.damage:actual.damage?.total,open:true,entered:false,claimed:false}:null;
   if(frame)frames.set(actor,frame);
   if(plan){actors.set(actor,scope);nonces.set(plan.nonce,scope);}
   try{return await native(actual);}
   finally{if(frame){frame.open=false;frames.delete(actor);if(frame.prevention)preventions.delete(frame.prevention.marker);}if(plan){actors.delete(actor);nonces.delete(plan.nonce);}}
  });
 }
 async function prepareMessage(data){
  const pf=data?.flags?.pf2e,options=pf?.context?.options??[],markers=options.filter(o=>typeof o==='string'&&o.startsWith(prefix));
  if(markers.length!==1||pf?.context?.type!=='damage-taken')return data;
  const scope=nonces.get(markers[0].slice(prefix.length));if(!scope)return data;
  const {actor,token,shield,plan,calculation}=scope;
  if(scope.captured||!calculation||data.speaker?.actor!==actor.id||data.speaker?.scene!==token.parent?.id||data.speaker?.token!==token.id)return data;
  scope.captured=true;scope.nativePhase=false;
  const liveShield=token.actor?.heldShield;
  if(actor.items.get(shield.id)!==shield||actor.heldShield?.id!==shield.id||shield._source.system.hp.value!==scope.shieldHP||liveShield?.id!==shield.id||liveShield._source.system.hp.value!==scope.shieldHP)throw Error('盾牌在伤害结算期间发生变化，生命值已经按原生结果结算；请检查盾牌伤害回执。');
  if(calculation.shieldDamage>0)await shield.update({'system.hp.value':scope.shieldHP-calculation.shieldDamage},{render:calculation.actorDamage===0});
  const nativeMarkers=options.filter(o=>typeof o==='string'&&o.startsWith(nativePrefix));
  const proof={kind:'destructive-block',nonce:plan.nonce,nativeBlockNonce:nativeMarkers.length===1?nativeMarkers[0].slice(nativePrefix.length):null,shieldId:shield.id,...calculation,blocked:calculation.incoming>0};
  const applied=pf.appliedDamage??(calculation.shieldDamage>0?{uuid:actor.uuid,isHealing:false,shield:null,persistent:[],updates:[]}:null);
  if(applied&&calculation.shieldDamage>0)applied.shield={id:shield.id,damage:calculation.shieldDamage};
  const actual={...data,flags:{...data.flags,pf2e:{...pf,appliedDamage:applied},[MODULE_ID]:{...data.flags?.[MODULE_ID],shieldBlock:proof}}};
  const hp=scope.shieldHP-calculation.shieldDamage,shieldState={value:hp,destroyed:hp===0,broken:hp>0&&hp<=scope.brokenThreshold};
  try{actual.content=await renderContent({data:actual,calculation,canUndo:!!applied,shield,shieldState});}catch(error){report(error);}
  return actual;
 }
 function register({libWrapper}){
  if(registered)return;registered=true;
  libWrapper.register(MODULE_ID,'CONFIG.PF2E.Actor.documentClasses.character.prototype.hardness',function(wrapped){const value=wrapped();return actors.get(this)?.nativePhase?0:value;},'WRAPPER');
  libWrapper.register(MODULE_ID,'CONFIG.Actor.documentClass.prototype.calculateHealthDelta',function(wrapped,params){
   const scope=actors.get(this);if(!scope?.nativePhase)return wrapped(params);
   scope.calculation??=destructiveBlockAmounts({incoming:Math.max(0,params.delta),shieldHardness:scope.shieldHardness,shieldHP:scope.shieldHP,actorHardness:scope.actorHardness});
   scope.nativePhase=false;return wrapped({...params,delta:scope.calculation.actorDamage});
  },'WRAPPER');
  libWrapper.register(MODULE_ID,'ChatMessage.create',async function(wrapped,data,...args){
   let actual=preparePreventionMessage(data);try{actual=await prepareMessage(actual);}catch(error){
    report(error);
    actual={...data,flags:{...data.flags,[MODULE_ID]:{...data.flags?.[MODULE_ID],shieldBlock:{kind:'destructive-block',uncertain:true}}}};
   }
   return createMessageMiddleware?createMessageMiddleware(wrapped,actual,...args):wrapped(actual,...args);
  },'MIXED');
 }
 return {applyDamage,register,addNativeInterceptor,nativeBridgeAvailable,nativeBridgeDiagnostic,withNativeFrame,nativeDamageIWR};
}
