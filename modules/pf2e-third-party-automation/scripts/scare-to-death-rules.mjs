import {MODULE_ID} from './rules.mjs';

export const SCARE_SOURCE='Compendium.pf2e.feats-srd.Item.mZttsiWl1ql5NvrH';
export const SCARE_OUTCOMES=Object.freeze(['criticalFailure','failure','success','criticalSuccess']);
export const SCARE_TRAITS=Object.freeze(['emotion','fear','mental','incapacitation']);
const values=c=>Array.from(c?.values?.()??c??[]);
const degree=n=>Number.isInteger(n)&&n>=0&&n<=3;
export const scareState=doc=>doc?.flags?.[MODULE_ID]?.scare??{};

/** Degrees are already final native degrees, including incapacitation. */
export function scareResult({check,fortitude,effectImmune=false,deathImmune=false}){
 if(!degree(check)||fortitude!==undefined&&!degree(fortitude))throw Error('肝胆俱裂的原生成功度无效。');
 const empty={frightened:0,fleeing:false,death:false,save:false};
 if(effectImmune)return empty;
 if(check<3)return {...empty,frightened:check===2?2:check===1?1:0};
 if(fortitude===undefined)return {...empty,save:true};
 return fortitude===0?{...empty,death:!deathImmune}:{...empty,frightened:2,fleeing:true};
}
export function scareLanguagePenalty({heard,understood}){
 if(typeof heard!=='boolean'||typeof understood!=='boolean')throw Error('缺少实际听觉或语言事实。');
 return heard&&understood?0:-4;
}
export const scareImmunityExpired=(item,now)=>typeof item?.remainingDuration?.expired==='boolean'?item.remainingDuration.expired:scareState(item).expiresAt<=now;
export const scareTemporarilyImmune=(actor,now)=>values(actor?.items).some(i=>scareState(i).kind==='immunity'&&!scareImmunityExpired(i,now));

export function scareEffectData({kind,item,origin,now,usageId,checkId}){
 if(!['fleeing','immunity','death','test'].includes(kind)||!Number.isFinite(now)||!item?.uuid||!origin?.actor?.uuid)throw Error('肝胆俱裂派生效果来源无效。');
 const immunity=kind==='immunity';
 return {name:`肝胆俱裂：${({fleeing:'逃跑',immunity:'暂时免疫',death:'死亡',test:'恐惧'})[kind]}`,type:'effect',img:item.img??'icons/svg/terror.svg',
  system:{slug:`scare-to-death-${kind}`,traits:{value:immunity?[]:[...SCARE_TRAITS,...kind==='death'?['death']:[]]},
   duration:{value:1,unit:immunity?'minutes':'rounds',expiry:'turn-start',sustained:false},start:{value:now,initiative:origin.actor.combatant?.initiative??null},
   context:{origin:{actor:origin.actor.uuid,item:item.uuid,token:origin.uuid}},tokenIcon:{show:!immunity},
   rules:kind==='fleeing'?[{key:'GrantItem',uuid:'Compendium.pf2e.conditionitems.Item.sDPxOjQ9kx2RZE8D',onDeleteActions:{granter:'cascade'}}]:[]},
  flags:{[MODULE_ID]:{scare:{kind,...usageId?{usageId}:{},...checkId?{checkId}:{},...immunity?{expiresAt:now+60}:{}}}}};
}

/** PF2e 8.5's death-effects predicate deliberately has no generic item match.
 * Add its missing base selector only for an actual death effect; let the real
 * native predicate continue to decide all exception clauses. */
export function scareDeathImmune(actor,rollOptions,iwrEnabled){
 const options=new Set(rollOptions);
 if(!options.has('item:trait:death'))throw Error('只有实际死亡分支可以检查死亡免疫。');
 if(!iwrEnabled)return false;
 options.add('unhandled:death-effects');
 return values(actor?.attributes?.immunities).some(i=>{
  if(i.type!=='death-effects')return false;
  if(typeof i.test!=='function')throw Error('缺少死亡免疫的原生例外谓词。');
  return i.test(options);
 });
}

/** StatisticCheck chooses the first active origin for saves and first active
 * target for skills. Reject a different linked token before it can roll. */
export function assertScareTokens(origin,target){
 const nativeToken=t=>t?.document??t;
 if(!origin?.object||!target?.object||origin.parent!==target.parent||origin.parent?.tokens?.get(origin.id)!==origin||target.parent?.tokens?.get(target.id)!==target||
  nativeToken(origin.actor?.getActiveTokens?.(true,true)?.[0])?.uuid!==origin.uuid||nativeToken(target.actor?.getActiveTokens?.(true,true)?.[0])?.uuid!==target.uuid)throw Error('肝胆俱裂的原生来源或目标 Token 不匹配。');
}

/** Test the two token-specific native detection sources independently. Do not
 * consult combined GM visibility, which includes unrelated vision sources. */
export function scareSensePair(origin,target,{canvas=globalThis.canvas,CONFIG=globalThis.CONFIG}={}){
 const visibility=canvas?.visibility,modes=CONFIG?.Canvas?.detectionModes;
 if(!visibility?._createVisibilityTestConfig||!modes||origin?.parent!==target?.parent)throw Error('无法确定肝胆俱裂双方的原生感知。');
 function detect(observer,subject){
  const token=observer?.object;
  if(!token||!subject?.object||subject.hidden||observer.actor?.isDead||observer.actor?.hasCondition?.('unconscious'))return {senses:false,hears:false};
  function hearing(){
   if(observer.actor?.hasCondition?.('deafened')||!subject.actor?.emitsSound||subject.actor?.hasCondition?.('undetected','unnoticed'))return false;
   const range=observer.parent.flags?.pf2e?.hearingRange??Infinity,distance=token.distanceTo?.(subject.object);
   if(!Number.isFinite(distance)||distance>range)return false;
   const backend=CONFIG.Canvas.polygonBackends?.sound,level=observer.parent.levels?.get(observer._source?.level??observer.level);
   if(!backend?.testCollision||!level)throw Error('无法确认本场景层级的原生声音传播。');
   return !backend.testCollision({...token.center,elevation:observer.elevation??0},{...subject.object.center,elevation:subject.elevation??0},{type:'sound',mode:'any',level});
  }
  // Without token vision, core deliberately omits NPC detection modes. Sight
  // simulation is off, but actual blindness, invisible subjects and scene
  // walls still apply; ordinary hearing is not removed by that UI setting.
  if(observer.parent.tokenVision===false){
   const hears=hearing(),hidden=['invisible','hidden','undetected','unnoticed'].some(c=>subject.actor?.hasCondition?.(c));
   const sees=observer.actor?.canSee!==false&&!observer.actor?.hasCondition?.('blinded')&&!hidden&&typeof token.checkCollision==='function'&&!token.checkCollision(subject.object.center,{origin:token.center,type:'sight',mode:'any'});
   return {senses:hears||sees,hears};
  }
  const temporary=!token.vision,source=token.vision??token._createSharedFogVisionSource?.();
  if(!source)throw Error('缺少本次 Token 的独立感知源。');
  try{
   if(temporary){Object.assign(source.blinded,token._getVisionBlindedStates());source.initialize(token._getVisionSourceData());}
   const detected=[];
   for(const [id,mode]of Object.entries(observer.detectionModes??{})){
    if(!mode?.enabled||typeof modes[id]?.testVisibility!=='function')continue;
    if(id==='hearing'&&observer.actor?.hasCondition?.('deafened'))continue;
    const config=visibility._createVisibilityTestConfig([subject.object.center],{object:subject.object,tolerance:2});
    if(modes[id].testVisibility(source,mode,config)===true)detected.push(id);
   }
   const hears=detected.includes('hearing')||!Object.hasOwn(observer.detectionModes??{},'hearing')&&hearing();
   return {senses:detected.length>0||hears,hears};
  }finally{if(temporary)source.destroy();}
 }
 const forward=detect(origin,target),reverse=detect(target,origin);
 return {originSensesTarget:forward.senses,targetSensesOrigin:reverse.senses,targetHearsOrigin:reverse.hears};
}
