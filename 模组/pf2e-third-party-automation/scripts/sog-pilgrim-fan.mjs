import {MODULE_ID as M} from './rules.mjs';
import {isActiveGM} from './native-context.mjs';
import {PILGRIM_FLAG,PilgrimError,EFFECTS,values,rewardKey,usableReward,serializeReward} from './sog-pilgrim-rules.mjs';

const duration={value:1,unit:'minutes',expiry:'turn-start',sustained:false};
const same=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
const kinds=new Set(['leaves','light']);
const traits=actor=>actor?.traits instanceof Set?actor.traits:new Set(actor?.system?.traits?.value??[]);
const creature=actor=>actor?.isOfType?.('character','npc')===true;
const validStart=start=>Number.isFinite(start?.value)&&start.value>=0&&(start.initiative===null||Number.isFinite(start.initiative));

/** The physical fan holds both hit receipts and the independent leaf starts. */
export function getFanState(item) {
 const stored=item?.flags?.[M]?.[PILGRIM_FLAG]?.fan;
 return {
  leaves:Array.isArray(stored?.leaves)?stored.leaves.filter(leaf=>typeof leaf?.messageId==='string'&&validStart(leaf.start)).slice(0,3).map(leaf=>({messageId:leaf.messageId,start:{...leaf.start}})):[],
  hitIds:Array.isArray(stored?.hitIds)?[...new Set(stored.hitIds.filter(id=>typeof id==='string'&&id))]:[],
 };
}

export function createPilgrimFan({game,fromUuid=globalThis.fromUuid,media={},onError=error=>console.error(M,error),
 nativeRemaining=(data,actor)=>new globalThis.CONFIG.Item.documentClass(data,{parent:actor}).remainingDuration}={}) {
 const known=new Map();
 const currentToken=token=>token?.documentName==='Token'&&game.scenes?.get(token.parent?.id)===token.parent&&token.parent.tokens?.get(token.id)===token;
 const currentActor=actor=>actor?.isToken?currentToken(actor.token)&&actor.token.actor===actor:game.actors?.get(actor?.id)===actor;
 const owned=item=>rewardKey(item)==='fan'&&currentActor(item.actor)&&item.actor.type==='character'&&!traits(item.actor).has('eidolon')&&item.actor.items?.get(item.id)===item;
 const authority=()=>{if(!isActiveGM(game))throw new PilgrimError('请由当前主持人处理灵扇。');};
 const source=item=>{authority();if(!owned(item)||!usableReward(item))throw new PilgrimError('请先持握灵扇。');};
 const ownedEffects=(actor,uuid,kind)=>values(actor.items).filter(effect=>effect.type==='effect'&&effect.flags?.[M]?.[PILGRIM_FLAG]?.source===uuid&&(kind?effect.flags[M][PILGRIM_FLAG].kind===kind:kinds.has(effect.flags[M][PILGRIM_FLAG]?.kind)));
 async function play(method,...args) {
  try{await media[method]?.(...args)}catch(error){try{onError(error)}catch(reportError){console.error(M,reportError)}}
 }
 function remaining(item,leaf) {
  const data={type:'effect',name:'金叶',system:{duration:{...duration},start:{...leaf.start},context:{origin:{actor:item.actor.uuid,item:item.uuid,rollOptions:[]}},traits:{value:[],otherTags:[]},rules:[],badge:null}};
  const result=nativeRemaining(data,item.actor);
  if(typeof result?.expired!=='boolean'||!Number.isFinite(result.remaining))throw new PilgrimError('无法确认金叶的剩余时间，请核对灵扇。');
  return result;
 }
 function validLeaves(item,state=getFanState(item)) {
  return state.leaves.filter(leaf=>leaf.start.value<=game.time.worldTime&&!remaining(item,leaf).expired);
 }
 async function save(item,state,validate=()=>source(item)) {
  validate();
  await item.update({[`flags.${M}.${PILGRIM_FLAG}.fan`]:state});
  authority();
  if(!owned(item)||!same(getFanState(item),state))throw new PilgrimError('金叶未能保存，请核对灵扇。');
 }
 async function remove(actor,uuid) {
  authority();const effects=ownedEffects(actor,uuid);
  if(effects.length){await actor.deleteEmbeddedDocuments('Item',effects.map(effect=>effect.id));authority();if(ownedEffects(actor,uuid).length)throw new PilgrimError('灵扇光芒未能熄灭，请核对效果。');}
 }
 async function syncEffect(item,kind,state) {
  source(item);
  const template=await fromUuid(`Item.${EFFECTS[kind]}`);source(item);
  if(template?.type!=='effect'||typeof template.toObject!=='function')throw new PilgrimError('缺少金叶或光芒效果，请联系主持人。');
  const data=template.toObject();delete data._id;delete data.folder;delete data.ownership;delete data._stats;
  const latest=state.leaves.reduce((last,leaf)=>leaf.start.value>=last.start.value?leaf:last);
  data.system.duration={...duration};data.system.start={...latest.start};
  data.system.context={...data.system.context,origin:{...data.system.context?.origin,actor:item.actor.uuid,item:item.uuid,rollOptions:[]}};
  if(kind==='leaves')data.system.badge={...data.system.badge,type:'counter',value:state.leaves.length,min:1,max:3};
  data.flags={...data.flags,[M]:{...data.flags?.[M],[PILGRIM_FLAG]:{kind,source:item.uuid}}};
  const existing=ownedEffects(item.actor,item.uuid,kind);let effect=existing[0];
  if(effect){
   const patch={};
   for(const key of ['duration','start','context','badge'])if(!same(effect._source?.system?.[key]??effect.system[key],data.system[key]))patch[`system.${key}`]=data.system[key];
   if(Object.keys(patch).length){await effect.update(patch);source(item);}
  }else{
   [effect]=await item.actor.createEmbeddedDocuments('Item',[data]);source(item);
   if(!effect||item.actor.items.get(effect.id)!==effect)throw new PilgrimError('金叶效果未能保存，请核对灵扇。');
   // PF2e resets an embedded effect's start on creation, including restoration.
   if(!same(effect._source?.system?.start??effect.system.start,data.system.start)){await effect.update({'system.start':data.system.start});source(item);}
  }
  if(!same(effect._source?.system?.start??effect.system.start,data.system.start)||kind==='leaves'&&effect.system.badge?.value!==state.leaves.length)throw new PilgrimError('金叶数量或时限未能保存，请核对灵扇。');
  if(existing.length>1){await item.actor.deleteEmbeddedDocuments('Item',existing.slice(1).map(other=>other.id));source(item);}
 }
 async function sync(item,state) {
  if(!state.leaves.length)await remove(item.actor,item.uuid);
  else{await syncEffect(item,'leaves',state);await syncEffect(item,'light',state);}
  await play('leaves',item,state.leaves.length);
 }
 async function hitSource(message) {
  if(!isActiveGM(game)||!message?.id||game.messages?.get(message.id)!==message||message.isCheckRoll!==true)return null;
  const item=message.item,actor=message.actor,pf=message.flags?.pf2e,c=pf?.context,roll=message.rolls?.[0],degree=c?.outcome==='success'?2:c?.outcome==='criticalSuccess'?3:null;
  if(!item||!owned(item)||!usableReward(item)||actor!==item.actor||pf.origin?.actor!==actor.uuid||pf.origin.uuid!==item.uuid||c?.type!=='attack-roll'||degree===null||c.isReroll||message.isReroll||roll?._evaluated!==true||!Number.isFinite(roll.total)||roll.total<0||roll.options?.degreeOfSuccess!==degree||!Number.isFinite(c.dc?.value)||c.dc.value<=0)return null;
  const target=await fromUuid(c.target?.token);
  if(!isActiveGM(game)||!currentToken(target)||!creature(target.actor)||target.actor.isDead===true||c.target.actor!==target.actor.uuid||!owned(item)||message.actor!==item.actor||game.messages.get(message.id)!==message)return null;
  return {item,target};
 }
 async function handleStrike(message) {
  const binding=await hitSource(message);if(!binding)return null;
  const {item,target}=binding;known.set(item.uuid,item);
  return serializeReward(item.uuid,async()=>{
   const current=await hitSource(message);if(!current||current.item!==item||current.target!==target)return null;
   const state=getFanState(item),leaves=validLeaves(item,state);
   if(state.hitIds.includes(message.id)){
    if(!same(leaves,state.leaves)){state.leaves=leaves;await save(item,state);await sync(item,state);}
    return null;
   }
   const start={value:game.time.worldTime,initiative:game.combat?.combatant?.initiative??null};
   if(!validStart(start))throw new PilgrimError('无法确认金叶亮起的时刻，请核对灵扇。');
   const count=traits(target.actor).has('undead')?3:Math.min(leaves.length+1,3);
   while(leaves.length<count)leaves.push({messageId:message.id,start:{...start}});
   const next={leaves,hitIds:[...state.hitIds,message.id]};
   // One persisted update binds the hit receipt to its leaf starts.
   await save(item,next);await sync(item,next);return next.leaves.length;
  });
 }
 async function reconcile(actor) {
  if(!isActiveGM(game)||!currentActor(actor))return;
  const items=values(actor.items).filter(item=>rewardKey(item)==='fan');
  for(const item of items){
   known.set(item.uuid,item);
   await serializeReward(item.uuid,async()=>{
    if(!isActiveGM(game)||!owned(item))return;
    if(!usableReward(item)){await remove(actor,item.uuid);await play('leaves',item,0);return;}
    const state=getFanState(item),leaves=validLeaves(item,state);
    if(!same(state.leaves,leaves)){state.leaves=leaves;await save(item,state);}
    await sync(item,state);
   });
  }
  const present=new Set(items.map(item=>item.uuid));
  const orphanSources=new Set(values(actor.items).filter(effect=>{
   const mark=effect.flags?.[M]?.[PILGRIM_FLAG],origin=effect.system?.context?.origin;
   return effect.type==='effect'&&kinds.has(mark?.kind)&&typeof mark.source==='string'&&mark.source.startsWith(`${actor.uuid}.Item.`)&&origin?.actor===actor.uuid&&origin.item===mark.source&&!present.has(mark.source);
  }).map(effect=>effect.flags[M][PILGRIM_FLAG].source));
  for(const [uuid,item]of known)if(item.actor===actor&&!present.has(uuid))orphanSources.add(uuid);
  for(const uuid of orphanSources)await serializeReward(uuid,async()=>{
   if(!isActiveGM(game))return;await remove(actor,uuid);const old=known.get(uuid);if(old){await play('leaves',old,0);known.delete(uuid);}
  });
 }
 async function clearAfterRelease(item) {
  const validate=()=>{authority();if(!owned(item))throw new PilgrimError('灵扇已改变，请联系主持人核对金叶。');};
  validate();const state=getFanState(item);state.leaves=[];
  await save(item,state,validate);await remove(item.actor,item.uuid);await play('leaves',item,0);
 }
 function releaseTarget(item,target,selectedSource) {
  source(item);
  if(!currentToken(target)||!creature(target.actor)||target.actor.isDead||traits(target.actor).has('construct')&&!traits(target.actor).has('undead'))throw new PilgrimError('请选择一个活物或不死生物。');
  const tokens=(selectedSource?[selectedSource]:item.actor.token?[item.actor.token]:item.actor.getActiveTokens?.(true,true)??[]).map(token=>token.document??token).filter(token=>currentToken(token)&&token.actor===item.actor&&token.parent===target.parent);
  const token=tokens.length===1?tokens[0]:null;
  if(!token?.object||!target.object||target.hidden||item.actor.canSee===false||item.actor.hasCondition?.('blinded')||item.actor.getCondition?.('blinded')||typeof token.object.distanceTo!=='function'||typeof token.object.checkCollision!=='function')throw new PilgrimError('请选择你能看见、且在 30 尺内的目标。');
  const distance=token.object.distanceTo(target.object);
  if(!Number.isFinite(distance)||distance<0||distance>30||token.object.checkCollision(target.object.center,{origin:token.object.center,type:'sight',mode:'any'}))throw new PilgrimError('请选择你能看见、且在 30 尺内的目标。');
  if(validLeaves(item).length!==3)throw new PilgrimError('需要三片金叶全部亮起。');
  return traits(target.actor).has('undead');
 }
 /** The provider already owns the reward queue and its frequency transaction. */
 async function release({item,source:selectedSource,target,nonce,card}) {
  const undead=releaseTarget(item,target,selectedSource),targetActor=target.actor;
  if(typeof nonce!=='string'||!nonce||typeof card!=='function')throw new PilgrimError('本次启动未能确认，请重试。');
  const validate=()=>{if(target.actor!==targetActor||releaseTarget(item,target,selectedSource)!==undead)throw new PilgrimError('目标已改变，请重新选择。');};
  const message=await card({item,target,nonce,label:'灵魂连携',formula:undead?'(3d8+8)[vitality]':'(3d8+8)[healing]',...undead?{save:{type:'fortitude',dc:23,basic:true}}:{},validate});
  if(!message)return null;
  if(!message.id||game.messages?.get(message.id)!==message)throw new PilgrimError('启动结果未能保存，请联系主持人核对。');
  // Publication is the release boundary; later time changes cannot undo it.
  await clearAfterRelease(item);await play('release',item,target);return {messageId:message.id};
 }
 return {handleStrike,reconcile,release,clearAfterRelease,getState:getFanState};
}
