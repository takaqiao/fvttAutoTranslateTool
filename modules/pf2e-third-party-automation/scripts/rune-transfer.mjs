import {MODULE_ID,hasSource} from './rules.mjs';
import {SerialActions,requireOwner} from './runtime.mjs';
import {isEatFortuneProbe} from './eat-fortune.mjs';

export const RUNE_TRANSFER_SOURCE='Compendium.pf2e.feats-srd.Item.pe8a7WDz0MIY45uO';
export const RUNE_TRANSFER_KEY='ThirdPartyCuttingHeavenRunes';
const KIND='rune-transfer',values=value=>Array.from(value?.values?.()??value?.contents??value??[]);
const own=item=>item?.flags?.[MODULE_ID]??{};
const feature=actor=>values(actor?.items).find(item=>hasSource(item,RUNE_TRANSFER_SOURCE));
const states=actor=>values(actor?.items).filter(item=>item.type==='effect'&&own(item).kind===KIND);
const get=(actor,id)=>actor?.items?.get?.(id)??values(actor?.items).find(item=>item.id===id);
const worlds=new Set(['-','sog','pnvfcgjbf2cjp7gz','team-automation-qa2']);
// PF2e 8.5 equipment pack Usage entries. Mechanical rune effects remain native.
const ordinaryRunes=new Set('ancestralEchoing anchoring ashen astral authorized bane brilliant calledLastwall coating conducting corrosive deathdrinking decaying demolishing earthbinding energizing energyVulnerability fearsome flaming flickering frost ghostTouch giantKilling greaterAnchoring greaterAshen greaterAstral greaterBrilliant greaterCorrosive greaterDecaying greaterEnergyVulnerability greaterFearsome greaterFlaming greaterFrost greaterGiantKilling greaterHauling greaterImpactful greaterShock greaterThundering grievous hauling hopeful impactful impossible merciful nightmare pacifying quickstrike shock thundering underwater vitalizing'.split(' '));
const meleeRunes=new Set('animated deflecting extending fanged greaterExtending greaterFanged greaterRooting greaterVitalizing hooked majorFanged majorRooting rooting shifting spellReservoir trueRooting'.split(' '));
export function runeCanApply(rune,weapon){
 const type=weapon?.baseDamage?.damageType??weapon?.system?.damage?.damageType,traits=new Set(weapon?.traits??weapon?.system?.traits?.value??[]),melee=weapon?.isMelee===true;
 if(ordinaryRunes.has(rune))return true;
 if(meleeRunes.has(rune))return melee;
 if(['crushing','greaterCrushing','shockwave'].includes(rune))return type==='bludgeoning';
 if(['bloodthirsty','keen','wounding'].includes(rune))return melee&&['slashing','piercing'].includes(type);
 if(rune==='cunning')return ['slashing','piercing'].includes(type);
 if(rune==='vorpal')return melee&&type==='slashing';
 if(rune==='flurrying')return melee&&traits.has('monk');
 if(['returning','swarming'].includes(rune))return weapon.isThrown||weapon.isThrowable||[...traits].some(t=>/^thrown(?:-|$)/.test(t));
 if(['holy','unholy'].includes(rune))return !weapon.system.runes.property.includes(rune==='holy'?'unholy':'holy');
 return false;
}

export function isCuttingWeapon(item){
 if(item?.type!=='weapon'||(item.category??item.system?.category)==='unarmed'||item.isMelee!==true||item.isEquipped!==true||item.isHeld!==true||item.isStowed)return false;
 const traits=new Set(item.traits??item.system?.traits?.value??[]),hands=item.hands??({'held-in-one-hand':'1','held-in-one-plus-hands':'1+','held-in-two-hands':'2'}[item.system?.usage?.value]);
 const required=hands==='2'?2:hands==='1+'?2:1;
 return item.handsHeld>=required&&(hands==='1'||traits.has('agile')||traits.has('finesse'));
}
const eligibleUsage=item=>isCuttingWeapon(item)?item:(item?.getAltUsages?.()??[]).find(isCuttingWeapon);
const handwraps=actor=>values(actor.itemTypes?.weapon??actor.items).find(item=>item.type==='weapon'&&(item.category??item.system?.category)==='unarmed'&&item.system?.traits?.otherTags?.includes('handwraps-of-mighty-blows')&&item.isEquipped&&item.isInvested);
const prune=runes=>{const set=new Set(runes),capital=value=>value[0].toUpperCase()+value.slice(1);return [...set].filter(rune=>!set.has(`greater${capital(rune)}`)&&!set.has(`major${capital(rune.replace(/^greater/,''))}`)&&!set.has(`true${capital(rune.replace(/^greater|^major/,''))}`))};
function propertyPlan(actor,weapon,wraps,game){
 const existing=[...weapon.system.runes.property],accepted=wraps.system.runes.property.filter(rune=>runeCanApply(rune,weapon));
 const combined=prune([...existing,...accepted]),abp=game.pf2e.variantRules.AutomaticBonusProgression;
 const potency=abp.isEnabled(actor)?abp.getAttackPotency(actor.level):Math.max(weapon.system.runes.potency,wraps.system.runes.potency);
 const capacity=weapon.system.grade?0:potency+Number(weapon.system.material?.type==='orichalcum');
 const conflict=combined.length>capacity;
 return {status:conflict?'capacity-conflict':'ready',capacity,properties:conflict?existing:combined,transfer:conflict?[]:accepted,skipped:wraps.system.runes.property.filter(rune=>!accepted.includes(rune))};
}
export function runeTransferStatus(actor,game){
 const id=selectedRuneWeaponId(actor),weapon=eligibleUsage(get(actor,id)),wraps=handwraps(actor);
 return id&&weapon&&wraps?{...propertyPlan(actor,weapon,wraps,game),selectedWeaponId:id}:{status:'inactive',selectedWeaponId:id};
}

export function selectedRuneWeaponId(actor){
 const feat=feature(actor),list=states(actor);if(!feat||list.length!==1)return null;
 const state=own(list[0]).runeTransfer;
 return state?.version===1&&state.featId===feat.id&&state.source===RUNE_TRANSFER_SOURCE&&typeof state.selectedWeaponId==='string'&&get(actor,state.selectedWeaponId)?.type==='weapon'?state.selectedWeaponId:null;
}

export function buildRuneTransferEffect({actor,feat,selectedWeaponId=null,revision=0}){
 if(!hasSource(feat,RUNE_TRANSFER_SOURCE)||!values(actor?.items).includes(feat))throw Error('断天剑碎地拳需要角色实际持有的准确专长。');
 return {name:'断天剑·碎地拳：符文传递',type:'effect',img:feat.img??'icons/svg/aura.svg',system:{description:{value:'符文随当前投入的重拳缠手带自动更新；正常使用断天剑碎地拳可更换选定武器。'},level:{value:1},traits:{value:[],rarity:'common'},duration:{value:-1,unit:'unlimited',expiry:null,sustained:false},tokenIcon:{show:false},rules:[{key:RUNE_TRANSFER_KEY}]},flags:{[MODULE_ID]:{kind:KIND,runeTransfer:{version:1,source:RUNE_TRANSFER_SOURCE,featId:feat.id,selectedWeaponId,revision}}}};
}

const registrations=new WeakMap();
export function registerRuneTransferRuleElement(game){
 const registry=game?.pf2e?.RuleElements,Base=game?.pf2e?.RuleElement;
 if(!registry?.custom||!Base)throw Error('PF2e 原生规则元素尚未初始化。');
 if(registrations.has(registry))return registrations.get(registry);
 const {ItemAlteration,AdjustStrike}=registry.builtin;
 class CuttingHeavenRunes extends Base{
  constructor(data,options){super({...data,priority:1000},options)}
  beforePrepareData(){
   if(this.ignored||!worlds.has(game.world?.id)||states(this.actor).length!==1||states(this.actor)[0]!==this.item)return;
   const id=selectedRuneWeaponId(this.actor);if(!id)return;
   if(this.actor.rollOptions?.all)this.actor.rollOptions.all[`${MODULE_ID}:cutting-weapon:${id}`]=true;
   const apply=weapon=>{
    if(weapon?.id!==id||selectedRuneWeaponId(this.actor)!==id||!isCuttingWeapon(weapon))return;
    const wraps=handwraps(this.actor);
    if(!wraps)return;
    const runes=wraps.system.runes;
    for(const property of ['potency','striking'])new ItemAlteration({key:'ItemAlteration',itemId:id,mode:'upgrade',property:`runes-${property}`,value:runes[property],fromEquipment:true,predicate:['item:melee']},{parent:this.item}).applyAlteration({singleItem:weapon});
    for(const rune of new Set(propertyPlan(this.actor,weapon,wraps,game).transfer)){
     // AdjustStrike registers its native adjustment; invoke it during late item
     // preparation so prepareStrike also sees the rune's own trait adjustments.
     const index=this.actor.synthetics.strikeAdjustments.length;
     new AdjustStrike({key:'AdjustStrike',mode:'add',property:'property-runes',value:rune,definition:[`item:id:${id}`,'item:melee']},{parent:this.item}).beforePrepareData();
     const adjustments=this.actor.synthetics.strikeAdjustments.splice(index);
     for(const adjustment of adjustments)adjustment.adjustWeapon?.(weapon);
    }
   };
   apply(get(this.actor,id));
   // PF2e 8.5 performLatePreparation invokes this for same-ID alternative usages
   // cloned from raw source. Never copy the prepared weapon into a ranged usage.
   this.actor.synthetics.itemAlterations.push({isLazy:false,applyAlteration:({singleItem}={})=>{if(singleItem)apply(singleItem)}});
  }
 }
 registry.custom[RUNE_TRANSFER_KEY]=CuttingHeavenRunes;registrations.set(registry,CuttingHeavenRunes);return CuttingHeavenRunes;
}

export function createRuneTransfer({game,fromUuid,choose,onError=()=>{}}){
 const queue=new SerialActions(),readyMarker=Symbol('rune-transfer-ready');let socket,registered=false;
 const activeGM=()=>game.user?.isGM&&game.user.id===game.users?.activeGM?.id;
 const real=actor=>actor?.type==='character'&&worlds.has(game.world?.id)&&(actor.isToken?actor.token?.actor===actor:game.actors?.get?.(actor.id)===actor);
 const guard=(actor,user)=>{if(!activeGM())throw Error('符文选择必须由当前主GM处理。');if(!real(actor))throw Error('符文选择只能保存到实际角色。');if(user)requireOwner(actor,user)};
 const candidates=actor=>values(actor.items).filter(item=>!!eligibleUsage(item));
 const fingerprint=actor=>candidates(actor).map(i=>i.id).sort().join(',');
 async function write(actor,item,changes,user){guard(actor,user);await item.update(changes);guard(actor,user);return item}
 async function maintainInside(actor){
  guard(actor);const feat=feature(actor),list=states(actor);
  if(!feat){if(list.length){guard(actor);await actor.deleteEmbeddedDocuments('Item',list.map(i=>i.id))}return null}
  if(list.length>1){guard(actor);await actor.deleteEmbeddedDocuments('Item',list.slice(1).map(i=>i.id));guard(actor)}
  let effect=list[0],state=own(effect).runeTransfer;
  const compatible=state?.version===1&&state.source===RUNE_TRANSFER_SOURCE&&state.featId===feat.id;
  const saved=compatible&&typeof state.selectedWeaponId==='string'&&get(actor,state.selectedWeaponId)?.type==='weapon'?state.selectedWeaponId:null;
  const options=candidates(actor),selected=saved??(options.length===1?options[0].id:null);
  if(!effect){guard(actor);[effect]=await actor.createEmbeddedDocuments('Item',[buildRuneTransferEffect({actor,feat,selectedWeaponId:selected})]);guard(actor);return effect}
  const rules=[{key:RUNE_TRANSFER_KEY}];
  if(!compatible||state.selectedWeaponId!==selected||JSON.stringify(effect.system?.rules)!==JSON.stringify(rules)){
   const next=buildRuneTransferEffect({actor,feat,selectedWeaponId:selected,revision:(state?.revision??0)+1});
   await write(actor,effect,{[`flags.${MODULE_ID}.runeTransfer`]:next.flags[MODULE_ID].runeTransfer,'system.rules':rules});
  }
  return effect;
 }
 async function maintain(actor){if(!activeGM()||!real(actor)||!feature(actor)&&!states(actor).length)return;return queue.run(actor.uuid,()=>maintainInside(actor))}
 async function select(actor,user,force=false){
  guard(actor,user);return queue.run(actor.uuid,async()=>{
   guard(actor,user);if(!feature(actor))return null;
   const effect=await maintainInside(actor),state=own(effect).runeTransfer;
   if(!force&&selectedRuneWeaponId(actor))return effect;
   const options=candidates(actor),key=fingerprint(actor);
   if(!force&&state.declined===key)return effect;
   if(!options.length)return effect;
   const id=options.length===1?options[0].id:await choose?.({actor,user,title:'断天剑·碎地拳：选择传递符文的武器',choices:options.map(i=>({value:i.id,label:i.name}))});
   guard(actor,user);
   if(!id){await write(actor,effect,{[`flags.${MODULE_ID}.runeTransfer.declined`]:key},user);return effect}
   if(!options.some(i=>i.id===id)||!eligibleUsage(get(actor,id))||!feature(actor)||states(actor)[0]!==effect)throw Error('选择期间武器或专长已经改变，请重新使用专长选择。');
   await write(actor,effect,{[`flags.${MODULE_ID}.runeTransfer`]:{...state,selectedWeaponId:id,revision:state.revision+1,declined:null}},user);return effect;
  });
 }
 async function ensureReady(actor,user=game.user){
  if(!feature(actor)||!real(actor))return;
  if(selectedRuneWeaponId(actor))return runeTransferStatus(actor,game);
  const current=states(actor);if(current.length===1&&own(current[0]).runeTransfer?.declined===fingerprint(actor))return runeTransferStatus(actor,game);
  requireOwner(actor,user);
  if(activeGM()){await select(actor,user);return runeTransferStatus(actor,game)}
  if(user.id!==game.user.id||!socket||!game.users?.activeGM?.id)throw Error('需要在线主GM来保存首次符文选择。');
  const response=await socket.executeAsUser('rune-transfer:ready',game.users.activeGM.id,{actorUuid:actor.uuid});
  if(!response?.ok)throw Error(response?.error??'无法保存符文选择。');
  const receipt=response.value;if(!receipt)return;
  const start=Date.now();
  while(!states(actor).some(e=>e.id===receipt.effectId&&(own(e).runeTransfer?.revision??-1)>=receipt.revision)){
   if(Date.now()-start>5000)throw Error('符文选择尚未同步，本次攻击未执行。');
   await new Promise(resolve=>setTimeout(resolve,20));
  }
  return runeTransferStatus(actor,game);
 }
 function wrapStrike(strike,actor){
  if(!feature(actor))return strike;
  for(const[index,variant]of (strike?.variants??[]).entries()){
   const native=variant.roll;if(typeof native!=='function'||native.runeTransferWrapped)continue;
   const wrapped=async(params={})=>{
    if(isEatFortuneProbe(params))return native.call(variant,params);
    if(params[readyMarker])return native.call(variant,params);
    await ensureReady(actor,game.user);
    const usages=values(actor.system?.actions).flatMap(s=>[s,...s.altUsages??[]]);
    const current=usages.find(s=>s.item?.id===strike.item?.id&&(s.item.altUsageType??null)===(strike.item.altUsageType??null));
    if(current&&current!==strike&&current.variants?.[index]?.roll)return current.variants[index].roll({...params,[readyMarker]:true});
    return native.call(variant,{...params,[readyMarker]:true});
   };wrapped.runeTransferWrapped=true;variant.roll=wrapped;
  }
  return strike;
 }
 const resolveAction=item=>hasSource(item,RUNE_TRANSFER_SOURCE)?'rune-transfer:select':null;
 async function executeUsage({actor,item,user,action}){
  if(action!=='rune-transfer:select'||resolveAction(item)!==action||get(actor,item.id)!==item)throw Error('没有对应的符文选择专长。');
  await select(actor,user,true);const status=runeTransferStatus(actor,game);
  if(status.status==='capacity-conflict')return '已保存武器。属性符文合并超过容量，组合规则待裁定；保留原有属性符文，未自动覆盖或额外添加。基础符文仍按较高值生效。';
  return selectedRuneWeaponId(actor)?`已保存符文传递武器；装备和符文变化会自动更新。${status.skipped?.length?`以下符文不符合此用法或尚不支持：${status.skipped.join('、')}。`:''}`:'没有更改符文传递武器。';
 }
 function register({Hooks,socket:socketApi}={}){
  if(registered)return ()=>{};registered=true;socket=socketApi;
  socket?.register('rune-transfer:ready',async function(payload){
   try{if(typeof payload?.actorUuid!=='string')throw Error('角色来源无效。');const actor=await fromUuid(payload.actorUuid),user=game.users.get(this.socketdata.userId);guard(actor,user);const effect=await select(actor,user);return {ok:true,value:effect?{effectId:effect.id,revision:own(effect).runeTransfer.revision}:null}}catch(error){return {ok:false,error:error.message}}
  });
  const ids=[];
  const on=(name,fn)=>ids.push([name,Hooks.on(name,(...args)=>Promise.resolve().then(()=>fn(...args)).catch(onError))]);
  const itemChanged=item=>{if(item?.type==='weapon'||hasSource(item,RUNE_TRANSFER_SOURCE)||own(item).kind===KIND)return maintain(item.actor)};
  for(const name of ['createItem','updateItem','deleteItem'])on(name,itemChanged);
  return ()=>{for(const[name,id]of ids)Hooks.off(name,id);registered=false};
 }
 return {resolveAction,executeUsage,maintain,ensureReady,wrapStrike,register};
}
