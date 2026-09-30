import {MODULE_ID as ID} from './rules.mjs';
import {isActiveGM,markUnappliedDamageError} from './native-context.mjs';
import {SerialActions} from './runtime.mjs';

const values=collection=>Array.from(collection?.values?.()??collection??[]);
const own=message=>message.flags?.[ID]??{};
const author=message=>message.author?.id??message.user?.id??message.user;
const attackProof=message=>own(message).spellCombinationAttack?{...own(message).spellCombinationAttack,type:'spell'}:own(message).dualStrikeAttack?{...own(message).dualStrikeAttack,activityMessageId:own(message).dualStrikeAttack.usageMessageId,type:'dual'}:null;
const damageActivity=message=>own(message).spellCombinationDamage?.activityMessageId??own(message).dualStrike?.usageMessageId;
const nativeReroll=message=>message.isCheckRoll&&message.rolls?.some(roll=>roll._evaluated===true)&&(message.flags?.pf2e?.context?.isReroll===true||message.rolls.some(roll=>roll.options?.isReroll===true));
const sourceIds=message=>values(message.flags?.pf2e?.context?.options).filter(option=>typeof option==='string'&&option.startsWith(`${ID}:source:`)).map(option=>option.slice(`${ID}:source:`.length).split(':')[0]);
const hasApplication=message=>{
 const native=message.flags?.pf2e?.appliedDamage;if(native&&!native.isReverted)return true;
 const applied=message.flags?.['pf2e-toolbelt']?.targetHelper?.applied;
 const seen=value=>value===true||typeof value==='number'&&value!==0||value&&typeof value==='object'&&Object.values(value).some(seen);
 return !!seen(applied);
};

/** A kept PF2e reroll replaces the original check document. Retain a small
 * source index so its already-generated damage cannot continue using old DoS.
 * Resource payment and HP are never replayed or compensated here. */
export function createActivityResultLifecycle({game,getRollContext=()=>null,onError=()=>{},onReroll=async()=>{}}={}){
 const attacks=new Map(),damage=new Map(),receipts=new Map(),stale=new Set(),queue=new SerialActions();
 function track(message){
  const proof=attackProof(message),context=message.flags?.pf2e?.context;
  if(proof&&context?.type==='attack-roll'){
   const key=`${proof.activityMessageId}:${proof.index}`;
   const activity=game.messages.get(proof.activityMessageId);
   attacks.set(key,{...proof,messageId:message.id,actorUuid:message.flags.pf2e.origin?.actor,weaponUuid:message.flags.pf2e.origin?.uuid,targetUuid:context.target?.token,targetActorUuid:context.target?.actor,userId:author(activity??message),speakerActor:message.speaker?.actor});
  }
  const activity=damageActivity(message);if(activity){const set=damage.get(activity)??new Set();set.add(message.id);damage.set(activity,set);}
  if(own(message).activityResult)stale.add(message.id);
  if(context?.type==='damage-taken'&&message.flags?.pf2e?.appliedDamage)for(const id of sourceIds(message)){
   const set=receipts.get(id)??new Set();set.add(message.id);receipts.set(id,set);
  }
 }
 function affectedDamage(previous){
  return [...damage.get(previous.activityMessageId)??[]].map(id=>game.messages.get(id)).filter(card=>{
   const sources=own(card??{}).spellCombinationDamage?.attacks??own(card??{}).dualStrike?.attacks??[];
   return card&&sources.some(source=>source.messageId===previous.messageId);
  });
 }
 async function process(message){
  if(!nativeReroll(message)||game.messages.get(message.id)!==message)return;
  const context=message.flags?.pf2e?.context;if(context?.type!=='attack-roll')return;
  const options=values(context.options),ids=options.flatMap(option=>{
   if(typeof option!=='string')return [];
   for(const prefix of [`${ID}:spell-combination:`,`${ID}:dual-strike:`])if(option.startsWith(prefix))return [option.slice(prefix.length)];return [];
  });
  if(new Set(ids).size!==1)return;const activityId=ids[0],activity=game.messages.get(activityId);
  if(!activity||![author(activity),game.users.activeGM?.id].includes(author(message)))return;
  const candidates=[...attacks.values()].filter(record=>record.activityMessageId===activityId&&record.messageId!==message.id&&record.userId===author(activity)&&record.actorUuid===message.flags.pf2e.origin?.actor&&record.weaponUuid===message.flags.pf2e.origin?.uuid&&record.targetUuid===context.target?.token&&record.targetActorUuid===context.target?.actor&&record.speakerActor===message.speaker?.actor);
  if(candidates.length!==1)return;
  // Foundry does not await chat hooks. Every client must block the previous
  // damage before any GM write, queue wait, or broadcast can leave a gap.
  for(const card of affectedDamage(candidates[0]))stale.add(card.id);
  if(!isActiveGM(game))return;
  return queue.run(activityId,async()=>{
   if(!isActiveGM(game))return;const previous=candidates[0];
   const linkage=previous.type==='spell'?{activityMessageId:activityId,index:previous.index,kind:previous.kind}:{usageMessageId:activityId,index:previous.index};
   await message.update({[`flags.${ID}.${previous.type==='spell'?'spellCombinationAttack':'dualStrikeAttack'}`]:linkage});
   if(!isActiveGM(game))return;
   let applied=false;
   for(const card of affectedDamage(previous)){
    const used=hasApplication(card)||[...receipts.get(card.id)??[]].some(receipt=>hasApplication(game.messages.get(receipt)??{}));applied||=used;
    await card.update({[`flags.${ID}.activityResult`]:{status:used?'undo-required':'superseded',activityMessageId:activityId,previousCheckId:previous.messageId,keptCheckId:message.id}});
   }
   const continuation=previous.type==='spell'?'已付款的法术可沿原法术卡继续；大成功时请核对法术倍伤和附加持续伤害。':'双重切割的精确伤害保留一次。';
   await activity.update({[`flags.${ID}.usage`]:{...own(activity).usage,status:'waiting',result:`检定已重掷；旧组合伤害已失效。${applied?'请先沿原生流程撤销旧伤害。':'如旧伤害已应用，请先沿原生流程撤销。'}${continuation}请GM按保留结果核对并合并伤害，对每个目标应用一次。`}});
   track(message);await onReroll({activity,previous,message,applied});
  });
 }
 function beforeDamage(_actor,params){
  const ids=new Set(),tracked=getRollContext(params.damage);if(tracked?.messageId)ids.add(tracked.messageId);
  for(const option of values(params.rollOptions))if(typeof option==='string'&&option.startsWith(`${ID}:source:`))ids.add(option.slice(`${ID}:source:`.length).split(':')[0]);
  if([...ids].some(id=>stale.has(id)||own(game.messages.get(id)??{}).activityResult))throw markUnappliedDamageError(Error('组合检定已重掷，旧伤害已失效；如已应用，请先沿原生流程撤销。'));
  return null;
 }
 function applyNativeDamage(actor,params,native){
  // Earlier providers can await a reaction while a kept reroll supersedes this
  // source. Check again at the actual native HP-write boundary.
  beforeDamage(actor,params);return native(params);
 }
 function register({Hooks}={}){
  const registrations=[],on=(event,fn)=>registrations.push([event,Hooks.on(event,fn)]);
  for(const message of values(game.messages))track(message);
  on('preCreateChatMessage',message=>{
   if(!nativeReroll(message))return;
   const options=values(message.flags?.pf2e?.context?.options);
   if(options.some(option=>typeof option==='string'&&(option.startsWith(`${ID}:spell-combination:`)||option.startsWith(`${ID}:dual-strike:`))))message.updateSource({'flags.xdy-pf2e-workbench.noAutoDamageRoll':true});
  });
  on('createChatMessage',async message=>{await process(message).catch(onError);track(message);});
  on('updateChatMessage',track);
  on('deleteChatMessage',message=>{stale.delete(message.id);const activity=damageActivity(message);if(activity)damage.get(activity)?.delete(message.id);for(const id of sourceIds(message)){const set=receipts.get(id);set?.delete(message.id);if(!set?.size)receipts.delete(id);}if(damage.has(message.id)){damage.delete(message.id);for(const[key,record]of attacks)if(record.activityMessageId===message.id)attacks.delete(key);}});
  on('renderChatMessageHTML',(message,html)=>{
   const state=own(message).activityResult??(stale.has(message.id)?{status:'superseded'}:null),root=html?.[0]??html;root?.querySelector?.('.third-party-stale-damage')?.remove();if(!state)return;
   for(const button of root?.querySelectorAll?.('[data-action="apply-damage"],[data-action="applyDamage"],.damage-buttons button,.damage-application button,.target-damage-application button')??[])button.disabled=true;
   (root?.querySelector?.('.message-content')??root)?.insertAdjacentHTML?.('beforeend',`<p class="third-party-stale-damage" role="status">检定已重掷，此卡伤害已失效。${state.status==='undo-required'?'请先使用原生撤销。':'如旧伤害已应用，请先使用原生撤销。'}</p>`);
  });
  return()=>{for(const[event,id]of registrations)Hooks.off(event,id);attacks.clear();damage.clear();receipts.clear();stale.clear();};
 }
 return {register,beforeDamage,applyNativeDamage,process};
}
