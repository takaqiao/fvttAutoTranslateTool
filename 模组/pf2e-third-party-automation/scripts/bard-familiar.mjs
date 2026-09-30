import {MODULE_ID as ID} from './rules.mjs';
import {getNativeCastEvents} from './amp-cast-events.mjs';
import {getSourceId,isActiveGM,showNativeChoice} from './native-context.mjs';
import {isActualUseMessage} from './usage-events.mjs';

export const BARD_FAMILIAR_SOURCES=Object.freeze({
 focus:'Compendium.pf2e.familiar-abilities.Item.jdlefpPcSCIe27vO',
 accompanist:'Compendium.pf2e.familiar-abilities.Item.92lgSEPFIDLvKOCF',
});
const S=BARD_FAMILIAR_SOURCES,ACTION='bard-familiar:focus',BONUS=`${ID}-accompanist`;
const values=collection=>Array.from(collection?.values?.()??collection??[]);
const states=new WeakMap();
const stateFor=game=>{if(!states.has(game))states.set(game,{payments:new Map()});return states.get(game);};
const ability=(item,source)=>item?.type==='action'&&item.actor?.type==='familiar'&&getSourceId(item)===source;
const owner=(actor,user)=>!!user&&actor?.testUserPermission?.(user,'OWNER')===true;
const authorId=message=>message?.author?.id??message?.user?.id??message?.user;
const boundedId=id=>typeof id==='string'&&/^[A-Za-z0-9_-]{1,100}$/.test(id);
const recordMatches=(record,expected)=>record&&Object.entries(expected).every(([key,value])=>record[key]===value);

/** These two original abilities use native owned items, native Use and native Check.roll. */
export function createBardFamiliarProvider({game,fromUuid=globalThis.fromUuid,confirm=showNativeChoice,onError=()=>{},randomId=()=>globalThis.foundry?.utils?.randomID?.()??globalThis.crypto.randomUUID()}={}){
 const state=stateFor(game),castEvents=getNativeCastEvents({game,fromUuid});
 const resolveAction=item=>ability(item,S.focus)?ACTION:undefined;
 function masterOf(familiar){
  const master=game.actors.get(familiar?.system?.master?.id);
  return familiar?.type==='familiar'&&master?.type==='character'&&familiar.master?.uuid===master.uuid?master:null;
 }
 function focusParts(item,user){
  const familiar=item?.actor,master=masterOf(familiar);
  if(!ability(item,S.focus)||game.actors.get(familiar.id)!==familiar||familiar.items.get(item.id)!==item||!master)throw Error('魔宠聚能的原始能力或当前主人关系已改变。');
  if(!owner(familiar,user)||!owner(master,user))throw Error('没有魔宠及其主人的所有者权限。');
  const frequency=item.system.frequency;
  if(frequency?.max!==1||frequency.per!=='day')throw Error('魔宠聚能的原生每日次数配置不符。');
  return {familiar,master};
 }
 function poolOf(master){
  const pool=master.system.resources?.focus;
  if(!Number.isInteger(pool?.max)||pool.max<1||!Number.isInteger(pool.value)||pool.value<0||pool.value>pool.max)throw Error('主人没有可用的原生聚能池。');
  return pool;
 }
 function canAct(familiar){if(familiar.canAct!==true||familiar.isDead===true)throw Error('魔宠目前不能行动。');}
 function beforeUse(item,user=game.user){
  if(!resolveAction(item))return true;
  const {familiar,master}=focusParts(item,user);canAct(familiar);
  const pool=poolOf(master);if(pool.value>=pool.max)throw Error('主人的聚能池已满，未使用魔宠聚能。');
  if(item.system.frequency.value!==1)throw Error('魔宠聚能没有剩余的每日次数。');
  return true;
 }
 function requireGM(){if(!isActiveGM(game))throw Error('魔宠聚能只能由当前主GM结算。');}
 function usageContext({actor,item,message,user,action,frequencyReceipt:r}){
  requireGM();const {familiar,master}=focusParts(item,user);
  const origin=message?.flags?.pf2e?.origin,input=message?.flags?.[ID]?.usageInput;
  if(actor!==familiar||action!==ACTION||game.messages.get(message?.id)!==message||authorId(message)!==user.id||message.speaker?.actor!==familiar.id||origin?.uuid!==item.uuid||origin.actor!==familiar.uuid||origin.type!=='action'||!isActualUseMessage(message)||message.isRoll||message.isCheckRoll||message.rolls?.length)throw Error('需要由原操作者实际使用的原始魔宠聚能消息。');
  if(!boundedId(r?.id)||input?.frequencyReceiptId!==r.id||r.itemUuid!==item.uuid||r.userId!==user.id||r.before!==1||r.after!==0)throw Error('魔宠聚能的次数回执无效。');
  const expected={id:r.id,itemUuid:item.uuid,familiarUuid:familiar.uuid,masterUuid:master.uuid,messageUuid:message.uuid,userId:user.id};
  return {familiar,master,r,expected};
 }
 function paymentIsCurrent(item,ctx){
  const payment=item.flags?.[ID]?.bardFamiliar?.payment,observed=state.payments.get(ctx.r.id);
  return boundedId(payment?.nonce)&&observed?.nonce===payment.nonce&&recordMatches(payment,{itemUuid:item.uuid,masterUuid:ctx.master.uuid,userId:ctx.expected.userId,before:1,after:0})&&recordMatches(observed,{itemUuid:item.uuid,userId:ctx.expected.userId})&&item.system.frequency.value===0;
 }
 async function executeUsage(context){
  const initial=usageContext(context);
  return castEvents.withActorResourceLock(initial.master,async()=>{
   const ctx=usageContext(context),{item}=context,{master,familiar,r,expected}=ctx;
   if(master.uuid!==initial.master.uuid)throw Error('魔宠的主人已改变，请核对本次使用。');
   const prior=master.flags?.[ID]?.bardFamiliar?.focus?.[r.id]??item.flags?.[ID]?.bardFamiliar?.refunds?.[r.id];
   if(prior){if(!recordMatches(prior,expected))throw Error('次数回执已经关联另一次使用。');return prior.result;}
   canAct(familiar);const pool=poolOf(master);
   if(!paymentIsCurrent(item,ctx))throw Error('本次付款标记已改变，不能恢复聚能或退回其他使用的次数。');
   requireGM();
   if(pool.value>=pool.max){
    const result='主人的聚能池已满；已退回本次魔宠聚能的每日次数。';
    // The refund and its original receipt share one native item update. A later
    // payment has a different durable nonce even if its counter is zero again.
    await item.update({'system.frequency.value':1,[`flags.${ID}.bardFamiliar.refunds.${r.id}`]:{...expected,status:'refunded',paymentNonce:item.flags[ID].bardFamiliar.payment.nonce,result}});
    return result;
   }
   const result='已为魔宠的主人恢复1点聚能。';
   // A single document update commits both the resource and replay protection.
   // If the transport reply is lost, never compensate or blindly apply again.
   await master.update({'system.resources.focus.value':pool.value+1,[`flags.${ID}.bardFamiliar.focus.${r.id}`]:{...expected,status:'done',before:pool.value,after:pool.value+1,paymentNonce:item.flags[ID].bardFamiliar.payment.nonce,result}});
   return result;
  });
 }
 function accompanistFor(master){
  const matches=values(game.actors).filter(f=>masterOf(f)===master&&f.canAct===true&&f.isDead!==true).flatMap(f=>values(f.items).filter(i=>ability(i,S.accompanist)).map(item=>({familiar:f,item})));
  return matches.length===1?matches[0]:null;
 }
 async function interceptCheck(native,check,context={},...args){
  if(context.type!=='skill-check'||context.isReroll||check?.slug!=='performance'||!values(context.domains).includes('performance')||check.modifiers?.some(m=>m.slug===BONUS))return native(check,context,...args);
  const actor=context.actor??(context.origin?.self?context.origin?.actor:context.target?.actor);
  const master=actor?.uuid?await fromUuid(actor.uuid):null,match=master&&accompanistFor(master);
  if(!match||master.type!=='character'||!owner(master,game.user))return native(check,context,...args);
  const answer=await confirm({actor:master,user:game.user,title:'伴奏者：确认本次表演检定',choices:[{value:'yes',label:'魔宠在身边，并且能够行动'},{value:'no',label:'不满足本次条件／不使用'}]});
  const current=accompanistFor(master);
  if(answer!=='yes'||!owner(master,game.user)||game.actors.get(master.id)!==master||current?.item!==match.item||current?.familiar!==match.familiar)return native(check,context,...args);
  const rank=master.getStatistic?.('performance')?.rank??master.skills?.performance?.rank;
  if(!Number.isInteger(rank)||rank<0||rank>4)return native(check,context,...args);
  const bonus=new game.pf2e.Modifier({slug:BONUS,label:'伴奏者',modifier:rank>=3?2:1,type:'circumstance'});
  const next=new game.pf2e.CheckModifier(check.slug,{modifiers:check.modifiers},[bonus],context.options);
  return native(next,context,...args);
 }
 function register({Hooks}={}){
  const hooks=[],listeners=[],elements=new WeakSet(),on=(event,fn)=>hooks.push([event,Hooks.on(event,fn)]);
  on('preUpdateItem',(item,changes,options,userId)=>{
   if(!resolveAction(item)||item.system.frequency?.value!==1||(changes['system.frequency.value']??changes.system?.frequency?.value)!==0)return;
   const nonce=randomId(),master=masterOf(item.actor);if(!boundedId(nonce)||!master)return;
   // Augment the existing payment write; never perform another frequency write.
   changes[`flags.${ID}.bardFamiliar.payment`]={nonce,itemUuid:item.uuid,masterUuid:master.uuid,userId,before:1,after:0};
   options[ID]={...options[ID],familiarFocusPayment:nonce};
  });
  on('updateItem',(item,_changes,options,userId)=>{
   const proof=options?.[ID]?.frequencyReceipt,nonce=options?.[ID]?.familiarFocusPayment;
   if(!resolveAction(item)||!boundedId(proof?.id)||!boundedId(nonce)||proof.itemUuid!==item.uuid||proof.userId!==userId||proof.before!==1||proof.after!==0||item.flags?.[ID]?.bardFamiliar?.payment?.nonce!==nonce||item.system.frequency.value!==0)return;
   // One current observation per familiar; claimed old receipts cannot refund a
   // newer payment after a daily recharge or a manual resource adjustment.
   for(const[id,payment]of state.payments)if(payment.itemUuid===item.uuid)state.payments.delete(id);
   state.payments.set(proof.id,{nonce,itemUuid:item.uuid,userId});
  });
  const capture=(app,html)=>{
   const element=html?.[0]??html,actor=app.actor??app.document;if(actor?.type!=='familiar'||!element?.addEventListener||elements.has(element))return;elements.add(element);
   const listener=event=>{
    const button=event.target?.closest?.('[data-action="use-action"],button.use-action'),id=button?.closest?.('[data-item-id]')?.dataset.itemId,item=actor.items.get(id);if(!resolveAction(item))return;
    try{beforeUse(item);}catch(error){event.preventDefault();event.stopImmediatePropagation();onError(error);}
   };
   element.addEventListener('click',listener,true);listeners.push([element,listener]);
  };
  for(const event of ['renderFamiliarSheetPF2e','renderActorSheetPF2e','renderActorSheetV2'])on(event,capture);
  return()=>{for(const[name,id]of hooks)Hooks.off(name,id);for(const[element,listener]of listeners)element.removeEventListener('click',listener,true);};
 }
 return {resolveAction,requiresActualUse:item=>!!resolveAction(item),tracksFrequency:item=>!!resolveAction(item),beforeUse,executeUsage,interceptCheck,register};
}
