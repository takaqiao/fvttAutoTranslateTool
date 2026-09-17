import {MODULE_ID as ID} from './rules.mjs';
import {getSourceId,isActiveGM} from './native-context.mjs';
import {isActualUseMessage} from './usage-events.mjs';
import {getNativeCastEvents} from './amp-cast-events.mjs';

export const PRAYER_SOURCES=Object.freeze({
 prayer:'Compendium.pf2e.feats-srd.Item.WYaKRREZUSH0jel5',
 devotion:'Compendium.pf2e.classfeatures.Item.Q1VfQZp49hkhY0HY',
 domain:'Compendium.pf2e.feats-srd.Item.FXKIALDXAzEBfj5A',
 lay:'Compendium.pf2e.spells-srd.Item.zNN9212H2FGfM7VS',
 surge:'Compendium.pf2e.spells-srd.Item.W37iBXLsY2trJ1rS',
});
const S=PRAYER_SOURCES,PATH=`flags.${ID}.desperatePrayer`,ACTION='desperate-prayer:use';
const values=c=>Array.from(c?.values?.()??c??[]),copy=v=>structuredClone(v);
const data=a=>a?.flags?.[ID]?.desperatePrayer??{},focus=a=>a?.system?.resources?.focus;
const bounded=id=>typeof id==='string'&&/^[A-Za-z0-9_-]{1,100}$/.test(id);
const owner=(a,u)=>!!u&&a?.testUserPermission?.(u,'OWNER')===true;
const source=(a,s)=>values(a?.items).find(i=>getSourceId(i)===s);
const resolveAction=i=>i?.type==='feat'&&i.actor?.type==='character'&&getSourceId(i)===S.prayer?ACTION:undefined;
const match=(a,b)=>!!a&&Object.entries(b).every(([k,v])=>a[k]===v);
const author=m=>m.author?.id??m.user?.id??m.user;

/** A one-turn restricted point, paid through PF2e's original focus update. */
export function createDesperatePrayerProvider({game,fromUuid=globalThis.fromUuid,choose,useOriginal,onError=()=>{},randomId=()=>globalThis.foundry?.utils?.randomID?.()??globalThis.crypto.randomUUID(),castEvents=getNativeCastEvents({game,fromUuid})}={}){
 const payments=new Map(),pendingUses=new Map();let socket,installed=false;
 const managed=a=>a?.type==='character'&&!!source(a,S.prayer);
 const lock=(a,fn)=>castEvents.withActorResourceLock(a,fn);
 function requireGM(){if(!isActiveGM(game))throw Error('绝境祷告资源只能由当前主GM结算。');}
 function checkActor(a,u){requireGM();if(game.actors.get(a?.id)!==a||!owner(a,u))throw Error('绝境祷告角色或所有者权限不符。');}
 function pool(a){const p=focus(a);if(!Number.isInteger(p?.value)||!Number.isInteger(p.max)||p.max<1||p.value<0||p.value>p.max)throw Error('无法确认原生聚能池。');return p;}
 function current(a,w){
  const c=game.combats?.get(w?.combatId)??(game.combat?.id===w?.combatId?game.combat:null),t=c?.combatant;
  return !!(w&&c?.started&&c.round===w.round&&t?.id===w.combatantId&&t.actor===a&&t.token?.uuid===w.tokenUuid&&c.turns?.[c.turn]===t);
 }
 async function save(a,next,extra={}){requireGM();return a.update({...extra,[PATH]:next});}
 function beforeUse(item,user=game.user){
  if(!resolveAction(item))return true;
  const a=item.actor,w=data(a).window;
  if(!owner(a,user)||game.actors.get(a.id)!==a||a.items.get(item.id)!==item||a.canAct!==true||a.isDead===true)throw Error('原始绝境祷告角色、能力或权限已改变。');
  if(w?.state!=='open'||w.gmId!==game.users.activeGM?.id||w.userId!==user.id||!current(a,w))throw Error('绝境祷告只可在本次真实起回合机会使用；主GM不能中途改变。');
  if(pool(a).value!==0||item.system.frequency?.value!==1||item.system.frequency.max!==1||item.system.frequency.per!=='day')throw Error('绝境祷告需要0聚能且有本次每日次数。');
  return true;
 }
 async function closeWindow(a,nonce){
  return lock(a,async()=>{const d=data(a);if(d.window?.nonce===nonce&&d.window.state==='open')await save(a,{...d,window:{...d.window,state:'closed'}});});
 }
 async function beforeAction(a){
  const w=data(a).window;if(w?.state!=='open')return;
  if(isActiveGM(game))return closeWindow(a,w.nonce);
  if(!socket||!game.users.activeGM)throw Error('需要主GM关闭起回合选择后再行动。');
  const response=await socket.executeAsUser('desperate-prayer:close',game.users.activeGM.id,{actorUuid:a.uuid,nonce:w.nonce});
  if(!response?.ok)throw Error(response?.error??'无法关闭起回合选择。');
 }
 async function interceptCast({item,options={}},native){
  // A silent cast (message:false), including macro casts, still spends actions
  // and resources. Only explicit non-consuming previews bypass this boundary.
  if(options.consume!==false)await beforeAction(item.actor);
  return native();
 }
 async function interceptCheck(native,check,context={},...args){
  await beforeAction(context.actor??context.origin?.actor);
  return native(check,context,...args);
 }
 function captureUsage(item){
  if(!resolveAction(item))return null;
  const payment=item.flags?.[ID]?.prayerPayment;
  return {prayerInput:{opportunityNonce:payment?.opportunityNonce??null,paymentNonce:payment?.nonce??null}};
 }
 function paymentContext(ctx){
  const {actor:a,item,message:m,user,frequencyReceipt:r}=ctx;checkActor(a,user);
  const origin=m?.flags?.pf2e?.origin,input=m?.flags?.[ID]?.prayerInput,payment=item?.flags?.[ID]?.prayerPayment;
  if(resolveAction(item)!==ACTION||ctx.action!==ACTION||item.actor!==a||a.items.get(item.id)!==item||game.messages.get(m?.id)!==m||author(m)!==user.id||m.speaker?.actor!==a.id||origin?.uuid!==item.uuid||origin.actor!==a.uuid||origin.type!=='feat'||!isActualUseMessage(m)||m.isRoll||m.isCheckRoll||m.rolls?.length)throw Error('需要原作者实际使用的原始绝境祷告消息。');
  if(item.system.frequency?.max!==1||item.system.frequency.per!=='day')throw Error('原生日次数配置已改变。');
  if(!bounded(r?.id)||m.flags[ID].usageInput?.frequencyReceiptId!==r.id||!match(r,{itemUuid:item.uuid,userId:user.id,before:1,after:0})||!bounded(input?.opportunityNonce)||!bounded(input?.paymentNonce))throw Error('绝境祷告每日次数回执不符。');
  return {a,item,m,user,r,input,payment};
 }
 async function executeUsage(ctx){
  const initial=paymentContext(ctx);
  try{return await lock(initial.a,async()=>{
   const {a,item,m,user,r,input,payment}=paymentContext(ctx),d=data(a),c=d.credit;
   const identity={frequencyReceiptId:r.id,messageUuid:m.uuid,itemUuid:item.uuid,userId:user.id};
   if(c?.frequencyReceiptId===r.id){if(!match(c,identity))throw Error('绝境祷告回执已关联另一消息。');return '本次绝境祷告已经结算。';}
   const refund=item.flags?.[ID]?.prayerRefund;if(refund?.frequencyReceiptId===r.id){if(!match(refund,identity))throw Error('退款回执来源不符。');return '本次起回合机会已失效，次数已退回。';}
   if(!match(payment,{nonce:input.paymentNonce,opportunityNonce:input.opportunityNonce,userId:user.id,itemUuid:item.uuid})||payments.get(r.id)!==input.paymentNonce||item.system.frequency.value!==0)throw Error('本次原生日次数付款无法确认，未重复扣费或退款。');
   const w=d.window;
   if(w?.nonce!==input.opportunityNonce||w.gmId!==game.users.activeGM?.id||w.userId!==user.id||w.state!=='open'||!current(a,w)||pool(a).value!==0||a.canAct!==true||a.isDead===true){
    await item.update({'system.frequency.value':1,[`flags.${ID}.prayerRefund`]:{...identity,paymentNonce:payment.nonce}});
    return '本次起回合机会已失效，已退回这一次每日次数。';
   }
   const credit={...identity,nonce:w.nonce,combatId:w.combatId,combatantId:w.combatantId,tokenUuid:w.tokenUuid,round:w.round,state:'available',remaining:1,totalObserved:1,payments:[]};
   await save(a,{...d,window:{...w,state:'used'},credit},{'system.resources.focus.value':1});
   return '已获得本回合的1点临时聚能，仅可用于当前虔诚法术。';
  });}finally{pendingUses.get(initial.input.opportunityNonce)?.();}
 }
 function devotion(item){
  const a=item.actor,entry=item.spellcasting??a.items.get(item.system.location?.value);
  if(item.type!=='spell'||entry?.type!=='spellcastingEntry'||entry.actor!==a||entry.system.prepared?.value!=='focus'||a.items.get(entry.id)!==entry||item.system.location?.value!==entry.id)return false;
  if(getSourceId(item)===S.lay)return !!source(a,S.devotion);
  const domain=source(a,S.domain);
  return getSourceId(item)===S.surge&&!!source(a,S.devotion)&&domain?.system.rules?.some(r=>r.key==='ChoiceSet'&&r.flag==='deitysDomain'&&r.selection==='zeal')===true;
 }
 async function expire(a,turn){
  const d=data(a),c=d.credit;
  if(c?.state!=='available'||c.remaining!==1||turn&&!match(c,turn))return;
  if(pool(a).value!==c.totalObserved||c.totalObserved<1){await save(a,{...d,credit:{...c,state:'uncertain'}});onError(Error('绝境祷告的聚能来源已改变，未自动扣除到期点。'));return;}
  await save(a,{...d,credit:{...c,state:'expired',remaining:0,totalObserved:c.totalObserved-1}},{'system.resources.focus.value':c.totalObserved-1});
 }
 async function onEndTurn(combatant,round=combatant.encounter?.round){
  if(!isActiveGM(game)||!combatant.actor)return;
  const a=combatant.actor;await lock(a,async()=>{const d=data(a);if(d.window?.combatId===combatant.encounter?.id&&d.window.round===round&&d.window.state==='open')await save(a,{...d,window:{...d.window,state:'closed'}});await expire(a,{combatId:combatant.encounter?.id,combatantId:combatant.id,round});});
 }
 async function consumePolicy(context,next){
  const {actor:a,item,user,payload}=context;if(!managed(a)&&!data(a).credit)return next();checkActor(a,user);
  if(data(a).window?.state==='open')throw Error('请先完成本次起回合的绝境祷告选择。');
  if(data(a).credit?.state==='available'&&!current(a,data(a).credit))await expire(a);
  const d=data(a),c=d.credit,cost=item.system.cast?.focusPoints??0;
  if(!c||c.remaining!==1||!['available','uncertain'].includes(c.state)||cost===0)return next();
  const total=pool(a).value;
  if(!Number.isInteger(cost)||cost<0||cost!==payload.focusPoints)throw Error('原生聚能费用已改变。');
  const temporary=c.state==='available'&&current(a,c)&&devotion(item)?1:0;
  if(total-(temporary?0:1)<cost)throw Error('本次法术没有足够普通聚能；临时点仅可用于当前虔诚法术。');
  if(typeof context.expectFocusCommit!=='function')throw Error('无法确认原生聚能提交接口，未消费资源。');
  context.expectFocusCommit({before:total,cost,changes:proof=>{
   const now=data(a),credit=now.credit;
   if(credit?.nonce!==c.nonce||credit.state!==c.state||credit.remaining!==1||credit.totalObserved!==total||proof.castNonce!==payload.id||proof.itemUuid!==item.uuid||proof.before!==total||proof.after!==total-cost)throw Error('本次临时聚能付款来源已改变。');
   const payment={...proof,temporarySpent:temporary,ordinarySpent:cost-temporary};
   return {[PATH]:{...now,credit:{...credit,remaining:1-temporary,state:temporary?'spent':credit.state,totalObserved:proof.after,payments:[...(credit.payments??[]).slice(-19),payment]}}};
  }});
  let result;
  try{result=await next();}catch(error){
   if(!data(a).credit?.payments?.some(p=>p.castNonce===payload.id)&&data(a).credit?.nonce===c.nonce)await save(a,{...data(a),credit:{...data(a).credit,state:'uncertain'}});
   throw error;
  }
  if(result&&!data(a).credit?.payments?.some(p=>p.castNonce===payload.id)){
   if(data(a).credit?.nonce===c.nonce)await save(a,{...data(a),credit:{...data(a).credit,state:'uncertain'}});
   throw Error('原生聚能支付没有绑定回执；保留不确定状态，不会到期误扣。');
  }
  return result;
 }
 async function originalUse(item,user,window){
  if(useOriginal)return useOriginal({item,user,opportunity:window});
  const payload={itemUuid:item.uuid,opportunityNonce:window.nonce};
  if(user.id===game.user.id)return localUse(payload);
  if(!socket)throw Error('无法请求所有者使用原始绝境祷告。');
  const response=await socket.executeAsUser('desperate-prayer:use',user.id,payload);if(!response?.ok)throw Error(response?.error??'原始绝境祷告使用失败。');
 }
 async function localUse(payload){
  const item=await fromUuid(payload.itemUuid);beforeUse(item);
  if(data(item.actor).window.nonce!==payload.opportunityNonce)throw Error('起回合选择已经过期。');
  return game.pf2e.rollItemMacro(item.uuid);
 }
 async function onStartTurn(combatant){
  if(!isActiveGM(game))return;
  const a=combatant.actor,c=combatant.encounter,item=source(a,S.prayer);
  if(!item||!managed(a)||!c?.started||c.combatant!==combatant||c.turns?.[c.turn]!==combatant||!combatant.token?.uuid||!Number.isInteger(c.round))return;
  const prior=data(a).lastStart,key={combatId:c.id,combatantId:combatant.id,round:c.round};if(match(prior,key))return;
  const user=values(game.users).find(u=>u.active&&!u.isGM&&owner(a,u))??game.user;
  let window;
  await lock(a,async()=>{
   requireGM();if(!current(a,{...key,tokenUuid:combatant.token.uuid}))return;
   const d=data(a);if(match(d.lastStart,key))return;
   if(d.credit?.state==='available')await expire(a);
   window={...key,nonce:randomId(),tokenUuid:combatant.token.uuid,userId:user.id,gmId:game.user.id,state:'closed'};
   if(bounded(window.nonce)&&owner(a,user)&&a.canAct===true&&a.isDead!==true&&pool(a).value===0&&item.system.frequency?.value===1)window.state='open';
   await save(a,{...data(a),lastStart:key,window});
  });
  if(window?.state!=='open')return;
  let timer;
  try{
   const answer=await choose({actor:a,user,title:'绝境祷告：本次回合开始',choices:[{value:'use',label:'使用原始绝境祷告'},{value:'skip',label:'不使用'}]});
   if(answer!=='use'||data(a).window?.state!=='open'||data(a).window.nonce!==window.nonce||!current(a,window))return;
   beforeUse(item,user);
   // Only wait for this use's settlement. The timeout is a transport boundary,
   // never a rule window allowing a later turn action to claim the feature.
   const settled=new Promise(resolve=>pendingUses.set(window.nonce,resolve));
   await originalUse(item,user,window);
   await Promise.race([settled,new Promise((_,reject)=>{timer=setTimeout(()=>reject(Error('原始绝境祷告结算等待超时，请核对这一次日次数。')),15000)})]);
  }finally{clearTimeout(timer);pendingUses.delete(window.nonce);await closeWindow(a,window.nonce);}
 }
 function register({Hooks,libWrapper,socket:api}={}){
  if(installed)return()=>{};installed=true;socket=api;const hooks=[],paths=[],listeners=[],restores=[],elements=new WeakSet();
  const on=(k,fn)=>hooks.push([k,Hooks.on(k,fn)]),wrap=(p,fn)=>{libWrapper?.register(ID,p,fn,'MIXED');paths.push(p)};
  on('preUpdateItem',(item,changes,options,userId)=>{
   if(!resolveAction(item)||item.system.frequency?.value!==1||(changes['system.frequency.value']??changes.system?.frequency?.value)!==0)return;
   beforeUse(item,game.users.get(userId));const w=data(item.actor).window,nonce=randomId();if(!bounded(nonce))throw Error('付款nonce无效。');
   changes[`flags.${ID}.prayerPayment`]={nonce,opportunityNonce:w.nonce,userId,itemUuid:item.uuid};options[ID]={...options[ID],prayerPaymentNonce:nonce};
  });
  on('updateItem',(item,_changes,options,userId)=>{
   const r=options?.[ID]?.frequencyReceipt,p=item.flags?.[ID]?.prayerPayment;
   if(resolveAction(item)&&bounded(r?.id)&&match(r,{itemUuid:item.uuid,userId,before:1,after:0})&&p?.nonce===options[ID].prayerPaymentNonce&&item.system.frequency.value===0)payments.set(r.id,p.nonce);
  });
  on('preUpdateActor',(actor,changes)=>{
   const c=data(actor).credit,after=changes['system.resources.focus.value']??changes.system?.resources?.focus?.value;
   if(!c||c.remaining!==1||!['available','uncertain'].includes(c.state)||after===undefined||changes[PATH]||changes.flags?.[ID]?.desperatePrayer)return;
   const before=focus(actor)?.value;
   const knownIncrease=c.state==='available'&&before===c.totalObserved&&Number.isInteger(after)&&after>=before&&after<=focus(actor).max;
   changes[PATH]={...data(actor),credit:{...c,totalObserved:after,state:knownIncrease?'available':'uncertain'}};
  });
  wrap('CONFIG.Combatant.documentClass.prototype.onStartTurn',async function(native,...args){const result=await native(...args);await onStartTurn(this);return result});
  wrap('CONFIG.Combatant.documentClass.prototype.onEndTurn',async function(native,...args){const result=await native(...args);await onEndTurn(this,args[0]?.round??this.encounter?.round);return result});
  wrap('CONFIG.Token.documentClass.prototype._preUpdateMovement',async function(native,...args){await beforeAction(this.actor);return native(...args)});
  // Native basic actions are not owned Item.toMessage calls (e.g. Raise a Shield).
  // Brand their actual variants, including future variants made by the original
  // factory, rather than guessing an action from a generic chat-card title.
  const variants=new WeakSet(),prototypes=new Set();
  for(const action of values(game.pf2e.actions)){
   if(typeof action.toActionVariant!=='function')continue;
   const original=action.toActionVariant,variant=original.call(action);if(!variant?.use)continue;
   prototypes.add(Object.getPrototypeOf(variant));variants.add(variant);for(const v of values(action.variants))if(v&&typeof v==='object')variants.add(v);
   const factory=function(...args){const v=original.apply(this,args);if(v&&typeof v==='object')variants.add(v);return v};action.toActionVariant=factory;
   restores.push(()=>{if(action.toActionVariant===factory)action.toActionVariant=original;});
  }
  for(const prototype of prototypes){
   const descriptor=Object.getOwnPropertyDescriptor(prototype,'use'),native=prototype.use;
   const wrapper=async function(params={}){
    if(installed&&variants.has(this)&&params.message?.create!==false){
     const actors=params.actors?(Array.isArray(params.actors)?params.actors:[params.actors]):values(game.user.getActiveTokens?.()).map(t=>(t.document??t).actor);
     if(!actors.length&&game.user.character)actors.push(game.user.character);
     for(const a of new Set(actors))await beforeAction(a);
    }
    return native.call(this,params);
   };
   Object.defineProperty(prototype,'use',{configurable:true,writable:true,value:wrapper});restores.push(()=>{if(prototype.use===wrapper){if(descriptor)Object.defineProperty(prototype,'use',descriptor);else delete prototype.use;}});
  }
  on('preUpdateToken',(token,changes)=>{if(['x','y','elevation'].some(k=>Object.hasOwn(changes,k))&&data(token.actor).window?.state==='open'){onError(Error('请先完成起回合选择，再直接调整Token位置。'));return false;}});
  const capture=(app,html)=>{const element=html?.[0]??html,a=app.actor??app.document;if(!a?.items||!element?.addEventListener||elements.has(element))return;elements.add(element);
   const fn=event=>{const button=event.target?.closest?.('[data-action="use-action"],button.use-action'),item=a.items.get(button?.closest?.('[data-item-id]')?.dataset.itemId);if(!item)return;
    try{if(resolveAction(item))beforeUse(item);else if(data(a).window?.state==='open')throw Error('请先完成起回合的绝境祷告选择。');}catch(e){event.preventDefault();event.stopImmediatePropagation();onError(e);}};
   element.addEventListener('click',fn,true);listeners.push([element,fn]);};
  for(const k of ['renderCharacterSheetPF2e','renderActorSheetPF2e','renderActorSheetV2'])on(k,capture);
  socket?.register('desperate-prayer:use',async function(payload){try{if(this.socketdata.userId!==game.users.activeGM?.id)throw Error('只有主GM可请求原始起回合使用。');await localUse(payload);return {ok:true}}catch(error){return {ok:false,error:error.message}}});
  socket?.register('desperate-prayer:close',async function(payload){try{const a=await fromUuid(payload.actorUuid);checkActor(a,game.users.get(this.socketdata.userId));await closeWindow(a,payload.nonce);return {ok:true}}catch(error){return {ok:false,error:error.message}}});
  on('updateCombat',c=>{if(isActiveGM(game))for(const a of values(game.actors)){const d=data(a);if(d.window?.state==='open'&&d.window.combatId===c.id&&!current(a,d.window))closeWindow(a,d.window.nonce).catch(onError);if(d.credit?.state==='available'&&d.credit.combatId===c.id&&!current(a,d.credit))lock(a,()=>expire(a)).catch(onError);}});
  on('deleteCombat',c=>{if(isActiveGM(game))for(const a of values(game.actors))if(data(a).credit?.combatId===c.id)lock(a,()=>expire(a)).catch(onError);});
  return()=>{for(const[k,id]of hooks)Hooks.off(k,id);for(const p of paths)libWrapper?.unregister(ID,p);for(const[e,f]of listeners)e.removeEventListener('click',f,true);for(const restore of restores.reverse())restore();installed=false;};
 }
 return {resolveAction,requiresActualUse:item=>!!resolveAction(item),tracksFrequency:item=>!!resolveAction(item),beforeUse,beforeAction,interceptCast,interceptCheck,captureUsage,executeUsage,consumePolicy,isManagedActor:a=>managed(a)||data(a).credit?.remaining===1,onStartTurn,onEndTurn,register};
}
