import {SerialActions} from '../runtime.mjs';
import {metapowerKind,powerProfile,sourceUuid,buildChannelSnapshot} from './rules.mjs';
export const MODULE_ID='pf2e-third-party-automation';
const copy=value=>structuredClone(value);
const same=(a,b)=>a===b||!!a&&!!b&&typeof a==='object'&&typeof b==='object'&&Array.isArray(a)===Array.isArray(b)&&Object.keys(a).length===Object.keys(b).length&&Object.keys(a).every(key=>Object.hasOwn(b,key)&&same(a[key],b[key]));
export const turnIdentity=(game,actor)=>{
 const encounters=game.combats&&actor?Array.from(game.combats.values()).filter(c=>c.started&&Array.from(c.combatants?.values?.()??c.turns??[]).some(t=>t.actor?.uuid===actor.uuid)):null;
 if(encounters?.length>1)throw Error('角色属于多个已开始的遭遇；请先确定当前遭遇，再使用威能调整。');
 const c=encounters?encounters[0]:game.combat;return c?`${c.id}:${c.round}:${c.turn}:${c.combatant?.id??''}`:null};
export const ledgerState=actor=>copy(actor.flags?.[MODULE_ID]?.metapower??{version:1,sequence:0,armed:null,pending:null,receipts:{}});
export const chargedEffect=actor=>Array.from(actor?.items?.values?.()??actor?.items??[]).find(i=>['Compendium.battlezoo-eldamon-pf2e.conditions.Item.Bi2aHykg6CZrQCnR','Compendium.battlezoo-eldamon-pf2e.conditions.Bi2aHykg6CZrQCnR'].includes(sourceUuid(i))&&i.system.badge?.value>0);

/** Prepared roll options are evaluated by PF2e; raw ChoiceSet selections are not
 * evidence that a suppressed or inactive preparation slot is available. */
export function validatePowerAdmission(item,{kind,selection={}}={}){
 const profile=powerProfile(item);if(!profile)return null;
 const options=item.actor.getRollOptions?.(['all'])??[];
 if(!options.some(o=>/^active-power-(one|two|three|four|reactive(?:-two)?|refresh(?:-two)?):/.test(o)&&o.endsWith(`:${profile.id}`)))throw Error('此威能当前未准备。');
 if(item.system.frequency&&!(item.system.frequency.value>0))throw Error('此威能的原生使用次数已耗尽。');
 if(profile.reaction&&selection.triggerConfirmed!==true)throw Error('使用此威能前须声明实际发生的反应触发。');
 if(profile.id==='reactive-chain'&&(!Number.isFinite(selection.triggerDamage)||selection.triggerDamage<=0||selection.eligibleTargetConfirmed!==true))throw Error('反应电链需要真实触发的电击伤害及符合条件的目标。');
 if(kind==='siphoning'&&selection.discharge&&profile.id==='reactive-chain')throw Error('虹吸会移除此放电分支的唯一收益；请选择普通分支。');
 return profile;
}

/** Short GM mutation queue. No native invocation or remote dialog runs under it.
 * A durable lease prevents another client from overtaking an unfinished use;
 * ambiguous native completion is archived, never refunded or automatically retried. */
export function createMetapowerLedger({game,fromUuid,queue=new SerialActions(),validateSelection=async()=>{}}){
 const mutate=(payload,user,fn,{render=true}={})=>queue.run(payload.actorUuid,async()=>{
  const gm=game.user;
  if(!gm?.id||gm!==game.users.activeGM||game.users.get(gm.id)!==gm)throw Error('只有主GM可以修改威能调整状态。');
  const actor=await fromUuid(payload.actorUuid);
  if(!actor||!user||!actor.testUserPermission(user,'OWNER'))throw Error('需要拥有此角色的权限。');
  const state=ledgerState(actor);
  let saved=copy(state);const guards=[];
  const assertLive=()=>{
   if(game.user!==gm||game.users.activeGM!==gm||game.users.get(gm.id)!==gm)throw Error('受理期间主GM已改变；请核对原操作记录。');
   const token=actor.token,current=actor.isToken?token&&game.scenes?.get(token.parent?.id)?.tokens?.get(token.id)===token&&token.actor===actor:game.actors?.get(actor.id)===actor;
   if(actor.uuid!==payload.actorUuid||!current||game.users.get(user.id)!==user||!actor.testUserPermission(user,'OWNER'))throw Error('原始角色来源或拥有权限已改变；请核对原操作记录。');
   if(!same(actor.flags?.[MODULE_ID]?.metapower??{version:1,sequence:0,armed:null,pending:null,receipts:{}},saved))throw Error('原始待处理操作或付款记录已改变；请核对原卡。');
   for(const guard of guards)guard();
  };
  const saveState=async()=>{assertLive();saved=copy(state);await actor.update({[`flags.${MODULE_ID}.metapower`]:state},{render:false});assertLive();};
  assertLive();
  if(state.armed?.turn!==turnIdentity(game,actor))state.armed=null;
  const result=await fn(actor,state,{assertLive,saveState,guard:check=>guards.push(check)});
  // Keep source/channel, uncertain and live proofs indefinitely. Ordinary
  // completed actions have no downstream card consumer; their client high-water
  // marks reject replay after the bounded detail archive has been pruned.
  const ordinary=Object.values(state.receipts).filter(r=>r.clientId&&!r.kind&&!r.powerId&&!r.snapshot&&r.nonce!==state.pending&&(!r.delivery||r.delivery.status==='done')&&['committed','cancelled'].includes(r.status)).sort((a,b)=>b.sequence-a.sequence);
  const pruned=ordinary.slice(64);for(const r of pruned)delete state.receipts[r.nonce];
  assertLive();
  // Foundry merges nested flags: omission alone cannot remove an old receipt.
  const update=copy(state);for(const r of pruned)update.receipts[`-=${r.nonce}`]=null;
  saved=copy(state);await actor.update({[`flags.${MODULE_ID}.metapower`]:update},{render});assertLive();return copy(result);
 });
 const bound=(state,payload,user)=>{
  const r=state.receipts[payload.nonce];if(!r||r.userId!==user.id)throw Error('操作绑定无效。');return r;
 };
 return {
  begin:(payload,user)=>mutate(payload,user,async(actor,state,live)=>{
   if(typeof payload.nonce!=='string'||!payload.nonce||payload.nonce.length>100)throw Error('操作标识无效。');
   const old=state.receipts[payload.nonce];
   if(old){if(old.userId!==user.id||old.itemUuid!==(payload.itemUuid??null))throw Error('操作来源绑定不匹配。');if(payload.startNative===true)throw Error('此操作已执行；不会重复执行原生操作。');return old;}
   let clientKey;
   if(payload.clientId){
    if(typeof payload.clientId!=='string'||payload.clientId.length>100||!Number.isSafeInteger(payload.clientSequence)||payload.clientSequence<1)throw Error('客户端操作序号无效。');
    clientKey=`${user.id}:${payload.clientId}`;state.clients??={};
    if(payload.clientSequence<=(state.clients[clientKey]??0))throw Error('拒绝重放已归档操作或乱序的客户端操作。');
   }
   if(state.pending)throw Error('另一项原生动作正在处理；请先完成它，再使用下一项动作。');
   if(Object.values(state.receipts).some(r=>r.delivery&&r.delivery.status!=='done'))throw Error('已提交动作的后续结算等待GM恢复；完成后才能使用下一项动作。');
   const item=payload.itemUuid?await fromUuid(payload.itemUuid):null;
   live.assertLive();
   if(payload.itemUuid&&(!item||item.actor!==actor||actor.items.get(item.id)!==item))throw Error('需要此角色拥有的原始嵌入条目。');
   if(item){const source=sourceUuid(item);live.guard(()=>{if(item.actor!==actor||actor.items.get(item.id)!==item||item.uuid!==payload.itemUuid||sourceUuid(item)!==source)throw Error('原始嵌入条目的来源已改变。');});}
   const kind=metapowerKind(item),profile=validatePowerAdmission(item,{kind:state.armed?.kind,selection:payload.selection});
   if(profile)await validateSelection({actor,item,selection:payload.selection??{},kind:state.armed?.kind??'normal',user});
   live.assertLive();validatePowerAdmission(item,{kind:state.armed?.kind,selection:payload.selection});
   const built=profile?buildChannelSnapshot({kind:state.armed?.kind??'normal',item,selection:payload.selection,policy:{dischargeNonDamage:'remove',dischargeArea:'retain',dischargeRange:'retain',dischargeSaveDowngrade:'retain',highVoltage:'convert'}}):null;
   const snapshot=built?{...built,...(profile.id==='reactive-chain'?{triggerDamage:payload.selection.triggerDamage}:{})}:null;
   const charge=snapshot?.dischargeCost?chargedEffect(actor):null;
   if(snapshot?.dischargeCost&&!charge)throw Error('所选放电分支需要蓄电。');
   const r={nonce:payload.nonce,sequence:++state.sequence,actorUuid:actor.uuid,itemUuid:item?.uuid??null,sourceUuid:sourceUuid(item),userId:user.id,turn:turnIdentity(game,actor),activationNonce:state.armed?.nonce??null,kind,snapshot,selection:copy(payload.selection??{}),status:'reserved',messageUuid:null};
   r.entry=payload.entry??'item';
   r.powerId=profile?.id??null;
   if(clientKey){r.clientId=payload.clientId;r.clientSequence=payload.clientSequence;state.clients[clientKey]=payload.clientSequence;}
   if(charge)r.charge={itemUuid:charge.uuid,before:charge.system.badge.value,after:charge.system.badge.value-1};
   // Persist admission and native-start intent together. A lost response keeps
   // the durable lease; the same nonce can never authorize native execution twice.
   if(payload.startNative===true)r.status='started';
   state.pending=r.nonce;state.receipts[r.nonce]=r;return payload.startNative===true?{...r,nativeStartAuthorized:true}:r;
  }),
  start:(payload,user)=>mutate(payload,user,(actor,state)=>{const r=bound(state,payload,user);if(r.status!=='reserved')throw Error('此操作已开始或已完成。');if(r.turn!==turnIdentity(game,actor))throw Error('原生执行前回合已改变；请在当前回合重新使用此动作。');r.status='started';return r},{render:false}),
  finish:(payload,user)=>mutate(payload,user,async(actor,state,live)=>{
   const r=bound(state,payload,user);
   if(!['reserved','started'].includes(r.status)){
    if(r.status!==payload.status||r.messageUuid!==(payload.messageUuid??null))throw Error('原始聊天卡绑定与已完成操作不匹配。');return r;
   }
   if(!['committed','cancelled','uncertain'].includes(payload.status))throw Error('原生操作完成状态无效。');
   if(payload.status==='cancelled'&&r.status==='started'&&!(r.entry==='native-check'&&payload.confirmation==='native-check-no-result'))throw Error('已开始的原生动作需要可核验的取消凭据；不会自动退款。');
   if(payload.messageUuid){
    const m=await fromUuid(payload.messageUuid);live.assertLive();
    live.guard(()=>{
     const proof=m?.flags?.[MODULE_ID]?.metapowerUse,originMatches=m?.flags?.pf2e?.origin?.uuid===r.itemUuid||m?.flags?.pf2e?.context?.type==='self-effect'&&actor.items.get(m.flags.pf2e.context.item)?.uuid===r.itemUuid;
     if(!m||m.uuid!==payload.messageUuid||game.messages?.get(m.id)!==m||proof?.nonce!==r.nonce||proof.actorUuid!==actor.uuid||proof.itemUuid!==r.itemUuid||m.speaker?.actor!==actor.id||(m.author?.id??m.user?.id??m.user)!==user.id||!originMatches)throw Error('原始原生聊天卡绑定无效。');
    });live.assertLive();
   }else if(payload.status==='committed'&&r.itemUuid)throw Error('提交条目使用需要原始原生聊天卡。');
   if(payload.status==='committed'&&r.itemUuid){const item=actor.items.get(r.itemUuid.split('.').at(-1));live.guard(()=>{if(!item||item.actor!==actor||actor.items.get(item.id)!==item||item.uuid!==r.itemUuid||sourceUuid(item)!==r.sourceUuid)throw Error('原始威能条目的来源已改变。');});live.assertLive();}
   if(payload.status==='committed'&&r.charge){
    const charge=await fromUuid(r.charge.itemUuid);live.assertLive();
    if(charge?.flags?.[MODULE_ID]?.payment?.nonce!==r.nonce){
     if(r.paymentStarted)throw Error('放电付款结果不确定；须由GM核对原卡，不得再次付款。');
     if(!charge||charge.system.badge.value!==r.charge.before)throw Error('原生使用期间蓄电已改变；请先核对原卡，再继续。');
     const source=sourceUuid(charge),payment=copy(charge.flags?.[MODULE_ID]?.payment??null);let paying=true;
     live.guard(()=>{
      const current=actor.items.get(charge.id),deleted=!paying&&r.charge.after===0&&!current;
      if(!deleted&&(charge.actor!==actor||current!==charge||charge.uuid!==r.charge.itemUuid||sourceUuid(charge)!==source||!['Compendium.battlezoo-eldamon-pf2e.conditions.Item.Bi2aHykg6CZrQCnR','Compendium.battlezoo-eldamon-pf2e.conditions.Bi2aHykg6CZrQCnR'].includes(source)||charge.system.badge.value!==(paying?r.charge.before:r.charge.after)||!same(charge.flags?.[MODULE_ID]?.payment??null,paying?payment:{nonce:r.nonce,after:r.charge.after})))throw Error('原始蓄电来源、计数或付款凭据已改变。');
     });
     // Native PF2e deletes a counter effect at zero, including its GrantItem
     // children. Persist intent before that deletion so a lost response never
     // permits a second payment or silently assumes an unrelated deletion paid.
     r.paymentStarted=true;r.messageUuid=payload.messageUuid;
     await live.saveState();
     // Counter and payment proof share one embedded document update. A retry or
     // GM handover sees the proof and cannot charge a second time.
     live.assertLive();await charge.update({'system.badge.value':r.charge.after,[`flags.${MODULE_ID}.payment`]:{nonce:r.nonce,after:r.charge.after}});paying=false;live.assertLive();
    }else{
     const source=sourceUuid(charge);live.guard(()=>{if(charge.actor!==actor||actor.items.get(charge.id)!==charge||charge.uuid!==r.charge.itemUuid||sourceUuid(charge)!==source||!['Compendium.battlezoo-eldamon-pf2e.conditions.Item.Bi2aHykg6CZrQCnR','Compendium.battlezoo-eldamon-pf2e.conditions.Bi2aHykg6CZrQCnR'].includes(source)||charge.flags?.[MODULE_ID]?.payment?.nonce!==r.nonce||charge.flags[MODULE_ID].payment.after!==r.charge.after||charge.system.badge.value!==r.charge.after)throw Error('原始放电付款凭据无法确认；须由GM核对原卡。');});live.assertLive();
    }
    r.paymentPaid=true;
   }
   r.status=payload.status;r.messageUuid=payload.messageUuid??r.messageUuid??null;state.pending=null;
   if(r.status==='committed'&&r.messageUuid)r.delivery={status:'pending',attempts:0};
   if(r.status!=='cancelled'){
    if(state.armed?.nonce===r.activationNonce)state.armed=null;
    if(r.status==='committed'&&r.kind&&r.turn===turnIdentity(game,actor))state.armed={nonce:r.nonce,kind:r.kind,itemUuid:r.itemUuid,sourceUuid:r.sourceUuid,sequence:r.sequence,turn:r.turn,messageUuid:r.messageUuid};
   }
   return r;
  },{render:payload.status!=='committed'||!payload.messageUuid}),
  clear:(payload,user)=>mutate(payload,user,(_actor,state)=>{if(state.pending)throw Error('原生动作正在处理，暂时不能清除待用的威能调整。');if(!payload.activationNonce||state.armed?.nonce===payload.activationNonce)state.armed=null;return state.armed}),
  reconcile:(payload,user)=>mutate(payload,user,(_actor,state)=>{
   if(user.id!==game.users.activeGM?.id)throw Error('只有主GM可以核对已中断的原生操作。');
   const r=state.receipts[payload.nonce];if(!r||state.pending!==r.nonce||!['reserved','started'].includes(r.status)||payload.confirmation!=='archive-uncertain')throw Error('需要原始待处理操作，以及明确保留结果不确定的处理选择。');
   r.status='uncertain';r.reconciledBy=user.id;state.pending=null;if(state.armed?.nonce===r.activationNonce)state.armed=null;return r;
  }),
  delivery:(payload,user)=>mutate(payload,user,(_actor,state)=>{
   if(user.id!==game.users.activeGM?.id)throw Error('只有主GM可以执行已提交原生动作的后续结算。');
   const r=state.receipts[payload.nonce];if(r?.status!=='committed'||!r.messageUuid||!r.delivery)throw Error('无法取得原始已提交威能的后续结算记录。');
   if(r.delivery.status==='done')return r;
   if(!['started','pending','done'].includes(payload.status))throw Error('威能后续结算状态无效。');
   r.delivery={...r.delivery,status:payload.status,...(payload.status==='started'?{attempts:r.delivery.attempts+1}:{}),...(payload.confirmation==='gm-manual-effects-settled'?{manuallySettled:true,resolvedBy:user.id}:{}),error:payload.status==='pending'?String(payload.error??'Interrupted native follow-up').slice(0,500):null};return r;
  },{render:payload.status!=='started'}),
  expire:(payload,user)=>mutate(payload,user,(_actor,state)=>state.armed),
 };
}
