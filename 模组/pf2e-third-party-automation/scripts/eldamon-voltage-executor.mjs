import {createVoltageLedger,currentVoltageToken,HIGH_VOLTAGE_SOURCE,ELEMENTAL_POWERS_SOURCE,VOLTAGE_MODULE_ID as ID,VOLTAGE_APPLY_PREFIX,voltageRollOption,voltageReceiptMatches} from './eldamon-voltage.mjs';
import {sourceUuid,siphonMultiplier} from './metapower/rules.mjs';
import {convertSiphonRoll} from './metapower/damage.mjs';
import {showNativeChoice,publicTargetName} from './native-context.mjs';
import {withDamageMessageTarget} from './damage-message-targets.mjs';
import {createVoltageContinuationIndex,canReadVoltageMessage} from './eldamon-voltage-continuations.mjs';
import {manualDamageRoll as rollNativeDamageManually,manualDamagePrivacy as nativeDamagePrivacy} from './manual-native-roll.mjs';
import {beforeNativeRoll} from './native-owner-operations.mjs';

const values=c=>Array.from(c?.values?.()??c??[]),prefix=`${ID}:voltage:`,outcomes={criticalSuccess:0,success:0.5,failure:1,criticalFailure:2};
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const random=()=>globalThis.foundry?.utils?.randomID?.(24)??globalThis.crypto.randomUUID();
const binding=a=>({actorUuid:a.actorUuid,nonce:a.nonce,messageUuid:a.messageUuid});

/** Native interactions belong to the clicking owner. The GM only claims and
 * settles durable phases; observing an attack never rolls or applies damage. */
export function createEldamonVoltageProvider({game,fromUuid,observe,onRefresh,getRollContext,DamageRoll=globalThis.CONFIG?.Dice?.rolls?.find(C=>C.name==='DamageRoll'),manualDamageRoll=rollNativeDamageManually,manualDamagePrivacy=nativeDamagePrivacy,convertRoll=convertSiphonRoll,selectChoice=showNativeChoice,onError=console.error}={}){
 const index=createVoltageContinuationIndex({game}),ledger=createVoltageLedger({game,fromUuid,onRefresh,getSaveCandidates:index.saves}),applications=new Map(),views=new Map(),continuing=new Set();let socket,registered=false,rollHooks=globalThis.Hooks;
 const activeActors={refresh:index.refresh,register:index.register,values:index.actors};
 const active=()=>game.user?.id===game.users.activeGM?.id;
 async function notifyCard(actorUuid,nonce){
  const actor=await fromUuid(actorUuid);activeActors.refresh(actor);const a=actor?.flags?.[ID]?.voltage?.activations?.[nonce];if(!a)return;
  const message=await fromUuid(a.messageUuid),phase=a.native?.phase??null;
  if(message?.flags?.[ID]?.voltageStatus!==a.status||message?.flags?.[ID]?.voltagePhase!==phase)await message?.update?.({[`flags.${ID}.voltageStatus`]:a.status,[`flags.${ID}.voltagePhase`]:phase});
  refreshViews(actorUuid);
 }
 async function coordinate(method,payload,user){
  if(method==='trigger')return trigger(payload,user);
  try{return await ledger[method](payload,user)}
  finally{if(payload.nonce)await notifyCard(payload.actorUuid,payload.nonce).catch(onError)}
 }
 async function request(method,payload){
  if(active())return coordinate(method,payload,game.user);
  if(!socket||!game.users.activeGM)throw Error('高电压需要套接字模块及在线的当前主持人。');
  const r=await socket.executeAsUser(`voltage:${method}`,game.users.activeGM.id,payload);if(!r?.ok)throw Error(r?.error??'高电压结算器没有响应，请核对原活动后再继续。');return r.value;
 }
 async function trigger(payload,user){
  if(payload.kind!=='attack'&&payload.confirmed!==true)throw Error('请声明本次实际触碰或未知金属武器命中。');
  const result=await ledger.claim({...payload,nativeInteraction:true},user);await notifyCard(payload.actorUuid,payload.nonce).catch(onError);return result;
 }
 async function abort(payload,status){await request('abortNative',{...payload,status}).catch(onError)}
 async function nativeScope(a,payload,phase,{user,gm}){
  const [actor,item,origin,target,message,save]=await Promise.all([a.actorUuid,a.itemUuid,a.originUuid,a.trigger.targetUuid,a.messageUuid,...phase==='rolling-damage'?[a.native.save.messageUuid]:[]].map(fromUuid));
  const targetActor=target?.actor,roller=phase==='rolling-save'?targetActor:actor,kind=phase==='rolling-save'?'save':'damage';
  const input=current=>JSON.stringify(current&&{actorUuid:current.actorUuid,nonce:current.nonce,messageUuid:current.messageUuid,itemUuid:current.itemUuid,originUuid:current.originUuid,userId:current.userId,status:current.status,level:current.level,dc:current.dc,traits:current.traits,snapshot:current.snapshot,trigger:current.trigger,native:{version:current.native?.version,phase:current.native?.phase,operation:current.native?.[kind],...kind==='damage'?{save:current.native?.save}:{}}});
  const expected=input(a),saveEvidence=save&&JSON.stringify({pf:save.flags?.pf2e,rolls:save.rolls?.map(r=>r.toJSON?.()??{total:r.total,evaluated:r._evaluated})});
  const assertLive=()=>{
   if(game.user!==user||game.users.get(user?.id)!==user||!user.active||game.users.activeGM!==gm||game.users.get(gm?.id)!==gm||!gm.active||!gm.isGM)throw Error('高电压原操作者或主GM已交接；结果未确认，不会自动重投。');
   const current=actor?.flags?.[ID]?.voltage?.activations?.[a.nonce],receipt=actor?.flags?.[ID]?.metapower?.receipts?.[a.nonce],proof=message?.flags?.[ID]?.metapowerUse;
   if(!actor||!item||item.actor!==actor||actor.items.get(item.id)!==item||sourceUuid(item)!==HIGH_VOLTAGE_SOURCE||[actor,targetActor].some(doc=>!doc||!doc.isToken&&game.actors.get(doc.id)!==doc)||!currentVoltageToken(origin)||origin.actor!==actor||game.scenes.get(origin.parent?.id)!==origin.parent||!currentVoltageToken(target)||target.parent!==origin.parent||target.actor!==targetActor||targetActor.uuid!==a.trigger.targetActorUuid)throw Error('高电压原角色、能力或绑定Token已改变；结果未确认，不会自动重投。');
   if(roller?.testUserPermission(user,'OWNER')!==true||game.messages.get(message?.id)!==message||message.uuid!==a.messageUuid||proof?.nonce!==a.nonce||proof.actorUuid!==actor.uuid||proof.itemUuid!==item.uuid||message.flags?.pf2e?.origin?.uuid!==item.uuid||message.flags.pf2e.origin.actor!==actor.uuid||message.speaker?.actor!==actor.id||`Scene.${message.speaker?.scene}.Token.${message.speaker?.token}`!==origin.uuid||(message.author?.id??message.user?.id??message.user)!==a.userId||receipt?.status!=='committed'||receipt.nonce!==a.nonce||receipt.userId!==a.userId||receipt.actorUuid!==actor.uuid||receipt.itemUuid!==item.uuid||receipt.messageUuid!==message.uuid||receipt.sourceUuid!==HIGH_VOLTAGE_SOURCE)throw Error('高电压原活动、费用或拥有者权限已改变；结果未确认，不会自动重投。');
   if(current?.status!=='claimed'||current.native?.phase!==phase||current.native?.[kind]?.id!==payload.operationId||current.native[kind].userId!==user.id||input(current)!==expected)throw Error('高电压本次操作认领已改变；结果未确认，不会自动重投。');
   if(kind==='damage'&&(!save||game.messages.get(save.id)!==save||save.uuid!==a.native.save.messageUuid||JSON.stringify({pf:save.flags?.pf2e,rolls:save.rolls?.map(r=>r.toJSON?.()??{total:r.total,evaluated:r._evaluated})})!==saveEvidence))throw Error('高电压绑定原生豁免已改变；结果未确认，不会自动重投。');
  };
  assertLive();return {actor,item,origin,target,assertLive};
 }
 async function rollSave(activation){
  const started={user:game.user,gm:game.users.activeGM},payload={...binding(activation),operationId:random()},a=await request('beginSave',payload);
  try{
   const {actor,item,target,assertLive}=await nativeScope(a,payload,'rolling-save',started),statistic=target.actor.getStatistic('reflex'),roller=statistic?.check??statistic;
   if(typeof roller?.roll!=='function')throw Error('原生反射豁免统计不可用。');
   let save;await beforeNativeRoll({Hooks:rollHooks,marker:voltageRollOption(a),showDialog:true,commit:async()=>{},assertLive,native:()=>roller.roll({token:target,origin:actor,item,action:'high-voltage',dc:{slug:'eldamon',value:a.dc},traits:a.traits,skipDialog:false,event:null,
    extraRollOptions:[voltageRollOption(a),'action:high-voltage','damaging-effect',...a.traits.map(t=>'item:trait:'+t)],createMessage:true,
    callback:(_roll,_outcome,message)=>{save=message}})});assertLive();
   if(!save){await abort(payload,'cancelled');return null}
   return await request('finishSave',{...payload,saveUuid:save.uuid});
  }catch(error){await abort(payload,'uncertain');throw error}
 }
 async function rollDamage(activation){
  const started={user:game.user,gm:game.users.activeGM},payload={...binding(activation),operationId:random()},a=await request('beginRoll',payload);
  try{
   const {item,assertLive}=await nativeScope(a,payload,'rolling-damage',started);if(typeof DamageRoll!=='function')throw Error('原生伤害掷骰不可用。');
   let damage=await manualDamageRoll({game,roll:new DamageRoll(`${a.level+1}d6[electricity]`,{},{rollerId:game.user.id}),assertLive});assertLive();
   if(!damage){await abort(payload,'cancelled');return null}
   const privacy=manualDamagePrivacy(damage),messageOptions=privacy?{messageMode:privacy.messageMode}:{};
   if(a.snapshot?.siphon?.applies)damage=convertRoll(damage,{DamageRoll,rejectMixedPartitions:true});
   damage=damage.alter(outcomes[a.native.save.outcome],0);
   const proof={...binding(a),targetUuid:a.trigger.targetUuid,operationId:payload.operationId};damage.options[ID]={...damage.options[ID],voltageDamage:proof};
   const origin=item.getOriginData(),options=[voltageRollOption(a),'action:high-voltage',...a.traits.map(t=>'item:trait:'+t),...origin.rollOptions??[]];
   // This is already the kept basic-save amount. No Toolbelt saveVariants are
   // copied from the action card, so its target row offers ordinary full apply.
   assertLive();let publicationError;
   const publicationHook=rollHooks.on('preCreateChatMessage',message=>{
    const tagged=message.flags?.[ID]?.voltageDamage;
    if(tagged?.actorUuid!==proof.actorUuid||tagged.nonce!==proof.nonce||tagged.messageUuid!==proof.messageUuid||tagged.operationId!==proof.operationId)return;
    try{assertLive();}catch(error){publicationError=error;return false;}
   });
   try{
    const message=await damage.toMessage(withDamageMessageTarget({...(privacy?{blind:privacy.blind,whisper:privacy.whisper}:{}),speaker:a.speaker,
     flavor:`${esc(item.name)} · 反射基础豁免已计入；对绑定目标按全额应用${a.snapshot?.siphon?.applies?'（虹吸按目标特征调整）':''}`,
     flags:{pf2e:{origin,context:{type:'damage-roll',sourceType:'save',outcome:a.native.save.outcome,target:{actor:a.trigger.targetActorUuid,token:a.trigger.targetUuid},options,...(privacy?{messageMode:privacy.messageMode}:{})}},[ID]:{voltageDamage:proof}}},a.trigger.targetUuid),messageOptions);
    if(publicationError)throw publicationError;
    if(!message?.uuid)throw Error('高电压原生伤害卡未保存；结果未确认，不会自动重投。');
    return await request('finishRoll',{...payload,damageUuid:message.uuid});
   }finally{rollHooks.off('preCreateChatMessage',publicationHook);}
  }catch(error){await abort(payload,'uncertain');throw error}
 }
 async function beforeDamage(actor,params){
  const options=values(params.rollOptions),tracked=getRollContext?.(params.damage),trackedMessage=game.messages.get(tracked?.messageId);
  const tagged=options.filter(o=>typeof o==='string'&&o.startsWith(prefix)),privateProof=params.damage?.options?.[ID]?.voltageDamage;
  if(!tagged.length&&!privateProof&&!trackedMessage?.flags?.[ID]?.voltageDamage)return null;
  const sourceOptions=options.filter(o=>typeof o==='string'&&o.startsWith(`${ID}:source:`)),fallback=sourceOptions.length===1?sourceOptions[0].slice(`${ID}:source:`.length).split(':'):[];
  const source=getRollContext?tracked:fallback.length===2?{messageId:fallback[0],rollIndex:Number(fallback[1])}:null,message=game.messages.get(source?.messageId),proof=message?.flags?.[ID]?.voltageDamage;
  if(typeof DamageRoll!=='function'||!(params.damage instanceof DamageRoll)||params.damage._evaluated!==true||!Number.isFinite(params.damage.total)||params.damage.total<0||!source||source.rollIndex!==0||!proof||tagged.length!==1)throw Error('高电压需要已授权并记录的原生伤害卡；已消耗或其他来源的骰子不能应用。');
  const target=params.token?.document??params.token;
  if(!currentVoltageToken(target)||target.uuid!==proof.targetUuid||target.actor.uuid!==actor.uuid||params.item?.uuid!==message.flags?.pf2e?.origin?.uuid||!target.actor.testUserPermission(game.user,'OWNER')||params.final||params.skipIWR)throw Error('高电压伤害绑定原目标及其拥有者，必须进入原生抗免结算。');
  const actorSource=await fromUuid(proof.actorUuid),a=actorSource?.flags?.[ID]?.voltage?.activations?.[proof.nonce];
  if(!a||tagged[0]!==voltageRollOption(a)||a.native?.damage?.messageUuid!==message.uuid)throw Error('高电压原生伤害来源绑定无效。');
  const payload={...binding(a),operationId:random(),damageUuid:message.uuid,targetUuid:target.uuid,targetActorUuid:actor.uuid,rollIndex:0};
  const claimed=await request('beginDamage',payload),multiplier=siphonMultiplier(claimed.snapshot,actor.traits??actor.system?.traits?.value??[]);
  applications.set(payload.operationId,{activation:claimed,payload,message:null});
  return {params:{...params,damage:multiplier===1?params.damage:params.damage.alter(multiplier,0),rollOptions:new Set([...options,VOLTAGE_APPLY_PREFIX+payload.operationId])},receipt:payload};
 }
 function capture(message,_options,creator){
  if(creator!==game.user.id||message.flags?.pf2e?.context?.type!=='damage-taken')return;
  const tags=(message.flags.pf2e.context.options??[]).filter(o=>typeof o==='string'&&o.startsWith(VOLTAGE_APPLY_PREFIX));if(tags.length!==1)return;
  const scope=applications.get(tags[0].slice(VOLTAGE_APPLY_PREFIX.length));if(scope&&!scope.message&&voltageReceiptMatches(message,scope.activation))scope.message=message;
 }
 async function afterDamage(receipt,{applied,uncertain}){
  const scope=applications.get(receipt?.operationId);if(!scope)return;
  try{
   if((applied||uncertain)&&scope.message)await request('finishDamage',{...scope.payload,receiptUuid:scope.message.uuid});
   else await abort(scope.payload,applied||uncertain?'uncertain':'cancelled');
  }finally{applications.delete(receipt.operationId)}
 }
 async function onCommittedChannel({receipt,message,user=game.users.get(receipt.userId)}){
  const payload={actorUuid:receipt.actorUuid,nonce:receipt.nonce,messageUuid:message.uuid},deliver=method=>active()?coordinate(method,payload,user):request(method,payload);
  if(receipt.sourceUuid===HIGH_VOLTAGE_SOURCE)return deliver('channel');
  if(receipt.sourceUuid===ELEMENTAL_POWERS_SOURCE&&message.flags?.[ID]?.voltageRefreshActivity)return deliver('refreshActivity');return null;
 }
 async function useRefresh(actor,event){
  const item=values(actor.items).find(i=>sourceUuid(i)===ELEMENTAL_POWERS_SOURCE);
  if(!item||!actor.testUserPermission(game.user,'OWNER')||typeof observe!=='function')throw Error('需要本角色拥有的元素威能及原生动作观察器。');
  return observe({actor,item},async()=>{
   const draft=await item.toMessage(event,{create:false}),data=draft.toObject();
    data.content='<div class="pf2e chat-card"><header><h3>刷新威能 · 2动作</h3></header><p>恢复本角色已准备的启动与反应威能使用次数。</p></div>';
   data.flags??={};data.flags[ID]={...data.flags[ID],voltageRefreshActivity:{actions:2}};return globalThis.CONFIG.ChatMessage.documentClass.create(data);
  });
 }
 async function confirmTouch(a,selectedUuid){
  const origin=await fromUuid(a.originUuid),selected=values(game.user.targets).map(t=>t.document??t);
  let targetUuid=selectedUuid??(selected.length===1?selected[0].uuid:null);
  if(!targetUuid){const choices=values(origin?.parent?.tokens).filter(t=>currentVoltageToken(t)&&t.actor.uuid!==a.actorUuid).map((t,i)=>({value:t.uuid,label:publicTargetName(t,{game})+`（${i+1}）`}));
   if(!choices.length)throw Error('当前场景没有可声明触碰的其他生物。');
   targetUuid=await selectChoice({title:'选择实际触碰高电压使用者的生物',choices});}
  if(!targetUuid)return null;
  return request('trigger',{...binding(a),targetUuid,kind:'touch',confirmed:true});
 }
 const nativeHit=(message,a)=>{
  const c=message?.flags?.pf2e?.context,item=message?.item;
  return game.messages.get(message?.id)===message&&message?.isCheckRoll&&message.rolls?.[0]?._evaluated&&c?.type==='attack-roll'&&
   ['success','criticalSuccess'].includes(c.outcome)&&c.target?.token===a.originUuid&&c.target?.actor===a.actorUuid&&message.timestamp>=a.channelTimestamp&&
   (item?.isMelee===true||c.options?.includes('item:melee')||c.options?.includes('melee'))&&canReadVoltageMessage(message,game.user);
 };
 async function confirmAttack(a,metal=false,selectedUuid){
  let attackUuid=selectedUuid;
  if(!attackUuid){const candidates=index.attacks(a).filter(m=>nativeHit(m,a));if(!candidates.length)throw Error('没有当前可读取、绑定本角色的真实原生近战命中。');
   attackUuid=await selectChoice({title:metal?'选择本次金属武器命中':'选择本次原生近战命中',choices:candidates.map((m,i)=>({value:m.uuid,label:`原生近战命中（${i+1}）`}))});}
  if(!attackUuid)return null;
  const message=await fromUuid(attackUuid);if(!nativeHit(message,a))throw Error('本次命中记录不可读取，或不是本活动绑定的原生近战命中。');
  return request('trigger',{...binding(a),attackUuid,kind:metal?'metal-hit':'attack',...(metal?{confirmed:true}:{})});
 }
 const tokenFor=uuid=>{const parts=uuid?.split('.');return parts?game.scenes?.get(parts[1])?.tokens?.get(parts[3]):null};
 const targetOf=a=>tokenFor(a.trigger?.targetUuid);
 const owner=actor=>actor?.testUserPermission?.(game.user,'OWNER')===true;
 const statusText=a=>a.expiryReason==='no-encounter'?`当前没有已开始的遭遇，反击窗口请由主持人裁定；${a.refreshSuppressed?'本次未刷新威能':'已即时刷新威能'}。`:
  a.status==='armed'?'已就绪；实际触发后选择一次反击。':a.status==='claimed'?({'awaiting-save':'等待绑定目标的原生反射豁免。','awaiting-damage':'豁免已记录；可先英雄点重掷，再由施法者点击投掷伤害。','awaiting-application':'伤害已发布；请在原生伤害卡对绑定目标按全额应用。','rolling-save':'原生豁免处理中，请勿重复操作。','rolling-damage':'伤害投掷处理中，请勿重复操作。',applying:'伤害应用处理中，请勿重复操作。'}[a.native?.phase]??'等待主持人核对原始记录，请勿重新操作。'):
  a.status==='done'?'已按原生伤害回执结算。':a.status==='uncertain'?'结果不确定；等待主持人核对原记录，请勿重复操作。':a.status==='cancelled'?'已取消，本次触发已消耗。':'已到期。';
 function describe(entry,message){
  const {actor,activation:a}=entry,target=targetOf(a),sourceOwner=owner(actor),targetOwner=!!target&&target.actor?.uuid===a.trigger?.targetActorUuid&&owner(target.actor);
  if(!sourceOwner&&!targetOwner)return null;
  const item=actor.items?.get(a.itemUuid?.split('.').at(-1)),origin=tokenFor(a.originUuid),original=game.messages.get(a.messageUuid?.split('.').at(-1)),changed=original?.uuid!==a.messageUuid||!currentVoltageToken(origin)||origin.actor?.uuid!==a.actorUuid||sourceUuid(item)!==HIGH_VOLTAGE_SOURCE||!!a.trigger&&(!currentVoltageToken(target)||target.actor?.uuid!==a.trigger.targetActorUuid);
  const actions=[];
  if(changed)return {...binding(a),status:a.status,phase:a.native?.phase??null,label:'高电压原来源或目标已改变，等待主持人核对原活动。',actions};
  if(a.status==='armed'&&sourceOwner){
   if(message&&nativeHit(message,a))actions.push({action:'hit',label:'高电压反击',attackUuid:message.uuid},{action:'metal-hit',label:'金属武器命中反击',attackUuid:message.uuid});
   else if(!message||message.uuid===a.messageUuid){if(index.attacks(a).some(m=>nativeHit(m,a)))actions.push({action:'hit',label:'选择原生命中反击'},{action:'metal-hit',label:'选择金属武器命中反击'});actions.push({action:'touch',label:'声明实际触碰反击'});}
  }
  if(a.status==='claimed'&&a.native?.phase==='awaiting-save'&&targetOwner)actions.push({action:'save',label:'绑定目标：原生反射豁免'});
  if(a.status==='claimed'&&a.native?.phase==='awaiting-damage'&&sourceOwner)actions.push({action:'damage',label:'投掷原生伤害'});
  return {...binding(a),status:a.status,phase:a.native?.phase??null,label:`${sourceOwner?item?.name??'高电压':'高电压'}：${statusText(a)}`,actions};
 }
 function getContinuations(actorOrUuid){return index.forActor(typeof actorOrUuid==='string'?actorOrUuid:actorOrUuid?.uuid).map(e=>describe(e)).filter(Boolean)}
 async function continueActivity(input={}){
  if(!input.nonce){const choices=getContinuations(input.actorUuid).flatMap(entry=>entry.actions.map(action=>({...entry,activityLabel:entry.label,...action})));if(!choices.length)throw Error('当前角色没有可继续的高电压步骤。');
   const choice=await selectChoice({title:'继续本次高电压',choices:choices.map((c,i)=>({value:String(i),label:c.activityLabel+' · '+c.label}))});if(choice===null||choice===undefined)return null;input=choices[Number(choice)];if(!input)return null;}
  const actor=await fromUuid(input.actorUuid),a=actor?.flags?.[ID]?.voltage?.activations?.[input.nonce];
  if(!a||input.messageUuid&&input.messageUuid!==a.messageUuid)throw Error('原高电压活动已改变，不能继续此入口。');
  const available=describe({actor,activation:a});if(!available?.actions.some(c=>c.action===input.action))throw Error('本步骤已结束、处理中或结果不确定；请核对原活动，不要重新执行。');
  const key=a.actorUuid+':'+a.nonce;if(continuing.has(key))return null;continuing.add(key);
  try{
   if(input.action==='hit'||input.action==='metal-hit')return await confirmAttack(a,input.action==='metal-hit',input.attackUuid);
   if(input.action==='touch')return await confirmTouch(a,input.targetUuid);
   if(input.action==='save')return await rollSave(a);
   if(input.action==='damage')return await rollDamage(a);
  }finally{continuing.delete(key);refreshViews(a.actorUuid)}
 }
 function appendControls(root,entries,message){
  root.querySelector('[data-voltage-state]')?.remove();if(!root.ownerDocument?.createElement)return;
  const projected=entries.map(e=>describe(e,message)).filter(Boolean);if(!projected.length)return;
  const box=root.ownerDocument.createElement('div');box.dataset.voltageState='';
  for(const entry of projected){const row=root.ownerDocument.createElement('div');row.textContent=entry.label;
   for(const action of entry.actions){const button=root.ownerDocument.createElement('button');button.type='button';button.dataset.voltageAction=action.action;button.textContent=action.label;
    button.addEventListener('click',async event=>{event.preventDefault();event.stopPropagation();button.disabled=true;try{await continueActivity({...binding(entry),...action})}catch(error){onError(error)}finally{button.disabled=false}});row.append(button);}
   box.append(row);}
  (root.querySelector('.message-content')??root).append(box);
 }
 function renderCard(message,html){
  const root=html?.[0]??html;if(!root?.querySelectorAll)return;
  const proof=message.flags?.[ID]?.metapowerUse,actor=message.actor??game.actors.get(message.speaker?.actor),receipt=actor?.flags?.[ID]?.metapower?.receipts?.[proof?.nonce];
  const entries=index.forCard(message);
  if(receipt?.sourceUuid===HIGH_VOLTAGE_SOURCE&&receipt.messageUuid===message.uuid){
   for(const node of root.querySelectorAll('[data-damage-roll],[data-pf2-check]')){node.removeAttribute('data-damage-roll');node.removeAttribute('data-pf2-check');node.setAttribute('aria-disabled','true');node.style.pointerEvents='none';}
   const a=actor.flags?.[ID]?.voltage?.activations?.[receipt.nonce];if(a&&!entries.some(e=>e.activation.nonce===a.nonce&&e.actor.uuid===actor.uuid))entries.push({actor,activation:a});
  }
  if(entries.some(e=>['armed','claimed','uncertain'].includes(e.activation.status))||receipt?.sourceUuid===HIGH_VOLTAGE_SOURCE&&!actor?.flags?.[ID]?.voltage?.activations?.[receipt.nonce])views.set(root,{message,connected:views.get(root)?.connected||root.isConnected===true,sourceUuids:new Set(entries.map(e=>e.actor.uuid).concat(actor?.uuid??[]))});else views.delete(root);
  appendControls(root,canReadVoltageMessage(message,game.user)?entries:[],message);
 }
 function renderActorContinuations(actor,html,{app}={}){const root=html?.[0]??html;if(!root?.querySelector)return;views.set(root,{actor,app,connected:views.get(root)?.connected||root.isConnected===true});appendControls(root,index.forActor(actor.uuid))}
 function refreshViews(actorUuid){for(const [root,view]of [...views]){if(root.isConnected===true)view.connected=true;if(view.connected&&root.isConnected===false){views.delete(root);continue}if(actorUuid&&view.message&&!view.sourceUuids.has(actorUuid)||actorUuid&&view.actor&&view.actor.uuid!==actorUuid&&!index.forActor(view.actor.uuid).some(e=>e.actor.uuid===actorUuid))continue;if(view.message)renderCard(view.message,root);else renderActorContinuations(view.actor,root,{app:view.app})}}
 function closeSheet(app){for(const [root,view]of views)if(view.app===app||!view.app&&view.actor===(app.actor??app.document))views.delete(root)}
 function register({Hooks,socket:api}={}){
  if(registered)return;registered=true;socket=api;rollHooks=Hooks;activeActors.register(Hooks);
  for(const method of ['channel','refreshActivity','trigger','beginSave','finishSave','beginRoll','finishRoll','beginDamage','finishDamage','abortNative'])socket.register(`voltage:${method}`,async function(payload){try{return {ok:true,value:await coordinate(method,payload,game.users.get(this.socketdata.userId))}}catch(error){return {ok:false,error:error.message}}});
  const sweep=()=>{if(!active())return;return Promise.all(activeActors.values().map(actor=>{const nonce=actor.flags?.[ID]?.voltage?.activeNonce;return nonce?ledger.expire({actorUuid:actor.uuid},game.user).then(()=>notifyCard(actor.uuid,nonce)).catch(onError):null}))};
  Hooks.on('createChatMessage',capture);
  Hooks.on('updateActor',actor=>refreshViews(actor.uuid));
  Hooks.on('deleteChatMessage',message=>{for(const actorUuid of new Set(index.forCard(message).map(entry=>entry.actor.uuid)))refreshViews(actorUuid)});
  const refreshToken=token=>{for(const entry of index.forToken(token.uuid))refreshViews(entry.actor.uuid)};
  Hooks.on('deleteToken',refreshToken);Hooks.on('updateToken',(token,changes)=>{if(Object.hasOwn(changes,'actorId')||Object.hasOwn(changes,'actorLink'))refreshToken(token)});
  for(const hook of ['closeApplication','closeApplicationV2','closeActorSheetPF2e','closeCharacterSheetPF2e','closeActorSheetV2'])Hooks.on(hook,closeSheet);
  for(const hook of ['pf2e.startTurn','updateCombat','deleteCombat','deleteItem','deleteToken','userConnected','updateUser'])Hooks.on(hook,sweep);
  Hooks.on('renderChatMessageHTML',renderCard);
  for(const hook of ['renderActorSheetPF2e','renderCharacterSheetPF2e','renderActorSheetV2'])Hooks.on(hook,(app,html)=>{
   const root=html?.[0]??html,actor=app.actor;if(!root?.querySelector||!owner(actor))return;
   const host=root.querySelector('.tab.actions[data-tab="actions"],section[data-tab="actions"]')??root;renderActorContinuations(actor,host,{app});
   if(!values(actor.items).some(i=>sourceUuid(i)===ELEMENTAL_POWERS_SOURCE)||root.querySelector('[data-voltage-refresh]'))return;
   const button=root.ownerDocument.createElement('button');button.type='button';button.dataset.voltageRefresh='';button.textContent='刷新威能 · 2动作';button.addEventListener('click',event=>{event.preventDefault();event.stopPropagation();button.disabled=true;useRefresh(actor,event).catch(onError).finally(()=>{button.disabled=false})});host.append(button);
  });sweep();
 }
 const refreshOutsideEncounter=({actor,nonce})=>ledger.refreshOutsideEncounter({actorUuid:actor.uuid,nonce},game.user);
 return {register,onCommittedChannel,beforeDamage,afterDamage,rollSave,rollDamage,useRefresh,trigger,ledger,renderCard,renderActorContinuations,getContinuations,continueActivity,refreshOutsideEncounter};
}
