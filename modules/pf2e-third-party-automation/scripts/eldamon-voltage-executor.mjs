import {createVoltageLedger,voltageState,currentVoltageToken,HIGH_VOLTAGE_SOURCE,ELEMENTAL_POWERS_SOURCE,VOLTAGE_MODULE_ID as ID} from './eldamon-voltage.mjs';
import {sourceUuid,siphonMultiplier} from './metapower/rules.mjs';
import {convertSiphonRoll} from './metapower/damage.mjs';
import {showNativeChoice} from './native-context.mjs';

const values=c=>Array.from(c?.values?.()??c??[]),prefix=`${ID}:voltage:`,outcomes={criticalSuccess:0,success:0.5,failure:1,criticalFailure:2};
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const speakerToken=s=>`Scene.${s?.scene}.Token.${s?.token}`;

/** Explicit registration; importing this module never changes game or Hooks. */
export function createEldamonVoltageProvider({game,fromUuid,observe,onRefresh,DamageRoll=globalThis.CONFIG?.Dice?.rolls?.find(C=>C.name==='DamageRoll'),convertRoll=convertSiphonRoll,selectChoice=showNativeChoice,onError=console.error}={}){
 const ledger=createVoltageLedger({game,fromUuid,onRefresh}),grants=new WeakMap();let socket,registered=false;
 const activeGM=()=>{if(game.user?.id!==game.users.activeGM?.id)throw Error('High Voltage execution requires the active GM.');};
 async function notifyCard(actorUuid,nonce){
  const actor=await fromUuid(actorUuid),activation=voltageState(actor).activations[nonce];if(!activation)return;
  const message=await fromUuid(activation.messageUuid);if(message?.flags?.[ID]?.voltageStatus!==activation.status)await message?.update?.({[`flags.${ID}.voltageStatus`]:activation.status});
 }
 async function request(method,payload){
  if(!socket||!game.users.activeGM)throw Error('High Voltage requires socketlib and an active GM.');
  const r=await socket.executeAsUser(`voltage:${method}`,game.users.activeGM.id,payload);if(!r?.ok)throw Error(r?.error??'High Voltage coordinator did not respond.');return r.value;
 }
 async function context(activation){
  activeGM();const [actor,item,origin,target,message]=await Promise.all([activation.actorUuid,activation.itemUuid,activation.originUuid,activation.trigger.targetUuid,activation.messageUuid].map(fromUuid));
  const saved=voltageState(actor).activations[activation.nonce],receipt=actor?.flags?.[ID]?.metapower?.receipts?.[activation.nonce];
  if(saved?.status!=='claimed'||saved.trigger?.targetUuid!==activation.trigger.targetUuid||saved.messageUuid!==activation.messageUuid||receipt?.status!=='committed'||receipt.messageUuid!==message?.uuid||
   game.messages.get(message?.id)!==message||item?.actor!==actor||actor.items.get(item.id)!==item||sourceUuid(item)!==HIGH_VOLTAGE_SOURCE||!currentVoltageToken(origin)||origin.actor!==actor||!currentVoltageToken(target)||target.actor.uuid!==activation.trigger.targetActorUuid||target.parent!==origin.parent)throw Error('High Voltage execution source, target or claim changed.');
  return {actor,item,origin,target,message};
 }
 async function contextualDamage({actor,item,target},options){
  const recipient=target.actor,originOptions=options.filter(o=>o.startsWith('self:')).map(o=>o.replace(/^self\b/,'origin'));
  const test=[...options,...actor.getRollOptions?.(['damage-received'])??[],...recipient.getSelfRollOptions?.('target')??[]];
  const deferred=actor.synthetics?.ephemeralEffects?.['damage-received']?.target??[];
  const effects=(await Promise.all(deferred.map(fn=>fn({test,resolvables:{weapon:item}})))).filter(Boolean).map(effect=>{
   const data=structuredClone(effect);if(data.type==='effect'){data.system.context={origin:{actor:actor.uuid,token:null,item:null,spellcasting:null,rollOptions:[]},target:{actor:recipient.uuid,token:null},roll:null};data.system.duration={value:-1,unit:'unlimited',expiry:null,sustained:false};}return data;
  });
  return {recipient:recipient.getContextualClone(originOptions,effects),originOptions};
 }
 async function execute(activation){
  const payload={actorUuid:activation.actorUuid,nonce:activation.nonce,messageUuid:activation.messageUuid};
  try{
   const ctx=await context(activation),{actor,item,target}=ctx,statistic=target.actor.getStatistic('reflex'),roller=statistic?.check??statistic;
   if(typeof roller?.roll!=='function'||!Number.isFinite(activation.dc))throw Error('Native Reflex statistic or original Eldamon power DC is unavailable.');
   const option=prefix+activation.nonce,traits=activation.traits,options=[option,'action:high-voltage',...traits.map(t=>'item:trait:'+t)];let save;
   await roller.roll({token:target,origin:actor,item,action:'high-voltage',dc:{slug:'eldamon',value:activation.dc},traits,extraRollOptions:options,createMessage:true,
    callback:(_roll,outcome,message)=>{save={outcome,message};}});
   if(!save){await ledger.settle({...payload,status:'cancelled'},game.user);return {status:'cancelled'};}
   const check=save.message,c=check?.flags?.pf2e?.context;
   if(game.messages.get(check?.id)!==check||c?.type!=='saving-throw'||!Object.hasOwn(outcomes,c.outcome)||c.outcome!==save.outcome||check.speaker?.actor!==target.actor.id||speakerToken(check.speaker)!==target.uuid||check.flags?.pf2e?.origin?.uuid!==item.uuid||!c.options?.includes(option))throw Error('Original native Reflex result binding is invalid.');
   await context(activation);if(typeof DamageRoll!=='function')throw Error('Native DamageRoll is unavailable.');
   let damage=await new DamageRoll(`${activation.level+1}d6[electricity]`,{},{rollerId:game.user.id}).evaluate();
   if(activation.snapshot?.siphon?.applies)damage=convertRoll(damage,{DamageRoll,rejectMixedPartitions:true});
   const proof={actorUuid:actor.uuid,nonce:activation.nonce,messageUuid:activation.messageUuid,targetUuid:target.uuid};
   damage.options[ID]={...damage.options[ID],voltageDamage:proof};
   const origin=item.getOriginData(),originRollOptions=origin.rollOptions??[],allOptions=[...options,...originRollOptions];
   const message=await damage.toMessage({speaker:activation.speaker,flavor:`${esc(item.name)} · 反射基础豁免`,flags:{pf2e:{origin,context:{type:'damage-roll',sourceType:'save',outcome:c.outcome,target:{actor:target.actor.uuid,token:target.uuid},options:allOptions}},[ID]:{voltageDamage:proof}}});
   if(game.messages.get(message?.id)!==message)throw Error('Native High Voltage damage card was not persisted.');
   const multiplier=siphonMultiplier(activation.snapshot,target.actor.traits??target.actor.system?.traits?.value??[]);
   damage=damage.alter(outcomes[c.outcome],0);if(multiplier!==1)damage=damage.alter(multiplier,0);
   damage.options[ID]={...damage.options[ID],voltageDamage:proof};
   const {recipient,originOptions}=await contextualDamage(ctx,allOptions);await context(activation);
   const rollOptions=new Set([...allOptions.filter(o=>! /^(?:self|target)(?::|$)/.test(o)),...originOptions,...recipient.getSelfRollOptions()]);
   const params={damage,token:target,item,skipIWR:false,rollOptions,shieldBlockRequest:false,outcome:c.outcome};
   grants.set(damage,{actorUuid:target.actor.uuid,nonce:activation.nonce});
   try{await recipient.applyDamage(params);}finally{grants.delete(damage);}
   await ledger.settle({...payload,status:'done',result:{saveUuid:check.uuid,damageUuid:message.uuid,outcome:c.outcome}},game.user);
   return {status:'done',saveUuid:check.uuid,damageUuid:message.uuid};
  }catch(error){await ledger.settle({...payload,status:'uncertain'},game.user).catch(()=>{});throw error;}
 }
 async function trigger(payload,user){
  const activation=await ledger.claim(payload,user);
  try{return activation?await execute(activation):null;}finally{await notifyCard(payload.actorUuid,payload.nonce).catch(onError);}
 }
 async function beforeDamage(actor,params){
  const proof=params.damage?.options?.[ID]?.voltageDamage,option=[...params.rollOptions??[]].find(o=>o.startsWith(prefix));
  if(!proof&&!option)return null;
  activeGM();const grant=grants.get(params.damage);
  if(!proof||!grant||grant.actorUuid!==actor.uuid||grant.nonce!==proof.nonce)throw Error('High Voltage damage is already consumed or lacks its authorized application grant.');
  grants.delete(params.damage);return null;
 }
 async function onCommittedChannel({receipt,message,user=game.users.get(receipt.userId)}){
  const payload={actorUuid:receipt.actorUuid,nonce:receipt.nonce,messageUuid:message.uuid};
  const deliver=method=>game.user?.id===game.users.activeGM?.id?ledger[method](payload,user):request(method,payload);
  if(receipt.sourceUuid===HIGH_VOLTAGE_SOURCE){const result=await deliver('channel');await notifyCard(payload.actorUuid,payload.nonce).catch(onError);return result;}
  if(receipt.sourceUuid===ELEMENTAL_POWERS_SOURCE&&message.flags?.[ID]?.voltageRefreshActivity)return deliver('refreshActivity');
  return null;
 }
 async function useRefresh(actor,event){
  const item=values(actor.items).find(i=>sourceUuid(i)===ELEMENTAL_POWERS_SOURCE);
  if(!item||!actor.testUserPermission(game.user,'OWNER')||typeof observe!=='function')throw Error('Owned Elemental Powers and the native action observer are required.');
  return observe({actor,item},async()=>{
   const draft=await item.toMessage(event,{create:false}),data=draft.toObject();
   data.content='<div class="pf2e chat-card"><header><h3>刷新威能 Refresh · 2动作</h3></header><p>恢复本角色已准备的启动与反应威能使用次数。</p></div>';
   data.flags??={};data.flags[ID]={...data.flags[ID],voltageRefreshActivity:{actions:2}};
   return globalThis.CONFIG.ChatMessage.documentClass.create(data);
  });
 }
 async function confirmTouch(activation){
  const origin=await fromUuid(activation.originUuid),choices=values(origin?.parent?.tokens).filter(t=>currentVoltageToken(t)&&t.actor.uuid!==activation.actorUuid).map(t=>({value:t.uuid,label:t.name??t.actor.name??t.id}));
  const targetUuid=await selectChoice({title:'确认该生物确实触碰了高电压使用者（并非仅靠近或选为目标）',choices});if(!targetUuid)return null;
  return request('trigger',{actorUuid:activation.actorUuid,nonce:activation.nonce,messageUuid:activation.messageUuid,targetUuid,kind:'touch',confirmed:true});
 }
 async function confirmMetal(activation){
  const choices=values(game.messages).filter(m=>m.flags?.pf2e?.context?.type==='attack-roll'&&m.flags.pf2e.context.target?.token===activation.originUuid&&['success','criticalSuccess'].includes(m.flags.pf2e.context.outcome)&&m.timestamp>=activation.channelTimestamp).map(m=>({value:m.uuid,label:`${m.alias??m.speaker?.alias??m.id} · ${m.item?.name??'近战攻击'}`}));
  const attackUuid=await selectChoice({title:'确认这次原生近战命中使用金属武器',choices});if(!attackUuid)return null;
  return request('trigger',{actorUuid:activation.actorUuid,nonce:activation.nonce,messageUuid:activation.messageUuid,attackUuid,kind:'metal-hit',confirmed:true});
 }
 function renderCard(message,html){
  const root=html?.[0]??html;if(!root?.querySelectorAll)return;
  const proof=message.flags?.[ID]?.metapowerUse,actor=game.actors.get(message.speaker?.actor),receipt=actor?.flags?.[ID]?.metapower?.receipts?.[proof?.nonce];
  if(receipt?.sourceUuid!==HIGH_VOLTAGE_SOURCE)return;
  // Original inline damage/save controls must never bypass the durable trigger.
  for(const node of root.querySelectorAll('[data-damage-roll],[data-pf2-check]')){node.removeAttribute('data-damage-roll');node.removeAttribute('data-pf2-check');node.setAttribute('aria-disabled','true');node.style.pointerEvents='none';}
  if(root.querySelector('[data-voltage-state]')||receipt.messageUuid!==message.uuid)return;
  const activation=voltageState(actor).activations[receipt.nonce];if(!activation)return;
  const box=root.ownerDocument.createElement('div');box.dataset.voltageState='';
  box.textContent=activation.expiryReason==='no-encounter'?`当前没有已开始的遭遇，无法自动追踪下回合前的反击窗口；请由GM裁定。${activation.refreshSuppressed?'本次虹吸未刷新威能。':'已即时刷新威能。'}`:activation.status==='armed'?`高电压已就绪；${activation.refreshSuppressed?'本次虹吸不刷新威能':'已即时刷新威能'}。首次有效命中或实际触碰后放电。`:`高电压：${activation.status==='expired'?'已到期':activation.status==='done'?'已放电':'窗口已关闭'}。`;
  if(activation.status==='armed'&&actor.testUserPermission(game.user,'OWNER'))for(const [label,run]of [['确认实际触碰并放电',confirmTouch],['确认金属武器命中',confirmMetal]]){const button=root.ownerDocument.createElement('button');button.type='button';button.textContent=label;button.addEventListener('click',event=>{event.preventDefault();event.stopPropagation();button.disabled=true;run(activation).catch(onError).finally(()=>{button.disabled=false})});box.append(button);}
  root.append(box);
 }
 function register({Hooks,socket:api}={}){
  if(registered)return;registered=true;socket=api;
  for(const method of ['channel','refreshActivity','trigger'])socket.register(`voltage:${method}`,async function(payload){try{return {ok:true,value:await(method==='trigger'?trigger(payload,game.users.get(this.socketdata.userId)):ledger[method](payload,game.users.get(this.socketdata.userId)))}}catch(error){return {ok:false,error:error.message}}});
  const actors=()=>[...new Map([...values(game.actors),...values(game.combat?.combatants??game.combat?.turns).map(c=>c.actor).filter(Boolean)].map(a=>[a.uuid,a])).values()];
  const sweep=()=>{if(game.user?.id!==game.users.activeGM?.id)return;for(const actor of actors()){const nonce=voltageState(actor).activeNonce;if(nonce)ledger.expire({actorUuid:actor.uuid},game.user).then(()=>notifyCard(actor.uuid,nonce)).catch(onError)}};
  Hooks.on('createChatMessage',message=>{
   if(game.user?.id!==game.users.activeGM?.id||message.flags?.pf2e?.context?.type!=='attack-roll')return;
   const actor=actors().find(a=>a.uuid===message.flags.pf2e.context.target?.actor),state=voltageState(actor),a=state.activations[state.activeNonce];if(!a)return;
   trigger({actorUuid:actor.uuid,nonce:a.nonce,messageUuid:a.messageUuid,kind:'attack',attackUuid:message.uuid},game.user).catch(onError);
  });
  for(const hook of ['pf2e.startTurn','updateCombat','deleteCombat','deleteItem','deleteToken'])Hooks.on(hook,sweep);
  Hooks.on('renderChatMessageHTML',renderCard);
  for(const hook of ['renderActorSheetPF2e','renderCharacterSheetPF2e','renderActorSheetV2'])Hooks.on(hook,(app,html)=>{
   const root=html?.[0]??html,actor=app.actor;if(!root?.querySelector||!actor?.testUserPermission(game.user,'OWNER')||!values(actor.items).some(i=>sourceUuid(i)===ELEMENTAL_POWERS_SOURCE)||root.querySelector('[data-voltage-refresh]'))return;
   const host=root.querySelector('.tab.actions[data-tab="actions"],section[data-tab="actions"]')??root,button=root.ownerDocument.createElement('button');button.type='button';button.dataset.voltageRefresh='';button.textContent='刷新威能 Refresh · 2动作';button.addEventListener('click',event=>{event.preventDefault();event.stopPropagation();button.disabled=true;useRefresh(actor,event).catch(onError).finally(()=>{button.disabled=false})});host.append(button);
  });
  sweep();
 }
 const refreshOutsideEncounter=({actor,nonce})=>ledger.refreshOutsideEncounter({actorUuid:actor.uuid,nonce},game.user);
 return {register,onCommittedChannel,beforeDamage,useRefresh,trigger,ledger,renderCard,refreshOutsideEncounter};
}
