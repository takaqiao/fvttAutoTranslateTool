import {createVoltageLedger,currentVoltageToken,HIGH_VOLTAGE_SOURCE,ELEMENTAL_POWERS_SOURCE,VOLTAGE_MODULE_ID as ID,VOLTAGE_APPLY_PREFIX,voltageRollOption,voltageReceiptMatches} from './eldamon-voltage.mjs';
import {sourceUuid,siphonMultiplier} from './metapower/rules.mjs';
import {convertSiphonRoll} from './metapower/damage.mjs';
import {showNativeChoice} from './native-context.mjs';
import {withDamageMessageTarget} from './damage-message-targets.mjs';
import {createActorStateIndex} from './actor-state-index.mjs';

const values=c=>Array.from(c?.values?.()??c??[]),prefix=`${ID}:voltage:`,outcomes={criticalSuccess:0,success:0.5,failure:1,criticalFailure:2};
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const random=()=>globalThis.foundry?.utils?.randomID?.(24)??globalThis.crypto.randomUUID();
const binding=a=>({actorUuid:a.actorUuid,nonce:a.nonce,messageUuid:a.messageUuid});

/** Native interactions belong to the clicking owner. The GM only claims and
 * settles durable phases; observing an attack never rolls or applies damage. */
export function createEldamonVoltageProvider({game,fromUuid,observe,onRefresh,getRollContext,DamageRoll=globalThis.CONFIG?.Dice?.rolls?.find(C=>C.name==='DamageRoll'),convertRoll=convertSiphonRoll,selectChoice=showNativeChoice,onError=console.error}={}){
 const ledger=createVoltageLedger({game,fromUuid,onRefresh}),applications=new Map();let socket,registered=false;
 const activeActors=createActorStateIndex({game,matches:actor=>!!actor.flags?.[ID]?.voltage?.activeNonce});
 const active=()=>game.user?.id===game.users.activeGM?.id;
 async function notifyCard(actorUuid,nonce){
  const actor=await fromUuid(actorUuid);activeActors.refresh(actor);const a=actor?.flags?.[ID]?.voltage?.activations?.[nonce];if(!a)return;
  const message=await fromUuid(a.messageUuid),phase=a.native?.phase??null;
  if(message?.flags?.[ID]?.voltageStatus!==a.status||message?.flags?.[ID]?.voltagePhase!==phase)await message?.update?.({[`flags.${ID}.voltageStatus`]:a.status,[`flags.${ID}.voltagePhase`]:phase});
 }
 async function coordinate(method,payload,user){
  if(method==='trigger')return trigger(payload,user);
  try{return await ledger[method](payload,user)}
  finally{if(payload.nonce)await notifyCard(payload.actorUuid,payload.nonce).catch(onError)}
 }
 async function request(method,payload){
  if(active())return coordinate(method,payload,game.user);
  if(!socket||!game.users.activeGM)throw Error('High Voltage requires socketlib and an active GM.');
  const r=await socket.executeAsUser(`voltage:${method}`,game.users.activeGM.id,payload);if(!r?.ok)throw Error(r?.error??'High Voltage coordinator did not respond.');return r.value;
 }
 async function trigger(payload,user){
  if(payload.confirmed!==true)throw Error('Confirm the actual native hit or touch on the original High Voltage card.');
  const result=await ledger.claim({...payload,nativeInteraction:true},user);await notifyCard(payload.actorUuid,payload.nonce).catch(onError);return result;
 }
 async function abort(payload,status){await request('abortNative',{...payload,status}).catch(onError)}
 async function rollSave(activation){
  const payload={...binding(activation),operationId:random()},a=await request('beginSave',payload);
  try{
   const [actor,item,target]=await Promise.all([a.actorUuid,a.itemUuid,a.trigger.targetUuid].map(fromUuid)),statistic=target.actor.getStatistic('reflex'),roller=statistic?.check??statistic;
   if(typeof roller?.roll!=='function')throw Error('Native Reflex statistic is unavailable.');
   let save;await roller.roll({token:target,origin:actor,item,action:'high-voltage',dc:{slug:'eldamon',value:a.dc},traits:a.traits,
    extraRollOptions:[voltageRollOption(a),'action:high-voltage','damaging-effect',...a.traits.map(t=>'item:trait:'+t)],createMessage:true,
    callback:(_roll,_outcome,message)=>{save=message}});
   if(!save){await abort(payload,'cancelled');return null}
   return await request('finishSave',{...payload,saveUuid:save.uuid});
  }catch(error){await abort(payload,'uncertain');throw error}
 }
 async function rollDamage(activation){
  const payload={...binding(activation),operationId:random()},a=await request('beginRoll',payload);
  try{
   const item=await fromUuid(a.itemUuid);if(typeof DamageRoll!=='function')throw Error('Native DamageRoll is unavailable.');
   let damage=await new DamageRoll(`${a.level+1}d6[electricity]`,{},{rollerId:game.user.id}).evaluate();
   if(a.snapshot?.siphon?.applies)damage=convertRoll(damage,{DamageRoll,rejectMixedPartitions:true});
   damage=damage.alter(outcomes[a.native.save.outcome],0);
   const proof={...binding(a),targetUuid:a.trigger.targetUuid,operationId:payload.operationId};damage.options[ID]={...damage.options[ID],voltageDamage:proof};
   const origin=item.getOriginData(),options=[voltageRollOption(a),'action:high-voltage',...a.traits.map(t=>'item:trait:'+t),...origin.rollOptions??[]];
   // This is already the kept basic-save amount. No Toolbelt saveVariants are
   // copied from the action card, so its target row offers ordinary full apply.
   const message=await damage.toMessage(withDamageMessageTarget({speaker:a.speaker,
    flavor:`${esc(item.name)} · 反射基础豁免已计入；对绑定目标按全额应用${a.snapshot?.siphon?.applies?'（虹吸按目标特征调整）':''}`,
    flags:{pf2e:{origin,context:{type:'damage-roll',sourceType:'save',outcome:a.native.save.outcome,target:{actor:a.trigger.targetActorUuid,token:a.trigger.targetUuid},options}},[ID]:{voltageDamage:proof}}},a.trigger.targetUuid));
   return await request('finishRoll',{...payload,damageUuid:message?.uuid});
  }catch(error){await abort(payload,'uncertain');throw error}
 }
 async function beforeDamage(actor,params){
  const options=values(params.rollOptions),tracked=getRollContext?.(params.damage),trackedMessage=game.messages.get(tracked?.messageId);
  const tagged=options.filter(o=>typeof o==='string'&&o.startsWith(prefix)),privateProof=params.damage?.options?.[ID]?.voltageDamage;
  if(!tagged.length&&!privateProof&&!trackedMessage?.flags?.[ID]?.voltageDamage)return null;
  const sourceOptions=options.filter(o=>typeof o==='string'&&o.startsWith(`${ID}:source:`)),fallback=sourceOptions.length===1?sourceOptions[0].slice(`${ID}:source:`.length).split(':'):[];
  const source=getRollContext?tracked:fallback.length===2?{messageId:fallback[0],rollIndex:Number(fallback[1])}:null,message=game.messages.get(source?.messageId),proof=message?.flags?.[ID]?.voltageDamage;
  if(typeof DamageRoll!=='function'||!(params.damage instanceof DamageRoll)||params.damage._evaluated!==true||!Number.isFinite(params.damage.total)||params.damage.total<0||!source||source.rollIndex!==0||!proof||tagged.length!==1)throw Error('High Voltage requires the authorized tracked native damage card; consumed or foreign rolls cannot apply.');
  const target=params.token?.document??params.token;
  if(!currentVoltageToken(target)||target.uuid!==proof.targetUuid||target.actor.uuid!==actor.uuid||params.item?.uuid!==message.flags?.pf2e?.origin?.uuid||!target.actor.testUserPermission(game.user,'OWNER')||params.final||params.skipIWR)throw Error('High Voltage damage is bound to its original target, owner and native IWR.');
  const actorSource=await fromUuid(proof.actorUuid),a=actorSource?.flags?.[ID]?.voltage?.activations?.[proof.nonce];
  if(!a||tagged[0]!==voltageRollOption(a)||a.native?.damage?.messageUuid!==message.uuid)throw Error('High Voltage native damage source binding is invalid.');
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
  if(!item||!actor.testUserPermission(game.user,'OWNER')||typeof observe!=='function')throw Error('Owned Elemental Powers and the native action observer are required.');
  return observe({actor,item},async()=>{
   const draft=await item.toMessage(event,{create:false}),data=draft.toObject();
   data.content='<div class="pf2e chat-card"><header><h3>刷新威能 Refresh · 2动作</h3></header><p>恢复本角色已准备的启动与反应威能使用次数。</p></div>';
   data.flags??={};data.flags[ID]={...data.flags[ID],voltageRefreshActivity:{actions:2}};return globalThis.CONFIG.ChatMessage.documentClass.create(data);
  });
 }
 async function confirmTouch(a){
  const origin=await fromUuid(a.originUuid),choices=values(origin?.parent?.tokens).filter(t=>currentVoltageToken(t)&&t.actor.uuid!==a.actorUuid).map(t=>({value:t.uuid,label:t.name??t.actor.name??t.id}));
  const targetUuid=await selectChoice({title:'确认该生物确实触碰了高电压使用者（并非仅靠近或选为目标）',choices});if(!targetUuid)return null;
  return request('trigger',{...binding(a),targetUuid,kind:'touch',confirmed:true});
 }
 async function confirmAttack(a,metal=false){
  const choices=values(game.messages).filter(m=>m.flags?.pf2e?.context?.type==='attack-roll'&&m.flags.pf2e.context.target?.token===a.originUuid&&['success','criticalSuccess'].includes(m.flags.pf2e.context.outcome)&&m.timestamp>=a.channelTimestamp).map(m=>({value:m.uuid,label:`${m.alias??m.speaker?.alias??m.id} · ${m.item?.name??'近战攻击'}`}));
  const attackUuid=await selectChoice({title:metal?'确认这次原生近战命中使用金属武器':'选择实际命中的原生近战攻击（相邻、徒手或已知金属武器）',choices});if(!attackUuid)return null;
  return request('trigger',{...binding(a),attackUuid,kind:metal?'metal-hit':'attack',confirmed:true});
 }
 function renderCard(message,html){
  const root=html?.[0]??html;if(!root?.querySelectorAll)return;
  const proof=message.flags?.[ID]?.metapowerUse,actor=message.actor??game.actors.get(message.speaker?.actor),receipt=actor?.flags?.[ID]?.metapower?.receipts?.[proof?.nonce];
  if(receipt?.sourceUuid!==HIGH_VOLTAGE_SOURCE)return;
  for(const node of root.querySelectorAll('[data-damage-roll],[data-pf2-check]')){node.removeAttribute('data-damage-roll');node.removeAttribute('data-pf2-check');node.setAttribute('aria-disabled','true');node.style.pointerEvents='none';}
  if(root.querySelector('[data-voltage-state]')||receipt.messageUuid!==message.uuid)return;
  const a=actor.flags?.[ID]?.voltage?.activations?.[receipt.nonce];if(!a)return;
  const box=root.ownerDocument.createElement('div');box.dataset.voltageState='';
  const labels={'awaiting-save':'触发已确认；等待绑定目标的拥有者或GM进行原生反射豁免。','awaiting-damage':'豁免已记录；等待施法者拥有者或GM投掷伤害。','awaiting-application':'伤害卡已发布；请对绑定目标按全额应用。','rolling-save':'原生豁免处理中；中断后请GM核对原记录。','rolling-damage':'伤害卡处理中；中断后请GM核对原记录。',applying:'伤害应用处理中；中断后请GM核对原回执。'};
  box.textContent=a.expiryReason==='no-encounter'?`当前没有已开始的遭遇，无法追踪下回合前的反击窗口；请由GM裁定。${a.refreshSuppressed?'本次虹吸未刷新威能。':'已即时刷新威能。'}`:
   a.status==='armed'?`高电压已就绪；${a.refreshSuppressed?'本次虹吸不刷新威能':'已即时刷新威能'}。实际命中或触碰后，请在此卡确认触发。`:
   a.status==='claimed'?(labels[a.native?.phase]??'此前触发已认领；请GM核对原保存与伤害记录，勿重复应用。'):
   `高电压：${a.status==='done'?'已按原生伤害回执结算':a.status==='uncertain'?'结果不确定；请GM核对原回执，勿重复应用':a.status==='cancelled'?'已取消，本次触发已消耗':'已到期'}。`;
  const button=(label,run)=>{const element=root.ownerDocument.createElement('button');element.type='button';element.textContent=label;element.addEventListener('click',event=>{event.preventDefault();event.stopPropagation();element.disabled=true;Promise.resolve().then(run).catch(onError).finally(()=>{element.disabled=false})});box.append(element)};
  if(a.status==='armed'&&actor.testUserPermission(game.user,'OWNER')){
   button('确认原生命中',()=>confirmAttack(a));button('确认金属武器命中',()=>confirmAttack(a,true));button('确认实际触碰',()=>confirmTouch(a));
  }
  const targetParts=a.trigger?.targetUuid?.split('.'),target=targetParts&&game.scenes?.get(targetParts[1])?.tokens?.get(targetParts[3]);
  if(a.status==='claimed'&&a.native?.phase==='awaiting-save'&&target?.actor?.testUserPermission(game.user,'OWNER'))button('绑定目标：原生反射豁免',()=>rollSave(a));
  if(a.status==='claimed'&&a.native?.phase==='awaiting-damage'&&actor.testUserPermission(game.user,'OWNER'))button('投掷原生伤害',()=>rollDamage(a));
  root.append(box);
 }
 function register({Hooks,socket:api}={}){
  if(registered)return;registered=true;socket=api;activeActors.register(Hooks);
  for(const method of ['channel','refreshActivity','trigger','beginSave','finishSave','beginRoll','finishRoll','beginDamage','finishDamage','abortNative'])socket.register(`voltage:${method}`,async function(payload){try{return {ok:true,value:await coordinate(method,payload,game.users.get(this.socketdata.userId))}}catch(error){return {ok:false,error:error.message}}});
  const sweep=()=>{if(!active())return;return Promise.all(activeActors.values().map(actor=>{const nonce=actor.flags?.[ID]?.voltage?.activeNonce;return nonce?ledger.expire({actorUuid:actor.uuid},game.user).then(()=>notifyCard(actor.uuid,nonce)).catch(onError):null}))};
  Hooks.on('createChatMessage',capture);
  for(const hook of ['pf2e.startTurn','updateCombat','deleteCombat','deleteItem','deleteToken','userConnected','updateUser'])Hooks.on(hook,sweep);
  Hooks.on('renderChatMessageHTML',renderCard);
  for(const hook of ['renderActorSheetPF2e','renderCharacterSheetPF2e','renderActorSheetV2'])Hooks.on(hook,(app,html)=>{
   const root=html?.[0]??html,actor=app.actor;if(!root?.querySelector||!actor?.testUserPermission(game.user,'OWNER')||!values(actor.items).some(i=>sourceUuid(i)===ELEMENTAL_POWERS_SOURCE)||root.querySelector('[data-voltage-refresh]'))return;
   const host=root.querySelector('.tab.actions[data-tab="actions"],section[data-tab="actions"]')??root,button=root.ownerDocument.createElement('button');button.type='button';button.dataset.voltageRefresh='';button.textContent='刷新威能 Refresh · 2动作';button.addEventListener('click',event=>{event.preventDefault();event.stopPropagation();button.disabled=true;useRefresh(actor,event).catch(onError).finally(()=>{button.disabled=false})});host.append(button);
  });sweep();
 }
 const refreshOutsideEncounter=({actor,nonce})=>ledger.refreshOutsideEncounter({actorUuid:actor.uuid,nonce},game.user);
 return {register,onCommittedChannel,beforeDamage,afterDamage,rollSave,rollDamage,useRefresh,trigger,ledger,renderCard,refreshOutsideEncounter};
}
