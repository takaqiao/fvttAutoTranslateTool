import {MODULE_ID} from './rules.mjs';
import {showNativeChoice} from './native-context.mjs';
import {EFFECTS,PILGRIM_FLAG,PilgrimError,values,rewardKey,usableReward,scarfBranch,serializeReward} from './sog-pilgrim-rules.mjs';

const names={release:'灵魂连携 ◆◆',shift:'变型 ◆',tree:'余烬重生 ◆◆◆',restore:'恢复武器 ◇',scarf:'启动围巾 ◆',storm:'花瓣风暴 ◆◆'};
const state=item=>item.flags?.[MODULE_ID]?.[PILGRIM_FLAG]??{};
const own=doc=>doc.flags?.[MODULE_ID]?.[PILGRIM_FLAG];
export const playerError=error=>error instanceof PilgrimError?error.message:'启动未完成，请联系GM核对。';
export function prepareScarfChoices(data,actor,templates) {
 const key=data?.flags?.world?.sogWontonNativeAutomation?.key;
 const branch=['ghost','astral','ward'].find(branch=>key&&templates?.get(EFFECTS[branch])?.flags?.world?.sogWontonNativeAutomation?.key===key);
 if(data?.type!=='effect'||!branch||!actor)return;
 const choices=values(actor.items).filter(item=>item.type==='weapon'&&item.system.equipped?.carryType!=='dropped'&&scarfBranch(item)===branch).map(item=>({value:item.id,label:item.name}));
 if(!choices.length)throw new PilgrimError('需要一件符合条件的携带武器。');
 for(const rule of data.system.rules??[])if(rule.key==='ChoiceSet'&&rule.flag==='weapon'){
  rule.choices=choices;
  if(rule.selection&&!choices.some(choice=>choice.value===rule.selection))delete rule.selection;
 }
}
export function hasPublishedUse(item,use,game) {
 if(use.operation==='scarf')return values(item.actor?.items).some(effect=>effect.type==='effect'&&own(effect)?.kind==='scarf'&&own(effect).source===item.uuid&&own(effect).nonce===use.nonce);
 return ['release','tree','storm'].includes(use.operation)&&values(game.messages).some(message=>message.isDamageRoll===true&&own(message)?.generated===true&&own(message).source===item.uuid&&own(message).nonce===use.nonce);
}
export function activationOperations(key) { return ({fan:['release'],branch:['shift','tree','restore'],scarf:['scarf'],hairpin:['storm']})[key]??[]; }
export function assertAvailable(current,operation,now) {
 if(current.use?.status==='pending')throw new PilgrimError('上次启动尚待核对，请联系GM。');
 if(operation==='release'&&Number.isFinite(current.lastUse?.release)&&now-current.lastUse.release<3600)throw new PilgrimError('灵魂连携每小时只能使用一次。');
 if(['tree','scarf','storm'].includes(operation)&&current.spent)throw new PilgrimError('今日已经启动；完成每日休息后恢复。');
}

export async function createPilgrimRewards({game,fromUuid,choose,onError=console.error,canvas=globalThis.canvas}={}) {
 const enabled=game.system?.id==='pf2e'&&['sog','sog-pilgrim-qa'].includes(game.world?.id);
 if(!enabled)return {register(){},enabled:false};
 const [{createPilgrimFan},{createPilgrimAreas},{createPilgrimMedia},{createPilgrimShift}]=await Promise.all([import('./sog-pilgrim-fan.mjs'),import('./sog-pilgrim-areas.mjs'),import('./sog-pilgrim-media.mjs'),import('./sog-pilgrim-shift.mjs')]);
 const media=createPilgrimMedia({game,canvas,fromUuid,onError});
 const fan=createPilgrimFan({game,fromUuid,media,onError});
 const areas=createPilgrimAreas({game,fromUuid,media,onError,canvas});
 const shift=createPilgrimShift({game,fromUuid});
 const tracked=new Map();let socket;
 const active=()=>game.user?.id===game.users.activeGM?.id;
 const report=error=>{onError(error);};
 const quietMedia=async(fn)=>{try{await fn();}catch(error){report(error);}};
 const index=actor=>{if(!actor)return;const relevant=values(actor.items).filter(i=>rewardKey(i));if(relevant.length)tracked.set(actor.uuid,actor);else tracked.delete(actor.uuid);};
 const cosmeticChange=item=>['fan','scarf'].includes(rewardKey(item))||own(item)?.kind==='leaves';
 async function resetDaily(actor,user) {
  if(!actor?.testUserPermission(user,'OWNER'))throw new PilgrimError('请使用你持有的物品。');
  for(const item of values(actor.items).filter(item=>['branch','scarf','hairpin'].includes(rewardKey(item))))await serializeReward(item.uuid,async()=>{
   if(actor.items.get(item.id)===item&&state(item).spent===true)await item.update({[`flags.${MODULE_ID}.${PILGRIM_FLAG}.spent`]:false});
  });
 }
 const liveItem=async(uuid,user)=>{
  const item=await fromUuid(uuid);
  if(!rewardKey(item)||!item.actor||item.actor.items.get(item.id)!==item||!item.actor.testUserPermission(user,'OWNER'))throw new PilgrimError('请使用你持有的物品。');
  return item;
 };
 function sourceToken(item,uuid) {
  const token=values(game.scenes).flatMap(s=>values(s.tokens)).find(t=>t.uuid===uuid);
  if(!token?.actor||token.actor.uuid!==item.actor.uuid||token.parent.tokens.get(token.id)!==token)throw new PilgrimError('请在场景中选择使用物品的角色。');
  return token;
 }
 function within(source,target,range,{visible=false}={}) {
  if(!source||!target||target.parent!==source.parent||target.parent.tokens.get(target.id)!==target||canvas?.scene!==source.parent||!source.object||!target.object)throw new PilgrimError('请让GM打开角色所在场景，再选择目标。');
  const distance=source.object.distanceTo(target.object);
  if(!Number.isFinite(distance)||distance>range)throw new PilgrimError('目标超出范围。');
  if(visible&&(source.actor.canSee===false||source.actor.hasCondition?.('blinded')||source.object.checkCollision?.(target.object.center,{origin:source.object.center,type:'sight',mode:'any'})!==false))throw new PilgrimError('请选择角色能看见的目标。');
 }
 async function card({item,formula,label,nonce,target,save,gm=false,validate}) {
  await validate?.();
  const speaker=ChatMessage.getSpeaker({actor:item.actor,token:target?.actor===item.actor?target:null});
  const flavor=await foundry.applications.ux.TextEditor.enrichHTML(`<p><strong>${label}</strong>${save?` @Check[type:${save.type}|dc:${save.dc}|basic:true]`:''}</p>`,{async:true});
  const flags={pf2e:{origin:{uuid:item.uuid,actor:item.actor.uuid,type:item.type},context:{type:'damage-roll',options:[],...(target?{target:{actor:target.actor.uuid,token:target.uuid}}:{})}},[MODULE_ID]:{[PILGRIM_FLAG]:{generated:true,source:item.uuid,nonce}}};
  const DamageRoll=globalThis.CONFIG?.Dice?.rolls?.find(cls=>cls.name==='DamageRoll');
  if(!DamageRoll)throw new PilgrimError('原生掷骰暂时不可用。');
  const roll=new DamageRoll(formula);
  await roll.evaluate();
  await validate?.();
  const message=await roll.toMessage({speaker,flavor,flags,...gm?{whisper:ChatMessage.getWhisperRecipients('GM').map(u=>u.id)}:{}},{rollMode:gm?'gmroll':game.settings.get('core','rollMode')});
  if(!message?.id)throw new PilgrimError('启动尚未完成，请联系GM核对。');
  return message;
 }
 async function applyScarf({item,weapon,nonce}) {
  const branch=scarfBranch(weapon),source=await fromUuid('Item.'+EFFECTS[branch]);
  if(!source||source.type!=='effect')throw new PilgrimError('效果暂时不可用，请联系GM。');
  const data=source.toObject();delete data._id;
  data.flags={...data.flags,pf2e:{...data.flags?.pf2e,rulesSelections:{...data.flags?.pf2e?.rulesSelections,weapon:weapon.id}},[MODULE_ID]:{...data.flags?.[MODULE_ID],[PILGRIM_FLAG]:{kind:'scarf',source:item.uuid,nonce,weapon:weapon.id}}};
  for(const rule of data.system.rules)if(rule.key==='ChoiceSet'&&rule.flag==='weapon'){
   rule.selection=weapon.id;
   rule.choices=[{value:weapon.id,label:weapon.name}];
  }
  data.system.start={value:game.time.worldTime,initiative:game.combat?.combatant?.initiative??null};
  const created=await item.actor.createEmbeddedDocuments('Item',[data]);
  if(!created[0])throw new PilgrimError('启动尚未完成，请联系GM核对。');
  await quietMedia(()=>media.scarf(item,item.actor.rollOptions?.all?.['wonton-ghost-scarf:lit']===true));
  return {effectId:created[0].id};
 }
 async function execute(payload,user) {
  if(!active())throw new PilgrimError('请等待GM在线后使用。');
  const item=await liveItem(payload.itemUuid,user),key=rewardKey(item),operation=payload.operation;
  if(!activationOperations(key).includes(operation))throw new PilgrimError('该物品没有此项启动。');
  return serializeReward(item.uuid,async()=>{
   if(operation==='restore')return areas.restore({item,user});
   if(!usableReward(item))throw new PilgrimError(item.type==='equipment'?'请先佩戴并投资该物品。':'请先持用该武器。');
   assertAvailable(state(item),operation,game.time.worldTime);
   if(operation==='shift'){await shift.apply({item,formUuid:payload.formUuid});return {status:'done'};}
   const origin=payload.sourceUuid?sourceToken(item,payload.sourceUuid):null;
   const target=payload.targetUuid?await fromUuid(payload.targetUuid):null;
   let weapon;
   if(operation==='release'){
    within(origin,target,30,{visible:true});
    if(target.hidden&&!user.isGM)throw new PilgrimError('请选择你能看见的目标。');
    if(!target.actor?.isOfType('character','npc')||target.actor.isDead||target.actor.traits?.has('construct')&&!target.actor.traits?.has('undead'))throw new PilgrimError('请选择活物或不死生物。');
    if((state(item).fan?.leaves??[]).length!==3)throw new PilgrimError('需要三片亮起的金叶。');
   }else if(operation==='scarf'){
    weapon=item.actor.items.get(payload.weaponId);
    if(weapon?.type!=='weapon'||weapon.system.equipped?.carryType==='dropped')throw new PilgrimError('请选择你携带的武器。');
   }else if(operation==='tree'){
    within(origin,target,item.actor.getReach?.({action:'interact'})??5);
    const traits=target.actor?.traits;
    if(!(traits?.has('undead')||traits?.has('fiend'))||!target.actor.isDead)throw new PilgrimError('请选择触及内不死生物或魔族的尸体。');
   }else if(operation==='storm')areas.validatePlacement({item,source:origin,position:payload.position,range:60});
   const nonce=payload.nonce;
   if(typeof nonce!=='string'||!/^[A-Za-z0-9_-]{8,80}$/.test(nonce))throw new PilgrimError('请重新点击启动。');
   await item.update({[`flags.${MODULE_ID}.${PILGRIM_FLAG}.use`]:{nonce,operation,status:'pending',startedAt:game.time.worldTime}});
   let result;
   if(operation==='release')result=await fan.release({item,source:origin,target,nonce,card});
   else if(operation==='scarf')result=await applyScarf({item,weapon,nonce});
   else result=await areas[operation]({item,source:origin,target,position:payload.position,nonce,card});
   if(result===null){await item.update({[`flags.${MODULE_ID}.${PILGRIM_FLAG}.use`]:{nonce,operation,status:'cancelled'}});return {status:'cancelled'};}
   await item.update({[`flags.${MODULE_ID}.${PILGRIM_FLAG}.use`]:{nonce,operation,status:'done',...result},[`flags.${MODULE_ID}.${PILGRIM_FLAG}.lastUse.${operation}`]:game.time.worldTime,...operation!=='release'?{[`flags.${MODULE_ID}.${PILGRIM_FLAG}.spent`]:true}:{}});
   return {status:'done',...result};
  });
 }
 async function request(item,operation) {
  if(!game.users.activeGM)throw new PilgrimError('请等待GM在线后使用。');
  const actor=item.actor;
  if(!actor?.isOwner)throw new PilgrimError('请使用你持有的物品。');
  if(operation!=='restore'&&!usableReward(item))throw new PilgrimError(item.type==='equipment'?'请先佩戴并投资该物品。':'请先持用该武器。');
  if(operation!=='restore')assertAvailable(state(item),operation,game.time.worldTime);
  const token=values(canvas?.tokens?.controlled).find(t=>t.actor?.uuid===actor.uuid)?.document??values(canvas?.scene?.tokens).find(t=>t.actor?.uuid===actor.uuid&&!t.hidden);
  const targets=values(game.user.targets).map(t=>t.document??t);
  const payload={itemUuid:item.uuid,operation,nonce:foundry.utils.randomID(),sourceUuid:token?.uuid};
  if(['release','tree'].includes(operation)){
   if(targets.length!==1)throw new PilgrimError(operation==='tree'?'请先选中一具不死生物或魔族的尸体。':'请先选中一个目标。');
   if(targets[0].object?.isVisible===false)throw new PilgrimError('请选择你能看见的目标。');
   payload.targetUuid=targets[0].uuid;
  }else if(operation==='shift'){
   payload.formUuid=await shift.select(item);
   if(!payload.formUuid)return {status:'cancelled'};
  }else if(operation==='scarf'){
   const weapons=values(actor.items).filter(i=>i.type==='weapon'&&i.system.equipped?.carryType!=='dropped');
   if(!weapons.length)throw new PilgrimError('需要一件携带的武器。');
   payload.weaponId=weapons.length===1?weapons[0].id:await showNativeChoice({title:'选择围巾缠绕的武器',choices:weapons.map(w=>({value:w.id,label:w.name}))});
   if(!payload.weaponId)return {status:'cancelled'};
  }else if(operation==='storm'){
   if(!token)throw new PilgrimError('请在场景中选择使用物品的角色。');
   payload.position=await areas.pick({item,source:token});
   if(!payload.position)return {status:'cancelled'};
  }
  const response=await socket.executeAsUser('pilgrim-use',game.users.activeGM.id,payload);
  if(!response.ok)throw new PilgrimError(response.error);
  return response.value;
 }
 function controls(item,element) {
  if(!rewardKey(item)||!item.actor?.isOwner||!element?.querySelector||element.querySelector('.sog-pilgrim-actions'))return;
  const bar=document.createElement('div');bar.className='sog-pilgrim-actions';
  const tree=values(item.actor.items).some(effect=>own(effect)?.kind==='tree'&&own(effect).source===item.uuid);
  const operations=activationOperations(rewardKey(item)).filter(operation=>operation==='restore'?tree:operation==='tree'?!tree:true);
  for(const operation of operations){const button=document.createElement('button');button.type='button';button.textContent=names[operation];button.addEventListener('click',async event=>{event.preventDefault();event.stopPropagation();button.disabled=true;try{await request(item,operation);}catch(error){report(error);ui.notifications.warn(playerError(error));}finally{button.disabled=false;}});bar.append(button);}
  (element.querySelector('.message-content')??element.querySelector('.window-content')??element).append(bar);
 }
 async function reconcile(actor) {
  index(actor);if(!active())return;
  await fan.reconcile(actor);await areas.reconcile(actor);
  for(const item of values(actor.items).filter(rewardKey)){
   const use=state(item).use;
   if(use?.status==='pending'){
    if(hasPublishedUse(item,use,game))await serializeReward(item.uuid,async()=>{const current=state(item).use;if(current?.status!=='pending'||current.nonce!==use.nonce||current.operation!==use.operation||!hasPublishedUse(item,current,game))return;if(use.operation==='release')await fan.clearAfterRelease(item);await item.update({[`flags.${MODULE_ID}.${PILGRIM_FLAG}.use`]:{...use,status:'done',reconciled:true},[`flags.${MODULE_ID}.${PILGRIM_FLAG}.lastUse.${use.operation}`]:use.startedAt,...use.operation!=='release'?{[`flags.${MODULE_ID}.${PILGRIM_FLAG}.spent`]:true}:{}});});
   }
  }
 }
 function register({Hooks,socket:channel}={}) {
  socket=channel;
  socket.register('pilgrim-use',async function(payload){try{return {ok:true,value:await execute(payload,game.users.get(this.socketdata.userId))};}catch(error){report(error);return {ok:false,error:playerError(error)};}});
  socket.register('pilgrim-rest',async function(payload){try{
   if(!active())throw new PilgrimError('请等待GM在线后使用。');
   const actor=await fromUuid(payload.actorUuid);
   if(actor?.uuid!==payload.actorUuid)throw new PilgrimError('请使用你持有的物品。');
   await resetDaily(actor,game.users.get(this.socketdata.userId));return {ok:true};
  }catch(error){report(error);return {ok:false,error:playerError(error)};}});
  globalThis.libWrapper?.register(MODULE_ID,'CONFIG.Item.documentClass.createDocuments',function(wrapped,data=[],operation={}){
   if(operation.parent)for(const row of data)prepareScarfChoices(row,operation.parent,game.items);
   return wrapped(data,operation);
  },'WRAPPER');
  for(const actor of values(game.actors))index(actor);
  for(const scene of values(game.scenes))for(const token of values(scene.tokens))if(token.actor?.isToken)index(token.actor);
  const on=(name,callback)=>Hooks.on(name,(...args)=>{
   try{return Promise.resolve(callback(...args)).catch(report);}catch(error){report(error);}
  });
  const restoreMedia=async actor=>{if(active())await reconcile(actor);await quietMedia(()=>media.reconcile(actor));};
  on('createChatMessage',async message=>{if(active()){await fan.handleStrike(message);await quietMedia(()=>media.strike(message));}});
  on('updateChatMessage',async message=>{if(active()){await fan.handleStrike(message);await quietMedia(()=>media.strike(message));}});
  on('renderChatMessageHTML',(message,html)=>{if(!message.isRoll&&!message.rolls?.length)controls(message.item,html?.[0]??html);});
  for(const event of ['renderItemSheetPF2e','renderItemSheetV2'])on(event,(app,html)=>controls(app.item??app.document,html?.[0]??html));
  on('createItem',async item=>{index(item.actor);if(item.actor&&cosmeticChange(item))await quietMedia(()=>media.reconcile(item.actor));});
  on('updateItem',async(item,changes)=>{
   if(!item.actor)return;index(item.actor);
   if(own(item)?.kind&&active())await areas.reconcile(item.actor);
   if(cosmeticChange(item))await quietMedia(()=>media.reconcile(item.actor));
  });
  on('deleteItem',async item=>{if(item.actor){index(item.actor);if(active()){await areas.deleted(item);if(rewardKey(item)==='fan')await fan.reconcile(item.actor);}await quietMedia(()=>media.deleted(item));if(cosmeticChange(item))await quietMedia(()=>media.reconcile(item.actor));}});
  for(const event of ['updateWorldTime','updateCombat','deleteCombat'])on(event,async()=>{if(active())for(const actor of tracked.values())await reconcile(actor);});
  on('pf2e.restForTheNight',async actor=>{
   if(active()||!game.users.activeGM)await resetDaily(actor,game.user);
   else {const response=await socket.executeAsUser('pilgrim-rest',game.users.activeGM.id,{actorUuid:actor.uuid});if(!response.ok)throw new PilgrimError(response.error);}
  });
  for(const event of ['updateUser','userConnected'])on(event,async()=>{if(active())for(const actor of tracked.values())await reconcile(actor);});
  on('canvasReady',async()=>{for(const token of values(canvas?.scene?.tokens))index(token.actor);if(active())for(const actor of tracked.values())await reconcile(actor);});
  on('sequencerEffectManagerReady',async()=>{for(const token of values(canvas?.scene?.tokens))index(token.actor);for(const actor of tracked.values())await restoreMedia(actor);});
  areas.register({Hooks,card});
  // Module initialization may finish after Sequencer's first ready event.
  for(const actor of tracked.values())restoreMedia(actor).catch(report);
 }
 return {enabled,register,request,execute,reconcile,card,fan,areas};
}
