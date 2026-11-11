import {MODULE_ID} from './rules.mjs';
import {showNativeChoice} from './native-context.mjs';
import {EFFECTS,PILGRIM_FLAG,PILGRIM_SYNC,PilgrimError,activationOf,durationText,escapeHTML,values,rewardKey,usableReward,scarfBranch,serializeReward,sightBlocked,tokenDistance} from './sog-pilgrim-rules.mjs';
import {SOG_PILGRIM_WORLDS} from './world-gates.mjs';

const names={release:'灵魂连携 ◆◆',shift:'变型 ◆',tree:'余烬重生 ◆◆◆',restore:'恢复武器 ◇',scarf:'启动围巾 ◆',storm:'花瓣风暴 ◆◆'};
const branches={ghost:'幽冥',astral:'星魂',ward:'护佑'};
const frequencies={release:'PT1H',tree:'day',scarf:'day',storm:'day'};
const HAUNT='wonton-ghost-scarf:haunt-context';
const synced={[PILGRIM_SYNC]:true};
const state=item=>item.flags?.[MODULE_ID]?.[PILGRIM_FLAG]??{};
const own=doc=>doc.flags?.[MODULE_ID]?.[PILGRIM_FLAG];
const unusable=item=>item.type==='equipment'?'请先佩戴并投资该物品。':'请先持用该武器。';
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
 const marked=message=>own(message)?.source===item.uuid&&own(message).nonce===use.nonce;
 // The storm's damage cards come later, one per creature; its notice is what the activation published.
 if(use.operation==='storm')return values(game.messages).some(message=>marked(message)&&own(message).announce===true);
 return ['release','tree'].includes(use.operation)&&values(game.messages).some(message=>message.isDamageRoll===true&&marked(message)&&own(message).generated===true);
}
export function activationOperations(key) { return ({fan:['release'],branch:['shift','tree','restore'],scarf:['scarf'],hairpin:['storm']})[key]??[]; }
export function assertAvailable(current,operation,now) {
 if(current.use?.status==='pending')throw new PilgrimError('上次启动尚待核对，请联系GM。');
 if(operation==='release'&&Number.isFinite(current.lastUse?.release)&&now-current.lastUse.release<3600)throw new PilgrimError('灵魂连携每小时只能使用一次。');
 if(['tree','scarf','storm'].includes(operation)&&current.spent)throw new PilgrimError('今日已经启动；完成每日休息后恢复。');
}
/** Button text and, when the activation cannot start now, the reason. */
export function operationStatus(item,operation,{now,leaves=0,tree=false}) {
 const current=state(item),label=names[operation],blocked=(suffix,reason)=>({label:suffix?`${label} · ${suffix}`:label,reason});
 if(operation==='restore')return tree?{label}:blocked('','大杉枝已经处于武器形态。');
 if(current.use?.status==='pending')return blocked('待核对','上次启动尚待核对，请联系GM。');
 if(!usableReward(item))return blocked('',unusable(item));
 if(operation==='shift'&&tree)return blocked('','请先将大杉枝恢复为武器。');
 if(operation==='release'){
  const wait=Number.isFinite(current.lastUse?.release)?current.lastUse.release+3600-now:0;
  if(wait>0)return blocked(`冷却 ${Math.ceil(wait/60)} 分钟`,'灵魂连携每小时只能使用一次。');
  return leaves>=3?{label:`${label} · 金叶 ${leaves}/3`}:blocked(`金叶 ${leaves}/3`,'需要三片亮起的金叶。');
 }
 if(['tree','scarf','storm'].includes(operation)&&current.spent)return blocked('今日已用','今日已经启动；完成每日休息后恢复。');
 return {label};
}
export function scarfWeaponChoices(weapons) {
 const held=weapon=>weapon.system.equipped?.carryType==='held';
 return [...weapons.filter(held),...weapons.filter(weapon=>!held(weapon))].map(weapon=>({value:weapon.id,label:`${weapon.name}${held(weapon)?'（手持）':''} · ${branches[scarfBranch(weapon)]}`}));
}

export async function createPilgrimRewards({game,fromUuid,choose,onError=console.error,canvas=globalThis.canvas}={}) {
 const enabled=game.system?.id==='pf2e'&&SOG_PILGRIM_WORLDS.has(game.world?.id);
 if(!enabled)return {register(){},enabled:false};
 const [{createPilgrimFan},{createPilgrimAreas},{createPilgrimMedia},{createPilgrimShift}]=await Promise.all([import('./sog-pilgrim-fan.mjs'),import('./sog-pilgrim-areas.mjs'),import('./sog-pilgrim-media.mjs'),import('./sog-pilgrim-shift.mjs')]);
 const media=createPilgrimMedia({game,canvas,fromUuid,onError});
 const fan=createPilgrimFan({game,fromUuid,media,onError,canvas});
 const areas=createPilgrimAreas({game,fromUuid,media,onError,canvas});
 const shift=createPilgrimShift({game,fromUuid});
 const tracked=new Map(),bars=new Set();let socket;
 const active=()=>game.user?.id===game.users.activeGM?.id;
 const report=error=>{onError(error);};
 const quiet=async(fn)=>{try{await fn();}catch(error){report(error);}};
 const index=actor=>{if(!actor)return;const relevant=values(actor.items).filter(i=>rewardKey(i));if(relevant.length)tracked.set(actor.uuid,actor);else tracked.delete(actor.uuid);};
 const cosmeticChange=item=>['fan','scarf'].includes(rewardKey(item))||own(item)?.kind==='leaves';
 const hasTree=item=>values(item.actor?.items).some(effect=>own(effect)?.kind==='tree'&&own(effect).source===item.uuid);
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
  if(!source||!target||target.parent!==source.parent||target.parent.tokens.get(target.id)!==target)throw new PilgrimError('请在角色所在的场景中选择目标。');
  const distance=tokenDistance(source,target);
  if(!Number.isFinite(distance)||distance>range)throw new PilgrimError('目标超出范围。');
  // Walls are known only where the scene is drawn; without that the requester's own check stands.
  if(visible&&(source.actor.canSee===false||source.actor.hasCondition?.('blinded')||sightBlocked(canvas,source,target)===true))throw new PilgrimError('请选择角色能看见的目标。');
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
  const message=await roll.toMessage({speaker,flavor,flags,...gm?{whisper:ChatMessage.getWhisperRecipients('GM').map(u=>u.id)}:{}},{messageMode:gm?'gm':'public'});
  if(!message?.id)throw new PilgrimError('启动尚未完成，请联系GM核对。');
  return message;
 }
 /** A public notice of what an activation did, without any roll. */
 async function announce({item,nonce,text}) {
  const content=`<div class="sog-pilgrim-notice"><p><strong>${escapeHTML(item.actor.name)}</strong> · ${escapeHTML(item.name)}</p><p>${escapeHTML(text)}</p></div>`;
  const message=await ChatMessage.create({speaker:ChatMessage.getSpeaker({actor:item.actor}),content,flags:{[MODULE_ID]:{[PILGRIM_FLAG]:{announce:true,source:item.uuid,nonce}}}},{messageMode:'public'});
  if(!message?.id)throw new PilgrimError('启动说明未能发布，请联系GM核对。');
  return message;
 }
 async function applyScarf({item,weapon,nonce,scene}) {
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
  await quiet(()=>media.scarf(item,item.actor.rollOptions?.all?.['wonton-ghost-scarf:lit']===true));
  await quiet(()=>media.scarfCast(item,{scene}));
  await quiet(()=>announce({item,nonce,text:`围巾缠上${weapon.name}，获得${branches[branch]}效果，${durationText(data.system.duration)}。`}));
  return {effectId:created[0].id};
 }
 async function complete(item,use) {
  if(use.operation==='release')await fan.clearAfterRelease(item);
  await item.update({[`flags.${MODULE_ID}.${PILGRIM_FLAG}.use`]:{...use,status:'done',reconciled:true},[`flags.${MODULE_ID}.${PILGRIM_FLAG}.lastUse.${use.operation}`]:use.startedAt,...use.operation!=='release'?{[`flags.${MODULE_ID}.${PILGRIM_FLAG}.spent`]:true}:{}});
 }
 async function execute(payload,user) {
  if(!active())throw new PilgrimError('请等待GM在线后使用。');
  const item=await liveItem(payload.itemUuid,user),key=rewardKey(item),operation=payload.operation;
  if(!activationOperations(key).includes(operation))throw new PilgrimError('该物品没有此项启动。');
  return serializeReward(item.uuid,async()=>{
   if(operation==='restore')return areas.restore({item,user});
   if(!usableReward(item))throw new PilgrimError(unusable(item));
   assertAvailable(state(item),operation,game.time.worldTime);
   const nonce=payload.nonce;
   if(typeof nonce!=='string'||!/^[A-Za-z0-9_-]{8,80}$/.test(nonce))throw new PilgrimError('请重新点击启动。');
   if(operation==='shift'){
    await shift.apply({item,formUuid:payload.formUuid});
    const form=await fromUuid(payload.formUuid);
    await quiet(()=>announce({item,nonce,text:`大杉枝化为${form?.name??'新的武器'}的形态，直到再次变型。`}));
    return {status:'done'};
   }
   const origin=payload.sourceUuid?sourceToken(item,payload.sourceUuid):null;
   const target=payload.targetUuid?await fromUuid(payload.targetUuid):null;
   let weapon;
   if(operation==='release'){
    within(origin,target,30,{visible:true});
    if(target.hidden&&!user.isGM)throw new PilgrimError('请选择你能看见的目标。');
    if(!target.actor?.isOfType('character','npc')||target.actor.isDead||target.actor.traits?.has('construct')&&!target.actor.traits?.has('undead'))throw new PilgrimError('请选择活物或不死生物。');
    if(fan.validLeaves(item).length!==3)throw new PilgrimError('需要三片亮起的金叶。');
   }else if(operation==='scarf'){
    weapon=item.actor.items.get(payload.weaponId);
    if(weapon?.type!=='weapon'||weapon.system.equipped?.carryType==='dropped')throw new PilgrimError('请选择你携带的武器。');
   }else if(operation==='tree'){
    within(origin,target,item.actor.getReach?.({action:'interact'})??5);
    const traits=target.actor?.traits;
    if(!(traits?.has('undead')||traits?.has('fiend'))||!target.actor.isDead)throw new PilgrimError('请选择触及内不死生物或魔族的尸体。');
   }else if(operation==='storm')areas.validatePlacement({item,source:origin,position:payload.position,range:60});
   const use={nonce,operation,status:'pending',startedAt:game.time.worldTime};
   await item.update({[`flags.${MODULE_ID}.${PILGRIM_FLAG}.use`]:use});
   // Sequencer delivers the GM's broadcast only to the GM's scene; otherwise the requester plays it.
   const shown=!!origin&&media.shows(origin.parent);
   let result;
   try{
    if(operation==='release')result=await fan.release({item,source:origin,target,nonce,card});
    else if(operation==='scarf')result=await applyScarf({item,weapon,nonce,scene:origin?.parent});
    else result=await areas[operation]({item,source:origin,target,position:payload.position,nonce,card,announce});
    if(result===null){await item.update({[`flags.${MODULE_ID}.${PILGRIM_FLAG}.use`]:{nonce,operation,status:'cancelled'}});return {status:'cancelled'};}
    await item.update({[`flags.${MODULE_ID}.${PILGRIM_FLAG}.use`]:{nonce,operation,status:'done',...result},[`flags.${MODULE_ID}.${PILGRIM_FLAG}.lastUse.${operation}`]:game.time.worldTime,...operation!=='release'?{[`flags.${MODULE_ID}.${PILGRIM_FLAG}.spent`]:true}:{}});
   }catch(error){
    // A published result already happened; anything earlier must not leave the reward locked as pending.
    if(!hasPublishedUse(item,use,game)){await quiet(()=>item.update({[`flags.${MODULE_ID}.${PILGRIM_FLAG}.use`]:{...use,status:'failed'}}));throw error;}
    report(error);await complete(item,use);
   }
   return {status:'done',...result,shown};
  });
 }
 async function replay(item,operation,value,{token,target}) {
  if(operation==='release')return media.release(item,target,{requester:true});
  if(operation==='scarf')return media.scarfCast(item,{scene:token?.parent,requester:true});
  if(!['tree','storm'].includes(operation))return;
  const scene=game.scenes.get(value.sceneId);
  return media.area(item,{kind:operation,scene,position:value.position,nonce:value.nonce,effect:item.actor.items.get(value.effectId),template:scene?.templates.get(value.templateId),requester:true});
 }
 async function request(item,operation,{position}={}) {
  if(!game.users.activeGM)throw new PilgrimError('请等待GM在线后使用。');
  const actor=item.actor;
  if(!actor?.isOwner)throw new PilgrimError('请使用你持有的物品。');
  if(operation!=='restore'&&!usableReward(item))throw new PilgrimError(unusable(item));
  if(operation!=='restore')assertAvailable(state(item),operation,game.time.worldTime);
  const token=values(canvas?.tokens?.controlled).find(t=>t.actor?.uuid===actor.uuid)?.document??values(canvas?.scene?.tokens).find(t=>t.actor?.uuid===actor.uuid&&!t.hidden);
  const targets=values(game.user.targets).map(t=>t.document??t);
  const payload={itemUuid:item.uuid,operation,nonce:foundry.utils.randomID(),sourceUuid:token?.uuid};
  if(['release','tree'].includes(operation)){
   if(targets.length!==1)throw new PilgrimError(operation==='tree'?'请先选中一具不死生物或魔族的尸体。':'请先选中一个目标。');
   if(targets[0].object?.isVisible===false)throw new PilgrimError('请选择你能看见的目标。');
   // This client has the scene drawn, so it judges walls for the GM.
   if(operation==='release'&&token&&sightBlocked(canvas,token,targets[0])===true)throw new PilgrimError('请选择角色能看见的目标。');
   payload.targetUuid=targets[0].uuid;
  }else if(operation==='shift'){
   payload.formUuid=await shift.select(item);
   if(!payload.formUuid)return {status:'cancelled'};
  }else if(operation==='scarf'){
   const weapons=values(actor.items).filter(i=>i.type==='weapon'&&i.system.equipped?.carryType!=='dropped');
   if(!weapons.length)throw new PilgrimError('需要一件携带的武器。');
   payload.weaponId=await showNativeChoice({title:'选择围巾缠绕的武器',prompt:'选择围巾缠绕的武器，每项后面是武器会获得的效果。',choices:scarfWeaponChoices(weapons)});
   if(!payload.weaponId)return {status:'cancelled'};
  }else if(operation==='storm'){
   if(!token)throw new PilgrimError('请在场景中选择使用物品的角色。');
   payload.position=position??await areas.pick({item,source:token});
   if(!payload.position)return {status:'cancelled'};
  }
  const response=await socket.executeAsUser('pilgrim-use',game.users.activeGM.id,payload);
  if(!response.ok)throw new PilgrimError(response.error);
  if(response.value?.shown===false)await replay(item,operation,response.value,{token,target:targets[0]});
  return response.value;
 }
 async function attempt(item,operation,options) {
  try{return await request(item,operation,options);}catch(error){report(error);ui.notifications.warn(playerError(error));return null;}
 }
 async function clearPending(item) {
  if(!game.user.isGM)throw new PilgrimError('只有GM可以清除待核对。');
  return serializeReward(item.uuid,async()=>{
   const use=state(item).use;
   if(use?.status==='pending')await item.update({[`flags.${MODULE_ID}.${PILGRIM_FLAG}.use`]:{...use,status:'cleared'}});
  });
 }
 function leafCount(item) {
  if(rewardKey(item)!=='fan')return 0;
  try{return fan.validLeaves(item).length;}catch{return 0;}
 }
 function paint(entry) {
  const {bar,item,bound}=entry,tree=hasTree(item),now=game.time.worldTime,leaves=leafCount(item);
  const operations=bound?[bound]:activationOperations(rewardKey(item)).filter(operation=>operation==='restore'?tree:operation==='tree'?!tree:true);
  const button=(label,click)=>{
   const node=document.createElement('button');node.type='button';node.textContent=label;
   node.addEventListener('click',async event=>{event.preventDefault();event.stopPropagation();node.disabled=true;try{await click();}finally{paint(entry);}});
   return node;
  };
  const nodes=operations.map(operation=>{
   const status=operationStatus(item,operation,{now,leaves,tree}),wrap=document.createElement('span'),node=button(status.label,()=>attempt(item,operation));
   // A disabled button gets no pointer events, so the reason goes on its wrapper.
   node.disabled=!!status.reason;if(status.reason)wrap.dataset.tooltip=status.reason;
   wrap.append(node);return wrap;
  });
  if(game.user.isGM&&state(item).use?.status==='pending')nodes.push(button('清除待核对',async()=>{try{await clearPending(item);}catch(error){report(error);ui.notifications.warn(playerError(error));}}));
  bar.replaceChildren(...nodes);
 }
 function controls(item,element,bound=null) {
  if(!rewardKey(item)||!item.actor?.isOwner||!element?.querySelector||element.querySelector('.sog-pilgrim-actions'))return;
  const bar=document.createElement('div');bar.className='sog-pilgrim-actions';
  const entry={bar,item,bound};bars.add(entry);paint(entry);
  (element.querySelector('.message-content')??element.querySelector('.window-content')??element).append(bar);
 }
 function refreshControls() {
  for(const entry of bars){if(entry.bar.isConnected)paint(entry);else bars.delete(entry);}
 }
 /** The client that posted an activation's own card starts it, as its TPA button would. */
 async function activated(message) {
  const binding=activationOf(message?.item);
  if(!binding||message.author?.id!==game.user?.id||message.flags?.pf2e?.origin?.uuid!==message.item.uuid||!/\baction-card\b/.test(message.content??''))return;
  await attempt(binding.reward,binding.operation);
 }
 /** An area placed from the hairpin's description becomes the storm at the same spot. */
 async function placed(region,userId) {
  const scene=region?.parent,shape=region?.shapes?.[0],origin=region?.flags?.pf2e?.origin?.uuid;
  if(userId!==game.user?.id||own(region)||typeof origin!=='string'||region.flags.pf2e.areaShape!=='burst'||region.shapes.length!==1||shape?.type!=='circle'||Math.abs(shape.radius-scene.grid.size*15/scene.grid.distance)>1)return;
  const source=await fromUuid(origin),binding=activationOf(source);
  const item=rewardKey(source)==='hairpin'?source:binding?.operation==='storm'?binding.reward:null;
  if(!item)return;
  const result=await attempt(item,'storm',{position:{x:shape.x,y:shape.y}});
  if(result?.status==='done')await region.delete();
 }
 function frequencyOf(item,operation) {
  const per=frequencies[operation];if(!per)return null;
  const current=state(item),now=game.time.worldTime;
  const used=current.use?.status==='pending'||(operation==='release'?Number.isFinite(current.lastUse?.release)&&now-current.lastUse.release<3600:current.spent===true);
  return {max:1,per,value:used?0:1};
 }
 /** pf2e-item-activations parses the frequency text on its own; the generated actions mirror TPA's state instead. */
 async function mirror(actor) {
  if(!active())return;
  for(const action of values(actor?.items)){
   const binding=activationOf(action),frequency=binding&&frequencyOf(binding.reward,binding.operation),current=action.system?.frequency;
   if(frequency&&(current?.max!==frequency.max||current.per!==frequency.per||current.value!==frequency.value))await action.update({'system.frequency':frequency},synced);
  }
 }
 /** The scarf's haunt bonus follows an encounter that includes a haunt. */
 async function haunted(combat,on) {
  if(!active())return;
  const actors=[...new Set(values(combat?.combatants).map(combatant=>combatant.actor).filter(Boolean))];
  if(on&&!actors.some(actor=>actor.traits?.has?.('haunt')))return;
  for(const actor of actors)for(const scarf of values(actor.items).filter(item=>rewardKey(item)==='scarf'&&(!on||usableReward(item))))
   if((actor.rollOptions?.all?.[HAUNT]===true)!==on)await actor.toggleRollOption('all',HAUNT,scarf.id,on);
 }
 async function reconcile(actor) {
  index(actor);if(!active())return;
  await fan.reconcile(actor);await areas.reconcile(actor);
  for(const item of values(actor.items).filter(rewardKey)){
   const use=state(item).use;
   if(use?.status==='pending'&&hasPublishedUse(item,use,game))await serializeReward(item.uuid,async()=>{const current=state(item).use;if(current?.status!=='pending'||current.nonce!==use.nonce||current.operation!==use.operation||!hasPublishedUse(item,current,game))return;await complete(item,use);});
  }
  await mirror(actor);
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
  const restoreMedia=async actor=>{if(active())await reconcile(actor);await quiet(()=>media.reconcile(actor));};
  on('createChatMessage',async message=>{if(active()){await fan.handleStrike(message);await quiet(()=>media.strike(message));}await activated(message);});
  on('updateChatMessage',async message=>{if(active()){await fan.handleStrike(message);await quiet(()=>media.strike(message));}});
  // PF2e deletes the original card right before posting its reroll.
  on('deleteChatMessage',message=>fan.recordDeleted(message));
  on('renderChatMessageHTML',(message,html)=>{
   if(message.isRoll||message.rolls?.length)return;
   const binding=activationOf(message.item);
   if(binding)controls(binding.reward,html?.[0]??html,binding.operation);else controls(message.item,html?.[0]??html);
  });
  for(const event of ['renderItemSheetPF2e','renderItemSheetV2'])on(event,(app,html)=>controls(app.item??app.document,html?.[0]??html));
  on('createRegion',(region,_options,userId)=>placed(region,userId));
  on('createItem',async item=>{refreshControls();index(item.actor);if(item.actor&&activationOf(item))await mirror(item.actor);if(item.actor&&cosmeticChange(item))await quiet(()=>media.reconcile(item.actor));});
  on('updateItem',async(item,changes,options,userId)=>{
   refreshControls();
   if(!item.actor)return;index(item.actor);
   const mine=options?.[PILGRIM_SYNC]===true;
   if(own(item)?.kind&&active())await areas.reconcile(item.actor);
   if(own(item)?.kind==='leaves'&&active()&&!mine&&game.users.get(userId)?.isGM&&Number.isInteger(changes?.system?.badge?.value))await fan.adjust(item,changes.system.badge.value);
   if(!mine&&(rewardKey(item)||activationOf(item)))await mirror(item.actor);
   if(cosmeticChange(item))await quiet(()=>media.reconcile(item.actor));
  });
  on('deleteItem',async(item,options,userId)=>{
   refreshControls();
   if(!item.actor)return;index(item.actor);
   if(active()){
    await areas.deleted(item);if(rewardKey(item)==='fan')await fan.reconcile(item.actor);
    if(own(item)?.kind==='leaves'&&options?.[PILGRIM_SYNC]!==true&&game.users.get(userId)?.isGM)await fan.adjust(item,0);
   }
   await quiet(()=>media.deleted(item));if(cosmeticChange(item))await quiet(()=>media.reconcile(item.actor));
  });
  for(const event of ['updateWorldTime','updateCombat','deleteCombat'])on(event,async()=>{refreshControls();if(active())for(const actor of tracked.values())await reconcile(actor);});
  on('updateCombat',async(combat,changes)=>{if(changes?.round===1)await haunted(combat,true);});
  on('createCombatant',async combatant=>{if(combatant.parent?.started&&combatant.actor?.traits?.has?.('haunt'))await haunted(combatant.parent,true);});
  on('deleteCombat',combat=>haunted(combat,false));
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
 return {enabled,register,request,execute,reconcile,card,clearPending,fan,areas};
}
