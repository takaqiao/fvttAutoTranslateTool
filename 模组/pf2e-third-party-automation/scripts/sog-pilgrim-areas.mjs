import {MODULE_ID} from './rules.mjs';
import {EFFECTS,PILGRIM_FLAG,PilgrimError,values,rewardKey,serializeReward} from './sog-pilgrim-rules.mjs';

const EVENT='tpaSogPilgrimAreaEvent';
const TREE_TEXTURE='modules/pf2e-season-of-ghosts/assets/maps/other/treetops/tree-top01narchy.webp';
const own=document=>document?.flags?.[MODULE_ID]?.[PILGRIM_FLAG];
const flags=data=>({[MODULE_ID]:{[PILGRIM_FLAG]:data}});
const collections={MeasuredTemplate:'templates',Region:'regions',Tile:'tiles'};
const isArea=effect=>effect?.type==='effect'&&['tree','storm'].includes(own(effect)?.kind);

function eventKey(event) {
 const {token,movement,combat,round,turn,skipped}=event.data??{};
 if(event.name==='tokenTurnStart'&&!skipped&&combat?.uuid&&Number.isInteger(round)&&Number.isInteger(turn))return `turn:${token.uuid}:${combat.uuid}:${round}:${turn}`;
 if(event.name==='tokenMoveIn'&&movement?.id){
  const path=JSON.stringify([movement.origin,movement.destination,movement.passed?.waypoints]);
  return `move:${token.uuid}:${movement.id}:${path}`;
 }
 return null;
}

export function createPilgrimAreas({game,fromUuid,media,onError=console.error,canvas=globalThis.canvas}={}) {
 const active=()=>game.user?.id===game.users.activeGM?.id&&game.system?.id==='pf2e'&&['sog','sog-pilgrim-qa'].includes(game.world?.id);
 const ending=new Set();
 const quietMedia=async operation=>{try{await operation();}catch(error){onError(error);}};
 const currentScene=source=>{
  const scene=source?.parent;
  if(!scene||canvas?.scene!==scene||scene.tokens.get(source.id)!==source||!source.object?.center)throw new PilgrimError('请让GM打开角色所在场景。');
  return scene;
 };
 function validatePlacement({item,source,position,range}) {
  currentScene(source);
  if(source.actor?.uuid!==item.actor?.uuid||!Number.isFinite(position?.x)||!Number.isFinite(position?.y))throw new PilgrimError('请重新选择区域位置。');
  const distance=canvas.grid.measurePath([source.object.center,position]).distance;
  if(!Number.isFinite(distance)||distance>range)throw new PilgrimError('所选位置超出范围。');
 }
 async function pick({item,source}) {
  currentScene(source);
  if(!game.modules.get('sequencer')?.active||!globalThis.Sequencer?.Crosshair?.show)throw new PilgrimError('暂时无法放置区域，请联系GM。');
  const position=await Sequencer.Crosshair.show({
   t:'circle',distance:15,distanceMin:15,distanceMax:15,lockManualRotation:true,
   label:{text:'花瓣风暴'},icon:{texture:item.img},
   location:{obj:source.object,limitMaxRange:60,showRange:true},
  });
  if(!position)return null;
  validatePlacement({item,source,position,range:60});
  return {x:position.x,y:position.y};
 }
 async function removeResources(data) {
  const scene=game.scenes.get(data.sceneId);
  if(scene){
   for(const [type,collection] of Object.entries(collections)){
    const ids=(data.resourceRefs??[]).filter(ref=>ref.type===type).map(ref=>ref.id).filter(id=>{
     const document=scene[collection].get(id),record=own(document);
     return record?.source===data.source&&record.nonce===data.nonce;
    });
    if(ids.length)await scene.deleteEmbeddedDocuments(type,ids);
   }
  }
  const source=await fromUuid(data.source);
  if(source)await quietMedia(()=>media.clearArea(source,data.nonce));
 }
 async function restoreEquipment(item,data) {
  if(!item||!data.equippedBefore||item.actor?.items.get(item.id)!==item)return;
  const current=item.system.equipped;
  if(current.carryType==='dropped'&&current.handsHeld===0)await item.update({
   'system.equipped.carryType':data.equippedBefore.carryType,
   'system.equipped.handsHeld':data.equippedBefore.handsHeld,
  });
 }
 async function end(effect,{remove=true}={}) {
  const data=own(effect);if(!isArea(effect)||ending.has(data.nonce))return;
  ending.add(data.nonce);
  try{
   await removeResources(data);
   if(data.kind==='tree')await restoreEquipment(await fromUuid(data.source),data);
   if(remove&&effect.actor?.items.get(effect.id)===effect)await effect.actor.deleteEmbeddedDocuments('Item',[effect.id]);
  }finally{ending.delete(data.nonce);}
 }
 async function create({item,source,position,nonce,kind,card}) {
  if(!active())throw new PilgrimError('请等待GM在线后使用。');
  const scene=currentScene(source),resources=[];
  const effectSource=await fromUuid('Item.'+EFFECTS[kind]);
  if(effectSource?.type!=='effect')throw new PilgrimError('效果暂时不可用，请联系GM。');
  if(values(item.actor.items).some(effect=>isArea(effect)&&own(effect).source===item.uuid&&!effect.isExpired))throw new PilgrimError('此物品的效果仍在持续。');
  const data={kind,source:item.uuid,nonce,sceneId:scene.id,position:{x:position.x,y:position.y},resourceRefs:resources};
  if(kind==='tree')data.equippedBefore={carryType:item.system.equipped.carryType,handsHeld:item.system.equipped.handsHeld};
  const add=async(type,row)=>{
   const [document]=await scene.createEmbeddedDocuments(type,[{...row,flags:flags({kind,source:item.uuid,nonce})}]);
   if(!document)throw new PilgrimError('区域尚未放置，请联系GM核对。');
   resources.push({type,id:document.id,uuid:document.uuid});return document;
  };
  let effect;
  try{
   const template=await add('MeasuredTemplate',{t:'circle',x:position.x,y:position.y,distance:15,direction:0,user:game.user.id,borderColor:kind==='tree'?'#7da56d':'#db91b4',fillColor:kind==='tree'?'#7da56d':'#db91b4'});
   if(kind==='tree'){
    const size=scene.grid.size*15/scene.grid.distance;
    await add('Tile',{x:position.x,y:position.y,width:size,height:size,texture:{src:TREE_TEXTURE,anchorX:0.5,anchorY:0.5},hidden:false});
   }else{
    const radius=scene.grid.size*15/scene.grid.distance;
    await add('Region',{name:'花瓣风暴',color:'#db91b4',shapes:[{type:'circle',x:position.x,y:position.y,radius,gridBased:true}],behaviors:[{
     name:'花瓣风暴',type:'executeScript',system:{events:['tokenMoveIn','tokenTurnStart'],source:`if (game.user.id === game.users.activeGM?.id) Hooks.callAll('${EVENT}', event);`},
    }]});
   }
   const row=effectSource.toObject();delete row._id;
   row.flags={...row.flags,pf2e:{...row.flags?.pf2e,origin:{uuid:item.uuid,actor:item.actor.uuid,type:item.type}},[MODULE_ID]:{...row.flags?.[MODULE_ID],[PILGRIM_FLAG]:data}};
   row.system.start={value:game.time.worldTime,initiative:game.combat?.combatant?.initiative??null};
   [effect]=await item.actor.createEmbeddedDocuments('Item',[row]);
   if(!effect)throw new PilgrimError('效果尚未生效，请联系GM核对。');
   if(kind==='tree')await item.update({'system.equipped.carryType':'dropped','system.equipped.handsHeld':0});
   await card({item,nonce,gm:true,...kind==='tree'?{
    formula:'(3d8+8)[healing]',label:'余烬重生：15尺内所有生物恢复生命值。能看见杉树的不死生物与魔族获得惊惧1；保留更高的惊惧值，且无法降至1以下。结束条件由GM裁定。',
   }:{formula:'{1d10[slashing],1d10[vitality]}',label:'花瓣风暴：生物进入风暴或在其中开始回合时结算。',save:{type:'reflex',dc:23}}});
   await quietMedia(()=>media.area(item,{kind,scene,position,nonce,template,effect}));
   return {effectId:effect.id,templateId:template.id,...data};
  }catch(error){
   try{
    await removeResources(data);
    if(effect&&item.actor.items.get(effect.id)===effect)await item.actor.deleteEmbeddedDocuments('Item',[effect.id]);
    if(kind==='tree')await restoreEquipment(item,data);
   }catch(cleanupError){onError(cleanupError);}
   throw error;
  }
 }
 async function tree({item,source,target,nonce,card}) {
  const scene=currentScene(source);
  if(target?.parent!==scene||scene.tokens.get(target.id)!==target||!target.object?.center)throw new PilgrimError('请重新选择触及内的尸体。');
  return create({item,source,position:target.object.center,nonce,kind:'tree',card});
 }
 async function storm({item,source,position,nonce,card}) {
  if(!position)return null;
  validatePlacement({item,source,position,range:60});
  return create({item,source,position,nonce,kind:'storm',card});
 }
 async function restore({item,user}) {
  if(!active()||!item.actor.testUserPermission(user,'OWNER'))throw new PilgrimError('请使用你持有的物品。');
  const effect=values(item.actor.items).find(effect=>isArea(effect)&&own(effect).kind==='tree'&&own(effect).source===item.uuid);
  if(!effect)throw new PilgrimError('大杉枝已经处于武器形态。');
  await end(effect);return {status:'done'};
 }
 async function reconcile(actor) {
  if(!active())return;
  for(const effect of values(actor?.items).filter(isArea)){
   const source=await fromUuid(own(effect).source);
   if(effect.isExpired===true||effect.remainingDuration?.expired===true||!source||source.actor?.items.get(source.id)!==source)await serializeReward(own(effect).source,()=>end(effect));
  }
 }
 async function deleted(item) {
  if(!active())return;
  if(isArea(item))await serializeReward(own(item).source,()=>end(item,{remove:false}));
  else if(rewardKey(item))for(const effect of values(item.actor?.items).filter(effect=>isArea(effect)&&own(effect).source===item.uuid))await serializeReward(item.uuid,()=>end(effect));
 }
 function register({Hooks,card}) {
  Hooks.on(EVENT,event=>Promise.resolve(handleEvent(event,card)).catch(error=>{onError(error);globalThis.ui?.notifications?.warn('花瓣风暴尚未结算，请GM使用伤害卡处理。');}));
 }
 async function handleEvent(event,card) {
  if(!active())return;
  const region=event?.region,data=own(region),token=event?.data?.token;
  if(data?.kind!=='storm'||!data.source||!data.nonce||!token?.actor||region.parent?.regions.get(region.id)!==region||token.parent!==region.parent||token.parent.tokens.get(token.id)!==token)return;
  if(!token.actor.isOfType?.('character','npc'))return;
  const key=eventKey(event);if(!key)return;
  await serializeReward(data.source,async()=>{
   const item=await fromUuid(data.source);
   if(rewardKey(item)!=='hairpin'||item.actor?.items.get(item.id)!==item)return;
   const effect=values(item.actor.items).find(effect=>isArea(effect)&&own(effect).kind==='storm'&&own(effect).nonce===data.nonce&&own(effect).source===data.source);
   if(!effect||effect.isExpired||effect.remainingDuration?.expired)return;
   if(!(own(effect).resourceRefs??[]).some(ref=>ref.type==='Region'&&ref.id===region.id))return;
   const nonce=`${data.nonce}:${key}`;
   if(values(game.messages).some(message=>own(message)?.source===item.uuid&&own(message).nonce===nonce))return;
   await card({item,formula:'{1d10[slashing],1d10[vitality]}',label:'花瓣风暴',nonce,target:token,save:{type:'reflex',dc:23},gm:true});
  });
 }
 return {pick,validatePlacement,tree,storm,restore,reconcile,deleted,register};
}
