import {MODULE_ID} from './rules.mjs';
import {PILGRIM_FLAG,rewardKey,values} from './sog-pilgrim-rules.mjs';

export const PILGRIM_MEDIA = Object.freeze({
 leaf:`modules/${MODULE_ID}/assets/pilgrim-golden-leaf.png`,
 heal:'modules/jb2a_patreon/Library/Generic/Healing/HealingAbility_01_Yellow_400x400.webm',
 leaves:'modules/jb2a_patreon/Library/Generic/Nature/SwirlingLeavesOutburst_01_01_Regular_GreenOrange_400x400.webm',
 storm:'modules/jb2a_patreon/Library/Generic/Nature/SwirlingLeavesLoop02_01_Regular_Pink_400x400.webm',
 petals:'modules/eskie-effects/assets/Nature/Flower/Particle/Flower_Particle_01_Pink.webm',
 club:'modules/jb2a_patreon/Library/Generic/Weapon_Attacks/Melee/Group02/MeleeAttack02_Club01_01_800x600.webm',
 sword:'modules/jb2a_patreon/Library/Generic/Weapon_Attacks/Melee/Sword01_01_Regular_White_800x600.webm',
 swordSound:'modules/psfx-patreon/library/weapon-attacks/sword/v1/sword-001-00.ogg',
 releaseSound:'modules/psfx-patreon/library/1st-level-spells/cure-wounds/v1/cure-wounds-00.ogg',
 treeSound:'modules/psfx-patreon/library/1st-level-spells/entangle/vines/v1/entangle-intro.ogg',
 stormSound:'modules/psfx-patreon/library/cantrips/gust/v1/gust-001.ogg',
 natureSound:'modules/soundfxlibrary/Combat/Single/Spell Whoosh/spell-whoosh-1.mp3',
});
export function createPilgrimMedia({game,canvas=globalThis.canvas,fromUuid=globalThis.fromUuid,onError=console.error}={}) {
 const strikes=new Set(),displays=new Map();
 const leader=()=>game.user?.id===game.users.activeGM?.id;
 const localReady=()=>game.modules.get('sequencer')?.active&&typeof globalThis.Sequence==='function'&&canvas?.ready&&!!canvas.scene;
 const ready=()=>leader()&&localReady();
 const tokens=item=>values(canvas?.scene?.tokens).filter(t=>t.parent===canvas.scene&&t.actor?.uuid===item.actor?.uuid&&t.object);
 const name=(item,kind,suffix='')=>`${MODULE_ID}.pilgrim.${item.uuid}.${kind}${suffix}`;
 async function end(item,kind,suffix='',local=false) {
  if(!(local?localReady():ready()))return;
  const label=name(item,kind,suffix),sceneId=canvas.scene.id;
  // Sequencer 4.2.3 ignores sceneId when filtering visible effects.
  const effects=Sequencer.EffectManager.getEffects({name:label}).filter(effect=>effect.data.name===label&&effect.data.sceneId===sceneId).map(effect=>effect.id);
  await Sequencer.EffectManager.endEffects({effects,sceneId},!local);
 }
 function refresh(item,kind,render) {
  const key=name(item,kind),scene=canvas?.scene;
  const next=(displays.get(key)??Promise.resolve()).catch(()=>{}).then(async()=>{
   if(!localReady()||canvas.scene!==scene)return;
   await end(item,kind,'',true);
   if(localReady()&&canvas.scene===scene&&item.actor?.items.get(item.id)===item)await render();
  });
  displays.set(key,next);
  return next.finally(()=>{if(displays.get(key)===next)displays.delete(key);});
 }
 const leafEffect=item=>values(item.actor?.items).find(effect=>effect.type==='effect'&&effect.flags?.[MODULE_ID]?.[PILGRIM_FLAG]?.kind==='leaves'&&effect.flags[MODULE_ID][PILGRIM_FLAG].source===item.uuid&&effect.isExpired!==true&&effect.remainingDuration?.expired!==true);
 const sound=(sequence,file,volume=0.25)=>sequence.sound().file(file).volume(volume).fadeOutAudio(200);
 async function leaves(item,count) {
  return refresh(item,'leaves',async()=>{
   if(count<1)return;
   const sequence=new Sequence(),effect=leafEffect(item),tied=[item.uuid,...(effect?[effect.uuid]:[])];
   // Rebuild token decorations locally; saved effects can restore an obsolete count or light state.
   for(const token of tokens(item))for(let index=0;index<Math.min(count,3);index++)sequence.effect().file(PILGRIM_MEDIA.leaf).name(name(item,'leaves')).attachTo(token.object,{bindAlpha:true,bindVisibility:true}).tieToDocuments(tied).scaleToObject(0.25).spriteOffset({x:(index-(count-1)/2)*0.26,y:-0.65},{gridUnits:true}).opacity(0.9).persist().temporary().fadeIn(120).fadeOut(120);
   await sequence.play({local:true});
  });
 }
 async function scarf(item,lit) {
  return refresh(item,'scarf',async()=>{
   if(!lit)return;
   const sequence=new Sequence();
   for(const token of tokens(item))sequence.effect().name(name(item,'scarf')).attachTo(token.object,{bindAlpha:true,bindVisibility:true}).tieToDocuments(item.uuid).shape('circle',{radius:0.55,gridUnits:true,fillColor:'#e5edf5',fillAlpha:0.08,lineColor:'#e4ecf5',lineSize:1.5}).persist().temporary().opacity(0.45).loopProperty('alphaFilter','alpha',{from:0.3,to:0.6,duration:2400,pingPong:true}).fadeIn(200).fadeOut(200);
   await sequence.play({local:true});
  });
 }
 async function release(item,target) {
  if(!ready()||target.parent!==canvas.scene||!target.object)return;
  const sequence=new Sequence();
  sequence.effect().file(PILGRIM_MEDIA.heal).atLocation(target.object).scaleToObject(1.25).opacity(0.75);
  sound(sequence,PILGRIM_MEDIA.releaseSound,0.22);await sequence.play();
 }
 async function area(item,{kind,scene,position,nonce,effect,template}) {
  if(!ready()||scene!==canvas.scene)return;
  if(kind==='storm'&&(!effect?.uuid||!template?.uuid))throw new Error('花瓣风暴缺少绑定的效果或区域文档。');
  const sequence=new Sequence();
  if(kind==='storm'){
   const mask=new PIXI.Circle(position.x,position.y,scene.grid.size*15/scene.grid.distance),diameter=30/scene.grid.distance;
   for(const [file,scale,opacity]of [[PILGRIM_MEDIA.storm,1.17,0.65],[PILGRIM_MEDIA.petals,1.2,0.8]])sequence.effect().file(file).name(name(item,'area',nonce)).atLocation(position).tieToDocuments([item.uuid,effect.uuid,template.uuid]).size(diameter*scale,{gridUnits:true}).mask(mask).opacity(opacity).persist().fadeIn(300).fadeOut(300);
  }
  else sequence.effect().file(PILGRIM_MEDIA.leaves).atLocation(position).size(2,{gridUnits:true}).opacity(0.8);
  sound(sequence,kind==='storm'?PILGRIM_MEDIA.stormSound:PILGRIM_MEDIA.treeSound,kind==='storm'?0.46:0.5);await sequence.play();
 }
 async function clearArea(item,nonce){await end(item,'area',nonce);}
 function nativeStrike(message) {
  const item=message?.item,actor=message?.actor,pf=message?.flags?.pf2e,context=pf?.context,roll=message?.rolls?.[0];
  if(!ready()||!message?.id||game.messages?.get(message.id)!==message||!item||!actor||item.actor!==actor)return false;
  const owned=actor.items?.get(item.id);
  if(!owned||owned.actor!==actor||owned.uuid!==item.uuid||rewardKey(owned)!=='branch')return false;
  // PF2e prepares alternate usage Items outside the actor's embedded collection.
  if(owned!==item&&!values(actor.system?.actions).some(action=>[action,...action.altUsages??[]].some(usage=>usage.type==='strike'&&usage.item===item)))return false;
  const degree=context?.outcome==='success'?2:context?.outcome==='criticalSuccess'?3:null;
  return message.isCheckRoll===true&&context?.type==='attack-roll'&&pf.origin?.uuid===item.uuid&&pf.origin?.actor===actor.uuid&&degree!==null&&roll?._evaluated===true&&Number.isFinite(roll.total)&&roll.options?.degreeOfSuccess===degree;
 }
 async function existingAttack(item) {
  const api=globalThis.triggerAnimations?.api;
  if(typeof api?.matchTrigger!=='function')return false;
  const names=[item.uuid,item.slug,item.system.baseItem,item.system.group].filter(Boolean).map(value=>`attack:${value}`).join(',');
  try{return !!await api.matchTrigger(names);}catch(error){onError(error);return true;}
 }
 function strikeSourceUuid(message) {
  const context=message.flags.pf2e.context,origin=context.origin?.token;
  const tokenUuid=value=>typeof value==='string'&&/^Scene\.[A-Za-z0-9]+\.Token\.[A-Za-z0-9]+$/.test(value);
  const id=value=>typeof value==='string'&&/^[A-Za-z0-9]+$/.test(value);
  if(origin!==undefined&&origin!==null)return tokenUuid(origin)?origin:null;
  const token=context.token??message.speaker?.token;
  if(tokenUuid(token))return token;
  return id(token)&&id(message.speaker?.scene)?`Scene.${message.speaker.scene}.Token.${token}`:null;
 }
 async function strike(message) {
  if(!nativeStrike(message)||message.flags?.[MODULE_ID]?.pilgrimStrike||strikes.has(message.id))return;
  strikes.add(message.id);
  try{
   const context=message.flags.pf2e.context,item=message.item;
   const sourceUuid=strikeSourceUuid(message);
   if(!sourceUuid||!context.target?.token)return;
   const source=await fromUuid(sourceUuid),target=await fromUuid(context.target.token);
   const currentToken=token=>token?.parent===canvas.scene&&canvas.scene.tokens?.get(token.id)===token&&!!token.object;
   const current=()=>nativeStrike(message)&&currentToken(source)&&currentToken(target)&&source.actor===message.actor&&target.actor?.uuid===context.target.actor;
   if(!current())return;
   const provided=await existingAttack(item);
   if(!current()||message.flags?.[MODULE_ID]?.pilgrimStrike)return;
   const sequence=new Sequence(),sword=item.system.group==='sword';
   const attack=item.system.baseItem==='whip'?PILGRIM_MEDIA.club:sword?PILGRIM_MEDIA.sword:null;
   if(!provided&&attack)sequence.effect().file(attack).attachTo(source.object,{bindAlpha:true,bindVisibility:true}).rotateTowards(target.object).scaleToObject(1.5);
   sequence.effect().file(PILGRIM_MEDIA.leaves).attachTo(target.object,{bindAlpha:true,bindVisibility:true}).scaleToObject(0.65).opacity(0.7);
   if(!provided)sound(sequence,sword?PILGRIM_MEDIA.swordSound:PILGRIM_MEDIA.natureSound);
   // Claim before playback: an asset failure may already have played another section.
   await message.update({[`flags.${MODULE_ID}.pilgrimStrike`]:true});
   if(current()&&message.flags?.[MODULE_ID]?.pilgrimStrike===true)await sequence.play();
  }finally{strikes.delete(message.id);}
 }
 async function deleted(item){const key=rewardKey(item);if(key==='fan')await leaves(item,0);if(key==='scarf')await scarf(item,false);}
 async function reconcile(actor) {
  if(!localReady())return;
  for(const item of values(actor.items)){if(rewardKey(item)==='fan')await leaves(item,leafEffect(item)?.system.badge?.value??0);if(rewardKey(item)==='scarf')await scarf(item,actor.rollOptions?.all?.['wonton-ghost-scarf:lit']===true);}
 }
 const quiet=operation=>async(...args)=>{try{return await operation(...args);}catch(error){onError(error);}};
 return {leaves:quiet(leaves),scarf:quiet(scarf),release:quiet(release),area:quiet(area),clearArea:quiet(clearArea),strike:quiet(strike),deleted:quiet(deleted),reconcile:quiet(reconcile)};
}
