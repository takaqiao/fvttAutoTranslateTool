import {MODULE_ID as ID} from './rules.mjs';
import {getSourceId,isActiveGM,showNativeChoice} from './native-context.mjs';
import {SerialActions} from './runtime.mjs';
import {BOB2_AUTOMATION_WORLDS} from './world-gates.mjs';

export const DEATHS_CALL_SOURCE='Compendium.pf2e.spells-srd.Item.HG4afO9EOGEU9bZN';
const KEY='deathsCall',INPUT='deathsCallInput';
const values=collection=>Array.from(collection?.values?.()??collection??[]);
const copy=value=>structuredClone(value);
const author=message=>message?.author?.id??message?.user?.id??message?.user;
const demand=(condition,text)=>{if(!condition)throw Error(`斩杀号令：${text}`);};
const eligible=actor=>['character','npc','familiar'].includes(actor?.type)&&['living','undead'].includes(actor.modeOfBeing);
const hpSource=actor=>actor.system.attributes.hp.tempSource??actor.system.attributes.hp.tempsource;

/** Selection precedes native Cast. Only its paid message can create the effect. */
export function createDeathsCall({game,fromUuid=globalThis.fromUuid,nativeCasts,chooseCast=showNativeChoice,onError=console.error}={}){
 const scopes=new Map(),queue=new SerialActions(),trackedActors=new Map();
 const enabled=()=>BOB2_AUTOMATION_WORLDS.has(game.world?.id)&&game.system?.id==='pf2e';
 const matches=item=>enabled()&&item?.type==='spell'&&getSourceId(item)===DEATHS_CALL_SOURCE;
 const ownEffects=actor=>values(actor.items).filter(effect=>{
  const marker=effect.flags?.[ID]?.[KEY];
  return effect.type==='effect'&&marker?.sourceId===DEATHS_CALL_SOURCE&&marker.actorUuid===actor.uuid&&typeof marker.messageId==='string'&&typeof marker.paymentId==='string'&&effect.system?.context?.origin?.actor===actor.uuid;
 });
 const track=actor=>{if(ownEffects(actor).length)trackedActors.set(actor.uuid,actor);else trackedActors.delete(actor.uuid);};
 function assertGM(ctx){
  const original=ctx.item.original??ctx.item;
  demand(isActiveGM(game)&&game.messages.get(ctx.message.id)===ctx.message&&author(ctx.message)===ctx.user.id&&ctx.actor.testUserPermission(ctx.user,'OWNER')&&matches(ctx.item)&&ctx.actor.items.get(original.id)===original,'施法来源、所有者或主GM已改变。');
 }
 function candidates(){
  return values(game.scenes?.current?.tokens).filter(token=>eligible(token.actor)&&(game.user.isGM||token.hidden!==true&&token.object?.isVisible===true));
 }
 function targetLabel(token,index){
  const visible=!game.pf2e?.settings?.tokens?.nameVisibility||token.playersCanSeeName===true||game.user.isGM||token.actor.testUserPermission?.(game.user,'OWNER');
  return visible?token.name??token.actor.name:`目标 ${index+1}`;
 }
 async function inspect(item,input){
  const token=await fromUuid(input?.triggerTokenUuid),entry=item.spellcasting;
  demand(input?.confirmed===true&&token?.documentName==='Token'&&token.actor?.uuid===input.triggerActorUuid&&eligible(token.actor),'触发生物来源已改变。');
  demand(entry?.uuid===input.entryUuid&&entry.actor===item.actor&&item.actor.items.get(entry.id)===entry,'原生施法条目已改变。');
  const level=token.actor.level,modifier=entry.statistic?.attributeModifier?.value;
  demand(Number.isInteger(level)&&Number.isInteger(modifier),'生物等级或原生施法属性调整值不可用。');
  return {amount:Math.max(0,(level+modifier)*(token.actor.modeOfBeing==='undead'?2:1))};
 }
 async function interceptCast({item,options={}},native){
  if(!matches(item)||options.consume===false||options.message===false)return native();
  demand(item.actor?.testUserPermission(game.user,'OWNER')&&game.users.activeGM?.id,'需要原施法者与在线主GM。');
  demand(!scopes.has(item.uuid),'本次施法选择尚未完成。');
  const available=candidates();demand(available.length,'没有可选的可见触发生物。');
  const selected=await chooseCast({title:'斩杀号令：选择触发生物',choices:[...available.map((token,index)=>({value:token.uuid,label:targetLabel(token,index)})),{value:'cancel',label:'取消施法'}]});
  const token=available.find(token=>token.uuid===selected);if(!token)return;
  const answer=await chooseCast({title:'斩杀号令：确认触发与反应',choices:[{value:'confirm',label:'使用反应：所选活物刚刚死亡，或不死生物刚刚被摧毁，且位于20尺内'},{value:'cancel',label:'取消施法'}]});
  if(answer!=='confirm')return;
  const input={triggerTokenUuid:token.uuid,triggerActorUuid:token.actor.uuid,entryUuid:item.spellcasting?.uuid,confirmed:true};
  await inspect(item,input);scopes.set(item.uuid,input);
  try{return await native();}finally{scopes.delete(item.uuid);}
 }
 const captureCast=item=>matches(item)&&scopes.has(item.uuid)?copy(scopes.get(item.uuid)):undefined;
 async function executeUsage(ctx){
  const {actor,item,message}=ctx,input=message.flags?.[ID]?.[INPUT],proof=message.flags?.[ID]?.nativeCast;
  if(!matches(item)||!proof||!input)return {status:'cancelled',result:'展示卡未绑定实际斩杀号令施法。'};
  return queue.run(actor.uuid,async()=>{
   assertGM(ctx);const prior=message.flags?.[ID]?.[KEY];
   if(prior)return {status:prior.status==='done'?'done':'waiting',result:prior.result??'本次施法已开始结算，请核对原卡；不会重复授予临时生命值。'};
   const {amount}=await inspect(item,input);assertGM(ctx);
   const receipt=await nativeCasts.ensurePaid(ctx);assertGM(ctx);
   demand(receipt?.id===proof.id&&receipt.state==='used'&&receipt.messageId===message.id,'原生施法支付回执未绑定原卡。');
   await message.update({[`flags.${ID}.${KEY}`]:{status:'creating',paymentId:receipt.id}});
   try{
    assertGM(ctx);
    const [effect]=await actor.createEmbeddedDocuments('Item',[{
     name:'斩杀号令',type:'effect',img:item.img,
     flags:{[ID]:{[KEY]:{sourceId:DEATHS_CALL_SOURCE,actorUuid:actor.uuid,messageId:message.id,paymentId:receipt.id}}},
     system:{slug:'tpa-deaths-call',level:{value:item.rank},description:{value:'<p>本次斩杀号令的临时生命值；耗尽或到期后结束。</p>'},duration:{value:1,unit:'minutes',expiry:'turn-start',sustained:false},start:{value:game.time.worldTime,initiative:game.combat?.combatant?.initiative??null},context:{origin:{actor:actor.uuid,item:item.uuid}},rules:[{key:'TempHP',value:amount}]}
    }]);
    assertGM(ctx);demand(effect&&actor.items.get(effect.id)===effect,'原生临时生命值效果创建结果未确认。');track(actor);
    // Native Item._onCreate starts its Actor update without awaiting it.
    const result=`已创建数值为 ${amount} 的临时生命值效果，持续1分钟。原生规则保留较高值；本次临时生命值耗尽时效果结束。`;
    await message.update({[`flags.${ID}.${KEY}`]:{status:'done',paymentId:receipt.id,effectId:effect.id,amount,result}});
    return {status:'done',result};
   }catch(error){if(isActiveGM(game))await message.update({[`flags.${ID}.${KEY}`]:{...message.flags[ID][KEY],status:'uncertain',result:`${error.message} 请核对本次效果，不会自动重新创建。`}});throw error;}
  });
 }
 async function cleanup(actor){
  if(!enabled()||!isActiveGM(game))return;
  return queue.run(actor.uuid,async()=>{
   if(!enabled()||!isActiveGM(game))return;
   const expired=ownEffects(actor).filter(effect=>{
    const hasExpired=effect.isExpired||effect.remainingDuration?.expired;
    // PF2e's effect tracker owns expired effects while automatic removal is enabled.
    if(hasExpired&&game.pf2e?.settings?.automation?.removeEffects)return false;
    return hasExpired||hpSource(actor)!==effect.id||actor.system.attributes.hp.temp<=0;
   });
   // Native TempHP.onDelete clears HP only while this exact effect owns it.
   if(expired.length)await actor.deleteEmbeddedDocuments('Item',expired.map(effect=>effect.id));
   track(actor);
  });
 }
 function register({Hooks}={}){
  if(!enabled())return()=>{};
  if(game.user.isGM){
   const actors=new Set([...values(game.actors),...values(game.scenes).flatMap(scene=>values(scene.tokens).map(token=>token.actor)).filter(Boolean)]);
   for(const actor of actors)track(actor);
  }
  const run=actor=>cleanup(actor).catch(onError),listeners=[];
  const on=(name,fn)=>listeners.push([name,Hooks.on(name,fn)]);
  on('createItem',item=>{if(game.user.isGM&&item.actor&&item.flags?.[ID]?.[KEY]?.sourceId===DEATHS_CALL_SOURCE)track(item.actor);});
  on('deleteItem',item=>{if(item.actor&&trackedActors.has(item.actor.uuid))track(item.actor);});
  on('updateActor',(actor,changes)=>{
   if(!isActiveGM(game)||!trackedActors.has(actor.uuid))return;
   if(changes.system?.attributes?.hp||Object.keys(changes).some(key=>key.startsWith('system.attributes.hp.')))return run(actor);
  });
  for(const event of ['updateWorldTime','updateCombat'])on(event,()=>isActiveGM(game)?Promise.all([...trackedActors.values()].map(run)):undefined);
  return()=>{for(const[name,id]of listeners)Hooks.off(name,id);trackedActors.clear();};
 }
 return {matches,captureCast,interceptCast,resolveAction:item=>matches(item)?'deaths-call':null,executeUsage,register};
}
