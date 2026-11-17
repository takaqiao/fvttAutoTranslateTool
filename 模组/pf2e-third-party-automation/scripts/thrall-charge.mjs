import {MODULE_ID,hasSource} from './rules.mjs';
import {SerialActions} from './runtime.mjs';
import {isActiveGM,publicTargetName,upsertOwnedEffect} from './native-context.mjs';
import {ownedLivingThrall,inRange} from './thrall-automation.mjs';

export const THRALL_CHARGE_SPELL='Compendium.pf2e.spells-srd.Item.NWDTNTpfPEc821pu';
export const THRALL_CHARGE_EFFECT='Compendium.pf2e.spell-effects.Item.Wd1fYphfUNpZlgcH';
// Summons Assistant 2.20.6 adds this to the caster's first target once its
// setting lookup works; it must never stack with the native effect.
export const SUMMONS_THRALL_CHARGE_EFFECT='Compendium.pf2e-summons-assistant.pf2e-summons-assistant-items.Item.zFyeEWPXwfeQgYku';
const KEY='thrall-charge',PATH=`flags.${MODULE_ID}.thrallCharge`;
const DESTROY_OPTION='thrall-charge:destroy-thrall',CAST_OPTION='origin:action:slug:cast-a-spell';
const CHARGE_SLUGS=new Set(['spell-effect-thrall-charge','effect-thrall-charge']);
const values=collection=>Array.from(collection?.values?.()??collection??[]);
const authorId=message=>message?.author?.id??message?.user?.id??message?.user;
const marker=document=>document?.flags?.[MODULE_ID]?.thrallCharge;
const chargeEffect=item=>item?.type==='effect'&&(hasSource(item,THRALL_CHARGE_EFFECT)||hasSource(item,SUMMONS_THRALL_CHARGE_EFFECT)||CHARGE_SLUGS.has(item.system?.slug));
const ownEffect=actor=>values(actor?.items).find(item=>item.type==='effect'&&item.flags?.[MODULE_ID]?.nativeEffectKey===KEY&&marker(item));
const extraDice=rank=>rank>=10?4:rank>=6?3:rank>=2?2:1;

function strikeRoll(message){
 const context=message?.flags?.pf2e?.context;
 if(context?.type==='attack-roll'&&context.action==='strike'&&context.domains?.includes('strike-attack-roll'))return 'attack';
 if(context?.type==='damage-roll'&&context.sourceType==='attack'&&context.domains?.includes('strike-damage'))return 'damage';
 return null;
}

/**
 * Thrall Charge: the active GM puts the native spell effect on one of the
 * caster's thralls and ends it after that thrall's next Strike. The Step is
 * left to the player.
 */
export function createThrallCharge({game,fromUuid=globalThis.fromUuid,choose,onError=console.error}={}){
 const queue=new SerialActions(),handled=new Set(),settled=new WeakSet();

 // Only a native cast pushes the cast-a-spell option; posting the spell to
 // chat also records `casting` whenever the spell has a spellcasting entry.
 async function castContext(message){
  const pf2e=message.flags?.pf2e,origin=pf2e?.origin;
  if(origin?.type!=='spell'||!pf2e.casting||!origin.rollOptions?.includes(CAST_OPTION))return null;
  const spell=await fromUuid(origin.uuid),caster=spell?.actor;
  if(spell?.type!=='spell'||!hasSource(spell,THRALL_CHARGE_SPELL)||!caster||caster.uuid!==origin.actor)return null;
  const user=game.users.get(authorId(message));
  if(!user||!caster.testUserPermission(user,'OWNER'))return null;
  return {caster,user,rank:origin.castRank};
 }

 async function pickThrall({caster,user,source,candidates}){
  const targeted=values(user.targets).map(target=>target?.document??target).filter(token=>candidates.includes(token));
  if(targeted.length===1)return targeted[0];
  if(candidates.length===1)return candidates[0];
  const answer=await choose({actor:caster,user,title:'奴仆冲锋：选择冲锋的奴仆',choices:candidates.map(token=>({
   value:token.uuid,label:`${publicTargetName(token,{game,user})} · ${Math.round(source.object.distanceTo(token.object))}尺`,
  }))});
  return candidates.find(token=>token.uuid===answer)??null;
 }

 async function applyCharge(message,{caster,user,rank}){
  if(!Number.isInteger(rank)||rank<1||rank>10)throw Error('奴仆冲锋：施法卡没有记录有效的施放环级。');
  const scene=game.scenes.get(message.speaker?.scene),source=scene?.tokens.get(message.speaker?.token);
  if(source?.actor?.uuid!==caster.uuid)throw Error('奴仆冲锋：施法卡没有记录施法者在场景中的Token。');
  if(!source.object)throw Error('奴仆冲锋：主GM当前没有打开施法者所在的场景，无法测量30尺内的奴仆。');
  const candidates=values(scene.tokens).filter(token=>ownedLivingThrall(token,caster)&&inRange(source,token));
  if(!candidates.length)throw Error('奴仆冲锋：施法者30尺内没有属于自己的存活奴仆（奴仆需由 PF2e Summons Assistant 召出）。');
  const data=(await fromUuid(THRALL_CHARGE_EFFECT))?.toObject();
  const choiceSet=data?.system?.rules?.find(rule=>rule.key==='ChoiceSet'&&rule.rollOption===DESTROY_OPTION);
  if(!choiceSet)throw Error('奴仆冲锋：找不到 PF2e 的奴仆冲锋法术效果或其中摧毁奴仆的选择。');

  const token=await pickThrall({caster,user,source,candidates});
  if(!token)return {status:'cancelled',rank};
  const dice=`${extraDice(rank)}d6`;
  const answer=await choose({actor:caster,user,title:'奴仆冲锋：是否在这次打击中摧毁奴仆',choices:[
   {value:'keep',label:`不摧毁：打击额外 ${dice} 伤害`},
   {value:'destroy',label:`摧毁：打击额外 ${dice} 伤害，伤害 +${rank} 状态加值，打击后移除奴仆`},
  ]});
  if(answer==null)return {status:'cancelled',rank,thrallUuid:token.uuid};
  if(scene.tokens.get(token.id)!==token||!ownedLivingThrall(token,caster))throw Error('奴仆冲锋：所选奴仆已不在场或已被摧毁。');

  const destroy=answer==='destroy',thrall=token.actor;
  data.system.level={...data.system.level,value:rank};
  // A preset selection keeps PF2e from prompting on the GM client.
  choiceSet.selection=destroy?'yes':'no';
  data._stats={...data._stats,compendiumSource:THRALL_CHARGE_EFFECT};
  data.flags={...data.flags,[MODULE_ID]:{...data.flags?.[MODULE_ID],thrallCharge:{messageId:message.id,casterUuid:caster.uuid,destroy,rank}}};
  // An earlier effect of ours is recreated too, so PF2e restarts its duration.
  const stale=values(thrall.items).filter(chargeEffect).map(item=>item.id);
  if(stale.length)await thrall.deleteEmbeddedDocuments('Item',stale);
  const effect=await upsertOwnedEffect(thrall,KEY,data);
  return {status:'applied',rank,destroy,thrallUuid:token.uuid,effectId:effect.id};
 }

 async function cast(message){
  if(marker(message)||handled.has(message.id))return;
  const context=await castContext(message);
  if(!context||marker(message)||handled.has(message.id))return;
  handled.add(message.id);
  return queue.run(context.caster.uuid,async()=>{
   let result;
   try{result=await applyCharge(message,context);}
   catch(error){await message.update({[PATH]:{status:'failed',rank:context.rank,error:error.message}}).catch(onError);throw error;}
   await message.update({[PATH]:result});
  });
 }

 // A hit waits for its damage roll so the effect's dice and bonus reach it.
 async function strike(message,kind){
  const token=game.scenes.get(message.speaker?.scene)?.tokens.get(message.speaker?.token),thrall=token?.actor;
  if(!ownEffect(thrall))return;
  return queue.run(thrall.uuid,async()=>{
   const effect=ownEffect(thrall);
   if(!effect||settled.has(effect))return;
   const outcome=message.flags.pf2e.context.outcome??null;
   if(kind==='attack'&&!['failure','criticalFailure'].includes(outcome)){
    await effect.update({[`${PATH}.attack`]:{messageId:message.id,outcome}});
    return;
   }
   settled.add(effect);
   const {destroy}=marker(effect);
   await effect.delete();
   if(!destroy)return;
   // Native Token deletion preserves Summons Assistant's destruction hooks.
   try{await token.delete();}
   catch(error){throw Error('奴仆冲锋：效果已结束，但移除奴仆Token失败，请手动移除。',{cause:error});}
  });
 }

 async function handleMessage(message){
  if(!isActiveGM(game))return;
  const kind=strikeRoll(message);
  if(kind)return strike(message,kind);
  if(message?.flags?.pf2e?.origin?.type==='spell')return cast(message);
 }

 // preCreateItem runs on the creating client, which may be a player's.
 function blockDuplicate(item){
  if(chargeEffect(item)&&!marker(item)&&ownEffect(item.parent))return false;
 }

 function register({Hooks}={}){
  const listeners=[
   ['createChatMessage',Hooks.on('createChatMessage',message=>handleMessage(message).catch(onError))],
   ['preCreateItem',Hooks.on('preCreateItem',blockDuplicate)],
  ];
  return()=>{for(const [name,id] of listeners)Hooks.off(name,id);};
 }

 return {register,handleMessage};
}
