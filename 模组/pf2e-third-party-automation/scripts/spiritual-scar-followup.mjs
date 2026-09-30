import {MODULE_ID as M} from './rules.mjs';
import {getSourceId,isActiveGM} from './native-context.mjs';
import {SerialActions} from './runtime.mjs';
import {SPIRITUAL_SCAR_SOURCE} from './spiritual-scar-native.mjs';
import {SCAR_SLOWED,createSpiritualScarExpiry} from './spiritual-scar-expiry.mjs';
const OUTCOMES=['criticalFailure','failure','success','criticalSuccess'],PATH=`flags.${M}.spiritualScarFollowup`;
const values=c=>Array.from(c?.values?.()??c??[]),same=(a,b)=>JSON.stringify(a)===JSON.stringify(b),author=m=>m?.author?.id??m?.user?.id??m?.user;
const fail=reason=>Error(`精神伤痕后续结算未确认：${reason}；不会自动重掷或重复创建缓慢。`);
const demand=(ok,reason)=>{if(!ok)throw fail(reason)};
const privacy=p=>({blind:p?.blind===true,whisper:[...new Set(p?.whisper??[])].sort()});
const marker=nonce=>`${M}:spiritual-scar-save:${nonce}`;

/** One native non-basic Will, including native degree adjustments and the
 * installed check/reaction middleware. Persist the attempt before the die. */
export function createSpiritualScarFollowup({game,fromUuid=globalThis.fromUuid,Hooks=globalThis.Hooks,expiry=createSpiritualScarExpiry({game}),onError=console.error}={}){
 const queue=new SerialActions();
 const ready=()=>game.world?.id==='ujx5r8oipw7ercdr'&&game.system?.id==='pf2e'&&game.system.version==='8.5.1'&&typeof Hooks?.on==='function'&&typeof Hooks?.off==='function';
 const read=message=>message.flags?.[M]?.spiritualScarFollowup??null;
 async function apply(context){return queue.run(context.damageMessage?.id,async()=>{
  const {claim,actor,fiend,fiendToken,damageMessage,authorize}=context,leader=game.users.activeGM?.id,privateCard=privacy(context.privacy);
  const identity={schema:1,nonce:claim.nonce,actorUuid:actor.uuid,itemUuid:claim.itemUuid,targetActorUuid:fiend.uuid,targetTokenUuid:fiendToken?.uuid??null,damageMessageId:damageMessage.id};
  async function check(){
   demand(ready()&&isActiveGM(game)&&game.user.isGM&&game.users.activeGM?.id===leader,'主GM或兼容层已改变');
   demand(game.messages.get(damageMessage.id)===damageMessage&&!damageMessage.flags?.pf2e?.appliedDamage?.isReverted&&await authorize()===true,'原伤害或调用授权已改变');
   demand(isActiveGM(game)&&game.users.activeGM?.id===leader&&await fromUuid(actor.uuid)===actor&&await fromUuid(fiend.uuid)===fiend,'来源或目标已改变');
   const ability=await fromUuid(claim.itemUuid),token=await fromUuid(claim.tokenUuid);
   demand(ability?.actor===actor&&actor.items.get(ability.id)===ability&&getSourceId(ability)===SPIRITUAL_SCAR_SOURCE&&token?.actor===actor&&claim.actorUuid===actor.uuid&&claim.source.sourceActorUuid===fiend.uuid,'原始精神伤痕动作或来源 token 已改变');
   if(fiendToken)demand(await fromUuid(fiendToken.uuid)===fiendToken&&fiendToken.actor===fiend&&claim.source.sourceTokenUuid===fiendToken.uuid,'实际魔族 token 已改变');
   const dc=typeof actor.classDC?.dc==='number'?actor.classDC.dc:actor.classDC?.dc?.value;
   demand(Number.isFinite(dc)&&dc>0,'缺少原生职业 DC');
   demand(isActiveGM(game)&&game.users.activeGM?.id===leader,'主GM已改变');return ability;
  }
  const matches=r=>r&&Object.entries(identity).every(([k,v])=>r[k]===v)&&same(r.privacy,privateCard);
  async function save(before,after){
   await check();demand(same(read(damageMessage),before),'后续认领已改变');
   const result=await damageMessage.update({[PATH]:after});
   demand(result===damageMessage&&same(read(damageMessage),after),'后续认领没有保存');await check();return after;
  }
  let record=read(damageMessage);await check();
  if(record){demand(matches(record),'后续认领范围不一致');demand(record.status==='done','已有未确认的检定或效果操作');return {status:'done',messageId:record.messageId,effectId:record.effectId??null}}
  const ability=await check(),statistic=fiend.getStatistic?.('will');demand(typeof statistic?.roll==='function','魔族缺少原生意志检定');
  demand(claim.expiry?.actorUuid===actor.uuid&&claim.expiry?.tokenUuid===claim.tokenUuid&&Number.isFinite(claim.expiry?.startTime),'缺少原始一轮到期范围');
  record=await save(null,{...identity,privacy:privateCard,status:'rolling'});
  try{
   let card,creations=0;
   // Hook the final published native card, preserving check middleware that
   // stages drafts or modifies degrees before its original callback runs.
   const hook=Hooks.on('preCreateChatMessage',(message,_data,_options,userId)=>{
    const pf=message.flags?.pf2e,c=pf?.context,opts=c?.options??[];
    if(!opts.includes(marker(claim.nonce))&&!(c?.action==='spiritual-scar'&&pf?.origin?.uuid===ability.uuid))return;
    try{
     demand(isActiveGM(game)&&game.users.activeGM?.id===leader&&userId===game.user.id&&author(message)===game.user.id&&opts.includes(marker(claim.nonce))&&c.type==='saving-throw'&&pf.origin?.uuid===ability.uuid&&pf.origin.actor===actor.uuid&&creations===0,'原生豁免卡的范围或隐私无法验证');
     message.updateSource({...privateCard,[`flags.${M}.spiritualScarSave`]:{nonce:claim.nonce,damageMessageId:damageMessage.id}});creations++;return true;
    }catch(error){onError(error);return false}
   });
   try{
    await check();
    // Native check middleware must know the privacy before it rolls or offers
    // reactions. The publication hook above preserves the exact recipients.
    await statistic.roll({token:fiendToken??undefined,origin:actor,item:ability,action:'spiritual-scar',dc:{slug:'class'},traits:[...ability.system.traits.value],extraRollOptions:['action:spiritual-scar',marker(claim.nonce),'inflicts:slowed'],messageMode:privateCard.blind?'blind':privateCard.whisper.length?'gm':'public',skipDialog:true,createMessage:true,callback:(_roll,_outcome,message)=>{demand(!card,'原生豁免回调重复');card=message}});
   }finally{Hooks.off('preCreateChatMessage',hook)}
   await check();
   const pf=card?.flags?.pf2e,c=pf?.context,roll=card?.rolls?.[0],degree=OUTCOMES.indexOf(c?.outcome);
   demand(creations===1&&game.messages.get(card?.id)===card&&card?.isCheckRoll===true&&author(card)===game.user.id&&card.actor?.uuid===fiend.uuid&&card.speaker?.actor===fiend.id&&card.rolls.length===1&&c?.type==='saving-throw'&&c.action==='spiritual-scar'&&c.origin?.actor===actor.uuid&&c.target?.actor===fiend.uuid&&(!fiendToken||c.target.token===fiendToken.uuid&&`Scene.${card.speaker.scene}.Token.${card.speaker.token}`===fiendToken.uuid)&&pf.origin?.actor===actor.uuid&&pf.origin.uuid===ability.uuid&&c.dc?.slug==='class'&&Number.isFinite(c.dc.value)&&c.dc.value>0&&c.options?.includes(marker(claim.nonce))&&roll._evaluated===true&&roll.toJSON?.()?.evaluated===true&&Number.isFinite(roll.total)&&degree>=0&&roll.options?.degreeOfSuccess===degree&&same(privacy(card),privateCard)&&card.flags?.[M]?.spiritualScarSave?.nonce===claim.nonce,'最终原生意志结果不一致');
   record=await save(record,{...record,status:'rolled',messageId:card.id,degree,dc:c.dc.value});
   if(degree<2){
    const timing={...structuredClone(claim.expiry),nonce:claim.nonce,targetActorUuid:fiend.uuid,itemUuid:ability.uuid,damageMessageId:damageMessage.id,status:'armed'};
    const data={name:'精神伤痕：缓慢 1',type:'effect',img:ability.img,system:{slug:`tpa-spiritual-scar-${claim.nonce.toLowerCase()}`,duration:{value:-1,unit:'unlimited',expiry:null,sustained:false},context:{origin:{actor:actor.uuid,token:claim.tokenUuid,item:ability.uuid}},rules:[{key:'GrantItem',uuid:SCAR_SLOWED,allowDuplicate:true,inMemoryOnly:true,alterations:[{mode:'override',property:'badge-value',value:1}]}]},flags:{[M]:{spiritualScarExpiry:timing}}};
    record=await save(record,{...record,status:'creating'});
    const created=await fiend.createEmbeddedDocuments('Item',[data]);await check();
    const matches=values(fiend.items).filter(i=>i.flags?.[M]?.spiritualScarExpiry?.nonce===claim.nonce);
    demand(matches.length===1&&created?.includes(matches[0])&&matches[0].type==='effect'&&matches[0].system.context?.origin?.item===ability.uuid,'本次缓慢效果未确认创建');
    record=await save(record,{...record,status:'done',effectId:matches[0].id});await expiry.settle(matches[0]);
   }else record=await save(record,{...record,status:'done',effectId:null});
   return {status:'done',messageId:record.messageId,effectId:record.effectId};
  }catch(error){
   if(record&&same(read(damageMessage),record))await save(record,{...record,status:'uncertain',reason:String(error.message??error).slice(0,500)}).catch(onError);
   throw error;
  }
 })}
 return {ready,apply,register:args=>expiry.register?.(args),unregister:()=>expiry.unregister?.(),reconcile:()=>expiry.reconcile?.()};
}
