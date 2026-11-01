import {getSourceId,isActiveGM,resolveMessageTargets,publicTargetName} from './native-context.mjs';
import {MODULE_ID} from './rules.mjs';
import {SerialActions,requireOwner} from './runtime.mjs';
import {BOB2_AUTOMATION_WORLDS} from './world-gates.mjs';
import {isActualUseMessage} from './usage-events.mjs';

export const BOB2_RULE_SOURCES=Object.freeze({
 blood:'Compendium.pf2e.feat-effects.Item.52owimBcCQNBuNfM',
 overture:'Compendium.pf2e.spells-srd.Item.f0Z5mqGA6Yu79B8x',
 overtureEffect:'Compendium.pf2e.spell-effects.Item.ITErgFRfydm1xmnW',
 aid:'Compendium.pf2e.actionspf2e.Item.HCl3pzVefiv9ZKQW',
 aidEffect:'Compendium.pf2e.other-effects.Item.AHMUpMbaVkZ5A1KX',
});
const values=c=>Array.from(c?.values?.()??c??[]);
const liveEffect=i=>i?.type==='effect'&&!i.isExpired&&!i.system?.expired&&!i.suppressed;
const mind=actor=>values(actor?.items).some(i=>liveEffect(i)&&getSourceId(i)===BOB2_RULE_SOURCES.blood&&i.system.rules?.some(r=>r.key==='ChoiceSet'&&r.rollOption==='blooded-by-the-dead'&&r.selection==='bolster-the-mind'));

/** StatisticCheck already added ordinary incapacitation. Add only the comparison
 * newly crossed by Bolster the Mind; the prepared actor and roll options stay intact. */
export function bolsterMindCheckContext(context){
 const options=new Set(context.options??[]),has=trait=>options.has(trait)||options.has(`item:trait:${trait}`);
 if(!context.dc||!has('mental')||!has('incapacitation'))return context;
 const actor=context.actor??(context.origin?.self?context.origin.actor:context.target?.actor),item=context.item;
 const level=item?.type==='spell'?item.rank*2:item?.isOfType?.('physical')?item.level:context.origin?.actor?.level??actor?.level;
 const saving=context.type==='saving-throw',opposed=['attack-roll','spell-attack-roll','skill-check'].includes(context.type);
 const protectedActor=saving?actor:opposed?context.target?.actor:null;
 if(!mind(protectedActor)||!Number.isFinite(level)||!Number.isFinite(protectedActor.level)||protectedActor.level>level||protectedActor.level+2<=level)return context;
 const prior=context.dosAdjustments??[];
 if(prior.some(r=>r.adjustments?.all?.label==='PF2E.TraitIncapacitation'))return context;
 return {...context,dosAdjustments:[...prior,{adjustments:{all:{label:'PF2E.TraitIncapacitation',amount:saving?1:-1}}}]};
}

const enabled=game=>game.system?.id==='pf2e'&&BOB2_AUTOMATION_WORLDS.has(game.world?.id);
const own=item=>item?.flags?.[MODULE_ID]?.overture;
const skills=['acrobatics','arcana','athletics','crafting','deception','diplomacy','intimidation','medicine','nature','occultism','performance','religion','society','stealth','survival','thievery'];
const skillKeys=actor=>[...new Set([...skills,...Object.keys(actor.skills??{})])];
const route=`${MODULE_ID}:usage:bob2:aid`;

export function buildBob2RuleLinkRepairs(original,{game}={}){
 const rules=structuredClone(original),changes=[];
 if(!enabled(game)||game.modules?.get('patreon-v3')?.active!==true)return {rules,changes};
 const group=rules?.['5tMRJEGX3HmeZu3s'];
 if(group?.uuid!=='5tMRJEGX3HmeZu3s'||JSON.stringify(group.source)!==JSON.stringify([BOB2_RULE_SOURCES.aid]))return {rules,changes};
 for(const [index,rule]of (Array.isArray(group.baseRules)?group.baseRules:[]).entries()){
  if(rule.type!=='base'||rule.triggerType!=='postInfo'||rule.target!=='TargetEffect'||rule.value!==BOB2_RULE_SOURCES.aidEffect||!Array.isArray(rule.predicate)||!rule.predicate.includes('origin:item:aid')||rule.predicate.some(p=>p?.not===route))continue;
  const before=structuredClone(rule.predicate);rule.predicate.push({not:route});
  changes.push({path:`5tMRJEGX3HmeZu3s.baseRules.${index}.predicate`,before,after:structuredClone(rule.predicate),reason:'序曲协助由原生表演结果授予一次收益；普通协助保留原处理。'});
 }
 return {rules,changes};
}

/** Native Cast prepares the ally's skill; the later original Aid Use owns the
 * reaction, Performance roll and one native Aid effect. */
export function createBob2RuleLinks({game,fromUuid=globalThis.fromUuid,choose,runNative,castEvents}){
 const queue=new SerialActions();
 const overture=item=>enabled(game)&&item?.type==='spell'&&getSourceId(item)===BOB2_RULE_SOURCES.overture;
 const prepared=actor=>values(actor?.items).filter(i=>liveEffect(i)&&getSourceId(i)===BOB2_RULE_SOURCES.overtureEffect&&own(i)?.castMessageId);
 const resolveAction=item=>overture(item)?'bob2:overture':enabled(game)&&item?.type==='action'&&getSourceId(item)===BOB2_RULE_SOURCES.aid&&prepared(item.actor).length?'bob2:aid':undefined;
 castEvents.addMatcher(overture);
 const save=(message,record)=>message.update({[`flags.${MODULE_ID}.overtureUse`]:record});
 const load=async uuid=>{const source=await fromUuid(uuid);if(source?.type!=='effect'||!source.toObject)throw Error('无法读取序曲或协助的原生效果。');const data=source.toObject();delete data._id;data._stats={...data._stats,compendiumSource:uuid};return data;};
 function sourceContext({actor,item,message,user,action}){
  if(!enabled(game)||!isActiveGM(game))throw Error('序曲协助必须由当前主GM结算。');requireOwner(actor,user);
  const origin=message?.flags?.pf2e?.origin,author=message?.author?.id??message?.user?.id??message?.user;
  const source=action==='bob2:overture'?BOB2_RULE_SOURCES.overture:action==='bob2:aid'?BOB2_RULE_SOURCES.aid:null;
  if(!source||getSourceId(item)!==source||item.actor!==actor||actor.items.get(item.id)!==item||game.messages.get(message?.id)!==message||author!==user.id||message.speaker?.actor!==actor.id||origin?.uuid!==item.uuid||origin.actor!==actor.uuid||message.isRoll||message.rolls?.length)throw Error('序曲协助需要原操作者的原始使用消息。');
  if(action==='bob2:overture'&&!message.flags?.[MODULE_ID]?.nativeCast)throw Error('需要本次真实施法回执，展示法术卡不准备协助。');
  if(action==='bob2:aid'&&!isActualUseMessage(message))throw Error('需要实际使用协助反应。');
 }
 async function executeUsage(context){
  const {actor,item,message,user,action}=context;sourceContext(context);
  return queue.run(actor.uuid,async()=>{
   sourceContext(context);const prior=message.flags?.[MODULE_ID]?.overtureUse;
   if(prior?.status==='done'||prior?.status==='cancelled')return prior.result;
   if(prior)throw Error('此序曲活动已开始但结果未确认；请核对原卡，不会自动重试。');
   const cancel=async()=>{const result='已取消；未使用序曲协助。';await save(message,{status:'cancelled',result});return result;};
   if(action==='bob2:overture'){
    await castEvents.ensurePaid(context);sourceContext(context);
    const scene=game.scenes.get(message.speaker.scene),recorded=await resolveMessageTargets(message,{fromUuid});
    const candidates=(recorded.length?recorded:values(scene?.tokens)).filter(t=>t.parent===scene&&t.actor&&t.actor.uuid!==actor.uuid);
    if(!candidates.length)throw Error('请选择一名接受本次序曲协助的盟友。');
    const targetUuid=await choose({actor,user,title:'振奋序曲：选择接受协助的盟友（60尺内）',choices:candidates.map(t=>({value:t.uuid,label:publicTargetName(t,{game,user})}))});if(!targetUuid)return cancel();
    const target=candidates.find(t=>t.uuid===targetUuid);
    if(!target)throw Error('序曲受益人选择无效。');
    const beneficiary=target.actor;
    const skill=await choose({actor,user,title:'振奋序曲：准备协助盟友的哪项技能',choices:skillKeys(beneficiary).map(value=>({value,label:beneficiary.skills?.[value]?.label??game.i18n?.localize?.(globalThis.CONFIG?.PF2E?.skills?.[value]?.label??value)??value}))});if(!skill)return cancel();
    sourceContext(context);if(scene.tokens.get(target.id)!==target||target.actor!==beneficiary||!skillKeys(beneficiary).includes(skill))throw Error('序曲受益人或技能选择已改变。');
    const data=await load(BOB2_RULE_SOURCES.overtureEffect);sourceContext(context);
    data.system.start={value:game.time.worldTime,initiative:game.combat?.combatant?.initiative??null};
    data.flags={...data.flags,[MODULE_ID]:{...data.flags?.[MODULE_ID],overture:{castMessageId:message.id,targetUuid:target.uuid,targetActorUuid:target.actor.uuid,skill}}};
    await save(message,{status:'preparing'});
    const old=prepared(actor);if(old.length)await actor.deleteEmbeddedDocuments('Item',old.map(i=>i.id));
    await actor.createEmbeddedDocuments('Item',[data]);
    const result='已准备振奋序曲；盟友进行所选技能检定前，使用原生协助反应。';await save(message,{status:'done',result});return result;
   }
   const effects=prepared(actor);if(effects.length!==1)throw Error('没有唯一且未到期的序曲准备效果。');
   const effect=effects[0],preparation=structuredClone(own(effect)),target=await fromUuid(preparation.targetUuid);
   const assertReady=()=>{
    sourceContext(context);
    if(!liveEffect(effect)||actor.items.get(effect.id)!==effect||!target?.actor||target.actor.uuid!==preparation.targetActorUuid||game.scenes.get(target.parent?.id)?.tokens.get(target.id)!==target||target.parent.id!==message.speaker.scene||!skillKeys(target.actor).includes(preparation.skill)||own(effect)?.aidMessageId&&own(effect).aidMessageId!==message.id)throw Error('序曲效果、受益人或本次协助已改变。');
   };assertReady();
   const answer=await choose({actor,user,title:'振奋序曲：盟友即将进行所选技能检定；确认其自愿、表演适用且可以使用协助反应（DC 15）',choices:[{value:'aid',label:'使用协助反应并投表演'},{value:'cancel',label:'取消'}]});if(answer!=='aid')return cancel();assertReady();
   const native=await runNative(context,{type:'check',nativeAction:'aid',statistic:'performance',action:'aid',itemUuid:item.uuid,sourceUuid:BOB2_RULE_SOURCES.aid,tokenUuid:`Scene.${message.speaker.scene}.Token.${message.speaker.token}`,targetUuid:target.uuid,targetActorUuid:target.actor.uuid,dc:{value:15},options:['action:aid','action:aid:performance',route]},undefined,async()=>{
    assertReady();await effect.update({[`flags.${MODULE_ID}.overture.aidMessageId`]:message.id});await save(message,{status:'rolling'});
   });
   if(native?.status==='cancelled')return cancel();assertReady();
   const outcome=native?.check?.rolls?.[0]?.options?.degreeOfSuccess,rank=actor.getStatistic('performance')?.rank;
   if(native?.status!=='rolled'||![0,1,2,3].includes(outcome)||!Number.isInteger(rank)||rank<0||rank>4)throw Error('未收到本次协助的原生成功度。');
   const bonus=outcome===0?-1:outcome===2?1:outcome===3?rank>=3?rank:2:null;
   await save(message,{status:'applying',outcome});
   if(bonus!==null){
    const data=await load(BOB2_RULE_SOURCES.aidEffect);assertReady();
    data.system.rules=[{key:'FlatModifier',selector:preparation.skill,type:'circumstance',value:bonus,removeAfterRoll:[`check:statistic:${preparation.skill}`]}];
    data.system.duration={value:1,unit:'rounds',expiry:'turn-start',sustained:false};data.system.start={value:game.time.worldTime,initiative:game.combat?.combatant?.initiative??null};
    data.flags={...data.flags,[MODULE_ID]:{...data.flags?.[MODULE_ID],overtureAid:{messageId:message.id,sourceActorUuid:actor.uuid,skill:preparation.skill}}};
    await target.actor.createEmbeddedDocuments('Item',[data]);
   }
   await actor.deleteEmbeddedDocuments('Item',[effect.id]);
   const result=bonus===null?'本次协助失败，没有收益；已消耗序曲准备。':`已结算协助（${bonus>0?'+':''}${bonus}环境修正）；盟友下一次所选技能检定后自动移除。`;
   await save(message,{status:'done',result});return result;
  });
 }
 return {resolveAction,requiresActualUse:(_item,action)=>action==='bob2:aid',executeUsage,
  interceptCheck:(native,check,context,...args)=>native(check,enabled(game)?bolsterMindCheckContext(context):context,...args)};
}
