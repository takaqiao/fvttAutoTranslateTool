import {MODULE_ID} from './rules.mjs';
import {SerialActions,requireOwner} from './runtime.mjs';
import {getSourceId,isActiveGM,resolveMessageTargets,upsertOwnedEffect} from './native-context.mjs';
import {getNativeCastEvents} from './amp-cast-events.mjs';

export const PARTY_SOURCES=Object.freeze({
 clue:'Compendium.pf2e.actionspf2e.Item.25WDi1cVUrW92sUj',clueEffect:'Compendium.pf2e.feat-effects.Item.vhSYlQiAQMLuXqoc',
 anoint:'Compendium.pf2e.feats-srd.Item.bRftzbFvSF1pilIo',anointEffect:'Compendium.pf2e.feat-effects.Item.nnF7RSVlC6swbSw8',
 guardian:'Compendium.pf2e.feats-srd.Item.YwVdaszwpDJd6kf9',
 imperial:'Compendium.pf2e.classfeatures.Item.ZEtJJ5UOlV5oTWWp',imperialEffect:'Compendium.pf2e.feat-effects.Item.vguxP8ukwVTWWWaA',
 knownEffect:'Compendium.pf2e.feat-effects.Item.DvyyA11a63FBwV7x',
 raiseShieldEffect:'Compendium.pf2e.equipment-effects.Item.2YgXoHvJfrDHucMr',
});
const values=x=>Array.from(x?.values?.()??x??[]),own=x=>x?.flags?.[MODULE_ID]?.party;
const has=(actor,key)=>values(actor?.items).some(i=>getSourceId(i)===PARTY_SOURCES[key]);
const clone=structuredClone;
const tokenDoc=t=>t?.document??t;
export function buildPartyPatreonRepairs(original){
 const rules=clone(original),changes=[];
 for(const[id,group]of Object.entries(rules??{}))for(const list of ['baseRules','handlerRules'])for(const[index,rule]of(group[list]??[]).entries()){
  if(!Array.isArray(rule.predicate))continue;
  let exclusion;
  if(group.source?.includes(PARTY_SOURCES.clue)&&rule.triggerType==='postInfo'&&rule.target==='SelfEffect'&&rule.value===PARTY_SOURCES.clueEffect&&rule.predicate.includes('origin:item:clue-in'))exclusion={not:'origin:item:sourceId:'+PARTY_SOURCES.clue};
  if(group.source?.includes(PARTY_SOURCES.anoint)&&rule.triggerType==='postInfo'&&rule.value===PARTY_SOURCES.anointEffect&&rule.predicate.includes('origin:item:anoint-ally'))exclusion={not:'origin:item:sourceId:'+PARTY_SOURCES.anoint};
  if(group.source?.includes(PARTY_SOURCES.guardian)&&rule.triggerType==='postInfo'&&rule.value==='devotedGuardian')exclusion={not:'origin:item:sourceId:'+PARTY_SOURCES.guardian};
  if(rule.triggerType==='spell-cast'&&rule.value==='bloodlines'&&rule.predicate.includes('feature:bloodline-spells'))exclusion={not:MODULE_ID+':usage:party:imperial'};
  if(!exclusion||rule.predicate.some(p=>JSON.stringify(p)===JSON.stringify(exclusion)))continue;
  const before=clone(rule.predicate);rule.predicate.push(exclusion);changes.push({path:`${id}.${list}.${index}.predicate`,before,after:clone(rule.predicate),reason:'由原始使用者与目标明确关联的协作能力统一结算'});
 }
 return {rules,changes};
}
const needsKnownWeaknessCheck=r=>r==null||(r.key==='TokenMark'&&r.slug==='known-weakness');
export function buildKnownWeaknessRepair(item){
 // Most inventory has no legacy TokenMark. Check current rules before the
 // PF2e sourceId getter; retain every maintenance hook and avoid a stale cache.
 const currentRules=item?.system?.rules;
 if(Array.isArray(currentRules)&&!currentRules.some(needsKnownWeaknessCheck))return null;
 if(getSourceId(item)!==PARTY_SOURCES.knownEffect)return null;
 const rules=clone(item.system.rules??[]),mark=rules.find(r=>r.key==='TokenMark'&&r.slug==='known-weakness');
 if(!mark||!rules.some(r=>r.key==='FlatModifier'&&r.predicate?.includes('target:mark:known-weaknesses')))return null;
 mark.slug='known-weaknesses';return rules;
}
export function isImperialBloodMagic(item){
 if(item?.type!=='spell'||!has(item.actor,'imperial')||!item.system?.traits?.otherTags?.includes('blood-magic-spell'))return false;
 const options=values(item.actor.getRollOptions?.());
 return options.includes('blood-magic:imperial')&&!options.some(o=>o.startsWith('second-blood-magic:'));
}
function distance(source,target){
 if(!source?.object||!target?.object||source.parent?.id!==target.parent?.id)return null;
 const d=source.object.distanceTo?.(target.object);return Number.isFinite(d)?d:null;
}
export function guardianActive(source,target){
 const shield=source?.actor?.attributes?.shield,d=distance(source,target);
 return !!shield?.raised&&!shield.broken&&!shield.destroyed&&d!==null&&d<=5;
}
export function createPartyAutomation({game,fromUuid=globalThis.fromUuid,choose,onError=console.error,castEvents=getNativeCastEvents({game,fromUuid})}={}){
 castEvents.addMatcher(isImperialBloodMagic);
 const queue=new SerialActions(),lastActions=new Map();
 const now=()=>game.time.worldTime;
 const gm=()=>{if(!isActiveGM(game))throw Error('协作能力必须由当前主GM结算。');};
 const resolveAction=item=>{const key=['clue','anoint','guardian'].find(k=>getSourceId(item)===PARTY_SOURCES[k]);return key?'party:'+key:isImperialBloodMagic(item)?'party:imperial':undefined};
 const load=async uuid=>{const i=await fromUuid(uuid);if(!i?.toObject)throw Error('无法读取协作能力的原生效果。');const d=i.toObject();delete d._id;return d;};
 const actors=()=>{
  const seen=new Set();
  return values(game.actors).concat(values(globalThis.canvas?.scene?.tokens).map(t=>t.actor)).filter(actor=>{
   const uuid=actor?.uuid;
   // Strict equality in the old findIndex never matched NaN, even to itself.
   if(uuid!==uuid)return false;
   const first=!seen.has(uuid);seen.add(uuid);
   // Null entries still reserve undefined, matching the original first-index rule.
   return !!actor&&first;
  });
 };
 const sourceToken=async(actor,message)=>{
  const scene=message.speaker?.scene,token=message.speaker?.token;
  if(scene&&token){const t=await fromUuid(`Scene.${scene}.Token.${token}`);if(t?.actor?.uuid===actor.uuid)return t;}
  const active=values(actor.getActiveTokens?.(true,true)).map(tokenDoc);return active.length===1?active[0]:null;
 };
 async function targetFor(context,maxDistance){
  const targets=await resolveMessageTargets(context.message,{fromUuid}),source=await sourceToken(context.actor,context.message);
  if(targets.length!==1||targets[0].actor.uuid===context.actor.uuid)throw Error('使用此能力时请选中一个其他生物作为目标。');
  const d=distance(source,targets[0]);if(maxDistance!==null&&(d===null||d>maxDistance))throw Error(`目标必须在${maxDistance}尺范围内，且双方Token位于同一场景。`);
  return {source,target:targets[0]};
 }
 function origin(data,actor,item,token){
  data.system.context={origin:{actor:actor.uuid,item:item.uuid,token:token?.uuid??null,rollOptions:item.getOriginData?.().rollOptions??[]},target:null,roll:null};
  data.system.start={value:now(),initiative:game.combat?.combatants?.find?.(c=>c.actor?.uuid===actor.uuid)?.initiative??null};
  return data;
 }
 async function removeForSource(kind,uuid){for(const a of actors()){const ids=values(a.items).filter(i=>own(i)?.kind===kind&&own(i).sourceActor===uuid).map(i=>i.id);if(ids.length)await a.deleteEmbeddedDocuments('Item',ids);}}
 async function executeUsage(context){
  const {actor,item,message,user,action,frequencyReceipt}=context;gm();requireOwner(actor,user);
  if(resolveAction(item)!==action)throw Error('能力来源与使用事件不匹配。');
  return queue.run(actor.uuid,async()=>{
   let delivered=false;
   try{
   if(action==='party:clue'){
    const {source,target}=await targetFor(context,null),cooldown=actor.flags?.[MODULE_ID]?.party?.clueUntil??0;
    if(cooldown>now())throw Error('线索指引尚未结束10分钟冷却。');
    const uses=item.system.frequency?.value??item.system.frequency?.max??0;
    if(!frequencyReceipt&&uses<1)throw Error('线索指引没有可用次数。');
    const data=origin(await load(PARTY_SOURCES.clueEffect),actor,item,source);
    await upsertOwnedEffect(target.actor,'clue:'+actor.uuid,data);
    delivered=true;
    if(!frequencyReceipt)await item.update({'system.frequency.value':uses-1},{[MODULE_ID]:{usageInternal:true}});
    await actor.update({[`flags.${MODULE_ID}.party.clueUntil`]:now()+600});return `已向${target.actor.name}提供下一次检定加值。`;
   }
   if(action==='party:anoint'){
    const {source,target}=await targetFor(context,5);if(target.actor.isAllyOf&&!target.actor.isAllyOf(actor))throw Error('符血点化的目标必须是盟友。');
    const data=origin(await load(PARTY_SOURCES.anointEffect),actor,item,source);
    data.flags??={};data.flags[MODULE_ID]={party:{kind:'anoint',sourceActor:actor.uuid,sourceToken:source.uuid,targetToken:target.uuid,expiresAt:now()+60}};
    await removeForSource('anoint',actor.uuid);await upsertOwnedEffect(target.actor,'anoint:'+actor.uuid,data);return `已点化${target.actor.name}，持续1分钟。`;
   }
   if(action==='party:guardian'){
    const {source,target}=await targetFor(context,5);
    if(!guardianActive(source,target))throw Error('需要举起可用的盾牌并与目标相邻。');
    const last=lastActions.get(actor.uuid);if(last?.kind!=='raise-shield')throw Error('忠诚卫士要求上一个动作是举盾；请通过角色卡或HUD举盾后使用。');
    const shield=actor.items.get(actor.attributes.shield.itemId),tower=shield?.baseType==='tower-shield'||shield?.system?.baseItem==='tower-shield';
    const data=origin({name:'忠诚卫士',type:'effect',img:item.img,system:{description:{value:''},rules:[{key:'FlatModifier',selector:'ac',type:'circumstance',value:tower?2:1,predicate:['parent:origin:shield:raised']}],duration:{value:-1,unit:'unlimited',expiry:null,sustained:false},level:{value:actor.level},traits:{value:[]},tokenIcon:{show:true}},flags:{[MODULE_ID]:{party:{kind:'guardian',sourceActor:actor.uuid,sourceToken:source.uuid,targetToken:target.uuid,shieldId:shield.id}}}},actor,item,source);
    await removeForSource('guardian',actor.uuid);await upsertOwnedEffect(target.actor,'guardian:'+actor.uuid,data);lastActions.set(actor.uuid,{kind:'guardian'});return `已守护${target.actor.name}；盾牌放下或不再相邻时自动结束。`;
   }
   if(action==='party:imperial'){
    const marks=actors().flatMap(a=>values(a.items).filter(i=>own(i)?.kind==='anoint'&&own(i).sourceActor===actor.uuid&&!i.isExpired&&own(i).expiresAt>now()).map(i=>({actor:a,item:i})));
    const choices=[{value:actor.uuid,label:`自己 · ${actor.name}`},...marks.map(m=>({value:m.actor.uuid,label:`已点化盟友 · ${m.actor.name}`}))];
    const selected=await choose({actor,user,title:'帝皇血统奥秘 · 受益者',choices});if(!selected)return '已取消血统奥秘。';
    const recipient=selected===actor.uuid?actor:marks.find(m=>m.actor.uuid===selected)?.actor;if(!recipient)throw Error('符血点化已失效。');
    const defense=await choose({actor,user,title:'帝皇血统奥秘 · 防护',choices:[{value:'ac',label:'AC +1状态加值'},{value:'saving-throw',label:'豁免 +1状态加值'}]});if(!defense)return '已取消血统奥秘。';
    if(recipient!==actor&&!values(recipient.items).some(i=>own(i)?.kind==='anoint'&&own(i).sourceActor===actor.uuid&&!i.isExpired&&own(i).expiresAt>now()))throw Error('选择期间符血点化已失效，未转交血统奥秘。');
    await castEvents.ensurePaid(context);
    const data=origin(await load(PARTY_SOURCES.imperialEffect),actor,item,await sourceToken(actor,message));
    const combat=game.combat,index=combat?.turns?.findIndex(c=>c.actor?.uuid===actor.uuid);
    if(combat?.started&&index>=0)data.system.duration.value=index>combat.turn?0:1;
    for(const rule of data.system.rules)if(rule.key==='ChoiceSet'&&rule.flag==='defense')rule.selection=defense;
    data.flags??={};data.flags.system={...data.flags.system,rulesSelections:{...data.flags.system?.rulesSelections,defense}};
    data.flags[MODULE_ID]={party:{kind:'imperial',sourceActor:actor.uuid}};
    await upsertOwnedEffect(recipient,'imperial:'+actor.uuid,data);return `已为${recipient.name}提供${defense==='ac'?'AC':'豁免'}加值，至施法者下回合开始。`;
   }
   }catch(error){
    if(action==='party:clue'&&frequencyReceipt&&!delivered)await item.update({'system.frequency.value':Math.min(item.system.frequency.max,(item.system.frequency.value??0)+1)},{[MODULE_ID]:{usageInternal:true}});
    throw error;
   }
  });
 }
 async function maintain(actor){
  if(!isActiveGM(game)||!actor?.items)return;
  return queue.run(actor.uuid,async()=>{
   for(const item of values(actor.items)){
    const repaired=buildKnownWeaknessRepair(item);if(repaired){await item.update({[`flags.${MODULE_ID}.knownWeaknessBefore`]:clone(item.system.rules),'system.rules':repaired});continue;}
    const state=own(item);if(!state)continue;
    let expired=state.expiresAt&&state.expiresAt<=now();
    if(state.kind==='guardian'){
     const s=await fromUuid(state.sourceToken),t=await fromUuid(state.targetToken);
     expired=!guardianActive(s,t)||s?.actor?.attributes?.shield?.itemId!==state.shieldId||!has(s?.actor,'guardian');
    }
    if(expired)await item.delete();
   }
   const until=actor.flags?.[MODULE_ID]?.party?.clueUntil;if(until&&until<=now()){
    const clue=values(actor.items).find(i=>getSourceId(i)===PARTY_SOURCES.clue);
    if(clue?.system.frequency)await clue.update({'system.frequency.value':clue.system.frequency.max},{[MODULE_ID]:{usageInternal:true}});
    await actor.update({[`flags.${MODULE_ID}.party.-=clueUntil`]:null});
   }
  });
 }
 function register({Hooks}={}){
  const registrations=[],on=(name,fn)=>registrations.push([name,Hooks.on(name,fn)]);
  const all=()=>{if(isActiveGM(game))for(const a of actors())maintain(a).catch(onError);};
  on('createItem',item=>{
   if(isActiveGM(game)&&item.actor&&getSourceId(item)===PARTY_SOURCES.raiseShieldEffect)lastActions.set(item.actor.uuid,{kind:'raise-shield'});
   if(item.type==='effect')maintain(item.actor).catch(onError);
  });
  on('createChatMessage',m=>{
   if(!isActiveGM(game)||m.flags?.[MODULE_ID]?.usageGenerated)return;
   const actor=m.actor;if(!actor)return;
   const source=m.item?.sourceId,context=m.flags?.pf2e?.context;
   if(m.item?.slug==='raise-a-shield'){lastActions.set(actor.uuid,{kind:'raise-shield'});return;}
   if(source===PARTY_SOURCES.guardian)return;
   if(['attack-roll','skill-check'].includes(context?.type)||m.item?.type==='spell'||['feat','action'].includes(m.item?.type))lastActions.set(actor.uuid,{kind:'other'});
  });
  on('updateToken',(token,changes)=>{if(['x','y','elevation'].some(k=>Object.hasOwn(changes,k))) {if(token.actor)lastActions.set(token.actor.uuid,{kind:'move'});all();}});
  on('deleteToken',all);on('deleteItem',all);on('updateItem',(item,changes)=>{if(item.type==='shield'||getSourceId(item)===PARTY_SOURCES.raiseShieldEffect)all();});on('updateWorldTime',all);on('updateCombat',(_combat,changes={})=>{if('round'in changes||'turn'in changes)lastActions.clear();all();});on('deleteCombat',()=>{lastActions.clear();all();});
  return()=>{for(const[n,id]of registrations)Hooks.off(n,id);};
 }
 return {resolveAction,executeUsage,maintain,register,captureUsage:(item,context)=>isImperialBloodMagic(item)?castEvents.captureUsage(item,context):null};
}
