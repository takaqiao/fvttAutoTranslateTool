import { MODULE_ID, hasSource } from './rules.mjs';
import {withDamageMessageTarget} from './damage-message-targets.mjs';
import {selectedRuneWeaponId,isCuttingWeapon} from './rune-transfer.mjs';

export const CAMPAIGN_SOURCES=Object.freeze({
 keenEye:'Compendium.pf2e.feats-srd.Item.X9UprPmeU3ovOgwb',
 walls:'Compendium.pf2e.actionspf2e.Item.IBEbFlMmDbGsrN4G',
 paragon:'Compendium.pf2e.feats-srd.Item.xOMwuKCf02aFzyp3',
 battleMedicine:'Compendium.pf2e.feats-srd.Item.wYerMk6F1RZb0Fwt',
 cutting:'Compendium.pf2e.feats-srd.Item.pe8a7WDz0MIY45uO',
 drink:'Compendium.pf2e.actionspf2e.Item.vZwa0PZLQvm3X5ME',
 barrowsEdge:'Compendium.pf2e.classfeatures.Item.LfgeEJgJdA8WAKV8',
 shatter:'Compendium.pf2e.feats-srd.Item.zbxqYhmn7KbqR2Sb',
 slam:'Compendium.pf2e.feats-srd.Item.39CqlOzlHjEhh0E4',
 crashing:'Compendium.pf2e.feats-srd.Item.yTh9QwAf0hadP91j',
 godless:'Compendium.pf2e.feats-srd.Item.tnzZvaJ97t0N9g6y',
 mortal:'Compendium.pf2e.feats-srd.Item.mEk2POFNU1Q0TQg2',
});
const OFF_GUARD='Compendium.pf2e.conditionitems.Item.AJh5ex99aV6VTggg';
const FRIGHTENED='Compendium.pf2e.conditionitems.Item.TBSHQspnbcqxsmjL';
const WALLS_EFFECT='Compendium.pf2e.feat-effects.Item.dn7P5RzMZugxiWri';
const BM_IMMUNITY='Compendium.pf2e.feat-effects.Item.2XEYQNZTCGpdkyR6';
const values=value=>Array.from(value?.values?.()??value?.contents??value??[]);
const feature=(actor,key)=>values(actor?.items).find(item=>hasSource(item,CAMPAIGN_SOURCES[key]));
const own=item=>item?.flags?.[MODULE_ID]??{};
const opts=message=>message?.flags?.pf2e?.context?.options??[];
const degree=message=>message?.rolls?.[0]?.options?.degreeOfSuccess??({criticalFailure:0,failure:1,success:2,criticalSuccess:3}[message?.flags?.pf2e?.context?.outcome]);
const tokenDoc=token=>token?.document??token;
const actorTokens=actor=>values(actor?.getActiveTokens?.(true,true)).map(tokenDoc);
const isFist=item=>item?.system?.baseItem==='fist'||item?.baseType==='fist'||item?.slug==='fist';
const traits=item=>new Set(item?.traits??item?.system?.traits?.value??[]);
const melee=item=>item?.isMelee===true||item?.system?.range===null;
const eligibleWeapon=isCuttingWeapon;
const activeGM=game=>Boolean(game?.user?.isGM&&game.user.id===game.users?.activeGM?.id);
const parentCondition=c=>c.isLocked||c.inMemoryOnly||c.system?.references?.parent?.id||c.flags?.pf2e?.grantedBy;
const conditionValue=c=>c.value??c.system?.value?.value??0;
const labels={sickened:'恶心',enfeebled:'衰弱',clumsy:'笨拙',frightened:'惊惧',stunned:'震慑',stupefied:'呆滞',drained:'流失'};
const randomId=()=>globalThis.foundry?.utils?.randomID?.()??globalThis.crypto.randomUUID().replaceAll('-','').slice(0,16);

export function paragonConditionChoices(actor,target){
 const allowed=new Set(['sickened','enfeebled','clumsy']);
 if((actor.skills?.medicine?.rank??actor.system?.skills?.med?.rank??0)>=4){allowed.add('frightened');allowed.add('stunned')}
 if(feature(actor,'godless')){allowed.add('stupefied');allowed.add('drained')}
 return values(target.conditions).filter(c=>allowed.has(c.slug)&&conditionValue(c)>0&&!parentCondition(c))
  .map(c=>({value:c.id,label:`${labels[c.slug]} ${conditionValue(c)} → ${Math.max(0,conditionValue(c)-1)}`}));
}

export function nextOwnTurnTiming(actor,combat,expiry='turn-end'){
 if(!combat?.started)return null;
 const index=combat.turns?.findIndex(c=>c.actor?.uuid===actor.uuid)??-1;
 if(index<0||combat.turns[index].initiative==null)return null;
 const rounds=index>combat.turn?0:1;
 return {combatId:combat.id,combatantId:combat.turns[index].id,endRound:combat.round+rounds,rounds,initiative:combat.turns[index].initiative,expiry};
}

function baseEffect(name,kind,rules=[],duration={value:-1,unit:'unlimited',expiry:null,sustained:false}){
 return {name,type:'effect',img:'icons/svg/aura.svg',system:{description:{value:''},level:{value:1},traits:{value:[],rarity:'common'},duration,start:{value:0,initiative:null},tokenIcon:{show:true},rules},flags:{[MODULE_ID]:{kind}}};
}

export function buildCuttingEffect({targetSignature,targetUuid,next,weaponIds=[],sourceMessageId,worldTime,initiative,rounds=1}){
 if(!targetSignature||!targetUuid||!['fist','weapon'].includes(next))throw Error('断天剑碎地拳需要明确目标及下一次攻击类型。');
 const predicate=[`target:signature:${targetSignature}`,...next==='fist'?['item:base:fist']:['item:type:weapon','item:melee',{not:'item:category:unarmed'},{or:['item:usage:hands:1','item:trait:agile','item:trait:finesse']}]];
 const data=baseEffect('断天剑·碎地拳：下一次攻击','campaign-cutting',[{key:'EphemeralEffect',selectors:['strike-attack-roll'],uuid:OFF_GUARD,predicate}],{value:rounds,unit:'rounds',expiry:'turn-end',sustained:false});
 data.system.start={value:worldTime,initiative};Object.assign(data.flags[MODULE_ID],{targetUuid,next,sourceMessageId});return data;
}

export function parseCampaignStrike(message){
 const context=message?.flags?.pf2e?.context;
 if(context?.type!=='attack-roll'||context.action!=='strike')return null;
 const prefix=`${MODULE_ID}:campaign:`;
 const marker=opts(message).find(o=>typeof o==='string'&&o.startsWith(prefix));
 if(!marker)return null;
 const [action,usageId,...rest]=marker.slice(prefix.length).split(':');
 return ['shatter','slam'].includes(action)&&/^[A-Za-z0-9_-]+$/.test(usageId??'')&&!rest.length?{action,usageId}:null;
}

export function appliedDamageAmount(receipt){
 if(!receipt||receipt.isHealing||receipt.isReverted)return 0;
 return (receipt.updates??[]).filter(u=>['system.attributes.hp.value','system.attributes.hp.temp','system.attributes.hp.sp.value'].includes(u.path))
  .reduce((sum,u)=>sum+(Number.isFinite(u.value)&&u.value>0?u.value:0),0);
}

/** Normal-use provider. All document mutations run on the elected GM. */
export function createCampaignFeats({game,fromUuid=globalThis.fromUuid,choose,onError=error=>console.error(MODULE_ID,error)}={}){
 const queues=new Map(),hooks=[];
 const serial=(key,callback)=>{const pending=(queues.get(key)??Promise.resolve()).catch(()=>{}).then(callback);queues.set(key,pending);pending.finally(()=>{if(queues.get(key)===pending)queues.delete(key)}).catch(()=>{});return pending};
 const gm=()=>{if(!activeGM(game))throw Error('此能力必须由当前主GM统一结算。')};
 const owner=(actor,user)=>{gm();if(actor?.type!=='character'||!user||!actor.testUserPermission?.(user,'OWNER'))throw Error('没有此角色的所有者权限。')};
 const pick=async(actor,user,title,choices)=>{if(!choices.length)throw Error(`${title}：没有合法选项。`);const value=choices.length===1?choices[0].value:await choose?.({actor,user,title,choices});if(!choices.some(c=>c.value===value))throw Error('已取消选择。');return value};
 const saveFlag=(doc,key,value)=>doc.update({[`flags.${MODULE_ID}.${key}`]:value});
 const upsert=async(actor,key,data)=>{
  const found=values(actor.items).filter(i=>own(i).campaignKey===key);
  if(!data){if(found.length)await actor.deleteEmbeddedDocuments('Item',found.map(i=>i.id));return}
  data.flags??={};data.flags[MODULE_ID]={...data.flags[MODULE_ID],campaignKey:key};
  if(found.length){await found[0].update(data);if(found.length>1)await actor.deleteEmbeddedDocuments('Item',found.slice(1).map(i=>i.id));return found[0]}
  return (await actor.createEmbeddedDocuments('Item',[data]))[0];
 };
 const targets=async message=>{
  const uuids=[...own(message).usageInput?.targetUuids??[]];
  const context=message?.flags?.pf2e?.context;
  if(!uuids.length){const target=context?.target?.token??context?.target?.uuid??message?.target?.token?.uuid;if(target)uuids.push(target)}
  return (await Promise.all([...new Set(uuids)].map(async uuid=>typeof uuid==='string'?fromUuid(uuid):tokenDoc(uuid)))).filter(t=>t?.actor).map(tokenDoc);
 };
 const oneTarget=async(actor,message,user,title,filter=()=>true)=>{
  const all=(await targets(message)).filter(filter);const id=await pick(actor,user,title,all.map(t=>({value:t.uuid,label:t.name??t.actor.name})));return all.find(t=>t.uuid===id);
 };
 const distance=(actor,target)=>{
  const source=actorTokens(actor)[0],a=source?.object,b=target?.object;
  return a&&b&&typeof a.distanceTo==='function'?a.distanceTo(b):null;
 };
 const requireMeleeTarget=(actor,target,item)=>{
  const d=distance(actor,target),reach=actor.getReach?.({action:'attack',weapon:item})??item?.reach??actor.system?.attributes?.reach?.general??5;
  if(d===null||d>reach)throw Error('目标不在这次近战打击触及范围内。');
 };
 const nativeEffect=async uuid=>{const item=await fromUuid(uuid);if(!item?.toObject)throw Error('无法读取原生能力效果。');const data=item.toObject();delete data._id;return data};
 const userFor=message=>message.author??game.users?.get?.(message.user?.id??message.user);
 const messageItem=async message=>message.item??(message.flags?.pf2e?.origin?.uuid?await fromUuid(message.flags.pf2e.origin.uuid):null);
 const recordResult=(message,text)=>message.update({[`flags.${MODULE_ID}.campaignResult`]:text,...own(message).usage?{[`flags.${MODULE_ID}.usage`]:{...own(message).usage,result:text}}:{}});
 const damageRollClass=()=>game.pf2e?.DamageRoll??globalThis.CONFIG?.Dice?.rolls?.find(c=>c.name==='DamageRoll');
 async function postDamage({actor,item,target,formula,options=[],checkId=null,usageId=null,outcome='success'}){
  const Roll=damageRollClass();if(!Roll)throw Error('当前系统缺少原生DamageRoll。');
  const roll=await new Roll(formula).evaluate();
  const message=await roll.toMessage(withDamageMessageTarget({speaker:globalThis.ChatMessage?.getSpeaker?.({actor,token:actorTokens(actor)[0]})??{actor:actor.id},flags:{
   pf2e:{origin:item.getOriginData?.()??{uuid:item.uuid,type:item.type,actor:actor.uuid},context:{type:'damage-roll',sourceType:'attack',domains:['damage','strike-damage'],options:[...options],outcome,target:{actor:target.actor.uuid,token:target.uuid}}},
   [MODULE_ID]:{usageGenerated:true,campaignAttackMessageId:checkId,campaignUsageId:usageId,usageInput:{targetUuids:[target.uuid]}}
  }},target.uuid));
  return {roll,message};
 }

 async function maintain(actor){
  if(!activeGM(game)||actor?.type!=='character')return;
  return serial(`maintain:${actor.uuid}`,async()=>{
   const f=feature(actor,'keenEye');
   const already=values(actor.items).some(i=>own(i).kind!=='campaign-keen-eye'&&(i.system?.rules?.some(r=>r.key==='Sense'&&r.selector==='lifesense'&&Number(r.range)>=30)||Number(i.system?.subfeatures?.senses?.lifesense?.range)>=30));
   const data=f&&!already?baseEffect('灵识：生命感知','campaign-keen-eye',[{key:'Sense',selector:'lifesense',acuity:'imprecise',range:30}]):null;
   if(data)data.system.tokenIcon.show=false;
   const old=values(actor.items).find(i=>own(i).campaignKey==='keen-eye');
   if(data&&old&&JSON.stringify(old.system?.rules)===JSON.stringify(data.system.rules))return;
   await upsert(actor,'keen-eye',data);
  });
 }

 async function walls(ctx){
  const {actor,message,user}=ctx;
  const target=await oneTarget(actor,message,user,'竖起盾墙：选择15尺内盟友',t=>t.actor.uuid!==actor.uuid&&(actor.isAllyOf?.(t.actor)??t.actor.type==='character')&&distance(actor,t)!==null&&distance(actor,t)<=15);
  const data=await nativeEffect(WALLS_EFFECT);data.system.duration={value:1,unit:'minutes',expiry:'turn-start',sustained:false};data.system.start={value:game.time.worldTime,initiative:actor.combatant?.initiative??null};
  data.flags={...data.flags,[MODULE_ID]:{kind:'campaign-walls',sourceActorUuid:actor.uuid,sourceMessageId:message.id}};
  await upsert(actor,`walls:${actor.uuid}`,structuredClone(data));await upsert(target.actor,`walls:${actor.uuid}`,structuredClone(data));
  return `已为${actor.name}与${target.actor.name}施加竖起盾墙，持续1分钟。`;
 }

 async function applyParagon(actor,target,user,message){
  owner(actor,user);
  if(!feature(actor,'paragon'))return '';
  const choices=paragonConditionChoices(actor,target);if(!choices.length)return '没有可降低的独立条件。';
  const id=await pick(actor,user,'典范战地医疗：选择减1的条件',choices);
  const condition=values(target.conditions).find(c=>c.id===id);
  if(!condition||parentCondition(condition)||!paragonConditionChoices(actor,target).some(c=>c.value===id))throw Error('所选条件已变化，未降低条件。');
  await target.decreaseCondition(condition);
  const text=`${target.name}的${labels[condition.slug]}降低1。`;if(message)await recordResult(message,text);return text;
 }

 async function battleMedicine({actor,item,message,user}){
  const target=await oneTarget(actor,message,user,'战地医疗：选择触及内目标',t=>distance(actor,t)!==null&&distance(actor,t)<=5);
  const stat=actor.skills?.medicine??actor.getStatistic?.('medicine');
  if(!stat?.check?.roll||stat.rank<1)throw Error('战地医疗需要医疗受训。');
  const immune=values(target.actor.items).find(i=>hasSource(i,BM_IMMUNITY)&&i.isExpired!==true&&(own(i).healerUuid===actor.uuid||i.system?.context?.origin?.actor===actor.uuid));
  if(immune)throw Error('目标仍然免疫你的战地医疗。');
  const rank=Number(await pick(actor,user,'战地医疗：选择DC',[15,20,30,40].slice(0,stat.rank).map((dc,index)=>({value:String(index+1),label:`DC ${dc} · ${['受训','专家','大师','传奇'][index]}`}))));
  const medic=values(actor.items).some(i=>hasSource(i,'Compendium.pf2e.feats-srd.Item.MJg24e9fJd7OASvF'));
  const bonus=[0,10,30,50][rank-1]+(medic?(rank-1)*5:0);
  let resultText='';
  const checkRoll=await stat.check.roll({dc:{value:[15,20,30,40][rank-1],visible:true},target:target.actor,skipDialog:true,
   extraRollOptions:['action:battle-medicine',`${MODULE_ID}:battle-medicine:${message.id}`],
   callback:async(roll,outcome,check)=>{
    const dos=roll.options?.degreeOfSuccess??({criticalFailure:0,failure:1,success:2,criticalSuccess:3}[outcome]);
    const immunity=await nativeEffect(BM_IMMUNITY);
    const robust=values(target.actor.items).some(i=>hasSource(i,'Compendium.pf2e.feats-srd.Item.yTLGclKtWVFZLKIz'));
    immunity._stats={...immunity._stats,compendiumSource:BM_IMMUNITY};
    immunity.system.duration={value:1,unit:robust?'hours':'days',expiry:'turn-start',sustained:false};immunity.system.start={value:game.time.worldTime,initiative:null};
    immunity.system.context={...immunity.system.context,origin:{actor:actor.uuid,item:item.uuid}};
    immunity.flags={...immunity.flags,[MODULE_ID]:{kind:'campaign-battle-medicine-immunity',healerUuid:actor.uuid,sourceMessageId:message.id}};
    await upsert(target.actor,`battle-medicine:${actor.uuid}`,immunity);
    if(dos===1){resultText='战地医疗失败，未恢复生命值；已记录免疫。';return}
    const formula=dos===0?'1d8':`(${dos===3?'4d8':'2d8'}${bonus?`+${bonus}`:''})[healing]`;
    const {roll:damage}=await postDamage({actor,item,target,formula,options:['action:battle-medicine'],usageId:message.id});
    // PF2e's native healing button passes a negative number: DamageRoll.alter
    // retains damage instances which IWR can reinterpret as positive damage.
    await target.actor.applyDamage({damage:dos>=2?-damage.total:damage,token:target,item,rollOptions:new Set(['action:battle-medicine'])});
    const paragon=dos>=2?await applyParagon(actor,target.actor,user,check):'';
    resultText=dos>=2?`战地医疗已恢复生命值并记录免疫。${paragon}`:'战地医疗大失败，已结算1d8伤害并记录免疫。';
   }
  });
  if(!checkRoll)throw Error('已取消战地医疗。');return resultText;
 }

 async function processCheck(message){
  if(!activeGM(game)||message.flags?.pf2e?.context?.type!=='attack-roll'||message.flags?.pf2e?.context?.action!=='strike')return;
  return serial('campaign-checks',async()=>{
   if(own(message).campaignCheck)return;
   const item=await messageItem(message),actor=item?.actor??message.actor,user=userFor(message);
   if(typeof item?.uuid!=='string'||!item.uuid||!actor||!user||!actor.testUserPermission?.(user,'OWNER'))return;
   if(!feature(actor,'cutting')&&!feature(actor,'barrowsEdge'))return;
   const [target]=await targets(message);if(!target)return;
   await saveFlag(message,'campaignCheck',true);
   const type=isFist(item)?'fist':eligibleWeapon(item)?'weapon':null;
   const consume=values(actor.items).filter(i=>own(i).kind==='campaign-cutting'&&own(i).targetUuid===target.actor.uuid&&own(i).next===type);
   if(consume.length)await actor.deleteEmbeddedDocuments('Item',consume.map(i=>i.id));
   if(feature(actor,'barrowsEdge'))await saveFlag(actor,'campaignLastStrike',{messageId:message.id,itemUuid:item.uuid,targetUuid:target.actor.uuid,degree:degree(message),activity:message.id,combatId:game.combat?.id??null,round:game.combat?.round??null,turn:game.combat?.turn??null});
   if(!feature(actor,'cutting')||degree(message)<2||!type)return;
   const selectedOptions=opts(message).filter(o=>typeof o==='string'&&o.startsWith(`${MODULE_ID}:cutting-weapon:`));
   const selected=selectedOptions.length===1?selectedOptions[0].slice(`${MODULE_ID}:cutting-weapon:`.length):selectedOptions.length?null:selectedRuneWeaponId(actor);
   if(type==='weapon'&&item.id!==selected)return;
   const timing=nextOwnTurnTiming(actor,game.combat);if(!timing)return;
   const weaponIds=values(actor.system?.actions).filter(s=>eligibleWeapon(s.item)).map(s=>s.item.id);
   const next=type==='weapon'?'fist':'weapon';
   const effect=buildCuttingEffect({targetSignature:target.actor.signature??target.actor.id,targetUuid:target.actor.uuid,next,weaponIds,sourceMessageId:message.id,worldTime:game.time.worldTime,initiative:timing.initiative,rounds:timing.rounds});
   Object.assign(effect.flags[MODULE_ID],{timing});await upsert(actor,`cutting:${target.actor.uuid}:${next}`,effect);
  });
 }

 async function drinkFoes({actor,item,message}){
  return serial(`drink:${actor.uuid}`,async()=>{
   const last=own(actor).campaignLastStrike;
   if(!feature(actor,'barrowsEdge')||!last||last.degree<2||last.activity!==last.messageId||last.consumed)throw Error('上一动作不是尚未使用的墓刃成功打击，或已经结算。');
   if(last.combatId&&(last.combatId!==game.combat?.id||last.round!==game.combat?.round||last.turn!==game.combat?.turn))throw Error('墓刃打击已不是本次回合的上一动作。');
   const weapon=values(actor.items).find(i=>i.uuid===last.itemUuid);
   const tags=weapon?.system?.traits?.otherTags??[];
   const edge=feature(actor,'barrowsEdge'),selected=edge?.flags?.system?.rulesSelections?.existingIkon??edge?.system?.rules?.find(r=>r.key==='ChoiceSet'&&r.flag==='existingIkon')?.selection;
   if(!weapon||!tags.includes('physical-ikon:barrows-edge')&&selected!==weapon.id)throw Error('上次武器不是绑定的墓刃。');
   const attack=game.messages.get(last.messageId),receipts=own(attack).campaignDamage??[];
   const valid=receipts.filter(r=>r.targetUuid===last.targetUuid&&!r.reverted&&r.amount>0);
   if(!valid.length)throw Error('没有该次打击的实际伤害回执；请先应用原伤害卡。');
   const amount=Math.floor(valid.reduce((sum,r)=>sum+r.amount,0)/2);
   await saveFlag(actor,'campaignLastStrike',{...last,consumed:message.id});
   if(amount>0){
    if(actor.applyDamage&&actorTokens(actor)[0])await actor.applyDamage({damage:-amount,token:actorTokens(actor)[0],item,final:true,rollOptions:new Set([`${MODULE_ID}:drink:${message.id}`])});
    else await actor.update({'system.attributes.hp.value':Math.min(actor.system.attributes.hp.max,actor.system.attributes.hp.value+amount)});
   }
   return `渴饮吾敌之血恢复${amount}点生命值（该次实际伤害的一半）。`;
  });
 }

 async function compositeStrike(ctx){
  const {actor,item,message,user}=ctx,action=ctx.action==='campaign:shatter-defenses'?'shatter':'slam';
  const target=await oneTarget(actor,message,user,'选择本次近战打击目标',t=>action!=='shatter'||t.actor.hasCondition?.('frightened'));
  const strikes=values(actor.system?.actions).filter(s=>s.ready!==false&&s.item?.isMelee===true&&s.variants?.length);
  const choices=strikes.flatMap((strike,index)=>strike.variants.map((variant,map)=>({value:`${index}:${map}`,label:`${strike.label??strike.item.name} · ${map===0?'无MAP':map===1?'第二次攻击':'第三次攻击'} ${variant.label??''}`})));
  const selection=await pick(actor,user,action==='shatter'?'粉碎防御：选择武器及当前MAP':'摔击：选择武器及当前MAP',choices);
  const [index,map]=selection.split(':').map(Number),strike=strikes[index];requireMeleeTarget(actor,target,strike.item);
  const timing=nextOwnTurnTiming(actor,game.combat,'turn-start');
  if(action==='shatter'&&!timing)throw Error('粉碎防御需要当前遭遇先攻，以追踪惊惧下限到期。');
  const marker=`${MODULE_ID}:campaign:${action}:${message.id}`;
  let state={action,actorUuid:actor.uuid,itemUuid:strike.item.uuid,targetUuid:target.actor.uuid,targetTokenUuid:target.uuid,map,checkId:null,degree:null,offGuardBefore:target.actor.hasCondition?.('off-guard')??false,timing,status:'rolling'};
  await saveFlag(message,'campaignStrike',state);
  const roll=await strike.variants[map].roll({target:target.object,options:[marker],event:{ctrlKey:false,metaKey:false,shiftKey:!!game.user.settings?.showCheckDialogs},callback:async(result,outcome,check)=>{
   const degreeValue=result.options?.degreeOfSuccess??({criticalFailure:0,failure:1,success:2,criticalSuccess:3}[outcome]);
   state={...state,checkId:check.id,degree:degreeValue,offGuardBefore:state.offGuardBefore||opts(check).includes('target:condition:off-guard'),status:degreeValue>=2?'awaiting-damage':'miss'};
   await saveFlag(message,'campaignStrike',state);
   await saveFlag(check,'campaignUsageId',message.id);
   if(degreeValue<2)return;
   const method=degreeValue===3?'critical':'damage';
   const damage=await strike[method]({target:target.object,options:[marker,`${MODULE_ID}:campaign-damage:${message.id}:${check.id}`],checkContext:check.flags?.pf2e?.context,mapIncreases:map,createMessage:true,event:{ctrlKey:false,metaKey:false,shiftKey:!!game.user.settings?.showDamageDialogs}});
   if(!damage)throw Error('原生伤害掷骰未完成。');
  }});
  if(!roll)throw Error('已取消本次打击。');
  return state.degree>=2?`${action==='shatter'?'粉碎防御':'猛烈摔击'}已命中；应用原生伤害卡时自动结算附加效果。${action==='slam'?'打击和摔绊均计入后续MAP。':''}`:'本次打击未命中，没有附加效果。';
 }

 async function linkDamageCard(message){
  if(!activeGM(game)||!message.isDamageRoll||own(message).campaignAttackMessageId)return;
  const item=await messageItem(message),actor=item?.actor??message.actor;
  // Native ChatMessage.item can be null for an expired temporary infusion.
  // A speaker and exact check marker do not establish that missing item source.
  if(typeof item?.uuid!=='string'||!item.uuid||!actor||!feature(actor,'barrowsEdge'))return;
  const [target]=await targets(message);if(!target)return;
  // The shared native prepareStrike wrapper stamps this only when checkContext is
  // the actual saved check's context object. Time/order/name similarity is insufficient.
  const prefix=`${MODULE_ID}:bear-attack:`,ids=[...new Set(opts(message).filter(o=>typeof o==='string'&&o.startsWith(prefix)).map(o=>o.slice(prefix.length)))];
  if(ids.length!==1)return;
  const check=game.messages.get(ids[0]);
  if(check?.flags?.pf2e?.context?.type!=='attack-roll'||check.flags.pf2e.context.action!=='strike'||![2,3].includes(degree(check)))return;
  const checkItem=await messageItem(check),[checkTarget]=await targets(check);
  if(checkItem?.uuid!==item.uuid||checkTarget?.uuid!==target.uuid||(checkItem.actor??check.actor)?.uuid!==actor.uuid)return;
  await saveFlag(message,'campaignAttackMessageId',check.id);
 }

 async function processDamageReceipt(message){
  if(!activeGM(game)||message.flags?.pf2e?.context?.type!=='damage-taken')return;
  return serial(`receipt:${message.id}`,async()=>{
   if(own(message).campaignReceipt)return;
   const sourceOption=opts(message).find(o=>typeof o==='string'&&o.startsWith(`${MODULE_ID}:source:`));
   if(!sourceOption)return;
   const [sourceId]=sourceOption.slice(`${MODULE_ID}:source:`.length).split(':'),source=game.messages.get(sourceId);
   if(!source?.isDamageRoll)return;
   await linkDamageCard(source);
   const checkId=own(source).campaignAttackMessageId;if(!checkId)return;
   const check=game.messages.get(checkId);if(!check)return;
   const sourceItem=await messageItem(source),actor=sourceItem?.actor??source.actor;
   if(typeof sourceItem?.uuid!=='string'||!sourceItem.uuid||!actor)return;
   const [target]=await targets(source);if(!target)return;
   const author=userFor(message);
   if(!author||!author.isGM&&!target.actor.testUserPermission?.(author,'OWNER'))return;
   const receipt=message.flags.pf2e.appliedDamage;
   if(receipt?.uuid&&receipt.uuid!==target.actor.uuid)return;
   if(message.speaker?.actor&&message.speaker.actor!==target.actor.id)return;
   await saveFlag(message,'campaignReceipt',true);
   const amount=appliedDamageAmount(receipt);
   const old=own(check).campaignDamage??[];
   if(!old.some(r=>r.receiptId===message.id))await saveFlag(check,'campaignDamage',[...old,{receiptId:message.id,amount,targetUuid:target.actor.uuid,reverted:!!receipt?.isReverted}]);
   const usageId=own(source).campaignUsageId,usage=usageId?game.messages.get(usageId):null,state=own(usage).campaignStrike;
   if(!state||state.status!=='awaiting-damage'||state.checkId!==check.id||state.actorUuid!==actor.uuid||state.itemUuid!==sourceItem.uuid||state.targetUuid!==target.actor.uuid||state.degree<2)return;
   if(state.action==='shatter'&&amount<=0)return;
   await saveFlag(usage,'campaignStrike',{...state,status:'done',receiptId:message.id});
   if(state.action==='shatter'){
    const effect=baseEffect('粉碎防御：措手不及','campaign-shatter',[{key:'GrantItem',uuid:OFF_GUARD,inMemoryOnly:true,predicate:['self:condition:frightened']}]);
    effect.flags[MODULE_ID]={kind:'campaign-shatter',sourceActorUuid:actor.uuid,sourceMessageId:usage.id};
    await upsert(target.actor,`shatter:${actor.uuid}`,effect);
    if(state.offGuardBefore||opts(message).includes(`${MODULE_ID}:offguard-before-damage`)){
     const timing=state.timing,data=baseEffect('粉碎防御：惊惧下限1','campaign-frightened-floor',[{key:'GrantItem',uuid:FRIGHTENED,inMemoryOnly:true}],{value:timing.rounds,unit:'rounds',expiry:'turn-start',sustained:false});
     data.system.start={value:game.time.worldTime,initiative:timing.initiative};data.flags[MODULE_ID]={kind:'campaign-frightened-floor',timing,sourceActorUuid:actor.uuid,sourceMessageId:usage.id};
     await upsert(target.actor,`shatter-floor:${actor.uuid}`,data);
    }
    await recordResult(usage,'粉碎防御已使目标措手不及，持续至惊惧结束。');
   }else if(state.action==='slam'){
    if(!target.actor.hasCondition?.('prone'))await target.actor.increaseCondition('prone');
    const die=sourceItem.system?.equipped?.handsHeld===2?sourceItem.system?.damage?.die:'d6';
    const {roll}=await postDamage({actor,item:feature(actor,'crashing'),target,formula:`1${/^d(?:4|6|8|10|12)$/.test(die)?die:'d6'}[bludgeoning]`,options:[`${MODULE_ID}:trip:${usage.id}`],usageId:usage.id});
    await target.actor.applyDamage({damage:roll,token:target,item:feature(actor,'crashing'),rollOptions:new Set(['item:trait:attack',`${MODULE_ID}:trip:${usage.id}`])});
    await recordResult(usage,'猛烈摔击已自动绊倒目标并结算绊摔大成功的单骰伤害。');
   }
  });
 }

 async function beforeDamage(actor,params){
  if(!actor.hasCondition?.('off-guard'))return null;
  return {params:{...params,rollOptions:new Set([...params.rollOptions??[],`${MODULE_ID}:offguard-before-damage`])}};
 }

 async function cleanEffects(actor){
  if(!activeGM(game)||!actor?.items)return;
  return serial(`clean:${actor.uuid}`,async()=>{
  const combat=game.combat;
  const expired=values(actor.items).filter(item=>{
   const data=own(item),timing=data.timing;
   if(!timing||!['campaign-cutting','campaign-frightened-floor'].includes(data.kind))return false;
   const index=combat?.turns?.findIndex(c=>c.id===timing.combatantId)??-1;
   return !combat?.started||combat.id!==timing.combatId||index<0||combat.round>timing.endRound||(combat.round===timing.endRound&&(timing.expiry==='turn-start'?combat.turn>=index:combat.turn>index));
  });
  if(expired.length)await actor.deleteEmbeddedDocuments('Item',expired.map(i=>i.id));
  if(!actor.hasCondition?.('frightened')){
   const remove=values(actor.items).filter(i=>own(i).kind==='campaign-shatter');
   if(remove.length)await actor.deleteEmbeddedDocuments('Item',remove.map(i=>i.id));
  }
  });
 }

 async function recordOtherAction(message){
  if(!activeGM(game)||message.isRoll||message.rolls?.length||own(message).usageGenerated)return;
  const item=await messageItem(message),actor=item?.actor;
  if(!actor||!feature(actor,'barrowsEdge')||hasSource(item,CAMPAIGN_SOURCES.drink)||!['feat','action','spell'].includes(item.type))return;
  const last=own(actor).campaignLastStrike;
  if(last&&last.activity===last.messageId)await saveFlag(actor,'campaignLastStrike',{...last,activity:message.id});
 }

 async function undoReceipt(message){
  if(!activeGM(game)||!message.flags?.pf2e?.appliedDamage?.isReverted)return;
  for(const check of values(game.messages)){
   const receipts=own(check).campaignDamage;
   if(!receipts?.some(r=>r.receiptId===message.id&&!r.reverted))continue;
   await saveFlag(check,'campaignDamage',receipts.map(r=>r.receiptId===message.id?{...r,reverted:true}:r));
  }
  for(const usage of values(game.messages)){
   const state=own(usage).campaignStrike;if(state?.receiptId!==message.id||state.status!=='done')continue;
   if(state.action==='shatter'){
    const target=await fromUuid(state.targetTokenUuid);
    const effects=values(target?.actor?.items).filter(i=>own(i).sourceMessageId===usage.id&&['campaign-shatter','campaign-frightened-floor'].includes(own(i).kind));
    if(effects.length)await target.actor.deleteEmbeddedDocuments('Item',effects.map(i=>i.id));
   }
   await saveFlag(usage,'campaignStrike',{...state,status:'reverted'});
   await recordResult(usage,state.action==='slam'?'原打击伤害已撤销；绊摔的独立伤害请使用其原生撤销，俯卧请按后续行动核对。':'原打击伤害已撤销，粉碎防御的附加效果已清理。');
  }
 }

 function resolveAction(item){
  if(hasSource(item,CAMPAIGN_SOURCES.walls))return 'campaign:raise-walls';
  if(hasSource(item,CAMPAIGN_SOURCES.drink))return 'campaign:drink-foes';
  if(hasSource(item,CAMPAIGN_SOURCES.shatter))return 'campaign:shatter-defenses';
  if(hasSource(item,CAMPAIGN_SOURCES.slam)&&feature(item.actor,'crashing'))return 'campaign:crashing-slam';
  if(hasSource(item,CAMPAIGN_SOURCES.battleMedicine)&&feature(item.actor,'paragon'))return 'campaign:battle-medicine';
  return null;
 }

 async function executeUsage(ctx){
  owner(ctx.actor,ctx.user);
  if(resolveAction(ctx.item)!==ctx.action)throw Error('技能来源与自动化路由不符。');
  if(ctx.action==='campaign:raise-walls')return walls(ctx);
  if(ctx.action==='campaign:drink-foes')return drinkFoes(ctx);
  if(['campaign:shatter-defenses','campaign:crashing-slam'].includes(ctx.action))return compositeStrike(ctx);
  if(ctx.action==='campaign:battle-medicine')return battleMedicine(ctx);
  throw Error('此能力的自动结算尚未就绪。');
 }
 function register({Hooks,onError:report=onError}={}){
  const on=(name,callback)=>hooks.push([name,Hooks.on(name,callback)]);
  const safe=callback=>(...args)=>{try{Promise.resolve(callback(...args)).catch(report)}catch(e){report(e)}};
  on('createChatMessage',safe(async message=>{await recordOtherAction(message);await processCheck(message);await linkDamageCard(message);await processDamageReceipt(message)}));
  on('updateChatMessage',safe(undoReceipt));
  on('preCreateChatMessage',message=>{
   if(parseCampaignStrike(message))message.updateSource({'flags.xdy-pf2e-workbench.noAutoDamageRoll':true});
   const prefix=`${MODULE_ID}:campaign-damage:`,markers=opts(message).filter(o=>typeof o==='string'&&o.startsWith(prefix));
   if(message.flags?.pf2e?.context?.type!=='damage-roll'||markers.length!==1)return;
   const[usageId,checkId,...extra]=markers[0].slice(prefix.length).split(':'),usage=game.messages.get(usageId),state=own(usage).campaignStrike;
   if(extra.length||state?.checkId!==checkId||state.itemUuid!==message.flags?.pf2e?.origin?.uuid||state.actorUuid!==message.actor?.uuid)return;
   const target=message.flags.pf2e.context.target;
   if(!state.targetTokenUuid||target?.token!==state.targetTokenUuid||target?.actor!==state.targetUuid)return;
   const bound=withDamageMessageTarget({flags:message.flags},state.targetTokenUuid);
   message.updateSource({[`flags.${MODULE_ID}.usageGenerated`]:true,[`flags.${MODULE_ID}.campaignAttackMessageId`]:checkId,[`flags.${MODULE_ID}.campaignUsageId`]:usageId,'flags.pf2e-toolbelt.targetHelper.targets':bound.flags['pf2e-toolbelt'].targetHelper.targets});
  });
  const maintenance=safe(async item=>{if(!activeGM(game)||!item?.actor)return;await maintain(item.actor);await cleanEffects(item.actor)});
  on('createItem',maintenance);on('updateItem',maintenance);on('deleteItem',maintenance);
  const allActors=()=>[...values(game.actors),...values(game.scenes).flatMap(scene=>values(scene.tokens).map(t=>t.actor).filter(Boolean))];
  on('updateCombat',safe(async()=>{for(const actor of new Map(allActors().map(a=>[a.uuid,a])).values())await cleanEffects(actor)}));
  on('deleteCombat',safe(async()=>{for(const actor of new Map(allActors().map(a=>[a.uuid,a])).values())await cleanEffects(actor)}));
  return ()=>{for(const [name,id]of hooks.splice(0))Hooks.off(name,id)};
 }
 return {resolveAction,executeUsage,register,maintain,processCheck,applyParagon,processDamageReceipt,beforeDamage,cleanEffects,linkDamageCard};
}
