import {MODULE_ID} from './rules.mjs';
import {withDamageMessageTarget} from './damage-message-targets.mjs';
import {getSourceId,isActiveGM,resolveMessageTargets} from './native-context.mjs';
import {SerialActions} from './runtime.mjs';
import {getNativeCastEvents} from './amp-cast-events.mjs';
import {preserveDamagePartForMerge,preserveMergedDamageBypass} from './native-damage-components.mjs';
import {isCuttingWeapon} from './rune-transfer.mjs';
import {isActualUseMessage} from './usage-events.mjs';
import {createAttackSequence} from './activity-attack-sequence.mjs';
export {preserveDamagePartForMerge} from './native-damage-components.mjs';

export const SPELL_COMBINATION_SOURCES=Object.freeze({
 strike:'Compendium.pf2e.actionspf2e.Item.QDW9H8XLIjuW2fE4',
 swipe:'Compendium.pf2e.feats-srd.Item.Fs88vjez9px2mmrC',
 combination:'Compendium.pf2e.actionspf2e.Item.zUWj4zmBNWOTzeFJ',
 disintegrate:'Compendium.pf2e.spells-srd.Item.r7ihOgKv19eJQnik',
 ignition:'Compendium.pf2e.spells-srd.Item.6DfLZBl8wKIV03Iq',
 needleDarts:'Compendium.pf2e.spells-srd.Item.iYRDFxeVpJ5KIjmr',
});
const S=SPELL_COMBINATION_SOURCES,values=c=>Array.from(c?.values?.()??c??[]),own=d=>d?.flags?.[MODULE_ID]??{};
const Message=()=>globalThis.CONFIG?.ChatMessage?.documentClass??globalThis.ChatMessage;
const clone=data=>globalThis.foundry?.utils?.deepClone?.(data)??structuredClone(data);
const escape=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const hit=outcome=>['success','criticalSuccess'].includes(outcome);
const hasSpellstrike=actor=>values(actor?.items).some(i=>getSourceId(i)===S.strike);
const isConflux=item=>item?.type==='spell'&&hasSpellstrike(item.actor)&&item.system.traits.value.includes('focus')&&item.system.traits.value.includes('magus')&&/[1-3]/.test(item.system.time?.value??'');
const actionKind=item=>['action','feat'].includes(item?.type)?['strike','swipe','combination'].find(kind=>getSourceId(item)===S[kind]):null;
const hasAttackOrSave=spell=>spell.isAttack||spell.system.traits?.value.includes('attack')||!!spell.system.defense?.save;
const hasEligibleTime=spell=>/^[12]$|^[12] (?:or|to) [23]$/.test(String(spell.system.time?.value??''));
const eligibleSpell=spell=>spell?.type==='spell'&&hasAttackOrSave(spell)&&hasEligibleTime(spell);
const canAffectSeveral=spell=>{
 if(spell.system.area)return true;
 const count=String(spell.system.target?.value??'').split(/creature|target|生物|目标/i)[0];
 return /\b(all|each|any number of)\b|所有|任意数量/i.test(count)||[...count.matchAll(/\d+/g)].some(m=>Number(m[0])>=2);
};
const melee=strike=>strike?.type==='strike'&&strike.ready!==false&&strike.item?.type==='weapon'&&strike.item.isMelee;
const unarmed=item=>(item.category??item.system.category)==='unarmed';
const fist=strike=>melee(strike)&&unarmed(strike.item)&&['fist','basic-unarmed'].includes(strike.item.slug??strike.item.system.slug);
const held=strike=>melee(strike)&&!unarmed(strike.item)&&strike.item.system.equipped?.carryType==='held'&&strike.item.system.equipped.handsHeld>0;
const allowedCombinationWeapon=strike=>held(strike)&&isCuttingWeapon(strike.item);
const strikeKey=strike=>`${strike.item.id}:${strike.item.altUsageType??''}`;
const skipEvent=(game,kind)=>({ctrlKey:false,metaKey:false,shiftKey:game.user.settings?.[kind==='attack'?'showCheckDialogs':'showDamageDialogs']??true});

export function criticalSpellPersistentFormula(spell){
 if(getSourceId(spell)===S.ignition)return `${spell.rank}d${spell.system.range?.value==='touch'?6:4}[persistent,fire]`;
 if(getSourceId(spell)===S.needleDarts)return `${spell.rank}[persistent,bleed]`;
 return null;
}

/** DamageRoll.alter(2) normally marks a critical hit. A critically failed basic
 * save doubles damage without granting critical-hit immunity that protection. */
export function scaleSpellDamage(roll,multiplier,{critical=false}={}){
 if(multiplier===1)return roll;
 const scaled=roll.alter(multiplier,0);
 if(multiplier!==2||critical||!scaled.instances.length)return scaled;
 const data=clone(scaled.toJSON());
 for(const instance of data.terms?.[0]?.rolls??[])for(const term of instance.terms??[])if(term.class==='ArithmeticExpression'&&term.operator==='*'&&term.operands?.[0]?.number===2){
  // PF2e recognizes a left-hand literal 2 as critical doubling regardless of
  // options.crit. Reuse the evaluated base twice as addition: no reroll and no
  // accidental critical-hit exemption on a basic save's failure multiplier.
  const base=term.operands[1];term.operator='+';term.operands=[base,clone(base)];if(term.options)delete term.options.crit;
 }
 const D=globalThis.CONFIG.Dice.rolls.find(c=>c.name==='DamageRoll');return D.fromData(data);
}

/** Source based activities; normal player choices finish before any resource or die is spent. */
export function createSpellCombination({game,fromUuid=globalThis.fromUuid,choose,onError=()=>{},afterAttack=async()=>{},nativeCasts=getNativeCastEvents({game,fromUuid})}={}){
 const queue=new SerialActions(),damageContexts=new WeakMap();
 const requireGM=()=>{if(!isActiveGM(game))throw Error('主 GM 已交接；旧客户端停止组合活动，已发生的消耗与攻击不会重试。');};
 const resolveAction=item=>{
  const kind=actionKind(item);if(kind)return `spell-combination:${kind}`;
  if(item?.type==='action'&&own(item).spellstrikeRecharge===true&&hasSpellstrike(item.actor))return 'spell-combination:recharge';
  return isConflux(item)?'spell-combination:conflux':null;
 };
 const requiresActualUse=(item,action)=>action==='spell-combination:combination'&&actionKind(item)==='combination';
 nativeCasts.addMatcher(isConflux);
 nativeCasts.addActorMatcher?.(hasSpellstrike);
 nativeCasts.addActivityMatcher?.(item=>['strike','swipe'].includes(actionKind(item)));
 const captureUsage=(item,context)=>nativeCasts.captureUsage(item,context);
 async function select(actor,user,title,choices){
  requireGM();
  if(!choices.length)throw Error(`${title}：没有合法选项。`);
  if(choices.length===1)return choices[0].value;
  const result=await choose({actor,user,title,choices});
  requireGM();
  if(result===null||result===undefined)return null;
  if(!choices.some(c=>c.value===result))throw Error('组合活动选择已失效。');return result;
 }
 function assertUse(actor,item,message,user,action){
  if(!isActiveGM(game)||!actor?.testUserPermission(user,'OWNER')||item?.actor?.uuid!==actor.uuid||resolveAction(item)!==action||game.messages.get(message?.id)!==message||(message.author?.id??message.user?.id??message.user)!==user.id||message.flags?.pf2e?.origin?.uuid!==item.uuid)throw Error('无权执行此组合活动。');
  if(requiresActualUse(item,action)&&!isActualUseMessage(message))throw Error('神威连击需要从原技能的实际使用入口执行。');
 }
 async function sourceToken(actor,message,target){
  if(message.speaker?.scene&&message.speaker?.token){
   const doc=await fromUuid(`Scene.${message.speaker.scene}.Token.${message.speaker.token}`);
   if(doc?.actor?.uuid===actor.uuid&&doc.parent?.id===target.parent?.id)return doc;
   throw Error('组合活动的来源 Token 不匹配。');
  }
  const docs=values(actor.getActiveTokens?.(true,true)).map(t=>t.document??t).filter(t=>t.actor?.uuid===actor.uuid&&t.parent?.id===target.parent?.id);
  if(docs.length!==1)throw Error('无法唯一确定组合活动来源 Token；请从场景角色使用。');return docs[0];
 }
 function requireSceneTarget(actor,origin,target){
  // Preserve the original native Strike recipient without policing spatial legality.
  const scene=origin.parent;
  if(origin.actor?.uuid!==actor.uuid||!scene||game.scenes?.get(scene.id)!==scene||scene.tokens?.get(origin.id)!==origin||target.parent!==scene||scene.tokens.get(target.id)!==target||!target.object||!target.actor)throw Error('组合活动的来源或场景目标已改变。');
 }
 const strikes=actor=>values(actor.system.actions).flatMap(s=>[s,...s.altUsages??[]]).filter(melee);
 async function spellChoices(actor){
  const entries=values(actor.spellcasting?.contents).filter(e=>e.type==='spellcastingEntry');
  const available=[];
  for(const entry of entries){
   const data=await entry.getSheetData();
   for(const group of data.groups??[]){
    if(group.uses?.value===0&&!(entry.isPrepared&&!entry.isFlexible)&&group.id!=='cantrips')continue;
    for(const [slotId,slot] of (group.active??[]).entries()){
     const spell=slot?.spell;if(!spell||slot.expended&&!spell.atWill||!hasAttackOrSave(spell))continue;
     const rank=slot.castRank??group.maxRank??spell.rank;
     const variants=values(spell.overlays).filter(o=>o.overlayType==='override');
     const hasLegalVariant=variants.some(o=>eligibleSpell(spell.loadVariant({castRank:rank,overlayIds:[o._id??[...spell.overlays.entries()].find(([,v])=>v===o)?.[0]]})));
     if(!eligibleSpell(spell)&&!hasLegalVariant)continue;
     available.push({entry,spell,rank,slotId:entry.isPrepared&&!entry.isFlexible&&group.id!=='cantrips'?slotId:null,key:`${entry.id}:${spell.id}:${rank}:${slotId}`,label:`${spell.name} · ${rank}环（${entry.name??'施法栏'}${entry.isPrepared&&!entry.isFlexible&&group.id!=='cantrips'?`，第${slotId+1}位`:''}）`});
    }
   }
  }
  return available;
 }
 async function chooseSpell(actor,user){
  const available=await spellChoices(actor),key=await select(actor,user,'选择灌注的法术与法术位',available.map(s=>({value:s.key,label:s.label})));
  if(key===null)return null;
  const choice=available.find(s=>s.key===key),overlays=values(choice.spell.overlays).filter(o=>o.overlayType==='override');
  let selected=choice.spell.loadVariant?.({castRank:choice.rank})??choice.spell;
  if(overlays.length){
   const variants=overlays.map(o=>{const id=o._id??[...choice.spell.overlays.entries()].find(([,v])=>v===o)[0];return {id,sort:o.sort??0,item:choice.spell.loadVariant({castRank:choice.rank,overlayIds:[id]})}}).filter(v=>eligibleSpell(v.item)).sort((a,b)=>a.sort-b.sort);
   // Native variants may share their spell's name; action time distinguishes
   // one-action damage from the two-action version without rewriting formulas.
   const id=await select(actor,user,'选择法术变体',variants.map(v=>({value:v.id,label:`${v.item.name} · ${String(v.item.system.time.value).replace(/\bor\b/g,'或').replace(/\bto\b/g,'至')} 动作`})));
   if(id===null)return null;selected=variants.find(v=>v.id===id).item;
  }
  if(!eligibleSpell(selected))throw Error('所选法术不满足一或二动作及攻击或豁免要求。');
  return {...choice,spell:selected};
 }
 async function record(message,state,extra={}){requireGM();await message.update({[`flags.${MODULE_ID}.spellCombinationUse`]:{...own(message).spellCombinationUse,state,...extra}});requireGM();}
 async function attack(actor,strike,target,map,message,index,kind,sequence){
  requireGM();
  let created;
  if(kind!=='combination'){
   const infused=actor.clone({items:[...clone(actor._source.items),{_id:globalThis.foundry?.utils?.randomID?.()??'ComboArcane00001',name:'Spellstrike infusion',type:'effect',system:{duration:{value:-1,unit:'unlimited'},rules:[{key:'AdjustStrike',mode:'add',property:'traits',value:'arcane',definition:[`item:id:${strike.item.id}`]},{key:'AdjustStrike',mode:'add',property:'weapon-traits',value:'magical',definition:[`item:id:${strike.item.id}`]}]}}]},{keepId:true});
   strike=strikes(infused).find(s=>strikeKey(s)===strikeKey(strike));if(!strike)throw Error('无法构建灌注奥术能量的原生打击。');
  }
  const frame=sequence.begin(strike,target);
  const options=new Set([`action:${kind==='combination'?'overwhelming-combination':kind==='swipe'?'spell-swipe':'spellstrike'}`,`${MODULE_ID}:spell-combination:${message.id}`,...frame.attackOptions]);
  if(kind==='combination')options.add('overwhelming-combination');else{options.add('arcane');options.add('magical');options.add('item:trait:magical');}
  if(kind==='swipe'&&strike.item.system.traits.value.includes('sweep'))options.add('sweep-bonus');
  const check=await strike.variants[map].roll({target:target.object,options,event:skipEvent(game,'attack'),createMessage:false,callback:async(_roll,_outcome,raw)=>{
   requireGM();
   const data=raw.toObject();delete data._id;
   data.author=message.author?.id??message.user?.id??message.user;
   data.flags={...data.flags,'xdy-pf2e-workbench':{...data.flags?.['xdy-pf2e-workbench'],noAutoDamageRoll:true},[MODULE_ID]:{...data.flags?.[MODULE_ID],usageGenerated:true,spellCombinationAttack:{activityMessageId:message.id,index,kind}}};
   created=await Message().create(data);
   requireGM();
  }});
  requireGM();
  if(!check||!created)throw Error('组合活动的原生攻击未完成，已发生的攻击不会重试。');
  frame.capture(created);
  await afterAttack(created);
  requireGM();
  const outcome=created.flags.pf2e.context.outcome;
  sequence.record(frame,outcome);
  await frame.consume();
  requireGM();
  return {strike,target,map,message:created,outcome,frame};
 }
 async function weaponDamage(attack){
  requireGM();
  const {strike,target,map,message,outcome}=attack;if(!hit(outcome))return null;
  const {strike:damageStrike,options:sequenceOptions}=attack.frame.damage(strike);
  const options=new Set([`${MODULE_ID}:bear-attack:${message.id}`,...sequenceOptions]);
  const result=await damageStrike[outcome==='criticalSuccess'?'critical':'damage']({target:target.object,checkContext:message.flags.pf2e.context,mapIncreases:map,options,event:skipEvent(game,'damage'),createMessage:false});
  requireGM();
  if(!result)throw Error('攻击已发生，但原生武器伤害尚未完成。');
  damageContexts.set(result,{...clone(message.flags.pf2e.context),sourceType:'attack',domains:['damage','strike-damage'],options:[...attack.frame.damageOptions([...(message.flags.pf2e.context.options??[]),...strike.item.getRollOptions?.('item')??[],...strike.item.actor.getRollOptions?.(['damage','strike-damage'])??[],...options])]});
  return result;
 }
 async function publishSpell({actor,user,message,choice,payment,targets,kind}){
  requireGM();
  const raw=await choice.spell.toMessage(null,{create:false,data:{castRank:choice.rank}}),data=raw.toObject();delete data._id;
  requireGM();
  data.author=user.id;data.flags??={};data.flags[MODULE_ID]={...data.flags[MODULE_ID],...payment.flags,usageInput:{...data.flags[MODULE_ID]?.usageInput,targetUuids:targets.map(t=>t.uuid)},spellCombination:{activityMessageId:message.id,kind}};
  data.flags.pf2e??={};data.flags.pf2e.origin={...data.flags.pf2e.origin,uuid:choice.spell.uuid,type:'spell',actor:actor.uuid,castRank:choice.rank};
  const card=await Message().create(data);
  requireGM();
  await nativeCasts.ensurePaid({actor,item:choice.spell,message:card,user});
  requireGM();
  return card;
 }
 async function save(spell,target,attack,card){
  requireGM();
  const defense=spell.system.defense.save,dc=spell.spellcasting?.statistic?.getChatData({item:spell})?.dc?.value;
  const adjust=getSourceId(spell)===S.disintegrate&&attack.outcome==='criticalSuccess';
  const marker=`${MODULE_ID}:spell-combination-save:${card.id}`;
  const roller=adjust?target.actor.clone({items:[...clone(target.actor._source.items),{_id:globalThis.foundry?.utils?.randomID?.()??'ComboSave0000001',name:spell.name,type:'effect',system:{duration:{value:-1,unit:'unlimited'},rules:[{key:'AdjustDegreeOfSuccess',selector:'saving-throw',predicate:[marker],adjustment:{all:'one-degree-worse'}}]}}]},{keepId:true}):target.actor;
  const statistic=roller.getStatistic(defense.statistic);if(!statistic?.check||!Number.isFinite(dc))throw Error('无法确定该法术的原生豁免与 DC。');
  let outcome;
  const result=await statistic.check.roll({origin:spell.actor,item:spell,token:target,dc:{value:dc},extraRollOptions:[...spell.getRollOptions?.('item')??[],...(defense.basic?['damaging-effect']:[]),marker],skipDialog:true,createMessage:false,callback:async(_roll,result,raw)=>{
    requireGM();
    outcome=result;const data=raw.toObject();delete data._id;
    data.flags??={};data.flags[MODULE_ID]={...data.flags[MODULE_ID],usageGenerated:true,spellCombinationSave:{spellMessageId:card.id,targetUuid:target.uuid,activityMessageId:own(card).spellCombination.activityMessageId}};
    await Message().create(data);
    requireGM();
   }});
  requireGM();
  if(!result||!['criticalFailure','failure','success','criticalSuccess'].includes(outcome))throw Error('原生豁免尚未完成，不会重复投骰。');
  return outcome;
 }
 async function spellDamage(spell,target,outcome,{saveOutcome,shared}={}){
  requireGM();
  const native=shared?.native??await spell.getDamage({target,skipDialog:true});if(!native)return null;
  requireGM();
  let damage=native.template.damage.roll;if(!damage)throw Error('该法术没有可识别的原生伤害骰。');
  damage=shared?.roll??await damage.evaluate();if(shared){shared.native=native;shared.roll=damage;}
  requireGM();
  const multiplier=saveOutcome?{criticalSuccess:0,success:0.5,failure:1,criticalFailure:2}[saveOutcome]:outcome==='criticalSuccess'?2:1;
  if(multiplier===0)return null;
  if(multiplier!==1)damage=scaleSpellDamage(damage,multiplier,{critical:!saveOutcome&&outcome==='criticalSuccess'});
  damageContexts.set(damage,{...native.context,options:[...native.context.options??[]],outcome:saveOutcome??outcome});
  return damage;
 }
 async function damageCard({actor,message,target,parts,attacks,kind}){
  requireGM();
  if(!parts.length)return;
  const docs=[];
  for(const part of parts){
   preserveDamagePartForMerge(part.roll);
   const context=damageContexts.get(part.roll)??{options:part.item.getRollOptions?.('item')??[]};
   const data=await part.roll.toMessage({speaker:message.speaker,flavor:escape(part.item.name),flags:{pf2e:{origin:{uuid:part.item.uuid,type:part.item.type,actor:actor.uuid},context:{type:'damage-roll',sourceType:context.sourceType??(part.item.type==='spell'&&part.item.system.defense?.save?'save':'attack'),outcome:context.outcome??'success',domains:context.domains??['damage'],options:[...context.options??[]],target:{actor:target.actor.uuid,token:target.uuid}}}}},{create:false});
   requireGM();
   docs.push(new (Message())(typeof data.toObject==='function'?data.toObject():data));
  }
  let combined=docs[0];for(const next of docs.slice(1)){requireGM();combined=await game.toolbelt.api.betterChat.mergeDamageMessages(combined,next,{updateMessages:false});requireGM();if(!combined)throw Error('原生伤害合并未完成。');}
  if(docs.length>1)preserveMergedDamageBypass(combined.rolls[0],parts.map(part=>part.roll));
  if(attacks.some(a=>a.outcome==='criticalSuccess'))combined.rolls[0].options.degreeOfSuccess=3;
  const data=combined.toObject();delete data._id;data.flags??={};data.flags.pf2e??={};
  // Persist corrections to derived rolls: Foundry toObject() reads _source.
  data.rolls=combined.rolls.map(roll=>roll.toJSON());
  const markers=attacks.filter(a=>hit(a.outcome)).map(a=>`${MODULE_ID}:bear-attack:${a.message.id}`);
  if(attacks.some(a=>a.outcome==='criticalSuccess'))markers.push('check:outcome:critical-success');
  data.flags.pf2e.context={...data.flags.pf2e.context,type:'damage-roll',sourceType:'attack',outcome:'success',options:[...new Set([...(data.flags.pf2e.context?.options??[]),...parts.flatMap(p=>damageContexts.get(p.roll)?.options??p.item.getRollOptions?.('item')??[]),...markers])],target:{actor:target.actor.uuid,token:target.uuid}};
  data.flags[MODULE_ID]={...data.flags[MODULE_ID],usageGenerated:true,spellCombinationDamage:{activityMessageId:message.id,kind,targetUuid:target.uuid,attacks:attacks.map(a=>({messageId:a.message.id,weaponUuid:a.strike.item.uuid,outcome:a.outcome})),parts:parts.map(p=>({itemUuid:p.item.uuid,total:p.roll.total}))}};
  data.flavor=`<h4>${escape(kind==='combination'?'神威连击':kind==='swipe'?'法术横扫':'法术打击')}：${escape(target.name??target.actor.name??'目标')} · 合并伤害</h4>${data.flavor??''}`;
  await Message().create(withDamageMessageTarget(data,target.uuid));
  requireGM();
 }
 async function executeUsage({actor,item,message,user,action}){
  assertUse(actor,item,message,user,action);
  return queue.run(actor.uuid,async()=>{
   assertUse(actor,item,message,user,action);
   const prior=own(message).spellCombinationUse;
   if(prior?.state==='done')return '本次组合活动已经结算。';
   if(prior)throw Error('本次组合活动已开始但未完成；为避免重复消耗与攻击，不会自动重试。');
   const kind=action.split(':')[1];
   if(kind==='recharge'||kind==='conflux'){
    if(kind==='conflux')await nativeCasts.ensurePaid({actor,item,message,user});
    requireGM();
    await actor.update({[`flags.${MODULE_ID}.spellstrike`]:{charged:true,messageId:message.id}});await record(message,'done');return '法术打击已充能。';
   }
   if(!game.modules.get('pf2e-toolbelt')?.active||typeof game.toolbelt?.api?.betterChat?.mergeDamageMessages!=='function')throw Error('需要已启用的 Toolbelt 原生伤害合并接口，尚未攻击。');
   if(kind!=='combination'&&own(actor).spellstrike?.charged===false)throw Error('法术打击尚未充能；请使用充能动作或施放汇聚法术。');
   const targets=await resolveMessageTargets(message,{game,fromUuid});
   if(targets.length!==(kind==='swipe'?2:1)||targets.some(t=>!t.object||!t.actor))throw Error(`请选定${kind==='swipe'?'两个相邻的':'一个'}场景目标。`);
   const origin=await sourceToken(actor,message,targets[0]),available=strikes(actor).filter(kind==='combination'?allowedCombinationWeapon:s=>held(s)||unarmed(s.item));
   const key=await select(actor,user,'选择近战武器或无武装攻击',available.map(s=>({value:strikeKey(s),label:s.item.name})));if(key===null)return '已取消。';
   const selected=available.find(s=>strikeKey(s)===key);let second=null;
   if(kind==='combination'){
    const fists=strikes(actor).filter(fist),fistKey=await select(actor,user,'选择拳头攻击',fists.map(s=>({value:strikeKey(s),label:s.item.name})));if(fistKey===null)return '已取消。';second=fists.find(s=>strikeKey(s)===fistKey);
   }
   for(const target of targets)requireSceneTarget(actor,origin,target);
   const choice=kind==='combination'?null:await chooseSpell(actor,user);if(kind!=='combination'&&!choice)return '已取消。';
   let spellTarget=null;
   if(kind==='swipe'&&!canAffectSeveral(choice.spell)){spellTarget=await select(actor,user,'选择承受法术的目标',targets.map(t=>({value:t.uuid,label:t.name??t.actor.name??t.id})));if(spellTarget===null)return '已取消。';}
   const tier=await select(actor,user,'当前多重攻击惩罚档位',[{value:'0',label:'本回合尚未攻击（MAP 0）'},{value:'1',label:'已攻击一次（MAP 1）'},{value:'2',label:'已攻击两次或更多（MAP 2）'}]);if(tier===null)return '已取消。';
   const order=kind==='combination'?await select(actor,user,'神威连击：攻击顺序',[{value:'weapon',label:'先武器，后拳头'},{value:'fist',label:'先拳头，后武器'}]):null;
   if(kind==='combination'&&order===null)return '已取消。';
   const map=Number(tier);let payment;
   assertUse(actor,item,message,user,action);
   // Recheck equipped state and recipient identity after normal player choices.
   const current=strikes(actor).find(s=>strikeKey(s)===key);
   if(!current||!(kind==='combination'?allowedCombinationWeapon(current):held(current)||unarmed(current.item)))throw Error('所选武器的持用状态已改变，尚未攻击。');
   for(const target of targets)requireSceneTarget(actor,origin,target);
   if(choice){payment=await nativeCasts.payForActivity({actor,item:choice.spell,message,user,rank:choice.rank,slotId:choice.slotId});}
   requireGM();
   await record(message,'started',{kind,weaponUuid:current.item.uuid,spellUuid:choice?.spell.uuid??null});
   try{
    requireGM();if(choice)await actor.update({[`flags.${MODULE_ID}.spellstrike`]:{charged:false,messageId:message.id}});requireGM();
    const attacks=[],sequence=createAttackSequence({actor});
    if(kind==='combination'){
     for(const [index,selected]of (order==='fist'?[second,current]:[current,second]).entries()){
      requireGM();
      const strike=strikes(actor).find(s=>strikeKey(s)===strikeKey(selected));
      if(!strike||!(fist(strike)||allowedCombinationWeapon(strike)))throw Error('连击的下一把武器已不可用。');
      requireSceneTarget(actor,origin,targets[0]);attacks.push(await attack(actor,strike,targets[0],Math.min(map+index,2),message,index,kind,sequence));
     }
    }else for(const [index,target]of targets.entries()){requireGM();requireSceneTarget(actor,origin,target);attacks.push(await attack(actor,current,target,map,message,index,kind,sequence));}
    const receivesSpell=attack=>choice&&(!spellTarget||attack.target.uuid===spellTarget)&&(kind==='swipe'&&!spellTarget?hit(attack.outcome):choice.spell.isAttack||choice.spell.system.traits.value.includes('attack')?hit(attack.outcome):attack.outcome!=='criticalFailure');
    const eligible=attacks.filter(receivesSpell),spellCard=eligible.length?await publishSpell({actor,user,message,choice,payment,targets:eligible.map(a=>a.target),kind}):null;
    const sharedSpellDamage=choice?.spell.system.defense?.save&&!choice.spell.isAttack&&!choice.spell.system.traits.value.includes('attack')?{}:null;
    for(const target of targets){
     requireGM();
     const forTarget=attacks.filter(a=>a.target.uuid===target.uuid),parts=[];
     for(const a of forTarget){const damage=await weaponDamage(a);if(damage)parts.push({roll:damage,item:a.strike.item});}
     const spellAttack=forTarget.find(receivesSpell);
     if(spellAttack){
      const spell=choice.spell,defense=spell.system.defense?.save;
      const outcome=defense?await save(spell,target,spellAttack,spellCard):null;
      const damage=await spellDamage(spell,target,spellAttack.outcome,{saveOutcome:defense?.basic?outcome:null,shared:sharedSpellDamage});if(damage)parts.push({roll:damage,item:spell});
      const extra=spellAttack.outcome==='criticalSuccess'&&!defense?criticalSpellPersistentFormula(spell):null;
      if(extra){requireGM();const D=globalThis.CONFIG.Dice.rolls.find(c=>c.name==='DamageRoll'),roll=await new D(extra).evaluate();requireGM();damageContexts.set(roll,damageContexts.get(damage));parts.push({roll,item:spell});}
     }
     await damageCard({actor,message,target,parts,attacks:forTarget,kind});
    }
    requireGM();if(choice&&!eligible.length)await nativeCasts.finishActivityWithoutSpell({actor,message,user});
    await record(message,'done');return '已完成攻击、法术支付与合并伤害；每个目标按原生伤害卡应用一次。';
   }catch(error){if(isActiveGM(game))await record(message,'error',{error:String(error.message??error)});throw error;}
  });
 }
 async function maintain(actor){
  if(!isActiveGM(game)||!hasSpellstrike(actor)||values(actor.items).some(i=>own(i).spellstrikeRecharge===true))return;
  await actor.createEmbeddedDocuments('Item',[{type:'action',name:'法术打击充能 Recharge Spellstrike',img:'systems/pf2e/icons/actions/OneAction.webp',system:{actionType:{value:'action'},actions:{value:1},category:'interaction',description:{value:'<p>以一个具有专注特征的动作，为你的法术打击充能。</p>'},traits:{value:['concentrate']},rules:[]},flags:{[MODULE_ID]:{spellstrikeRecharge:true}}}]);
 }
 function register({Hooks,libWrapper,socket}){
  const unregister=nativeCasts.register({libWrapper,socket});
  const id=Hooks.on('pf2e.restForTheNight',actor=>{
   if(!actor?.testUserPermission(game.user,'OWNER')||!hasSpellstrike(actor))return;
   const startedAsGM=isActiveGM(game);
   queue.run(actor.uuid,()=>{if(startedAsGM)requireGM();return actor.update({[`flags.${MODULE_ID}.spellstrike`]:{charged:true}})}).catch(onError);
  });
  return()=>{Hooks.off('pf2e.restForTheNight',id);unregister();};
 }
 return {resolveAction,requiresActualUse,captureUsage,executeUsage,maintain,register};
}
