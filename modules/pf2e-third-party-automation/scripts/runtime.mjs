import {MODULE_ID,SOURCES,ATTUNEMENT_DAMAGE,findFeature,hasSource,focusRecovery,levelDC,restRecovery,buildLegendEffect,buildCycleEffect,buildAttunementEffect} from './rules.mjs';
import {buildRepairPlan} from './repairs.mjs';

export function requireOwner(actor,user){
 if(actor?.type!=='character'||!user||!actor.testUserPermission(user,'OWNER'))throw Error('没有此角色的所有者权限。');
}
export class SerialActions{
 #pending=new Map();
 run(key,fn){const next=(this.#pending.get(key)??Promise.resolve()).catch(()=>{}).then(fn);this.#pending.set(key,next);next.finally(()=>{if(this.#pending.get(key)===next)this.#pending.delete(key)}).catch(()=>{});return next}
}
export function resolveProviderAction(providers,item){
 for(const provider of providers){const action=provider.resolveAction?.(item);if(action)return action;}
}
export function isLegendEligible(item){return ['weapon','armor','equipment','shield','backpack'].includes(item.type)&&item.isMagical===true&&(item.quantity??item.system?.quantity??0)>0}
const queue=new SerialActions();
const itemsOf=a=>Array.from(a.items??[]);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const feature=(actor,key)=>{const f=findFeature(actor,key);if(!f)throw Error('角色没有对应的第三方专长。');return f};
const kinds=new Set(['legend','attunement','cycle']);
const damageLabels={air:'气',cold:'寒冷',earth:'土',electricity:'电击',fire:'火焰',metal:'金',poison:'毒素',sonic:'音波',vitality:'命能',void:'虚能',water:'水',wood:'木',slashing:'挥砍',bludgeoning:'钝击',piercing:'穿刺'};
const cultivatorSpells=actor=>itemsOf(actor).filter(i=>i.type==='spell'&&hasSource(i,SOURCES.swordQiSpell));

async function replaceEffect(actor,kind,data){
 const existing=itemsOf(actor).filter(i=>i.type==='effect'&&i.flags?.[MODULE_ID]?.kind===kind);
 if(!data){if(existing.length)await actor.deleteEmbeddedDocuments('Item',existing.map(i=>i.id));return}
 if(existing.length){await existing[0].update(data);if(existing.length>1)await actor.deleteEmbeddedDocuments('Item',existing.slice(1).map(i=>i.id))}
 else await actor.createEmbeddedDocuments('Item',[data]);
}

export async function executeActorAction(actor,action,payload={},user=game.user,context={}){
 requireOwner(actor,user);
 return queue.run(actor.uuid,async()=>{
  requireOwner(actor,user);
  if(action==='breath'){
   const f=feature(actor,'breath'),focus=actor.system.resources.focus;
   const oldUses=f.system.frequency?.value??f.system.frequency?.max;
   const receipt=context.frequencyReceipt;
   if(receipt&&(receipt.itemUuid!==f.uuid||receipt.userId!==user.id||receipt.before-receipt.after!==1))throw Error('使用次数回执与本角色、本次使用不符。');
   const refund=()=>f.update({'system.frequency.value':Math.min(f.system.frequency.max,(f.system.frequency.value??0)+1)});
   if(receipt&&focus.value>=focus.max){await refund();return '聚能点已满；已返还使用按钮扣除的本次次数。'}
   const result=focusRecovery({value:focus.value,max:focus.max,uses:receipt?1:oldUses});
   if(!receipt)await f.update({'system.frequency.value':result.uses});
   try{await actor.update({'system.resources.focus.value':result.value})}
   catch(e){if(receipt)await refund();else await f.update({'system.frequency.value':oldUses});throw e}
   return '已恢复1点聚能点，并消耗鼓舞之息的每日次数。';
  }
  if(action==='legend'){
   feature(actor,'legend');const ids=[...new Set(payload.itemIds??[])];
   if(ids.length>2)throw Error('最多选择两件物品。');
   for(const id of ids){const i=actor.items.get(id);if(!i||!isLegendEligible(i)||!/@Check\[[^\]]*dc:\d+/.test(i.system.description.value))throw Error('请选择持有、非消耗品且具有固定启动DC的魔法物品。')}
   await replaceEffect(actor,'legend',ids.length?buildLegendEffect(ids):null);
   return ids.length?'魔法物品传奇已生效；完整休息后清除并重新准备。':'已清除魔法物品传奇选择。';
  }
  if(action==='attunement'){
   feature(actor,'attunement');
   const spellIds=cultivatorSpells(actor).map(i=>i.id);
   await replaceEffect(actor,'attunement',buildAttunementEffect(payload.trait,spellIds));
   return `已记录能量调谐，限定作用于${spellIds.length}道天夏修炼者法术。反应抗力须按触发效果的特征判定。`;
  }
  if(action==='cycle'){
   feature(actor,'cycle');feature(actor,'attunement');
   const att=itemsOf(actor).find(i=>i.flags?.[MODULE_ID]?.kind==='attunement');
   const trait=actor.flags?.pf2e?.cultivator?.energy??att?.flags?.[MODULE_ID]?.trait;
   const damageType=ATTUNEMENT_DAMAGE[trait];
   if(!damageType)throw Error('请先选择每日能量调谐。');
   const allowed=['void','vitality'].includes(trait)?['void','vitality']:[damageType];
   if(!allowed.includes(payload.damageType))throw Error('伤害类型与已调谐特征不符。');
   if(payload.triggerConfirmed!==true)throw Error('请确认触发伤害具有已调谐特征，并消耗反应。');
   const combatant=game.combat?.combatants.find(c=>c.actor?.uuid===actor.uuid);
   if(!game.combat?.started||combatant?.initiative==null)throw Error('请在已掷先攻的遭遇中使用，以正确追踪下一回合结束。');
   const index=game.combat.turns.findIndex(c=>c.id===combatant.id);
   const timing=context.cycleTiming;
   if(timing&&timing.combatId!==game.combat.id)throw Error('循环能量的原遭遇已经结束。');
   const rounds=timing?.rounds??(index>game.combat.turn?0:1);
   const effect=buildCycleEffect(payload.damageType,{worldTime:timing?.worldTime??game.time.worldTime,initiative:timing?.initiative??combatant.initiative});
   effect.system.duration.value=rounds;
   Object.assign(effect.flags[MODULE_ID],{combatId:game.combat.id,combatantId:combatant.id,endRound:timing?.endRound??game.combat.round+rounds});
   await replaceEffect(actor,'cycle',effect);
   return '已自动登记循环能量的打击加伤，持续到下一次自己的回合结束。';
  }
  if(action==='rest'){
   feature(actor,'circadian');
   if(payload.eligible!==true)throw Error('必须确认休息资格与所需时间。');
   const now=game.time.worldTime,last=actor.flags?.[MODULE_ID]?.restAt;
   if(Number.isFinite(last)&&now-last<86400)throw Error('记录显示24小时内已经获得休息效果。时间记录有误时由GM重置。');
   const message=game.messages.get(payload.messageId),context=message?.flags?.pf2e?.context;
   if(message?.speaker.actor!==actor.id||!context?.options?.includes('action:third-party-circadian')||context.dc?.value!==levelDC(actor.level)||context.type!=='skill-check')throw Error('未找到该角色的有效昼夜规律师生存检定。');
   if(!context.domains?.includes('survival')||(!user.isGM&&message.author?.id!==user.id&&!message.author?.isGM))throw Error('必须使用当前操作者或执行GM的生存检定。');
   if(message.flags?.[MODULE_ID]?.restApplied)throw Error('此检定已经结算过，不能重复使用。');
   const degree=message.rolls[0]?.options.degreeOfSuccess;
   if(![0,1,2,3].includes(degree))throw Error('检定缺少成功等级。');
   if(degree<2)return '检定失败，没有获得休息效果。';
   const healing=restRecovery({level:actor.level,constitution:actor.system.abilities.con.mod,degree});
   // Reserve before multi-document effects; a failed operation remains visible for GM review.
   await message.update({[`flags.${MODULE_ID}.restApplied`]:actor.uuid});
   await actor.update({[`flags.${MODULE_ID}.restAt`]:now,[`flags.${MODULE_ID}.restMessage`]:message.id});
   if(actor.hasCondition('fatigued'))await actor.decreaseCondition('fatigued',{forceRemove:true});
   if(actor.hasCondition('drained'))await actor.decreaseCondition('drained');
   const hp=actor.system.attributes.hp;
   await actor.update({'system.attributes.hp.value':Math.min(hp.max,hp.value+healing)});
   return `已结算休息：治疗${healing}，移除疲乏，流失减1；未恢复法术或每日次数。`;
  }
  if(action==='repair'){
   const plan=buildRepairPlan(actor.toObject());let n=0;
   for(const update of plan.updates){
    const item=actor.items.get(update.itemId);if(!item)continue;
    for(const path of Object.keys(update.changes))if(foundry.utils.getProperty(item.toObject(),path)!==update.originals[path])throw Error(`${item.name}已被同时编辑，已停止修复。`);
    const prior=item.flags?.[MODULE_ID]?.repairBackup??[];
    const additions=Object.keys(update.changes).map(path=>({path,before:update.originals[path],after:update.changes[path]}));
    await item.update({...update.changes,[`flags.${MODULE_ID}.repairBackup`]:[...prior.filter(p=>!additions.some(a=>a.path===p.path)),...additions]});n++;
   }
   for(const missing of plan.missingSpells){
    if(!missing.spellcastingEntryId)continue;
    const source=await fromUuid(missing.sourceUuid);
    if(!source)throw Error('无法读取剑气波源法术，请确认天夏+已启用。');
    if(itemsOf(actor).some(i=>hasSource(i,missing.sourceUuid)))continue;
    const data=source.toObject();delete data._id;data.system.location={...data.system.location,value:missing.spellcastingEntryId};
    data.flags={...data.flags,[MODULE_ID]:{importedSpell:true}};
    data._stats={...data._stats,compendiumSource:source.uuid};
    await actor.createEmbeddedDocuments('Item',[data]);n++;
   }
   return `已修复${n}项。修复前值已保存在各条目中；未明确的龙种条件保留供GM核对。`;
  }
  if(action==='undo-repairs'){
   await actor.update({[`flags.${MODULE_ID}.autoRepairDisabled`]:true});
   let n=0;for(const item of itemsOf(actor)){
    const backup=item.flags?.[MODULE_ID]?.repairBackup;if(!backup)continue;
    const changes={};for(const {path,...v}of backup){
     if(foundry.utils.getProperty(item.toObject(),path)!==v.after)throw Error(`${item.name}已被再次编辑，停止回滚以保留新修改。`);
     changes[path]=v.before;
    }
    await item.update({...changes,[`flags.${MODULE_ID}.-=repairBackup`]:null});n++;
   }
   return `已回滚${n}个条目的数据修复。补入的法术需按需从角色卡手动删除。`;
  }
  if(action==='clear'){
   for(const kind of kinds)await replaceEffect(actor,kind,null);return '已清除本模组创建的准备与反应效果。';
  }
  if(action==='reset-rest'){
   if(!user.isGM)throw Error('只有GM可以重置休息记录。');
   await actor.update({[`flags.${MODULE_ID}.-=restAt`]:null,[`flags.${MODULE_ID}.-=restMessage`]:null});return '休息记录已重置。';
  }
  if(action==='full-rest'){
   // Dailies owns temporary and legacy daily effect cleanup. Do not race its rest hook.
   if(findFeature(actor,'circadian'))await actor.update({[`flags.${MODULE_ID}.restAt`]:game.time.worldTime,[`flags.${MODULE_ID}.lastNightAt`]:game.time.worldTime});
   return '第三方每日选择与休息记录已更新。';
  }
  throw Error('未知操作。');
 });
}

/** Invoked by the authenticated, idempotent native/chat usage adapter on the active GM. */
export function createUsageExecutor({cycleUse}={}){
 const usageQueue=new SerialActions();
 return context=>usageQueue.run(context.actor?.uuid,async()=>{
  const {actor,item,message,user,action,frequencyReceipt}=context;
  requireOwner(actor,user);
  if(action==='breath')return executeActorAction(actor,'breath',{},user,{frequencyReceipt});
  if(action==='cycle'){
   if(!cycleUse)throw Error('循环能量的伤害入口尚未就绪。');
   return cycleUse(actor,message,user);
  }
  if(action==='rest'){
   feature(actor,'circadian');
   const now=game.time.worldTime,last=actor.flags?.[MODULE_ID]?.restAt,night=actor.flags?.[MODULE_ID]?.lastNightAt;
   if(Number.isFinite(last)&&now-last<86400)throw Error('24小时内已经获得休息效果。');
   if(Number.isFinite(night)&&now-night>86400)throw Error('已超过1日未整夜休息，昼夜规律师本次自动失败。');
   let result='生存检定未完成。';
   await actor.skills.survival.roll({
    dc:{value:levelDC(actor.level)},skipDialog:true,
    extraRollOptions:['action:third-party-circadian'],
    callback:async(_roll,_outcome,rollMessage)=>{result=await executeActorAction(actor,'rest',{eligible:true,messageId:rollMessage.id},user);},
   });
   return result;
  }
  throw Error('没有此技能的自动执行入口。');
 });
}

export function createPanel(request){
 return async function open(actor=canvas.tokens.controlled[0]?.actor??game.user.character){
  requireOwner(actor,game.user);
  const present=Object.keys(SOURCES).filter(k=>findFeature(actor,k));
  const plan=buildRepairPlan(actor.toObject());
  const magical=itemsOf(actor).filter(i=>isLegendEligible(i)&&/@Check\[[^\]]*dc:\d+/.test(i.system.description.value));
  const att=itemsOf(actor).find(i=>i.flags?.[MODULE_ID]?.kind==='attunement')?.flags?.[MODULE_ID];
  const legendSelection=itemsOf(actor).find(i=>i.flags?.[MODULE_ID]?.kind==='legend')?.flags?.[MODULE_ID]?.itemIds??[];
  const section=(key,html)=>present.includes(key)?html:'';
  const checkbox=(name,label,checked=false)=>`<label style="display:block"><input type="checkbox" name="${name}" ${checked?'checked':''}> ${label}</label>`;
  const content=`<div style="display:grid;gap:12px;max-height:70vh;overflow:auto"><p>仅结算这张角色卡拥有的第三方能力。操作会由在线GM统一执行。</p>
   ${section('breath',`<fieldset><legend>鼓舞之息</legend><p>恢复1聚能点，消耗每日次数；满值不消耗。</p></fieldset>`)}
   ${section('legend',`<fieldset><legend>魔法物品传奇 · 每日准备</legend><p>选择最多两件物品。固定启动DC = 职业／法术DC较高值 − 2。</p>${magical.map(i=>checkbox('legend-'+i.id,esc(i.name),legendSelection.includes(i.id))).join('')||'<p>没有符合条件的物品。</p>'}</fieldset>`)}
   ${section('attunement',`<fieldset><legend>能量调谐 · 每日准备</legend><select name="trait">${[['air','气（挥砍）'],['cold','寒冷'],['earth','土（钝击）'],['electricity','电击'],['fire','火焰'],['metal','金（挥砍）'],['poison','毒素'],['sonic','音波'],['vitality','命能'],['void','虚能'],['water','水（钝击）'],['wood','木（穿刺）']].map(([v,n])=>`<option value="${v}" ${att?.trait===v?'selected':''}>${n}</option>`).join('')}</select></fieldset>`)}
   ${section('cycle',`<fieldset><legend>循环能量 · 反应</legend><p>当前调谐：${esc(damageLabels[att?.trait]??'尚未选择')}。本次触发伤害抗力${actor.level}须在伤害结算时应用；此按钮登记后续打击加伤。</p>${checkbox('trigger','已确认伤害具有调谐特征，并使用反应')}<select name="cycleType">${(att&&['void','vitality'].includes(att.trait)?['void','vitality']:[att?.damageType??'']).map(t=>`<option value="${t}">${damageLabels[t]||'先进行调谐'}</option>`).join('')}</select></fieldset>`)}
   ${section('circadian',`<fieldset><legend>昼夜规律师 · 部分休息</legend><p>生存DC ${levelDC(actor.level)}；生存大师10分钟，否则1小时。成功恢复一半，大成功恢复全部正常休息生命值，并移除疲乏、流失减1。</p>${checkbox('eligible','已休息规定时间；距离上次整夜休息未超过1日，且24小时内未获得休息效果')}</fieldset>`)}
   ${(plan.updates.length||plan.missingSpells.length)?`<fieldset><legend>检测到的数据缺口</legend><ul>${plan.updates.map(u=>`<li>${esc(actor.items.get(u.itemId)?.name)}：${esc(u.reason)}</li>`).join('')}${plan.missingSpells.map(s=>`<li>补入已学剑气波聚能法术${s.spellcastingEntryId?'':'（无法确定施法栏，暂不执行）'}</li>`).join('')}</ul></fieldset>`:''}
   ${plan.warnings.length?`<details><summary>需要核对的条目</summary><ul>${plan.warnings.map(w=>`<li>${esc(w.message)}</li>`).join('')}</ul></details>`:''}
   </div>`;
  const run=async(action,payload)=>{try{const result=await request(actor.uuid,action,payload);ui.notifications.info(result);return result}catch(e){ui.notifications.error(e.message);throw e}};
  const form=b=>b.form;
  const buttons=[];
  const add=(action,label,callback)=>buttons.push({action,label,callback});
  if(present.includes('breath'))add('breath','使用鼓舞之息',()=>run('breath',{}));
  if(present.includes('legend'))add('legend','准备物品传奇',(_e,b)=>run('legend',{itemIds:magical.filter(i=>form(b).elements['legend-'+i.id]?.checked).map(i=>i.id)}));
  if(present.includes('attunement'))add('attunement','准备能量调谐',(_e,b)=>run('attunement',{trait:form(b).elements.trait.value}));
  if(present.includes('cycle'))add('cycle','使用循环能量',(_e,b)=>run('cycle',{damageType:form(b).elements.cycleType.value,triggerConfirmed:form(b).elements.trigger.checked}));
  if(present.includes('circadian'))add('circadian','检定部分休息',async(_e,b)=>{
   if(!form(b).elements.eligible.checked)throw Error('请先确认休息资格。');
   const last=actor.flags?.[MODULE_ID]?.restAt;if(Number.isFinite(last)&&game.time.worldTime-last<86400)throw Error('24小时内已有休息效果。');
   return actor.skills.survival.roll({dc:{value:levelDC(actor.level)},extraRollOptions:['action:third-party-circadian'],callback:async(_roll,_outcome,message)=>run('rest',{eligible:true,messageId:message.id})});
  });
  if(plan.updates.length||plan.missingSpells.some(s=>s.spellcastingEntryId))add('repair','备份并修复所列缺口',()=>run('repair',{}));
  if(itemsOf(actor).some(i=>i.flags?.[MODULE_ID]?.repairBackup))add('undo','回滚数据修复',()=>run('undo-repairs',{}));
  if(itemsOf(actor).some(i=>kinds.has(i.flags?.[MODULE_ID]?.kind)))add('clear','清除本模组效果',()=>run('clear',{}));
  if(game.user.isGM&&Number.isFinite(actor.flags?.[MODULE_ID]?.restAt))add('reset','GM重置休息记录',()=>run('reset-rest',{}));
  add('close','关闭',()=>null);
  return foundry.applications.api.DialogV2.wait({window:{title:`第三方自动化 · ${actor.name}`,resizable:true},position:{width:640},content,buttons,rejectClose:false});
 };
}

export async function onFullRest(actor,request){
 if(actor.type!=='character')return;
 if(!findFeature(actor,'circadian')&&!itemsOf(actor).some(i=>['legend','attunement'].includes(i.flags?.[MODULE_ID]?.kind)))return;
 return request(actor.uuid,'full-rest',{});
}

export async function expireCycles(combat,deleted=false){
 if(game.user!==game.users.activeGM)return;
 const actors=new Map([...game.actors,...combat.combatants.map(c=>c.actor).filter(Boolean)].map(a=>[a.uuid,a]));
 for(const actor of actors.values()){
  const shouldExpire=i=>{
   const f=i.flags?.[MODULE_ID];if(f?.kind!=='cycle'||f.combatId!==combat.id)return false;
   const index=combat.turns.findIndex(c=>c.id===f.combatantId);
   return deleted||index<0||combat.round>f.endRound||(combat.round===f.endRound&&combat.turn>index);
  };
  if(itemsOf(actor).some(shouldExpire))await queue.run(actor.uuid,async()=>{
   const ids=itemsOf(actor).filter(shouldExpire).map(i=>i.id);
   if(ids.length)await actor.deleteEmbeddedDocuments('Item',ids);
  });
 }
}
