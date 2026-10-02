import {reconstructEarliest} from './timeline.mjs';
import {checkpointBinding,activityCheckpointBinding,sameActivityCheckpoint} from './schema.mjs';
import {scheduleCheckpointActivities} from './policy.mjs';
import {normalizeCheckpointActivity} from './manual-time.mjs';
import {canonicalJSON} from './revision-codec.mjs';
export const escapeHTML=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const labels={'goals-met':'恢复目标完成',running:'恢复中',recording:'手动记录中',paused:'已停止',confirmed:'已确认',started:'进行中',completing:'结算中','awaiting-evidence':'待核对证据',uncertain:'结果未知',blocked:'无法继续',cancelled:'已取消','treat-wounds':'治疗伤势',refocus:'再聚能',manual:'手动行动','focus-healing':'圣疗','user-stopped':'已手动停止',budget:'达到预算',closed:'已核对结束','client-context-lost':'客户端已重载，请核对后接续','gm-reviewed-closed':'GM 已核对结束，未知结果保留','refocus-recovery-unadapted':'特殊再聚能规则尚未适配，请手动处理'};
export function formatSessionResult(record){return `${record.certainty==='observed'?'已确认经过':record.certainty==='contradictory'?'记录存在矛盾，推算':record.certainty==='incomplete'?'证据不完整，按假设推算最早':'按并行假设推算最早'} ${Math.round(record.durationSeconds/60*10)/10} 分钟；尚缺 ${record.remainingHP??0} HP；${labels[record.status]??record.status??''}`}
export function sessionTimeSummary(data){return data.session.manual?reconstructEarliest({startedAt:data.session.startedAt,activities:data.activities,assumptions:data.session.assumptions}):{durationSeconds:data.session.cursorAt-data.session.startedAt,certainty:'observed'}}
export function recoveryBudgetPolicy({minutes,activities}){if(!Number.isFinite(minutes)||minutes<=0)throw Error('时间预算应为正数分钟。');if(!Number.isInteger(activities)||activities<1||activities>100)throw Error('行动次数应为1至100。');return {budgetSeconds:minutes*60,maxActivities:activities}}
export function recoveryDefaults(actors){return {goalsByPool:[...new Map(actors.map(a=>[a.pool.poolUUID,{poolUUID:a.pool.poolUUID,targetHP:a.hp.max}])).values()],allowFiniteResources:false,riskySurgery:false,requireFullFocus:false,budgetSeconds:7200,maxActivities:100}}
const evidenceLabels={
 'native-application-receipt':'缺少原生 HP 应用回执','native-immunity-receipt':'缺少原生免疫回执',
 'checkpoint-time-confirmation':'等待本轮十分钟时间确认','native-manual-source-unavailable':'尚未完成本轮 Workbench 治疗',
 'native-immunity-timing-unconfirmed':'原生免疫的开始时刻或冷却时长尚未确认','native-source-identity-unproven':'原生治疗来源尚未确认',
 'ambiguous-native-application':'同一治疗结果的 HP 应用次数尚未确认','unreserved-manual-source':'出现未预约的手动治疗，请核对后结束会话',
 'manual-source-requires-review':'人工登记的行动来源需 GM 核对',
 'shared-hp-completion-unavailable':'共用血池主人的写入尚未确认','hp-pool-source-unavailable':'共用血池来源尚未确认',
 'missing-activity-source':'缺少行动来源','missing-actor-order':'缺少执行者行动顺序','unproven-group':'群体行动缺少同次来源证明',
 'inconsistent-group':'群体行动的执行者或时长不一致','missing-dependency':'缺少前置行动','dependency-cycle':'前置行动顺序存在循环',
 'observed-before-ready':'记录时间早于可开始时间','activity-uncertain':'行动结果未知','activity-blocked':'行动未能继续','activity-cancelled':'行动已取消',
 'world-time-source-unconfirmed':'世界时间来源回执尚未确认','world-time-conflict':'世界时间与计划不一致',
 'world-time-conflict-after-claim':'保存提交声明后世界时间发生变化','clock-never-replayed':'时间提交尚未确认，保留原尝试',
 'clock-in-flight':'已有时间提交正在执行','time-effects-unconfirmed':'时间相关效果尚未完成确认',
 'passive-completion-unavailable':'被动恢复尚无可确认完成的适配','passive-completion-unproven':'被动恢复缺少完成回执',
 'missing-passive-checkpoint':'缺少被动恢复检查点','passive-rules-changed':'检查点内的被动恢复规则发生变化'
};
const evidenceText=value=>labels[value]??evidenceLabels[value]??value;
function ledgerSetupControls(status){
 if(!status||status.state==='ready')return '';
 const migrating=['legacy','migration-required'].includes(status.state),text=status.state==='unconfigured'?'先准备探索账本，再启动恢复。':migrating?'已有探索历史需要升级账本。升级会保留记录，暂停旧的恢复任务，并保留尚未确认的尝试。恢复任务由你另行启动。':`探索账本暂不可用：${evidenceText(status.reason)}。请选择仍存在且仅供 GM 管理的账本。`;
 return `<section class="recovery-unresolved"><p>${escapeHTML(text)}</p>${status.state==='unconfigured'?'<button type="button" data-recovery="provision">创建并准备探索账本</button>':migrating?'<button type="button" data-recovery="initialize">升级探索账本</button>':''}<button type="button" data-recovery="selectLedger">选择已有账本</button></section>`;
}
async function setupChoices({game,select=false}){
 const candidates=Array.from(game.journal?.contents??[]).filter(j=>j.flags?.['pf2e-third-party-automation']?.explorationLedger||j.flags?.['pf2e-third-party-automation']?.explorationLedgerRoot);
 return foundry.applications.api.DialogV2.wait({window:{title:select?'选择探索账本':'准备探索账本'},content:`${select?`<label>账本 <select name="rootUUID">${candidates.map(j=>`<option value="${escapeHTML(j.uuid)}">${escapeHTML(j.name)} (${escapeHTML(j.id)})</option>`).join('')}</select></label>`:''}<p>先停止正在运行的恢复，并让相关 GM 和玩家重载到当前版本。准备完成后不会自动开始恢复。</p><label><input type="checkbox" name="issuersStopped" required> 所有旧恢复操作均已停止</label><label><input type="checkbox" name="clientsReloaded" required> 相关客户端均已重载</label><label><input type="checkbox" name="recoveryDisabled" required> 当前没有自动恢复正在执行</label>`,buttons:[{action:'prepare',label:select?'选择账本':'准备账本',callback:(_event,b)=>{const form=new FormData(b.form);return {issuersStopped:form.has('issuersStopped'),clientsReloaded:form.has('clientsReloaded'),recoveryDisabled:form.has('recoveryDisabled'),...select?{rootUUID:form.get('rootUUID')}:{}}}},{action:'cancel',label:'取消',callback:()=>null}],rejectClose:false});
}
const minutes=value=>Math.round(value/60*10)/10;
function historyAnnotations(activity,sessionStart){
 const lines=[],duration=activity.durationSource,temporal=activity.temporalSource,options=activity.options??{};
 const durationLabels={'user-declared':'玩家声明','item-text':'条目说明','table-convention':'本桌约定'};
 if(duration)lines.push(`时长出处：${durationLabels[duration.type]??'未识别的出处'}${duration.detail?` · ${duration.detail}`:''}`);
 if(temporal?.type==='checkpoint-declaration')lines.push(`${activity.state==='confirmed'?'声明活动时间已计入':'声明活动时间尚未完成'}；规则效果未核验。`);
 if(temporal?.type==='user-declared'){
  const times=[];
  if(Number.isFinite(activity.observedStart)&&Number.isFinite(activity.observedEnd))times.push(`声明区间：第 ${minutes(activity.observedStart-sessionStart)} → ${minutes(activity.observedEnd-sessionStart)} 分钟`);
  if(Number.isFinite(activity.notBefore))times.push(`不得早于第 ${minutes(activity.notBefore-sessionStart)} 分钟`);
  if(Number.isFinite(temporal.recordedAt))times.push(`登记于第 ${minutes(temporal.recordedAt-sessionStart)} 分钟`);
  lines.push(`时间来源：人工声明${times.length?` · ${times.join('；')}`:''}（相对本次休整开始，不代表原生时间回执）`);
 }
 if(options.estimate){
  const dc={trained:15,expert:20,master:30,legendary:40}[options.rank],reason=options.estimateReason;
  const reasons={
   'healing-model-unverified':'患者的治疗修正规则尚未核验','native-healing-model-unverified':'额外治疗规则尚未核验',
   'patient-hp-model-unverified':'患者的 HP 状态尚未核验','inconsistent-patient-pool':'共用血池的状态不一致',
   'self-treatment-context-unverified':'自我治疗情境尚未核验','damage-model-unverified':'伤害修正或临时 HP 尚未核验',
   'action-capacity-after-damage-unverified':'割伤后能否继续治疗尚未核验','skill-assurance-unavailable':'对应技能的 Assurance 不可用',
   'native-before-roll-unverified':'掷骰前的动态规则尚未核验','native-substitution-conflict':'存在其他替代掷骰规则',
   'native-roll-twice-unverified':'存在取高或取低掷骰规则'
  };
  const verified=!reason&&['verified-prepared-native-context','verified-unconditional-native-context','fixed-selected-dc'].includes(options.estimate);
  if(!verified)lines.push(`治疗估算暂不可用：${reasons[reason]??'当前条件或规则尚未核验'}${dc?`；采用 DC ${dc}`:''}。`);
  else{
   const source=options.estimateSource,skill={medicine:'医疗',nature:'自然',occultism:'神秘'}[source?.skill??options.skill],basis=[skill,dc?`DC ${dc}`:null,(source?.assurance??options.assurance)?'Assurance':null,(source?.riskySurgery??options.riskySurgery)?'激进治疗':null].filter(Boolean);
   const amounts=[];
   if(Number.isFinite(activity.expectedNetHealing))amounts.push(`预计净恢复 ${Math.round(activity.expectedNetHealing*10)/10} HP`);
   if(Number.isFinite(activity.expectedDamage))amounts.push(`预计受到 ${Math.round(activity.expectedDamage*10)/10} HP 伤害`);
   lines.push(`行动前估算${basis.length?`（${basis.join(' · ')}）`:''}${amounts.length?`：${amounts.join('；')}`:''}。按当时的目标与血池合计，实际结果以原生回执为准。`);
  }
 }
 return lines.map(line=>`<p>${escapeHTML(line)}</p>`).join('');
}
function visibleMessage(messages,id){const message=messages?.get?.(id);return message?.id===id&&message.visible===true&&message.isContentVisible===true?message:null}
/** Render saved facts only; missing proof never grants an execution or application. */
export function renderRecoveryHistory(data,{actors=[],messages}={}){
 if(!data?.session)return '';
 const e=escapeHTML,names=new Map([...(data.actors??[]),...actors].map(a=>[a.actorUUID,a.name])),history=data.activities??[],reconstruction=sessionTimeSummary(data);
 const title=a=>`${e(names.get(a.actorUUID)??a.actorUUID)}${a.patientUUIDs?.length?` → ${[...new Set(a.patientUUIDs)].map(id=>e(names.get(id)??id)).join('、')}`:''} · ${e(a.options?.label??labels[a.providerId]??a.providerId)}`;
 const reason=a=>a.reason?` · ${e(evidenceText(a.reason))}`:'';
 const unresolved=history.filter(a=>['uncertain','awaiting-evidence','completing','started'].includes(a.state)),clocks=(data.clocks??[]).filter(c=>c.state!=='confirmed');
 const context=(data.session.status==='paused'||data.session.review)&&(unresolved.length||clocks.length)?`<section class="recovery-unresolved"><p>尚未确认</p><ul>${unresolved.map(a=>`<li>${title(a)} · ${e(evidenceText(a.state))}${reason(a)}</li>`).join('')}${clocks.map(c=>`<li>时间提交 · 第 ${e(minutes(c.from-data.session.startedAt))} → ${e(minutes(c.to-data.session.startedAt))} 分钟 · ${e(evidenceText(c.state))}${reason(c)} <code>${e(c.id)}</code></li>`).join('')}</ul><p>核对只读取已有回执，不重掷、不应用、不推进时间；未知项会保留。</p></section>`:'';
 const rows=history.map(a=>{
  const keys=new Set([a.id,...a.groupId&&a.groupProof?[`${a.groupId}:${a.groupProof}`]:[]]);
  const missing=[...new Set([...(a.options?.missing??[]),...(reconstruction.missing??[]).filter(g=>keys.has(g.id)).map(g=>g.reason)])];
  const seen=new Set(),sources=[];
  for(const [label,ids,chat]of [['检定',a.proof?.checkIds,true],[a.source?.type==='user-record'?'人工登记':'骰点结果',a.proof?.resultIds,true],['完成回执',a.proof?.receiptIds,true],['免疫效果',a.proof?.immunityIds,false],['行动来源',a.source?.messageId?[a.source.messageId]:[],true]]){
   for(const id of ids??[]){if(typeof id!=='string'||!id||seen.has(id))continue;seen.add(id);const text=`${e(label)} <code>${e(id)}</code>`;sources.push(`<li>${chat&&visibleMessage(messages,id)?`<button type="button" class="recovery-evidence-link" data-recovery-message="${e(id)}" aria-label="在聊天中查看${e(label)} ${e(id)}">${text}</button>`:text}</li>`)}
  }
  return `<li class="recovery-history-entry" data-recovery-activity-id="${e(a.id)}"><p class="recovery-activity-title">${title(a)} · ${e(minutes(a.durationSeconds??a.endsAt-a.startedAt))} 分钟 <strong>${e(evidenceText(a.state))}</strong></p>${historyAnnotations(a,data.session.startedAt)}${a.reason?`<p>${e(evidenceText(a.reason))}</p>`:''}${missing.length?`<ul class="recovery-missing">${missing.map(m=>`<li>${e(evidenceText(m))}</li>`).join('')}</ul>`:''}<details class="recovery-evidence"><summary>来源与回执</summary><p>行动 ID：<code>${e(a.id)}</code></p>${sources.length?`<ul>${sources.join('')}</ul>`:'<p>尚无已保存的来源回执。</p>'}</details></li>`;
 }).join('');
 return `${context}<ol class="recovery-history">${rows}</ol>${data.session.review?.note?`<section class="recovery-review"><p>GM 核对说明</p><p>${e(data.session.review.note)}</p></section>`:''}`;
}
/** Navigate an existing visible source through V14's native ChatLog; never repost it. */
export async function showRecoveryMessage(id,{game=globalThis.game,chat=globalThis.ui?.chat}={}){
 const message=visibleMessage(game?.messages,id);if(!message||!chat)return false;
 const current=()=>visibleMessage(game.messages,id)===message;
 if(!chat.rendered){if(typeof chat.render!=='function')return false;await chat.render({force:true});if(!current())return false}
 if(!chat.rendered||typeof chat.activate!=='function')return false;chat.activate();if(!current())return false;
 const cards=()=>Array.from(chat.element?.querySelectorAll?.('.chat-log .message[data-message-id]')??[]);
 let card=cards().find(el=>el.dataset.messageId===id);
 if(!card){
  if(typeof chat.renderBatch!=='function')return false;
  const contents=game.messages.contents??[],target=contents.indexOf(message),indices=cards().map(el=>contents.findIndex(m=>m.id===el.dataset.messageId)).filter(i=>i>=0),oldest=indices.length?Math.min(...indices):contents.length;
  if(target<0||oldest<=target)return false;
  await chat.renderBatch(oldest-target+1);if(!current())return false;card=cards().find(el=>el.dataset.messageId===id);
 }
 if(!current()||typeof card?.scrollIntoView!=='function')return false;
 card.scrollIntoView({block:'center',behavior:'auto'});return true;
}
const activityBinding=checkpoint=>activityCheckpointBinding(Object.fromEntries(['id','sessionId','rootUUID','epoch','observationNonce','from'].map(key=>[key,checkpoint[key]])));
export async function promptActivityDeclaration({actors,activities=[],session,window,record}){
 const binding=window?activityCheckpointBinding(window.binding):null;if(window&&window.phase!=='open')throw Error('探索登记窗口已关闭。');
 const names=new Map(actors.map(a=>[a.actorUUID,a.name]));
 const value=await foundry.applications.api.DialogV2.wait({window:{title:'登记探索行动'},content:`
  <label>执行者 <select name="actor">${actors.map(a=>`<option value="${escapeHTML(a.actorUUID)}">${escapeHTML(a.name)}</option>`).join('')}</select></label>
  <label>行动名称 <input name="label" required></label>
  <label>耗时 <input name="duration" type="number" min="0" step="any" required><select name="unit"><option value="60">分钟</option><option value="1">秒</option></select></label>
  <label>时长出处 <select name="durationSource"><option value="user-declared">玩家声明</option><option value="item-text">条目说明</option><option value="table-convention">本桌约定</option></select></label>
  <label>出处说明 <input name="durationDetail" maxlength="500" placeholder="例如：修理条目写明十分钟"></label>
  <details><summary>时间与前置行动</summary>
   <p>${binding?`当前起点为世界时间 ${escapeHTML(binding.from)} 秒。不得早于以此起点计分钟，已流逝时间不能回填。`:'以下时刻以本次休整开始为第零分钟，可留空。填写相同的开始时刻可声明并行；账本仍会核对执行者、治疗冷却及前置关系。'}</p>
   ${binding?'':'<label>已知开始（分钟） <input name="observedStart" type="number" step="any"></label><label>已知结束（分钟） <input name="observedEnd" type="number" step="any"></label>'}
   <label>不得早于（分钟） <input name="notBefore" type="number" min="${binding?'0':''}" step="any"></label>
   <label>执行顺序（从零开始，可留空） <input name="order" type="number" min="0" step="1"></label>
   <label>须先完成 <select name="dependsOn" multiple>${activities.map(a=>`<option value="${escapeHTML(a.id)}">${escapeHTML(names.get(a.actorUUID)??'')} · ${escapeHTML(a.label??a.options?.label??labels[a.providerId]??a.providerId)}</option>`).join('')}</select></label>
  </details><p>登记仅声明耗时与执行者占用，规则效果不会自动执行。封口继续时共同计时；无需所有成员登记。</p>`,buttons:[{action:'record',label:'登记',callback:(_event,b)=>{const form=new FormData(b.form);return {...Object.fromEntries(form),dependsOn:form.getAll('dependsOn')}}},{action:'cancel',label:'取消',callback:()=>null}],rejectClose:false});
 if(!value)return null;
 const duration=Number(value.duration),unit=Number(value.unit),seconds=duration*unit;
 if(!actors.some(a=>a.actorUUID===value.actor)||!value.label?.trim()||value.duration?.trim()===''||![1,60].includes(unit)||!Number.isFinite(seconds)||seconds<0)throw Error('请输入行动名称和有效耗时。');
 const declaration={...binding?{registrationId:crypto.randomUUID(),checkpointBinding:binding}:{sessionId:session.id},actorUUID:value.actor,label:value.label.trim(),durationSeconds:seconds,durationSource:{type:value.durationSource,...value.durationDetail?.trim()?{detail:value.durationDetail.trim()}:{}},dependsOn:value.dependsOn};
 for(const field of binding?['notBefore']:['notBefore','observedStart','observedEnd'])if(value[field]?.trim()){const absolute=(binding?.from??session.startedAt)+Number(value[field])*60;if(!Number.isFinite(absolute))throw Error('请输入有效的行动时刻。');declaration[field]=absolute}
 if(value.order?.trim()){const order=Number(value.order);if(!Number.isSafeInteger(order)||order<0)throw Error('执行顺序须为非负整数。');declaration.order=order}
 if(binding)normalizeCheckpointActivity(declaration);
 try{return await record(declaration)}catch(error){if(binding)error.declaration=structuredClone(declaration);throw error}
}
function checkpointPreview(data){
 const checkpoint=data?.session?.activityCheckpoint;if(checkpoint?.phase!=='open')return '';
 try{const planned=scheduleCheckpointActivities({registrations:checkpoint.registrations,activities:data.activities,session:data.session,from:checkpoint.from}),end=Math.max(checkpoint.from,...planned.map(a=>a.endsAt),...data.activities.filter(a=>['started','planned'].includes(a.state)).map(a=>a.endsAt));return `<p>登记起点：世界时间 ${escapeHTML(checkpoint.from)} 秒；预计共同结束：${escapeHTML(end)} 秒。封口时按最新登记核对预算；规则效果未核验。</p>`}
 catch(error){return `<p class="recovery-unresolved">${error.message==='activity-checkpoint-budget'?'登记超过剩余时间预算，请调整后继续。':'登记的前置行动或执行顺序存在冲突，请核对后继续。'}</p>`}
}
export function createRecoveryPanel({game,coordinator,capabilities,start,storage,record,getActivityCheckpoint,lookupCheckpointActivity,getSessionId,onError=console.error}){
 let panel;const unconfirmed=new Map();
 async function submitDeclaration(options){try{return await promptActivityDeclaration({...options,record})}catch(error){if(error.declaration&&error.declarationRejected!==true){unconfirmed.set(error.declaration.actorUUID,error.declaration);panel?.render(true)}throw error}}
 async function lookupDeclaration(declaration){
  const saved=await lookupCheckpointActivity(declaration.checkpointBinding,declaration.registrationId,declaration.actorUUID),normalized=normalizeCheckpointActivity(declaration),{registrationId,checkpointBinding,...expected}=normalized;
  if(!saved)throw Error(`登记 ${registrationId} 尚未确认，保留原尝试，不会重发。`);
  if(saved.registrationId!==registrationId||!sameActivityCheckpoint(saved.checkpointBinding,checkpointBinding)||canonicalJSON(saved.declaration)!==canonicalJSON(expected)||game.user.id&&saved.source?.userId!==game.user.id)throw Error('登记查回内容与原提交不符。');
  unconfirmed.delete(declaration.actorUUID);return saved;
 }
 async function openActivityDeclaration(actorUUID){
  if(unconfirmed.has(actorUUID))return lookupDeclaration(unconfirmed.get(actorUUID));
  const window=await getActivityCheckpoint(actorUUID);return submitDeclaration({actors:[window.actor],activities:window.dependencies,window});
 }
 function open(actorUUIDs){
  if(!game.user.isGM)throw Error('恢复面板由 GM 启动；玩家使用原生治疗即可记录。');
  if(!panel){const {ApplicationV2}=foundry.applications.api;
   class RecoveryPanel extends ApplicationV2{
    static DEFAULT_OPTIONS={id:'exploration-recovery',classes:['exploration-recovery'],window:{title:'探索恢复',resizable:true},position:{width:610,height:'auto'}};
    actorUUIDs=[];
    async _prepareContext(){const actors=await capabilities.snapshot(this.actorUUIDs),ledgerStatus=await storage?.status();const id=getSessionId();return {actors,ledgerStatus,policy:game.user.getFlag?.('pf2e-third-party-automation','explorationPolicy')??{},data:id&&ledgerStatus?.state!=='blocked'?await coordinator.snapshot(id):null}}
    async _renderHTML({actors,data,policy,ledgerStatus}){
     const e=escapeHTML,pools=[...new Map(actors.map(a=>[a.pool.poolUUID,a])).values()];
     const rows=pools.map(a=>`<tr><td>${actors.filter(p=>p.pool.poolUUID===a.pool.poolUUID).map(p=>e(p.name)).join(' / ')}</td><td>${a.hp.value} / ${a.hp.max}</td><td><input aria-label="目标血量" data-pool="${e(a.pool.poolUUID)}" type="number" min="0" max="${a.hp.max}" value="${data?.session?.goalsByPool.find(g=>g.poolUUID===a.pool.poolUUID)?.targetHP??a.hp.max}"></td></tr>`).join('');
     const active=data?.session?.status==='running',manual=data?.session?.status==='recording',ready=!storage||ledgerStatus?.state==='ready',activityOpen=active&&data.session.activityCheckpoint?.phase==='open';
     const history=data?.activities??[],paused=data?.session?.status==='paused',unresolved=history.filter(a=>['uncertain','awaiting-evidence','completing','started'].includes(a.state)),unknownClocks=(data?.clocks??[]).filter(c=>c.state!=='confirmed');let result='选择成员后，一次启动恢复。';
     if(data){const remaining=data.session.goalsByPool.reduce((n,g)=>n+Math.max(0,g.targetHP-(data.actors.find(a=>a.pool.poolUUID===g.poolUUID)?.hp.value??0)),0);const reconstruction=sessionTimeSummary(data);result=formatSessionResult({...reconstruction,remainingHP:remaining,status:data.session.stopReason??data.session.status})}
     return `<div class="recovery-content">${ledgerSetupControls(ledgerStatus)}<p class="recovery-time" role="status">${e(result)}</p><table><thead><tr><th>成员 / 共用血池</th><th>当前 HP</th><th>目标 HP</th></tr></thead><tbody>${rows}</tbody></table><fieldset ${active||!ready?'disabled':''}><legend>恢复策略</legend><label><input type="checkbox" name="activityFirstRound" ${data?.session?.waitForActivityFirstRound?'checked':''}> 启动后先登记其他行动</label><p>登记耗时后封口继续；无需所有成员登记。与首轮手动治疗窗口择一使用。</p><label><input type="checkbox" name="manualFirstRound" ${data?.session?.waitForManualFirstRound?'checked':''}> 首轮等待手动治疗</label><p>首轮可用 Workbench 治疗一位非共用血池患者，原生应用 HP 并创建免疫后继续本轮；同轮的普通再聚能共用十分钟。等待时世界时间不推进。</p><label><input type="checkbox" name="risky" ${(data?.session?.riskySurgery??policy.riskySurgery)?'checked':''}> 激进治疗：先受到真实 1d8 伤害</label><label><input type="checkbox" name="focus" ${(data?.session?.requireFullFocus??policy.requireFullFocus)?'checked':''}> 结束时补满聚能</label><label><input type="checkbox" name="extension" ${(data?.session?.extendTreatment??policy.extendTreatment)?'checked':''}> 成功后允许延长至一小时</label><label><input type="checkbox" name="assurance" ${(data?.session?.useAssurance??policy.useAssurance)?'checked':''}> 优先使用对应技能的 Assurance</label><label>治疗 DC <select name="rank">${['auto','trained','expert','master','legendary'].map((rank,i)=>`<option value="${rank}" ${(data?.session?.treatmentRank??policy.treatmentRank??'trained')===rank?'selected':''}>${['自动',15,20,30,40][i]}（${['未知情境回落15','受训','专家','大师','传奇'][i]}）</option>`).join('')}</select></label><label>时间预算（分钟） <input name="budget" type="number" min="1" value="${(data?.session?.budgetEndsAt&&data?.session?.startedAt!==undefined?(data.session.budgetEndsAt-data.session.startedAt):policy.budgetSeconds??7200)/60}"></label><label>最多行动次数 <input name="activityBudget" type="number" min="1" max="100" value="${data?.session?.maxActivities??policy.maxActivities??100}"></label><p>圣疗与再聚能可循环使用。每日次数、法术位和消耗品保持关闭。条件修正未核验时采用选定 DC。</p></fieldset>${active&&data.session.manualCheckpoint?.phase==='open'?'<p class="recovery-unresolved">正在等待本轮 Workbench 手动治疗。完成原生 HP 应用和免疫后，点击继续本轮。</p>':''}${checkpointPreview(data)}${data?.activityCheckpointRequested?'<p role="status">已请求登记，等待当前原生行动与时间完成。</p>':''}<div class="recovery-actions">${active&&ready&&data.session.manualCheckpoint?.phase==='open'?'<button type="button" data-recovery="closeManualCheckpoint">继续本轮</button>':''}<button type="button" data-recovery="start" ${active||manual||!ready?'disabled':''}>按真实掷骰恢复</button><button type="button" data-recovery="record" ${active||manual||!ready?'disabled':''}>记录手动行动</button>${active&&ready&&!activityOpen&&!data.activityCheckpointRequested&&(!data.session.manualCheckpoint||data.session.manualCheckpoint.phase==='settled')?'<button type="button" data-recovery="openActivityCheckpoint">下一检查点登记</button>':''}${activityOpen&&ready?'<button type="button" data-recovery="closeActivityCheckpoint">封口并继续</button>':''}<button type="button" data-recovery="activity" ${(manual||activityOpen)&&ready&&!unconfirmed.size?'':'disabled'}>登记其他行动</button>${unconfirmed.size?'<button type="button" data-recovery="lookupActivity">查回上次登记</button>':''}<button type="button" data-recovery="stop" ${(active||manual)&&ready?'':'disabled'}>停止</button><button type="button" data-recovery="reconcile" ${paused&&ready?'':'disabled'}>核对已保存回执</button><button type="button" data-recovery="resume" ${paused&&ready&&!data.session.manual&&!unresolved.length&&!unknownClocks.length?'':'disabled'}>继续已确认会话</button><button type="button" data-recovery="review" ${paused&&ready?'':'disabled'}>核对后结束会话</button>${active&&ready&&!data.session.manual?'<button type="button" data-recovery="takeover">接管并停止</button>':''}<button type="button" data-recovery="refresh">刷新</button></div>${renderRecoveryHistory(data,{actors,messages:game.messages})}<details><summary>能力与限制</summary>${actors.map(a=>`<p>${e(a.name)}：医疗 ${a.medicine?.rank??0}；群体治疗 ${a.wardCapacity} 人；聚能 ${a.focus.value}/${a.focus.max}${a.pool.ready?'':`；${e(a.pool.reason)}`}${a.refocusUnsupported?.length?`；特殊再聚能规则需手动处理 ${e(a.refocusUnsupported.join(', '))}`:''}${a.unsupported?.length?`；需手动核对 ${e(a.unsupported.join(', '))}`:''}</p>`).join('')}</details></div>`;
    }
    _replaceHTML(result,content){content.innerHTML=result;this.listener?.abort();this.listener=new AbortController();content.addEventListener('click',event=>{const source=event.target.closest('[data-recovery-message]');if(source){event.preventDefault();void showRecoveryMessage(source.dataset.recoveryMessage,{game}).then(shown=>{if(!shown)globalThis.ui?.notifications?.warn?.('来源消息已删除、不可见或未能载入，请核对保存的来源 ID。')}).catch(onError);return}const action=event.target.closest('[data-recovery]')?.dataset.recovery;if(!action)return;void this.act(action,content).catch(onError)},{signal:this.listener.signal})}
    async act(action,content){if(action==='refresh')return this.render(true);if(['provision','initialize','selectLedger'].includes(action)){
      if(!storage)throw Error('exploration-storage-unavailable');const options=await setupChoices({game,select:action==='selectLedger'});if(!options)return;
      const result=await storage[action==='selectLedger'?'select':action](options);
      if(result.requiresReload)globalThis.ui?.notifications?.info?.('账本已选择，请重载页面后再准备恢复。');else if(action==='provision')await storage.initialize(options);
      return this.render(true);
     }if(action==='closeManualCheckpoint'){
      const selected=await coordinator.snapshot(getSessionId());if(selected.session?.status!=='running'||selected.session.manualCheckpoint?.phase!=='open')throw Error('本轮手动治疗窗口已关闭。');
      await coordinator.closeManualCheckpoint(checkpointBinding(selected.session.manualCheckpoint));return this.render(true);
     }if(action==='openActivityCheckpoint'){await coordinator.openActivityCheckpoint(getSessionId());return this.render(true)}if(action==='closeActivityCheckpoint'){
      const selected=await coordinator.snapshot(getSessionId());if(selected.session?.status!=='running'||selected.session.activityCheckpoint?.phase!=='open')throw Error('探索登记窗口已关闭。');await coordinator.closeActivityCheckpoint(activityBinding(selected.session.activityCheckpoint));return this.render(true);
     }if(action==='lookupActivity'){for(const declaration of unconfirmed.values())await lookupDeclaration(declaration);return this.render(true)}if(action==='takeover'){await coordinator.takeover(getSessionId());return this.render(true)}if(action==='activity'){
      const selected=await coordinator.snapshot(getSessionId());
      const checkpoint=selected.session?.activityCheckpoint;if(selected.session?.status!=='recording'&&(selected.session?.status!=='running'||checkpoint?.phase!=='open'))throw Error('探索登记窗口已关闭。');
      if(unconfirmed.size)throw Error('请先查回上次未确认的登记。');
      const actors=await capabilities.snapshot(this.actorUUIDs);await submitDeclaration({actors,activities:selected.activities??[],session:selected.session,...selected.session.status==='running'?{window:{binding:activityBinding(checkpoint),phase:'open'}}:{}});return this.render(true);
     }if(action==='reconcile'){await coordinator.reconcile(getSessionId());return this.render(true)}if(action==='resume'){await coordinator.resume(getSessionId());return this.render(true)}if(action==='review'){const note=await foundry.applications.api.DialogV2.wait({window:{title:'核对后结束恢复会话'},content:'<p>请先核对实际 HP、资源、聊天回执和世界时间。结束会话会保留未知记录；旧尝试不会重放。尚未确认的执行仍会占用相应时间与治疗目标，核对说明不会解除这些限制。</p><label>核对说明 <textarea name="note" required></textarea></label>',buttons:[{action:'close',label:'保存核对说明并结束',callback:(_e,b)=>new FormData(b.form).get('note')},{action:'cancel',label:'取消',callback:()=>null}],rejectClose:false});if(note===null)return;await coordinator.review(getSessionId(),{note,userId:game.user.id});return this.render(true)}if(action==='stop'){await coordinator.stop(getSessionId());return this.render(true)}
     const config={...recoveryDefaults(await capabilities.snapshot(this.actorUUIDs)),...recoveryBudgetPolicy({minutes:Number(content.querySelector('[name=budget]').value),activities:Number(content.querySelector('[name=activityBudget]').value)}),actorUUIDs:this.actorUUIDs,goalsByPool:[...content.querySelectorAll('[data-pool]')].map(el=>({poolUUID:el.dataset.pool,targetHP:Number(el.value)})),riskySurgery:content.querySelector('[name=risky]').checked,requireFullFocus:content.querySelector('[name=focus]').checked,extendTreatment:content.querySelector('[name=extension]').checked,useAssurance:content.querySelector('[name=assurance]').checked,treatmentRank:content.querySelector('[name=rank]').value,waitForManualFirstRound:action==='start'&&content.querySelector('[name=manualFirstRound]')?.checked===true,waitForActivityFirstRound:action==='start'&&content.querySelector('[name=activityFirstRound]')?.checked===true,manual:action==='record'};
     if(config.waitForManualFirstRound&&config.waitForActivityFirstRound)throw Error('请选择一种初始登记窗口。');
     await game.user.setFlag?.('pf2e-third-party-automation','explorationPolicy',{riskySurgery:config.riskySurgery,requireFullFocus:config.requireFullFocus,useAssurance:config.useAssurance,extendTreatment:config.extendTreatment,treatmentRank:config.treatmentRank,budgetSeconds:config.budgetSeconds,maxActivities:config.maxActivities});await start(config);return this.render(true);
    }
    async close(options){this.listener?.abort();return super.close(options)}
   }panel=new RecoveryPanel();
  }
  panel.actorUUIDs=actorUUIDs??Array.from(game.actors.party?.members??[]).filter(a=>a.type==='character').map(a=>a.uuid);return panel.render(true);
 }
 return {open,openActivityDeclaration,refresh:()=>panel?.render(true)};
}
