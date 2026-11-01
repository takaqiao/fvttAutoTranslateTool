import {activityCheckpointBinding,sameActivityCheckpoint} from './schema.mjs';
import {normalizeRecoveryPreferences,resolveRecoveryGoals,encodeRecoveryPreferences,decodeRecoveryPreferences} from './recovery-goals.mjs';
import {normalizeCheckpointActivity} from './manual-time.mjs';
import {canonicalJSON} from './revision-codec.mjs';
export const escapeHTML=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const labels={'goals-met':'恢复目标完成',running:'恢复中',recording:'手动记录中',paused:'已停止',confirmed:'已确认',started:'进行中',completing:'结算中','awaiting-evidence':'待核对证据',uncertain:'结果未知',blocked:'无法继续',cancelled:'已取消','treat-wounds':'治疗伤势',refocus:'再聚能',manual:'手动行动','focus-healing':'圣疗','user-stopped':'已手动停止','encounter-started':'遭遇已开始，恢复已停止',budget:'达到预算',closed:'已核对结束','client-context-lost':'客户端已重载，请核对后接续','gm-reviewed-closed':'GM 已核对结束，未知结果保留','refocus-recovery-unadapted':'特殊再聚能规则尚未适配，请手动处理','consecutive-treatment-failures':'达到连续治疗失败上限'};
export function formatSessionResult(record){return `已确认经过 ${Math.round(record.durationSeconds/60*10)/10} 分钟；尚缺 ${record.remainingHP??0} HP；${evidenceText(record.status)??''}`}
export function sessionTimeSummary(data){return {durationSeconds:data.session.cursorAt-data.session.startedAt}}
export function recoveryBudgetPolicy({minutes,activities}){if(!Number.isFinite(minutes)||minutes<=0)throw Error('时间预算应为正数分钟。');if(!Number.isInteger(activities)||activities<1||activities>100)throw Error('行动次数应为1至100。');return {budgetSeconds:minutes*60,maxActivities:activities}}
export function recoveryDefaults(actors){return {goalsByPool:[...new Map(actors.map(a=>[a.pool.poolUUID,{poolUUID:a.pool.poolUUID,targetHP:a.hp.max}])).values()],recoveryStrategy:'success-first',allowFiniteResources:false,riskySurgery:false,requireFullFocus:false,budgetSeconds:7200,maxActivities:100}}
function savedRecoveryPolicy(policy){
 try{
  const field=Object.getOwnPropertyDescriptor(policy,'recovery');
  if(!field){if('recovery' in policy)throw Error('invalid-recovery-preferences');return {preferences:normalizeRecoveryPreferences()}}
  if(!field.enumerable||!Object.hasOwn(field,'value'))throw Error('invalid-recovery-preferences');
  return {preferences:decodeRecoveryPreferences(field.value)};
 }
 catch(error){return {preferences:normalizeRecoveryPreferences(),reason:error.message}}
}
function selectedRecoveryPreferences(preferences,actorUUIDs){
 return {...preferences,targetIntentsByActor:Object.fromEntries(actorUUIDs.map(uuid=>[uuid,preferences.targetIntentsByActor[uuid]??{mode:'max',value:null}]))};
}
function targetControls(content,actorUUIDs,preferences){
 const modes=[...content.querySelectorAll('[data-target-mode]')].filter(el=>Object.hasOwn(el.dataset,'targetMode'));
 const targetIntentsByActor={...preferences.targetIntentsByActor};
 if(modes.length){
  if(modes.length!==actorUUIDs.length||new Set(modes.map(el=>el.dataset.actor)).size!==modes.length||modes.some(el=>!actorUUIDs.includes(el.dataset.actor)))throw Error('invalid-recovery-preferences');
  for(const el of modes){const value=content.querySelector(`[data-target-value][data-actor="${el.dataset.actor}"]`)?.value;if(el.value!=='max'&&(value===undefined||!String(value).trim()))throw Error('invalid-recovery-preferences');targetIntentsByActor[el.dataset.actor]={mode:el.value,value:el.value==='max'?null:Number(value)}}
 }
 const limit=content.querySelector('[name=failureLimit]')?.value;
 if(limit!==undefined&&!String(limit).trim())throw Error('invalid-recovery-preferences');
 return {patientControls:modes.length>0,preferences:normalizeRecoveryPreferences({version:1,targetIntentsByActor,requireNoWounded:content.querySelector('[name=wounded]')?.checked??preferences.requireNoWounded,failureStop:{enabled:content.querySelector('[name=failureStop]')?.checked??preferences.failureStop.enabled,limit:limit===undefined?preferences.failureStop.limit:Number(limit)}})};
}
function targetExplanation(target){
 return target.intent.mode==='absolute'&&target.intent.value>target.basisMaxHP?`原目标 ${target.intent.value} HP，当前最大 HP ${target.basisMaxHP}，本次目标 ${target.targetHP} HP。`:`按当前最大 HP ${target.basisMaxHP}，本次目标 ${target.targetHP} HP。`;
}
function treatmentFailureSummary(data){
 const remaining=data?.treatmentFailures?.remaining??[];if(!remaining.length)return '';
 return `<section class="recovery-failures" aria-label="连续治疗失败"><p>以下患者已停止新的恢复；已保存结果保留。</p><ul>${remaining.map(p=>`<li>${escapeHTML(data.actors.find(a=>a.actorUUID===p.patientUUID)?.name??p.patientUUID)}：连续失败 ${p.streak} / ${p.limit}；HP ${p.currentHP} / ${p.targetHP}（尚缺 ${p.hpGap} HP）${p.woundedGap?'；wounded 尚未清除':''}。</li>`).join('')}</ul></section>`;
}
const evidenceLabels={
 'legacy-recovery-view-only':'旧版恢复会话不能继续，请核对后关闭，再开始新的计算恢复',
 'computed-batch-unresolved':'计算批次仍有未确认结果，请先核对已保存回执',
 'unresolved-computed-batch':'计算批次仍有未确认结果，请先核对已保存回执',
 'computed-boundary-unsettled':'时间已推进，本段不自动重抽；请核对当前状态后关闭会话',
 'computed-calculation-unsettled':'恢复计算已开始，结果尚未确认，不自动重抽',
 'computed-model-unavailable':'角色恢复规则模型不可用',
 'computed-treatment-model-unverified':'治疗规则模型尚未核验',
 'native-healing-model-unverified':'额外治疗规则尚未适配（如 Magic Hands），请手动处理',
 'healing-received-modifier-unverified':'患者的受疗修正尚未适配，请手动处理',
 'computed-treatment-context-changed':'治疗情境已改变，停止本次治疗',
 'computed-actor-changed':'角色状态已改变，请核对已保存结果',
 'computed-actor-domain-changed':'参与角色或共享血池已改变',
 'computed-pool-changed':'共享血池绑定已改变',
 'computed-clock-unconfirmed':'计算恢复的时间提交尚未确认',
 'computed-result-unknown':'计算恢复的写入结果未知',
 'computed-items-unconfirmed':'恢复效果写入尚未确认',
 'computed-wounded-unconfirmed':'wounded 移除尚未确认',
 'computed-driver-changed':'恢复执行权已改变',
 'actor-cannot-act':'角色目前无法行动',
 'focus-healing-context-changed':'聚能治疗情境已改变',
 'rest-full-hp-changed':'休息期间 HP 已改变，未清除 wounded',
 'activity-limit':'达到行动次数上限',
 'failure-stop':'达到连续治疗失败上限',
 unreachable:'无可执行的恢复行动',
 rest:'休息',
 'needs-review':'角色状态需要人工核对，恢复已停止',
 'closed-without-replay':'GM 已核对结束，旧结果不重放',
 remaining:'尚未达到恢复目标',
 'game-pause-failed':'未能暂停游戏，请由 GM 暂停后重试',
 'exploration-clock-pause-unavailable':'无法确认日历停钟状态。Calendaria 的时间提交间隔需大于 1 秒，请核对后重试',
 'exploration-clock-still-running':'游戏已暂停，但日历仍在运行；请停止日历时钟后重试',
 'exploration-clock-pause-unconfirmed':'日历停钟前的时间尚未提交完成，请稍后重试',
 'accepted-current-state':'已确认当前状态，可开始新的恢复',
 'native-application-receipt':'缺少原生 HP 应用回执','native-immunity-receipt':'缺少原生免疫回执',
 'checkpoint-time-confirmation':'等待本轮十分钟时间确认','native-manual-source-unavailable':'尚未完成本轮 Workbench 治疗',
 'native-immunity-timing-unconfirmed':'原生免疫的开始时刻或冷却时长尚未确认','native-source-identity-unproven':'原生治疗来源尚未确认',
 'ambiguous-native-application':'同一治疗结果的 HP 应用次数尚未确认','unreserved-manual-source':'出现未预约的手动治疗，请核对后结束会话',
 'manual-source-requires-review':'人工登记的行动来源需 GM 核对',
 'shared-hp-completion-unavailable':'共用血池主人的写入尚未确认','hp-pool-source-unavailable':'共用血池来源尚未确认',
 'world-time-source-unconfirmed':'世界时间来源回执尚未确认','world-time-conflict':'世界时间与计划不一致',
 'external-world-time-change':'世界时间已被其他操作改变，请核对当前状态','external-rest-time-authority':'世界时间由外部休息操作管理，恢复已停止',
 'world-time-conflict-after-claim':'保存提交声明后世界时间发生变化','clock-never-replayed':'时间提交尚未确认，保留原尝试',
 'clock-in-flight':'已有时间提交正在执行','time-effects-unconfirmed':'时间相关效果尚未完成确认',
 'passive-completion-unavailable':'被动恢复尚无可确认完成的适配','passive-completion-unproven':'被动恢复缺少完成回执',
 'missing-passive-checkpoint':'缺少被动恢复检查点','passive-rules-changed':'检查点内的被动恢复规则发生变化',
 'unresolved-evidence-no-replay':'旧会话还有未确认的结果或时间；先在那场会话里处理，处理不了可换新账本',
 'replace-ledger-session-active':'仍有恢复会话在进行，请先停止再换新账本','replace-ledger-ready-required':'探索账本尚未就绪，暂不能换新'
};
const evidenceText=value=>labels[value]??evidenceLabels[value]??value;
function ledgerSetupControls(status){
 if(!status||status.state==='ready')return '';
 // Once the configured ledger changes, this page can use neither ledger until it reloads; the page that made the switch also refuses another selection.
 if(status.state==='blocked'&&status.reason==='revision-root-changed')return '<section class="recovery-unresolved"><p>探索账本已切换，请重载页面后再继续。</p></section>';
 const migrating=['legacy','migration-required'].includes(status.state),text=status.state==='unconfigured'?'先准备探索账本，再启动恢复。':migrating?'已有探索历史需要升级账本。升级会保留记录，暂停旧的恢复任务，并保留尚未确认的尝试。恢复任务由你另行启动。':`探索账本暂不可用：${evidenceText(status.reason)}。请选择仍存在且仅供 GM 管理的账本。`;
 return `<section class="recovery-unresolved"><p>${escapeHTML(text)}</p>${status.state==='unconfigured'?'<button type="button" data-recovery="provision">创建并准备探索账本</button>':migrating?'<button type="button" data-recovery="initialize">升级探索账本</button>':'<button type="button" data-recovery="selectLedger">选择已有账本</button>'}</section>`;
}
const setupConfirmations='<label class="recovery-option"><input type="checkbox" name="issuersStopped" required> 所有旧恢复操作均已停止</label><label class="recovery-option"><input type="checkbox" name="clientsReloaded" required> 相关客户端均已重载</label><label class="recovery-option"><input type="checkbox" name="recoveryDisabled" required> 当前没有自动恢复正在执行</label>';
const confirmedSetup=form=>({issuersStopped:form.has('issuersStopped'),clientsReloaded:form.has('clientsReloaded'),recoveryDisabled:form.has('recoveryDisabled')});
async function setupChoices({game,select=false}){
 const candidates=Array.from(game.journal?.contents??[]).filter(j=>j.flags?.['pf2e-third-party-automation']?.explorationLedger||j.flags?.['pf2e-third-party-automation']?.explorationLedgerRoot);
 return foundry.applications.api.DialogV2.wait({window:{title:select?'选择探索账本':'准备探索账本'},content:`${select?`<label>账本 <select name="rootUUID">${candidates.map(j=>`<option value="${escapeHTML(j.uuid)}">${escapeHTML(j.name)} (${escapeHTML(j.id)})</option>`).join('')}</select></label>`:''}<p>先停止正在运行的恢复，并让相关 GM 和玩家重载到当前版本。准备完成后不会自动开始恢复。</p>${setupConfirmations}`,buttons:[{action:'prepare',label:select?'选择账本':'准备账本',callback:(_event,b)=>{const form=new FormData(b.form);return {...confirmedSetup(form),...select?{rootUUID:form.get('rootUUID')}:{}}}},{action:'cancel',label:'取消',callback:()=>null}],rejectClose:false});
}
async function replaceChoices({unfinishedSessions,unconfirmedClocks}){
 return foundry.applications.api.DialogV2.wait({window:{title:'换新探索账本'},content:`<p>旧账本里还有 ${escapeHTML(unfinishedSessions)} 场没结束的会话、${escapeHTML(unconfirmedClocks)} 个未确认的时钟。</p><p>换新之后它们不再阻挡，也不会被重放；以角色卡上现在的 HP 和世界时间为准。</p><p>旧账本原样保留为归档。新账本的名字带今天的日期和时间，清理归档时别删错。</p><p>有限药品的跨会话用量会从新账本重新算起。</p>${setupConfirmations}`,buttons:[{action:'replace',label:'换新账本',callback:(_event,b)=>confirmedSetup(new FormData(b.form))},{action:'cancel',label:'取消',callback:()=>null}],rejectClose:false});
}
const minutes=value=>Math.round(value/60*10)/10;
const isComputed=data=>data?.session?.executionMode==='computed-v1';
function canResumeComputed(data,game){
 return isComputed(data)&&!data.session.manual&&data.session.status==='paused'&&data.canResumeComputed!==false&&
  !data.session.computedRecovery?.pendingBoundary&&
  !data.session.computedRecovery?.pendingCalculation&&
  !(data.session.computedRecovery?.batches??[]).some(batch=>batch.state!=='confirmed')&&
  !(data.activities??[]).some(activity=>['uncertain','awaiting-evidence','completing','started'].includes(activity.state))&&
  !(data.clocks??[]).some(clock=>clock.state!=='confirmed')&&
  (!Number.isFinite(game.time?.worldTime)||game.time.worldTime===data.session.cursorAt);
}
function computedResultText(result){
 const parts=[];
 if(result.check)parts.push(`${result.check.assurance?'Assurance':'检定'} ${result.check.total} / DC ${result.check.dc}`);
 const outcome=result.outcome??result.degree;if(Number.isInteger(outcome))parts.push(['大失败','失败','成功','大成功'][outcome]);
 if(result.healing)parts.push(`治疗 ${result.healing} HP`);
 if(result.damage)parts.push(`伤害 ${result.damage} HP`);
 if(Number.isFinite(result.hpBefore)&&Number.isFinite(result.hpAfter))parts.push(`HP ${result.hpBefore} → ${result.hpAfter}`);
 if(Number.isFinite(result.focusBefore)&&Number.isFinite(result.focusAfter))parts.push(`聚能 ${result.focusBefore} → ${result.focusAfter}`);
 if(result.reason)parts.push(evidenceText(result.reason));
 if(result.check&&!result.check.assurance)parts.push(`d20：${result.check.face}`);
 for(const [key,label]of [['riskyDice','手术骰'],['healingDice','治疗骰'],['damageDice','伤害骰']])if(result[key]?.length)parts.push(`${label}：${result[key].join('、')}`);
 return parts.filter(Boolean).join('；');
}
function renderComputedEvidence(data){
 const e=escapeHTML,batches=(data.session.computedRecovery?.batches??[]).filter(batch=>batch.state!=='confirmed'),clocks=(data.clocks??[]).filter(clock=>clock.state!=='confirmed');
 const boundary=data.session.computedRecovery?.pendingBoundary;
 const pending=[...boundary?[{...boundary,label:'恢复计算',state:'awaiting-evidence',reason:'computed-boundary-unsettled'}]:[],...batches.map(batch=>({...batch,label:'计算批次'})),...clocks.map(clock=>({...clock,label:'时间提交'}))];
 const recovery=data.session.computedRecovery,diagnostics=[...new Set([...(recovery?.summary?.diagnostics??[]),...(recovery?.simulationState?.diagnostics??[])])];
 if(recovery?.pendingCalculation&&!boundary&&!batches.length)pending.unshift({...recovery.pendingCalculation,label:'恢复计算',state:'awaiting-evidence',reason:'computed-calculation-unsettled'});
 return `${pending.length?`<section class="recovery-unresolved"><p>尚未确认；已保存的尝试不会重放。</p><ul>${pending.map(record=>`<li>${record.label}${record.id?` <code>${e(record.id)}</code>`:''} · 第 ${e(minutes(record.from-data.session.startedAt))}${record.to===undefined?' 分钟起':` → ${e(minutes(record.to-data.session.startedAt))} 分钟`} · ${e(evidenceText(record.state))}${record.reason?`；${e(evidenceText(record.reason))}`:''}</li>`).join('')}</ul></section>`:''}${diagnostics.length?`<details><summary>规则限制</summary><ul>${diagnostics.map(reason=>`<li>${e(evidenceText(reason))}</li>`).join('')}</ul></details>`:''}${recovery?.review?.note?`<section class="recovery-review"><p>GM 核对说明</p><p>${e(recovery.review.note)}</p></section>`:''}`;
}
function renderComputedReadback(readback,actors){
 if(!readback)return '';
 const e=escapeHTML,names=new Map(actors.map(actor=>[actor.actorUUID,actor.name]));
 const rows=readback.results.map(batch=>`<li>批次 <code>${e(batch.batchId)}</code><ul>${[['actors','角色状态'],['items','效果']].flatMap(([key,label])=>batch[key].map(row=>`<li>${e(names.get(row.actorUUID)??row.actorUUID)} · ${label}：${row.state==='confirmed'?'已应用（当前值符合保存结果）':'尚无法确认'}</li>`)).join('')}</ul></li>`).join('');
 return `<details class="recovery-readback"><summary>本次只读核对</summary>${rows?`<ul>${rows}</ul>`:'<p>没有待核对的计算批次。</p>'}<p>本次核对仅查看当前状态，不重试写入；未确认批次仍然保留。</p></details>`;
}
function renderComputedSummary(data){
 const summary=data.session.computedRecovery?.summary;if(!summary)return '';
 const e=escapeHTML,names=new Map(summary.actors.map(actor=>[actor.actorUUID,actor.name])),wounded=value=>typeof value==='boolean'?Number(value):value;
 const events=(summary.events??(data.session.computedRecovery.batches??[]).filter(batch=>batch.state==='confirmed').flatMap(batch=>batch.events??[])).map(event=>{
  const patients=event.patientUUIDs??(event.patientUUID?[event.patientUUID]:[]),results=event.results??[event];
  return `<li><p>${e(names.get(event.actorUUID)??event.actorUUID)}${patients.length?` → ${patients.map(uuid=>e(names.get(uuid)??uuid)).join('、')}`:''} · ${e(evidenceText(event.providerId))} · 第 ${e(minutes(event.startedAt-data.session.startedAt))} → ${e(minutes(event.endsAt-data.session.startedAt))} 分钟</p>${results.map(result=>`<p>${e(computedResultText(result))}</p>`).join('')}</li>`;
 }).join('');
 return `<section class="recovery-computed-summary" aria-label="恢复结果"><div class="recovery-results-scroll"><table class="recovery-results"><thead><tr><th>成员</th><th>HP（目标）</th><th>聚能</th><th>wounded</th><th>状态</th></tr></thead><tbody>${summary.actors.map(actor=>`<tr><td>${e(actor.name)}</td><td>${e(actor.hpBefore)} → ${e(actor.hpAfter)} / ${e(actor.hpMax)}（${e(actor.targetHP)}）</td><td>${e(actor.focusBefore)} → ${e(actor.focusAfter)} / ${e(actor.focusMax)}</td><td>${e(wounded(actor.woundedBefore))} → ${e(wounded(actor.woundedAfter))}</td><td>${e(evidenceText(actor.status))}</td></tr>`).join('')}</tbody></table></div><details class="recovery-computed-events"><summary>过程详情（${e(summary.activityCount)} 项行动）</summary><ol>${events}</ol></details></section>`;
}
function historyAnnotations(activity){
 const lines=[],duration=activity.durationSource,temporal=activity.temporalSource,options=activity.options??{};
 const durationLabels={'user-declared':'玩家声明','item-text':'条目说明','table-convention':'本桌约定'};
 if(duration)lines.push(`时长出处：${durationLabels[duration.type]??'未识别的出处'}${duration.detail?` · ${duration.detail}`:''}`);
 if(temporal?.type==='checkpoint-declaration')lines.push(`${activity.state==='confirmed'?'声明活动时间已计入':'声明活动时间尚未完成'}；规则效果未核验。`);
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
 const e=escapeHTML,names=new Map([...(data.actors??[]),...actors].map(a=>[a.actorUUID,a.name])),history=data.activities??[];
 const title=a=>`${e(names.get(a.actorUUID)??a.actorUUID)}${a.patientUUIDs?.length?` → ${[...new Set(a.patientUUIDs)].map(id=>e(names.get(id)??id)).join('、')}`:''} · ${e(a.options?.label??labels[a.providerId]??a.providerId)}`;
 const reason=a=>a.reason?` · ${e(evidenceText(a.reason))}`:'';
 const unresolved=history.filter(a=>['uncertain','awaiting-evidence','completing','started'].includes(a.state)),clocks=(data.clocks??[]).filter(c=>c.state!=='confirmed');
 const context=(data.session.status==='paused'||data.session.review)&&(unresolved.length||clocks.length)?`<section class="recovery-unresolved"><p>尚未确认</p><ul>${unresolved.map(a=>`<li>${title(a)} · ${e(evidenceText(a.state))}${reason(a)}</li>`).join('')}${clocks.map(c=>`<li>时间提交 · 第 ${e(minutes(c.from-data.session.startedAt))} → ${e(minutes(c.to-data.session.startedAt))} 分钟 · ${e(evidenceText(c.state))}${reason(c)} <code>${e(c.id)}</code></li>`).join('')}</ul><p>核对只读取已有回执，不重掷、不应用、不推进时间；未知项会保留。</p></section>`:'';
 const rows=history.map(a=>{
  const missing=[...new Set(a.options?.missing??[])];
  const seen=new Set(),sources=[];
  for(const [label,ids,chat]of [['检定',a.proof?.checkIds,true],[a.source?.type==='user-record'?'人工登记':'骰点结果',a.proof?.resultIds,true],['完成回执',a.proof?.receiptIds,true],['免疫效果',a.proof?.immunityIds,false],['行动来源',a.source?.messageId?[a.source.messageId]:[],true]]){
   for(const id of ids??[]){if(typeof id!=='string'||!id||seen.has(id))continue;seen.add(id);const text=`${e(label)} <code>${e(id)}</code>`;sources.push(`<li>${chat&&visibleMessage(messages,id)?`<button type="button" class="recovery-evidence-link" data-recovery-message="${e(id)}" aria-label="在聊天中查看${e(label)} ${e(id)}">${text}</button>`:text}</li>`)}
  }
  return `<li class="recovery-history-entry" data-recovery-activity-id="${e(a.id)}"><p class="recovery-activity-title">${title(a)} · ${e(minutes(a.durationSeconds??a.endsAt-a.startedAt))} 分钟 <strong>${e(evidenceText(a.state))}</strong></p>${historyAnnotations(a)}${a.reason?`<p>${e(evidenceText(a.reason))}</p>`:''}${missing.length?`<ul class="recovery-missing">${missing.map(m=>`<li>${e(evidenceText(m))}</li>`).join('')}</ul>`:''}<details class="recovery-evidence"><summary>来源与回执</summary><p>行动 ID：<code>${e(a.id)}</code></p>${sources.length?`<ul>${sources.join('')}</ul>`:'<p>尚无已保存的来源回执。</p>'}</details></li>`;
 }).join('');
 return `<details class="recovery-history-details"><summary>完整记录（${history.length} 项行动）</summary>${context}<ol class="recovery-history">${rows}</ol>${data.session.review?.note?`<section class="recovery-review"><p>GM 核对说明</p><p>${e(data.session.review.note)}</p></section>`:''}</details>`;
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
export async function promptActivityDeclaration({actors,activities=[],window,record}){
 if(window?.phase!=='open')throw Error('探索登记窗口已关闭。');const binding=activityCheckpointBinding(window.binding);
 const names=new Map(actors.map(a=>[a.actorUUID,a.name]));
 const value=await foundry.applications.api.DialogV2.wait({window:{title:'登记探索行动'},content:`
  <label>执行者 <select name="actor">${actors.map(a=>`<option value="${escapeHTML(a.actorUUID)}">${escapeHTML(a.name)}</option>`).join('')}</select></label>
  <label>行动名称 <input name="label" required></label>
  <label>耗时 <input name="duration" type="number" min="0" step="any" required><select name="unit"><option value="60">分钟</option><option value="1">秒</option></select></label>
  <label>时长出处 <select name="durationSource"><option value="user-declared">玩家声明</option><option value="item-text">条目说明</option><option value="table-convention">本桌约定</option></select></label>
  <label>出处说明 <input name="durationDetail" maxlength="500" placeholder="例如：修理条目写明十分钟"></label>
  <details><summary>时间与前置行动</summary>
   <p>当前起点为世界时间 ${escapeHTML(binding.from)} 秒。不得早于以此起点计分钟，已流逝时间不能回填。</p>
   <label>不得早于（分钟） <input name="notBefore" type="number" min="0" step="any"></label>
   <label>执行顺序（从零开始，可留空） <input name="order" type="number" min="0" step="1"></label>
   <label>须先完成 <select name="dependsOn" multiple>${activities.map(a=>`<option value="${escapeHTML(a.id)}">${escapeHTML(names.get(a.actorUUID)??'')} · ${escapeHTML(a.label??a.options?.label??labels[a.providerId]??a.providerId)}</option>`).join('')}</select></label>
  </details><p>登记仅声明耗时与执行者占用，规则效果不会自动执行。封口继续时共同计时；无需所有成员登记。</p>`,buttons:[{action:'record',label:'登记',callback:(_event,b)=>{const form=new FormData(b.form);return {...Object.fromEntries(form),dependsOn:form.getAll('dependsOn')}}},{action:'cancel',label:'取消',callback:()=>false}],rejectClose:false});
 if(!value)return null;
 const duration=Number(value.duration),unit=Number(value.unit),seconds=duration*unit;
 if(!actors.some(a=>a.actorUUID===value.actor)||!value.label?.trim()||value.duration?.trim()===''||![1,60].includes(unit)||!Number.isFinite(seconds)||seconds<0)throw Error('请输入行动名称和有效耗时。');
 const declaration={registrationId:crypto.randomUUID(),checkpointBinding:binding,actorUUID:value.actor,label:value.label.trim(),durationSeconds:seconds,durationSource:{type:value.durationSource,...value.durationDetail?.trim()?{detail:value.durationDetail.trim()}:{}},dependsOn:value.dependsOn};
 if(value.notBefore?.trim()){const absolute=binding.from+Number(value.notBefore)*60;if(!Number.isFinite(absolute))throw Error('请输入有效的行动时刻。');declaration.notBefore=absolute}
 if(value.order?.trim()){const order=Number(value.order);if(!Number.isSafeInteger(order)||order<0)throw Error('执行顺序须为非负整数。');declaration.order=order}
 normalizeCheckpointActivity(declaration);
 try{return await record(declaration)}catch(error){error.declaration=structuredClone(declaration);throw error}
}
export function createRecoveryPanel({game,coordinator,capabilities,start,storage,prepareContext,record,getActivityCheckpoint,lookupCheckpointActivity,getSessionId,onError=console.error}){
 const reportError=error=>onError(evidenceLabels[error.message]?new Error(evidenceLabels[error.message],{cause:error}):error);
 let panel;const unconfirmed=new Map();
 async function submitDeclaration(options){try{return await promptActivityDeclaration({...options,record})}catch(error){if(error.declaration&&error.declarationRejected!==true){unconfirmed.set(error.declaration.actorUUID,error.declaration);panel?.render({force:false})}throw error}}
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
  if(!game.user.isGM)throw Error('恢复面板由 GM 启动。');
  if(!panel){const {ApplicationV2}=foundry.applications.api;
   class RecoveryPanel extends ApplicationV2{
    static DEFAULT_OPTIONS={id:'exploration-recovery',classes:['exploration-recovery'],window:{title:'探索恢复',resizable:true},position:{width:720,height:'auto'}};
    actorUUIDs=[];
    async _prepareContext(){if(prepareContext)return {...await prepareContext(this.actorUUIDs),policy:game.user.getFlag?.('pf2e-third-party-automation','explorationPolicy')??{}};const actors=await capabilities.snapshot(this.actorUUIDs),ledgerStatus=await storage?.status();const id=getSessionId();return {actors,ledgerStatus,policy:game.user.getFlag?.('pf2e-third-party-automation','explorationPolicy')??{},data:id&&ledgerStatus?.state!=='blocked'?await coordinator.snapshot(id):null}}
    async _renderHTML({actors,data,policy,ledgerStatus}){
     // A selected id the ledger does not hold, such as a session of the ledger this one replaced, is no session.
     if(!data?.session)data=null;
     if(this.computedReadback?.sessionId!==data?.session.id)this.computedReadback=null;
     const e=escapeHTML,pools=[...new Map(actors.map(a=>[a.pool.poolUUID,a])).values()];
     const saved=savedRecoveryPolicy(policy),preferences=selectedRecoveryPreferences(saved.preferences,actors.map(a=>a.actorUUID));
     const active=data?.session?.status==='running',manual=data?.session?.status==='recording',ready=ledgerStatus?ledgerStatus.state==='ready':!storage,computed=isComputed(data);
     const resolved=actors.every(a=>a.pool.ready)?resolveRecoveryGoals(actors,preferences):null,preview=resolved?.goalsByPool??recoveryDefaults(actors).goalsByPool;
     this.targetActors=structuredClone(actors);this.targetPreview={actorUUIDs:actors.map(a=>a.actorUUID),goalsByPool:preview.map(g=>({...g}))};
     const rows=actors.map(a=>{
      const intent=preferences.targetIntentsByActor[a.actorUUID],target=resolved?.recoveryGoals.patientTargets.find(t=>t.patientUUID===a.actorUUID),frozen=data?.session?.recoveryGoals?.patientTargets.find(t=>t.patientUUID===a.actorUUID),poolGoal=data?.session?.goalsByPool.find(g=>g.poolUUID===(frozen?.poolUUID??a.pool.poolUUID)),disabled=active||manual||!ready?'disabled':'';
      return `<tr><td>${e(a.name)}</td><td>${a.hp.value} / ${a.hp.max}</td><td><div class="recovery-target-controls"><select aria-label="${e(a.name)} 目标模式" data-target-mode data-actor="${e(a.actorUUID)}" ${disabled}>${['max','percent','absolute'].map((mode,i)=>`<option value="${mode}" ${intent.mode===mode?'selected':''}>${['最大 HP','百分比','绝对 HP'][i]}</option>`).join('')}</select><input aria-label="${e(a.name)} 目标值" data-target-value data-actor="${e(a.actorUUID)}" type="number" min="0" step="${intent.mode==='percent'?'any':'1'}" ${intent.mode==='percent'?'max="100"':''} value="${intent.value??''}" ${intent.mode==='max'?'hidden disabled':disabled}></div><small data-target-explanation data-actor="${e(a.actorUUID)}">${target?e(targetExplanation(target)):e(a.pool.reason??'血池尚未确认')}</small>${poolGoal?`<small>会话目标 ${frozen?.targetHP??poolGoal.targetHP} HP${frozen?`；启动时最大 HP ${frozen.basisMaxHP}`:''}${frozen&&frozen.targetHP!==poolGoal.targetHP?`；共享执行目标 ${poolGoal.targetHP} HP`:''}。</small>`:''}</td></tr>`;
     }).join('');
     const targetPreview=`<details class="recovery-target-preview"><summary>查看血池目标汇总</summary>${pools.map(a=>`<p>${actors.filter(p=>p.pool.poolUUID===a.pool.poolUUID).map(p=>e(p.name)).join(' / ')}：下次启动${actors.filter(p=>p.pool.poolUUID===a.pool.poolUUID).length>1?'共享':''}目标 <output data-pool="${e(a.pool.poolUUID)}" value="${preview.find(g=>g.poolUUID===a.pool.poolUUID).targetHP}">${preview.find(g=>g.poolUUID===a.pool.poolUUID).targetHP}</output> HP。</p>`).join('')}</details><p data-target-error role="status"></p>`;
     const stopControls=`<label class="recovery-option"><input type="checkbox" name="wounded" ${preferences.requireNoWounded?'checked':''}> 结束时要求所有患者无 wounded</label><label class="recovery-option"><input type="checkbox" name="failureStop" ${preferences.failureStop.enabled?'checked':''}> 连续治疗失败时停止该患者的新恢复</label><label class="recovery-field"><span>连续失败次数</span><input name="failureLimit" type="number" min="1" max="100" value="${preferences.failureStop.limit}"></label><p>失败次数按患者计算，更换治疗者不会重置；成功后清零。其他患者和独立再聚能可以继续。</p>`;
     const paused=data?.session?.status==='paused',summary=computed?data.session.computedRecovery?.summary:null;let result='选择成员和策略，计算恢复并推进时间。';
     if(summary)result=`已确认经过 ${minutes(summary.elapsedSeconds)} 分钟；${evidenceText(data.session.stopReason??summary.stopReason??data.session.status)}`;
     else if(data){const remaining=data.session.goalsByPool.reduce((n,g)=>n+Math.max(0,g.targetHP-(data.actors.find(a=>a.pool.poolUUID===g.poolUUID)?.hp.value??0)),0);result=formatSessionResult({...sessionTimeSummary(data),remainingHP:remaining,status:data.session.stopReason??data.session.status})}
     if(saved.reason)result+=` 保存的恢复偏好无效，已使用最大 HP：${saved.reason}`;
     const selectedStrategy=(computed&&(active||paused)?data.session.recoveryStrategy:policy.recoveryStrategy)==='time-first'?'time-first':'success-first';
     const strategyChoice=`<fieldset class="recovery-policy" ${active||manual||!ready?'disabled':''}><legend>恢复策略</legend><label class="recovery-field"><span>选择策略</span><select name="recoveryStrategy">${['success-first','time-first'].map((value,i)=>`<option value="${value}" ${selectedStrategy===value?'selected':''}>${['成功优先','时间优先'][i]}</option>`).join('')}</select></label><p>成功优先自动选择合法且足够的 Assurance 或聚能治疗；时间优先比较治疗、冷却、再聚能和并行活动的预计耗时。实际检定与治疗量在后台随机计算。</p></fieldset>`;
     const strategy=`<fieldset class="recovery-policy" ${active||manual||!ready?'disabled':''}><legend>治疗与停止条件</legend>
      <div class="recovery-policy-grid"><section class="recovery-policy-group"><h3>治疗与聚能</h3><label class="recovery-option"><input type="checkbox" name="risky" ${(data?.session?.riskySurgery??policy.riskySurgery)?'checked':''}> 激进治疗：先受到真实 1d8 伤害</label><label class="recovery-option"><input type="checkbox" name="focus" ${(data?.session?.requireFullFocus??policy.requireFullFocus)?'checked':''}> 结束时补满聚能</label><label class="recovery-option"><input type="checkbox" name="extension" ${(data?.session?.extendTreatment??policy.extendTreatment)?'checked':''}> 成功后允许延长至一小时</label><label class="recovery-field recovery-field-wide"><span>治疗 DC</span><select name="rank">${['auto','trained','expert','master','legendary'].map((rank,i)=>`<option value="${rank}" ${(data?.session?.treatmentRank??policy.treatmentRank??'auto')===rank?'selected':''}>${['自动',15,20,30,40][i]}（${['按策略选择','受训','专家','大师','传奇'][i]}）</option>`).join('')}</select></label><p>符合资格的聚能治疗与再聚能可循环使用。每日次数、法术位和消耗品保持关闭。无法核验的规则会停止计算并说明原因。</p></section>
      <section class="recovery-policy-group"><h3>停止条件与预算</h3>${stopControls}<label class="recovery-field"><span>时间预算（分钟）</span><input name="budget" type="number" min="1" value="${(data?.session?.budgetEndsAt&&data?.session?.startedAt!==undefined?(data.session.budgetEndsAt-data.session.startedAt):policy.budgetSeconds??7200)/60}"></label><label class="recovery-field"><span>最多行动次数</span><input name="activityBudget" type="number" min="1" max="100" value="${data?.session?.maxActivities??policy.maxActivities??100}"></label></section></div>
      </fieldset>`;
     const canStart=ready&&!active&&!manual&&!paused&&!data?.canAcceptCurrentTime,canResume=ready&&canResumeComputed(data,game);
     const mainActions=[
      canStart?'<button type="button" class="recovery-primary" data-recovery="start">计算恢复并推进时间</button>':'',
      (active||manual)&&ready?'<button type="button" data-recovery="stop">停止</button>':'',
      paused&&ready?'<button type="button" data-recovery="reconcile">核对已保存回执</button>':'',
      canResume?'<button type="button" data-recovery="resume">继续已确认会话</button>':'',
      ready&&data?.canAcceptCurrentTime?'<button type="button" data-recovery="acceptCurrentTime">按当前状态继续</button>':'',
      paused&&ready?'<button type="button" data-recovery="review">核对后结束会话</button>':''
     ].join('');
     const moreActions=[
      unconfirmed.size?'<button type="button" data-recovery="lookupActivity">查回上次登记</button>':'',
      active&&ready&&!data.session.manual?'<button type="button" data-recovery="takeover">接管并停止</button>':'',
      ledgerStatus&&ledgerStatus.state!=='blocked'?'<button type="button" data-recovery="selectLedger">选择已有账本</button>':'',
      storage&&ledgerStatus?.state==='ready'?`<button type="button" data-recovery="replaceLedger"${active||manual?' disabled':''}>换新账本</button><span>当前账本 ${e(ledgerStatus.revision+1)} 页</span>`:'',
      '<button type="button" data-recovery="refresh">刷新</button>'
     ].join('');
     return `<div class="recovery-content">${ledgerSetupControls(ledgerStatus)}<p class="recovery-time" role="status">${e(result)}</p>${treatmentFailureSummary(data)}
      ${summary?renderComputedSummary(data):`<ul class="recovery-member-summary">${actors.map(a=>`<li><span>${e(a.name)}</span><span>${a.hp.value} / ${a.hp.max} HP</span></li>`).join('')}</ul>`}
      ${computed?renderComputedEvidence(data):''}
      ${computed?renderComputedReadback(this.computedReadback,actors):''}
      ${data&&!computed?'<p class="recovery-legacy">旧版恢复会话仅供查看、核对、停止和关闭；请核对后关闭，再开始新的计算恢复。旧骰点、HP 应用和时间不会重放。</p>':''}
      ${data?.session?.recoveryReview?'<p role="status">已保留旧记录并确认当前状态。点击“计算恢复并推进时间”开始新的恢复。</p>':''}
      ${strategyChoice}
      <div class="recovery-actions recovery-main-actions">${mainActions}</div>
      <details class="recovery-advanced"><summary>恢复目标与高级选项</summary><table class="recovery-members"><thead><tr><th>成员 / 共用血池</th><th>当前 HP</th><th>下次启动目标</th></tr></thead><tbody>${rows}</tbody></table>${targetPreview}${strategy}</details>
      <details class="recovery-more"><summary>更多操作</summary><div class="recovery-actions">${moreActions}</div><details><summary>能力与限制</summary>${actors.map(a=>`<p>${e(a.name)}：医疗 ${a.medicine?.rank??0}；群体治疗 ${a.wardCapacity} 人；聚能 ${a.focus.value}/${a.focus.max}${a.pool.ready?'':`；${e(a.pool.reason)}`}${a.refocusUnsupported?.length?`；特殊再聚能规则需手动处理 ${e(a.refocusUnsupported.join(', '))}`:''}${a.unsupported?.length?`；需手动核对 ${e(a.unsupported.join(', '))}`:''}</p>`).join('')}</details></details>
      ${computed?'':renderRecoveryHistory(data,{actors,messages:game.messages})}</div>`;
    }
    updateTargetPreview(content){
     const error=content.querySelector('[data-target-error]');
     try{
      const saved=savedRecoveryPolicy(game.user.getFlag?.('pf2e-third-party-automation','explorationPolicy')??{}),{preferences}=targetControls(content,this.actorUUIDs,selectedRecoveryPreferences(saved.preferences,this.actorUUIDs)),resolved=resolveRecoveryGoals(this.targetActors,preferences);
      for(const goal of resolved.goalsByPool){const output=content.querySelector(`[data-pool="${goal.poolUUID}"]`);if(output){output.value=String(goal.targetHP);output.textContent=String(goal.targetHP)}}
      for(const target of resolved.recoveryGoals.patientTargets){const explanation=content.querySelector(`[data-target-explanation][data-actor="${target.patientUUID}"]`);if(explanation)explanation.textContent=targetExplanation(target)}
      if(error)error.textContent='';
     }catch{if(error)error.textContent='请核对目标和共用血池：百分比为0–100，绝对 HP 为非负整数，连续失败次数为1–100。'}
    }
    _replaceHTML(result,content){content.innerHTML=result;this.listener?.abort();this.listener=new AbortController();content.addEventListener('click',event=>{const source=event.target.closest('[data-recovery-message]');if(source){event.preventDefault();void showRecoveryMessage(source.dataset.recoveryMessage,{game}).then(shown=>{if(!shown)globalThis.ui?.notifications?.warn?.('来源消息已删除、不可见或未能载入，请核对保存的来源 ID。')}).catch(reportError);return}const action=event.target.closest('[data-recovery]')?.dataset.recovery;if(!action)return;void this.act(action,content).catch(reportError)},{signal:this.listener.signal});
     const update=event=>{
      if(!event.target.closest('[data-target-mode],[data-target-value],[name=wounded],[name=failureStop],[name=failureLimit]'))return;
      if(Object.hasOwn(event.target.dataset,'targetMode')){const mode=event.target.value,input=content.querySelector(`[data-target-value][data-actor="${event.target.dataset.actor}"]`);input.hidden=mode==='max';input.disabled=mode==='max';input.step=mode==='percent'?'any':'1';if(mode==='percent')input.max='100';else input.removeAttribute('max');if(mode!=='max'&&!String(input.value).trim())input.value=mode==='percent'?'100':String(this.targetActors.find(a=>a.actorUUID===event.target.dataset.actor).hp.max)}
      this.updateTargetPreview(content);
     };for(const type of ['input','change'])content.addEventListener(type,update,{signal:this.listener.signal});
    }
    async act(action,content){if(action==='acceptCurrentTime'){
      const id=getSessionId(),selected=await coordinator.snapshot(id);if(!selected.canAcceptCurrentTime)throw Error('当前状态暂不能接续，请刷新后核对未确认的治疗和时间。');
      const expectedWorldTime=selected.worldTime;
      const confirmed=await foundry.applications.api.DialogV2.wait({window:{title:'按当前状态继续'},content:'<p>请确认当前 HP、资源和游戏内时间已核对。旧的附带效果仍未核实；旧骰点、HP 应用和时间推进均不会重放。</p><p>确认后保留旧记录并结束旧会话，再点击“计算恢复并推进时间”开始新的恢复。</p>',buttons:[{action:'accept',label:'确认当前状态',callback:()=>true},{action:'cancel',label:'取消',callback:()=>false}],rejectClose:false});
      if(confirmed!==true)return;
      await coordinator.acceptCurrentTime(id,{expectedWorldTime,note:'GM 已核对当前 HP、资源和世界时间；旧附带效果未核实，保留旧记录，不重放骰点、HP 应用或时间推进。'});return this.render(true);
     }if(action==='refresh')return this.render(true);if(['provision','initialize','selectLedger'].includes(action)){
      if(!storage)throw Error('exploration-storage-unavailable');const options=await setupChoices({game,select:action==='selectLedger'});if(!options)return;
      const result=await storage[action==='selectLedger'?'select':action](options);
      if(result.requiresReload)globalThis.ui?.notifications?.info?.('账本已选择，请重载页面后再准备恢复。');else if(action==='provision')await storage.initialize(options);
      return this.render(true);
     }if(action==='replaceLedger'){
      if(!storage)throw Error('exploration-storage-unavailable');const options=await replaceChoices(await storage.replacePreview());if(!options)return;
      await storage.replace(options);globalThis.ui?.notifications?.info?.('账本已更换，请重载页面后再准备恢复。');return this.render(true);
     }if(['openActivityCheckpoint','closeActivityCheckpoint','activity'].includes(action))throw Error('恢复会话不再开放行动登记；旧登记仅可查看和核对。');
     if(action==='lookupActivity'){for(const declaration of unconfirmed.values())await lookupDeclaration(declaration);return this.render(true)}if(action==='takeover'){await coordinator.takeover(getSessionId());return this.render(true)}
     if(action==='reconcile'){
      const id=getSessionId(),result=await coordinator.reconcile(id);
      this.computedReadback=isComputed(result)&&getSessionId()===id?{sessionId:id,results:structuredClone(result.computedReadbacks??[])}:null;
      return this.render(true);
     }if(action==='resume'){
      if(game.users&&game.users.activeGM?.id!==game.user.id)throw Error('active-gm-required');
      const id=getSessionId(),selected=await coordinator.snapshot(id);
      if(!isComputed(selected))throw Error(evidenceLabels['legacy-recovery-view-only']);
      if(!canResumeComputed(selected,game))throw Error('当前计算会话不能继续，请核对未确认结果和世界时间。');
      await coordinator.resume(id);return this.render(true);
     }if(action==='review'){
      const id=getSessionId(),selected=await coordinator.snapshot(id),explanation=isComputed(selected)?'未知记录仍会保留，旧尝试不会重放；关闭后可开始新的计算恢复。':'结束会话会保留未知记录；旧尝试不会重放。尚未确认的执行仍会占用相应时间与治疗目标，核对说明不会解除这些限制。';
      const note=await foundry.applications.api.DialogV2.wait({window:{title:'核对后结束恢复会话'},content:`<p>请先核对实际 HP、资源、已保存结果和世界时间。${explanation}</p><label>核对说明 <textarea name="note" required></textarea></label>`,buttons:[{action:'close',label:'保存核对说明并结束',callback:(_e,b)=>new FormData(b.form).get('note')},{action:'cancel',label:'取消',callback:()=>null}],rejectClose:false});
      if(note===null)return;await coordinator.review(id,{note,userId:game.user.id});return this.render(true);
     }if(action==='stop'){await coordinator.stop(getSessionId());return this.render(true)}
     if(action!=='start')throw Error('未知的恢复操作。');
     const actorUUIDs=[...this.actorUUIDs],saved=savedRecoveryPolicy(game.user.getFlag?.('pf2e-third-party-automation','explorationPolicy')??{}),controls=targetControls(content,actorUUIDs,selectedRecoveryPreferences(saved.preferences,actorUUIDs)),recovery=controls.preferences;
     const recoveryStrategy=content.querySelector('[name=recoveryStrategy]')?.value??'success-first';
     if(!['success-first','time-first'].includes(recoveryStrategy))throw Error('请选择有效的恢复策略。');
     const options={recoveryStrategy,...recoveryBudgetPolicy({minutes:Number(content.querySelector('[name=budget]').value),activities:Number(content.querySelector('[name=activityBudget]').value)}),riskySurgery:content.querySelector('[name=risky]').checked,requireFullFocus:content.querySelector('[name=focus]').checked,extendTreatment:content.querySelector('[name=extension]').checked,treatmentRank:content.querySelector('[name=rank]').value};
     const targetPreview=this.targetPreview?structuredClone(this.targetPreview):null;
     const submitted=controls.patientControls?[]:[...content.querySelectorAll('[data-pool]')].map(el=>{const targetHP=Number(el.value);if(!String(el.value).trim()||!Number.isSafeInteger(targetHP)||targetHP<0)throw Error('invalid-recovery-goals');return {poolUUID:el.dataset.pool,targetHP}});
     const actors=await capabilities.snapshot(actorUUIDs),resolved=resolveRecoveryGoals(actors,recovery),sameSelection=targetPreview&&canonicalJSON(targetPreview.actorUUIDs)===canonicalJSON(actorUUIDs),preview=sameSelection?targetPreview.goalsByPool:resolved.goalsByPool;
     if(!controls.patientControls&&(submitted.length!==resolved.goalsByPool.length||new Set(submitted.map(g=>g.poolUUID)).size!==submitted.length||submitted.some(g=>!resolved.goalsByPool.some(p=>p.poolUUID===g.poolUUID))))throw Error('invalid-recovery-goals');
     for(const goal of submitted){if(goal.targetHP===preview.find(p=>p.poolUUID===goal.poolUUID)?.targetHP)continue;for(const actor of actors.filter(a=>a.pool.poolUUID===goal.poolUUID))recovery.targetIntentsByActor[actor.actorUUID]={mode:'absolute',value:goal.targetHP}}
     const {goalsByPool}=resolveRecoveryGoals(actors,recovery),config={...recoveryDefaults(actors),...options,goalsByPool,recovery,actorUUIDs};
     const latest=savedRecoveryPolicy(game.user.getFlag?.('pf2e-third-party-automation','explorationPolicy')??{}).preferences;
     await game.user.setFlag?.('pf2e-third-party-automation','explorationPolicy',{...options,recovery:encodeRecoveryPreferences({...latest,...recovery,targetIntentsByActor:{...latest.targetIntentsByActor,...recovery.targetIntentsByActor}})});await start(config);return this.render(true);
    }
    async close(options){this.listener?.abort();return super.close(options)}
   }panel=new RecoveryPanel();
  }
  panel.actorUUIDs=actorUUIDs??Array.from(game.actors.party?.members??[]).filter(a=>a.type==='character').map(a=>a.uuid);return panel.render(true);
 }
 // Background updates respect a closed or minimized window; open() is the explicit entry.
 return {open,openActivityDeclaration,refresh:()=>panel?.rendered?panel.render({force:false}):undefined};
}
