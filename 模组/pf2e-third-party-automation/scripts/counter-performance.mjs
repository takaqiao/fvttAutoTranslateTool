import {MODULE_ID as ID} from './rules.mjs';
import {getSourceId,isActiveGM,publicTargetName,showNativeChoice} from './native-context.mjs';
import {SerialActions} from './runtime.mjs';
import {BOB2_AUTOMATION_WORLDS} from './world-gates.mjs';

export const COUNTER_PERFORMANCE_SOURCE='Compendium.pf2e.spells-srd.Item.WILXkjU5Yq3yw10r';
const TOOL='pf2e-toolbelt',KEY='counterPerformance';
const outcomes=['criticalFailure','failure','success','criticalSuccess'];
const labels=['大失败','失败','成功','大成功'];
const performances={
 acting:{label:'表演 Acting（听觉、语言、视觉）',traits:['auditory','linguistic','visual']},
 comedy:{label:'喜剧 Comedy（听觉、语言、视觉）',traits:['auditory','linguistic','visual']},
 dance:{label:'舞蹈 Dance（移动、视觉）',traits:['move','visual']},
 keyboards:{label:'键盘乐器 Keyboards（听觉、操控）',traits:['auditory','manipulate']},
 oratory:{label:'演说 Oratory（听觉、语言）',traits:['auditory','linguistic']},
 percussion:{label:'打击乐器 Percussion（听觉、操控）',traits:['auditory','manipulate']},
 singing:{label:'歌唱 Singing（听觉、语言）',traits:['auditory','linguistic']},
 strings:{label:'弦乐器 Strings（听觉、操控）',traits:['auditory','manipulate']},
 winds:{label:'管乐器 Winds（听觉、操控）',traits:['auditory','manipulate']},
};
const copy=value=>structuredClone(value),same=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
const values=collection=>Array.from(collection?.values?.()??collection??[]);
const helper=message=>message?.flags?.[TOOL]?.targetHelper;
const author=message=>message?.author?.id??message?.user?.id??message?.user;
const publicMessage=message=>message?.blind===false&&Array.isArray(message.whisper)&&message.whisper.length===0;
const sourceMatches=item=>item?.type==='spell'&&getSourceId(item)===COUNTER_PERFORMANCE_SOURCE;
const demand=(condition,text)=>{if(!condition)throw Error(`反制表演：${text}`);};
const escape=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const link=(uuid,label)=>`<a data-counter-performance-message="${escape(uuid.split('.').at(-1))}">${escape(label)}</a>`;
const saveLabels={fortitude:'强韧',reflex:'反射',will:'意志'};

// The trigger must be auditory or visual, and the performance has to match it.
function performancesFor(traits){
 const auditory=traits.has('auditory'),visual=traits.has('visual');
 const need=auditory&&!visual?'auditory':visual&&!auditory?'visual':null;
 const hint=auditory&&visual?'触发效果既是听觉也是视觉的，两类表演都可以。':
  auditory?'触发效果是听觉的，下面只列听觉表演。':visual?'触发效果是视觉的，下面只列视觉表演。':
  '这张卡没有记下听觉或视觉特征。反制表演只对听觉或视觉效果有用，先确认触发效果属于哪一种，再选对应的表演。';
 return {hint,choices:Object.entries(performances).filter(([,data])=>!need||data.traits.includes(need)).map(([value,data])=>({value,label:data.label}))};
}

/** One form replaces the per-target and confirmation dialogs. */
export function showCounterPerformanceSetup({trigger,hint,performances:choices,performance,targets}){
 const options=choices.map(({value,label})=>`<option value="${escape(value)}"${value===performance?' selected':''}>${escape(label)}</option>`).join('');
 const rows=targets.map(row=>`<label style="display:block;margin:2px 0"><input type="checkbox" name="target" value="${escape(row.uuid)}"${row.checked?' checked':''}> ${escape(row.label)}</label>`).join('');
 const content=`<p>触发：<strong>${escape(trigger)}</strong></p>
<p>检定都是表演（Performance）。这里选的是你用哪种表演，它决定这次表演算听觉还是视觉。${escape(hint)}</p>
<div class="form-group"><label>表演方式</label><select name="performance">${options}</select></div>
<fieldset><legend>谁用表演结果</legend><p>你自己和 60 尺内的盟友；每人取原豁免和表演里较好的一个，不会变差。</p>${rows}</fieldset>`;
 return globalThis.foundry.applications.api.DialogV2.wait({
  classes:['tpa-native-choice','tpa-counter-performance-setup'],position:{width:520},window:{title:'反制表演'},content,
  buttons:[
   {action:'cast',label:'使用反应',default:true,callback:(_event,button)=>{const data=new FormData(button.form);return {performance:data.get('performance'),targets:data.getAll('target')};}},
   {action:'cancel',label:'取消',callback:()=>null},
  ],rejectClose:false,
 });
}

function readSave(row){
 if(!row||row.private===true||row.rerolled||!Number.isFinite(row.value)||!Number.isInteger(row.die)||row.die<1||row.die>20||!outcomes.includes(row.success))return null;
 try{
  const roll=JSON.parse(row.roll);
  if(roll.class!=='CheckRoll'||roll.options?.type!=='saving-throw'||roll.total!==row.value||outcomes[roll.options.degreeOfSuccess]!==row.success)return null;
  return roll;
 }catch{return null;}
}

// Reuse the original save's already-filtered degree adjustments. No statistic
// is rolled again, and neither the original roll JSON nor its die is changed.
export function counterPerformanceOutcome(total,die,dc,adjustments={}){
 let degree=total>=dc+10?3:total>=dc?2:total<=dc-10?0:1;
 const clamp=n=>Math.min(3,Math.max(0,n));
 degree=clamp(degree+(die===20?1:die===1?-1:0));
 const unadjusted=outcomes[degree];
 for(const key of ['all',...outcomes]){
  const adjustment=adjustments[key];
  if(!adjustment?.label||!adjustment.amount||key!=='all'&&key!==unadjusted||degree===3&&adjustment.amount===1||degree===0&&adjustment.amount===-1)continue;
  degree=typeof adjustment.amount==='number'?clamp(degree+adjustment.amount):outcomes.indexOf(adjustment.amount);
  demand(degree>=0,'原豁免成功度修正无效。');break;
 }
 return {success:outcomes[degree],unadjustedOutcome:unadjusted};
}

/** The Cast chooser is local to the casting owner. The normal usage dispatcher
 * then runs one authenticated Performance check on that owner's client. */
export function createCounterPerformance({game,fromUuid=globalThis.fromUuid,nativeCasts,runNative,chooseCast=showNativeChoice,chooseSetup=showCounterPerformanceSetup}={}){
 const scopes=new Map(),queue=new SerialActions(),lastPerformance=new Map();
 const enabled=()=>BOB2_AUTOMATION_WORLDS.has(game.world?.id)&&game.system?.id==='pf2e';
 const matches=item=>enabled()&&sourceMatches(item);
 const compatible=()=>game.system?.id==='pf2e'&&game.modules?.get(TOOL)?.active===true;
 const assertGM=ctx=>demand(isActiveGM(game)&&game.messages.get(ctx.message.id)===ctx.message&&author(ctx.message)===ctx.user.id&&ctx.actor.testUserPermission(ctx.user,'OWNER')&&matches(ctx.item)&&ctx.actor.items.get(ctx.item.id)===ctx.item,'施法来源、所有者或主GM已改变。');
 function candidates(){
  const result=[];
  for(const message of values(game.messages).slice(-100).reverse()){
   const h=helper(message);if(!publicMessage(message)||h?.private===true||!Array.isArray(h?.targets))continue;
   for(const [variantId,variant]of Object.entries(h.saveVariants??{})){
    if(!/^[A-Za-z0-9-]+$/.test(variantId)||!Number.isFinite(variant.dc)||!['fortitude','reflex','will'].includes(variant.statistic))continue;
    const used=Object.values(message.flags?.[ID]?.[KEY]??{}).flatMap(record=>record.results??[]).map(row=>row.targetUuid);
    const targets=h.targets.filter(uuid=>typeof uuid==='string'&&/^Scene\.[^.]+\.Token\.[^.]+$/.test(uuid)&&!used.includes(uuid)&&readSave(variant.saves?.[uuid.split('.').at(-1)]));
    if(targets.length)result.push({message,variantId,variant,targets});
   }
  }
  return result.slice(0,40);
 }
 async function inspect(input){
  demand(compatible(),'需要启用 Toolbelt 的原豁免结果卡。');
  const source=game.messages.get(input?.sourceMessageId),h=helper(source),variant=h?.saveVariants?.[input?.variantId];
  demand(publicMessage(source)&&h?.private!==true&&variant&&same({dc:variant.dc,statistic:variant.statistic,basic:variant.basic??false,item:h.item??null},input.source),'原豁免卡或DC已改变。');
  demand(Array.isArray(input.targets)&&input.targets.length>0&&input.targets.length<=100&&new Set(input.targets.map(row=>row.targetUuid)).size===input.targets.length,'没有明确的原豁免目标。');
  demand(!Object.values(source.flags?.[ID]?.[KEY]??{}).some(record=>(record.results??[]).some(row=>input.targets.some(target=>target.targetUuid===row.targetUuid))),'该原豁免已经使用过反制表演。');
  for(const saved of input.targets){
   const token=await fromUuid(saved.targetUuid),row=variant.saves?.[saved.targetUuid.split('.').at(-1)];
   demand(token?.actor?.uuid===saved.actorUuid&&h.targets?.includes(saved.targetUuid)&&same(row,saved.original)&&readSave(row),'原豁免目标或骰值已改变，请按原卡手工核对。');
  }
  return {source,h,variant};
 }
 const lookup=async uuid=>{try{return typeof uuid==='string'?await fromUuid(uuid):null;}catch{return null;}};
 // Name the save card the way the chat log shows it; never fall back to an ID.
 async function describe(candidate,caster){
  const {message,variantId,variant}=candidate,h=helper(message);
  const item=await lookup(h.item)??message.item??null;
  const traits=new Set([...(Array.isArray(h.traits)?h.traits:[]),...(Array.isArray(item?.system?.traits?.value)?item.system.traits.value:[])]);
  const tokens=[];
  for(const uuid of candidate.targets){const token=await lookup(uuid);if(token?.actor)tokens.push(token);}
  const speakerToken=message.speaker?.token?message.token??null:null;
  const speaker=message.speaker?.token?(speakerToken?publicTargetName(speakerToken,{game}):null):message.alias;
  const names=tokens.map(token=>publicTargetName(token,{game}));
  const who=names.length>4?`${names.slice(0,4).join('、')} 等 ${names.length} 人`:names.join('、');
  const sense=[traits.has('auditory')&&'听觉',traits.has('visual')&&'视觉'].filter(Boolean).join('、');
  const label=`${[item?.name??'未命名效果',speaker&&speaker!=='目标'?speaker:null,`${saveLabels[variant.statistic]}豁免`,sense].filter(Boolean).join(' · ')}（${who}）`.slice(0,300);
  const rows=tokens.map(token=>{
   const actor=token.actor,ally=actor===caster||actor.uuid===caster.uuid||(typeof caster.isAllyOf==='function'?caster.isAllyOf(actor)===true:actor.hasPlayerOwner===true);
   const save=variant.saves[token.id],name=publicTargetName(token,{game});
   return {uuid:token.uuid,id:token.id,actorUuid:actor.uuid,name:token.name??actor.name,checked:ally,
    label:ally?`${name}（原豁免 ${save.value}，${labels[outcomes.indexOf(save.success)]}）`:`${name}（不是盟友）`};
  }).sort((a,b)=>b.checked-a.checked);
  return {key:`${message.id}:${variantId}`,candidate,label,traits,rows};
 }
 async function interceptCast({item,options={}},native){
  if(!matches(item)||options.consume===false||options.message===false)return native();
  demand(compatible(),'需要启用 Toolbelt 的原豁免结果卡。');
  demand(item.actor?.testUserPermission(game.user,'OWNER')&&game.users.activeGM?.id,'需要原施法者与在线主GM。');
  demand(!scopes.has(item.uuid),'本次施法选择尚未完成。');
  const available=candidates();demand(available.length,'聊天栏里没有能用的公开豁免。等触发效果的豁免掷完，再施放反制表演。');
  const caster=item.actor,described=[];
  for(const candidate of available){const entry=await describe(candidate,caster);if(entry.rows.length)described.push(entry);}
  demand(described.length,'聊天栏里没有能用的公开豁免。等触发效果的豁免掷完，再施放反制表演。');
  let picked=described[0];
  if(described.length>1){
   const selected=await chooseCast({title:'反制表演：帮哪一次豁免？',prompt:'聊天栏里最近的公开豁免，最新的在最上面。',choices:[...described.map(({key,label})=>({value:key,label})),{value:'cancel',label:'取消'}]});
   picked=described.find(entry=>entry.key===selected);if(!picked)return;
  }
  const {hint,choices}=performancesFor(picked.traits),remembered=lastPerformance.get(caster.uuid);
  const setup=await chooseSetup({trigger:picked.label,hint,performances:choices,performance:choices.some(c=>c.value===remembered)?remembered:choices[0]?.value,targets:picked.rows.map(({uuid,label,checked})=>({uuid,label,checked}))});
  if(!setup)return;
  const performance=setup.performance;
  demand(choices.some(choice=>choice.value===performance),'所选表演方式与触发效果不符。');
  const chosen=new Set(Array.isArray(setup.targets)?setup.targets:[]);
  demand(chosen.size,'没有勾选任何人，本次没有施法。');
  const selectedRows=picked.rows.filter(row=>chosen.has(row.uuid));
  demand(selectedRows.length===chosen.size,'勾选的角色不在这张豁免卡上。');
  const candidate=picked.candidate;
  const targets=selectedRows.map(row=>({targetUuid:row.uuid,actorUuid:row.actorUuid,name:row.name,original:copy(candidate.variant.saves[row.id])}));
  const h=helper(candidate.message),input={performance,sourceMessageId:candidate.message.id,variantId:candidate.variantId,source:{dc:candidate.variant.dc,statistic:candidate.variant.statistic,basic:candidate.variant.basic??false,item:h.item??null},targets};
  await inspect(input);scopes.set(item.uuid,input);lastPerformance.set(caster.uuid,performance);
  try{return await native();}finally{scopes.delete(item.uuid);}
 }
 const captureCast=item=>matches(item)&&scopes.has(item.uuid)?copy(scopes.get(item.uuid)):undefined;
 async function executeUsage(ctx){
  const {actor,item,message}=ctx,input=message.flags?.[ID]?.counterPerformanceInput;
  if(!matches(item)||!message.flags?.[ID]?.nativeCast||!input)return {status:'cancelled',result:'展示卡未绑定实际反制表演施法。'};
  return queue.run(input.sourceMessageId,async()=>{
   assertGM(ctx);const prior=message.flags?.[ID]?.[KEY];
   if(prior)return {status:prior.status==='cancelled'?'cancelled':prior.status==='done'?'done':'waiting',result:prior.result??'本次已开始处理；请核对原卡，不能重复投骰。'};
   demand(performances[input.performance],'本次表演方式无效。');
   await inspect(input);assertGM(ctx);
   await nativeCasts.ensurePaid(ctx);assertGM(ctx);
   await message.update({[`flags.${ID}.${KEY}`]:{status:'rolling',sourceMessageId:input.sourceMessageId}});
   try{
    const performance=performances[input.performance];
    const rolled=await runNative(ctx,{type:'check',statistic:'performance',checkKind:'skill-check',itemUuid:item.uuid,sourceUuid:COUNTER_PERFORMANCE_SOURCE,action:'counter-performance',label:'反制表演',traits:[...new Set([...(item.system.traits?.value??[]),...performance.traits])],options:['action:counter-performance',`action:perform:${input.performance}`],messageMode:'public'});
    assertGM(ctx);
    if(rolled?.status==='cancelled'){
     const result='已施放反制表演并支付聚能；表演检定已取消，原豁免未修改。请手工完成本次表演，不要重新 Cast。';
     await message.update({[`flags.${ID}.${KEY}`]:{status:'cancelled',sourceMessageId:input.sourceMessageId,result}});return {status:'cancelled',result};
    }
    const check=rolled?.check,roll=check?.rolls?.[0];
    demand(rolled?.status==='rolled'&&game.messages.get(check?.id)===check&&publicMessage(check)&&check.speaker?.actor===actor.id&&author(check)===ctx.user.id&&roll?._evaluated===true&&Number.isFinite(roll.total),'原生表演结果未确认或并非公开结果；请手工核对，不会重新投骰。');
    const performanceDie=roll.dice?.find(die=>die.faces===20)?.total;
    demand(Number.isInteger(performanceDie)&&performanceDie>=1&&performanceDie<=20,'表演骰缺少原始d20，请手工核对。');
    const {source,h}=await inspect(input);assertGM(ctx);
    const changes={},results=[];
    for(const saved of input.targets){
     const targetId=saved.targetUuid.split('.').at(-1),original=saved.original;
     const candidate=counterPerformanceOutcome(roll.total,performanceDie,input.source.dc,original.dosAdjustments);
     const degree=outcomes.indexOf(candidate.success),originalDegree=outcomes.indexOf(original.success);
     const better=degree>originalDegree||degree===originalDegree&&roll.total>original.value;
     const outcome=better?candidate:{success:original.success,unadjustedOutcome:original.unadjustedOutcome};
     if(better){
      const path=`flags.${TOOL}.targetHelper.saveVariants.${input.variantId}.saves.${targetId}`;
      changes[`${path}.value`]=roll.total;changes[`${path}.success`]=outcome.success;changes[`${path}.unadjustedOutcome`]=outcome.unadjustedOutcome;
     }
     results.push({...copy(saved),effectiveTotal:better?roll.total:original.value,success:outcome.success,usedPerformance:better,damageAlreadyApplied:Object.values(h.applied?.[targetId]??{}).some(Boolean)});
    }
    const record={castMessageUuid:message.uuid,performanceMessageUuid:check.uuid,sourceMessageUuid:source.uuid,variantId:input.variantId,dc:input.source.dc,performance:input.performance,performanceRoll:copy(roll.toJSON()),results};
    changes[`flags.${ID}.${KEY}.${message.id}`]=record;
    // One document update stores the effective rows and their original dice.
    // Toolbelt's existing apply/Undo controls remain the settlement owner.
    await source.update(changes);assertGM(ctx);
    const result='已更新所选原豁免结果。伤害使用原卡按钮；已应用伤害先 Undo 再应用一次。已有 BUFF 按新成功度人工核对。';
    await message.update({[`flags.${ID}.${KEY}`]:{status:'done',...record,result}});
    return {status:'done',result};
   }catch(error){await message.update({[`flags.${ID}.${KEY}`]:{...message.flags[ID][KEY],status:'uncertain',result:`${error.message} 不会自动重投或再次应用后果。`}});throw error;}
  });
 }
 function resultHTML(message){
  const data=message.flags?.[ID]?.[KEY];if(!data)return '';
  const records=data.status==='done'?[data]:data.status?[]:Object.values(data);
  const rows=records.map(record=>`<p>${link(record.sourceMessageUuid,'原豁免卡')} · ${link(record.performanceMessageUuid,'表演骰')} · ${link(record.castMessageUuid,'反制表演')}</p><ul>${record.results.map(row=>`<li>${escape(row.name)}：原总值 ${row.original.value} → ${row.effectiveTotal}（${labels[outcomes.indexOf(row.success)]}，采用${row.usedPerformance?'表演':'原豁免'}）${row.damageAlreadyApplied?'；已有伤害应用记录，先 Undo':''}</li>`).join('')}</ul>`).join('');
  return `<div class="tpa-counter-performance"><strong>反制表演</strong>${rows}<p>${escape(data.result??'原骰保留，结果栏已更新。已应用伤害先 Undo，再按新结果应用一次；已有 BUFF 按新成功度人工核对。')}</p></div>`;
 }
 function register({Hooks}={}){
  if(!enabled())return()=>{};
  const hook=Hooks.on('renderChatMessageHTML',(message,html)=>{
   const element=html?.[0]??html,markup=resultHTML(message);element?.querySelector?.('.tpa-counter-performance')?.remove();
   if(markup)(element?.querySelector?.('.message-content')??element)?.insertAdjacentHTML?.('beforeend',markup);
   for(const node of element?.querySelectorAll?.('[data-counter-performance-message]')??[])node.addEventListener('click',event=>{
    event.preventDefault();const original=game.messages.get(node.dataset.counterPerformanceMessage);
    if(original)new globalThis.CONFIG.ChatMessage.popoutClass({message:original}).render({force:true});
   });
  });return()=>Hooks.off('renderChatMessageHTML',hook);
 }
 return {matches,resolveAction:item=>matches(item)?'counter-performance':null,interceptCast,captureCast,executeUsage,resultHTML,register};
}
