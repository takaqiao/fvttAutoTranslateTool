import {MODULE_ID} from './rules.mjs';
import {hasSkillAssurance} from './knowledge-automatic.mjs';
import {withKnowledgeProbe,consumeKnowledgePrimary} from './knowledge-probes.mjs';
import {createWorkbenchDisplay,knowledgeNativeLabel,knowledgeProficiencyLabel} from './knowledge-display.mjs';
export const WORKBENCH_RECALL_UUID='Compendium.xdy-pf2e-workbench.asymonous-benefactor-macros-internal.Macro.xcFr7PWwG5OVALNJ';
const skills=['arcana','crafting','medicine','nature','occultism','religion','society'];
const identify={aberration:['occultism'],astral:['occultism'],animal:['nature'],beast:['arcana','nature'],celestial:['religion'],construct:['arcana','crafting'],dragon:['arcana'],elemental:['arcana','nature'],ethereal:['occultism'],fey:['nature'],fiend:['religion'],fungus:['nature'],giant:['society'],humanoid:['society'],monitor:['religion'],ooze:['occultism'],plant:['nature'],spirit:['occultism'],undead:['religion']};
const outcomes=['criticalFailure','failure','success','criticalSuccess'];
const escape=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const values=x=>Array.from(x?.values?.()??x??[]),doc=x=>x?.document??x,own=m=>m?.flags?.[MODULE_ID]?.workbenchRecall;
// A fresh facade avoids invariants on immutable native own properties. Bind all
// native getters/methods to their real receiver so private fields still work.
const scoped=(original,overrides)=>new Proxy(Object.create(Object.getPrototypeOf(original)),{
 get:(_target,key)=>{if(Object.hasOwn(overrides,key))return overrides[key];const value=Reflect.get(original,key,original);return typeof value==='function'?value.bind(original):value;},
 has:(_target,key)=>Object.hasOwn(overrides,key)||key in original,
 ownKeys:()=>[...new Set([...Reflect.ownKeys(original),...Reflect.ownKeys(overrides)])],
 getOwnPropertyDescriptor:(_target,key)=>key in original||Object.hasOwn(overrides,key)?{value:Object.hasOwn(overrides,key)?overrides[key]:Reflect.get(original,key,original),writable:false,enumerable:true,configurable:true}:undefined,
});
function targetDC(actor){const level=actor.level;if(!Number.isInteger(level))return null;const base=level>20?level*2:14+level+(level<0?0:Math.floor(level/3));const adjustment={common:0,uncommon:2,rare:5,unique:10}[actor.rarity??'common'];return Number.isFinite(adjustment)?base+adjustment:null;}
function naturalRollOptions(roll){if(!Array.isArray(roll?.dice))return null;const natural=roll.dice.map(die=>die.results?.find(result=>result.active&&!result.discarded)?.result??null).find(value=>value!=null);return [`check:total:natural:${natural}`,`check:roll:total:natural:${natural}`];}
export function recallDegree({total,die,dc,domains=[],rollOptions=[],actor,roll,dosAdjustments,Predicate=globalThis.game?.pf2e?.Predicate}){
 if(!Number.isFinite(total)||!Number.isFinite(dc))return null;
 let degree=total-dc>=10?3:total>=dc?2:total-dc<=-10?0:1;
 if(die===20)degree=Math.min(3,degree+1);else if(die===1)degree=Math.max(0,degree-1);
 const nativeNatural=rollOptions.some(option=>option.startsWith('check:total:natural:')||option.startsWith('check:roll:total:natural:'))?[]:roll===undefined?[`check:total:natural:${die??undefined}`,`check:roll:total:natural:${die??undefined}`]:naturalRollOptions(roll);if(!nativeNatural)return null;
 const checkOptions=[...nativeNatural,...rollOptions.filter(option=>option.startsWith('check:total:natural:')||option.startsWith('check:roll:total:natural:')),`check:total:${total}`,`check:total:delta:${total-dc}`],options=new Set([...rollOptions,...checkOptions]),adjustments={};
 const entries=dosAdjustments??domains.flatMap(domain=>actor?.synthetics?.degreeOfSuccessAdjustments?.[domain]??[]);
 for(const entry of entries){const entryOptions=entry.options?new Set([...checkOptions,...entry.options]):options,predicate=entry.predicate;if(predicate&&typeof predicate.test!=='function'&&!Predicate?.test)return null;if(predicate&&!(typeof predicate.test==='function'?predicate.test(entryOptions):Predicate.test(predicate,entryOptions)))continue;for(const key of ['all',...outcomes])if(entry.adjustments?.[key])adjustments[key]=entry.adjustments[key];}
 for(const key of ['all',...outcomes]){const {amount,label}=adjustments[key]??{};if(!amount||!label||degree===3&&amount===1||degree===0&&amount===-1||key!=='all'&&key!==outcomes[degree])continue;const explicit=outcomes.indexOf(amount);return explicit>=0?explicit:Number.isFinite(amount)?Math.max(0,Math.min(3,degree+amount)):degree;}
 return degree;
}
function savedAdjustments(context){
 // StatisticCheck skips extraction without a DC. Preserve the contextual rules
 // before afterRoll consumes them so a later GM DC can still use that check.
 const entries=Number.isFinite(context.dc?.value)?context.dosAdjustments:[...new Set(context.domains??[])].flatMap(domain=>context.actor?.synthetics?.degreeOfSuccessAdjustments?.[domain]??[]);
 return entries?.map(entry=>({adjustments:structuredClone(entry.adjustments),...(entry.predicate?{predicate:entry.predicate.toObject?.()??(Array.isArray(entry.predicate)?structuredClone(entry.predicate):entry.predicate)}:{}),...(entry.options?{options:[...entry.options]}:{})}));
}
function primarySkills(actor,target){const relevant=new Set();for(const [trait,list]of Object.entries(identify))if(target?.traits?.has?.(trait)||values(target?.traits).includes(trait))for(const skill of list)relevant.add(skill);if(actor.itemTypes?.feat?.some(f=>(f.slug??f.system?.slug)==='unified-theory')&&['religion','occultism','nature'].some(s=>relevant.has(s)))relevant.add('arcana');return [...relevant];}
function recallDC(actor,target,statistic,dc){if(Number.isFinite(dc))return dc;if(!target)return null;if(statistic&&(actor.skills[statistic]?.lore||!primarySkills(actor,target).includes(statistic)))return null;return targetDC(target);}
function authorization(game,actor,user){if(game.user!==user||game.users.get(user.id)!==user||!actor?.testUserPermission?.(user,'OWNER'))throw Error('回忆知识需要原操作者及角色所有权。');}
/** Execute the installed macro unchanged. Scoped native probes supply structured results; no HTML parsing or global latest-message lookup. */
export async function captureWorkbenchRecall({game,actor,token,user=game.user,targetUuids=[],requestId,origin=null,statistic=null,assurance=false,dc=null,fromUuid=globalThis.fromUuid,globals=globalThis}){
 authorization(game,actor,user);if(!requestId||typeof requestId!=='string')throw Error('回忆知识缺少操作来源。');if(assurance&&(!statistic||actor.skills?.[statistic]?.rank<1||!hasSkillAssurance(actor,statistic)))throw Error('驾轻就熟必须具有对应专长并保留指定技能。');
 const tokenDocument=doc(token);if(tokenDocument?.actor!==actor)throw Error('回忆知识的原始 Token 与角色不匹配。');
  const targets=[];for(const uuid of [...new Set(targetUuids)]){const target=doc(await fromUuid(uuid));if(target?.documentName!=='Token'||!target.actor)throw Error('回忆知识目标已不存在。');targets.push(target);}
  const targetActors=targets.map(target=>({tokenUuid:target.uuid,actorUuid:target.actor.uuid}));
  const assertTargets=()=>{if(targets.some((target,index)=>target.actor?.uuid!==targetActors[index].actorUuid))throw Error('回忆知识目标关联角色已改变；原始秘骰保留，不会再次投骰。');};
 const previous=values(game.messages).find(message=>own(message)?.requestId===requestId&&own(message).actorUuid===actor.uuid&&own(message).userId===user.id);
 if(previous)throw Error('本次回忆知识已开始或已保存；不会重复投骰。');
 const probes=new Map(),receipts=new Map();let assuranceApplied=assurance,currentTarget=null,created=null,macroPublished=false,primary=null,primaryRoll=null,primaryNativeContext=null,rawRoll=null,fail;const failed=new Promise((_,reject)=>{fail=reject;});failed.catch(()=>{});
 let display={content:value=>value,notice:value=>value};
 const cancelUnrolledReservation=async()=>{if(created&&game.messages.get(created.id)===created&&own(created)?.requestId===requestId&&own(created)?.status==='rolling'&&!created.rolls?.length)await created.delete();};
 const preparedSkills={};
 for(const [slug,skill]of Object.entries(actor.skills??{}))preparedSkills[slug]=scoped(skill,{get label(){return display.enabled?knowledgeNativeLabel(game,slug,skill.label):skill.label;},roll:async options=>{
  authorization(game,actor,user);const targetUuid=currentTarget;
  const key=`${targetUuid??''}:${slug}`,cached=receipts.get(key);if(cached){await options.callback?.(cached.roll,undefined,cached.message);return cached.roll;}
  const probeTarget=targets.find(target=>target.uuid===targetUuid),probeDC=recallDC(actor,probeTarget?.actor,statistic,dc);
  try{const captured=await withKnowledgeProbe({actor,statistic:slug,globals},marker=>skill.roll({...options,dc:Number.isFinite(probeDC)?{value:probeDC,visible:false}:null,extraRollOptions:[...new Set([...(options.extraRollOptions??[]),...(origin?.rollOptions??[]),'secret',marker])],callback:(roll,outcome,message)=>{const context=message?.flags?.pf2e?.context??{};if(!Number.isFinite(roll?.options?.totalModifier)||context.type!=='skill-check'||!context.options?.includes('action:recall-knowledge'))throw Error('Workbench 原生技能回执不完整。');probes.set(key,{statistic:slug,label:skill.label??slug,modifier:roll.options.totalModifier,domains:context.domains??[],rollOptions:context.options??[],targetUuid,lore:!!skill.lore});return options.callback?.(roll,outcome,message);}}));if(!captured.receipt.captured)throw Error('原生回忆知识探测接口未接入或已取消；不会再次投骰。');probes.get(key).dosAdjustments=savedAdjustments(captured.receipt.context);receipts.set(key,captured.receipt);return captured.receipt.roll;}catch(error){fail(error);return null;}
 }});
 const scopedActor=scoped(actor,{skills:preparedSkills});
 const scopedToken=scoped(tokenDocument.object??tokenDocument,{actor:scopedActor,document:tokenDocument});
 const scopedTargets=new Set(targets.map(target=>scoped(target.object??target,{document:target,actor:scoped(target.actor,{getSelfRollOptions:(...args)=>{currentTarget=target.uuid;return target.actor.getSelfRollOptions(...args);}})})));if(typeof scopedTargets.first!=='function')Object.defineProperty(scopedTargets,'first',{value:()=>scopedTargets.values().next().value});
 const scopedUser=scoped(user,{targets:scopedTargets}),scopedGame=scoped(game,{user:scopedUser,userId:user.id});
 const BaseMessages=globals.ChatMessage;
 const notifications=scoped(globals.ui.notifications,Object.fromEntries(['info','error'].filter(method=>typeof globals.ui.notifications[method]==='function').map(method=>[method,(message,...args)=>globals.ui.notifications[method](display.notice(message),...args)]))),scopedUI=scoped(globals.ui,{notifications});
 class Messages extends BaseMessages {static getSpeaker(){return {actor:actor.id,token:tokenDocument.id,scene:tokenDocument.parent?.id};}static async create(data){if(macroPublished)throw Error('Workbench 同次回忆知识生成了重复结果卡。');authorization(game,actor,user);const context=data.flags?.pf2e?.context;if(context?.type!=='skill-check'||!context.options?.includes('action:recall-knowledge'))throw Error('Workbench 结果来源不匹配。');const content={...data,content:display.content(data.content),user:user.id,author:user.id,speaker:Messages.getSpeaker(),blind:true,whisper:BaseMessages.getWhisperRecipients('GM').map(u=>u.id)};if(created){const flags={...created.flags,...data.flags,[MODULE_ID]:created.flags?.[MODULE_ID]};if(primaryNativeContext)flags.pf2e={...data.flags.pf2e,context:{...primaryNativeContext,...context,domains:primaryNativeContext.domains,options:[...new Set([...primaryNativeContext.options,...context.options])],outcome:null,unadjustedOutcome:null,dc:null}};delete content.user;delete content.author;await created.update({...content,flags});}else created=await BaseMessages.create(content);macroPublished=true;return created;}}
 try{
 if(assurance){
  const skill=actor.skills[statistic];currentTarget=targets.length===1?targets[0].uuid:null;
  // Capture the ordinary native check first. Assurance's AdjustModifier can
  // suppress an empty-predicate ability modifier on the contextual clone; its
  // ignored flag survives later fortune/misfortune cancellation recalculation.
  const extraRollOptions=['action:recall-knowledge',`action:recall-knowledge:${statistic}`,`skill:rank:${skill.rank}`,...(targets.length===1?targets[0].actor.getSelfRollOptions('target'):[])];
  await Promise.race([failed,preparedSkills[statistic].roll({createMessage:false,skipDialog:true,extraRollOptions,callback(){}})]);
  primary=probes.get(`${currentTarget??''}:${statistic}`);const receipt=receipts.get(`${currentTarget??''}:${statistic}`);
  const conflict=receipt.context.rollTwice==='keep-lower'||receipt.rollOptions.has('misfortune')||receipt.context.substitutions?.some(substitution=>substitution.selected&&substitution.effectType==='misfortune');assuranceApplied=!conflict;
  // Assurance uses the native proficiency values, with no bonuses, penalties or
  // modifier adjustments. Unused status/circumstance rules cannot be if-enabled.
  const proficiencyModifiers=(skill.modifiers??[]).filter(modifier=>modifier.type==='proficiency'&&!modifier.ignored).map(modifier=>new game.pf2e.Modifier({slug:modifier.slug??'proficiency',label:modifier.label??knowledgeProficiencyLabel(game),modifier:modifier.modifier,type:'proficiency',rule:modifier.rule,predicate:[],adjustments:[]}));
  const assuranceCheck=new game.pf2e.CheckModifier(statistic,{modifiers:proficiencyModifiers}),proficiency=assuranceCheck.totalModifier;if(!Number.isFinite(proficiency))throw Error('驾轻就熟的熟练加值不可用。');if(!conflict)receipt.check=assuranceCheck;
  const primaryDC=recallDC(actor,targets.length===1?targets[0].actor:null,statistic,dc);
  created=await BaseMessages.create({content:'<strong>回忆知识 — 驾轻就熟</strong>',rolls:[],user:user.id,author:user.id,speaker:Messages.getSpeaker(),blind:true,whisper:BaseMessages.getWhisperRecipients('GM').map(u=>u.id),flags:{pf2e:{context:{type:'skill-check',options:extraRollOptions,traits:['concentrate','secret']}},[MODULE_ID]:{workbenchRecall:{schema:1,requestId,actorUuid:actor.uuid,tokenUuid:tokenDocument.uuid,userId:user.id,targetUuids:targets.map(t=>t.uuid),targetActors,origin,statistic,assurance:assuranceApplied,assuranceRequested:true,die:null,candidates:[],status:'rolling'}}}});
  const primaryOptions=new Set([...receipt.rollOptions,'fortune']);if(!conflict){primaryOptions.add('assurance');primaryOptions.add('substitute:assurance');}
  receipt.primaryContext={...receipt.context,options:primaryOptions,createMessage:false,skipDialog:true,messageMode:'blind',traits:['concentrate','secret'],dc:Number.isFinite(primaryDC)?{value:primaryDC,visible:false}:null,rollTwice:false,substitutions:[{slug:'assurance',label:'驾轻就熟',value:10,required:true,selected:true,effectType:'fortune'}]};
  for(const rule of receipt.actor.rules?.filter(rule=>!rule.ignored)??[])rule.beforeRoll?.(receipt.domains,receipt.primaryContext.options);
  if(conflict)receipt.primaryContext.options.add('misfortune');
  primaryRoll=await game.pf2e.Check.roll(receipt.check,receipt.primaryContext,null,async(roll,_outcome,nativeMessage)=>{
   primaryNativeContext=nativeMessage.flags.pf2e.context;primary.rollOptions=primaryNativeContext.options;primary.modifier=roll.options.totalModifier;primary.nativeDegree=roll.options.degreeOfSuccess;primary.nativeDC=primaryDC;primary.dosAdjustments=savedAdjustments(receipt.primaryContext);
   if(!conflict&&(roll.dice.length||roll.total!==10+proficiency||roll.options.totalModifier!==proficiency))throw Error('原生驾轻就熟未保留 10 加熟练值规则。');
   rawRoll=conflict?globals.Roll.fromTerms(roll.dice):await new globals.Roll('10').evaluate({allowInteractive:false});await created.update({rolls:[rawRoll],[`flags.${MODULE_ID}.workbenchRecall.die`]:conflict?rawRoll.total:null});
  });
  if(primaryRoll===null&&!rawRoll){await cancelUnrolledReservation();throw Error('原生驾轻就熟检定已取消。');}
  if(!primaryRoll||!rawRoll)throw Error('原生驾轻就熟检定未完成；已保存操作不会重复执行。');
  await Messages.create({content:`<strong>回忆知识 — ${conflict?'驾轻就熟与厄运抵消：原生普通检定':'驾轻就熟'}</strong><p>${escape(knowledgeNativeLabel(game,statistic,skill.label))}: ${primaryRoll.total}</p>`,rolls:[rawRoll],flags:{pf2e:{context:{type:'skill-check',options:['action:recall-knowledge','secret',...(assuranceApplied?['assurance']:[])],traits:['concentrate','secret'],rollMode:'blindroll',target:targets.length===1?{token:targets[0].uuid,actor:targets[0].actor.uuid}:undefined}}}});
 }else{
  if(!game.modules.get('xdy-pf2e-workbench')?.active)throw Error('回忆知识需要启用 Workbench。');let macro=await fromUuid(WORKBENCH_RECALL_UUID);if(!macro?.execute||macro.type!=='script')throw Error('Workbench 回忆知识宏接口不可用。');
  display=await createWorkbenchDisplay({game,macro,actor,token:tokenDocument.object??tokenDocument,targets,globals});
  if(macro.canExecute===false){const data=macro.toObject();delete data._id;data.ownership={...data.ownership,default:globals.CONST.DOCUMENT_OWNERSHIP_LEVELS.OWNER};macro=new macro.constructor(data);}
  for(const target of targets.length?targets:[null]){
   currentTarget=target?.uuid??null;
   const allowed=statistic?[statistic]:target?primarySkills(actor,target.actor):skills;
   const slugs=[...new Set([...allowed,...Object.entries(actor.skills).filter(([,skill])=>skill.lore).map(([slug])=>slug)])];
   // Unknown traits still need one real check to consume generic next-check
   // effects. Its highest ordinary skill is not a creature identification DC.
   if(!slugs.some(slug=>!actor.skills[slug]?.lore))slugs.push(...skills);
   for(const slug of slugs){const skill=preparedSkills[slug];if(!skill)throw Error('指定技能不存在。');const extraRollOptions=['action:recall-knowledge',`action:recall-knowledge:${slug}`,`skill:rank:${actor.skills[slug].rank}`,...(target?target.actor.getSelfRollOptions('target'):[])];await Promise.race([failed,skill.roll({createMessage:false,skipDialog:true,extraRollOptions,callback(){}})]);}
  }
  primary=[...probes.values()].filter(probe=>statistic?probe.statistic===statistic:!probe.lore).sort((a,b)=>b.modifier-a.modifier)[0];if(!primary)throw Error('原生回忆知识主技能不可用。');
  const primaryReceipt=receipts.get(`${primary.targetUuid??''}:${primary.statistic}`),primaryTarget=targets.find(target=>target.uuid===primary.targetUuid),primaryDC=Number.isFinite(dc)?dc:targets.length===1&&primaryTarget&&(statistic||primarySkills(actor,primaryTarget.actor).includes(primary.statistic))?recallDC(actor,primaryTarget.actor,statistic,dc):null;
  const reservation={schema:1,requestId,actorUuid:actor.uuid,tokenUuid:tokenDocument.uuid,userId:user.id,targetUuids:targets.map(t=>t.uuid),targetActors,origin,statistic,assurance:false,die:null,candidates:[],status:'rolling'};
  // Save the operation before its sole real roll, so an interrupted rendering
  // or rule write cannot cause a retry to throw another secret die.
  created=await BaseMessages.create({content:'<strong>回忆知识</strong>',rolls:[],user:user.id,author:user.id,speaker:Messages.getSpeaker(),blind:true,whisper:BaseMessages.getWhisperRecipients('GM').map(u=>u.id),flags:{pf2e:{context:{type:'skill-check',options:['action:recall-knowledge','secret'],traits:['concentrate','secret']}},[MODULE_ID]:{workbenchRecall:reservation}}});
  primaryReceipt.primaryContext={...primaryReceipt.context,options:new Set(primaryReceipt.rollOptions),createMessage:false,skipDialog:true,messageMode:'blind',traits:['concentrate','secret'],dc:Number.isFinite(primaryDC)?{value:primaryDC,visible:false}:null};
  // Restore the chosen rule's own beforeRoll state after comparing other skills.
  for(const rule of primaryReceipt.actor.rules?.filter(rule=>!rule.ignored)??[])rule.beforeRoll?.(primaryReceipt.domains,primaryReceipt.primaryContext.options);
  // libWrapper wrapped continuations expire when their frame returns. Re-enter
  // the installed public Check boundary with a fresh live wrapper chain.
  primaryRoll=await game.pf2e.Check.roll(primaryReceipt.check,primaryReceipt.primaryContext,null,async(roll,_outcome,nativeMessage)=>{
   primaryNativeContext=nativeMessage.flags.pf2e.context;primary.rollOptions=primaryNativeContext.options;
   primary.modifier=roll.options.totalModifier;primaryReceipt.roll.options.totalModifier=roll.options.totalModifier;primary.nativeDegree=roll.options.degreeOfSuccess;primary.nativeDC=primaryDC;primary.dosAdjustments=savedAdjustments(primaryReceipt.primaryContext);
   rawRoll=roll.dice.length?globals.Roll.fromTerms(roll.dice):await new globals.Roll(String(roll.total-roll.options.totalModifier)).evaluate({allowInteractive:false});
   if(!Number.isInteger(rawRoll.total)||rawRoll.total<1||rawRoll.total>20)throw Error('原生回忆知识选中骰点不可验证。');
   await created.update({rolls:[rawRoll],[`flags.${MODULE_ID}.workbenchRecall.die`]:rawRoll.total});
  });
  if(primaryRoll===null&&!rawRoll){await cancelUnrolledReservation();throw Error('原生回忆知识技能检定已取消。');}
  if(!primaryRoll||!rawRoll)throw Error('原生回忆知识技能检定未完成；已保存操作不会再次投骰。');
  assertTargets();
  class SharedRoll {constructor(formula){if(formula!=='1d20')throw Error('Workbench 原始骰子接口改变。');return scoped(rawRoll,{roll:async()=>rawRoll});}}
  await Promise.race([failed,macro.execute({actor:scopedActor,token:scopedToken,game:scopedGame,ChatMessage:Messages,Roll:SharedRoll,CONST:globals.CONST,CONFIG:globals.CONFIG,ui:scopedUI,document:globals.document})]);
  if(!macroPublished)throw Error('Workbench 未生成本次回忆知识卡；原生秘骰已保存，不会再次投骰。');
 }
  assertTargets();
  const die=assuranceApplied?null:created.rolls?.[0]?.total;if(!assuranceApplied&&(!Number.isInteger(die)||die<1||die>20))throw Error('Workbench 原始 d20 不可验证。');
 const candidates=[];const scopeTargets=targets.length?targets:[null];
 const nativeNatural=naturalRollOptions(primaryRoll);
 for(const target of scopeTargets){const allowed=statistic?[statistic]:target?primarySkills(actor,target.actor):skills;for(const probe of probes.values()){if(probe.targetUuid!==(target?.uuid??null)||!allowed.includes(probe.statistic)&&!probe.lore)continue;if(statistic&&probe.statistic!==statistic)continue;const total=(assuranceApplied?10:die)+probe.modifier,effectiveDC=recallDC(actor,target?.actor,statistic,dc);probe.rollOptions=[...new Set([...probe.rollOptions,...(nativeNatural??[])])];candidates.push({...probe,total,dc:probe.lore&&!statistic?null:effectiveDC,degree:Number.isInteger(probe.nativeDegree)&&probe.nativeDC===effectiveDC?probe.nativeDegree:recallDegree({total,die,dc:probe.lore&&!statistic?null:effectiveDC,domains:probe.domains,rollOptions:probe.rollOptions,dosAdjustments:probe.dosAdjustments,Predicate:game.pf2e.Predicate,actor,roll:primaryRoll})});}}
  const state={schema:1,requestId,actorUuid:actor.uuid,tokenUuid:tokenDocument.uuid,userId:user.id,targetUuids:targets.map(t=>t.uuid),targetActors,origin,statistic,assurance:assuranceApplied,assuranceRequested:assurance,die,candidates,primary:primary?{statistic:primary.statistic,targetUuid:primary.targetUuid}:null,status:primaryRoll?'consuming':'pending'};
 await created.update({[`flags.${MODULE_ID}.workbenchRecall`]:state});
 if(primaryRoll)await consumeKnowledgePrimary({message:created,candidate:candidates.find(candidate=>candidate.statistic===primary.statistic&&candidate.targetUuid===primary.targetUuid)??primary,receipt:receipts.get(`${primary.targetUuid??''}:${primary.statistic}`),roll:primaryRoll});
 if(primaryRoll)await created.update({[`flags.${MODULE_ID}.workbenchRecall.status`]:'pending'});
 return {message:created,die,candidates};
 }catch(error){
  // Rendering/target validation can fail after a completed native check. Its
  // saved die still spent the next-check effects; output failure is no replay.
  if(primaryRoll&&rawRoll&&created?.rolls?.[0]?.total===rawRoll.total&&own(created)?.die===(assuranceApplied?null:rawRoll.total)){
   try{await consumeKnowledgePrimary({message:created,candidate:primary,receipt:receipts.get(`${primary.targetUuid??''}:${primary.statistic}`),roll:primaryRoll});}
   catch(cleanupError){throw Error(`${error.message}；原生一次性效果处理未确认：${cleanupError.message}`,{cause:error});}
  }
  throw error;
 }
}
export async function finalizeWorkbenchRecall({game,message,user=game.user,statistic=null,dc=null,fromUuid=globalThis.fromUuid}){
 if(!user?.isGM||game.user!==user||game.users.get(user.id)!==user)throw Error('只有 GM 可裁定回忆知识结果。');
 const state=own(message);if(game.messages.get(message?.id)!==message||state?.schema!==1||state.actorUuid!==message.actor?.uuid||message.flags?.pf2e?.context?.type!=='skill-check'||!message.flags?.pf2e?.context?.options?.includes('action:recall-knowledge')||message.author?.id!==state.userId||!message.actor?.testUserPermission?.(message.author,'OWNER')||!message.blind)throw Error('回忆知识原卡或原操作者不可验证。');
 if(!['pending','done'].includes(state.status)||state.probeUse?.status==='claimed')throw Error('原生回忆知识已保存，原生规则处理尚未完成；不会再次投骰。');
 if((state.assurance?null:message.rolls?.[0]?.total)!==state.die||state.candidates.some(c=>!Number.isFinite(c.modifier)||c.total!==(state.assurance?10:state.die)+c.modifier))throw Error('回忆知识原始骰点或候选结果不可验证。');
  if(!Array.isArray(state.targetActors)||state.targetActors.length!==state.targetUuids.length)throw Error('回忆知识目标角色来源不可验证。');
  for(const [index,binding]of state.targetActors.entries()){
   const target=typeof fromUuid==='function'?doc(await fromUuid(binding.tokenUuid)):null;
   if(binding.tokenUuid!==state.targetUuids[index]||!binding.actorUuid||target?.documentName!=='Token'||target.actor?.uuid!==binding.actorUuid)throw Error('回忆知识目标关联角色已改变；原始秘骰保留，不会再次投骰。');
  }
  if(state.targetActors.length===1){const native=message.flags.pf2e.context.target,binding=state.targetActors[0];if(native?.token!==binding.tokenUuid||native.actor!==binding.actorUuid)throw Error('回忆知识原生目标角色与保存来源不匹配。');}
 const available=state.candidates.filter(c=>state.targetUuids.length===1?c.targetUuid===state.targetUuids[0]:state.targetUuids.length===0&&!c.targetUuid);
 const selection=statistic??state.statistic;
 const candidate=selection?available.find(c=>c.statistic===selection):available.find(c=>c.statistic===state.primary?.statistic&&c.targetUuid===state.primary.targetUuid)??available.filter(c=>!c.lore&&Number.isFinite(c.dc)).sort((a,b)=>b.modifier-a.modifier)[0];
 if(!candidate&&!selection)return null;
 if(!candidate)throw Error('请选择本次已保存的候选技能；不能接收外部检定总值。');
 if(state.result){if(state.result.statistic!==candidate.statistic||Number.isFinite(dc)&&dc!==state.result.dc)throw Error('该次机械结果已锁定；其他技能与 DC 供 GM 信息裁定，不会再次触发收益。');return state.result;}
 const effectiveDC=Number.isFinite(dc)?dc:candidate.dc;if(!Number.isFinite(effectiveDC))return null;
 const result={statistic:candidate.statistic,dc:effectiveDC,total:candidate.total,die:state.die,degree:effectiveDC===candidate.dc?candidate.degree:recallDegree({...candidate,dc:effectiveDC,die:state.die,actor:message.actor,roll:message.rolls?.[0],Predicate:game.pf2e?.Predicate}),targetUuid:candidate.targetUuid,assurance:state.assurance};
 if(!Number.isInteger(result.degree))return null;
 const options=[...new Set([...message.flags.pf2e.context.options,...candidate.rollOptions,...(state.origin?.rollOptions??[])])];
 await message.update({[`flags.${MODULE_ID}.workbenchRecall`]:{...state,status:'done',result},'flags.pf2e.context.outcome':outcomes[result.degree],'flags.pf2e.context.dc':{value:effectiveDC,visible:false},'flags.pf2e.context.options':options,'flags.pf2e.context.domains':candidate.domains,'flags.pf2e.context.statistic':candidate.statistic});return result;
}
