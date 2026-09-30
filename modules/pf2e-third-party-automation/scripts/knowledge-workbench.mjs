import {MODULE_ID} from './rules.mjs';
import {hasSkillAssurance} from './knowledge-automatic.mjs';
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
export function recallDegree({total,die,dc,domains=[],rollOptions=[],actor}){
 if(!Number.isFinite(total)||!Number.isFinite(dc))return null;
 let degree=total-dc>=10?3:total>=dc?2:total-dc<=-10?0:1;
 if(die===20)degree=Math.min(3,degree+1);else if(die===1)degree=Math.max(0,degree-1);
 const options=[...rollOptions,`check:total:${total}`,`check:total:natural:${die??10}`,`check:total:delta:${total-dc}`];
 const adjustments=domains.flatMap(domain=>(actor?.synthetics?.degreeOfSuccessAdjustments?.[domain]??[]).filter(a=>a.predicate?.test?.(options)??true).flatMap(a=>[a.adjustments?.all??[],a.adjustments?.[outcomes[degree]]??[]].flat()));
 if(adjustments.length)degree=Math.max(...adjustments.map(adjustment=>{const explicit=outcomes.indexOf(adjustment.amount);return explicit>=0?explicit:Number.isFinite(adjustment.amount)?Math.max(0,Math.min(3,degree+adjustment.amount)):degree;}));
 return degree;
}
function primarySkills(actor,target){const relevant=new Set();for(const [trait,list]of Object.entries(identify))if(target?.traits?.has?.(trait)||values(target?.traits).includes(trait))for(const skill of list)relevant.add(skill);if(actor.itemTypes?.feat?.some(f=>(f.slug??f.system?.slug)==='unified-theory')&&['religion','occultism','nature'].some(s=>relevant.has(s)))relevant.add('arcana');return [...relevant];}
function authorization(game,actor,user){if(game.user!==user||game.users.get(user.id)!==user||!actor?.testUserPermission?.(user,'OWNER'))throw Error('回忆知识需要原操作者及角色所有权。');}
/** Execute the installed macro unchanged. Scoped native probes supply structured results; no HTML parsing or global latest-message lookup. */
export async function captureWorkbenchRecall({game,actor,token,user=game.user,targetUuids=[],requestId,origin=null,statistic=null,assurance=false,dc=null,fromUuid=globalThis.fromUuid,globals=globalThis}){
 authorization(game,actor,user);if(!requestId||typeof requestId!=='string')throw Error('回忆知识缺少操作来源。');if(assurance&&(!statistic||actor.skills?.[statistic]?.rank<1||!hasSkillAssurance(actor,statistic)))throw Error('Assurance 必须具有对应专长并保留指定技能。');
 const tokenDocument=doc(token);if(tokenDocument?.actor!==actor)throw Error('回忆知识的原始 Token 与角色不匹配。');
 const targets=[];for(const uuid of [...new Set(targetUuids)]){const target=doc(await fromUuid(uuid));if(target?.documentName!=='Token'||!target.actor)throw Error('回忆知识目标已不存在。');targets.push(target);}
 const probes=new Map();let currentTarget=null,created=null,fail;const failed=new Promise((_,reject)=>{fail=reject;});failed.catch(()=>{});
 const preparedSkills={};
 for(const [slug,skill]of Object.entries(actor.skills??{}))preparedSkills[slug]=scoped(skill,{roll:async options=>{
  authorization(game,actor,user);const targetUuid=currentTarget;
  try{const result=await skill.roll({...options,extraRollOptions:[...new Set([...(options.extraRollOptions??[]),...(origin?.rollOptions??[])])],callback:(roll,outcome,message)=>{const context=message?.flags?.pf2e?.context??{};if(!Number.isFinite(roll?.options?.totalModifier)||context.type!=='skill-check'||!context.options?.includes('action:recall-knowledge'))throw Error('Workbench 原生技能回执不完整。');probes.set(`${targetUuid??''}:${slug}`,{statistic:slug,label:skill.label??slug,modifier:roll.options.totalModifier,domains:context.domains??[],rollOptions:context.options??[],targetUuid,lore:!!skill.lore});return options.callback?.(roll,outcome,message);}});if(!result)fail(Error('原生回忆知识技能检定已取消；不会再次投骰。'));return result;}catch(error){fail(error);return null;}
 }});
 const scopedActor=scoped(actor,{skills:preparedSkills});
 const scopedToken=scoped(tokenDocument.object??tokenDocument,{actor:scopedActor,document:tokenDocument});
 const scopedTargets=new Set(targets.map(target=>scoped(target.object??target,{document:target,actor:scoped(target.actor,{getSelfRollOptions:(...args)=>{currentTarget=target.uuid;return target.actor.getSelfRollOptions(...args);}})})));scopedTargets.first=()=>scopedTargets.values().next().value;
 const scopedUser=scoped(user,{targets:scopedTargets}),scopedGame=scoped(game,{user:scopedUser,userId:user.id});
 const BaseMessages=globals.ChatMessage;
 class Messages extends BaseMessages {static getSpeaker(){return {actor:actor.id,token:tokenDocument.id,scene:tokenDocument.parent?.id};}static async create(data){if(created)throw Error('Workbench 同次回忆知识生成了重复结果卡。');authorization(game,actor,user);const context=data.flags?.pf2e?.context;if(context?.type!=='skill-check'||!context.options?.includes('action:recall-knowledge'))throw Error('Workbench 结果来源不匹配。');created=await BaseMessages.create({...data,user:user.id,author:user.id,speaker:Messages.getSpeaker(),blind:true,whisper:BaseMessages.getWhisperRecipients('GM').map(u=>u.id)});return created;}}
 if(assurance){const skill=actor.skills?.[statistic];if(!skill)throw Error('指定技能不存在。');const proficiency=(skill.modifiers??[]).filter(m=>m.type==='proficiency'&&!m.ignored).reduce((n,m)=>n+(m.modifier??0),0);if(!Number.isFinite(proficiency))throw Error('Assurance 熟练加值不可用。');currentTarget=targets.length===1?targets[0].uuid:null;probes.set(`${currentTarget??''}:${statistic}`,{statistic,label:skill.label??statistic,modifier:proficiency,domains:[statistic,'skill-check'],rollOptions:[...(origin?.rollOptions??[]),...(actor.getRollOptions?.([statistic,'skill-check'])??[]),'action:recall-knowledge',`action:recall-knowledge:${statistic}`,`skill:rank:${skill.rank}`,'assurance','substitute:assurance','fortune',...(targets.length===1?targets[0].actor.getSelfRollOptions('target'):[])],targetUuid:currentTarget,lore:!!skill.lore});await Messages.create({content:`<strong>Recall Knowledge — Assurance</strong><p>${escape(skill.label??statistic)}: ${10+proficiency}</p>`,rolls:[],flags:{pf2e:{context:{type:'skill-check',options:['action:recall-knowledge','secret','assurance'],traits:['concentrate','secret'],rollMode:'blindroll',target:targets.length===1?{token:targets[0].uuid,actor:targets[0].actor.uuid}:undefined}}}});
 }else{
  if(!game.modules.get('xdy-pf2e-workbench')?.active)throw Error('回忆知识需要启用 Workbench。');let macro=await fromUuid(WORKBENCH_RECALL_UUID);if(!macro?.execute||macro.type!=='script')throw Error('Workbench 回忆知识宏接口不可用。');
  if(macro.canExecute===false){const data=macro.toObject();delete data._id;data.ownership={...data.ownership,default:globals.CONST.DOCUMENT_OWNERSHIP_LEVELS.OWNER};macro=new macro.constructor(data);}
  await Promise.race([failed,macro.execute({actor:scopedActor,token:scopedToken,game:scopedGame,ChatMessage:Messages,Roll:globals.Roll,CONST:globals.CONST,CONFIG:globals.CONFIG,ui:globals.ui,document:globals.document})]);
  if(!created)throw Error('Workbench 未生成本次回忆知识卡；不会再次投骰。');
  if(statistic&&!probes.has(`${targets.length===1?targets[0].uuid:''}:${statistic}`)){const skill=preparedSkills[statistic];if(!skill)throw Error('指定技能不存在。');currentTarget=targets.length===1?targets[0].uuid:null;const options=['action:recall-knowledge',`action:recall-knowledge:${statistic}`,...(targets.length===1?targets[0].actor.getSelfRollOptions('target'):[])];await Promise.race([failed,skill.roll({createMessage:false,rollMode:'blindroll',skipDialog:true,extraRollOptions:options,callback(){}})]);}
 }
 const die=assurance?null:created.rolls?.[0]?.total;if(!assurance&&(!Number.isInteger(die)||die<1||die>20))throw Error('Workbench 原始 d20 不可验证。');
 const candidates=[];const scopeTargets=targets.length?targets:[null];
 for(const target of scopeTargets){const allowed=statistic?[statistic]:target?primarySkills(actor,target.actor):skills;for(const probe of probes.values()){if(probe.targetUuid!==(target?.uuid??null)||!allowed.includes(probe.statistic)&&!probe.lore)continue;if(statistic&&probe.statistic!==statistic)continue;const total=(assurance?10:die)+probe.modifier,effectiveDC=Number.isFinite(dc)?dc:target?targetDC(target.actor):null;candidates.push({...probe,total,dc:probe.lore&&!statistic?null:effectiveDC,degree:recallDegree({total,die,dc:probe.lore&&!statistic?null:effectiveDC,domains:probe.domains,rollOptions:probe.rollOptions,actor})});}}
 const state={schema:1,requestId,actorUuid:actor.uuid,tokenUuid:tokenDocument.uuid,userId:user.id,targetUuids:targets.map(t=>t.uuid),origin,statistic,assurance,die,candidates,status:'pending'};
 await created.update({[`flags.${MODULE_ID}.workbenchRecall`]:state});return {message:created,die,candidates};
}
export async function finalizeWorkbenchRecall({game,message,user=game.user,statistic=null,dc=null}){
 if(!user?.isGM||game.user!==user||game.users.get(user.id)!==user)throw Error('只有 GM 可裁定回忆知识结果。');
 const state=own(message);if(game.messages.get(message?.id)!==message||state?.schema!==1||state.actorUuid!==message.actor?.uuid||message.flags?.pf2e?.context?.type!=='skill-check'||!message.flags?.pf2e?.context?.options?.includes('action:recall-knowledge')||message.author?.id!==state.userId||!message.actor?.testUserPermission?.(message.author,'OWNER')||!message.blind)throw Error('回忆知识原卡或原操作者不可验证。');
 if((state.assurance?null:message.rolls?.[0]?.total)!==state.die||state.candidates.some(c=>!Number.isFinite(c.modifier)||c.total!==(state.assurance?10:state.die)+c.modifier))throw Error('回忆知识原始骰点或候选结果不可验证。');
 const available=state.candidates.filter(c=>state.targetUuids.length===1?c.targetUuid===state.targetUuids[0]:state.targetUuids.length===0&&!c.targetUuid);
 const selection=statistic??state.statistic;
 const candidate=selection?available.find(c=>c.statistic===selection):available.filter(c=>!c.lore&&Number.isFinite(c.dc)).sort((a,b)=>b.modifier-a.modifier)[0];
 if(!candidate&&!selection)return null;
 if(!candidate)throw Error('请选择本次已保存的候选技能；不能接收外部检定总值。');
 if(state.result){if(state.result.statistic!==candidate.statistic||Number.isFinite(dc)&&dc!==state.result.dc)throw Error('该次机械结果已锁定；其他技能与 DC 供 GM 信息裁定，不会再次触发收益。');return state.result;}
 const effectiveDC=Number.isFinite(dc)?dc:candidate.dc;if(!Number.isFinite(effectiveDC))return null;
 const result={statistic:candidate.statistic,dc:effectiveDC,total:candidate.total,die:state.die,degree:recallDegree({...candidate,dc:effectiveDC,die:state.die,actor:message.actor}),targetUuid:candidate.targetUuid,assurance:state.assurance};
 const options=[...new Set([...message.flags.pf2e.context.options,...candidate.rollOptions,...(state.origin?.rollOptions??[])])];
 await message.update({[`flags.${MODULE_ID}.workbenchRecall`]:{...state,status:'done',result},'flags.pf2e.context.outcome':outcomes[result.degree],'flags.pf2e.context.dc':{value:effectiveDC,visible:false},'flags.pf2e.context.options':options,'flags.pf2e.context.domains':candidate.domains,'flags.pf2e.context.statistic':candidate.statistic});return result;
}
