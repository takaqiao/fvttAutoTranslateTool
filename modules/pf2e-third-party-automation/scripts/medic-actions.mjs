import {MODULE_ID} from './rules.mjs';
import {SerialActions} from './runtime.mjs';
import {isActiveGM,resolveMessageTargets,showNativeChoice} from './native-context.mjs';
import {medicAction,medicFeat,visitationBranches,values,usableToolkit,validateTreatment,conditionValue,conditionBound,conditionSnapshot,treatmentValue} from './medic-rules.mjs';
import {createMedicNative,pinnedMedicTarget} from './medic-native.mjs';
import {createMedicOwner} from './medic-owner.mjs';

const sessions=new WeakMap();
const own=m=>m?.flags?.[MODULE_ID]?.medic;
const input=m=>m?.flags?.[MODULE_ID]?.medicInput;
const uid=()=>globalThis.foundry?.utils?.randomID?.(16)??globalThis.crypto.randomUUID();
const author=m=>m.author?.id??m.user?.id??m.user;
const terminal=s=>['done','cancelled','failed','uncertain'].includes(s);
const rollData=roll=>JSON.stringify(roll?.toJSON?.()??roll);
const turn=game=>game.combat?.started?`${game.combat.id}:${game.combat.round}:${game.combat.turn}`:null;
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

/** This dialog exists only on the authority client. No facts are persisted on the public activity. */
async function nativeFacts({condition}){
 const Dialog=globalThis.foundry?.applications?.api?.DialogV2;
 if(!Dialog)throw Error('缺少GM来源事实对话框。');
 return Dialog.wait({window:{title:'处理状态 · GM来源事实'},content:`<p>状态：${esc(condition.name??condition.slug)}。请输入真实来源事实；原生检定自动决定成功度。接触距离由GM裁定。</p><label>来源DC <input name="dc" type="number" min="1" required></label><label><input name="restricted" type="checkbox">神器或20级以上效果</label><label><input name="continuous" type="checkbox">产生状态的持续情境仍存在（处理无效）</label>`,buttons:[{action:'confirm',label:'确认事实',callback:(_event,button)=>{const f=new FormData(button.form);return {dc:Number(f.get('dc')),restricted:f.has('restricted'),continuous:f.has('continuous')};}},{action:'cancel',label:'取消',callback:()=>null}],rejectClose:false});
}

/** Original Use is the sole entry. Authority and durable nonce checks also apply to card continuation. */
export function createMedicActions({game,fromUuid=globalThis.fromUuid,choose,requestFacts=nativeFacts,rollCheck,delegateTreatment,commitActivity,onError=console.error}={}){
 delegateTreatment??=createMedicNative({game,choose:args=>game.user.id===args.user.id?showNativeChoice(args):choose?.(args)});
 if(!sessions.has(game))sessions.set(game,{actors:new SerialActions(),targets:new SerialActions()});
 const queues=sessions.get(game),hooks=[],movementCards=new Map();let hookApi,socket,movementsIndexed=false;
 const indexMovement=message=>{if(message?.id&&game.messages.get(message.id)===message&&own(message)?.status==='movement')movementCards.set(message.id,message);else if(message?.id)movementCards.delete(message.id);};
 const restoreMovements=()=>{if(movementsIndexed)return;movementsIndexed=true;for(const message of values(game.messages))indexMovement(message);};
 const gm=()=>{if(!isActiveGM(game)||!game.user?.isGM)throw Error('医疗结算需要当前主GM。');};
 const save=async(m,changes)=>{gm();if(game.messages.get(m.id)!==m)throw Error('原始动作消息不存在。');await m.update({[`flags.${MODULE_ID}.medic`]:{...own(m),...changes}});indexMovement(m);gm();};
 function validate(ctx,{authority=true}={}){
  if(authority)gm();const {actor,item,message,user,action}=ctx;
  if(!user?.active||game.users.get(user.id)!==user||!actor?.testUserPermission?.(user,'OWNER'))throw Error('需要原使用者的角色所有权。');
  if(actor.items.get(item?.id)!==item||item.actor!==actor||medicAction(item)!==action)throw Error('医疗专长来源已变化。');
  if(!message?.id||game.messages.get(message.id)!==message||author(message)!==user.id||message.speaker?.actor!==actor.id||message.flags?.pf2e?.origin?.uuid!==item.uuid||message.flags?.[MODULE_ID]?.usageInput?.actualUse!==true||typeof input(message)?.nonce!=='string')throw Error('需要原始专长的真实Use回执。');
  if(own(message)&&(actor.flags?.[MODULE_ID]?.medicUses?.[input(message).nonce]!==message.id||own(message).actorUuid!==actor.uuid||own(message).itemUuid!==item.uuid||own(message).userId!==user.id))throw Error('医疗回执不属于此原始消息。');
  return ctx;
 }
 const pick=async(actor,user,title,choices)=>{if(!choices.length)throw Error('没有合法医疗选项。');const selected=choices.length===1?choices[0].value:await choose?.({actor,user,title,choices});if(selected==null||selected===false)return null;if(!choices.some(c=>c.value===selected))throw Error('无效的医疗选择。');return selected;};
 function requireTurn(actor){if(game.combat?.started&&game.combat.combatant?.actor?.uuid!==actor.uuid)throw Error('医师探访需要角色自己的回合。');return turn(game);}
 async function healerToken(actor,message){const uuid=message.speaker?.scene&&message.speaker?.token?`Scene.${message.speaker.scene}.Token.${message.speaker.token}`:null;const token=uuid?await fromUuid(uuid):null;if(token?.actor?.uuid!==actor.uuid)throw Error('需要原卡所指的场景施术者Token。');return token;}
 function validatePatientContext(actor,healer,target){
  if(!usableToolkit(actor))throw Error('需要持握医疗工具包，或穿戴工具包且有空手。');
  const scene=healer.parent;
  if(healer.actor?.uuid!==actor.uuid||!scene||game.scenes?.get(scene.id)!==scene||scene.tokens?.get(healer.id)!==healer||target.parent!==scene||scene.tokens.get(target.id)!==target)throw Error('需要原卡所指场景中仍存在的施术者和医疗目标 Token。');
 }
 async function targetFor(ctx){const targets=await resolveMessageTargets(ctx.message,{fromUuid});if(targets.length!==1)throw Error('请在原始Use时选定一个医疗目标。');return targets[0];}
 async function commit(ctx,cost,flourish){
  validate(ctx);const {actor,message}=ctx,key=input(message).nonce,ledger=actor.flags?.[MODULE_ID]?.medicUses??{};
  if(ledger[key]&&ledger[key]!==message.id)throw Error('此医疗Use回执已属于另一个原始消息。');
  const currentTurn=flourish?requireTurn(actor):turn(game);
  if(flourish&&currentTurn){
   const existing=actor.flags?.[MODULE_ID]?.medicFlourish;
   const other=values(game.messages).some(m=>m.id!==message.id&&m.speaker?.actor===actor.id&&m.flags?.[MODULE_ID]?.medicObservedFlourishTurn===currentTurn&&m.flags?.pf2e?.origin?.rollOptions?.includes('origin:action:slug:use-action')&&m.flags?.pf2e?.origin?.rollOptions?.includes('origin:item:trait:flourish'));
   if(existing?.turn===currentTurn||other)throw Error('本回合已使用华丽动作。');
  }
  await actor.update({[`flags.${MODULE_ID}.medicUses`]:{...ledger,[key]:message.id},...(flourish?{[`flags.${MODULE_ID}.medicFlourish`]:{turn:currentTurn,messageId:message.id}}:{})});
  await save(message,{status:'committed',nonce:key,userId:ctx.user.id,actorUuid:actor.uuid,itemUuid:ctx.item.uuid,cost,turn:currentTurn,flourish});
  // No system-wide action pool is invented. Integrations can account for this single activity here.
  await commitActivity?.({...ctx,cost,flourish,nonce:key});
 }
 async function nativeRoll({actor,item,healer,target,nonce,dc}){
  const stat=actor.getStatistic?.('medicine')??actor.skills?.medicine;
  if(!stat?.clone)throw Error('缺少原生医疗Statistic接口。');
  let receipt=null;
  await stat.clone({check:{domains:['counteract-check']}}).check.roll({item,token:healer,dc:{value:dc,visible:false,slug:'medicine'},extraRollOptions:['action:treat-condition',`${MODULE_ID}:medic:${nonce}`],action:'treat-condition',traits:['healing','manipulate'],target:pinnedMedicTarget(target),createMessage:true,skipDialog:false,messageMode:'blind',callback:(roll,_outcome,message)=>{receipt={roll,message};}});
  return receipt;
 }
 const owner=createMedicOwner({game,fromUuid,context:async message=>{
  const state=own(message),item=await fromUuid(state.itemUuid);
  return validate({actor:item?.actor,item,message,user:game.users.get(state.userId),action:medicAction(item)},{authority:false});
 },execute:async ctx=>{
  const {actor,message,healer,target,state}=ctx;
  if(state.nativeKind==='treat-condition'){
   if(!Number.isInteger(ctx.dc)||ctx.dc<1)throw Error('缺少GM提供的医疗来源DC。');
   const receipt=await (rollCheck??nativeRoll)({...ctx,item:medicFeat(actor,'treatCondition'),nonce:state.nonce});
   if(receipt&&rollData(receipt.roll)!==rollData(receipt.message?.rolls?.[0]))throw Error('原生医疗回调与保存的骰子不一致。');
   return receipt?{status:'delegated',checkId:receipt.message?.id}:{status:'cancelled'};
  }
  return delegateTreatment({...ctx,branch:state.branch,continuation:{actorUuid:actor.uuid,cardId:message.id,nonce:state.nonce}});
 }});
 function validateDelegated(ctx,target,result){
  const state=own(ctx.message),check=result.check,branch=state.branch,workbench=branch==='battle-medicine',key=workbench?'medicWorkbench':'medicNative',proof=check?.flags?.[MODULE_ID]?.[key];
  const matches=p=>p?.nonce===state.nonce&&p.cardId===ctx.message.id&&p.actorUuid===ctx.actor.uuid&&p.targetUuid===target.uuid&&p.branch===branch;
  if(!check?.id||game.messages.get(check.id)!==check||author(check)!==ctx.user.id||check.speaker?.actor!==ctx.actor.id||!matches(proof)||check.flags?.[MODULE_ID]?.medicReceipt)throw Error('医疗原生结果不属于本次操作者和患者。');
  const pf=check.flags?.pf2e?.context,roll=check.rolls?.[0],degree=roll?.options?.degreeOfSuccess;
  if(!Number.isFinite(roll?.total)||check.rolls.length!==1)throw Error('医疗原生骰子尚未完成。');
  if(!workbench||proof.checkKind!=='assurance'){
   const marker=`${MODULE_ID}:medic-${workbench?'workbench':'native'}:${state.nonce}`;
   const hasOutcome=workbench||pf?.dc!=null||degree!=null||pf?.outcome!=null;
   if(pf?.type!=='skill-check'||pf.isReroll||!pf.options?.includes(marker)||pf.target?.actor!==target.actor.uuid||pf.target?.token!==target.uuid||hasOutcome&&(!Number.isInteger(degree)||degree<0||degree>3||pf.outcome!==['criticalFailure','failure','success','criticalSuccess'][degree]))throw Error('医疗检定来源、目标或成功度不匹配。');
  }
  if(workbench){
   const card=result.resultMessage,receipt=card?.flags?.treat_wounds_battle_medicine;
   if(!card?.id||game.messages.get(card.id)!==card||author(card)!==ctx.user.id||card.speaker?.actor!==ctx.actor.id||!matches(card.flags?.[MODULE_ID]?.medicWorkbench)||card.flags[MODULE_ID].medicWorkbench.checkId!==check.id||receipt?.id!==target.id||receipt.healerId!==ctx.actor.id||!Number.isInteger(receipt.dos)||receipt.dos<0||receipt.dos>3||proof.checkKind!=='assurance'&&receipt.dos!==degree)throw Error('Workbench结果卡与原生医疗检定不匹配。');
  }
 }
 async function treat(ctx,healer,target){
  const {actor,message,user}=ctx;
  const options=values(target.actor.items).filter(c=>c.type==='condition'&&['clumsy','enfeebled','sickened'].includes(c.slug)&&conditionValue(c)>0&&!conditionBound(c)).map(c=>({value:c.id,label:`${c.name??c.slug} ${conditionValue(c)}`}));
  const selected=await pick(actor,user,'处理状态：选择一种状态',options);if(!selected){await save(message,{status:'cancelled'});return '已取消处理状态。';}
  return queues.targets.run(target.actor.uuid,async()=>{
   validate(ctx);const condition=target.actor.items.get(selected);const before=condition&&conditionSnapshot(condition);
   const facts=await requestFacts({actor,target,condition,user,message});if(!facts){await save(message,{status:'cancelled'});return '已取消处理状态。';}
   validatePatientContext(actor,healer,target);const {dc}=validateTreatment({actor,condition,facts});
   await save(message,{status:'rolling',nativeKind:'treat-condition',healerUuid:healer.uuid,targetUuid:target.uuid});
   // Counteract's generic global dialog fields are not used; the native Check API receives exact local DC.
   const receipt=await owner.run(ctx,{dc});
   validate(ctx);validatePatientContext(actor,healer,target);
   if(receipt.status==='cancelled'){await save(message,{status:'cancelled'});return '已取消原生检定；已承诺动作不回退。';}
   const check=receipt.check,roll=check?.rolls?.[0],pf=check?.flags?.pf2e?.context,degree=roll?.options?.degreeOfSuccess;
   if(!check?.id||game.messages.get(check.id)!==check||author(check)!==user.id||check.speaker?.actor!==actor.id||pf?.type!=='skill-check'||pf.isReroll||!pf.options?.includes(`${MODULE_ID}:medic:${own(message).nonce}`)||pf.dc?.value!==dc||pf.target?.actor!==target.actor.uuid||pf.target?.token!==target.uuid||check.rolls.length!==1||!Number.isFinite(roll?.total)||!Number.isInteger(degree)||degree<0||degree>3||pf.outcome!==['criticalFailure','failure','success','criticalSuccess'][degree]||check.flags?.[MODULE_ID]?.medicReceipt)throw Error('原生医疗检定回执不匹配或已经使用。');
   const current=target.actor.items.get(selected);
   if(current!==condition||conditionSnapshot(current)!==before)throw Error('目标状态或来源在检定期间已变化，未覆盖新状态。');
   validateTreatment({actor,condition:current,facts});
   const after=treatmentValue(conditionValue(current),degree);
   await check.update({[`flags.${MODULE_ID}.medicReceipt`]:{messageId:message.id,nonce:own(message).nonce}});
   await save(message,{status:'applying',checkId:check.id});
   gm();if(after===0)await current.delete();else if(after!==conditionValue(current))await current.update({'system.value.value':after});
   const result=['大失败：状态值增加1。','失败：状态未改变。','成功：状态值降低1。','大成功：状态值降低2。'][degree];
   await save(message,{status:'done',result});return result;
  });
 }
 async function executeUsage(ctx){return queues.actors.run(ctx.actor?.uuid,async()=>{
  validate(ctx);if(own(ctx.message)){if(own(ctx.message).actorUuid!==ctx.actor.uuid||own(ctx.message).nonce!==input(ctx.message).nonce||ctx.actor.flags?.[MODULE_ID]?.medicUses?.[input(ctx.message).nonce]!==ctx.message.id)throw Error('无效原卡回执。');return own(ctx.message).result??'此医疗动作已开始。';}
  const {actor,message,user}=ctx,target=await targetFor(ctx),healer=await healerToken(actor,message);
  if(ctx.action==='medic:treat-condition'){
   validatePatientContext(actor,healer,target);await commit(ctx,2,false);
   try{return await treat(ctx,healer,target);}catch(error){await save(message,{status:own(message).nativeKind?'uncertain':'failed',result:error.message});throw error;}
  }
  const branch=await pick(actor,user,'医师探访：选择本次医疗动作',visitationBranches(actor));if(!branch)return '已取消医师探访。';
  const cost=visitationBranches(actor).find(b=>b.value===branch).cost;
  await commit(ctx,cost,true);await save(message,{status:'movement',branch,healerUuid:healer.uuid,targetUuid:target.uuid,result:'请自行完成本次原生行走，再在本卡确认“移动完成”。行走与接触距离由GM裁定；取消治疗不会退还华丽动作。'});
  return '医师探访已承诺；请完成行走后在本卡确认。';
 });}
 async function continuationContext(message,user){const state=own(message);if(!state||state.userId!==user?.id)throw Error('只能由原使用者继续医疗。');const item=await fromUuid(state.itemUuid);return validate({actor:item?.actor,item,message,user,action:medicAction(item)});}
 async function continueUsage(message,user,{cancel=false,movementConfirmed=false}={}){
  const ctx=await continuationContext(message,user);
  return queues.actors.run(ctx.actor.uuid,async()=>{
   validate(ctx);const state=own(message);if(terminal(state.status))return state.result;
   if(state.status!=='movement')throw Error('此活动正在处理或不可继续。');
   if(turn(game)!==state.turn){await save(message,{status:'cancelled',result:'回合已改变，后续医疗已过期。'});throw Error('医疗延续已过期。');}
   if(cancel){await save(message,{status:'cancelled',result:'已取消后续医疗；已承诺动作与华丽不回退。'});return;}
   requireTurn(ctx.actor);if(movementConfirmed!==true)throw Error('请由原使用者在本卡确认已完成本次行走。');
   const healer=await fromUuid(state.healerUuid),target=await fromUuid(state.targetUuid);if(healer?.actor!==ctx.actor||!target?.actor)throw Error('原始医疗Token不存在。');
   validatePatientContext(ctx.actor,healer,target);if(!visitationBranches(ctx.actor).some(b=>b.value===state.branch))throw Error('医疗分支资格已变化。');
   await save(message,{status:'treatment',nativeKind:state.branch,movementConfirmation:{userId:user.id,turn:state.turn,healerUuid:healer.uuid,targetUuid:target.uuid}});
   try{
    if(state.branch==='treat-condition')return await treat(ctx,healer,target);
    const result=await owner.run(ctx);validate(ctx);
    if(result.status!=='cancelled'){validateDelegated(ctx,target,result);await result.check.update({[`flags.${MODULE_ID}.medicReceipt`]:{messageId:message.id,nonce:state.nonce}});}
    await save(message,{status:result?.status==='cancelled'?'cancelled':'done',result:result?.text??'已调用原生医疗；其结果与免疫由原医疗提供者处理。'});return own(message).result;
   }catch(error){await save(message,{status:own(message).nativeKind?'uncertain':'failed',result:error.message});throw error;}
  });
 }
 function captureUsage(item){return medicAction(item)?{medicInput:{nonce:uid()}}:null;}
 function renderChatMessage(message,html){const state=own(message),el=html?.[0]??html;if(!state||!el?.querySelector)return;el.querySelector('.medic-activity')?.remove();const glyph=el.querySelector('.action-glyph');if(glyph)glyph.textContent=state.cost===2?'D':'A';const can=game.user.id===state.userId&&state.status==='movement';const panel=document.createElement('div');panel.className='medic-activity';panel.innerHTML=`<p>${esc(state.result??state.status)} · ${state.cost}动作</p>${can?'<button type="button" data-medic="continue">移动完成，继续医疗</button><button type="button" data-medic="cancel">取消后续医疗</button>':''}`;panel.addEventListener('click',event=>{const action=event.target?.closest?.('[data-medic]')?.dataset.medic;if(!action)return;const args={messageId:message.id,nonce:state.nonce,userId:game.user.id,cancel:action==='cancel',movementConfirmed:action==='continue'};(isActiveGM(game)?continueUsage(message,game.user,args):socket?.executeAsGM?.('continueMedic',args))?.catch(onError);});(el.querySelector('.message-content')??el).append(panel);}
 async function maintain(){if(!isActiveGM(game))return;restoreMovements();for(const m of [...movementCards.values()]){if(game.messages.get(m.id)!==m||own(m)?.status!=='movement'){movementCards.delete(m.id);continue;}if(own(m).turn!==turn(game))await save(m,{status:'cancelled',result:'回合已改变，后续医疗已过期。'});}}
 function register({Hooks=globalThis.Hooks,socket:providedSocket}={}){
  if(hookApi)return;hookApi=Hooks;socket=providedSocket??globalThis.socketlib?.registerModule?.(MODULE_ID);
  owner.register({Hooks,socket});
  socket?.register?.('continueMedic',async function(args){gm();const message=game.messages.get(args.messageId),user=game.users.get(this.socketdata?.userId);if(own(message)?.nonce!==args.nonce)throw Error('原卡医疗回执不匹配。');return continueUsage(message,user,args);});
  const observeFlourish=message=>{const opts=message.flags?.pf2e?.origin?.rollOptions??[];if(turn(game)&&opts.includes('origin:action:slug:use-action')&&opts.includes('origin:item:trait:flourish'))message.updateSource({[`flags.${MODULE_ID}.medicObservedFlourishTurn`]:turn(game)});};
  for(const [event,fn]of [['preCreateChatMessage',observeFlourish],['createChatMessage',indexMovement],['updateChatMessage',indexMovement],['deleteChatMessage',message=>movementCards.delete(message.id)],['renderChatMessageHTML',renderChatMessage],['updateCombat',()=>maintain().catch(onError)]])hooks.push([event,Hooks.on(event,fn)]);
  if(isActiveGM(game))restoreMovements();
  return ()=>{for(const [event,id]of hooks)Hooks.off(event,id);};
 }
 return {resolveAction:medicAction,captureUsage,executeUsage,continueUsage,register,renderChatMessage,maintain};
}
