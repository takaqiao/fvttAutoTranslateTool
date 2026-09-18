import {MODULE_ID} from './rules.mjs';
import {isActiveGM} from './native-context.mjs';
import {getDisruptPreyMeleeOptions,isCurrentDisruptToken} from './disrupt-prey-rules.mjs';
import {genericReactionAvailable,reactionEpoch,withReactionReservation} from './reaction-budget.mjs';

const values=c=>Array.from(c?.values?.()??c??[]),own=actor=>actor?.flags?.[MODULE_ID]?.disruptPrey??{events:[],reactions:[]};
const outcomes=['criticalFailure','failure','success','criticalSuccess'];
const fields=['eventId','nonce','actorUuid','tokenUuid','sourceActorUuid','sourceTokenUuid','sourceUserId','kind','phase'];
const authorId=message=>message?.author?.id??message?.user?.id??message?.user;
const speakerToken=message=>`Scene.${message?.speaker?.scene}.Token.${message?.speaker?.token}`;
const turnKey=game=>game.combat?.started?`${game.combat.id}:${game.combat.round}:${game.combat.turn}`:null;
const eventRecord=(actor,id)=>own(actor).events?.find(e=>e.eventId===id);
const reactionRecord=(actor,nonce)=>own(actor).reactions?.find(r=>r.nonce===nonce);
const validEvent=e=>fields.every(k=>typeof e?.[k]==='string'&&e[k].length>0&&e[k].length<=256)&&/^[A-Za-z0-9_-]{1,80}$/.test(e.nonce)&&({spell:'before-effects',stand:'after-action','drop-prone':'after-action',stride:'before-departure'}[e.kind]===e.phase);
const sameSource=(a,b)=>fields.filter(k=>k!=='nonce').every(k=>a[k]===b[k]);
// uncertain/choosing is not permission to continue a source action. The adapter
// must stop its continuation even though disrupted is not yet a proven result;
// an already confirmed critical hit must not be bypassed by uncertain damage.
const resultOf=(event,claim)=>({status:event?.status??'ineligible',disrupted:event?.status==='done'&&claim?.degree===3&&event.phase!=='after-action',eventId:event?.eventId,nonce:event?.nonce,...(claim?{degree:claim.degree,checkId:claim.checkId,damageMessageId:claim.damageMessageId,receiptId:claim.receiptId}:{})});
const recordResult=(actor,id)=>{const event=eventRecord(actor,id);return resultOf(event,reactionRecord(actor,event?.nonce));};

/** One proved triggering action and one normally chosen reaction. Native execution is injected. */
export function createDisruptPrey({game,reactionRestriction,fromUuid=globalThis.fromUuid,choose,validateSource=()=>false,performStrike,onError=()=>{}}={}){
 const report=error=>{try{onError(error)}catch{/* Reporting cannot repeat a mechanical operation. */}};
 const gm=()=>{if(!isActiveGM(game))throw Error('扰乱狩猎主GM已经改变。');};
 const owner=(actor,user)=>!!(user?.active&&actor.testUserPermission?.(user,'OWNER'));
 const chooseUser=actor=>{const users=values(game.users).filter(u=>owner(actor,u));return users.find(u=>!u.isGM&&(u.character?.uuid===actor.uuid||u.character?.id===actor.id))??users.find(u=>!u.isGM)??(owner(actor,game.users.activeGM)?game.users.activeGM:null);};
 async function save(actor,change){
  return withReactionReservation(actor,game,async()=>{
   gm();const state=structuredClone(own(actor));state.events??=[];state.reactions??=[];
   const changed=change(state);if(changed===false)return false;
   // No adapter has yet proved that an old source continuation has expired.
   // Keep every event/claim identity and terminal result. Only remove bulky dice
   // from completed older-epoch receipts; current/uncertain evidence stays intact.
   // A bounded archive needs a separate, explicit source-expiry contract.
   const epoch=reactionEpoch(actor,game),doneEvents=new Set(state.events.filter(e=>e.status==='done').map(e=>e.eventId));
   for(const claim of state.reactions)if(claim.state==='done'&&claim.epoch!==epoch&&doneEvents.has(claim.eventId)){delete claim.attackRoll;delete claim.damageRoll;}
   gm();await actor.update({[`flags.${MODULE_ID}.disruptPrey`]:state});gm();return true;
  });
 }
 async function resolve(event){
  const actor=await fromUuid?.(event.actorUuid),token=await fromUuid?.(event.tokenUuid),target=await fromUuid?.(event.sourceTokenUuid);gm();
  if(actor?.uuid!==event.actorUuid)return null;
  return {actor,token,target};
 }
 const currentContext=(context,event)=>!!(context&&isCurrentDisruptToken(context.token,game)&&isCurrentDisruptToken(context.target,game)&&context.token.actor.uuid===event.actorUuid&&context.target.actor.uuid===event.sourceActorUuid&&context.token.parent===context.target.parent);
 async function validSource(event,context,stage,choice=null){
  gm();const allowed=await validateSource(event,{...context,stage,choice});gm();return allowed===true;
 }
 async function finishEvent(actor,event,status,extra={}){
  await save(actor,state=>{const saved=state.events.find(e=>e.eventId===event.eventId);if(!saved||saved.status!=='choosing')return false;Object.assign(saved,{status,...extra});});return recordResult(actor,event.eventId);
 }
 function rollData(roll){
  if(!roll||roll._evaluated!==true||!Number.isFinite(roll.total)||typeof roll.toJSON!=='function')throw Error('缺少已评估的真实原生骰子。');
  const data=roll.toJSON();if(!data||typeof data!=='object'||data.evaluated!==true||!Number.isFinite(data.total))throw Error('原生骰子无法保存为已评估回执。');
  const serialized=JSON.stringify(data);if(serialized.length>100000)throw Error('原生骰子回执过大。');return JSON.parse(serialized);
 }
 function liveMessage(id){const message=typeof id==='string'?game.messages.get(id):null;if(!message?.id||message.id!==id)throw Error('没有本次真实聊天回执。');return message;}
 function attackProof(message,claim){
  const pf=message.flags?.pf2e,context=pf?.context,proof=message.flags?.[MODULE_ID]?.disruptPreyReaction;
  if(authorId(message)!==claim.userId||message.actor?.uuid!==claim.actorUuid||message.speaker?.actor!==claim.actorId||speakerToken(message)!==claim.tokenUuid||context?.type!=='attack-roll'||context.action!=='strike'||pf.origin?.actor!==claim.actorUuid||pf.origin.uuid!==claim.itemUuid||context.target?.token!==claim.targetUuid||context.target.actor!==claim.targetActorUuid||!context.options?.includes('action:reaction')||!context.options.includes('action:disrupt-prey')||proof?.nonce!==claim.nonce||proof.claimKey!==claim.claimKey||proof.actorUuid!==claim.actorUuid||proof.weaponKey&&proof.weaponKey!==claim.weaponKey)throw Error('扰乱狩猎攻击卡来源不符。');
  const degree=outcomes.indexOf(context.outcome),roll=message.rolls?.[0];if(degree<0||roll?.options?.degreeOfSuccess!==degree)throw Error('原生攻击成功度与骰子不一致。');
  const attackRoll=rollData(roll);
  if(claim.attackRoll&&(claim.degree!==degree||JSON.stringify(claim.attackRoll)!==JSON.stringify(attackRoll)))throw Error('已确认的攻击骰子发生变化。');
  return {degree,attackRoll};
 }
 function damageProof(message,claim){
  const pf=message.flags?.pf2e,c=pf?.context,options=c?.options??[];
  if(authorId(message)!==claim.userId||message.actor?.uuid!==claim.actorUuid||message.speaker?.actor!==claim.actorId||speakerToken(message)!==claim.tokenUuid||c?.type!=='damage-roll'||pf.origin?.actor!==claim.actorUuid||pf.origin.uuid!==claim.itemUuid||c.target?.token!==claim.targetUuid||c.target.actor!==claim.targetActorUuid||!options.includes(`${MODULE_ID}:disrupt-damage:${claim.nonce}`)||!options.includes(`${MODULE_ID}:bear-attack:${claim.checkId}`)||options.includes('action:reaction')||options.includes('trait:reaction'))throw Error('扰乱狩猎伤害卡来源不符。');
  const damageRoll=rollData(message.rolls?.[0]);
  if(claim.damageRoll&&JSON.stringify(claim.damageRoll)!==JSON.stringify(damageRoll))throw Error('已确认的伤害骰子发生变化。');
  return {damageRoll};
 }
 function appliedProof(message,claim){
  const context=message.flags?.pf2e?.context,options=context?.options??[],applied=message.flags?.pf2e?.appliedDamage,applicationUser=game.users.get(claim.applicationUserId);
  if(!applicationUser?.isGM||authorId(message)!==applicationUser.id||message.speaker?.actor!==claim.targetActorId||speakerToken(message)!==claim.targetUuid||context?.type!=='damage-taken'||!options.includes(`${MODULE_ID}:disrupt-apply:${claim.nonce}`)||!options.includes(`${MODULE_ID}:source:${claim.damageMessageId}:0`)||applied?.uuid&&applied.uuid!==claim.targetActorUuid||applied?.isReverted)throw Error('扰乱狩猎实际伤害回执不符。');
 }
 function callbacks(context,event,passedClaim){
  const {actor}=context,nonce=passedClaim.nonce;
  const change=async modify=>{
   await save(actor,state=>{const claim=state.reactions.find(r=>r.nonce===nonce),saved=state.events.find(e=>e.eventId===event.eventId);if(!claim||!saved||claim.eventId!==event.eventId)throw Error('扰乱狩猎认领已失效。');return modify(claim,saved);});
   const current=reactionRecord(actor,nonce);Object.assign(passedClaim,structuredClone(current));return structuredClone(current);
  };
  const onAttack=async({messageId}={})=>change((claim,saved)=>{
   const proof=attackProof(liveMessage(messageId),claim);
   if(claim.checkId){if(claim.checkId!==messageId)throw Error('本次反应已经绑定另一张攻击卡。');return false;}
   if(!['claimed','uncertain'].includes(claim.state))throw Error('扰乱狩猎攻击阶段错误。');
   Object.assign(claim,proof,{checkId:messageId,state:proof.degree<2?'done':'attack-rolled'});
   if(proof.degree<2)saved.status='done';saved.stage=claim.state;
  });
  const onDamage=async({messageId}={})=>change((claim,saved)=>{
   if(!claim.checkId||claim.degree<2)throw Error('没有命中的本次攻击，不能结算伤害。');
   attackProof(liveMessage(claim.checkId),claim);const proof=damageProof(liveMessage(messageId),claim);
   if(claim.damageMessageId){if(claim.damageMessageId!==messageId)throw Error('本次反应已经绑定另一张伤害卡。');return false;}
   if(!['attack-rolled','uncertain'].includes(claim.state))throw Error('扰乱狩猎伤害阶段错误。');
   Object.assign(claim,proof,{damageMessageId:messageId,state:'damage-rolled'});saved.stage=claim.state;
  });
  const onApplying=async()=>change((claim,saved)=>{
   if(claim.state!=='damage-rolled'||claim.degree<2||!claim.damageMessageId)throw Error('不能重复或提前应用扰乱狩猎伤害。');
   attackProof(liveMessage(claim.checkId),claim);damageProof(liveMessage(claim.damageMessageId),claim);
   claim.applicationUserId=game.user.id;claim.state='applying';saved.stage=claim.state;
  });
  const onApplied=async({messageId}={})=>change((claim,saved)=>{
   if(!claim.checkId||claim.degree<2||!claim.damageMessageId)throw Error('没有可应用的本次伤害。');
   appliedProof(liveMessage(messageId),claim);
   if(claim.receiptId){if(claim.receiptId!==messageId)throw Error('本次反应已经绑定另一张应用回执。');return false;}
   if(claim.state!=='applying'&&!(claim.state==='uncertain'&&claim.lastConfirmedState==='applying'))throw Error('伤害尚未进入原生应用阶段。');
   claim.receiptId=messageId;claim.state='done';saved.status='done';saved.stage='done';
  });
  return {onAttack,onDamage,onApplying,onApplied};
 }
 async function recover(context,event,handlers){
  const findUnique=predicate=>{const matches=values(game.messages).filter(message=>{try{return predicate(message)}catch{return false;}});return matches.length===1?matches[0]:null;};
  let claim=reactionRecord(context.actor,event.nonce);if(!claim||claim.state==='done')return;
  if(!claim.checkId){const message=findUnique(m=>!!attackProof(m,claim));if(message)await handlers.onAttack({messageId:message.id});}
  claim=reactionRecord(context.actor,event.nonce);if(claim.degree<2||!claim.checkId)return;
  if(!claim.damageMessageId){const message=findUnique(m=>!!damageProof(m,claim));if(message)await handlers.onDamage({messageId:message.id});}
  claim=reactionRecord(context.actor,event.nonce);
  if(claim.state==='applying'||claim.state==='uncertain'&&claim.lastConfirmedState==='applying'){
   const message=findUnique(m=>{appliedProof(m,claim);return true;});if(message)await handlers.onApplied({messageId:message.id});
  }
 }
 async function handleConfirmed(input){
  if(!isActiveGM(game))return {status:'not-authority',disrupted:false};
  if(!validEvent(input))return {status:'ineligible',disrupted:false};
  const event=Object.fromEntries(fields.map(k=>[k,input[k]]));let context,entered=false,handlers;
  try{
   context=await resolve(event);if(!context)return {status:'ineligible',disrupted:false};
   const {actor,token,target}=context,previous=eventRecord(actor,event.eventId);
    if(previous){
     if(!sameSource(previous,event))return {status:'ineligible',disrupted:false};
     const claim=reactionRecord(actor,previous.nonce);
     if(!['choosing','uncertain'].includes(previous.status)||!claim||claim.actorUuid!==actor.uuid||claim.eventId!==previous.eventId||!currentContext(context,previous))return recordResult(actor,event.eventId);
     // A repeated source may already have lost its native continuation. Recover
     // only existing native proof, using the persisted nonce; never choose, roll,
     // pay or apply again, nor invalidate payment when recovery itself fails.
     const recordedEvent=Object.fromEntries(fields.map(k=>[k,previous[k]]));
     try{await recover(context,recordedEvent,callbacks(context,recordedEvent,structuredClone(claim)));}catch(error){report(error);}
     return isActiveGM(game)?recordResult(actor,event.eventId):{status:'uncertain',disrupted:false,eventId:previous.eventId,nonce:previous.nonce};
    }
    if(!currentContext(context,event))return {status:'ineligible',disrupted:false};
   if(own(actor).events?.some(e=>e.nonce===event.nonce)||reactionRecord(actor,event.nonce))return {status:'ineligible',disrupted:false};
   if(!await validSource(event,context,'candidate'))return {status:'ineligible',disrupted:false};
   if(typeof performStrike!=='function')return {status:'ineligible',disrupted:false};
   const epoch=reactionEpoch(actor,game),turn=turnKey(game),user=chooseUser(actor),options=getDisruptPreyMeleeOptions({...context,game});
   let claimedChoice=false;
   await save(actor,state=>{
    if(state.events.some(e=>e.eventId===event.eventId||e.nonce===event.nonce)||state.reactions.some(r=>r.nonce===event.nonce))return false;
    claimedChoice=!!(epoch&&turn&&user&&options.length&&reactionEpoch(actor,game)===epoch&&turnKey(game)===turn&&owner(actor,user)&&genericReactionAvailable(actor,game,{reactionRestriction})&&getDisruptPreyMeleeOptions({...context,game}).length);
    state.events.push({...event,status:claimedChoice?'choosing':'ineligible',epoch,turn,userId:user?.id??null});
   });
   if(!claimedChoice)return recordResult(actor,event.eventId);
   const ownTurn=game.combat.turns[game.combat.turn]?.actor?.uuid===actor.uuid,choices=options.flatMap((option,index)=>(ownTurn?[0,1,2]:[0]).filter(map=>typeof option.strike.variants?.[map]?.roll==='function').map(map=>({value:`strike:${index}:${map}`,label:`${option.strike.item.name}${ownTurn?` · MAP ${map}`:''}`,key:option.key,map})));
   const selected=await choose?.({actor,user,title:'扰乱狩猎：选择近战打击或不使用',choices:[...choices.map(({value,label})=>({value,label})),{value:'decline',label:'不使用本次反应'}]});gm();
   if(selected==null||selected==='decline')return finishEvent(actor,event,'declined');
   const choice=choices.find(c=>c.value===selected);if(!choice)throw Error('无效的扰乱狩猎选择。');
   context=await resolve(event);
    const current=currentContext(context,event)&&getDisruptPreyMeleeOptions({...context,game}).find(o=>o.key===choice.key);
   if(!context||!current||!owner(actor,user)||reactionEpoch(actor,game)!==epoch||turnKey(game)!==turn||!await validSource(event,context,'chosen',choice))return finishEvent(actor,event,'ineligible');
   let claim;
   await withReactionReservation(actor,game,async()=>{
    gm();const state=structuredClone(own(actor)),saved=state.events.find(e=>e.eventId===event.eventId);
    if(saved?.status!=='choosing'||reactionEpoch(actor,game)!==epoch||turnKey(game)!==turn||!owner(actor,user)||!genericReactionAvailable(actor,game,{reactionRestriction})||!getDisruptPreyMeleeOptions({...context,game}).some(o=>o.key===choice.key))return;
    if(!await validSource(event,context,'claim',choice))return;
    if(reactionEpoch(actor,game)!==epoch||turnKey(game)!==turn||!owner(actor,user)||!genericReactionAvailable(actor,game,{reactionRestriction})||!getDisruptPreyMeleeOptions({...context,game}).some(o=>o.key===choice.key))return;
    claim={nonce:event.nonce,eventId:event.eventId,actorUuid:actor.uuid,actorId:actor.id,tokenUuid:token.uuid,targetUuid:target.uuid,targetTokenUuid:target.uuid,targetActorUuid:target.actor.uuid,targetActorId:target.actor.id,weaponKey:current.key,itemUuid:current.itemUuid,userId:user.id,epoch,claimKey:`disrupt:${event.nonce}`,slug:'disrupt-prey',cost:1,state:'claimed',checkId:null,map:choice.map};
    state.reactions??=[];state.reactions.push(claim);saved.stage='claimed';saved.weaponKey=current.key;saved.map=choice.map;
    gm();await actor.update({[`flags.${MODULE_ID}.disruptPrey`]:state});gm();
   });
   if(!claim)return finishEvent(actor,event,'ineligible');
   const passedClaim=structuredClone(claim);handlers=callbacks(context,event,passedClaim);
   if(!await validSource(event,context,'perform',choice))throw Error('已认领的反应来源在执行前发生变化。');
   const ready=getDisruptPreyMeleeOptions({...context,game}).find(o=>o.key===choice.key);
   if(!ready||!owner(actor,user)||reactionEpoch(actor,game)!==epoch||turnKey(game)!==turn)throw Error('已认领的反应来源在执行前发生变化。');
   gm();
   entered=true;await performStrike({...context,user,option:{...ready,map:choice.map},claim:passedClaim,...handlers});gm();
   if(eventRecord(actor,event.eventId)?.status!=='done')throw Error('原生反应未取得完整攻击及伤害回执。');
   return recordResult(actor,event.eventId);
  }catch(error){
   report(error);if(!context?.actor||!isActiveGM(game))return {status:'uncertain',disrupted:false,eventId:event.eventId,nonce:event.nonce};
   const {actor}=context;
   if(entered&&handlers)try{await recover(context,event,handlers);}catch(recoveryError){report(recoveryError);}
   if(eventRecord(actor,event.eventId)?.status==='done')return recordResult(actor,event.eventId);
   if(isActiveGM(game))try{await save(actor,state=>{
    const saved=state.events.find(e=>e.eventId===event.eventId),claim=state.reactions.find(r=>r.nonce===event.nonce&&r.eventId===event.eventId&&r.actorUuid===event.actorUuid);if(!saved)return false;
    saved.status=entered&&claim?'uncertain':'ineligible';saved.reason=String(error.message??error).slice(0,500);
    if(claim&&entered){claim.lastConfirmedState=claim.state;claim.state='uncertain';}
    // This local flag proves the injected executor was never invoked. Once
    // invoked, an exception (including RPC timeout) never proves non-execution.
    else if(claim){state.reactions=state.reactions.filter(r=>r!==claim);saved.stage='ineligible';}
   });}catch(writeError){report(writeError);}
   return recordResult(actor,event.eventId);
  }
 }
 return {handleConfirmed};
}
