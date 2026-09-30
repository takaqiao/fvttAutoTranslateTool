import {sameSalubriousPrivacy,treatmentPrivacyForPatient} from './salubrious-privacy.mjs';
import {MODULE_ID} from './rules.mjs';
import {SerialActions} from './runtime.mjs';
import {salubriousFeat,treatmentTiers,treatmentOutcome,treatmentImmunityData,TREAT_WOUNDS_IMMUNITY,kissState,values} from './salubrious-kiss-rules.mjs';
import {assertSource,assertPatient,currentToken,contextFor,claimOf,fail} from './salubrious-kiss-context.mjs';

/** Isolated draft subscriber. It never invokes Refocus or updates focus/time.
 * Native Workbench completion is the sole entry; optional decisions run outside
 * the shared actor resource queue. Every unknown operation stays non-retryable. */
export function createSalubriousKiss({game,fromUuid=globalThis.fromUuid,choose,executor,validateRefocusNote,runExclusive}={}){
 const queue=new SerialActions(),live=new Map(),run=runExclusive??((key,fn)=>queue.run(key,fn));
 const gm=()=>{if(game.user!==game.users.activeGM||!game.user?.isGM)throw fail('只有当前主GM能结算');};
 const write=async fn=>{gm();const result=await fn();gm();return result};
 const save=(actor,claim)=>run(actor.uuid,()=>write(()=>actor.update({[`flags.${MODULE_ID}.salubriousKiss.claims`]:[...(kissState(actor).claims??[]).filter(c=>c.nonce!==claim.nonce),{...structuredClone(claimOf(actor,claim.nonce)??{}),...structuredClone(claim)}]})));
 async function verifyEvent(actor,user,proof){
  gm();const token=await fromUuid(proof?.tokenUuid),item=salubriousFeat(actor);assertSource({game,actor,item,token,user,privacy:proof?.privacy});
  const keys=['nonce','actorUuid','itemUuid','userId','before','after','tokenUuid','startedAt'],intent=actor.flags?.[MODULE_ID]?.avRefocusIntent,receipt=actor.flags?.[MODULE_ID]?.refocusEvents?.find(r=>r.nonce===proof?.nonce);
  if(!proof||!sameSalubriousPrivacy(proof.privacy,intent?.privacy)||!sameSalubriousPrivacy(proof.privacy,receipt?.privacy)||keys.some(k=>proof[k]!==intent?.[k]||proof[k]!==receipt?.[k])||receipt.state!=='claimed'||proof.actorUuid!==actor.uuid||proof.userId!==user.id||!Number.isFinite(proof.before)||proof.after<proof.before||proof.after!==actor.system.resources.focus.value||!Number.isFinite(proof.startedAt)||game.time.worldTime<proof.startedAt||game.time.worldTime>=proof.startedAt+3600)throw fail('没有本次正常重新聚能的已提交回执');
  if(proof.privacy&&validateRefocusNote?.({actor,user,proof})!==true)throw fail('没有本次原生再聚能完成卡');
  return {token,item};
 }
 function onRefocus({actor,user,proof}){
  const key=actor?.uuid+':'+proof?.nonce;if(live.has(key))return live.get(key);
  const promise=(async()=>{
   gm();const previous=claimOf(actor,proof?.nonce);
   if(previous){if(['done','declined'].includes(previous.state))return structuredClone(previous);throw fail('本次重新聚能已有未确认医疗记录，不能重复');}
   const {token,item}=await verifyEvent(actor,user,proof);
   let claim={nonce:proof.nonce,actorUuid:actor.uuid,itemUuid:item.uuid,tokenUuid:token.uuid,userId:user.id,startedAt:proof.startedAt,state:'choosing',...proof.privacy?{privacy:structuredClone(proof.privacy),refocusNoteId:proof.noteId}:{},skill:'occultism',activityMinutes:10},target,executionStarted=false;
   await run(actor.uuid,async()=>{gm();if((kissState(actor).claims??[]).some(c=>!['done','declined'].includes(c.state)))throw fail('该角色仍有未确认的医疗');await write(()=>actor.update({[`flags.${MODULE_ID}.salubriousKiss.claims`]:[...(kissState(actor).claims??[]),structuredClone(claim)]}));});
   try{
    const candidates=values(token.parent.tokens).filter(t=>{if(t.hidden&&!user.isGM)return false;try{assertPatient({game,actor,token,target:t,user});return true}catch{return false}});
    if(!candidates.length)throw fail('没有可确认的合格患者');
    const selected=await choose({kind:'patient',actor,user,title:'仙露三吻：重新聚能时同时医疗',choices:[{value:'only-refocus',label:'仅重新聚能'},...candidates.map(t=>({value:t.uuid,label:t.name??t.actor.name??t.actor.id}))]});
    if(selected==null||selected==='only-refocus'){claim.state='declined';await save(actor,claim);return claim}
    target=candidates.find(t=>t.uuid===selected);if(!target)throw fail('患者选择不属于本次真实候选');
    await verifyEvent(actor,user,proof);assertPatient({game,actor,token,target,user});
    const tiers=treatmentTiers(actor),tier=tiers.length===1?String(tiers[0].tier):await choose({kind:'tier',actor,user,title:'仙露三吻：医疗DC',choices:tiers.map(t=>({value:String(t.tier),label:`DC ${t.dc}`}))});
    if(tier==null){claim.state='declined';await save(actor,claim);return claim}
    const selectedTier=tiers.find(t=>String(t.tier)===tier);if(!selectedTier)throw fail('非法神秘医疗DC');
    claim={...claim,targetUuid:target.uuid,targetActorUuid:target.actor.uuid,...selectedTier,...claim.privacy?{privacy:treatmentPrivacyForPatient({game,user,token,item,target,privacy:proof.privacy})}:{}};
    await run(target.actor.uuid,async()=>{await verifyEvent(actor,user,proof);assertPatient({game,actor,token,target,user});if(kissState(target.actor).pending)throw fail('该患者已有进行中或未确认的医疗');await write(()=>target.actor.update({[`flags.${MODULE_ID}.salubriousKiss.pending`]:{actorUuid:actor.uuid,nonce:claim.nonce}}));});
    claim.state='rolling';await save(actor,claim);
    executionStarted=true;
    const result=await executor.roll(claim);gm();await contextFor({game,fromUuid,claim});
    if(![0,1,2,3].includes(result?.degree)||!result.checkId||result.degree!==1&&!result.damageId||result.degree===1&&result.damageId)throw fail('原生医疗结果不完整');
    claim={...claim,result:{checkId:result.checkId,damageId:result.damageId},state:'applying'};await save(actor,claim);
    const reservation=kissState(target.actor).pending;if(reservation?.nonce!==claim.nonce||reservation.actorUuid!==actor.uuid)throw fail('患者保留记录已改变');
    const source=await fromUuid(TREAT_WOUNDS_IMMUNITY);await write(()=>target.actor.createEmbeddedDocuments('Item',[treatmentImmunityData(source.toObject(),claim,game.time.worldTime)]));
    if(result.degree!==1)claim.receipt=await executor.apply(claim,result);
    gm();if(treatmentOutcome({degree:result.degree,tier:claim.tier}).removeWounded)await write(()=>target.actor.decreaseCondition('wounded',{forceRemove:true}));
    claim.state='done';await save(actor,claim);await write(()=>target.actor.update({[`flags.${MODULE_ID}.salubriousKiss.pending`]:null}));return claim;
   }catch(error){
    // No remote/native call was started: a vanished candidate or canceled choice
    // is a known refusal. It must not strand this actor or a reserved patient.
    if(!executionStarted&&game.user===game.users.activeGM){
     if(target&&currentToken(target,game))await run(target.actor.uuid,async()=>{const pending=kissState(target.actor).pending;if(pending?.actorUuid===actor.uuid&&pending.nonce===claim.nonce)await write(()=>target.actor.update({[`flags.${MODULE_ID}.salubriousKiss.pending`]:null}));});
     claim.state='declined';claim.reason=claim.privacy?'known-pre-roll-refusal':String(error.message??error);await save(actor,claim);return claim;
    }
    claim.state='uncertain';claim.error=claim.privacy?'execution-uncertain':String(error.message??error);if(game.user===game.users.activeGM)await save(actor,claim);throw error;
   }
  })();live.set(key,promise);promise.finally(()=>{if(live.get(key)===promise)live.delete(key)}).catch(()=>{});return promise;
 }
 return {matchesActor:actor=>!!salubriousFeat(actor),onRefocus};
}
