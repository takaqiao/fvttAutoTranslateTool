import {sameSalubriousPrivacy,treatmentPrivacyForPatient,captureSalubriousPrivacy} from './salubrious-privacy.mjs';
import {MODULE_ID} from './rules.mjs';
import {SerialActions} from './runtime.mjs';
import {publicTargetName} from './native-context.mjs';
import {salubriousFeat,treatmentTiers,treatmentOutcome,treatmentImmunityData,TREAT_WOUNDS_IMMUNITY,kissState,values} from './salubrious-kiss-rules.mjs';
import {assertSource,assertPatient,assertClaimPrivacy,currentToken,contextFor,claimOf,fail} from './salubrious-kiss-context.mjs';

/** Isolated draft subscriber. It never invokes Refocus or updates focus/time.
 * Native Workbench completion is the sole entry; optional decisions run outside
 * the shared actor resource queue. Every unknown operation stays non-retryable. */
export function createSalubriousKiss({game,fromUuid=globalThis.fromUuid,choose,executor,validateRefocusNote,runExclusive,isExplorationContext=()=>false}={}){
 const queue=new SerialActions(),live=new Map(),activities=new Map(),activityResults=new Map(),run=runExclusive??((key,fn)=>queue.run(key,fn));
 const gm=()=>{if(game.user!==game.users.activeGM||!game.user?.isGM||!game.user.active||game.users.get(game.user.id)!==game.user)throw fail('只有当前主GM能结算');};
 const write=async(fn,validate)=>{gm();validate?.();const result=await fn();gm();validate?.();return result};
 const save=(actor,claim,validate)=>run(actor.uuid,()=>write(()=>actor.update({[`flags.${MODULE_ID}.salubriousKiss.claims`]:[...(kissState(actor).claims??[]).filter(c=>c.nonce!==claim.nonce),{...structuredClone(claimOf(actor,claim.nonce)??{}),...structuredClone(claim)}]}),validate));
 async function verifyEvent(actor,user,proof){
  gm();const token=await fromUuid(proof?.tokenUuid),item=salubriousFeat(actor);assertSource({game,actor,item,token,user,privacy:proof?.privacy});
  const keys=['nonce','actorUuid','itemUuid','userId','before','after','tokenUuid','startedAt'],intent=actor.flags?.[MODULE_ID]?.avRefocusIntent,receipt=actor.flags?.[MODULE_ID]?.refocusEvents?.find(r=>r.nonce===proof?.nonce);
  if(!proof||!sameSalubriousPrivacy(proof.privacy,intent?.privacy)||!sameSalubriousPrivacy(proof.privacy,receipt?.privacy)||keys.some(k=>proof[k]!==intent?.[k]||proof[k]!==receipt?.[k])||receipt.state!=='claimed'||proof.actorUuid!==actor.uuid||proof.userId!==user.id||!Number.isFinite(proof.before)||proof.after<proof.before||proof.after!==actor.system.resources.focus.value||!Number.isFinite(proof.startedAt)||game.time.worldTime<proof.startedAt||game.time.worldTime>=proof.startedAt+3600)throw fail('没有本次正常重新聚能的已提交回执');
  if(proof.privacy&&validateRefocusNote?.({actor,user,proof})!==true)throw fail('没有本次原生再聚能完成卡');
  return {token,item};
 }
 function onRefocus({actor,user,proof}){
  const activityBinding=activities.get(actor?.uuid);
  const binding=activityBinding?.activity.id===proof?.nonce?activityBinding:null;
  const explorationScope=binding?{activity:binding.activity,ctx:binding.ctx}:undefined;
  const validateBinding=()=>{gm();if(binding){const activity=binding.activity;if(!isExplorationContext(binding.ctx,activity.id)||proof.nonce!==activity.id||proof.actorUuid!==activity.actorUUID||proof.userId!==game.user.id||proof.startedAt!==activity.startedAt||game.time.worldTime!==activity.endsAt||game.combat?.started)throw fail('仙露活动没有本次开始／完成上下文');binding.ctx.validate();}};
  if(binding)validateBinding();
  const key=actor?.uuid+':'+proof?.nonce;if(live.has(key))return live.get(key);
  const promise=(async()=>{
   gm();const previous=claimOf(actor,proof?.nonce);
   if(previous){if(['done','declined'].includes(previous.state))return structuredClone(previous);throw fail('本次重新聚能已有未确认医疗记录，不能重复');}
   const {token,item}=await verifyEvent(actor,user,proof);validateBinding();const originalProof=structuredClone(proof);
   let claim={nonce:proof.nonce,actorUuid:actor.uuid,itemUuid:item.uuid,tokenUuid:token.uuid,userId:user.id,startedAt:proof.startedAt,state:'choosing',...proof.privacy?{privacy:structuredClone(proof.privacy),refocusNoteId:proof.noteId}:{},skill:'occultism',activityMinutes:10},target,targetActor,targetScene,targetUuid,targetActorUuid,executionStarted=false;
   const sourceKeys=['nonce','actorUuid','itemUuid','tokenUuid','userId','startedAt'];
   const originalActor=current=>current?.isToken?current.token?.actor===current&&currentToken(current.token,game):game.actors.get(current?.id)===current;
   const validateSource=()=>{validateBinding();assertSource({game,actor,item,token,user,privacy:originalProof.privacy});};
   const validateRecord=()=>{gm();const saved=claimOf(actor,claim.nonce);if(!originalActor(actor)||!saved||sourceKeys.some(key=>saved[key]!==claim[key]))throw fail('原施治者或医疗认领已改变');return saved;};
   const validateTarget=({states=[claim.state],pending=true,pendingOptional=false}={})=>{
    validateBinding();const saved=validateRecord();assertSource({game,actor,item,token,user,privacy:proof.privacy});
    const intent=actor.flags?.[MODULE_ID]?.avRefocusIntent,receipt=actor.flags?.[MODULE_ID]?.refocusEvents?.find(r=>r.nonce===originalProof.nonce),keys=['nonce','actorUuid','itemUuid','userId','before','after','tokenUuid','startedAt'];
    if(keys.some(key=>intent?.[key]!==originalProof[key]||receipt?.[key]!==originalProof[key])||receipt?.state!=='claimed'||!sameSalubriousPrivacy(originalProof.privacy,intent?.privacy)||!sameSalubriousPrivacy(originalProof.privacy,receipt?.privacy)||actor.system.resources.focus.value!==originalProof.after)throw fail('原重新聚能回执已改变');
    if(!states.includes(saved.state)||!currentToken(target,game)||target.parent!==targetScene||game.scenes.get(targetScene?.id)!==targetScene||target.uuid!==targetUuid||target.actor!==targetActor||targetActor.uuid!==targetActorUuid||saved.targetUuid!=null&&(saved.targetUuid!==targetUuid||saved.targetActorUuid!==targetActorUuid))throw fail('原患者文档或本次医疗阶段已改变');
    const reservation=kissState(targetActor).pending;
    if(pending&&!(pendingOptional&&reservation==null)&&(!reservation||reservation.nonce!==claim.nonce||reservation.actorUuid!==actor.uuid))throw fail('原患者保留记录已改变');
    assertClaimPrivacy({game,claim,token,item,target,user});
   };
   try{
    await run(actor.uuid,async()=>{validateSource();if((kissState(actor).claims??[]).some(c=>!['done','declined'].includes(c.state)))throw fail('该角色仍有未确认的医疗');await write(()=>actor.update({[`flags.${MODULE_ID}.salubriousKiss.claims`]:[...(kissState(actor).claims??[]),structuredClone(claim)]}),validateSource);});
    const candidates=values(token.parent.tokens).filter(t=>{if(t.hidden&&!user.isGM)return false;try{assertPatient({game,actor,token,target:t,user});return true}catch{return false}});
    const candidateActors=new Map(candidates.map(t=>[t,t.actor]));
    if(!candidates.length)throw fail('没有可确认的合格患者');
    const selected=binding?binding.patientTokenUuid:await choose({kind:'patient',actor,user,title:'仙露三吻：重新聚能时同时医疗',choices:[{value:'only-refocus',label:'仅重新聚能'},...candidates.map(t=>({value:t.uuid,label:publicTargetName(t,{game,user})}))]});
    if(selected==null||selected==='only-refocus'){claim.state='declined';await save(actor,claim,validateRecord);return claim}
    target=candidates.find(t=>t.uuid===selected);if(!target)throw fail('患者选择不属于本次真实候选');
    targetActor=candidateActors.get(target);targetScene=target.parent;targetUuid=target.uuid;targetActorUuid=targetActor.uuid;
    await verifyEvent(actor,user,proof);validateBinding();assertPatient({game,actor,token,target,user});
    const tiers=treatmentTiers(actor),tier=binding?String(binding.tier):tiers.length===1?String(tiers[0].tier):await choose({kind:'tier',actor,user,title:'仙露三吻：医疗DC',choices:tiers.map(t=>({value:String(t.tier),label:`DC ${t.dc}`}))});
    if(tier==null){claim.state='declined';await save(actor,claim,validateRecord);return claim}
    const selectedTier=tiers.find(t=>String(t.tier)===tier);if(!selectedTier)throw fail('非法神秘医疗DC');
    claim={...claim,targetUuid,targetActorUuid,...selectedTier,...claim.privacy?{privacy:treatmentPrivacyForPatient({game,user,token,item,target,privacy:proof.privacy})}:{}};
    await run(targetActor.uuid,async()=>{await verifyEvent(actor,user,proof);validateTarget({pending:false});assertPatient({game,actor,token,target,user});if(kissState(targetActor).pending)throw fail('该患者已有进行中或未确认的医疗');await write(()=>targetActor.update({[`flags.${MODULE_ID}.salubriousKiss.pending`]:{actorUuid:actor.uuid,nonce:claim.nonce}}),()=>validateTarget({pendingOptional:true}));});
    claim.state='rolling';await save(actor,claim,()=>validateTarget({states:['choosing','rolling']}));
    executionStarted=true;
    const result=await executor.roll(claim,explorationScope);
    if(![0,1,2,3].includes(result?.degree)||!result.checkId||result.degree!==1&&!result.damageId||result.degree===1&&result.damageId)throw fail('原生医疗结果不完整');
    claim={...claim,result:{checkId:result.checkId,damageId:result.damageId,degree:result.degree},state:'applying'};validateTarget({states:['rolling']});await contextFor({game,fromUuid,claim});validateTarget({states:['rolling']});await save(actor,claim,()=>validateTarget({states:['rolling','applying']}));
    const source=await fromUuid(TREAT_WOUNDS_IMMUNITY);validateTarget();const immunity=treatmentImmunityData(source.toObject(),claim,game.time.worldTime);validateTarget();
    await write(async()=>{const immunities=await targetActor.createEmbeddedDocuments('Item',[immunity]);claim.immunityIds=immunities.map(i=>i.uuid);return immunities;},()=>validateTarget());
    if(result.degree!==1){validateTarget();claim.receipt=await executor.apply(claim,result,explorationScope);validateTarget();}
    validateTarget();if(treatmentOutcome({degree:result.degree,tier:claim.tier}).removeWounded)await write(()=>targetActor.decreaseCondition('wounded',{forceRemove:true}),()=>validateTarget());
    claim.state='done';await save(actor,claim,()=>validateTarget({states:['applying','done']}));await write(()=>{validateTarget();return targetActor.update({[`flags.${MODULE_ID}.salubriousKiss.pending`]:null});},()=>validateTarget({pendingOptional:true}));return claim;
   }catch(error){
    // No remote/native call was started: a vanished candidate or canceled choice
    // is a known refusal. It must not strand this actor or a reserved patient.
    if(!executionStarted&&game.user===game.users.activeGM){
     if(targetActor&&originalActor(targetActor))await run(targetActor.uuid,async()=>{const pending=kissState(targetActor).pending;if(pending?.actorUuid===actor.uuid&&pending.nonce===claim.nonce)await write(()=>targetActor.update({[`flags.${MODULE_ID}.salubriousKiss.pending`]:null}),()=>{gm();if(!originalActor(targetActor))throw fail('原患者已不是当前文档');});});
     claim.state='declined';claim.reason=claim.privacy?'known-pre-roll-refusal':String(error.message??error);await save(actor,claim,validateRecord);return claim;
    }
    if(error.salubriousReceipt)claim.receipt=error.salubriousReceipt;
    claim.state='uncertain';claim.error=claim.privacy?'execution-uncertain':String(error.message??error);if(game.user===game.users.activeGM)await save(actor,claim,validateRecord);throw error;
   }
  })();
  if(binding)promise.then(claim=>{
   const result={status:claim.state==='done'?'confirmed':'blocked',proof:{useId:claim.nonce,checkIds:claim.result?.checkId?[claim.result.checkId]:[],resultIds:claim.result?.damageId?[claim.result.damageId]:[],receiptIds:claim.receipt?.messageId?[claim.receipt.messageId]:[],immunityIds:claim.immunityIds??[],poolReceipts:claim.receipt?.poolReceipt?[claim.receipt.poolReceipt]:[]},sourceDegree:claim.result?.degree,effectiveOutcome:['criticalFailure','failure','success','criticalSuccess'][claim.result?.degree],rolledHealing:claim.result?.degree>=2?game.messages.get(claim.result.damageId)?.rolls?.[0]?.total??null:null};
   activityResults.set(binding.activity.id,result);binding.resolve(result);activities.delete(actor.uuid);
  },error=>{binding.reject(error);activities.delete(actor.uuid)});
  live.set(key,promise);promise.finally(()=>{if(live.get(key)===promise)live.delete(key)}).catch(()=>{});return promise;
 }
 async function claimActivity(activity,ctx){
  gm();if(!isExplorationContext(ctx,activity.id))throw fail('缺少私有活动范围');ctx.validate();
  if(activities.has(activity.actorUUID))return {status:'blocked',reason:'three-pecks-already-reserved'};
  const actor=await fromUuid(activity.actorUUID),patient=await fromUuid(activity.patientUUIDs[0]);ctx.validate();
  if(activity.patientUUIDs.length!==1)return {status:'blocked',reason:'three-pecks-group-not-adapted'};
  let tiers;try{tiers=treatmentTiers(actor)}catch(error){return {status:'blocked',reason:error.message}}
  const scene=game.scenes.active??globalThis.canvas?.scene;
  const token=values(scene?.tokens).find(t=>t.actor===actor),target=values(scene?.tokens).find(t=>t.actor===patient);
  if(!token||!target)return {status:'blocked',reason:'three-pecks-current-tokens-required'};
  const item=salubriousFeat(actor);try{const privacy=captureSalubriousPrivacy({game,user:game.user,token,item,requestedMode:game.settings?.get('core','messageMode')??'public'});assertSource({game,actor,item,token,user:game.user,privacy});assertPatient({game,actor,token,target})}catch(error){return {status:'blocked',reason:error.message}}
  const tier=({trained:1,expert:2,master:3,legendary:4}[activity.options.rank]??1);if(!tiers.some(t=>t.tier===tier))return {status:'blocked',reason:'three-pecks-dc-unqualified'};
  let resolve,reject;const promise=new Promise((r,j)=>{resolve=r;reject=j});promise.catch(()=>{});
  activities.set(actor.uuid,{activity:structuredClone(activity),ctx,patientTokenUuid:target.uuid,tier,promise,resolve,reject});return {status:'started'};
 }
 async function completeActivity(activity,ctx){
  if(!isExplorationContext(ctx,activity.id))throw fail('缺少私有完成范围');ctx.validate();
  const result=activityResults.get(activity.id);if(result)return structuredClone(result);
  const binding=activities.get(activity.actorUUID);if(binding?.activity.id!==activity.id)throw fail('没有此仙露预留');return binding.promise;
 }
 function cancelActivity(activity){const binding=activities.get(activity.actorUUID);if(binding?.activity.id!==activity.id)return;activities.delete(activity.actorUUID);binding.reject(fail('探索活动已停止，尚未执行仙露治疗'));}
 return {matchesActor:actor=>!!salubriousFeat(actor),onRefocus,claimActivity,completeActivity,cancelActivity,getActivityResult:id=>structuredClone(activityResults.get(id)??null)};
}
