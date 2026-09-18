import {assessRoaringCast,validateRoaringTarget,roaringOwnTurn,roaringTurnPriority,ROARING_APPLAUSE_SOURCE} from './roaring-applause-rules.mjs';
import {createRoaringSource,reduceRoaringSource,projectRoaringConditions} from './roaring-lifecycle.mjs';
import {createRoaringSaveEvidence} from './roaring-save-evidence.mjs';
import {isActiveGM,getSourceId} from './native-context.mjs';
import {SerialActions} from './runtime.mjs';

const ID='pf2e-third-party-automation',KIND='roaring-applause';
const PROOF='roaring-applause:proof',COMPLETE='roaring-applause:complete',NOTICE='roaring-applause:notice';
const CONDITIONS={slowed:'Compendium.pf2e.conditionitems.Item.xYTAsEpcJE1Ccni3',fascinated:'Compendium.pf2e.conditionitems.Item.AdPVz7rbaVSRxHFg'};
const values=c=>Array.from(c?.values?.()??c??[]),copy=v=>structuredClone(v);
const canonical=v=>JSON.stringify(v,(_k,x)=>x&&typeof x==='object'&&!Array.isArray(x)?Object.fromEntries(Object.keys(x).sort().map(k=>[k,x[k]])):x);
const same=(a,b)=>canonical(a)===canonical(b),bounded=s=>typeof s==='string'&&/^[A-Za-z0-9-]{1,80}$/.test(s);
const author=m=>m?.author?.id??m?.user?.id??m?.user;
const demand=(value,message)=>{if(!value)throw Error(`Roaring Applause: ${message}`)};
async function hash(v){const b=await crypto.subtle.digest('SHA-256',new TextEncoder().encode(canonical(v)));return [...new Uint8Array(b)].map(n=>n.toString(16).padStart(2,'0')).join('');}
const shape=item=>canonical({source:getSourceId(item),system:item.system});
const transport=o=>({status:o.status,castNonce:o.castNonce,input:copy(o.input),receipt:copy(o.receipt),messageUuid:o.message?.uuid??null,completedWorldTime:o.completedWorldTime});

async function chooseFacts(){
 return globalThis.foundry.applications.api.DialogV2.wait({window:{title:'轰然喝彩：确认本次目标'},content:'<p>确认具有法术效线，并选择目标理解施法者的方式。不同高度、私密结果或不确定规则请取消并手工处理。</p>',buttons:[
  ...[['sees','目标能看见施法者，且有法术效线'],['hears','目标能听见施法者，且有法术效线'],['understands','目标能以其他方式理解施法者，且有法术效线']].map(([perception,label])=>({action:perception,label,callback:()=>({perception,lineOfEffectConfirmed:true})})),
  {action:'cancel',label:'取消',type:'button',callback:()=>false}],rejectClose:false});
}

/** Current public rank-3 source coordinator. No matcher, extra payment, native
 * save, Sustain completion, reaction-resource mutation or reconstructed roll. */
export function createRoaringApplause({game,fromUuid=globalThis.fromUuid,nativeCasts,effects,choose=chooseFacts,saveEvidence,onError=()=>{},onManual:manualNotice=()=>{},onClap=()=>globalThis.ui?.notifications?.info?.('轰然喝彩：目标起回合需要鼓掌（操控）。请GM核对可触发的反应。'),randomId=()=>globalThis.crypto.randomUUID()}={}){
 demand(nativeCasts&&effects,'native Cast and owned effect store required');
 const scopes=new Map(),byItem=new Map(),queue=new SerialActions(),hooks=[],reported=new Set(),resume=new Set(),lastStarts=new Map();let socket,Hooks,installed=false,electedGM=null,authorityGeneration=0;
 const evidence=saveEvidence??createRoaringSaveEvidence({game,fromUuid,lookupSource,onVerified,onManual,onError,randomId});
 const allActors=()=>[...new Map([...values(game.actors),...values(game.scenes).flatMap(s=>values(s.tokens).map(t=>t.actor))].filter(Boolean).map(a=>[a.uuid,a])).values()];
 const liveActor=uuid=>allActors().find(a=>a.uuid===uuid);
 const records=()=>allActors().flatMap(actor=>(effects.list(actor)??[]).map(record=>({actor,record})));
 const find=nonce=>{const found=records().filter(x=>x.record.state.sourceNonce===nonce);return found.length===1?found[0]:null;};
 function ownContext(actor){
  const tokens=values(actor.getActiveTokens?.(false,true)).map(t=>t.document??t),token=tokens[0],sceneId=game.scenes?.current?.id??globalThis.canvas?.scene?.id;
  demand(tokens.length===1&&token?.parent?.id===sceneId,'需要场景中唯一的实际施法者Token。');
  return {token,targets:values(game.user.targets).map(t=>t.document??t)};
 }
 function immunity(actor,item){
  demand(game.pf2e?.settings?.iwr===true&&typeof actor.isImmuneTo==='function','免疫自动化关闭或未知，请手工核对。');
  const conditions=game.pf2e.ConditionManager?.conditions,slowed=conditions?.get('slowed'),fascinated=conditions?.get('fascinated');
  demand(slowed?.type==='condition'&&slowed.system?.slug==='slowed'&&fascinated?.type==='condition'&&fascinated.system?.slug==='fascinated','缺少准确原生条件来源。');
  demand([[slowed,'slowed'],[fascinated,'fascinated']].every(([item,role])=>item.uuid===CONDITIONS[role]||getSourceId(item)===CONDITIONS[role]),'原生条件来源已改变，不能推断免疫。');
  const assessment={spell:actor.isImmuneTo(item),slowed:actor.isImmuneTo(slowed),fascinated:actor.isImmuneTo(fascinated),checked:true,systemVersion:'8.5.1'};
  demand(['spell','slowed','fascinated'].every(k=>typeof assessment[k]==='boolean'),'免疫结果无法证明。');return assessment;
 }
 function liveScope(s,{beforePayment=false}={}){
  demand(installed&&s.user===game.user&&s.user.active&&game.users.activeGM?.id===s.gmId&&game.actors.get(s.actor.id)===s.actor&&s.actor.items.get(s.item.id)===s.item&&s.actor.items.get(s.entry.id)===s.entry&&s.actor.testUserPermission(s.user,'OWNER')===true&&shape(s.item)===s.itemShape,'原客户端来源、所有者或主GM已改变。');
  if(beforePayment){const a=assessRoaringCast({...s,item:s.castItem,user:s.user});demand(a.eligible,a.reason??'施法入口已改变。');}
  const c=ownContext(s.actor);demand(c.token===s.token,'原施法者Token已改变。');
  if(beforePayment)demand(same(c.targets.map(t=>t.uuid),s.targets.map(t=>t.uuid)),'付款前原单目标选择已改变。');
  validateRoaringTarget({...s,...s.facts});demand(same(roaringOwnTurn(s),s.turn),'准确施法回合已改变。');
 }
 const identity=s=>({sourceNonce:s.sourceNonce,actorUuid:s.actor.uuid,itemUuid:s.item.uuid,entryUuid:s.entry.uuid,sourceTokenUuid:s.token.uuid,targetUuid:s.targets[0].uuid,targetActorUuid:s.targets[0].actor.uuid,userId:s.user.id,gmId:s.gmId,rank:3,dc:s.dc,turn:s.turn,facts:s.facts,itemShape:s.itemShape});
 async function localProof(payload,sender){
  const s=scopes.get(payload?.sourceNonce);demand(s&&sender===s.gmId&&sender===game.users.activeGM?.id&&s.fingerprint===payload.fingerprint&&['casting','completed'].includes(payload.phase)&&s.stage===payload.phase,'缺少准确原客户端施法scope。');
  liveScope(s,{beforePayment:payload.phase==='casting'});demand(await hash(identity(s))===s.fingerprint,'施法scope内容已改变。');
  return {identity:copy(identity(s)),fingerprint:s.fingerprint,phase:s.stage,...s.stage==='completed'?{outcome:transport(s.outcome)}:{}};
 }
 async function askProof(data,userId,phase){
  demand(installed&&isActiveGM(game)&&bounded(data?.sourceNonce)&&/^[a-f0-9]{64}$/.test(data.fingerprint??''),'主GM或施法证明参数无效。');
  const payload={sourceNonce:data.sourceNonce,fingerprint:data.fingerprint,phase};
  const response=userId===game.user.id?{ok:true,value:await localProof(payload,userId)}:await socket.executeAsUser(PROOF,userId,payload);
  demand(isActiveGM(game)&&response?.ok&&response.value?.identity?.gmId===game.user.id&&response.value.identity.userId===userId&&response.value.phase===phase&&response.value.fingerprint===data.fingerprint&&await hash(response.value.identity)===data.fingerprint,'原客户端真实调用没有确认。');
  return response.value;
 }
 async function resolveIdentity(i,{beforePayment=false}={}){
  const actor=await fromUuid(i.actorUuid),item=await fromUuid(i.itemUuid),entry=await fromUuid(i.entryUuid),token=await fromUuid(i.sourceTokenUuid),target=await fromUuid(i.targetUuid),user=game.users.get(i.userId);
  demand(isActiveGM(game)&&game.user.id===i.gmId&&user?.active&&actor?.items?.get(item?.id)===item&&actor.items.get(entry?.id)===entry&&actor.testUserPermission(user,'OWNER')===true&&item.actor===actor&&entry.actor===actor&&getSourceId(item)===ROARING_APPLAUSE_SOURCE&&shape(item)===i.itemShape,'原生来源或权限不再匹配。');
  if(beforePayment){const a=assessRoaringCast({game,actor,item,entry,user,options:{rank:3,messageMode:'public'}});demand(a.eligible,a.reason??'原生施法参数无效。');}
  validateRoaringTarget({game,actor,token,targets:[target],...i.facts});demand(target.actor.uuid===i.targetActorUuid&&same(roaringOwnTurn({game,actor,token}),i.turn),'原目标或准确施法回合不再匹配。');
  const dc=entry.statistic?.withRollOptions?.({item})?.dc?.value??entry.statistic?.dc?.value;demand(dc===i.dc,'原施法DC已经改变。');
  return {actor,item,entry,token,target,user};
 }
 function noOtherSource(target,nonce){demand((effects.list(target.actor)??[]).every(r=>r.state.sourceNonce===nonce||r.state.status==='ended'),'同一目标已有受管来源，第二来源请手工核对叠加。');}
 async function validateInvocation(context){
  const {invocation,user,payload,castNonce}=context,data=invocation?.data;
  demand(invocation?.kind===KIND&&data.captureCompletionTime===true&&data.messageMode==='public','本次调用没有完成时间/公开范围。');
  const proof=await askProof(data,user.id,'casting'),i=proof.identity,resolved=await resolveIdentity(i,{beforePayment:true});
  demand(data.sourceTokenUuid===i.sourceTokenUuid&&payload.itemUuid===i.itemUuid&&payload.entryUuid===i.entryUuid&&payload.actorUuid===i.actorUuid&&payload.rank===3&&payload.slotId===null&&context.actor===resolved.actor,'施法原生调用与scope不符。');
  demand(!(resolved.actor.flags?.[ID]?.nativeCasts??[]).some(r=>r.invocation?.kind===KIND&&r.invocation.data?.sourceNonce===i.sourceNonce&&r.id!==castNonce),'同一来源已有原生支付尝试，禁止重付。');
  noOtherSource(resolved.target,i.sourceNonce);immunity(resolved.target.actor,resolved.item);return true;
 }
 async function consumePolicy(context,next){context.expectSlotCommit({before:context.entry.system.slots.slot3.value,cost:1,changes:()=>({})});return next();}
 function paidBinding(i,outcome,{actor,item,entry},message){
  demand(outcome?.status==='completed'&&bounded(outcome.castNonce)&&message?.id&&game.messages.get(message.id)===message&&author(message)===i.userId&&message.rolls?.length===0&&message.blind===false&&Array.isArray(message.whisper)&&message.whisper.length===0,'缺少准确公开原卡。');
  const list=actor.flags?.[ID]?.nativeCasts??[],matches=list.filter(p=>p.id===outcome.castNonce),p=matches[0];
  demand(matches.length===1&&p.state==='used'&&p.messageId===message.id&&p.userId===i.userId&&p.invocation?.kind===KIND&&p.invocation.gmId===i.gmId&&p.invocation.data.sourceNonce===i.sourceNonce&&same(p,outcome.receipt),'原生已付回执不符。');
  demand(list.filter(p=>p.invocation?.kind===KIND&&p.invocation.data.sourceNonce===i.sourceNonce).length===1,'来源存在重复支付证据。');
  const expected={actorUuid:actor.uuid,itemUuid:item.uuid,entryUuid:entry.uuid,sourceId:ROARING_APPLAUSE_SOURCE,rank:3,slotId:null,focusPoints:0,overlayIds:[],messageMode:'public'};
  for(const value of [p,outcome.input,message.flags?.[ID]?.nativeCastInput])demand(value&&Object.entries(expected).every(([k,v])=>same(value[k],v)),'原生卡、来源或环级不符。');
  const slot=p.slotCommit;
  demand(slot?.castNonce===p.id&&slot.itemUuid===item.uuid&&slot.entryUuid===entry.uuid&&slot.rank===3&&slot.cost===1&&Number.isInteger(slot.before)&&slot.before>0&&slot.after===slot.before-1&&slot.userId===i.userId&&slot.gmId===i.gmId,'没有准确一次法术位支付证据。');
  demand(Number.isFinite(p.completedWorldTime)&&p.completedWorldTime===outcome.completedWorldTime&&p.completedWorldTime<=game.time.worldTime&&p.invocation.data.captureCompletionTime===true&&p.nativeCastScope?.castNonce===p.id&&p.nativeCastScope.tokenUuid===i.sourceTokenUuid,'缺少原施法完成时间或Token证明。');
  const marker=message.flags?.[ID]?.nativeCast,origin=message.flags?.pf2e?.origin,parts=i.sourceTokenUuid.split('.');
  demand(marker?.id===p.id&&marker.itemUuid===i.itemUuid&&marker.actorUuid===i.actorUuid&&marker.userId===i.userId&&origin?.uuid===i.itemUuid&&origin.actor===i.actorUuid&&origin.castRank===3&&message.speaker?.actor===actor.id&&message.speaker.scene===parts[1]&&message.speaker.token===parts[3]&&same(message.flags?.['pf2e-toolbelt']?.targetHelper?.targets,[i.targetUuid]),'原卡来源、speaker或单目标不符。');
  return p;
 }
 async function complete(payload,sender){
  const proof=await askProof(payload,sender,'completed');demand(same(payload.outcome,proof.outcome),'客户端结果不是原native outcome。');
  const i=proof.identity,resolved=await resolveIdentity(i),message=await fromUuid(proof.outcome.messageUuid),p=paidBinding(i,proof.outcome,resolved,message);
  return queue.run(resolved.target.actor.uuid,async()=>{
   demand(isActiveGM(game)&&game.user.id===i.gmId,'主GM已改变。');demand(same(roaringOwnTurn({game,actor:resolved.actor,token:resolved.token}),i.turn),'排队期间准确施法回合已改变。');noOtherSource(resolved.target,i.sourceNonce);
   const assessment=immunity(resolved.target.actor,resolved.item),state=createRoaringSource({sourceNonce:i.sourceNonce,castNonce:p.id,sourceId:ROARING_APPLAUSE_SOURCE,itemUuid:i.itemUuid,entryUuid:i.entryUuid,casterActorUuid:i.actorUuid,casterTokenUuid:i.sourceTokenUuid,targetActorUuid:i.targetActorUuid,targetTokenUuid:i.targetUuid,originalMessageUuid:message.uuid,rank:3,completedWorldTime:p.completedWorldTime,turn:i.turn,finiteEnvelope:{start:{value:p.completedWorldTime,initiative:i.turn.order.find(c=>c.id===i.turn.combatantId).initiative},duration:{value:1,unit:'rounds',expiry:'turn-end',sustained:false}}});
   const context={userId:i.userId,gmId:i.gmId,dc:i.dc,paymentId:p.id,immunity:assessment};
   const prior=effects.get(resolved.target.actor,i.sourceNonce);
   demand(!prior||same(prior.state,state)&&same(prior.context,context),'已存在不同来源状态，不能重建。');
   const record=prior??await effects.claim({actor:resolved.target.actor,state,context});
   demand(isActiveGM(game)&&game.user.id===i.gmId&&record&&same(record.state,state)&&same(record.context,context)&&same(effects.get(resolved.target.actor,i.sourceNonce),record),'来源认领未持久保存。');
   const marker={schema:1,sourceNonce:i.sourceNonce,castNonce:p.id,targetUuid:i.targetUuid,actorUuid:i.targetActorUuid};
   demand(!message.flags?.[ID]?.roaringSource||same(message.flags[ID].roaringSource,marker),'原卡已有冲突来源标记。');
   const returned=await message.update({[`flags.${ID}.roaringSource`]:marker});
   demand(isActiveGM(game)&&game.user.id===i.gmId&&returned===message&&game.messages.get(message.id)===message&&same(message.flags?.[ID]?.roaringSource,marker),'原卡来源标记未持久绑定；不会重施法。');
   await socket.executeForEveryone(NOTICE,{messageUuid:message.uuid});return {sourceNonce:i.sourceNonce,messageUuid:message.uuid};
  }).catch(async error=>{
   // The paid Cast is never retried. If a source was already persisted before a
   // later card/transport failure, keep that exact record visibly unverified.
   if(isActiveGM(game)&&game.user.id===i.gmId&&effects.get(resolved.target.actor,i.sourceNonce)){
    try{await applyLifecycleEvent({actor:resolved.target.actor,nonce:i.sourceNonce,event:{type:'save-unverified',receiptId:randomId(),reason:'source-binding-uncertain'}});}catch(secondary){onError(secondary);}
   }
   throw error;
  });
 }
 async function interceptCast({item:castItem,entry,options={}},next){
  const actor=castItem.actor,user=game.user,a=assessRoaringCast({game,actor,item:castItem,entry,user,options});if(!a.handled)return next();demand(a.eligible,a.reason);
  const item=a.base;demand(!byItem.has(item.uuid),'这个法术已有正在进行的施法。');
  const s={game,actor,item,castItem,entry,options,user,gmId:game.users.activeGM.id,sourceNonce:randomId(),stage:'choosing',itemShape:shape(item),...ownContext(actor)};
  demand(bounded(s.sourceNonce),'来源nonce无效。');s.turn=roaringOwnTurn(s);s.dc=entry.statistic?.withRollOptions?.({item:castItem})?.dc?.value??entry.statistic?.dc?.value;demand(Number.isFinite(s.dc),'原生施法DC未知。');
  scopes.set(s.sourceNonce,s);byItem.set(item.uuid,s);
  try{
   const facts=await choose(s);if(!facts)return; s.facts={perception:facts.perception,lineOfEffectConfirmed:facts.lineOfEffectConfirmed};
   liveScope(s,{beforePayment:true});s.fingerprint=await hash(identity(s));s.stage='casting';
   demand(typeof next.withOutcome==='function','准确原生施法接口不可用。');
   s.outcome=await next.withOutcome({kind:KIND,data:{sourceNonce:s.sourceNonce,fingerprint:s.fingerprint,sourceTokenUuid:s.token.uuid,messageMode:'public',captureCompletionTime:true}});
   if(s.outcome?.status==='disrupted'){s.stage='disrupted';return s.outcome.nativeResult;}
   demand(s.outcome?.status==='completed','本次原生施法结果未确认。');s.stage='completed';liveScope(s);
   const payload={sourceNonce:s.sourceNonce,fingerprint:s.fingerprint,outcome:transport(s.outcome)};
   const response=isActiveGM(game)?{ok:true,value:await complete(payload,user.id)}:await socket.executeAsUser(COMPLETE,s.gmId,payload);
   demand(response?.ok,'来源建立未确认；原卡和原生付款保持，不重试。');s.stage='registered';return s.outcome.nativeResult;
  }finally{scopes.delete(s.sourceNonce);if(byItem.get(item.uuid)===s)byItem.delete(item.uuid);}
 }
 function lookupSource(message){
  if(!installed||game.messages.get(message?.id)!==message)return null;const m=message.flags?.[ID]?.roaringSource;if(m?.schema!==1||!bounded(m.sourceNonce))return null;
  const actor=liveActor(m.actorUuid),r=actor&&effects.get(actor,m.sourceNonce),s=r?.state,scope=s?.source;
  if(!s||!['awaiting-save','active'].includes(s.status)||r.context.gmId!==game.users.activeGM?.id||scope.originalMessageUuid!==message.uuid||scope.castNonce!==m.castNonce||scope.targetTokenUuid!==m.targetUuid||scope.targetActorUuid!==actor.uuid||message.flags?.[ID]?.nativeCast?.id!==scope.castNonce||message.flags.pf2e?.origin?.uuid!==scope.itemUuid||author(message)!==r.context.userId)return null;
  return {sourceNonce:s.sourceNonce,castNonce:scope.castNonce,originalMessageUuid:message.uuid,casterActorUuid:scope.casterActorUuid,casterTokenUuid:scope.casterTokenUuid,entryUuid:scope.entryUuid,itemUuid:scope.itemUuid,targetUuid:scope.targetTokenUuid,targetActorUuid:scope.targetActorUuid,rank:3,dc:r.context.dc,status:s.status,gmId:r.context.gmId};
 }
 function observation(state){
  const d=state.timing.deadline,combat=game.combats?.get(d.combatId),turns=values(combat?.turns),c=turns.find(c=>c.id===d.combatantId),actor=liveActor(d.actorUuid),token=values(game.scenes).flatMap(s=>values(s.tokens)).find(t=>t.uuid===d.tokenUuid);
  const worldTime=game.time.worldTime;
  if(!combat?.started||!c||!actor||c.actor!==actor||c.token!==token||token?.actor!==actor||!Number.isInteger(combat.round)||combat.round<1||!Number.isInteger(combat.turn)||!turns[combat.turn])return {worldTime,turn:null};
  const matching=values(game.combats).filter(c=>c.started&&c.scene?.id===token.parent?.id).flatMap(c=>values(c.turns).filter(t=>t.token?.uuid===token.uuid));
  if(matching.length!==1||matching[0]!==c)return {worldTime,turn:null};
  const ended=turns.map(c=>c.flags?.pf2e?.roundOfLastTurnEnd).filter(Number.isInteger);
  return {worldTime,turn:{combatId:combat.id,combatantId:c.id,actorUuid:actor.uuid,tokenUuid:token.uuid,started:true,round:combat.round,turn:combat.turn,order:turns.map(c=>({id:c.id,initiative:Number.isFinite(c.initiative)?c.initiative:null,overridePriority:roaringTurnPriority(c)})),lastTurnEnd:c.flags?.pf2e?.roundOfLastTurnEnd??null,latestTurnEndRound:ended.length?Math.max(...ended):null}};
 }
 /** Synchronous read at a reaction's precommit boundary. The accepted source
  * owns payment; this query neither reconstructs Cast nor changes resources. */
 function reactionRestriction(actor){
  const entry=(sourceNonce,status,reason)=>({sourceNonce,status,reason});
  const unresolved=reason=>({status:'manual',sources:[entry(null,'manual',reason)]});
  if(!installed||game.world?.id!=='ujx5r8oipw7ercdr'||game.system?.id!=='pf2e'||game.system.version!=='8.5.1')return unresolved('provider-unavailable');
  if(!actor?.uuid||liveActor(actor.uuid)!==actor)return unresolved('actor-not-live');
  const sources=[];
  try{
   const candidates=effects.list(actor)??[];
   // The store intentionally filters unsafe/incomplete keys. A persisted but
   // unreadable candidate must not disappear into an apparently clear result.
   const raw=actor.flags?.[ID]?.roaringApplause?.sources;
   if(raw!==undefined){
    if(!raw||typeof raw!=='object'||Array.isArray(raw))sources.push(entry(null,'manual','source-unproven'));
    else for(const nonce of Object.keys(raw))if(!candidates.some(r=>r?.state?.sourceNonce===nonce))sources.push(entry(nonce,'manual','source-unproven'));
   }
   for(const r of candidates){
    const nonce=r?.state?.sourceNonce;
    try{
     const s=r?.state,i=r?.context?.immunity;
     demand(r?.schema===1&&Number.isSafeInteger(r.revision)&&r.revision>=0&&bounded(nonce)&&!['constructor','prototype'].includes(nonce)&&s.source?.targetActorUuid===actor.uuid,'source-unproven');
     projectRoaringConditions(s);
     demand(r.context?.paymentId===s.source.castNonce&&typeof r.context.userId==='string'&&typeof r.context.gmId==='string'&&Number.isFinite(r.context.dc)&&i?.checked===true&&i.systemVersion==='8.5.1'&&['spell','slowed','fascinated'].every(k=>typeof i[k]==='boolean'),'source-unproven');
     if(s.status==='ended'||i.spell){sources.push(entry(nonce,'clear',s.status==='ended'?'source-ended':'whole-spell-immune'));continue;}
     const preview=reduceRoaringSource(s,{type:'reconcile',sourceNonce:nonce,observation:observation(s)}).source;
     if(preview.status==='ended'){sources.push(entry(nonce,'clear',preview.termination?.reason??'source-ended'));continue;}
     const parent=effects.inspectReactionParent?.({actor,nonce});
     // Removing a confirmed own parent ends this source independently of an
     // unverified result/GM epoch, even before its asynchronous delete hook.
     if(parent?.status==='removed'){sources.push(entry(nonce,'clear','parent-removed'));continue;}
     const projection=projectRoaringConditions(preview),gm=game.users?.activeGM;
     if(resume.has(nonce)||!gm?.active||!gm.isGM||gm.id!==r.context.gmId||electedGM!==gm.id){sources.push(entry(nonce,'manual','continuity-unproven'));continue;}
     if(preview.status!=='active'||projection.manualReview||!projection.noReactions||preview.result?.revision!==1||!bounded(preview.result.receiptId)){sources.push(entry(nonce,'manual','result-or-timing-unproven'));continue;}
     const result=evidence.inspectResultContinuity?.({sourceNonce:nonce,originalMessageUuid:s.source.originalMessageUuid,result:copy(s.result)});
     if(result?.status!=='current'){sources.push(entry(nonce,'manual',result?.reason??'result-continuity-unproven'));continue;}
     sources.push(parent?.status==='present'?entry(nonce,'restricted','active-source'):entry(nonce,'manual',parent?.reason??'parent-unproven'));
    }catch{sources.push(entry(nonce??null,'manual','source-unproven'));}
   }
  }catch{return unresolved('source-unproven')}
  return {status:sources.some(s=>s.status==='restricted')?'restricted':sources.some(s=>s.status==='manual')?'manual':'clear',sources};
 }
 async function applyLifecycleEvent({actor,nonce,event}){
  demand(installed&&isActiveGM(game),'生命周期事实必须由当前主GM执行。');
  return queue.run(actor.uuid,async()=>{
   const generation=authorityGeneration;
   async function applyStep(step){
   demand(isActiveGM(game),'主GM已改变。');const r=effects.get(actor,nonce);demand(r,'缺少持久来源。');
   const reduced=reduceRoaringSource(r.state,{...step,sourceNonce:nonce,observation:observation(r.state)});
   if(!same(reduced.source,r.state))await effects.saveState({actor,nonce,state:reduced.source,expectedRevision:r.revision});
   demand(isActiveGM(game)&&same(effects.get(actor,nonce)?.state,reduced.source),'来源状态保存未确认。');
   for(const command of reduced.commands){
    demand(isActiveGM(game),'主GM已改变。');const args={actor,nonce};
    if(command.type==='condition-sync')await effects.materialize(args);
    else if(command.type==='source-end')await effects.end(args);
    else if(command.type==='end-fascination')await effects.endFascination(args);
    else if(command.type==='renew-source')await effects.renew(args);
    else if(command.type==='restore-finite')await effects.restoreFinite(args);
    else if(command.type==='clap-prompt')await onClap({actor,sourceNonce:nonce,state:copy(reduced.source),command:copy(command)});
    else if(command.type==='manual-review')await manualNotice({sourceNonce:nonce,reason:command.reason});
   }
   return reduced;
   }
   // A save may arrive while ready is still awaiting other providers. Keep the
   // recovery barrier in the same source queue as every incoming native fact.
   if(resume.has(nonce)&&event.type!=='continuity-unverified')await applyStep({type:'continuity-unverified'});
   const result=await applyStep(event);
   // An older write may return after the same GM disconnects and returns.
   // It cannot consume the new barrier raised during that await.
   if(generation===authorityGeneration)resume.delete(nonce);return result;
  });
 }
 async function onVerified(event){
  demand(isActiveGM(game),'只有主GM可接受保存证据。');const found=find(event.sourceNonce),s=found?.record.state;
  demand(found&&s.source.originalMessageUuid===event.originalMessageUuid&&s.source.targetTokenUuid===event.targetUuid&&s.source.castNonce===event.castNonce&&bounded(event.proof?.invocationId),'保存证据与来源不符。');
  const item=await fromUuid(s.source.itemUuid),assessment=immunity(found.actor,item);
  if(!same(assessment,found.record.context.immunity))return onManual({...event,reason:'immunity-changed-needs-review'});
  return applyLifecycleEvent({actor:found.actor,nonce:event.sourceNonce,event:{type:'save-confirmed',revision:event.revision,receiptId:event.proof.invocationId,outcome:event.adjustedOutcome}});
 }
 async function onManual(event){
  if(!isActiveGM(game))return;const f=find(event.sourceNonce);if(!f)return;
  return applyLifecycleEvent({actor:f.actor,nonce:event.sourceNonce,event:{type:'save-unverified',receiptId:event.proof?.invocationId??randomId(),reason:String(event.reason??'unverified-save')}});
 }
 async function reconcile(){
  if(!installed||!isActiveGM(game))return;
  const failures=[];
  for(const {actor,record}of records()){
   try{
   if(record.state.status==='ended'){
    // Logical termination is not evidence of completed document deletion. The
    // store observes its exact existing delete operation; it must not repeat an
    // uncertain native deletion or create any replacement item.
    if(record.effects.status!=='ended')await effects.end({actor,nonce:record.state.sourceNonce});
    continue;
   }
   const nonce=record.state.sourceNonce;
   if(record.state.status!=='ended')await applyLifecycleEvent({actor,nonce,event:{type:resume.has(nonce)||record.context.gmId!==game.users.activeGM?.id?'continuity-unverified':'reconcile'}});
   const fresh=effects.get(actor,record.state.sourceNonce);
   if(fresh?.state.status==='active'&&['creating','uncertain'].includes(fresh.effects.status)){
    // A lost native create reply may leave parentId null despite a real parent.
    // In these two phases the store only observes the exact operation marker;
    // it cannot enter a new create. No unique live result stays manual.
    try{await effects.materialize({actor,nonce:record.state.sourceNonce});}
    catch(error){
     await applyLifecycleEvent({actor,nonce:record.state.sourceNonce,event:{type:'save-unverified',receiptId:randomId(),reason:'materialization-uncertain-no-proven-parent'}});
     const key=`${record.state.sourceNonce}:${error.message??error}`;if(!reported.has(key)){reported.add(key);onError(error);}
    }
   }
   if(fresh?.state.status!=='ended'&&record.context.gmId!==game.users.activeGM?.id)await applyLifecycleEvent({actor,nonce:record.state.sourceNonce,event:{type:'save-unverified',receiptId:randomId(),reason:'GM-changed-unverified-scope'}});
   const message=game.messages.get(record.state.source.originalMessageUuid.split('.').at(-1));if(message&&lookupSource(message))evidence.track(message);
   }catch(error){failures.push(new Error(`${record.state.sourceNonce}: ${error.message??error}`,{cause:error}));}
  }
  if(failures.length)throw new AggregateError(failures,failures.map(e=>e.message).join('\n'));
 }
 const startKey=(combat,c)=>`${combat.id}/${c.id}`;
 function seedStarts(){for(const combat of values(game.combats))for(const c of values(combat.turns)){const key=startKey(combat,c);if(!lastStarts.has(key))lastStarts.set(key,c.flags?.pf2e?.roundOfLastTurn??null);}}
 function captureTargetStart(c,changes,userId){
  const changed=changes?.['flags.pf2e.roundOfLastTurn']??changes?.flags?.pf2e?.roundOfLastTurn;
  if(changed===undefined)return null;
  const combats=values(game.combats).filter(combat=>values(combat.turns).includes(c));if(combats.length!==1)return;
  const combat=combats[0],key=startKey(combat,c),previous=lastStarts.get(key),current=c.flags?.pf2e?.roundOfLastTurn??null;lastStarts.set(key,current);
  if(!isActiveGM(game)||!game.users.get(userId)||previous===undefined||previous===current||changed!==current||!Number.isInteger(current)||current!==combat.round||!combat.started||values(combat.turns)[combat.turn]!==c)return;
  const token=c.token,actor=c.actor;
  if(!token?.uuid||token.actor!==actor||!actor?.uuid||values(game.combats).filter(x=>x.started).flatMap(x=>values(x.turns).filter(t=>t.token?.uuid===token.uuid)).length!==1)return;
  return {combatId:combat.id,combatantId:c.id,round:combat.round,actorUuid:actor.uuid,tokenUuid:token.uuid,lastTurnStart:current};
 }
 async function targetStarted(targetTurn){
  if(!targetTurn||!isActiveGM(game))return;
  const combat=game.combats.get(targetTurn.combatId),c=values(combat?.turns).find(c=>c.id===targetTurn.combatantId);
  if(!combat?.started||combat.round!==targetTurn.round||values(combat.turns)[combat.turn]!==c||c?.actor?.uuid!==targetTurn.actorUuid||c.token?.uuid!==targetTurn.tokenUuid||c.flags?.pf2e?.roundOfLastTurn!==targetTurn.lastTurnStart)return;
  for(const found of records())if(found.record.state.source.targetTokenUuid===targetTurn.tokenUuid&&found.record.state.status==='active'&&found.record.context.immunity.spell===false){
   await applyLifecycleEvent({actor:found.actor,nonce:found.record.state.sourceNonce,event:{type:'target-start',targetTurn}});
  }
 }
 function register({Hooks:api,socket:rpc}){
  if(installed)return cleanup;installed=true;Hooks=api;socket=rpc;nativeCasts.addInvocationAdapter(KIND,{validate:validateInvocation,consumePolicy});evidence.register({Hooks,socket});
  electedGM=game.users.activeGM?.id??null;authorityGeneration++;
  for(const {record}of records())if(record.state.status!=='ended')resume.add(record.state.sourceNonce);seedStarts();
  const on=(name,fn)=>hooks.push([name,Hooks.on(name,fn)]),changed=()=>reconcile().catch(error=>{const key=String(error?.message??error);if(!reported.has(key)){reported.add(key);onError(error);}});
  for(const name of ['pf2e.endTurn','pf2e.startTurn','updateCombat','deleteCombat','deleteCombatant','deleteToken','deleteScene','deleteActor','updateWorldTime'])on(name,changed);
  // Core emits userConnected after changing user.active. Invalidate before
  // any await, even when the same GM returns before reconciliation finishes.
  const authorityChanged=()=>{
   const current=game.users.activeGM?.id??null;
   if(current!==electedGM){electedGM=current;authorityGeneration++;for(const {record}of records())if(record.state.status!=='ended')resume.add(record.state.sourceNonce);}
   return changed();
  };
  for(const name of ['userConnected','updateUser'])on(name,authorityChanged);
  for(const name of ['createCombat','createCombatant'])on(name,seedStarts);
  on('updateCombatant',async(c,changes,_options,userId)=>{const start=captureTargetStart(c,changes,userId);await changed();await targetStarted(start).catch(onError);});
  on('updateChatMessage',message=>{if(lookupSource(message))evidence.track(message);});
  // Actor source flags and the original-card marker are separate broadcasts.
  // Whichever arrives second can enroll the still-empty row on this client;
  // a late already-populated row remains manual in the evidence adapter.
  on('updateActor',actor=>{for(const r of effects.list(actor)??[]){const message=game.messages.get(r.state.source.originalMessageUuid.split('.').at(-1));if(message&&lookupSource(message))evidence.track(message);}});
  on('deleteItem',(item,_options,userId)=>{
   if(!isActiveGM(game)||!game.users.get(userId))return;const own=effects.identifyOwnedItem?.(item,{deleted:true});if(!own)return;
   applyLifecycleEvent({actor:own.actor,nonce:own.nonce,event:{type:own.isParent?'own-parent-deleted':'own-child-deleted',condition:own.condition,itemUuid:own.itemUuid,receiptId:randomId()}}).catch(onError);
  });
  socket.register(PROOF,async function(p){try{return {ok:true,value:await localProof(p,this.socketdata?.userId)}}catch(e){return {ok:false,error:e.message}}});
  socket.register(COMPLETE,async function(p){try{return {ok:true,value:await complete(p,this.socketdata?.userId)}}catch(e){return {ok:false,error:e.message}}});
  socket.register(NOTICE,async function(p){try{demand(this.socketdata?.userId===game.users.activeGM?.id,'来源广播必须来自当前主GM。');const message=await fromUuid(p?.messageUuid);demand(lookupSource(message),'广播来源尚未持久生效。');return {ok:evidence.track(message)};}catch(e){return {ok:false,error:e.message}}});
  return cleanup;
 }
 function cleanup(){installed=false;for(const[n,id]of hooks.splice(0))Hooks.off(n,id);scopes.clear();byItem.clear();resume.clear();lastStarts.clear();evidence.cleanup();}
 return {interceptCast,lookupSource,onVerified,onManual,applyLifecycleEvent,reconcile,reactionRestriction,register,cleanup,listSources:()=>records().map(({actor,record})=>({actor,record:copy(record)})),diagnostic:()=>({installed,activeScopes:scopes.size,sourceCount:records().length,reactionRestrictionQuery:installed,automaticSustain:false})};
}
