import {MODULE_ID as M} from './rules.mjs';
import {SerialActions} from './runtime.mjs';
import {getSourceId,isActiveGM,showNativeChoice,validateNativeChoices,markUnappliedDamageError} from './native-context.mjs';
import {reactionPermitted,requireReactionPermitted} from './reaction-restriction.mjs';
import {withReactionReservation,genericReactionAvailable,genericReactionSpent,shieldEncounter} from './reaction-budget.mjs';
import {createShieldReactionResources} from './shield-reaction-resources.mjs';
import {resolveSpiritualScarSource,validateSpiritualScarSource,hasIncomingScarDamage} from './spiritual-scar-source.mjs';
import {SPIRITUAL_SCAR_SOURCE,compileSpiritualScarResistance,withSpiritualScarResistance,spiritualScarMarker} from './spiritual-scar-native.mjs';
import {createSpiritualScarUseLedger} from './spiritual-scar-use-ledger.mjs';
import {spiritualScarClaims,findSpiritualScarClaim} from './spiritual-scar-reaction-proof.mjs';
import {spiritualScarExpiryFor} from './spiritual-scar-expiry.mjs';
const values=c=>Array.from(c?.values?.()??c??[]),same=(a,b)=>JSON.stringify(a)===JSON.stringify(b),author=m=>m?.author?.id??m?.user?.id??m?.user;
const random=()=>globalThis.foundry?.utils?.randomID?.(24)??crypto.randomUUID(),brand=Symbol('spiritual-scar-native-scope');
const key=s=>`${s.damageMessageId}:${s.rollIndex}:${s.tokenUuid}`;
const action=actor=>values(actor?.items).find(i=>i.type==='action'&&getSourceId(i)===SPIRITUAL_SCAR_SOURCE&&i.system?.actionType?.value==='reaction');
const owned=(actor,user)=>user?.active===true&&actor?.testUserPermission?.(user,'OWNER')===true;

/** Scoped native damage orchestration. No retrospective damage-card action and
 * no public resource grant. A verified follow-up implementation is required. */
export function createSpiritualScarProvider({game,fromUuid=globalThis.fromUuid,getRollContext,nativeAdapter,followup,reactionRestriction,
 reactionResources=createShieldReactionResources({game,reactionRestriction}),useLedger=createSpiritualScarUseLedger({game,fromUuid}),choose,show=showNativeChoice,originalUse,
 compileResistance=compileSpiritualScarResistance,withResistance=withSpiritualScarResistance,onError=console.error,onManual=()=>globalThis.ui?.notifications?.warn?.('精神伤痕本次抗力来源无法区分，后续意志豁免请手动核对。')}={}){
 const live=new Map(),plans=new WeakSet(),authorizations=new Map(),pendingWaits=new Set(),queue=new SerialActions();let socket,installation,disposeObserver,closed=false;
 const ready=()=>!closed&&game.world?.id==='ujx5r8oipw7ercdr'&&game.system?.id==='pf2e'&&game.system.version==='8.5.1'&&nativeAdapter?.nativeBridgeDiagnostic?.().ready===true&&followup?.ready?.()===true&&game.pf2e?.settings?.iwr!==false&&!game.modules?.get('pf2e-auto-action-tracker')?.active;
 const handlesActor=actor=>ready()&&actor?.type==='character'&&!!action(actor);
 const resolveAction=item=>handlesActor(item?.actor)&&action(item.actor)===item?'spiritual-scar:use':undefined;
 const gm=()=>{if(!isActiveGM(game)||!ready())throw Error('精神伤痕主GM或已验证兼容层已改变。');};
 const preferred=actor=>{const users=values(game.users).filter(u=>owned(actor,u));return users.find(u=>!u.isGM&&(u.character?.id===actor.id||u.character?.uuid===actor.uuid))??users.find(u=>!u.isGM)??users.find(u=>u.id===game.users.activeGM?.id)};
 const bounded=combat=>({combat,modules:game.modules,messages:game.messages,users:game.users});
 async function verified(snapshot){const context=await validateSpiritualScarSource({game,fromUuid,snapshot});if(!context.verified)throw Error(`精神伤痕伤害来源已改变：${context.unsupportedReason}`);return context;}
 function scopeProof(payload,user){
  const p=live.get(payload?.scopeId);
  if(!p||p.userId!==user?.id||p.leader!==game.users.activeGM?.id||!owned(p.actor,user)||!same(p.snapshot,payload.snapshot)||JSON.stringify(p.damage.toJSON())!==p.damageEvidence)throw Error('没有本操作者仍有效的精神伤痕原生伤害调用。');
  if(payload.phase==='receipt')return {scopeId:p.scopeId,snapshot:p.snapshot,nonce:p.claim?.nonce,nativeEntered:p.nativeEntered===true,nativeReturned:p.nativeReturned===true,messageId:p.messageId??null,multiple:!!p.multiple,observations:p.observations,label:p.label??null,uniqueLabel:p.uniqueLabel===true};
  if(p.state!=='waiting')throw Error('本次精神伤痕伤害调用已关闭。');return {scopeId:p.scopeId,snapshot:p.snapshot};
 }
 async function authenticate(payload,user,phase){
  gm();const context=await verified(payload?.snapshot);gm();if(!owned(context.actor,user))throw Error('精神伤痕伤害申请者不是实际受伤角色拥有者。');
  const args={scopeId:payload.scopeId,snapshot:payload.snapshot,phase},local=user.id===game.user.id;
  const reply=local?scopeProof(args,user):await socket?.executeAsUser('scar:scope',user.id,args),proof=local?reply:reply?.ok?reply.value:null;gm();
  if(!proof||proof.scopeId!==payload.scopeId||!same(proof.snapshot,payload.snapshot))throw Error('精神伤痕原生调用证明失效。');return {...context,proof};
 }
 function option(context,expected,{frequency=1,canAct=true}={}){
  const actor=context.token.actor,ability=action(actor),encounter=shieldEncounter(actor,context.token,game);
  if(!handlesActor(actor)||!ability||ability.system?.frequency?.max!==1||ability.system.frequency.per!=='day'||ability.system.frequency.value!==frequency||actor.rollOptions?.all?.['spiritual-scar']||canAct&&(actor.canAct!==true||actor.isDead||!reactionPermitted(actor,reactionRestriction))||!encounter)return null;
  const result={actor,ability,token:context.token,...encounter};
  if(expected&&(ability.uuid!==expected.itemUuid||actor.uuid!==expected.actorUuid||context.token.uuid!==expected.tokenUuid||encounter.combat.id!==expected.combatId||encounter.combatant.id!==expected.combatantId||encounter.epoch!==expected.epoch||encounter.combat.round!==expected.round||encounter.combat.turn!==expected.turn||!owned(actor,game.users.get(expected.userId))))return null;
  return result;
 }
 const requireOption=(context,expected,options)=>{const result=option(context,expected,options);if(!result)throw Error('精神伤痕能力、日次数、反应资格或实际回合已改变。');return result;};
 async function updateClaim(nonce,change){
  gm();const initial=findSpiritualScarClaim(game,nonce);if(!initial)throw Error('精神伤痕反应认领不存在。');
  return withReactionReservation(initial.combatant.actor,game,async()=>{
   gm();const bound=findSpiritualScarClaim(game,nonce);if(!bound)throw Error('精神伤痕反应认领已改变。');
   const claims=structuredClone(spiritualScarClaims(bound.combatant)),record=claims.find(c=>c.nonce===nonce);Object.assign(record,change);
   const result=await bound.combatant.update({[`flags.${M}.spiritualScarClaims`]:claims});gm();
   if(result!==bound.combatant||!same(findSpiritualScarClaim(game,nonce)?.claim,record))throw Error('精神伤痕反应状态没有提交。');return record;
  });
 }
 async function waitFor(predicate,event,description){
  const current=predicate();if(current)return current;if(!installation)throw Error(description);
  const {Hooks}=installation;
  return new Promise((resolve,reject)=>{
   let id,timer,done=false;
   const cancel=()=>finish(Error('精神伤痕自动流程已停止。'));
   function finish(error,value){if(done)return;done=true;clearTimeout(timer);if(id!==undefined)Hooks.off(event,id);pendingWaits.delete(cancel);error?reject(error):resolve(value)}
   const check=()=>{try{const value=predicate();if(value)finish(null,value)}catch(error){finish(error)}};
   pendingWaits.add(cancel);id=Hooks.on(event,check);timer=setTimeout(()=>finish(Error(description)),15000);check();
  });
 }
 function beforeUse(item){
  const auth=authorizations.get(item?.uuid);if(!auth)return true;const record=useLedger.current(item);
  if(!ready()||game.users.activeGM?.id!==auth.gmId||game.user.id!==auth.claim.userId||!owned(item.actor,game.user)||record?.nonce!==auth.claim.nonce||!['paying','paid','ready'].includes(record.status))throw Error('本次精神伤痕原生使用授权已失效。');return true;
 }
 function captureUsage(item){const auth=authorizations.get(item?.uuid);if(!auth)return null;beforeUse(item);const record=useLedger.current(item);return ['paid','ready'].includes(record.status)?{spiritualScarInput:{nonce:record.nonce,paymentNonce:record.paymentNonce}}:null;}
 async function payLocal({nonce},sender){
  if(sender?.id!==game.users.activeGM?.id)throw Error('只有主GM可请求本次精神伤痕原生使用。');
  const claim=await waitFor(()=>findSpiritualScarClaim(game,nonce)?.claim,'updateCombatant','精神伤痕反应认领尚未同步。'),ability=await fromUuid(claim.itemUuid),token=await fromUuid(claim.tokenUuid);
  if(!ready()||claim.status!=='reserved'||claim.userId!==game.user.id||ability?.actor!==token?.actor||action(ability.actor)!==ability||!owned(ability.actor,game.user)||authorizations.has(ability.uuid))throw Error('精神伤痕原始能力或本次使用者不匹配。');
  await waitFor(()=>useLedger.current(ability)?.nonce===nonce&&useLedger.current(ability)?.status==='paying','updateItem','精神伤痕准备使用状态尚未同步。');
  const auth={claim,token,gmId:sender.id};authorizations.set(ability.uuid,auth);
  try{
   useLedger.authorizePayment(ability,nonce);beforeUse(ability);
   const message=await(originalUse?originalUse(ability):game.pf2e.rollItemMacro(ability.uuid));
   if(!message?.id)throw Error('精神伤痕原生使用未创建准确卡片；不会重试付款。');
   const record=await waitFor(()=>{const r=useLedger.current(ability);return r?.nonce===nonce&&r.status==='ready'?r:null},'updateItem','精神伤痕原生付款回执等待超时；不会重试。');
   beforeUse(ability);if(!message?.id||record.messageId!==message.id)throw Error('精神伤痕原始使用卡未与本次付款一致。');return message.id;
  }finally{authorizations.delete(ability.uuid);useLedger.clearAuthorization(ability,nonce)}
 }
 async function executeUsage(context){
  if(!context.message?.flags?.[M]?.spiritualScarInput)return '手动使用保留原生规则；没有本次自动伤害调用，不追溯伤害。';
  gm();await useLedger.bindUsage(context);return '已确认本次原生每日次数与反应，等待本次伤害结算。';
 }
 async function chooseUse(actor,user){
  const input={actor,user,title:'精神伤痕：抵抗本次魔族精魂伤害？',choices:[{value:'use',label:'使用精神伤痕（反应；每日一次）'},{value:'decline',label:'不使用'}]};
  let answer;if(choose)answer=await choose(input);else if(user.id===game.user.id)answer=await show(input);else{const reply=await socket?.executeAsUser('scar:choice',user.id,{actorUuid:actor.uuid,title:input.title,choices:input.choices});if(!reply?.ok)throw Error(reply?.error??'精神伤痕选择未完成。');answer=reply.value}
  gm();if(answer!=null&&!input.choices.some(c=>c.value===answer))throw Error('精神伤痕选择无效。');return answer;
 }
 async function uncertain(claim,reason){
  const item=await fromUuid(claim.itemUuid),user=game.users.get(claim.userId);
  await useLedger.uncertain({actor:item?.actor,item,user,nonce:claim.nonce,reason:String(reason?.message??reason).slice(0,1000)}).catch(onError);
  await updateClaim(claim.nonce,{status:'uncertain'}).catch(onError);
 }
 async function cancelUnattempted(scope,claim){
  const record=useLedger.current(scope.item);if(record?.nonce!==scope.nonce||record.status!=='claimed'||scope.item.system.frequency.value!==1)return false;
  await withReactionReservation(scope.actor,game,async()=>{
   gm();const current=useLedger.current(scope.item);if(current?.nonce!==scope.nonce||current.status!=='claimed')throw Error('精神伤痕已开始原生使用，不能取消预留。');
   let bound=claim?findSpiritualScarClaim(game,claim.nonce):null;
   if(bound&&bound.claim.status!=='reserved')throw Error('精神伤痕反应预留已经推进。');
   // Cancelling this still-unattempted daily claim never changes its frequency.
   await useLedger.cancelClaim(scope);gm();
   bound=claim?findSpiritualScarClaim(game,claim.nonce):null;
   let changes={};
   if(bound){
    const {combatant,combat}=bound,currentEpoch=()=>{const e=shieldEncounter(scope.actor,combatant.token,game);return e?.combat===combat&&e.epoch===claim.epoch},spent=()=>genericReactionSpent(scope.actor,bounded(combat),{excludeClaimKeys:[claim.claimKey]});
    if(currentEpoch()&&!spent())changes=await reactionResources.release(combatant,claim.resource);
    gm();if(findSpiritualScarClaim(game,claim.nonce)?.claim.status!=='reserved')throw Error('精神伤痕取消期间反应认领已经改变。');
    // PF2e can refresh a turn independently while our item write or the
    // resource adapter awaits. Never restore that newer turn's resource.
    if(!currentEpoch()||spent())changes={};
    else for(const path of Object.keys(changes))if(path.split('.').reduce((v,k)=>v?.[k],combatant)!==claim.resource.after)delete changes[path];
    const budget=combatant.flags?.[M]?.reactionBudget;
    if(currentEpoch()&&budget?.epoch===claim.epoch)changes[`flags.${M}.reactionBudget`]={...structuredClone(budget),entries:budget.entries.filter(e=>e.claimKey!==claim.claimKey)};
    changes[`flags.${M}.spiritualScarClaims`]=structuredClone(spiritualScarClaims(combatant)).map(c=>c.nonce===claim.nonce?{...c,status:'refunded'}:c);
   }
   if(bound){const result=await bound.combatant.update(changes);if(result!==bound.combatant||findSpiritualScarClaim(game,claim.nonce)?.claim.status!=='refunded')throw Error('精神伤痕未尝试使用的反应预留退回未确认。');}
  });return true;
 }
 async function request(payload,user){
  const initial=await authenticate(payload,user);
  return queue.run(initial.token.uuid,async()=>{
   let context=await authenticate(payload,user),candidate=option(context);if(!candidate)return {status:'passed'};
   if(values(game.combats).some(c=>values(c.turns).some(t=>spiritualScarClaims(t).some(r=>r.sourceKey===key(context.snapshot)&&r.status!=='refunded'))))throw Error('本次精神伤痕原伤害已处理，不能重复支付。');
   if(!genericReactionAvailable(candidate.actor,bounded(candidate.combat),{reactionRestriction}))return {status:'passed'};
   const resources=await reactionResources.snapshot(candidate.combatant);gm();if(!reactionResources.available(resources,'generic'))return {status:'passed'};
   const decisionUser=preferred(candidate.actor);if(!decisionUser)throw Error('精神伤痕没有在线拥有者。');
   const expected={actorUuid:candidate.actor.uuid,tokenUuid:candidate.token.uuid,itemUuid:candidate.ability.uuid,userId:decisionUser.id,combatId:candidate.combat.id,combatantId:candidate.combatant.id,epoch:candidate.epoch,round:candidate.combat.round,turn:candidate.combat.turn};
   if(await chooseUse(candidate.actor,decisionUser)!=='use')return {status:'passed'};
   let claim,dailyClaim;
   try{
    await withReactionReservation(candidate.actor,game,async()=>{
     context=await authenticate(payload,user);candidate=requireOption(context,expected);requireReactionPermitted(candidate.actor,reactionRestriction);
     if(!genericReactionAvailable(candidate.actor,bounded(candidate.combat),{reactionRestriction}))throw Error('精神伤痕的通用反应已经消耗。');
     const snapshot=await reactionResources.snapshot(candidate.combatant);gm();candidate=requireOption(await authenticate(payload,user),expected);
     if(!reactionResources.available(snapshot,'generic'))throw Error('精神伤痕的 Reaction Checker 反应已经消耗。');
     const use=await useLedger.claim({actor:candidate.actor,item:candidate.ability,user:decisionUser,invocationId:payload.scopeId,fingerprint:context.snapshot.evidence,privacy:context.privacy});
     dailyClaim={actor:candidate.actor,item:candidate.ability,user:decisionUser,nonce:use.nonce};
     candidate=requireOption(await authenticate(payload,user),expected);const resource=reactionResources.reserve(snapshot,'generic');
     claim={...expected,nonce:use.nonce,claimKey:`scar:${use.nonce}`,scopeId:payload.scopeId,sourceUserId:user.id,source:context.snapshot,sourceKey:key(context.snapshot),status:'reserved',resource:resource.proof,expiry:spiritualScarExpiryFor(candidate.token,game)};
     const previous=candidate.combatant.flags?.[M]?.reactionBudget,entries=previous?.epoch===candidate.epoch?[...previous.entries??[]]:[];entries.push({type:'reaction',cost:1,slug:'spiritual-scar',claimKey:claim.claimKey});
     const result=await candidate.combatant.update({[`flags.${M}.reactionBudget`]:{epoch:candidate.epoch,entries},[`flags.${M}.spiritualScarClaims`]:[...structuredClone(spiritualScarClaims(candidate.combatant)),claim],...resource.changes});gm();
     const actualBudget=candidate.combatant.flags?.[M]?.reactionBudget,actual=path=>path.split('.').reduce((value,key)=>value?.[key],candidate.combatant);
     if(result!==candidate.combatant||!same(findSpiritualScarClaim(game,claim.nonce)?.claim,claim)||!same(actualBudget,{epoch:candidate.epoch,entries})||Object.entries(resource.changes).some(([path,value])=>!same(actual(path),value)))throw Error('精神伤痕反应预留没有提交。');
    });
    candidate=requireOption(await authenticate(payload,user),expected);
    await useLedger.beginPayment({actor:candidate.actor,item:candidate.ability,user:decisionUser,nonce:claim.nonce});
    const local=decisionUser.id===game.user.id,result=local?await payLocal({nonce:claim.nonce},game.user):await socket?.executeAsUser('scar:pay',decisionUser.id,{nonce:claim.nonce});gm();
    const messageId=local?result:result?.ok?result.value:null;if(!messageId)throw Error(result?.error??'精神伤痕原生使用结果不确定。');
    candidate=requireOption(await authenticate(payload,user),expected,{frequency:0});const use=useLedger.current(candidate.ability);
    if(use?.nonce!==claim.nonce||use.status!=='ready'||use.messageId!==messageId)throw Error('精神伤痕准确日频付款尚未确认。');
    await updateClaim(claim.nonce,{status:'paid',messageId,paymentNonce:use.paymentNonce});return {status:'paid',nonce:claim.nonce};
   }catch(error){let cancelled=false;if(dailyClaim)cancelled=await cancelUnattempted(dailyClaim,claim).catch(e=>{onError(e);return false});if(claim&&!cancelled)await uncertain(claim,error);throw error;}
  });
 }
 async function beginNative(payload,user){
  const context=await authenticate(payload,user),claim=findSpiritualScarClaim(game,payload.nonce)?.claim;
  if(!claim||claim.status!=='paid'||claim.scopeId!==payload.scopeId||claim.sourceUserId!==user.id||!same(claim.source,payload.snapshot))throw Error('精神伤痕本次付款不能重复进入伤害。');
  const candidate=requireOption(context,claim,{frequency:0});await useLedger.consume({actor:candidate.actor,item:candidate.ability,user:game.users.get(claim.userId),nonce:claim.nonce});await updateClaim(claim.nonce,{status:'native'});return true;
 }
 function receiptMatches(message,context,claim,user){
  const pf=message?.flags?.pf2e,proof=message?.flags?.[M]?.spiritualScarDamage;
  return game.messages.get(message?.id)===message&&author(message)===user.id&&message.speaker?.actor===context.actor.id&&`Scene.${message.speaker?.scene}.Token.${message.speaker?.token}`===context.token.uuid&&pf?.context?.type==='damage-taken'&&pf.context.options?.includes(spiritualScarMarker(claim.nonce))&&pf.origin?.actor===context.sourceActor.uuid&&(context.item?pf.origin?.uuid===context.item.uuid:!pf.origin?.uuid)&&proof?.nonce===claim.nonce&&proof.scopeId===claim.scopeId&&!pf.appliedDamage?.isReverted&&pf.appliedDamage?.isHealing!==true;
 }
 function prevention(proof){
  if(proof.observations?.length!==1)return {status:'uncertain'};const observation=proof.observations[0],iwr=observation.iwr;
  if(iwr.finalDamage!==0||iwr.persistent.length)return {status:'none'};
  const negative=iwr.applications.filter(a=>a.adjustment<0),scar=negative.filter(a=>a.category==='resistance'&&a.type===proof.label&&!a.ignored&&!a.redirect);
  if(!scar.length)return {status:'none'};
  if(!proof.uniqueLabel)return {status:'manual',reason:'same-label-resistance'};
  if(scar.length!==negative.length)return {status:'manual',reason:'combined-damage-prevention'};
  return {status:'prevented',observation};
 }
 async function complete(payload,user){
  const context=await authenticate(payload,user,'receipt'),claim=findSpiritualScarClaim(game,payload.nonce)?.claim;
  if(!claim||claim.scopeId!==payload.scopeId||claim.sourceUserId!==user.id||!same(claim.source,payload.snapshot))throw Error('精神伤痕伤害完成回执不属于本次付款。');if(claim.status==='done')return true;
  const proof=context.proof;
  try{
   if(!payload.applied||!proof.nativeEntered||!proof.nativeReturned||proof.nonce!==claim.nonce||proof.multiple||!proof.messageId||claim.status!=='native')throw Error('精神伤痕缺少唯一完成的原生伤害回执；保留已付资源。');
   const message=await waitFor(()=>game.messages.get(proof.messageId),'createChatMessage','精神伤痕原生伤害卡同步超时。');gm();
   if(!receiptMatches(message,context,claim,user))throw Error('精神伤痕原生伤害回执不一致。');
   const result=prevention(proof);if(result.status==='uncertain')throw Error('精神伤痕未得到唯一可信原生 IWR 结果。');
   if(result.status==='prevented'){
    await updateClaim(claim.nonce,{status:'followup',receiptId:message.id});
    const followed=await followup.apply({claim,actor:context.actor,fiend:context.sourceActor,fiendToken:context.sourceToken,privacy:context.privacy,damageMessage:message,
     authorize:async()=>{gm();const refreshed=await verified(claim.source);gm();const current=findSpiritualScarClaim(game,claim.nonce)?.claim;return current?.status==='followup'&&current.receiptId===message.id&&receiptMatches(message,refreshed,current,user)}});
    if(!followed||!['done','manual'].includes(followed.status))throw Error('精神伤痕后续豁免或条件没有确认完成。');
    if(followed.status==='manual'){result.status='manual';result.reason=followed.reason??'followup-manual';}
   }
   if(result.status==='manual')onManual({claim,message,reason:result.reason});
   await updateClaim(claim.nonce,{status:'done',receiptId:message.id,followup:result.status,manualReason:result.reason??null});return true;
  }catch(error){await uncertain(claim,error);throw error;}
 }
 async function rpc(method,payload){
  if(isActiveGM(game))return ({request,native:beginNative,complete})[method](payload,game.user);
  const leader=game.users.activeGM?.id;if(!socket||!leader)throw Error('精神伤痕需要在线主GM。');const result=await socket.executeAsUser(`scar:${method}`,leader,payload);
  if(game.users.activeGM?.id!==leader||!result?.ok)throw Error(result?.error??'精神伤痕结算未确认。');return result.value;
 }
 async function beforeDamage(actor,params){
  const token=params.token?.document??params.token,actual=token?.actor;
  if(!handlesActor(actual)||!hasIncomingScarDamage(params)||actual.rollOptions?.all?.['spiritual-scar']||action(actual)?.system?.frequency?.value!==1||actual.canAct!==true||actual.isDead||!reactionPermitted(actual,reactionRestriction))return {params};
  const binding=await resolveSpiritualScarSource({game,fromUuid,actor,params,source:getRollContext?.(params.damage)});
  if(!binding.verified)throw markUnappliedDamageError(Error(`精神伤痕无法确认原伤害来源：${binding.unsupportedReason}。请手动确认来源后结算。`));
  // Validate the original rule before any choice or payment; the final native
  // call recompiles against its actual contextual actor and current rules.
  try{compileResistance({actor,ability:actor.items.get(action(actual).id),nonce:random(),options:params.rollOptions})}catch(error){throw markUnappliedDamageError(error)}
  const p={scopeId:random(),snapshot:binding.snapshot,actor,token,damage:params.damage,damageEvidence:JSON.stringify(params.damage.toJSON()),outcome:params.outcome,userId:game.user.id,leader:game.users.activeGM?.id,state:'waiting',observations:[]};live.set(p.scopeId,p);plans.add(p);
  try{
   const result=await rpc('request',{scopeId:p.scopeId,snapshot:p.snapshot});if(result.status!=='paid'){live.delete(p.scopeId);plans.delete(p);return {params}}
   const claim=await waitFor(()=>findSpiritualScarClaim(game,result.nonce)?.claim,'updateCombatant','精神伤痕准确付款尚未同步。');
   if(claim.status!=='paid'||claim.scopeId!==p.scopeId||claim.sourceUserId!==p.userId||!same(claim.source,p.snapshot))throw Error('精神伤痕付款范围不匹配。');p.claim=claim;
   const actual={...params,rollOptions:new Set([...params.rollOptions??[],spiritualScarMarker(claim.nonce)])};actual[brand]=p;return {params:actual,receipt:p};
  }catch(error){live.delete(p.scopeId);plans.delete(p);throw markUnappliedDamageError(error)}
 }
 async function wrapNativeDamage(actor,params,native){
  const p=params?.[brand];if(!plans.has(p))return native(params);
  try{
   if(!ready()||p.state!=='waiting'||p.leader!==game.users.activeGM?.id||p.userId!==game.user.id||actor!==p.actor||(params.token?.document??params.token)!==p.token||params.damage!==p.damage||params.outcome!==p.outcome||params.final||params.skipIWR||!new Set(params.rollOptions).has(spiritualScarMarker(p.claim.nonce))||JSON.stringify(params.damage.toJSON())!==p.damageEvidence)throw Error('精神伤痕最终原生伤害范围已改变。');
   const context=await verified(p.snapshot),candidate=requireOption(context,p.claim,{frequency:0});
   if((params.item?.uuid??null)!==p.snapshot.itemUuid)throw Error('精神伤痕最终伤害物品已改变。');
   const resistance=compileResistance({actor,ability:actor.items.get(candidate.ability.id),nonce:p.claim.nonce,options:params.rollOptions});
   p.label=resistance.applicationLabel;p.uniqueLabel=typeof p.label==='string'&&!actor.attributes.resistances.some(r=>r.applicationLabel===p.label);
   await rpc('native',{scopeId:p.scopeId,snapshot:p.snapshot,nonce:p.claim.nonce});requireOption(await verified(p.snapshot),p.claim,{frequency:0});
   if(!ready()||p.leader!==game.users.activeGM?.id)throw Error('精神伤痕进入原生前兼容层或主GM已改变。');
   p.state='native';p.nativeEntered=true;const result=await withResistance(actor,resistance,()=>native(params));p.nativeReturned=true;return result;
  }catch(error){if(!p.nativeEntered)throw markUnappliedDamageError(error);throw error}
 }
 function observe(snapshot){
  for(const p of live.values())if(p.state==='native'&&p.nativeEntered&&snapshot.rollOptions.includes(spiritualScarMarker(p.claim.nonce))&&snapshot.actorUuid===p.actor.uuid&&snapshot.tokenUuid===p.token.uuid&&snapshot.itemUuid===p.snapshot.itemUuid&&snapshot.total===p.damage.total)p.observations.push(snapshot);
 }
 function decorateUse(message){
  const pf=message.flags?.pf2e,proof=message.flags?.[M]?.spiritualScarInput,auth=[...authorizations.values()].find(a=>a.claim.nonce===proof?.nonce||pf?.origin?.uuid===a.claim.itemUuid||pf?.context?.type==='self-effect'&&pf.context.item===a.claim.itemUuid.split('.').at(-1)&&message.speaker?.actor===a.token.actor.id);if(!auth)return;
  const item=auth.token.actor.items.get(auth.claim.itemUuid.split('.').at(-1));beforeUse(item);const record=useLedger.current(item);
  if(proof?.nonce!==auth.claim.nonce||author(message)!==game.user.id||pf?.origin?.uuid!==item.uuid||record.paymentNonce!==proof.paymentNonce)throw Error('精神伤痕原生使用卡来源已改变。');
  message.updateSource({...record.privacy,speaker:{actor:item.actor.id,scene:auth.token.parent.id,token:auth.token.id}});
 }
 function decorateDamage(message,_data,_options,userId){
  if(userId!==game.user.id||author(message)!==game.user.id||message.flags?.pf2e?.context?.type!=='damage-taken')return;
  const options=new Set(message.flags.pf2e.context.options??[]),matches=[...live.values()].filter(p=>p.state==='native'&&p.nativeEntered&&options.has(spiritualScarMarker(p.claim.nonce))&&message.speaker?.actor===p.actor.id&&`Scene.${message.speaker?.scene}.Token.${message.speaker?.token}`===p.token.uuid);
  if(matches.length!==1)return;const p=matches[0],receiptNonce=random();message.updateSource({[`flags.${M}.spiritualScarDamage`]:{nonce:p.claim.nonce,scopeId:p.scopeId,receiptNonce}});p.creations??=new Set();p.creations.add(receiptNonce);
 }
 function capture(message,_options,userId){const proof=message.flags?.[M]?.spiritualScarDamage,p=live.get(proof?.scopeId);if(!p||p.state!=='native'||userId!==p.userId||author(message)!==p.userId||!p.creations?.delete(proof.receiptNonce))return;if(p.messageId&&p.messageId!==message.id)p.multiple=true;else p.messageId=message.id;}
 async function afterDamage(p,{applied}){if(!plans.has(p))return;try{await rpc('complete',{scopeId:p.scopeId,snapshot:p.snapshot,nonce:p.claim.nonce,applied})}finally{live.delete(p.scopeId);plans.delete(p)}}
 function register({Hooks,socket:api}={}){
  if(installation)return;closed=false;socket=api;const ids=[];const on=(event,fn)=>ids.push([event,Hooks.on(event,fn)]);installation={Hooks,ids};
  on('preUpdateItem',(item,changes,options,userId)=>useLedger.preparePayment(item,changes,options,userId));on('updateItem',(item,changes,options,userId)=>useLedger.observePayment(item,changes,options,userId));on('preCreateChatMessage',(...args)=>{try{return decorateUse(...args)}catch(error){onError(error);return false}});on('preCreateChatMessage',decorateDamage);on('createChatMessage',capture);
  disposeObserver=nativeAdapter.addNativeObserver(observe,{matches:(_actor,params)=>plans.has(params?.[brand])});
  const handle=(name,fn)=>socket?.register(`scar:${name}`,async function(payload){try{return {ok:true,value:await fn(payload,game.users.get(this.socketdata.userId))}}catch(error){return {ok:false,error:error.message}}});
  for(const[name,fn]of [['request',request],['native',beginNative],['complete',complete],['pay',payLocal]])handle(name,fn);
  handle('scope',(payload,user)=>{if(user?.id!==game.users.activeGM?.id)throw Error('只有主GM可核验原生伤害范围。');return scopeProof(payload,game.user)});
  handle('choice',async(payload,user)=>{const actor=await fromUuid(payload.actorUuid);if(user?.id!==game.users.activeGM?.id||!owned(actor,game.user))throw Error('精神伤痕选择来源或拥有者无效。');return show({title:payload.title,choices:validateNativeChoices(payload.choices)})});
  followup.register?.({Hooks});
 }
 function unregister(){closed=true;for(const cancel of [...pendingWaits])cancel();disposeObserver?.();followup.unregister?.();if(installation)for(const[event,id]of installation.ids)installation.Hooks.off(event,id);installation=null;live.clear();authorizations.clear()}
 return {ready,handlesActor,resolveAction,requiresActualUse:item=>!!resolveAction(item),tracksFrequency:item=>!!resolveAction(item),beforeUse,captureUsage,executeUsage,beforeDamage,wrapNativeDamage,afterDamage,register,unregister};
}
