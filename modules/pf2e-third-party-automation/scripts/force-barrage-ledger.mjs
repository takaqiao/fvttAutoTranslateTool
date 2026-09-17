import {MODULE_ID as ID} from './rules.mjs';
import {getSourceId,isActiveGM} from './native-context.mjs';

export const FORCE_BARRAGE_SOURCE='Compendium.pf2e.spells-srd.Item.gKKqvLohtrSJj3BM';
const PATH=`flags.${ID}.forceBarrage`,terminal=new Set(['cancelled','rejected','disrupted','delivered','uncertain']);
const copy=value=>structuredClone(value),validId=value=>typeof value==='string'&&/^[A-Za-z0-9_-]{1,100}$/.test(value)&&!['__proto__','constructor','prototype'].includes(value);
const ordered=value=>Array.isArray(value)?value.map(ordered):value&&typeof value==='object'?Object.fromEntries(Object.keys(value).sort().map(k=>[k,ordered(value[k])])):value;
const equal=(a,b)=>JSON.stringify(ordered(a))===JSON.stringify(ordered(b));
const requireTrue=(value,reason)=>{if(!value)throw Error(reason?`力场飞弹：${reason}`:'力场飞弹的来源、原生付款或当前执行许可无法确认。');};
const author=message=>message?.author?.id??message?.user?.id??message?.user;
const stateOf=actor=>actor?.flags?.[ID]?.forceBarrage??{currentByItem:{},operations:{}};
const nativeReceipts=actor=>actor.flags?.[ID]?.nativeCasts??[];

/** The caller supplies the original evaluated native DamageRoll JSON. DSN adds
 * only these two confirmed presentation fields. In particular, damage-instance
 * type/flavor and Roll.options.type remain part of the mechanical witness. */
export function forceBarrageRollWitness(rollJSON){
 requireTrue(rollJSON?.class==='DamageRoll'&&rollJSON.evaluated===true&&typeof rollJSON.formula==='string'&&rollJSON.formula.length>0&&Number.isFinite(rollJSON.total)&&rollJSON.total>=0&&Array.isArray(rollJSON.terms)&&rollJSON.terms.length>0);
 requireTrue(JSON.stringify(rollJSON).length<=100000);
 const clean=(value,result=false)=>{
  if(Array.isArray(value))return value.map(v=>clean(v,result));
  if(!value||typeof value!=='object')return value;
  const term=typeof value.class==='string'&&!value.class.endsWith('Roll')&&value.class!=='DamageInstance';
  return Object.fromEntries(Object.entries(value).filter(([k])=>!(result&&k==='indexThrow')).map(([k,v])=>{
   if(k==='options'&&term&&v&&typeof v==='object')return[k,clean(Object.fromEntries(Object.entries(v).filter(([name])=>name!=='type')))];
   return[k,clean(v,k==='results')];
  }));
 };
 return clean(copy(rollJSON));
}

/** Durable permissions, not a resource or dice executor. Every mutation uses the
 * native Cast resource queue, but no caller continuation runs inside that lock. */
export function createForceBarrageLedger({game,fromUuid=globalThis.fromUuid,withActorResourceLock,randomId=()=>globalThis.crypto.randomUUID()}={}){
 requireTrue(game&&typeof fromUuid==='function'&&typeof withActorResourceLock==='function');
 const read=(actor,nonce)=>{const r=stateOf(actor).operations?.[nonce];return r?copy(r):null;};
 const current=(actor,item)=>read(actor,stateOf(actor).currentByItem?.[item?.id]);
 const gm=()=>requireTrue(game.user?.isGM===true&&isActiveGM(game));
 function source({actor,item,entry,user}){
  gm();requireTrue(game.world?.id==='ujx5r8oipw7ercdr'&&game.system?.id==='pf2e'&&game.system.version==='8.5.1');
  requireTrue(actor?.type==='character'&&!actor.isToken&&game.actors?.get(actor.id)===actor);
  requireTrue(item?.type==='spell'&&item.actor===actor&&actor.items?.get(item.id)===item&&getSourceId(item)===FORCE_BARRAGE_SOURCE);
  requireTrue(entry?.type==='spellcastingEntry'&&entry.actor===actor&&actor.items.get(entry.id)===entry&&item.system?.location?.value===entry.id&&item.system.location.signature===true&&item.system.level?.value===1);
  requireTrue(entry.system?.prepared?.value==='spontaneous'&&entry.system.tradition?.value==='occult');
  requireTrue(user?.active===true&&game.users?.get(user.id)===user&&actor.testUserPermission?.(user,'OWNER')===true);
 }
 const canAct=actor=>requireTrue(actor.canAct===true&&actor.isDead!==true);
 async function resolve(scope){source(scope);for(const doc of [scope.actor,scope.item,scope.entry])requireTrue(await fromUuid(doc.uuid)===doc);source(scope);}
 function recordFor(scope){
  source(scope);requireTrue(validId(scope.nonce));const state=stateOf(scope.actor),r=state.operations?.[scope.nonce];
  requireTrue(state.currentByItem?.[scope.item.id]===scope.nonce&&r?.nonce===scope.nonce&&r.actorUuid===scope.actor.uuid&&r.itemUuid===scope.item.uuid&&r.entryUuid===scope.entry.uuid&&r.userId===scope.user.id&&r.gmId===game.users.activeGM.id);
  if(scope.fingerprint!==undefined)requireTrue(scope.fingerprint===r.fingerprint);
  return r;
 }
 function allocationShape(allocation){
  requireTrue(allocation&&[1,2,3].includes(allocation.rank)&&[1,2,3].includes(allocation.actions)&&typeof allocation.sourceTokenUuid==='string');
  requireTrue(Array.isArray(allocation.targets)&&allocation.targets.length>0&&allocation.targets.length<=100);
  requireTrue(allocation.targets.every(t=>t&&typeof t.targetUuid==='string'&&Number.isSafeInteger(t.count)&&t.count>=0));
  requireTrue(new Set(allocation.targets.map(t=>t.targetUuid)).size===allocation.targets.length);
  const sum=allocation.targets.reduce((n,t)=>n+t.count,0);requireTrue(Number.isSafeInteger(sum)&&sum>0);
 }
 async function allocationLive(scope,allocation){
  allocationShape(allocation);const sourceToken=await fromUuid(allocation.sourceTokenUuid),scene=sourceToken?.parent;
  requireTrue(sourceToken?.documentName==='Token'&&sourceToken.actor===scope.actor&&sourceToken.actorLink===true&&game.scenes?.get(scene?.id)===scene&&scene.tokens?.get(sourceToken.id)===sourceToken);
  for(const {targetUuid}of allocation.targets){const t=await fromUuid(targetUuid);requireTrue(t?.documentName==='Token'&&t.parent===scene&&scene.tokens.get(t.id)===t&&['character','npc','familiar'].includes(t.actor?.type));}
 }
 const slots=(entry,rank)=>entry.system.slots?.[`slot${rank}`]?.value;
 async function save(scope,state){
  source(scope);const expected=copy(state),result=await scope.actor.update({[PATH]:expected});
  requireTrue(result===scope.actor);await resolve(scope);requireTrue(equal(stateOf(scope.actor),expected));
  const nonce=expected.currentByItem[scope.item.id];recordFor({...scope,nonce});return read(scope.actor,nonce);
 }
 const mutate=(scope,run)=>withActorResourceLock(scope.actor,async()=>{await resolve(scope);return run();});
 const replace=(scope,record)=>{const state=copy(stateOf(scope.actor));state.operations[record.nonce]=copy(record);return save(scope,state);};
 const claim=scope=>mutate(scope,async()=>{
  const {actor,item,user,entry,invocationId,fingerprint,allocation}=scope;canAct(actor);
  requireTrue(validId(invocationId)&&/^[a-f0-9]{64}$/.test(fingerprint??''));await allocationLive(scope,allocation);source(scope);canAct(actor);
  requireTrue(Number.isInteger(slots(entry,allocation.rank))&&slots(entry,allocation.rank)>0);
  const state=copy(stateOf(actor)),records=Object.values(state.operations);
  const prior=records.find(r=>r.invocationId===invocationId);
  if(prior){requireTrue(prior.status==='claimed'&&prior.fingerprint===fingerprint&&equal(prior.allocation,allocation));recordFor({...scope,nonce:prior.nonce});return copy(prior);}
  const active=state.operations[state.currentByItem[item.id]];requireTrue(!active||terminal.has(active.status));
  const nonce=randomId();requireTrue(validId(nonce)&&!Object.hasOwn(state.operations,nonce));
  state.currentByItem[item.id]=nonce;state.operations[nonce]={nonce,status:'claimed',invocationId,fingerprint,actorUuid:actor.uuid,itemUuid:item.uuid,entryUuid:entry.uuid,sourceId:FORCE_BARRAGE_SOURCE,userId:user.id,gmId:game.users.activeGM.id,allocation:copy(allocation),targets:allocation.targets.filter(t=>t.count>0).map(t=>({...copy(t),status:'not-started'}))};
  const saved=await save(scope,state);canAct(actor);return saved;
 });
 const cancelClaim=scope=>mutate(scope,()=>{const r=recordFor(scope);requireTrue(r.status==='claimed');return replace(scope,{...r,status:'cancelled'});});
 const startCast=scope=>mutate(scope,async()=>{const r=recordFor(scope);requireTrue(r.status==='claimed');canAct(scope.actor);await allocationLive(scope,r.allocation);recordFor(scope);canAct(scope.actor);requireTrue(Number.isInteger(slots(scope.entry,r.allocation.rank))&&slots(scope.entry,r.allocation.rank)>0);const saved=await replace(scope,{...r,status:'casting'});canAct(scope.actor);return saved;});
 function assertCastIntent(scope){
  const r=recordFor(scope);canAct(scope.actor);requireTrue(r.status==='casting');requireTrue(Number.isInteger(slots(scope.entry,r.allocation.rank))&&slots(scope.entry,r.allocation.rank)>0);
  requireTrue(!nativeReceipts(scope.actor).some(p=>p.invocation?.kind==='force-barrage'&&p.invocation.data?.bridgeNonce===r.nonce));return true;
 }
 function nativeBinding(scope,outcome){
  const r=recordFor(scope),m=outcome?.message,castNonce=outcome?.castNonce;
  requireTrue(outcome?.status==='completed'&&validId(castNonce)&&m?.id&&game.messages?.get(m.id)===m&&author(m)===r.userId&&m.rolls?.length===0);
  const matches=nativeReceipts(scope.actor).filter(p=>p.id===castNonce),input=outcome.input;requireTrue(matches.length===1);const p=matches[0];
  requireTrue(p.state==='used'&&p.messageId===m.id&&p.userId===r.userId&&p.invocation?.kind==='force-barrage'&&p.invocation.gmId===r.gmId&&p.invocation.data?.bridgeNonce===r.nonce&&p.invocation.data.fingerprint===r.fingerprint);
  requireTrue(nativeReceipts(scope.actor).filter(v=>v.invocation?.kind==='force-barrage'&&v.invocation.data?.bridgeNonce===r.nonce).length===1);
  const expected={actorUuid:r.actorUuid,itemUuid:r.itemUuid,entryUuid:r.entryUuid,sourceId:r.sourceId,rank:r.allocation.rank,slotId:null,focusPoints:0,overlayIds:[]};
  const sameInput=value=>value&&Object.entries(expected).every(([k,v])=>equal(value[k],v));
  requireTrue(sameInput(p)&&sameInput(input)&&sameInput(m.flags?.[ID]?.nativeCastInput)&&equal(p,outcome.receipt));
  const payment=p.slotCommit;requireTrue(payment?.castNonce===castNonce&&payment.itemUuid===r.itemUuid&&payment.entryUuid===r.entryUuid&&payment.rank===r.allocation.rank&&Number.isInteger(payment.before)&&payment.before>0&&payment.cost===1&&payment.after===payment.before-1&&payment.userId===r.userId&&payment.gmId===r.gmId);
  requireTrue(p.nativeCastScope?.castNonce===castNonce&&p.nativeCastScope.tokenUuid===r.allocation.sourceTokenUuid);
  const proof=m.flags?.[ID]?.nativeCast,pf=m.flags?.pf2e?.origin;
  requireTrue(proof?.id===castNonce&&proof.actorUuid===r.actorUuid&&proof.itemUuid===r.itemUuid&&proof.userId===r.userId&&pf?.uuid===r.itemUuid&&pf.castRank===r.allocation.rank&&(!pf.actor||pf.actor===r.actorUuid));
  const tokenParts=r.allocation.sourceTokenUuid.split('.');requireTrue(m.speaker?.actor===scope.actor.id&&m.speaker.scene===tokenParts[1]&&m.speaker.token===tokenParts[3]&&m.blind===false&&Array.isArray(m.whisper)&&m.whisper.length===0);
  return {castNonce,originalMessageUuid:m.uuid,nativeInput:copy(input),slotCommit:copy(payment)};
 }
 const bindCast=scope=>mutate(scope,()=>{const r=recordFor(scope),binding=nativeBinding(scope,scope.outcome);if(['paid','producing','delivered'].includes(r.status)){requireTrue(Object.entries(binding).every(([k,v])=>equal(r[k],v)));return copy(r);}requireTrue(r.status==='casting');return replace(scope,{...r,...binding,status:'paid'});});
 function targetFor(scope){
  const r=recordFor(scope);requireTrue(['paid','producing','delivered'].includes(r.status));const t=r.targets.find(t=>t.targetUuid===scope.targetUuid);requireTrue(t?.count>0);
  const receipt=nativeReceipts(scope.actor).find(p=>p.id===r.castNonce),message=game.messages?.get(r.originalMessageUuid?.split('.').at(-1));
  const binding=nativeBinding(scope,{status:'completed',castNonce:r.castNonce,input:r.nativeInput,receipt,message});requireTrue(Object.entries(binding).every(([k,v])=>equal(r[k],v)));
  return {r,t};
 }
 async function liveTarget(scope,r){
  const sourceToken=await fromUuid(r.allocation.sourceTokenUuid),target=await fromUuid(scope.targetUuid),scene=sourceToken?.parent;
  requireTrue(sourceToken?.documentName==='Token'&&sourceToken.actor===scope.actor&&sourceToken.actorLink===true&&game.scenes?.get(scene?.id)===scene&&scene.tokens?.get(sourceToken.id)===sourceToken);
  requireTrue(target?.documentName==='Token'&&target.parent===scene&&scene.tokens.get(target.id)===target&&['character','npc','familiar'].includes(target.actor?.type));recordFor(scope);
 }
 function replaceTarget(scope,r,t,changes){
  const targets=r.targets.map(target=>target.targetUuid===t.targetUuid?{...target,...changes}:target),status=targets.every(target=>target.status==='published')?'delivered':'producing';
  return replace(scope,{...r,status,targets});
 }
 const startTarget=scope=>mutate(scope,async()=>{const {r,t}=targetFor(scope);requireTrue(t.status==='not-started');await liveTarget(scope,r);return replaceTarget(scope,r,t,{status:'rolling'});});
 const recordRoll=scope=>mutate(scope,()=>{
  const {r,t}=targetFor(scope),witness=forceBarrageRollWitness(scope.rollJSON);
  if(['rolled','publishing','published'].includes(t.status)){requireTrue(equal(t.rollWitness,witness));return copy(r);}
  requireTrue(t.status==='rolling');return replaceTarget(scope,r,t,{status:'rolled',rollJSON:copy(scope.rollJSON),rollWitness:witness});
 });
 const beginPublication=scope=>mutate(scope,async()=>{const {r,t}=targetFor(scope);requireTrue(t.status==='rolled');await liveTarget(scope,r);return replaceTarget(scope,r,t,{status:'publishing'});});
 const finishPublication=scope=>mutate(scope,async()=>{
  const {r,t}=targetFor(scope),m=scope.message;requireTrue(['publishing','published'].includes(t.status));
  requireTrue(equal(t.rollWitness,forceBarrageRollWitness(scope.rollJSON)),'原掷骰回执已改变，未确认交付。');
  requireTrue(m?.id&&game.messages?.get(m.id)===m,'准确伤害卡尚未同步到主GM，未确认交付。');
  requireTrue(m.documentName==='ChatMessage'&&m.isDamageRoll===true&&author(m)===r.userId&&m.rolls?.length===1&&typeof m.rolls[0]?.toJSON==='function','伤害卡类型、作者或原生骰子无法确认。');
  requireTrue(equal(t.rollWitness,forceBarrageRollWitness(m.rolls[0].toJSON())),'主GM伤害卡的原生骰子与已记录结果不同。');
  const expected={bridgeNonce:r.nonce,castNonce:r.castNonce,originalMessageUuid:r.originalMessageUuid,targetUuid:t.targetUuid,count:t.count,fingerprint:r.fingerprint},proof=m.flags?.[ID]?.forceBarrage;
  requireTrue(equal(proof,expected),'伤害卡的本次分弹标记不匹配。');
  const pf=m.flags?.pf2e?.origin,parts=r.allocation.sourceTokenUuid.split('.');
  requireTrue(pf?.uuid===r.itemUuid&&pf.castRank===r.allocation.rank&&(!pf.actor||pf.actor===r.actorUuid)&&m.speaker?.actor===scope.actor.id&&m.speaker.scene===parts[1]&&m.speaker.token===parts[3]);
  requireTrue(equal(m.flags?.['pf2e-toolbelt']?.targetHelper?.targets,[t.targetUuid])&&m.blind===false&&Array.isArray(m.whisper)&&m.whisper.length===0);
  requireTrue(!r.targets.some(target=>target.targetUuid!==t.targetUuid&&target.messageUuid===m.uuid));
  if(t.status==='published'){requireTrue(t.messageUuid===m.uuid);return copy(r);}
  await liveTarget(scope,r);return replaceTarget(scope,r,t,{status:'published',messageUuid:m.uuid});
 });
 const uncertain=scope=>mutate(scope,()=>{const r=recordFor(scope);if(r.status==='uncertain')return copy(r);requireTrue(!terminal.has(r.status));return replace(scope,{...r,status:'uncertain',reason:String(scope.reason??'未知结果').slice(0,500)});});
 const finishWithoutDamage=scope=>mutate(scope,()=>{
  const r=recordFor(scope),o=scope.outcome;requireTrue(['casting','disrupted'].includes(r.status)&&o?.status==='disrupted'&&o.message===null&&validId(o.castNonce));
  const matches=nativeReceipts(scope.actor).filter(p=>p.id===o.castNonce);requireTrue(matches.length===1);const p=matches[0];
  requireTrue(nativeReceipts(scope.actor).filter(p=>p.invocation?.kind==='force-barrage'&&p.invocation.data?.bridgeNonce===r.nonce).length===1);
  requireTrue(p.state==='disrupted'&&!p.messageId&&p.userId===r.userId&&p.itemUuid===r.itemUuid&&p.actorUuid===r.actorUuid&&p.entryUuid===r.entryUuid&&p.sourceId===r.sourceId&&p.rank===r.allocation.rank&&p.invocation?.kind==='force-barrage'&&p.invocation.gmId===r.gmId&&p.invocation.data?.bridgeNonce===r.nonce&&p.invocation.data.fingerprint===r.fingerprint&&equal(p,o.receipt));
  requireTrue(p.nativeCastScope?.castNonce===o.castNonce&&p.nativeCastScope.tokenUuid===r.allocation.sourceTokenUuid);
  const payment=p.slotCommit;requireTrue(payment?.castNonce===o.castNonce&&payment.itemUuid===r.itemUuid&&payment.entryUuid===r.entryUuid&&payment.rank===r.allocation.rank&&payment.userId===r.userId&&payment.gmId===r.gmId&&payment.cost===1&&Number.isInteger(payment.before)&&payment.before>0&&payment.after===payment.before-1);
  if(r.status==='disrupted'){requireTrue(r.castNonce===o.castNonce&&equal(r.slotCommit,payment));return copy(r);}
  return replace(scope,{...r,status:'disrupted',castNonce:o.castNonce,slotCommit:copy(payment)});
 });
 return {read,current,claim,cancelClaim,startCast,assertCastIntent,bindCast,startTarget,recordRoll,beginPublication,finishPublication,uncertain,finishWithoutDamage};
}
