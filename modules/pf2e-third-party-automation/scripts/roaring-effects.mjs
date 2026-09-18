import {MODULE_ID as ID} from './rules.mjs';
import {getSourceId,isActiveGM} from './native-context.mjs';
import {SerialActions} from './runtime.mjs';
import {projectRoaringConditions} from './roaring-lifecycle.mjs';

const PATH=`flags.${ID}.roaringApplause.sources`;
const CONDITIONS={slowed:'Compendium.pf2e.conditionitems.Item.xYTAsEpcJE1Ccni3',fascinated:'Compendium.pf2e.conditionitems.Item.AdPVz7rbaVSRxHFg'};
const copy=value=>structuredClone(value),values=c=>Array.from(c?.values?.()??c??[]);
const ordered=v=>Array.isArray(v)?v.map(ordered):v&&typeof v==='object'?Object.fromEntries(Object.keys(v).sort().map(k=>[k,ordered(v[k])])):v;
const equal=(a,b)=>JSON.stringify(ordered(a))===JSON.stringify(ordered(b));
const demand=(ok,reason)=>{if(!ok)throw Error(`轰然喝彩：${reason}`)};
const safe=v=>typeof v==='string'&&/^[A-Za-z0-9-]{1,80}$/.test(v)&&!['constructor','prototype'].includes(v);
const table=actor=>actor?.flags?.[ID]?.roaringApplause?.sources??{};
const marker=item=>item?.flags?.[ID]?.roaringEffect;

/** Source-local documents only. Native grants supply condition mechanics and
 * cascade behavior; operation receipts prevent repeated creation after doubt. */
export function createRoaringEffects({game,fromUuid=globalThis.fromUuid,randomId=()=>globalThis.foundry?.utils?.randomID?.()??globalThis.crypto.randomUUID(),onError=()=>{}}={}){
 const queue=new SerialActions();
 const get=(actor,nonce)=>{const r=safe(nonce)?table(actor)[nonce]:null;return r?copy(r):null};
 const list=actor=>Object.entries(table(actor)).filter(([nonce,r])=>safe(nonce)&&r?.state?.sourceNonce===nonce).map(([,r])=>copy(r));
 function gm(){demand(game.user?.isGM===true&&isActiveGM(game),'需要当前主GM处理来源效果。')}
 async function live(actor){gm();demand(actor?.uuid&&await fromUuid(actor.uuid)===actor,'目标角色已改变。');gm()}
 function valid(actor,r){
  demand(r?.schema===1&&Number.isSafeInteger(r.revision)&&r.revision>=0&&safe(r.state?.sourceNonce)&&r.state.source?.targetActorUuid===actor.uuid,'来源记录不完整。');
  projectRoaringConditions(r.state);
  const i=r.context?.immunity;
  demand(r.context?.paymentId===r.state.source.castNonce&&typeof r.context.userId==='string'&&typeof r.context.gmId==='string'&&Number.isFinite(r.context.dc),'施法及DC记录不完整。');
  demand(i?.checked===true&&i.systemVersion==='8.5.1'&&['spell','slowed','fascinated'].every(k=>typeof i[k]==='boolean'),'免疫结果尚未确认。');
 }
 function current(actor,nonce){const r=get(actor,nonce);valid(actor,r);return r}
 async function persist(actor,before,after){
  await live(actor);valid(actor,after);
  demand(equal(get(actor,after.state.sourceNonce),before),'来源记录在操作期间已改变。');
  const result=await actor.update({[`${PATH}.${after.state.sourceNonce}`]:copy(after)});
  await live(actor);demand(result===actor&&equal(get(actor,after.state.sourceNonce),after),'来源记录未完整保存，请核对本次操作。');return copy(after);
 }
 const change=(actor,before,patch)=>persist(actor,before,{...before,...copy(patch),revision:before.revision+1});
 const run=(actor,fn)=>queue.run(actor.uuid,async()=>{await live(actor);return fn()});
 async function claim({actor,state,context}){return run(actor,async()=>{
  const before=get(actor,state?.sourceNonce);
  const next={schema:1,revision:0,state:copy(state),context:copy(context),effects:{status:'not-started',operationId:null,parentId:null,children:{slowed:null,fascinated:null},reason:null}};
  valid(actor,next);
  if(before){demand(equal(before.state,state)&&equal(before.context,context),'同一来源已经推进，不能重新认领。');return before}
  return persist(actor,null,next);
 })}
 async function saveState({actor,nonce,state,expectedRevision}){return run(actor,async()=>{
  const before=current(actor,nonce);
  demand(before.revision===expectedRevision&&state.sourceNonce===nonce&&equal(state.source,before.state.source)&&state.completedWorldTime===before.state.completedWorldTime&&state.hardStopAt===before.state.hardStopAt&&state.clockHighWater>=before.state.clockHighWater&&state.revision>=before.state.revision,'来源修订或时间范围已改变。');
  demand(before.state.status!=='ended'||state.status==='ended','已结束的来源不能重新生效。');
  if(equal(before.state,state))return before;
  return change(actor,before,{state});
 })}
 function proofFor(r){return {schema:1,sourceNonce:r.state.sourceNonce,castNonce:r.state.source.castNonce,operationId:r.effects.operationId,targetActorUuid:r.state.source.targetActorUuid,originalMessageUuid:r.state.source.originalMessageUuid}}
 function grantKey(r,role){return `roaring${role}${r.state.sourceNonce.replaceAll('-','')}`}
 function desired(r){const p=projectRoaringConditions(r.state),i=r.context.immunity;return {projection:p,roles:i.spell?[]:['slowed','fascinated'].filter(role=>!!p[role]&&!i[role])}}
 // PF2e's ItemAlteration preparation adds fromEquipment:true. Declare the
 // native default so the complete alteration remains strictly comparable.
 function expectedRules(r){return desired(r).roles.map(role=>({key:'GrantItem',uuid:CONDITIONS[role],flag:grantKey(r,role),allowDuplicate:true,inMemoryOnly:false,reevaluateOnUpdate:false,onDeleteActions:{granter:'cascade',grantee:'detach'},...(role==='slowed'?{alterations:[{mode:'override',property:'badge-value',value:1,fromEquipment:true}]}:{})}))}
 function ownParent(actor,r,item,{deleted=false}={}){
  return item?.type==='effect'&&item.actor===actor&&(deleted||actor.items.get(item.id)===item)&&item.id===r.effects.parentId&&equal(marker(item),proofFor(r))&&item.system?.slug===`tpa-roaring-${r.state.sourceNonce.toLowerCase()}`&&item.system.context?.origin?.actor===r.state.source.casterActorUuid&&item.system.context.origin.token===r.state.source.casterTokenUuid&&item.system.context.origin.item===r.state.source.itemUuid;
 }
 function ownChild(actor,r,role,item,{deleted=false}={}){
  const stored=r.effects.children[role],parent=actor.items.get(r.effects.parentId),grant=parent?.flags?.pf2e?.itemGrants?.[stored?.grantKey];
  return !!stored&&item?.type==='condition'&&item.actor===actor&&item.id===stored.id&&item.uuid===stored.uuid&&getSourceId(item)===CONDITIONS[role]&&item.flags?.pf2e?.grantedBy?.id===r.effects.parentId&&item.flags.pf2e.grantedBy.onDelete==='cascade'&&(deleted||actor.items.get(item.id)===item&&ownParent(actor,r,parent)&&grant?.id===item.id&&grant.onDelete==='detach');
 }
 function identifyOwnedItem(item,{deleted=false}={}){
  const actor=item?.actor;if(!actor)return null;
  for(const r of list(actor)){
   if(ownParent(actor,r,item,{deleted}))return {actor,nonce:r.state.sourceNonce,condition:null,isParent:true,itemUuid:item.uuid};
   for(const condition of ['slowed','fascinated'])if(ownChild(actor,r,condition,item,{deleted}))return {actor,nonce:r.state.sourceNonce,condition,isParent:false,itemUuid:item.uuid};
  }
  return null;
 }
 function verifyParentRules(r,parent,{cleanup=false}={}){
  const wanted=r.effects.rules;
  demand(Array.isArray(wanted)&&Array.isArray(parent.system?.rules)&&parent.system.rules.length===wanted.length,'来源效果规则已改变。');
  for(let n=0;n<wanted.length;n++)for(const [key,value]of Object.entries(wanted[n]))demand(equal(parent.system.rules[n][key],value),'来源效果授予规则已改变。');
  if(parent.system.rules.some(rule=>rule.ignored===true)){
   // Native finite expiry disables prepared rules without editing their source.
   // This permits cleanup only, never a fresh mechanical grant or renewal.
   const raw=parent._source?.system?.rules;
   demand(cleanup&&r.state.timing.mode==='manual-finite'&&parent.isExpired===true&&equal(parent.system.duration,r.state.timing.finiteEnvelope.duration)&&equal(parent.system.start,r.state.timing.finiteEnvelope.start)&&Array.isArray(raw)&&raw.length===wanted.length&&raw.every(rule=>rule.ignored!==true),'来源效果授予规则已停用。');
   for(let n=0;n<wanted.length;n++)for(const [key,value]of Object.entries(wanted[n]))demand(equal(raw[n][key],value),'来源效果原始授予规则已改变。');
  }
 }
 function grantReceipt(actor,r,parent,{observe=false,cleanup=false}={}){
  verifyParentRules(r,parent,{cleanup});
  const children={slowed:null,fascinated:null};
  for(const rule of r.effects.rules){
   const role=Object.keys(CONDITIONS).find(role=>CONDITIONS[role]===rule.uuid),grant=parent.flags?.pf2e?.itemGrants?.[rule.flag],child=actor.items.get(grant?.id);
   demand(role&&safe(grant?.id)&&grant.onDelete==='detach','原生条件授予关系未确认。');
   demand(observe&&!child||child?.type==='condition'&&child.actor===actor&&getSourceId(child)===rule.uuid&&child.flags?.pf2e?.grantedBy?.id===parent.id&&child.flags.pf2e.grantedBy.onDelete==='cascade','原生条件授予关系未确认。');
   children[role]={id:grant.id,uuid:child?.uuid??`${actor.uuid}.Item.${grant.id}`,grantKey:rule.flag};
  }
  demand(Object.keys(parent.flags?.pf2e?.itemGrants??{}).length===r.effects.rules.length,'来源效果出现额外授予关系。');return children;
 }
 function effectData(r){
  const s=r.state,source=s.source;
  return {name:'轰然喝彩',type:'effect',img:'systems/pf2e/icons/spells/roaring-applause.webp',flags:{[ID]:{roaringEffect:proofFor(r),roaringTiming:copy(s.timing),roaringRestriction:{noReactions:true,automaticConsumers:false},fascinationSubject:desired(r).projection.subject}},system:{slug:`tpa-roaring-${s.sourceNonce.toLowerCase()}`,level:{value:3},description:{value:'<p>本次轰然喝彩的来源效果。禁反应与鼓掌触发需按提示处理；维持是否完成由GM裁定。</p>'},traits:{value:['emotion','mental'],rarity:'common'},duration:{value:-1,unit:'unlimited',expiry:null,sustained:false},start:copy(s.timing.finiteEnvelope.start),context:{origin:{actor:source.casterActorUuid,token:source.casterTokenUuid,item:source.itemUuid}},rules:copy(r.effects.rules)}};
 }
 async function uncertain(actor,r,error){
  const latest=get(actor,r.state.sourceNonce);
  if(latest&&equal(latest,r)&&isActiveGM(game))try{await change(actor,r,{effects:{...r.effects,status:'uncertain',reason:String(error.message??error).slice(0,300)}})}catch(saveError){onError(saveError)}
  throw error;
 }
 async function observeCreated(actor,r){
  demand(['creating','uncertain'].includes(r.effects.status)&&safe(r.effects.operationId)&&Array.isArray(r.effects.rules),'没有可核对的原始效果创建操作。');
  const matching=values(actor.items).filter(item=>equal(marker(item),proofFor(r)));
  demand(matching.length===1,'原始效果创建没有唯一的持久结果，不会重试。');
  const candidate={...r,effects:{...r.effects,parentId:matching[0].id}};
  demand(ownParent(actor,candidate,matching[0]),'原始效果创建的持久内容未确认。');
  const children=grantReceipt(actor,candidate,matching[0],{observe:true,cleanup:r.state.status==='ended'}),state=copy(r.state);
  // The native grant receipt survives manual child deletion. Preserve its
  // absence in the same durable write; observing never repeats a native create.
  if(state.status!=='ended')for(const [role,child]of Object.entries(children))if(child&&!actor.items.has(child.id)&&!state.tombstones[role]){
   state.tombstones[role]={receiptId:r.effects.operationId,itemUuid:child.uuid,reason:'original-grant-now-absent'};state.revision++;
  }
  return change(actor,r,{state,effects:{...candidate.effects,status:'created',children,reason:'observed-original-operation'}});
 }
 async function materialize({actor,nonce}){return run(actor,async()=>{
  let r=current(actor,nonce);
  if(['creating','uncertain'].includes(r.effects.status))r=await observeCreated(actor,r);
  if(['ended','immune'].includes(r.effects.status))return r;
  if(r.effects.status==='created'){
   demand(ownParent(actor,r,actor.items.get(r.effects.parentId)),'原来源效果已不存在，请核对人工删除。');return r;
  }
  demand(r.effects.status==='not-started','本次效果创建结果不确定，不会重复创建。');
  if(r.state.status==='ended')return change(actor,r,{effects:{...r.effects,status:'ended'}});
  if(r.state.status!=='active')return r;
  if(r.context.immunity.spell)return change(actor,r,{effects:{...r.effects,status:'immune'}});
  demand(!projectRoaringConditions(r.state).manualReview,'当前来源需要人工核对，不能首次自动创建效果。');
  const operationId=randomId();demand(safe(operationId),'效果操作标记无效。');
  r=await change(actor,r,{effects:{...r.effects,status:'creating',operationId,rules:expectedRules(r)}});
  try{
   const data=effectData(r),returned=await actor.createEmbeddedDocuments('Item',[data]);await live(actor);
   demand(equal(current(actor,nonce),r),'来源在效果创建期间已改变。');
   const matching=values(actor.items).filter(item=>equal(marker(item),proofFor(r)));
   demand(matching.length===1&&Array.isArray(returned)&&returned.includes(matching[0]),'本次原生效果创建未返回准确的持久文档。');
   const parent=matching[0],candidate={...r,effects:{...r.effects,parentId:parent.id}};
   demand(ownParent(actor,candidate,parent),'原生来源效果内容未确认。');
   const children=grantReceipt(actor,candidate,parent);
   return change(actor,r,{effects:{...candidate.effects,status:'created',children}});
  }catch(error){return uncertain(actor,r,error)}
 })}
 function safeCascade(actor,r,parent){
  demand(ownParent(actor,r,parent),'原来源效果身份已改变。');verifyParentRules(r,parent,{cleanup:true});
  for(const [key,grant]of Object.entries(parent.flags?.pf2e?.itemGrants??{})){
   const role=Object.keys(r.effects.children).find(role=>r.effects.children[role]?.grantKey===key&&r.effects.children[role]?.id===grant.id);
   demand(role&&grant.onDelete==='detach','来源效果挂接了未确认的条件，不能级联删除。');
   const child=actor.items.get(grant.id);if(child)demand(ownChild(actor,r,role,child),'条件已经改属其他来源，不能级联删除。');
  }
 }
 async function end({actor,nonce}){return run(actor,async()=>{
  let r=current(actor,nonce);demand(r.state.status==='ended','来源仍在持续，不能免费解除整个法术。');
  if(r.effects.status==='ended')return r;
  if(['not-started','immune'].includes(r.effects.status))return change(actor,r,{effects:{...r.effects,status:'ended'}});
  if(['creating','uncertain'].includes(r.effects.status))r=await observeCreated(actor,r);
  demand(['created','deleting'].includes(r.effects.status),'无法证明原来源效果的创建结果，请手工清理。');
  const parent=actor.items.get(r.effects.parentId);
  if(parent){
   demand(r.effects.status==='created','本次原生删除结果尚未确认，不会重复删除。');safeCascade(actor,r,parent);
   r=await change(actor,r,{effects:{...r.effects,status:'deleting'}});
   const result=await actor.deleteEmbeddedDocuments('Item',[parent.id]);await live(actor);
   demand(Array.isArray(result)&&result.some(item=>item===parent)&&!actor.items.has(parent.id),'原生效果删除未完成。');
  }
  for(const child of Object.values(r.effects.children))if(child)demand(!actor.items.has(child.id),'本来源仍有未清除的授予条件。');
  return change(actor,r,{effects:{...r.effects,status:'ended'}});
 })}
 async function endFascination({actor,nonce}){return run(actor,async()=>{
  const r=current(actor,nonce);demand(r.state.tombstones.fascinated,'尚未记录本来源迷魂的解除事实。');
  const info=r.effects.children.fascinated,child=info?actor.items.get(info.id):null;if(!child)return r;
  demand(ownChild(actor,r,'fascinated',child),'迷魂条件的归属已改变。');
  const result=await actor.deleteEmbeddedDocuments('Item',[child.id]);await live(actor);
  demand(Array.isArray(result)&&result.includes(child)&&!actor.items.has(child.id),'本来源迷魂删除未完成。');return current(actor,nonce);
 })}
 async function timing({actor,nonce},finite){return run(actor,async()=>{
  const r=current(actor,nonce);if(['not-started','immune','ended'].includes(r.effects.status))return r;
  demand(r.effects.status==='created','效果创建尚未确认。');
  const parent=actor.items.get(r.effects.parentId);demand(ownParent(actor,r,parent),'原来源效果已改变。');verifyParentRules(r,parent);
  const changes={[`flags.${ID}.roaringTiming`]:copy(r.state.timing)};
  if(finite){demand(r.state.timing.mode==='manual-finite','来源没有进入有限人工计时。');changes['system.start']=copy(r.state.timing.finiteEnvelope.start);changes['system.duration']=copy(r.state.timing.finiteEnvelope.duration);}
  const result=await parent.update(changes);await live(actor);demand(result===parent&&ownParent(actor,r,parent)&&equal(parent.flags?.[ID]?.roaringTiming,r.state.timing),'来源时长未完整保存。');
  if(finite)demand(equal(parent.system.start,r.state.timing.finiteEnvelope.start)&&equal(parent.system.duration,r.state.timing.finiteEnvelope.duration),'有限后备时长未确认。');
  return current(actor,nonce);
 })}
 return {claim,get,list,saveState,materialize,end,endFascination,renew:args=>timing(args,false),restoreFinite:args=>timing(args,true),identifyOwnedItem};
}
