import {MODULE_ID as ID} from './rules.mjs';
import {isActiveGM} from './native-context.mjs';
import {assessForceBarrageCast,validateForceBarrageTargets,validateForceBarrageAllocation} from './force-barrage-rules.mjs';
import {loadForceBarrageWorkbench} from './force-barrage-workbench-compat.mjs';
import {createForceBarrageLedger} from './force-barrage-ledger.mjs';

const RPC='force-barrage:ledger',PROOF='force-barrage:proof';
const operations=new Set(['claim','startCast','bindCast','startTarget','recordRoll','beginPublication','finishPublication','uncertain','finishWithoutDamage']);
const values=c=>Array.from(c?.values?.()??c??[]),copy=v=>structuredClone(v);
const bounded=v=>typeof v==='string'&&/^[A-Za-z0-9_-]{1,100}$/.test(v);
const ordered=v=>Array.isArray(v)?v.map(ordered):v&&typeof v==='object'?Object.fromEntries(Object.keys(v).sort().map(k=>[k,ordered(v[k])])):v;
const equal=(a,b)=>JSON.stringify(ordered(a))===JSON.stringify(ordered(b));
async function fingerprint(data){return [...new Uint8Array(await globalThis.crypto.subtle.digest('SHA-256',new TextEncoder().encode(JSON.stringify(ordered(data)))))].map(n=>n.toString(16).padStart(2,'0')).join('');}
const tokenDoc=t=>t?.document??t;
function originalContext(actor){
 const controlled=values(globalThis.canvas?.tokens?.controlled).map(tokenDoc).filter(t=>t.actor===actor);
 const candidates=values(actor.getActiveTokens?.(false,true)).map(tokenDoc);
 if(candidates.length!==1||controlled.length>1||controlled.length===1&&controlled[0]!==candidates[0])throw Error('需要当前场景唯一的本角色Token，再从原法术条目施放。');
 return {token:candidates[0],targets:values(globalThis.game?.user?.targets).map(tokenDoc)};
}
async function chooseAllocation({rank,targets,adapter}){
 const Dialog=globalThis.foundry?.applications?.api?.DialogV2;
 if(!Dialog)throw Error('无法打开本次施法的分弹选择。');
 const actions=await Dialog.wait({window:{title:`力场飞弹 · ${rank}环 · 动作数`},content:'<p>选择本次原始施法使用的动作数。</p>',buttons:[1,2,3].map(n=>({action:String(n),label:`${n}动作`,callback:()=>n})),rejectClose:false});
 if(![1,2,3].includes(actions))return null;
 const missiles=adapter.getMissileCount({rank,actions});
 const answer=await Dialog.wait({window:{title:`力场飞弹 · 分配${missiles}枚飞弹`},content:`<p>目标按本次选择顺序列出；确认后再支付法术位。</p>${targets.map((_t,i)=>`<label>目标 ${i+1} <input name="target${i}" type="number" min="0" max="${missiles}" step="1" value="${i?0:missiles}" required></label>`).join('')}<label><input name="visible" type="checkbox" required>施法者能看见所有所选目标（不以GM视角为准）</label>`,buttons:[{action:'cast',label:'确认分弹并施法',callback:(_event,button)=>{const data=new FormData(button.form);return {actions,visibilityConfirmed:data.has('visible'),allocations:targets.map((t,i)=>({targetUuid:t.uuid,count:data.get(`target${i}`)===''?NaN:Number(data.get(`target${i}`))}))};}},{action:'cancel',label:'取消施法',callback:()=>null}],rejectClose:false});
 return answer||null;
}

/** Call-local original Cast bridge. Native payment and Workbench arithmetic are
 * separate awaited capabilities; neither is inferred from a slot delta/card. */
export function createForceBarrageBridge({game,fromUuid=globalThis.fromUuid,nativeCasts,choose=chooseAllocation,loadWorkbench=loadForceBarrageWorkbench,assess=assessForceBarrageCast,validateTargets=validateForceBarrageTargets,getContext=originalContext,randomId=()=>globalThis.foundry?.utils?.randomID?.()??globalThis.crypto.randomUUID(),onError=()=>{},ledger=createForceBarrageLedger({game,fromUuid,withActorResourceLock:nativeCasts.withActorResourceLock})}={}){
 const scopes=new Map(),byItem=new Map();let socket,installed=false;
 function requireScope(s,beforePayment=false){
  if(scopes.get(s.invocationId)!==s||byItem.get(s.item.uuid)!==s||game.user!==s.user||!s.user.active||game.users.get(s.user.id)!==s.user||game.users.activeGM?.id!==s.gmId||game.actors.get(s.actor.id)!==s.actor||s.actor.items.get(s.item.id)!==s.item||s.actor.items.get(s.entry.id)!==s.entry||s.actor.testUserPermission(s.user,'OWNER')!==true)throw Error('本次力场飞弹的原操作者、物品或主GM已经改变；不会重试。');
  if(beforePayment){const result=assess({...s,item:s.castItem});if(!result.eligible)throw Error(result.reason??'本次力场飞弹已经不能施放。');}
 }
 const identity=s=>({invocationId:s.invocationId,actorUuid:s.actor.uuid,itemUuid:s.item.uuid,entryUuid:s.entry.uuid,userId:s.user.id,gmId:s.gmId,allocation:s.allocation});
 async function prove(payload,sender){
  const s=scopes.get(payload?.invocationId);
  if(!s||s.stage!=='claiming'||sender!==s.gmId||payload.fingerprint!==s.fingerprint||!equal(payload.identity,identity(s)))throw Error('缺少原客户端当前Cast的准确分弹证明。');
  requireScope(s,true);validateTargets({...s,visibilityConfirmed:true});
  if(await fingerprint(identity(s))!==s.fingerprint)throw Error('分弹选择已改变。');
  return {fingerprint:s.fingerprint,identity:identity(s)};
 }
 async function dispatch(payload,sender){
  if(!installed||!isActiveGM(game)||!operations.has(payload?.operation))throw Error('力场飞弹操作或主GM权限无效。');
  const user=game.users.get(sender),actor=await fromUuid(payload.actorUuid),item=await fromUuid(payload.itemUuid),entry=await fromUuid(payload.entryUuid);
  if(!user?.active||actor?.items?.get(item?.id)!==item||actor.items.get(entry?.id)!==entry||item?.actor!==actor||entry?.actor!==actor||actor.testUserPermission(user,'OWNER')!==true)throw Error('力场飞弹角色、物品或所有者无效。');
  const args={actor,item,entry,user,nonce:payload.nonce};
  if(payload.operation==='claim'){
   if(!bounded(payload.invocationId)||!/^[a-f0-9]{64}$/.test(payload.fingerprint??''))throw Error('分弹调用标记无效。');
   const allocation=copy(payload.allocation),proofInput={invocationId:payload.invocationId,fingerprint:payload.fingerprint,identity:{invocationId:payload.invocationId,actorUuid:actor.uuid,itemUuid:item.uuid,entryUuid:entry.uuid,userId:sender,gmId:game.user.id,allocation}};
   const reply=sender===game.user.id?await prove(proofInput,sender):await socket?.executeAsUser(PROOF,sender,proofInput),proof=sender===game.user.id?reply:reply?.ok?reply.value:null;
   if(!proof||!equal(proof,{fingerprint:payload.fingerprint,identity:proofInput.identity})||await fingerprint(proofInput.identity)!==payload.fingerprint)throw Error('原客户端的实际Cast未确认。');
   const token=await fromUuid(allocation.sourceTokenUuid),targets=await Promise.all(allocation.targets.map(t=>fromUuid(t.targetUuid)));
   const check=assess({game,actor,item,entry,user,options:{rank:allocation.rank,messageMode:'public'}});if(!check.eligible)throw Error(check.reason??'法术条件已改变。');
   validateTargets({game,actor,token,targets,visibilityConfirmed:true});
   const adapter=await loadWorkbench({game,fromUuid});validateForceBarrageAllocation({targets,allocations:allocation.targets,missiles:adapter.getMissileCount(allocation)});
   return ledger.claim({...args,invocationId:payload.invocationId,fingerprint:payload.fingerprint,allocation});
  }
  if(!bounded(payload.nonce))throw Error('分弹回执无效。');
  if(['bindCast','finishWithoutDamage'].includes(payload.operation))return ledger[payload.operation]({...args,outcome:{...payload.outcome,message:payload.outcome.messageUuid?await fromUuid(payload.outcome.messageUuid):null}});
  if(payload.operation==='finishPublication')return ledger.finishPublication({...args,targetUuid:payload.targetUuid,message:await fromUuid(payload.messageUuid),rollJSON:payload.rollJSON});
  if(['startTarget','recordRoll','beginPublication'].includes(payload.operation))return ledger[payload.operation]({...args,targetUuid:payload.targetUuid,...payload.operation==='recordRoll'?{rollJSON:payload.rollJSON}:{}});
  if(payload.operation==='uncertain')return ledger.uncertain({...args,reason:String(payload.reason??'结果不确定').slice(0,500)});
  return ledger[payload.operation](args);
 }
 async function call(s,operation,extra={}){
  requireScope(s,['claim','startCast'].includes(operation));
  const payload={operation,actorUuid:s.actor.uuid,itemUuid:s.item.uuid,entryUuid:s.entry.uuid,nonce:s.nonce,invocationId:s.invocationId,fingerprint:s.fingerprint,...extra};
  const local=isActiveGM(game),reply=local?await dispatch(payload,s.user.id):await socket?.executeAsUser(RPC,s.gmId,payload),result=local?reply:reply?.ok?reply.value:null;
  if(!result)throw Error(reply?.error??'分弹操作回执未确认；不会自动重试。');
  requireScope(s,false);return result;
 }
 const transportOutcome=o=>({status:o.status,castNonce:o.castNonce,input:o.input,receipt:o.receipt,messageUuid:o.message?.uuid??null});
 async function interceptCast({item:castItem,entry,options={}},next){
  const actor=castItem.actor,assessment=assess({game,actor,item:castItem,entry,user:game.user,options});if(!assessment.handled)return next();
  if(!assessment.eligible)throw Error(assessment.reason??'当前施法需要人工处理。');
  const item=assessment.base??castItem.original??castItem;
  if(byItem.has(item.uuid))throw Error('这个法术已有一次分弹选择或施法正在处理。');
  const mode=options.messageMode??game.settings?.get('core','messageMode');if(mode!=='public')throw Error('当前分弹接线只支持公开施法；请先选择公开模式。');
  const s={game,actor,item,castItem,entry,options,user:game.user,gmId:game.users.activeGM?.id,rank:assessment.rank,invocationId:randomId(),stage:'choosing'};
  if(!bounded(s.invocationId))throw Error('本次施法标记无效。');scopes.set(s.invocationId,s);byItem.set(item.uuid,s);
  try{
   if(!isActiveGM(game)&&!socket)throw Error('无法连接当前主GM。');
   Object.assign(s,getContext(actor));
   const adapter=await loadWorkbench({game,fromUuid}),answer=await choose({...s,adapter});if(!answer)return;
   requireScope(s,true);if(![1,2,3].includes(answer.actions)||answer.visibilityConfirmed!==true)throw Error('动作数或施法者可见性未确认。');
   validateTargets({...s,visibilityConfirmed:true});
   const allocations=validateForceBarrageAllocation({targets:s.targets,allocations:answer.allocations,missiles:adapter.getMissileCount({rank:s.rank,actions:answer.actions})});
   s.allocation={rank:s.rank,actions:answer.actions,sourceTokenUuid:s.token.uuid,targets:allocations};s.fingerprint=await fingerprint(identity(s));s.stage='claiming';
   const claimed=await call(s,'claim',{allocation:s.allocation});if(!bounded(claimed.nonce))throw Error('分弹认领回执无效。');s.nonce=claimed.nonce;
   let originalOutcome;
   await adapter.run({actor,token:s.token,item:castItem,entry,rank:s.rank,actions:answer.actions,allocations:allocations.map(a=>({...a,targetToken:s.targets.find(t=>t.uuid===a.targetUuid)})),bridge:{
    payAndBindOriginalCast:async()=>{
     requireScope(s,true);validateTargets({...s,visibilityConfirmed:true});await call(s,'startCast');s.stage='casting';
     if(typeof next.withOutcome!=='function')throw Error('原生Cast准确回执接口尚未就绪。');
     const outcome=originalOutcome=await next.withOutcome({kind:'force-barrage',data:{bridgeNonce:s.nonce,fingerprint:s.fingerprint,sourceTokenUuid:s.token.uuid,messageMode:'public'}});
     if(outcome?.status==='disrupted'){await call(s,'finishWithoutDamage',{outcome:transportOutcome(outcome)});s.stage='disrupted';throw Error('本次原生施法已被打断，未生成分弹伤害。');}
     if(outcome?.status!=='completed')throw Error('原生Cast未返回准确的付款及原卡完成证明。');
     s.record=await call(s,'bindCast',{outcome:transportOutcome(outcome)});s.stage='paid';return s.record;
    },
    publishTarget:async({roll,messageData,targetUuid})=>{
     requireScope(s);if(!s.record||!['paid','producing'].includes(s.stage))throw Error('分弹付款尚未绑定。');s.stage='producing';
     const allocation=allocations.find(a=>a.targetUuid===targetUuid);if(!allocation||allocation.count<1)throw Error('无效的分弹目标。');
     await call(s,'startTarget',{targetUuid});requireScope(s);await roll.evaluate();const rollJSON=roll.toJSON();await call(s,'recordRoll',{targetUuid,rollJSON});
     const flags={...messageData.flags};delete flags['pf2e-toolbelt.targetHelper.targets'];
     flags['pf2e-toolbelt']={...flags['pf2e-toolbelt'],targetHelper:{...flags['pf2e-toolbelt']?.targetHelper,targets:[targetUuid]}};
     flags.pf2e={...flags.pf2e,origin:{...flags.pf2e?.origin,uuid:item.uuid,actor:actor.uuid,type:'spell',castRank:s.rank}};
     flags[ID]={...flags[ID],forceBarrage:{bridgeNonce:s.nonce,castNonce:originalOutcome.castNonce,originalMessageUuid:originalOutcome.message.uuid,targetUuid,count:allocation.count,fingerprint:s.fingerprint}};
     await call(s,'beginPublication',{targetUuid});requireScope(s);
     s.publishing=flags[ID].forceBarrage;
     let message;try{message=await roll.toMessage({...messageData,flags,author:s.user.id,blind:false,whisper:[]},{messageMode:'public'});}finally{s.publishing=null;}
     if(!message?.uuid)throw Error('本次目标伤害卡没有返回准确消息。');
     s.record=await call(s,'finishPublication',{targetUuid,messageUuid:message.uuid,rollJSON});return message;
    },
   }});
   if(s.record?.status!=='delivered'||s.record.targets?.length!==allocations.filter(a=>a.count>0).length||s.record.targets.some(t=>t.status!=='published'||!t.messageUuid))throw Error('仍有分弹目标未确认交付；不会重发。');
   s.stage='delivered';return originalOutcome?.nativeResult;
  }catch(error){if(s.nonce&&!['delivered','disrupted'].includes(s.stage)){try{await call(s,'uncertain',{reason:String(error?.message??error)});}catch(recordError){onError(recordError);}}throw error;}
  finally{scopes.delete(s.invocationId);if(byItem.get(item.uuid)===s)byItem.delete(item.uuid);}
 }
 function captureUsage(item){const s=byItem.get(item?.uuid);return s?.stage==='casting'&&s.nonce?{forceBarrageCast:{bridgeNonce:s.nonce,fingerprint:s.fingerprint}}:null;}
 async function validateInvocation(context){
  const {actor,item:variant,entry,user,invocation,payload}=context,item=variant.original??variant,data=invocation?.data;
  if(invocation?.kind!=='force-barrage'||!bounded(data?.bridgeNonce)||!/^[a-f0-9]{64}$/.test(data.fingerprint??''))return false;
  ledger.assertCastIntent({actor,item,entry,user,nonce:data.bridgeNonce,fingerprint:data.fingerprint,castNonce:context.castNonce});
  const r=ledger.current(actor,item);if(r.nonce!==data.bridgeNonce||r.allocation.rank!==payload.rank||r.allocation.sourceTokenUuid!==data.sourceTokenUuid)return false;
  const assessment=assess({game,actor,item:variant,entry,user,options:{rank:payload.rank,messageMode:'public'}});if(!assessment.eligible)return false;
  const token=await fromUuid(r.allocation.sourceTokenUuid),targets=await Promise.all(r.allocation.targets.map(t=>fromUuid(t.targetUuid)));
  validateTargets({game,actor,token,targets,visibilityConfirmed:true});const adapter=await loadWorkbench({game,fromUuid});validateForceBarrageAllocation({targets,allocations:r.allocation.targets,missiles:adapter.getMissileCount(r.allocation)});return true;
 }
 async function consumePolicy(context,next){context.expectSlotCommit({before:context.entry.system.slots[`slot${context.payload.rank}`].value,cost:1,changes:()=>({})});return next();}
 function renderOriginal(message,html){
  const proof=message.flags?.[ID]?.forceBarrageCast,native=message.flags?.[ID]?.nativeCast;
  if(!bounded(proof?.bridgeNonce)||!native?.id||native.itemUuid!==message.flags?.pf2e?.origin?.uuid)return;
  for(const button of html.querySelectorAll?.('[data-action="spell-damage"]')??[]){button.disabled=true;button.removeAttribute('data-action');button.textContent='请按本次分弹卡结算';}
 }
 function preCreateDamage(message){
  const proof=message.flags?.[ID]?.forceBarrage;if(!proof)return;
  const s=byItem.get(message.flags?.pf2e?.origin?.uuid);
  try{
   if(!s||s.stage!=='producing'||!equal(s.publishing,proof))throw Error('这张分弹卡没有本次准确发布许可。');
   requireScope(s);const author=message.author?.id??message.user?.id??message.user??message.author;
   if(author!==s.user.id||message.blind!==false||message.whisper?.length!==0||message.speaker?.actor!==s.actor.id||message.flags.pf2e.origin.castRank!==s.rank||!equal(message.flags['pf2e-toolbelt']?.targetHelper?.targets,[proof.targetUuid]))throw Error('分弹消息在发布前改变了来源或隐私。');
  }catch(error){onError(error);return false;}
 }
 function register({Hooks,socket:api}={}){
  if(installed)return()=>{};installed=true;socket=api;
  nativeCasts.addInvocationAdapter('force-barrage',{validate:validateInvocation,consumePolicy});
  const hook=Hooks?.on('renderChatMessageHTML',renderOriginal);
  const publicationHook=Hooks?.on('preCreateChatMessage',preCreateDamage);
  socket?.register(PROOF,async function(payload){try{return {ok:true,value:await prove(payload,this.socketdata?.userId)}}catch(error){return {ok:false,error:String(error.message??error)}}});
  socket?.register(RPC,async function(payload){try{return {ok:true,value:await dispatch(payload,this.socketdata?.userId)}}catch(error){return {ok:false,error:String(error.message??error)}}});
  return()=>{installed=false;if(hook!==undefined)Hooks.off('renderChatMessageHTML',hook);if(publicationHook!==undefined)Hooks.off('preCreateChatMessage',publicationHook);};
 }
 return {interceptCast,captureUsage,register};
}
