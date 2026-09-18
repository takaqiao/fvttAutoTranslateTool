import {getNativeActionEvents} from './native-action-events.mjs';
import {roaringOwnTurn,ROARING_APPLAUSE_SOURCE} from './roaring-applause-rules.mjs';
import {isActiveGM} from './native-context.mjs';
import {projectRoaringConditions} from './roaring-lifecycle.mjs';
import {SerialActions} from './runtime.mjs';

const PROOF='roaring-sustain:proof',RECORD='roaring-sustain:record';
const copy=structuredClone,values=c=>Array.from(c?.values?.()??c??[]);
const canonical=v=>JSON.stringify(v,(_k,x)=>x&&typeof x==='object'&&!Array.isArray(x)?Object.fromEntries(Object.keys(x).sort().map(k=>[k,x[k]])):x);
const same=(a,b)=>canonical(a)===canonical(b),bounded=s=>typeof s==='string'&&/^[A-Za-z0-9-]{1,80}$/.test(s)&&!['constructor','prototype'].includes(s);
const author=m=>m?.author?.id??m?.user?.id??m?.user;
const demand=(v,m)=>{if(!v)throw Error(`轰然喝彩维持：${m}`)};
const escape=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function chooseSource({choices}){
 return globalThis.foundry.applications.api.DialogV2.wait({window:{title:'维持哪个持续效果？'},content:'<p>本次仍使用原生维持动作。轰然喝彩须由主GM另行确认这次维持已完成或被打断。</p>',buttons:[
  ...choices.map((c,i)=>({action:`source-${i}`,label:escape(c.label),callback:()=>c.value})),
  {action:'manual',label:'其他持续效果（手工处理）',callback:()=>'manual'},
  {action:'cancel',label:'取消',callback:()=>null}],rejectClose:false});
}

/** Subscribe after Prayer. A branded native card is only a Use fact; only an
 * active-GM verdict renews/ends its exact source. No resource or Dismiss API. */
export function createRoaringSustain({game,fromUuid=globalThis.fromUuid,provider,actionEvents=getNativeActionEvents({game}),choose=chooseSource,onError=()=>{},randomId=()=>globalThis.crypto.randomUUID()}={}){
 demand(provider?.listSources&&provider?.lookupSource&&provider?.applyLifecycleEvent,'缺少来源查询和生命周期接口。');
 const scopes=new Map(),queue=new SerialActions(),hooks=[];let installed=false,socket,Hooks,unsubscribe;
 const sources=()=>provider.listSources();
 const find=nonce=>{const rows=sources().filter(x=>x.record.state.sourceNonce===nonce);demand(rows.length===1,'来源不存在或不唯一。');return rows[0]};
 const original=record=>{const uuid=record.state.source.originalMessageUuid,m=game.messages?.get(uuid.split('.').at(-1));return m?.uuid===uuid?m:null};
 function sourceValid(found,{active=true,renew=true}={}){
  const {actor,record:r}=found,s=r.state,m=original(r),proof=m&&provider.lookupSource(m);
  demand(installed&&r.context.gmId===game.users.activeGM?.id&&game.users.activeGM?.active&&s.source.sourceId===ROARING_APPLAUSE_SOURCE&&s.source.rank===3&&s.source.targetActorUuid===actor.uuid,'来源或主GM已改变。');
  if(active){demand(s.status==='active'&&r.context.immunity?.spell===false&&proof?.sourceNonce===s.sourceNonce&&proof.castNonce===s.source.castNonce,'该来源当前须人工核对。');if(renew)demand(s.timing.mode==='exact'&&!s.manualReview,'当前时长或结果不能自动续期。');}
  demand(m&&m.blind===false&&m.whisper?.length===0,'原卡不再是准确公开消息。');return m;
 }
 function actorValid(actor,token,user,{before=false}={}){
  demand(user?.active&&game.users.get(user.id)===user&&actor?.type==='character'&&!actor.isToken&&game.actors?.get(actor.id)===actor&&actor.testUserPermission?.(user,'OWNER')===true,'需要准确角色及原操作者所有权。');
  demand(token?.actor===actor&&token.documentName==='Token'&&game.scenes?.get(token.parent?.id)?.tokens?.get(token.id)===token,'原施法者Token已改变。');
  if(before)demand(actor.canAct===true&&actor.isDead!==true,'无法开始新的维持动作。');
 }
 const turnOf=(actor,token)=>roaringOwnTurn({game,actor,token});
 function actionValid(scope){
  demand(scope.user===game.user&&scope.action===game.pf2e?.actions?.get('sustain')&&scope.action.slug==='sustain'&&scope.variant.slug==='sustain'&&scope.action.cost===1&&scope.variant.cost===1&&same(scope.action.traits,['concentrate'])&&same(scope.variant.traits,['concentrate'])&&!scope.variant.effect&&(scope.params.effect===undefined||scope.params.effect===false)&&!scope.params.traits?.length,'修改过的维持活动请手工处理。');
 }
 function nativeCard(message,actor,token,userId){
  const C=globalThis.CONFIG?.ChatMessage?.documentClass;
  demand(C&&message instanceof C&&message.id&&game.messages?.get(message.id)===message&&author(message)===userId&&message.uuid===`ChatMessage.${message.id}`&&message.rolls?.length===0&&message.blind===false&&Array.isArray(message.whisper)&&!message.whisper.length,'缺少本次真实公开维持卡；草稿、取消和未知发布需手工核对。');
  demand(message.speaker?.actor===actor.id&&message.speaker.token===token.id&&message.speaker.scene===token.parent.id,'维持卡操作者或Token不符。');
 }
 const identity=s=>({useNonce:s.useNonce,sourceNonce:s.sourceNonce,source:copy(s.source),sourceRevision:s.sourceRevision,userId:s.user.id,gmId:s.gmId,turn:copy(s.turn),finiteEnvelope:copy(s.finiteEnvelope),messageUuid:s.message.uuid});
 function localValid(s){
  demand(installed&&scopes.get(s.useNonce)===s&&s.user===game.user&&s.gmId===game.users.activeGM?.id,'原生Use scope或主GM已失效。');actionValid(s.scope);
  const f=find(s.sourceNonce);sourceValid(f);demand(same(f.record.state.source,s.source)&&f.record.state.revision===s.sourceRevision,'维持期间来源已改变。');
  actorValid(s.actor,s.token,s.user);demand(same(turnOf(s.actor,s.token),s.turn)&&game.time.worldTime===s.finiteEnvelope.start.value,'原生Use准确回合或时间已改变。');
  demand(Array.isArray(s.result)&&s.result.length===1&&s.result[0].actor===s.actor&&!s.result[0].effect&&s.result[0].message===s.message&&!s.priorMessages.has(s.message?.id),'原生返回未证明该角色的新单次维持。');nativeCard(s.message,s.actor,s.token,s.user.id);
 }
 async function prove(p,sender){
  demand(sender===game.users.activeGM?.id,'只有当前主GM可请求原始Use证明。');
  const s=scopes.get(p?.useNonce);demand(s&&s.sourceNonce===p.sourceNonce,'不存在本次真实Use scope。');localValid(s);return identity(s);
 }
 async function recordUse(p,sender){
  demand(installed&&isActiveGM(game)&&bounded(p?.useNonce)&&bounded(p.sourceNonce),'只有主GM可登记准确Use。');
  return queue.run(p.sourceNonce,async()=>{
   const f=find(p.sourceNonce);sourceValid(f);const user=game.users.get(sender),actor=await fromUuid(f.record.state.source.casterActorUuid);
   demand(user?.active&&actor?.testUserPermission?.(user,'OWNER')===true,'原操作者无此来源权限。');
   const response=sender===game.user.id?{ok:true,value:await prove(p,sender)}:await socket.executeAsUser(PROOF,sender,p);
   const i=response?.value;demand(installed&&isActiveGM(game)&&response?.ok&&i?.useNonce===p.useNonce&&i.sourceNonce===p.sourceNonce&&i.userId===sender&&i.gmId===game.user.id,'原客户端没有确认本次Use。');
   const latest=find(p.sourceNonce);sourceValid(latest);demand(same(i.source,latest.record.state.source)&&i.sourceRevision===latest.record.state.revision,'原生Use来源版本已改变。');
   const token=await fromUuid(i.source.casterTokenUuid),message=await fromUuid(i.messageUuid);actorValid(actor,token,user);nativeCard(message,actor,token,sender);
   demand(installed&&isActiveGM(game)&&game.user.id===i.gmId&&same(turnOf(actor,token),i.turn)&&game.time.worldTime===i.finiteEnvelope?.start?.value,'登记前主GM、回合或时间已改变。');
   const ready=find(p.sourceNonce);sourceValid(ready);demand(same(ready.record.state,latest.record.state),'登记前来源修订已改变。');
   const out=await provider.applyLifecycleEvent({actor:latest.actor,nonce:p.sourceNonce,event:{type:'sustain-use',useNonce:i.useNonce,invocationId:i.useNonce,userId:sender,messageUuid:i.messageUuid,turn:copy(i.turn),finiteEnvelope:copy(i.finiteEnvelope)}});
   demand(out?.source?.sustainUses?.[i.useNonce]?.messageUuid===i.messageUuid,'本次Use未保存，不能声称已维持。');return {useNonce:i.useNonce};
  });
 }
 async function observe(scope,next){
  if(!installed||scope.action!==game.pf2e?.actions?.get('sustain')||scope.slug!=='sustain'||scope.params.message?.create===false)return next();
  if(game.world?.id!=='ujx5r8oipw7ercdr'||game.system?.version!=='8.5.1')return next();
  const candidates=sources().filter(({record:r})=>scope.actors.some(a=>a.uuid===r.state.source.casterActorUuid)&&r.state.status==='active');
  if(!candidates.length)return next();
  demand(scope.actors.length===1&&scope.user===game.user,'多个角色或不明操作者的维持请手工处理。');
  actionValid(scope);
  const selected=await choose({choices:candidates.map(({actor,record:r})=>({value:r.state.sourceNonce,label:`轰然喝彩 · ${actor.name??'本次目标'}`}))});
  if(selected===null||selected===undefined||selected===false)return undefined;if(selected==='manual')return next();
  demand(candidates.some(x=>x.record.state.sourceNonce===selected),'所选来源不在本次列表。');
  const found=find(selected);sourceValid(found);const actor=scope.actors[0],token=await fromUuid(found.record.state.source.casterTokenUuid),user=game.user;
  actorValid(actor,token,user,{before:true});demand(actor.uuid===found.record.state.source.casterActorUuid,'所选来源不属于本次角色。');
  actionValid(scope);sourceValid(found);const fresh=find(selected);sourceValid(fresh);demand(same(fresh.record.state,found.record.state),'准备维持期间来源已改变。');
  const turn=turnOf(actor,token),useNonce=randomId(),worldTime=game.time.worldTime;demand(bounded(useNonce)&&!scopes.has(useNonce)&&!Object.hasOwn(found.record.state.sustainUses,useNonce)&&Number.isFinite(worldTime),'Use nonce或时间无效。');
  const state=found.record.state;demand(worldTime>=state.clockHighWater&&worldTime<state.hardStopAt&&(turn.lastTurnEnd===null||turn.lastTurnEnd<state.timing.deadline.endRound),'来源已到期或时间倒退，请手工核对。');
  const s={scope,priorMessages:new Set(values(game.messages).map(m=>m.id)),useNonce,sourceNonce:selected,source:copy(found.record.state.source),sourceRevision:found.record.state.revision,actor,token,user,gmId:game.users.activeGM.id,turn:copy(turn),finiteEnvelope:{start:{value:worldTime,initiative:turn.order.find(c=>c.id===turn.combatantId).initiative},duration:{value:1,unit:'rounds',expiry:'turn-end',sustained:false}}};
  // Native throws retain their identity. After it returns, failures only report
  // uncertainty: never replay, refund, or replace its successful return value.
  const result=await next();s.result=result;s.message=result?.[0]?.message;scopes.set(useNonce,s);
  try{localValid(s);const p={useNonce,sourceNonce:selected},r=isActiveGM(game)?{ok:true,value:await recordUse(p,user.id)}:await socket.executeAsUser(RECORD,s.gmId,p);demand(r?.ok,'主GM未确认Use登记，请手工核对。');}
  catch(error){try{onError(error)}catch{/* A notice cannot replay or replace native Use. */}}finally{scopes.delete(useNonce)}return result;
 }
 async function adjudicate({sourceNonce,useNonce,terminal,expectedUse}){
  demand(installed&&isActiveGM(game),'只有当前主GM可裁定维持。');
  if(terminal==='unknown')return null;demand(['completed','disrupted'].includes(terminal)&&bounded(useNonce),'裁定或Use标识无效。');
  return queue.run(sourceNonce,async()=>{
   demand(installed&&isActiveGM(game),'主GM已改变。');const f=find(sourceNonce);sourceValid(f,{renew:terminal==='completed'});const use=f.record.state.sustainUses[useNonce];demand(use,'缺少已保存原生Use。');
   if(expectedUse!==undefined)demand(same(use,expectedUse),'这张原卡控件对应的Use已改变，请重新核对。');
   if(use.verdict){demand(use.verdict.terminal===terminal&&use.verdict.gmId===game.user.id,'该Use已有不同终态，需人工核对。');return null;}
   const actor=await fromUuid(f.record.state.source.casterActorUuid),token=await fromUuid(f.record.state.source.casterTokenUuid),message=await fromUuid(use.messageUuid);
   demand(actor&&token,'原始施法者结构已丢失。');nativeCard(message,actor,token,use.userId);
   demand(installed&&isActiveGM(game)&&f.record.context.gmId===game.user.id,'主GM已改变。');
   const ready=find(sourceNonce);sourceValid(ready,{renew:terminal==='completed'});demand(same(ready.record.state.sustainUses[useNonce],use),'裁定前准确Use已改变。');
   return provider.applyLifecycleEvent({actor:f.actor,nonce:sourceNonce,event:{type:'sustain-settled',useNonce,terminal,verdictId:randomId(),gmId:game.user.id}});
  });
 }
 async function endFascination(sourceNonce){
  demand(installed&&isActiveGM(game),'只有当前主GM可结束本源迷魂。');return queue.run(sourceNonce,async()=>{
   const f=find(sourceNonce);sourceValid(f,{active:false});demand(isActiveGM(game)&&f.record.state.status==='active'&&f.record.context.immunity?.spell===false&&f.record.context.immunity?.fascinated===false&&projectRoaringConditions(f.record.state).fascinated,'本源没有尚存的迷魂记录。');
   return provider.applyLifecycleEvent({actor:f.actor,nonce:sourceNonce,event:{type:'end-fascination',receiptId:randomId(),gmId:game.user.id}});
  });
 }
 function render(message,html){
  const root=html?.[0]??html;root?.querySelector?.('[data-roaring-controls]')?.remove();if(!installed||!root?.ownerDocument)return;
  const found=sources().filter(x=>original(x.record)===message);if(found.length!==1||message.blind!==false||message.whisper?.length)return;
  const {actor,record:r}=found[0],s=r.state,doc=root.ownerDocument,box=doc.createElement('section');box.dataset.roaringControls='';
  const add=(tag,text)=>{const e=doc.createElement(tag);e.textContent=text;box.append(e);return e};
  add('p',`轰然喝彩：${s.status==='ended'?'已结束':s.status==='active'?'生效中':'等待首次豁免'}。${s.manualReview||s.timing.mode!=='exact'?'本源结果或时长需GM人工核对。':''}`);
  if(r.context.immunity?.spell===false&&projectRoaringConditions(s).noReactions)add('p','禁反应为本源规则事实；此处尚未机械拦截其他反应入口。');
  if(r.context.immunity?.spell===true)add('p','目标对此法术免疫；此来源不施加条件或禁反应。');
  const stored=r.effects.children?.fascinated,remaining=stored&&actor.items?.get(stored.id);
  if(s.tombstones.fascinated&&remaining&&typeof stored.uuid==='string'&&remaining.uuid===stored.uuid)add('p','迷魂已标记结束，但条件尚未清除，请GM手工核对。');
  if(s.status==='ended'&&r.effects.status!=='ended'&&r.effects.status!=='immune')add('p','来源已逻辑结束，但效果清理尚未确认，请GM手工核对。');
  const gm=isActiveGM(game)&&r.context.gmId===game.user.id;
  const button=(label,fn)=>{const b=add('button',label);b.type='button';b.addEventListener('click',event=>{event.preventDefault();event.stopPropagation();b.disabled=true;Promise.resolve().then(fn).catch(onError).finally(()=>{b.disabled=false})});};
  for(const use of Object.values(s.sustainUses)){
   const a=add('a',`维持使用：第${use.turn.round}轮（${use.status==='use-recorded'?'待GM裁定':use.status==='completed'?'已完成':'已打断'}）`);a.className='content-link';a.dataset.uuid=use.messageUuid;
   if(gm&&r.context.immunity?.spell===false&&s.status==='active'&&!use.verdict){
    for(const [terminal,label]of [['completed','确认这次维持已完成'],['disrupted','确认这次维持已打断'],['unknown','暂不裁定']])if(terminal!=='completed'||s.timing.mode==='exact'&&!s.manualReview)button(label,()=>adjudicate({sourceNonce:s.sourceNonce,useNonce:use.useNonce,terminal,expectedUse:copy(use)}));
   }
  }
  if(gm&&s.status==='active'&&r.context.immunity?.spell===false&&r.context.immunity?.fascinated===false&&projectRoaringConditions(s).fascinated)button('结束本源迷魂',()=>endFascination(s.sourceNonce));
  (root.querySelector('.message-content')??root).append(box);
 }
 function refresh(actor){
  if(!installed)return;
  for(const f of sources().filter(x=>x.actor===actor)){const m=original(f.record);if(m)Promise.resolve().then(()=>installed&&globalThis.ui?.chat?.updateMessage?.(m,{notify:false})).catch(onError)}
 }
 function register({Hooks:api,socket:rpc}){
  if(installed)return cleanup;installed=true;Hooks=api;socket=rpc;
  unsubscribe=actionEvents.addMiddleware(observe);actionEvents.register();
  const on=(n,fn)=>hooks.push([n,Hooks.on(n,fn)]);
  on('renderChatMessageHTML',render);on('updateActor',refresh);on('deleteItem',item=>refresh(item.actor));
  for(const [name,fn]of [[PROOF,prove],[RECORD,recordUse]])socket.register(name,async function(p){try{return {ok:true,value:await fn(p,this.socketdata?.userId)}}catch(e){return {ok:false,error:e.message}}});
  return cleanup;
 }
 function cleanup(){installed=false;unsubscribe?.();unsubscribe=null;for(const[n,id]of hooks.splice(0))Hooks.off(n,id);scopes.clear();}
 return {register,cleanup,adjudicate,endFascination,diagnostic:()=>({installed,activeScopes:scopes.size,automaticCompletion:false})};
}
