import {MODULE_ID as ID} from './rules.mjs';
import {isActiveGM} from './native-context.mjs';
import {runCheckReactionPipeline} from './reaction-checks.mjs';
import {assessHalflingLuck,isHalflingLuckItem} from './halfling-luck-rules.mjs';
import {createHalflingLuckLedger} from './halfling-luck-ledger.mjs';

const ACTION='halfling-luck:use',RPC='halfling-luck:ledger',PROOF='halfling-luck:proof';
const values=c=>Array.from(c?.values?.()??c??[]);
const bounded=s=>typeof s==='string'&&/^[A-Za-z0-9_-]{1,100}$/.test(s);
const operations=new Set(['claim','cancelClaim','startRolling','recordResult','beginDelivery','finishDelivery','uncertain']);
const canonical=value=>JSON.stringify(value,(_key,v)=>v instanceof Set?[...v].sort():v&&typeof v==='object'&&!Array.isArray(v)?Object.fromEntries(Object.keys(v).sort().map(k=>[k,v[k]])):v);
async function sha256(value){return [...new Uint8Array(await globalThis.crypto.subtle.digest('SHA-256',new TextEncoder().encode(canonical(value))))].map(b=>b.toString(16).padStart(2,'0')).join('');}

/** One live failed Check invocation, one original daily Use, one final delivery.
 * No reaction reservation and no retrospective lookup of a previous check card.
 */
export function createHalflingLuckProvider({game,fromUuid=globalThis.fromUuid,choose,originalUse,ledger=createHalflingLuckLedger({game,fromUuid}),assess=assessHalflingLuck,randomId=()=>globalThis.foundry?.utils?.randomID?.()??globalThis.crypto.randomUUID(),onError=()=>{},publish}={}){
 const scopes=new Map(),authorizations=new Map(),waiters=new Map();let socket,installed=false;
 const feature=a=>values(a?.items).find(isHalflingLuckItem);
 const handlesActor=a=>a?.type==='character'&&!a.isToken&&game.actors?.get(a.id)===a&&!!feature(a);
 const resolveAction=i=>isHalflingLuckItem(i)&&handlesActor(i.actor)&&i.actor.items.get(i.id)===i?ACTION:undefined;
 const live=(s,canAct=true)=>handlesActor(s.actor)&&s.actor.items.get(s.item.id)===s.item&&game.users.get(s.user.id)===s.user&&s.user.active&&s.actor.testUserPermission(s.user,'OWNER')===true&&(!canAct||s.actor.canAct===true&&s.actor.isDead!==true);
 function requireScope(s,canAct=true){if(!live(s,canAct)||game.user.id!==s.user.id||game.users.activeGM?.id!==s.gmId)throw Error('半身人幸运的原操作者、能力或主GM已改变；不会重试本次使用。');}
 function evaluate(s){return assess({game,actor:s.actor,item:s.item,user:s.user,check:s.check,...s.captured,requestedCreateMessage:s.requestedCreateMessage});}
 function fingerprintData(s){
  const c=s.captured.context;
  return {invocationId:s.invocationId,actorUuid:s.actor.uuid,itemUuid:s.item.uuid,userId:s.user.id,gmId:s.gmId,
   tokenUuid:c.token?.uuid??(c.origin?.self?c.origin.token?.uuid:c.target?.self?c.target.token?.uuid:null)??null,
   check:{slug:s.check.slug,modifiers:values(s.check.modifiers).map(m=>({slug:m.slug,modifier:m.modifier,type:m.type,enabled:m.enabled,ignored:m.ignored}))},
   context:{type:c.type,domains:c.domains,options:c.options,dc:c.dc,isReroll:c.isReroll,rollTwice:c.rollTwice,substitutions:c.substitutions,messageMode:c.messageMode},
   roll:s.captured.roll.toJSON(),outcome:s.captured.outcome,card:s.captured.card.toObject()};
 }
 async function prove(payload,sender){
  if(sender!==game.users.activeGM?.id)throw Error('只有当前主GM可以验证本次检定。');
  const s=scopes.get(payload?.invocationId);
  if(!s||s.stage!=='claiming'||payload.proofNonce!==s.invocationId||payload.actorUuid!==s.actor.uuid||payload.itemUuid!==s.item.uuid||payload.userId!==s.user.id||payload.gmId!==s.gmId||payload.fingerprint!==s.fingerprint)throw Error('半身人幸运的实时检定证明不匹配。');
  requireScope(s);
  if(!evaluate(s)?.eligible||await sha256(fingerprintData(s))!==s.fingerprint)throw Error('原生失败检定已改变，未支付半身人幸运。');
  return {invocationId:s.invocationId,proofNonce:s.invocationId,actorUuid:s.actor.uuid,itemUuid:s.item.uuid,userId:s.user.id,gmId:s.gmId,fingerprint:s.fingerprint};
 }
 async function dispatch(payload,sender){
  if(!isActiveGM(game)||!operations.has(payload?.operation))throw Error('半身人幸运请求或主GM权限无效。');
  const user=game.users.get(sender),actor=await fromUuid(payload.actorUuid),item=await fromUuid(payload.itemUuid);
  if(!user?.active||!handlesActor(actor)||item?.actor!==actor||actor.items.get(item.id)!==item||!isHalflingLuckItem(item)||actor.testUserPermission(user,'OWNER')!==true)throw Error('半身人幸运原物品或所有者权限无效。');
  // Never accept a user supplied in the payload as the authenticated sender.
  const args={actor,item,user,nonce:payload.nonce};
  if(payload.operation==='claim'){
   if(!bounded(payload.invocationId)||!/^[a-f0-9]{64}$/.test(payload.fingerprint??''))throw Error('检定证明格式无效。');
   const proofInput={invocationId:payload.invocationId,proofNonce:payload.invocationId,actorUuid:actor.uuid,itemUuid:item.uuid,userId:sender,gmId:game.user.id,fingerprint:payload.fingerprint};
   const proof=sender===game.user.id?await prove(proofInput,game.user.id):await socket?.executeAsUser(PROOF,sender,proofInput);
   const value=sender===game.user.id?proof:proof?.ok?proof.value:null;
   if(!value||Object.entries(proofInput).some(([k,v])=>value[k]!==v)||!isActiveGM(game))throw Error('没有原客户端实时失败证明，未认领。');
   return ledger.claim({...args,invocationId:payload.invocationId,fingerprint:payload.fingerprint});
  }
  if(!bounded(payload.nonce))throw Error('半身人幸运使用凭据无效。');
  if(payload.operation==='recordResult')return ledger.recordResult({...args,rollJSON:payload.rollJSON,outcome:payload.outcome});
  if(payload.operation==='uncertain')return ledger.uncertain({...args,reason:String(payload.reason??'未知传输结果').slice(0,500)});
  return ledger[payload.operation](args);
 }
 async function call(s,operation,extra={}){
  // Only starting an action/die requires current ability to act. Recording or
  // delivering an already evaluated result retains all identity/GM checks.
  const canAct=operation==='claim'||operation==='startRolling';
  requireScope(s,canAct);const local=isActiveGM(game);
  const payload={operation,actorUuid:s.actor.uuid,itemUuid:s.item.uuid,nonce:s.nonce,invocationId:s.invocationId,fingerprint:s.fingerprint,...extra};
  const reply=local?await dispatch(payload,game.user.id):await socket?.executeAsUser(RPC,s.gmId,payload);
  const value=local?reply:reply?.ok?reply.value:null;
  if(!value)throw Error(reply?.error??'半身人幸运主GM回执未确认；不会重试。');
  requireScope(s,canAct);return value;
 }
 function beforeUse(item,user=game.user){
  const s=authorizations.get(item?.uuid);if(!s)return true;
  requireScope(s);const r=ledger.current(item);
  if(user!==s.user||s.stage!=='paying'||r?.nonce!==s.nonce||!['claimed','paid','ready'].includes(r.status))throw Error('本次半身人幸运自动付款授权已失效。');
  return true;
 }
 function captureUsage(item){
  const s=authorizations.get(item?.uuid);if(!s)return null;
  beforeUse(item);const r=ledger.current(item);
  if(!['paid','ready'].includes(r?.status)||!bounded(r.paymentNonce))return null;
  return {halflingLuckInput:{nonce:s.nonce,paymentNonce:r.paymentNonce}};
 }
 function wake(item){
  const w=waiters.get(item?.uuid);if(!w)return;
  const r=ledger.current(item);
  if(r?.status==='ready'&&r.nonce===w.nonce&&bounded(r.paymentNonce))w.resolve(r);
 }
 async function executeUsage(ctx){
  if(!ctx.message?.flags?.[ID]?.halflingLuckInput)return '原生手工使用未绑定本次自动检定；不会追溯或重复投骰。';
  if(!isActiveGM(game))throw Error('半身人幸运付款卡只能由主GM核验。');
  const r=await ledger.bindUsage(ctx);wake(ctx.item);
  return r?.status==='ready'?'已绑定本次原生付款；等待原操作者完成检定。':'本次付款记录已处理，不会重新启动投骰。';
 }
 async function pay(s){
  requireScope(s);s.stage='paying';authorizations.set(s.item.uuid,s);
  ledger.authorizePayment(s.item,s.nonce,s.user);
  let timer;
  const ready=new Promise(resolve=>waiters.set(s.item.uuid,{nonce:s.nonce,resolve}));
  const deadline=new Promise((_,reject)=>{timer=setTimeout(()=>reject(Error('半身人幸运原生付款回执等待超时；请核对本次记录，不会重试。')),15000);});
  try{
   // Establish the waiter before original Use. Its returned card alone is not
   // evidence that the native frequency update and GM binding both committed.
   await Promise.race([(async()=>{await (originalUse?originalUse(s.item):game.pf2e.rollItemMacro(s.item.uuid));wake(s.item);return ready;})(),deadline]);
   requireScope(s);const r=ledger.current(s.item);
   if(r?.nonce!==s.nonce||r.status!=='ready'||!bounded(r.paymentNonce))throw Error('半身人幸运准确付款尚未确认。');
   s.paymentNonce=r.paymentNonce;
  }finally{clearTimeout(timer);waiters.delete(s.item.uuid);authorizations.delete(s.item.uuid);ledger.clearAuthorization(s.item,s.nonce);}
 }
 async function interceptCheck(native,check,context={},event=null,callback){
  const candidate=context.actor??(context.origin?.self?context.origin.actor:context.target?.self?context.target.actor:null),actor=game.actors?.get(candidate?.id);
  if(!handlesActor(actor)||actor.uuid!==candidate?.uuid||!game.users.activeGM?.id||!ledger||!isActiveGM(game)&&!socket)return native(check,context,event,callback);
  const invocationId=randomId();if(!bounded(invocationId)||scopes.has(invocationId))throw Error('本次原生检定标记无效。');
  const s={invocationId,actor,item:feature(actor),user:game.user,gmId:game.users.activeGM.id,check,stage:'original',nonce:null,requestedCreateMessage:context.createMessage!==false};scopes.set(invocationId,s);
  let originalCallbacks=0,rerollCallbacks=0,nativeInvocations=0,delivered=false;
  try{
   return await runCheckReactionPipeline({game,check,context,event,
    publish:data=>{if(s.nonce)requireScope(s,false);return publish?publish(data):globalThis.ChatMessage.create(data);},
    native:async(c,ctx,e,collect)=>{const reroll=++nativeInvocations>1;if(reroll)requireScope(s);return native(c,ctx,e,async(roll,outcome,card,ce)=>{
     if(!reroll){if(++originalCallbacks!==1)throw Error('原生检定重复返回，未继续半身人幸运。');return collect(roll,outcome,card,ce);}
     if(s.stage!=='rolling'||++rerollCallbacks!==1||!ctx.options?.has?.(`${ID}:halfling-luck:${s.nonce}`))throw Error('半身人幸运重投不属于本次一次性许可。');
     requireScope(s,false);roll.options.halflingLuckNonce=s.nonce;
     const data=card.toObject();data.flags.pf2e.context.halflingLuckNonce=s.nonce;card.updateSource(data);
     await call(s,'recordResult',{rollJSON:roll.toJSON(),outcome});s.stage='result-ready';
     return collect(roll,outcome,card,ce);
    });},
    decide:async captured=>{
     s.captured=captured;if(!live(s)||!evaluate(s)?.eligible)return null;
     // Unsupported serialization or unavailable WebCrypto cannot turn a normal
     // native result into a lost check, nor authorize an unverifiable payment.
     try{s.fingerprint=await sha256(fingerprintData(s));}catch{return null;}
     const answer=await choose({actor,user:s.user,title:'半身人幸运：是否重投本次检定',choices:[{value:'use',label:'使用原始半身人幸运（免费动作；每日1次；必须保留新结果）'},{value:'decline',label:'保留原结果'}]});
     if(answer!=='use')return null;
     requireScope(s);if(!evaluate(s)?.eligible||await sha256(fingerprintData(s))!==s.fingerprint)return null;
     s.stage='claiming';const claim=await call(s,'claim');
     if(!bounded(claim.nonce)||claim.status!=='claimed')throw Error('半身人幸运认领回执无效。');s.nonce=claim.nonce;
     await pay(s);await call(s,'startRolling');s.stage='rolling';
     captured.context.options.add(`${ID}:halfling-luck:${s.nonce}`);
     return {reaction:'halfling-luck',nonce:s.nonce,actorUuid:actor.uuid};
    },
    callback:async(...args)=>{
     if(delivered)throw Error('原生检定调用方已接收本次结果。');delivered=true;
     if(s.nonce)await call(s,'beginDelivery');
     if(callback)await callback(...args);
     // Awaited callback return is not Toolbelt's detached target-row persistence.
     if(s.nonce)await call(s,'finishDelivery');
    },
   });
  }catch(error){
   if(s.nonce){try{await call(s,'uncertain',{reason:String(error?.message??error)});}catch(reportError){onError(reportError);}}
   throw error;
  }finally{scopes.delete(invocationId);if(authorizations.get(s.item.uuid)===s){authorizations.delete(s.item.uuid);waiters.delete(s.item.uuid);ledger.clearAuthorization(s.item,s.nonce);}}
 }
 function register({Hooks,socket:api}={}){
  if(installed)return()=>{};installed=true;socket=api;const hooks=[];
  const on=(name,fn)=>hooks.push([name,Hooks.on(name,fn)]);
  on('preUpdateItem',(item,changes,options,userId)=>{
   if(!isHalflingLuckItem(item))return;
   if(!Object.hasOwn(changes??{},'system.frequency.value')&&!Object.hasOwn(changes?.system?.frequency??{},'value'))return;
   try{beforeUse(item,game.users.get(userId));return ledger.preparePayment(item,changes,options,userId);}catch(error){onError(error);return false;}
  });
  on('updateItem',(item,changes,options,userId)=>{
   if(!isHalflingLuckItem(item))return;
   try{ledger.observePayment(item,changes,options,userId);wake(item);}catch(error){onError(error);}
  });
  socket?.register(PROOF,async function(payload){try{if(!installed)throw Error('Provider已停用。');return {ok:true,value:await prove(payload,this.socketdata?.userId)};}catch(error){return {ok:false,error:String(error.message??error)};}});
  socket?.register(RPC,async function(payload){try{if(!installed)throw Error('Provider已停用。');return {ok:true,value:await dispatch(payload,this.socketdata?.userId)};}catch(error){return {ok:false,error:String(error.message??error)};}});
  return()=>{installed=false;for(const[name,id]of hooks)Hooks.off(name,id);};
 }
 return {handlesActor,interceptCheck,resolveAction,requiresActualUse:item=>!!resolveAction(item),tracksFrequency:isHalflingLuckItem,beforeUse,captureUsage,executeUsage,register};
}
