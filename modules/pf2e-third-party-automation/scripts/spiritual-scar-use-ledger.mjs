import {MODULE_ID as ID} from './rules.mjs';
import {SerialActions} from './runtime.mjs';
import {getSourceId,isActiveGM} from './native-context.mjs';
import {isActualUseMessage} from './usage-events.mjs';
import {SPIRITUAL_SCAR_SOURCE} from './spiritual-scar-native.mjs';

const PATH=`flags.${ID}.spiritualScarUse`,queues=new WeakMap(),copy=v=>structuredClone(v);
const validId=v=>typeof v==='string'&&/^[A-Za-z0-9_-]{1,100}$/.test(v)&&!['__proto__','prototype','constructor'].includes(v);
const ordered=v=>Array.isArray(v)?v.map(ordered):v&&typeof v==='object'?Object.fromEntries(Object.keys(v).sort().map(k=>[k,ordered(v[k])])):v;
const equal=(a,b)=>JSON.stringify(ordered(a))===JSON.stringify(ordered(b));
const changed=(changes,path)=>Object.hasOwn(changes??{},path)?changes[path]:path.split('.').reduce((v,k)=>v?.[k],changes);
const author=m=>m?.author?.id??m?.user?.id??m?.user;
const requireTrue=v=>{if(!v)throw Error('精神伤痕的原始能力、使用次数或本次付款凭据无法确认。');};
const stateOf=i=>i?.flags?.[ID]?.spiritualScarUse??{currentNonce:null,operations:{}};
const paidStatuses=new Set(['paid','ready','consumed']);
const privacy=value=>{
 requireTrue(typeof value?.blind==='boolean'&&Array.isArray(value.whisper)&&value.whisper.every(validId));
 return {blind:value.blind,whisper:[...new Set(value.whisper)].sort()};
};

/** Internal daily-Use evidence, not damage or reaction authorization. The
 * provider must bind its private live damage scope and reserve the reaction.
 * Only the original native Use changes frequency; no refund or retry is inferred. */
export function createSpiritualScarUseLedger({game,fromUuid=globalThis.fromUuid,queue,randomId=()=>globalThis.foundry?.utils?.randomID?.()??crypto.randomUUID()}={}){
 requireTrue(game&&typeof fromUuid==='function');
 if(!queue){if(!queues.has(game))queues.set(game,new SerialActions());queue=queues.get(game);}
 const authorizations=new Map(),witnesses=new Map(),retired=new Set(),key=(item,nonce)=>`${item.uuid}:${nonce}`;
 const current=item=>{const state=stateOf(item),record=state.operations?.[state.currentNonce];return record?copy(record):null;};
 function source(actor,item,user){
  const token=actor?.token;
  const live=actor?.isToken===true?token?.actor===actor&&game.scenes?.get(token?.parent?.id)===token?.parent&&token?.parent?.tokens?.get(token.id)===token:game.actors?.get(actor?.id)===actor;
  requireTrue(actor?.type==='character'&&live&&item?.actor===actor&&actor.items?.get(item.id)===item&&item.type==='action'&&getSourceId(item)===SPIRITUAL_SCAR_SOURCE);
  requireTrue(item.system?.actionType?.value==='reaction'&&item.system?.frequency?.max===1&&item.system.frequency.per==='day');
  requireTrue(user?.id&&game.users?.get(user.id)===user&&user.active===true&&actor.testUserPermission?.(user,'OWNER')===true&&game.users.activeGM?.isGM===true);
 }
 const gm=()=>requireTrue(game.user?.isGM===true&&isActiveGM(game));
 const canPay=actor=>requireTrue(actor.canAct===true&&actor.isDead!==true);
 async function resolve(scope){gm();source(scope.actor,scope.item,scope.user);requireTrue(await fromUuid(scope.actor.uuid)===scope.actor&&await fromUuid(scope.item.uuid)===scope.item);gm();source(scope.actor,scope.item,scope.user);}
 function recordFor(scope,frequency){
  const {actor,item,user,nonce}=scope;source(actor,item,user);requireTrue(validId(nonce));
  const state=stateOf(item),record=state.operations?.[nonce];
  requireTrue(state.currentNonce===nonce&&record?.nonce===nonce&&record.actorUuid===actor.uuid&&record.itemUuid===item.uuid&&record.userId===user.id&&record.gmId===game.users.activeGM.id);
  if(frequency!==undefined)requireTrue(item.system.frequency.value===frequency);
  return record;
 }
 function observed(scope){
  const record=recordFor(scope,0),witness=witnesses.get(key(scope.item,scope.nonce));
  requireTrue(witness&&!retired.has(key(scope.item,scope.nonce))&&witness.paymentNonce===record.paymentNonce&&witness.userId===record.userId&&witness.gmId===record.gmId&&paidStatuses.has(record.status));
  return {record,witness};
 }
 async function save(scope,state){
  gm();source(scope.actor,scope.item,scope.user);const expected=copy(state),result=await scope.item.update({[PATH]:expected});
  requireTrue(result===scope.item);await resolve(scope);requireTrue(equal(stateOf(scope.item),expected));
  return current(scope.item);
 }
 const mutate=(scope,fn)=>queue.run(scope.item?.uuid??'',async()=>{await resolve(scope);return fn();});
 const replace=(scope,record)=>{const state=copy(stateOf(scope.item));state.operations[record.nonce]=copy(record);return save(scope,state);};
 const claim=scope=>mutate(scope,async()=>{
  const {actor,item,user,invocationId,fingerprint}=scope;canPay(actor);
  requireTrue(validId(invocationId)&&/^[a-f0-9]{64}$/.test(fingerprint??'')&&item.system.frequency.value===1);
  const privacyValue=privacy(scope.privacy),state=copy(stateOf(item));requireTrue(state.operations&&typeof state.operations==='object'&&!Array.isArray(state.operations));
  const prior=Object.values(state.operations).find(r=>r.invocationId===invocationId);
  if(prior){requireTrue(prior.status==='claimed'&&state.currentNonce===prior.nonce&&prior.fingerprint===fingerprint&&equal(prior.privacy,privacyValue));recordFor({...scope,nonce:prior.nonce},1);return copy(prior);}
  requireTrue(!['claimed','paying','paid','ready'].includes(state.operations[state.currentNonce]?.status));
  const nonce=randomId();requireTrue(validId(nonce)&&!Object.hasOwn(state.operations,nonce));
  state.currentNonce=nonce;state.operations[nonce]={nonce,status:'claimed',actorUuid:actor.uuid,itemUuid:item.uuid,userId:user.id,gmId:game.users.activeGM.id,invocationId,fingerprint,privacy:privacyValue};
  const result=await save(scope,state);recordFor({...scope,nonce},1);return result;
 });
 const cancelClaim=scope=>mutate(scope,async()=>{const record=recordFor(scope,1);requireTrue(record.status==='claimed');const result=await replace(scope,{...record,status:'cancelled'});recordFor(scope,1);return result;});
 const beginPayment=scope=>mutate(scope,async()=>{const record=recordFor(scope,1);requireTrue(record.status==='claimed');canPay(scope.actor);const result=await replace(scope,{...record,status:'paying'});recordFor(scope,1);return result;});
 function authorizePayment(item,nonce){
  const scope={actor:item?.actor,item,user:game.user,nonce},record=recordFor(scope,1);requireTrue(record.status==='paying');canPay(scope.actor);
  const previous=authorizations.get(item.uuid);requireTrue(!previous||previous.nonce===nonce&&previous.status==='authorized');
  authorizations.set(item.uuid,{...scope,gmId:record.gmId,status:'authorized'});return copy(record);
 }
 const clearAuthorization=(item,nonce)=>{if(authorizations.get(item?.uuid)?.nonce!==nonce)return false;authorizations.delete(item.uuid);return true;};
 function retire(item){
  for(const k of witnesses.keys())if(k.startsWith(item.uuid+':')){retired.add(k);witnesses.delete(k);}
  const r=current(item);if(r&&paidStatuses.has(r.status))retired.add(key(item,r.nonce));authorizations.delete(item.uuid);
 }
 function preparePayment(item,changes,options,userId){
  if(changed(changes,'system.frequency.value')===undefined)return;
  const auth=authorizations.get(item?.uuid);if(!auth)return;
  try{
   requireTrue(auth.item===item&&auth.user===game.user&&auth.user.id===userId&&auth.status==='authorized');
   const record=recordFor(auth,1);requireTrue(record.status==='paying'&&record.gmId===auth.gmId&&changed(changes,'system.frequency.value')===0);canPay(auth.actor);
   const paymentNonce=randomId();requireTrue(validId(paymentNonce));const state=copy(stateOf(item));state.operations[record.nonce]={...record,status:'paid',paymentNonce};
   changes[PATH]=state;options[ID]={...options[ID],spiritualScarPayment:{nonce:record.nonce,paymentNonce}};auth.status='prepared';return true;
  }catch{return false;}
 }
 function observePayment(item,changes,options,userId){
  try{
   const proof=options?.[ID]?.spiritualScarPayment,receipt=options?.[ID]?.frequencyReceipt,user=game.users?.get(userId);
   requireTrue(proof&&receipt&&validId(proof.nonce)&&validId(proof.paymentNonce));
   const record=recordFor({actor:item.actor,item,user,nonce:proof.nonce},0),updated=changed(changes,PATH),payment=updated?.operations?.[proof.nonce];
   requireTrue(record.status==='paid'&&!retired.has(key(item,record.nonce))&&record.paymentNonce===proof.paymentNonce&&changed(changes,'system.frequency.value')===0);
   requireTrue(payment?.status==='paid'&&payment.paymentNonce===record.paymentNonce&&(updated.currentNonce===undefined||updated.currentNonce===record.nonce));
   requireTrue(validId(receipt.id)&&receipt.itemUuid===item.uuid&&receipt.userId===userId&&receipt.before===1&&receipt.after===0&&Number.isFinite(receipt.createdAt));
   const witness={nonce:record.nonce,paymentNonce:record.paymentNonce,userId,gmId:record.gmId,receipt:copy(receipt)},prior=witnesses.get(key(item,record.nonce));requireTrue(!prior||equal(prior,witness));
   witnesses.set(key(item,record.nonce),witness);clearAuthorization(item,record.nonce);return copy(witness);
  }catch{if(item?.uuid&&changed(changes,'system.frequency.value')!==undefined)retire(item);return null;}
 }
 const bindUsage=scope=>mutate(scope,async()=>{
  const {actor,item,user,message,frequencyReceipt}=scope,proof=message?.flags?.[ID]?.spiritualScarInput,nonce=proof?.nonce;
  requireTrue(scope.nonce===undefined||scope.nonce===nonce);const actual={...scope,nonce},{record,witness}=observed(actual);
  requireTrue(equal(frequencyReceipt,witness.receipt)&&message?.id&&game.messages?.get(message.id)===message&&await fromUuid(message.uuid)===message);
  gm();requireTrue(equal(observed(actual).record,record)&&game.messages.get(message.id)===message);
  const origin=message.flags?.pf2e?.origin,input=message.flags?.[ID]?.usageInput;
  requireTrue(author(message)===user.id&&message.speaker?.actor===actor.id&&origin?.uuid===item.uuid&&origin.actor===actor.uuid&&origin.type==='action');
  requireTrue(isActualUseMessage(message)&&!message.isCheckRoll&&!message.isRoll&&!message.rolls?.length&&!message.flags?.[ID]?.usageGenerated&&(!message.flags?.pf2e?.context||message.flags.pf2e.context.type==='self-effect'));
  requireTrue(proof.paymentNonce===record.paymentNonce&&input?.frequencyReceiptId===frequencyReceipt.id&&equal(privacy({blind:message.blind===true,whisper:message.whisper??[]}),record.privacy));
  if(record.messageUuid){requireTrue(record.messageUuid===message.uuid&&record.messageId===message.id&&record.frequencyReceiptId===frequencyReceipt.id);return copy(record);}
  requireTrue(record.status==='paid');const result=await replace(actual,{...record,status:'ready',messageUuid:message.uuid,messageId:message.id,frequencyReceiptId:frequencyReceipt.id});observed(actual);return result;
 });
 const consume=scope=>mutate(scope,async()=>{const {record}=observed(scope);requireTrue(record.status==='ready');canPay(scope.actor);const result=await replace(scope,{...record,status:'consumed'});observed(scope);return result;});
 const uncertain=scope=>mutate(scope,async()=>{
  const record=recordFor(scope);requireTrue(['claimed','paying','paid','ready','consumed','uncertain'].includes(record.status)&&typeof scope.reason==='string'&&scope.reason.length>0&&scope.reason.length<=1000);
  if(record.status==='uncertain')return copy(record);
  const result=await replace(scope,{...record,status:'uncertain',reason:scope.reason});retire(scope.item);return result;
 });
 return {current,claim,cancelClaim,beginPayment,authorizePayment,clearAuthorization,preparePayment,observePayment,bindUsage,consume,uncertain};
}
