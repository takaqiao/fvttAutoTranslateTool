import {MODULE_ID as ID} from './rules.mjs';
import {SerialActions} from './runtime.mjs';
import {getSourceId,isActiveGM} from './native-context.mjs';
import {isActualUseMessage} from './usage-events.mjs';

export const HALFLING_LUCK_SOURCE='Compendium.pf2e.feats-srd.Item.ZbRVqf14RTJJIZXG';
const PATH=`flags.${ID}.halflingLuck`,queues=new WeakMap();
const copy=value=>structuredClone(value);
const validId=value=>typeof value==='string'&&/^[A-Za-z0-9_-]{1,100}$/.test(value)&&!['__proto__','prototype','constructor'].includes(value);
const ordered=value=>Array.isArray(value)?value.map(ordered):value&&typeof value==='object'?Object.fromEntries(Object.keys(value).sort().map(key=>[key,ordered(value[key])])):value;
const equal=(a,b)=>JSON.stringify(ordered(a))===JSON.stringify(ordered(b));
const changed=(changes,path)=>Object.hasOwn(changes??{},path)?changes[path]:path.split('.').reduce((value,key)=>value?.[key],changes);
const authorId=message=>message?.author?.id??message?.user?.id??message?.user;
const fail=()=>{throw Error('半身人幸运的来源、付款或当前处理状态无法确认。');};
const requireTrue=value=>{if(!value)fail();};
const stateOf=item=>item?.flags?.[ID]?.halflingLuck??{currentNonce:null,operations:{}};
const paidStatuses=new Set(['paid','ready','rolling','result-ready','delivering','callback-returned','uncertain']);
const outcomes=new Set(['criticalFailure','failure','success','criticalSuccess']);

/** Durable permissions only. Native Use owns the frequency debit; the caller owns dice and callbacks. */
export function createHalflingLuckLedger({game,fromUuid=globalThis.fromUuid,queue,randomId=()=>globalThis.foundry?.utils?.randomID?.()??globalThis.crypto.randomUUID()}={}){
 requireTrue(game&&typeof fromUuid==='function');
 if(!queue){if(!queues.has(game))queues.set(game,new SerialActions());queue=queues.get(game);}
 const authorizations=new Map(),witnesses=new Map(),retired=new Set();
 const key=(item,nonce)=>`${item.uuid}:${nonce}`;
 const current=item=>{const state=stateOf(item),record=state.operations?.[state.currentNonce];return record?copy(record):null;};
 const source=(actor,item,user)=>{
  requireTrue(actor?.type==='character'&&!actor.isToken&&game.actors?.get(actor.id)===actor);
  requireTrue(item?.actor===actor&&actor.items?.get(item.id)===item&&item.type==='feat'&&getSourceId(item)===HALFLING_LUCK_SOURCE);
  requireTrue(item.system?.actionType?.value==='free'&&item.system?.frequency?.max===1&&item.system.frequency.per==='day');
  requireTrue(user?.id&&game.users?.get(user.id)===user&&actor.testUserPermission?.(user,'OWNER')===true);
  requireTrue(game.users?.activeGM?.isGM===true);
 };
 const gm=()=>requireTrue(game.user?.isGM===true&&isActiveGM(game));
 const canPay=actor=>requireTrue(actor.canAct===true&&actor.isDead!==true);
 const resolve=async({actor,item,user})=>{
  gm();source(actor,item,user);
  requireTrue(await fromUuid(actor.uuid)===actor);requireTrue(await fromUuid(item.uuid)===item);
  gm();source(actor,item,user);
 };
 const recordFor=({actor,item,user,nonce},frequency)=>{
  source(actor,item,user);requireTrue(validId(nonce));
  const state=stateOf(item),record=state.operations?.[nonce];
  requireTrue(state.currentNonce===nonce&&record?.nonce===nonce&&record.actorUuid===actor.uuid&&record.itemUuid===item.uuid&&record.userId===user.id&&record.gmId===game.users.activeGM.id);
  if(frequency!==undefined)requireTrue(item.system.frequency.value===frequency);
  return record;
 };
 const observed=scope=>{
  const record=recordFor(scope,0),witness=witnesses.get(key(scope.item,scope.nonce));
  requireTrue(witness&&witness.paymentNonce===record.paymentNonce&&witness.userId===record.userId&&witness.gmId===record.gmId&&paidStatuses.has(record.status));
  return {record,witness};
 };
 const save=async(scope,state)=>{
  gm();source(scope.actor,scope.item,scope.user);
  const expected=copy(state),result=await scope.item.update({[PATH]:expected});
  // A cancelled/no-op Document update is not a durable permission, even if its Promise resolved.
  requireTrue(result===scope.item);await resolve(scope);requireTrue(equal(stateOf(scope.item),expected));
  const saved=expected.operations[expected.currentNonce];
  if(paidStatuses.has(saved.status))observed({...scope,nonce:saved.nonce});
  else recordFor({...scope,nonce:saved.nonce},1);
  return current(scope.item);
 };
 const mutate=async(scope,run)=>queue.run(scope.item?.uuid??'',async()=>{await resolve(scope);return run();});
 const replace=async(scope,record)=>{const state=copy(stateOf(scope.item));state.operations[record.nonce]=copy(record);return save(scope,state);};
 const claim=scope=>mutate(scope,async()=>{
  const {actor,item,user,invocationId,fingerprint}=scope;
  canPay(actor);
  requireTrue(validId(invocationId)&&typeof fingerprint==='string'&&fingerprint.length>0&&fingerprint.length<=65536&&item.system.frequency.value===1);
  const state=copy(stateOf(item));requireTrue(state.operations&&typeof state.operations==='object'&&!Array.isArray(state.operations));
  const prior=Object.values(state.operations).find(record=>record.invocationId===invocationId);
  if(prior){requireTrue(state.currentNonce===prior.nonce&&prior.status==='claimed');recordFor({...scope,nonce:prior.nonce},1);requireTrue(prior.fingerprint===fingerprint);return copy(prior);}
  requireTrue(state.operations[state.currentNonce]?.status!=='claimed');
  const nonce=randomId();requireTrue(validId(nonce)&&!Object.hasOwn(state.operations,nonce));
  state.currentNonce=nonce;state.operations[nonce]={nonce,status:'claimed',actorUuid:actor.uuid,itemUuid:item.uuid,userId:user.id,gmId:game.users.activeGM.id,invocationId,fingerprint};
  return save(scope,state);
 });
 const cancelClaim=scope=>mutate(scope,async()=>{
  const record=recordFor(scope,1);requireTrue(record.status==='claimed');
  return replace(scope,{...record,status:'cancelled'});
 });
 const authorizePayment=(item,nonce,user=game.user)=>{
  requireTrue(game.user===user);const scope={actor:item?.actor,item,user,nonce},record=recordFor(scope,1);requireTrue(record.status==='claimed');
  canPay(scope.actor);
  const existing=authorizations.get(item.uuid);
  requireTrue(!existing||existing.nonce!==nonce||existing.status==='authorized');
  authorizations.set(item.uuid,{...scope,gmId:record.gmId,status:'authorized'});return copy(record);
 };
 const clearAuthorization=(item,nonce)=>{
  if(authorizations.get(item?.uuid)?.nonce!==nonce)return false;
  authorizations.delete(item.uuid);return true;
 };
 const retirePayments=item=>{
  for(const witnessKey of witnesses.keys())if(witnessKey.startsWith(`${item.uuid}:`)){retired.add(witnessKey);witnesses.delete(witnessKey);}
  const record=current(item);if(record&&paidStatuses.has(record.status))retired.add(key(item,record.nonce));
  authorizations.delete(item.uuid);
 };
 const preparePayment=(item,changes,options,userId)=>{
  if(changed(changes,'system.frequency.value')===undefined)return undefined;
  const authorization=authorizations.get(item?.uuid);if(!authorization)return undefined;
  try{
   requireTrue(authorization.item===item&&authorization.user===game.user&&authorization.user.id===userId&&authorization.status==='authorized');
   const record=recordFor(authorization,1);requireTrue(record.status==='claimed'&&record.gmId===authorization.gmId&&changed(changes,'system.frequency.value')===0);
   canPay(authorization.actor);
   const paymentNonce=randomId();requireTrue(validId(paymentNonce));
   const state=copy(stateOf(item));state.operations[record.nonce]={...record,status:'paid',paymentNonce};
   changes[PATH]=state;options[ID]={...options[ID],halflingLuckPayment:{nonce:record.nonce,paymentNonce}};
   authorization.status='prepared';return true;
  }catch{return false;}
 };
 const observePayment=(item,changes,options,userId)=>{
  try{
   const proof=options?.[ID]?.halflingLuckPayment,receipt=options?.[ID]?.frequencyReceipt,user=game.users?.get(userId);
   requireTrue(proof&&receipt&&validId(proof.nonce)&&validId(proof.paymentNonce));
   const record=recordFor({actor:item.actor,item,user,nonce:proof.nonce},0);
   requireTrue(record.status==='paid'&&!retired.has(key(item,record.nonce))&&record.paymentNonce===proof.paymentNonce&&changed(changes,'system.frequency.value')===0);
   // updateItem receives the native diff: unchanged claim fields/currentNonce are normally omitted.
   const updated=changed(changes,PATH),payment=updated?.operations?.[proof.nonce];
   requireTrue(payment?.status==='paid'&&payment.paymentNonce===record.paymentNonce&&(updated.currentNonce===undefined||updated.currentNonce===proof.nonce));
   requireTrue(validId(receipt.id)&&receipt.itemUuid===item.uuid&&receipt.userId===userId&&receipt.before===1&&receipt.after===0&&Number.isFinite(receipt.createdAt));
   const witness={nonce:record.nonce,paymentNonce:record.paymentNonce,userId,gmId:record.gmId,receipt:copy(receipt)};
   const previous=witnesses.get(key(item,record.nonce));requireTrue(!previous||equal(previous,witness));
   witnesses.set(key(item,record.nonce),witness);clearAuthorization(item,record.nonce);return copy(witness);
  }catch{
   // A later native recharge/manual debit cannot revive an earlier pending invocation.
   if(item?.uuid&&changed(changes,'system.frequency.value')!==undefined)retirePayments(item);
   return null;
  }
 };
 const bindUsage=scope=>mutate(scope,async()=>{
  const {actor,item,user,message,frequencyReceipt}=scope,proof=message?.flags?.[ID]?.halflingLuckInput;
  const nonce=proof?.nonce;requireTrue(scope.nonce===undefined||scope.nonce===nonce);
  const {record,witness}=observed({...scope,nonce});
  requireTrue(equal(frequencyReceipt,witness.receipt)&&message?.id&&game.messages?.get(message.id)===message&&await fromUuid(message.uuid)===message);
  // Recheck after UUID resolution; a recharge or a second claim invalidates the old card.
  gm();requireTrue(equal(observed({...scope,nonce}).record,record)&&game.messages?.get(message.id)===message);
  const origin=message.flags?.pf2e?.origin,input=message.flags?.[ID]?.usageInput;
  requireTrue(authorId(message)===user.id&&message.speaker?.actor===actor.id&&origin?.uuid===item.uuid&&origin.actor===actor.uuid&&origin.type==='feat');
  requireTrue(isActualUseMessage(message)&&!message.isCheckRoll&&!message.isRoll&&!message.rolls?.length&&!message.flags?.[ID]?.usageGenerated);
  requireTrue(!message.flags?.pf2e?.context||message.flags.pf2e.context.type==='self-effect');
  requireTrue(proof.paymentNonce===record.paymentNonce&&input?.frequencyReceiptId===frequencyReceipt.id);
  if(record.messageUuid){requireTrue(record.messageUuid===message.uuid&&record.messageId===message.id&&record.frequencyReceiptId===frequencyReceipt.id);return copy(record);}
  requireTrue(record.status==='paid');
  return replace(scope,{...record,status:'ready',messageUuid:message.uuid,messageId:message.id,frequencyReceiptId:frequencyReceipt.id});
 });
 const transition=(scope,before,after)=>mutate(scope,async()=>{
  const {record}=observed(scope);requireTrue(record.status===before);return replace(scope,{...record,status:after});
 });
 const startRolling=scope=>transition(scope,'ready','rolling');
 const recordResult=scope=>mutate(scope,async()=>{
  const {record}=observed(scope);requireTrue(outcomes.has(scope.outcome)&&scope.rollJSON&&typeof scope.rollJSON==='object'&&!Array.isArray(scope.rollJSON));
  const rollJSON=JSON.parse(JSON.stringify(scope.rollJSON));requireTrue(equal(rollJSON,scope.rollJSON));
  if(record.status==='result-ready'){requireTrue(equal(record.rollJSON,rollJSON)&&record.outcome===scope.outcome);return copy(record);}
  requireTrue(record.status==='rolling');return replace(scope,{...record,status:'result-ready',rollJSON,outcome:scope.outcome});
 });
 const beginDelivery=scope=>transition(scope,'result-ready','delivering');
 const finishDelivery=scope=>mutate(scope,async()=>{
  const {record}=observed(scope);if(record.status==='callback-returned')return copy(record);
  requireTrue(record.status==='delivering');return replace(scope,{...record,status:'callback-returned'});
 });
 const uncertain=scope=>mutate(scope,async()=>{
  const {record}=observed(scope);requireTrue(record.status!=='callback-returned'&&typeof scope.reason==='string'&&scope.reason.length>0&&scope.reason.length<=1000);
  if(record.status==='uncertain')return copy(record);
  return replace(scope,{...record,status:'uncertain',reason:scope.reason});
 });
 return {current,claim,cancelClaim,authorizePayment,clearAuthorization,preparePayment,observePayment,bindUsage,startRolling,recordResult,beginDelivery,finishDelivery,uncertain};
}
