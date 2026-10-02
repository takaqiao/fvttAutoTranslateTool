import {MODULE_ID} from './schema.mjs';
import {canonicalJSON} from './revision-codec.mjs';
import {manualPoolBatchModel} from './manual-pool-model.mjs';

const author=message=>message?.author?.id??message?.author??message?.user?.id??message?.user;
const receiptSource=data=>canonicalJSON({actor:data.speaker?.actor,pf2e:data.flags?.pf2e});
function originalOptions(frame,options){
 const original=frame.paramsSnapshot.rollOptions;
 if(!Array.isArray(original)||!Array.isArray(options)&&!(options instanceof Set))return false;
 const source=`${MODULE_ID}:source:${frame.message.id}:${frame.rollIndex}`;
 if(original.some(option=>typeof option!=='string'||option.startsWith(`${MODULE_ID}:source:`)&&option!==source||option.startsWith(`${MODULE_ID}:manual-pool-native:`))||new Set(original).size!==original.length)return false;
 const expected=new Set([...original,source]),actual=[...options];
 return actual.length===expected.size&&new Set(actual).size===actual.length&&actual.every(option=>expected.has(option));
}

/** The only native leaf consumes a captured source frame. The create observer
 * waits on the original receipt Promise; reading a card cannot create a half. */
export function createManualPoolApplication({game,completion,getProvider=()=>game.pf2e?.thirdPartyManualPoolBatch}){
 const captured=new WeakMap(),receipts=new Map();let stopped=false;
 function captureFrame(actor,params){
  const provider=getProvider();if(stopped||!manualPoolBatchModel(provider?.descriptor)||typeof provider.currentCall!=='function')return null;
  const frame=provider.currentCall(actor,params);if(!frame)return null;
  captured.set(frame,{actor,params,provider,used:false});return frame;
 }
 async function observeCreate(wrapped,data,...args){
  const c=data?.flags?.pf2e?.context,keys=c?.options?.filter(option=>option.startsWith(`${MODULE_ID}:manual-pool-native:`))??[],scope=keys.length===1?receipts.get(keys[0]):null;
  if(!scope)return wrapped(data,...args);
  if(scope.receiptPromise||!scope.isCurrent()||c.type!=='damage-taken'||data.speaker?.actor!==scope.frame.patient.id)throw Error('manual-pool-receipt-call-mismatch');
  // Core cleans creation input in place. The original PF2e receipt fields must
  // survive that workflow; unrelated document defaults are allowed to settle.
  const before=receiptSource(data),writer=game.user;
  const promise=wrapped(data,...args);if(!promise||typeof promise.then!=='function')throw Error('manual-pool-receipt-promise-required');
  scope.receiptPromise=(async()=>{
   const receipt=await promise;
   if(!scope.isCurrent()||game.user!==writer||receiptSource(data)!==before||receiptSource(receipt)!==before||game.messages.get(receipt?.id)!==receipt||author(receipt)!==writer.id||receipt.speaker?.actor!==scope.frame.patient.id||receipt.flags?.pf2e?.context?.type!=='damage-taken'||!receipt.flags.pf2e.context.options?.includes(scope.tag))throw Error('manual-pool-receipt-call-changed');
   return receipt;
  })();scope.receiptPromise.catch(()=>{});return scope.receiptPromise;
 }
 async function applyNativeDamage(actor,params,native,frame){
  if(!frame)return native(params);
  const entry=captured.get(frame);if(!entry||entry.used||entry.actor!==actor||frame.contextualActor!==actor||frame.token?.actor!==frame.patient||typeof completion?.withApplication!=='function')throw Error('manual-pool-original-frame-required');entry.used=true;
  const isCurrent=()=>!stopped&&getProvider()===entry.provider&&manualPoolBatchModel(entry.provider.descriptor)&&frame.isCurrent?.()===true&&frame.token.actor===frame.patient&&game.messages.get(frame.message.id)===frame.message&&frame.message.rolls[frame.rollIndex]===frame.roll;
  const expected=frame.paramsSnapshot;
  const checkParams=()=>{if(typeof params.damage!=='number'||!Number.isFinite(params.damage)||params.damage>=0||params.damage!==expected.damage||params.token!==frame.token||params.item!==frame.item||params.skipIWR!==expected.skipIWR||params.outcome!==expected.outcome||params.shieldBlockRequest!==expected.shieldBlockRequest||!originalOptions(frame,params.rollOptions))throw Error('manual-pool-native-params-changed')};
  if(!isCurrent())throw Error('manual-pool-original-frame-changed');checkParams();
  const tag=`${MODULE_ID}:manual-pool-native:${crypto.randomUUID()}`,scope={frame,tag,isCurrent,receiptPromise:null};
  const result=await completion.withApplication(frame.grant,frame.patient,async()=>{
   if(!isCurrent())throw Error('manual-pool-original-frame-changed');checkParams();receipts.set(tag,scope);
   try{
    const nativeResult=await native({...params,rollOptions:new Set([...(params.rollOptions??[]),tag])});
    if(!isCurrent()||nativeResult!==actor||!scope.receiptPromise)throw Error('manual-pool-original-receipt-unavailable');
    const receipt=await scope.receiptPromise;if(!isCurrent())throw Error('manual-pool-original-frame-changed');return {nativeResult,receipt};
   }finally{receipts.delete(tag)}
  },{updateActor:actor});
  return result.result.nativeResult;
 }
 return {captureFrame,observeCreate,applyNativeDamage,stop(){stopped=true;receipts.clear()}};
}
