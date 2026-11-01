import {clone,finite,id} from './schema.mjs';
import {canonicalJSON} from './revision-codec.mjs';

export const computedSession=session=>session?.executionMode==='computed-v1';
export const computedBatch=activity=>activity?.kind==='computed-recovery-batch';
export const computedBatchIndex=({id,state,from,to,reason})=>({id,state,from,to,...reason===undefined?{}:{reason}});
export const pendingComputed=session=>!!session?.computedRecovery?.pendingBoundary||!!session?.computedRecovery?.pendingCalculation||session?.computedRecovery?.batches.some(batch=>batch.state!=='confirmed')===true;
export const reviewedComputed=session=>computedSession(session)&&session.status==='closed'&&session.computedRecovery?.review?.disposition==='closed-without-replay';

export function computedGuard(guard){
 if(typeof guard!=='function')throw Error('synchronous-computed-guard-required');
 const valid=guard();if(valid?.then||valid!==true)throw Error('computed-source-changed');
}

export function captureComputedBatch(input){
 const fields=['id','from','to','beforeActors','afterActors','events','documents'];
 if(!input||Object.keys(input).some(key=>!fields.includes(key)))throw Error('invalid-computed-payload');
 const payload=JSON.parse(canonicalJSON(input));
 id(payload.id,'computed-batch');finite(payload.from,'computed-from');finite(payload.to,'computed-to');
 if(payload.to<payload.from)throw Error('computed-boundary-conflict');
 for(const key of ['beforeActors','afterActors','events'])if(!Array.isArray(payload[key]))throw Error('invalid-computed-payload');
 const documents=payload.documents;
 if(!documents||documents.id!==payload.id||documents.at!==payload.to||!Array.isArray(documents.actors)||!Array.isArray(documents.items))throw Error('invalid-computed-documents');
 const actors=documents.actors.map(row=>id(row.uuid,'computed-actor'));
 if(new Set(actors).size!==actors.length)throw Error('invalid-computed-documents');
 const items=documents.items.map(row=>id(row.actorUUID,'computed-item-actor'));
 if(new Set(items).size!==items.length||documents.items.some(row=>!Array.isArray(row.create)||row.create.some(item=>typeof item._id!=='string')))throw Error('invalid-computed-documents');
 return payload;
}

export async function computedDigest(payload){
 const digest=await crypto.subtle.digest('SHA-256',new TextEncoder().encode(canonicalJSON(payload)));
 return Array.from(new Uint8Array(digest),byte=>byte.toString(16).padStart(2,'0')).join('');
}

export function validateComputedReceipt(batch,phase,receipt){
 let expected;
 if(phase==='actors')expected={actors:batch.documents.actors.map(row=>({actorUUID:row.uuid,batchId:batch.id,after:row.changes}))};
 else if(phase==='items')expected={items:batch.documents.items.flatMap(row=>row.create.map(item=>({actorUUID:row.actorUUID,itemUUID:row.actorUUID+'.Item.'+item._id,batchId:batch.id}))),wounded:batch.documents.items.filter(row=>row.removeWounded).map(row=>row.actorUUID)};
 else throw Error('computed-phase-conflict');
 if(canonicalJSON(receipt??null)!==canonicalJSON(expected))throw Error('computed-receipt-mismatch');
}

export function captureComputedTransition(input){
 const fields=['expected','state','phase','receipt','receipts','summary','simulationState','reason'];
 if(!input||Object.keys(input).some(key=>!fields.includes(key)))throw Error('immutable-computed-payload');
 const change=JSON.parse(canonicalJSON(input));
 if(!Array.isArray(change.expected)||!change.expected.length||change.expected.some(state=>!['claimed','uncertain','confirmed'].includes(state)))throw Error('computed-state-conflict');
 return change;
}

function recordComputedReceipts(batch,receipts){
 if(!receipts||typeof receipts!=='object'||Array.isArray(receipts)||Object.keys(receipts).some(phase=>!['actors','items'].includes(phase)))throw Error('computed-receipt-mismatch');
 for(const phase of ['actors','items'])if(Object.hasOwn(receipts,phase)){
  if(phase==='items'&&!Object.hasOwn(batch.receipts,'actors'))throw Error('computed-phase-conflict');
  validateComputedReceipt(batch,phase,receipts[phase]);
  if(Object.hasOwn(batch.receipts,phase)&&canonicalJSON(batch.receipts[phase])!==canonicalJSON(receipts[phase]))throw Error('computed-receipt-mismatch');
  batch.receipts[phase]=clone(receipts[phase]);
 }
}

export function transitionComputed(session,batch,change){
 if(!change.expected.includes(batch.state)||batch.state!=='claimed'||!['claimed','uncertain','confirmed'].includes(change.state))throw Error('computed-state-conflict');
 if(change.state==='claimed'){
  const next=Object.hasOwn(batch.receipts,'actors')?'items':'actors';
  if(change.phase!==next||Object.hasOwn(batch.receipts,change.phase)||'receipts' in change||'summary' in change||'simulationState' in change||'reason' in change)throw Error('computed-phase-conflict');
  validateComputedReceipt(batch,change.phase,change.receipt);batch.receipts[change.phase]=clone(change.receipt);return batch;
 }
 if('phase' in change||'receipt' in change)throw Error('computed-phase-conflict');
 if('receipts' in change)recordComputedReceipts(batch,change.receipts);
 if(change.state==='uncertain'){
  if('summary' in change||'simulationState' in change)throw Error('computed-state-conflict');
  batch.state='uncertain';batch.reason=change.reason??'computed-result-unknown';return batch;
 }
 if(!Object.hasOwn(batch.receipts,'actors')||!Object.hasOwn(batch.receipts,'items'))throw Error('computed-phase-incomplete');
 if(session.cursorAt!==batch.to)throw Error('computed-cursor-conflict');
 const boundary=session.computedRecovery.pendingBoundary,calculation=session.computedRecovery.pendingCalculation;
 if(boundary&&(boundary.from!==batch.from||boundary.to!==batch.to))throw Error('computed-boundary-conflict');
 if(calculation&&calculation.from!==batch.from)throw Error('computed-boundary-conflict');
 saveComputedSummary(session,change);batch.state='confirmed';
 if(boundary)delete session.computedRecovery.pendingBoundary;if(calculation)delete session.computedRecovery.pendingCalculation;return batch;
}

export function saveComputedSummary(session,input){
 if('simulationState' in input){
  if(input.simulationState?.cursorAt!==session.cursorAt)throw Error('computed-cursor-conflict');
  session.computedRecovery.simulationState=clone(input.simulationState);
 }
 if('summary' in input)session.computedRecovery.summary=clone(input.summary);
}
