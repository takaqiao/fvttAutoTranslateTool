import {test} from 'node:test';
import assert from 'node:assert/strict';
import {validateNativeResult} from '../../scripts/exploration/owner-command.mjs';

const MODULE_ID='pf2e-third-party-automation',P='Actor.P',Q='Actor.Q';
const enabled={enabled:true,limit:3};
const context=activities=>({sessionId:'S',activityIds:[...new Set(activities.map(a=>a.id))]});
async function helper(){
  try{return (await import('../../scripts/exploration/treatment-streaks.mjs')).treatmentFailureState}
  catch(error){if(error.code==='ERR_MODULE_NOT_FOUND')assert.fail('treatment failure derivation is not implemented');throw error}
}

async function savedTreatment({id='A',startedAt=0,healer='Actor.H',patients=[{uuid:P,outcome:'failure'}],poolUUIDs=patients.map(p=>p.uuid),rolledHealing}={}){
  const user={id:'G',active:true,isGM:true},actor={id:healer.split('.').at(-1),uuid:healer,testUserPermission:()=>true},documents=new Map([[healer,actor]]),messages=new Map();
  const activity={id,sessionId:'S',providerId:'treat-wounds',actorUUID:healer,patientUUIDs:patients.map(p=>p.uuid),hpPoolUUIDs:poolUUIDs,startedAt,endsAt:startedAt+600,durationSeconds:600,state:'confirmed',source:{type:'coordinator',ownerId:'G'},options:{skill:'medicine',rank:'trained',riskySurgery:false,continualRecovery:false}};
  const permit={protocol:'pf2e-third-party-automation.exploration-owner.v1',rootUUID:'JournalEntry.ROOT',epoch:'E',revision:1,sessionId:'S',activityId:id,operationId:'treat-wounds',actorUUID:healer,ownerUserId:'G',ownerClientNonce:'owner',attemptNonce:'attempt-'+id,permitNonce:'permit-'+id,leaseNonce:'lease',offerId:'offer-'+id,requestId:'request-'+id,commandDigest:'digest-'+id};
  const summaries=patients.map(({uuid,outcome},index)=>{
    const patient={uuid,id:uuid.split('.').at(-1),testUserPermission:()=>true};documents.set(uuid,patient);
    const checkId=`${id}-C${index}`,resultId=`${id}-R${index}`,receiptId=`${id}-HP${index}`,immunityId=`${uuid}.Item.${id}-I${index}`;
    const flags={pf2e:{context:{outcome,options:['exploration-activity:'+id]}},[MODULE_ID]:{exploration:{activityId:id,patientUUID:uuid}}};
    messages.set(checkId,{id:checkId,author:user,speaker:{actor:actor.id},flags,rolls:[{_evaluated:true,total:20}]});
    const hasResult=outcome!=='failure',healing=['success','criticalSuccess'].includes(outcome),total=healing?rolledHealing??8:3;
    if(hasResult){
      messages.set(resultId,{id:resultId,author:user,speaker:{actor:actor.id},flags:{...structuredClone(flags),pf2e:{...structuredClone(flags.pf2e),origin:{messageId:checkId}}},rolls:[{_evaluated:true,total,toJSON:()=>({formula:healing?'{2d8[healing]}':'{(1d8)}'})}]});
      messages.set(receiptId,{id:receiptId,author:user,speaker:{actor:patient.id},flags:{pf2e:{context:{type:'damage-taken',options:[`${MODULE_ID}:source:${resultId}:0`,`${MODULE_ID}:exploration-apply:${id}:${resultId}:${uuid}`]},appliedDamage:{uuid,isReverted:false}}}});
    }
    documents.set(immunityId,{uuid:immunityId,actor:patient,flags:{[MODULE_ID]:{exploration:{activityId:id,kind:'immunity'}}}});
    return {status:'confirmed',proof:{useId:id,checkIds:[checkId],resultIds:hasResult?[resultId]:[],receiptIds:hasResult?[receiptId]:[],immunityIds:[immunityId],poolReceipts:hasResult?[{activityId:id,actorUUID:poolUUIDs[index]??poolUUIDs[0],patientUUID:uuid,before:{value:10},after:{value:10}}]:[]},sourceDegree:['criticalFailure','failure','success','criticalSuccess'].indexOf(outcome),effectiveOutcome:outcome,rolledHealing:healing?total:null,medicBonus:0,expiresAt:startedAt+3600,resourceReceiptIds:[],patientUUID:uuid};
  });
  const result={...summaries[0],results:summaries,proof:{useId:id,...Object.fromEntries(['checkIds','resultIds','receiptIds','immunityIds','poolReceipts'].map(key=>[key,summaries.flatMap(summary=>summary.proof[key])]))}};
  const validated=await validateNativeResult({game:{users:new Map([['G',user]]),messages},fromUuid:async uuid=>documents.get(uuid),activity,permit,result});
  return {...activity,...structuredClone(validated),proof:structuredClone(validated.proof),executor:{...permit,state:'settled'},executionResult:validated};
}

test('failure stopping defaults off and does not derive an unproved zero for patients',async()=>{
  const state=await helper(),a=await savedTreatment();
  assert.deepEqual(state([a]),{streakByPatient:{},blockedPatientUUIDs:[]});
  assert.deepEqual(state([a],{enabled:false,limit:1}),{streakByPatient:{},blockedPatientUUIDs:[]});
});

test('confirmed native outcomes derive consecutive failures in persisted treatment order',async()=>{
  const state=await helper(),activities=await Promise.all(['failure','criticalFailure','success','failure'].map((outcome,index)=>savedTreatment({id:'A'+index,startedAt:index*600,patients:[{uuid:P,outcome}]})));
  assert.deepEqual(state([...activities].reverse(),enabled,context(activities)),{streakByPatient:{[P]:1},blockedPatientUUIDs:[]});
});

test('failures follow the patient across healers and shared HP does not merge streaks',async()=>{
  const state=await helper(),activities=await Promise.all([
    savedTreatment({id:'A',healer:'Actor.H1',patients:[{uuid:P,outcome:'failure'}],poolUUIDs:['Actor.M']}),
    savedTreatment({id:'B',startedAt:600,healer:'Actor.H2',patients:[{uuid:P,outcome:'criticalFailure'}],poolUUIDs:['Actor.M']}),
    savedTreatment({id:'C',startedAt:1200,patients:[{uuid:P,outcome:'failure'}],poolUUIDs:['Actor.M']}),
    savedTreatment({id:'D',startedAt:1800,patients:[{uuid:Q,outcome:'success'}],poolUUIDs:['Actor.M']})
  ]);
  assert.deepEqual(state(activities,enabled,context(activities)),{streakByPatient:{[P]:3,[Q]:0},blockedPatientUUIDs:[P]});
  assert.equal(Object.hasOwn(state(activities,enabled,context(activities)).streakByPatient,'Actor.H2'),false);
});

test('Ward results read each patient rather than copying the first top-level outcome',async()=>{
  const state=await helper(),a=await savedTreatment({patients:[{uuid:P,outcome:'failure'},{uuid:Q,outcome:'success'}]});
  a.effectiveOutcome='success';a.executionResult.effectiveOutcome='criticalSuccess';
  assert.deepEqual(state([a],{enabled:true,limit:1},context([a])),{streakByPatient:{[P]:1,[Q]:0},blockedPatientUUIDs:[P]});
});

test('the same original use and check for a patient counts once after reload',async()=>{
  const state=await helper(),a=await savedTreatment(),duplicate=structuredClone(a);
  assert.deepEqual(state([a,duplicate,a],enabled,context([a])),{streakByPatient:{[P]:1},blockedPatientUUIDs:[]});
});

test('conflicting outcomes for the same original patient check reject both arrival orders',async()=>{
  const state=await helper(),a=await savedTreatment(),conflict=structuredClone(a);conflict.executionResult.results[0].effectiveOutcome='criticalFailure';
  for(const activities of [[a,conflict],[conflict,a]])assert.throws(()=>state(activities,enabled,context([a])),/conflicting-treatment-check/);
});

test('the same original check cannot be remapped to a different patient',async()=>{
  const state=await helper(),a=await savedTreatment(),other=structuredClone(a);
  other.patientUUIDs=[Q];other.patientUUID=Q;other.executionResult.patientUUID=Q;other.executionResult.results[0].patientUUID=Q;
  for(const activities of [[a,other],[other,a]])assert.throws(()=>state(activities,enabled,context([a])),/conflicting-treatment-check/);
});

test('an extension cannot reset or increment the original treatment streak',async()=>{
  const state=await helper(),activities=await Promise.all(['success','failure','failure','failure'].map((outcome,index)=>savedTreatment({id:'A'+index,startedAt:index*600,patients:[{uuid:P,outcome}]}))),extension=structuredClone(activities[0]);
  extension.id='EXT';extension.startedAt=2400;extension.endsAt=3600;extension.options.extensionOf='A0';extension.executor.activityId='EXT';extension.executor.operationId='treatment-extension';
  activities.push(extension);
  assert.deepEqual(state(activities,enabled,context(activities)),{streakByPatient:{[P]:3},blockedPatientUUIDs:[P]});
});

test('missing, unconfirmed and other-provider records cannot reset a real failure',async()=>{
  const state=await helper(),a=await savedTreatment(),success=await savedTreatment({id:'B',startedAt:600,patients:[{uuid:P,outcome:'success'}]});
  for(const change of [row=>row.state='uncertain',row=>row.state='completing',row=>row.state='awaiting-evidence',row=>row.executor.state='granted',row=>delete row.executor,row=>delete row.executionResult,row=>row.executionResult.status='uncertain',row=>row.providerId='refocus',row=>row.providerId='focus-healing',row=>row.providerId='manual',row=>{row.providerId='generic';row.source={manual:true}},row=>{row.executionResult.proof.checkIds=[];row.executionResult.results[0].proof.checkIds=[]}]){
    const other=structuredClone(success);change(other);
    assert.deepEqual(state([a,other],enabled,context([a,other])),{streakByPatient:{[P]:1},blockedPatientUUIDs:[]});
  }
});

test('effective outcome controls streaks independently of source degree or HP delta',async()=>{
  const state=await helper(),a=await savedTreatment(),success=await savedTreatment({id:'B',startedAt:600,patients:[{uuid:P,outcome:'success'}],rolledHealing:0});
  a.executionResult.results[0].sourceDegree=3;success.executionResult.results[0].sourceDegree=0;
  assert.deepEqual(state([a],enabled,context([a])),{streakByPatient:{[P]:1},blockedPatientUUIDs:[]});
  assert.deepEqual(state([a,success],enabled,context([a,success])),{streakByPatient:{[P]:0},blockedPatientUUIDs:[]});
});

test('numeric, absent and unknown effective degrees cannot supply a counting fact',async()=>{
  const state=await helper(),a=await savedTreatment();
  for(const outcome of [0,1,2,3,null,undefined,'Failure','unknown']){
    const other=structuredClone(a);other.executionResult.results[0].effectiveOutcome=outcome;
    assert.deepEqual(state([other],enabled,context([other])),{streakByPatient:{},blockedPatientUUIDs:[]});
  }
});

test('validated single-patient legacy results count only with exact patient and native proof',async()=>{
  const state=await helper(),a=await savedTreatment();delete a.executionResult.results;
  assert.deepEqual(state([a],{enabled:true,limit:1},context([a])),{streakByPatient:{[P]:1},blockedPatientUUIDs:[P]});
  const missing=structuredClone(a);delete missing.executionResult.patientUUID;
  assert.deepEqual(state([missing],enabled,context([missing])),{streakByPatient:{},blockedPatientUUIDs:[]});
});

test('patient summaries require matching merged original use and check proof',async()=>{
  const state=await helper(),a=await savedTreatment({patients:[{uuid:P,outcome:'failure'},{uuid:Q,outcome:'success'}]});
  for(const change of [row=>row.executionResult.results[0].proof.useId='foreign',row=>row.executionResult.results[0].proof.checkIds=['foreign'],row=>row.executionResult.results[1].proof.checkIds=[row.executionResult.results[0].proof.checkIds[0]],row=>row.executionResult.results[0].patientUUID='Actor.FOREIGN',row=>row.executionResult.results[0].status='uncertain',row=>row.executionResult.results[1].patientUUID=P,row=>row.executionResult.proof.checkIds.pop(),row=>row.executor.actorUUID='Actor.FOREIGN',row=>row.executor.activityId='foreign',row=>row.proof.useId='foreign']){
    const other=structuredClone(a);change(other);assert.deepEqual(state([other],enabled,context([other])),{streakByPatient:{},blockedPatientUUIDs:[]});
  }
});

test('failure thresholds default to three and accept only limits one through one hundred',async()=>{
  const state=await helper(),activities=await Promise.all([0,1,2].map(index=>savedTreatment({id:'A'+index,startedAt:index*600})));
  assert.deepEqual(state(activities,{enabled:true},context(activities)).blockedPatientUUIDs,[P]);
  assert.deepEqual(state(activities,{enabled:true,limit:100},context(activities)).blockedPatientUUIDs,[]);
  assert.deepEqual(state([activities[0]],{enabled:true,limit:1},context(activities)).blockedPatientUUIDs,[P]);
  for(const stop of [null,[],{enabled:1},{limit:0},{limit:101},{limit:1.5},{limit:NaN},{limit:Infinity},{limit:'3'},{extra:true}])assert.throws(()=>state([],stop),/invalid-treatment-failure-stop/);
  let reads=0;const stop=Object.defineProperty({},'enabled',{enumerable:true,get(){reads++;return true}});assert.throws(()=>state([],stop),/invalid-treatment-failure-stop/);assert.equal(reads,0);
});

test('enabled derivation requires explicit session scope and persistent activity ordering',async()=>{
  const state=await helper(),a=await savedTreatment();
  for(const scope of [undefined,{}, {sessionId:'S',activityIds:[]},{sessionId:'S',activityIds:['A','A']},{sessionId:'S',activityIds:['A'],extra:true}])assert.throws(()=>state([a],enabled,scope),/invalid-treatment-failure-context/);
  const foreign=structuredClone(a);foreign.sessionId='OTHER';foreign.executor.sessionId='OTHER';
  assert.throws(()=>state([a,foreign],enabled,context([a])),/mixed-treatment-sessions/);
});

test('started time precedes persistent ordinal and completion arrival order has no effect',async()=>{
  const state=await helper(),a=await savedTreatment({id:'A',startedAt:0}),b=await savedTreatment({id:'B',startedAt:600,patients:[{uuid:P,outcome:'success'}]}),scope={sessionId:'S',activityIds:['B','A']};
  for(const activities of [[a,b],[b,a]])assert.deepEqual(state(activities,enabled,scope),{streakByPatient:{[P]:0},blockedPatientUUIDs:[]});
});

test('simultaneous treatments use the saved session ordinal',async()=>{
  const state=await helper(),a=await savedTreatment({id:'A'}),b=await savedTreatment({id:'B',patients:[{uuid:P,outcome:'success'}]});
  for(const activities of [[a,b],[b,a]])assert.deepEqual(state(activities,enabled,{sessionId:'S',activityIds:['A','B']}),{streakByPatient:{[P]:0},blockedPatientUUIDs:[]});
  assert.deepEqual(state([a,b],enabled,{sessionId:'S',activityIds:['B','A']}),{streakByPatient:{[P]:1},blockedPatientUUIDs:[]});
});

test('duplicate check facts with incompatible saved times cannot choose an arrival order',async()=>{
  const state=await helper(),a=await savedTreatment(),other=structuredClone(a);other.startedAt=600;
  for(const activities of [[a,other],[other,a]])assert.throws(()=>state(activities,enabled,context([a])),/conflicting-treatment-check/);
});

test('derivation leaves saved receipts and session order unchanged and returns detached state',async()=>{
  const state=await helper(),a=await savedTreatment(),activities=[a],scope=context(activities),before=structuredClone({activities,scope}),result=state(activities,{enabled:true,limit:1},scope);
  result.streakByPatient[P]=99;result.blockedPatientUUIDs.push(Q);
  assert.deepEqual({activities,scope},before);assert.deepEqual(state(activities,{enabled:true,limit:1},scope),{streakByPatient:{[P]:1},blockedPatientUUIDs:[P]});
});
