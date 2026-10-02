import {test} from 'node:test';
import assert from 'node:assert/strict';
import {authorityFixture} from './authority-fixture.mjs';
import {createCoordinator} from '../../scripts/exploration/coordinator.mjs';
import {createClock} from '../../scripts/exploration/clock.mjs';
import {chooseNext,recoveryProposals} from '../../scripts/exploration/policy.mjs';
import {validateNativeResult} from '../../scripts/exploration/owner-command.mjs';
import {MODULE_ID} from '../../scripts/exploration/schema.mjs';

const H='Actor.H',H2='Actor.H2',P='Actor.P',B='Actor.B',POOL='Actor.POOL';
const healing=new Set(['success','criticalSuccess']);

async function treatmentResult({game,documents,activity,permit,outcomes}){
 const summaries=activity.patientUUIDs.map((uuid,index)=>{
  const outcome=outcomes[uuid]??'success',patient=documents.get(uuid),checkId=`${activity.id}-C${index}`,resultId=`${activity.id}-R${index}`,receiptId=`${activity.id}-HP${index}`,immunityId=`${uuid}.Item.${activity.id}-I${index}`;
  const flags={pf2e:{context:{outcome,options:['exploration-activity:'+activity.id]}},[MODULE_ID]:{exploration:{activityId:activity.id,patientUUID:uuid}}},hasResult=outcome!=='failure';
  game.messages.set(checkId,{id:checkId,author:game.user,speaker:{actor:documents.get(activity.actorUUID).id},flags});
  if(hasResult){
   game.messages.set(resultId,{id:resultId,author:game.user,speaker:{actor:documents.get(activity.actorUUID).id},flags:{...structuredClone(flags),pf2e:{...structuredClone(flags.pf2e),origin:{messageId:checkId}}},rolls:[{_evaluated:true,total:healing.has(outcome)?20:3,toJSON:()=>({formula:healing.has(outcome)?'{2d8[healing]}':'{(1d8)}'})}]});
   game.messages.set(receiptId,{id:receiptId,author:game.user,speaker:{actor:patient.id},flags:{pf2e:{context:{type:'damage-taken',options:[`${MODULE_ID}:source:${resultId}:0`,`${MODULE_ID}:exploration-apply:${activity.id}:${resultId}:${uuid}`]},appliedDamage:{uuid,isReverted:false}}}});
  }
  documents.set(immunityId,{uuid:immunityId,actor:patient,flags:{[MODULE_ID]:{exploration:{activityId:activity.id,kind:'immunity'}}}});
  return {status:'confirmed',proof:{useId:activity.id,checkIds:[checkId],resultIds:hasResult?[resultId]:[],receiptIds:hasResult?[receiptId]:[],immunityIds:[immunityId],poolReceipts:[]},sourceDegree:['criticalFailure','failure','success','criticalSuccess'].indexOf(outcome),effectiveOutcome:outcome,rolledHealing:healing.has(outcome)?20:null,medicBonus:0,expiresAt:activity.startedAt+3600,resourceReceiptIds:[],patientUUID:uuid};
 });
 const result={...summaries[0],results:summaries,proof:{useId:activity.id,...Object.fromEntries(['checkIds','resultIds','receiptIds','immunityIds','poolReceipts'].map(key=>[key,summaries.flatMap(r=>r.proof[key])]))}};
 return validateNativeResult({game,fromUuid:async uuid=>documents.get(uuid),activity,permit,result});
}

async function fixture({shared=false,otherHP=20,enabled=true,limit=3,fullFocus=false,patientHealer=false,extraPatient=false}={}){
 const server=await authorityFixture(),ledger=server.client('driver'),user={id:'G',active:true,isGM:true},users=new Map([['G',user]]);users.activeGM=user;
 const documents=new Map([H,H2,P,B,POOL,...extraPatient?['Actor.C']:[]].map(uuid=>[uuid,{id:uuid.slice(6),uuid,name:uuid,flags:{},testUserPermission:()=>true,system:{attributes:{hp:{value:[P,POOL,'Actor.C'].includes(uuid)?1:uuid===B?otherHP:20,max:20}},resources:{focus:{value:1,max:1}}}}]));
 const actors=new Map([...documents.values()].map(a=>[a.id,a])),calls={begin:[],complete:[],advance:[]},hooks=new Map();let hookId=0,c;
 const game={user,users,actors,messages:new Map(),time:{worldTime:0,advance:async(dt,options)=>{calls.advance.push(dt);game.time.worldTime+=dt;for(const handler of hooks.values())handler(game.time.worldTime,dt,options,'G')}}};
 const poolUUID=uuid=>shared&&[P,B].includes(uuid)?POOL:uuid,getHpPool=actor=>({ready:true,poolUUID:poolUUID(actor.uuid)});
 const capabilities={snapshot:async uuids=>uuids.map(uuid=>{const actor=documents.get(uuid),master=documents.get(poolUUID(uuid));return {actorUUID:uuid,name:uuid,pool:getHpPool(actor),hp:structuredClone(master.system.attributes.hp),focus:structuredClone(actor.system.resources.focus),medicine:{rank:[H,H2].includes(uuid)||uuid===P&&patientHealer?1:0},nature:{rank:0},slugs:[],items:[],assuranceSkills:[],wardCapacity:2,modeOfBeing:'living',isDead:false,unconscious:false,unsupported:[],refocusUnsupported:[],vitalityHealingReady:true,wounded:uuid===P}})};
 const outcomesByActivity=new Map(),providers=['treat-wounds','refocus','focus-healing'].map(providerId=>({id:providerId,begin:async a=>{calls.begin.push(a);return {status:'started'}},cancel:async()=>{},complete:async a=>{
  calls.complete.push(a);const operationId=a.options?.extensionOf?'treatment-extension':providerId,permit=await ledger.claimExecution(a.id,{...c.executionScope('S'),operationId,ownerUserId:'G',ownerClientNonce:'owner',attemptNonce:'attempt-'+a.id,permitNonce:'permit-'+a.id});
  let result;
  if(providerId==='treat-wounds')result=await treatmentResult({game,documents,activity:a,permit,outcomes:outcomesByActivity.get(a.id)??{}});
  else {const actor=documents.get(a.actorUUID),before=actor.system.resources.focus.value;actor.system.resources.focus.value=Math.min(actor.system.resources.focus.max,before+1);actor.flags[MODULE_ID]={avRefocusIntent:{nonce:a.id,actorUuid:actor.uuid,userId:'G',startedAt:a.startedAt,before,after:actor.system.resources.focus.value}};result=await validateNativeResult({game,fromUuid:async uuid=>documents.get(uuid),activity:a,permit,result:{status:'confirmed',focusBefore:before,focusAfter:actor.system.resources.focus.value,proof:{useId:a.id,checkIds:[],resultIds:[],receiptIds:[a.id],immunityIds:[]}}})}
  await ledger.recordExecutionResult(a.id,{permit,result});for(const summary of result.results??[])if(healing.has(summary.effectiveOutcome))documents.get(poolUUID(summary.patientUUID)).system.attributes.hp.value=20;return result;
 }}));
 const Hooks={on:(_event,fn)=>{hooks.set(++hookId,fn);return hookId},off:(_event,id)=>hooks.delete(id)},clock=createClock({game,Hooks,ledger,isAuthority:()=>true,timeEffects:{beforeAdvance:async()=>({status:'ready'}),settle:async()=>({status:'ready',proof:[]})},confirmationTimeoutMs:100});
 c=createCoordinator({game,fromUuid:async uuid=>documents.get(uuid),getHpPool,ledger,capabilities,providers,clock,policy:chooseNext,isAuthority:()=>true,now:()=>game.time.worldTime,ownerOperations:{createActivityContext:async()=>({}),cancelActivity:async()=>{}}});
 const selected=[H,H2,P,B,...extraPatient?['Actor.C']:[]];await c.start({id:'S',actorUUIDs:selected,budgetSeconds:21600,maxActivities:100,requireFullFocus:fullFocus,autoRun:false,recovery:{version:1,targetIntentsByActor:{},requireNoWounded:false,failureStop:{enabled,limit}}});
 const waitUntil=async to=>{const receipt=await clock.advanceTo({id:crypto.randomUUID(),sessionId:'S',from:game.time.worldTime,to},c.executionScope('S'));assert.equal(receipt.status,'confirmed');await ledger.updateSession('S',{cursorAt:game.time.worldTime},c.executionScope('S'))};
 const admit=(id,patientUUIDs=[P],actorUUID=H,options={})=>c.addActivity('S',{id,providerId:'treat-wounds',actorUUID,patientUUIDs,hpPoolUUIDs:[...new Set(patientUUIDs.map(poolUUID))],startedAt:game.time.worldTime,endsAt:game.time.worldTime+600,options:{skill:'medicine',rank:'trained',continualRecovery:false,...options}});
 const seed=async(id,outcomes,actorUUID=H)=>{const a=await admit(id,Object.keys(outcomes),actorUUID);outcomesByActivity.set(id,outcomes);await waitUntil(a.endsAt);const claimed=await ledger.transitionActivity(id,{expected:['started'],patch:{state:'completing'},...c.executionScope('S')}),result=await providers[0].complete(claimed);return ledger.transitionActivity(id,{expected:['completing'],patch:{...result,state:'confirmed'},...c.executionScope('S')})};
 const failures=async()=>{for(let i=0;i<3;i++){if(i)await waitUntil(game.time.worldTime+3000);await seed('F'+i,{[P]:i===1?'criticalFailure':'failure'},i===1?H2:H)}};
 const proposals=async()=>{const data=await c.snapshot('S');return recoveryProposals({actors:data.actors,activities:data.activities,session:data.session,now:game.time.worldTime,providerIds:providers.map(p=>p.id)})};
 return {server,ledger,c,game,documents,calls,capabilities,providers,poolUUID,waitUntil,admit,seed,failures,proposals};
}

test('policy blocks patient recovery across healers while preserving that actor as a healer and independent Refocus',async()=>{
 const f=await fixture({otherHP:1,patientHealer:true,fullFocus:true});await f.failures();f.documents.get(P).system.resources.focus.value=0;
 const proposals=await f.proposals();assert.ok(proposals.length);assert.ok(proposals.every(a=>!a.patientUUIDs.includes(P)));
 assert.ok(proposals.some(a=>a.actorUUID===P&&a.providerId==='treat-wounds'&&a.patientUUIDs.includes(B)));
 assert.ok(proposals.some(a=>a.actorUUID===P&&a.providerId==='refocus'&&a.patientUUIDs.length===0));
});

test('shared HP eligibility is per patient and never lowers the frozen pool goal',async()=>{
 const f=await fixture({shared:true});await f.failures();const before=await f.ledger.getSession('S'),proposals=await f.proposals();
 assert.ok(proposals.some(a=>a.patientUUIDs.includes(B)));assert.ok(proposals.every(a=>!a.patientUUIDs.includes(P)));
 await f.c.step('S');await f.c.step('S');assert.equal((await f.ledger.getSession('S')).status,'complete');assert.deepEqual((await f.ledger.getSession('S')).goalsByPool,before.goalsByPool);
 assert.equal((await f.ledger.snapshot('S')).activities.filter(a=>a.patientUUIDs.includes(P)).length,3);assert.equal(f.documents.get(P).system.attributes.hp.value,1);
});

test('Ward regrouping contains only remaining eligible patients and preserves separate pools',async()=>{
 const f=await fixture({otherHP:1,extraPatient:true});await f.failures();const groups=(await f.proposals()).filter(a=>a.patientUUIDs.length>1);
 assert.ok(groups.length);for(const group of groups){assert.deepEqual(new Set(group.patientUUIDs),new Set([B,'Actor.C']));assert.equal(new Set(group.hpPoolUUIDs).size,2)}
 await f.c.step('S');assert.equal(f.documents.get(B).system.attributes.hp.value,20);assert.equal(f.documents.get('Actor.C').system.attributes.hp.value,20);
 assert.equal((await f.c.step('S')).stopReason,'consecutive-treatment-failures');const data=await f.c.snapshot('S');assert.equal(data.treatmentFailures.streakByPatient[P],3);assert.equal(data.treatmentFailures.streakByPatient[B],0);assert.equal(data.treatmentFailures.streakByPatient['Actor.C'],0);
});

test('other patients and their reachable cooldowns finish before the failure pause',async()=>{
 const f=await fixture({otherHP:1});await f.failures();await f.seed('B-cooldown',{[B]:'failure'});const at=f.game.time.worldTime;
 await f.c.step('S');assert.equal(f.game.time.worldTime,at+3000);assert.equal((await f.ledger.getSession('S')).status,'running');
 await f.c.step('S');assert.equal(f.documents.get(B).system.attributes.hp.value,20);const stopped=await f.c.step('S');assert.equal(stopped.stopReason,'consecutive-treatment-failures');
 const data=await f.c.snapshot('S');assert.deepEqual(data.treatmentFailures,{streakByPatient:{[P]:3,[B]:0},blockedPatientUUIDs:[P],remaining:[{patientUUID:P,poolUUID:P,streak:3,limit:3,currentHP:1,targetHP:20,hpGap:19}]});
 assert.equal(Object.hasOwn(await f.ledger.getSession('S'),'treatmentFailures'),false);
});

test('an exhausted blocked patient pauses without waiting for that patient cooldown or retrying after resume',async()=>{
 const f=await fixture();await f.failures();const before=f.game.time.worldTime,calls=f.calls.begin.length;
 const stopped=await f.c.step('S');assert.equal(stopped.stopReason,'consecutive-treatment-failures');assert.equal(f.game.time.worldTime,before);assert.equal(f.calls.begin.length,calls);
 await f.c.resume('S',{autoRun:false});assert.equal((await f.c.step('S')).stopReason,'consecutive-treatment-failures');assert.equal(f.calls.begin.length,calls);
 f.documents.get(P).system.attributes.hp.value=18;assert.equal((await f.c.snapshot('S')).treatmentFailures.remaining[0].hpGap,2);
});

test('an unavailable unrelated recovery source does not hide the exhausted failure stop',async()=>{
 const f=await fixture();await f.failures();const original=f.capabilities.snapshot;
 f.capabilities.snapshot=async uuids=>(await original(uuids)).map(a=>a.actorUUID===H?{...a,threePecks:true,refocusUnsupported:['special-refocus']}:a);
 assert.equal((await f.c.step('S')).stopReason,'consecutive-treatment-failures');assert.equal((await f.c.snapshot('S')).treatmentFailures.remaining[0].patientUUID,P);
});

test('unknown evidence remains the first pause and starts no replacement work or time',async()=>{
 const f=await fixture();await f.failures();const scope=f.c.executionScope('S'),a=await f.ledger.insertActivity({id:'unknown',sessionId:'S',providerId:'refocus',actorUUID:H,patientUUIDs:[],hpPoolUUIDs:[],state:'planned',startedAt:f.game.time.worldTime,endsAt:f.game.time.worldTime+600,options:{},source:{type:'coordinator'}},scope);
 await f.ledger.transitionActivity(a.id,{expected:['planned'],patch:{state:'started'},...scope});await f.ledger.transitionActivity(a.id,{expected:['started'],patch:{state:'uncertain'},...scope});const before=f.game.time.worldTime,calls=f.calls.begin.length;
 assert.equal((await f.c.step('S')).stopReason,'unresolved-evidence');assert.equal(f.game.time.worldTime,before);assert.equal(f.calls.begin.length,calls);assert.equal((await f.c.snapshot('S')).treatmentFailures,undefined);
});

test('common admission prevents a new native patient proposal from bypassing the threshold',async()=>{
 const f=await fixture();await f.failures();const before=(await f.ledger.snapshot('S')).activities.length,calls=f.calls.begin.length;
 await assert.rejects(f.admit('retry',[P],H2),/consecutive-treatment-failures/);await assert.rejects(f.admit('bare-extension',[P],H2,{extensionOf:'invented'}),/extension|consecutive-treatment-failures/);
 assert.equal((await f.ledger.snapshot('S')).activities.length,before);assert.equal(f.calls.begin.length,calls);
});

test('common admission applies to other patient recovery sources while independent Refocus still completes',async()=>{
 const f=await fixture({fullFocus:true});await f.failures();const at=f.game.time.worldTime;
 for(const input of [{providerId:'focus-healing',options:{itemUUID:H+'.Item.LOH'}},{providerId:'refocus',options:{threePecks:true}}])await assert.rejects(f.c.addActivity('S',{id:input.providerId,actorUUID:H,patientUUIDs:[P],hpPoolUUIDs:[P],startedAt:at,endsAt:at+600,...input}),/consecutive-treatment-failures/);
 f.documents.get(P).system.resources.focus.value=0;
 await f.c.addActivity('S',{id:'independent-refocus',providerId:'refocus',actorUUID:P,patientUUIDs:[],hpPoolUUIDs:[],startedAt:at,endsAt:at+600,options:{}});
 await f.c.step('S');assert.equal((await f.ledger.getActivity('independent-refocus')).state,'confirmed');assert.equal(f.game.time.worldTime,at+600);
 assert.equal((await f.c.step('S')).stopReason,'consecutive-treatment-failures');
});

test('an old successful check cannot disguise a backdated new proposal as a continuation',async()=>{
 const f=await fixture({limit:1}),original=await f.seed('success',{[P]:'success'});f.documents.get(P).system.attributes.hp.value=1;
 await f.waitUntil(3600);await f.seed('failure',{[P]:'failure'},H2);
 await assert.rejects(f.c.addActivity('S',{id:'backdated',providerId:'treat-wounds',actorUUID:H,patientUUIDs:[P],hpPoolUUIDs:[P],startedAt:original.endsAt,endsAt:original.startedAt+3600,options:{extensionOf:original.id}}),/extension|consecutive-treatment-failures/);
 assert.equal(await f.ledger.getActivity('backdated'),null);
});

test('registered generic activity by a blocked patient completes before the failure pause',async()=>{
 const f=await fixture();await f.failures();const at=f.game.time.worldTime,binding=await f.c.openActivityCheckpoint('S');
 await f.ledger.enrollCheckpointActivity(binding,{registrationId:'search',checkpointBinding:binding,actorUUID:P,label:'Search',durationSeconds:300},{...f.c.executionScope('S'),authenticatedCaller:'G',guard:()=>true});
 await f.c.closeActivityCheckpoint(binding,{autoRun:false});await f.c.step('S');assert.equal(f.game.time.worldTime,at+300);assert.equal((await f.ledger.getSession('S')).status,'running');
 const declaration=(await f.ledger.snapshot('S')).activities.find(a=>a.registrationId==='search');assert.equal(declaration.state,'confirmed');assert.equal(declaration.executor,undefined);
 assert.equal((await f.c.step('S')).stopReason,'consecutive-treatment-failures');
});

test('patient filtering also applies to Natural Medicine, focus healing and Three Pecks proposals',async()=>{
 const f=await fixture({otherHP:1});await f.failures();const data=await f.c.snapshot('S'),healer=data.actors.find(a=>a.actorUUID===H);
 healer.slugs=['natural-medicine'];healer.nature={rank:1};healer.threePecks=true;healer.items=[{uuid:H+'.Item.LOH',sourceId:'Compendium.pf2e.spells-srd.Item.zNN9212H2FGfM7VS'}];
 const proposals=recoveryProposals({actors:data.actors,activities:data.activities,session:data.session,now:f.game.time.worldTime,providerIds:['treat-wounds','focus-healing','refocus']});
 assert.ok(proposals.every(a=>!a.patientUUIDs.includes(P)));assert.ok(proposals.some(a=>a.options?.skill==='nature'&&a.patientUUIDs.includes(B)));
 assert.ok(proposals.some(a=>a.providerId==='focus-healing'&&a.patientUUIDs.includes(B)));assert.ok(proposals.some(a=>a.options?.threePecks&&a.patientUUIDs.includes(B)));
});

test('disabled failure stopping keeps original proposal eligibility after the same confirmed failures',async()=>{
 const f=await fixture({enabled:false});await f.failures();assert.ok((await f.proposals()).some(a=>a.patientUUIDs.includes(P)));
 await f.c.step('S');assert.equal((await f.ledger.getSession('S')).status,'running');assert.equal(f.game.time.worldTime,10800);
});
