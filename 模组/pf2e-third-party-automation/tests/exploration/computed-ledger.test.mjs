import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {createLedger} from '../../scripts/exploration/ledger.mjs';
import {canonicalJSON} from '../../scripts/exploration/revision-codec.mjs';
import {authorityFixture} from './authority-fixture.mjs';

const config=(id='S',patch={})=>({id,executionMode:'computed-v1',actorUUIDs:['Actor.A'],startedAt:0,cursorAt:0,budgetEndsAt:3600,goalsByPool:[{poolUUID:'Actor.A',targetHP:20}],...patch});
const payload=(id='B',patch={})=>({id,from:0,to:0,beforeActors:[{actorUUID:'Actor.A',hp:{value:5}}],afterActors:[{actorUUID:'Actor.A',hp:{value:9}}],events:[{die:12,healing:4}],documents:{id,at:patch.to??0,actors:[],items:[],pools:[]},...patch});
async function fixture(){
 const server=await authorityFixture();let valid=true,authority=true,finalHook;
 const store=server.storage('driver');
 const ledger=createLedger({...store,isAuthority:()=>authority,identity:()=>({userId:'G',clientNonce:'driver'}),transact:(fn,options)=>store.transact(fn,options?.validateCommit?{validateCommit:()=>{finalHook?.();return options.validateCommit()}}:options)});
 const session=await ledger.createSession(config()),scope={leaseNonce:session.driver.leaseNonce,guard:()=>valid};
 return {server,ledger,session,scope,onFinal:fn=>finalHook=fn,invalidate:()=>valid=false,loseAuthority:()=>authority=false,
  advance:async to=>{await store.transact(state=>{state.clocks['clock-'+to]={id:'clock-'+to,sessionId:'S',from:0,to,state:'confirmed'};return null});await ledger.updateSession('S',{cursorAt:to},scope)}};
}
function requireAPI(ledger){assert.equal(typeof ledger.claimComputedBatch,'function','computed batch claim must exist');assert.equal(typeof ledger.transitionComputedBatch,'function','computed batch transition must exist')}
async function claim(f,input=payload()){requireAPI(f.ledger);return f.ledger.claimComputedBatch('S',input,f.scope)}
async function phases(f,id='B'){
 await f.ledger.transitionComputedBatch('S',id,{expected:['claimed'],state:'claimed',phase:'actors',receipt:{actors:[]}},f.scope);
 await f.ledger.transitionComputedBatch('S',id,{expected:['claimed'],state:'claimed',phase:'items',receipt:{items:[],wounded:[]}},f.scope);
}
async function confirm(f,id='B'){
 await phases(f,id);return f.ledger.transitionComputedBatch('S',id,{expected:['claimed'],state:'confirmed',summary:{healing:4},simulationState:{cursorAt:0}},f.scope);
}

test('computed creation initializes protected execution state and rejects injected batches',async()=>{
 const f=await fixture();assert.deepEqual(f.session.computedRecovery,{version:1,batches:[],summary:null});
 await assert.rejects(f.ledger.createSession(config('injected',{computedRecovery:{version:1,batches:[],summary:{forged:true}}})),/computed.*required|invalid-computed/);
 for(const patch of [{computedRecovery:{}},{executionMode:'native'},{computedInitialActors:[]},{actorUUIDs:[]}])await assert.rejects(f.ledger.updateSession('S',patch,f.scope),/immutable-session/);
});

test('creation cannot inject a completed or already advanced computed session',async()=>{
 for(const patch of [{status:'complete'},{manual:true},{cursorAt:600},{activityIds:['old']},{executionMode:'computed-v2'}]){
  const server=await authorityFixture();await assert.rejects(server.client('driver').createSession(config('S',patch)),/invalid.*session|invalid.*mode|computed.*required/);
 }
});

test('sampled payload is detached, hashed, and cannot be claimed twice with different samples',async()=>{
 const f=await fixture(),input=payload(),expected=structuredClone(input),pending=claim(f,input);input.events[0].die=20;
 const batch=await pending;assert.deepEqual(batch.events,[{die:12,healing:4}]);assert.equal(batch.state,'claimed');assert.deepEqual(batch.receipts,{});
 assert.equal(batch.digest,createHash('sha256').update(canonicalJSON(expected)).digest('hex'));
 await assert.rejects(claim(f,payload('B',{events:[{die:1,healing:0}]})),/duplicate-computed-batch/);
 assert.deepEqual((await f.ledger.getComputedBatch('S','B')).events,[{die:12,healing:4}]);
});

test('two tabs cannot race new sessions or claim another driver batch',async()=>{
 const server=await authorityFixture(),one=server.client('one'),two=server.client('two');
 const results=await Promise.allSettled([one.createSession(config('one')),two.createSession(config('two'))]);assert.equal(results.filter(row=>row.status==='fulfilled').length,1);
 const session=results.find(row=>row.status==='fulfilled').value,other=session.id==='one'?two:one;requireAPI(other);
 await assert.rejects(other.claimComputedBatch(session.id,payload(),{leaseNonce:session.driver.leaseNonce,guard:()=>true}),/session-driver-required/);
 assert.equal(Object.keys((await server.read()).sessions).length,1);
});

test('claim rechecks its source guard and authority immediately before revision submission',async()=>{
 for(const loss of ['invalidate','loseAuthority']){
  const f=await fixture();f.onFinal(()=>f[loss]());
  await assert.rejects(claim(f),/computed.*changed|active-gm-required/);assert.equal((await f.server.read()).sessions.S.computedRecovery.batches.length,0);
 }
});

test('claims reject foreign epochs and legacy sessions',async()=>{
 const f=await fixture();requireAPI(f.ledger);
 await f.server.storage('editor').transact(state=>{state.sessions.S.protocol.epoch='other';return null});
 await assert.rejects(claim(f),/computed.*scope/);
 const old=await authorityFixture(),ledger=old.client('old');await ledger.createSession({...config(),executionMode:undefined});
 await assert.rejects(ledger.claimComputedBatch('S',payload(),{guard:()=>true}),/computed-session-required/);
});

test('pre-clock and post-clock batches bind their boundary and cannot confirm early',async()=>{
 const f=await fixture();await claim(f,payload('B',{to:600,documents:{id:'B',at:600,actors:[],items:[],pools:[]}}));await phases(f);
 await assert.rejects(f.ledger.transitionComputedBatch('S','B',{expected:['claimed'],state:'confirmed'},f.scope),/computed.*cursor/);
 await f.advance(600);await f.ledger.transitionComputedBatch('S','B',{expected:['claimed'],state:'confirmed'},f.scope);
 await f.advance(1200);await claim(f,payload('C',{from:600,to:1200,documents:{id:'C',at:1200,actors:[],items:[],pools:[]}}));
 await assert.rejects(claim(f,payload('D',{from:1200,to:1800})),/unresolved-computed-batch/);
 const g=await fixture();await assert.rejects(claim(g,payload('bad',{from:10,to:20})),/computed.*boundary/);
 await assert.rejects(claim(g,payload('backward',{from:0,to:-1})),/computed.*boundary/);
});

test('phase order, immutable payload, complete receipts and predecessor state are enforced',async()=>{
 const f=await fixture(),documents={id:'B',at:0,actors:[{uuid:'Actor.A',changes:{hp:9}}],items:[{actorUUID:'Actor.A',create:[{_id:'I'}],removeWounded:true}],pools:[]};await claim(f,payload('B',{documents}));
 for(const transition of [
  {state:'confirmed'},
  {state:'claimed',phase:'unknown',receipt:{}},
  {state:'claimed',phase:'items',receipt:{items:[],wounded:[]}},
  {state:'claimed',phase:'actors',receipt:{actors:[]}},
  {state:'claimed',events:[]}
 ])await assert.rejects(f.ledger.transitionComputedBatch('S','B',{expected:['claimed'],...transition},f.scope),/computed.*phase|computed.*receipt|immutable-computed/);
 const actors={actors:[{actorUUID:'Actor.A',batchId:'B',after:{hp:9}}]};
 await f.ledger.transitionComputedBatch('S','B',{expected:['claimed'],state:'claimed',phase:'actors',receipt:actors},f.scope);
 await assert.rejects(f.ledger.transitionComputedBatch('S','B',{expected:['claimed'],state:'claimed',phase:'actors',receipt:actors},f.scope),/computed.*phase/);
 await assert.rejects(f.ledger.transitionComputedBatch('S','B',{expected:['claimed'],state:'claimed',phase:'items',receipt:{items:[{actorUUID:'Actor.A',itemUUID:'Actor.A.Item.I',batchId:'B'}],wounded:[]}},f.scope),/computed.*receipt/);
 await f.ledger.transitionComputedBatch('S','B',{expected:['claimed'],state:'claimed',phase:'items',receipt:{items:[{actorUUID:'Actor.A',itemUUID:'Actor.A.Item.I',batchId:'B'}],wounded:['Actor.A']}},f.scope);
 const result=await f.ledger.transitionComputedBatch('S','B',{expected:['claimed'],state:'confirmed',summary:{healing:4},simulationState:{cursorAt:0}},f.scope);
 assert.equal(result.state,'confirmed');const saved=await f.ledger.getSession('S');assert.deepEqual(saved.computedRecovery.summary,{healing:4});assert.deepEqual(saved.computedRecovery.simulationState,{cursorAt:0});assert.deepEqual((await f.ledger.getComputedBatch('S','B')).documents,documents);
 await assert.rejects(f.ledger.transitionComputedBatch('S','B',{expected:['claimed'],state:'uncertain'},f.scope),/computed.*state/);
});

test('phase confirmation rechecks its guard at commit and leaves the receipt absent on failure',async()=>{
 const f=await fixture();await claim(f);f.onFinal(()=>f.invalidate());
 await assert.rejects(f.ledger.transitionComputedBatch('S','B',{expected:['claimed'],state:'claimed',phase:'actors',receipt:{actors:[]}},f.scope),/computed.*changed/);
 assert.deepEqual((await f.ledger.getComputedBatch('S','B')).receipts,{});
});

test('unknown claim acknowledgement persists the sampled claim and prevents a fresh replay',async()=>{
 const f=await fixture();f.server.setAcknowledgement(()=>({}));await assert.rejects(claim(f),/acknowledgement-unknown/);
 f.server.setAcknowledgement(ack=>ack);await assert.rejects(claim(f),/duplicate-computed-batch/);
 await f.ledger.updateSession('S',{status:'paused'},f.scope);
 await assert.rejects(f.server.client('other').createSession(config('next')),/unresolved-evidence-no-replay/);
 assert.equal((await f.ledger.getSession('S')).computedRecovery.batches[0].state,'claimed');
});

test('unknown phase acknowledgement cannot grant a duplicate phase or promote unknown results',async()=>{
 const f=await fixture();await claim(f);f.server.setAcknowledgement(()=>({}));
 await assert.rejects(f.ledger.transitionComputedBatch('S','B',{expected:['claimed'],state:'claimed',phase:'actors',receipt:{actors:[]}},f.scope),/acknowledgement-unknown/);
 f.server.setAcknowledgement(ack=>ack);
 await assert.rejects(f.ledger.transitionComputedBatch('S','B',{expected:['claimed'],state:'claimed',phase:'actors',receipt:{actors:[]}},f.scope),/computed.*phase/);
 await f.ledger.transitionComputedBatch('S','B',{expected:['claimed'],state:'uncertain',reason:'unknown-ack'},{leaseNonce:f.scope.leaseNonce});
 await assert.rejects(f.ledger.transitionComputedBatch('S','B',{expected:['uncertain'],state:'confirmed'},f.scope),/computed.*state/);
});

test('takeover preserves unknown samples, blocks resume, and explicit review releases only their occupancy',async()=>{
 const f=await fixture();await claim(f);const other=f.server.client('other');await other.takeoverSession('S');
 const unknown=(await f.ledger.getSession('S')).computedRecovery.batches[0];assert.equal(unknown.state,'uncertain');
 await assert.rejects(other.resumeSession('S',{cursorAt:0}),/unresolved-computed-batch/);
 await assert.rejects(other.createSession(config('next')),/unresolved-evidence-no-replay/);
 assert.equal(typeof other.reviewComputedSession,'function');
 await assert.rejects(other.reviewComputedSession('S',{note:'checked',userId:'wrong',at:0}),/review.*identity/);
 await assert.rejects(other.reviewComputedSession('S',{note:' ',userId:'G',at:0}),/review-note-required/);
 const reviewed=await other.reviewComputedSession('S',{note:'checked Actor and items',userId:'G',at:0});assert.equal(reviewed.status,'closed');assert.equal(reviewed.computedRecovery.review.disposition,'closed-without-replay');
 assert.deepEqual(reviewed.computedRecovery.batches[0],unknown);await other.createSession(config('next'));
});

test('finish persists a summary without activities and cannot hide pending work',async()=>{
 const f=await fixture();assert.equal(typeof f.ledger.finishComputedSession,'function');
 await claim(f);await assert.rejects(f.ledger.finishComputedSession('S',{status:'complete',stopReason:'goals-met',summary:{},simulationState:{cursorAt:0}},f.scope),/unresolved-computed-batch/);
 await assert.rejects(f.ledger.updateSession('S',{status:'complete'},f.scope),/computed.*finish/);
 await confirm(f);const saved=await f.ledger.finishComputedSession('S',{status:'complete',stopReason:'goals-met',summary:{hp:9},simulationState:{cursorAt:0}},f.scope);assert.equal(saved.status,'complete');assert.deepEqual(saved.computedRecovery.summary,{hp:9});
 const g=await fixture();const complete=await g.ledger.finishComputedSession('S',{status:'complete',stopReason:'goals-met',summary:{hp:20},simulationState:{cursorAt:0}},g.scope);assert.equal(complete.computedRecovery.batches.length,0);assert.deepEqual(complete.computedRecovery.summary,{hp:20});
});

test('review closes uncertain time and batch occupancy without changing either saved result',async()=>{
 const f=await fixture();await claim(f);
 await f.server.storage('clock').transact(state=>{state.clocks.C={id:'C',sessionId:'S',from:0,to:600,state:'uncertain',reason:'unknown-clock-ack'};return null});
 await f.ledger.takeoverSession('S');const before=await f.server.read();
 await f.ledger.reviewComputedSession('S',{note:'checked time and Actor state',at:600,userId:'G'});
 const other=f.server.client('other'),next=await other.createSession(config('next',{startedAt:600,cursorAt:600}));
 await other.upsertClockCommit({id:'next-clock',sessionId:'next',gmId:'G',from:600,to:1200,state:'started'},{leaseNonce:next.driver.leaseNonce});
 const after=await f.server.read();assert.deepEqual(after.clocks.C,before.clocks.C);assert.deepEqual(after.sessions.S.computedRecovery.batches,before.sessions.S.computedRecovery.batches);assert.deepEqual(after.activities.B,before.activities.B);
});

test('a stale lease cannot claim, record receipts, or finish a computed session',async()=>{
 const f=await fixture(),stale={leaseNonce:'stale',guard:()=>true};requireAPI(f.ledger);
 await assert.rejects(f.ledger.claimComputedBatch('S',payload(),stale),/session-driver-required/);await claim(f);
 await assert.rejects(f.ledger.transitionComputedBatch('S','B',{expected:['claimed'],state:'uncertain'},stale),/session-driver-required/);
 await assert.rejects(f.ledger.finishComputedSession('S',{status:'paused',summary:{}},stale),/session-driver-required/);
});

test('finish rechecks its guard before storing a no-activity summary',async()=>{
 const f=await fixture();assert.equal(typeof f.ledger.finishComputedSession,'function');f.onFinal(()=>f.invalidate());
 await assert.rejects(f.ledger.finishComputedSession('S',{status:'complete',summary:{hp:20},simulationState:{cursorAt:0}},f.scope),/computed.*changed/);
 const saved=(await f.server.read()).sessions.S;assert.equal(saved.status,'running');assert.equal(saved.computedRecovery.summary,null);
});

test('session revisions keep compact batch indexes and snapshot hydrates independent payloads',async()=>{
 const f=await fixture(),first=await claim(f,payload('B',{events:[{detail:'first-sample-'.repeat(1000)}]}));
 assert.deepEqual((await f.ledger.getSession('S')).computedRecovery.batches,[{id:'B',state:'claimed',from:0,to:0}]);
 const record=(await f.server.read()).activities.B;assert.equal(record.kind,'computed-recovery-batch');assert.equal(record.providerId,'computed-recovery');assert.deepEqual(record.patientUUIDs,['Actor.A']);assert.deepEqual(record.hpPoolUUIDs,['Actor.A']);assert.equal(record.actorUUID,'Actor.A');
 assert.deepEqual(await f.ledger.getComputedBatch('S','B'),first);
 await f.ledger.transitionComputedBatch('S','B',{expected:['claimed'],state:'confirmed',receipts:{actors:{actors:[]},items:{items:[],wounded:[]}}},f.scope);
 await claim(f,payload('C'));
 const revision=JSON.parse(f.server.raw.pages.at(-1).flags['pf2e-third-party-automation'].explorationRevision);
 assert.deepEqual(Object.keys(revision.changes.activities),['C']);assert.equal(JSON.stringify(revision.changes.sessions).includes('first-sample-'),false);
 const snapshot=await f.ledger.snapshot('S');assert.deepEqual(snapshot.session.computedRecovery.batches[0].events,first.events);assert.deepEqual(snapshot.activities,[]);
});

test('a bundled confirmation stores both receipts and matching compact state in one revision',async()=>{
 const f=await fixture();await claim(f);const count=f.server.raw.pages.length;
 const batch=await f.ledger.transitionComputedBatch('S','B',{expected:['claimed'],state:'confirmed',receipts:{actors:{actors:[]},items:{items:[],wounded:[]}},summary:{healing:4},simulationState:{cursorAt:0}},f.scope);
 assert.equal(f.server.raw.pages.length,count+1);assert.equal(batch.state,'confirmed');assert.deepEqual(batch.receipts,{actors:{actors:[]},items:{items:[],wounded:[]}});
 assert.deepEqual((await f.ledger.getSession('S')).computedRecovery.batches,[{id:'B',state:'confirmed',from:0,to:0}]);assert.equal((await f.ledger.getComputedBatch('S','B')).state,'confirmed');
});

test('bundled confirmation rejects incomplete or unknown receipts without changing either record',async()=>{
 const f=await fixture();await claim(f);
 for(const receipts of [{actors:{actors:[]}},{items:{items:[],wounded:[]}},{actors:{actors:[]},items:{items:[],wounded:[]},other:{}},{actors:{actors:[{unknown:true}]},items:{items:[],wounded:[]}}]){
  await assert.rejects(f.ledger.transitionComputedBatch('S','B',{expected:['claimed'],state:'confirmed',receipts},f.scope),/computed.*phase|computed.*receipt/);
  assert.equal((await f.ledger.getSession('S')).computedRecovery.batches[0].state,'claimed');assert.deepEqual((await f.ledger.getComputedBatch('S','B')).receipts,{});
 }
});

test('uncertain records preserve completed bundled phases and takeover keeps the index aligned',async()=>{
 const f=await fixture();await claim(f);
 await f.ledger.transitionComputedBatch('S','B',{expected:['claimed'],state:'uncertain',reason:'items-ack-unknown',receipts:{actors:{actors:[]}}},{leaseNonce:f.scope.leaseNonce});
 const detail=await f.ledger.getComputedBatch('S','B');assert.equal(detail.state,'uncertain');assert.deepEqual(detail.receipts,{actors:{actors:[]}});
 assert.deepEqual((await f.ledger.getSession('S')).computedRecovery.batches,[{id:'B',state:'uncertain',from:0,to:0,reason:'items-ack-unknown'}]);
 const g=await fixture();await claim(g);await g.server.client('peer').takeoverSession('S');assert.equal((await g.ledger.getComputedBatch('S','B')).state,'uncertain');assert.equal((await g.ledger.getSession('S')).computedRecovery.batches[0].state,'uncertain');
});

test('unknown bundled acknowledgement cannot retry either the full batch or its compact state',async()=>{
 const f=await fixture();await claim(f);f.server.setAcknowledgement(()=>({}));
 const input={expected:['claimed'],state:'confirmed',receipts:{actors:{actors:[]},items:{items:[],wounded:[]}}};
 await assert.rejects(f.ledger.transitionComputedBatch('S','B',input,f.scope),/acknowledgement-unknown/);f.server.setAcknowledgement(ack=>ack);
 const pages=f.server.raw.pages.length;await assert.rejects(f.ledger.transitionComputedBatch('S','B',input,f.scope),/computed.*state/);assert.equal(f.server.raw.pages.length,pages);
 assert.equal((await f.ledger.getSession('S')).computedRecovery.batches[0].state,'confirmed');assert.equal((await f.ledger.getComputedBatch('S','B')).state,'confirmed');
});

test('generic activity APIs cannot insert, transition, or settle a computed batch record',async()=>{
 const f=await fixture();await claim(f);
 await assert.rejects(f.ledger.insertActivity({id:'forged',sessionId:'S',kind:'computed-recovery-batch',providerId:'computed-recovery',actorUUID:'Actor.A',patientUUIDs:[],hpPoolUUIDs:[],startedAt:0,endsAt:0,state:'planned'},f.scope),/computed-batch-lifecycle-required/);
 await assert.rejects(f.ledger.transitionActivity('B',{expected:['claimed'],patch:{state:'confirmed'},...f.scope}),/computed-batch-lifecycle-required/);
 await assert.rejects(f.ledger.claimExecution('B',f.scope),/computed-batch-lifecycle-required/);
 await assert.rejects(f.ledger.recordExecutionResult('B',{permit:{},result:{status:'confirmed'}}),/computed-batch-lifecycle-required/);
 assert.equal((await f.ledger.getComputedBatch('S','B')).state,'claimed');
});

test('a generic native activity cannot acquire the computed discriminator to bypass occupancy',async()=>{
 const server=await authorityFixture(),ledger=server.client('driver'),session=await ledger.createSession({...config(),executionMode:undefined}),scope={leaseNonce:session.driver.leaseNonce};
 await ledger.insertActivity({id:'A',sessionId:'S',providerId:'refocus',actorUUID:'Actor.A',patientUUIDs:[],hpPoolUUIDs:[],startedAt:0,endsAt:600,state:'planned',source:{type:'coordinator'}},scope);
 await assert.rejects(ledger.transitionActivity('A',{...scope,expected:['planned'],patch:{kind:'computed-recovery-batch'}}),/immutable-provenance/);
 assert.equal((await ledger.getActivity('A')).kind,undefined);
});

test('post-clock sampling boundaries persist with the cursor and block replay before a batch exists',async()=>{
 const f=await fixture(),before=f.server.raw.pages.length;await f.advance(600);
 assert.equal(f.server.raw.pages.length,before+2,'the clock fixture and cursor write must not gain a boundary-only transaction');
 let session=await f.ledger.getSession('S');assert.deepEqual(session.computedRecovery.pendingBoundary,{from:0,to:600});assert.deepEqual(session.computedRecovery.batches,[]);
 await assert.rejects(f.ledger.finishComputedSession('S',{status:'complete',summary:{},simulationState:{cursorAt:600}},f.scope),/unresolved-computed/);
 await f.ledger.updateSession('S',{status:'paused',stopReason:'computed-document-preparation-failed'},f.scope);
 await assert.rejects(f.ledger.resumeSession('S',{cursorAt:600}),/unresolved-computed/);
 await assert.rejects(f.server.client('other').createSession(config('other',{startedAt:600,cursorAt:600})),/unresolved-evidence-no-replay/);
 await f.server.client('other').takeoverSession('S');session=await f.ledger.getSession('S');assert.deepEqual(session.computedRecovery.pendingBoundary,{from:0,to:600});
 const reviewed=await f.server.client('other').reviewComputedSession('S',{note:'checked current Actor state after failed preparation',userId:'G',at:600});assert.deepEqual(reviewed.computedRecovery.pendingBoundary,{from:0,to:600});
 await f.server.client('other').createSession(config('other',{startedAt:600,cursorAt:600}));
});

test('a pending clock boundary accepts only its exact batch and clears in the confirmation transaction',async()=>{
 const f=await fixture();await f.advance(600);
 for(const interval of [{from:1,to:600},{from:600,to:600},{from:600,to:1200}])await assert.rejects(claim(f,payload('wrong',interval)),/computed-boundary-conflict/);
 await claim(f,payload('B',{from:0,to:600}));assert.deepEqual((await f.ledger.getSession('S')).computedRecovery.pendingBoundary,{from:0,to:600});
 const before=f.server.raw.pages.length;await f.ledger.transitionComputedBatch('S','B',{expected:['claimed'],state:'confirmed',receipts:{actors:{actors:[]},items:{items:[],wounded:[]}},simulationState:{cursorAt:600}},f.scope);
 assert.equal(f.server.raw.pages.length,before+1);assert.equal((await f.ledger.getSession('S')).computedRecovery.pendingBoundary,undefined);
 await f.ledger.finishComputedSession('S',{status:'complete',summary:{},simulationState:{cursorAt:600}},f.scope);
});

test('an unsettled post-clock boundary cannot be advanced again',async()=>{
 const f=await fixture();await f.advance(600);
 await assert.rejects(f.advance(1200),/unresolved-computed-boundary/);
 const session=await f.ledger.getSession('S');assert.equal(session.cursorAt,600);assert.deepEqual(session.computedRecovery.pendingBoundary,{from:0,to:600});
});

test('pre-clock claimed samples do not need a second pending-boundary marker',async()=>{
 const f=await fixture();await claim(f,payload('B',{to:600}));await f.advance(600);
 assert.equal((await f.ledger.getSession('S')).computedRecovery.pendingBoundary,undefined);
 await f.ledger.transitionComputedBatch('S','B',{expected:['claimed'],state:'confirmed',receipts:{actors:{actors:[]},items:{items:[],wounded:[]}},simulationState:{cursorAt:600}},f.scope);
 assert.equal((await f.ledger.getSession('S')).computedRecovery.batches[0].state,'confirmed');
});

test('an effects-only boundary can confirm an empty event batch before the next interval',async()=>{
 const f=await fixture(),actors=[{actorUUID:'Actor.A',hp:{value:5}}];await f.advance(300);
 await claim(f,payload('effects',{from:0,to:300,beforeActors:actors,afterActors:actors,events:[]}));
 await f.ledger.transitionComputedBatch('S','effects',{expected:['claimed'],state:'confirmed',receipts:{actors:{actors:[]},items:{items:[],wounded:[]}},simulationState:{cursorAt:300}},f.scope);
 assert.equal((await f.ledger.getSession('S')).computedRecovery.pendingBoundary,undefined);
 await f.advance(600);assert.deepEqual((await f.ledger.getSession('S')).computedRecovery.pendingBoundary,{from:300,to:600});
});

for(const persisted of [false,true])test(`an unknown post-clock claim ACK blocks replay when the batch ${persisted?'was':'was not'} persisted`,async()=>{
 const f=await fixture();await f.advance(600);
 f.server.setAcknowledgement(()=>{if(!persisted)f.server.raw.pages.pop();return {}});
 await assert.rejects(claim(f,payload('B',{to:600})),/acknowledgement-unknown/);f.server.setAcknowledgement(ack=>ack);
 const saved=await f.ledger.getSession('S');assert.deepEqual(saved.computedRecovery.pendingBoundary,{from:0,to:600});assert.equal(saved.computedRecovery.batches.length,persisted?1:0);
 await f.ledger.updateSession('S',{status:'paused'},f.scope);await assert.rejects(f.ledger.resumeSession('S',{cursorAt:600}),/unresolved-computed/);
 await assert.rejects(f.server.client('other').createSession(config('other',{startedAt:600,cursorAt:600})),/unresolved-evidence-no-replay/);
});

test('stable calculation intent is acknowledged once and requires its authenticated running driver',async()=>{
 const f=await fixture();assert.equal(typeof f.ledger.beginComputedCalculation,'function');
 const before=f.server.raw.pages.length;assert.deepEqual(await f.ledger.beginComputedCalculation('S',f.scope),{from:0});assert.equal(f.server.raw.pages.length,before+1);
 assert.deepEqual((await f.ledger.getSession('S')).computedRecovery.pendingCalculation,{from:0});
 await assert.rejects(f.ledger.beginComputedCalculation('S',f.scope),/unresolved-computed/);
 await assert.rejects(f.server.client('other').beginComputedCalculation('S',f.scope),/session-driver-required/);
 await assert.rejects(f.ledger.finishComputedSession('S',{status:'complete',summary:{},simulationState:{cursorAt:0}},f.scope),/unresolved-computed/);
 await f.ledger.updateSession('S',{status:'paused'},f.scope);await assert.rejects(f.ledger.beginComputedCalculation('S',f.scope),/recovery-session-stopped/);
});

test('calculation intent rechecks the guard before its revision can authorize sampling',async()=>{
 const f=await fixture();assert.equal(typeof f.ledger.beginComputedCalculation,'function');f.onFinal(()=>f.invalidate());
 await assert.rejects(f.ledger.beginComputedCalculation('S',f.scope),/computed-source-changed/);
 assert.equal((await f.ledger.getSession('S')).computedRecovery.pendingCalculation,undefined);
});

test('another GM takeover preserves unsaved calculation intent and blocks resume or replacement',async()=>{
 const f=await fixture();assert.equal(typeof f.ledger.beginComputedCalculation,'function');await f.ledger.beginComputedCalculation('S',f.scope);
 const other=f.server.client('other');await other.takeoverSession('S');assert.deepEqual((await f.ledger.getSession('S')).computedRecovery.pendingCalculation,{from:0});
 await assert.rejects(other.resumeSession('S',{cursorAt:0}),/unresolved-computed/);await assert.rejects(other.createSession(config('next')),/unresolved-evidence-no-replay/);
 await other.reviewComputedSession('S',{note:'checked cards after interrupted calculation',at:0,userId:'G'});assert.deepEqual((await f.ledger.getSession('S')).computedRecovery.pendingCalculation,{from:0});
 await other.createSession(config('next'));
});

test('only a matching calculation batch can claim and clear the intent on confirmation',async()=>{
 const f=await fixture();assert.equal(typeof f.ledger.beginComputedCalculation,'function');await f.ledger.beginComputedCalculation('S',f.scope);
 await assert.rejects(claim(f,payload('wrong',{from:-1,to:0})),/computed-boundary-conflict/);
 await claim(f,payload('B',{to:600}));await assert.rejects(f.ledger.beginComputedCalculation('S',f.scope),/unresolved-computed/);
 await f.advance(600);assert.deepEqual((await f.ledger.getSession('S')).computedRecovery.pendingCalculation,{from:0});
 await f.ledger.transitionComputedBatch('S','B',{expected:['claimed'],state:'confirmed',receipts:{actors:{actors:[]},items:{items:[],wounded:[]}},simulationState:{cursorAt:600}},f.scope);
 assert.equal((await f.ledger.getSession('S')).computedRecovery.pendingCalculation,undefined);assert.deepEqual(await f.ledger.beginComputedCalculation('S',f.scope),{from:600});
});

for(const persisted of [false,true])test(`an unknown calculation intent ACK never grants sampling when it ${persisted?'was':'was not'} persisted`,async()=>{
 const f=await fixture();assert.equal(typeof f.ledger.beginComputedCalculation,'function');let samples=0;
 f.server.setAcknowledgement(()=>{if(!persisted)f.server.raw.pages.pop();return {}});
 await assert.rejects(f.ledger.beginComputedCalculation('S',f.scope).then(()=>samples++),/acknowledgement-unknown/);f.server.setAcknowledgement(ack=>ack);
 assert.equal(samples,0);assert.deepEqual((await f.ledger.getSession('S')).computedRecovery.pendingCalculation,persisted?{from:0}:undefined);
 if(persisted)await assert.rejects(f.ledger.beginComputedCalculation('S',f.scope),/unresolved-computed/);
});
