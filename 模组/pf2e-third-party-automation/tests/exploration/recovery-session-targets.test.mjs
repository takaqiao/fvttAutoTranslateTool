import {test} from 'node:test';
import assert from 'node:assert/strict';
import {authorityFixture} from './authority-fixture.mjs';
import {createLedger} from '../../scripts/exploration/ledger.mjs';
import {createCoordinator} from '../../scripts/exploration/coordinator.mjs';
import {validateSession} from '../../scripts/exploration/schema.mjs';

const P='Actor.P',MASTER='Actor.MASTER',OTHER='Actor.OTHER';
const preferences=(value=25)=>({version:1,targetIntentsByActor:{[P]:{mode:'percent',value}},requireNoWounded:false,failureStop:{enabled:false,limit:3}});
const deferred=()=>{let resolve;return {promise:new Promise(r=>resolve=r),resolve}};

async function fixture({shared=false,snapshotGate,beforeCommit,resolveHook}={}){
 const server=await authorityFixture(),G={id:'G',active:true,isGM:true},users=new Map([['G',G]]);users.activeGM=G;
 const documents=new Map([P,MASTER,OTHER].map(uuid=>[uuid,{uuid,id:uuid.slice(6),system:{attributes:{hp:{value:1,max:uuid===OTHER?80:41}}},testUserPermission:()=>true}]));
 const actors=new Map([...documents.values()].map(a=>[a.id,a])),pools=new Map([[P,shared?MASTER:P],[MASTER,MASTER],[OTHER,OTHER]]);
 const game={user:G,users,actors,time:{worldTime:0}},getHpPool=actor=>({poolUUID:pools.get(actor.uuid),ready:true});let snapshots=0;
 const capabilities={snapshot:async uuids=>{await snapshotGate?.(++snapshots);return uuids.map(actorUUID=>{const pool=getHpPool(documents.get(actorUUID)),hp=documents.get(pool.poolUUID).system.attributes.hp;return {actorUUID,pool,hp:structuredClone(hp),focus:{value:0,max:0},medicine:{rank:1},slugs:[],items:[],assuranceSkills:[],modeOfBeing:'living'}})}};
 const storage=server.storage('driver'),ledger=createLedger({...storage,transact:(fn,options)=>storage.transact((state,context)=>{const result=fn(state,context);beforeCommit?.({documents,pools,actors,state,result});return result},options),isAuthority:()=>true,identity:()=>({userId:'G',clientNonce:'driver'})});
 const c=createCoordinator({game,fromUuid:async uuid=>{await resolveHook?.({uuid,documents,pools,actors});return documents.get(uuid)},getHpPool,ledger,capabilities,providers:[],clock:{stop(){}},ownerOperations:{},isAuthority:()=>true,now:()=>0});
 const config={id:'S',actorUUIDs:[P],autoRun:false,recovery:preferences()};
 return {server,ledger,c,game,documents,pools,actors,config,start:input=>c.start(input??config)};
}

test('start saves detached patient intent, prepared basis and absolute pool goal',async()=>{
 const f=await fixture(),s=await f.start();
 assert.deepEqual(s.goalsByPool,[{poolUUID:P,targetHP:11}]);
 assert.deepEqual(s.recoveryGoals,{version:1,patientTargets:[{patientUUID:P,poolUUID:P,intent:{mode:'percent',value:25},basisMaxHP:41,targetHP:11}],requireNoWounded:false,failureStop:{enabled:false,limit:3}});
 f.config.recovery.targetIntentsByActor[P].value=90;s.recoveryGoals.patientTargets[0].intent.value=1;
 assert.equal((await f.ledger.getSession('S')).recoveryGoals.patientTargets[0].intent.value,25);
 assert.equal(Object.hasOwn(await f.ledger.getSession('S'),'recovery'),false);
});

test('start captures recovery input before its first snapshot await',async()=>{
 const entered=deferred(),release=deferred(),f=await fixture({snapshotGate:async count=>{if(count===1){entered.resolve();await release.promise}}});
 const pending=f.start();await entered.promise;f.config.recovery.targetIntentsByActor[P].value=80;release.resolve();
 assert.equal((await pending).goalsByPool[0].targetHP,11);
});

test('a fresh start resolves the current max after the old evidence barrier',async()=>{
 const f=await fixture(),original=f.ledger.all;let first=true;
 f.ledger.all=async()=>{const state=await original();if(first){first=false;f.documents.get(P).system.attributes.hp.max=81}return state};
 const s=await f.start();assert.equal(s.goalsByPool[0].targetHP,21);assert.equal(s.recoveryGoals.patientTargets[0].basisMaxHP,81);
});

for(const change of ['max','pool','patient-document','master-document'])test('final session commit rejects a changed '+change,async()=>{
 const f=await fixture({shared:change==='master-document',beforeCommit:({documents,pools,actors,result})=>{
  if(result?.id!=='S')return;
  if(change==='max')documents.get(P).system.attributes.hp.max=81;
  else if(change==='pool')pools.set(P,OTHER);
  else {const uuid=change==='master-document'?MASTER:P;actors.set(documents.get(uuid).id,{...documents.get(uuid)})}
 }});
 await assert.rejects(f.start(),/recovery.*changed|document/);
 assert.deepEqual((await f.ledger.all()).sessions,{});assert.equal(f.server.raw.pages.length,1);
});

test('a shared master change during document resolution cannot save an old target basis',async()=>{
 let changed=false;const f=await fixture({shared:true,resolveHook:({uuid,pools})=>{if(uuid===MASTER&&!changed){changed=true;pools.set(P,OTHER)}}});
 await assert.rejects(f.start(),/recovery.*changed|pool/);assert.deepEqual((await f.ledger.all()).sessions,{});
});

test('resume and GM takeover preserve saved intent and max even after preferences change',async()=>{
 const f=await fixture(),s=await f.start(),saved=structuredClone(s.recoveryGoals);await f.c.stop(s.id);
 f.documents.get(P).system.attributes.hp.max=81;f.config.recovery.targetIntentsByActor[P].value=90;
 await f.c.resume(s.id,{autoRun:false});assert.deepEqual((await f.ledger.getSession(s.id)).recoveryGoals,saved);
 const peer=f.server.client('peer');await peer.takeoverSession(s.id,{cursorAt:0});
 assert.deepEqual((await peer.getSession(s.id)).recoveryGoals,saved);assert.equal((await peer.getSession(s.id)).goalsByPool[0].targetHP,11);
});

test('generic patches cannot rewrite either saved target representation',async()=>{
 const f=await fixture(),s=await f.start();
 for(const patch of [{goalsByPool:[{poolUUID:P,targetHP:30}]},{recoveryGoals:{...s.recoveryGoals,patientTargets:[]}}])await assert.rejects(f.ledger.updateSession(s.id,patch,f.c.executionScope(s.id)),/immutable-session/);
 assert.equal((await f.ledger.getSession(s.id)).goalsByPool[0].targetHP,11);
});

test('legacy absolute goals stay absolute and missing recovery metadata stays missing',async()=>{
 const f=await fixture(),config={...f.config,goalsByPool:[{poolUUID:P,targetHP:12}]};delete config.recovery;
 const s=await f.start(config);assert.deepEqual(s.goalsByPool,config.goalsByPool);assert.equal(Object.hasOwn(s,'recoveryGoals'),false);
});

test('new goal intentions cannot bypass unresolved old recovery work',async()=>{
 const f=await fixture(),legacy={...f.config};delete legacy.recovery;const s=await f.start(legacy);await f.c.stop(s.id);
 const state=await f.server.read();state.activities.A={id:'A',sessionId:s.id,state:'uncertain',actorUUID:P,patientUUIDs:[P],hpPoolUUIDs:[P],source:{type:'coordinator'}};
 const storage=f.server.storage('seed');await storage.transact(current=>{current.activities.A=state.activities.A});
 await assert.rejects(f.start({...f.config,id:'NEW'}),/unresolved-evidence-no-replay/);
 assert.equal(Object.keys((await f.ledger.all()).sessions).length,1);
});

test('session shape rejects a forged intent resolution instead of preserving it',()=>{
 const input={id:'S',actorUUIDs:[P],startedAt:0,budgetEndsAt:600,goalsByPool:[{poolUUID:P,targetHP:11}],recoveryGoals:{version:1,patientTargets:[{patientUUID:P,poolUUID:P,intent:{mode:'percent',value:25},basisMaxHP:41,targetHP:40}],requireNoWounded:false,failureStop:{enabled:false,limit:3}}};
 assert.throws(()=>validateSession(input),/invalid-recovery-goals/);
});

test('createSession requires a synchronous true guard and rechecks it before the revision write',async()=>{
 const f=await authorityFixture(),ledger=f.client('guard'),input={id:'S',actorUUIDs:[P],startedAt:0,budgetEndsAt:600,goalsByPool:[{poolUUID:P,targetHP:10}]};
 for(const guard of [()=>false,async()=>true])await assert.rejects(ledger.createSession(input,{guard}),/guard|recovery.*changed/);
 assert.equal(f.raw.pages.length,1);assert.deepEqual((await ledger.all()).sessions,{});
});
