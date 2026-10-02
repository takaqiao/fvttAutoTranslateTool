import {test} from 'node:test';import assert from 'node:assert/strict';import {createManualRecordBridge} from '../../scripts/exploration/manual-record.mjs';
test('player generic records use authenticated caller, selected owned actor and fixed manual provenance',async()=>{
 const users=new Map([['G',{id:'G',isGM:true,active:true}],['P',{id:'P',active:true}]]);users.activeGM=users.get('G');const game={user:users.get('G'),users};let handler,saved;const socket={register:(n,f)=>handler=f};const actor={uuid:'Actor.A',testUserPermission:u=>u?.id==='P'};const bridge=createManualRecordBridge({game,fromUuid:async()=>actor,getSession:async()=>({status:'recording',actorUUIDs:['Actor.A']}),observe:async e=>{saved=e;return e}});bridge.register(socket);
 const result=await handler.call({socketdata:{userId:'P'}},{actorUUID:'Actor.A',label:'辨识魔法',durationSeconds:37,id:'fake',source:{type:'native'},receiptIds:['forged']});assert.equal(result.ok,true);assert.equal(saved.durationSeconds,37);assert.equal(saved.source.type,'user-record');assert.deepEqual(saved.receiptIds,undefined);assert.equal(saved.kind,'activity');assert.notEqual(saved.id,'fake');
 actor.testUserPermission=()=>false;assert.equal((await handler.call({socketdata:{userId:'P'}},{actorUUID:'Actor.A',label:'搜索',durationSeconds:600})).ok,false);
});

function windowFixture(change=()=>{}){
 const users=new Map([['G',{id:'G',isGM:true,active:true}],['P',{id:'P',active:true}],['X',{id:'X',active:true}]]);users.activeGM=users.get('G');
 const actor={id:'A',uuid:'Actor.A',name:'A',testUserPermission:user=>['P','G'].includes(user?.id)},game={user:users.get('G'),users,actors:new Map([['A',actor]])},binding={id:'C',sessionId:'S',rootUUID:'JournalEntry.ROOT',epoch:'E',observationNonce:'N',from:0},session={id:'S',status:'running',actorUUIDs:['Actor.A'],driver:{leaseNonce:'L'},budgetEndsAt:900,activityCheckpoint:{...binding,phase:'open'}},context={sessionId:'S',worldTime:0,leaseNonce:'L'},handlers=new Map();
 const f={game,actor,session,context,binding,handlers},bridge=createManualRecordBridge({game,fromUuid:async()=>actor,getSession:async()=>structuredClone(session),checkpointContext:()=>({...context}),getActivities:async()=>{change(f);return [{id:'own',actorUUID:'Actor.A',sessionId:'S',providerId:'refocus',state:'confirmed',proof:{receiptIds:['private']}},{id:'other',actorUUID:'Actor.B',sessionId:'S',providerId:'refocus',state:'confirmed',proof:{receiptIds:['other-private']}}]}});
 bridge.register({register:(name,fn)=>handlers.set(name,fn)});return {...f,bridge,query:(userId='P',uuid='Actor.A')=>handlers.get('exploration:activityCheckpoint').call({socketdata:{userId}},uuid)};
}
test('the authenticated window DTO contains only one owned actor and its dependency summaries',async()=>{
 const f=windowFixture(),response=await f.query();assert.equal(response.ok,true);assert.deepEqual(response.value,{binding:f.binding,phase:'open',actor:{actorUUID:'Actor.A',name:'A'},budgetEndsAt:900,dependencies:[{id:'own',label:'refocus',state:'confirmed'}]});assert.equal((await f.query('X')).ok,false);assert.equal((await f.query('P',{actorUUID:'Actor.A',authenticatedCaller:'G'})).ok,false);
});
for(const [name,change] of Object.entries({permission:f=>{f.actor.testUserPermission=()=>false},offline:f=>{f.game.users.get('P').active=false},canonical:f=>{f.game.actors.set('A',{...f.actor})},membership:f=>{f.session.actorUUIDs=[]},lease:f=>{f.context.leaseNonce='changed'},closed:f=>{f.session.activityCheckpoint.phase='sealed'},binding:f=>{f.session.activityCheckpoint.observationNonce='changed'}}))test(`window DTO rejects ${name} drift during its final read`,async()=>{
 const f=windowFixture(change);assert.equal((await f.query()).ok,false);
});
test('querying after scope loss cannot borrow the saved lease or create a window',async()=>{
 const f=windowFixture();delete f.context.leaseNonce;assert.equal((await f.query()).ok,false);assert.equal(f.session.activityCheckpoint.phase,'open');assert.deepEqual(f.session.driver,{leaseNonce:'L'});
});
