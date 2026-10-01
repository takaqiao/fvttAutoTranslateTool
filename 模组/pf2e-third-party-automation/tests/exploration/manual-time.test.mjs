import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createManualRecordBridge} from '../../scripts/exploration/manual-record.mjs';
import {createManualEvents} from '../../scripts/exploration/manual-events.mjs';
import {createLedger} from '../../scripts/exploration/ledger.mjs';
import {reconstructEarliest} from '../../scripts/exploration/timeline.mjs';
import {normalizeManualActivity} from '../../scripts/exploration/manual-time.mjs';

async function fixture(){
 let state={sessions:{},activities:{},clocks:{}};
 const ledger=createLedger({read:async()=>structuredClone(state),write:async next=>{state=structuredClone(next)},isAuthority:()=>true});
 await ledger.createSession({id:'S',manual:true,status:'recording',actorUUIDs:['Actor.A','Actor.B'],startedAt:-1000,budgetEndsAt:10000,assumptions:['different-actors-may-overlap']});
 const users=new Map([['G',{id:'G',active:true,isGM:true}],['P',{id:'P',active:true,isGM:false}]]);users.activeGM=users.get('G');
 const game={user:users.get('G'),users,time:{worldTime:-1000}};
 const actors=new Map(['Actor.A','Actor.B'].map(uuid=>[uuid,{uuid,testUserPermission:u=>u?.active&&['P','G'].includes(u.id)}]));
 const changes=[];const resolve=async uuid=>actors.get(uuid);
 const recorder=createManualEvents({game,ledger,fromUuid:resolve,sessionId:()=> 'S',onChange:e=>changes.push(e)});
 const bridge=createManualRecordBridge({game,fromUuid:resolve,getSession:()=>ledger.getSession('S'),observe:recorder.observe});
 let handler;bridge.register({register:(_n,fn)=>handler=fn});
 const record=event=>handler.call({socketdata:{userId:'P'}},event);
 return {game,users,actors,ledger,recorder,bridge,record,changes,resolve,snapshot:()=>ledger.snapshot('S')};
}
const declaration=patch=>({actorUUID:'Actor.A',label:' 搜索 ',durationSeconds:37,...patch});

test('public declaration persists timing, authenticated provenance and actual saved dependency into timeline',async()=>{
 const f=await fixture();const first=await f.record(declaration({actorUUID:'Actor.B',durationSeconds:10}));assert.equal(first.ok,true);
 const response=await f.record(declaration({durationSource:{type:'item-text',detail:' 物品条目所述37秒 '},order:2,dependsOn:[first.value.id,first.value.id],notBefore:-980,observedStart:-980,observedEnd:-943,temporalSource:{type:'native',userId:'forged',recordedAt:0}}));
 assert.equal(response.ok,true);const a=(await f.snapshot()).activities[1];
 assert.deepEqual(a.durationSource,{type:'item-text',detail:'物品条目所述37秒'});assert.equal(a.order,2);assert.deepEqual(a.dependsOn,[first.value.id]);
 assert.equal(a.notBefore,-980);assert.equal(a.observedStart,-980);assert.equal(a.observedEnd,-943);
 assert.deepEqual(a.temporalSource,{type:'user-declared',userId:'P',recordedAt:-1000});
 const timeline=reconstructEarliest({startedAt:-1000,activities:(await f.snapshot()).activities,assumptions:['different-actors-may-overlap']});
 assert.equal(timeline.endsAt,-943);assert.notEqual(timeline.certainty,'observed');
 const row=timeline.scheduled.find(row=>row.activityIds.includes(a.id));assert.deepEqual(row.temporalSource,a.temporalSource);assert.deepEqual(row.durationSource,a.durationSource);
});

test('generic zero seconds can be recorded without implying native completion',async()=>{
 const f=await fixture();const response=await f.record(declaration({durationSeconds:0,observedStart:-1000,observedEnd:-1000}));assert.equal(response.ok,true);
 const [a]=(await f.snapshot()).activities;assert.equal(a.durationSeconds,0);assert.equal(a.startedAt,a.endsAt);assert.equal(a.state,'awaiting-evidence');assert.equal(a.source.unverified,true);
 assert.deepEqual(a.proof,{useId:null,checkIds:[],resultIds:[],receiptIds:[],immunityIds:[]});
});

test('public record freezes nested declarations before actor lookup awaits',async()=>{
 const f=await fixture();let release;const gate=new Promise(r=>release=r);let entered;
 const began=new Promise(r=>entered=r);const bridge=createManualRecordBridge({game:f.game,fromUuid:async uuid=>{entered();await gate;return f.resolve(uuid)},getSession:()=>f.ledger.getSession('S'),observe:f.recorder.observe});
 const input=declaration({durationSource:{type:'table-convention',detail:'原说明'},observedStart:-1000,observedEnd:-963,dependsOn:[]});
 const pending=bridge.record(input);await began;input.durationSeconds=999;input.durationSource.detail='变更';input.dependsOn.push('forged');input.observedEnd=999;release();await pending;
 const [a]=(await f.snapshot()).activities;assert.equal(a.durationSeconds,37);assert.equal(a.durationSource.detail,'原说明');assert.deepEqual(a.dependsOn,[]);assert.equal(a.observedEnd,-963);
});

test('recorder freezes a generic event before the enrollment queue runs',async()=>{
 const f=await fixture();const input={...declaration(),id:'generic',kind:'activity',source:{type:'user-record',userId:'P',unverified:true},durationSource:{type:'user-declared',detail:'原说明'}};
 const pending=f.recorder.observe(input);input.label='变更';input.durationSeconds=900;input.durationSource.detail='变更';await pending;
 const [a]=(await f.snapshot()).activities;assert.equal(a.options.label,'搜索');assert.equal(a.durationSeconds,37);assert.equal(a.durationSource.detail,'原说明');
});

test('player socket projection preserves allowed times but cannot carry treatment proof',async()=>{
 const f=await fixture();f.game.user=f.users.get('P');let transported;
 f.bridge.register({register(){},executeAsGM:async(_name,event)=>{transported=event;return {ok:true,value:'ok'}}});
 await f.bridge.record(declaration({durationSource:{type:'table-convention',detail:'本桌约定'},order:0,dependsOn:['saved'],notBefore:-1000,observedStart:-1000,observedEnd:-963,patientUUIDs:['Actor.B'],receiptIds:['fake'],source:{type:'native'},parallelWith:['fake']}));
 assert.equal(transported.observedStart,-1000);assert.deepEqual(transported.dependsOn,['saved']);assert.equal(transported.durationSource.type,'table-convention');
 for(const key of ['patientUUIDs','receiptIds','source','parallelWith'])assert.equal(transported[key],undefined);
});

test('delayed public dialog only accepts the currently selected recording session',async()=>{
 const f=await fixture();const response=await f.record(declaration({sessionId:'Previous'}));assert.equal(response.ok,false);assert.equal((await f.snapshot()).activities.length,0);
});

test('recorder rejects session switch during its own awaited enrollment',async()=>{
 const f=await fixture();let sid='S';const base=f.ledger.getSession;
 const ledger={...f.ledger,getSession:async id=>{const value=await base(id);sid='Other';return value}};
 const recorder=createManualEvents({game:f.game,ledger,fromUuid:f.resolve,sessionId:()=>sid});
 await assert.rejects(recorder.observe({...declaration(),id:'stale',kind:'activity',expectedSessionId:'S',source:{type:'user-record',userId:'P',unverified:true}}),/session/);
 assert.equal((await f.snapshot()).activities.length,0);
});

test('recorder final boundary rejects permission loss during its awaited source validation',async()=>{
 const f=await fixture();const recorder=createManualEvents({game:f.game,ledger:f.ledger,fromUuid:async uuid=>{const actor=await f.resolve(uuid);actor.testUserPermission=()=>false;return actor},sessionId:()=> 'S'});
 await assert.rejects(recorder.observe({...declaration(),id:'revoked',kind:'activity',source:{type:'user-record',userId:'P',unverified:true}}),/actor/);
 assert.equal((await f.snapshot()).activities.length,0);
});

test('closing recording before queued enrollment reports failure instead of a null success',async()=>{
 const f=await fixture();const base=f.ledger.getSession;let reads=0;
 f.ledger.getSession=async id=>{const session=await base(id);return ++reads===1?session:{...session,status:'closed'}};
 const response=await f.record(declaration());assert.equal(response.ok,false);assert.match(response.error,/session/);assert.equal((await f.snapshot()).activities.length,0);
});

test('explicit generic order does not replace the native observation counter after hydration',async()=>{
 const f=await fixture();assert.equal((await f.record(declaration({order:Number.MAX_SAFE_INTEGER}))).ok,true);
 const next=createManualEvents({game:f.game,ledger:f.ledger,fromUuid:f.resolve,sessionId:()=> 'S'});
 await next.observe({id:'native-check',actorUUID:'Actor.A',kind:'medicine-check',durationSeconds:6,source:{type:'native-action'}});
 const a=await f.ledger.getActivity('manual:native-check');assert.equal(a.order,1);assert.equal(a.source.type,'native-action');assert.equal(a.temporalSource,undefined);
});

test('hydrating an old generic record leaves its persisted source and proof unchanged',async()=>{
 const f=await fixture();const old={id:'manual:old',sessionId:'S',providerId:'manual',actorUUID:'Actor.A',patientUUIDs:[],hpPoolUUIDs:[],state:'awaiting-evidence',startedAt:-1000,endsAt:-963,kind:'activity',order:0,durationSeconds:37,source:{manual:true,type:'user-record',userId:'P',messageId:'old',unverified:true},proof:{resultIds:['old']},options:{label:'旧记录',missing:['manual-source-requires-review']}};
 await f.ledger.insertActivity(old);const before=await f.ledger.getActivity(old.id);await f.record(declaration());assert.deepEqual(await f.ledger.getActivity(old.id),before);
});

for(const patch of [{durationSeconds:-1},{durationSeconds:NaN},{durationSeconds:Infinity},{order:-1},{order:0.5},{order:Number.MAX_SAFE_INTEGER+1},{notBefore:Infinity},{observedStart:-20},{observedEnd:-10},{observedStart:-10,observedEnd:-20},{observedStart:NaN,observedEnd:0},{durationSource:{type:'native'}},{durationSource:{type:'item-text',detail:23}},{dependsOn:['']},{dependsOn:'id'}])test('invalid declaration rejected before persistence '+JSON.stringify(patch),async()=>{
 const f=await fixture();const response=await f.record(declaration(patch));assert.equal(response.ok,false);assert.equal((await f.snapshot()).activities.length,0);
});

test('unknown and cross-session dependencies cannot silently disappear',async()=>{
 const f=await fixture();await f.ledger.createSession({id:'Other',manual:true,status:'recording',actorUUIDs:['Actor.A'],startedAt:0,budgetEndsAt:100});
 await f.ledger.insertActivity({id:'foreign',sessionId:'Other',providerId:'manual',actorUUID:'Actor.A',patientUUIDs:[],hpPoolUUIDs:[],state:'awaiting-evidence',startedAt:0,endsAt:10,source:{manual:true},kind:'activity',durationSeconds:10});
 for(const dependency of ['unknown','foreign']){const response=await f.record(declaration({dependsOn:[dependency]}));assert.equal(response.ok,false);assert.match(response.error,/dependency/)}
 assert.equal((await f.snapshot()).activities.length,0);
});

test('sparse dependency arrays are invalid rather than creating an undefined reference',async()=>{
 assert.throws(()=>normalizeManualActivity(declaration({dependsOn:new Array(1)})),/invalid-manual-dependency/);
 const f=await fixture();const response=await f.record(declaration({dependsOn:new Array(1)}));assert.equal(response.ok,false);assert.match(response.error,/invalid-manual-dependency/);assert.equal((await f.snapshot()).activities.length,0);
});

test('generic evidence never acquires patient, native proof, grouping or receipt completion',async()=>{
 const f=await fixture();const input={...declaration(),id:'generic',kind:'activity',source:{type:'user-record',userId:'P',unverified:true},patientUUIDs:['Actor.B'],hpPoolUUIDs:['Actor.B'],useId:'fake',resultIds:['fake'],receiptIds:['fake'],groupProof:'fake',treatmentImmunitySeconds:0,missing:[]};
 await f.recorder.observe(input);await f.recorder.observe({...input,kind:'treatment',durationSeconds:0,source:{type:'native-action'},effectiveOutcome:'success'});
 const [a]=(await f.snapshot()).activities;assert.equal(a.kind,'activity');assert.equal(a.state,'awaiting-evidence');assert.deepEqual(a.patientUUIDs,[]);assert.deepEqual(a.hpPoolUUIDs,[]);assert.equal(a.groupProof,undefined);assert.equal(a.treatmentImmunitySeconds,undefined);assert.deepEqual(a.proof.resultIds,[]);assert.ok(a.options.missing.includes('manual-source-requires-review'));
});

test('duration shorter than a declared interval is allowed but interval shorter than duration conflicts',async()=>{
 const f=await fixture();const response=await f.record(declaration({durationSeconds:37,observedStart:-1000,observedEnd:-990}));assert.equal(response.ok,true);
 const timeline=reconstructEarliest({startedAt:-1000,activities:(await f.snapshot()).activities});assert.equal(timeline.certainty,'contradictory');assert.ok(timeline.missing.some(m=>m.reason==='observed-before-ready'));
});

const solve=activities=>reconstructEarliest({startedAt:-1000,activities,assumptions:['different-actors-may-overlap']});
const generic=patch=>({id:'a',actorUUID:'Actor.A',kind:'activity',source:{type:'user-record'},order:0,durationSeconds:37,...patch});
for(const patch of [{durationSeconds:-1},{durationSeconds:Infinity},{notBefore:NaN},{observedStart:-1000},{observedStart:-1000,observedEnd:Infinity},{order:-1}])test('malformed saved timing cannot produce successful reconstruction '+JSON.stringify(patch),()=>{
 const result=solve([generic(patch)]);assert.equal(result.certainty,'incomplete');assert.ok(result.missing.length);assert.ok(Number.isFinite(result.endsAt));
});
test('self dependency remains a cycle instead of disappearing',()=>{const result=solve([generic({dependsOn:['a']})]);assert.equal(result.certainty,'incomplete');assert.ok(result.missing.some(m=>m.reason==='dependency-cycle'));assert.deepEqual(result.scheduled,[])});
test('user-declared complete interval is never labeled observed',()=>{const result=solve([generic({observedStart:-1000,observedEnd:-963,temporalSource:{type:'user-declared',userId:'P',recordedAt:-900}})]);assert.notEqual(result.certainty,'observed');assert.equal(result.endsAt,-963);assert.equal(result.scheduled[0].temporalSource.type,'user-declared')});
test('zero native treatment duration is incomplete while genuine legacy observed order stays unchanged',()=>{
 const zero=solve([{id:'t',actorUUID:'Actor.A',order:0,kind:'treatment',durationSeconds:0}]);assert.equal(zero.certainty,'incomplete');
 const native=solve([{id:'t',actorUUID:'Actor.A',order:0,kind:'treatment',durationSeconds:600,observedStart:-1000,observedEnd:-400}]);assert.equal(native.certainty,'observed');assert.equal(native.endsAt,-400);
});
test('finite inputs that overflow reconstructed world seconds remain incomplete',()=>{
 const result=reconstructEarliest({startedAt:Number.MAX_VALUE,activities:[generic({durationSeconds:Number.MAX_VALUE})]});assert.equal(result.certainty,'incomplete');assert.ok(Number.isFinite(result.endsAt));assert.deepEqual(result.scheduled,[]);
});
