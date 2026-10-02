import {test} from 'node:test';
import assert from 'node:assert/strict';
import {authorityFixture} from './authority-fixture.mjs';
import {createManualPoolCompletion} from '../../scripts/exploration/manual-pool-completion.mjs';
import {createLedger} from '../../scripts/exploration/ledger.mjs';

const session={id:'S',manual:true,status:'recording',actorUUIDs:['Actor.H','Actor.P','Actor.M'],startedAt:0,budgetEndsAt:600};
const activity=(id='A',useId='U')=>({id,sessionId:'S',providerId:'manual',kind:'treatment',actorUUID:'Actor.H',patientUUIDs:['Actor.P'],hpPoolUUIDs:['Actor.M'],startedAt:0,endsAt:600,state:'awaiting-evidence',source:{manual:true,type:'native-action',useId},proof:{useId,checkIds:['C'],resultIds:['R']}});
async function base(sourceType='native-action'){const f=await authorityFixture(),ledger=f.client('issuer');await ledger.createSession(session);const a=activity();a.source.type=sourceType;await ledger.insertActivity(a);return {...f,ledger}}

test('generic activity insertion cannot mint a shared application grant',async()=>{
 const f=await base();
 await assert.rejects(f.ledger.insertActivity({...activity('B','U2'),proof:{...activity('B','U2').proof,poolApplications:{forged:{state:'granted',permitNonce:'forged'}}}}),/manual-pool/);
});

test('generic evidence transition cannot mint a shared application grant',async()=>{
 const f=await base(),a=await f.ledger.getActivity('A');
 await assert.rejects(f.ledger.transitionActivity('A',{expected:['awaiting-evidence'],patch:{proof:{...a.proof,poolApplications:{forged:{state:'granted',permitNonce:'forged'}}}}}),/manual-pool/);
});

const request=(patch={})=>({sessionId:'S',activityId:'A',actorUUID:'Actor.H',sourceType:'native-action',useId:'U',checkId:'C',resultId:'R',rollIndex:0,stage:'healing',poolUUID:'Actor.M',patientUUIDs:['Actor.P'],batchId:'batch',ownerClientNonce:'owner-1',attemptNonce:'attempt',...patch});
async function fixture(sourceType){
 const f=await base(sourceType),users=new Map(['G','O','OTHER'].map(id=>[id,{id,active:true,isGM:id==='G'}]));users.activeGM=users.get('G');
 const actors=new Map(['H','P','M'].map(id=>[`Actor.${id}`,{id,uuid:`Actor.${id}`,owners:new Set(id==='M'?['G']:['G','O']),testUserPermission(user,level){return level==='OWNER'&&this.owners.has(user.id)}}]));
 const messages=new Map(['C','R'].map(id=>[id,{id,author:users.get('O'),speaker:{actor:'H'},rolls:[{_evaluated:true,total:id==='C'?24:12}],toObject(){return {id:this.id,author:this.author.id,speaker:this.speaker,rolls:this.rolls}}}]));
 const clients=[],packets=[],errors=[];let pool='Actor.M',sourceCurrent=true,reads=0,beforeResolve=async()=>{},beforeCommit=()=>{},dropGrant=false;
 const make=(id,clientNonce,issuer=false)=>{
  const listeners=new Set(),game={user:users.get(id),users,messages,time:{worldTime:0},socket:{on:(_,fn)=>listeners.add(fn),off:(_,fn)=>listeners.delete(fn),emit(channel,packet,options,ack){
   packets.push({sender:id,clientNonce,packet:structuredClone(packet)});ack?.({relayed:true});
   if(dropGrant&&packet.kind==='grant')return;
   for(const c of clients.filter(c=>options.recipients.includes(c.game.user.id)))for(const receive of c.listeners)receive(structuredClone(packet),id);
  }}};
  const store=f.storage(clientNonce),ledger=id==='G'?createLedger({...store,transact:(fn,options)=>store.transact(fn,{...options,validateCommit:()=>{beforeCommit();return options?.validateCommit?.()}}),isAuthority:()=>true,identity:()=>({userId:id,clientNonce})}):undefined;
  const broker=createManualPoolCompletion({game,clientNonce,isIssuer:()=>issuer,ledger,
   hpPools:{discover:()=>({ready:true,poolUUID:pool,memberUUIDs:['Actor.M','Actor.P']})},fromUuid:async uuid=>{reads++;return actors.get(uuid)},
   resolveSource:async input=>{await beforeResolve();return {binding:{sessionId:input.sessionId,activityId:input.activityId,actorUUID:input.actorUUID,sourceType:input.sourceType,useId:input.useId,checkId:input.checkId,resultId:input.resultId,rollIndex:input.rollIndex,stage:input.stage,poolUUID:'Actor.M',patientUUIDs:['Actor.P'],batchId:input.batchId,effectId:input.resultId,selectedPatientUUID:'Actor.P',worldTime:0},isCurrent:()=>sourceCurrent}},
   timeoutMs:100,onError:error=>errors.push(error)});
  const value={game,listeners,broker};clients.push(value);broker.start();return value;
 };
 const gm=make('G','issuer',true),peer=make('G','peer',false),owner=make('O','owner-1'),owner2=make('O','owner-2');
 return {...f,gm,peer,owner,owner2,users,actors,messages,packets,errors,make,setPool:value=>{pool=value},setSource:value=>{sourceCurrent=value},setBeforeResolve:fn=>{beforeResolve=fn},setBeforeCommit:fn=>{beforeCommit=fn},dropGrants:()=>{dropGrant=true},readCount:()=>reads,close(){for(const c of clients)c.broker.stop()}};
}

test('two owner clients receive exactly one grant through the authenticated transport',async t=>{
 const f=await fixture();t.after(()=>f.close());
 const results=await Promise.allSettled([f.owner.broker.claim(request()),f.owner2.broker.claim(request({ownerClientNonce:'owner-2',attemptNonce:'second'}))]);
 assert.equal(results.filter(r=>r.status==='fulfilled').length,1);
 const grant=results.find(r=>r.status==='fulfilled').value;assert.ok(grant.permitNonce);assert.equal(grant.poolUUID,'Actor.M');
 assert.equal(Object.keys((await f.ledger.getActivity('A')).proof.poolApplications).length,1);
 assert.equal(f.packets.some(({packet})=>packet.proof?.ledger||packet.proof?.activity),false);
 assert.equal(f.packets.filter(p=>p.clientNonce==='peer').length,0);
});

test('a distinct verified use obtains a distinct claim without merging the effects',async t=>{
 const f=await fixture();t.after(()=>f.close());await f.ledger.insertActivity(activity('B','U2'));
 const first=await f.owner.broker.claim(request()),second=await f.owner.broker.claim(request({activityId:'B',useId:'U2',attemptNonce:'next'}));
 assert.notEqual(first.effectKey,second.effectKey);assert.notEqual(first.permitNonce,second.permitNonce);
});

test('unknown committed claim ACK never becomes a grant through lookup or another attempt',async t=>{
 const f=await fixture();t.after(()=>f.close());let executions=0;
 f.setAcknowledgement(()=>null);
 await assert.rejects(f.owner.broker.claim(request()).then(()=>executions++));
 f.setAcknowledgement(ack=>ack);
 const pages=f.raw.pages.length,proof=await f.owner.broker.lookup(request());
 assert.deepEqual(Object.keys(proof).sort(),['effectKey','poolUUID','sourceDigest','status']);assert.equal(proof.status,'reserved');assert.equal(proof.permitNonce,undefined);
 await assert.rejects(f.owner2.broker.claim(request({ownerClientNonce:'owner-2',attemptNonce:'retry'})));
 assert.equal(executions,0);assert.equal(f.raw.pages.length,pages);
});

for(const [label,change] of [
 ['pool relink',f=>f.setPool('Actor.OTHER')],['source changed',f=>f.setSource(false)],
 ['patient ownership',f=>f.actors.get('Actor.P').owners.delete('O')],['healer ownership',f=>f.actors.get('Actor.H').owners.delete('O')],
 ['inactive owner',f=>{f.users.get('O').active=false}],['deleted result',f=>f.messages.delete('R')],
 ['unevaluated check',f=>{f.messages.get('C').rolls[0]._evaluated=false}],['world time',f=>{f.gm.game.time.worldTime=1}]
])test(`source qualification rejects ${label}`,async t=>{
 const f=await fixture();t.after(()=>f.close());change(f);const pages=f.raw.pages.length;
 await assert.rejects(f.owner.broker.claim(request()));assert.equal(f.raw.pages.length,pages);
});

test('Stop while source qualification awaits prevents the atomic claim',async t=>{
 const f=await fixture();t.after(()=>f.close());f.setBeforeResolve(()=>f.ledger.updateSession('S',{status:'closed'}));
 await assert.rejects(f.owner.broker.claim(request()));assert.equal((await f.ledger.getActivity('A')).proof.poolApplications,undefined);
});

test('request input is detached before the first asynchronous resolver boundary',async t=>{
 const f=await fixture();t.after(()=>f.close());const input=request();
 const task=f.owner.broker.claim(input);input.patientUUIDs[0]='Actor.Evil';input.useId='forged';
 assert.equal((await task).poolUUID,'Actor.M');
});

test('client nonce mismatch and unexpected request fields fail without network or ledger reads',async t=>{
 const f=await fixture();t.after(()=>f.close());
 for(const value of [request({ownerClientNonce:'other-tab'}),request({proof:{permitNonce:'forged'}}),request({stage:'damage'}),request({sourceType:'user-record'})])await assert.rejects(f.owner.broker.claim(value));
 assert.equal(f.packets.length,0);assert.equal(f.readCount(),0);
});

test('lookup remains read only and does not expose another client claim',async t=>{
 const f=await fixture();t.after(()=>f.close());await f.owner.broker.claim(request());const pages=f.raw.pages.length;
 await assert.rejects(f.owner2.broker.lookup(request({ownerClientNonce:'owner-2',attemptNonce:'second'})));
 const result=await f.owner.broker.lookup(request());assert.equal(result.status,'reserved');assert.equal(f.raw.pages.length,pages);
});

for(const [label,change] of [
 ['pool',f=>f.setPool('Actor.OTHER')],['permission',f=>f.actors.get('Actor.P').owners.delete('O')],
 ['source',f=>f.setSource(false)],['result edit',f=>{f.messages.get('R').rolls[0].total=99}],
 ['result delete',f=>f.messages.delete('R')],['time',f=>{f.gm.game.time.worldTime=2}]
])test(`final synchronous guard rejects ${label} after revision digest awaits`,async t=>{
 const f=await fixture();t.after(()=>f.close());const pages=f.raw.pages.length;f.setBeforeCommit(()=>change(f));
 await assert.rejects(f.owner.broker.claim(request()));assert.equal(f.raw.pages.length,pages);
});

test('unknown manual claim retains its pool domain against automatic recovery',async t=>{
 const f=await fixture();t.after(()=>f.close());await f.owner.broker.claim(request());await f.ledger.updateSession('S',{status:'closed'});
 await assert.rejects(f.ledger.createSession({...session,id:'Auto',manual:false,status:'running',goalsByPool:[{poolUUID:'Actor.M',targetHP:30}]}),/unresolved-evidence-no-replay/);
});

test('generic proof replacement cannot remove a saved claim or turn it into a terminal',async t=>{
 const f=await fixture();t.after(()=>f.close());await f.owner.broker.claim(request());const a=await f.ledger.getActivity('A');
 const removed={...a.proof};delete removed.poolApplications;
 const forged=structuredClone(a.proof);Object.values(forged.poolApplications)[0].state='fulfilled';
 for(const proof of [removed,forged])await assert.rejects(f.ledger.transitionActivity('A',{expected:['awaiting-evidence'],patch:{proof}}),/manual-pool/);
});

test('a claimed source cannot be replaced or confirmed through generic transitions',async t=>{
 const f=await fixture();t.after(()=>f.close());await f.owner.broker.claim(request());const a=await f.ledger.getActivity('A');
 for(const patch of [{proof:{...a.proof,useId:'OTHER'}},{proof:{...a.proof,resultIds:[]}},{state:'confirmed'}])await assert.rejects(f.ledger.transitionActivity('A',{expected:['awaiting-evidence'],patch}),/manual-pool/);
});

test('remote shared source requires a currently permitted master writer',async t=>{
 const f=await fixture();t.after(()=>f.close());f.actors.get('Actor.M').owners.clear();
 await assert.rejects(f.owner.broker.claim(request()));
});

test('two independent GM ledger instances cannot claim the same saved source',async()=>{
 const f=await base(),input={request:request(),effectId:'R',selectedPatientUUID:'Actor.P',sourceDigest:'a'.repeat(64),ownerUserId:'O',permitNonce:'first',worldTime:0};
 const results=await Promise.allSettled([f.client('one').claimManualPoolApplication(input,{evidenceGuard:()=>true}),f.client('two').claimManualPoolApplication({...input,permitNonce:'second'},{evidenceGuard:()=>true})]);
 assert.equal(results.filter(x=>x.status==='fulfilled').length,1);assert.match(results.find(x=>x.status==='rejected').reason.message,/already-claimed/);
});

test('a caller-supplied owner id cannot override the authenticated socket sender',async t=>{
 const f=await fixture();t.after(()=>f.close());await f.owner.broker.claim(request());
 const saved=f.packets.find(x=>x.packet.kind==='claim').packet,forged={...saved,requestId:'forged'};
 const pages=f.raw.pages.length,packets=f.packets.length;for(const handler of f.gm.listeners)handler(forged,'OTHER');
 await new Promise(resolve=>setImmediate(resolve));assert.equal(f.raw.pages.length,pages);assert.equal(f.packets.length,packets);
});

test('a player can request a bounded lookup without a private ledger dependency',async t=>{
 const f=await fixture();t.after(()=>f.close());const other=f.make('OTHER','other');
 await assert.rejects(other.broker.lookup(request({ownerClientNonce:'other'})));
 assert.equal(f.packets.filter(x=>x.clientNonce==='other').length,1);assert.equal(f.raw.pages.length,3);
});

test('the same bounded claim accepts a privately verified Workbench source',async t=>{
 const f=await fixture('workbench');t.after(()=>f.close());
 assert.equal((await f.owner.broker.claim(request({sourceType:'workbench'}))).state,'granted');
});

test('risky activities cannot enter the ordinary shared claim',async t=>{
 const f=await fixture();t.after(()=>f.close());await f.ledger.transitionActivity('A',{expected:['awaiting-evidence'],patch:{options:{riskySurgery:true}}});
 await assert.rejects(f.owner.broker.claim(request()));assert.equal((await f.ledger.getActivity('A')).proof.poolApplications,undefined);
});

test('automatic checkpoint reservations cannot enter the ordinary shared claim',async t=>{
 const f=await fixture('workbench');t.after(()=>f.close());
 const a=activity('Checkpoint');a.source.type='workbench';a.temporalSource={type:'checkpoint-reservation'};await f.ledger.insertActivity(a);
 await assert.rejects(f.owner.broker.claim(request({activityId:a.id,sourceType:'workbench'})));
 assert.equal((await f.ledger.getActivity(a.id)).proof.poolApplications,undefined);
});

test('a lost transport reply permits only explicit read-only lookup, never a resend',async t=>{
 const f=await fixture();t.after(()=>f.close());f.dropGrants();let executions=0;
 await assert.rejects(f.owner.broker.claim(request()).then(()=>executions++),/timeout.*unknown/);
 const pages=f.raw.pages.length;
 await assert.rejects(f.owner.broker.claim(request()),/unknown-no-retry/);
 const proof=await f.owner.broker.lookup(request());assert.equal(proof.status,'reserved');assert.equal(proof.permitNonce,undefined);
 assert.equal(f.packets.filter(x=>x.packet.kind==='claim').length,1);assert.equal(f.raw.pages.length,pages);assert.equal(executions,0);
});

test('an authority change after storage ACK returns no executable grant',async t=>{
 const f=await fixture();t.after(()=>f.close());f.setAcknowledgement(ack=>{f.users.get('G').active=false;return ack});let executions=0;
 await assert.rejects(f.owner.broker.claim(request()).then(()=>executions++));
 assert.equal(executions,0);assert.equal(f.packets.filter(x=>x.packet.kind==='grant').length,0);
 assert.equal(Object.values((await f.ledger.getActivity('A')).proof.poolApplications).length,1);
});

test('claims require atomic storage and a synchronous final evidence guard',async()=>{
 const f=await base(),input={request:request(),effectId:'R',selectedPatientUUID:'Actor.P',sourceDigest:'a'.repeat(64),ownerUserId:'O',permitNonce:'P',worldTime:0};
 const legacy=createLedger({read:f.read,write:()=>{throw Error('must not write')},isAuthority:()=>true});
 assert.throws(()=>legacy.claimManualPoolApplication(input,{evidenceGuard:()=>true}),/atomic-manual-pool/);
 assert.throws(()=>f.ledger.claimManualPoolApplication(input),/evidence-guard/);
 await assert.rejects(f.ledger.claimManualPoolApplication(input,{evidenceGuard:async()=>true}),/synchronous/);
 assert.equal((await f.ledger.getActivity('A')).proof.poolApplications,undefined);
});

test('actor flags and edited message data cannot replace the private source observer',async t=>{
 const f=await fixture();t.after(()=>f.close());f.messages.get('C').flags={claimedSource:true};f.setSource(false);
 await assert.rejects(f.owner.broker.claim(request()));assert.equal((await f.ledger.getActivity('A')).proof.poolApplications,undefined);
});
