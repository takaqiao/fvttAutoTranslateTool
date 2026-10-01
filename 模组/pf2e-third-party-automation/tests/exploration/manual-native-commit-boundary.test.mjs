import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {createLedger} from '../../scripts/exploration/ledger.mjs';
import {createRevisionStore} from '../../scripts/exploration/revision-store.mjs';
import {canonicalJSON,normalizeLedger,decodeRevision} from '../../scripts/exploration/revision-codec.mjs';
import {createManualEvents} from '../../scripts/exploration/manual-events.mjs';
import {createPatreonManualImmunity} from '../../scripts/exploration/patreon-manual-immunity.mjs';
import {manualEvidenceFixture,flush,M} from './manual-evidence-fixture.mjs';

const ROOT='JournalEntry.ROOT000000000001';
const descriptor={version:1,providerId:'patreon-v3',providerVersion:'3.2.29',
 baseSourceSHA256:'89ded325b92fa6b03dcf9257337ae2d628b3e99c987fe22cef9fff4e1837f4e9',
 pf2eSourceSHA256:'d63da8312831b84905e6866b1dd3f9d93e95c1012955b0177ad2ce8ccf246157'};
function deferred(){let resolve;const promise=new Promise(done=>{resolve=done});return {promise,resolve}}
const gate=()=>({entered:deferred(),release:deferred()});

async function revisionFixture(){
 const seed={sessions:{S:{id:'S',manual:true,status:'recording',startedAt:100,actorUUIDs:['Actor.H','Actor.P'],activityIds:[]}},activities:{},clocks:{}};
 const raw={_id:'ROOT000000000001',ownership:{default:0,G:3},pages:[],flags:{[M]:{explorationLedger:seed}}};
 let sequence=0,authority=true;
 function client(name){
  let nextRead,heldRead,lastTransaction=Promise.resolve();
  const store=createRevisionStore({getRootUUID:()=>ROOT,isAuthority:()=>authority,canUseRoot:()=>true,
   writerUserId:()=> 'G',writerClientId:()=>name,nonce:()=>`${name}-${++sequence}`,
   readRoot:async uuid=>{
    assert.equal(uuid,ROOT);const snapshot=structuredClone(raw),hold=heldRead;heldRead=null;
    if(hold){hold.entered.resolve();await hold.release.promise}return snapshot;
   },
   createPage:async({rootUUID,page})=>{
    assert.equal(rootUUID,ROOT);const envelope={type:'JournalEntryPage',action:'create',broadcast:false,operation:{parentUuid:ROOT},userId:'G'};
    if(raw.pages.some(saved=>saved._id===page._id))return {...envelope,error:{class:'ServerError',message:`The _id [${page._id}] already exists within the parent collection: JournalEntry [${raw._id}] pages`}};
    raw.pages.push(structuredClone(page));return {...envelope,result:[structuredClone(page)]};
   }});
  const ledger=createLedger({read:store.read,transact:(fn,options)=>{
   if(nextRead){heldRead=nextRead;nextRead=null}const transaction=store.transact(fn,options);
   lastTransaction=transaction.then(()=>{},()=>{});return transaction;
  },isAuthority:()=>authority,identity:()=>({userId:'G',clientNonce:name})});
  return {store,ledger,settled:()=>lastTransaction,holdTransactionRead:()=>{const hold=gate();nextRead=hold;return hold}};
 }
 const recorder=client('recorder'),stopper=client('stopper');
 await recorder.store.initialize({epoch:'native-commit-boundary',expectedSourceDigest:createHash('sha256').update(canonicalJSON(normalizeLedger(seed))).digest('hex')});
 return {raw,recorder,stopper,revokeAuthority:()=>{authority=false}};
}

async function waitForActivity(f,predicate){
 const until=Date.now()+5000;let activity;
 do{activity=await f.activity();if(predicate(activity))return activity;await flush()}while(Date.now()<until);
 assert.fail(`ledger did not settle: ${JSON.stringify({activity,errors:f.errors})}`);
}

async function nativeFixture(){
 const f=manualEvidenceFixture(),server=await revisionFixture();
 f.messages.clear();f.game.time.worldTime=100;f.game.system={version:'8.5.1'};
 f.healer.type='character';f.patient.type='character';
 const binding={invocationId:'INV',messageId:'C',useId:'U',tag:'exploration-manual:U',actorUUID:'Actor.H',patientUUID:'Actor.P',sourceUserId:'HUSER',startedAt:100,recordingSessionId:'S'};
 const metadata={tag:binding.tag,useId:'U',patientUUID:'Actor.P',startedAt:100,recordingSessionId:'S',riskySurgery:false,continualRecovery:false,patreonImmunity:{...binding}};
 const observers=new Set(),provider={descriptor,subscribe:fn=>{observers.add(fn);return ()=>observers.delete(fn)}};
 f.game.modules=new Map([['patreon-v3',{active:true,version:'3.2.29',api:{explorationManualImmunity:provider}}]]);
 f.check={id:'C',isCheckRoll:true,isReroll:false,author:f.users.get('HUSER'),speaker:{actor:'H'},rolls:[{_evaluated:true,formula:'1d20 + 9',total:19}],
  flags:{[M]:{explorationManualNative:metadata},pf2e:{modifiers:[],context:{type:'skill-check',origin:{actor:'Actor.H'},target:{actor:'Actor.P'},options:['action:treat-wounds',binding.tag],outcome:'success'}}}};
 f.child={id:'D',isCheckRoll:false,author:f.users.get('HUSER'),speaker:{actor:'H'},rolls:[{_evaluated:true,formula:'{2d8[healing]}'}],
  flags:{[M]:{explorationManualNative:{...metadata}},pf2e:{origin:{messageId:'C'},context:{origin:{actor:'Actor.H'},target:{actor:'Actor.P'},options:[binding.tag]}}},
  async update(changes){this.flags.pf2e.context.options=changes['flags.pf2e.context.options'];return this}};
 f.hpReceipt={id:'R',author:f.users.get('PUSER'),speaker:{actor:'P'},flags:{pf2e:{context:{type:'damage-taken',options:[`${M}:source:D:0`]},appliedDamage:{uuid:'Actor.P',isHealing:true,isReverted:false}}}};
 f.item={id:'I',uuid:'Actor.P.Item.I',actor:f.patient,parent:f.patient,type:'effect',sourceId:'Compendium.pf2e.feat-effects.Item.Lb4q2bBAgxamtix5',
  system:{start:{value:100},duration:{value:60,unit:'minutes',expiry:'turn-start',sustained:false},context:{origin:{actor:'Actor.H'}}},
  flags:{[M]:{explorationManualPatreonImmunity:{...binding,creatorId:'G'}}}};
 f.terminal={descriptor,binding,creatorId:'G',itemUUID:f.item.uuid,start:100,duration:{...f.item.system.duration},expiresAt:3700};
 f.ledger=server.recorder.ledger;
 f.options={...f.options,ledger:f.ledger,hpPools:{discover:actor=>({poolUUID:actor.uuid,ready:true})},
  fromUuid:async uuid=>f.actors.get(uuid)??[...f.patient.items.values()].find(item=>item.uuid===uuid)};
 f.options.patreonImmunity=createPatreonManualImmunity({game:f.game,fromUuid:f.options.fromUuid});
 f.recorder=createManualEvents(f.options);f.recorder.start();
 f.activity=()=>f.ledger.getActivity('manual:C');
 await f.fire(f.check);await waitForActivity(f,a=>a?.state==='awaiting-evidence');
 await f.fire(f.child);await waitForActivity(f,a=>a?.proof.resultIds.includes('D'));
 await f.fire(f.hpReceipt);await waitForActivity(f,a=>a?.proof.receiptIds.includes('R'));f.patient.items.set('I',f.item);
 f.finish=()=>{for(const observer of observers)observer({descriptor,binding,terminalPromise:Promise.resolve(f.terminal)})};
 return Object.assign(f,{server});
}

function holdConfirmedDigest(t){
 const hold=gate(),original=globalThis.crypto.subtle.digest;let armed=true;
 globalThis.crypto.subtle.digest=async function(...args){
  const result=await original.apply(this,args);let body;
  try{body=JSON.parse(new TextDecoder().decode(args[1]))}catch{}
  if(armed&&body?.changes?.activities?.['manual:C']?.state==='confirmed'){
   armed=false;hold.entered.resolve();await hold.release.promise;
  }
  return result;
 };
 t.after(()=>{hold.release.resolve();globalThis.crypto.subtle.digest=original});return hold;
}

test('ordinary native completion commits through the real revision store',async t=>{
 const f=await nativeFixture();t.after(()=>f.recorder.stop());f.finish();
 const activity=await waitForActivity(f,a=>a?.state==='confirmed');assert.deepEqual(activity.options.missing,[]);
 assert.deepEqual(activity.proof.receiptIds,['R']);assert.deepEqual(activity.proof.immunityIds,['Actor.P.Item.I']);
 assert.equal(f.server.raw.pages.map(page=>decodeRevision(page.flags[M].explorationRevision)).at(-1).changes.activities['manual:C'].state,'confirmed');
 assert.deepEqual(f.errors,[]);
});

test('Stop committed during the transaction read prevents native confirmation after a real revision conflict',async t=>{
 const f=await nativeFixture();t.after(()=>f.recorder.stop());const hold=f.server.recorder.holdTransactionRead();
 t.after(()=>hold.release.resolve());f.finish();await hold.entered.promise;
 await f.server.stopper.ledger.updateSession('S',{status:'stopped'});hold.release.resolve();await f.server.recorder.settled();
 assert.equal((await f.ledger.getSession('S')).status,'stopped');assert.equal((await f.activity()).state,'awaiting-evidence');
});

test('Stop committed after the confirmation mutation while its successor digest is pending prevents confirmation',async t=>{
 const f=await nativeFixture();t.after(()=>f.recorder.stop());const hold=holdConfirmedDigest(t);f.finish();await hold.entered.promise;
 await f.server.stopper.ledger.updateSession('S',{status:'stopped'});hold.release.resolve();await f.server.recorder.settled();
 assert.equal((await f.ledger.getSession('S')).status,'stopped');assert.equal((await f.activity()).state,'awaiting-evidence');
});

for(const [name,invalidate] of [
 ['the patient HP pool changes',f=>{f.options.hpPools.discover=()=>({poolUUID:'Actor.SHARED',ready:true})}],
 ['the HP receipt author loses patient OWNER',f=>{f.patient.testUserPermission=user=>user?.isGM===true}],
 ['the persisted native check useId changes',f=>{f.check.flags[M].explorationManualNative.useId='OTHER'}],
])test(`native confirmation cannot submit when ${name} during successor hashing`,async t=>{
 const f=await nativeFixture();t.after(()=>f.recorder.stop());const before=f.server.raw.pages.length,hold=holdConfirmedDigest(t);
 f.finish();await hold.entered.promise;invalidate(f);hold.release.resolve();await f.server.recorder.settled();
 assert.equal((await f.activity()).state,'awaiting-evidence');assert.equal(f.server.raw.pages.length,before);
});

test('a second saved HP receipt for the same native child blocks commit without a create hook',async t=>{
 const f=await nativeFixture();t.after(()=>f.recorder.stop());const before=f.server.raw.pages.length,hold=holdConfirmedDigest(t);
 assert.deepEqual((await f.activity()).proof.receiptIds,['R']);f.finish();await hold.entered.promise;
 f.messages.set('R2',{...f.hpReceipt,id:'R2'});hold.release.resolve();await f.server.recorder.settled();
 const activity=await f.activity();assert.equal(activity.state,'awaiting-evidence');
 assert.deepEqual(activity.proof.receiptIds,['R']);assert.equal(f.server.raw.pages.length,before);
});

test('a private synchronous evidence guard runs at mutation and after hashing without entering revision JSON',async t=>{
 const f=await nativeFixture();t.after(()=>f.recorder.stop());let calls=0;
 const activity=await f.ledger.transitionActivity('manual:C',{expected:['awaiting-evidence'],patch:{review:{note:'guard accepted'}},
  expectedSessionStatus:'recording',evidenceGuard:()=>{calls++;return true}});
 assert.equal(activity.review.note,'guard accepted');assert.ok(calls>=2,`guard calls: ${calls}`);
 assert.equal(JSON.stringify(f.server.raw.pages).includes('evidenceGuard'),false);
 assert.equal(JSON.stringify(f.server.raw.pages).includes('validateCommit'),false);
});

for(const stage of ['mutation','pre-submit'])test(`a Promise returned by the private evidence guard at ${stage} rejects without a revision`,async t=>{
 const f=await nativeFixture();t.after(()=>f.recorder.stop());const before=f.server.raw.pages.length;let calls=0;
 await assert.rejects(f.ledger.transitionActivity('manual:C',{expected:['awaiting-evidence'],patch:{review:{note:'must not persist'}},
  expectedSessionStatus:'recording',evidenceGuard:()=>{calls++;return stage==='mutation'||calls>1?Promise.resolve(true):true}}),/synchronous-evidence-guard-required/);
 assert.equal(f.server.raw.pages.length,before);assert.equal((await f.activity()).review?.note,undefined);
});

test('a JSON object cannot supply the private evidence guard',async t=>{
 const f=await nativeFixture();t.after(()=>f.recorder.stop());const before=f.server.raw.pages.length;
 await assert.rejects(async()=>f.ledger.transitionActivity('manual:C',{expected:['awaiting-evidence'],patch:{review:{note:'must not persist'}},
  expectedSessionStatus:'recording',evidenceGuard:{approved:true}}),/guard/);
 assert.equal(f.server.raw.pages.length,before);assert.equal((await f.activity()).review?.note,undefined);
});

test('GM authority lost during native successor hashing still prevents submission',async t=>{
 const f=await nativeFixture();t.after(()=>f.recorder.stop());const before=f.server.raw.pages.length,hold=holdConfirmedDigest(t);
 f.finish();await hold.entered.promise;f.server.revokeAuthority();hold.release.resolve();await f.server.recorder.settled();
 assert.equal((await f.activity()).state,'awaiting-evidence');assert.equal(f.server.raw.pages.length,before);
});
