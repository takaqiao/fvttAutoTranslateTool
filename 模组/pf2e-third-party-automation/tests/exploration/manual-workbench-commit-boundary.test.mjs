import {test} from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {createLedger} from '../../scripts/exploration/ledger.mjs';
import {createRevisionStore} from '../../scripts/exploration/revision-store.mjs';
import {canonicalJSON,normalizeLedger,decodeRevision} from '../../scripts/exploration/revision-codec.mjs';
import {createManualEvents,IMMUNITY_SOURCES} from '../../scripts/exploration/manual-events.mjs';
import {manualEvidenceFixture,flush,M} from './manual-evidence-fixture.mjs';

const ROOT='JournalEntry.ROOT000000000001';
function deferred(){let resolve;const promise=new Promise(done=>{resolve=done});return {promise,resolve}}

async function revisionFixture(){
 const seed={sessions:{S:{id:'S',manual:true,status:'recording',startedAt:100,actorUUIDs:['Actor.H','Actor.P'],activityIds:[]}},activities:{},clocks:{}};
 const raw={_id:'ROOT000000000001',ownership:{default:0,G:3},pages:[],flags:{[M]:{explorationLedger:seed}}};
 let sequence=0;
 function client(name){
  let lastTransaction=Promise.resolve();
  const store=createRevisionStore({getRootUUID:()=>ROOT,isAuthority:()=>true,canUseRoot:()=>true,
   writerUserId:()=> 'G',writerClientId:()=>name,nonce:()=>`${name}-${++sequence}`,
   readRoot:async uuid=>{assert.equal(uuid,ROOT);return structuredClone(raw)},
   createPage:async({rootUUID,page})=>{
    assert.equal(rootUUID,ROOT);const envelope={type:'JournalEntryPage',action:'create',broadcast:false,operation:{parentUuid:ROOT},userId:'G'};
    if(raw.pages.some(saved=>saved._id===page._id))return {...envelope,error:{class:'ServerError',message:`The _id [${page._id}] already exists within the parent collection: JournalEntry [${raw._id}] pages`}};
    raw.pages.push(structuredClone(page));return {...envelope,result:[structuredClone(page)]};
   }});
  const ledger=createLedger({read:store.read,transact:(fn,options)=>{
   const transaction=store.transact(fn,options);lastTransaction=transaction.then(()=>{},()=>{});return transaction;
  },isAuthority:()=>true,identity:()=>({userId:'G',clientNonce:name})});
  return {store,ledger,settled:()=>lastTransaction};
 }
 const recorder=client('workbench-recorder'),stopper=client('stopper');
 await recorder.store.initialize({epoch:'workbench-commit-boundary',expectedSourceDigest:createHash('sha256').update(canonicalJSON(normalizeLedger(seed))).digest('hex')});
 return {raw,recorder,stopper};
}

async function waitForActivity(f,predicate){
 const until=Date.now()+5000;let activity;
 do{activity=await f.activity();if(predicate(activity))return activity;await flush()}while(Date.now()<until);
 assert.fail(`ledger did not settle: ${JSON.stringify({activity,errors:f.errors})}`);
}

async function workbenchFixture(){
 const f=manualEvidenceFixture(),server=await revisionFixture();f.game.time.worldTime=100;f.ledger=server.recorder.ledger;
 f.result.flags[M].explorationManual.continualRecovery=false;
 f.result.update=async function(changes){this.flags.pf2e??={context:{}};this.flags.pf2e.context.options=changes['flags.pf2e.context.options'];return this};
 f.options={...f.options,ledger:f.ledger,hpPools:{discover:actor=>({poolUUID:actor.uuid,ready:true})},
  fromUuid:async uuid=>f.actors.get(uuid)??[...f.patient.items.values()].find(item=>item.uuid===uuid)};
 f.recorder=createManualEvents(f.options);f.recorder.start();f.activity=()=>f.ledger.getActivity('manual:W');
 await f.fire(f.result);await waitForActivity(f,a=>a?.state==='awaiting-evidence');
 f.hpReceipt=f.receipt();await f.fire(f.hpReceipt);await waitForActivity(f,a=>a?.proof.receiptIds.includes('R'));
 assert.deepEqual((await f.activity()).options.missing,['native-immunity-receipt']);
 f.nativeCreates=0;
 f.patient.createEmbeddedDocuments=async(type,data)=>{
  f.nativeCreates++;assert.equal(type,'Item');assert.equal(data.length,1);
  const saved={...f.immunity(),...data[0],id:'I',uuid:'Actor.P.Item.I',actor:f.patient,parent:f.patient};
  f.patient.items.set('I',saved);return [saved];
 };
 const observer=f.recorder.bindImmunity({message:f.result,token:{id:'T',actor:f.patient},kind:'treatment',
  sourceSHA:IMMUNITY_SOURCES['XDY DO_NOT_IMPORT TW Immunity CD'].sha});
 f.finish=()=>observer.createEmbeddedDocuments('Item',[{type:'effect',sourceId:'Compendium.pf2e.feat-effects.Item.Lb4q2bBAgxamtix5',
  system:{slug:'treat-wounds-immunity',start:{value:100},duration:{value:60,unit:'minutes',expiry:'turn-start',sustained:false}},flags:{}}]);
 return Object.assign(f,{server});
}

function holdConfirmedDigest(t){
 const hold={entered:deferred(),release:deferred()},original=globalThis.crypto.subtle.digest;let armed=true;
 globalThis.crypto.subtle.digest=async function(...args){
  const result=await original.apply(this,args);let body;
  try{body=JSON.parse(new TextDecoder().decode(args[1]))}catch{}
  if(armed&&body?.changes?.activities?.['manual:W']?.state==='confirmed'){
   armed=false;hold.entered.resolve();await hold.release.promise;
  }
  return result;
 };
 t.after(()=>{hold.release.resolve();globalThis.crypto.subtle.digest=original});return hold;
}

test('ordinary source-pinned Workbench completion commits through the real revision store',async t=>{
 const f=await workbenchFixture();t.after(()=>f.recorder.stop());const saved=await f.finish(),activity=await f.activity();
 assert.equal(f.nativeCreates,1);assert.equal(saved[0],f.patient.items.get('I'));assert.equal(activity.state,'confirmed');
 assert.deepEqual(activity.options.missing,[]);assert.deepEqual(activity.proof.checkIds,['C']);assert.deepEqual(activity.proof.resultIds,['W']);
 assert.deepEqual(activity.proof.receiptIds,['R']);assert.deepEqual(activity.proof.immunityIds,['Actor.P.Item.I']);
 assert.equal(f.server.raw.pages.map(page=>decodeRevision(page.flags[M].explorationRevision)).at(-1).changes.activities['manual:W'].state,'confirmed');
 assert.deepEqual(f.errors,[]);
});

test('ordinary Workbench completion cannot commit after Stop wins during successor hashing',async t=>{
 const f=await workbenchFixture();t.after(()=>f.recorder.stop());const hold=holdConfirmedDigest(t);
 const pending=f.finish().then(value=>({value}),error=>({error}));await hold.entered.promise;
 await f.server.stopper.ledger.updateSession('S',{status:'stopped'});hold.release.resolve();await pending;await f.server.recorder.settled();
 assert.equal(f.nativeCreates,1);assert.equal((await f.ledger.getSession('S')).status,'stopped');
 assert.equal((await f.activity()).state,'awaiting-evidence');
});
test('an ordinary one-result Workbench use cannot fold two independent HP receipts into one application',async t=>{
 const f=await workbenchFixture();t.after(()=>f.recorder.stop());await f.fire(f.receipt('R2'));await waitForActivity(f,a=>a.proof.receiptIds.length===2);await f.finish();assert.equal((await f.activity()).state,'awaiting-evidence');assert.ok((await f.activity()).options.missing.includes('native-application-receipt'));assert.equal(f.nativeCreates,1);
});

for(const [name,invalidate] of [
 ['the patient HP pool is rebound',f=>{f.options.hpPools.discover=()=>({poolUUID:'Actor.SHARED',ready:true})}],
 ['the HP receipt author loses patient OWNER',f=>{f.patient.testUserPermission=user=>user?.isGM===true}],
 ['the saved HP receipt is reverted',f=>{f.hpReceipt.flags.pf2e.appliedDamage.isReverted=true}],
])test(`ordinary Workbench completion cannot submit when ${name} during successor hashing`,async t=>{
 const f=await workbenchFixture();t.after(()=>f.recorder.stop());const before=f.server.raw.pages.length,hold=holdConfirmedDigest(t);
 const pending=f.finish().then(value=>({value}),error=>({error}));await hold.entered.promise;invalidate(f);hold.release.resolve();
 await pending;await f.server.recorder.settled();assert.equal(f.nativeCreates,1);
 const activity=await f.activity();assert.equal(activity.state,'awaiting-evidence');
 assert.deepEqual(activity.proof.receiptIds,['R']);assert.equal(f.server.raw.pages.length,before);
});
