import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createRevisionStore} from '../../scripts/exploration/revision-store.mjs';
import {canonicalJSON,createGenesisRevision,createSuccessorRevision,projectRevisionPages,revisionId} from '../../scripts/exploration/revision-codec.mjs';

const nativeRoot=process.env.FVTT_NATIVE_APP??'C:/Program Files/Foundry Virtual Tabletop/resources/app';
const {ObjectField}=await import(pathToFileURL(path.join(nativeRoot,'common/data/fields.mjs')));
const M='pf2e-third-party-automation',rootUUID='JournalEntry.ROOT000000000001';
const empty=()=>({sessions:{},activities:{},clocks:{}});
const hash=value=>createHash('sha256').update(canonicalJSON(value)).digest('hex');
const page=metadata=>({_id:revisionId(metadata.revision),flags:{[M]:{explorationRevision:metadata}}});
const clean=flags=>new ObjectField().clean(structuredClone(flags),{expand:true},{creation:true});
const special=()=>({id:'S',nativeOwnerByActor:{'Actor.Healer':'Player'},nested:[{'a.b':1,'-=literal':'kept','==replacement':'literal'}]});
async function genesis(){return createGenesisRevision({rootUUID,epoch:'epoch',nonce:'initial',writerUserId:'GM',writerClientId:'C0',seed:empty()})}
function fixture(){
 const raw={_id:rootUUID.slice(13),ownership:{default:0},flags:{[M]:{explorationLedger:empty()}},pages:[]};
 let creates=0,sequence=0,afterCreate;
 const client=()=>createRevisionStore({getRootUUID:()=>rootUUID,readRoot:async()=>structuredClone(raw),isAuthority:()=>true,canUseRoot:()=>true,writerUserId:()=> 'GM',writerClientId:'client-'+(++sequence),nonce:()=> 'nonce-'+(++sequence),createPage:async({page:input})=>{
  creates++;const operation={parentUuid:rootUUID};
  if(raw.pages.some(p=>p._id===input._id))return {type:'JournalEntryPage',action:'create',broadcast:false,operation,userId:'GM',error:{class:'ServerError',message:'The _id ['+input._id+'] already exists within the parent collection: JournalEntry ['+raw._id+'] pages'}};
  const saved={...structuredClone(input),flags:clean(input.flags)};raw.pages.push(saved);
  const ack={type:'JournalEntryPage',action:'create',broadcast:false,operation,userId:'GM',result:[structuredClone(saved)]};
  return afterCreate?afterCreate(ack):ack;
 }});
 return {raw,client,get creates(){return creates},setAfterCreate:fn=>{afterCreate=fn}};
}

test('native flags cleaner preserves new signed owner maps and literal JSON operator keys',async()=>{
 const f=fixture(),store=f.client();await store.initialize({epoch:'epoch',expectedSourceDigest:hash(empty())});
 await store.transact(state=>{state.sessions.S=special()});
 assert.equal(typeof f.raw.pages[1].flags[M].explorationRevision,'string');
 assert.deepEqual((await store.read()).sessions.S,special());
 assert.equal((await store.status()).revision,1);
});

test('legacy object genesis and canonical string successors project one unchanged logical chain',async()=>{
 const g=await genesis(),state=empty();state.sessions.S=special();
 const r=await createSuccessorRevision({previous:g,state:empty(),nextState:state,nonce:'n',writerUserId:'GM',writerClientId:'C'});
 const result=await projectRevisionPages({rootUUID,pages:[page(g),{...page(r),flags:{[M]:{explorationRevision:canonicalJSON(r)}}}],knownHead:g});
 assert.deepEqual(result,{state,head:r});
 const f=fixture();f.raw.pages.push(page(g));const store=f.client();await store.transact(s=>{s.sessions.S=special()});
 assert.deepEqual((await store.read()).sessions.S,special());assert.equal((await store.status()).sourceDigest,g.sourceDigest);
});

test('native-cleaned independent clients preserve both pure changes through a unique page conflict',async()=>{
 const f=fixture(),setup=f.client();await setup.initialize({epoch:'epoch',expectedSourceDigest:hash(empty())});
 const a=f.client(),b=f.client();await Promise.all([a.transact(s=>{s.sessions.S=special()}),b.transact(s=>{s.activities.A={id:'A',source:{'Actor.Patient':'proof'}}})]);
 const state=await a.read();assert.deepEqual(state.sessions.S,special());assert.equal(state.activities.A.source['Actor.Patient'],'proof');assert.equal(f.creates,4);assert.equal(f.raw.pages.length,3);
});

test('native-cleaned persisted unknown ACK neither returns a grant nor retries',async()=>{
 const f=fixture(),store=f.client();await store.initialize({epoch:'epoch',expectedSourceDigest:hash(empty())});f.setAfterCreate(()=>{throw Error('lost-ack')});let grant=0;
 await assert.rejects(store.transact(s=>{s.sessions.S=special();return true}).then(()=>grant++),/lost-ack/);
 assert.equal(grant,0);assert.equal(f.creates,2);assert.equal(f.raw.pages.length,2);f.setAfterCreate(null);assert.deepEqual((await f.client().read()).sessions.S,special());
});

for(const [label,change]of [
 ['whitespace',wire=>' '+wire],['malformed',()=>'{'],['array',()=> '[]'],['null',()=> 'null'],
 ['duplicate key',wire=>wire.replace('{','{"epoch":"shadow",')],
 ['wrong schema',wire=>canonicalJSON({...JSON.parse(wire),schemaVersion:2})],
 ['bad digest',wire=>canonicalJSON({...JSON.parse(wire),digest:'0'.repeat(64)})],
 ['prototype key',wire=>wire.replace('{','{"__proto__":{},')]
])test('string revision rejects '+label,async()=>{
 const g=await genesis();await assert.rejects(projectRevisionPages({rootUUID,pages:[{...page(g),flags:{[M]:{explorationRevision:change(canonicalJSON(g))}}}]}));
});

for(const [label,alter]of [
 ['noncanonical text',wire=>' '+wire],
 ['logical nonce',wire=>canonicalJSON({...JSON.parse(wire),nonce:'different'})],
 ['object substitution',wire=>JSON.parse(wire)]
])test('physical ACK '+label+' cannot grant or retry',async()=>{
 const f=fixture(),s=f.client();await s.initialize({epoch:'epoch',expectedSourceDigest:hash(empty())});f.setAfterCreate(ack=>{ack.result[0].flags[M].explorationRevision=alter(ack.result[0].flags[M].explorationRevision);return ack});let grant=0;
 await assert.rejects(s.transact(state=>{state.sessions.S=special()}).then(()=>grant++),/revision-acknowledgement-mismatch/);assert.equal(grant,0);assert.equal(f.creates,2);
});

test('historical expanded owner map is rejected instead of repaired or rehashed',async()=>{
 const g=await genesis(),state=empty();state.sessions.S={id:'S',nativeOwnerByActor:{'Actor.Healer':'Player'}};
 const r=await createSuccessorRevision({previous:g,state:empty(),nextState:state,nonce:'n',writerUserId:'GM',writerClientId:'C'});
 const good=page(r);assert.deepEqual((await projectRevisionPages({rootUUID,pages:[page(g),good]})).state,state);
 const bad={...good,flags:clean(good.flags)};await assert.rejects(projectRevisionPages({rootUUID,pages:[page(g),bad]}),/revision-digest-mismatch/);
});
