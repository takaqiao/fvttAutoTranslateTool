import test from 'node:test';
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {canonicalJSON, normalizeLedger, revisionId, createGenesisRevision, createSuccessorRevision, projectRevisionPages} from '../../scripts/exploration/revision-codec.mjs';

const M='pf2e-third-party-automation', rootUUID='JournalEntry.RootLedger000001';
const identity={rootUUID,epoch:'epoch-1',nonce:'genesis',writerUserId:'GM',writerClientId:'client-a'};
const empty=()=>({sessions:{},activities:{},clocks:{}});
const hash=value=>createHash('sha256').update(value).digest('hex');
const page=metadata=>({_id:revisionId(metadata.revision),flags:{[M]:{explorationRevision:metadata}}});
const seal=metadata=>{const {digest,...body}=metadata;return {...body,digest:hash(canonicalJSON(body))}};
const seed=()=>({sessions:{S:{id:'S',startedAt:-600,activityIds:['A'],status:'paused'}},activities:{A:{id:'A',state:'uncertain',proof:{useId:'use-1',checkIds:['C2','C1']}}},clocks:{C:{id:'C',state:'uncertain',from:-600,to:0}}});
async function chain(){
  const initial=seed(),genesis=await createGenesisRevision({...identity,seed:initial});assert.ok(genesis,'genesis metadata required');
  const first=structuredClone(initial);first.sessions.S.status='closed';first.activities.B={id:'B',state:'planned'};
  const r1=await createSuccessorRevision({previous:genesis,state:initial,nextState:first,nonce:'one',writerUserId:'GM',writerClientId:'client-a'});assert.ok(r1,'successor metadata required');
  const second=structuredClone(first);second.activities.A.proof.checkIds=['C1','C2'];second.clocks.C.state='confirmed';second.sessions.T={id:'T',status:'recording'};
  const r2=await createSuccessorRevision({previous:r1,state:first,nextState:second,nonce:'two',writerUserId:'GM2',writerClientId:'client-b'});assert.ok(r2,'second successor metadata required');
  return {initial,first,second,genesis,r1,r2,pages:[page(genesis),page(r1),page(r2)]};
}

test('canonical JSON sorts object keys lexically, keeps array order and omits object undefined',()=>{
  assert.equal(canonicalJSON({z:undefined,b:[{y:2,x:1},null,false,'雪'],a:{'2':'two','10':'ten'},zero:-0}),'{"a":{"10":"ten","2":"two"},"b":[{"x":1,"y":2},null,false,"雪"],"zero":0}');
  assert.notEqual(canonicalJSON([1,2]),canonicalJSON([2,1]));
});
for(const [label,value]of [['NaN',NaN],['Infinity',Infinity],['undefined',undefined],['bigint',1n],['function',()=>{}],['symbol',Symbol('x')],['Date',new Date(0)],['Map',new Map()],['array undefined',[undefined]],['sparse array',Array(1)]])test(`canonical JSON rejects ${label}`,()=>assert.throws(()=>canonicalJSON(value)));
test('canonical JSON rejects prototype keys, cycles and accessor records without invoking getters',()=>{
  for(const key of ['__proto__','constructor','prototype'])assert.throws(()=>canonicalJSON(JSON.parse(`{"${key}":1}`)));
  const cycle={};cycle.self=cycle;assert.throws(()=>canonicalJSON(cycle));
  let invoked=false;const record={get a(){invoked=true;return 1}};assert.throws(()=>canonicalJSON(record));assert.equal(invoked,false);
});
test('canonical JSON rejects hidden array entries and unsupported symbol keys instead of hashing ambiguous data',()=>{
  const extra=[1];extra.extra=2;assert.throws(()=>canonicalJSON(extra));
  const symbol={a:1,[Symbol('secret')]:2};assert.throws(()=>canonicalJSON(symbol));
});
test('normalization preserves legacy records and returns detached complete collections',()=>{
  const input={sessions:{S:{id:'S',note:undefined,events:['b','a']}}};const result=normalizeLedger(input);
  assert.deepEqual(result,{sessions:{S:{id:'S',events:['b','a']}},activities:{},clocks:{}});
  result.sessions.S.events.push('c');assert.deepEqual(input.sessions.S.events,['b','a']);assert.ok(Object.hasOwn(input.sessions.S,'note'));
});
for(const [label,value]of [['null',null],['array',[]],['unknown collection',{sessions:{},other:{}}],['null collection',{sessions:null}],['array collection',{activities:[]}],['null record',{clocks:{C:null}}],['array record',{activities:{A:[]}}],['undefined record',{sessions:{S:undefined}}],['prototype key',JSON.parse('{"sessions":{"__proto__":{}}}')]])test(`normalization rejects ${label}`,()=>assert.throws(()=>normalizeLedger(value)));
test('revision IDs are injective sixteen-character encodings across safe integer boundaries',()=>{
  assert.equal(revisionId(0),'er00000000000000');assert.equal(revisionId(35),'er0000000000000z');assert.equal(revisionId(36),'er00000000000010');
  const last=revisionId(Number.MAX_SAFE_INTEGER);assert.equal(last.length,16);assert.equal(parseInt(last.slice(2),36),Number.MAX_SAFE_INTEGER);assert.notEqual(last,revisionId(Number.MAX_SAFE_INTEGER-1));
  for(const bad of [-1,0.5,NaN,Infinity,Number.MAX_SAFE_INTEGER+1,'1',null])assert.throws(()=>revisionId(bad));
});
test('genesis digest binds the normalized seed, source fingerprint and writer identity',async()=>{
  const g=await createGenesisRevision({...identity,seed:empty()});assert.ok(g);
  const sourceDigest=hash('{"activities":{},"clocks":{},"sessions":{}}');
  const expected={schemaVersion:1,...identity,revision:0,previousDigest:null,seed:empty(),sourceDigest};
  assert.deepEqual(g,{...expected,digest:hash(`{"epoch":"epoch-1","nonce":"genesis","previousDigest":null,"revision":0,"rootUUID":"${rootUUID}","schemaVersion":1,"seed":{"activities":{},"clocks":{},"sessions":{}},"sourceDigest":"${sourceDigest}","writerClientId":"client-a","writerUserId":"GM"}`)});
});
test('legacy seed plus two successors preserve unrelated records and replace whole changed records',async()=>{
  const c=await chain();assert.deepEqual(c.r1.changes,{sessions:{S:c.first.sessions.S},activities:{B:c.first.activities.B},clocks:{}});
  assert.deepEqual(c.r2.changes,{sessions:{T:c.second.sessions.T},activities:{A:c.second.activities.A},clocks:{C:c.second.clocks.C}});
  assert.equal(c.r1.previousDigest,c.genesis.digest);assert.equal(c.r2.previousDigest,c.r1.digest);assert.equal(c.r2.epoch,c.genesis.epoch);
  const projected=await projectRevisionPages({rootUUID,pages:[c.pages[2],c.pages[0],c.pages[1]]});assert.deepEqual(projected,{state:c.second,head:c.r2});
});
test('successor rejects record deletion while permitting property replacement inside a record',async()=>{
  const state=seed(),g=await createGenesisRevision({...identity,seed:state});assert.ok(g);
  const deleted=structuredClone(state);delete deleted.activities.A;
  await assert.rejects(createSuccessorRevision({previous:g,state,nextState:deleted,nonce:'bad',writerUserId:'GM',writerClientId:'c'}),/delet/i);
  const replaced=structuredClone(state);replaced.activities.A={id:'A',state:'confirmed'};
  const r=await createSuccessorRevision({previous:g,state,nextState:replaced,nonce:'ok',writerUserId:'GM',writerClientId:'c'});
  assert.deepEqual((await projectRevisionPages({rootUUID,pages:[page(g),page(r)]})).state.activities.A,{id:'A',state:'confirmed'});
});
test('object key order and omitted undefined do not create spurious record changes',async()=>{
  const state={sessions:{S:{a:1,b:2}}},g=await createGenesisRevision({...identity,seed:state});assert.ok(g);
  const r=await createSuccessorRevision({previous:g,state,nextState:{sessions:{S:{b:2,a:1,unused:undefined}}},nonce:'same',writerUserId:'GM',writerClientId:'c'});
  assert.deepEqual(r.changes,empty());
});
test('genesis rejects invalid root and blank identity before hashing',async()=>{
  for(const patch of [{rootUUID:'Actor.RootLedger000001'},{rootUUID:'JournalEntry.short'},{epoch:''},{nonce:''},{writerUserId:''},{writerClientId:' '}])await assert.rejects(createGenesisRevision({...identity,...patch,seed:empty()}));
});
test('project requires genesis and refuses a gap or duplicate revision page',async()=>{
  const c=await chain();
  await assert.rejects(projectRevisionPages({rootUUID,pages:[]}),/protocol-not-initialized/);
  await assert.rejects(projectRevisionPages({rootUUID,pages:[c.pages[1]]}),/protocol-not-initialized/);
  await assert.rejects(projectRevisionPages({rootUUID,pages:[c.pages[0],c.pages[2]]}),/gap|contigu/i);
  await assert.rejects(projectRevisionPages({rootUUID,pages:[...c.pages,c.pages[1]]}),/duplicate/i);
});
test('projection ignores unrelated pages but rejects damaged revision namespace pages',async()=>{
  const c=await chain();const unrelated=[{_id:'notes00000000001',flags:{}},{_id:'notes00000000002',flags:{other:{value:1}}}];
  assert.deepEqual((await projectRevisionPages({rootUUID,pages:[...unrelated,...c.pages]})).state,c.second);
  for(const damaged of [{_id:'er00000000000003',flags:{}},{_id:'erBad',flags:{}},{_id:'notes00000000003',flags:{[M]:{explorationRevision:c.r1}}}])await assert.rejects(projectRevisionPages({rootUUID,pages:[...c.pages,damaged]}));
});
for(const [label,mutate]of [
  ['schema',r=>{r.schemaVersion=2}],['root',r=>{r.rootUUID='JournalEntry.OtherRoot0000001'}],['epoch',r=>{r.epoch='other-epoch'}],
  ['predecessor',r=>{r.previousDigest='0'.repeat(64)}],['revision',r=>{r.revision=8}],['extra field',r=>{r.unrecognized=true}],
  ['null record',r=>{r.changes.activities.A=null}],['unknown collection',r=>{r.changes.others={}}]
])test(`projection rejects a correctly rehashed successor with invalid ${label}`,async()=>{
  const c=await chain(),bad=structuredClone(c.r2);mutate(bad);const changed=seal(bad);
  await assert.rejects(projectRevisionPages({rootUUID,pages:[c.pages[0],c.pages[1],{...c.pages[2],flags:{[M]:{explorationRevision:changed}}}]}));
});
test('projection rejects seed tampering, source fingerprint mismatch and successor digest corruption',async()=>{
  const c=await chain(),seedChanged=structuredClone(c.genesis);seedChanged.seed.sessions.S.status='running';
  await assert.rejects(projectRevisionPages({rootUUID,pages:[page(seedChanged)]}));
  const wrongSource=seal({...c.genesis,sourceDigest:'0'.repeat(64)});await assert.rejects(projectRevisionPages({rootUUID,pages:[page(wrongSource)]}));
  const wrongDigest={...c.r2,digest:'0'.repeat(64)};await assert.rejects(projectRevisionPages({rootUUID,pages:[c.pages[0],c.pages[1],page(wrongDigest)]}));
});
test('known head detects rollback and replacement while a cold valid prefix remains readable',async()=>{
  const c=await chain();await assert.rejects(projectRevisionPages({rootUUID,pages:c.pages.slice(0,2),knownHead:c.r2}),/regress|rollback/i);
  const changed=await createSuccessorRevision({previous:c.r1,state:c.first,nextState:c.second,nonce:'replacement',writerUserId:'GM2',writerClientId:'client-b'});
  await assert.rejects(projectRevisionPages({rootUUID,pages:[c.pages[0],c.pages[1],page(changed)],knownHead:c.r2}),/replac|head/i);
  assert.deepEqual((await projectRevisionPages({rootUUID,pages:c.pages,knownHead:c.r1})).head,c.r2);
  assert.deepEqual((await projectRevisionPages({rootUUID,pages:c.pages.slice(0,2)})).state,c.first);
});
test('known head itself must have a valid digest and matching root and epoch',async()=>{
  const c=await chain();for(const head of [{...c.r1,digest:'0'.repeat(64)},seal({...c.r1,rootUUID:'JournalEntry.OtherRoot0000001'}),seal({...c.r1,epoch:'other-epoch'})])await assert.rejects(projectRevisionPages({rootUUID,pages:c.pages,knownHead:head}));
});
test('async creation snapshots inputs before the caller can change them',async()=>{
  const input={...identity,seed:seed()},before=structuredClone(input);const pending=createGenesisRevision(input);input.seed.sessions.S.status='running';input.nonce='changed';
  const g=await pending;assert.ok(g);assert.deepEqual(g.seed,before.seed);assert.equal(g.nonce,before.nonce);
  const previous=structuredClone(g),state=structuredClone(g.seed),nextState=structuredClone(state);nextState.sessions.S.status='closed';
  const next=createSuccessorRevision({previous,state,nextState,nonce:'next',writerUserId:'GM',writerClientId:'c'});previous.digest='bad';state.activities.A.state='bad';nextState.sessions.S.status='bad';
  const r=await next;assert.ok(r);assert.equal(r.previousDigest,g.digest);assert.equal(r.changes.sessions.S.status,'closed');assert.deepEqual(r.changes.activities,{});
});
test('projection snapshots pages and known head before hashing and never edits caller data',async()=>{
  const c=await chain(),pages=structuredClone(c.pages),knownHead=structuredClone(c.r1),before=structuredClone(pages);
  const pending=projectRevisionPages({rootUUID,pages,knownHead});pages[0].flags[M].explorationRevision.seed.sessions.S.status='bad';knownHead.digest='bad';
  const projected=await pending;assert.deepEqual(projected.state,c.second);
  const untouched=structuredClone(before),known=structuredClone(c.r1);await projectRevisionPages({rootUUID,pages:untouched,knownHead:known});assert.deepEqual(untouched,before);assert.deepEqual(known,c.r1);
  projected.state.sessions.S.status='mutated-result';projected.head.changes.activities.A.state='mutated-result';assert.deepEqual(c.pages,before);
});
