import {registerBabeleIndex,captureBabeleCore} from '../scripts/patches/babele.mjs';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import {test} from 'node:test';

const readFixture=name=>JSON.parse(readFileSync(new URL('./fixtures/'+name,import.meta.url),'utf8'));
const source=readFixture('babele-upstream-3.1.2.json');
const baseline=source;
const core=readFixture('babele-core-14.367.json');
const nativeBabele=readFixture('babele-native-2.9.1.json');
function fn(text,name){return text.functions[name]??'';}
function cls(name){return core[name];}
const wrapper=source.wrapper;

class Collection extends Map { [Symbol.iterator]() { return this.values(); } }
class Item { static metadata = {indexed:true, collection:'items', embedded:{}}; static schema = {has:()=>true}; }
function document(id, name, pack='example.items') {
  return Object.assign(new Item(), {id, _id:id, name, pack, isEmbedded:false, documentName:'Item', uuid:pack ? `Compendium.${pack}.Item.${id}` : `Item.${id}`});
}
async function harness({beforeCapture=()=>{},beforeRegister=()=>{},expectedState='applied',instrument=true,registrationType='WRAPPER',helperOverride='native',priorities={},registrationOrder=['babele','pf2e_compendium_chn'],translationVersion='3.1.2'}={}) {
  const timers=new Map(),hooks=new Map();let serial=0;
  const Hooks={on(name,callback){const id=++serial;hooks.set(id,{name,callback});return id;},off(_name,id){hooks.delete(id);}};
  const emit=(name,...args)=>{for(const h of [...hooks.values()])if(h.name===name)h.callback(...args);};
  const state={titleIndex:{'example.items':{titles:{a:'烈焰 长剑',b:'寒霜 法杖'}}}};
  const game={version:'14.368',ready:true,release:{generation:14},packs:new Collection(),items:[],i18n:{lang:'en'},
    modules:new Map([['pf2e_compendium_chn',{active:true,version:translationVersion}],['babele',{active:true,version:'2.9.1'}],['lib-wrapper',{active:true,version:'1.13.5.1'}]]),
    settings:{get:(namespace,key)=>namespace==='lib-wrapper'&&key==='module-priorities'?priorities:'ondemand'},babele:{__ondemandPatch:state}};
  const foundry={utils:{getDocumentClass:()=>Item,mergeObject:(a,b)=>({...a,...b}),getProperty:(data,key)=>data[key],setProperty:(data,key,value)=>data[key]=value},applications:{ux:{SearchFilter:{cleanQuery:s=>s}}},documents:{collections:{}}};
  const c=vm.createContext({game,foundry,Hooks,CONST:{WORLD_DOCUMENT_TYPES:['Item'],vtt:'test'},CONFIG:{Item:{documentClass:Item},i18n:{searchStopWords:new Set()}},performance,console:{debug(){}},tracePatch(){},logPatch(){},isOnDemandMode:()=>game.settings.get('babele','loadingMode')==='ondemand',PATCH_ID:'pf2e_compendium_chn',Collection,
    libWrapper:{register(_id,_path,callback){c.wrapper=callback;}},setTimeout(callback){const id=++serial;timers.set(id,callback);return id;},clearTimeout:id=>timers.delete(id)});
  const method=core.compendiumIndexDocument;
  vm.runInContext(`Array.prototype.findSplice=function(fn){const i=this.findIndex(fn);if(i>=0)return this.splice(i,1)[0];};\n${cls('StringTree')}\n${cls('WordTree')}\nfoundry.utils.StringTree=StringTree;foundry.utils.WordTree=WordTree;\n${cls('DocumentIndex')}\nclass NativeCompendium {#indexedFields=new Set(['name']);constructor(){this.index=new Collection();} ${method}}\nfoundry.documents.collections.CompendiumCollection=NativeCompendium;game.documentIndex=new DocumentIndex();`,c);
  const pack=new foundry.documents.collections.CompendiumCollection();Object.assign(pack,{collection:'example.items',documentName:'Item',documentClass:Item});
  game.packs.set(pack.collection,pack);
  const docs={a:document('a','Old Sword'),b:document('b','Old Staff'),u:document('u','Unrelated Shield')},world=document('w','World Treasure',null);game.items.push(world);
  for(const doc of Object.values(docs)){doc._source={name:doc.name};pack.index.set(doc.id,{_id:doc.id,uuid:doc.uuid,name:doc.name});}
  await game.documentIndex.index();
  for(const name of ['scheduleDocumentIndexRebuild','rebuildDocumentIndexCompat','normalizePackId','translateIndexTitles'])vm.runInContext(fn(source,name),c);
  game.babele.translateIndexTitles=(index,packId)=>c.translateIndexTitles(state,index,packId);
  const proto=foundry.documents.collections.CompendiumCollection.prototype;
  beforeCapture({g:c,pack,index:game.documentIndex,proto,emit});
  const capture=captureBabeleCore(c),native=proto.indexDocument;
  vm.runInContext(nativeBabele.helper,c);
  vm.runInContext(nativeBabele.titleReset.code,c);
  const nativeWrapper=vm.runInContext('('+nativeBabele.callback+')',c);
  vm.runInContext(wrapper,c);
  const callbacks={babele:nativeWrapper,pf2e_compendium_chn:c.wrapper};
  for(const owner of registrationOrder){
    const previous=proto.indexDocument,callback=callbacks[owner];
    proto.indexDocument=function(...args){return callback.call(this,previous.bind(this),...args);};
    emit('libWrapper.Register',owner,'foundry.documents.collections.CompendiumCollection.prototype.indexDocument',registrationType);
  }
  beforeRegister({g:c,pack,index:game.documentIndex,proto,emit,capture});
  const registration=registerBabeleIndex({g:c,capture,preserveIndexFlags:helperOverride==='native'?c.preserveIndexFlags:helperOverride});
  assert.equal(registration.status().state,expectedState,JSON.stringify(registration.status()));
  const counts={full:0,replace:0,removed:0},index=game.documentIndex;
  const full=index.index.bind(index),replace=index.replaceDocument.bind(index),remove=index.removeDocument.bind(index);
  if(instrument){index.index=async()=>{counts.full++;return full();};index.replaceDocument=d=>{counts.replace++;return replace(d);};index.removeDocument=d=>{counts.removed++;return remove(d);};}
  const update=d=>{d._source={name:d.name};return pack.indexDocument(d);};
  // A real title refresh clears previous translation markers before applying changed titles.
  // Babele 2.9.1 deliberately preserves those markers on ordinary document indexing.
  const retitle=(id,name)=>{state.titleIndex[pack.collection].titles[id]=name;c.restorePackIndexCompat(pack);};
  const takeTimer=()=>{const first=timers.entries().next().value;assert.ok(first,'a flush was scheduled');timers.delete(first[0]);return first[1]();};
  const flush=async()=>{for(let i=0;timers.size;i++){assert.ok(i<12,'queue must settle');await takeTimer();}};
  const search=query=>[...(index.lookup(query,{limit:50}).Item??[])].map(x=>x.uuid);
  return {state,pack,docs,world,game,index,c,counts,update,retitle,takeTimer,flush,timers,search,registration,emit,capture,native,proto,hooks,nativeWrapper};
}

test('translated title replaces old leaves; unrelated pack/world search stays intact without rebuilding',async()=>{
  const h=await harness(); const tree=h.index.trees.Item;
  assert.equal(h.update(h.docs.a),undefined); await h.flush();
  assert.deepEqual(h.search('Old Sword'),[]);
  assert.deepEqual(h.search('烈焰'),[h.docs.a.uuid]);
  assert.deepEqual(h.search('Unrelated'),[h.docs.u.uuid]);
  assert.deepEqual(h.search('World'),[h.world.uuid]);
  assert.equal(h.index.trees.Item,tree,'unrelated index tree is retained');
  assert.equal(h.counts.full,0); assert.equal(h.counts.replace,1);
});
test('repeated UUIDs use the latest translated entry once per batch',async()=>{
  const h=await harness();h.update(h.docs.a);h.update(h.docs.a);
  h.retitle('a','星辰 长剑');h.update(h.docs.a);await h.flush();
  assert.deepEqual(h.search('烈焰'),[]);assert.deepEqual(h.search('星辰'),[h.docs.a.uuid]);
  assert.equal(h.counts.replace,1);assert.equal(h.counts.full,0);
  h.retitle('a','清晨 长剑');h.update(h.docs.a);await h.flush();
  assert.deepEqual(h.search('星辰'),[]);assert.deepEqual(h.search('清晨'),[h.docs.a.uuid]);
  assert.equal(h.index.trees.Item.lookup('清晨').length,1,'raw word tree contains no duplicate leaves');
});
test('waits for ready and includes documents queued while waiting',async()=>{
  const h=await harness();let ready;Object.defineProperty(h.index,'ready',{get:()=>new Promise(r=>ready=r),configurable:true});
  h.update(h.docs.a);const active=h.takeTimer();await Promise.resolve();await Promise.resolve();
  assert.equal(h.counts.replace,0);h.update(h.docs.b);h.update(h.docs.a);ready();await active;delete h.index.ready;await h.flush();
  assert.deepEqual(h.search('烈焰'),[h.docs.a.uuid]);assert.deepEqual(h.search('寒霜'),[h.docs.b.uuid]);
  assert.equal(h.counts.replace,2);assert.equal(h.counts.full,0);
});
test('a reentrant update during replacement runs in the next batch',async()=>{
  const h=await harness();const replace=h.index.replaceDocument;let once=false;
  h.index.replaceDocument=d=>{replace(d);if(!once){once=true;h.retitle('a','雷霆 长剑');h.update(h.docs.a);}};
  h.update(h.docs.a);await h.flush();
  assert.deepEqual(h.search('烈焰'),[]);assert.deepEqual(h.search('雷霆'),[h.docs.a.uuid]);
  assert.equal(h.counts.replace,2);assert.equal(h.counts.full,0);
});
test('a document removed from its pack before the flush is not reintroduced',async()=>{
  const h=await harness();h.update(h.docs.a);h.pack.index.delete('a');await h.flush();
  assert.deepEqual(h.search('Old Sword'),[]);assert.deepEqual(h.search('烈焰'),[]);assert.equal(h.counts.full,0);
});
test('replacement failure requests the existing full rebuild and later batches still work',async()=>{
  const h=await harness();const replace=h.index.replaceDocument;let fail=true;
  h.index.replaceDocument=d=>{if(fail){fail=false;throw Error('simulated failure');}return replace(d);};
  h.update(h.docs.a);h.update(h.docs.b);await h.flush();
  assert.equal(h.counts.full,1);assert.deepEqual(h.search('烈焰'),[h.docs.a.uuid]);assert.deepEqual(h.search('寒霜'),[h.docs.b.uuid]);
  h.retitle('a','晨光 长剑');h.update(h.docs.a);await h.flush();
  assert.equal(h.counts.full,1);assert.deepEqual(h.search('晨光'),[h.docs.a.uuid]);
});
test('unsupported generation retains the original full rebuild',async()=>{
  const h=await harness();h.game.release.generation=15;h.update(h.docs.a);await h.flush();
  assert.equal(h.counts.full,1);assert.deepEqual(h.search('烈焰'),[h.docs.a.uuid]);
});
test('missing replacement API falls back without losing translated search',async()=>{
  const h=await harness();h.index.replaceDocument=undefined;h.update(h.docs.a);await h.flush();
  assert.equal(h.counts.full,1);assert.deepEqual(h.search('烈焰'),[h.docs.a.uuid]);
});
test('a rejected ready promise requests one fallback, settles, and allows recovery',async()=>{
  const h=await harness();let reject;Object.defineProperty(h.index,'ready',{get:()=>new Promise((_resolve,r)=>reject=r),configurable:true});
  h.update(h.docs.a);const active=h.takeTimer();await Promise.resolve();await Promise.resolve();h.update(h.docs.b);
  reject(Error('simulated ready failure'));await active;delete h.index.ready;await h.flush();
  assert.equal(h.counts.full,1);assert.deepEqual(h.search('烈焰'),[h.docs.a.uuid]);assert.deepEqual(h.search('寒霜'),[h.docs.b.uuid]);
  h.update(h.docs.a);await h.flush();assert.equal(h.counts.full,1);assert.equal(h.counts.replace,1);
});
test('a concurrent initial full rebuild cannot leave duplicate translated leaves',async()=>{
  const h=await harness();h.update(h.docs.a);
  const rebuilding=h.c.rebuildDocumentIndexCompat();const replacing=h.takeTimer();
  await Promise.all([rebuilding,replacing]);await h.flush();
  assert.equal(h.index.trees.Item.lookup('烈焰').length,1);
  h.retitle('a','繁星 长剑');h.update(h.docs.a);await h.flush();
  assert.deepEqual(h.search('烈焰'),[]);assert.deepEqual(h.search('繁星'),[h.docs.a.uuid]);
});
test('a full rebuild started while replacement waits for ready is also awaited',async()=>{
  const h=await harness();let ready;const pending=new Promise(r=>ready=r);
  Object.defineProperty(h.index,'ready',{get:()=>pending,configurable:true});
  h.update(h.docs.a);const replacing=h.takeTimer();await Promise.resolve();await Promise.resolve();
  const rebuilding=h.c.rebuildDocumentIndexCompat();ready();await Promise.all([rebuilding,replacing]);delete h.index.ready;await h.flush();
  assert.equal(h.index.trees.Item.lookup('烈焰').length,1);
  h.retitle('a','月光 长剑');h.update(h.docs.a);await h.flush();assert.deepEqual(h.search('烈焰'),[]);
});
test('overlapping full rebuilds serialize and retain one set of leaves',async()=>{
  const h=await harness();h.update(h.docs.a);
  await Promise.all([h.c.rebuildDocumentIndexCompat(),h.c.rebuildDocumentIndexCompat(),h.takeTimer()]);await h.flush();
  assert.equal(h.counts.full,2);assert.equal(h.index.trees.Item.lookup('烈焰').length,1);
  assert.deepEqual(h.search('World'),[h.world.uuid]);
});
test('initial/all-pack translation still requests a full rebuild and its callers stay unchanged',async()=>{
  for(const name of ['applyLightRuntimeTranslations','scheduleDocumentIndexRebuild'])assert.equal(fn(source,name),fn(baseline,name));
  const h=await harness();h.c.scheduleDocumentIndexRebuild(h.state,'all-pack-test');await h.flush();assert.equal(h.counts.full,1);
});

test('an unknown target registration disables native bypass and calls the new prototype wrapper',async()=>{
  const h=await harness();let calls=0;const upstream=h.proto.indexDocument;
  h.proto.indexDocument=function(...args){calls++;return upstream.apply(this,args);};
  h.emit('libWrapper.Register','other-module','foundry.documents.collections.CompendiumCollection.prototype.indexDocument','MIXED');
  assert.equal(h.registration.status().state,'unsupported');
  h.update(h.docs.a);await h.flush();
  assert.equal(calls,1);assert.equal(h.counts.full,1);assert.equal(h.counts.replace,0);
  assert.equal(h.registration.status().metrics.nativeCalls,0);assert.deepEqual(h.search('烈焰'),[h.docs.a.uuid]);
});
test('an unknown wrapper observed before registration prevents any own-method installation',async()=>{
  const h=await harness({instrument:false,expectedState:'unsupported',beforeRegister({emit}){
    emit('libWrapper.Register','other-module','CompendiumCollection.prototype.indexDocument','WRAPPER');
  }});
  assert.equal(Object.hasOwn(h.pack,'indexDocument'),false);assert.equal(Object.hasOwn(h.index,'index'),false);
});
test('already wrapped native capture and version drift both fail closed',async()=>{
  const late=await harness({instrument:false,expectedState:'unsupported',beforeCapture({proto}){
    const original=proto.indexDocument;proto.indexDocument=function(...args){return original.apply(this,args);};
  }});
  assert.equal(Object.hasOwn(late.pack,'indexDocument'),false);assert.match(late.registration.status().reason,/capture/);
  const changed=await harness({instrument:false,expectedState:'unsupported',beforeRegister({g}){g.game.modules.get('babele').version='2.9.2';}});
  assert.equal(Object.hasOwn(changed.pack,'indexDocument'),false);assert.equal(Object.hasOwn(changed.index,'index'),false);
});
test('full mode installs no instance methods and retains the native bypass',async()=>{
  const h=await harness({instrument:false,expectedState:'not-applicable',beforeRegister({g}){g.game.settings.get=()=>'full';}});
  assert.equal(Object.hasOwn(h.pack,'indexDocument'),false);assert.equal(Object.hasOwn(h.index,'index'),false);
  assert.equal(Object.hasOwn(h.index,'ready'),false);h.update(h.docs.a);assert.equal(h.timers.size,0);
  assert.equal(h.pack.index.get('a').name,'Old Sword');
});
test('unknown own pack methods remain intact and new native packs attach on createCompendium',async()=>{
  let own;
  const h=await harness({beforeRegister({pack,proto}){own=function(...args){return proto.indexDocument.apply(this,args);};pack.indexDocument=own;}});
  assert.equal(h.pack.indexDocument,own);assert.deepEqual(h.registration.status().skippedPacks,['example.items']);
  const pack=new h.c.foundry.documents.collections.CompendiumCollection();Object.assign(pack,{collection:'example.new',documentClass:Item,documentName:'Item'});
  h.game.packs.set(pack.collection,pack);h.emit('createCompendium',pack);
  assert.equal(Object.hasOwn(pack,'indexDocument'),true);assert.equal(h.registration.status().attachedPacks,1);
  assert.equal(registerBabeleIndex({g:h.c,capture:h.capture}),h.registration);
});
test('a customized DocumentIndex method prevents installing any pack bypass',async()=>{
  const h=await harness({instrument:false,expectedState:'unsupported',beforeRegister({index}){const indexFn=index.index;index.index=function(...args){return indexFn.apply(this,args);};}});
  assert.equal(Object.hasOwn(h.pack,'indexDocument'),false);assert.match(h.registration.status().reason,/unknown own/);
});
test('dispose flushes queued search changes, restores only owned methods, and removes all hooks',async()=>{
  const h=await harness({instrument:false}),protoMethod=h.proto.indexDocument;
  h.update(h.docs.a);assert.equal(h.timers.size,1);await h.registration.dispose();
  assert.deepEqual(h.search('烈焰'),[h.docs.a.uuid]);assert.equal(h.timers.size,0);
  assert.equal(h.registration.status().state,'disposed');assert.equal(h.pack.indexDocument,protoMethod);
  assert.equal(Object.hasOwn(h.index,'index'),false);assert.equal(Object.hasOwn(h.index,'ready'),false);
  assert.equal(h.hooks.size,0);
});
test('disposing never overwrites a later own method installed by another module',async()=>{
  const h=await harness({instrument:false}),foreign=()=>42;h.pack.indexDocument=foreign;
  await h.registration.dispose();assert.equal(h.pack.indexDocument,foreign);
});
test('an update arriving after a fallback build snapshot schedules a second full build',async()=>{
  const h=await harness();h.index.replaceDocument=undefined;
  const build=h.index._indexCompendium;let once=false;
  h.index._indexCompendium=function(pack){build.call(this,pack);if(!once){once=true;h.retitle('a','雷霆 长剑');h.update(h.docs.a);}};
  h.update(h.docs.a);await h.flush();
  assert.equal(h.counts.full,2);assert.deepEqual(h.search('烈焰'),[]);assert.deepEqual(h.search('雷霆'),[h.docs.a.uuid]);
});
test('accepts the actual libWrapper enum object supplied to libWrapper.Register',async()=>{
  const enumFactory=readFixture('babele-lib-wrapper-1.13.5.1.json').enumFactory;
  const context=vm.createContext({});
  vm.runInContext(enumFactory+';globalThis.actualTypes=s("WrapperType",{WRAPPER:1,MIXED:2,OVERRIDE:3,LISTENER:4});',context);
  const type=context.actualTypes.WRAPPER;assert.equal(typeof type,'object');assert.equal(String(type),'WRAPPER');
  const h=await harness({registrationType:type});h.update(h.docs.a);await h.flush();
  assert.equal(h.counts.full,0);assert.equal(h.counts.replace,1);
});

test('the verified Babele helper preserves prior index metadata and document Babele flags before title translation',async()=>{
  const h=await harness();
  Object.assign(h.pack.index.get('a'),{originalName:'Original Sword',customMetadata:{keep:true},hasTranslation:false});
  h.docs.a.flags={babele:{hasTranslation:true,translated:true,name:'已有完整译名',customFlag:17}};
  assert.equal(h.update(h.docs.a),undefined);await h.flush();
  const entry=h.pack.index.get('a');
  assert.equal(entry.originalName,'Original Sword');assert.deepEqual(entry.customMetadata,{keep:true});
  assert.equal(entry.hasTranslation,true);assert.equal(entry.customFlag,17);assert.equal(entry.name,'已有完整译名');
  assert.deepEqual(h.search('已有完整译名'),[h.docs.a.uuid]);assert.deepEqual(h.search('烈焰'),[]);
  assert.equal(h.counts.full,0);assert.equal(h.counts.replace,1);
});

test('a changed or absent Babele helper cannot enable native bypass and keeps both upstream wrappers',async()=>{
  for(const helper of [null,function preserveIndexFlags(collection,wrapped,args){return wrapped(...args);}]){
    const h=await harness({helperOverride:helper,expectedState:'unsupported',instrument:false});
    assert.equal(Object.hasOwn(h.pack,'indexDocument'),false);assert.match(h.registration.status().reason,/preserveIndexFlags/);
    Object.assign(h.pack.index.get('a'),{customMetadata:42});
    h.docs.a.flags={babele:{customFlag:17}};h.update(h.docs.a);await h.flush();
    assert.equal(h.pack.index.get('a').customMetadata,42);assert.equal(h.pack.index.get('a').customFlag,17);
    assert.deepEqual(h.search('烈焰'),[h.docs.a.uuid]);assert.equal(h.registration.status().metrics.nativeCalls,0);
  }
});

test('unregistering the audited Babele wrapper disables instance bypass',async()=>{
  const h=await harness();
  h.emit('libWrapper.Unregister','babele','foundry.documents.collections.CompendiumCollection.prototype.indexDocument');
  assert.equal(h.registration.status().state,'unsupported');h.update(h.docs.a);await h.flush();
  assert.equal(h.counts.full,1);assert.equal(h.registration.status().metrics.nativeCalls,0);
});

test('ordinary reindexing preserves translated markers exactly like the original two-wrapper chain',async()=>{
  const adapted=await harness(),upstream=await harness({helperOverride:null,expectedState:'unsupported'});
  for(const h of [adapted,upstream]){
    h.update(h.docs.a);await h.flush();
    h.state.titleIndex['example.items'].titles.a='尚未刷新译名';h.update(h.docs.a);await h.flush();
    assert.equal(h.pack.index.get('a').name,'烈焰 长剑');
    assert.deepEqual(h.search('烈焰'),[h.docs.a.uuid]);assert.deepEqual(h.search('尚未刷新译名'),[]);
  }
  assert.deepEqual(JSON.parse(JSON.stringify(adapted.pack.index.get('a'))),JSON.parse(JSON.stringify(upstream.pack.index.get('a'))));
});

test('the Babele helper retains the captured native return identity, receiver and all arguments',async()=>{
  const expected={nativeResult:true},options={test:true};let observed,calls=0;
  const h=await harness({beforeRegister({capture}){
    const native=capture.native;
    capture.native=function(...args){calls++;observed={receiver:this,args};native.apply(this,args);return expected;};
  }});
  assert.equal(h.pack.indexDocument(h.docs.a,options,'extra'),expected);
  assert.equal(calls,1);assert.equal(observed.receiver,h.pack);assert.equal(observed.args[0],h.docs.a);
  assert.equal(observed.args[1],options);assert.equal(observed.args[2],'extra');await h.flush();
  assert.deepEqual(h.search('烈焰'),[h.docs.a.uuid]);
});

test('accepts equal effective priorities and rejects different priorities using the real libWrapper configuration format',async()=>{
  const row=(id,index)=>({id,title:id,index});
  const cases=[
    [{},'applied'],
    [{prioritized:{'module:babele':row('babele',4),'module:pf2e_compendium_chn':row('pf2e_compendium_chn',4)}},'applied'],
    [{deprioritized:{'module:babele':row('babele',2),'module:pf2e_compendium_chn':row('pf2e_compendium_chn',2)}},'applied'],
    [{prioritized:{'module:babele':row('babele',0)}},'unsupported'],
    [{prioritized:{'module:babele':row('babele',0),'module:pf2e_compendium_chn':row('pf2e_compendium_chn',1)}},'unsupported'],
    [{prioritized:{'module:babele':row('babele',0),'module:pf2e_compendium_chn':row('pf2e_compendium_chn',0)},deprioritized:{'module:babele':row('babele',3)}},'applied'],
    [{prioritized:{babele:row('babele',0)}},'applied'],
    [{prioritized:{'module:babele':row('babele','invalid')}},'applied']
  ];
  const priorityLoader=readFixture('babele-lib-wrapper-1.13.5.1.json').priorityLoader;
  for(const [priorities,expectedState] of cases){
    const context=vm.createContext({PackageInfo:{is_valid_key_or_id:()=>true},Log:{},configuration:priorities});
    vm.runInContext(priorityLoader+';ae(configuration);globalThis.actual=[ie.get("module:babele")??0,ie.get("module:pf2e_compendium_chn")??0];',context);
    assert.equal(context.actual[0]===context.actual[1],expectedState==='applied');
    const h=await harness({priorities,expectedState,instrument:false});
    if(expectedState==='unsupported'){
      assert.equal(Object.hasOwn(h.pack,'indexDocument'),false);assert.match(h.registration.status().reason,/priorit/i);
    }else{h.update(h.docs.a);await h.flush();assert.deepEqual(h.search('烈焰'),[h.docs.a.uuid]);}
  }
});

test('equal priorities with reversed upstream registration order cannot enable the instance bypass',async()=>{
  const h=await harness({registrationOrder:['pf2e_compendium_chn','babele'],expectedState:'unsupported',instrument:false});
  assert.equal(Object.hasOwn(h.pack,'indexDocument'),false);assert.match(h.registration.status().reason,/order/i);
});

test('a later priority-setting update disables bypass and pending batches fall back through the original chain',async()=>{
  const h=await harness();h.update(h.docs.a);
  h.emit('updateSetting',{key:'lib-wrapper.module-priorities',value:{prioritized:{}}});
  assert.equal(h.registration.status().state,'unsupported');assert.match(h.registration.status().reason,/priorit.*reload/i);
  h.update(h.docs.b);await h.flush();assert.equal(h.counts.replace,0);assert.ok(h.counts.full>=1);
  assert.equal(h.registration.status().metrics.nativeCalls,1);
  assert.deepEqual(h.search('烈焰'),[h.docs.a.uuid]);assert.deepEqual(h.search('寒霜'),[h.docs.b.uuid]);
});

test('unrelated setting updates leave the verified wrapper order enabled',async()=>{
  const h=await harness();h.emit('updateSetting',{key:'other-module.setting'});
  assert.equal(h.registration.status().state,'applied');h.update(h.docs.a);await h.flush();assert.equal(h.counts.full,0);
});

test('unreadable or nonfinite priority configurations fail closed',async()=>{
  for(const priorities of [null,'unknown',{prioritized:{'module:babele':{id:'babele',title:'Babele',index:NaN}}}]){
    const h=await harness({priorities,expectedState:'unsupported',instrument:false});
    assert.equal(Object.hasOwn(h.pack,'indexDocument'),false);assert.match(h.registration.status().reason,/priorit/i);
  }
});


test('3.2.1 identical translation runtime keeps single-entry indexing and translated search',async()=>{
  const h=await harness({translationVersion:'3.2.1'});
  h.update(h.docs.a);h.update(h.docs.a);await h.flush();
  assert.deepEqual(h.search('Old Sword'),[]);
  assert.deepEqual(h.search('烈焰'),[h.docs.a.uuid]);
  assert.deepEqual(h.search('World'),[h.world.uuid]);
  assert.equal(h.counts.full,0);assert.equal(h.counts.replace,1);
});
test('a newer unverified translation release retains the upstream rebuild',async()=>{
  const h=await harness({translationVersion:'3.2.2',expectedState:'unsupported'});
  h.update(h.docs.a);await h.flush();
  assert.equal(h.counts.full,1);assert.equal(h.counts.replace,0);
  assert.deepEqual(h.search('烈焰'),[h.docs.a.uuid]);
});
