import {registerBabeleIndex} from '../scripts/patches/babele.mjs';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import {test} from 'node:test';

const readFixture=name=>JSON.parse(readFileSync(new URL('./fixtures/'+name,import.meta.url),'utf8'));
const source=readFixture('babele-upstream-3.1.2.json');
const baseline=source;
const core=readFixture('babele-core-14.367.json');
function fn(text,name){return text.functions[name]??'';}
function cls(name){return core[name];}
const wrapper=source.wrapper;

class Collection extends Map { [Symbol.iterator]() { return this.values(); } }
class Item { static metadata = {indexed:true, collection:'items', embedded:{}}; static schema = {has:()=>true}; }
function document(id, name, pack='example.items') {
  return Object.assign(new Item(), {id, _id:id, name, pack, isEmbedded:false, documentName:'Item', uuid:pack ? `Compendium.${pack}.Item.${id}` : `Item.${id}`});
}
async function harness() {
  const timers = new Map(); let serial = 0;
  const state = {titleIndex:{'example.items':{titles:{a:'烈焰 长剑', b:'寒霜 法杖'}}}};
  const pack = {collection:'example.items', documentName:'Item', documentClass:Item, index:new Collection()};
  const docs = {a:document('a','Old Sword'), b:document('b','Old Staff'), u:document('u','Unrelated Shield')};
  const world = document('w','World Treasure',null);
  for (const doc of Object.values(docs)) pack.index.set(doc.id, {_id:doc.id, uuid:doc.uuid, name:doc.name});
  const game = {release:{generation:14}, packs:new Collection([[pack.collection,pack]]), items:[world], i18n:{lang:'en'}, babele:{__ondemandPatch:state}};
  const foundry = {utils:{getDocumentClass:()=>Item, mergeObject:(a,b)=>({...a,...b})}, applications:{ux:{SearchFilter:{cleanQuery:s=>s}}}};
  const c = vm.createContext({game, foundry, CONST:{WORLD_DOCUMENT_TYPES:['Item'],vtt:'test'}, CONFIG:{Item:{documentClass:Item},i18n:{searchStopWords:new Set()}}, performance, console:{debug(){}}, tracePatch(){}, logPatch(){}, isOnDemandMode:()=>true, PATCH_ID:'test', libWrapper:{register(_id,_path,callback){c.wrapper=callback;}}, setTimeout(callback){const id=++serial;timers.set(id,callback);return id;}});
  vm.runInContext(`Array.prototype.findSplice=function(fn){const i=this.findIndex(fn);if(i>=0)return this.splice(i,1)[0];};\n${cls('StringTree')}\n${cls('WordTree')}\nfoundry.utils.StringTree=StringTree;foundry.utils.WordTree=WordTree;\n${cls('DocumentIndex')}\ngame.documentIndex=new DocumentIndex();`,c);
  await game.documentIndex.index();
  const counts = {full:0, replace:0, removed:0};
  const index = game.documentIndex;
  const full = index.index.bind(index), replace = index.replaceDocument.bind(index), remove = index.removeDocument.bind(index);
  index.index = async () => {counts.full++;return full();};
  index.replaceDocument = d => {counts.replace++;return replace(d);};
  index.removeDocument = d => {counts.removed++;return remove(d);};
  for (const name of ['scheduleDocumentIndexRebuild','rebuildDocumentIndexCompat','scheduleDocumentIndexEntryReplace','normalizePackId','translateIndexTitles']) {
    const code = fn(source,name); if(code)vm.runInContext(code,c);
  }
  vm.runInContext(wrapper,c);
  const update = d => c.wrapper.call(pack,()=>{pack.index.set(d.id,{_id:d.id,uuid:d.uuid,name:d.name});return 'wrapped-result';},d);
  const takeTimer = () => {const first=timers.entries().next().value;assert.ok(first,'a flush was scheduled');timers.delete(first[0]);return first[1]();};
  const flush = async () => {for(let i=0;timers.size;i++){assert.ok(i<12,'queue must settle');await takeTimer();}};
  const search = query => [...(index.lookup(query,{limit:50}).Item??[])].map(x=>x.uuid);
  return {state,pack,docs,world,game,index,c,counts,update,takeTimer,flush,timers,search};
}

test('unsupported adapter leaves upstream title translation and full rebuild behavior intact',async()=>{
  const h=await harness();h.game.modules=new Map([['pf2e_compendium_chn',{active:true}]]);
  const original=h.index.index;
  const registration=registerBabeleIndex({g:{game:h.game}});
  assert.equal(registration.status().state,'unsupported');assert.equal(registration.status().applied,false);
  assert.equal(h.index.index,original);assert.equal(h.timers.size,0);
  h.update(h.docs.a);await h.flush();
  assert.equal(h.counts.full,1);assert.equal(h.counts.replace,0);
  assert.deepEqual(h.search('烈焰'),[h.docs.a.uuid]);assert.deepEqual(h.search('World'),[h.world.uuid]);
});

test('full-mode setting precedence retains the upstream not-applicable branch',()=>{
  const game={modules:new Map([['pf2e_compendium_chn',{active:true}]]),babele:{__ondemandPatch:{}},settings:{get:ns=>ns==='babele'?'full':'ondemand'}};
  const registration=registerBabeleIndex({g:{game}});assert.equal(registration.status().state,'not-applicable');
  assert.equal(registration.status().mode,'full');assert.equal(registration.status().applied,false);
  game.settings.get=ns=>ns==='babele'?'invalid':'ondemand';assert.equal(registration.status().state,'unsupported');
});

test('a per-instance index-only serializer still duplicates leaves after overlapping upstream clears',async()=>{
  const h=await harness(),original=h.index.index;let pending=Promise.resolve();
  h.index.index=function(){return pending=pending.then(()=>original.call(this));};
  await Promise.all([h.c.rebuildDocumentIndexCompat(),h.c.rebuildDocumentIndexCompat()]);
  assert.equal(h.counts.full,2);assert.equal(h.index.trees.Item.lookup('sword').length,2);
});

test('the actual libWrapper package check rejects unregister calls made under a different package ID',()=>{
  const packageCheck=readFixture('babele-lib-wrapper-1.13.5.1.json').packageCheck;
  class PackageInfo{
    constructor(id='av-v14-hotfix'){this.id=id;this.exists=true;this.type_plus_id_capitalized=id;}
    static is_valid_key_or_id(){return true;}
    equals(other){return other.id===this.id;}
  }
  const context=vm.createContext({PackageInfo,t:'lib-wrapper',e:{package:Error},globalThis:{game:{modules:new Map([['av-v14-hotfix',{}]])}}});
  vm.runInContext(packageCheck,context);
  assert.throws(()=>context.be('pf2e_compendium_chn'),/not allowed to call libWrapper/);
  assert.equal(context.be('av-v14-hotfix').id,'av-v14-hotfix');
});
