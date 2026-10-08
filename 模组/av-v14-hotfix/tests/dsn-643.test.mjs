import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import vm from 'node:vm';
import {setup,settle,observe} from './dsn-queue-harness.mjs';

const read=name=>JSON.parse(fs.readFileSync(new URL('./fixtures/'+name,import.meta.url)));
const current=read('dsn-queue-6.4.3-native.json'),legacy=read('dsn-queue-native.json');
test('6.4.3 capture retains independent source and unchanged chat, model and quality contracts',()=>{
  assert.equal(current.provenance.version,'6.4.3');
  assert.equal(current.provenance.sha256,'c5d68e23c907f11e63c008066c639dce7a6f32c3f14edb90088364d4fe3dbdbe');
  assert.equal(current.provenance.manifest.sha256,'d8438372056c131a615aca7e3b5ef1b09618298375c54b03f5dcd37833988182');
  assert.equal(legacy.provenance.version,'6.4.2');
  assert.equal(legacy.provenance.sha256,'8ea57ede46b6b0f6e4367c1d9b1574c93ead3a95e981810945b53e7daf18e236');
  for(const[name,fragments]of Object.entries(current.provenance.unchangedContracts)){
    const fixture=read(name);
    for(const[key,text]of Object.entries(fixture.methods??fixture.dsn))
      assert.equal(createHash('sha256').update(text).digest('hex'),fragments[key].sha256,name+'/'+key);
  }
});
test('native 6.4.3 simulate rejection leaves rolling, preparation, binds and collisions behind',async()=>{
  const f=setup(current);
  f.engine.persistentDiceList.push(f.held,f.other);
  f.engine.persistentDiceManager={buildImpulseMap:()=>({}),_buildMeshByIdMap:()=>new Map()};
  f.held.quaternion={set(){}};
  f.fail('simulate');
  const first=observe(f.queue.enqueuePersistent({heldDice:[f.held],forcedByMesh:new Map(),velocity:{}}));
  await settle();
  assert.equal(first.value,false);
  assert.equal(f.engine.rolling,true);
  assert.equal(f.box._preparingThrow,true);
  assert.equal(f.held.userData.pendingBind,'current');
  assert.equal(f.workers.filter(([name,args])=>name==='setCollisionResponse'&&args.enabled).length,0);
  const second=observe(f.enqueue());
  await settle();
  assert.equal(second.status,'pending');
});
test('6.4.3 startup recovery settles affected binds, restores collisions and accepts the next throw',async()=>{
  const f=setup(current);
  assert.equal(f.install().status,'installed');
  f.engine.persistentDiceList.push(f.held,f.other);
  f.engine.persistentDiceManager={buildImpulseMap:()=>({}),_buildMeshByIdMap:()=>new Map()};
  f.held.quaternion={set(){}};
  f.fail('simulate');
  const first=observe(f.queue.enqueuePersistent({heldDice:[f.held],forcedByMesh:new Map(),velocity:{}}));
  const second=observe(f.enqueue());
  await settle();
  assert.equal(first.value,false);
  assert.equal(second.status,'pending');
  assert.equal(f.box._preparingThrow,false);
  assert.deepEqual(f.landed,['current']);
  assert.equal(f.other.userData.pendingBind,'other');
  assert.deepEqual(f.workers.filter(([name])=>name==='setCollisionResponse').map(([,args])=>[Array.from(args.ids),args.enabled]),
    [[[2],false],[[2],true],[[2],false]]);
  await f.finish();
  assert.equal(second.value,true);
  await f.queue.idle();
});
test('native 6.4.3 effects rejection reports success and skips ghost cleanup',async()=>{
  const f=setup(current);
  f.engine.persistentDiceList.push(f.other);
  const state=observe(f.enqueue());
  await settle();
  f.engine.diceList.push({userData:{system:'standard'},specialEffects:[{}]});
  f.fail('effects');
  await f.finish();
  assert.equal(state.value,true);
  assert.deepEqual(Array.from(f.engine._ghostifiedIds),[2]);
  assert.equal(f.workers.filter(([name,args])=>name==='setCollisionResponse'&&args.enabled).length,0);
});
test('native 6.4.3 late effects rejection invokes the next batch callback',async()=>{
  const f=setup(current),release=f.holdEffects(),first=observe(f.enqueue()),second=observe(f.enqueue());
  await settle();
  f.engine.diceList.push({userData:{system:'standard'},specialEffects:[{}]});
  await f.finish();
  f.engine.rolling=false;
  f.engine.callback(f.engine.throws);
  await settle();
  assert.equal(first.value,true);
  assert.equal(second.status,'pending');
  release.reject(Error('late effects failure'));
  await settle();
  assert.equal(second.value,true);
  assert.equal(f.engine.rolling,false);
  await f.queue.idle();
});
for(const failure of [false,true])test('6.4.3 late effects '+(failure?'rejection':'success')+' cannot finish a later batch',async()=>{
  const f=setup(current);
  assert.equal(f.install().completionStatus,'installed');
  const release=f.holdEffects(),first=observe(f.enqueue()),second=observe(f.enqueue());
  await settle();
  f.engine.diceList.push({userData:{system:'standard'},specialEffects:[{}]});
  await f.finish();
  f.engine.rolling=false;
  f.engine.callback(f.engine.throws);
  await settle();
  const nextCallback=f.engine.callback,nextThrows=f.engine.throws;
  if(failure)release.reject(Error('late effects failure'));else release();
  await settle();
  assert.equal(first.value,true);
  assert.equal(second.status,'pending');
  assert.equal(f.engine.rolling,true);
  assert.equal(f.engine.callback,nextCallback);
  assert.equal(f.engine.throws,nextThrows);
  await f.finish();
  assert.equal(second.value,true);
  await f.queue.idle();
});
for(const field of ['onEnd','boxAnimate'])test('mixed legacy/current '+field+' contracts are skipped',()=>{
  const f=setup(current),old=setup(legacy);
  if(field==='onEnd')f.queue.nextAnimation._onEnd=old.queue.nextAnimation._onEnd;
  else Object.getPrototypeOf(f.box).animateThrow=Object.getPrototypeOf(old.box).animateThrow;
  assert.equal(f.install().status,'unsupported-queue');
  assert.equal(Object.hasOwn(f.box,'startUnifiedBatch'),false);
  assert.equal(Object.hasOwn(f.engine,'handlePersistentThrowCompletion'),false);
});
test('changing throw identity during effects leaves foreign physics untouched and releases only its batch',async()=>{
  const f=setup(current);
  assert.equal(f.install().completionStatus,'installed');
  const release=f.holdEffects(),state=observe(f.enqueue());
  await settle();
  f.engine.diceList.push({userData:{system:'standard'},specialEffects:[{}]});
  await f.finish();
  const foreign=[];
  f.engine.throws=foreign;
  release();
  await settle();
  assert.equal(state.value,false);
  assert.equal(f.engine.rolling,true);
  assert.equal(f.engine.throws,foreign);
  assert.equal(f.workers.filter(([name])=>name==='setBodyPositions').length,0);
  await f.queue.idle();
});
for(const stage of ['collisions','positions'])for(const replacement of ['box','engine','throws'])
  test(stage+' cleanup awaiting its worker preserves a replacement '+replacement,async()=>{
    const f=setup(current);
    assert.equal(f.install().completionStatus,'installed');
    f.engine.persistentDiceList.push(f.other);
    const state=observe(f.enqueue());
    await settle();
    const release=f.holdWorker(stage);
    await f.finish();
    assert.equal(state.status,'pending');
    const callback=function foreign(){throw Error('foreign callback must not run');},foreign={rolling:true,callback,throws:[]};
    if(replacement==='box')f.queue.box={throwEngine:foreign};
    else if(replacement==='engine')f.box.throwEngine=foreign;
    else {f.engine.throws=foreign.throws;f.engine.callback=callback;}
    release();
    await settle();
    assert.equal(state.value,false);
    assert.equal(f.engine.rolling,true);
    assert.equal(foreign.rolling,true);
    if(replacement==='throws')assert.equal(f.engine.callback,callback);
    if(stage==='collisions')assert.equal(f.workers.filter(([name])=>name==='setBodyPositions').length,0);
    await f.queue.idle();
  });
test('an unknown queue constructor retains all native functions',()=>{
  const f=setup(current),onEnd=f.queue.nextAnimation._onEnd,start=f.box.startUnifiedBatch;
  f.queue.constructor=function foreign(){};
  assert.equal(f.install().status,'unsupported-queue');
  assert.equal(f.queue.nextAnimation._onEnd,onEnd);
  assert.equal(f.box.startUnifiedBatch,start);
});
for(const field of ['worker','animate','completion','effects'])test('pending cleanup preserves a replaced '+field+' contract',async()=>{
  const f=setup(current);
  assert.equal(f.install().completionStatus,'installed');
  f.engine.persistentDiceList.push(f.other);
  const state=observe(f.enqueue());
  await settle();
  const release=f.holdWorker('collisions');
  await f.finish();
  if(field==='worker')f.box.physicsWorker={exec(){throw Error('foreign worker must not run');}};
  if(field==='animate')Object.getPrototypeOf(f.box).animateThrow=function foreign(){};
  if(field==='completion')Object.getPrototypeOf(f.engine).handlePersistentThrowCompletion=function foreign(){};
  if(field==='effects')Object.getPrototypeOf(f.engine).handleSpecialEffectsInit=function foreign(){};
  release();
  await settle();
  assert.equal(state.value,false);
  assert.equal(f.engine.rolling,true);
  assert.equal(f.workers.filter(([name])=>name==='setBodyPositions').length,0);
  await f.queue.idle();
});
for(const replacement of ['box','engine','both'])test('startup rejection preserves a replacement '+replacement+' worker',async()=>{
  const f=setup(current),release=f.holdWorker('simulate'),foreignCalls=[];
  f.install();f.engine.persistentDiceList.push(f.other);
  const state=observe(f.enqueue());
  await settle();
  const ghostified=f.engine._ghostifiedIds,foreign={exec:(...args)=>{foreignCalls.push(args);return Promise.resolve(true);}};
  if(replacement==='box'||replacement==='both')f.box.physicsWorker=foreign;
  if(replacement==='engine'||replacement==='both')f.engine.physicsWorker=foreign;
  release.reject(Error('old simulation failed'));
  await settle();
  assert.equal(state.value,false);
  assert.equal(f.engine.rolling,true);
  assert.equal(f.box._preparingThrow,true);
  assert.equal(f.engine._ghostifiedIds,ghostified);
  assert.deepEqual(foreignCalls,[]);
  await f.queue.idle();
});
for(const replacement of ['worker','exec'])test('startup recovery rechecks '+replacement+' identity after restoring collisions',async()=>{
  const f=setup(current);f.install();f.engine.persistentDiceList.push(f.other);f.fail('simulate');
  const release=f.holdWorker('collisions'),state=observe(f.enqueue());await settle();
  assert.equal(state.status,'pending');const ghostified=f.engine._ghostifiedIds;
  if(replacement==='worker')f.engine.physicsWorker={exec(){throw Error('foreign worker');}};
  else f.box.physicsWorker.exec=function foreign(){throw Error('foreign exec');};
  assert.doesNotThrow(()=>f.ticker.frame());release();await settle();
  assert.equal(state.value,false);assert.equal(f.engine.rolling,true);assert.equal(f.box._preparingThrow,true);
  assert.equal(f.engine._ghostifiedIds,ghostified);await f.queue.idle();
});
for(const stage of ['collisions','positions'])test('intermediate ticker frames during stale '+stage+' cleanup neither throw nor touch replacement owners',async()=>{
  const f=setup(current);
  f.install();f.engine.persistentDiceList.push(f.other);
  const state=observe(f.enqueue());await settle();
  const release=f.holdWorker(stage);await f.finish();
  const foreignCalls=[],foreign={exec:(...args)=>{foreignCalls.push(args);return Promise.resolve(true);}};
  f.box.physicsWorker=foreign;f.engine.physicsWorker=foreign;
  assert.doesNotThrow(()=>f.ticker.frame());
  assert.doesNotThrow(()=>f.ticker.frame());
  assert.equal(state.status,'pending');
  assert.equal(f.engine.rolling,true);assert.deepEqual(foreignCalls,[]);
  release();await settle();
  assert.equal(state.value,false);assert.equal(f.engine.rolling,true);
  assert.deepEqual(foreignCalls,[]);await f.queue.idle();
});
test('native delayed persistent fade re-registration retains cleanup ownership protection',async()=>{
  const f=setup(current);
  f.install();f.engine.persistentDiceList.push(f.other);
  f.other.userData.persistentId='older';
  f.box._startMeshFade=()=>{};
  f.box.persistentDiceManager.removePersistentDie=async()=>true;
  const state=observe(f.enqueue());await settle();
  await f.box.fadeOutPersistentDie('older',1000);
  assert.equal(f.ticks.at(-1)[1],f.box);
  const release=f.holdWorker('collisions');await f.finish();
  let foreignCallbacks=0;
  const foreign=Object.assign(Object.create(Object.getPrototypeOf(f.engine)),f.engine,
    {rolling:true,throws:[],_ghostifiedIds:[77],callback(){foreignCallbacks++;}});
  f.box.throwEngine=foreign;
  assert.doesNotThrow(()=>f.ticker.frame());
  release();await settle();
  assert.equal(state.value,false);assert.equal(foreignCallbacks,0);
  assert.equal(foreign.rolling,true);assert.deepEqual(foreign._ghostifiedIds,[77]);
  await f.queue.idle();
});
const tickerConsumers=['spawnPersistentDie','removePersistentDie','fadeOutEphemeral','fadeOutPersistentDie','clearAll','clearScene'];
test('an unknown complete box profile with an added ticker consumer is refused',()=>{
  const f=setup(current),text=current.boxClass.slice(0,-1)+'extraTicker(){canvas.app.ticker.add(this.animateThrow,this)}}';
  const Foreign=vm.runInContext('('+text+')',f.context);
  Object.setPrototypeOf(f.box,Foreign.prototype);
  assert.equal(f.install().completionStatus,'unsupported-source');
  assert.equal(Object.hasOwn(f.box,'animateThrow'),false);
});
test('an unrelated own native completion alias is retained and refused',()=>{
  const f=setup(current),native=f.engine.handlePersistentThrowCompletion;
  f.engine.handlePersistentThrowCompletion=native;
  assert.equal(f.install().completionStatus,'unsupported-source');
  assert.equal(f.engine.handlePersistentThrowCompletion,native);
  assert.equal(Object.hasOwn(f.box,'animateThrow'),false);
});
for(const location of ['box-instance','box-prototype','engine-prototype'])test('pending cleanup checks changed '+location+' start consumer',async()=>{
  const f=setup(current);f.install();f.engine.persistentDiceList.push(f.other);
  const state=observe(f.enqueue());await settle();const release=f.holdWorker('collisions');await f.finish();
  const owner=location==='box-instance'?f.box:Object.getPrototypeOf(location==='box-prototype'?f.box:f.engine);
  owner.startUnifiedBatch=function foreign(){};
  assert.doesNotThrow(()=>f.ticker.frame());release();await settle();
  assert.equal(state.value,false);assert.equal(f.engine.rolling,true);await f.queue.idle();
});
for(const method of ['spawnPersistentDie','fadeOutEphemeral'])test('native '+method+' retains its stable ticker registration during cleanup',async()=>{
  const f=setup(current);f.install();f.engine.persistentDiceList.push(f.other);
  const state=observe(f.enqueue());await settle();const wrapped=f.box.animateThrow;
  const release=f.holdWorker('collisions');await f.finish();
  if(method==='spawnPersistentDie'){
    f.box.persistentDiceEnabled=true;f.box.persistentDiceManager.spawnPersistentDie=async()=>f.held;
    await f.box.spawnPersistentDie('d20',{});
  }else{
    f.engine.diceList.push(f.held);f.box._startMeshFade=()=>{};f.box.fadeOutEphemeral(1000);
  }
  assert.equal(f.ticks.at(-1)[0],wrapped);assert.equal(f.ticks.at(-1)[1],f.box);
  const foreign={exec(){throw Error('foreign worker must not run');}};
  f.box.physicsWorker=foreign;f.engine.physicsWorker=foreign;
  assert.doesNotThrow(()=>f.ticker.frame());release();await settle();
  assert.equal(state.value,false);assert.equal(f.engine.rolling,true);await f.queue.idle();
});
for(const method of ['removePersistentDie','clearAll','clearScene'])test('native '+method+' removes the owned ticker identity',async()=>{
  const f=setup(current);f.install();const wrapped=f.box.animateThrow;f.ticker.add(wrapped,f.box);
  if(method==='removePersistentDie')f.box.persistentDiceManager.removePersistentDie=async()=>true;
  else{
    f.box.initialized=true;f.box.cancelFade=()=>{};f.engine.clearAll=async()=>{};
    f.box._remoteOutlinePasses=new Map();f.box.diceScene.clearScene=()=>{};
  }
  await f.box[method]('older');assert.equal(f.ticker.has(wrapped),false);
});
for(const field of tickerConsumers)for(const location of ['prototype','instance'])
  test('installation refuses a foreign '+location+' '+field+' ticker consumer',()=>{
    const f=setup(current),native=f.box.animateThrow;
    (location==='prototype'?Object.getPrototypeOf(f.box):f.box)[field]=function foreign(){};
    assert.equal(f.install().completionStatus,'unsupported-source');
    assert.equal(f.box.animateThrow,native);
    assert.equal(Object.hasOwn(f.engine,'handlePersistentThrowCompletion'),false);
  });
for(const field of tickerConsumers)for(const location of ['prototype','instance'])
  test('pending cleanup preserves a changed '+location+' '+field+' ticker consumer',async()=>{
    const f=setup(current);f.install();f.engine.persistentDiceList.push(f.other);
    const state=observe(f.enqueue());await settle();
    const release=f.holdWorker('collisions');await f.finish();
    (location==='prototype'?Object.getPrototypeOf(f.box):f.box)[field]=function foreign(){};
    assert.doesNotThrow(()=>f.ticker.frame());
    release();await settle();
    assert.equal(state.value,false);assert.equal(f.engine.rolling,true);
    assert.equal(f.workers.filter(([name])=>name==='setBodyPositions').length,0);
    await f.queue.idle();
  });
test('ticker restore removes its owned registration and resumes the native function once',async()=>{
  const f=setup(current),native=f.box.animateThrow,patch=f.install(),state=observe(f.enqueue());
  await settle();
  const wrapped=f.box.animateThrow;
  assert.notEqual(wrapped,native);assert(f.ticker.has(wrapped));
  patch.restore();
  assert.equal(f.box.animateThrow,native);assert(!f.ticker.has(wrapped));assert(f.ticker.has(native));
  await f.finish();assert.equal(state.value,true);await f.queue.idle();
  assert.equal(f.ticker.has(native),false);
});
test('ticker restore leaves a later foreign animate function and registration alone',async()=>{
  const f=setup(current),patch=f.install(),state=observe(f.enqueue());await settle();
  const wrapped=f.box.animateThrow,foreign=function foreign(){};
  f.box.animateThrow=foreign;f.ticker.add(foreign,f.box);
  patch.restore();
  assert.equal(f.box.animateThrow,foreign);assert(f.ticker.has(foreign));assert(!f.ticker.has(wrapped));
  f.engine.rolling=false;f.engine.callback(f.engine.throws);await settle();
  assert.equal(state.value,true);await f.queue.idle();
});
test('native ticker business exceptions retain their original error',async()=>{
  const f=setup(current);f.install();const state=observe(f.enqueue());await settle();
  const error=Error('native stats failure');f.box.stats={update(){throw error;}};
  assert.throws(()=>f.ticker.frame(),value=>value===error);
  f.box.stats=null;await f.finish();assert.equal(state.value,true);await f.queue.idle();
});
