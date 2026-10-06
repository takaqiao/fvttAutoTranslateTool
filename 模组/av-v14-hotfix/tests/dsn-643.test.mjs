import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
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
