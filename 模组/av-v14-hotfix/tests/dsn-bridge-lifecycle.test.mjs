import test,{describe} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {setup,settle,observe} from './dsn-queue-harness.mjs';
import {prepareBridge,prepareRegisteredSettings} from './dsn-persistent-bridge-harness.mjs';
const worker=JSON.parse(fs.readFileSync(new URL('./fixtures/dsn-worker-native.json',import.meta.url)));
test('worker RPC fixture has the same exact source in both captured DsN profiles',()=>{
  assert.equal(createHash('sha256').update(worker.exec).digest('hex'),worker.sha256);
  assert.equal(worker.sha256,'9f910bea44df0fe32a3ad4153b4525e556b5ace83ea99ac47124ee8165aa5e91');
  assert.deepEqual(worker.provenance.map(pin=>pin.version),['6.4.3','6.4.2']);
  assert.deepEqual(worker.provenance.map(pin=>pin.sha256),[
    'c5d68e23c907f11e63c008066c639dce7a6f32c3f14edb90088364d4fe3dbdbe',
    '8ea57ede46b6b0f6e4367c1d9b1574c93ead3a95e981810945b53e7daf18e236']);
});

for(const file of ['dsn-queue-native.json','dsn-queue-6.4.3-native.json','dsn-queue-6.4.4-native.json'])describe(file,()=>{
  const fixture=JSON.parse(fs.readFileSync(new URL('./fixtures/'+file,import.meta.url)));
  async function retainTicker(f){
    f.engine.persistentDiceList.push(f.other);f.other.userData.persistentId='legacy';
    const prior=observe(f.enqueue());await settle();await f.finish();
    assert.equal(prior.value,true);assert.equal(f.ticker.has(f.box.animateThrow),true);
  }
  async function finishFollowing(f,first,second,release){
    release();await settle();for(let i=0;i<3;i++)await f.finish();
    assert.equal(first.value,true);assert.equal(second.value,true);
    const next=observe(f.enqueue());await settle();await f.finish();
    assert.equal(next.value,true);assert.equal(f.engine.rolling,false);assert.equal(f.engine.running,false);
    assert.equal(f.box._preparingThrow,false);assert.equal(f.errors.length,0);await f.queue.idle();
  }
  for(const patched of [false,true])test((patched?'recovery':'native')+' retained persistent ticker permits an unbound startup and following throws',async()=>{
    const f=setup(fixture);if(patched)f.install();await retainTicker(f);
    const release=f.holdWorker('simulate'),first=observe(f.enqueue()),second=observe(f.enqueue());await settle();
    assert.equal(f.box._preparingThrow,true);assert.equal(f.engine.callback,null);assert.equal(f.engine.throws,null);
    assert.doesNotThrow(()=>f.ticker.frame());await settle();
    assert.equal(first.status,'pending');assert.equal(second.status,'pending');
    await finishFollowing(f,first,second,release);
  });
  for(const enabled of [false,true])test('registered enabled='+enabled+' with a retained ticker preserves startup and the next throw',async()=>{
    const f=setup(fixture);prepareBridge(f);f.install();const settings=prepareRegisteredSettings(f);
    if(!enabled)await settings.setEnabled(true);await retainTicker(f);
    const release=f.holdWorker('simulate'),first=observe(f.enqueue()),second=observe(f.enqueue());await settle();
    await settings.setEnabled(enabled);assert.doesNotThrow(()=>f.ticker.frame());await settle();
    assert.equal(first.status,'pending');assert.equal(second.status,'pending');
    await finishFollowing(f,first,second,release);
    f.engine.persistentDiceList.length=0;await settings.setEnabled(false);
  });
  for(const field of ['worker','exec'])test('retained ticker rejects unknown '+field+' replacement during unbound startup',async()=>{
    const f=setup(fixture);f.install();await retainTicker(f);
    const release=f.holdWorker('simulate'),first=observe(f.enqueue()),second=observe(f.enqueue());await settle();
    const calls=[],foreign={rolling:true,exec:(...args)=>{calls.push(args);return Promise.resolve(true);}};
    if(field==='worker'){f.box.physicsWorker=foreign;f.engine.physicsWorker=foreign;}else f.box.physicsWorker.exec=foreign.exec;
    assert.doesNotThrow(()=>f.ticker.frame());await settle();
    assert.equal(first.value,false);assert.equal(second.value,false);assert.deepEqual(calls,[]);assert.equal(foreign.rolling,true);
    release();await settle();await f.queue.idle();
  });
  test('a retained ticker cannot reopen startup after its callback and throws were acquired',async()=>{
    const f=setup(fixture);f.install();await retainTicker(f);
    const first=observe(f.enqueue()),second=observe(f.enqueue());await settle();
    assert.equal(typeof f.engine.callback,'function');assert.notEqual(f.engine.throws,null);
    f.box._preparingThrow=true;f.engine.callback=null;f.engine.throws=null;
    assert.doesNotThrow(()=>f.ticker.frame());await settle();
    assert.equal(first.value,false);assert.equal(second.value,false);assert.equal(f.engine.rolling,true);
    await f.queue.idle();
  });
  for(const reenable of [false,true])for(const stage of ['simulate','playback','effects','collisions','positions'])
    test('registered enabled setting '+(reenable?'off/on':'off')+' during '+stage+' continues both batches through native bridge teardown',async()=>{
      const f=setup(fixture);prepareBridge(f);f.install();const settings=prepareRegisteredSettings(f);
      assert.equal(settings.registered.get('enabled').scope,'world');await settings.setEnabled(true);
      assert.equal(settings.bridge.diagnose().capabilities.enabled,true);
      f.engine.persistentDiceList.push(f.other);f.other.userData.persistentId='legacy';let release;
      if(stage==='simulate')release=f.holdWorker(stage);
      const first=observe(f.enqueue()),second=observe(f.enqueue());await settle();
      if(stage!=='simulate'&&stage!=='playback'){
        if(stage==='effects'){release=f.holdEffects();f.engine.diceList.push({userData:{system:'standard'},specialEffects:[{}]});}
        else release=f.holdWorker(stage);
        await f.finish();
      }
      await settings.setEnabled(false);
      assert.equal(settings.bridge.diagnose().capabilities.enabled,false);assert.equal(settings.bridge.diagnose().capabilities.dsn,false);
      if(reenable){await settings.setEnabled(true);assert.equal(settings.bridge.diagnose().capabilities.dsn,true);}
      if(release)release();await settle();for(let i=0;i<3;i++)await f.finish();
      assert.equal(first.value,true);assert.equal(second.value,true);assert.equal(f.engine.rolling,false);
      assert.equal(f.engine.running,false);assert.equal(f.box._preparingThrow,false);assert.equal(f.errors.length,0);
      await f.queue.idle();f.engine.persistentDiceList.length=0;await settings.setEnabled(false);
    });
  async function run({patched=true,bridgeFirst=false,change,stage,reject=false}){
    const f=setup(fixture),pd=prepareBridge(f);
    if(patched&&!bridgeFirst)assert.equal(f.install().completionStatus,'installed');
    if(change==='dispose')assert.equal(await pd.ready(),true);
    if(patched&&bridgeFirst)assert.equal(f.install().completionStatus,'installed');
    f.engine.persistentDiceList.push(f.other);f.other.userData.persistentId='legacy';
    let release;
    if(stage==='simulate')release=f.holdWorker(stage);
    const first=observe(f.enqueue()),second=observe(f.enqueue());await settle();
    if(stage!=='simulate'&&stage!=='playback'){
      if(stage==='effects'){
        release=f.holdEffects();f.engine.diceList.push({userData:{system:'standard'},specialEffects:[{}]});
      }else release=f.holdWorker(stage);
      await f.finish();
    }
    assert.equal(first.status,'pending');assert.equal(second.status,'pending');
    if(change==='ready')assert.equal(await pd.ready(),true);else await pd.dispose();
    assert.doesNotThrow(()=>f.ticker.frame());
    if(release){if(reject)release.reject(Error('switched '+stage+' rejection'));else release();}
    await settle();
    for(let i=0;i<3;i++)await f.finish();
    assert.equal(first.status,'resolved');assert.equal(first.value,!reject);
    assert.equal(second.status,'resolved');assert.equal(second.value,true);
    assert.equal(f.engine.rolling,false);assert.equal(f.engine.running,false);assert.equal(f.box._preparingThrow,false);
    assert.equal(f.errors.length,reject?1:0);await f.queue.idle();
    f.engine.persistentDiceList.length=0;await pd.dispose();
  }
  for(const patched of [false,true])for(const change of ['ready','dispose'])
    for(const stage of ['simulate','playback','effects','collisions','positions'])
      test((patched?'recovery':'native')+' actual '+change+' during '+stage+' completes both queued batches',
        ()=>run({patched,change,stage}));
  for(const change of ['ready','dispose'])for(const stage of ['simulate','effects','collisions','positions'])
    test('actual '+change+' during '+stage+' rejection releases failure and permits the next batch',
      ()=>run({change,stage,reject:true}));
  for(const stage of ['simulate','playback','effects','collisions','positions'])
    test('bridge-first installation survives actual dispose during '+stage,
      ()=>run({bridgeFirst:true,change:'dispose',stage}));
  for(const stage of ['simulate','effects','collisions','positions'])
    test('bridge-first disposal during '+stage+' rejection permits the next batch',
      ()=>run({bridgeFirst:true,change:'dispose',stage,reject:true}));
  for(const initial of [false,true])test('in-flight setting off/on cycles preserve both batches from '+(initial?'enabled':'disabled'),async()=>{
    const f=setup(fixture);let pd=prepareBridge(f);f.install();
    if(initial)await pd.ready();
    const first=observe(f.enqueue()),second=observe(f.enqueue());await settle();
    for(let i=0;i<2;i++){
      if(!initial||i>0)await pd.ready();
      await pd.dispose();pd=prepareBridge(f);await pd.ready();
    }
    for(let i=0;i<3;i++)await f.finish();
    assert.equal(first.value,true);assert.equal(second.value,true);assert.equal(f.engine.rolling,false);
    assert.equal(f.engine.running,false);assert.equal(f.errors.length,0);await f.queue.idle();await pd.dispose();
  });
  for(const field of ['worker','exec','effects','engine'])test('unknown pre-completion '+field+' replacement settles captured waits without clearing foreign state',async()=>{
    const f=setup(fixture),pd=prepareBridge(f);f.install();await pd.ready();
    const first=observe(f.enqueue()),second=observe(f.enqueue());await settle();
    const foreign={rolling:true,running:true,callback(){throw Error('foreign callback');},throws:[],exec(){throw Error('foreign exec');}};
    if(field==='worker'){f.box.physicsWorker=foreign;f.engine.physicsWorker=foreign;}
    if(field==='exec')f.box.physicsWorker.exec=foreign.exec;
    if(field==='effects')f.engine.handleSpecialEffectsInit=function foreign(){};
    if(field==='engine')f.box.throwEngine=foreign;
    for(let i=0;i<3;i++)assert.doesNotThrow(()=>f.ticker.frame());
    await settle();assert.equal(first.value,false);assert.equal(second.value,false);
    assert.equal(foreign.rolling,true);assert.equal(foreign.running,true);assert.equal(f.engine.rolling,true);
    await f.queue.idle();await pd.dispose();
  });
  for(const bridgeFirst of [false,true])test('coherent dispose does not authorize an unrelated restored executor with '+(bridgeFirst?'bridge':'hotfix')+' installed first',async()=>{
    const f=setup(fixture),pd=prepareBridge(f);
    if(!bridgeFirst)f.install();await pd.ready();if(bridgeFirst)f.install();
    f.engine.persistentDiceList.push(f.other);f.other.userData.persistentId='legacy';
    const first=observe(f.enqueue()),second=observe(f.enqueue());await settle();
    const release=f.holdWorker('collisions');await f.finish();await pd.dispose();
    const foreignCalls=[];f.box.physicsWorker.exec=(...args)=>{foreignCalls.push(args);return Promise.resolve(true);};
    assert.doesNotThrow(()=>f.ticker.frame());release();await settle();
    assert.equal(first.value,false);assert.equal(second.value,false);assert.equal(f.engine.rolling,true);
    assert.deepEqual(foreignCalls,[]);await f.queue.idle();
  });
  test('bridge-first installation refuses an unknown original worker RPC method',async()=>{
    const f=setup(fixture),pd=prepareBridge(f);await pd.ready();
    Object.getPrototypeOf(f.box.physicsWorker).exec=function foreign(){};
    assert.equal(f.install().completionStatus,'unsupported-source');await pd.dispose();
  });
  for(const field of ['worker','callback'])test('synchronous '+field+' revocation inside the native frame settles the captured waits',async()=>{
    const f=setup(fixture),pd=prepareBridge(f);f.install();await pd.ready();
    const first=observe(f.enqueue()),second=observe(f.enqueue());await settle();const calls=[];
    const foreign={exec:(...args)=>{calls.push(args);return Promise.resolve(true);}};
    f.box.stats={update(){
      f.box.stats=null;
      if(field==='worker'){f.box.physicsWorker=foreign;f.engine.physicsWorker=foreign;}
      else{f.engine.callback=()=>calls.push('foreign callback');f.engine.throws=[];}
    }};
    await f.finish();
    for(let i=0;i<2;i++){assert.doesNotThrow(()=>f.ticker.frame());await settle();}
    assert.equal(first.value,false);assert.equal(second.value,false);assert.deepEqual(calls,[]);
    assert.equal(f.engine.rolling,true);await f.queue.idle();await pd.dispose();
  });
});
