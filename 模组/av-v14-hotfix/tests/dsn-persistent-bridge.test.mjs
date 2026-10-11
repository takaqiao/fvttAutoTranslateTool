import test,{describe} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {createHash} from 'node:crypto';
import {installDsnChatRecovery} from '../scripts/patches/dsn-chat.mjs';
import {setup,settle,observe} from './dsn-queue-harness.mjs';
import {prepareBridge,bridgeFixture} from './dsn-persistent-bridge-harness.mjs';
const sha=fn=>createHash('sha256').update(fn.toString()).digest('hex');
test('bridge fixture is the independently captured 0.5.4 adapter',()=>{
  assert.equal(bridgeFixture.provenance.version,'0.5.4');
  assert.equal(createHash('sha256').update(bridgeFixture.source).digest('hex'),bridgeFixture.provenance.sha256);
  assert.equal(bridgeFixture.provenance.sha256,'82f6035a87e0f4269732a798ed88cb622ba8f036b3d40283bcfc041f9de9f0d8');
});
for(const name of ['dsn-queue-native.json','dsn-queue-6.4.3-native.json','dsn-queue-6.4.4-native.json'])describe(name,()=>{
  const fixture=JSON.parse(fs.readFileSync(new URL('./fixtures/'+name,import.meta.url)));
  test('a batch-local worker rejection injection restores startup state with the installed bridge',async()=>{
    const f=setup(fixture),pd=prepareBridge(f);f.install();await pd.ready();
    f.game.modules.get('pf2e-dsn-persistent-bridge').version='0.5.99';
    f.engine.persistentDiceList.push(f.other);
    const worker=f.box.physicsWorker,exec=worker.exec;let rejected=false;
    worker.exec=function(name,...args){
      if(name==='simulateThrow'&&!rejected){rejected=true;return Promise.reject(Error('injected simulation rejection'));}
      return exec.call(this,name,...args);
    };
    const first=observe(f.enqueue());await settle();
    assert.equal(first.value,false);assert.equal(f.engine.rolling,false);assert.equal(f.box._preparingThrow,false);
    assert.equal(f.errors.length,1);assert.deepEqual(Array.from(f.engine._ghostifiedIds),[]);
    worker.exec=exec;
    const second=observe(f.enqueue());await settle();await f.finish();assert.equal(second.value,true);await f.queue.idle();
    f.engine.persistentDiceList.length=0;await pd.dispose();
  });
  test('external observers of the actual bridge completion keep the native rejection',async()=>{
    const f=setup(fixture),pd=prepareBridge(f);f.install();await pd.ready();
    const state=observe(f.enqueue());await settle();f.held.persistentThrow={};f.held.specialEffects=[{}];
    f.engine.persistentDiceList.push(f.held);f.fail('effects');
    await assert.rejects(f.engine.handlePersistentThrowCompletion(),/effects failed/);
    assert.equal(state.status,'pending');assert.equal(f.errors.length,0);
    await f.finish();assert.equal(state.value,true);await f.queue.idle();
    f.engine.persistentDiceList.length=0;await pd.dispose();
  });
  for(const order of ['recovery-first','bridge-first'])test('native attach after quality rebuild keeps installed completion with '+order,async()=>{
    const f=setup(fixture),pd=prepareBridge(f),chat=JSON.parse(fs.readFileSync(new URL('./fixtures/dsn-chat-native.json',import.meta.url)));
    const Pipeline=vm.runInContext('(class Pipeline{'+Object.values(chat.methods).join('\n')+'})',f.context);
    const pipeline=new Pipeline();pipeline.queue=f.queue;f.game.dice3d.pipeline=pipeline;
    f.game.modules.set('dice-so-nice',{active:true,version:fixture.provenance.version});
    f.game.version='14.368';f.game.release={generation:14};
    const g={game:f.game,canvas:f.context.canvas,console:{error(){}}},result=installDsnChatRecovery({g});
    assert.equal(result.queueCompletionStatus,'installed');assert.equal(await pd.ready(),true);
    f.queue.attach(f.box);assert.equal(result.queueCompletionStatus,'installed');
    const next=setup(fixture);next.context.canvas.app.ticker=f.ticker;next.context.Utils=f.context.Utils;
    prepareBridge(next); // Supply the real adapter's prerequisites on the new box.
    f.game.dice3d.box=next.box;
    if(order==='recovery-first')f.queue.attach(next.box);
    await pd.ready();
    if(order==='bridge-first')f.queue.attach(next.box);
    assert.equal(result.queueStatus,'installed');assert.equal(result.queueCompletionStatus,'installed');
    next.fail('simulate');const first=observe(f.enqueue());await settle();
    assert.equal(first.value,false);assert.equal(next.engine.rolling,false);assert.equal(next.box._preparingThrow,false);
    next.engine.persistentDiceList.push(next.other);next.fail('collisions');
    const second=observe(f.enqueue());await settle();await next.finish();
    assert.equal(second.value,false);await f.queue.idle();
    const third=observe(f.enqueue());await settle();await next.finish();assert.equal(third.value,true);await f.queue.idle();
    assert.equal(result.queueCompletionStatus,'installed');assert.equal(result.stats.recovered,2);
    next.engine.persistentDiceList.length=0;result.restore();await pd.dispose();
  });
  for(const changed of ['inactive','incomplete'])test('installation refuses '+changed+' bridge wrapper profile',async()=>{
    const f=setup(fixture),pd=prepareBridge(f);await pd.ready();
    if(changed==='inactive')f.game.modules.get('pf2e-dsn-persistent-bridge').active=false;
    else delete f.box.clearScene;
    assert.equal(f.install().completionStatus,'unsupported-source');await pd.dispose();
  });
  for(const order of ['hotfix-first','bridge-first'])for(const stage of ['effects','collisions','positions'])
    test(order+' actual bridge preserves failure recovery at '+stage,async()=>{
      const f=setup(fixture),pd=prepareBridge(f);
      if(order==='hotfix-first')assert.equal(f.install().completionStatus,'installed');
      assert.equal(await pd.ready(),true);
      const refs=[f.box.spawnPersistentDie,f.box.clearScene,f.engine.handlePersistentThrowCompletion,f.box.physicsWorker.exec];
      assert.deepEqual(refs.map(sha),['d5e1bead9c55dd00887bb905b2db0b0395ea785ea46b84c307656294d6d7492f',
        '811288073cc835a47f1f9e8000d3c18a2078941da39b5c1dc77894a02b3f20be',
        'da8629905de5cce5f1c194baebc8f261cf3b7121b3909ce79698c1ba3ed5f162',
        'f5d4ddf7ec2834991653d054dfb962f65498d57af099e9c52823e309244e3afd']);
      if(order==='bridge-first')assert.equal(f.install().completionStatus,'installed');
      f.engine.persistentDiceList.push(f.other);
      const first=observe(f.enqueue()),second=observe(f.enqueue());await settle();
      if(stage==='effects')f.engine.diceList.push({userData:{system:'standard'},specialEffects:[{}]});
      f.fail(stage);await f.finish();
      assert.equal(first.value,false);assert.equal(second.status,'pending');assert.equal(f.errors.length,1);
      await f.finish();assert.equal(second.value,true);await f.queue.idle();
      assert.deepEqual([f.box.spawnPersistentDie,f.box.clearScene,f.engine.handlePersistentThrowCompletion,f.box.physicsWorker.exec],refs);
      f.engine.persistentDiceList.length=0;await pd.dispose();
      assert.equal(Object.hasOwn(f.box,'spawnPersistentDie'),false);assert.equal(Object.hasOwn(f.box,'clearScene'),false);
    });
  test('installed bridge survives repeated same-box recovery reinstall and its own dispose',async()=>{
    const f=setup(fixture),pd=prepareBridge(f),native=f.engine.handlePersistentThrowCompletion;
    let patch=f.install();assert.equal(await pd.ready(),true);
    const refs=[f.box.spawnPersistentDie,f.box.clearScene,f.engine.handlePersistentThrowCompletion,f.box.physicsWorker.exec];
    for(let i=0;i<3;i++){
      patch.restore();patch=f.install();assert.equal(patch.completionStatus,'installed');
      assert.deepEqual([f.box.spawnPersistentDie,f.box.clearScene,f.engine.handlePersistentThrowCompletion,f.box.physicsWorker.exec],refs);
      f.fail('collisions');f.engine.persistentDiceList.push(f.other);
      const state=observe(f.enqueue());await settle();await f.finish();
      assert.equal(state.value,false);await f.queue.idle();f.engine.persistentDiceList.length=0;
    }
    patch.restore();await pd.dispose();
    const next=f.install();assert.equal(next.completionStatus,'installed');next.restore();
    assert.equal(f.engine.handlePersistentThrowCompletion,native);
  });
  for(const stage of ['collisions','positions'])test('bridge fade registration retains async '+stage+' owner checks',async()=>{
    const f=setup(fixture),pd=prepareBridge(f);f.install();await pd.ready();
    f.engine.persistentDiceList.push(f.other);f.other.userData.persistentId='older';
    f.box._startMeshFade=()=>{};f.box.persistentDiceManager.removePersistentDie=async()=>true;
    const state=observe(f.enqueue());await settle();await f.box.fadeOutPersistentDie('older',1000);
    const release=f.holdWorker(stage);await f.finish();
    let foreignCallbacks=0;
    const foreign=Object.assign(Object.create(Object.getPrototypeOf(f.engine)),f.engine,
      {rolling:true,throws:[],_ghostifiedIds:[77],callback(){foreignCallbacks++;}});
    f.box.throwEngine=foreign;assert.doesNotThrow(()=>f.ticker.frame());release();await settle();
    assert.equal(state.value,false);assert.equal(foreignCallbacks,0);assert.equal(foreign.rolling,true);
    assert.deepEqual(foreign._ghostifiedIds,[77]);await f.queue.idle();
    f.engine.persistentDiceList.length=0;await pd.dispose();
  });
  test('bridge startup rejection never restores old collisions through a replacement worker',async()=>{
    const f=setup(fixture),pd=prepareBridge(f);f.install();await pd.ready();f.engine.persistentDiceList.push(f.other);
    const release=f.holdWorker('simulate'),state=observe(f.enqueue());await settle();
    const calls=[],foreign={exec:(...args)=>{calls.push(args);return Promise.resolve();}};
    f.box.physicsWorker=foreign;f.engine.physicsWorker=foreign;release.reject(Error('old worker failed'));await settle();
    assert.equal(state.value,false);assert.deepEqual(calls,[]);assert.equal(f.engine.rolling,true);
    await f.queue.idle();f.engine.persistentDiceList.length=0;await pd.dispose();
  });
  for(const field of ['spawnPersistentDie','clearScene','handlePersistentThrowCompletion'])
    test('unknown replacement of installed bridge '+field+' revokes pending cleanup',async()=>{
      const f=setup(fixture),pd=prepareBridge(f);f.install();await pd.ready();f.engine.persistentDiceList.push(f.other);
      const state=observe(f.enqueue());await settle();const release=f.holdWorker('collisions');await f.finish();
      (field==='handlePersistentThrowCompletion'?f.engine:f.box)[field]=function foreign(){};
      assert.doesNotThrow(()=>f.ticker.frame());release();await settle();
      assert.equal(state.value,false);assert.equal(f.engine.rolling,true);await f.queue.idle();
      f.engine.persistentDiceList.length=0;await pd.dispose();
    });
});
