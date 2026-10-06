import test, {describe} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {setup as nativeSetup, settle, observe} from './dsn-queue-harness.mjs';
for(const name of ['dsn-queue-native.json','dsn-queue-6.4.3-native.json'])describe(name,()=>{
const fixture=JSON.parse(fs.readFileSync(new URL('./fixtures/'+name,import.meta.url)));
const setup=()=>nativeSetup(fixture);
for (const stage of ['collisions', 'positions', 'effects'])
  test(
    'native playback ' + stage + ' rejection resolves false and starts the next batch',
    async () => {
      const f = setup(),
        native = f.box.animateThrow;
      assert.equal(f.install().status, 'installed');
      f.engine.persistentDiceList.push(f.other);
      const first = observe(f.enqueue()),
        second = observe(f.enqueue());
      await settle();
      assert.equal(first.status, 'pending');
      assert.equal(f.box.animateThrow, native);
      assert.equal(f.ticks[0][0], native);
      if (stage === 'effects')
        f.engine.diceList.push({ userData: { system: 'standard' }, specialEffects: [{}] });
      f.fail(stage);
      await f.finish();
      assert.equal(first.status, 'resolved');
      assert.equal(first.value, false);
      assert.equal(second.status, 'pending');
      assert.equal(f.errors.length, 1);
      await f.finish();
      assert.equal(second.status, 'resolved');
      assert.equal(second.value, true);
      assert.equal(f.engine.rolling, false);
      await f.queue.idle();
    }
  );
test('persistent SFX rejection preserves affected binds and still restores collisions', async () => {
  const f = setup();
  f.install();
  f.engine.persistentDiceList.push(f.held, f.other);
  const state = observe(f.enqueue());
  await settle();
  f.held.persistentThrow = {};
  f.held.specialEffects = [{}];
  f.fail('effects');
  await f.finish();
  assert.equal(state.status, 'resolved');
  assert.equal(state.value, false);
  assert.deepEqual(f.landed, ['current']);
  assert.equal(f.held.userData.pendingBind, undefined);
  assert.equal(f.other.userData.pendingBind, 'other');
  assert.deepEqual(
    f.workers
      .filter(([name]) => name === 'setCollisionResponse')
      .map(([, args]) => [Array.from(args.ids), args.enabled]),
    [
      [[2], false],
      [[2], true]
    ]
  );
  await f.queue.idle();
});
test('successful native playback waits for effects and preserves the ticker function', async () => {
  const f = setup(),
    native = f.box.animateThrow;
  f.install();
  const release = f.holdEffects(),
    state = observe(f.enqueue());
  await settle();
  f.engine.diceList.push({ userData: { system: 'standard' }, specialEffects: [{}] });
  await f.finish();
  assert.equal(state.status, 'pending');
  assert.equal(f.engine.rolling, true);
  release();
  await settle();
  assert.equal(state.value, true);
  assert.equal(f.box.animateThrow, native);
  assert.equal(f.errors.length, 0);
  await f.queue.idle();
});
test('failed playback keeps exploding batches in order and resolves the roll false', async () => {
  const f = setup();
  f.install();
  f.engine.persistentDiceList.push(f.other);
  const state = observe(f.enqueue(3));
  await settle();
  f.fail('collisions');
  await f.finish();
  assert.equal(state.status, 'pending');
  await f.finish();
  assert.equal(state.status, 'pending');
  await f.finish();
  assert.equal(state.value, false);
  assert.equal(f.workers.filter(([name]) => name === 'simulateThrow').length, 3);
  await f.queue.idle();
});
test('external completion await keeps the native result and callback timing', async () => {
  const f = setup();
  f.install();
  const state = observe(f.enqueue());
  await settle();
  assert.deepEqual(Array.from(await f.engine.handlePersistentThrowCompletion()), []);
  assert.equal(state.status, 'pending');
  await f.finish();
  assert.equal(state.value, true);
  await f.queue.idle();
});
test('unknown completion sources retain native methods', () => {
  for (const field of [
    'animateThrow',
    'handlePersistentThrowCompletion',
    'handleSpecialEffectsInit'
  ]) {
    const f = setup(),
      target =
        field === 'animateThrow' ? Object.getPrototypeOf(f.box) : Object.getPrototypeOf(f.engine);
    target[field] = function foreign() {};
    const result=f.install();
    assert.equal(result.status,'installed');
    assert.equal(result.completionStatus,'unsupported-source');
    assert.equal(Object.hasOwn(f.engine, 'handlePersistentThrowCompletion'), false);
  }
});
test('restore removes only its own completion adapter and leaves a later replacement', () => {
  const f = setup(),
    native = f.engine.handlePersistentThrowCompletion,
    result = f.install();
  assert.notEqual(f.engine.handlePersistentThrowCompletion, native);
  result.restore();
  assert.equal(f.engine.handlePersistentThrowCompletion, native);
  const next = f.install(),
    foreign = function foreign() {};
  f.engine.handlePersistentThrowCompletion = foreign;
  next.restore();
  assert.equal(f.engine.handlePersistentThrowCompletion, foreign);
});

test('external catch observes the native rejection without failing its queue scope', async () => {
  const f = setup();
  f.install();
  const state = observe(f.enqueue());
  await settle();
  f.held.persistentThrow = {};
  f.held.specialEffects = [{}];
  f.engine.persistentDiceList.push(f.held);
  f.fail('effects');
  const promise = f.engine.handlePersistentThrowCompletion();
  assert.equal(Object.prototype.toString.call(promise), '[object Promise]');
  await assert.rejects(promise, /effects failed/);
  assert.equal(state.status, 'pending');
  assert.equal(f.errors.length, 0);
  await f.finish();
  assert.equal(state.value, true);
  await f.queue.idle();
});
for (const stage of [null, 'collisions'])
  test(
    'restore during native effects still completes the captured batch' +
      (stage ? ' after cleanup failure' : ''),
    async () => {
      const f = setup(),
        patch = f.install(),
        release = f.holdEffects(),
        state = observe(f.enqueue());
      f.engine.persistentDiceList.push(f.other);
      await settle();
      f.engine.diceList.push({ userData: { system: 'standard' }, specialEffects: [{}] });
      await f.finish();
      assert.equal(state.status, 'pending');
      patch.restore();
      if (stage) f.fail(stage);
      release();
      await settle();
      assert.equal(state.value, !stage);
      assert.equal(f.engine.rolling, false);
      assert.equal(f.errors.length, stage ? 1 : 0);
      assert.equal(Object.hasOwn(f.engine, 'handlePersistentThrowCompletion'), false);
      await f.queue.idle();
    }
  );
for (const replacement of ['box', 'engine'])
  test(
    'effects finishing after ' + replacement + ' replacement settles only the captured batch',
    async () => {
      const f = setup();
      f.install();
      const release = f.holdEffects(),
        state = observe(f.enqueue());
      await settle();
      f.engine.diceList.push({ userData: { system: 'standard' }, specialEffects: [{}] });
      await f.finish();
      const other = {
        rolling: true,
        callback() {
          throw Error('new callback must not run');
        }
      };
      if (replacement === 'box') f.queue.box = { throwEngine: other };
      else f.box.throwEngine = other;
      release();
      await settle();
      assert.equal(state.value, false);
      assert.equal(other.rolling, true);
      assert.equal(f.engine.rolling, true);
      assert.equal(f.workers.filter(([name]) => name === 'setBodyPositions').length, 0);
      await f.queue.idle();
    }
  );
test('late effects rejection after the original callback leaves the next batch pending', async () => {
  const f = setup();
  f.install();
  const release = f.holdEffects(),
    first = observe(f.enqueue()),
    second = observe(f.enqueue());
  await settle();
  f.engine.diceList.push({ userData: { system: 'standard' }, specialEffects: [{}] });
  await f.finish();
  f.engine.rolling = false;
  f.engine.callback(f.engine.throws);
  await settle();
  assert.equal(first.value, true);
  assert.equal(second.status, 'pending');
  assert.equal(f.engine.rolling, true);
  release.reject(Error('late effects failure'));
  await settle();
  assert.equal(second.status, 'pending');
  assert.equal(f.engine.rolling, true);
  assert.equal(f.errors.length, 1);
  await f.finish();
  assert.equal(second.value, true);
  await f.queue.idle();
});
test('changed native completion source bypasses the adapter for later calls', async () => {
  const f = setup();
  f.install();
  const state = observe(f.enqueue());
  await settle();
  const proto = Object.getPrototypeOf(f.box),
    native = proto.animateThrow;
  proto.animateThrow = function foreign() {};
  const promise = f.engine.handlePersistentThrowCompletion();
  assert.equal(Object.hasOwn(promise, 'then'), false);
  await promise;
  proto.animateThrow = native;
  await f.finish();
  assert.equal(state.value, true);
  await f.queue.idle();
});

test('native completion failure reveals its chat message and keeps the following chat queued', async () => {
  const f = setup();
  f.install();
  f.engine.persistentDiceList.push(f.other);
  const native = JSON.parse(
      fs.readFileSync(new URL('./fixtures/dsn-chat-native.json', import.meta.url))
    ),
    events = [],
    user = { id: 'player' },
    nodes = new Map();
  const message = (id) => {
    const classes = new Set(['dsn-hide']);
    nodes.set(id, { classList: { remove: (value) => classes.delete(value) }, classes });
    return {
      id,
      author: user,
      speaker: {},
      whisper: [],
      isContentVisible: true,
      _dice3dMessageHidden: true,
      _dice3dPendingRenders: 1,
      _dice3danimating: true
    };
  };
  const first = message('first'),
    second = message('second');
  Object.assign(f.game, {
    view: 'game',
    user,
    modules: new Map(),
    messages: new Map([
      [first.id, first],
      [second.id, second]
    ]),
    actors: new Map(),
    users: []
  });
  const ui = {
      chat: {
        element: {
          querySelector: (selector) => nodes.get(selector.includes('first') ? 'first' : 'second')
        },
        _shouldShowNotifications: () => false,
        scrollBottom() {}
      },
      sidebar: { popouts: {} }
    },
    merge = f.context.DiceNotation.mergeQueuedRollCommands;
  Object.assign(f.context, {
    ui,
    window: { ui, document: { hidden: false } },
    document: { querySelector: () => null },
    Hooks: { callAll: (...args) => events.push(args) },
    InitiativeMask: { release() {} },
    CompanionLink: { release: () => [] },
    ChatMessage: { getSpeakerActor: () => null },
    CONST: { DOCUMENT_OWNERSHIP_LEVELS: { OWNER: 3 } },
    DiceNotation: class {
      constructor() {
        this.throws = [{ dice: [] }];
      }
      static mergeQueuedRollCommands = merge;
    }
  });
  Object.assign(f.context.DsnSettings, {
    CONFIG: () => ({ visibility: 'all' }),
    ALL_CONFIG: () => ({}),
    ALL_CUSTOMIZATION: () => ({})
  });
  const Native = vm.runInContext(
      '(class Native {' + Object.values(native.methods).join('\n') + '})',
      f.context
    ),
    pipeline = new Native();
  Object.assign(pipeline, {
    queue: f.queue,
    _assignDependentRollOrder() {},
    _stampRole() {},
    _stampAppearance() {},
    _buildRollList: (rolls) => rolls,
    _showNestedParts() {},
    pendingThrows: { isPending: () => false }
  });
  const roll = { dice: [{}], total: 12 };
  pipeline.renderRolls(first, [roll]);
  pipeline.renderRolls(second, [roll]);
  await settle();
  f.fail('collisions');
  await f.finish();
  assert.equal(first._dice3dPendingRenders, 0);
  assert.equal(nodes.get('first').classes.has('dsn-hide'), false);
  assert.equal(second._dice3dPendingRenders, 1);
  assert.equal(nodes.get('second').classes.has('dsn-hide'), true);
  await f.finish();
  assert.equal(second._dice3dPendingRenders, 0);
  assert.equal(nodes.get('second').classes.has('dsn-hide'), false);
  assert.deepEqual(
    events.filter(([name]) => name === 'diceSoNiceRollComplete').map(([, id]) => id),
    ['first', 'second']
  );
  assert.equal(f.errors.length, 1);
  await f.queue.idle();
});

test('unknown observers of the first native consumer retain its rejection', async () => {
  const f = setup();
  f.install();
  const state = observe(f.enqueue());
  await settle();
  f.engine.diceList.push({ userData: { system: 'standard' }, specialEffects: [{}] });
  f.fail('effects');
  const Observer = vm.runInContext(
      '(class Observer {consume(p){return p.then(()=>this.throwEngine.handleSpecialEffectsInit());}})',
      f.context
    ),
    observer = new Observer();
  observer.throwEngine = f.engine;
  await assert.rejects(
    observer.consume(f.engine.handlePersistentThrowCompletion()),
    /effects failed/
  );
  assert.equal(f.errors.length, 0);
  assert.equal(state.status, 'pending');
  await f.finish();
  assert.equal(state.value, true);
  await f.queue.idle();
});

});
