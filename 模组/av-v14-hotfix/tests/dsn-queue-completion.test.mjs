import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import { installDsnQueueRecovery } from '../scripts/patches/dsn-queue.mjs';
const fixture = JSON.parse(
  fs.readFileSync(new URL('./fixtures/dsn-queue-native.json', import.meta.url))
);
const settle = () => new Promise((resolve) => setImmediate(resolve));
const observe = (promise) => {
  const state = { status: 'pending' };
  Promise.resolve(promise).then(
    (value) => Object.assign(state, { status: 'resolved', value }),
    (error) => Object.assign(state, { status: 'rejected', error })
  );
  return state;
};
function setup() {
  const errors = [],
    workers = [],
    ticks = [],
    landed = [];
  let failure = null,
    release;
  const game = {
    settings: { get: (_module, key) => (key === 'maxDiceNumber' ? 20 : false) },
    dice3d: { pendingThrows: { noteBindsLanded: (binds) => landed.push(...binds) } }
  };
  const bo = {
    renderQueue: [],
    renderSFX() {},
    async playSFX() {
      if (failure === 'effects') {
        failure = null;
        throw Error('effects failed');
      }
      if (release) await release.promise;
    }
  };
  const context = vm.createContext({
    game,
    bo,
    r: { DICE_EVENT_TYPE: { RESULT: 1 } },
    setTimeout,
    clearTimeout,
    DsnSettings: { isEnabled: () => true },
    DiceNotation: {
      mergeQueuedRollCommands: (items) =>
        items.flatMap((item) => item.params.throws.map((throwData) => [throwData]))
    },
    Utils: { removeTicker() {} },
    canvas: {
      app: {
        ticker: {
          add(fn, box) {
            ticks.push([fn, box]);
          }
        }
      }
    }
  });
  const classes = vm.runInContext(
    '(()=>{' +
      fixture.accumulator +
      ';' +
      fixture.queue +
      ';' +
      fixture.boxClass +
      ';' +
      fixture.engineClass +
      ';return {AnimationQueue,DiceBox,ThrowEngine};})()',
    context
  );
  const worker = {
    async exec(name, args) {
      workers.push([name, args]);
      if (
        (name === 'setCollisionResponse' && args.enabled && failure === 'collisions') ||
        (name === 'setBodyPositions' && failure === 'positions')
      ) {
        failure = null;
        throw Error(name + ' failed');
      }
      if (name === 'simulateThrow')
        return {
          ids: [],
          quaternionsBuffers: [],
          positionsBuffers: [],
          detectedCollides: [],
          deads: [],
          iterationsNeeded: 0,
          faceValues: {},
          finalQuaternions: {}
        };
      return true;
    }
  };
  const scene = { display: { innerWidth: 1000, innerHeight: 800 }, animatedDiceDetected: false },
    factory = { systems: new Map([['standard', { fire() {} }]]) };
  const engine = new classes.ThrowEngine(scene, worker, factory, {
    generateCollisionSounds: () => []
  });
  const box = Object.assign(Object.create(classes.DiceBox.prototype), {
    throwEngine: engine,
    physicsWorker: worker,
    diceScene: scene,
    dicefactory: factory,
    fadingDice: [],
    inputHandler: { clearPendingThrowDice() {}, updatePreRoll() {} },
    persistentDiceManager: {
      persistentDiceList: engine.persistentDiceList,
      updateRemoteAnimations() {}
    },
    renderScene() {}
  });
  const queue = new classes.AnimationQueue({
    canvasVisibility: { show() {}, hide() {} },
    pendingThrows: { noteBindsLanded: (binds) => landed.push(...binds) }
  });
  queue.attach(box);
  const install = () => installDsnQueueRecovery({ queue, recover: (error) => errors.push(error) });
  const enqueue = (count = 1) =>
    queue.enqueue(
      { throws: Array.from({ length: count }, () => ({ dice: [], dsnConfig: {} })) },
      {}
    );
  const finish = async () => {
    engine.iteration = 2;
    engine.minIterations = 0;
    engine.iterationsNeeded = 1;
    box.last_time = Date.now();
    box.isVisible = false;
    box.animateThrow();
    await settle();
  };
  const held = { id: 1, userData: { pendingBind: 'current' }, traverse() {} },
    other = {
      id: 2,
      userData: { constrained: true, pendingBind: 'other' },
      traverse() {},
      parent: { position: { x: 1, y: 2, z: 3 } }
    };
  return {
    context,
    game,
    queue,
    engine,
    box,
    install,
    enqueue,
    finish,
    errors,
    workers,
    ticks,
    landed,
    held,
    other,
    fail: (value) => {
      failure = value;
    },
    holdEffects() {
      let resolve, reject;
      const promise = new Promise((r, j) => {
        resolve = r;
        reject = j;
      });
      release = { promise };
      return Object.assign(resolve, { reject });
    }
  };
}
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
