import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';

const viewport = await import('../scripts/viewport.mjs').catch((error) => {
  if (error.code === 'ERR_MODULE_NOT_FOUND') return {};
  throw error;
});

// Preserve Foundry's public registry shape and callback identity semantics.
class HookRegistry {
  events = {};
  #nextId = 1;

  on(hook, fn, { once = false } = {}) {
    if (!(hook in this.events)) {
      Object.defineProperty(this.events, hook, { value: [], writable: false });
    }
    const id = this.#nextId++;
    this.events[hook].push({ hook, id, fn, once });
    return id;
  }

  off(hook, fnOrId) {
    const entries = this.events[hook];
    if (!entries) return;
    const index = entries.findIndex((entry) => typeof fnOrId === 'number'
      ? entry.id === fnOrId : entry.fn === fnOrId);
    if (index >= 0) entries.splice(index, 1);
  }

  callAll(hook, ...args) {
    for (const entry of [...(this.events[hook] ?? [])]) {
      if (entry.once) this.off(hook, entry.id);
      entry.fn(...args);
    }
  }
}

function fixture() {
  const hooks = new HookRegistry();
  const socket = new EventEmitter();
  const counts = { sends: 0, ordinaryPan: 0, incoming: 0, ready: 0, requests: 0 };
  const game = { modules: new Map([['obs-utils', { active: true, version: '5.3.0' }]]) };

  function makeSender() {
    const getSetting = () => 'smooth';
    const dragReleaseEmit = () => { counts.sends++; };
    const throttledEmit = () => { counts.sends++; };
    const smoothedEmit = () => { counts.sends++; };
    // This is the registered sender body shipped by OBS Utils 5.3.0.
    function socketCanvas(_canvas, position) {
      const mode = getSetting("cameraTrackingMode") ?? "smooth";
      if (mode === "dragRelease") dragReleaseEmit(position);
      else if (mode === "raw") throttledEmit(position);
      else smoothedEmit(position);
    }
    return socketCanvas;
  }

  const sender = makeSender();
  const ordinaryPan = () => { counts.ordinaryPan++; };
  hooks.on('canvasPan', ordinaryPan);
  hooks.on('canvasPan', sender);
  hooks.on('canvasReady', () => { counts.ready++; });
  socket.on('viewport', (position) => {
    counts.incoming++;
    // Incoming camera following continues to pan the local canvas.
    hooks.callAll('canvasPan', {}, position);
  });
  socket.on('requestViewport', () => { counts.requests++; });

  const pan = (times = 1) => {
    for (let index = 0; index < times; index++) {
      hooks.callAll('canvasPan', {}, { x: index, y: 20, scale: 1 });
    }
  };
  const suspend = (options = {}) => {
    assert.equal(typeof viewport.suspendRecorderViewportBroadcast, 'function',
      'suspendRecorderViewportBroadcast must be implemented');
    return viewport.suspendRecorderViewportBroadcast({ game, hooks, isActive: () => true, ...options });
  };
  return { game, hooks, socket, counts, sender, ordinaryPan, makeSender, pan, suspend };
}

test('recorder stops its 30 pan sends while other callbacks and inbound following continue', () => {
  const f = fixture();
  f.pan(30);
  assert.equal(f.counts.sends, 30);
  assert.equal(f.counts.ordinaryPan, 30);
  f.counts.sends = 0;
  f.counts.ordinaryPan = 0;

  const restore = f.suspend();
  assert.equal(typeof restore, 'function');
  f.pan(30);
  assert.equal(f.counts.sends, 0);
  assert.equal(f.counts.ordinaryPan, 30);
  f.socket.emit('viewport', { x: 40, y: 80, scale: 0.8 });
  assert.equal(f.counts.incoming, 1);
  assert.equal(f.counts.ordinaryPan, 31);
  assert.equal(f.counts.sends, 0);
  f.hooks.callAll('canvasReady');
  f.socket.emit('requestViewport');
  assert.equal(f.counts.ready, 1);
  assert.equal(f.counts.requests, 1);

  restore();
  restore();
  f.pan();
  assert.equal(f.counts.sends, 1);
  assert.equal(f.counts.ordinaryPan, 32);
  assert.equal(f.hooks.events.canvasPan.filter(({ fn }) => fn === f.sender).length, 1);
});

test('restore remains valid after the recorder stops being active', () => {
  const f = fixture();
  let active = true;
  const restore = f.suspend({ isActive: () => active });
  active = false;
  restore();
  f.pan();
  assert.equal(f.counts.sends, 1);
  assert.equal(f.counts.ordinaryPan, 1);
});

test('restore does not duplicate the original listener restored by another module', () => {
  const f = fixture();
  const restore = f.suspend();
  f.hooks.on('canvasPan', f.sender);
  restore();
  f.pan();
  assert.equal(f.counts.sends, 1);
  assert.equal(f.hooks.events.canvasPan.length, 2);
});

test('restore does not add a second sender after an equivalent sender was registered', () => {
  const f = fixture();
  const restore = f.suspend();
  f.hooks.on('canvasPan', f.makeSender());
  restore();
  f.pan();
  assert.equal(f.counts.sends, 1);
  assert.equal(f.hooks.events.canvasPan.length, 2);
});

test('an unknown module label with the audited sender still pauses recorder broadcasts', () => {
  for (const version of ['5.3.1', undefined]) {
    const f = fixture();
    f.game.modules.get('obs-utils').version = version;
    const restore = f.suspend();
    f.pan();
    assert.equal(f.counts.sends, 0);
    assert.equal(f.counts.ordinaryPan, 1);
    restore();
    f.pan();
    assert.equal(f.counts.sends, 1);
  }
});

test('non-recorder and unavailable modules keep all original callbacks', () => {
  for (const configure of [
    (f) => ({ isActive: () => false }),
    (f) => ({ isActive: undefined }),
    (f) => ({ isActive: () => { throw new Error('recorder unavailable'); } }),
    (f) => { f.game.modules.get('obs-utils').active = false; },
    (f) => { f.game.modules.delete('obs-utils'); },
    (f) => { delete f.game.modules; },
  ]) {
    const f = fixture();
    const original = [...f.hooks.events.canvasPan];
    const restore = f.suspend(configure(f));
    assert.equal(typeof restore, 'function');
    restore();
    assert.deepEqual(f.hooks.events.canvasPan, original);
    f.pan();
    assert.equal(f.counts.sends, 1);
    assert.equal(f.counts.ordinaryPan, 1);
  }
});

test('multiple senders fail closed without removing either', () => {
  const f = fixture();
  f.hooks.on('canvasPan', f.makeSender());
  const original = [...f.hooks.events.canvasPan];
  f.suspend()();
  assert.deepEqual(f.hooks.events.canvasPan, original);
  f.pan();
  assert.equal(f.counts.sends, 2);
  assert.equal(f.counts.ordinaryPan, 1);
});

test('the OBS Utils 5.3.1 sender, whose body changed, is still paused and restored', () => {
  const f = fixture();
  f.hooks.off('canvasPan', f.sender);
  const getLocalViewport = () => ({ x: 0, y: 0, scale: 1 });
  const getSetting = () => 'smooth';
  const dragReleaseEmit = () => { f.counts.sends++; };
  const throttledEmit = () => { f.counts.sends++; };
  const smoothedEmit = () => { f.counts.sends++; };
  // The registered sender body shipped by OBS Utils 5.3.1 (dist/socket-gYkh4oxK.js).
  function socketCanvas(_canvas, position) {
    const local = getLocalViewport();
    const x = position.x ?? local?.x;
    const y = position.y ?? local?.y;
    const scale = position.scale ?? local?.scale;
    if (typeof x !== "number" || typeof y !== "number" || typeof scale !== "number") return;
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(scale)) return;
    const viewport = {
      x,
      y,
      scale
    };
    const mode = getSetting("cameraTrackingMode") ?? "smooth";
    if (mode === "dragRelease") dragReleaseEmit(viewport);
    else if (mode === "raw") throttledEmit(viewport);
    else smoothedEmit(viewport);
  }
  f.hooks.on('canvasPan', socketCanvas);
  f.game.modules.get('obs-utils').version = '5.3.1';
  const restore = f.suspend();
  f.pan(30);
  assert.equal(f.counts.sends, 0);
  assert.equal(f.counts.ordinaryPan, 30);
  restore();
  f.pan();
  assert.equal(f.counts.sends, 1);
  assert.equal(f.hooks.events.canvasPan.filter(({ fn }) => fn === socketCanvas).length, 1);
});

test('listeners under another name are never removed, even when they send', () => {
  const f = fixture();
  f.hooks.off('canvasPan', f.sender);
  function socketCanvasInternal(...args) { f.sender(...args); }
  const bound = f.sender.bind(null);
  for (const listener of [socketCanvasInternal, bound, (...args) => f.sender(...args)]) f.hooks.on('canvasPan', listener);
  const original = [...f.hooks.events.canvasPan];
  f.suspend()();
  assert.deepEqual(f.hooks.events.canvasPan, original);
  f.pan();
  assert.equal(f.counts.sends, 3);
  assert.equal(f.counts.ordinaryPan, 1);
});

test('sender is recognized by its name, independent of hook order', () => {
  const f = fixture();
  f.hooks.off('canvasPan', f.sender);
  f.hooks.off('canvasPan', f.ordinaryPan);
  f.hooks.on('canvasPan', f.sender);
  f.hooks.on('canvasPan', f.ordinaryPan);
  const restore = f.suspend();
  f.pan();
  assert.equal(f.counts.sends, 0);
  assert.equal(f.counts.ordinaryPan, 1);
  restore();
  f.pan();
  assert.equal(f.counts.sends, 1);
  assert.equal(f.counts.ordinaryPan, 2);
});

test('unsupported hook registries and missing public methods are unchanged', () => {
  const f = fixture();
  for (const hooks of [
    undefined,
    { on() { throw new Error('must not register'); }, off() { throw new Error('must not unregister'); } },
    { events: { canvasPan: null }, on() {}, off() {} },
    { events: f.hooks.events, on() {} },
    { events: f.hooks.events, off() {} },
  ]) {
    const restore = f.suspend({ hooks });
    assert.equal(typeof restore, 'function');
    restore();
  }
  f.pan();
  assert.equal(f.counts.sends, 1);
  assert.equal(f.counts.ordinaryPan, 1);
});
