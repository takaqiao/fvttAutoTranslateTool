import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { parseHTML } from 'linkedom';

const runtime = await import('../scripts/runtime.mjs').catch((error) => {
  if (error.code === 'ERR_MODULE_NOT_FOUND') return {};
  throw error;
});
const entry = await import('../main.mjs').catch((error) => {
  if (error.code === 'ERR_MODULE_NOT_FOUND') return {};
  throw error;
});
function deferred() {
  let resolve;
  const promise = new Promise((done) => { resolve = done; });
  return { promise, resolve };
}
const tick = () => new Promise((resolve) => setImmediate(resolve));
function actor(id, pending) {
  return {
    id, type: 'character', name: `Hero ${id}`, img: `${id}.webp`,
    attributes: { hp: { value: 0, max: 20, temp: 8 }, ac: { value: 24 } },
    system: { resources: { focus: { value: 1, max: 3 } } },
    testUserPermission: () => true,
    spellcasting: { collections: new Map(pending ? [['spells', { entry: { getSheetData: () => pending.promise } }]] : []) },
  };
}
function fixture() {
  const { document } = parseHTML('<html><body><canvas id="board" style="color:red"></canvas><div id="interface"><div id="camera-views"><video></video><audio id="audio"></audio></div></div></body></html>');
  const hooks = new EventEmitter();
  hooks.off = hooks.removeListener.bind(hooks);
  const a = actor('a'), b = actor('b');
  const users = [{ id: 'obs', name: 'Recorder', isGM: false }, { id: 'gm', name: 'GM', isGM: true, active: true }, { id: 'one', name: 'One', isGM: false, active: true, character: a }, { id: 'two', name: 'Two', isGM: false, active: true, character: b }];
  users.get = (id) => users.find((user) => user.id === id);
  const settings = new Map([['enabled', true], ['recorderUserId', 'obs'], ['seatOrder', ['gm', 'one', 'two']], ['portraitOverrides', { a: 'original-a.webp' }], ['skin', 'auto'], ['manualMode', 'auto']]);
  const game = {
    ready: true, view: 'game', system: { id: 'pf2e' }, world: { id: 'cotct' }, user: users[0], users,
    settings: { get: (namespace, key) => namespace === 'obs-utils' ? 'obs' : settings.get(key), register: (namespace, key, definition) => { if (!settings.has(key)) settings.set(key, definition.default); } },
    combat: { started: true, combatant: { actorId: 'a', actor: a, name: 'Public A', img: 'public-a.webp', visible: true } },
    webrtc: { client: {} },
  };
  const renderer = { width: 1920, height: 1080, resize(w, h) { this.width = w; this.height = h; } };
  const canvas = { ready: true, app: { renderer }, screenDimensions: [1920, 1080], stage: { pivot: { x: 321, y: 456 }, position: { set(x, y) { this.x = x; this.y = y; } } }, pans: [], pan(point) { this.pans.push({ ...point }); }, _onResize() { renderer.resize(1920, 1080); this.screenDimensions = [1920, 1080]; this.stage.position.set(960, 540); this.pan(this.stage.pivot); } };
  let wrapper;
  const libWrapper = { register(_id, target, handler, type) { assert.equal(target, 'foundry.canvas.Canvas.prototype._onResize'); assert.equal(type, 'WRAPPER'); wrapper = handler; }, unregister(_id, target) { assert.equal(target, 'foundry.canvas.Canvas.prototype._onResize'); wrapper = null; } };
  const start = (options = {}) => {
    assert.equal(typeof runtime.startDirector, 'function', 'startDirector must be implemented');
    return runtime.startDirector({ game, canvas, ui: {}, hooks, document, libWrapper, ...options });
  };
  const root = () => document.getElementById('taka-obs-director');
  return { document, hooks, a, b, users, game, settings, canvas, renderer, libWrapper, start, root, resize: () => wrapper.call(canvas, canvas._onResize.bind(canvas)) };
}
function roomFixture(f, participants = []) {
  const room = new EventEmitter();
  room.state = 'connected';
  room.remoteParticipants = new Map();
  const map = new Map(participants);
  f.game.webrtc.client._liveKitClient = { liveKitRoom: room, liveKitParticipants: map };
  return { room, map };
}
function participant(speaking = false, metadata) {
  const p = new EventEmitter();
  p.isSpeaking = speaking;
  p.identity = 'opaque-livekit-id';
  p.metadata = metadata;
  return p;
}
const speakers = (root) => [...root.querySelectorAll('.speaking')].map((node) => node.dataset.userId).sort();

test('focus and cast use approved original art without touching token visuals', async () => {
  const f = fixture();
  const token = { hidden: false };
  for (const key of ['texture', 'ring', 'object']) Object.defineProperty(token, key, { get() { throw new Error('Token visual access'); } });
  f.game.combat.combatant.token = token;
  const stop = f.start();
  try {
    await tick();
    assert.equal(f.root().querySelector('.focus-portrait').getAttribute('src'), 'original-a.webp');
    assert.equal(f.root().querySelector('[data-user-id="one"] .figure').getAttribute('src'), 'original-a.webp');
    f.hooks.emit('updateToken', token, { x: 200, rotation: 90 }); await tick();
    assert.equal(f.root().querySelector('.focus-portrait').getAttribute('src'), 'original-a.webp');
    token.hidden = true; f.hooks.emit('updateToken', token, { hidden: true }); await tick();
    assert.equal(f.root().querySelector('.focus-portrait'), null);
  } finally { stop(); }
});

test('portrait calibration repaints only the selected original and rejects changed image settings', async () => {
  const f = fixture(), counter = countSheets([f.a, f.b]);
  f.settings.set('portraitLayouts', { a: { source: 'original-a.webp', x: -7, y: 3, scale: 1.1 }, b: { source: 'b.webp', x: 20, y: 0, scale: 1 } });
  const stop = f.start();
  try {
    await tick(); counter.reset();
    assert.equal(f.root().querySelector('.focus-portrait').style.getPropertyValue('--portrait-shift-x'), '-7%');
    f.settings.get('portraitLayouts').a.x = -9;
    f.hooks.emit('updateSetting', { key: `${runtime.MODULE_ID}.portraitLayouts` }); await tick();
    assert.equal(f.root().querySelector('.focus-portrait').style.getPropertyValue('--portrait-shift-x'), '-9%');
    assert.equal(counter.calls.get(f.a), 0); assert.equal(counter.calls.get(f.b), 0);
    assert.equal(f.root().querySelector('[data-user-id="one"] .figure').style.getPropertyValue('--portrait-shift-x'), '');
    f.settings.set('portraitOverrides', { a: 'replacement.webp' });
    f.hooks.emit('updateSetting', { key: `${runtime.MODULE_ID}.portraitOverrides` }); await tick();
    const image = f.root().querySelector('.focus-portrait');
    assert.equal(image.getAttribute('src'), 'replacement.webp');
    assert.equal(image.style.getPropertyValue('--portrait-shift-x'), '0%');
    assert.equal(image.style.getPropertyValue('--portrait-scale'), '1');
  } finally { stop(); }
});

test('the mounted recorder applies portable native preferences and restores them when disabled', async () => {
  const f = fixture();
  const values = new Map([['core.maxFPS', 60], ['pf2e-hud.tracker.enabled', true]]);
  const read = f.game.settings.get;
  f.game.settings.settings = new Map([['core.maxFPS', { scope: 'client' }], ['pf2e-hud.tracker.enabled', { scope: 'user' }]]);
  f.game.settings.get = (namespace, key) => values.has(`${namespace}.${key}`) ? values.get(`${namespace}.${key}`) : read(namespace, key);
  f.game.settings.set = async (namespace, key, value) => { values.set(`${namespace}.${key}`, value); };
  f.game.modules = new Map([['pf2e-hud', { active: true }]]);
  const av = { disableVideo: false, muteAll: false };
  f.game.webrtc.settings = { get: (_scope, key) => av[key], set: (_scope, key, value) => { av[key] = value; } };
  const stop = f.start(); await tick();
  assert.equal(av.disableVideo, true);
  assert.equal(av.muteAll, false);
  assert.equal(values.get('core.maxFPS'), 30);
  assert.equal(values.get('pf2e-hud.tracker.enabled'), false);
  f.settings.set('enabled', false); f.hooks.emit('updateSetting', { key: `${runtime.MODULE_ID}.enabled` });
  await tick();
  assert.equal(f.root(), null);
  assert.equal(av.disableVideo, false);
  assert.equal(values.get('core.maxFPS'), 60);
  assert.equal(values.get('pf2e-hud.tracker.enabled'), true);
  stop();
});

function countSheets(actors, sources = 1) {
  const calls = new Map(actors.map(actor => [actor, 0]));
  for (const actor of actors) actor.spellcasting.collections = new Map(Array.from({ length: sources }, (_, index) => [String(index), { entry: { getSheetData() {
    calls.set(actor, calls.get(actor) + 1);
    return Promise.resolve({ id: `${actor.id}-${index}`, name: `${actor.id} spells`, statistic: { dc: { value: 20 } }, groups: [] });
  } } }]));
  return { calls, reset() { for (const actor of actors) calls.set(actor, 0); } };
}

test('unrelated documents and coordinate/combat changes keep actor snapshots without collecting spells', async () => {
  const f = fixture(), counter = countSheets([f.a, f.b]), stop = f.start();
  try {
    await tick(); counter.reset();
    const outsider = actor('outside');
    f.hooks.emit('updateActor', outsider, { 'system.attributes.hp.value': 8 });
    f.hooks.emit('updateItem', { parent: outsider }, { 'system.uses.value': 1 });
    f.hooks.emit('updateToken', { actorId: 'outside' }, { x: 100, y: 200 });
    f.hooks.emit('updateToken', { actorId: 'a' }, { x: 300, y: 400 });
    f.game.combat.combatant = { actorId: 'b', actor: f.b, name: 'Public B', visible: true };
    f.hooks.emit('updateCombat', f.game.combat, { turn: 1 });
    await tick();
    assert.deepEqual([...counter.calls.values()], [0, 0]);
    assert.equal(f.root().querySelector('.focus-name').textContent, 'Hero b');
    assert.equal(f.root().querySelector('.focus-hp .hp-value').textContent, '0/20');
  } finally { stop(); }
});

test('an actor update recollects that actor and preserves other authorized HP while pending', async () => {
  const f = fixture(), counter = countSheets([f.a, f.b]), stop = f.start();
  try {
    await tick(); counter.reset();
    const pending = deferred();
    f.a.spellcasting.collections.get('0').entry.getSheetData = () => { counter.calls.set(f.a, counter.calls.get(f.a) + 1); return pending.promise; };
    f.a.attributes.hp.value = 7;
    f.hooks.emit('updateActor', f.a, { 'system.attributes.hp.value': 7 });
    await tick();
    assert.deepEqual([...counter.calls.values()], [1, 0]);
    assert.equal(f.root().querySelector('[data-user-id="two"] .hp-value')?.textContent, '0/20');
    assert.equal(f.root().querySelector('.focus-hp'), null);
    pending.resolve({ groups: [] }); await tick();
    assert.equal(f.root().querySelector('.focus-hp .hp-value').textContent, '7/20');
  } finally { stop(); }
});

test('five-player multi-source batches collect each changed actor once and coalesce same-turn completions', async () => {
  const f = fixture(), actors = [f.a, f.b, actor('c'), actor('d'), actor('e')];
  for (const extra of actors.slice(2)) { f.users.push({ id: extra.id, name: extra.id, isGM: false, active: true, character: extra }); f.settings.get('seatOrder').push(extra.id); }
  const counter = countSheets(actors, 3), stop = f.start();
  try {
    await tick(); counter.reset();
    const panel = f.root().querySelector('.focus-panel'), paint = panel.replaceChildren.bind(panel);
    let paints = 0; panel.replaceChildren = (...nodes) => { paints++; return paint(...nodes); };
    actors.forEach((actor, index) => { actor.attributes.hp.value = index + 1; f.hooks.emit('updateItem', { parent: actor }, { 'system.uses.value': 1 }); });
    f.hooks.emit('updateActor', f.a, { 'system.attributes.hp.value': 1 });
    await tick();
    assert.deepEqual([...counter.calls.values()], [3, 3, 3, 3, 3]);
    assert.equal(paints, 2, 'one pending paint and one coalesced completion paint');
    assert.deepEqual([...f.root().querySelectorAll('.cast .hp-value')].map(node => node.textContent), ['1/20', '2/20', '3/20', '4/20', '5/20']);
    counter.reset(); paints = 0;
    f.hooks.emit('updateWorldTime', 100, 1); await tick();
    assert.deepEqual([...counter.calls.values()], [0, 0, 0, 0, 0]);
  } finally { stop(); }
});

test('five-player three-source fixture limits a single HP update to three sheet reads', async (t) => {
  const f = fixture(), actors = [f.a, f.b, actor('c'), actor('d'), actor('e')];
  for (const extra of actors.slice(2)) { f.users.push({ id: extra.id, name: extra.id, isGM: false, active: true, character: extra }); f.settings.get('seatOrder').push(extra.id); }
  const counter = countSheets(actors, 3), stop = f.start();
  try {
    await tick(); counter.reset();
    const panel = f.root().querySelector('.focus-panel'), paint = panel.replaceChildren.bind(panel);
    let paints = 0; panel.replaceChildren = (...nodes) => { paints++; return paint(...nodes); };
    f.a.attributes.hp.value = 11; f.hooks.emit('updateActor', f.a, { 'system.attributes.hp.value': 11 }); await tick();
    t.diagnostic(`5 PCs / 3 sources: single HP update = ${[...counter.calls.values()].reduce((total, n) => total + n, 0)} sheet reads, ${paints} paints`);
    assert.deepEqual([...counter.calls.values()], [3, 0, 0, 0, 0]);
    assert.equal(paints, 2);
    assert.equal(f.root().querySelector('.focus-hp .hp-value').textContent, '11/20');
  } finally { stop(); }
});

test('targeted revocation clears denied nodes synchronously while another authorized snapshot survives', async () => {
  const f = fixture(), counter = countSheets([f.a, f.b]), stop = f.start();
  try {
    await tick(); counter.reset();
    const pending = deferred();
    f.a.spellcasting.collections.get('0').entry.getSheetData = () => pending.promise;
    f.hooks.emit('updateActor', f.a, {}); await tick();
    f.a.testUserPermission = () => false;
    for (const key of ['name', 'img', 'attributes', 'system', 'spellcasting', 'type', 'itemTypes', 'conditions']) Object.defineProperty(f.a, key, { get() { throw Error(`denied ${key}`); } });
    f.hooks.emit('updateActor', f.a, { ownership: {} });
    assert.equal(f.root().querySelector('.focus-hp'), null);
    assert.equal(f.root().querySelector('[data-user-id="one"] .figure'), null);
    assert.equal(f.root().querySelector('[data-user-id="two"] .hp-value')?.textContent, '0/20');
    pending.resolve({ id: 'secret', name: 'Secret source', statistic: { dc: { value: 30 } }, groups: [] }); await tick();
    assert.equal(f.root().textContent.includes('Secret source'), false);
    assert.equal(counter.calls.get(f.b), 0);
  } finally { stop(); }
});

test('a completed focus actor renders while another actor sheet remains unresolved', async (t) => {
  const f = fixture(), counter = countSheets([f.a, f.b]), stop = f.start();
  try {
    await tick(); counter.reset();
    const slow = deferred(), focus = deferred();
    f.a.spellcasting.collections.get('0').entry.getSheetData = () => slow.promise;
    f.b.spellcasting.collections.get('0').entry.getSheetData = () => focus.promise;
    f.a.attributes.hp.value = 6; f.b.attributes.hp.value = 9;
    f.game.combat.combatant = { actorId: 'b', actor: f.b, name: 'Public B', visible: true };
    const panel = f.root().querySelector('.focus-panel'), paint = panel.replaceChildren.bind(panel);
    let paints = 0; panel.replaceChildren = (...nodes) => { paints++; return paint(...nodes); };
    f.hooks.emit('updateActor', f.a, {}); f.hooks.emit('updateActor', f.b, {});
    f.hooks.emit('updateCombat', f.game.combat, { turn: 1 }); await tick();
    assert.equal(paints, 1);
    focus.resolve({ groups: [] }); await tick();
    assert.equal(f.root().querySelector('.focus-name').textContent, 'Hero b');
    assert.equal(f.root().querySelector('.focus-hp .hp-value')?.textContent, '9/20');
    assert.equal(f.root().querySelector('[data-user-id="one"] .hp-value'), null);
    assert.equal(paints, 2);
    slow.resolve({ groups: [] }); await tick();
    assert.equal(f.root().querySelector('[data-user-id="one"] .hp-value').textContent, '6/20');
    assert.equal(f.root().querySelector('.focus-hp .hp-value').textContent, '9/20');
    assert.equal(paints, 3);
    t.diagnostic('Two separately completed actors: one pending paint, one focus paint before the slow sheet, one slow-actor paint');
  } finally { stop(); }
});

test('clock expiry only recollects assigned actors with effects and preserves rollback', async () => {
  const f = fixture(), effect = { id: 'clock', name: 'Blessing', isIdentified: true, system: { expired: false } };
  f.a.itemTypes = { effect: [effect] };
  const counter = countSheets([f.a, f.b]), stop = f.start();
  try {
    await tick(); counter.reset(); effect.system.expired = true;
    f.hooks.emit('updateWorldTime', 160, 60); await tick();
    assert.deepEqual([...counter.calls.values()], [1, 0]); assert.equal(f.root().querySelector('.condition'), null);
    effect.system.expired = false; f.hooks.emit('updateWorldTime', 100, -60); await tick();
    assert.deepEqual([...counter.calls.values()], [2, 0]); assert.equal(f.root().querySelector('.condition').textContent, 'Blessing');
  } finally { stop(); }
});

test('a newer request for the same actor wins without discarding another actor snapshot', async () => {
  const f = fixture(), counter = countSheets([f.a, f.b]), stop = f.start();
  try {
    await tick(); counter.reset();
    const old = deferred(), fresh = deferred(); let requests = 0;
    f.a.spellcasting.collections.get('0').entry.getSheetData = () => (++requests === 1 ? old : fresh).promise;
    f.a.attributes.hp.value = 4; f.hooks.emit('updateActor', f.a, {}); await tick();
    f.a.attributes.hp.value = 9; f.hooks.emit('updateItem', { parent: f.a }, {}); await tick();
    fresh.resolve({ id: 'fresh', name: 'Fresh source', statistic: { dc: { value: 24 } }, groups: [] }); await tick();
    assert.equal(f.root().querySelector('.focus-hp .hp-value').textContent, '9/20');
    old.resolve({ id: 'old', name: 'Old source', statistic: { dc: { value: 21 } }, groups: [] }); await tick();
    assert.equal(f.root().querySelector('.focus-hp .hp-value').textContent, '9/20');
    assert.equal(f.root().querySelector('.source-label').textContent, 'Fresh source');
    assert.equal(counter.calls.get(f.b), 0);
  } finally { stop(); }
});

test('native settings form CSV array mounts GM logo and assigned player portraits with live data', async () => {
  const f = fixture();
  f.settings.set('seatOrder', ['gm,one,two']);
  const stop = f.start();
  try {
    await tick();
    assert.equal(f.root().querySelectorAll('.cast-person').length, 3);
    assert.ok(f.root().querySelector('[data-user-id="gm"] .gm-logo-base'));
    assert.equal(f.root().querySelector('[data-user-id="one"] .figure').getAttribute('src'), 'original-a.webp');
    assert.equal(f.root().querySelector('[data-user-id="two"] .figure').getAttribute('src'), 'b.webp');
    assert.match(f.root().querySelector('.cast').textContent, /0\/20/);
  } finally { stop(); }
});

test('world-clock prepared expiry removes a retained public effect and rollback restores it without document updates', async () => {
  const f = fixture();
  const effect = { id: 'timed-public', name: 'Public blessing', isIdentified: true, system: { expired: false } };
  f.a.itemTypes = { effect: [effect] };
  f.a.reset = () => { throw Error('Recorder must not prepare or mutate actors'); };
  f.a.update = () => { throw Error('Recorder must not update actors'); };
  f.game.time = { worldTime: 100 };
  const stop = f.start();
  try {
    await tick(); assert.equal(f.root().querySelector('.condition')?.textContent, 'Public blessing');
    // PF2e effectTracker has already recomputed this prepared flag; no Item/Actor hook follows.
    effect.system.expired = true; f.game.time.worldTime = 160;
    f.hooks.emit('updateWorldTime', 160, 60); await tick();
    assert.equal(!!f.root().querySelector('.condition'), false);
    effect.system.expired = false; f.game.time.worldTime = 100;
    f.hooks.emit('updateWorldTime', 100, -60); await tick();
    assert.equal(f.root().querySelector('.condition')?.textContent, 'Public blessing');
    stop(); assert.equal(f.hooks.listenerCount('updateWorldTime'), 0);
    Object.defineProperty(effect, 'system', { get() { throw Error('Disposed runtime must not collect effects'); } });
    f.hooks.emit('updateWorldTime', 200, 100); await tick();
    assert.equal(f.root(), null);
  } finally { stop(); }
});

test('resource detail is GM-configurable world scope and changing it preserves the received story image', async () => {
  const definitions = new Map();
  runtime.registerSettings({ settings: { register: (_namespace, key, definition) => definitions.set(key, definition) } });
  const definition = definitions.get('resourceDetail');
  assert.equal(definition?.scope, 'world');
  assert.equal(definition?.default, 'compact');
  assert.deepEqual(definition?.choices, { compact: '简洁（最高三个法术环阶）', full: '详细（全部法术环阶）' });
  const f = fixture(), stop = f.start();
  try {
    await tick();
    f.hooks.emit('renderImagePopout', { id: 'received', options: { src: 'public.jpg' }, title: 'Public' });
    await tick();
    assert.equal(f.root().dataset.resourceDetail, 'compact');
    f.settings.set('resourceDetail', 'full');
    f.hooks.emit('updateSetting', { key: 'taka-obs-director.resourceDetail' });
    await tick();
    assert.equal(f.root().dataset.resourceDetail, 'full');
    assert.equal(f.root().dataset.mode, 'story');
    assert.equal(f.root().querySelector('.story-image').getAttribute('src'), 'public.jpg');
  } finally { stop(); }
});

test('only enabled PF2e /game exact non-GM recorder mounts; missing seats remain empty', async () => {
  for (const change of [(f) => f.settings.set('enabled', false), (f) => f.game.system.id = 'alienrpg', (f) => f.game.view = 'stream', (f) => f.game.user = f.users[1], (f) => f.settings.set('recorderUserId', 'other')]) {
    const f = fixture(); change(f); const stop = f.start(); await tick();
    assert.equal(f.root(), null); assert.equal(f.document.body.classList.contains('taka-obs-director-active'), false); stop();
  }
  const f = fixture(); f.settings.set('seatOrder', []); const stop = f.start(); await tick();
  assert.equal(f.root().querySelectorAll('.cast-person').length, 0); stop();
});

test('an unavailable or rejected native resize wrapper leaves the original client layout mounted', async () => {
  for (const libWrapper of [null, { register() { throw new Error('Unknown native target'); }, unregister() {} }]) {
    const f = fixture(); assert.equal(typeof runtime.startDirector, 'function');
    const stop = runtime.startDirector({ ...f, ui: {}, libWrapper });
    try {
      await tick(); assert.ok(!f.root(), 'a missing native wrapper must preserve the original layout');
      assert.equal(f.document.body.classList.contains('taka-obs-director-active'), false);
      assert.equal(f.document.getElementById('board').getAttribute('style'), 'color:red'); assert.ok(f.document.getElementById('audio').isConnected);
    } finally { stop(); }
  }
});

test('background room discovery follows late initialization/replacement without Foundry hooks and removes listeners on dispose', async () => {
  const f = fixture(), stop = f.start(); await tick();
  const old = participant(true); const first = roomFixture(f, [['one', old]]);
  await new Promise((resolve) => setTimeout(resolve, 1050)); assert.deepEqual(speakers(f.root()), ['one']);
  const next = participant(true); roomFixture(f, [['two', next]]);
  await new Promise((resolve) => setTimeout(resolve, 1050)); assert.deepEqual(speakers(f.root()), ['two']);
  assert.equal(old.listenerCount('isSpeakingChanged'), 0); assert.equal(first.room.eventNames().length, 0);
  stop(); assert.equal(next.listenerCount('isSpeakingChanged'), 0);
});

test('slow turn A cannot overwrite completed turn B and binding change clears old details immediately', async () => {
  const f = fixture(), slow = deferred(); f.users[2].character = actor('a', slow); f.game.combat.combatant.actor = f.users[2].character;
  const stop = f.start(); await tick();
  f.game.combat.combatant = { actorId: 'b', actor: f.b, name: 'B', img: 'b.webp', visible: true };
  f.hooks.emit('updateCombat', f.game.combat); await tick();
  assert.equal(f.root().querySelector('.focus-name').textContent, 'Hero b');
  slow.resolve({ name: 'Stale spell', groups: [] }); await tick();
  assert.equal(f.root().querySelector('.focus-name').textContent, 'Hero b');
  f.users[3].character = f.a; f.hooks.emit('updateUser', f.users[3]);
  assert.equal(f.root().querySelector('.focus-panel').textContent, '');
  await tick(); assert.equal(f.root().querySelector('.cast .figure').getAttribute('src'), 'original-a.webp'); stop();
});

test('a replaced plugin participant map removes stale listeners in the same room and fifth connection changes the cast', async () => {
  const f = fixture(), old = participant(true), next = participant(true);
  const { room, map } = roomFixture(f, [['one', old]]); const stop = f.start(); await tick();
  map.delete('one'); map.set('two', next); f.hooks.emit('renderCameraViews');
  assert.equal(old.listenerCount('isSpeakingChanged'), 0); assert.deepEqual(speakers(f.root()), ['two']);
  for (const id of ['three', 'four', 'five']) f.users.push({ id, name: id, isGM: false, active: false });
  f.settings.set('seatOrder', ['gm', 'one', 'two', 'three', 'four', 'five']); f.hooks.emit('updateSetting', { key: 'taka-obs-director.seatOrder' }); await tick();
  assert.equal(f.root().querySelectorAll('.cast-person').length, 5);
  const fifth = participant(false); map.set('five', fifth); f.hooks.emit('liveKitClientInitialized'); await tick();
  assert.equal(f.root().querySelectorAll('.cast-person').length, 6);
  room.emit('disconnected'); await tick(); assert.equal(f.root().querySelectorAll('.cast-person').length, 5); stop();
});

test('the same SDK participant adopts authoritative Map identity and drops invalid identity without duplicating listeners', async () => {
  const f = fixture(), same = participant(true, '{"fvttUserId":"one"}');
  const { room } = roomFixture(f); room.remoteParticipants.set(same.identity, same);
  const stop = f.start();
  try {
    await tick(); assert.deepEqual(speakers(f.root()), ['one']);
    const metadataHandler = same.listeners('isSpeakingChanged')[0];
    f.hooks.emit('liveKitClientInitialized');
    assert.equal(same.listeners('isSpeakingChanged')[0], metadataHandler);
    f.game.webrtc.client._liveKitClient.liveKitParticipants = new Map([['two', same]]);
    f.hooks.emit('liveKitClientInitialized');
    assert.deepEqual(speakers(f.root()), ['two']);
    assert.equal(same.listenerCount('isSpeakingChanged'), 1);
    const mappedHandler = same.listeners('isSpeakingChanged')[0];
    assert.notEqual(mappedHandler, metadataHandler);
    f.hooks.emit('renderCameraViews'); assert.equal(same.listeners('isSpeakingChanged')[0], mappedHandler);
    f.game.webrtc.client._liveKitClient.liveKitParticipants.clear(); same.metadata = '{"fvttUserId":"unknown-user"}';
    f.hooks.emit('liveKitClientInitialized');
    assert.deepEqual(speakers(f.root()), []); assert.equal(same.listenerCount('isSpeakingChanged'), 0);
    same.emit('isSpeakingChanged', true); assert.deepEqual(speakers(f.root()), []);
  } finally { stop(); }
});

test('native camera fallback follows class changes and is replaced on dock rerender without touching audio', async () => {
  const f = fixture(), dock = f.document.getElementById('camera-views');
  dock.insertAdjacentHTML('beforeend', '<div class="camera-view speaking" data-user="one"></div><div class="camera-view" data-user="two"></div>');
  const stop = f.start(); await tick(); assert.deepEqual(speakers(f.root()), ['one']);
  dock.querySelector('[data-user="two"]').classList.add('speaking'); await tick(); assert.deepEqual(speakers(f.root()), ['one', 'two']);
  const first = dock.querySelector('[data-user="one"]'); first.classList.remove('speaking'); await tick(); assert.deepEqual(speakers(f.root()), ['two']);
  const next = f.document.createElement('div'); next.id = 'camera-views'; next.innerHTML = '<div class="camera-view speaking" data-user="one"></div>';
  dock.id = 'old-dock'; f.document.body.append(next); f.hooks.emit('renderCameraViews'); await tick();
  assert.deepEqual(speakers(f.root()), ['one']); first.classList.add('speaking'); await tick(); assert.deepEqual(speakers(f.root()), ['one']);
  stop(); next.querySelector('.camera-view').classList.remove('speaking'); await tick(); assert.equal(f.root(), null); assert.ok(f.document.getElementById('audio').isConnected);
});

test('live assignment/document identity and Observer loss reject delayed results without a hook', async () => {
  for (const change of [(f) => f.users[2].character = actor('a'), (f) => f.users[2].character = f.b, (f) => f.users[2].character.testUserPermission = () => false]) {
    const f = fixture(), slow = deferred(); const a = actor('a', slow); f.users[2].character = a; f.game.combat.combatant.actor = a;
    const stop = f.start(); await tick(); change(f); slow.resolve({ name: 'Secret spell', groups: [] }); await tick();
    assert.equal(f.root().textContent.includes('Secret spell'), false);
    assert.equal(f.root().querySelector('.focus-hp'), null); stop();
  }
});

test('permission revocation clears visible sensitive nodes immediately and never reads denied actor getters', async () => {
  const f = fixture(); const stop = f.start(); await tick(); assert.match(f.root().textContent, /0\/20/);
  f.a.testUserPermission = () => false;
  for (const key of ['name', 'img', 'attributes', 'system', 'spellcasting', 'type']) Object.defineProperty(f.a, key, { get() { throw new Error(`denied ${key}`); } });
  f.hooks.emit('updateActor', f.a); assert.equal(f.root().querySelector('.focus-hp'), null); await tick();
  assert.equal(f.root().querySelector('.focus-name').textContent, 'Public A');
  assert.equal(f.root().querySelector('[data-user-id="one"] .figure'), null); stop();
});

test('actor-keyed portrait override does not follow a player rebinding to another actor', async () => {
  const f = fixture(); const stop = f.start(); await tick();
  assert.equal(f.root().querySelector('[data-user-id="one"] .figure').getAttribute('src'), 'original-a.webp');
  f.users[2].character = f.b; f.hooks.emit('updateUser', f.users[2]); await tick();
  assert.equal(f.root().querySelector('[data-user-id="one"] .figure').getAttribute('src'), 'b.webp'); stop();
});

test('received native images use app.options.src and filtered title; closing restores live combat state by app id', async () => {
  const f = fixture(); const stop = f.start(); await tick();
  const a = { id: 'img-a', options: { src: 'received-a.webp', window: { title: 'Private raw title' } }, title: '' };
  const b = { id: 'img-b', options: { src: 'received-b.webp' }, title: 'Public title' };
  f.hooks.emit('renderImagePopout', a); f.hooks.emit('renderImagePopout', b); await tick();
  assert.equal(f.root().dataset.mode, 'story'); assert.equal(f.root().querySelector('.story-image').getAttribute('src'), 'received-b.webp');
  f.hooks.emit('closeImagePopout', a); await tick(); assert.equal(f.root().querySelector('.story-image').getAttribute('src'), 'received-b.webp');
  f.game.combat = null; f.hooks.emit('deleteCombat'); f.hooks.emit('closeImagePopout', b); await tick();
  assert.equal(f.root().dataset.mode, 'explore'); assert.equal(f.root().querySelector('.focus-hp'), null);
  f.hooks.emit('renderImagePopout', a); await tick(); assert.equal(f.root().querySelector('.story-image').alt, ''); stop();
});

test('late LiveKit with opaque identity supports overlapping speakers without recollecting focus', async () => {
  const f = fixture(); let reads = 0; Object.defineProperty(f.a, 'name', { get() { reads++; return 'Hero a'; } });
  const stop = f.start(); await tick(); const before = reads;
  const one = participant(true), two = participant(true); const { room } = roomFixture(f, [['one', one], ['two', two]]);
  f.hooks.emit('liveKitClientAvailable'); f.hooks.emit('liveKitClientInitialized');
  assert.deepEqual(speakers(f.root()), ['one', 'two']); assert.equal(reads, before);
  one.isSpeaking = false; one.emit('isSpeakingChanged', false); assert.deepEqual(speakers(f.root()), ['two']);
  room.emit('activeSpeakersChanged', [one, two]); assert.deepEqual(speakers(f.root()), ['one', 'two']); assert.equal(reads, before);
  stop(); assert.equal(one.listenerCount('isSpeakingChanged'), 0); assert.equal(room.listenerCount('activeSpeakersChanged'), 0);
});

test('native muted and unpublished audio clears stale SDK speaking independently for overlapping seats', async () => {
  const f = fixture(), one = participant(true), two = participant(true);
  one.audioTrackPublications = new Map([['one-track', { isMuted: false }]]); two.audioTrackPublications = new Map([['two-track', { isMuted: false }]]);
  const { room } = roomFixture(f, [['one', one], ['two', two]]); const stop = f.start();
  try {
    await tick(); assert.deepEqual(speakers(f.root()), ['one', 'two']);
    one.audioTrackPublications.get('one-track').isMuted = true; room.emit('trackMuted');
    assert.deepEqual(speakers(f.root()), ['two']); assert.equal(one.isSpeaking, true);
    one.audioTrackPublications.get('one-track').isMuted = false; room.emit('trackUnmuted'); assert.deepEqual(speakers(f.root()), ['one', 'two']);
    one.audioTrackPublications.clear(); room.emit('trackUnpublished'); assert.deepEqual(speakers(f.root()), ['two']);
    room.emit('activeSpeakersChanged', [one, two]); assert.deepEqual(speakers(f.root()), ['two']);
    two.audioTrackPublications.clear(); f.hooks.emit('liveKitClientInitialized'); assert.deepEqual(speakers(f.root()), []);
    assert.equal(two.isSpeaking, true); assert.equal(f.root().querySelector('.focus-name')?.textContent, 'Hero a');
  } finally { stop(); }
});

test('room replacement removes old listeners; disconnect/offline clears speakers; metadata validates real user IDs', async () => {
  const f = fixture(), old = participant(true); const first = roomFixture(f, [['one', old]]); const stop = f.start(); await tick();
  assert.deepEqual(speakers(f.root()), ['one']);
  const valid = participant(true, '{"fvttUserId":"two"}'), unknown = participant(true, '{"fvttUserId":"stranger"}');
  const second = roomFixture(f); f.hooks.emit('liveKitClientInitialized'); second.room.emit('participantConnected', valid); second.room.emit('participantConnected', unknown);
  assert.equal(old.listenerCount('isSpeakingChanged'), 0); assert.equal(first.room.listenerCount('participantConnected'), 0); assert.deepEqual(speakers(f.root()), ['two']);
  f.users[3].active = false; f.hooks.emit('userConnected', f.users[3], false); assert.deepEqual(speakers(f.root()), []);
  second.room.emit('disconnected'); assert.deepEqual(speakers(f.root()), []);
  valid.isSpeaking = true; valid.emit('isSpeakingChanged', true); assert.deepEqual(speakers(f.root()), []);
  second.room.state = 'connected'; second.map.set('two', valid); f.users[3].active = true; second.room.emit('reconnected'); await tick();
  assert.deepEqual(speakers(f.root()), ['two']); second.map.delete('two'); second.room.emit('participantDisconnected', valid);
  assert.deepEqual(speakers(f.root()), []); assert.equal(valid.listenerCount('isSpeakingChanged'), 0); stop();
});

test('viewport wrapper calls native resize first, uses letterboxed CSS bounds, preserves pivot, restores board on dispose', async () => {
  const f = fixture(); const board = f.document.getElementById('board'), initialStyle = board.getAttribute('style'); const stop = f.start(); await tick();
  f.root().getBoundingClientRect = () => ({ left: 50, top: 100, width: 1920, height: 1080 });
  f.hooks.emit('canvasReady');
  assert.deepEqual(f.canvas.screenDimensions, [1332.48, 804.6]);
  assert.ok(Math.abs(parseFloat(board.style.left) - 157.52) < 1e-6); assert.ok(Math.abs(parseFloat(board.style.top) - 100) < 1e-6);
  assert.equal(f.canvas.stage.position.x, 666.24); assert.equal(f.canvas.stage.position.y, 402.3);
  assert.deepEqual(f.canvas.pans.at(-1), { x: 321, y: 456 });
  const before = f.canvas.pans.length; f.resize(); assert.equal(f.canvas.pans.length, before + 2);
  assert.equal(f.renderer.width, 1332.48);
  f.settings.set('manualMode', 'explore'); f.hooks.emit('updateSetting', { key: 'taka-obs-director.manualMode' }); await tick();
  assert.equal(f.renderer.width, 1704.96); stop();
  assert.deepEqual(f.canvas.screenDimensions, [1920, 1080]); assert.equal(board.getAttribute('style'), initialStyle);
  assert.equal(f.document.getElementById('audio').isConnected, true); assert.equal(f.document.body.classList.contains('taka-obs-director-active'), false);
  assert.equal(f.hooks.eventNames().length, 0); stop();
});

test('disable invalidates pending actor collection and disposes hooks/media-neutral lifecycle', async () => {
  const f = fixture(), slow = deferred(); f.users[2].character = actor('a', slow); const stop = f.start(); await tick();
  f.settings.set('enabled', false); f.hooks.emit('updateSetting', { key: 'taka-obs-director.enabled' });
  assert.equal(f.root(), null); slow.resolve({ name: 'Old data', groups: [] }); await tick(); assert.equal(f.root(), null);
  assert.equal(f.hooks.eventNames().length, 0); assert.equal(f.document.getElementById('audio').isConnected, true); stop();
});

test('init registers scope-correct settings; ready and setting changes reconcile activation without losing images on manual mode', async () => {
  assert.equal(typeof entry.installDirector, 'function', 'installDirector must be implemented');
  const f = fixture(), definitions = new Map(); f.settings.set('enabled', false); f.game.ready = false;
  f.game.settings.register = (namespace, key, definition) => { assert.equal(namespace, 'taka-obs-director'); definitions.set(key, definition); };
  const uninstall = entry.installDirector({ ...f, ui: {} }); f.hooks.emit('init');
  assert.equal(definitions.get('enabled').default, false); assert.equal(definitions.get('manualMode').scope, 'client'); assert.equal(definitions.get('skin').scope, 'world');
  assert.deepEqual(Object.keys(definitions.get('skin').choices), ['auto', 'cotct', 'sog', 'fotrp', 'av', 'bob']);
  f.game.ready = true; f.hooks.emit('ready'); assert.equal(f.root(), null);
  f.settings.set('enabled', true); f.hooks.emit('updateSetting', { key: 'taka-obs-director.enabled' }); await tick(); assert.ok(f.root());
  f.hooks.emit('renderImagePopout', { id: 'image', options: { src: 'public.webp' }, title: 'Received' }); await tick();
  f.settings.set('manualMode', 'combat'); f.hooks.emit('updateSetting', { key: 'taka-obs-director.manualMode' }); await tick();
  f.settings.set('manualMode', 'auto'); f.hooks.emit('updateSetting', { key: 'taka-obs-director.manualMode' }); await tick(); assert.equal(f.root().dataset.mode, 'story');
  f.settings.set('enabled', false); f.hooks.emit('updateSetting', { key: 'taka-obs-director.enabled' }); assert.equal(f.root(), null);
  uninstall(); assert.equal(f.hooks.eventNames().length, 0);
});

test('native entry resolves the real Game after its evaluation placeholder is replaced, mounting only the recorder at ready', async () => {
  const names = ['game', 'canvas', 'ui', 'Hooks', 'document', 'libWrapper'];
  const prior = new Map(names.map((name) => [name, Object.getOwnPropertyDescriptor(globalThis, name)]));
  try {
    for (const gm of [false, true]) {
      const f = fixture(), definitions = new Map();
      if (gm) f.game.user = f.users[1];
      f.game.ready = false;
      f.game.settings.register = (namespace, key, definition) => { assert.equal(namespace, 'taka-obs-director'); definitions.set(key, definition); };
      Object.assign(globalThis, { game: { view: 'game' }, canvas: undefined, ui: undefined, Hooks: f.hooks, document: f.document, libWrapper: undefined });
      try {
        await import(`../main.mjs?native-placeholder=${gm}`);
        assert.equal(!!f.root(), false); assert.equal(definitions.size, 0);
        Object.assign(globalThis, { game: f.game, canvas: f.canvas, ui: {}, libWrapper: f.libWrapper });
        assert.doesNotThrow(() => f.hooks.emit('init'));
        assert.equal(definitions.get('enabled').default, false); assert.equal(definitions.get('resourceDetail').scope, 'world');
        assert.equal(!!f.root(), false);
        f.game.ready = true; f.hooks.emit('ready'); await tick();
        assert.equal(!!f.root(), !gm);
        assert.equal(f.document.body.classList.contains('taka-obs-director-active'), !gm);
      } finally {
        f.document.defaultView.dispatchEvent(new f.document.defaultView.Event('beforeunload'));
        assert.equal(!!f.root(), false); assert.equal(f.hooks.eventNames().length, 0);
      }
    }
  } finally {
    for (const [name, descriptor] of prior) { if (descriptor) Object.defineProperty(globalThis, name, descriptor); else delete globalThis[name]; }
  }
});
