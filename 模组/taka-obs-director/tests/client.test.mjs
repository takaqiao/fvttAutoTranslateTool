import test from 'node:test';
import assert from 'node:assert/strict';
const client = await import('../scripts/client.mjs').catch(error => {
  if (error.code === 'ERR_MODULE_NOT_FOUND') return {};
  throw error;
});
function fixture() {
  const values = new Map([
    ['core.maxFPS', 60], ['core.visionAnimation', true], ['core.lightAnimation', true],
    ['pf2e-hud.persistent.display', 'always'], ['pf2e-hud.token.activation', 'hover'],
    ['pf2e-hud.tracker.enabled', true], ['pf2e-hud.tooltip.distance', 'feet'], ['pf2e-hud.tooltip.status', true],
  ]);
  const definitions = new Map([...values.keys()].map(key => [key, { scope: key.startsWith('core.') ? 'client' : 'user' }]));
  const av = new Map([['disableVideo', false], ['muteAll', false], ['audioSrc', 'disabled'], ['videoSrc', 'disabled']]);
  const writes = [];
  const game = {
    user: { id: 'recorder', isGM: false }, modules: new Map([['pf2e-hud', { active: true }]]),
    settings: { settings: definitions, get: (namespace, key) => values.get(`${namespace}.${key}`),
      set: async (namespace, key, value) => { writes.push(`${namespace}.${key}`); values.set(`${namespace}.${key}`, value); } },
    webrtc: { settings: { get: (scope, key) => av.get(key), set: async (scope, key, value) => {
      assert.equal(scope, 'client'); writes.push(`av.${key}`); av.set(key, value);
    } } },
  };
  return { game, values, definitions, av, writes };
}
function start(f, isActive = () => true) {
  assert.equal(typeof client.applyRecorderPreferences, 'function');
  return client.applyRecorderPreferences({ game: f.game, isActive });
}
test('every fresh recorder browser applies native video pause and 30fps without muting audio or disabling animations', async () => {
  for (let i = 0; i < 2; i++) {
    const f = fixture(); const policy = start(f); await policy.ready;
    assert.equal(f.av.get('disableVideo'), true);
    assert.equal(f.av.get('muteAll'), false);
    assert.equal(f.values.get('core.maxFPS'), 30);
    assert.equal(f.values.get('core.visionAnimation'), true);
    assert.equal(f.values.get('core.lightAnimation'), true);
    assert.equal(f.values.get('pf2e-hud.persistent.display'), 'disabled');
    assert.equal(f.values.get('pf2e-hud.token.activation'), 'disabled');
    assert.equal(f.values.get('pf2e-hud.tracker.enabled'), false);
    assert.equal(f.values.get('pf2e-hud.tooltip.distance'), 'never');
    assert.equal(f.values.get('pf2e-hud.tooltip.status'), false);
    assert.ok(f.writes.every(key => key === 'av.disableVideo' || key === 'core.maxFPS' || key.startsWith('pf2e-hud.')));
    await policy.dispose();
    assert.equal(f.av.get('disableVideo'), false); assert.equal(f.values.get('core.maxFPS'), 60);
  }
});
test('ordinary users, GM and unregistered or world-scoped HUD settings are left alone', async () => {
  const f = fixture(); const stopped = start(f, () => false); await stopped.ready; await stopped.dispose();
  assert.deepEqual(f.writes, []);
  f.game.user.isGM = true; const gm = start(f); await gm.ready; await gm.dispose(); assert.deepEqual(f.writes, []);
  f.game.user.isGM = false;
  f.definitions.set('pf2e-hud.tracker.enabled', { scope: 'world' });
  f.definitions.delete('pf2e-hud.token.activation');
  const policy = start(f); await policy.ready;
  assert.equal(f.values.get('pf2e-hud.tracker.enabled'), true);
  assert.equal(f.values.get('pf2e-hud.token.activation'), 'hover');
  await policy.dispose();
});
test('a lower existing canvas limit is preserved and dispose respects later manual edits', async () => {
  const f = fixture(); f.values.set('core.maxFPS', 20);
  const policy = start(f); await policy.ready;
  assert.equal(f.values.get('core.maxFPS'), 20);
  f.values.set('pf2e-hud.persistent.display', 'combat');
  await policy.dispose(); await policy.dispose();
  assert.equal(f.values.get('pf2e-hud.persistent.display'), 'combat');
  assert.equal(f.writes.filter(key => key === 'core.maxFPS').length, 0);
  assert.equal(f.writes.filter(key => key === 'av.disableVideo').length, 2);
});
test('dispose during a pending native write restores it and cancels remaining preferences', async () => {
  const f = fixture(); let finish;
  f.game.webrtc.settings.set = async (scope, key, value) => {
    f.writes.push(`av.${key}`);
    if (value === true) await new Promise(resolve => { finish = resolve; });
    f.av.set(key, value);
  };
  const policy = start(f); const closing = policy.dispose(); finish(); await closing;
  assert.equal(f.av.get('disableVideo'), false);
  assert.equal(f.values.get('core.maxFPS'), 60);
  assert.deepEqual(f.writes, ['av.disableVideo', 'av.disableVideo']);
});
test('preferences never restore into another user identity', async () => {
  const f = fixture(); const policy = start(f); await policy.ready;
  const count = f.writes.length; f.game.user = { id: 'other', isGM: false };
  await policy.dispose(); assert.equal(f.writes.length, count);
});

test('restoration stops if the account changes during an awaited setting write', async () => {
  const f = fixture(); const policy = start(f); await policy.ready;
  const originalSet = f.game.settings.set;
  const restored = [];
  f.game.settings.set = async (namespace, key, value) => {
    restored.push({ user: f.game.user.id, key });
    await originalSet(namespace, key, value);
    f.game.user = { id: 'ordinary-player', isGM: false };
  };
  await policy.dispose();
  assert.deepEqual(restored, [{ user: 'recorder', key: 'tooltip.status' }]);
  assert.equal(f.values.get('core.maxFPS'), 30);
});

test('a new active policy waits for the old policy restoration before reading preferences', async () => {
  const f = fixture(); const old = start(f); await old.ready;
  const closing = old.dispose(); const fresh = start(f);
  await Promise.all([closing, fresh.ready]);
  assert.equal(f.values.get('core.maxFPS'), 30);
  assert.equal(f.av.get('disableVideo'), true);
  assert.equal(f.values.get('pf2e-hud.persistent.display'), 'disabled');
  await fresh.dispose();
  assert.equal(f.values.get('core.maxFPS'), 60);
  assert.equal(f.av.get('disableVideo'), false);
});

test('a temporarily unavailable native preference does not block other settings or a later login', async () => {
  const f = fixture(), original = f.game.webrtc.settings.get;
  f.game.webrtc.settings.get = () => { throw Error('AV settings unavailable'); };
  const old = start(f);
  await old.ready;
  assert.equal(f.values.get('core.maxFPS'), 30);
  assert.equal(f.values.get('pf2e-hud.tracker.enabled'), false);
  f.game.webrtc.settings.get = original;
  const fresh = start(f); await fresh.ready;
  assert.equal(f.av.get('disableVideo'), true);
  assert.equal(f.values.get('core.maxFPS'), 30);
  await fresh.dispose();
});
