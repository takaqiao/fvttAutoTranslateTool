import test from 'node:test';
import assert from 'node:assert/strict';

// A missing module is the expected first RED state for this new module.
const model = await import('../scripts/model.mjs').catch((error) => {
  if (error.code === 'ERR_MODULE_NOT_FOUND') return {};
  throw error;
});
function api(name) {
  assert.equal(typeof model[name], 'function', `${name} must be implemented`);
  return model[name];
}

test('mode priority is manual lock, received public image, started combat, exploration', () => {
  const resolveMode = api('resolveMode');
  assert.equal(resolveMode({ manualMode: 'explore', publicImages: [{}], combatStarted: true }), 'explore');
  assert.equal(resolveMode({ manualMode: 'combat', publicImages: [{}] }), 'combat');
  assert.equal(resolveMode({ manualMode: 'story' }), 'story');
  assert.equal(resolveMode({ manualMode: 'auto', publicImages: [{}], combatStarted: true }), 'story');
  assert.equal(resolveMode({ publicImages: [], combatStarted: true }), 'combat');
  assert.equal(resolveMode({ publicImages: [], combatStarted: false }), 'explore');
});

test('counters preserve zero and reject absent, negative and nonfinite values', () => {
  const normalizeCounter = api('normalizeCounter');
  assert.deepEqual(normalizeCounter(0, 82), { value: 0, max: 82 });
  for (const [value, max] of [[1, -1], [1, NaN], [1, '3'], [null, 3], [Infinity, 3], [-1, 3], [1, 0]]) {
    assert.equal(normalizeCounter(value, max), null);
  }
});

const users = [
  { id: 'gm', isGM: true, active: true },
  { id: 'obs', isGM: false, active: true },
  { id: 'a', isGM: false, active: false, character: { id: 'pc-a' } },
  { id: 'b', isGM: false, active: false, character: { id: 'pc-b' } },
  { id: 'c', isGM: false, active: true, character: { id: 'pc-c' } },
  { id: 'd', isGM: false, active: true, character: { id: 'pc-d' } },
  { id: 'e', isGM: false, active: false, character: { id: 'pc-e' } },
  { id: 'other', isGM: false, active: true },
];

test('configured four players stay seated and fifth needs active or LiveKit connection', () => {
  const selectCastUsers = api('selectCastUsers');
  const input = { users, recorderId: 'obs', seatOrder: ['gm', 'obs', 'b', 'a', 'b', 'c', 'd', 'e'] };
  assert.deepEqual(selectCastUsers(input).map((user) => user.id), ['gm', 'b', 'a', 'c', 'd']);
  assert.deepEqual(selectCastUsers({ ...input, connectedUserIds: new Set(['e']) }).map((user) => user.id), ['gm', 'b', 'a', 'c', 'd', 'e']);
  assert.deepEqual(selectCastUsers({ ...input, users: users.map((user) => user.id === 'e' ? { ...user, active: true } : user) }).map((user) => user.id), ['gm', 'b', 'a', 'c', 'd', 'e']);
});

test('unconfigured users cannot become seats and changed character is resolved from current user', () => {
  const selectCastUsers = api('selectCastUsers');
  assert.deepEqual(selectCastUsers({ users, recorderId: 'obs', seatOrder: [] }), []);
  const currentUsers = users.map((user) => user.id === 'a' ? { ...user, character: { id: 'new-pc' } } : user);
  const cast = selectCastUsers({ users: currentUsers, recorderId: 'obs', seatOrder: ['gm', 'a'] });
  assert.equal(cast[1].character.id, 'new-pc');
});

test('native settings form comma-separated array entries retain only configured current users', () => {
  const selectCastUsers = api('selectCastUsers');
  const seatOrder = Object.freeze([' gm, obs, b, a, b, c, d, e, unknown, ', null, 7]);
  const input = { users, recorderId: 'obs', seatOrder };
  assert.deepEqual(selectCastUsers(input).map(user => user.id), ['gm', 'b', 'a', 'c', 'd']);
  assert.deepEqual(selectCastUsers({ ...input, connectedUserIds: ['e'] }).map(user => user.id), ['gm', 'b', 'a', 'c', 'd', 'e']);
  assert.deepEqual(selectCastUsers({ users, recorderId: 'obs', seatOrder: [null, 7, 'unknown,obs'] }), []);
  assert.deepEqual(seatOrder, [' gm, obs, b, a, b, c, d, e, unknown, ', null, 7]);
});

test('hidden, invisible or token-hidden combatants have no public focus', () => {
  const selectFocus = api('selectFocus');
  const actor = new Proxy({}, { get() { throw new Error('private actor read'); } });
  for (const visibility of [{ hidden: true, visible: true }, { hidden: false, visible: false }, { hidden: false }, { hidden: false, visible: true, token: { hidden: true } }]) {
    assert.equal(selectFocus({ combatant: { actor, name: 'Secret', img: 'secret.png', ...visibility } }), null);
  }
});

test('public NPC projection never reads combatant.actor or private actor identity', () => {
  const selectFocus = api('selectFocus');
  const combatant = { actorId: 'npc', name: 'Visible silhouette', img: 'public.png', hidden: false, visible: true };
  Object.defineProperty(combatant, 'actor', { get() { throw new Error('NPC actor must not be touched'); } });
  assert.deepEqual(selectFocus({ combatant, allowedActorIds: new Set(['pc']), viewer: {} }), {
    actor: null, publicOnly: true, name: 'Visible silhouette', portrait: 'public.png',
  });
});

test('whitelisted character still needs observer permission and permission errors fail closed', () => {
  const selectFocus = api('selectFocus');
  const actor = { id: 'pc', type: 'character', testUserPermission: () => false };
  Object.defineProperty(actor, 'system', { get() { throw new Error('private detail read'); } });
  const combatant = { actorId: 'pc', actor, name: 'Public PC', img: 'public-pc.png', hidden: false, visible: true };
  assert.equal(selectFocus({ combatant, allowedActorIds: ['pc'], viewer: {} }).actor, null);
  actor.testUserPermission = () => { throw new Error('permission unavailable'); };
  assert.equal(selectFocus({ combatant, allowedActorIds: ['pc'], viewer: {} }).actor, null);
  actor.testUserPermission = () => true;
  assert.equal(selectFocus({ combatant, allowedActorIds: ['pc'], viewer: {} }).actor, actor);
});

test('scene window uses stage dimensions and combat meets the quarter-width card', () => {
  const sceneRect = api('sceneRect');
  for (const [mode, expected] of [
    ['explore', { left: 107.52, top: 0, width: 1704.96, height: 804.6 }],
    ['combat', { left: 107.52, top: 0, width: 1332.48, height: 804.6 }],
  ]) {
    const actual = sceneRect({ width: 1920, height: 1080, mode });
    for (const key of ['left', 'top', 'width', 'height']) assert.ok(Math.abs(actual[key] - expected[key]) < 0.000001, `${mode} ${key}`);
  }
  assert.equal(sceneRect({ width: 0, height: 1080 }), null);
});
