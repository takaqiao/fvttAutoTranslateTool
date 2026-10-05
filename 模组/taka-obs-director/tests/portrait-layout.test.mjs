import test from 'node:test';
import assert from 'node:assert/strict';

const layout = await import('../scripts/portrait-layout.mjs').catch((error) => {
  if (error.code === 'ERR_MODULE_NOT_FOUND') return {};
  throw error;
});
const source = 'actors/portrait.webp';

function normalize(candidate, options = { source }) {
  assert.equal(typeof layout.normalizePortraitLayout, 'function', 'normalizePortraitLayout must be implemented');
  return layout.normalizePortraitLayout(candidate, options);
}

function resolve(options) {
  assert.equal(typeof layout.resolvePortraitLayout, 'function', 'resolvePortraitLayout must be implemented');
  return layout.resolvePortraitLayout({ worldId: 'world-a', actorId: 'actor-a', source, ...options });
}

test('matching original image calibration returns bounded numeric positions and scale only', () => {
  const candidate = { source, x: 120, y: -100, scale: 9 };
  Object.defineProperty(candidate, 'actor', { get() { throw new Error('must not read Actor'); } });
  const result = normalize(candidate);
  assert.deepEqual(result, { x: 75, y: -75, scale: 3 });
  assert.equal(Object.getPrototypeOf(result), Object.prototype);
  assert.deepEqual(normalize({ source, x: -75, y: 75, scale: 0.1 }), { x: -75, y: 75, scale: 0.25 });
  assert.deepEqual(normalize({ source, x: 10.5, y: -8.25, scale: 1.6 }), { x: 10.5, y: -8.25, scale: 1.6 });
});

test('missing or nonfinite calibration values fall back without converting strings', () => {
  for (const candidate of [
    { source }, { source, x: '15', y: Infinity, scale: NaN }, { source, x: null, y: true, scale: undefined },
  ]) assert.deepEqual(normalize(candidate), { x: 0, y: 0, scale: 1 });
  assert.deepEqual(normalize({ source, x: NaN, y: 12, scale: 0 }), { x: 0, y: 12, scale: 0.25 });
});

test('a changed image or invalid candidate cannot reuse an old portrait transform', () => {
  for (const candidate of [
    null, [], [{ source, x: 50 }], new Date(), 'layout', { source: 'old.webp', x: 50, y: 20, scale: 2 },
  ]) assert.deepEqual(normalize(candidate), { x: 0, y: 0, scale: 1 });
  for (const current of ['', null, 42, undefined]) {
    assert.deepEqual(normalize({ source: current, x: 50, scale: 2 }, { source: current }), { x: 0, y: 0, scale: 1 });
  }
});

test('accessor fields and hostile objects fail closed without evaluating their getters', () => {
  let getters = 0;
  for (const key of ['source', 'x', 'y', 'scale']) {
    const candidate = { source, x: 20, y: 30, scale: 2 };
    Object.defineProperty(candidate, key, { get() { getters++; throw new Error('private getter'); } });
    assert.deepEqual(normalize(candidate), { x: 0, y: 0, scale: 1 });
  }
  const hostile = new Proxy({}, { getPrototypeOf() { throw new Error('prototype unavailable'); } });
  assert.deepEqual(normalize(hostile), { x: 0, y: 0, scale: 1 });
  assert.equal(getters, 0);
});

test('calibration must own source and coordinates rather than inherit prototype values', () => {
  const inherited = Object.create({ source, x: 50, y: 20, scale: 2 });
  assert.deepEqual(normalize(inherited), { x: 0, y: 0, scale: 1 });
  const candidate = Object.assign(Object.create(null), { source, x: 12, y: 4, scale: 1.2 });
  assert.deepEqual(normalize(candidate), { x: 12, y: 4, scale: 1.2 });
});

test('manual actor calibration wins over the matching world default', () => {
  const overrides = { 'actor-a': { source, x: -18, y: 23, scale: 1.75 } };
  const defaults = { 'world-a': { 'actor-a': { source, x: 5, y: 10, scale: 1.3 } } };
  assert.deepEqual(resolve({ overrides, defaults }), { x: -18, y: 23, scale: 1.75 });
});

test('stale manual images fall back to a matching current world calibration', () => {
  const overrides = { 'actor-a': { source: 'old.webp', x: 50, y: 50, scale: 2 } };
  const defaults = { 'world-a': { 'actor-a': { source, x: -4, y: 18, scale: 1.4 } } };
  assert.deepEqual(resolve({ overrides, defaults }), { x: -4, y: 18, scale: 1.4 });
  defaults['world-a']['actor-a'].source = 'another.webp';
  assert.deepEqual(resolve({ overrides, defaults }), { x: 0, y: 0, scale: 1 });
});

test('other actors and worlds cannot supply the selected portrait layout', () => {
  const overrides = { 'actor-b': { source, x: 40, scale: 2 } };
  const defaults = {
    'world-a': { 'actor-b': { source, x: 40, scale: 2 } },
    'world-b': { 'actor-a': { source, x: 60, scale: 3 } },
  };
  assert.deepEqual(resolve({ overrides, defaults }), { x: 0, y: 0, scale: 1 });
  assert.deepEqual(resolve({ overrides, defaults, worldId: 'world-b' }), { x: 60, y: 0, scale: 3 });
  assert.deepEqual(resolve({ overrides, defaults, worldId: '' }), { x: 0, y: 0, scale: 1 });
});

test('mapping prototype and accessor entries cannot be used as actor or world calibrations', () => {
  let getters = 0;
  const overrides = {};
  Object.defineProperty(overrides, 'actor-a', { get() { getters++; throw new Error('private mapping'); } });
  assert.deepEqual(resolve({ overrides }), { x: 0, y: 0, scale: 1 });
  const defaults = {};
  Object.defineProperty(defaults, 'world-a', { get() { getters++; throw new Error('private world'); } });
  assert.deepEqual(resolve({ defaults }), { x: 0, y: 0, scale: 1 });
  assert.deepEqual(resolve({ overrides: {}, actorId: '__proto__', defaults: {} }), { x: 0, y: 0, scale: 1 });
  assert.deepEqual(resolve({ overrides: {}, worldId: 'constructor', defaults: {} }), { x: 0, y: 0, scale: 1 });
  assert.equal(getters, 0);
});

test('returned calibration is independent of shared settings and subsequent calls', () => {
  const overrides = { 'actor-a': { source, x: 10, y: 20, scale: 1.4 } };
  const first = resolve({ overrides });
  first.x = 55;
  assert.deepEqual(resolve({ overrides }), { x: 10, y: 20, scale: 1.4 });
  assert.equal(overrides['actor-a'].x, 10);
});
