import assert from 'node:assert/strict';
import test from 'node:test';
import {migrateQuenchFlags} from '../scripts/maintenance/quench-aa8.mjs';

// A disabled-extra, source-only AA4 configuration; no world document is included.
const legacy = () => ({
  version: 4, killAnim: false, animLevel: false, override: true,
  sourceToken: {enable: false}, targetToken: {enable: false}, levels3d: {type: ''},
  macro: {enable: false}, animType: 'static',
  options: {staticType: 'source', menuType: 'liquid', variant: '01', enableCustom: false,
    repeat: 3, delay: null, scale: null, opacity: null, unbindAlpha: false,
    unbindVisibility: false, persistent: false},
  animation: 'splash', color: 'blue', audio: {a01: {enable: false}},
  preview: false, explosions: {enable: false}
});

test('AA8 receives a customized source splash instead of name-based template fallback', () => {
  const result = migrateQuenchFlags(legacy());
  assert.equal(result.version, 5);
  assert.equal(result.isEnabled, true);
  assert.equal(result.isCustomized, true);
  assert.equal(result.menu, 'ontoken');
  assert.deepEqual(result.primary.video, {
    dbSection: 'static', menuType: 'liquid', animation: 'splash', variant: '01', color: 'blue',
    enableCustom: false, customPath: ''
  });
  assert.equal(result.primary.options.playOn, 'source');
  assert.equal(result.primary.options.repeat, 3);
});

test('null timing, size and opacity retain AA7 migration defaults rather than AA8 defaults', () => {
  const {options} = migrateQuenchFlags(legacy()).primary;
  assert.equal(options.delay, 0);
  assert.equal(options.repeatDelay, 0);
  assert.equal(options.size, 1);
  assert.equal(options.opacity, 1);
  assert.equal(options.persistent, false);
  assert.equal(options.unbindAlpha, false);
  assert.equal(options.unbindVisibility, false);
});

test('migration leaves audio, macros and extra effects disabled', () => {
  const result = migrateQuenchFlags(legacy());
  assert.equal(result.primary.sound.enable, false);
  for (const key of ['source', 'target', 'secondary', 'macro', 'levels3d']) {
    assert.equal(result[key].enable, false, key);
  }
  assert.equal(result.soundOnly.sound.enable, false);
  assert.equal(result.fromAmmo, false);
});

test('migration does not mutate its input or keep obsolete fields', () => {
  const before = legacy();
  const snapshot = structuredClone(before);
  const result = migrateQuenchFlags(before);
  assert.deepEqual(before, snapshot);
  for (const key of ['override', 'killAnim', 'options', 'audio', 'sourceToken', 'targetToken']) {
    assert.equal(Object.hasOwn(result, key), false, key);
  }
  result.primary.options.repeat = 9;
  assert.equal(migrateQuenchFlags(before).primary.options.repeat, 3);
});

test('unknown, active-extra or different legacy configurations are rejected', () => {
  const changes = [
    data => { data.audio.a01.enable = true; },
    data => { data.macro.enable = true; },
    data => { data.sourceToken.enable = true; },
    data => { data.targetToken.enable = true; },
    data => { data.explosions.enable = true; },
    data => { data.levels3d.type = 'sprite'; },
    data => { data.options.persistent = true; },
    data => { data.options.staticType = 'target'; },
    data => { data.options.repeat = 4; },
    data => { data.options.scale = 2; },
    data => { data.options.enableCustom = true; },
    data => { data.killAnim = true; },
    data => { data.color = 'green'; },
    data => { data.extra = {enable: true}; }
  ];
  for (const change of changes) {
    const input = legacy();
    change(input);
    assert.throws(() => migrateQuenchFlags(input), /Unsupported/);
  }
  for (const input of [null, {}, {version: 5}]) {
    assert.throws(() => migrateQuenchFlags(input), /Unsupported/);
  }
});

test('object key order does not change whether a supported configuration is accepted', () => {
  const input = Object.fromEntries(Object.entries(legacy()).reverse());
  assert.equal(migrateQuenchFlags(input).version, 5);
});
