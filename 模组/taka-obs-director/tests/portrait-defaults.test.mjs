import test from 'node:test';
import assert from 'node:assert/strict';
import { portraitDefaults } from '../scripts/portrait-defaults.mjs';
import { normalizePortraitLayout, resolvePortraitLayout } from '../scripts/portrait-layout.mjs';

// Source paths were checked against the original-art inventory; positions use
// square-slot percentages, including the source aspect ratio and scale.
const originals = [
  ['cotct', 'gWIa4TzF4sKUdi52', 'assets/constantine-token-20260919/mortal-source.png', -8.994, -4, 1],
  ['cotct', 'ubsGugmsOlQXuXub', 'assets/mishpat-token-20260920/source.png', -.747, -.5, 1],
  ['cotct', 'elTQO2mfkP7rjig7', 'assets/tavian-token-20260917/source.png', 0, -2, 1],
  ['cotct', '04POvuUzJrHHqIuT', 'assets/maomao-token-20260917/normal-source.png', 4, 3.5, 1],
  ['sog', 'i5JMNOGS6VXftxM0', 'tokenizer/pc-images/9591d892a14aa8d72a393ada3f77dcb5cb2ce782f2d9c421b745715fdbfc9572.Avatar.webp?1776523615868', -5, 0, 1],
  ['sog', 'QoVzp1GKBvYoZFWt', 'tokenizer/pc-images/4da33042812206b9dfaf7464bce99afe6dc0c10564a82a3852cf0b671b7649a5.Avatar.webp?1776446651636', 0, 0, 1],
  ['sog', 'mzV1KHpy21LSQkoo', 'tokenizer/pc-images/b5e0a0d9251cc5586122f0881426c90e114a0e712e58457fabd1bbe9d16e0800.Avatar.webp?1776446121605', 0, 0, 1],
  ['sog', 'yMb0KVxx1th4L3LQ', 'tokenizer/pc-images/2cb89aaa871c158539f0b4565bd83b1f58f2e9c58b6d7a137eefbc83b6a4b1e7.Avatar.webp?1776827511835', 0, 0, 1],
  ['pnvfcgjbf2cjp7gz', 'DE84KdZduIHoxvrQ', 'tokenizer/pc-images/6445fa496996dcdfda85c7d87afe2ce20e54633fee272aec945509ceed315c3a.Avatar.webp?1778818775711', -5, 0, 1],
  ['pnvfcgjbf2cjp7gz', 'fjY5SqLU5rfztZky', 'tokenizer/pc-images/9454a9b6b5f48204690bac9923f7518f22d0b48fd99bbc4b84b69894a4e8fffc.Avatar.webp?1779255519564', 1, 0, 1],
  ['pnvfcgjbf2cjp7gz', '7ZILSyyncy0W8uaO', 'tokenizer/pc-images/6bc5b208ac98ad0bfb5b709e1074bd1c0d5cfed87e9658c18ebe1e20f80f28ea.Avatar.webp?1779162776869', 4, 0, 1],
  ['pnvfcgjbf2cjp7gz', 'GzEJSZ5ecX4NW2L5', 'tokenizer/pc-images/7959e11e3ec205eae190bb84da9221d68857af8c70e4e0ba967ec1af343e4b14.Avatar.webp?1779158406393', 0, 0, 1],
  ['-', 'TV6lF7S1FcytHDcs', 'assets/QQ%E5%9B%BE%E7%89%8720260903124714-removebg.png', -.449, -1, 1],
  ['-', '2Zkv2wjkEdrO6WfE', 'assets/QQ%E5%9B%BE%E7%89%8720260903130056-removebg.png', -4.73, -1.5, 1],
  ['-', 'ZRiI3NpMNFlir0xn', 'assets/QQ%E5%9B%BE%E7%89%8720260903125824-removebg-preview.png', 11.119, .5, 1],
  ['-', 'H7rzh6LPWIVu4Zxg', 'assets/QQ%E5%9B%BE%E7%89%8720260903123733.png', -2.803, 0, 1],
  ['ujx5r8oipw7ercdr', 'sAJeWzzdMROned3V', 'tokenizer/pc-avatars/Avatar.%E6%99%AE%E8%8E%B1%E5%BE%B7.webp?v=1789621185555', -2.5, 0, 1],
  ['ujx5r8oipw7ercdr', 'oGvjoi8dKU6GlKa0', 'assets/BoB/797c5ea4be2deb774cba062ea0c4b742.png', -4.725, 0, 1.05],
  ['ujx5r8oipw7ercdr', 'ZOHMU4FoFUW9WxkQ', 'assets/BoB/8506585b5958b73b5ec5138fa7085a3f.png', -8, 0, 1],
  ['ujx5r8oipw7ercdr', 'bbcRRqa0XfhwEu1J', 'assets/BoB/1d6a834f6b8e92bb8a5e27599cae11d2_720-bgremoved.jpg', -.693, -1.872, 1.04],
];
const worlds = ['cotct', 'sog', 'pnvfcgjbf2cjp7gz', '-', 'ujx5r8oipw7ercdr'];
const centered = { x: 0, y: 0, scale: 1 };

test('defaults cover only four assigned PCs in each of the five approved worlds', () => {
  assert.deepEqual(Object.keys(portraitDefaults).sort(), [...worlds].sort());
  assert.equal(Object.values(portraitDefaults).flatMap(Object.keys).length, 20);
  for (const worldId of worlds) {
    const expectedIds = originals.filter(([world]) => world === worldId).map(([, actorId]) => actorId);
    assert.equal(expectedIds.length, 4);
    assert.deepEqual(Object.keys(portraitDefaults[worldId]).sort(), expectedIds.sort(), worldId);
  }
});

test('all approved original images resolve to their manually checked composition', () => {
  for (const [worldId, actorId, source, x, y, scale] of originals) {
    const expected = { x, y, scale };
    const candidate = portraitDefaults[worldId][actorId];
    assert.equal(candidate.source, source, `${worldId}/${actorId}: original image`);
    assert.deepEqual(normalizePortraitLayout(candidate, { source }), expected, `${actorId}: normalized layout`);
    assert.deepEqual(resolvePortraitLayout({ worldId, actorId, source, defaults: portraitDefaults }), expected, `${actorId}: effective layout`);
  }
});

test('defaults cannot follow the same actor and image into another world', () => {
  for (const [worldId, actorId, source] of originals) {
    for (const otherWorld of [...worlds.filter(world => world !== worldId), 'alien', '', undefined]) {
      assert.deepEqual(resolvePortraitLayout({ worldId: otherWorld, actorId, source, defaults: portraitDefaults }), centered, `${actorId}: ${otherWorld}`);
    }
  }
});

test('another assigned actor cannot borrow a portrait calibration in the same world', () => {
  for (const [worldId, actorId, source] of originals) {
    for (const [, otherActor] of originals.filter(([, id]) => id !== actorId)) {
      assert.deepEqual(resolvePortraitLayout({ worldId, actorId: otherActor, source, defaults: portraitDefaults }), centered, `${actorId} -> ${otherActor}`);
    }
  }
});

test('replaced images and new query versions start centered instead of reusing old art offsets', () => {
  for (const [worldId, actorId, source] of originals) {
    const changed = [source.replace(/^(assets|tokenizer)\//, '$1/replacement/'), `${source.split('?')[0]}?v=999`];
    if (decodeURIComponent(source) !== source) changed.push(decodeURIComponent(source));
    for (const currentSource of changed) {
      assert.deepEqual(normalizePortraitLayout(portraitDefaults[worldId][actorId], { source: currentSource }), centered, actorId);
      assert.deepEqual(resolvePortraitLayout({ worldId, actorId, source: currentSource, defaults: portraitDefaults }), centered, `${actorId}: ${currentSource}`);
    }
  }
});

test('manual calibration takes precedence only while its exact original image is current', () => {
  for (const [worldId, actorId, source, x, y, scale] of originals) {
    const manual = { x: 7.75, y: -4.5, scale: 1.125 };
    const overrides = { [actorId]: { source, ...manual } };
    assert.deepEqual(resolvePortraitLayout({ worldId, actorId, source, overrides, defaults: portraitDefaults }), manual, actorId);
    overrides[actorId].source = 'assets/old-image.png';
    assert.deepEqual(resolvePortraitLayout({ worldId, actorId, source, overrides, defaults: portraitDefaults }), { x, y, scale }, `${actorId}: stale manual image`);
    assert.deepEqual(resolvePortraitLayout({ worldId, actorId, source: 'assets/new-image.png', overrides, defaults: portraitDefaults }), centered, `${actorId}: replacement image`);
  }
});

test('shipped calibration entries contain only relative image sources and finite numeric transforms', () => {
  for (const entries of Object.values(portraitDefaults)) {
    for (const [actorId, candidate] of Object.entries(entries)) {
      assert.deepEqual(Object.keys(candidate).sort(), ['scale', 'source', 'x', 'y']);
      for (const descriptor of Object.values(Object.getOwnPropertyDescriptors(candidate))) assert.ok(Object.hasOwn(descriptor, 'value'), actorId);
      assert.match(candidate.source, /^(assets|tokenizer)\/[^\\#]*\.(png|webp|jpg)(\?(v=)?\d+)?$/, actorId);
      assert.ok(!candidate.source.split('?')[0].split('/').includes('..'), actorId);
      for (const key of ['x', 'y', 'scale']) assert.ok(Number.isFinite(candidate[key]), `${actorId}: ${key}`);
      assert.ok(Math.abs(candidate.x) <= 75 && Math.abs(candidate.y) <= 75 && candidate.scale >= .25 && candidate.scale <= 3, actorId);
    }
  }
});
