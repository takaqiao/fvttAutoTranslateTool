import test from 'node:test';
import assert from 'node:assert/strict';
import {
  migrateLegacyChatData,
  migrateLegacySceneCreateData,
  migrateLegacySceneUpdateData,
  registerLegacyCompat
} from '../scripts/compat/legacy.mjs';

const scene = {firstLevel: {id: 'existingLevel001'}};

test('legacy numeric message type is removed before v14 subtype validation, without mutating its caller', () => {
  const input = Object.freeze({type: 0, content: 'Importer notice', flags: {fixture: true}});
  const result = migrateLegacyChatData(input);
  assert.deepEqual(result, {style: 0, content: 'Importer notice', flags: {fixture: true}});
  assert.equal(result.flags, input.flags);
  assert.equal(input.type, 0);
});

test('an explicitly supplied message style survives numeric-type compatibility', () => {
  for (const style of [2, 0, null, undefined]) {
    const result = migrateLegacyChatData({type: 4, style});
    assert.equal(Object.hasOwn(result, 'type'), false);
    assert.equal(Object.hasOwn(result, 'style'), true);
    assert.equal(result.style, style);
  }
});

test('modern messages and modern create arrays pass through by reference', () => {
  const item = Object.freeze({type: 'base', style: 0, content: 'Modern'});
  const items = Object.freeze([item]);
  assert.equal(migrateLegacyChatData(item), item);
  assert.equal(migrateLegacyChatData(items), items);
  const mixed = Object.freeze([item, Object.freeze({type: 1})]);
  const result = migrateLegacyChatData(mixed);
  assert.equal(result[0], item);
  assert.deepEqual(result[1], {style: 1});
  assert.equal(mixed[1].type, 1);
});

test('old importer scene backgrounds survive v14 input pruning in a new embedded level', () => {
  const input = Object.freeze({
    _id: 'sceneId000000001', name: 'Imported', img: 'map.webp', backgroundColor: '#123456',
    background: Object.freeze({offsetX: 12, offsetY: -3, tint: '#ffffff', scaleX: 1.2}),
    foreground: 'roof.webp', foregroundElevation: 0,
    fog: Object.freeze({overlay: 'fog.webp', exploration: false})
  });
  assert.deepEqual(migrateLegacySceneCreateData(input), {
    _id: 'sceneId000000001', name: 'Imported', shiftX: 12, shiftY: -3,
    fog: {exploration: false}, initialLevel: 'defaultLevel0000',
    levels: [{_id: 'defaultLevel0000', name: 'Default',
      background: {src: 'map.webp', color: '#123456', tint: '#ffffff'}, textures: {scaleX: 1.2},
      foreground: {src: 'roof.webp'}, elevation: {top: 0}, fog: {src: 'fog.webp'}}]
  });
  assert.equal(input.img, 'map.webp');
  assert.equal(input.background.offsetX, 12);
});

test('modern scene payloads pass through with no new objects or arrays', () => {
  const input = Object.freeze({name: 'Modern', shiftX: 4, levels: [{_id: 'level00000000001', background: {src: 'map.webp'}}]});
  const inputs = Object.freeze([input]);
  assert.equal(migrateLegacySceneCreateData(input), input);
  assert.equal(migrateLegacySceneCreateData(inputs), inputs);
  assert.equal(migrateLegacySceneUpdateData(scene, input), input);
});

test('existing creation levels and explicit IDs win while missing legacy fields are filled', () => {
  const other = Object.freeze({_id: 'otherLevel000001', flags: {untouched: true}});
  const target = Object.freeze({_id: 'chosenLevel00001', name: 'Explicit', background: Object.freeze({src: 'modern.webp', alphaThreshold: 0}), flags: {keep: true}});
  const input = Object.freeze({initialLevel: target._id, levels: Object.freeze([other, target]), img: 'old.webp',
    background: Object.freeze({tint: '#223344', alphaThreshold: .5}), shiftX: 99, 'background.offsetX': 4});
  const result = migrateLegacySceneCreateData(input);
  assert.equal(result.initialLevel, target._id);
  assert.equal(result.levels[0], other);
  assert.deepEqual(result.levels[1], {_id: target._id, name: 'Explicit', background: {src: 'modern.webp', alphaThreshold: 0, tint: '#223344'}, flags: {keep: true}});
  assert.equal(result.levels[1].flags, target.flags);
  assert.equal(result.shiftX, 99);
  assert.deepEqual(target.background, {src: 'modern.webp', alphaThreshold: 0});
});

test('legacy scene updates merge into the same existing level without erasing modern siblings', () => {
  const other = {_id: 'otherLevel000001', name: 'Untouched'};
  const input = Object.freeze({
    img: 'old.webp', 'background.tint': '#abcdef', 'background.offsetX': 0,
    foreground: null, foregroundElevation: 0, fog: Object.freeze({overlay: null, exploration: true}),
    levels: Object.freeze([other, Object.freeze({_id: scene.firstLevel.id, background: Object.freeze({src: 'modern.webp'}), textures: {rotation: 90}, flags: {keep: true}})])
  });
  const result = migrateLegacySceneUpdateData(scene, input);
  assert.equal(result.levels[0], other);
  assert.deepEqual(result.levels[1], {
    _id: scene.firstLevel.id, background: {src: 'modern.webp', tint: '#abcdef'}, textures: {rotation: 90}, flags: {keep: true},
    foreground: {src: null}, elevation: {top: 0}, fog: {src: null}
  });
  assert.equal(result.shiftX, 0);
  assert.deepEqual(result.fog, {exploration: true});
  assert.equal(input.img, 'old.webp');
});

test('nested and dotted legacy texture fields are mapped, with dotted fields taking precedence', () => {
  const input = {background: {src: 'nested.webp', anchorX: .1, offsetY: 4},
    'background.src': 'dotted.webp', 'background.anchorY': .2, 'background.fit': 'cover',
    'background.scaleY': 2, 'background.rotation': 45, 'background.offsetY': 0,
    'fog.overlay': 'fog.webp'};
  assert.deepEqual(migrateLegacySceneUpdateData(scene, input), {
    shiftY: 0, levels: [{_id: scene.firstLevel.id,
      background: {src: 'dotted.webp'}, textures: {anchorX: .1, anchorY: .2, fit: 'cover', scaleY: 2, rotation: 45}, fog: {src: 'fog.webp'}}]
  });
});

test('explicit invalid or null modern levels are not replaced with an invented valid level', () => {
  for (const levels of [null, 'invalid']) {
    const input = {levels, img: 'legacy.webp'};
    assert.equal(migrateLegacySceneCreateData(input), input);
    assert.equal(migrateLegacySceneUpdateData(scene, input), input);
  }
});

test('explicit dotted fields inside a modern level also win over legacy peer fields', () => {
  const input = {img: 'legacy.webp', background: {tint: '#aaaaaa'},
    levels: [{_id: scene.firstLevel.id, 'background.src': 'modern.webp', flags: {keep: true}}]};
  assert.deepEqual(migrateLegacySceneUpdateData(scene, input), {
    levels: [{_id: scene.firstLevel.id, 'background.src': 'modern.webp', background: {tint: '#aaaaaa'}, flags: {keep: true}}]
  });
});

function environment(version = '13.2.0', active = true) {
  class Message { static create() {} static createDocuments() {} }
  class MessageImpl extends Message {}
  class Scene { static create() {} static createDocuments() {} update() {} }
  class SceneImpl extends Scene {}
  const registrations = [];
  const values = new Map([['present.autoOpenAdventures', {registered: true}]]);
  const g = {
    ChatMessage: Message, Scene,
    CONFIG: {ChatMessage: {documentClass: MessageImpl}, Scene: {documentClass: SceneImpl}},
    game: {modules: new Map([['sf2e-murder-in-metal-city', {active, version}]]),
      packs: [{metadata: {type: 'Adventure', packageName: 'missing'}}, {metadata: {type: 'Adventure', packageName: 'present'}}, {metadata: {type: 'Actor', packageName: 'actors'}}],
      settings: {settings: values, register(namespace, key, config) {registrations.push({namespace, key, config}); values.set(`${namespace}.${key}`, config);}}
    }
  };
  return {g, registrations};
}

test('installed wrappers migrate at the public API boundary and preserve receiver, extra args and return values', () => {
  const {g} = environment();
  const wrappers = new Map();
  registerLegacyCompat({moduleId: 'test', g, registerWrapper(target, fn, kind) {
    assert.equal(kind, 'WRAPPER'); wrappers.set(target, fn);
  }});
  assert.deepEqual([...wrappers.keys()].sort(), ['ChatMessage.createDocuments', 'Scene.create', 'Scene.createDocuments', 'Scene.prototype.update'].sort());
  const receiver = {firstLevel: scene.firstLevel};
  const operation = Object.freeze({render: false});
  const answer = Promise.resolve('created');
  let seen;
  const original = function(...args) {seen = {receiver: this, args}; return answer;};
  const result = wrappers.get('Scene.prototype.update').call(receiver, original, {img: 'map.webp'}, operation, 'extra');
  assert.equal(result, answer);
  assert.equal(seen.receiver, receiver);
  assert.equal(seen.args[1], operation);
  assert.equal(seen.args[2], 'extra');
  assert.deepEqual(seen.args[0], {levels: [{_id: scene.firstLevel.id, background: {src: 'map.webp'}}]});
});

test('a system override is wrapped separately, while inherited static APIs are not wrapped twice', () => {
  const {g} = environment();
  g.CONFIG.ChatMessage.documentClass.create = function() {};
  g.CONFIG.ChatMessage.documentClass.createDocuments = function() {};
  g.CONFIG.Scene.documentClass.prototype.update = function() {};
  const targets = [];
  registerLegacyCompat({g, registerWrapper(target) {targets.push(target);}});
  assert.equal(targets.includes('CONFIG.ChatMessage.documentClass.create'), false);
  assert.equal(targets.includes('CONFIG.ChatMessage.documentClass.createDocuments'), true);
  assert.equal(targets.includes('CONFIG.Scene.documentClass.prototype.update'), true);
});

test('only the active audited importer version receives missing adventure settings, without replacing existing ones', () => {
  for (const [version, active, expected] of [['13.2.0', true, 1], ['13.2.0', false, 0], ['14.0.0', true, 0]]) {
    const {g, registrations} = environment(version, active);
    const before = g.game.settings.settings.get('present.autoOpenAdventures');
    registerLegacyCompat({g, registerWrapper() {}});
    assert.equal(registrations.length, expected);
    assert.equal(g.game.settings.settings.get('present.autoOpenAdventures'), before);
    if (expected) assert.deepEqual(registrations[0], {namespace: 'missing', key: 'autoOpenAdventures', config: {scope: 'world', config: false, type: Boolean, default: false}});
  }
});
