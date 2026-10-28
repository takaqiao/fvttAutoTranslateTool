import test from 'node:test';
import assert from 'node:assert/strict';
import {applyCopy, applyForce, applyNew, previewPending, runApplyCheck, undoLast} from '../../scripts/main/apply.mjs';
import {OWN_FLAG} from '../../scripts/lib/rules.mjs';
import {actorDoc, applyEnvironment, makeCopy} from '../helpers/apply-environment.mjs';

const item = (id, type, moduleId) => ({_id: id, name: id, type, effects: [],
  system: {rules: [{key: 'GrantItem', uuid: `Compendium.${moduleId}.items.Item.abcdefghijklmnop`}]},
  flags: {}});
const usage = id => ({id, title: id, evidence: [{kind: 'reference', documentId: 'feat', documentName: 'feat', path: 'system.rules.0.uuid'}]});

for (const kind of ['level', 'force']) {
  test(`${kind} reports modules from the actual import scope and keeps undo state`, async () => {
    const env = applyEnvironment();
    const actor = env.mapped('source');
    game.modules = new Map([['build-addon', {id: 'build-addon', title: 'Build addon', active: false}]]);
    const source = actorDoc('source', {items: [item('feat', 'feat', 'build-addon'), item('gear', 'equipment', 'gear-addon')]});
    const copy = env.publishCopy(makeCopy('source', kind, {moduleUsage: [usage('build-addon'), usage('gear-addon')]}), source);
    const preview = await previewPending({actorId: actor.id, copy});
    assert.match(preview, /build-addon/);
    if (kind === 'level') assert.doesNotMatch(preview, /gear-addon/);
    if (kind === 'force') await applyForce(actor, copy);
    else await applyCopy(actor, copy, {manual: true});
    const saved = actor._source.flags[OWN_FLAG].applied.moduleUsage;
    assert.ok(saved.some(entry => entry.id === 'build-addon'));
    assert.equal(saved.some(entry => entry.id === 'gear-addon'), kind === 'force');
    const message = env.messages.at(-1).content;
    assert.match(message, /build-addon/);
    assert.match(message, /未启用/);
    if (kind === 'level') assert.doesNotMatch(message, /gear-addon/);
    else assert.match(message, /gear-addon/);
    await undoLast(actor);
    assert.equal(actor._source.flags[OWN_FLAG].applied?.moduleUsage, undefined);
  });
}

test('new-card group reports a module used only by its companion', async () => {
  const env = applyEnvironment();
  const copy = env.publishCopy(makeCopy('source', 'new', {pred: null,
    companions: [{sourceId: 'pet', copyId: 'pet-copy', rev: 200}]}));
  env.publishCopy(makeCopy('pet', 'new', {pred: null, companionOf: copy.copyId}),
    actorDoc('pet', {items: [item('pet-ability', 'feat', 'pet-addon')]}));
  const id = await applyNew(copy);
  assert.match(env.messages.at(-1).content, /pet-addon/);
  assert.match(env.messages.at(-1).content, /未安装或不兼容/);
  assert.ok(env.actors.get('pet')._source.flags[OWN_FLAG].applied.moduleUsage.some(entry => entry.id === 'pet-addon'));
  assert.deepEqual(env.actors.get(id)._source.flags[OWN_FLAG].applied.moduleUsage, []);
});

test('automatic apply reports modules and a forged usage record cannot invent a dependency', async () => {
  const env = applyEnvironment();
  const actor = env.mapped('source');
  actor.system.details.xp.value = 1000;
  const source = actorDoc('source', {items: [item('feat', 'feat', 'auto-addon')]});
  source.system.details.level.value = 3;
  env.publishCopy(makeCopy('source', 'level', {level: 3, moduleUsage: [usage('invented-addon')]}), source);
  const result = await runApplyCheck({trigger: 'tick'});
  assert.deepEqual(result.applied, [actor.id]);
  assert.match(env.messages.at(-1).content, /auto-addon/);
  assert.doesNotMatch(env.messages.at(-1).content, /invented-addon/);
  assert.ok(actor._source.flags[OWN_FLAG].applied.moduleUsage.some(entry => entry.id === 'auto-addon'));
});

test('a frozen module namespace identifies an imported flag even when unavailable on main', async () => {
  const env = applyEnvironment();
  const actor = env.mapped('source');
  const source = actorDoc('source', {items: [{_id: 'feat', type: 'feat', name: '<Borrowed>', system: {},
    flags: {'unknown-addon': {mode: true}}, effects: []}]});
  const copy = env.publishCopy(makeCopy('source', 'force', {moduleUsage: [{id: 'unknown-addon', title: '<Addon>', evidence: []}]}), source);
  await applyForce(actor, copy);
  assert.match(env.messages.at(-1).content, /unknown-addon/);
  assert.match(env.messages.at(-1).content, /需核对/);
  assert.doesNotMatch(env.messages.at(-1).content, /<Addon>|<Borrowed>/);
});

test('force import takes package evidence from immutable history rather than a changed same-revision listing', async () => {
  const env = applyEnvironment();
  const actor = env.mapped('source');
  const source = actorDoc('source', {items: [{_id: 'feat', type: 'feat', name: 'Feat', system: {},
    flags: {'frozen-addon': {mode: true}}, effects: []}]});
  const copy = env.publishCopy(makeCopy('source', 'force', {moduleUsage: [{id: 'frozen-addon', title: 'Frozen title', evidence: []}]}), source);
  env.files.set('copies/source.json', [{...copy, moduleUsage: [{id: 'frozen-addon', title: 'Later title', evidence: []}]}]);
  await applyForce(actor, copy);
  assert.match(env.messages.at(-1).content, /Frozen title/);
  assert.doesNotMatch(env.messages.at(-1).content, /Later title/);
});

test('a rejected replacement retains module identification for the old item that stays on the actor', async () => {
  const env = applyEnvironment();
  const actor = env.mapped('source');
  actor._source.items = [{_id: 'feat', name: 'Existing feat', type: 'feat', system: {},
    flags: {'old-addon': {value: true}}, effects: []}];
  actor._source.flags[OWN_FLAG] = {applied: {moduleUsage: [{id: 'old-addon', title: 'Old addon', evidence: []}]}};
  actor.refresh();
  const Item = CONFIG.Item.documentClass;
  CONFIG.Item.documentClass = class extends Item {
    constructor(data, options) {
      if (options?.strict && data.name === 'Invalid replacement') throw new Error('Invalid replacement');
      super(data);
    }
  };
  const copy = env.publishCopy(makeCopy('source', 'force'), actorDoc('source', {items: [{
    _id: 'feat', name: 'Invalid replacement', type: 'feat', system: {}, flags: {}, effects: [],
  }]}));
  await applyForce(actor, copy);
  assert.equal(actor._source.items[0].name, 'Existing feat');
  assert.match(env.messages.at(-1).content, /old-addon/);
  assert.match(env.messages.at(-1).content, /Existing feat/);
  assert.doesNotMatch(env.messages.at(-1).content, /规则引用的模组/);
});

test('deleting an old flagged item does not report its retained package catalog as a dependency', async () => {
  const env = applyEnvironment();
  const actor = env.mapped('source');
  actor._source.items = [{_id: 'feat', name: 'Old feat', type: 'feat', system: {}, flags: {'old-addon': {}}, effects: []}];
  actor._source.flags[OWN_FLAG] = {applied: {moduleUsage: [{id: 'old-addon', title: 'Old addon', evidence: []}]}};
  actor.refresh();
  const copy = env.publishCopy(makeCopy('source', 'force'), actorDoc('source'));
  await applyForce(actor, copy);
  assert.deepEqual(actor._source.items, []);
  assert.doesNotMatch(env.messages.at(-1).content, /old-addon/);
});
