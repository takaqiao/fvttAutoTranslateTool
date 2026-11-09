import test from 'node:test';
import assert from 'node:assert/strict';
import {catchUpCard} from '../../scripts/char/catch-up.mjs';
import {contentHash} from '../../scripts/lib/canon.mjs';
import {OWN_FLAG} from '../../scripts/lib/rules.mjs';
import {actorDoc, applyEnvironment} from '../helpers/apply-environment.mjs';
import {contentHash017} from '../helpers/content-hash-017.mjs';
import {activation, activationIds, charm} from '../helpers/item-activations.mjs';

const item = (_id, type, system = {}) => ({_id, type, name: _id, system, effects: []});
const byId = (doc, id) => doc.items.find(entry => entry._id === id);

function scenario({sameHash = false, type = 'character'} = {}) {
  const env = applyEnvironment();
  const base = actorDoc('main', {type});
  base.items = [item('coins', 'treasure', {quantity: 10}), item('feat', 'feat')];
  const main = structuredClone(base);
  const working = structuredClone(base);
  working._id = 'work'; working.folder = 'working-folder'; working.ownership = {charPlayer: 3};
  working.flags[OWN_FLAG] = {base: {kind: 'main', hash: sameHash ? 'latest' : 'base'}};
  env.meta.cards.work = {world: 'campaign', mainId: 'main', charOwner: 'char-player'};
  env.files.set('history/main-base.json', base);
  const latest = {hash: 'latest', readAt: 100, level: 2, applied: null, doc: main};
  env.files.set('main/campaign/main.json', latest);
  const actor = env.addActor(working);
  return {...env, actor, base, main, latest};
}

test('following main replaces char inventory/state but preserves unsubmitted build and appearance', async () => {
  const env = scenario();
  env.main.items = [item('new-gear', 'equipment'), item('feat', 'feat')];
  env.main.system.attributes.hp.value = 4;
  byId(env.actor._source, 'coins').system.quantity = 999;
  env.actor._source.items.push(item('local-gear', 'weapon'), item('draft', 'feat'));
  env.actor._source.name = 'Draft name';
  env.actor.refresh();
  const result = await catchUpCard(env.actor, {meta: env.meta});
  assert.equal(result.action, 'merge');
  assert.equal(byId(env.actor._source, 'coins'), undefined);
  assert.equal(byId(env.actor._source, 'local-gear'), undefined);
  assert.ok(byId(env.actor._source, 'new-gear'));
  assert.ok(byId(env.actor._source, 'draft'));
  assert.equal(env.actor.name, 'Draft name');
  assert.equal(env.actor.system.attributes.hp.value, 4);
  assert.equal(env.actor.id, 'work');
  assert.equal(env.actor._source.folder, 'working-folder');
  assert.deepEqual(env.actor._source.ownership, {charPlayer: 3});
  assert.deepEqual(env.actor._source.flags[OWN_FLAG].base, {kind: 'main', hash: 'latest'});
});

test('same-hash follow corrects game state once without losing build drafts or existing conflicts', async () => {
  const env = scenario({sameHash: true});
  byId(env.actor._source, 'coins').system.quantity = 999;
  env.actor._source.items.push(item('draft', 'feat'));
  env.actor._source.flags[OWN_FLAG].conflicts = {at: 1, mainHash: 'latest', list: [{scope: 'actor', path: 'name', winner: 'main'}]};
  env.actor.refresh();
  const result = await catchUpCard(env.actor, {meta: env.meta});
  assert.equal(result.action, 'merge');
  assert.equal(byId(env.actor._source, 'coins').system.quantity, 10);
  assert.ok(byId(env.actor._source, 'draft'));
  assert.equal(env.actor._source.flags[OWN_FLAG].conflicts.at, 1);
  const writes = env.writes.length;
  const second = await catchUpCard(env.actor, {meta: env.meta});
  assert.equal(second.action, 'none');
  assert.equal(env.writes.length, writes);
});

test('applied-version fast-forward retains a later build draft and main resources', async () => {
  const env = scenario();
  const submitted = structuredClone(env.base);
  submitted.system.details.level.value = 3;
  submitted.items.push(item('submitted-feat', 'feat'));
  const ref = {kind: 'char', cardId: 'work', hash: 'submitted'};
  env.files.set('history/char-submitted.json', submitted);
  Object.assign(env.latest, {level: 3, applied: ref, doc: structuredClone(submitted)});
  env.latest.doc.system.attributes.hp.value = 5;
  env.actor._source.system.details.level.value = 3;
  env.actor._source.items.push(item('submitted-feat', 'feat'), item('later-draft', 'feat'));
  env.actor.refresh();
  const result = await catchUpCard(env.actor, {meta: env.meta});
  assert.equal(result.action, 'merge');
  assert.ok(byId(env.actor._source, 'later-draft'));
  assert.ok(byId(env.actor._source, 'submitted-feat'));
  assert.equal(env.actor.system.attributes.hp.value, 5);
  assert.equal(env.writes.some(entry => entry.operation === 'create'), false);
});

test('familiar follow maps its master to the working owner and keeps its ability draft', async () => {
  const env = scenario({type: 'familiar', sameHash: true});
  env.meta.cards.work.companionOf = 'working-owner';
  env.main.system.master = {id: 'main-owner', ability: 'int'};
  env.actor._source.system.master = {id: 'working-owner', ability: 'cha'};
  env.actor._source.system.attributes.hp.value = 99;
  env.actor.refresh();
  await catchUpCard(env.actor, {meta: env.meta});
  assert.deepEqual(env.actor.system.master, {id: 'working-owner', ability: 'cha'});
  assert.equal(env.actor.system.attributes.hp.value, 12);
});

const migration = (version, previous) => ({version, previous: {schema: previous, foundry: '14.370', system: '8.7.0'}});
function migrate(doc, version, previous) {
  doc.system._migration = migration(version, previous);
  for (const entry of doc.items) entry.system._migration = migration(version, previous);
}

test('following a main card that just migrated keeps the working card migration records and reports no conflict', async () => {
  const env = scenario();
  migrate(env.base, 0.959, 0.958);
  migrate(env.main, 0.961, 0.959);
  env.main.system.attributes.hp.value = 4;
  migrate(env.actor._source, 0.961, 0.961);
  env.actor._source.items.push(item('draft', 'feat', {_migration: migration(0.961, 0.961)}));
  env.actor.refresh();
  const result = await catchUpCard(env.actor, {meta: env.meta});
  assert.equal(result.action, 'merge');
  assert.deepEqual(result.conflicts, []);
  assert.equal(env.actor.system.attributes.hp.value, 4);
  assert.ok(byId(env.actor._source, 'draft'));
  assert.deepEqual(env.actor._source.system._migration, migration(0.961, 0.961));
  for (const entry of env.actor._source.items) assert.deepEqual(entry.system._migration, migration(0.961, 0.961), entry._id);
  assert.equal(env.actor._source.flags[OWN_FLAG].conflicts, undefined);
});

test('a reset to the main version keeps the migration records of items the working card already has', async () => {
  const env = scenario();
  globalThis.Actor.create = data => Actor.implementation.create(data);
  migrate(env.main, 0.961, 0.959);
  env.main.system.details.level.value = 3;
  env.latest.level = 3;
  env.main.items.push(item('class-feature', 'feat', {_migration: migration(0.961, 0.959)}));
  migrate(env.actor._source, 0.961, 0.961);
  env.actor.refresh();
  const result = await catchUpCard(env.actor, {meta: env.meta});
  assert.equal(result.action, 'reset');
  assert.equal(env.actor._source.system.details.level.value, 3);
  assert.deepEqual(env.actor._source.system._migration, migration(0.961, 0.961));
  assert.deepEqual(byId(env.actor._source, 'feat').system._migration, migration(0.961, 0.961));
  assert.deepEqual(byId(env.actor._source, 'class-feature').system._migration, migration(0.961, 0.959),
    'an item new to the card keeps the record its data was migrated under');
});

test('a base published before the hash change is the same version when main republishes the same content', async () => {
  const env = scenario();
  for (const doc of [env.base, env.main]) migrate(doc, 0.961, 0.959);
  const oldHash = await contentHash017(env.base);
  const newHash = await contentHash(env.main);
  assert.notEqual(oldHash, newHash, 'the hash function did change for this card');
  env.files.delete('history/main-base.json');
  env.files.set(`history/main-${oldHash}.json`, env.base);
  env.latest.hash = newHash;
  migrate(env.actor._source, 0.961, 0.961);
  env.actor._source.items.push(item('draft', 'feat'));
  const pending = {at: 1, mainHash: oldHash, list: [{scope: 'item', id: 'feat', name: 'feat', path: 'name', winner: 'char'}]};
  env.actor._source.flags[OWN_FLAG] = {base: {kind: 'main', hash: oldHash}, conflicts: pending};
  env.actor.refresh();
  const before = structuredClone(env.actor._source);

  const result = await catchUpCard(env.actor, {meta: env.meta});

  assert.equal(result.action, 'none');
  assert.deepEqual(env.actor._source.flags[OWN_FLAG], {base: {kind: 'main', hash: newHash}, conflicts: pending});
  assert.deepEqual({...env.actor._source, flags: {}}, {...before, flags: {}}, 'the card content is untouched');
  assert.deepEqual(env.writes.map(entry => Object.keys(entry.data)), [['flags']]);
  assert.equal((await catchUpCard(env.actor, {meta: env.meta})).action, 'none');
  assert.equal(env.writes.length, 1);
});

test('following main brings main\'s activations and drops those the build server generated itself', async () => {
  const env = scenario();
  const gear = {...charm(), _id: 'test-gear', name: 'test-gear'};
  env.base.items.push(charm());
  env.main.items.push(charm(), activation('act-main', charm(), {enabled: true}));
  env.actor._source.items.push(charm(), activation('act-char', charm()), gear, activation('act-test', gear, {enabled: false}),
    item('draft', 'feat'));
  env.actor.refresh();
  const result = await catchUpCard(env.actor, {meta: env.meta});
  assert.equal(result.action, 'merge');
  assert.deepEqual(result.conflicts, []);
  assert.deepEqual(activationIds(env.actor._source), ['act-main']);
  assert.deepEqual(byId(env.actor._source, 'act-main'), byId(env.main, 'act-main'));
  assert.deepEqual(byId(env.actor._source, 'charm'), byId(env.main, 'charm'));
  assert.equal(byId(env.actor._source, 'test-gear'), undefined);
  assert.ok(byId(env.actor._source, 'draft'));
});

test('the one-off hash upgrade of a 0.1.8 base still holds when the card carries activations', async () => {
  const env = scenario();
  for (const doc of [env.base, env.main]) {
    doc.items.push(charm(), activation('act-main', charm(), {enabled: true}));
    migrate(doc, 0.961, 0.959);
  }
  const oldHash = await contentHash017(env.base);
  const newHash = await contentHash(env.main);
  env.files.delete('history/main-base.json');
  env.files.set(`history/main-${oldHash}.json`, env.base);
  env.latest.hash = newHash;
  const working = env.actor._source;
  working.items.push(charm(), activation('act-main', charm(), {enabled: true}), activation('act-char', charm(), {enabled: true}));
  migrate(working, 0.961, 0.961);
  working.items.push(item('draft', 'feat'));
  const pending = {at: 1, mainHash: oldHash, list: [{scope: 'item', id: 'feat', name: 'feat', path: 'name', winner: 'char'}]};
  working.flags[OWN_FLAG] = {base: {kind: 'main', hash: oldHash}, conflicts: pending};
  env.actor.refresh();

  const result = await catchUpCard(env.actor, {meta: env.meta});

  // 同版本跟上：只校正启动项这类主服状态，基线记号换成新哈希，上次的冲突记录还在
  assert.equal(result.action, 'merge');
  assert.deepEqual(result.conflicts, []);
  assert.deepEqual(env.actor._source.flags[OWN_FLAG], {base: {kind: 'main', hash: newHash}, conflicts: pending});
  assert.deepEqual(activationIds(env.actor._source), ['act-main']);
  assert.ok(byId(env.actor._source, 'draft'));
  const writes = env.writes.length;
  assert.equal((await catchUpCard(env.actor, {meta: env.meta})).action, 'none');
  assert.equal(env.writes.length, writes);
});
