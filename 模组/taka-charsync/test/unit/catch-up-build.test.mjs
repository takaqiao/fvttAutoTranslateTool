import test from 'node:test';
import assert from 'node:assert/strict';
import {catchUpCard} from '../../scripts/char/catch-up.mjs';
import {OWN_FLAG} from '../../scripts/lib/rules.mjs';
import {actorDoc, applyEnvironment} from '../helpers/apply-environment.mjs';

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
