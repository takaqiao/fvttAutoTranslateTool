import test from 'node:test';
import assert from 'node:assert/strict';
import {mergeBuildChanges, projectBuildChanges} from '../../scripts/lib/build-sync.mjs';
import {merge3} from '../../scripts/lib/merge.mjs';
import {OWN_FLAG} from '../../scripts/lib/rules.mjs';
import {actorDoc} from '../helpers/apply-environment.mjs';

const item = (_id, type, system = {}) => ({_id, type, name: _id, system, effects: []});
const byId = (doc, id) => doc.items.find(entry => entry._id === id);
const versions = () => {
  const base = actorDoc('main');
  base.items = [item('gold', 'treasure', {quantity: 10}), item('feat', 'feat', {frequency: {value: 2, max: 3}})];
  return {base, main: structuredClone(base), char: structuredClone(base)};
};

test('both directions of a same-id physical type change preserve the main item', () => {
  for (const [from, to] of [['feat', 'weapon'], ['weapon', 'feat']]) {
    const docs = versions();
    for (const doc of Object.values(docs)) byId(doc, 'feat').type = from;
    byId(docs.char, 'feat').type = to;
    assert.deepEqual(byId(mergeBuildChanges(docs).doc, 'feat'), byId(docs.main, 'feat'));
  }
});

test('state conflicts are excluded while a build conflict remains visible', () => {
  const docs = versions();
  byId(docs.main, 'feat').system.frequency.value = 1;
  byId(docs.char, 'feat').system.frequency.value = 0;
  byId(docs.main, 'feat').name = 'Main choice';
  byId(docs.char, 'feat').name = 'Draft choice';
  const result = mergeBuildChanges(docs);
  assert.equal(byId(result.doc, 'feat').system.frequency.value, 1);
  assert.equal(byId(result.doc, 'feat').name, 'Draft choice');
  assert.deepEqual(result.conflicts.map(entry => entry.path), ['name']);
});

test('main deletion is not overridden by a char edit to only the spent value', () => {
  const docs = versions();
  docs.main.items = docs.main.items.filter(entry => entry._id !== 'feat');
  byId(docs.char, 'feat').system.frequency.value = 0;
  const result = mergeBuildChanges({...docs, target: 'char'});
  assert.equal(byId(result.doc, 'feat'), undefined);
  assert.deepEqual(result.conflicts, []);
  byId(docs.char, 'feat').name = 'Real draft';
  const withDraft = mergeBuildChanges({...docs, target: 'char'});
  assert.equal(byId(withDraft.doc, 'feat').name, 'Real draft');
  assert.deepEqual(withDraft.conflicts.map(entry => [entry.id, entry.path]), [['feat', '']]);
});

test('char target adopts all main state and keeps build drafts and local document keys', () => {
  const docs = versions();
  const main = docs.main, char = docs.char;
  main.items = main.items.filter(entry => entry._id !== 'gold');
  main.items.push(item('main-gear', 'equipment'));
  main.system.attributes.hp.value = 5;
  main.system.resources = {focus: {value: 0, max: 2}};
  main.system.customModifiers = {ac: [{label: 'GM', modifier: 1}]};
  main.effects = [{_id: 'effect', disabled: false}];
  char._id = 'work'; char.folder = 'work-folder'; char.ownership = {charPlayer: 3};
  char.flags[OWN_FLAG] = {base: {kind: 'main', hash: 'base'}};
  byId(char, 'gold').system.quantity = 999;
  char.items.push(item('local-gear', 'weapon'), item('draft', 'feat'));
  byId(char, 'feat').sort = 123;
  char.name = 'Draft name';
  const before = structuredClone(docs);
  const result = mergeBuildChanges({...docs, target: 'char'});
  assert.equal(result.doc._id, 'work');
  assert.equal(result.doc.folder, 'work-folder');
  assert.deepEqual(result.doc.ownership, {charPlayer: 3});
  assert.deepEqual(result.doc.flags[OWN_FLAG], char.flags[OWN_FLAG]);
  assert.equal(byId(result.doc, 'feat').sort, 123);
  assert.equal(byId(result.doc, 'gold'), undefined);
  assert.equal(byId(result.doc, 'local-gear'), undefined);
  assert.ok(byId(result.doc, 'main-gear'));
  assert.ok(byId(result.doc, 'draft'));
  assert.equal(result.doc.name, 'Draft name');
  assert.equal(result.doc.system.attributes.hp.value, 5);
  assert.deepEqual(result.doc.system.resources, main.system.resources);
  assert.deepEqual(result.doc.system.customModifiers, main.system.customModifiers);
  assert.deepEqual(result.doc.effects, main.effects);
  assert.equal(result.changed, true);
  assert.deepEqual(result.conflicts, []);
  assert.deepEqual(docs, before);
});

test('manual ability mode, custom proficiencies and real source build paths are projected', () => {
  const before = actorDoc('hero');
  before.system.abilities = null;
  const candidate = structuredClone(before);
  candidate.system.abilities = {str: {mod: 4}};
  candidate.system.proficiencies = {attacks: {custom: {rank: 2, custom: true, definition: ['item:group:knife']}}};
  candidate.system.skills = {arcana: {rank: 2, modifier: 99}};
  candidate.system.traits = {value: ['undead']};
  candidate.system.perception = {rank: 4};
  candidate.system.saves = {reflex: {rank: 4}};
  const result = projectBuildChanges(before, candidate);
  assert.deepEqual(result.system.abilities, {str: {mod: 4}});
  assert.deepEqual(result.system.proficiencies, candidate.system.proficiencies);
  assert.deepEqual(result.system.skills, {arcana: {rank: 2}});
  for (const key of ['traits', 'perception', 'saves']) assert.equal(result.system[key], undefined);
  assert.equal(projectBuildChanges(result, before).system.abilities, null);
});

test('familiar ability and creature update, derived data and current health stay local', () => {
  const before = actorDoc('pet', {type: 'familiar'});
  before.system = {master: {id: 'main', ability: 'int'}, details: {creature: {value: 'cat'}}, attributes: {hp: {value: 5, temp: 2}}};
  before.items = [item('old-ability', 'action')];
  const candidate = structuredClone(before);
  candidate.system.master.ability = 'cha';
  candidate.system.details.creature.value = 'bird';
  candidate.system.attributes = {hp: {value: 30, max: 30, temp: 10}, speed: {value: 50}};
  candidate.system.traits = {value: ['flying']};
  candidate.items = [item('fly', 'action')];
  const result = projectBuildChanges(before, candidate);
  assert.deepEqual(result.system.master, {id: 'main', ability: 'cha'});
  assert.equal(result.system.details.creature.value, 'bird');
  assert.deepEqual(result.system.attributes, before.system.attributes);
  assert.equal(result.system.traits, undefined);
  assert.equal(result.items[0]._id, 'fly');
});

for (const target of ['main', 'char']) {
  test(`${target} target initializes a newly unlocked stored slot without refilling existing slots`, () => {
    const docs = versions();
    // PF2e 的 source 预存 slot0..slot11，即使尚未解锁也有 max/value 为 0 的对象。
    const slots = Object.fromEntries(Array.from({length: 12}, (_, rank) => [`slot${rank}`, {max: 0, value: 0, prepared: []}]));
    slots.slot1 = {max: 2, value: 1, prepared: [{id: 'spell', expended: true}]};
    for (const doc of Object.values(docs)) doc.items.push(item('casting', 'spellcastingEntry', {slots: structuredClone(slots)}));
    const incoming = byId(docs.char, 'casting').system.slots;
    incoming.slot1 = {max: 3, value: 3, prepared: [{id: 'spell', expended: false}]};
    incoming.slot2 = {max: 1, value: 1, prepared: []};
    const result = mergeBuildChanges({...docs, target});
    const after = byId(result.doc, 'casting').system.slots;
    assert.deepEqual(after.slot1, {max: 3, value: 1, prepared: [{id: 'spell', expended: true}]});
    assert.deepEqual(after.slot2, {max: 1, value: 1, prepared: []});
    assert.deepEqual(result.conflicts, []);
  });

  test(`${target} target preserves conflicting ability and appearance drafts while default merge stays unchanged`, () => {
    const docs = versions();
    docs.base.img = 'old.webp'; docs.base.system.abilities = {str: {mod: 1}};
    docs.main.img = 'main.webp'; docs.main.system.abilities = {str: {mod: 2}};
    docs.char.img = 'draft.webp'; docs.char.system.abilities = {str: {mod: 4}};
    const result = mergeBuildChanges({...docs, target});
    assert.equal(result.doc.img, 'draft.webp');
    assert.equal(result.doc.system.abilities.str.mod, 4);
    assert.deepEqual(result.conflicts.map(({path, winner}) => ({path, winner})), [
      {path: 'system.abilities.str.mod', winner: 'char'}, {path: 'img', winner: 'char'},
    ]);
    const ordinary = merge3({...docs, target});
    assert.equal(ordinary.doc.img, 'main.webp');
    assert.equal(ordinary.doc.system.abilities.str.mod, 2);
  });
}
