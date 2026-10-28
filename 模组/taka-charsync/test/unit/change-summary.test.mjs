import test from 'node:test';
import assert from 'node:assert/strict';
import { describeChanges } from '../../scripts/lib/change-summary.mjs';
import { ownerOfActorPath } from '../../scripts/lib/rules.mjs';

const character = () => ({
  _id: 'hero', name: 'Hero', type: 'character', items: [], effects: [], flags: {},
  system: {details: {level: {value: 5}, languages: {value: ['common']}},
    attributes: {hp: {value: 40, max: 50}}, skills: {}},
});

test('reports actor fields and effects even when no item or level changes', () => {
  const before = character();
  const after = structuredClone(before);
  after.name = 'Renamed';
  after.system.attributes.hp.value = 1;
  after.system.details.biography = {value: '<p>A new story</p>'};
  after.effects = [{_id: 'effect1', name: 'New effect'}];
  const result = describeChanges(before, after);
  assert.equal(result.changed, true);
  assert.equal(result.counts.fields, 3);
  assert.equal(result.counts.effects, 1);
  assert.match(result.html, /40.*1/);
  assert.match(result.html, /Renamed/);
  assert.match(result.html, /New effect/);
  assert.match(result.html, /A new story/);
  assert.doesNotMatch(result.html, /<p>A new story<\/p>/);
});

test('keeps actor field changes beside item changes', () => {
  const before = character();
  const after = structuredClone(before);
  after.items.push({_id: 'feat1', type: 'feat', name: 'New feat'});
  after.system.details.languages.value.push('elven');
  const result = describeChanges(before, after);
  assert.equal(result.counts.added, 1);
  assert.equal(result.counts.fields, 1);
  assert.match(result.html, /New feat/);
  assert.match(result.html, /语言/);
  assert.match(result.html, /elven/);
});

test('ignores local bookkeeping and does not mutate documents', () => {
  const before = character();
  const after = structuredClone(before);
  after._id = 'other-id';
  after.folder = 'folder';
  after.ownership = {owner: 3};
  after._stats = {modifiedTime: 123};
  after.flags['taka-charsync'] = {copy: {rev: 123}};
  const snapshot = structuredClone(after);
  const result = describeChanges(before, after);
  assert.equal(result.changed, false);
  assert.equal(result.html, '');
  assert.deepEqual(after, snapshot);
});

test('filters actor paths for build drafts and counts level and skill changes once', () => {
  const before = character();
  const after = structuredClone(before);
  after.name = 'Main-side name';
  after.system.attributes.hp.value = 1;
  after.effects.push({_id: 'effect1', name: 'Main-side effect'});
  after.system.details.level.value = 6;
  after.system.skills.arcana = {rank: 2};
  const result = describeChanges(before, after, {actorPathFilter: path => ownerOfActorPath(path) === 'char'});
  assert.equal(result.counts.fields, 2);
  assert.equal(result.counts.effects, 0);
  assert.match(result.html, /等级 5 → 6/);
  assert.match(result.html, /奥法/);
  assert.doesNotMatch(result.html, /Main-side name|血量|system\.skills/);
});

test('escapes dynamic values and includes unknown changes', () => {
  const before = character();
  const after = structuredClone(before);
  after.flags.plugin = {'<img src=x>': '<script>alert(1)</script>'};
  after.items.push({_id: 'feat1', type: 'feat', name: '<img src=x onerror=alert(1)>'});
  const result = describeChanges(before, after);
  assert.match(result.html, /&lt;script&gt;/);
  assert.match(result.html, /&lt;img/);
  assert.doesNotMatch(result.html, /<script|<img/);
  assert.equal(result.counts.fields, 1);
});

test('does not hide a replacement with a different item id', () => {
  const before = character();
  before.items = [{_id: 'old', type: 'feat', name: 'Same feat'}];
  const after = structuredClone(before);
  after.items[0]._id = 'new';
  const result = describeChanges(before, after);
  assert.equal(result.counts.added, 1);
  assert.equal(result.counts.removed, 1);
});

test('keeps long field differences available in the expanded details', () => {
  const before = character();
  before.system.details.biography = {value: 'A'.repeat(240) + '旧结局'};
  const after = structuredClone(before);
  after.system.details.biography.value = 'A'.repeat(240) + '新结局';
  const result = describeChanges(before, after);
  assert.match(result.html, /旧结局/);
  assert.match(result.html, /新结局/);
  assert.equal(result.text, '角色字段 1');
});
