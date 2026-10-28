import test from 'node:test';
import assert from 'node:assert/strict';
import {resolveActorCampaign} from '../../scripts/lib/campaign-rules.mjs';

const actor = (id, extra = {}) => ({_id: id, type: 'character', ...extra});
const own = flags => ({'taka-charsync': flags});

function resolve(doc, {cardWorld = {}, folderWorld = {}, actors = [], folders = []} = {}) {
  return resolveActorCampaign(doc, {
    snapshot: {cardWorld, folderWorld},
    actors: new Map(actors.map(row => [row._id, row])),
    folders: new Map(folders.map(row => [row._id, row])),
  });
}

test('working cards use explicit assignment and detect a conflicting campaign folder', () => {
  assert.equal(resolve(actor('A'), {cardWorld: {A: 'ecnu'}}).world, 'ecnu');
  const conflict = resolve(actor('A', {folder: 'other'}), {
    cardWorld: {A: 'ecnu'}, folderWorld: {other: 'sog'}, folders: [{_id: 'other'}],
  });
  assert.equal(conflict.ok, false);
  assert.equal(conflict.reason, 'assignment-conflict');
});

test('new cards inherit explicit parent folder mapping without guessing names or user characters', () => {
  assert.equal(resolve(actor('new', {folder: 'child'}), {
    folderWorld: {root: 'ecnu'}, folders: [{_id: 'child', folder: 'root'}, {_id: 'root'}],
  }).world, 'ecnu');
  assert.equal(resolve(actor('new', {name: 'ECNU', ownership: {gm: 3}})).reason, 'missing-assignment');
});

test('a frozen copy retains its target campaign after the working card changes campaign', () => {
  const copy = actor('copy', {folder: 'sog', flags: own({copy: {cardId: 'A', targetWorld: 'ecnu'}})});
  assert.equal(resolve(copy, {cardWorld: {A: 'sog', copy: 'sog'}, folderWorld: {sog: 'sog'}}).world, 'ecnu');
  assert.equal(resolve(actor('bad-copy', {flags: own({copy: {cardId: 'A'}})})).reason, 'missing-copy-target');
});

test('backups and familiars can resolve through a master that is a frozen copy', () => {
  const master = actor('copy', {flags: own({copy: {targetWorld: 'ecnu'}})});
  const pet = actor('pet', {type: 'familiar', system: {master: {id: 'copy'}}});
  const backup = actor('backup', {flags: own({backupOf: 'pet'})});
  assert.equal(resolve(pet, {actors: [master]}).world, 'ecnu');
  assert.equal(resolve(backup, {actors: [pet, master]}).world, 'ecnu');
});

test('missing references, reference cycles, and familiar assignment conflicts fail explicitly', () => {
  const a = actor('A', {flags: own({backupOf: 'B'})});
  const b = actor('B', {flags: own({backupOf: 'A'})});
  assert.equal(resolve(a, {actors: [a, b]}).reason, 'assignment-cycle');
  assert.equal(resolve(a).reason, 'missing-actor');
  const pet = actor('pet', {type: 'familiar', system: {master: {id: 'A'}}});
  assert.equal(resolve(pet, {actors: [actor('A')], cardWorld: {A: 'ecnu', pet: 'sog'}}).reason,
    'assignment-conflict');
});

test('folder cycles and contradictory ancestor mappings are not silently ignored', () => {
  const card = actor('A', {folder: 'child'});
  assert.equal(resolve(card, {
    folders: [{_id: 'child', folder: 'parent'}, {_id: 'parent', folder: 'child'}],
  }).reason, 'folder-cycle');
  assert.equal(resolve(card, {
    folderWorld: {child: 'sog', parent: 'ecnu'},
    folders: [{_id: 'child', folder: 'parent'}, {_id: 'parent'}],
  }).reason, 'assignment-conflict');
});

test('a configured folder id is unavailable when its document no longer exists', () => {
  const result = resolve(actor('A', {folder: 'removed'}), {folderWorld: {removed: 'ecnu'}});
  assert.equal(result.ok, false);
  assert.equal(result.reason, 'missing-folder');
});
