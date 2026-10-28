import test from 'node:test';
import assert from 'node:assert/strict';
import {applyNew, importNewFromWorking} from '../../scripts/main/apply.mjs';
import {OWN_FLAG} from '../../scripts/lib/rules.mjs';
import {actorDoc, applyEnvironment, makeCopy} from '../helpers/apply-environment.mjs';

function newGroup() {
  const env = applyEnvironment();
  const companion = makeCopy('pet', 'new', {companionOf: 'source-copy', pred: null});
  const copy = env.publishCopy(makeCopy('source', 'new', {pred: null,
    companions: [{copyId: companion.copyId, rev: companion.rev, sourceId: 'pet'}]}));
  const pet = actorDoc('pet', {type: 'familiar', system: {master: {id: 'source-copy'}, attributes: {hp: {value: 10, max: 10}}}});
  env.publishCopy(companion, pet);
  env.meta.working.source = {owner: 'char-player', hash: 'master-hash'};
  env.meta.working.pet = {owner: 'char-player', hash: 'pet-hash'};
  env.files.set('working/source.json', {hash: 'master-hash', owner: 'char-player', doc: actorDoc('source')});
  env.files.set('working/pet.json', {hash: 'pet-hash', owner: 'char-player', doc: {...pet, system: {...pet.system, master: {id: 'source'}}}});
  return {...env, copy};
}

for (const mode of ['copy', 'working']) {
  const apply = (env, options) => mode === 'copy' ? applyNew(env.copy, options) : importNewFromWorking('source', options);
  test(`${mode} imports the familiar with the owner's actual destination id`, async () => {
    const env = newGroup();
    const occupied = env.addActor(actorDoc('source', {name: 'Unrelated'}));
    const id = await apply(env);
    assert.equal(id, 'free-master');
    const master = env.actors.get(id), pet = env.actors.get('pet');
    assert.ok(pet, 'the familiar must be created with its master');
    assert.equal(pet.system.master.id, id);
    assert.deepEqual(pet.ownership, {default: 0, player: 3});
    assert.equal(pet._source.flags[OWN_FLAG].charId, 'pet');
    assert.deepEqual(master._source.flags[OWN_FLAG].applied.createdCompanions, ['pet']);
    assert.equal(occupied.name, 'Unrelated');
  });

  test(`${mode} rejects an occupied familiar id before creating the master`, async () => {
    const env = newGroup();
    env.addActor(actorDoc('pet'));
    await assert.rejects(apply(env), /同伴.*id.*占用/);
    assert.deepEqual(env.writes, []);
  });

  test(`${mode} removes the new master when familiar creation fails`, async () => {
    const env = newGroup();
    env.failures.create = doc => doc._id === 'pet';
    await assert.rejects(apply(env), /create failed/);
    assert.equal(env.actors.size, 0);
  });

  test(`${mode} restores the old group when archiving a companion fails`, async () => {
    const env = newGroup();
    const old = env.mapped('old', 'old-main');
    const oldPet = env.mapped('old-pet', 'old-pet-main', {companionOf: 'old'});
    old._source.folder = 'original'; old.refresh();
    oldPet._source.folder = 'original-pets'; oldPet.refresh();
    const before = [old.toObject(), oldPet.toObject()];
    env.failures.update = (actor, data) => actor.id === oldPet.id && data[`flags.${OWN_FLAG}.retired`] === true;
    await assert.rejects(apply(env, {replaceId: old.id}), /update failed/);
    assert.deepEqual([old.toObject(), oldPet.toObject()], before);
    assert.deepEqual([...env.actors.keys()].sort(), ['old-main', 'old-pet-main']);
  });

  test(`${mode} archives both the replaced owner and its companions`, async () => {
    const env = newGroup();
    const old = env.mapped('old', 'old-main');
    const oldPet = env.mapped('old-pet', 'old-pet-main', {companionOf: 'old'});
    await apply(env, {replaceId: old.id});
    for (const actor of [old, oldPet]) {
      assert.equal(actor._source.flags[OWN_FLAG].retired, true);
      assert.equal(actor._source.folder, 'archive');
    }
  });
}

test('a new-card group with a missing companion history writes nothing', async () => {
  const env = newGroup();
  env.files.delete('history/copy-pet-copy-200.json');
  await assert.rejects(applyNew(env.copy), /版本/);
  assert.deepEqual(env.writes, []);
});

test('a failed new-card cleanup tells the GM which cards remain', async () => {
  const env = newGroup();
  env.failures.create = doc => doc._id === 'pet';
  env.failures.delete = () => true;
  await assert.rejects(applyNew(env.copy), /恢复没有做完|没能删掉/);
  assert.equal(env.actors.has('source'), true);
  assert.match(env.messages.at(-1).content, /source/);
});
