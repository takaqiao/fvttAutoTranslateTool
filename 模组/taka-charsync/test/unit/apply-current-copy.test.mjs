import test from 'node:test';
import assert from 'node:assert/strict';
import {applyCopy, applyForce, applyNew, applyPending} from '../../scripts/main/apply.mjs';
import {OWN_FLAG} from '../../scripts/lib/rules.mjs';
import {actorDoc, applyEnvironment, makeCopy, refOf} from '../helpers/apply-environment.mjs';

for (const kind of ['level', 'force', 'new']) {
  for (const state of ['withdrawn', 'replaced', 'missing', 'wrong-owner', 'processed']) {
    test(`${kind} refuses ${state} copies before changing any actor`, async () => {
      const env = applyEnvironment();
      const actor = kind === 'new' ? null : env.mapped('source');
      const copy = env.publishCopy(makeCopy('source', kind));
      const latest = {...copy};
      if (state === 'withdrawn') latest.withdrawn = true;
      if (state === 'replaced') latest.rev++;
      if (state === 'wrong-owner') {
        if (kind === 'new') env.meta.working.source = {owner: 'somebody-else'};
        else env.meta.cards.source.mainId = 'another-main';
      }
      if (state === 'processed') {
        if (kind === 'new') env.addActor({_id: 'imported', flags: {[OWN_FLAG]: {charId: 'source'}}, system: {}});
        else actor._source.flags[OWN_FLAG] = {applied: {ref: refOf(copy)}};
      }
      env.files.set('copies/source.json', state === 'missing' ? [] : [latest]);
      const apply = kind === 'new' ? () => applyNew(copy) : kind === 'force' ? () => applyForce(actor, copy) : () => applyCopy(actor, copy, {manual: true});
      let rejected = false;
      try { rejected = (await apply()) === 'already'; } catch (error) { rejected = /刷新|撤回|归属|已经|已处理|对应|卡主/.test(error.message); }
      assert.equal(rejected, true);
      assert.deepEqual(env.writes, []);
    });
  }
}

for (const state of ['missing', 'revision', 'rules', 'master', 'mapping']) {
  test(`a ${state} companion prevents the master from being written`, async () => {
    const env = applyEnvironment();
    const actor = env.mapped('source');
    env.mapped('pet', 'main-pet', {companionOf: state === 'mapping' ? 'other-master' : 'source'});
    const pet = makeCopy('pet', 'force', {companionOf: state === 'master' ? 'other-copy' : 'source-copy'});
    const copy = env.publishCopy(makeCopy('source', 'force', {companions: [{sourceId: 'pet', copyId: pet.copyId, rev: 200}]}));
    if (state !== 'missing') env.publishCopy({...pet, rev: state === 'revision' ? 201 : 200, rulesHash: state === 'rules' ? 'b'.repeat(64) : pet.rulesHash});
    await assert.rejects(applyForce(actor, copy), /同伴|规则/);
    assert.deepEqual(env.writes, []);
  });
}

test('an old lower-level pending button cannot downgrade a card after a newer submission was applied', async () => {
  const env = applyEnvironment();
  const actor = env.mapped('source');
  actor.system.details.level.value = 6;
  actor._source.flags[OWN_FLAG] = {applied: {ref: {kind: 'copy', copyId: 'newer6', rev: 600}}};
  const base = actorDoc(actor.id);
  base.system.details.level.value = 4;
  env.files.set('history/main-source-base.json', base);
  const submitted = actorDoc('source');
  submitted.system.details.level.value = 5;
  const copy = env.publishCopy(makeCopy('source', 'level', {level: 5, rev: 500}), submitted);
  const before = actor.toObject();
  await assert.rejects(applyPending({actorId: actor.id, reason: 'session', copy}), /主服等级已提高.*刷新同步面板/);
  assert.deepEqual(env.writes, []);
  assert.deepEqual(actor.toObject(), before);
});
