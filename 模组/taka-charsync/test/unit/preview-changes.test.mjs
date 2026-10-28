import test from 'node:test';
import assert from 'node:assert/strict';
import {applyCopy, previewPending, runApplyCheck} from '../../scripts/main/apply.mjs';
import {actorDoc, applyEnvironment, makeCopy} from '../helpers/apply-environment.mjs';

test('force preview includes actor-only changes and companions while preserving main XP', async () => {
  const env = applyEnvironment();
  const actor = env.mapped('source');
  env.mapped('pet', 'main-pet', {companionOf: 'source'});
  const doc = actorDoc('source', {name: 'Renamed'});
  doc.system.details.biography = {appearance: 'scarlet cloak'};
  doc.system.attributes.hp.value = 7;
  doc.system.details.xp.value = 999;
  doc.effects = [{_id: 'effect', name: 'Glow', type: 'base'}];
  const copy = env.publishCopy(makeCopy('source', 'force', {name: 'Renamed',
    companions: [{sourceId: 'pet', copyId: 'pet-copy', rev: 200}]}), doc);
  const pet = actorDoc('pet');
  pet.system.details.biography = {appearance: 'silver familiar'};
  env.publishCopy(makeCopy('pet', 'force', {companionOf: copy.copyId}), pet);
  const html = await previewPending({actorId: actor.id, copy});
  assert.match(html, /Renamed/);
  assert.match(html, /scarlet cloak/);
  assert.match(html, /Glow/);
  assert.match(html, /silver familiar/);
  assert.match(html, /同伴/);
  assert.doesNotMatch(html, /999/);
  assert.deepEqual(env.writes, []);
});

test('same-level preview and apply use the three-way result for owner and companion', async () => {
  const env = applyEnvironment();
  const actor = env.mapped('source');
  const pet = env.mapped('pet', 'main-pet', {companionOf: 'source'});
  actor.system.attributes.hp.value = 4;
  actor.system.details.xp.value = 750;
  const doc = actorDoc('source');
  doc.system.details.biography = {appearance: 'blue coat'};
  doc.system.attributes.hp.value = 20;
  const petDoc = actorDoc('pet');
  petDoc.system.details.biography = {appearance: 'tiny owl'};
  const copy = env.publishCopy(makeCopy('source', 'level', {companions: [{sourceId: 'pet', copyId: 'pet-copy', rev: 200}]}), doc);
  env.publishCopy(makeCopy('pet', 'level', {companionOf: copy.copyId}), petDoc);
  const check = await runApplyCheck({trigger: 'tick'});
  assert.equal(check.pending[0].reason, 'same-level');
  assert.deepEqual(env.writes, []);
  const html = await previewPending({actorId: actor.id, copy});
  assert.match(html, /blue coat/);
  assert.match(html, /tiny owl/);
  await applyCopy(actor, copy, {manual: true});
  assert.equal(actor.system.attributes.hp.value, 4);
  assert.equal(actor.system.details.xp.value, 750);
  assert.equal(actor.system.details.biography.appearance, 'blue coat');
  assert.equal(pet.system.details.biography.appearance, 'tiny owl');
});

test('new-card preview names every card to be created, including an itemless familiar', async () => {
  const env = applyEnvironment();
  CONFIG.PF2E = {Actor: {documentClasses: {familiar: Actor.implementation}}};
  const copy = env.publishCopy(makeCopy('source', 'new', {name: 'New hero', pred: null,
    companions: [{sourceId: 'pet', copyId: 'pet-copy', rev: 200}]}));
  env.publishCopy(makeCopy('pet', 'new', {name: 'Little bat', pred: null, companionOf: copy.copyId}),
    actorDoc('pet', {type: 'familiar', system: {master: {id: copy.copyId}}}));
  const html = await previewPending({actorId: null, copy});
  assert.match(html, /New hero/);
  assert.match(html, /Little bat/);
  assert.match(html, /新增/);
  assert.deepEqual(env.writes, []);
});

test('same-level preview preserves source HP when its maximum does not change', async () => {
  const env = applyEnvironment();
  const actor = env.mapped('source');
  actor.system.attributes.hp.value = 56;
  const doc = actorDoc('source');
  doc.system.details.biography = {appearance: 'blue coat'};
  const copy = env.publishCopy(makeCopy('source', 'level'), doc);
  const html = await previewPending({actorId: actor.id, copy});
  assert.match(html, /blue coat/);
  assert.doesNotMatch(html, /56 → 20/);
  assert.deepEqual(env.writes, []);
  await applyCopy(actor, copy, {manual: true});
  assert.equal(actor._source.system.attributes.hp.value, 56);
});

test('familiar HP preview uses the upgraded temporary master and matches applying the group', async () => {
  const env = applyEnvironment();
  const Document = Actor.implementation;
  class Character extends Document {
    get level() { return this.system.details.level.value; }
    async update(...args) {
      const updated = await super.update(...args);
      for (const companion of env.actors.values()) {
        if (companion.type === 'familiar' && companion._source.system.master.id === this.id) companion.refresh();
      }
      return updated;
    }
  }
  class Familiar extends Document {
    get master() {
      const master = env.actors.get(this.system.master.id);
      if (master) master.familiar ??= this;
      return master ?? null;
    }
    refresh() {
      super.refresh();
      this.system = structuredClone(this._source.system);
      this.system.attributes.hp.max = (this.master?.level ?? 0) * 5;
    }
  }
  globalThis.CONFIG.PF2E = {Actor: {documentClasses: {character: Character, familiar: Familiar}}};
  globalThis.Actor.implementation = class {
    constructor(data) { return new (data.type === 'familiar' ? Familiar : Character)(data); }
  };
  const base = actorDoc('main-source');
  base.system.details.level.value = 5;
  const petBase = actorDoc('main-pet', {type: 'familiar', system: {
    master: {id: 'main-source'}, attributes: {hp: {value: 15}},
  }});
  env.mapped('source');
  env.mapped('pet', 'main-pet', {companionOf: 'source'});
  const actor = new Character(base);
  env.actors.set(actor.id, actor);
  const pet = new Familiar(petBase);
  env.actors.set(pet.id, pet);
  actor.familiar = null;
  env.files.set('history/main-source-base.json', base);
  env.files.set('history/main-pet-base.json', petBase);
  const upgraded = structuredClone(base);
  upgraded.system.details.level.value = 6;
  const copy = env.publishCopy(makeCopy('source', 'level', {level: 6,
    companions: [{sourceId: 'pet', copyId: 'pet-copy', rev: 200}]}), upgraded);
  env.publishCopy(makeCopy('pet', 'level', {level: 6, companionOf: copy.copyId}), petBase);
  const before = [actor.toObject(), pet.toObject()];
  const html = await previewPending({actorId: actor.id, copy});
  const familiarSection = html.split('<section>').find(section => section.includes('更新同伴'));
  assert.match(familiarSection, /15 → 20/);
  assert.deepEqual([actor.toObject(), pet.toObject()], before);
  assert.equal(actor.familiar, null, 'preview construction must not cache a temporary familiar on the live master');
  assert.equal(env.actors.get(actor.id), actor);
  assert.equal(env.actors.get(pet.id), pet);
  assert.deepEqual(env.writes, []);
  await applyCopy(actor, copy, {manual: true});
  assert.equal(pet.system.attributes.hp.value, 20);
  assert.equal(pet.system.attributes.hp.max, 30);
});

test('new-card pending lists the owner group once and never lists its frozen familiar separately', async () => {
  const env = applyEnvironment();
  env.meta.working = {source: {owner: 'char-player'}, pet: {owner: 'char-player'}};
  const copy = env.publishCopy(makeCopy('source', 'new', {pred: null,
    companions: [{sourceId: 'pet', copyId: 'pet-copy', rev: 200}]}));
  env.publishCopy(makeCopy('pet', 'new', {pred: null, companionOf: copy.copyId}),
    actorDoc('pet', {type: 'familiar', system: {master: {id: copy.copyId}}}));
  const result = await runApplyCheck({trigger: 'tick'});
  assert.deepEqual(result.pending.map(item => item.copy.cardId), ['source']);
  assert.deepEqual(result.pending[0].copy.companions, [{sourceId: 'pet', copyId: 'pet-copy', rev: 200}]);
  assert.deepEqual(env.writes, []);
});
