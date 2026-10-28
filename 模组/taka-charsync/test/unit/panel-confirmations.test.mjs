import test from 'node:test';
import assert from 'node:assert/strict';
import {runApplyCheck} from '../../scripts/main/apply.mjs';
import {OWN_FLAG} from '../../scripts/lib/rules.mjs';
import {actorDoc, applyEnvironment, makeCopy} from '../helpers/apply-environment.mjs';

class Application {
  async render() { this.body = await this._renderHTML(); this.rendered = true; return this; }
  bringToFront() {}
  element = {querySelectorAll: () => []};
}

async function show(env) {
  const confirmations = [];
  foundry.applications.api = {ApplicationV2: Application, DialogV2: {
    async confirm(options) { confirmations.push(options); return false; },
  }};
  globalThis.document = {createElement: () => ({innerHTML: ''}), querySelectorAll: () => []};
  await runApplyCheck({trigger: 'tick'});
  const {openPanel} = await import('../../scripts/main/panel.mjs');
  const panel = await openPanel();
  async function click(label) {
    const buttons = [...panel.body.innerHTML.matchAll(/<button[^>]*data-index="(\d+)"[^>]*>(.*?)<\/button>/g)];
    const button = buttons.find(([, , text]) => text === label);
    assert.ok(button, `missing button: ${label}`);
    await panel.runAction({dataset: {index: button[1]}});
  }
  return {confirmations, click};
}

test('adding or replacing a new submission confirms the new group and old archive scope', async () => {
  const env = applyEnvironment();
  env.meta.working.source = {owner: 'char-player', name: 'New hero'};
  const old = env.mapped('old', 'old-main');
  env.mapped('old-pet', 'old-pet-main', {companionOf: 'old'});
  const copy = env.publishCopy(makeCopy('source', 'new', {pred: null, replaces: 'old', name: 'New hero',
    companions: [{sourceId: 'pet', copyId: 'pet-copy', rev: 200}]}));
  env.publishCopy(makeCopy('pet', 'new', {pred: null, name: 'New pet', companionOf: copy.copyId}));
  const {click, confirmations} = await show(env);
  await click('新增');
  assert.equal(confirmations.length, 1);
  assert.match(confirmations[0].content, /New hero/);
  assert.match(confirmations[0].content, /New pet/);
  assert.deepEqual(env.writes, []);
  await click(`替换〈${old.name}〉`);
  assert.equal(confirmations.length, 2);
  assert.match(confirmations[1].content, /old-main/);
  assert.match(confirmations[1].content, /old-pet-main/);
  assert.match(confirmations[1].content, /归档/);
  assert.deepEqual(env.writes, []);
});

test('restoring an applied group names restored and removed companions before acting', async () => {
  const env = applyEnvironment();
  const actor = env.mapped('source');
  env.mapped('pet', 'main-pet', {companionOf: 'source'});
  env.addActor(actorDoc('new-pet'));
  actor._source.flags[OWN_FLAG] = {applied: {at: 123, undoPath: 'undo.json', companions: ['main-pet'], createdCompanions: ['new-pet']}};
  const {click, confirmations} = await show(env);
  await click('恢复到套用前');
  assert.equal(confirmations.length, 1);
  assert.match(confirmations[0].content, /main-pet/);
  assert.match(confirmations[0].content, /new-pet/);
  assert.match(confirmations[0].content, /删除/);
  assert.deepEqual(env.writes, []);
});

test('importing a working card confirms its familiar before creating anything', async () => {
  const env = applyEnvironment();
  env.meta.working.source = {name: 'Working hero', owner: 'char-player', hash: 'hero-hash'};
  env.meta.working.pet = {name: 'Working pet', owner: 'char-player', hash: 'pet-hash'};
  env.files.set('working/source.json', {doc: actorDoc('source')});
  env.files.set('working/pet.json', {doc: actorDoc('pet', {type: 'familiar', system: {master: {id: 'source'}}})});
  const {click, confirmations} = await show(env);
  await click('新增');
  assert.equal(confirmations.length, 1);
  assert.match(confirmations[0].content, /Working pet|pet/);
  assert.deepEqual(env.writes, []);
});
