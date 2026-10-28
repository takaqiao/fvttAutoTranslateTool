import test from 'node:test';
import assert from 'node:assert/strict';
import {applyForce, runApplyCheck} from '../../scripts/main/apply.mjs';
import {actorDoc, applyEnvironment, makeCopy} from '../helpers/apply-environment.mjs';

class Application {
  async render() { this.body = await this._renderHTML(); this.rendered = true; return this; }
  bringToFront() {}
  element = {querySelectorAll: () => []};
}

test('the panel retains companion module warnings after apply and opens native management without changing settings', async () => {
  const env = applyEnvironment();
  const actor = env.mapped('source');
  env.mapped('pet', 'main-pet', {companionOf: 'source'});
  game.user = {isGM: true};
  game.modules = new Map([['pet-addon', {id: 'pet-addon', title: '<Pet addon>', active: false}]]);
  const source = actorDoc('pet', {items: [{_id: 'pet-feat', name: 'Pet feat', type: 'feat', flags: {}, effects: [],
    system: {rules: [{key: 'GrantItem', uuid: 'Compendium.pet-addon.items.Item.abcdefghijklmnop'}]}}]});
  const copy = env.publishCopy(makeCopy('source', 'force', {
    companions: [{sourceId: 'pet', copyId: 'pet-copy', rev: 200}]}));
  env.publishCopy(makeCopy('pet', 'force', {companionOf: copy.copyId}), source);
  await applyForce(actor, copy);
  let opened = 0, settingWrites = 0;
  game.settings.set = async () => { settingWrites++; };
  foundry.applications.api = {ApplicationV2: Application, DialogV2: {}};
  foundry.applications.sidebar = {apps: {ModuleManagement: class {
    render(options) { assert.equal(options.force, true); opened++; }
  }}};
  globalThis.document = {createElement: () => ({innerHTML: ''}), querySelectorAll: () => []};
  await runApplyCheck({trigger: 'tick'});
  const {openPanel} = await import('../../scripts/main/panel.mjs');
  const panel = await openPanel();
  assert.match(panel.body.innerHTML, /pet-addon/);
  assert.match(panel.body.innerHTML, /未启用/);
  assert.doesNotMatch(panel.body.innerHTML, /<Pet addon>/);
  const button = [...panel.body.innerHTML.matchAll(/<button[^>]*data-index="(\d+)"[^>]*>(.*?)<\/button>/g)]
    .find(([, , label]) => label === '打开模组管理');
  assert.ok(button);
  const writesBefore = env.writes.length;
  await panel.runAction({dataset: {index: button[1]}});
  assert.equal(opened, 1);
  assert.equal(settingWrites, 0);
  assert.equal(env.writes.length, writesBefore);
  game.modules.get('pet-addon').active = true;
  await openPanel();
  assert.doesNotMatch(panel.body.innerHTML, /pet-addon/);
  game.modules.get('pet-addon').active = false;
  env.actors.get('main-pet')._source.items = [];
  await openPanel();
  assert.doesNotMatch(panel.body.innerHTML, /pet-addon/, 'removed content must not keep a stale warning');
});
