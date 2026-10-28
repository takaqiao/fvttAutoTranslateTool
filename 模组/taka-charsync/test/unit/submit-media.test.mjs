import test from 'node:test';
import assert from 'node:assert/strict';
import {OWN_FLAG} from '../../scripts/lib/rules.mjs';
import {makeCampaignEnvironment, ruleSnapshot} from '../helpers/campaign-environment.mjs';

test('a missing master media dependency or failed upload aborts every copy write, including companions', async () => {
  const {a, Familiar} = await makeCampaignEnvironment({isolated: false});
  const {loadRuleMeta} = await import('../../scripts/char/rules-session.mjs');
  const {submitNew} = await import('../../scripts/char/submit.mjs');
  game.user = {id: 'player', isGM: false};
  globalThis.CONST = {DOCUMENT_OWNERSHIP_LEVELS: {OWNER: 3}};
  game.actors[Symbol.iterator] = game.actors.values;
  game.actors.filter = predicate => [...game.actors.values()].filter(predicate);
  game.folders.find = predicate => [...game.folders.values()].find(predicate);
  game.folders.set('submitted', {id: 'submitted', type: 'Actor', getFlag: () => 'submitted'});
  game.folders.set('work', {id: 'work', type: 'Actor', folder: null, getFlag: () => null});
  foundry.utils.getRoute = value => value;
  const get = game.settings.get.bind(game.settings);
  game.settings.get = (namespace, key, options) => namespace === OWN_FLAG ? 'test-published' : get(namespace, key, options);
  const snapshot = ruleSnapshot();
  delete snapshot.cardWorld.A;
  snapshot.folderWorld = {work: 'a'};
  const meta = {ruleSnapshot: snapshot, cards: {}, users: {player: {a: 'main-player'}}};
  const writes = [];
  globalThis.Actor = {create: async data => { writes.push(data); return data; }};
  globalThis.fetch = async url => url === 'test-published/meta.json'
    ? {ok: true, status: 200, json: async () => structuredClone(meta)}
    : {ok: false, status: 404};
  const pet = new Familiar('pet', a.id);
  game.actors.clear();
  for (const actor of [a, pet]) {
    actor.name = actor.id;
    Object.assign(actor._source, {name: actor.name, folder: 'work', ownership: {player: 3}, items: [], effects: []});
    actor.toObject = () => structuredClone(actor._source);
    game.actors.set(actor.id, actor);
  }
  a._source.flags['tokenizer-2'] = {layerStack: [{masks: [{src: 'assets/missing-mask.png'}]}]};
  await loadRuleMeta();
  await assert.rejects(submitNew(a, {targetWorld: 'a'}), /404|媒体|图片/);
  assert.deepEqual(writes, []);
  globalThis.fetch = async url => {
    if (url === 'test-published/meta.json') return {ok: true, status: 200, json: async () => structuredClone(meta)};
    if (url === 'assets/missing-mask.png') return {ok: true, status: 200, arrayBuffer: async () => new TextEncoder().encode('mask').buffer};
    return {ok: false, status: 404};
  };
  foundry.applications.apps.FilePicker = {implementation: {upload: async () => false}};
  await assert.rejects(submitNew(a, {targetWorld: 'a'}), /媒体保存失败.*同步引擎.*文件上传权限/);
  assert.deepEqual(writes, []);
});
