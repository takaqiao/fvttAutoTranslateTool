import test from 'node:test';
import assert from 'node:assert/strict';
import {OWN_FLAG} from '../../scripts/lib/rules.mjs';
import {makeCampaignEnvironment, ruleSnapshot} from '../helpers/campaign-environment.mjs';

test('each submission freezes separate module evidence for the master and companion', async t => {
  const {a, Familiar} = await makeCampaignEnvironment({isolated: false});
  const {loadRuleMeta} = await import('../../scripts/char/rules-session.mjs');
  const {submitNew, submitLevel, submitForce} = await import('../../scripts/char/submit.mjs');
  game.user = {id: 'player', isGM: false};
  globalThis.CONST = {DOCUMENT_OWNERSHIP_LEVELS: {OWNER: 3}};
  game.actors[Symbol.iterator] = game.actors.values;
  game.actors.filter = predicate => [...game.actors.values()].filter(predicate);
  game.folders.find = predicate => [...game.folders.values()].find(predicate);
  for (const role of ['submitted', 'withdrawn']) {
    game.folders.set(role, {id: role, type: 'Actor', getFlag: () => role});
  }
  game.folders.set('work', {id: 'work', type: 'Actor', folder: null, getFlag: () => null});
  foundry.utils.getRoute = value => value;
  const get = game.settings.get.bind(game.settings);
  game.settings.get = (namespace, key, options) => namespace === OWN_FLAG ? 'test-published' : get(namespace, key, options);
  let meta;
  const writes = [];
  globalThis.Actor = {create: async data => { writes.push(structuredClone(data)); return data; }};
  globalThis.fetch = async url => {
    assert.equal(url, 'test-published/meta.json');
    return {ok: true, status: 200, json: async () => structuredClone(meta)};
  };

  function document(actor, name) {
    actor.name = name;
    actor._source = {
      _id: actor.id, name, type: actor.type, folder: 'work', ownership: {player: 3},
      flags: {pf2e: {}, [OWN_FLAG]: {base: {kind: 'main', hash: 'base'}}},
      system: {...actor._source.system, details: {level: {value: 6}}}, items: [], effects: [],
    };
    actor.flags = actor._source.flags;
    actor.toObject = () => structuredClone(actor._source);
    return actor;
  }

  for (const [kind, submit] of [
    ['new', () => submitNew(a, {targetWorld: 'a'})],
    ['level', () => submitLevel(a)],
    ['force', () => submitForce(a)],
  ]) {
    await t.test(`${kind} saves the source identities and does not follow later draft edits`, async () => {
      writes.length = 0;
      const pet = document(new Familiar('pet', a.id), '魔宠');
      document(a, '主人');
      a._source.items.push({
        _id: 'feat', name: '主人专长', type: 'feat',
        system: {rules: [{key: 'GrantItem', uuid: 'Compendium.master-module.items.Item.effect'}]},
      });
      pet._source.flags['pet-module'] = {enabled: true};
      game.modules = new Map([
        ['master-module', {id: 'master-module', title: '主人模块'}],
        ['pet-module', {id: 'pet-module', title: '魔宠模块'}],
      ]);
      game.actors.clear();
      for (const actor of [a, pet]) game.actors.set(actor.id, actor);
      const snapshot = ruleSnapshot();
      snapshot.folderWorld = {work: 'a'};
      if (kind === 'new') delete snapshot.cardWorld.A;
      meta = {
        ruleSnapshot: snapshot, users: {player: {a: 'main-player'}},
        cards: kind === 'new' ? {} : {A: {world: 'a', mainId: 'main-A', charOwner: 'player'}},
      };
      await loadRuleMeta();
      const copy = await submit();
      const [petCopy, masterCopy] = writes.map(data => data.flags[OWN_FLAG].copy);
      assert.deepEqual(copy.moduleUsage, [{
        id: 'master-module', title: '主人模块', evidence: [{
          kind: 'reference', documentId: 'feat', documentName: '主人专长', path: 'items.0.system.rules.0.uuid',
        }],
      }]);
      assert.deepEqual(masterCopy.moduleUsage, copy.moduleUsage);
      assert.deepEqual(petCopy.moduleUsage, [{
        id: 'pet-module', title: '魔宠模块', evidence: [{
          kind: 'flag', documentId: 'pet', documentName: '魔宠', path: 'flags.pet-module',
        }],
      }]);
      assert.equal(masterCopy.rulesHash, snapshot.hashes.a);
      assert.equal(petCopy.companionOf, copy.copyId);
      const frozen = structuredClone(writes);
      a._source.items[0].system.rules[0].uuid = 'Compendium.changed.items.Item.other';
      delete pet._source.flags['pet-module'];
      game.modules.get('master-module').title = '新名称';
      assert.deepEqual(writes, frozen);
      assert.deepEqual(copy.moduleUsage, masterCopy.moduleUsage);
      assert.equal(a._source.flags[OWN_FLAG].moduleUsage, undefined);
    });
  }
});
