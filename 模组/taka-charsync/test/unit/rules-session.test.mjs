import test from 'node:test';
import assert from 'node:assert/strict';
import {makeCampaignEnvironment, ruleSnapshot} from '../helpers/campaign-environment.mjs';

test('campaign session blocks unsafe edits and submits until current rules are prepared', async t => {
  const {api, a, b, Character, Familiar} = await makeCampaignEnvironment({isolated: false});
  a.name = 'A'; b.name = 'B';
  a.isOwner = b.isOwner = true;
  a.documentName = b.documentName = 'Actor';
  const missing = new Character('missing');
  missing.documentName = 'Actor';
  missing.name = 'Unassigned';
  game.actors.set(missing.id, missing);
  const get = game.settings.get.bind(game.settings);
  game.settings.get = (namespace, key, options) => namespace === 'taka-charsync'
    ? 'test-published' : get(namespace, key, options);
  foundry.utils.getRoute = value => value;
  let meta = {ruleSnapshot: ruleSnapshot()};
  let unavailable = false;
  globalThis.fetch = async url => {
    assert.equal(url, 'test-published/meta.json');
    if (unavailable) throw new Error('network unavailable');
    return {ok: true, status: 200, json: async () => structuredClone(meta)};
  };
  const hooks = new Map();
  Hooks.on = (name, handler) => {
    if (!hooks.has(name)) hooks.set(name, []);
    hooks.get(name).push(handler);
  };
  const call = (name, ...args) => hooks.get(name)?.map(handler => handler(...args));
  globalThis.ui = {notifications: {warn() {}}};
  const session = await import('../../scripts/char/rules-session.mjs');
  session.installRuleEditGuards();

  await t.test('current assigned card receives its hash; wrong targets and companions are rejected', async () => {
    const accepted = await session.prepareSubmissionRules(a, 'a');
    assert.equal(accepted.rulesHash, 'a'.repeat(64));
    await assert.rejects(session.prepareSubmissionRules(a, 'b'), /所属团/);
    await assert.rejects(session.prepareSubmissionRules(missing, 'a'), /尚未指定所属团/);
    await assert.rejects(session.prepareSubmissionRules(a, 'a', [b]), /所属团/);
  });

  await t.test('a profile change forces a review before a subsequent submission', async () => {
    const next = ruleSnapshot();
    next.generatedAt++;
    next.rules.a['pf2e.freeArchetypeVariant'] = false;
    next.hashes.a = 'c'.repeat(64);
    meta = {ruleSnapshot: next};
    await assert.rejects(session.prepareSubmissionRules(a, 'a'), /刚更新/);
    assert.equal(a.prepared.fa, false);
    const accepted = await session.prepareSubmissionRules(a, 'a');
    assert.equal(accepted.rulesHash, 'c'.repeat(64));
  });

  await t.test('unavailable publication blocks construction and recovers on successful read', async () => {
    unavailable = true;
    await assert.rejects(session.loadRuleMeta(), /network unavailable/);
    assert.equal(api.actorRuleState(a).ok, false);
    assert.ok(call('preUpdateActor', a, {'system.details.level.value': 7}, {}).includes(false));
    assert.ok(!call('preUpdateActor', missing, {folder: 'assigned-folder'}, {}).includes(false));
    assert.ok(call('preCreateItem', {parent: missing, documentName: 'Item'}, {}, {}).includes(false));
    assert.ok(call('preDeleteActiveEffect', {parent: missing, documentName: 'ActiveEffect'}, {}).includes(false));
    assert.ok(call('preDeleteActiveEffect', {
      parent: {parent: missing, documentName: 'Item'}, documentName: 'ActiveEffect',
    }, {}).includes(false));
    assert.ok(!call('preUpdateActor', missing, {system: {}}, {takaCharsync: true}).includes(false));
    unavailable = false;
    await session.loadRuleMeta();
    assert.equal(api.actorRuleState(a).ok, true);
  });

  await t.test('an unlinked familiar can repair its master without smuggling construction edits', () => {
    const familiar = new Familiar('unlinked-pet', null);
    familiar.documentName = 'Actor';
    for (const isGM of [true, false]) {
      game.user.isGM = isGM;
      assert.ok(!call('preUpdateActor', familiar, {'system.master.id': 'A'}, {}).includes(false));
      assert.ok(!call('preUpdateActor', familiar, {system: {master: {id: 'A', ability: 'cha'}}}, {}).includes(false));
      assert.ok(call('preUpdateActor', familiar, {'system.master.id': 'missing'}, {}).includes(false));
      assert.ok(call('preUpdateActor', familiar, {'system.master.id': 'A', 'system.attributes.hp.value': 99}, {}).includes(false));
    }
  });

  await t.test('folder and master changes reprepare dependent actors without waiting for a profile update', async () => {
    const next = structuredClone(meta.ruleSnapshot);
    next.generatedAt++;
    next.cardWorld = {};
    next.folderWorld = {folderA: 'a', folderB: 'b'};
    meta = {ruleSnapshot: next};
    game.folders.set('folderA', {_id: 'folderA', folder: null});
    game.folders.set('folderB', {_id: 'folderB', folder: null});
    a._source.folder = 'folderA';
    b._source.folder = 'folderB';
    const familiar = new Familiar('pet', 'A');
    game.actors.set(familiar.id, familiar);
    await session.loadRuleMeta();
    a._source.folder = 'folderB';
    call('updateActor', a, {folder: 'folderB'}, {});
    assert.equal(a.prepared.gradual, false);
    assert.equal(familiar.prepared.gradual, false);
    game.folders.set('child', {_id: 'child', folder: 'folderA'});
    a._source.folder = 'child';
    call('updateActor', a, {folder: 'child'}, {});
    assert.equal(a.prepared.gradual, true);
    game.folders.get('child').folder = 'folderB';
    call('updateFolder', game.folders.get('child'), {folder: 'folderB'}, {});
    assert.equal(a.prepared.gradual, false);
  });
});
