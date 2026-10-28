import test from 'node:test';
import assert from 'node:assert/strict';
import {copyIdFor, forceCopyIdFor} from '../../scripts/lib/chain.mjs';
import {OWN_FLAG} from '../../scripts/lib/rules.mjs';
import {makeCampaignEnvironment, ruleSnapshot} from '../helpers/campaign-environment.mjs';

test('companion discovery works before any card mappings have been published', async () => {
  const {companionsOf} = await import('../../scripts/char/submit.mjs');
  const master = {id: 'master', name: 'Master'};
  const pet = {id: 'pet', type: 'familiar', _source: {system: {master: {id: 'master'}}}};
  const actors = [pet];
  actors.get = id => actors.find(actor => actor.id === id);
  globalThis.game = {actors};
  assert.equal(typeof companionsOf, 'function');
  assert.deepEqual(companionsOf(master, {}), [pet]);
});

test('submission entry points freeze only cards prepared with current campaign rules', async t => {
  const {a, b, Character, Familiar} = await makeCampaignEnvironment({isolated: false});
  const session = await import('../../scripts/char/rules-session.mjs');
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
  game.settings.get = (namespace, key, options) => namespace === OWN_FLAG
    ? 'test-published' : get(namespace, key, options);

  let meta, generation = 100;
  const writes = [];
  const serialized = [];
  globalThis.fetch = async url => {
    assert.equal(url, 'test-published/meta.json');
    return {ok: true, status: 200, json: async () => structuredClone(meta)};
  };

  function document(actor) {
    actor.name = actor.id;
    actor._source = {
      _id: actor.id, name: actor.name, type: actor.type,
      ownership: {player: 3}, flags: {pf2e: {}, [OWN_FLAG]: {base: {kind: 'main', hash: 'base'}}},
      system: {...actor._source.system, details: {level: {value: 6}}}, items: [], effects: [],
    };
    actor.flags = actor._source.flags;
    actor.toObject = () => {
      serialized.push(actor.id);
      return structuredClone(actor._source);
    };
    for (const key of ['items', 'effects']) {
      actor[key] = new Map();
      actor[key].invalidDocumentIds = new Set();
    }
    actor.updateSource = () => {};
    actor.update = async data => {
      writes.push({operation: 'update', id: actor.id, data: structuredClone(data)});
      return actor;
    };
    return actor;
  }

  globalThis.Actor = {
    async create(data, options) {
      writes.push({operation: 'create', id: data._id, data: structuredClone(data), options});
      const copy = document(new Character(data._id));
      copy.type = data.type;
      copy._source = structuredClone(data);
      copy.flags = copy._source.flags;
      game.actors.set(copy.id, copy);
      return copy;
    },
  };

  async function prepare(kind, {existing = false} = {}) {
    writes.length = serialized.length = 0;
    game.actors.clear();
    for (const actor of [a, b]) game.actors.set(actor.id, document(actor));
    const pet = document(new Familiar('pet', a.id));
    const companions = [pet];
    for (const actor of companions) game.actors.set(actor.id, actor);
    const snapshot = ruleSnapshot();
    snapshot.generatedAt = ++generation;
    snapshot.folderWorld = {work: 'a'};
    a._source.folder = 'work';
    meta = {
      ruleSnapshot: snapshot, users: {player: {a: 'main-player', b: 'other-player'}},
      cards: kind === 'new' ? {} : {
        A: {world: 'a', mainId: 'main-A', charOwner: 'player'},
        pet: {world: 'a', mainId: 'main-pet', charOwner: 'player', companionOf: a.id},
      },
    };
    if (kind === 'new') delete snapshot.cardWorld.A;
    if (existing) {
      for (const actor of [...companions, a]) {
        const id = kind === 'force' ? await forceCopyIdFor(actor.id) : await copyIdFor(actor.id, 6);
        const copy = document(new Character(id));
        copy._source.flags[OWN_FLAG] = {copy: {copyId: id, cardId: actor.id, kind, level: 6, targetWorld: 'a'}};
        game.actors.set(id, copy);
      }
    }
    await session.loadRuleMeta();
    return {pet, snapshot};
  }

  const entryPoints = [
    ['new', () => submitNew(a, {targetWorld: 'a'})],
    ['level', () => submitLevel(a)],
    ['force', () => submitForce(a)],
  ];

  for (const [kind, submit] of entryPoints) {
    await t.test(`${kind} rejects a changed profile before creating or updating any copy`, async () => {
      const {snapshot} = await prepare(kind, {existing: true});
      snapshot.generatedAt = ++generation;
      snapshot.rules.a['pf2e.freeArchetypeVariant'] = false;
      snapshot.hashes.a = 'c'.repeat(64);
      await assert.rejects(submit(), /所属团规则刚更新/);
      assert.equal(a.prepared.fa, false);
      assert.deepEqual(writes, []);
      assert.deepEqual(serialized, []);
    });

    await t.test(`${kind} rejects a missing assignment before creating or updating any copy`, async () => {
      const {snapshot} = await prepare(kind, {existing: true});
      delete snapshot.cardWorld.A;
      a._source.folder = null;
      await session.loadRuleMeta();
      await assert.rejects(submit(), /尚未指定所属团/);
      assert.deepEqual(writes, []);
    });

    await t.test(`${kind} rejects an unpublished campaign before creating or updating any copy`, async () => {
      const {snapshot} = await prepare(kind, {existing: true});
      delete snapshot.rules.a;
      delete snapshot.hashes.a;
      await session.loadRuleMeta();
      await assert.rejects(submit(), /所属团规则尚未发布/);
      assert.deepEqual(writes, []);
    });

    await t.test(`${kind} rejects a different assigned world before creating or updating any copy`, async () => {
      const {snapshot} = await prepare(kind, {existing: true});
      snapshot.cardWorld.A = 'b';
      snapshot.folderWorld.work = 'b';
      await session.loadRuleMeta();
      await assert.rejects(submit(), /所属团与提交目标不同/);
      assert.deepEqual(writes, []);
    });

    await t.test(`${kind} rejects a missing target world before creating any copy`, async () => {
      await prepare(kind);
      if (kind !== 'new') delete meta.cards.A.world;
      await assert.rejects(kind === 'new' ? submitNew(a, {}) : submit(), /目标|所属团|对应的账号/);
      assert.deepEqual(writes, []);
    });

    await t.test(`${kind} saves the current hash and companion references`, async () => {
      const {snapshot} = await prepare(kind);
      snapshot.generatedAt = ++generation;
      snapshot.hashes.a = 'd'.repeat(64);
      await session.loadRuleMeta();
      const copy = await submit();
      const sources = ['pet', a.id];
      assert.deepEqual(writes.map(write => write.data.flags[OWN_FLAG].copy.cardId), sources);
      assert.deepEqual(serialized, sources);
      for (const write of writes) {
        assert.equal(write.operation, 'create');
        assert.deepEqual(write.options, {keepId: true, takaCharsync: true});
        assert.equal(write.data.folder, 'submitted');
        assert.equal(write.data.flags[OWN_FLAG].copy.rulesHash, snapshot.hashes.a);
        assert.equal(write.data.flags[OWN_FLAG].copy.targetWorld, 'a');
        assert.equal(write.data.flags[OWN_FLAG].copy.rev, copy.rev);
      }
      assert.deepEqual(writes.at(-1).data.flags[OWN_FLAG].copy, copy);
      assert.equal(copy.kind, kind);
      {
        const companion = writes[0].data;
        assert.equal(companion.system.master.id, copy.copyId);
        assert.equal(companion.flags[OWN_FLAG].copy.companionOf, copy.copyId);
        assert.deepEqual(copy.companions, [{copyId: companion._id, rev: copy.rev, sourceId: 'pet'}]);
      }
    });

    if (kind === 'new') continue;

    await t.test(`${kind} checks every companion before saving the first one`, async () => {
      await prepare(kind, {existing: true});
      meta.cards.B = {world: 'b', mainId: 'main-B', charOwner: 'player', companionOf: a.id};
      await assert.rejects(submit(), /所属团与提交目标不同/);
      assert.deepEqual(writes, []);
      assert.deepEqual(serialized, []);
    });

    await t.test(`${kind} serializes the master before saving any companion`, async () => {
      await prepare(kind);
      a.toObject = () => { throw new Error('master serialization failed'); };
      await assert.rejects(submit(), /master serialization failed/);
      assert.deepEqual(serialized, ['pet']);
      assert.deepEqual(writes, []);
    });
  }
});
