import test from 'node:test';
import assert from 'node:assert/strict';
import {makeCampaignEnvironment as environment, ruleSnapshot as snapshot} from '../helpers/campaign-environment.mjs';

test('interleaved prepare uses each Actor campaign for GM and player without changing world settings', async () => {
  const {a, b} = await environment();
  for (const isGM of [true, false]) {
    game.user.isGM = isGM;
    for (const actor of [a, b, a]) actor.prepareData();
    assert.deepEqual(a.prepared, {fa: true, gradual: true, stamina: true});
    assert.deepEqual(b.prepared, {fa: false, gradual: false, stamina: false});
    assert.equal(game.settings.get('pf2e', 'freeArchetypeVariant'), false);
    assert.equal(game.pf2e.settings.variants.gab, false);
  }
});

test('nested familiar preparation and thrown callbacks restore both enclosing and world contexts', async () => {
  const {api, a, b, Familiar} = await environment();
  const familiar = new Familiar('pet', 'B');
  game.actors.set(familiar.id, familiar);
  a.duringPrepare = () => {
    familiar.reset();
    assert.equal(familiar.prepared.fa, false);
    assert.throws(() => api.withActorRules(b, () => { throw new Error('nested'); }), /nested/);
    assert.equal(game.settings.get('pf2e', 'freeArchetypeVariant'), true);
    assert.equal(game.pf2e.settings.variants.gab, true);
  };
  a.prepareData();
  assert.equal(game.settings.get('pf2e', 'freeArchetypeVariant'), false);
  assert.equal(game.pf2e.settings.variants.gab, false);
});

test('document reads and unscoped module initialization keep the stored world setting', async () => {
  const {api, a} = await environment();
  assert.deepEqual(api.withActorRules(a, () => game.settings.get('pf2e', 'freeArchetypeVariant', {document: true})),
    {key: 'pf2e.freeArchetypeVariant', value: false});
  assert.equal(game.settings.get('pf2e', 'freeArchetypeVariant'), false);
  assert.equal(api.ruleValue(a, 'pf2e', 'freeArchetypeVariant'), true);
});

test('a Promise callback is rejected without retaining a shared context', async () => {
  const {api, a} = await environment();
  let observed;
  assert.throws(() => api.withActorRules(a, async () => {
    await Promise.resolve();
    observed = game.settings.get('pf2e', 'freeArchetypeVariant');
  }), /synchronous|同步/i);
  await Promise.resolve();
  assert.equal(observed, false);
  assert.equal(game.pf2e.settings.variants.gab, false);
});

test('invalid nested Actors use world values and remain unavailable instead of inheriting another campaign', async () => {
  const {api, a} = await environment();
  const unknown = {id: 'unknown', type: 'character', _source: {_id: 'unknown'}};
  api.withActorRules(a, () => api.withActorRules(unknown, () => {
    assert.equal(game.settings.get('pf2e', 'freeArchetypeVariant'), false);
    assert.equal(game.pf2e.settings.variants.gab, false);
  }));
  assert.equal(api.actorRuleState(unknown).ok, false);
  assert.equal(api.actorRuleState(a).ok, true);
});

test('rule updates reset only affected Actors, including changed assignment', async () => {
  const {api, a, b} = await environment();
  const next = snapshot();
  next.rules.a['pf2e.freeArchetypeVariant'] = false;
  next.hashes.a = 'c'.repeat(64);
  assert.equal(api.updateRuleSnapshot(next), true);
  assert.equal(a.resets, 1);
  assert.equal(b.resets, 0);
  assert.equal(a.prepared.fa, false);
  assert.equal(api.updateRuleSnapshot({...next, generatedAt: 200}), false);
  next.cardWorld.A = 'b';
  next.generatedAt = 300;
  assert.equal(api.updateRuleSnapshot(next), true);
  assert.equal(api.actorRuleState(a).world, 'b');
  assert.equal(a.resets, 2);
});

test('incomplete profiles and failed refresh cannot be reported as applied', async () => {
  const {api, a} = await environment();
  api.setRuleRefreshError('meta fetch failed');
  assert.equal(api.actorRuleState(a).ok, false);
  assert.ok(api.selfCheck(a).length > 0);
  api.setRuleRefreshError(null);
  assert.equal(api.actorRuleState(a).ok, true);
  const broken = snapshot();
  delete broken.rules.a['pf2e.gradualBoostsVariant'];
  api.updateRuleSnapshot(broken);
  assert.equal(api.actorRuleState(a).ok, false);
});

test('default sentinels use registered defaults rather than stored character world values', async () => {
  const {api, a} = await environment();
  const next = snapshot();
  next.rules.a['pf2e.freeArchetypeVariant'] = {$charsyncDefault: true};
  game.settings.settings.get('pf2e.freeArchetypeVariant').default = true;
  api.updateRuleSnapshot(next);
  assert.equal(api.ruleValue(a, 'pf2e', 'freeArchetypeVariant'), true);
  assert.equal(game.settings.get('pf2e', 'freeArchetypeVariant'), false);
});

test('ABP explicit Actor helper and standalone item preparation do not depend on a selected user character', async () => {
  const {a, b} = await environment();
  assert.equal(game.pf2e.variantRules.AutomaticBonusProgression.isEnabled(a), true);
  assert.equal(game.pf2e.variantRules.AutomaticBonusProgression.isEnabled(b), false);
  assert.equal(a.prepareDataFromItems(), true);
  assert.equal(b.prepareDataFromItems(), false);
  assert.equal(game.pf2e.settings.variants.abp, 'noABP');
});

test('malformed setting values cannot turn false strings into enabled variants', async () => {
  const {api, a} = await environment();
  const next = snapshot();
  next.rules.a['pf2e.freeArchetypeVariant'] = 'false';
  api.updateRuleSnapshot(next);
  assert.equal(api.actorRuleState(a).ok, false);
  a.prepareData();
  assert.equal(a.prepared.fa, false);
});

test('module-load installation tolerates the initial game placeholder and never invokes setting onChange', async () => {
  const {api, a} = await environment();
  const currentGame = game;
  game.settings.settings.get('pf2e.freeArchetypeVariant').onChange = () => {
    throw new Error('must not write derived world settings');
  };
  globalThis.game = {};
  api.installOverride(snapshot());
  globalThis.game = currentGame;
  a.prepareData();
  assert.equal(a.prepared.fa, true);
  assert.equal(game.settings.get('pf2e', 'freeArchetypeVariant'), false);
});

test('snapshot replacement only rerenders open apps of affected Actors', async () => {
  const {api, a, b} = await environment();
  const rendered = [];
  a.apps.open = {rendered: true, render: () => rendered.push('A')};
  a.apps.closed = {rendered: false, render: () => rendered.push('closed')};
  b.apps.open = {rendered: true, render: () => rendered.push('B')};
  const next = snapshot();
  next.hashes.a = 'd'.repeat(64);
  api.updateRuleSnapshot(next);
  assert.deepEqual(rendered, ['A']);
});

test('refresh uses ApplicationV2 render options for an open attribute builder', async () => {
  const {api, a, AttributeBuilder} = await environment();
  const builder = new AttributeBuilder({actor: a});
  builder.rendered = true;
  let options;
  builder.render = received => { options = received; };
  a.apps.builder = builder;
  api.refreshActorRules(a);
  assert.deepEqual(options, {force: false});
});

test('late older snapshots cannot overwrite a more recent campaign profile', async () => {
  const {api, a} = await environment();
  const recent = snapshot();
  recent.generatedAt = 200;
  recent.hashes.a = 'e'.repeat(64);
  recent.rules.a['pf2e.freeArchetypeVariant'] = false;
  api.updateRuleSnapshot(recent);
  assert.equal(api.updateRuleSnapshot(snapshot()), false);
  assert.equal(api.actorRuleState(a).hash, 'e'.repeat(64));
  assert.equal(a.prepared.fa, false);
});

test('refresh recovery reapplies campaign rules after an Actor was prepared while metadata was unavailable', async () => {
  const {api, a} = await environment();
  let renders = 0;
  a.apps.open = {rendered: true, render: () => { renders++; }};
  api.setRuleRefreshError('offline');
  a.prepareData();
  assert.equal(a.prepared.fa, false);
  assert.equal(renders, 1);
  api.setRuleRefreshError(null);
  api.updateRuleSnapshot(snapshot());
  assert.equal(a.prepared.fa, true);
  assert.equal(api.actorRuleState(a).ok, true);
  assert.equal(renders, 2);
});

test('explicit Actor refresh follows a changed master and updates only already opened applications', async () => {
  const {api, a, Familiar} = await environment();
  const pet = new Familiar('pet', 'A');
  game.actors.set('pet', pet);
  pet.prepareData();
  assert.equal(pet.prepared.fa, true);
  pet._source.system.master.id = 'B';
  api.refreshActorRules(pet);
  assert.equal(pet.prepared.fa, false);
  assert.equal(a.resets, 0);
});

test('libWrapper registration errors leave explicit unavailable state without interrupting later protection setup', async t => {
  t.mock.method(console, 'error', () => {});
  globalThis.libWrapper = {register() { throw new Error('registration rejected'); }};
  let env;
  try {
    await assert.doesNotReject(async () => { env = await environment(); });
    assert.equal(env.api.actorRuleState(env.a).ok, false);
    assert.match(env.api.actorRuleState(env.a).reason, /adapter-failed/);
  } finally {
    delete globalThis.libWrapper;
  }
});

test('failed Actor preparation remains unavailable until a complete successful retry', async () => {
  const {api, a} = await environment();
  a.duringPrepare = () => { throw new Error('invalid rule element'); };
  assert.throws(() => a.prepareData(), /invalid rule element/);
  assert.equal(api.actorRuleState(a).ok, false);
  assert.equal(game.pf2e.settings.variants.gab, false);
  delete a.duringPrepare;
  a.prepareData();
  assert.equal(api.actorRuleState(a).ok, true);
  assert.equal(a.prepared.fa, true);
});

test('a full retry after item preparation failure applies campaign rules to the outer preparation too', async () => {
  const {api, a} = await environment();
  a.duringPrepareFromItems = () => { throw new Error('invalid item'); };
  assert.throws(() => a.prepareData(), /invalid item/);
  assert.equal(api.actorRuleState(a).ok, false);
  delete a.duringPrepareFromItems;
  a.prepareData();
  assert.equal(api.actorRuleState(a).ok, true);
  assert.deepEqual(a.prepared, {fa: true, gradual: true, stamina: true});
});
