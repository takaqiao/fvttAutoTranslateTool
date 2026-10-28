import test from 'node:test';
import assert from 'node:assert/strict';
import {makeCampaignEnvironment, ruleSnapshot} from '../helpers/campaign-environment.mjs';
import * as adapters from '../../scripts/char/rules-sheets.mjs';

function deferred() {
  let resolve;
  const promise = new Promise(done => { resolve = done; });
  return {promise, resolve};
}

test('Actor preparation is available before setup and sheet adapters wait for core pending registrations', async () => {
  const {api, a, CharacterSheetPF2e, runSetup} = await makeCampaignEnvironment({setup: false});
  a.prepareData();
  assert.equal(a.prepared.fa, true);
  assert.equal(api.actorRuleState(a).ok, true);
  await runSetup();
  const data = await new CharacterSheetPF2e(a).getData();
  assert.equal(data.hasStamina, true);
  assert.equal(api.actorRuleState(a).ok, true);
});

test('concurrent character sheets keep Actor stamina, ABP and proficiency after different awaits', async () => {
  const {api, a, b, CharacterSheetPF2e} = await makeCampaignEnvironment();
  const next = ruleSnapshot();
  next.rules.a['pf2e.proficiencyVariant'] = true;
  next.rules.a['pf2e.proficiencyExpertModifier'] = 7;
  api.updateRuleSnapshot(next);
  const waitA = deferred(), waitB = deferred();
  const resultA = new CharacterSheetPF2e(a).getData({wait: waitA.promise});
  const resultB = new CharacterSheetPF2e(b).getData({wait: waitB.promise});
  assert.equal(game.pf2e.settings.variants.stamina, false);
  waitB.resolve();
  const dataB = await resultB;
  waitA.resolve();
  const dataA = await resultA;
  assert.equal(dataA.hasStamina, true);
  assert.equal(dataB.hasStamina, false);
  assert.equal(dataA.abpEnabled, true);
  assert.equal(dataB.abpEnabled, false);
  assert.equal(dataA.martialProficiencies.attacks.simple.value, 7);
  assert.equal(dataB.martialProficiencies.attacks.simple.value, 10);
  assert.deepEqual(dataA.apexAttributeOptions, []);
  assert.equal(game.pf2e.settings.variants.pwol.enabled, false);
});

test('AttributeBuilder follows render Actor for gradual boosts and its private Apex dependency', async () => {
  const {a, b, AttributeBuilder} = await makeCampaignEnvironment();
  const builder = new AttributeBuilder({actor: a});
  const pending = deferred();
  const first = builder.render({actor: a, wait: pending.promise});
  assert.equal(game.settings.get('pf2e', 'gradualBoostsVariant'), false);
  pending.resolve();
  const dataA = await first;
  assert.equal(dataA.state.gradualBoostsVariant, true);
  await builder.handleApexBoost('str');
  assert.equal(a.system.build.attributes.apex, 'str');
  const dataB = await builder.render({actor: b});
  assert.equal(dataB.state.gradualBoostsVariant, false);
  assert.equal(dataB.state.abpEnabled, false);
  assert.equal(adapters.ruleApplicationActor?.(builder), b);
  await builder.handleApexBoost('dex');
  assert.equal(b.system.build.attributes.apex, null);
});

test('a rules change during sheet preparation retries with the current profile', async () => {
  const {api, a, CharacterSheetPF2e} = await makeCampaignEnvironment();
  const pending = deferred();
  const sheet = new CharacterSheetPF2e(a);
  const result = sheet.getData({wait: pending.promise});
  const next = ruleSnapshot();
  next.hashes.a = 'd'.repeat(64);
  next.rules.a['pf2e.staminaVariant'] = false;
  api.updateRuleSnapshot(next);
  pending.resolve();
  const data = await result;
  assert.notEqual(data.editable, false);
  assert.equal(data.hasStamina, false);
  assert.equal(api.actorRuleState(a).ok, true);
});

test('an offline render completing after recovery automatically reads current sheet data', async () => {
  const {api, a, CharacterSheetPF2e} = await makeCampaignEnvironment();
  const pending = deferred();
  api.setRuleRefreshError('offline');
  const result = new CharacterSheetPF2e(a).getData({wait: pending.promise});
  api.setRuleRefreshError(null);
  assert.equal(api.actorRuleState(a).ok, true);
  pending.resolve();
  const data = await result;
  assert.notEqual(data.editable, false);
  assert.equal(data.hasStamina, true);
  assert.equal(api.actorRuleState(a).ok, true);
});

test('repeated profile changes stop sheet retries and keep the result read-only', async () => {
  const {api, a, CharacterSheetPF2e, runSetup} = await makeCampaignEnvironment({setup: false});
  const original = CharacterSheetPF2e.prototype.getData;
  let reads = 0;
  CharacterSheetPF2e.prototype.getData = async function (...args) {
    const data = await original.apply(this, args);
    const next = ruleSnapshot();
    next.hashes.a = (++reads).toString().repeat(64);
    api.updateRuleSnapshot(next);
    return data;
  };
  await runSetup();
  const data = await new CharacterSheetPF2e(a).getData();
  assert.equal(reads, 2);
  assert.equal(data.editable, false);
  assert.equal(api.actorRuleState(a).ok, false);
});

test('a recoverable sheet error does not make later Actor preparation silently use world rules', async () => {
  const {api, a, CharacterSheetPF2e} = await makeCampaignEnvironment();
  api.recordRuleAdapterFailure(a, 'CharacterSheet.getData');
  a.prepareData();
  assert.equal(api.actorRuleState(a).ok, false);
  await new CharacterSheetPF2e(a).getData();
  assert.equal(api.actorRuleState(a).ok, true);
  assert.deepEqual(a.prepared, {fa: true, gradual: true, stamina: true});
});

test('concurrent crafting formulas use Actor PWOL with rarity and update the original cached objects', async () => {
  const {api, a, b, CharacterSheetPF2e} = await makeCampaignEnvironment();
  const next = ruleSnapshot();
  next.rules.a['pf2e.proficiencyVariant'] = true;
  api.updateRuleSnapshot(next);
  const items = ['common', 'rare', 'unique'].map(rarity => ({uuid: `item-${rarity}`, level: 6, rarity}));
  a.formulaItems = items;
  b.formulaItems = items;
  const pendingA = deferred(), pendingB = deferred();
  a.formulaWait = pendingA.promise;
  b.formulaWait = pendingB.promise;
  const rawA = Object.getPrototypeOf(a.crafting).getFormulas.bind(a.crafting);
  const resultA = new CharacterSheetPF2e(a).getData();
  const resultB = new CharacterSheetPF2e(b).getData();
  pendingB.resolve();
  const dataB = await resultB;
  pendingA.resolve();
  const dataA = await resultA;
  assert.deepEqual(dataA.crafting.knownFormulas.map(formula => formula.dc), [16, 21, 26]);
  assert.deepEqual(dataB.crafting.knownFormulas.map(formula => formula.dc), [22, 27, 32]);
  assert.deepEqual((await rawA()).map(formula => formula.dc), [16, 21, 26]);
  assert.deepEqual((await a.crafting.getFormulas()).map(formula => formula.dc), [16, 21, 26]);
  assert.equal(game.pf2e.settings.variants.pwol.enabled, false);
});

test('an old crafting read cannot invalidate a newer successful profile read', async () => {
  const {api, a} = await makeCampaignEnvironment();
  a.formulaItems = [{uuid: 'item-common', level: 6, rarity: 'common'}];
  const pending = deferred();
  a.formulaWait = pending.promise;
  const oldRead = a.crafting.getFormulas();
  const next = ruleSnapshot();
  next.hashes.a = 'd'.repeat(64);
  next.rules.a['pf2e.proficiencyVariant'] = true;
  api.updateRuleSnapshot(next);
  a.formulaWait = Promise.resolve();
  assert.equal((await a.crafting.getFormulas())[0].dc, 16);
  pending.resolve();
  assert.equal((await oldRead)[0].dc, 16);
  assert.equal(api.actorRuleState(a).ok, true);
});

test('an obsolete crafting retry cannot poison recovery through an old render rejection', async t => {
  t.mock.method(console, 'error', () => {});
  const {api, a, CharacterSheetPF2e} = await makeCampaignEnvironment();
  const entered = [deferred(), deferred()];
  const pending = [deferred(), deferred()];
  let reads = 0;
  a.crafting = {
    initialize() {},
    async getFormulas() {
      const index = reads++;
      entered[index]?.resolve();
      await pending[index]?.promise;
      return [{item: {level: 6, rarity: 'common'}, dc: 22}];
    },
  };
  api.refreshActorRules(a);
  const sheet = new CharacterSheetPF2e(a);
  let oldRender;
  a.apps.sheet = {
    rendered: true,
    render() {
      this.rendered = false;
      oldRender = sheet.getData();
      return oldRender;
    },
  };
  api.refreshActorRules(a);
  await entered[0].promise;
  const next = ruleSnapshot();
  next.hashes.a = 'd'.repeat(64);
  api.updateRuleSnapshot(next);
  pending[0].resolve();
  await entered[1].promise;
  next.hashes.a = 'e'.repeat(64);
  next.rules.a['pf2e.proficiencyVariant'] = true;
  api.updateRuleSnapshot(next);
  const latest = await sheet.getData();
  assert.equal(latest.crafting.knownFormulas[0].dc, 16);
  assert.equal(api.actorRuleState(a).ok, true);
  pending[1].resolve();
  await assert.rejects(oldRender, /团规连续更新/);
  assert.equal(reads, 3);
  assert.equal(api.actorRuleState(a).ok, true);
});
