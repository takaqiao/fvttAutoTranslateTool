import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { DEFAULT_SENTINEL, RULE_KEYS, buildRulesTable, renderRulesModule } from '../../engine/rules-table.mjs';
import * as profiles from '../../engine/rules-table.mjs';

const SECTIONS_CHAR = [{id: 'team-plus', label: 'A', levels: [1, 2, 4]}];
const SECTIONS_SOG = [{id: 'ancestry-paragon', label: 'B', levels: [1, 3, 7]}];

function fixture() {
  const charSettings = new Map([
    ['pf2e.freeArchetypeVariant', false],
    ['pf2e.gradualBoostsVariant', true],
    ['pf2e.campaignFeatSections', SECTIONS_CHAR],
    ['core.unrelated', 'char'],
  ]);
  const worldSettings = {
    sog: new Map([
      ['pf2e.freeArchetypeVariant', true],
      ['pf2e.gradualBoostsVariant', true],
      ['pf2e.campaignFeatSections', structuredClone(SECTIONS_SOG)],
      ['core.unrelated', 'world'],
    ]),
    bob: new Map([
      ['pf2e.freeArchetypeVariant', false],
      // 键顺序不同、内容相同也算相同
      ['pf2e.campaignFeatSections', [{levels: [1, 2, 4], label: 'A', id: 'team-plus'}]],
      ['pf2e-team-plus-feats.basicPack', true],
    ]),
  };
  return {charSettings, worldSettings};
}

test('buildRulesTable publishes complete profiles, including equal true and false values', () => {
  const {charSettings, worldSettings} = fixture();
  const rules = buildRulesTable(charSettings, worldSettings);
  assert.equal(rules.sog['pf2e.gradualBoostsVariant'], true);
  assert.equal(rules.bob['pf2e.freeArchetypeVariant'], false);
  assert.deepEqual(rules.sog['pf2e.campaignFeatSections'], SECTIONS_SOG);
  assert.deepEqual(rules.bob['pf2e.campaignFeatSections'], SECTIONS_CHAR);
  for (const profile of Object.values(rules)) {
    assert.ok(RULE_KEYS.every(key => Object.hasOwn(profile, key)));
    assert.deepEqual(profile['pf2e.staminaVariant'], DEFAULT_SENTINEL);
    assert.ok(!Object.hasOwn(profile, 'core.unrelated'));
  }
  assert.deepEqual(rules.sog['pf2e-team-plus-feats.basicPack'], DEFAULT_SENTINEL);
  assert.equal(rules.bob['pf2e-team-plus-feats.basicPack'], true);
});

test('buildRulesTable marks prefixed settings that only the char world has as default', () => {
  const {charSettings, worldSettings} = fixture();
  charSettings.set('pf2e-sf2e-extra-feat-slots.enabled', true);
  const rules = buildRulesTable(charSettings, worldSettings);
  assert.deepEqual(rules.sog['pf2e-sf2e-extra-feat-slots.enabled'], DEFAULT_SENTINEL);
});

test('RULE_KEYS lists the campaign and variant settings', () => {
  assert.ok(RULE_KEYS.includes('pf2e.campaignFeatSections'));
  assert.ok(RULE_KEYS.includes('pf2e.staminaVariant'));
  assert.equal(RULE_KEYS.length, 14);
});

test('profile hashes depend on complete campaign rules and version, not char values, time or mappings', () => {
  const {charSettings, worldSettings} = fixture();
  const build = (extra = {}) => profiles.buildRuleSnapshot({
    charSettings, worldSettings, cardWorld: {cardHero00000001: 'sog'}, folderWorld: {},
    generatedAt: 100, enforcedSince: 90, ...extra,
  });
  const first = build();
  assert.equal(first.version, 2);
  assert.match(first.hashes.sog, /^[a-f0-9]{64}$/);
  charSettings.set('pf2e.freeArchetypeVariant', true);
  charSettings.set('pf2e.gradualBoostsVariant', false);
  const changedChar = build({generatedAt: 200, cardWorld: {}, folderWorld: {folderHero000001: 'sog'}});
  assert.deepEqual(changedChar.rules, first.rules);
  assert.deepEqual(changedChar.hashes, first.hashes);
  worldSettings.sog.set('pf2e.freeArchetypeVariant', false);
  assert.notEqual(build().hashes.sog, first.hashes.sog);
  assert.equal(build().hashes.bob, first.hashes.bob);
});

test('renderRulesModule produces a module that imports back to the same tables', async () => {
  const {charSettings, worldSettings} = fixture();
  const cardWorld = {cardHero00000001: 'sog', cardOther0000001: '-'};
  const generatedAt = 1760600000000;
  const snapshot = profiles.buildRuleSnapshot({charSettings, worldSettings, cardWorld, folderWorld: {}, generatedAt, enforcedSince: generatedAt});
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'charsync-rules-'));
  try {
    const file = path.join(dir, 'rules.mjs');
    fs.writeFileSync(file, renderRulesModule(snapshot));
    const mod = await import(pathToFileURL(file).href);
    assert.deepEqual(mod.SNAPSHOT, snapshot);
    assert.deepEqual(mod.RULES, snapshot.rules);
    assert.deepEqual(mod.CARD_WORLD, cardWorld);
    assert.equal(mod.GENERATED_AT, generatedAt);
  } finally {
    fs.rmSync(dir, {recursive: true, force: true});
  }
});
