import test from 'node:test';
import assert from 'node:assert/strict';
import { copyRuleError } from '../../scripts/lib/rule-validation.mjs';

const snapshot = {
  version: 2, enforcedSince: 100, rules: {ecnu: {'pf2e.freeArchetypeVariant': false}}, hashes: {ecnu: 'a'.repeat(64)},
};
const copy = {kind: 'level', rev: 101, targetWorld: 'ecnu', rulesHash: 'a'.repeat(64)};

test('all new submission kinds require the current target campaign rules', () => {
  for (const kind of ['new', 'level', 'force']) {
    assert.equal(copyRuleError({...copy, kind}, snapshot), null);
    assert.ok(copyRuleError({...copy, kind, rulesHash: undefined}, snapshot));
    assert.ok(copyRuleError({...copy, kind, rulesHash: 'b'.repeat(64)}, snapshot));
    assert.ok(copyRuleError({...copy, kind, targetWorld: 'other'}, snapshot));
  }
});

test('copies from before enforcement keep their history without fabricated rule hashes', () => {
  assert.equal(copyRuleError({...copy, rev: 99, rulesHash: undefined}, snapshot), null);
  assert.ok(copyRuleError({...copy, rev: 100, rulesHash: undefined}, snapshot));
  assert.ok(copyRuleError({...copy, rev: 99, rulesHash: 'b'.repeat(64)}, snapshot));
});

test('unavailable or invalid rule publication never authorizes a copy', () => {
  for (const value of [null, {}, {...snapshot, version: 1}, {...snapshot, enforcedSince: null},
    {...snapshot, rules: {}}, {...snapshot, hashes: {}}]) {
    assert.ok(copyRuleError(copy, value));
  }
  assert.ok(copyRuleError({...copy, rev: NaN}, snapshot));
});

test('a mapped working card cannot submit under another valid campaign profile', () => {
  const other = {
    ...snapshot,
    rules: {...snapshot.rules, other: {}},
    hashes: {...snapshot.hashes, other: 'b'.repeat(64)},
    cardWorld: {card000000000001: 'ecnu'},
  };
  assert.ok(copyRuleError({
    ...copy, cardId: 'card000000000001', targetWorld: 'other', rulesHash: 'b'.repeat(64),
  }, other));
});
