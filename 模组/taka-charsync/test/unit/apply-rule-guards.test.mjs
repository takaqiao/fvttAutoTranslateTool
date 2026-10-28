import test from 'node:test';
import assert from 'node:assert/strict';
import { applyCopy, applyForce, applyNew } from '../../scripts/main/apply.mjs';

test('all copy apply paths recheck current rules before reading or writing actor history', async () => {
  globalThis.game = {world: {id: 'ecnu'}, settings: {get: () => 'test-published'}};
  globalThis.foundry = {utils: {getRoute: value => value}};
  const copy = {copyId: 'copy000000000001', cardId: 'card000000000001', rev: 200, targetWorld: 'ecnu', rulesHash: 'a'.repeat(64)};
  const meta = {
    ruleSnapshot: {version: 2, enforcedSince: 100, rules: {ecnu: {}}, hashes: {ecnu: 'b'.repeat(64)}},
    published: ['copy:copy000000000001:200'],
  };
  const stale = {...meta, ruleSnapshot: {...meta.ruleSnapshot, hashes: {ecnu: 'a'.repeat(64)}}};
  for (const [kind, apply] of [
    ['level', value => applyCopy({}, value, {meta: stale})],
    ['force', value => applyForce({}, value, {meta: stale})],
    ['new', value => applyNew(value, {meta: stale})],
  ]) {
    const requests = [];
    globalThis.fetch = async url => {
      requests.push(url);
      assert.equal(url, 'test-published/meta.json', 'unverified copies must not reach history loading');
      return {ok: true, status: 200, json: async () => meta};
    };
    await assert.rejects(apply({...copy, kind}), /团规则/);
    assert.deepEqual(requests, ['test-published/meta.json']);
  }
});
