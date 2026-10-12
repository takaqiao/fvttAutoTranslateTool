import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {recallDegree, finalizeWorkbenchRecall} from '../scripts/knowledge-workbench.mjs';
import {MODULE_ID} from '../scripts/rules.mjs';
import * as native860 from './knowledge-pf2e-860-fixture.mjs';
import {fixture, Predicate, extract, degree} from './knowledge-pf2e-861-fixture.mjs';

const bundle = process.env.FVTT_PF2E_861_BUNDLE ?? process.env.PF2E_861_BUNDLE ?? '';
test('the 8.6.1 contract fixture contains unchanged functions from the release bundle', {skip: !fs.existsSync(bundle)}, () => {
  const bytes = fs.readFileSync(bundle), source = bytes.toString('utf8');
  assert.equal(createHash('sha256').update(bytes).digest('hex'), 'a48f919b8b9a791868ff3eb4eb560b210855932847e8c861ee914e16e3448ed3');
  assert.equal(fixture.sha256, 'a48f919b8b9a791868ff3eb4eb560b210855932847e8c861ee914e16e3448ed3');
  for (const key of ['degree', 'predicate', 'extract', 'totals', 'merge']) assert.ok(source.includes(fixture[key]), `${key} is a verbatim native excerpt`);
});

const adjustment = (predicate, options, amount = -1) => ({predicate: new Predicate(predicate), ...(options === undefined ? {} : {options: new Set(options)}), adjustments: {all: {amount, label: 'native adjustment'}}});
function compare(entries, expected, rollOptions = ['self:level:5', 'action:recall-knowledge']) {
  const input = {total: 21, die: 12, dc: 20, rollOptions, dosAdjustments: entries};
  assert.equal(degree(input), expected, 'expectation matches the actual 8.6.1 functions');
  assert.equal(recallDegree({...input, domains: ['society'], actor: {synthetics: {degreeOfSuccessAdjustments: {society: entries}}}, Predicate}), expected);
}

test('an opposing adjustment uses its own perspective with native 8.6.1 total and natural options', () => {
  compare([adjustment(['self:level:8', 'check:total:21', 'check:total:delta:1', 'check:roll:total:natural:12'], ['self:level:8'])], 1);
});

test('global actor options do not leak into an 8.6.1 opposing adjustment', () => {
  compare([adjustment(['self:level:5'], ['self:level:8'])], 2);
});

test('an empty 8.6.1 entry options set still excludes global actor options', () => {
  compare([adjustment(['action:recall-knowledge'], [])], 2);
});

test('8.6.1 entries without their own options retain the global-option behavior', () => {
  compare([adjustment(['action:recall-knowledge'], undefined)], 1);
});

const opposing = predicateOf => {
  const self = {synthetics: {degreeOfSuccessAdjustments: {society: [{predicate: predicateOf([]), adjustments: {all: {amount: 1, label: 'native adjustment'}}}]}}};
  const opposer = {_source: {flags: {pf2e: {rollOptions: {all: {'self:level:5': true}}}}}, getRollOptions: () => ['self:level:8', 'self:level:5'], synthetics: {opposingDegreeOfSuccessAdjustments: {origin: {society: [{predicate: predicateOf(['self:level:8', {not: 'self:level:5'}, 'check:type:skill-check']), adjustments: {all: {amount: -1, label: 'native adjustment'}}}]}}}};
  return {self, selfRole: 'origin', opposer, domains: ['society'], options: ['self:level:5', 'check:type:skill-check']};
};
test('native 8.6.1 opposing entries preserve their order and override earlier self adjustments', () => {
  compare(extract(opposing(statements => new Predicate(statements))), 1, ['self:level:5', 'check:type:skill-check']);
});

function savedRecall(entries) {
  const gm = {id: 'gm', isGM: true}, owner = {id: 'owner'};
  const actor = {uuid: 'Actor.hero', testUserPermission: user => user === owner, synthetics: {degreeOfSuccessAdjustments: {society: [adjustment([], undefined, 1)]}}};
  const candidate = {statistic: 'society', modifier: 9, total: 21, dc: 20, degree: 3, targetUuid: null, domains: ['society'], rollOptions: ['self:level:5'], dosAdjustments: entries};
  const state = JSON.parse(JSON.stringify({schema: 1, actorUuid: actor.uuid, userId: owner.id, status: 'pending', probeUse: {status: 'done'}, die: 12, assurance: false, candidates: [candidate], targetUuids: [], targetActors: [], primary: {statistic: 'society', targetUuid: null}}));
  let updates = 0;
  const message = {id: 'saved', actor, author: owner, blind: true, rolls: [{total: 12, dice: [{results: [{result: 12, active: true}]}]}], flags: {pf2e: {context: {type: 'skill-check', options: ['action:recall-knowledge']}}, [MODULE_ID]: {workbenchRecall: state}}, update: async () => {updates++;}};
  return {game: {user: gm, users: new Map([[gm.id, gm], [owner.id, owner]]), messages: new Map([[message.id, message]]), pf2e: {Predicate}}, message, updates: () => updates};
}

test('GM DC changes use serialized native 8.6.1 predicates and entry options after the original effect is gone', async () => {
  const entries = [{predicate: ['self:level:8', {gte: ['check:total:delta', 0]}, 'check:total:natural:12'], options: ['self:level:8'], adjustments: {all: {amount: -1, label: 'consumed effect'}}}];
  const f = savedRecall(entries);
  const result = await finalizeWorkbenchRecall({...f, dc: 21});
  assert.equal(degree({total: 21, die: 12, dc: 21, dosAdjustments: entries}), 1);
  assert.equal(result.degree, 1);
  assert.equal(result.total, 21);
  assert.equal(result.die, 12);
  assert.equal(f.message.rolls[0].total, 12);
  assert.equal(f.updates(), 1);
});

// Other suites use the 8.6.0 excerpts as their native oracle. 8.6.1 renames the
// bindings and rewrites one condition, so the two releases are compared by
// what they decide rather than by text.
test('the 8.6.1 degree and adjustment merge decide every roll as 8.6.0 does', () => {
  const entrySets = [
    [],
    [{adjustments: {all: {amount: 1, label: 'raise'}}}],
    [{adjustments: {all: {amount: -1, label: 'lower'}}}],
    [{adjustments: {all: {amount: 'criticalFailure', label: 'fixed'}}}],
    [{predicate: [{gte: ['check:total:delta', 0]}], adjustments: {success: {amount: 1, label: 'success raise'}, failure: {amount: 2, label: 'unused'}}}],
    [{predicate: [{or: ['check:total:natural:1', {not: 'action:recall-knowledge'}]}], options: ['self:level:8'], adjustments: {failure: {amount: -1, label: 'failure lower'}}}],
    [{predicate: [{lt: ['check:total:natural', 5]}], adjustments: {criticalSuccess: {amount: -2, label: 'critical lower'}}}, {adjustments: {criticalFailure: {amount: 1, label: 'later entry'}}}],
  ];
  let rolls = 0;
  for (const dosAdjustments of entrySets) for (let die = 1; die <= 20; die++) for (let total = 8; total <= 32; total++) {
    const input = {total, die, dc: 20, rollOptions: ['self:level:5', 'action:recall-knowledge'], dosAdjustments};
    assert.equal(degree(input), native860.degree(input), JSON.stringify(input));
    rolls++;
  }
  assert.equal(rolls, 3500);
});

test('the 8.6.1 predicate validates and tests statements as 8.6.0 does', () => {
  const options = [[], ['a'], ['a', 'b'], ['self:level:5', 'target:level:8'], ['self:level:9', 'target:level:8', 'a']];
  const predicates = [
    [], ['a'], ['a', 'b'], [{and: ['a', 'b']}], [{nand: ['a', 'b']}], [{or: ['a', 'b']}], [{xor: ['a', 'b']}], [{nor: ['a', 'b']}], [{not: 'a'}],
    [{if: 'a', then: 'b'}], [{iff: ['a', 'b']}], [{eq: ['a', 'a']}], [{eq: ['self:level', 5]}], [{gt: ['self:level', 'target:level']}],
    [{gte: ['self:level', 9]}], [{lt: ['self:level', 'target:level']}], [{lte: ['target:level', 8]}], [{or: [{and: ['a', {not: 'b'}]}, {gt: ['self:level', 6]}]}],
    [''], [5], [{and: 'a'}], [{gt: ['self:level']}], [{not: 'a', or: ['b']}], [{unknown: ['a']}],
  ];
  for (const statements of predicates) {
    const current = new Predicate(statements), previous = new native860.Predicate(statements);
    assert.equal(current.isValid, previous.isValid, JSON.stringify(statements));
    if (current.isValid) for (const set of options) assert.equal(current.test(set), previous.test(set), JSON.stringify([statements, set]));
  }
});

test('8.6.1 extracts opposing adjustments as 8.6.0 does', () => {
  const input = () => opposing(statements => statements);
  assert.deepEqual(extract(input()), native860.extract(input()));
  assert.deepEqual(extract({...input(), opposer: null}), native860.extract({...input(), opposer: null}));
});
