import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {recallDegree, finalizeWorkbenchRecall} from '../scripts/knowledge-workbench.mjs';
import {MODULE_ID} from '../scripts/rules.mjs';
import {fixture, Predicate, extract, degree} from './knowledge-pf2e-860-fixture.mjs';

const bundle = process.env.FVTT_PF2E_860_BUNDLE ?? process.env.PF2E_860_BUNDLE ?? '';
test('the 8.6.0 contract fixture contains unchanged functions from the release bundle', {skip: !fs.existsSync(bundle)}, () => {
  const bytes = fs.readFileSync(bundle), source = bytes.toString('utf8');
  assert.equal(createHash('sha256').update(bytes).digest('hex'), 'd908c2fa1741d692b00b44e6dacba0bf7e8724d05898d555f9b640b2581d34b4');
  assert.equal(fixture.sha256, 'd908c2fa1741d692b00b44e6dacba0bf7e8724d05898d555f9b640b2581d34b4');
  for (const key of ['degree', 'predicate', 'extract', 'totals', 'merge']) assert.ok(source.includes(fixture[key]), `${key} is a verbatim native excerpt`);
});

const adjustment = (predicate, options, amount = -1) => ({predicate: new Predicate(predicate), ...(options === undefined ? {} : {options: new Set(options)}), adjustments: {all: {amount, label: 'native adjustment'}}});
function compare(entries, expected, rollOptions = ['self:level:5', 'action:recall-knowledge']) {
  const input = {total: 21, die: 12, dc: 20, rollOptions, dosAdjustments: entries};
  assert.equal(degree(input), expected, 'expectation matches the actual 8.6.0 functions');
  assert.equal(recallDegree({...input, domains: ['society'], actor: {synthetics: {degreeOfSuccessAdjustments: {society: entries}}}, Predicate}), expected);
}

test('an opposing adjustment uses its own perspective with native total and natural options', () => {
  compare([adjustment(['self:level:8', 'check:total:21', 'check:total:delta:1', 'check:roll:total:natural:12'], ['self:level:8'])], 1);
});

test('global actor options do not leak into an opposing adjustment', () => {
  compare([adjustment(['self:level:5'], ['self:level:8'])], 2);
});

test('an empty entry options set still excludes global actor options', () => {
  compare([adjustment(['action:recall-knowledge'], [])], 2);
});

test('entries without their own options retain the 8.5.1 global-option behavior', () => {
  compare([adjustment(['action:recall-knowledge'], undefined)], 1);
});

test('native opposing entries preserve their order and override earlier self adjustments', () => {
  const self = {synthetics: {degreeOfSuccessAdjustments: {society: [adjustment([], undefined, 1)]}}};
  const opposer = {_source: {flags: {pf2e: {rollOptions: {all: {'self:level:5': true}}}}}, getRollOptions: () => ['self:level:8', 'self:level:5'], synthetics: {opposingDegreeOfSuccessAdjustments: {origin: {society: [adjustment(['self:level:8', {not: 'self:level:5'}, 'check:type:skill-check'], undefined)]}}}};
  const entries = extract({self, selfRole: 'origin', opposer, domains: ['society'], options: ['self:level:5', 'check:type:skill-check']});
  compare(entries, 1, ['self:level:5', 'check:type:skill-check']);
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

test('GM DC changes use serialized native predicates and entry options after the original effect is gone', async () => {
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

test('an empty saved adjustment list cannot fall back to newly added actor effects', async () => {
  const f = savedRecall([]);
  assert.equal((await finalizeWorkbenchRecall({...f, dc: 21})).degree, 2);
});

test('a legacy saved card without a snapshot still uses actor adjustments', async () => {
  const f = savedRecall(undefined);
  assert.equal((await finalizeWorkbenchRecall({...f, dc: 21})).degree, 3);
});
