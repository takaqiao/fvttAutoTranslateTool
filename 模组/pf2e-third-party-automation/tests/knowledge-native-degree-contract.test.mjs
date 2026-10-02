import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {recallDegree, finalizeWorkbenchRecall} from '../scripts/knowledge-workbench.mjs';
import {MODULE_ID} from '../scripts/rules.mjs';

const nativeBundle = process.env.FVTT_PF2E_BUNDLE ?? process.env.FVTT_PF2E_RUNTIME ?? '';
const nativeOptions = {skip: !fs.existsSync(nativeBundle)};
const outcomes = ['criticalFailure', 'failure', 'success', 'criticalSuccess'];
let contract;
function nativeContract() {
  if (contract) return contract;
  const bytes = fs.readFileSync(nativeBundle);
  assert.equal(createHash('sha256').update(bytes).digest('hex'), 'd63da8312831b84905e6866b1dd3f9d93e95c1012955b0177ad2ce8ccf246157', 'PF2e 8.5.1 source pin changed');
  const source = bytes.toString('utf8');
  const start = source.indexOf('class DegreeOfSuccess {');
  const end = source.indexOf('}, Vt = {', start);
  assert.ok(start >= 0 && end > start);
  const Degree = new Function('Roll', 'foundry', 'Math', 'Vt', 'Ut', 'sluggify', '_loc', `return (${source.slice(start, end + 1)});`)(class Roll {}, {}, {clamp: (value, min, max) => Math.min(max, Math.max(min, value))}, {INCREASE: 1, LOWER: -1}, outcomes, value => value, value => value);
  const mergeStart = source.indexOf('t.dosAdjustments?.filter((t) => t.predicate?.test(e) ?? !0).reduce(');
  const mergeEnd = source.indexOf(' ?? {};', mergeStart);
  assert.ok(mergeStart >= 0 && mergeEnd > mergeStart);
  const merge = new Function('t', 'e', 'Ut', 'foundry', `return ${source.slice(mergeStart, mergeEnd)} ?? {};`);
  const optionsStart = source.indexOf('g = h.dice.map(', source.indexOf('Sa = class Check {'));
  const optionsEnd = source.indexOf('\n\t\tlet v = (() => {', optionsStart);
  assert.ok(optionsStart >= 0 && optionsEnd > optionsStart);
  const options = new Function('h', 't', 'i', 'o', `let ${source.slice(optionsStart, optionsEnd)} return new Set([..._, ...i]);`);
  contract = {Degree, options: (roll, dc, rollOptions) => options(roll, {dc: {value: dc}}, new Set(rollOptions), value => value != null), merge: (entries, options) => merge({dosAdjustments: entries}, options, outcomes, {utils: {deepClone: structuredClone}})};
  return contract;
}
const entry = adjustments => ({adjustments});
function compare({total = 21, die = 12, dc = 20, entries, domains = ['society'], rollOptions = [], dice = die == null ? [] : [{results: [{result: die, active: true}]}]}, expected) {
  const native = nativeContract();
  const natural = die ?? 10;
  const actor = {synthetics: {degreeOfSuccessAdjustments: Object.fromEntries(domains.map(domain => [domain, entries]))}};
  const allEntries = domains.flatMap(() => entries);
  const roll = {total, dice};
  const options = native.options(roll, dc, rollOptions);
  const merged = native.merge(allEntries, options);
  const degree = new native.Degree({dieValue: natural, modifier: total - natural}, dc, merged).value;
  assert.equal(degree, expected, 'literal expectation matches pinned native DegreeOfSuccess');
  assert.equal(recallDegree({total, die, dc, domains, rollOptions, actor, roll}), expected);
}

test('later native adjustment for the same key overrides an earlier upgrade', nativeOptions, () => {
  compare({entries: [entry({all: {amount: 1, label: 'earlier'}}), entry({all: {amount: -1, label: 'later'}})]}, 1);
});
test('native all adjustment precedes the outcome-specific adjustment', nativeOptions, () => {
  compare({entries: [entry({all: {amount: -1, label: 'all'}, success: {amount: 1, label: 'success'}})]}, 1);
});
test('the same adjustment repeated across native domains applies only once', nativeOptions, () => {
  compare({total: 19, entries: [entry({all: {amount: 1, label: 'upgrade'}})], domains: ['society', 'skill-check']}, 2);
});
test('an ineffective critical-success all increase leaves the specific downgrade eligible', nativeOptions, () => {
  compare({total: 30, entries: [entry({all: {amount: 1, label: 'increase'}, criticalSuccess: {amount: -1, label: 'lower'}})]}, 2);
});
test('zero or unlabeled all adjustments leave a labeled outcome adjustment eligible', nativeOptions, () => {
  for (const all of [{amount: 0, label: 'zero'}, {amount: 1}]) compare({entries: [entry({all, success: {amount: -1, label: 'lower'}})]}, 1);
});
test('explicit native outcome replacements retain their declared degree', nativeOptions, () => {
  for (const [expected, amount] of outcomes.entries()) compare({entries: [entry({all: {amount, label: 'replacement'}})]}, expected);
});
test('native natural 20 and 1 adjustments are bounded before synthetic adjustment', nativeOptions, () => {
  for (const [total, die, expected] of [[19, 20, 2], [20, 1, 1], [40, 20, 3], [0, 1, 0]]) compare({total, die, entries: []}, expected);
});
test('native check total options include the natural-roll alias', nativeOptions, () => {
  const predicate = {test: options => ['check:roll:total:natural:12', 'check:total:delta:1', 'action:recall-knowledge'].every(option => Array.from(options).includes(option))};
  compare({entries: [{predicate, adjustments: {all: {amount: -1, label: 'matched'}}}], rollOptions: ['action:recall-knowledge']}, 1);
});
test('Assurance exposes native undefined natural-check options without null or ten aliases', nativeOptions, () => {
  const predicate = {test: options => ['check:total:natural:undefined', 'check:roll:total:natural:undefined'].every(option => Array.from(options).includes(option)) && !Array.from(options).some(option => /natural:(null|10)$/.test(option))};
  compare({die: null, entries: [{predicate, adjustments: {all: {amount: -1, label: 'matched'}}}]}, 1);
});
test('a deterministic native substitution never adds a saved-number natural alias', nativeOptions, () => {
  const predicate = {test: options => Array.from(options).includes('check:total:natural:undefined') && !Array.from(options).includes('check:total:natural:15')};
  compare({die: 15, dice: [], entries: [{predicate, adjustments: {all: {amount: -1, label: 'substitution'}}}]}, 1);
});
test('captured native natural aliases retain priority over a supplied raw saved die', () => {
  const actor = {synthetics: {degreeOfSuccessAdjustments: {society: [{predicate: {test: options => Array.from(options).includes('check:total:natural:undefined') && !Array.from(options).includes('check:total:natural:15')}, adjustments: {all: {amount: -1, label: 'captured'}}}]}}};
  assert.equal(recallDegree({total: 21, die: 15, dc: 20, domains: ['society'], rollOptions: ['check:total:natural:undefined', 'check:roll:total:natural:undefined'], actor}), 1);
});
test('failed predicates cannot override an earlier matching adjustment', nativeOptions, () => {
  compare({entries: [entry({all: {amount: -1, label: 'matching'}}), {predicate: {test: () => false}, adjustments: {all: {amount: 1, label: 'excluded'}}}]}, 1);
});

function savedRecall() {
  const gm = {id: 'gm', isGM: true};
  const owner = {id: 'owner'};
  const actor = {uuid: 'Actor.hero', testUserPermission: user => user === owner, synthetics: {degreeOfSuccessAdjustments: {society: [entry({all: {amount: -1, label: 'lower'}})]}}};
  const candidate = {statistic: 'society', modifier: 9, total: 21, dc: 20, degree: 3, targetUuid: null, domains: ['society'], rollOptions: []};
  const state = {schema: 1, actorUuid: actor.uuid, userId: owner.id, status: 'pending', probeUse: {status: 'done'}, die: 12, assurance: false, candidates: [candidate], targetUuids: [], targetActors: [], primary: {statistic: 'society', targetUuid: null}};
  let updates = 0;
  const message = {id: 'original', actor, author: owner, blind: true, rolls: [{total: 12, dice: [{results: [{result: 12, active: true}]}]}], flags: {pf2e: {context: {type: 'skill-check', options: ['action:recall-knowledge']}}, [MODULE_ID]: {workbenchRecall: state}}, update: async () => {updates++;}};
  const game = {user: gm, users: new Map([[gm.id, gm], [owner.id, owner]]), messages: new Map([[message.id, message]])};
  return {game, message, updates: () => updates};
}
test('an unchanged DC retains the saved native degree without another roll', async () => {
  const f = savedRecall();
  const result = await finalizeWorkbenchRecall(f);
  assert.equal(result.degree, 3);
  assert.equal(f.updates(), 1);
  assert.equal(f.message.rolls[0].total, 12);
});
test('a GM DC change recomputes only the degree using the original saved die', nativeOptions, async () => {
  const f = savedRecall();
  const result = await finalizeWorkbenchRecall({...f, dc: 21});
  compare({dc: 21, entries: [entry({all: {amount: -1, label: 'lower'}})]}, 1);
  assert.equal(result.degree, 1);
  assert.equal(result.total, 21);
  assert.equal(result.die, 12);
  assert.equal(f.message.rolls[0].total, 12);
  assert.equal(f.updates(), 1);
});
test('a changed DC uses a deterministic saved Roll with no native natural-number option', nativeOptions, async () => {
  const f = savedRecall(), state = f.message.flags[MODULE_ID].workbenchRecall;
  state.die = 15; state.candidates[0].modifier = 6;
  f.message.rolls = [{total: 15, dice: []}];
  f.message.actor.synthetics.degreeOfSuccessAdjustments.society = [{predicate: {test: options => Array.from(options).includes('check:total:natural:undefined') && !Array.from(options).includes('check:total:natural:15')}, adjustments: {all: {amount: -1, label: 'substitution'}}}];
  const result = await finalizeWorkbenchRecall({...f, dc: 21});
  assert.equal(result.degree, 1);
  assert.equal(result.die, 15);
  assert.equal(f.message.rolls[0].dice.length, 0);
});
test('missing saved Roll dice and native aliases leave a changed degree unavailable', async () => {
  const f = savedRecall();
  delete f.message.rolls[0].dice;
  assert.equal(await finalizeWorkbenchRecall({...f, dc: 21}), null);
  assert.equal(f.updates(), 0);
});
