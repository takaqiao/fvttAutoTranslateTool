import fs from 'node:fs';

export const fixture = JSON.parse(fs.readFileSync(new URL('./fixtures/knowledge-pf2e-8.6.1-native.json', import.meta.url), 'utf8'));
const outcomes = ['criticalFailure', 'failure', 'success', 'criticalSuccess'];
const foundry = {utils: {deepClone: structuredClone}};
// 8.6.1 binds the adjustment amounts to Ut, the outcome names to Gt, the
// plain-object test to M and the unique helper to j.
const Degree = new Function('Roll', 'foundry', 'Math', 'Ut', 'Gt', 'sluggify', '_loc', `return (${fixture.degree});`)(class Roll {}, foundry, {clamp: (value, min, max) => Math.min(max, Math.max(min, value))}, {INCREASE: 1, LOWER: -1}, outcomes, value => value, value => value);
export const Predicate = new Function('foundry', 'M', `const Predicate = ${fixture.predicate}; return Predicate;`)(foundry, value => value !== null && typeof value === 'object' && !Array.isArray(value));
export const extract = new Function('j', `${fixture.extract}; return extractDegreeOfSuccessAdjustments;`)(values => [...new Set(values)]);
const totals = new Function('h', 't', 'o', `let ${fixture.totals} return _;`);
const merge = new Function('t', 'e', '_', 'Gt', 'foundry', `return ${fixture.merge} ?? {};`);
export function adjustments({total, die, dc, rollOptions = [], dosAdjustments = [], roll = {total, dice: die == null ? [] : [{results: [{result: die, active: true}]}]}}) {
  const checkOptions = totals({...roll, total}, {dc: {value: dc}}, value => value != null);
  const entries = dosAdjustments.map(entry => ({...entry, predicate: entry.predicate ? new Predicate(entry.predicate) : undefined}));
  return merge({dosAdjustments: entries}, new Set([...checkOptions, ...rollOptions]), checkOptions, outcomes, foundry);
}
export function degree(input) {
  const {total, die, dc} = input;
  return new Degree({dieValue: die ?? 10, modifier: total - (die ?? 10)}, dc, adjustments(input)).value;
}
