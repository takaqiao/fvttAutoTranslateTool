/**
 * Round-22 probe (round-21 script, repointed at the round-22 extraction) ③c: does the mood panel actually translate the 42 optgroup group
 * names now, measured through the REAL translateText with the REAL MOOD_PANEL scope?
 *
 * How the table is obtained (no hand-copying, no regex over the table body):
 *   the whole ember-hardcoded-cn.mjs source is executed as a script inside node:vm
 *   with a stubbed `Hooks`, then the top-level bindings we care about are published
 *   out of the sandbox. So `translateText` and `MOOD_PANEL` are the module's own
 *   objects, not a reconstruction.
 *
 * Anti-空转 measures, all of which must pass or the run exits non-zero:
 *   1. inputs come from soundscapes.json, which probe_groups.mjs wrote after two
 *      independent extractions agreed; the file's counts are re-asserted here
 *      (42 groups / 212 unique arrangement labels). Missing file => NO-INPUT, exit 2.
 *   2. the probe prints how many strings it fed through translateText.
 *   3. a NEGATIVE control: translateText WITHOUT the scope table must translate 0
 *      group names (they are not in global EXACT). If that control shows coverage,
 *      the measurement is meaningless and we exit.
 *   4. a MUTATION control: one entry is deleted from the module SOURCE and the whole
 *      pipeline re-run; coverage must drop by exactly 1. If it does not, the probe
 *      is not actually measuring the table and we exit.
 */
import fs from 'fs';
import vm from 'node:vm';

const HC = 'C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs';
const DATA = 'C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/4-临时脚本/2026-08-16-round22/soundscapes_r22.json';

if (!fs.existsSync(DATA)) { console.log('NO-INPUT: soundscapes.json missing; run probe_groups.mjs first'); process.exit(2); }
const data = JSON.parse(fs.readFileSync(DATA, 'utf8'));
const groups = data.groupLabels, arrs = data.arrLabels;
console.log(`inputs: groupLabels=${groups.length} arrLabels(unique)=${arrs.length} dualRole=${data.both.length}`);
if (groups.length !== 42 || arrs.length !== 212) { console.log('INPUT SHAPE UNEXPECTED'); process.exit(2); }

const rawSrc = fs.readFileSync(HC, 'utf8');

function loadModule(src, tag) {
  const script = src.replace(/^export /gm, '')
    + '\n;globalThis.__out = {translateText, translateNode, MOOD_PANEL, ARRANGEMENTS, ARRANGEMENT_LEAVES, SOUNDSCAPE_GROUPS, INJECTED_SUBTREES};';
  const sandbox = {
    console,
    Hooks: { once() {} },
    Node: { TEXT_NODE: 3, ELEMENT_NODE: 1 },
    globalThis: null
  };
  sandbox.globalThis = sandbox;
  try { vm.runInNewContext(script, sandbox, { filename: `${tag}:ember-hardcoded-cn.mjs` }); }
  catch (e) { console.log(`VM LOAD FAILED (${tag}): ${e.message}`); process.exit(2); }
  const out = sandbox.__out;
  if (!out || typeof out.translateText !== 'function' || !out.MOOD_PANEL) {
    console.log(`VM PUBLISH FAILED (${tag})`); process.exit(2);
  }
  return out;
}

function measure(mod, label) {
  const T = mod.translateText, tbl = mod.MOOD_PANEL;
  let fed = 0;
  const cov = (list) => list.filter((s) => { fed++; return T(s, tbl) !== s; });
  const gCov = cov(groups), aCov = cov(arrs);
  // negative control: no scope table
  let bare = 0;
  for (const s of groups.concat(arrs)) { fed++; if (T(s, null) !== s) bare++; }
  return { fed, gCov, aCov, bare, keys: Object.keys(tbl).length, label };
}

const mod = loadModule(rawSrc, 'live');

// --- structural assertions on the split ---
const gk = Object.keys(mod.SOUNDSCAPE_GROUPS), ak = Object.keys(mod.ARRANGEMENTS);
const overlap = gk.filter((k) => ak.includes(k));
console.log(`\ntables: SOUNDSCAPE_GROUPS=${gk.length} ARRANGEMENTS=${ak.length} overlap=${overlap.length}${overlap.length ? ' -> ' + overlap : ''}`);
console.log(`        ARRANGEMENT_LEAVES=${Object.keys(mod.ARRANGEMENT_LEAVES).length} MOOD_PANEL=${Object.keys(mod.MOOD_PANEL).length}`);
const missingGroupKeys = groups.filter((g) => !(g in mod.SOUNDSCAPE_GROUPS));
const strayGroupKeys = gk.filter((k) => !groups.includes(k));
console.log(`        group keys missing from table=${missingGroupKeys.length}${missingGroupKeys.length ? ' -> ' + missingGroupKeys.join(' | ') : ''}`);
console.log(`        table keys that are NOT upstream group names=${strayGroupKeys.length}${strayGroupKeys.length ? ' -> ' + strayGroupKeys.join(' | ') : ''}`);
// every ARRANGEMENT_LEAVES key must be a real arrangement label or the literal "Reset"
const badLeaf = Object.keys(mod.ARRANGEMENT_LEAVES).filter((k) => k !== 'Reset' && !arrs.includes(k));
console.log(`        ARRANGEMENT_LEAVES keys that are not arrangement labels (Reset allowed)=${badLeaf.length}${badLeaf.length ? ' -> ' + badLeaf.join(' | ') : ''}`);
const undef = Object.entries(mod.ARRANGEMENT_LEAVES).filter(([, v]) => typeof v !== 'string');
console.log(`        ARRANGEMENT_LEAVES entries with non-string value=${undef.length}${undef.length ? ' -> ' + JSON.stringify(undef) : ''}`);
// the subtree registration must still point at MOOD_PANEL
const reg = mod.INJECTED_SUBTREES.find((r) => r[0] === 'form#ember-mood');
console.log(`        form#ember-mood registers MOOD_PANEL: ${reg ? reg[1] === mod.MOOD_PANEL : 'NOT REGISTERED'}`);

const m = measure(mod, 'live');
console.log(`\nfed ${m.fed} strings through the real translateText`);
console.log(`group names   : covered ${m.gCov.length}/${groups.length}  -> 恒英文 ${groups.length - m.gCov.length}`);
console.log(`arrangement   : covered ${m.aCov.length}/${arrs.length}  -> 恒英文 ${arrs.length - m.aCov.length}`);
console.log(`  arrangement labels still ENGLISH: ${JSON.stringify(arrs.filter((s) => mod.translateText(s, mod.MOOD_PANEL) === s))}`);
console.log(`negative control (no scope table): ${m.bare} of ${groups.length + arrs.length} translated (must be 0)`);
console.log('\nsample renderings:');
for (const s of ['Abyssal Combat', 'Cindaric Temple', 'Elemental Combat - Frost', 'Water Temple', 'Solemn Folk', 'Ordain', 'Ember Environment'])
  console.log(`  ${s.padEnd(28)} -> ${mod.translateText(s, mod.MOOD_PANEL)}`);
console.log('enricher path (ARRANGEMENT_LEAVES via PATTERNS):');
for (const s of ['Music: Ankarist Theme', 'Music: Shent Ruins (Tension)', 'Music: Reset', 'Environment: Ordain', 'Music: Bandit Fight Chorus'])
  console.log(`  ${s.padEnd(30)} -> ${mod.translateText(s)}`);

// --- DOM-path check: <optgroup label="…"> is an ATTRIBUTE, not a text node ---
// Builds the minimal element shape translateNode touches and runs the real function.
function fakeOptgroup(label) {
  const attrs = { label };
  return {
    nodeType: 1,
    childNodes: [],
    getAttribute: (a) => attrs[a] ?? null,
    setAttribute: (a, v) => { attrs[a] = v; },
    attrs
  };
}
const og = fakeOptgroup('Cindaric Temple');
mod.translateNode(og, mod.MOOD_PANEL);
console.log(`\nDOM path: <optgroup label="Cindaric Temple"> -> label="${og.attrs.label}"`);
const domOk = og.attrs.label === '辛达里克神殿';
// and prove the claim "removing `label` from the attribute whitelist kills all 42"
const noLabel = loadModule(rawSrc.replace('"placeholder", "alt", "label"', '"placeholder", "alt"'), 'no-label');
const og2 = fakeOptgroup('Cindaric Temple');
noLabel.translateNode(og2, noLabel.MOOD_PANEL);
console.log(`whitelist mutation: drop "label" -> label="${og2.attrs.label}" (must stay English)`);
const whitelistOk = og2.attrs.label === 'Cindaric Temple';
console.log(`DOM path ${domOk && whitelistOk ? 'PASSED' : 'FAILED'} (attribute channel proven, not asserted)`);

// --- MUTATION control ---
const victim2 = '  "Bandit Fight Chorus":              "强盗战斗 · 副歌",';
if (!rawSrc.includes(victim2)) { console.log('MUTATION CONTROL 2: arrangement victim line not found'); process.exit(2); }
const mutated2 = loadModule(rawSrc.replace(victim2 + String.fromCharCode(10), ''), 'mutated-arr');
const m3 = measure(mutated2, 'mutated-arr');
console.log(`mutation control 2: deleted the "Bandit Fight Chorus" row -> arrangement coverage ${m.aCov.length} => ${m3.aCov.length}`);
const ok2 = m3.aCov.length === m.aCov.length - 1;
console.log(`mutation control 2 ${ok2 ? 'PASSED (the probe really reads ARRANGEMENTS too)' : 'FAILED'}`);

const victim = '  "Water Temple": "水之神殿"';
if (!rawSrc.includes(victim)) { console.log('\nMUTATION CONTROL: victim line not found'); process.exit(2); }
const mutated = loadModule(rawSrc.replace(victim + '\n', ''), 'mutated');
const m2 = measure(mutated, 'mutated');
console.log(`\nmutation control: deleted the "Water Temple" row from the source -> group coverage ${m.gCov.length} => ${m2.gCov.length}`);
const ok = m2.gCov.length === m.gCov.length - 1;
console.log(`mutation control ${ok ? 'PASSED (the probe really reads the table)' : 'FAILED (probe is not measuring anything)'}`);

const fail = !ok || !ok2 || !domOk || !whitelistOk || m.bare !== 0 || overlap.length || missingGroupKeys.length || strayGroupKeys.length
  || badLeaf.length || undef.length || !reg || reg[1] !== mod.MOOD_PANEL;
console.log(`\nVERDICT: ${fail ? 'FAIL' : 'PASS'}`);
process.exit(fail ? 1 : 0);
