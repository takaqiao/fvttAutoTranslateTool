/**
 * Round-20 probe ③b: how many soundscape group names / arrangement option labels
 * does the playlist-sidebar mood panel actually put on screen, and how many of
 * them does ember-hardcoded-cn.mjs's ARRANGEMENTS table cover?
 *
 * Ground truth for the two <select>s (ember.mjs #updateSoundscapeForm, :15916-15922):
 *   optgroup label = soundscape `s.label`
 *   option  label  = arrangement `label`  (per entry of `s.arrangements`)
 * Only soundscapes whose `type` is "music" or "environment" are offered.
 *
 * Extraction: read the frozen registry (`var soundscapes=Object.freeze({...})`)
 * to get the 42 local var names, then brace-match each `var <name> = {` block
 * and pull `id`/`label`/`type` plus the labels inside its `arrangements` block.
 *
 * Anti-空转 self-report: prints how many var blocks were located. Any soundscape
 * in the registry that could not be located is listed as MISSING — a run with
 * MISSING > 0 is an incomplete extraction, not a result.
 */
import fs from 'fs';

const EMBER = 'C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/scripts/ember.mjs';
const HC = 'C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs';

const src = fs.readFileSync(EMBER, 'utf8');

// --- registry ---
const reg = src.match(/var soundscapes=\/\*#__PURE__\*\/Object\.freeze\(\{__proto__:null,([^}]*)\}\)/);
if (!reg) throw new Error('registry not found');
const pairs = reg[1].split(',').map(s => s.trim()).filter(Boolean).map(s => {
  const [k, v] = s.split(':');
  return { key: k, varName: v };
});
console.log(`registry entries=${pairs.length}`);

// --- brace matcher that skips strings/template literals/comments ---
function blockAt(s, openIdx) {
  let d = 0, i = openIdx, q = null;
  for (; i < s.length; i++) {
    const c = s[i];
    if (q) {
      if (c === '\\') { i++; continue; }
      if (c === q) q = null;
      continue;
    }
    if (c === '"' || c === "'" || c === '`') { q = c; continue; }
    if (c === '/' && s[i + 1] === '/') { i = s.indexOf('\n', i); if (i < 0) break; continue; }
    if (c === '/' && s[i + 1] === '*') { i = s.indexOf('*/', i) + 1; continue; }
    if (c === '{') d++;
    else if (c === '}') { d--; if (d === 0) return s.slice(openIdx, i + 1); }
  }
  return null;
}

const missing = [];
const groups = [];          // {key,label,type}
const arrangements = [];    // {soundscape,label}
for (const { key, varName } of pairs) {
  const re = new RegExp(`(?:^|[;}])var ${varName.replace(/\$/g, '\\$')} = \\{`, 'm');
  const m = re.exec(src);
  if (!m) { missing.push(`${key}(${varName})`); continue; }
  const open = src.indexOf('{', m.index);
  const body = blockAt(src, open);
  if (!body) { missing.push(`${key}(${varName}):unbalanced`); continue; }
  const label = body.match(/\n  label: "([^"]*)"/)?.[1] ?? null;
  const type = body.match(/\n  type: "([^"]*)"/)?.[1] ?? null;
  groups.push({ key, label, type });
  const aIdx = body.search(/\n  arrangements: \{/);
  if (aIdx >= 0) {
    const aBody = blockAt(body, body.indexOf('{', aIdx));
    // arrangement entries sit one level in; their labels are at 6-space indent
    for (const mm of aBody.matchAll(/\n      label: "([^"]*)"/g)) {
      arrangements.push({ soundscape: key, label: mm[1] });
    }
  }
}

console.log(`located=${groups.length} MISSING=${missing.length}${missing.length ? ' -> ' + missing.join(', ') : ''}`);
if (missing.length) { console.log('EXTRACTION INCOMPLETE'); process.exit(2); }

const onScreen = groups.filter(g => ['music', 'environment'].includes(g.type));
const groupLabels = [...new Set(onScreen.map(g => g.label))];
const arrOnScreen = arrangements.filter(a => onScreen.some(g => g.key === a.soundscape));
const arrLabels = [...new Set(arrOnScreen.map(a => a.label))];

console.log(`\nsoundscapes total=${groups.length}  offered in the two <select>s (type music|environment)=${onScreen.length}`);
console.log(`  by type: ${JSON.stringify(onScreen.reduce((o, g) => (o[g.type] = (o[g.type] || 0) + 1, o), {}))}`);
console.log(`optgroup labels: ${onScreen.length} objects / ${groupLabels.length} unique`);
console.log(`option (arrangement) labels: ${arrOnScreen.length} entries / ${arrLabels.length} unique`);

// --- what does ARRANGEMENTS cover? ---
const hc = fs.readFileSync(HC, 'utf8');
const tbl = hc.slice(hc.indexOf('const ARRANGEMENTS = {'));
const keys = [...blockAt(tbl, tbl.indexOf('{')).matchAll(/"([^"]+)":\s*"/g)].map(m => m[1]);
console.log(`\nARRANGEMENTS table keys=${keys.length}: ${JSON.stringify(keys)}`);

const gCov = groupLabels.filter(l => keys.includes(l));
const aCov = arrLabels.filter(l => keys.includes(l));
console.log(`\ngroup labels covered=${gCov.length} -> ${JSON.stringify(gCov)}`);
console.log(`group labels UNCOVERED=${groupLabels.length - gCov.length}`);
console.log(groupLabels.filter(l => !keys.includes(l)).sort().join(' | '));
console.log(`\narrangement labels covered=${aCov.length} -> ${JSON.stringify(aCov)}`);
console.log(`arrangement labels UNCOVERED=${arrLabels.length - aCov.length}`);
console.log(arrLabels.filter(l => !keys.includes(l)).sort().slice(0, 40).join(' | '));
const unused = keys.filter(k => !groupLabels.includes(k) && !arrLabels.includes(k));
console.log(`\ntable keys matching NEITHER list=${unused.length}: ${JSON.stringify(unused)}`);
