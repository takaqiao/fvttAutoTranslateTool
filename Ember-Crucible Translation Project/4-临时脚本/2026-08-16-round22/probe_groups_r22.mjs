/**
 * Round-21 probe ③: dump the 42 on-screen soundscape group names (optgroup labels)
 * and the 212 unique arrangement labels, and report which of them the mood-panel
 * scope table (MOOD_PANEL) covers.
 *
 * Extraction is done TWICE by two independent methods and the results must agree:
 *   A) brace-match each `var <name> = {` block, regex out label/type/arrangement labels
 *      (the round-20 method)
 *   B) node:vm — slice each top-level `var <name> = { ... \n};` and let the real JS
 *      parser build the object, then require obj.id === registry key
 * Any disagreement => exit(2). Any unresolved soundscape => exit(2).
 *
 * MOOD_PANEL is read out of ember-hardcoded-cn.mjs by importing it? No — the table is
 * not exported. Instead we import the module's translateText via a DOM-free shim is
 * impossible, so here we only read the table text; the real translateText check lives
 * in probe_translate.mjs.
 */
import fs from 'fs';
import vm from 'node:vm';

const EMBER = 'C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/scripts/ember.mjs';
const OUT = 'C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/4-临时脚本/2026-08-16-round22/soundscapes_r22.json';

const src = fs.readFileSync(EMBER, 'utf8');

const reg = src.match(/var soundscapes=\/\*#__PURE__\*\/Object\.freeze\(\{__proto__:null,([^}]*)\}\)/);
if (!reg) throw new Error('registry not found');
const pairs = reg[1].split(',').map(s => s.trim()).filter(Boolean).map(s => {
  const [k, v] = s.split(':');
  return { key: k, varName: v };
});
console.log(`registry entries=${pairs.length}`);
if (pairs.length !== 44) { console.log('UNEXPECTED registry size'); process.exit(2); }

function blockAt(s, openIdx) {
  let d = 0, i = openIdx, q = null;
  for (; i < s.length; i++) {
    const c = s[i];
    if (q) { if (c === '\\') { i++; continue; } if (c === q) q = null; continue; }
    if (c === '"' || c === "'" || c === '`') { q = c; continue; }
    if (c === '/' && s[i + 1] === '/') { i = s.indexOf('\n', i); if (i < 0) break; continue; }
    if (c === '/' && s[i + 1] === '*') { i = s.indexOf('*/', i) + 1; continue; }
    if (c === '{') d++;
    else if (c === '}') { d--; if (d === 0) return s.slice(openIdx, i + 1); }
  }
  return null;
}

const A = { groups: [], arr: [] };   // method A
const B = { groups: [], arr: [] };   // method B
const bad = [];

for (const { key, varName } of pairs) {
  const re = new RegExp(`(?:^|[;}])var ${varName.replace(/\$/g, '\\$')} = \\{`, 'm');
  const m = re.exec(src);
  if (!m) { bad.push(`${key}:notfound`); continue; }
  const open = src.indexOf('{', m.index);
  const body = blockAt(src, open);
  if (!body) { bad.push(`${key}:unbalanced`); continue; }

  // --- A ---
  A.groups.push({
    key,
    label: body.match(/\n  label: "([^"]*)"/)?.[1] ?? null,
    type: body.match(/\n  type: "([^"]*)"/)?.[1] ?? null
  });
  const aIdx = body.search(/\n  arrangements: \{/);
  if (aIdx >= 0) {
    const aBody = blockAt(body, body.indexOf('{', aIdx));
    for (const mm of aBody.matchAll(/\n      label: "([^"]*)"/g)) A.arr.push({ key, label: mm[1] });
  }

  // --- B: real parser ---
  let obj;
  try { obj = vm.runInNewContext('(' + body + ')'); }
  catch (e) { bad.push(`${key}:vm ${e.message}`); continue; }
  if (obj?.id !== key) { bad.push(`${key}:id-mismatch(${obj?.id})`); continue; }
  B.groups.push({ key, label: obj.label ?? null, type: obj.type ?? null });
  for (const a of Object.values(obj.arrangements ?? {})) {
    if (a && typeof a.label === 'string') B.arr.push({ key, label: a.label });
  }
}

console.log(`resolved: A=${A.groups.length} B=${B.groups.length} BAD=${bad.length}${bad.length ? ' -> ' + bad.join('; ') : ''}`);
if (bad.length || A.groups.length !== 44 || B.groups.length !== 44) { console.log('EXTRACTION INCOMPLETE'); process.exit(2); }

const norm = o => JSON.stringify(o.groups) + '||' + JSON.stringify([...o.arr].sort((x, y) => (x.key + x.label).localeCompare(y.key + y.label)));
if (norm(A) !== norm(B)) {
  console.log('METHOD DISAGREEMENT');
  const ag = new Set(A.groups.map(g => g.key + '=' + g.label)); const bg = new Set(B.groups.map(g => g.key + '=' + g.label));
  console.log('A-only groups:', [...ag].filter(x => !bg.has(x)));
  console.log('B-only groups:', [...bg].filter(x => !ag.has(x)));
  console.log(`arr counts A=${A.arr.length} B=${B.arr.length}`);
  process.exit(2);
}
console.log('two methods AGREE (groups + arrangement labels identical)');

const onScreen = B.groups.filter(g => ['music', 'environment'].includes(g.type));
const groupLabels = [...new Set(onScreen.map(g => g.label))];
const arrOn = B.arr.filter(a => onScreen.some(g => g.key === a.key));
const arrLabels = [...new Set(arrOn.map(a => a.label))];
console.log(`on-screen soundscapes=${onScreen.length} (excluded: ${B.groups.filter(g => !onScreen.includes(g)).map(g => g.label).join(', ')})`);
console.log(`group labels: ${onScreen.length} objects / ${groupLabels.length} unique`);
console.log(`arrangement labels: ${arrOn.length} entries / ${arrLabels.length} unique`);

// overlap between the two tiers (the "Ancient Ruins is both" claim)
const both = groupLabels.filter(l => arrLabels.includes(l));
console.log(`labels that are BOTH a group name and an arrangement name (${both.length}): ${JSON.stringify(both)}`);

fs.writeFileSync(OUT, JSON.stringify({
  groupLabels: groupLabels.sort(),
  arrLabels: arrLabels.sort(),
  both,
  perSoundscape: onScreen.map(g => ({ key: g.key, label: g.label, type: g.type,
    arrangements: arrOn.filter(a => a.key === g.key).map(a => a.label) }))
}, null, 2), 'utf8');
console.log(`wrote ${OUT}`);
