/**
 * Round-22 probe: a REAL ESM `import` of ember-hardcoded-cn.mjs (not a vm re-exec),
 * calling the exported translateText with the correct scope table.
 *
 * The module registers a `Hooks.once("ready", …)` at top level, so a Foundry-shaped
 * `Hooks` stub is planted on globalThis before importing.
 *
 * MOOD_PANEL is not exported, so the scope table is rebuilt here **from the module's
 * own source text** and then cross-checked: every key must round-trip through the
 * imported translateText, and the负控制 (no scope table) must translate none of them.
 *
 * Anti-空转: prints how many labels were fed and lists the ones that stay English.
 * Exits 2 on any structural surprise, 1 on a failed control.
 */
globalThis.Hooks = { once() { globalThis.__hookRegistered = true; } };
globalThis.Node = { TEXT_NODE: 3, ELEMENT_NODE: 1 };

import fs from 'fs';
import { pathToFileURL } from 'node:url';

const HC = 'C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/1-Ember汉化插件/scripts/ember-hardcoded-cn.mjs';
const DATA = new URL('./soundscapes_r22.json', import.meta.url);

const mod = await import(pathToFileURL(HC).href);
if (typeof mod.translateText !== 'function') { console.log('NO EXPORT translateText'); process.exit(2); }
console.log(`imported OK; Hooks.once registered: ${globalThis.__hookRegistered === true}`);

const data = JSON.parse(fs.readFileSync(DATA, 'utf8'));
const groups = data.groupLabels, arrs = data.arrLabels;
if (groups.length !== 42 || arrs.length !== 212) { console.log('INPUT SHAPE UNEXPECTED'); process.exit(2); }

// rebuild the scope table from the module source (same three parts MOOD_PANEL spreads)
const src = fs.readFileSync(HC, 'utf8');
function tableOf(name) {
  const m = src.match(new RegExp('const ' + name + ' = \\{([\\s\\S]*?)\\n\\};'));
  if (!m) { console.log('TABLE NOT FOUND: ' + name); process.exit(2); }
  const out = {};
  for (const r of m[1].matchAll(/^\s*"([^"]+)":\s*"([^"]*)"/gm)) out[r[1]] = r[2];
  return out;
}
const SCOPE = { ...tableOf('SOUNDSCAPE_GROUPS'), ...tableOf('ARRANGEMENTS'),
                'Ember Music': '余烬乐曲', 'Rearrange Music': '重新编排音乐', 'Ember Default': '余烬默认' };
console.log(`scope table rebuilt from source: ${Object.keys(SCOPE).length} keys ` +
            `(groups ${Object.keys(tableOf('SOUNDSCAPE_GROUPS')).length} / arrangements ${Object.keys(tableOf('ARRANGEMENTS')).length})`);

let fed = 0;
const stillEn = [];
for (const s of arrs) { fed++; if (mod.translateText(s, SCOPE) === s) stillEn.push(s); }
const gEn = [];
for (const s of groups) { fed++; if (mod.translateText(s, SCOPE) === s) gEn.push(s); }
let bare = 0;
for (const s of groups.concat(arrs)) { fed++; if (mod.translateText(s, null) !== s) bare++; }

console.log(`fed ${fed} strings through the imported translateText`);
console.log(`group names 恒英文 ${gEn.length}: ${JSON.stringify(gEn)}`);
console.log(`arrangements 恒英文 ${stillEn.length}: ${JSON.stringify(stillEn)}`);
console.log(`negative control (no scope table): ${bare} translated (must be 0)`);
console.log('spot checks:');
for (const s of ['Celestial Combat Section 3', 'Helkas Attack (Drakes)', 'Shrine of Nite Calm',
                 'Ooze Fight - Weird', 'Rustvar Valley Tension', 'The Teeth Night', 'Seven Sails'])
  console.log(`  ${s.padEnd(30)} -> ${mod.translateText(s, SCOPE)}`);
console.log('enricher path (global, no scope table):');
for (const s of ['Music: Bandit Fight Chorus', 'Environment: Ordain Interior Day',
                 'Music: Shent Ruins (Tension)', 'Music: Seven Sails'])
  console.log(`  ${s.padEnd(34)} -> ${mod.translateText(s)}`);

const fail = bare !== 0 || gEn.length !== 0 || stillEn.length !== 1 || stillEn[0] !== 'Seven Sails';
console.log(`\nVERDICT: ${fail ? 'FAIL' : 'PASS'}`);
process.exit(fail ? 1 : 0);
