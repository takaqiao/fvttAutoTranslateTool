/**
 * Exhaustive scan of EVERY executable string upstream ember 0.6.1 carries in the
 * two adventure packs (region-behaviour `system.script` / `system.source`, and
 * Macro `command`), looking for user-facing string literals.
 *
 * SELF-PROOF, two separate assertions:
 *   A. COUNT   : the number of executable strings found must equal the counts the
 *                type-inventory probe measured independently
 *                (executeScript.source 13, ember.trapTrigger.script 2, macros 6/7).
 *   B. IDENTITY: the one literal already known by hand
 *                ("You must select a single Grayling token.") must be found,
 *                exactly twice, both inside the macro named `Reveal Grayling`.
 */
import { createRequire } from 'module';
import fs from 'fs';
import path from 'path';

const require = createRequire('C:/Users/Taka/Desktop/fvtt/package.json');
const { ClassicLevel } = require('classic-level');
const PKG = 'C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember';

const TRUTH = {
  adventure: { source: 13, script: 2, macros: 6 },
  'crucible-adventure': { source: 13, script: 2, macros: 7 },
};

/** crude but sufficient JS string-literal lexer for these snippets */
function literals(code) {
  const out = [];
  const re = /(['"`])(?:\\.|(?!\1)[^\\])*\1/g;
  let m;
  while ((m = re.exec(code))) out.push(m[0].slice(1, -1));
  return out;
}
/** does a literal look like on-screen prose rather than an identifier/path? */
const looksLikeProse = (s) => /\s/.test(s) && /[a-z]{2}/.test(s) && !s.includes('/') && !s.startsWith('.') && !/^[a-z]+([A-Z][a-z]+)+$/.test(s);

const report = {};
let allOk = true;

for (const pack of ['adventure', 'crucible-adventure']) {
  const db = new ClassicLevel(path.join(PKG, 'packs', pack), { valueEncoding: 'json' });
  await db.open();
  const rows = [];
  for await (const [k, v] of db.iterator()) rows.push(v);
  await db.close();

  const execStrings = [];
  for (const v of rows) {
    for (const s of (Array.isArray(v?.scenes) ? v.scenes : [])) {
      for (const r of (s.regions ?? [])) {
        for (const b of (r.behaviors ?? [])) {
          if (typeof b.system?.source === 'string' && b.system.source.trim()) {
            execStrings.push({ kind: 'source', where: `${s.name}/${r.name}/${b.name}`, code: b.system.source });
          }
          if (typeof b.system?.script === 'string' && b.system.script.trim()) {
            execStrings.push({ kind: 'script', where: `${s.name}/${r.name}/${b.name}`, code: b.system.script });
          }
        }
      }
    }
    for (const mm of (v?.macros ?? [])) {
      if (typeof mm.command === 'string') execStrings.push({ kind: 'macro', where: mm.name, code: mm.command });
    }
  }

  const counts = { source: 0, script: 0, macros: 0 };
  for (const e of execStrings) counts[e.kind === 'macro' ? 'macros' : e.kind]++;
  const A_ok = counts.source === TRUTH[pack].source && counts.script === TRUTH[pack].script && counts.macros === TRUTH[pack].macros;

  const prose = [];
  for (const e of execStrings) for (const lit of literals(e.code)) if (looksLikeProse(lit)) prose.push({ ...e, literal: lit, code: undefined });

  const target = prose.filter((p) => p.literal === 'You must select a single Grayling token.');
  const B_ok = target.length === 2 && target.every((p) => p.where === 'Reveal Grayling');
  if (!A_ok || !B_ok) allOk = false;

  report[pack] = {
    A_counts: counts, A_expected: TRUTH[pack], A_ok,
    B_target_hits: target.length, B_all_in_reveal_grayling: target.every((p) => p.where === 'Reveal Grayling'), B_ok,
    proseLiterals: prose,
    allLiteralsSample: [...new Set(execStrings.flatMap((e) => literals(e.code)))].sort(),
  };
  console.log(`[${pack}] exec strings: source=${counts.source} script=${counts.script} macro=${counts.macros}  A_ok=${A_ok} B_ok=${B_ok}`);
  console.log(`   prose-looking literals (${prose.length}):`);
  for (const p of prose) console.log(`     ${p.kind} @ ${p.where}: ${JSON.stringify(p.literal)}`);
}

fs.writeFileSync('C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/4-临时脚本/2026-08-22-ui-round3/probe_scripts.json',
  JSON.stringify(report, null, 2), 'utf8');
if (!allOk) { console.error('SELF-PROOF FAILED'); process.exit(2); }
console.log('SELF-PROOF OK (A: counts match the independent type inventory; B: known literal found at its known coordinates)');
