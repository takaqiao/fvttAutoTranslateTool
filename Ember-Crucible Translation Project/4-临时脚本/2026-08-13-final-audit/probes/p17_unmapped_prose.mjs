#!/usr/bin/env node
/**
 * P17: prose in the LIVE Ember packs that never reaches the EN baseline.
 *
 * The extractor is mapping-driven, so `compendium/en` == "what the mapping
 * asks for". Any prose field the mapping omits is outside the domain of every
 * existing gate. Test: walk the live LevelDB, collect every string leaf that
 * looks like prose, and check whether that exact string occurs anywhere in the
 * project's EN extract for the same pack. Miss => unmapped.
 *
 * False positives: strings that are ids/enums/numbers (filtered), and strings
 * the extractor legitimately normalises. Every reported group is printed with
 * a sample so it can be judged. READ ONLY.
 */
import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';
const { ClassicLevel } = createRequire('C:/Users/Taka/Desktop/fvtt/package.json')('classic-level');

const PKG = 'C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/packs';
const EN = 'C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/1-Ember汉化插件/compendium/en';

const enText = {};
for (const f of fs.readdirSync(EN)) {
  if (f.endsWith('.json')) enText[f.replace(/^ember\./, '').replace(/\.json$/, '')] = fs.readFileSync(path.join(EN, f), 'utf8');
}

const PROSE = (s) => {
  const t = s.replace(/<[^>]+>/g, '').trim();
  if (t.length < 3) return false;
  if (!/[A-Za-z]/.test(t)) return false;
  if (/^[a-z][A-Za-z0-9]*$/.test(t)) return false;             // camelCase id
  if (/^[A-Za-z]+\.[A-Za-z0-9]+/.test(t) && !/\s/.test(t)) return false; // Doc.id
  if (/^[a-z0-9_\-.]+$/.test(t)) return false;                  // slug
  if (/^\{.*\}$/.test(t)) return false;                          // json blob
  return true;
};

const groups = {};
function leaf(pack, gpath, val) {
  if (typeof val !== 'string' || !PROSE(val)) return;
  const hay = enText[pack] ?? '';
  const needle = JSON.stringify(val).slice(1, -1);
  const found = hay.includes(needle) || hay.includes(needle.slice(0, 120));
  groups[gpath] ??= { n: 0, missing: 0, chars: 0, samples: [] };
  const g = groups[gpath];
  g.n++;
  if (!found) {
    g.missing++;
    g.chars += val.replace(/<[^>]+>/g, '').length;
    if (g.samples.length < 3) g.samples.push(`${pack}: ${val.slice(0, 150)}`);
  }
}
function walk(pack, o, p, d) {
  if (d > 8 || o == null) return;
  if (Array.isArray(o)) { for (const v of o) walk(pack, v, p + '[]', d + 1); return; }
  if (typeof o === 'object') { for (const [k, v] of Object.entries(o)) walk(pack, v, p ? p + '.' + k : k, d + 1); return; }
  leaf(pack, p, o);
}

for (const pack of fs.readdirSync(PKG).filter((d) => fs.statSync(path.join(PKG, d)).isDirectory())) {
  const db = new ClassicLevel(path.join(PKG, pack), { valueEncoding: 'json' });
  await db.open();
  for await (const [, v] of db.iterator()) walk(pack, v, '', 0);
  await db.close();
}

const rows = Object.entries(groups).filter(([, g]) => g.missing > 0).sort((a, b) => b[1].chars - a[1].chars);
console.log('field paths with prose missing from the EN baseline:', rows.length);
for (const [p, g] of rows.slice(0, 45)) {
  console.log(`\n${p}\n   values=${g.n} missing=${g.missing} missingChars=${g.chars}`);
  for (const s of g.samples) console.log('   e.g.', s.replace(/\n/g, ' '));
}
