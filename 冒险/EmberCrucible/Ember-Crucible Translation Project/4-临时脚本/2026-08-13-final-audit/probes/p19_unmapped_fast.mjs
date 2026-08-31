#!/usr/bin/env node
/**
 * P19: fast version of P17. Build a Set of every string leaf in the project's
 * EN baseline, then walk the live packs and report prose leaves absent from it,
 * grouped by field path. READ ONLY.
 */
import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';
const { ClassicLevel } = createRequire('C:/Users/Taka/Desktop/fvtt/package.json')('classic-level');
const PKG = 'C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/packs';
const EN = 'C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/1-Ember汉化插件/compendium/en';

const known = new Set();
function collect(o) {
  if (o == null) return;
  if (Array.isArray(o)) { o.forEach(collect); return; }
  if (typeof o === 'object') { Object.values(o).forEach(collect); return; }
  if (typeof o === 'string') known.add(o.trim());
}
for (const f of fs.readdirSync(EN)) if (f.endsWith('.json')) collect(JSON.parse(fs.readFileSync(path.join(EN, f), 'utf8')));
console.log('distinct strings in the EN baseline:', known.size);

const PROSE = (s) => {
  const t = s.replace(/<[^>]+>/g, '').trim();
  if (t.length < 4) return false;
  if (!/[A-Za-z]/.test(t)) return false;
  if (/^[a-z][A-Za-z0-9]*$/.test(t)) return false;
  if (/^[A-Za-z][A-Za-z0-9]*\.[A-Za-z0-9]{6,}/.test(t) && !/\s/.test(t)) return false;
  if (/^[a-z0-9_\-.]+$/.test(t)) return false;
  if (/^[0-9a-zA-Z]{16}$/.test(t)) return false;          // foundry id
  if (/^(modules|systems|icons|worlds|assets)\//.test(t)) return false;
  if (/^#[0-9a-fA-F]{3,8}$/.test(t)) return false;
  if (/^\{/.test(t)) return false;
  return true;
};

const g = {};
function walk(o, p, d) {
  if (d > 9 || o == null) return;
  if (Array.isArray(o)) { for (const v of o) walk(v, p + '[]', d + 1); return; }
  if (typeof o === 'object') { for (const [k, v] of Object.entries(o)) walk(v, p ? `${p}.${k}` : k, d + 1); return; }
  if (typeof o !== 'string' || !PROSE(o)) return;
  g[p] ??= { n: 0, miss: 0, chars: 0, s: new Set() };
  g[p].n++;
  if (!known.has(o.trim())) {
    g[p].miss++;
    g[p].chars += o.replace(/<[^>]+>/g, '').length;
    if (g[p].s.size < 3) g[p].s.add(o.slice(0, 130));
  }
}
for (const pack of fs.readdirSync(PKG).filter((d) => fs.statSync(path.join(PKG, d)).isDirectory())) {
  const db = new ClassicLevel(path.join(PKG, pack), { valueEncoding: 'json' });
  await db.open();
  for await (const [, v] of db.iterator()) walk(v, '', 0);
  await db.close();
}
const rows = Object.entries(g).filter(([, x]) => x.miss > 0).sort((a, b) => b[1].chars - a[1].chars);
console.log('field paths carrying prose absent from the EN baseline:', rows.length);
for (const [p, x] of rows.slice(0, 40)) {
  console.log(`\n${p}\n   values=${x.n} missing=${x.miss} missingChars=${x.chars}`);
  for (const s of x.s) console.log('   e.g.', s.replace(/\n/g, ' '));
}
