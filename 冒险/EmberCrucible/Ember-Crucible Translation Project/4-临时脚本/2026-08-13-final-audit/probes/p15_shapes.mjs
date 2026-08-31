#!/usr/bin/env node
/** P15: exact shape + fill rate of Ember's GM-shaped page fields. READ ONLY. */
import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';
const { ClassicLevel } = createRequire('C:/Users/Taka/Desktop/fvtt/package.json')('classic-level');
const PKG = 'C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/packs';

const stat = {};
function bump(f, filled, sample) {
  stat[f] ??= { seen: 0, filled: 0, sample: null, types: {} };
  stat[f].seen++;
  if (filled) { stat[f].filled++; stat[f].sample ??= sample; }
}
function walkSys(sub, sys, pref, depth) {
  for (const [k, v] of Object.entries(sys ?? {})) {
    const f = pref + k;
    const t = Array.isArray(v) ? 'array' : v === null ? 'null' : typeof v;
    stat[f] ??= { seen: 0, filled: 0, sample: null, types: {} };
    stat[f].types[t] = (stat[f].types[t] || 0) + 1;
    const filled = t === 'string' ? v.trim().length > 0
      : t === 'array' ? v.length > 0
      : t === 'object' ? Object.keys(v).length > 0 : v !== null && v !== undefined;
    bump(f, filled, typeof v === 'string' ? v.slice(0, 160) : JSON.stringify(v).slice(0, 200));
    if (t === 'object' && depth < 2) walkSys(sub, v, f + '.', depth + 1);
  }
}
const pages = [];
for (const pack of fs.readdirSync(PKG).filter((d) => fs.statSync(path.join(PKG, d)).isDirectory())) {
  const db = new ClassicLevel(path.join(PKG, pack), { valueEncoding: 'json' });
  await db.open();
  for await (const [, v] of db.iterator()) {
    const js = [];
    if (Array.isArray(v?.journal)) js.push(...v.journal);
    if (Array.isArray(v?.pages)) js.push(v);
    for (const j of js) for (const p of j.pages ?? []) { pages.push(p); walkSys(p.type, p.system, '', 0); }
  }
  await db.close();
}
console.log('pages scanned:', pages.length);
const keys = Object.keys(stat).filter((k) => /secret|hook|develop|gm|private|content/i.test(k));
for (const k of keys.sort()) {
  const s = stat[k];
  console.log(`${k}: seen=${s.seen} filled=${s.filled} types=${JSON.stringify(s.types)}`);
  if (s.sample) console.log('    e.g.', String(s.sample).replace(/\n/g, ' ').slice(0, 180));
}
