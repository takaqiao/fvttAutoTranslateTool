#!/usr/bin/env node
/** P16: full system key inventory for the two page types Ember marks gmOnly. READ ONLY. */
import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';
const { ClassicLevel } = createRequire('C:/Users/Taka/Desktop/fvtt/package.json')('classic-level');
const PKG = 'C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/packs';

const TYPES = new Set(['ember.questEvent', 'ember.standaloneEvent']);
const stat = {};
function walk(o, pref, depth) {
  for (const [k, v] of Object.entries(o ?? {})) {
    const f = pref + k;
    const t = Array.isArray(v) ? 'array' : v === null ? 'null' : typeof v;
    stat[f] ??= { seen: 0, filled: 0, chars: 0, sample: null, t: {} };
    stat[f].t[t] = (stat[f].t[t] || 0) + 1;
    stat[f].seen++;
    const txt = t === 'string' ? v.replace(/<[^>]+>/g, '').trim() : '';
    if (txt) { stat[f].filled++; stat[f].chars += txt.length; stat[f].sample ??= v.slice(0, 220); }
    if (t === 'array' && v.length) { stat[f].filled++; stat[f].sample ??= JSON.stringify(v).slice(0, 300); v.forEach((x) => { if (x && typeof x === 'object' && depth < 4) walk(x, f + '[].', depth + 1); }); }
    if (t === 'object' && depth < 4) walk(v, f + '.', depth + 1);
  }
}
let n = 0;
for (const pack of fs.readdirSync(PKG).filter((d) => fs.statSync(path.join(PKG, d)).isDirectory())) {
  const db = new ClassicLevel(path.join(PKG, pack), { valueEncoding: 'json' });
  await db.open();
  for await (const [, v] of db.iterator()) {
    const js = [];
    if (Array.isArray(v?.journal)) js.push(...v.journal);
    if (Array.isArray(v?.pages)) js.push(v);
    for (const j of js) for (const p of j.pages ?? []) if (TYPES.has(p.type)) { n++; walk(p.system, '', 0); }
  }
  await db.close();
}
console.log('questEvent/standaloneEvent pages:', n);
for (const k of Object.keys(stat).sort()) {
  const s = stat[k];
  console.log(`${k}  seen=${s.seen} filled=${s.filled} chars=${s.chars} ${JSON.stringify(s.t)}`);
  if (s.sample) console.log('     e.g.', String(s.sample).replace(/\n/g, ' ').slice(0, 200));
}
