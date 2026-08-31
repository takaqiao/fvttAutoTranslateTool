/**
 * Round-20 probe ③ (v2): is `Hexblade` still a LIVE key?
 *
 * v1 scanned only top-level pack docs and reported 0 pages — a FALSE NEGATIVE:
 * the deity journal pages live nested inside the Adventure doc
 * (`!adventures!emberAlphaOne000` → .journal[].pages[]). v2 walks the whole
 * object graph so nesting depth cannot hide a value.
 *
 * Anti-空转 self-report: prints nodes walked + pages carrying the field.
 * A run with pagesWithField == 0 is a broken probe, NOT evidence of absence.
 *
 * Usage: node probe_patrons2.mjs --package <dir>
 */
import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';
const { ClassicLevel } = createRequire('C:/Users/Taka/Desktop/fvtt/package.json')('classic-level');
const argv = process.argv.slice(2);
const arg = (n, d) => { const i = argv.indexOf(n); return i >= 0 ? argv[i + 1] : d; };
const PACKAGE_DIR = arg('--package');

const FIELDS = ['clerics', 'warlocks', 'sorcerers'];
const counts = Object.fromEntries(FIELDS.map(f => [f, new Map()]));
const where = new Map();
let nodes = 0, pagesWithField = 0, docsSeen = 0;

function walk(node, pack, trail) {
  if (node == null || typeof node !== 'object') return;
  nodes++;
  if (Array.isArray(node)) { for (const x of node) walk(x, pack, trail); return; }
  const name = typeof node.name === 'string' ? node.name : null;
  const t2 = name ? `${trail}/${name}` : trail;
  const sys = node.system;
  if (sys && typeof sys === 'object') {
    let hit = false;
    for (const f of FIELDS) {
      const v = sys[f];
      if (v == null) continue;
      hit = true;
      for (const item of (Array.isArray(v) ? v : [v])) {
        if (typeof item !== 'string' || !item) continue;
        counts[f].set(item, (counts[f].get(item) || 0) + 1);
        if (!where.has(item)) where.set(item, []);
        if (where.get(item).length < 6) where.get(item).push(`${pack}:${t2}`);
      }
    }
    if (hit) pagesWithField++;
  }
  for (const [k, v] of Object.entries(node)) {
    if (k === 'system' || typeof v !== 'object' || v == null) continue;
    walk(v, pack, t2);
  }
  if (sys && typeof sys === 'object') for (const v of Object.values(sys)) walk(v, pack, t2);
}

const packsDir = path.join(PACKAGE_DIR, 'packs');
for (const p of fs.readdirSync(packsDir)) {
  const dir = path.join(packsDir, p);
  if (!fs.statSync(dir).isDirectory()) continue;
  const db = new ClassicLevel(dir, { keyEncoding: 'utf8', valueEncoding: 'json' });
  await db.open();
  for await (const [, doc] of db.iterator()) { docsSeen++; walk(doc, p, ''); }
  await db.close();
}

console.log(`SCANNED docs=${docsSeen} nodes=${nodes} pagesWithField=${pagesWithField}`);
if (pagesWithField === 0) { console.log('PROBE FAILED: 0 pages carried the field'); process.exit(2); }
for (const f of FIELDS) {
  const m = [...counts[f].entries()].sort((a, b) => a[0].localeCompare(b[0]));
  console.log(`\n${f}: ${m.length} unique / ${m.reduce((s, [, n]) => s + n, 0)} leaves`);
  for (const [v, n] of m) console.log(`  ${JSON.stringify(v)} x${n}`);
}
console.log('\n--- Hexblade sightings ---');
console.log(where.has('Hexblade') ? where.get('Hexblade').join('\n') : '(none)');
