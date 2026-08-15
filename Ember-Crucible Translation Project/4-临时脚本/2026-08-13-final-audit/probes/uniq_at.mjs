#!/usr/bin/env node
/** Distinct values at a path-regex, with counts and the journal page they sit on. */
import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';
const { ClassicLevel } = createRequire('C:/Users/Taka/Desktop/fvtt/package.json')('classic-level');
const argv = process.argv.slice(2);
const rx = new RegExp(argv[0]);
const arg = (n, d) => { const i = argv.indexOf(n); return i >= 0 ? argv[i + 1] : d; };
const PACKFILTER = arg('--pack', '');
const PKGS = [
  ['ember', 'C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember', 'module.json'],
  ['crucible', 'C:/Users/Taka/AppData/Local/FoundryVTT/Data/systems/crucible', 'system.json'],
];
const counts = new Map();
function walk(node, p, hit) {
  if (node === null || node === undefined) return;
  if (Array.isArray(node)) { node.forEach((v) => walk(v, p + '[]', hit)); return; }
  if (typeof node === 'object') { for (const [k, v] of Object.entries(node)) walk(v, p ? `${p}.${k}` : k, hit); return; }
  if (typeof node === 'string' && node.length && rx.test(p)) hit(p, node);
}
for (const [pkgId, dir, manifest] of PKGS) {
  const m = JSON.parse(fs.readFileSync(path.join(dir, manifest), 'utf8'));
  for (const pk of m.packs ?? []) {
    const packId = `${pkgId}.${pk.name}`;
    if (PACKFILTER && !packId.includes(PACKFILTER)) continue;
    const pd = path.join(dir, 'packs', path.basename(pk.path ?? pk.name));
    if (!fs.existsSync(pd)) continue;
    const db = new ClassicLevel(pd, { createIfMissing: false });
    for await (const [k, v] of db.iterator()) {
      const mm = k.toString().match(/^!([^!]+)!(.+)$/);
      if (!mm) continue;
      let doc; try { doc = JSON.parse(v.toString()); } catch { continue; }
      walk(doc, '', (p, val) => counts.set(val, (counts.get(val) || 0) + 1));
    }
    await db.close();
  }
}
const rows = [...counts.entries()].sort((a, b) => b[1] - a[1]);
console.log(`# ${rows.length} distinct values, ${rows.reduce((s, r) => s + r[1], 0)} occurrences`);
for (const [v, n] of rows) console.log(`${String(n).padStart(5)}  ${JSON.stringify(v)}`);
