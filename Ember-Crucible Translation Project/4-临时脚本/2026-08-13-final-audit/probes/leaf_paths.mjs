#!/usr/bin/env node
/**
 * Enumerate every STRING leaf path present in the ember / crucible LevelDB packs,
 * with counts and samples, so we can diff "what the packs contain" against
 * "what mappings.mjs pulls out".  Read-only.
 *
 *   node leaf_paths.mjs > leaf_paths.txt
 */
import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';

const { ClassicLevel } = createRequire('C:/Users/Taka/Desktop/fvtt/package.json')('classic-level');

const PKGS = [
  ['ember', 'C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember', 'module.json'],
  ['crucible', 'C:/Users/Taka/AppData/Local/FoundryVTT/Data/systems/crucible', 'system.json'],
];

/** Numeric-looking or id-looking dict key -> `*` so paths aggregate. */
const norm = (k) => k;

const stats = new Map();   // pkg|bucket|path -> {n, chars, samples:Set}

function record(pkg, bucket, p, v) {
  const key = `${pkg}\t${bucket}\t${p}`;
  let s = stats.get(key);
  if (!s) { s = { n: 0, chars: 0, samples: [] }; stats.set(key, s); }
  s.n += 1;
  s.chars += v.length;
  if (s.samples.length < 3 && v.trim()) s.samples.push(v.length > 90 ? v.slice(0, 90) + '…' : v);
}

function walk(pkg, bucket, node, p) {
  if (node === null || node === undefined) return;
  if (Array.isArray(node)) {
    for (const v of node) walk(pkg, bucket, v, p + '[]');
    return;
  }
  if (typeof node === 'object') {
    for (const [k, v] of Object.entries(node)) walk(pkg, bucket, v, p ? `${p}.${norm(k)}` : norm(k));
    return;
  }
  if (typeof node === 'string' && node.length) record(pkg, bucket, p, node);
}

for (const [pkgId, dir, manifest] of PKGS) {
  const m = JSON.parse(fs.readFileSync(path.join(dir, manifest), 'utf8'));
  for (const pk of m.packs ?? []) {
    const pd = path.join(dir, 'packs', path.basename(pk.path ?? pk.name));
    if (!fs.existsSync(pd)) { console.error(`# MISSING PACK DIR ${pkgId}/${pk.name}`); continue; }
    const db = new ClassicLevel(pd, { createIfMissing: false });
    for await (const [k, v] of db.iterator()) {
      const mm = k.toString().match(/^!([^!]+)!(.+)$/);
      if (!mm) continue;
      let doc; try { doc = JSON.parse(v.toString()); } catch { continue; }
      walk(`${pkgId}.${pk.name}`, mm[1], doc, '');
    }
    await db.close();
  }
}

const rows = [...stats.entries()].map(([k, v]) => {
  const [pkg, bucket, p] = k.split('\t');
  return { pkg, bucket, path: p, ...v };
});
// aggregate across packs by bucket+path
const agg = new Map();
for (const r of rows) {
  const key = `${r.bucket}\t${r.path}`;
  let a = agg.get(key);
  if (!a) { a = { bucket: r.bucket, path: r.path, n: 0, chars: 0, packs: new Set(), samples: [] }; agg.set(key, a); }
  a.n += r.n; a.chars += r.chars; a.packs.add(r.pkg);
  for (const s of r.samples) if (a.samples.length < 3) a.samples.push(s);
}
const out = [...agg.values()].sort((x, y) => y.chars - x.chars);
for (const a of out) {
  console.log(`${String(a.n).padStart(7)} ${String(a.chars).padStart(9)}  ${a.bucket} :: ${a.path}`);
  console.log(`         packs=${[...a.packs].join(',')}`);
  for (const s of a.samples) console.log(`         · ${JSON.stringify(s)}`);
}
