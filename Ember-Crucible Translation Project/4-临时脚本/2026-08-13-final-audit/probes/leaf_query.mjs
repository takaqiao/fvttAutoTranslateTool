#!/usr/bin/env node
/**
 * Per-pack query over the LevelDB packs: show every string leaf whose dotted
 * path matches a regex, with pack / bucket / doc-name / value.  Read-only.
 *
 *   node leaf_query.mjs "<path-regex>" [--max N] [--pack <substr>] [--paths]
 */
import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';
const { ClassicLevel } = createRequire('C:/Users/Taka/Desktop/fvtt/package.json')('classic-level');

const argv = process.argv.slice(2);
const rx = new RegExp(argv[0]);
const arg = (n, d) => { const i = argv.indexOf(n); return i >= 0 ? argv[i + 1] : d; };
const MAX = Number(arg('--max', 60));
const PACKFILTER = arg('--pack', '');
const PATHS_ONLY = argv.includes('--paths');

const PKGS = [
  ['ember', 'C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember', 'module.json'],
  ['crucible', 'C:/Users/Taka/AppData/Local/FoundryVTT/Data/systems/crucible', 'system.json'],
];

const byPath = new Map();
let shown = 0;

function walk(ctx, node, p, out) {
  if (node === null || node === undefined) return;
  if (Array.isArray(node)) { node.forEach((v) => walk(ctx, v, p + '[]', out)); return; }
  if (typeof node === 'object') {
    for (const [k, v] of Object.entries(node)) walk(ctx, v, p ? `${p}.${k}` : k, out);
    return;
  }
  if (typeof node !== 'string' || !node.length) return;
  if (!rx.test(p)) return;
  const key = `${ctx.pack}\t${ctx.bucket}\t${p}`;
  let a = byPath.get(key);
  if (!a) { a = { n: 0, vals: [] }; byPath.set(key, a); }
  a.n += 1;
  if (a.vals.length < 4) a.vals.push(node.length > 160 ? node.slice(0, 160) + '…' : node);
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
      walk({ pack: packId, bucket: mm[1] }, doc, '', null);
    }
    await db.close();
  }
}

for (const [k, a] of [...byPath.entries()].sort((x, y) => y[1].n - x[1].n)) {
  const [pack, bucket, p] = k.split('\t');
  console.log(`${String(a.n).padStart(6)}  ${pack} [${bucket}] ${p}`);
  if (!PATHS_ONLY) for (const v of a.vals) console.log(`         · ${JSON.stringify(v)}`);
  if (++shown >= MAX) { console.log('… truncated'); break; }
}
