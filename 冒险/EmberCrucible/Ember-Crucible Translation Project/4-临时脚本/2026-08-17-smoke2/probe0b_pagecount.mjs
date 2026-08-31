#!/usr/bin/env node
/**
 * probe0b —— 诊断 probe0 里唯一一条对不上的真值（「两包 3208 个页面」）。
 * 逐包、逐来源（兄弟桶 vs adventure blob 内部）数 JournalEntryPage，
 * 好判断是「我少数了一处」还是「上游与记录不同」。
 */
import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';

const { ClassicLevel } = createRequire('C:/Users/Taka/Desktop/fvtt/package.json')('classic-level');
const DATA = 'C:/Users/Taka/AppData/Local/FoundryVTT/Data';
const PKGS = [
  { id: 'crucible', dir: path.join(DATA, 'systems/crucible'), manifest: 'system.json' },
  { id: 'ember', dir: path.join(DATA, 'modules/ember'), manifest: 'module.json' },
];

function countPagesIn(node, out, depthKey) {
  if (Array.isArray(node)) { for (const x of node) countPagesIn(x, out, depthKey); return; }
  if (!node || typeof node !== 'object') return;
  if (Array.isArray(node.pages)) {
    for (const p of node.pages) if (p && typeof p === 'object' && typeof p._id === 'string') {
      out.n += 1;
      out.types[p.type ?? '(none)'] = (out.types[p.type ?? '(none)'] ?? 0) + 1;
    }
  }
  for (const v of Object.values(node)) countPagesIn(v, out, depthKey);
}

const rows = [];
for (const pkg of PKGS) {
  const manifest = JSON.parse(fs.readFileSync(path.join(pkg.dir, pkg.manifest), 'utf8'));
  for (const p of manifest.packs ?? []) {
    const dir = path.join(pkg.dir, 'packs', path.basename(p.path ?? p.name));
    if (!fs.existsSync(dir)) continue;
    const db = new ClassicLevel(dir, { createIfMissing: false });
    const sibling = { n: 0, types: {} };
    const inBlob = { n: 0, types: {} };
    for await (const [k, v] of db.iterator()) {
      const key = k.toString();
      const m = key.match(/^!([^!]+)!(.+)$/);
      if (!m) continue;
      let doc; try { doc = JSON.parse(v.toString()); } catch { continue; }
      if (m[1] === 'journal.pages') {
        sibling.n += 1;
        sibling.types[doc.type ?? '(none)'] = (sibling.types[doc.type ?? '(none)'] ?? 0) + 1;
      } else if (m[1] === 'adventures') {
        countPagesIn(doc, inBlob);
      }
    }
    await db.close();
    if (sibling.n || inBlob.n) {
      rows.push({ pack: `${manifest.id}.${p.name}`, sibling: sibling.n, inBlob: inBlob.n,
        types: Object.keys(sibling.n ? sibling.types : inBlob.types).length ? (sibling.n ? sibling.types : inBlob.types) : {} });
    }
  }
}
let tot = 0;
for (const r of rows) { tot += r.sibling + r.inBlob; console.log(r.pack.padEnd(28), 'sibling=', r.sibling, 'inAdventureBlob=', r.inBlob, JSON.stringify(r.types)); }
console.log('TOTAL pages =', tot);
