/**
 * ⚠⚠ 作废，别用 —— 用同目录的 `probe_patrons2.mjs`。
 * 本版只看 pack 的顶层文档（和其 .pages），而 deity 页藏在 Adventure 文档里
 * （`!adventures!emberAlphaOne000` → .journal[].pages[]），于是跑出 pagesWithField=0、
 * 「Hexblade 不存在」的**假阴性**。留档是为了记住这个空转形态：
 * 判据没写坏、库也真读了（531 docs），但**层级没走到底**，照样报 0。
 * 判 0 之前必须有一个「已知非零」的对照量（这里是 148 页 / 130 叶）。
 *
 * Round-20 probe ③: is `Hexblade` still a LIVE key?
 *
 * Walks every pack of a Foundry package, finds journal pages that carry
 * `system.warlocks` / `system.clerics` / `system.sorcerers` (the three fields
 * EmberDeityPageSheet._getTags() renders), and counts every value.
 *
 * Self-check (anti-空转): prints how many packs opened, how many docs iterated,
 * how many pages carried the field. A run that scanned 0 docs must be treated
 * as a failed probe, not as "Hexblade is gone".
 *
 * Usage: node probe_patrons.mjs --package <dir>
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
const where = new Map(); // value -> [pack/docName]
let packsOpened = 0, docsSeen = 0, pagesWithField = 0;

const packsDir = path.join(PACKAGE_DIR, 'packs');
for (const p of fs.readdirSync(packsDir)) {
  const dir = path.join(packsDir, p);
  if (!fs.statSync(dir).isDirectory()) continue;
  let db;
  try {
    db = new ClassicLevel(dir, { keyEncoding: 'utf8', valueEncoding: 'json' });
    await db.open();
  } catch (e) { console.error(`SKIP ${p}: ${e.message}`); continue; }
  packsOpened++;
  for await (const [key, doc] of db.iterator()) {
    docsSeen++;
    const pages = Array.isArray(doc?.pages) ? doc.pages : [];
    const candidates = [doc, ...pages];
    for (const c of candidates) {
      const sys = c?.system;
      if (!sys) continue;
      let hit = false;
      for (const f of FIELDS) {
        const v = sys[f];
        if (v == null) continue;
        hit = true;
        const arr = Array.isArray(v) ? v : [v];
        for (const item of arr) {
          if (typeof item !== 'string' || !item) continue;
          counts[f].set(item, (counts[f].get(item) || 0) + 1);
          if (!where.has(item)) where.set(item, []);
          if (where.get(item).length < 5) where.get(item).push(`${p}/${doc?.name}#${c?.name ?? '(root)'}`);
        }
      }
      if (hit) pagesWithField++;
    }
  }
  await db.close();
}

console.log(`SCANNED packs=${packsOpened} docs=${docsSeen} pagesWithField=${pagesWithField}`);
if (docsSeen === 0) { console.log('PROBE FAILED: scanned 0 docs'); process.exit(2); }
for (const f of FIELDS) {
  const m = [...counts[f].entries()].sort((a, b) => a[0].localeCompare(b[0]));
  const total = m.reduce((s, [, n]) => s + n, 0);
  console.log(`\n${f}: ${m.length} unique / ${total} leaves`);
  for (const [v, n] of m) console.log(`  ${JSON.stringify(v)} x${n}`);
}
console.log('\n--- Hexblade sightings ---');
console.log(where.has('Hexblade') ? where.get('Hexblade').join('\n') : '(none)');
