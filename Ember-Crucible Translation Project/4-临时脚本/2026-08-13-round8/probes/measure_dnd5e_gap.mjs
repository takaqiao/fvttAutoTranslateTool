#!/usr/bin/env node
/**
 * 量化 dnd5e 侧「description.value / biography.value 没被抽」的实际损失。
 *
 * 光有字符数不够定严重度：Babele 的 `document` 转换器有 source-pack fallback
 * （`_stats.compendiumSource` -> 原包译文），所以**嵌在 actor 身上、来自别的包**
 * 的 dnd5e 物品理论上可以靠 dnd5e 中文模块兜底；而**顶层文档**和
 * **source 指回 ember 自己**的那些没有任何兜底，是死的。
 *
 * 用法: node measure_dnd5e_gap.mjs --package <emberdir> --out <json>
 */
import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';

const { ClassicLevel } = createRequire('C:/Users/Taka/Desktop/fvtt/package.json')('classic-level');
const argv = process.argv.slice(2);
const arg = (n, d) => { const i = argv.indexOf(n); return i >= 0 ? argv[i + 1] : d; };
const PACKAGE_DIR = arg('--package', 'C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember');
const OUT = arg('--out');

const isStr = (v) => typeof v === 'string' && v.trim().length > 0;
const TAG = /<[^>]+>/g;
const plain = (s) => s.replace(TAG, ' ').replace(/\s+/g, ' ').trim();

async function readPack(dir) {
  const db = new ClassicLevel(dir, { createIfMissing: false });
  const b = {};
  for await (const [k, v] of db.iterator()) {
    const m = k.toString().match(/^!([^!]+)!(.+)$/);
    if (!m) continue;
    let doc; try { doc = JSON.parse(v.toString()); } catch { continue; }
    (b[m[1]] ||= []).push({ idPart: m[2], doc });
  }
  await db.close();
  return b;
}
function attach(b, p, c, f) {
  const by = {};
  for (const { idPart, doc } of (b[c] || [])) (by[idPart.split('.')[0]] ||= []).push(doc);
  for (const { doc } of (b[p] || [])) { const k = by[doc._id]; if (k?.length) doc[f] = k; }
}

const tally = {};
function add(bucket, chars, docs = 1) {
  const t = (tally[bucket] ||= { docs: 0, chars: 0 });
  t.docs += docs; t.chars += chars;
}

const srcHosts = {};
function noteItem(item, position) {
  const v = item?.system?.description?.value;
  if (!isStr(v)) return;
  const chars = plain(v).length;
  const src = item?._stats?.compendiumSource ?? '';
  const host = src ? (src.split('.')[1] ?? '?') : '(none)';
  const key = `${position}|${host}`;
  (srcHosts[key] ||= { docs: 0, chars: 0, samples: [] });
  srcHosts[key].docs += 1; srcHosts[key].chars += chars;
  if (srcHosts[key].samples.length < 3) srcHosts[key].samples.push(item.name);
  add(position, chars);
}

const manifest = JSON.parse(fs.readFileSync(path.join(PACKAGE_DIR, 'module.json'), 'utf8'));
for (const pack of manifest.packs ?? []) {
  const dir = path.join(PACKAGE_DIR, 'packs', path.basename(pack.path ?? pack.name));
  if (!fs.existsSync(dir)) continue;
  const b = await readPack(dir);
  attach(b, 'actors', 'actors.items', 'items');
  attach(b, 'items', 'items.effects', 'effects');

  if (pack.type === 'Item') for (const { doc } of (b.items || [])) noteItem(doc, `${pack.name}:top-level-item`);
  if (pack.type === 'Actor') for (const { doc } of (b.actors || [])) for (const it of (doc.items || [])) noteItem(it, `${pack.name}:actor-embedded-item`);
  if (pack.type === 'Adventure') {
    for (const { doc } of (b.adventures || [])) {
      for (const it of (doc.items || [])) noteItem(it, `${pack.name}:top-level-item`);
      for (const a of (doc.actors || [])) {
        const bio = a?.system?.details?.biography?.value;
        if (isStr(bio)) add(`${pack.name}:actor-biography.value`, plain(bio).length);
        for (const it of (a.items || [])) noteItem(it, `${pack.name}:actor-embedded-item`);
      }
    }
  }
}

const rows = Object.entries(srcHosts).sort((a, b) => b[1].chars - a[1].chars)
  .map(([k, v]) => ({ where: k.split('|')[0], compendiumSourcePackage: k.split('|')[1], ...v }));
const res = { tally, bySourcePackage: rows };
if (OUT) fs.writeFileSync(OUT, `${JSON.stringify(res, null, 1)}\n`, 'utf8');
console.log(JSON.stringify(tally, null, 1));
console.log('\n--- description.value by where + compendiumSource package (plain chars) ---');
for (const r of rows) console.log(`${r.chars.toString().padStart(8)}  docs=${r.docs.toString().padEnd(5)} ${r.where}  src=${r.compendiumSourcePackage}  eg ${r.samples.join(' / ')}`);
