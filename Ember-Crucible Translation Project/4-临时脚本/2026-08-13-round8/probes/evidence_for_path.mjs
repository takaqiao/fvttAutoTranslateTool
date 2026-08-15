#!/usr/bin/env node
/**
 * 取证器：把 census 报出来的某条字段路径，落到具体文档上打印出来。
 *
 * 判据本身只给「路径 + 计数 + 样例」，写报告需要「哪个包、哪个文档、
 * 上级 name 是什么、值是什么」。这个脚本补的就是那一段。
 *
 * 用法:
 *   node evidence_for_path.mjs --package <dir> --path "system.description.value" \
 *        [--subtype feat] [--pack adventure] [--limit 5] [--chars 400]
 *
 * `--path` 用 census 的相对路径写法（数组元素写 `[]`），相对于**它所属的文档**，
 * 不是相对于 pack 根。
 */
import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';

const { ClassicLevel } = createRequire('C:/Users/Taka/Desktop/fvtt/package.json')('classic-level');
const argv = process.argv.slice(2);
const arg = (n, d) => { const i = argv.indexOf(n); return i >= 0 ? argv[i + 1] : d; };

const PACKAGE_DIR = arg('--package');
const WANT = arg('--path');
const WANT_SUB = arg('--subtype');
const ONLY_PACK = arg('--pack');
const LIMIT = Number(arg('--limit', 5));
const CHARS = Number(arg('--chars', 400));
if (!PACKAGE_DIR || !WANT) { console.error('need --package and --path'); process.exit(1); }

const isStr = (v) => typeof v === 'string' && v.trim().length > 0;
const out = [];

function walk(node, rel, doc, packName, trail) {
  if (out.length >= LIMIT) return;
  if (isStr(node)) {
    if (rel === WANT) out.push({ pack: packName, docName: doc.name, docId: doc._id, docType: doc.type, trail, value: node });
    return;
  }
  if (Array.isArray(node)) { node.forEach((it, i) => walk(it, `${rel}[]`, doc, packName, `${trail}[${i}]`)); return; }
  if (node && typeof node === 'object') {
    for (const [k, v] of Object.entries(node)) {
      walk(v, rel ? `${rel}.${k}` : k, doc, packName, trail);
    }
  }
}

/** 递归下潜到所有嵌套文档，每碰到一个带 name/_id 的对象就当作可能的文档根试一次。 */
function scanDoc(doc, packName, depth = 0) {
  if (out.length >= LIMIT || depth > 8) return;
  if (!WANT_SUB || doc.type === WANT_SUB) walk(doc, '', doc, packName, doc.name ?? doc._id ?? '?');
  for (const key of ['items', 'effects', 'pages', 'categories', 'results', 'regions',
                     'journal', 'scenes', 'actors', 'macros', 'tables', 'playlists', 'sounds', 'behaviors']) {
    const v = doc[key];
    if (!v) continue;
    for (const child of (Array.isArray(v) ? v : Object.values(v))) {
      if (child && typeof child === 'object') scanDoc(child, packName, depth + 1);
    }
  }
}

async function readPack(dir) {
  const db = new ClassicLevel(dir, { createIfMissing: false });
  const buckets = {};
  for await (const [k, v] of db.iterator()) {
    const m = k.toString().match(/^!([^!]+)!(.+)$/);
    if (!m) continue;
    let doc; try { doc = JSON.parse(v.toString()); } catch { continue; }
    (buckets[m[1]] ||= []).push({ idPart: m[2], doc });
  }
  await db.close();
  return buckets;
}
function attach(b, p, c, f) {
  const by = {};
  for (const { idPart, doc } of (b[c] || [])) (by[idPart.split('.')[0]] ||= []).push(doc);
  for (const { doc } of (b[p] || [])) { const k = by[doc._id]; if (k?.length) doc[f] = k; }
}

const BUCKET_FOR = { Item: 'items', Actor: 'actors', JournalEntry: 'journal', Adventure: 'adventures',
  ActiveEffect: 'effects', Macro: 'macros', RollTable: 'tables', Scene: 'scenes', Playlist: 'playlists' };

const manifestPath = ['system.json', 'module.json'].map((f) => path.join(PACKAGE_DIR, f)).find((p) => fs.existsSync(p));
const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
for (const pack of manifest.packs ?? []) {
  if (ONLY_PACK && pack.name !== ONLY_PACK) continue;
  const dir = path.join(PACKAGE_DIR, 'packs', path.basename(pack.path ?? pack.name));
  if (!fs.existsSync(dir)) continue;
  const b = await readPack(dir);
  attach(b, 'actors', 'actors.items', 'items'); attach(b, 'actors', 'actors.effects', 'effects');
  attach(b, 'items', 'items.effects', 'effects'); attach(b, 'journal', 'journal.pages', 'pages');
  attach(b, 'journal', 'journal.categories', 'categories'); attach(b, 'tables', 'tables.results', 'results');
  attach(b, 'scenes', 'scenes.regions', 'regions');
  for (const { doc } of (b[BUCKET_FOR[pack.type]] || [])) scanDoc(doc, pack.name);
  if (out.length >= LIMIT) break;
}

for (const o of out) {
  console.log(`--- [${o.pack}] ${o.docType ?? ''} "${o.docName}" (${o.docId})  @${o.trail}`);
  console.log(o.value.length > CHARS ? `${o.value.slice(0, CHARS)}…(+${o.value.length - CHARS})` : o.value);
}
console.log(`\n(${out.length} shown)`);
