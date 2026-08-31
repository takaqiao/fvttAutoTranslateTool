#!/usr/bin/env node
/**
 * 字段普查器（判据「抽取器没抽的字段」的数据采集端）
 *
 * 直接把 LevelDB packs 里**每一个文档的每一个字符串叶子**枚举出来，
 * 与 `3-常用脚本/extract/mappings.mjs` 声明的字段求差。
 *
 * 与 extract_en.mjs 的关键区别：extract_en 只走 mapping 指到的路径，
 * 所以 mapping 里没有的字段它**永远看不见**——这正是本判据要找的盲区。
 * 这里反过来：先无条件枚举全部字段，再减去 mapping 覆盖集。
 *
 * 覆盖集的计算必须复刻 extract_en.mjs 各 converter 的语义
 * （nameCollection -> `p[].name`，crucibleActions -> `p[].name/description/
 * condition/effects[].name`，document -> 递归成子文档根，等等），
 * 否则会把已抽的字段误报成盲区。
 *
 * 用法:
 *   node census_pack_fields.mjs --package <dir> [--target crucible|ember] --out <json>
 */
import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';
import { effectiveMappings } from 'file:///C:/Users/Taka/Desktop/fvtt/Ember-Crucible%20Translation%20Project/3-%E5%B8%B8%E7%94%A8%E8%84%9A%E6%9C%AC/extract/mappings.mjs';

const FVTT_NODE_ANCHOR = 'C:/Users/Taka/Desktop/fvtt/package.json';
const { ClassicLevel } = createRequire(FVTT_NODE_ANCHOR)('classic-level');

const argv = process.argv.slice(2);
const arg = (n, d) => { const i = argv.indexOf(n); return i >= 0 ? argv[i + 1] : d; };

const PACKAGE_DIR = arg('--package');
const OUT = arg('--out');
if (!PACKAGE_DIR || !OUT) {
  console.error('Usage: node census_pack_fields.mjs --package <dir> --out <json> [--target crucible|ember]');
  process.exit(1);
}

const readJSON = (p) => JSON.parse(fs.readFileSync(p, 'utf8'));
const isStr = (v) => typeof v === 'string' && v.trim().length > 0;

/* ------------------- mapping -> covered relative paths ------------------- */
/**
 * @returns {{covered:Set<string>, childDocs:Array<{path:string,documentType:string}>}}
 */
function coverageOf(mapping) {
  const covered = new Set();
  const childDocs = [];
  if (!mapping) return { covered, childDocs };
  for (const [field, spec] of Object.entries(mapping)) {
    if (field.startsWith('_')) continue;
    if (typeof spec === 'string') { covered.add(spec); continue; }
    if (!spec || typeof spec !== 'object') continue;
    const p = spec.path ?? field;
    switch (spec.converter) {
      case 'name': covered.add(p); break;
      case 'nameCollection': covered.add(`${p}[].name`); break;
      case 'textCollection': covered.add(`${p}[].text`); break;
      case 'crucibleDescription':
        covered.add(p); covered.add(`${p}.public`); covered.add(`${p}.private`); break;
      case 'crucibleNested':
        for (const k of ['name', 'description', 'public', 'private', 'appearance']) covered.add(`${p}.${k}`);
        break;
      case 'crucibleActions':
        covered.add(`${p}[].name`); covered.add(`${p}[].description`);
        covered.add(`${p}[].condition`); covered.add(`${p}[].effects[].name`);
        break;
      case 'structured':
        for (const sp of Object.values(spec.mapping ?? {})) covered.add(`${p}[].${sp}`);
        if (spec.valuePath) covered.add(`${p}[].${spec.valuePath}`);
        break;
      case 'document':
        childDocs.push({ path: p, documentType: spec.documentType });
        break;
      default: covered.add(p);
    }
  }
  return { covered, childDocs };
}

function mappingFor(mappings, documentType, doc) {
  const sub = doc?.type;
  if (sub && mappings[`${documentType}.${sub}`]) return mappings[`${documentType}.${sub}`];
  return mappings[documentType] ?? null;
}

const toArray = (v) => {
  if (v === null || v === undefined) return [];
  if (Array.isArray(v)) return v;
  if (Array.isArray(v.contents)) return v.contents;
  if (typeof v === 'object') return Object.values(v);
  return [];
};

/* ------------------------------ census ------------------------------ */
// bucket: Map<"documentType|subtype|relpath", {docs:Set, n, chars, uniq:Map<value,count>}>
const census = new Map();
let docsSeen = 0;

function record(documentType, subtype, relpath, value, docId, packName, covered) {
  const key = `${documentType}|${subtype ?? ''}|${relpath}`;
  let e = census.get(key);
  if (!e) {
    e = { documentType, subtype: subtype ?? '', relpath, mapped: covered.has(relpath),
          n: 0, chars: 0, docs: new Set(), packs: new Set(), uniq: new Map() };
    census.set(key, e);
  }
  e.n += 1;
  e.chars += value.length;
  e.docs.add(docId);
  e.packs.add(packName);
  e.uniq.set(value, (e.uniq.get(value) ?? 0) + 1);
}

function walkDoc(doc, documentType, mappings, packName) {
  if (!doc || typeof doc !== 'object') return;
  docsSeen += 1;
  const mapping = mappingFor(mappings, documentType, doc);
  const { covered, childDocs } = coverageOf(mapping);
  const childByPath = new Map(childDocs.map((c) => [c.path, c.documentType]));
  const subtype = isStr(doc.type) ? doc.type : '';
  const docId = `${packName}:${doc._id ?? doc.name ?? '?'}`;

  // 16 位 Foundry id 当键的集合（`system.activities.<id>.name`、
  // `system.advancement.<id>.title`）必须折成 `<id>`，否则同一个逻辑字段会按
  // 每个实例各报一行 —— 实测把 `system.activities.*.name` 摊成了 20 多条，
  // 看着像 20 个不同盲区，其实是一个。
  const norm = (k) => (/^[A-Za-z0-9]{16}$/.test(k) ? '<id>' : k);

  const walk = (node, rel, depth) => {
    if (depth > 12) return;
    if (isStr(node)) { record(documentType, subtype, rel, node, docId, packName, covered); return; }
    if (Array.isArray(node)) { for (const it of node) walk(it, `${rel}[]`, depth + 1); return; }
    if (node && typeof node === 'object') {
      for (const [k0, v] of Object.entries(node)) {
        const k = norm(k0);
        const next = rel ? `${rel}.${k}` : k;
        if (childByPath.has(next)) {
          for (const child of toArray(v)) walkDoc(child, childByPath.get(next), mappings, packName);
          continue;
        }
        walk(v, next, depth + 1);
      }
    }
  };
  walk(doc, '', 0);
}

/* --------------------------- pack reading --------------------------- */
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

function attachEmbedded(buckets, parent, child, field) {
  const byParent = {};
  for (const { idPart, doc } of (buckets[child] || [])) (byParent[idPart.split('.')[0]] ||= []).push(doc);
  for (const { doc } of (buckets[parent] || [])) {
    const kids = byParent[doc._id];
    if (kids?.length) doc[field] = kids;
  }
}

const BUCKET_FOR = {
  Item: 'items', Actor: 'actors', JournalEntry: 'journal', Adventure: 'adventures',
  ActiveEffect: 'effects', Macro: 'macros', RollTable: 'tables', Scene: 'scenes',
  Playlist: 'playlists', Cards: 'cards',
};

async function main() {
  const manifestPath = ['system.json', 'module.json'].map((f) => path.join(PACKAGE_DIR, f)).find((p) => fs.existsSync(p));
  const manifest = readJSON(manifestPath);
  const target = arg('--target', manifest.id === 'ember' ? 'ember' : 'crucible');
  const mappings = effectiveMappings(target);

  // 灵敏度回测钩子：`--drop Scene.levels,Scene.tokens` 从覆盖集里摘掉指定的
  // <documentType>.<field>，用来复现「这一档从来没被抽过」的历史状态，
  // 检验判据能不能把它重新报出来。只影响本进程的内存副本，不落盘。
  for (const spec of (arg('--drop', '') || '').split(',').filter(Boolean)) {
    const i = spec.lastIndexOf('.');
    const [dt, field] = [spec.slice(0, i), spec.slice(i + 1)];
    if (mappings[dt]) { mappings[dt] = { ...mappings[dt] }; delete mappings[dt][field];
      console.error(`  [drop] ${dt}.${field}`); }
  }
  const packsDir = path.join(PACKAGE_DIR, 'packs');
  const packStats = [];

  for (const pack of manifest.packs ?? []) {
    const dir = path.join(packsDir, path.basename(pack.path ?? pack.name));
    if (!fs.existsSync(dir)) { packStats.push({ pack: pack.name, missing: true }); continue; }
    const buckets = await readPack(dir);
    attachEmbedded(buckets, 'actors', 'actors.items', 'items');
    attachEmbedded(buckets, 'actors', 'actors.effects', 'effects');
    attachEmbedded(buckets, 'items', 'items.effects', 'effects');
    attachEmbedded(buckets, 'journal', 'journal.pages', 'pages');
    attachEmbedded(buckets, 'journal', 'journal.categories', 'categories');
    attachEmbedded(buckets, 'tables', 'tables.results', 'results');
    attachEmbedded(buckets, 'scenes', 'scenes.regions', 'regions');

    const bucketName = BUCKET_FOR[pack.type];
    const roots = buckets[bucketName] || [];
    for (const { doc } of roots) walkDoc(doc, pack.type, mappings, pack.name);
    // folders 桶：extract_en 只取 name，其余字段属于盲区候选
    for (const { doc } of (buckets.folders || [])) {
      walkDoc(doc, '_FolderBucket', { _FolderBucket: { name: 'name' } }, pack.name);
    }
    packStats.push({ pack: pack.name, type: pack.type, roots: roots.length,
                     buckets: Object.fromEntries(Object.entries(buckets).map(([k, v]) => [k, v.length])) });
    process.stderr.write(`  ${pack.name}: ${roots.length} roots\n`);
  }

  const rows = [...census.values()].map((e) => {
    const top = [...e.uniq.entries()].sort((a, b) => b[1] - a[1]).slice(0, 6)
      .map(([v, c]) => ({ v: v.length > 300 ? `${v.slice(0, 300)}…` : v, c }));
    return {
      documentType: e.documentType, subtype: e.subtype, relpath: e.relpath, mapped: e.mapped,
      n: e.n, docs: e.docs.size, uniq: e.uniq.size, chars: e.chars,
      packs: [...e.packs].sort(), samples: top,
    };
  }).sort((a, b) => b.chars - a.chars);

  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  fs.writeFileSync(OUT, `${JSON.stringify({
    package: manifest.id, version: manifest.version, target,
    docsWalked: docsSeen, fieldPaths: rows.length, packStats, rows,
  }, null, 1)}\n`, 'utf8');
  console.error(`docs=${docsSeen} paths=${rows.length} -> ${OUT}`);
}

main().catch((e) => { console.error(e); process.exit(1); });
