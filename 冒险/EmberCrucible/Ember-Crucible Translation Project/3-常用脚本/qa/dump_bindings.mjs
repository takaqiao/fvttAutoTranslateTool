#!/usr/bin/env node
/**
 * Dump the *id bindings* that Babele translation files cannot express.
 *
 * `compendium/en|cn/*.json` are keyed by NAME and carry only translatable
 * fields, so a Scene Note's `entryId`/`pageId` and a RollTable result's
 * `documentCollection`/`documentId` exist **only** in the LevelDB packs.
 * Without them, "does this pin's label match the page it opens?" can only be
 * guessed from English string identity (what the two one-shot probes did).
 *
 * Output shape:
 * {
 *   "ids":   { "<_id>": {name, type, pack, kind} },          # every document id -> English name
 *   "notes": [ {pack, adventure, scene, noteId, text, entryId, pageId, iconTooltip} ],
 *   "results":[{pack, adventure, table, resultId, range, type, name, text,
 *               documentCollection, documentId} ]
 * }
 *
 * `--package` 可以给多份（2026-08-15 修）
 * ------------------------------------
 * 只导一个包会把「目标在别的包里」误报成「上游删了目标文档 / 悬空 id」。
 * 实测：只导 ember 时 `scan_name_binding` 的 199 条 UNCERTAIN 里有 187 条是这么来的
 * （114 条指向 `Compendium.dnd5e.*`、73 条指向 `Compendium.crucible.*`），
 * 而它们 187/187 全都能解析 —— 只是 id 表里没有别的包而已。
 * 所以至少要把 **ember 模块 + crucible 系统 + dnd5e 系统** 三份一起导：
 *
 *   node dump_bindings.mjs --package <…/modules/ember> \
 *                          --package <…/systems/crucible> \
 *                          --package <…/systems/dnd5e> --out bindings.json
 *
 * 每条记录都带 `pkg` 字段（包目录名），下游按 `pkg` 区分同 id 的不同包。
 *
 * Usage: node dump_bindings.mjs --package <foundry package dir> [--package …] --out <file.json>
 */
import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';

const FVTT_NODE_ANCHOR = 'C:/Users/Taka/Desktop/fvtt/package.json';
const { ClassicLevel } = createRequire(FVTT_NODE_ANCHOR)('classic-level');

const argv = process.argv.slice(2);
const arg = (n, d) => { const i = argv.indexOf(n); return i >= 0 ? argv[i + 1] : d; };
/** 收集同名开关的**全部**取值，`--package a --package b` → ['a','b']。 */
const args = (n) => argv.reduce((acc, v, i) => (v === n && argv[i + 1] ? acc.concat(argv[i + 1]) : acc), []);
const PACKAGE_DIRS = args('--package');
const OUT = arg('--out');

if (!PACKAGE_DIRS.length || !OUT) {
  console.error('usage: node dump_bindings.mjs --package <dir> [--package <dir> …] --out <file.json>');
  process.exit(2);
}

/** 当前正在导的包目录名。`record()`/`harvest()` 都往记录里写它。 */
let PACKAGE_ID = null;
const ids = {};
const notes = [];
const results = [];

/**
 * One id can occur in several packs (the two Ember adventures are near-copies
 * that reuse ids, and every embedded `compendiumSource` copy carries the source
 * document's id). Keep every occurrence — the caller resolves by pack.
 */
function record(id, name, type, pack, kind) {
  if (!id || typeof id !== 'string') return;
  const bucket = (ids[id] ??= []);
  if (bucket.some(o => o.pkg === PACKAGE_ID && o.pack === pack
                    && o.name === (name ?? null) && o.kind === kind)) return;
  bucket.push({ name: name ?? null, type: type ?? null, pkg: PACKAGE_ID, pack, kind });
}

/** Record every `_id` -> name we meet, and note the container chain. */
function scanIds(doc, pack, kind) {
  if (Array.isArray(doc)) { for (const d of doc) scanIds(d, pack, kind); return; }
  if (!doc || typeof doc !== 'object') return;
  if (typeof doc._id === 'string') record(doc._id, doc.name ?? null, doc.type ?? null, pack, kind);
  for (const [k, v] of Object.entries(doc)) {
    if (v && typeof v === 'object') scanIds(v, pack, kind === '?' ? k : `${kind}.${k}`);
  }
}

/** Pull scene notes + table results, remembering the adventure they sit in. */
function harvest(doc, pack, adventure) {
  for (const sc of doc.scenes ?? []) {
    for (const n of sc.notes ?? []) {
      notes.push({
        pkg: PACKAGE_ID,
        pack, adventure, scene: sc.name ?? null, sceneId: sc._id ?? null,
        noteId: n._id ?? null,
        text: typeof n.text === 'string' ? n.text : null,
        entryId: n.entryId ?? null, pageId: n.pageId ?? null,
      });
    }
  }
  for (const t of doc.tables ?? []) {
    for (const r of t.results ?? []) {
      results.push({
        pkg: PACKAGE_ID,
        pack, adventure, table: t.name ?? null, tableId: t._id ?? null,
        resultId: r._id ?? null,
        range: Array.isArray(r.range) ? r.range.join('-') : null,
        type: r.type ?? null,
        name: typeof r.name === 'string' ? r.name : null,
        text: typeof r.text === 'string' ? r.text : null,
        documentCollection: r.documentCollection ?? null,
        documentId: r.documentId ?? null,
        documentUuid: r.documentUuid ?? null,
      });
    }
  }
}

const packages = [];
for (const packageDir of PACKAGE_DIRS) {
  PACKAGE_ID = path.basename(packageDir.replace(/[\\/]+$/, ''));
  const packsDir = path.join(packageDir, 'packs');
  if (!fs.existsSync(packsDir)) {
    console.error(`⚠ ${PACKAGE_ID}: 没有 packs/ 目录，跳过（${packsDir}）`);
    continue;
  }
  const before = { ids: Object.keys(ids).length, notes: notes.length, results: results.length };
  for (const name of fs.readdirSync(packsDir)) {
    const dir = path.join(packsDir, name);
    if (!fs.statSync(dir).isDirectory()) continue;
    // 只有 LevelDB 目录才有 CURRENT；.json 源目录 / 散装资源目录直接跳过。
    if (!fs.existsSync(path.join(dir, 'CURRENT'))) continue;
    const db = new ClassicLevel(dir, { valueEncoding: 'json' });
    await db.open();
    for await (const [key, value] of db.iterator()) {
      const kind = String(key).split('!')[1] ?? '?';
      scanIds(value, name, kind);
      // Adventure documents hold whole sub-collections; standalone scenes/tables
      // live in their own pack.
      if (Array.isArray(value.scenes) || Array.isArray(value.tables)) {
        harvest(value, name, value.name ?? null);
      }
      if (kind === 'scenes' && Array.isArray(value.notes)) {
        harvest({ scenes: [value] }, name, null);
      }
      if (kind === 'tables' && Array.isArray(value.results)) {
        harvest({ tables: [value] }, name, null);
      }
    }
    await db.close();
  }
  packages.push({
    pkg: PACKAGE_ID, dir: packageDir,
    ids: Object.keys(ids).length - before.ids,
    notes: notes.length - before.notes,
    results: results.length - before.results,
  });
  const p = packages[packages.length - 1];
  console.log(`  ${p.pkg}: +ids=${p.ids} +notes=${p.notes} +results=${p.results}`);
}

// `packages` 让下游知道**哪些包的 id 表在场** —— 解析不出来的目标究竟是「上游删了」
// 还是「这个包压根没导」，全靠它区分。
fs.writeFileSync(OUT, JSON.stringify({ packages, ids, notes, results }, null, 1), 'utf8');
console.log(`ids=${Object.keys(ids).length} notes=${notes.length} results=${results.length} -> ${OUT}`);
