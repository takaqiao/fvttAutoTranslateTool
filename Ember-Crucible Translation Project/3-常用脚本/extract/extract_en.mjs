#!/usr/bin/env node
/**
 * Extract a Foundry package's compendium packs (LevelDB) into Babele-shaped
 * English translation JSON files.
 *
 * Unlike the older per-project extractors, this one does not hard-code which
 * fields to pull. It *interprets* the mapping layers in `mappings.mjs` — the
 * same data the runtime hands to `babele.registerMapping()` — so the extracted
 * keys are by construction the keys Babele will look for.
 *
 * Usage:
 *   node extract_en.mjs --package <dir> --target <crucible|ember> --out <dir>
 *
 *   --package  a Foundry package dir containing system.json or module.json
 *   --target   which mapping layer to apply (default: inferred from package id)
 *   --out      output dir for <packageId>.<packName>.json files
 *   --pack     optional: only this pack name (repeatable)
 *
 * Requires `classic-level`, resolved from C:/Users/Taka/Desktop/fvtt.
 */
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { createRequire } from 'module';
import { effectiveMappings } from './mappings.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const FVTT_NODE_ANCHOR = 'C:/Users/Taka/Desktop/fvtt/package.json';
let ClassicLevel;
try {
  ({ ClassicLevel } = createRequire(FVTT_NODE_ANCHOR)('classic-level'));
} catch (e) {
  console.error(`Failed to load classic-level via ${FVTT_NODE_ANCHOR}: ${e.message}`);
  console.error('Fix: cd C:/Users/Taka/Desktop/fvtt && npm i classic-level');
  process.exit(1);
}

/* ------------------------------ CLI ------------------------------ */
const argv = process.argv.slice(2);
const arg = (name, def) => {
  const i = argv.indexOf(name);
  return i >= 0 ? argv[i + 1] : def;
};
const argAll = (name) => argv.reduce((a, v, i) => (v === name ? [...a, argv[i + 1]] : a), []);

const PACKAGE_DIR = arg('--package');
const OUT_DIR = arg('--out');
const ONLY_PACKS = argAll('--pack');
if (!PACKAGE_DIR || !OUT_DIR) {
  console.error('Usage: node extract_en.mjs --package <dir> --out <dir> [--target crucible|ember] [--pack <name>]');
  process.exit(1);
}

/* --------------------------- utilities --------------------------- */
const readJSON = (p) => JSON.parse(fs.readFileSync(p, 'utf8'));
function writeJSON(file, obj) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, `${JSON.stringify(obj, null, 2)}\n`, 'utf8');
}
const getPath = (obj, p) =>
  p.split('.').reduce((a, k) => (a === null || a === undefined ? a : a[k]), obj);

const isNonEmptyString = (v) => typeof v === 'string' && v.trim().length > 0;

/**
 * Key allocator, modelled on Babele's `ExportKeys` but with one deliberate
 * difference: when two documents want the same key and their extracted content
 * is identical, they COLLAPSE onto that one key instead of the second falling
 * back to `_id`.
 *
 * Why: Babele matches translations in the order `_id` -> `name` -> `sourceId`.
 * `crucible.pregens` ships 16 actors under 8 distinct names (each pregen twice).
 * A single name-keyed entry translates both copies; two `_id`-keyed entries
 * would double the translation surface AND take precedence over the name key,
 * so a later name-only fix would silently stop applying.
 *
 * Genuinely different documents sharing a name still get an `_id` key, so
 * nothing is lost.
 *
 * Returns `{key, mergeInto}` — when `mergeInto` is set the caller must deep-merge
 * the new entry into the entry already stored under `key` rather than writing a
 * second one.
 */

/** Union two extracted entries. Returns null if a string leaf genuinely differs. */
function mergeEntries(a, b) {
  if (a === undefined) return b;
  if (b === undefined) return a;
  if (typeof a === 'string' || typeof b === 'string') {
    return a === b ? a : null;
  }
  if (Array.isArray(a) || Array.isArray(b)) {
    return JSON.stringify(a) === JSON.stringify(b) ? a : null;
  }
  const out = { ...a };
  for (const [k, v] of Object.entries(b)) {
    if (!(k in out)) { out[k] = v; continue; }
    const m = mergeEntries(out[k], v);
    if (m === null) return null;
    out[k] = m;
  }
  return out;
}

function makeKeyAllocator() {
  const used = new Map(); // key -> stored entry
  return (doc, identity, entry, fallbackPrefix = 'entry') => {
    const tokens = identity?.export ?? ['name', '_id', 'id'];
    const candidates = [];
    for (const t of tokens) {
      if (t === 'range') {
        const r = doc?.range;
        if (Array.isArray(r) && Number.isInteger(r[0]) && Number.isInteger(r[1])) {
          candidates.push(`${r[0]}-${r[1]}`);
        }
        continue;
      }
      const v = doc?.[t];
      if (isNonEmptyString(v)) candidates.push(v);
    }

    for (const c of candidates) {
      if (!used.has(c)) { used.set(c, entry); return { key: c, mergeInto: null }; }
      // Same name, compatible content -> one translation should cover both
      // documents. Babele matches `_id` before `name`, so emitting a second
      // `_id`-keyed entry would both duplicate the work and take precedence
      // over the shared one.
      const merged = mergeEntries(used.get(c), entry);
      if (merged !== null) { used.set(c, merged); return { key: c, mergeInto: merged }; }
    }

    let i = used.size;
    let fb = `${fallbackPrefix}-${i}`;
    while (used.has(fb)) { i += 1; fb = `${fallbackPrefix}-${i}`; }
    used.set(fb, entry);
    return { key: fb, mergeInto: null };
  };
}

/* --------------------- mapping interpretation --------------------- */

/**
 * Resolve the mapping for a document type, honouring subtype keys
 * (`JournalEntryPage.ember.location`) exactly as Babele 2.9.1 does.
 */
function mappingFor(mappings, documentType, doc) {
  const subtype = doc?.type;
  if (subtype && mappings[`${documentType}.${subtype}`]) {
    return mappings[`${documentType}.${subtype}`];
  }
  return mappings[documentType] ?? null;
}

/** Normalize a document-valued field to an array. */
function toArray(value) {
  if (value === null || value === undefined) return [];
  if (Array.isArray(value)) return value;
  if (Array.isArray(value.contents)) return value.contents;
  if (typeof value === 'object') return Object.values(value);
  return [];
}

/**
 * Extract one document into a Babele translation entry, driven by the mapping.
 *
 * @returns {object|null} null when the document yields no translatable text
 */
function extractDocument(doc, documentType, mappings) {
  const mapping = mappingFor(mappings, documentType, doc);
  if (!mapping || !doc) return null;

  const out = {};

  for (const [field, spec] of Object.entries(mapping)) {
    if (field.startsWith('_')) continue; // _identity etc.

    // --- plain path ---
    if (typeof spec === 'string') {
      const v = getPath(doc, spec);
      if (isNonEmptyString(v)) out[field] = v;
      continue;
    }
    if (!spec || typeof spec !== 'object') continue;

    const value = getPath(doc, spec.path ?? field);

    switch (spec.converter) {
      case 'name': {
        if (isNonEmptyString(value)) out[field] = value;
        break;
      }

      case 'nameCollection': {
        const map = {};
        for (const it of toArray(value)) {
          if (isNonEmptyString(it?.name)) map[it.name] ??= it.name;
        }
        if (Object.keys(map).length) out[field] = map;
        break;
      }

      case 'textCollection': {
        const map = {};
        for (const it of toArray(value)) {
          // 键**必须**是 text 本身，不能用 _id。Babele 的 textCollection 就是
          // fieldCollection("text")，运行时查的是 `translations[data.text]`
          // （converter/converters.js: fieldCollection）。这里一度写成
          // `it?._id ?? it?.text`，于是英文基线按 ID 建键、中文包按文本建键 ——
          // 中文其实一直在正常生效，但任何拿两边比键的扫描都会把 284 条活的
          // scene note 判成「死键」，差点被 prune_dead 全删。
          const key = it?.text;
          if (!isNonEmptyString(key)) continue;
          map[key] ??= it.text;
        }
        if (Object.keys(map).length) out[field] = map;
        break;
      }

      case 'crucibleDescription': {
        // Polymorphic: plain string on most item types, {public, private} on
        // equipment-like ones. Emitted in whichever shape the source uses so
        // existing crucible-cn translations keep matching.
        if (isNonEmptyString(value)) { out[field] = value; break; }
        if (value && typeof value === 'object') {
          const d = {};
          if (isNonEmptyString(value.public)) d.public = value.public;
          if (isNonEmptyString(value.private)) d.private = value.private;
          if (Object.keys(d).length) out[field] = d;
        }
        break;
      }

      case 'crucibleNested': {
        // {name, description} or {public, private, appearance} sub-objects.
        if (!value || typeof value !== 'object') break;
        const d = {};
        for (const k of ['name', 'description', 'public', 'private', 'appearance']) {
          if (isNonEmptyString(value[k])) d[k] = value[k];
        }
        if (Object.keys(d).length) out[field] = d;
        break;
      }

      case 'crucibleActions': {
        // Keyed by action id; each action may carry an `effects` array whose
        // entries only need `name`.
        const map = {};
        for (const a of toArray(value)) {
          if (!isNonEmptyString(a?.id)) continue;
          const e = {};
          if (isNonEmptyString(a.name)) e.name = a.name;
          if (isNonEmptyString(a.description)) e.description = a.description;
          if (isNonEmptyString(a.condition)) e.condition = a.condition;
          if (Array.isArray(a.effects) && a.effects.length) {
            const eff = a.effects.map((x) => (isNonEmptyString(x?.name) ? { name: x.name } : {}));
            if (eff.some((x) => x.name)) e.effects = eff;
          }
          if (Object.keys(e).length) map[a.id] ??= e;
        }
        if (Object.keys(map).length) out[field] = map;
        break;
      }

      case 'structured': {
        // Only the array-container form is used by this project.
        const map = {};
        for (const it of toArray(value)) {
          const key = it?.[spec.key ?? 'id'];
          if (!isNonEmptyString(key)) continue;
          const sub = {};
          for (const [sf, sp] of Object.entries(spec.mapping ?? {})) {
            const sv = getPath(it, sp);
            if (isNonEmptyString(sv)) sub[sf] = sv;
          }
          if (Object.keys(sub).length) map[key] ??= sub;
        }
        if (Object.keys(map).length) out[field] = map;
        break;
      }

      case 'document': {
        const childType = spec.documentType;
        const childMapping = mappings[childType] ?? null;
        const identity = childMapping?._identity ?? null;
        const keyOf = makeKeyAllocator();
        const map = {};
        for (const child of toArray(value)) {
          const entry = extractDocument(child, childType, mappings);
          if (!entry || !Object.keys(entry).length) continue;
          const { key, mergeInto } = keyOf(child, identity, entry, 'embedded');
          map[key] = mergeInto ?? entry;
        }
        if (Object.keys(map).length) out[field] = map;
        break;
      }

      default: {
        // Unknown converter: fall back to a plain path read.
        if (isNonEmptyString(value)) out[field] = value;
      }
    }
  }

  return Object.keys(out).length ? out : null;
}

/* --------------------------- pack reading -------------------------- */

/** Read a LevelDB pack into buckets keyed by the `!prefix!` segment. */
async function readPack(packDir) {
  const db = new ClassicLevel(packDir, { createIfMissing: false });
  const buckets = {};
  for await (const [k, v] of db.iterator()) {
    const m = k.toString().match(/^!([^!]+)!(.+)$/);
    if (!m) continue;
    let doc;
    try { doc = JSON.parse(v.toString()); } catch { continue; }
    (buckets[m[1]] ||= []).push({ idPart: m[2], doc });
  }
  await db.close();
  return buckets;
}

/** Foundry's LevelDB layout stores embedded docs in sibling buckets. */
function attachEmbedded(buckets, parentBucket, childBucket, field) {
  const byParent = {};
  for (const { idPart, doc } of (buckets[childBucket] || [])) {
    (byParent[idPart.split('.')[0]] ||= []).push(doc);
  }
  for (const { doc } of (buckets[parentBucket] || [])) {
    const kids = byParent[doc._id];
    if (kids?.length) doc[field] = kids;
  }
}

const BUCKET_FOR = {
  Item: 'items',
  Actor: 'actors',
  JournalEntry: 'journal',
  Adventure: 'adventures',
  ActiveEffect: 'effects',
  Macro: 'macros',
  RollTable: 'tables',
  Scene: 'scenes',
  Playlist: 'playlists',
  Cards: 'cards',
};

async function processPack(pack, packsDir, mappings) {
  const packDir = path.join(packsDir, path.basename(pack.path ?? pack.name));
  if (!fs.existsSync(packDir)) {
    console.warn(`  skip (missing dir): ${packDir}`);
    return null;
  }
  const buckets = await readPack(packDir);

  // Re-attach embedded documents that Foundry stores in sibling buckets.
  attachEmbedded(buckets, 'actors', 'actors.items', 'items');
  attachEmbedded(buckets, 'actors', 'actors.effects', 'effects');
  attachEmbedded(buckets, 'items', 'items.effects', 'effects');
  attachEmbedded(buckets, 'journal', 'journal.pages', 'pages');
  attachEmbedded(buckets, 'journal', 'journal.categories', 'categories');
  attachEmbedded(buckets, 'tables', 'tables.results', 'results');
  attachEmbedded(buckets, 'scenes', 'scenes.regions', 'regions');

  const folders = {};
  for (const { doc } of (buckets.folders || [])) {
    if (isNonEmptyString(doc?.name)) folders[doc.name] ??= doc.name;
  }

  const bucketName = BUCKET_FOR[pack.type];
  if (!bucketName) {
    console.warn(`  unsupported pack type: ${pack.type} (${pack.name})`);
    return null;
  }

  const keyOf = makeKeyAllocator();
  const identity = mappings[pack.type]?._identity ?? null;
  const entries = {};
  let docCount = 0;
  let mergedCount = 0;
  for (const { doc } of (buckets[bucketName] || [])) {
    docCount += 1;
    const entry = extractDocument(doc, pack.type, mappings);
    if (!entry) continue;
    const { key, mergeInto } = keyOf(doc, identity, entry, 'entry');
    if (mergeInto) mergedCount += 1;
    entries[key] = mergeInto ?? entry;
  }

  return {
    label: pack.label,
    folders,
    entries,
    _meta: { documentType: pack.type, documents: docCount, mergedDuplicates: mergedCount },
  };
}

/* ------------------------------- main ------------------------------ */
async function main() {
  const manifestPath = ['system.json', 'module.json']
    .map((f) => path.join(PACKAGE_DIR, f))
    .find((p) => fs.existsSync(p));
  if (!manifestPath) {
    console.error(`No system.json or module.json in ${PACKAGE_DIR}`);
    process.exit(1);
  }
  const manifest = readJSON(manifestPath);
  const target = arg('--target', manifest.id === 'ember' ? 'ember' : 'crucible');
  const mappings = effectiveMappings(target);
  const packsDir = path.join(PACKAGE_DIR, 'packs');

  console.log(`Package : ${manifest.id} v${manifest.version}  (${path.basename(manifestPath)})`);
  console.log(`Mapping : ${target}`);
  console.log(`Output  : ${OUT_DIR}\n`);

  const summary = [];
  for (const pack of manifest.packs ?? []) {
    if (ONLY_PACKS.length && !ONLY_PACKS.includes(pack.name)) continue;
    process.stdout.write(`- ${pack.name} (${pack.type})\n`);
    const result = await processPack(pack, packsDir, mappings);
    if (!result) continue;
    const outFile = path.join(OUT_DIR, `${manifest.id}.${pack.name}.json`);
    const { _meta, ...payload } = result;
    writeJSON(outFile, payload);
    summary.push({ pack: pack.name, ...(_meta), entries: Object.keys(result.entries).length, folders: Object.keys(result.folders).length });
    console.log(`    -> ${Object.keys(result.entries).length} entries, ${Object.keys(result.folders).length} folders`);
  }

  writeJSON(path.join(OUT_DIR, '_source.json'), {
    packageId: manifest.id,
    packageVersion: manifest.version,
    packageType: manifest.type ?? 'system',
    mappingTarget: target,
    extractedAt: new Date().toISOString(),
    extractedBy: 'Ember-Crucible Translation Project / 3-常用脚本/extract/extract_en.mjs',
    packs: summary,
  });

  console.log(`\nDone. Wrote ${summary.length} pack files + _source.json`);
}

main().catch((e) => { console.error(e); process.exit(1); });
