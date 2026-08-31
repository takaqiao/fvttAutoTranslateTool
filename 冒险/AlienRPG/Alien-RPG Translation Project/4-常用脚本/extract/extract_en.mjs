#!/usr/bin/env node
/**
 * Extract a Foundry package's compendium packs (LevelDB) into Babele-shaped
 * English translation JSON files.
 *
 * It does not hard-code which fields to pull. It *interprets* the mapping
 * layers — the same data the runtime hands to `babele.registerMapping()` — so
 * the extracted keys are by construction the keys Babele will look for.
 *
 * Usage:
 *   node extract_en.mjs --package <dir> --out <dir>
 *                       [--target alienrpg|starterset|corerules]
 *                       [--mapping-source auto|project|babele]
 *                       [--mappings <file.mjs>] [--babele-defaults <file.js>]
 *                       [--pack <name>] [--strict-mappings]
 *
 *   --package         a Foundry package dir containing system.json or module.json
 *   --target          which mapping layer to apply (default: inferred from package id)
 *   --mapping-source  where the mapping comes from (default: auto, see below)
 *   --out             output dir for <packageId>.<packName>.json files
 *   --pack            optional: only this pack name (repeatable)
 *   --strict-mappings exit 3 instead of falling back to Babele raw defaults
 *
 * Requires `classic-level`, resolved from C:/Users/Taka/Desktop/fvtt.
 *
 * ---------------------------------------------------------------------------
 * MAPPING SOURCE — this file works with EITHER mapping module
 * ---------------------------------------------------------------------------
 * `--mapping-source auto` (the default):
 *   1. import `./mappings.mjs`;
 *   2. accept it ONLY if it positively declares it serves this target — i.e. it
 *      exports `ALIEN_TARGETS` / `SUPPORTED_TARGETS` (array containing the
 *      target) or `ALIEN_LAYER`. A bare `effectiveMappings(target)` is NOT
 *      enough: the Ember/Crucible module this project was ported from has that
 *      export and silently returns its *Crucible* layer for any unknown target,
 *      which would produce a plausible-looking but wrong English baseline.
 *   3. otherwise fall back to Babele 2.9.1's OWN `default-mappings.js`, read
 *      from the installed module, and say so loudly (banner + `mappingSource`
 *      in `_source.json`).
 *
 * The contract `mappings.mjs` must satisfy:
 *   export function effectiveMappings(target) -> object keyed by documentType
 *       (Babele defaults overlaid with this project's layer, merged PER FIELD)
 *   export const ALIEN_TARGETS = ['alienrpg', 'starterset', 'corerules']
 *   export const ALIEN_LAYER = { ... }        // project layer alone, no defaults
 *   export const EXTRACT_CONVERTERS = { ... } // OPTIONAL, see below
 *
 * `EXTRACT_CONVERTERS` is the extract-direction half of any custom functional
 * converter (Babele calls it `converter.extract`). Shape:
 *     { converterName: (value, ctx) => extracted | undefined }
 * with ctx = {spec, doc, documentType, mappings, extractDocument, toArray}.
 * Without it an unknown converter degrades to a plain path read AND is counted
 * into the `unknownConverters` warning block — never silently.
 *
 * ---------------------------------------------------------------------------
 * WHAT CHANGED vs the Ember/Crucible original (and why it had to)
 * ---------------------------------------------------------------------------
 * EC's `mappingFor()` understood only Babele's LEGACY `Type.subtype` keys, and
 * resolved them by WHOLE-BLOCK REPLACE. Both halves are wrong for this project:
 *   · The Alien mapping is `_variants`-driven (`_when: {all:[...]}`), because
 *     one translation key (`description`) has to point at a different source
 *     path per Item/Actor subtype. EC's extractor sees `_variants`, hits the
 *     `field.startsWith('_')` skip, and extracts NOTHING for those fields.
 *   · Babele does not replace, it takes a UNION with variant-wins-by-key
 *     (`mapping/mapping-block.js:126-140 #activeFields`), and legacy subtype
 *     keys are normalized INTO `_variants` and MERGED into the base definition
 *     (`mapping/document-mappings.js:293-344`), not swapped for it.
 * This file now reproduces the runtime's own resolution: `#normalized`,
 * `#matches` (all/any/equals/in/exists) and `#activeFields` are ported
 * verbatim in behaviour. Extracting a different key set than the runtime looks
 * up is the one defect class this whole two-consumers-one-definition design
 * exists to prevent.
 */
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { createRequire } from 'module';
import { pathToFileURL } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const FVTT_NODE_ANCHOR = 'C:/Users/Taka/Desktop/fvtt/package.json';
const BABELE_DEFAULTS_JS =
  'C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/babele/script/mapping/default-mappings.js';

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
const flag = (name) => argv.includes(name);
const argAll = (name) => argv.reduce((a, v, i) => (v === name ? [...a, argv[i + 1]] : a), []);

const PACKAGE_DIR = arg('--package');
const OUT_DIR = arg('--out');
const ONLY_PACKS = argAll('--pack');
const MAPPING_SOURCE = arg('--mapping-source', 'auto');
const MAPPINGS_FILE = arg('--mappings', path.join(__dirname, 'mappings.mjs'));
const BABELE_FILE = arg('--babele-defaults', BABELE_DEFAULTS_JS);
const STRICT = flag('--strict-mappings');

if (!PACKAGE_DIR || !OUT_DIR) {
  console.error('Usage: node extract_en.mjs --package <dir> --out <dir>'
    + ' [--target alienrpg|starterset|corerules] [--mapping-source auto|project|babele]'
    + ' [--pack <name>] [--strict-mappings]');
  process.exit(1);
}

const TARGETS = ['alienrpg', 'starterset', 'corerules'];
const TARGET_FOR_PACKAGE = {
  alienrpg: 'alienrpg',
  'alien-evolved-starterset': 'starterset',
  'alien-evolved-corerules': 'corerules',
};

/* --------------------------- utilities --------------------------- */
const readJSON = (p) => JSON.parse(fs.readFileSync(p, 'utf8'));
function writeJSON(file, obj) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, `${JSON.stringify(obj, null, 2)}\n`, 'utf8');
}
const getPath = (obj, p) =>
  p.split('.').reduce((a, k) => (a === null || a === undefined ? a : a[k]), obj);

const isNonEmptyString = (v) => typeof v === 'string' && v.trim().length > 0;

/* --------------------- key allocation (Babele-shaped) --------------------- */
/**
 * Key allocator, modelled on Babele's `ExportKeys` (identity/export-key-allocator.js)
 * plus its identity extractors (core/babele.js:85-92 registers `range`;
 * identity/identity-extractor-registry.js supplies `_id`/`id`/`name`/`sourceId`),
 * with ONE deliberate difference: when two documents want the same key and
 * their extracted content is identical, they COLLAPSE onto that one key instead
 * of the second falling back to `_id`.
 *
 * Why: Babele matches translations in the order `_id` -> `name` -> `sourceId`.
 * A single name-keyed entry translates both copies; two `_id`-keyed entries
 * would double the translation surface AND take precedence over the name key,
 * so a later name-only fix would silently stop applying.
 *
 * Genuinely different documents sharing a name still get an `_id` key, so
 * nothing is lost.
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

const DEFAULT_EXPORT_TOKENS = ['name', '_id', 'id'];

function identityCandidate(doc, token) {
  if (token === 'range') {
    const [start, end] = doc?.range ?? [];
    return Number.isInteger(start) && Number.isInteger(end) ? `${start}-${end}` : null;
  }
  if (token === 'sourceId') {
    // Babele parses a uuid here; the extract direction only needs a stable
    // string, and every Alien embedded source uuid is a world/dead-pack uuid
    // anyway (verify-mapping.json: 144 world + 36 dead-pack = 180/180).
    const raw = doc?.flags?.core?.sourceId || doc?._stats?.compendiumSource;
    if (!isNonEmptyString(raw)) return null;
    const tail = raw.split('.').pop();
    return isNonEmptyString(tail) ? tail : null;
  }
  const v = doc?.[token];
  return isNonEmptyString(v) ? v : null;
}

function makeKeyAllocator() {
  const used = new Map(); // key -> stored entry
  return (doc, identity, entry, fallbackPrefix = 'entry') => {
    const tokens = identity?.export ?? DEFAULT_EXPORT_TOKENS;
    const candidates = [];
    for (const t of tokens) {
      const c = identityCandidate(doc, t);
      if (c !== null && !candidates.includes(c)) candidates.push(c);
    }

    for (const c of candidates) {
      if (!used.has(c)) { used.set(c, entry); return { key: c, mergeInto: null }; }
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

/* ------------------- mapping normalization (Babele-faithful) ------------------- */

/**
 * `MappingBlock#normalized` — split `_variants` off the base mapping and drop
 * variants whose mapping body is empty (mapping/mapping-block.js:154-176).
 */
function normalizeDefinition(definition = {}) {
  const { _variants, ...base } = definition ?? {};
  const variants = Array.isArray(_variants)
    ? _variants
      .map((v) => { const { _when, ...mapping } = v ?? {}; return { when: _when ?? null, mapping }; })
      .filter((v) => Object.keys(v.mapping).length)
    : [];
  return { base, variants };
}

/**
 * `DocumentMappings#legacySubtypeDefinition` — a legacy `Type.subtype` key is
 * normalized into a variant gated on `{path:'type', equals: subtype}` and
 * MERGED into the base type's definition (document-mappings.js:321-344).
 * A subtype block's own `_variants` get their condition ANDed with the subtype
 * condition.
 */
function legacySubtypeVariants(subtype, definition = {}) {
  const cond = { path: 'type', equals: subtype };
  const { base, variants } = normalizeDefinition(definition);
  const out = [];
  if (Object.keys(base).length) out.push({ _when: cond, ...base });
  for (const v of variants) {
    out.push({ _when: v.when ? { all: [cond, v.when] } : cond, ...v.mapping });
  }
  return out;
}

/**
 * Fold every `Type.subtype` key of a mapping table into its base `Type` entry's
 * `_variants`, so downstream code only ever deals with base + variants.
 */
function foldLegacySubtypeKeys(mappings) {
  const out = {};
  const deferred = [];
  for (const [key, def] of Object.entries(mappings)) {
    const dot = key.indexOf('.');
    if (dot <= 0) { out[key] = def; continue; }
    deferred.push([key.slice(0, dot), key.slice(dot + 1), def]);
  }
  for (const [documentType, subtype, def] of deferred) {
    const current = out[documentType] ?? {};
    const { _variants = [], ...rest } = current;
    out[documentType] = { ...rest, _variants: [..._variants, ...legacySubtypeVariants(subtype, def)] };
  }
  return out;
}

/** `MappingBlock#matches` — mapping/mapping-block.js:176-206, verbatim behaviour. */
function whenMatches(condition, data) {
  if (!condition) return false;
  if (Array.isArray(condition.all)) return condition.all.every((c) => whenMatches(c, data));
  if (Array.isArray(condition.any)) return condition.any.some((c) => whenMatches(c, data));
  if (typeof condition.path !== 'string' || !condition.path.length) return false;

  const value = getPath(data, condition.path);
  const checks = [];
  if (Object.hasOwn(condition, 'equals')) checks.push(value === condition.equals);
  if (Array.isArray(condition.in)) checks.push(condition.in.includes(value));
  if (Object.hasOwn(condition, 'exists')) {
    checks.push((typeof value !== 'undefined') === Boolean(condition.exists));
  }
  return checks.length > 0 && checks.every(Boolean);
}

/**
 * `MappingBlock#activeFields` — base fields UNION matching-variant fields, with
 * later definitions winning by key (mapping/mapping-block.js:126-140).
 * Returns a plain `{field: spec}` object in effective order.
 */
function activeFields(normalized, doc) {
  const effective = new Map();
  const push = (obj) => {
    for (const [k, v] of Object.entries(obj)) {
      if (k.startsWith('_')) continue;
      effective.delete(k);
      effective.set(k, v);
    }
  };
  push(normalized.base);
  for (const variant of normalized.variants) {
    if (whenMatches(variant.when, doc)) push(variant.mapping);
  }
  return Object.fromEntries(effective);
}

/* --------------------- document extraction --------------------- */

/** Normalize a document-valued field to an array. */
function toArray(value) {
  if (value === null || value === undefined) return [];
  if (Array.isArray(value)) return value;
  if (Array.isArray(value.contents)) return value.contents;
  if (typeof value === 'object') return Object.values(value);
  return [];
}

const unknownConverters = new Map(); // `${documentType}.${field}:${converter}` -> count

/**
 * Extract one document into a Babele translation entry, driven by the mapping.
 *
 * @param {object} doc
 * @param {string} documentType
 * @param {object} ctx  {normalized, extractConverters}
 * @returns {object|null} null when the document yields no translatable text
 */
function extractDocument(doc, documentType, ctx) {
  const normalized = ctx.normalized[documentType];
  if (!normalized || !doc) return null;

  const fields = activeFields(normalized, doc);
  const out = {};

  for (const [field, spec] of Object.entries(fields)) {
    // --- plain path ---
    if (typeof spec === 'string') {
      const v = getPath(doc, spec);
      if (isNonEmptyString(v)) out[field] = v;
      continue;
    }
    if (!spec || typeof spec !== 'object') continue;

    const value = getPath(doc, spec.path ?? field);

    // --- project-supplied extract half of a custom functional converter ---
    const custom = ctx.extractConverters?.[spec.converter];
    if (typeof custom === 'function') {
      const v = custom(value, { spec, doc, documentType, ctx, extractDocument, toArray });
      if (v !== undefined && v !== null && !(typeof v === 'object' && !Object.keys(v).length)) {
        out[field] = v;
      }
      continue;
    }

    switch (spec.converter) {
      // `name` (Babele built-in) and the EC-era `crucibleTokenName` read the same
      // raw value in the EXTRACT direction; they differ only at translate time.
      case 'name':
      case 'crucibleTokenName':
      case 'alienTokenName': {
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
        // The key MUST be the text itself, not `_id`: Babele's textCollection is
        // fieldCollection("text") and looks up `translations[data.text]`
        // (converter/converters.js). Keying by id here would make every
        // en-vs-cn key comparison report live entries as dead keys.
        const map = {};
        for (const it of toArray(value)) {
          const key = it?.text;
          if (!isNonEmptyString(key)) continue;
          map[key] ??= it.text;
        }
        if (Object.keys(map).length) out[field] = map;
        break;
      }

      case 'structured': {
        // Only the array-container `mapping` form is implemented; the
        // `key`+`valuePath` form (ActiveEffect.changes) extracts to nothing,
        // which is the right outcome — AE change values are numbers/formulas.
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
        const childNormalized = ctx.normalized[childType] ?? null;
        const identity = childNormalized?.base?._identity ?? null;
        const keyOf = makeKeyAllocator();
        const map = {};
        for (const child of toArray(value)) {
          const entry = extractDocument(child, childType, ctx);
          if (!entry || !Object.keys(entry).length) continue;
          const { key, mergeInto } = keyOf(child, identity, entry, 'embedded');
          map[key] = mergeInto ?? entry;
        }
        if (Object.keys(map).length) out[field] = map;
        break;
      }

      /* ---- Ember/Crucible-era converters, kept so this file still runs
              against the EC mappings.mjs unchanged ---- */
      case 'crucibleDescription': {
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
        if (!value || typeof value !== 'object') break;
        const d = {};
        for (const k of ['name', 'description', 'public', 'private', 'appearance']) {
          if (isNonEmptyString(value[k])) d[k] = value[k];
        }
        if (Object.keys(d).length) out[field] = d;
        break;
      }
      case 'crucibleActions': {
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
      case 'emberEncounterTokenNames': {
        const map = {};
        for (const token of toArray(value)) {
          for (const actor of toArray(token?.actors)) {
            const n = actor?.tokenData?.name;
            if (isNonEmptyString(n)) map[n] ??= n;
          }
        }
        if (Object.keys(map).length) out[field] = map;
        break;
      }

      default: {
        // Unknown converter: plain path read, and COUNTED. Babele's own
        // `referencedDocumentField` legitimately lands here (extract direction
        // is byte-identical to a plain read of `spec.path`), but anything the
        // project invented and forgot to give an `EXTRACT_CONVERTERS` entry
        // lands here too, so this must never be silent.
        const tag = `${documentType}.${field}:${spec.converter ?? '(no converter)'}`;
        unknownConverters.set(tag, (unknownConverters.get(tag) ?? 0) + 1);
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

async function processPack(pack, packsDir, ctx) {
  const packDir = path.join(packsDir, path.basename(pack.path ?? pack.name));
  if (!fs.existsSync(packDir)) {
    // A warn is not enough: the manifest declared the pack, the disk has no
    // directory, so the output is short one pack file — and `_source.json`
    // would only record the successful ones, making the shortfall invisible.
    return { _skipped: 'missing pack directory', detail: packDir };
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

  // Pack-level Folder documents (`!folders!`). For an Adventure pack the folders
  // live INSIDE the Adventure document instead and are picked up by the
  // `Adventure.folders` nameCollection mapping — this stays empty there.
  const folders = {};
  for (const { doc } of (buckets.folders || [])) {
    if (isNonEmptyString(doc?.name)) folders[doc.name] ??= doc.name;
  }

  const bucketName = BUCKET_FOR[pack.type];
  if (!bucketName) return { _skipped: 'unsupported pack type', detail: pack.type };

  const keyOf = makeKeyAllocator();
  const identity = ctx.normalized[pack.type]?.base?._identity ?? null;
  const entries = {};
  let docCount = 0;
  let mergedCount = 0;
  for (const { doc } of (buckets[bucketName] || [])) {
    docCount += 1;
    const entry = extractDocument(doc, pack.type, ctx);
    if (!entry) continue;
    const { key, mergeInto } = keyOf(doc, identity, entry, 'entry');
    if (mergeInto) mergedCount += 1;
    entries[key] = mergeInto ?? entry;
  }

  return {
    label: pack.label,
    folders,
    entries,
    _meta: {
      documentType: pack.type,
      documents: docCount,
      mergedDuplicates: mergedCount,
      buckets: Object.fromEntries(Object.entries(buckets).map(([k, v]) => [k, v.length])),
    },
  };
}

/* ---------------------- mapping source resolution ---------------------- */

async function resolveMappings(target) {
  const wantProject = MAPPING_SOURCE === 'auto' || MAPPING_SOURCE === 'project';
  let fallbackReason = 'forced by --mapping-source babele';
  if (wantProject) {
    let mod = null;
    let err = null;
    try {
      mod = await import(pathToFileURL(MAPPINGS_FILE).href);
    } catch (e) { err = e; }

    if (mod) {
      const declared = mod.ALIEN_TARGETS ?? mod.SUPPORTED_TARGETS ?? null;
      const serves = (Array.isArray(declared) && declared.includes(target))
        || Boolean(mod.ALIEN_LAYER);
      if (serves && typeof mod.effectiveMappings === 'function') {
        return {
          mappings: mod.effectiveMappings(target),
          extractConverters: mod.EXTRACT_CONVERTERS ?? null,
          source: 'project',
          sourceDetail: MAPPINGS_FILE,
          note: null,
        };
      }
      if (MAPPING_SOURCE === 'project') {
        console.error(`--mapping-source project: ${MAPPINGS_FILE} does not serve target '${target}'.`);
        console.error(`  exports: ${Object.keys(mod).join(', ')}`);
        console.error("  needs: effectiveMappings(target) + ALIEN_TARGETS (or ALIEN_LAYER).");
        process.exit(3);
      }
      err = new Error(`does not declare target '${target}'`
        + ` (exports: ${Object.keys(mod).join(', ')})`);
    }
    if (STRICT) {
      console.error(`--strict-mappings: cannot use ${MAPPINGS_FILE}: ${err?.message}`);
      process.exit(3);
    }
    // fall through to Babele raw defaults
    fallbackReason = err?.message ?? 'unknown';
  }

  const babele = await import(pathToFileURL(BABELE_FILE).href);
  const defaults = babele.defaultMappings ?? babele.default ?? null;
  if (!defaults) {
    console.error(`No \`defaultMappings\` export in ${BABELE_FILE}`);
    process.exit(3);
  }
  return {
    mappings: defaults,
    extractConverters: null,
    source: 'babele-defaults',
    sourceDetail: BABELE_FILE,
    note: fallbackReason,
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
  const manifestKind = path.basename(manifestPath) === 'system.json' ? 'system' : 'module';
  const target = arg('--target', TARGET_FOR_PACKAGE[manifest.id] ?? null);
  if (!target || !TARGETS.includes(target)) {
    console.error(`--target must be one of ${TARGETS.join('|')}`
      + ` (could not infer from package id '${manifest.id}')`);
    process.exit(1);
  }

  const resolved = await resolveMappings(target);
  const normalized = {};
  for (const [dt, def] of Object.entries(foldLegacySubtypeKeys(resolved.mappings))) {
    normalized[dt] = normalizeDefinition(def);
  }
  const ctx = { normalized, extractConverters: resolved.extractConverters };

  console.log(`Package : ${manifest.id} v${manifest.version}  (${path.basename(manifestPath)}, ${manifestKind})`);
  console.log(`Target  : ${target}`);
  console.log(`Mapping : ${resolved.source}  <- ${resolved.sourceDetail}`);
  if (resolved.source === 'babele-defaults') {
    console.log('');
    console.log('  ############################################################');
    console.log('  #  BABELE RAW DEFAULTS — NOT the project mapping layer.    #');
    console.log(`  #  reason: ${resolved.note}`);
    console.log('  #  This output proves the LevelDB walk / key allocation /  #');
    console.log('  #  file shape only. Every alienrpg-specific field          #');
    console.log('  #  (system.notes, comment.value, special.value, ...) is    #');
    console.log('  #  ABSENT. Re-run once extract/mappings.mjs lands.         #');
    console.log('  ############################################################');
    console.log('');
  }
  console.log(`Output  : ${OUT_DIR}`);
  console.log(`Types   : ${Object.keys(normalized).length} document types`
    + ` (${Object.values(normalized).reduce((a, n) => a + n.variants.length, 0)} variants)\n`);

  const summary = [];
  const skipped = [];
  let declared = 0;
  for (const pack of manifest.packs ?? []) {
    if (ONLY_PACKS.length && !ONLY_PACKS.includes(pack.name)) continue;
    declared += 1;
    process.stdout.write(`- ${pack.name} (${pack.type})\n`);
    const result = await processPack(pack, path.join(PACKAGE_DIR, 'packs'), ctx);
    if (result?._skipped) {
      skipped.push({ pack: pack.name, type: pack.type, reason: result._skipped, detail: result.detail });
      console.warn(`    !! SKIPPED (${result._skipped}): ${result.detail}`);
      continue;
    }
    if (!result) continue;
    const outFile = path.join(OUT_DIR, `${manifest.id}.${pack.name}.json`);
    const { _meta, ...payload } = result;
    writeJSON(outFile, payload);
    summary.push({
      pack: pack.name,
      collection: `${manifest.id}.${pack.name}`,
      label: pack.label,
      file: path.basename(outFile),
      ..._meta,
      entries: Object.keys(result.entries).length,
      folders: Object.keys(result.folders).length,
    });
    console.log(`    -> ${Object.keys(result.entries).length} entries,`
      + ` ${Object.keys(result.folders).length} pack-level folders`);
  }

  writeJSON(path.join(OUT_DIR, '_source.json'), {
    packageId: manifest.id,
    packageVersion: manifest.version,
    packageType: manifestKind,
    packageManifest: path.basename(manifestPath),
    mappingTarget: target,
    mappingSource: resolved.source,
    mappingSourceDetail: resolved.sourceDetail,
    mappingSourceNote: resolved.note,
    extractedAt: new Date().toISOString(),
    extractedBy: 'Alien-RPG Translation Project / 4-常用脚本/extract/extract_en.mjs',
    declaredPacks: declared,
    extractedPacks: summary.length,
    skippedPacks: skipped,
    unknownConverters: Object.fromEntries([...unknownConverters.entries()].sort()),
    packs: summary,
    note: 'compendium/en/ 是英文基准，只用于跨版本算 drift 与 validate_translations，'
      + '进 git、不进发布 zip。此文件由 extract_en.mjs 生成，请勿手写。',
  });

  console.log(`\ncoverage: packs extracted ${summary.length}/${declared} declared in ${path.basename(manifestPath)}`);
  if (unknownConverters.size) {
    console.warn('\nunknown converters (fell back to a plain path read):');
    for (const [tag, n] of [...unknownConverters.entries()].sort()) console.warn(`    ${String(n).padStart(6)}  ${tag}`);
  }
  if (skipped.length) {
    console.warn(`\n!! ${path.basename(manifestPath)} declares ${declared} pack(s) but only ${summary.length}`
      + ` produced output; the following are recorded in _source.json.skippedPacks:`);
    for (const s of skipped) console.warn(`    - ${s.pack} (${s.type}): ${s.reason} -> ${s.detail}`);
    process.exitCode = 1;
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
