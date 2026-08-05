#!/usr/bin/env node
/**
 * Migrate shipped `compendium/cn/*.json` translation files onto the schema the
 * new declarative Babele mapping expects, WITHOUT losing any existing Chinese.
 *
 * What it does, per file:
 *   1. Delete the per-pack `mapping` block. Compendium-local mappings outrank
 *      `registerMapping()` layers, so leaving them would silently override the
 *      new global mapping and keep calling converters that no longer exist.
 *   2. `categories`: `{<_id>: {name: "译名"}}` -> `{"<EN name>": "译名"}`.
 *      Babele's built-in `nameCollection` looks entries up by NAME, so the
 *      id-keyed shape produced by the old extractor never matches. The EN name
 *      for each id is read straight out of the LevelDB pack.
 *   3. Ember legacy page keys: `soverview`/`sexposition`/`ssummary`/
 *      `scoverview`/`sgamemaster` -> `overview`/`exposition`/`summary`/
 *      `contentOverview`/`contentGamemaster`; drop the junk `path: {}` field
 *      (the mapping reserved word `path` leaked into 1447 page entries).
 *   4. Ember actors: `prototypeToken` -> `tokenName`; `biographypublic` /
 *      `biographyprivate` / `biographyappearance` -> `biography: {...}`.
 *   5. Ember items: `descriptionpublic`/`descriptionprivate` -> `description`;
 *      positional `actionname[]`/`actiondesc[]`/`actioneffectname[]` ->
 *      `actions: {<actionId>: {...}}`, using the English baseline's action key
 *      order to recover which id each position referred to.
 *
 * Usage:
 *   node migrate_cn_schema.mjs --repo <pluginRepo> --package <foundryPkgDir>
 *                              --target <crucible|ember> [--dry]
 */
import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';

const require_ = createRequire('C:/Users/Taka/Desktop/fvtt/package.json');
const { ClassicLevel } = require_('classic-level');

const argv = process.argv.slice(2);
const arg = (n, d) => { const i = argv.indexOf(n); return i >= 0 ? argv[i + 1] : d; };
const REPO = arg('--repo');
const PKG = arg('--package');
const TARGET = arg('--target', 'crucible');
const DRY = argv.includes('--dry');
if (!REPO || !PKG) {
  console.error('Usage: node migrate_cn_schema.mjs --repo <dir> --package <dir> --target <crucible|ember> [--dry]');
  process.exit(1);
}

const CN_DIR = path.join(REPO, 'compendium', 'cn');
const EN_DIR = path.join(REPO, 'compendium', 'en');
const readJSON = (p) => JSON.parse(fs.readFileSync(p, 'utf8'));
const isStr = (v) => typeof v === 'string' && v.trim().length > 0;

const stats = {
  mappingBlocksRemoved: 0,
  categoriesConverted: 0,
  categoriesUnmatched: 0,
  pageKeysRenamed: 0,
  pathJunkRemoved: 0,
  actorsMigrated: 0,
  itemsDescMigrated: 0,
  itemActionsMigrated: 0,
  itemActionsUnmatched: 0,
};
/** Items whose action translations could not be re-keyed; parked, not dropped. */
const unmatchedItems = [];

/* ---------- read journal category id -> EN name from the packs ---------- */
async function categoryNamesFromPacks(pkgDir) {
  const manifestPath = ['system.json', 'module.json']
    .map((f) => path.join(pkgDir, f)).find((p) => fs.existsSync(p));
  const manifest = readJSON(manifestPath);
  const byId = {};
  for (const pack of manifest.packs ?? []) {
    const dir = path.join(pkgDir, 'packs', path.basename(pack.path ?? pack.name));
    if (!fs.existsSync(dir)) continue;
    const db = new ClassicLevel(dir, { createIfMissing: false });
    for await (const [k, v] of db.iterator()) {
      const key = k.toString();
      if (!key.includes('!journal.categories!') && !key.startsWith('!journal.categories!')) continue;
      try {
        const doc = JSON.parse(v.toString());
        if (doc?._id && isStr(doc?.name)) byId[doc._id] = doc.name;
      } catch { /* ignore */ }
    }
    await db.close();
  }
  return byId;
}

/* ----------------------------- transforms ----------------------------- */

const PAGE_RENAME = {
  soverview: 'overview',
  sexposition: 'exposition',
  ssummary: 'summary',
  scoverview: 'contentOverview',
  coverview: 'contentOverview',
  sgamemaster: 'contentGamemaster',
  gamemaster: 'contentGamemaster',
};

function migratePage(page) {
  if (!page || typeof page !== 'object') return page;
  const out = {};
  for (const [k, v] of Object.entries(page)) {
    if (k === 'path') { stats.pathJunkRemoved += 1; continue; }
    const nk = PAGE_RENAME[k];
    if (nk) { out[nk] = v; stats.pageKeysRenamed += 1; continue; }
    out[k] = v;
  }
  return out;
}

function migrateCategories(cats, idToName) {
  if (!cats || typeof cats !== 'object') return cats;
  // Already in nameCollection form ({"EN": "译名"})? leave alone.
  if (Object.values(cats).every((v) => typeof v === 'string')) return cats;
  const out = {};
  for (const [k, v] of Object.entries(cats)) {
    const cn = typeof v === 'string' ? v : v?.name;
    if (!isStr(cn)) continue;
    const enName = idToName[k] ?? (typeof v === 'object' ? null : k);
    if (enName) { out[enName] = cn; stats.categoriesConverted += 1; }
    else { stats.categoriesUnmatched += 1; }
  }
  return Object.keys(out).length ? out : undefined;
}

function migrateActor(actor, enActor = null) {
  if (!actor || typeof actor !== 'object') return actor;
  const out = { ...actor };
  let touched = false;
  if (out.prototypeToken !== undefined) {
    out.tokenName ??= out.prototypeToken; delete out.prototypeToken; touched = true;
  }
  const bio = {};
  for (const [legacy, field] of [['biographypublic', 'public'],
                                 ['biographyprivate', 'private'],
                                 ['biographyappearance', 'appearance']]) {
    if (isStr(out[legacy])) { bio[field] = out[legacy]; delete out[legacy]; touched = true; }
  }
  if (Object.keys(bio).length) {
    out.biography = typeof out.biography === 'object' ? { ...bio, ...out.biography } : bio;
  }
  if (touched) stats.actorsMigrated += 1;
  if (out.items && typeof out.items === 'object') {
    out.items = Object.fromEntries(Object.entries(out.items).map(
      ([k, v]) => [k, migrateItem(v, enActor?.items?.[k])]));
  }
  return out;
}

function migrateItem(item, enItem) {
  if (!item || typeof item !== 'object') return item;
  const out = { ...item };

  // description
  const pub = out.descriptionpublic, priv = out.descriptionprivate;
  if (isStr(pub) || isStr(priv)) {
    delete out.descriptionpublic; delete out.descriptionprivate;
    if (isStr(pub) && isStr(priv)) out.description = { public: pub, private: priv };
    else if (isStr(pub)) out.description = { public: pub };
    else out.description = { private: priv };
    stats.itemsDescMigrated += 1;
  }

  // positional action arrays -> id-keyed map, using the EN baseline for ids
  const names = [].concat(out.actionname ?? []);
  const descs = [].concat(out.actiondesc ?? []);
  if (names.length || descs.length) {
    const legacy = { actionname: out.actionname, actiondesc: out.actiondesc,
                     actioneffectname: out.actioneffectname };
    delete out.actionname; delete out.actiondesc; delete out.actioneffectname;
    const ids = enItem?.actions ? Object.keys(enItem.actions) : [];
    if (!ids.length) {
      // No English counterpart to recover action ids from. Park the Chinese
      // under a clearly-marked key instead of dropping it, so a later pass can
      // rescue it by hand. Babele ignores unknown keys.
      stats.itemActionsUnmatched += 1;
      unmatchedItems.push(out.name ?? '(unnamed)');
      out._legacyActions = Object.fromEntries(
        Object.entries(legacy).filter(([, v]) => v !== undefined));
    } else {
      const acts = {};
      const n = Math.max(names.length, descs.length);
      for (let i = 0; i < n && i < ids.length; i += 1) {
        const e = {};
        if (isStr(names[i])) e.name = names[i];
        if (isStr(descs[i])) e.description = descs[i];
        if (Object.keys(e).length) acts[ids[i]] = e;
      }
      if (Object.keys(acts).length) { out.actions = acts; stats.itemActionsMigrated += 1; }
    }
  }
  return out;
}

/** Recursively walk a translation entry, applying every transform in scope. */
function migrateEntry(entry, enEntry, idToName) {
  if (!entry || typeof entry !== 'object') return entry;
  const out = { ...entry };

  if (out.categories) {
    const c = migrateCategories(out.categories, idToName);
    if (c) out.categories = c; else delete out.categories;
  }
  if (out.pages && typeof out.pages === 'object') {
    out.pages = Object.fromEntries(
      Object.entries(out.pages).map(([k, v]) => [k, migratePage(v)]));
  }
  for (const coll of ['journals']) {
    if (out[coll] && typeof out[coll] === 'object') {
      out[coll] = Object.fromEntries(Object.entries(out[coll]).map(
        ([k, v]) => [k, migrateEntry(v, enEntry?.[coll]?.[k], idToName)]));
    }
  }
  if (out.actors && typeof out.actors === 'object') {
    out.actors = Object.fromEntries(Object.entries(out.actors).map(
      ([k, v]) => [k, migrateActor(v, enEntry?.actors?.[k])]));
  }
  if (out.items && typeof out.items === 'object') {
    out.items = Object.fromEntries(Object.entries(out.items).map(
      ([k, v]) => [k, migrateItem(v, enEntry?.items?.[k])]));
  }
  // top-level item/actor shaped entries (non-adventure packs)
  return migrateItem(migrateActor(out, enEntry), enEntry);
}

/* -------------------------------- main -------------------------------- */
const idToName = await categoryNamesFromPacks(PKG);
console.log(`journal categories resolved from packs: ${Object.keys(idToName).length}`);

for (const fn of fs.readdirSync(CN_DIR).filter((f) => f.endsWith('.json'))) {
  const cnPath = path.join(CN_DIR, fn);
  let doc;
  try { doc = readJSON(cnPath); } catch (e) { console.error(`  ! ${fn}: ${e.message}`); continue; }

  let en = null;
  const enPath = path.join(EN_DIR, fn);
  if (fs.existsSync(enPath)) { try { en = readJSON(enPath); } catch { /* ignore */ } }

  if (doc.mapping !== undefined) { delete doc.mapping; stats.mappingBlocksRemoved += 1; }

  doc.entries = Object.fromEntries(Object.entries(doc.entries ?? {}).map(
    ([k, v]) => [k, migrateEntry(v, en?.entries?.[k], idToName)]));

  if (!DRY) fs.writeFileSync(cnPath, `${JSON.stringify(doc, null, 2)}\n`, 'utf8');
  console.log(`  ${DRY ? '[dry] ' : ''}${fn}`);
}

console.log(`\n--- ${TARGET} migration summary ---`);
for (const [k, v] of Object.entries(stats)) console.log(`  ${k.padEnd(26)} ${v}`);
if (unmatchedItems.length) {
  console.log(`\n  action translations parked under _legacyActions (no EN counterpart):`);
  for (const n of [...new Set(unmatchedItems)]) console.log(`    - ${n}`);
}
if (DRY) console.log('\n(dry run: nothing written)');
