#!/usr/bin/env node
/**
 * Extract Crucible system compendium packs (LevelDB) into Babele-compatible
 * English translation JSON files, matching the layout of compendium/en/.
 *
 * Usage:
 *   node scripts/extract_en_compendium.mjs [--system <path>] [--out <path>]
 *
 * Defaults:
 *   --system : %LOCALAPPDATA%/FoundryVTT/Data/systems/crucible
 *   --out    : <project>/compendium/en
 *
 * Requires `classic-level` (resolved from the user's existing fvtt toolchain
 * at C:/Users/Taka/Desktop/fvtt/node_modules, so no local install needed).
 */
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { createRequire } from 'module';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '..');

// ---- classic-level resolution ----
const FVTT_NODE_ANCHOR = 'C:/Users/Taka/Desktop/fvtt/package.json';
let ClassicLevel;
try {
  const fvttRequire = createRequire(FVTT_NODE_ANCHOR);
  ({ ClassicLevel } = fvttRequire('classic-level'));
} catch (e) {
  console.error(`Failed to load classic-level via ${FVTT_NODE_ANCHOR}`);
  console.error(e.message);
  console.error('\nFix: install classic-level somewhere reachable, e.g.:');
  console.error('  cd C:/Users/Taka/Desktop/fvtt && npm i classic-level');
  process.exit(1);
}

// ---- CLI args ----
const args = process.argv.slice(2);
const getArg = (name, def) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : def;
};
const DEFAULT_SYSTEM = 'C:/Users/Taka/Desktop/fvtt/crucible';
const SYSTEM_DIR = getArg('--system', DEFAULT_SYSTEM);
const OUT_DIR = getArg('--out', path.join(projectRoot, 'compendium/en_new'));

// ---- Helpers ----
const readJSON = p => JSON.parse(fs.readFileSync(p, 'utf8'));
function writeJSON(file, obj) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, JSON.stringify(obj, null, 2) + '\n', 'utf8');
}

async function readPack(packDir) {
  const db = new ClassicLevel(packDir, { createIfMissing: false });
  const buckets = {};
  for await (const [k, v] of db.iterator()) {
    const key = k.toString();
    const m = key.match(/^!([^!]+)!(.+)$/);
    if (!m) continue;
    const [, prefix, idPart] = m;
    let doc;
    try { doc = JSON.parse(v.toString()); } catch { continue; }
    (buckets[prefix] ||= []).push({ idPart, doc });
  }
  await db.close();
  return buckets;
}

// ---- Entry builders (mirror babele-register.js converters) ----
function buildActionEntry(a) {
  const e = {};
  if (a.name) e.name = a.name;
  if (a.description) e.description = a.description;
  if (a.condition) e.condition = a.condition;
  if (Array.isArray(a.effects) && a.effects.length) {
    e.effects = a.effects.map(ef => (ef?.name ? { name: ef.name } : {}));
  }
  return e;
}

function buildActionsMap(actions) {
  if (!Array.isArray(actions) || !actions.length) return null;
  const out = {};
  for (const a of actions) {
    const id = a?.id;
    if (!id) continue;
    out[id] = buildActionEntry(a);
  }
  return Object.keys(out).length ? out : null;
}

function buildItemEntry(item) {
  const e = { name: item.name };
  const desc = item.system?.description;
  if (typeof desc === 'string' && desc) {
    e.description = desc;
  } else if (desc && typeof desc === 'object') {
    const d = {};
    if (desc.public) d.public = desc.public;
    if (desc.private) d.private = desc.private;
    if (Object.keys(d).length) e.description = d;
  }
  const actions = buildActionsMap(item.system?.actions);
  if (actions) e.actions = actions;
  return e;
}

function buildActorEntry(actor, embeddedItems = []) {
  const e = { name: actor.name };
  const protoName = actor.prototypeToken?.name;
  if (protoName) e.tokenName = protoName;

  for (const k of ['ancestry', 'background', 'archetype', 'taxonomy']) {
    const d = actor.system?.details?.[k];
    if (d && typeof d === 'object') {
      const entry = {};
      if (d.name) entry.name = d.name;
      if (d.description) entry.description = d.description;
      if (Object.keys(entry).length) e[k] = entry;
    }
  }
  const bio = actor.system?.details?.biography;
  if (bio && typeof bio === 'object') {
    const entry = {};
    if (bio.public) entry.public = bio.public;
    if (bio.private) entry.private = bio.private;
    if (Object.keys(entry).length) e.biography = entry;
  }

  const actions = buildActionsMap(actor.system?.actions);
  if (actions) e.actions = actions;

  // items: from external actors.items bucket, or inline (adventure actors).
  const items = embeddedItems.length
    ? embeddedItems
    : (Array.isArray(actor.items) && actor.items.length && typeof actor.items[0] === 'object'
        ? actor.items : []);
  if (items.length) {
    const map = {};
    for (const it of items) {
      if (!it?.name) continue;
      if (!map[it.name]) map[it.name] = buildItemEntry(it);
    }
    if (Object.keys(map).length) e.items = map;
  }
  return e;
}

function buildJournalEntry(journal, pagesByParent = null, categoriesByParent = null) {
  const e = { name: journal.name };

  const categories = categoriesByParent?.[journal._id];
  if (categories && categories.length) {
    const sorted = [...categories].sort((a, b) => (a.sort ?? 0) - (b.sort ?? 0));
    const out = {};
    for (const c of sorted) {
      if (!c._id) continue;
      const ce = {};
      if (c.name) ce.name = c.name;
      if (Object.keys(ce).length) out[c._id] = ce;
    }
    if (Object.keys(out).length) e.categories = out;
  }

  let pages = null;
  if (pagesByParent && pagesByParent[journal._id]) {
    pages = pagesByParent[journal._id];
  } else if (Array.isArray(journal.pages) && journal.pages.length && typeof journal.pages[0] === 'object') {
    pages = journal.pages;
  }
  if (pages && pages.length) {
    const sorted = [...pages].sort((a, b) => (a.sort ?? 0) - (b.sort ?? 0));
    const out = {};
    for (const p of sorted) {
      if (!p.name) continue;
      const pe = { name: p.name };
      if (p.text?.content) pe.text = p.text.content;
      if (p.src) pe.src = p.src;
      if (p.image?.caption) pe.caption = p.image.caption;
      if (!out[p.name]) out[p.name] = pe;
    }
    if (Object.keys(out).length) e.pages = out;
  }
  return e;
}

function buildAdventureEntry(adv) {
  const e = { name: adv.name };
  if (adv.caption) e.caption = adv.caption;
  if (adv.description) e.description = adv.description;

  if (Array.isArray(adv.scenes) && adv.scenes.length) {
    e.scenes = {};
    for (const s of adv.scenes) if (s?.name && !e.scenes[s.name]) e.scenes[s.name] = { name: s.name };
  }
  if (Array.isArray(adv.macros) && adv.macros.length) {
    e.macros = {};
    for (const m of adv.macros) {
      if (!m?.name || e.macros[m.name]) continue;
      const me = { name: m.name };
      if (m.command) me.command = m.command;
      e.macros[m.name] = me;
    }
  }
  if (Array.isArray(adv.folders) && adv.folders.length) {
    e.folders = {};
    for (const f of adv.folders) if (f?.name && !e.folders[f.name]) e.folders[f.name] = f.name;
  }
  if (Array.isArray(adv.journal) && adv.journal.length) {
    e.journals = {};
    for (const j of adv.journal) {
      if (!j?.name || e.journals[j.name]) continue;
      e.journals[j.name] = buildJournalEntry(j);
    }
  }
  if (Array.isArray(adv.items) && adv.items.length) {
    e.items = {};
    for (const it of adv.items) {
      if (!it?.name || e.items[it.name]) continue;
      e.items[it.name] = buildItemEntry(it);
    }
  }
  if (Array.isArray(adv.actors) && adv.actors.length) {
    e.actors = {};
    for (const a of adv.actors) {
      if (!a?.name || e.actors[a.name]) continue;
      e.actors[a.name] = buildActorEntry(a);
    }
  }
  return e;
}

// ---- Mapping definitions (must stay in sync with babele-register.js) ----
const ACTOR_MAPPING = {
  items:      { path: 'items',                     converter: 'adventure_items_converter' },
  actions:    { path: 'system.actions',            converter: 'actions_converter' },
  ancestry:   { path: 'system.details.ancestry',   converter: 'nested_object_converter' },
  background: { path: 'system.details.background', converter: 'nested_object_converter' },
  biography:  { path: 'system.details.biography',  converter: 'nested_object_converter' },
  archetype:  { path: 'system.details.archetype',  converter: 'nested_object_converter' },
  taxonomy:   { path: 'system.details.taxonomy',   converter: 'nested_object_converter' }
};
const ITEM_MAPPING_DESC = { description: 'system.description' };
const ITEM_MAPPING_DESC_ACTIONS = {
  description: 'system.description',
  actions: { path: 'system.actions', converter: 'actions_converter' }
};
const JOURNAL_MAPPING = {
  categories: { path: 'categories', converter: 'categories_converter' }
};
const EFFECT_MAPPING = { description: 'description' };
const MACRO_MAPPING = {};
const ADVENTURE_MAPPING = {
  actors: ACTOR_MAPPING,
  items: {},
  journals: {},
  scenes: {},
  macros: {}
};

// ---- Pack processing ----
async function processPack(pack, systemPacksDir) {
  const packDir = path.join(systemPacksDir, pack.name);
  if (!fs.existsSync(packDir)) {
    console.warn(`  skip (missing dir): ${packDir}`);
    return null;
  }
  const buckets = await readPack(packDir);

  const folders = {};
  for (const { doc } of (buckets.folders || [])) {
    if (doc?.name) folders[doc.name] = doc.name;
  }

  const out = { label: pack.label, mapping: null, folders, entries: {} };

  if (pack.type === 'Item') {
    const items = buckets.items || [];
    const hasActions = items.some(x => Array.isArray(x.doc?.system?.actions) && x.doc.system.actions.length);
    out.mapping = hasActions ? ITEM_MAPPING_DESC_ACTIONS : ITEM_MAPPING_DESC;
    for (const { doc } of items) {
      if (!doc?.name) continue;
      if (!out.entries[doc.name]) out.entries[doc.name] = buildItemEntry(doc);
    }
  } else if (pack.type === 'Actor') {
    out.mapping = ACTOR_MAPPING;
    const itemsByParent = {};
    for (const { idPart, doc } of (buckets['actors.items'] || [])) {
      const parentId = idPart.split('.')[0];
      (itemsByParent[parentId] ||= []).push(doc);
    }
    for (const { doc } of (buckets.actors || [])) {
      if (!doc?.name) continue;
      if (!out.entries[doc.name]) {
        out.entries[doc.name] = buildActorEntry(doc, itemsByParent[doc._id] || []);
      }
    }
  } else if (pack.type === 'JournalEntry') {
    out.mapping = JOURNAL_MAPPING;
    const pagesByParent = {};
    for (const { idPart, doc } of (buckets['journal.pages'] || [])) {
      const parentId = idPart.split('.')[0];
      (pagesByParent[parentId] ||= []).push(doc);
    }
    const categoriesByParent = {};
    for (const { idPart, doc } of (buckets['journal.categories'] || [])) {
      const parentId = idPart.split('.')[0];
      (categoriesByParent[parentId] ||= []).push(doc);
    }
    for (const { doc } of (buckets.journal || [])) {
      if (!doc?.name) continue;
      if (!out.entries[doc.name]) out.entries[doc.name] = buildJournalEntry(doc, pagesByParent, categoriesByParent);
    }
  } else if (pack.type === 'ActiveEffect') {
    out.mapping = EFFECT_MAPPING;
    for (const { doc } of (buckets.effects || [])) {
      if (!doc?.name) continue;
      if (out.entries[doc.name]) continue;
      const e = { name: doc.name };
      if (doc.description) e.description = doc.description;
      out.entries[doc.name] = e;
    }
  } else if (pack.type === 'Macro') {
    out.mapping = MACRO_MAPPING;
    for (const { doc } of (buckets.macros || [])) {
      if (!doc?.name) continue;
      if (out.entries[doc.name]) continue;
      out.entries[doc.name] = { name: doc.name };
    }
  } else if (pack.type === 'Adventure') {
    out.mapping = ADVENTURE_MAPPING;
    for (const { doc } of (buckets.adventures || [])) {
      if (!doc?.name) continue;
      if (!out.entries[doc.name]) out.entries[doc.name] = buildAdventureEntry(doc);
    }
  } else {
    console.warn(`  unsupported pack type: ${pack.type} (${pack.name})`);
    return null;
  }

  return out;
}

// ---- Main ----
async function main() {
  const sysJsonPath = path.join(SYSTEM_DIR, 'system.json');
  if (!fs.existsSync(sysJsonPath)) {
    console.error(`system.json not found: ${sysJsonPath}`);
    process.exit(1);
  }
  const sysJson = readJSON(sysJsonPath);
  const packsDir = path.join(SYSTEM_DIR, 'packs');

  console.log(`System : ${sysJson.id} v${sysJson.version}`);
  console.log(`Packs  : ${packsDir}`);
  console.log(`Output : ${OUT_DIR}\n`);

  for (const pack of sysJson.packs) {
    console.log(`- ${pack.name} (${pack.type})`);
    const result = await processPack(pack, packsDir);
    if (!result) continue;
    const outFile = path.join(OUT_DIR, `${sysJson.id}.${pack.name}.json`);
    writeJSON(outFile, result);
    console.log(`    -> ${path.relative(projectRoot, outFile).replace(/\\/g, '/')}  (${Object.keys(result.entries).length} entries, ${Object.keys(result.folders).length} folders)`);
  }

  console.log('\nDone.');
}

main().catch(e => { console.error(e); process.exit(1); });
