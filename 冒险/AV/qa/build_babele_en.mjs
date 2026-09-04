/**
 * build_babele_en.mjs - emit a Babele-shaped ENGLISH baseline straight from the
 * installed LevelDB packs, so every downstream tool has an exact, uncontaminated
 * reference for the source text.
 *
 * Why this exists: every English-looking file already on disk is contaminated or
 * stale.  `冒险/AV/pf2e-abomination-vaults.av.json` was exported with Babele
 * active (255/255 actors carry Chinese); `模组/system/pf2e_compendium/en-US/...`
 * and `更新merge/...` describe a different AV version (19 lettered journals vs the
 * installed 13 numbered ones).  Only the pack itself is authoritative.
 *
 * Field mapping = babele defaults MERGED WITH the pack-local override, which is
 * how babele itself composes them (DocumentMappings#mergedDefinition ->
 * foundry.utils.mergeObject).  The override is NOT a replacement: actor `items`
 * and `tokenName` stay live alongside the PF2e statblock fields.
 *
 * Key allocation and traversal mirror dump_pack_keys.mjs (same babele rules).
 *
 * Foundry must be CLOSED.
 *
 * Usage:
 *   node build_babele_en.mjs --data-root <Data/modules> --modules a,b,c --out-dir <dir>
 */
import fs from 'node:fs';
import path from 'node:path';

import { loadPack } from './pack_loader.mjs';

// ---- scalar fields per document type: babele defaults ------------------------
const BASE_FIELDS = {
  Adventure:        { name: 'name', description: 'description', caption: 'caption' },
  Actor:            { name: 'name', description: 'system.details.biography.value', tokenName: 'prototypeToken.name' },
  Item:             { name: 'name', description: 'system.description.value' },
  ActiveEffect:     { name: 'name', description: 'description' },
  JournalEntry:     { name: 'name', description: 'content' },
  JournalEntryPage: { name: 'name', caption: 'image.caption', src: 'src', text: 'text.content',
                      width: 'video.width', height: 'video.height' },
  Macro:            { name: 'name', command: 'command' },
  Playlist:         { name: 'name', description: 'description' },
  PlaylistSound:    { name: 'name', description: 'description' },
  RollTable:        { name: 'name', description: 'description' },
  TableResult:      { name: 'name', description: 'description' },
  Scene:            { name: 'name' },
  Region:           { name: 'name' },
  RegionBehavior:   { name: 'name' },
  Cards:            { name: 'name', description: 'description' },
  Card:             { name: 'name', description: 'description', suit: 'suit' },
};

// ---- the pack-local override every AV-family pack ships ----------------------
const PF2E_OVERRIDE = {
  Actor: {
    name: 'name',
    publicNotes: 'system.details.publicNotes',
    privateNotes: 'system.details.privateNotes',
    disable: 'system.details.disable',
    hazarddescription: 'system.details.description',
    reset: 'system.details.reset',
    routine: 'system.details.routine',
    blurb: 'system.details.blurb',
    stealthdetails: 'system.attributes.stealth.details',
    hp: 'system.attributes.hp.details',
    senses: 'system.traits.senses.value',
    allSaves: 'system.attributes.allSaves.value',
    ac: 'system.attributes.ac.details',
    prototypeToken: 'prototypeToken.name',
  },
  Item: { name: 'name', description: 'system.description.value' },
};

const COLLECTIONS = {
  Adventure: {
    journals:  { path: 'journal',   type: 'JournalEntry' },
    scenes:    { path: 'scenes',    type: 'Scene' },
    macros:    { path: 'macros',    type: 'Macro' },
    playlists: { path: 'playlists', type: 'Playlist' },
    tables:    { path: 'tables',    type: 'RollTable' },
    items:     { path: 'items',     type: 'Item' },
    actors:    { path: 'actors',    type: 'Actor' },
    cards:     { path: 'cards',     type: 'Cards' },
  },
  Actor:        { items: { path: 'items', type: 'Item' }, effects: { path: 'effects', type: 'ActiveEffect' } },
  Item:         { effects: { path: 'effects', type: 'ActiveEffect' } },
  JournalEntry: { pages: { path: 'pages', type: 'JournalEntryPage' } },
  Playlist:     { sounds: { path: 'sounds', type: 'PlaylistSound' } },
  RollTable:    { results: { path: 'results', type: 'TableResult' } },
  Scene:        { regions: { path: 'regions', type: 'Region' } },
  Region:       { behaviors: { path: 'behaviors', type: 'RegionBehavior' } },
  Cards:        { cards: { path: 'cards', type: 'Card' } },
};
const VALUE_COLLECTIONS = {
  Adventure:    { folders: { path: 'folders', by: 'name' } },
  JournalEntry: { categories: { path: 'categories', by: 'name' } },
  Scene:        { notes: { path: 'notes', by: 'text' }, drawings: { path: 'drawings', by: 'text' } },
};

const IDENTITY = { TableResult: { export: ['range', '_id'] } };
const DEFAULT_EXPORT = ['name', '_id', 'id'];

function extract(doc, token) {
  if (token === '_id') return doc && doc._id ? [doc._id] : [];
  if (token === 'id') return doc && doc.id ? [doc.id] : [];
  if (token === 'name') return doc && doc.name ? [doc.name] : [];
  return [];
}
function exportCandidates(doc, type) {
  const tokens = (IDENTITY[type] && IDENTITY[type].export) || DEFAULT_EXPORT;
  const out = [];
  for (const t of tokens) for (const k of extract(doc, t)) if (!out.includes(k)) out.push(k);
  return out;
}
function allocateKeys(docs, type) {
  const used = new Set(); const out = [];
  for (const doc of docs || []) {
    let key = null;
    for (const c of exportCandidates(doc, type)) if (!used.has(c)) { used.add(c); key = c; break; }
    if (key === null) {
      let i = used.size; let fb = 'entry-' + i;
      while (used.has(fb)) { i += 1; fb = 'entry-' + i; }
      used.add(fb); key = fb;
    }
    out.push(key);
  }
  return out;
}

function getPath(obj, dotted) {
  let cur = obj;
  for (const seg of dotted.split('.')) {
    if (cur === null || cur === undefined || typeof cur !== 'object') return undefined;
    cur = cur[seg];
  }
  return cur;
}

// A pack may declare mapping fields beyond the AV-family override: tianzes-gauntlight-extras
// adds speed / di / languages / dr. Missing them means those leaves never reach the English
// baseline, so the unit system cannot see they are untranslated.
let PACK_MAPPING = {};

function fieldsFor(type, usePf2eOverride) {
  const base = BASE_FIELDS[type] || {};
  const over = usePf2eOverride ? (PF2E_OVERRIDE[type] || {}) : {};
  const key = type === 'Actor' ? 'actors' : (type === 'Item' ? 'items' : null);
  const declared = key && PACK_MAPPING[key] ? PACK_MAPPING[key] : {};
  const extra = {};
  for (const field of Object.keys(declared)) {
    if (typeof declared[field] === 'string') extra[field] = declared[field];
  }
  return Object.assign({}, base, over, extra);
}

const stats = { emitted: 0, skippedNonString: 0, skippedEmpty: 0 };

function buildDoc(doc, type, usePf2eOverride) {
  const out = {};
  const fields = fieldsFor(type, usePf2eOverride);
  for (const key of Object.keys(fields)) {
    const value = getPath(doc, fields[key]);
    if (typeof value === 'string') {
      if (value.trim()) { out[key] = value; stats.emitted += 1; }
      else stats.skippedEmpty += 1;
    } else if (value !== undefined && value !== null) {
      stats.skippedNonString += 1;      // e.g. system.traits.senses.value can be an array
    }
  }

  const subs = COLLECTIONS[type] || {};
  for (const outKey of Object.keys(subs)) {
    const spec = subs[outKey];
    const children = doc && doc[spec.path];
    if (!Array.isArray(children) || !children.length) continue;
    const keys = allocateKeys(children, spec.type);
    const block = {};
    children.forEach(function (child, i) {
      const built = buildDoc(child, spec.type, usePf2eOverride);
      if (Object.keys(built).length) block[keys[i]] = built;
    });
    if (Object.keys(block).length) out[outKey] = block;
  }

  const valueSubs = VALUE_COLLECTIONS[type] || {};
  for (const outKey of Object.keys(valueSubs)) {
    const spec = valueSubs[outKey];
    const children = doc && doc[spec.path];
    if (!Array.isArray(children) || !children.length) continue;
    const block = {};
    for (const child of children) {
      const v = child && child[spec.by];
      if (typeof v === 'string' && v.trim()) block[v] = v;   // identity: EN -> EN
    }
    if (Object.keys(block).length) out[outKey] = block;
  }
  return out;
}

function arg(name, def) {
  const i = process.argv.indexOf('--' + name);
  return (i >= 0 && process.argv[i + 1]) ? process.argv[i + 1] : (def === undefined ? null : def);
}
const dataRoot = arg('data-root');
const modules = (arg('modules') || '').split(',').map(function (s) { return s.trim(); }).filter(Boolean);
const outDir = arg('out-dir');
const mappingFrom = arg('mapping-from');
const noOverride = process.argv.includes('--no-pf2e-override');
if (!dataRoot || !modules.length || !outDir) {
  console.error('usage: --data-root <dir> --modules a,b,c --out-dir <dir> [--no-pf2e-override]');
  process.exit(2);
}
fs.mkdirSync(outDir, { recursive: true });

let declared = 0, built = 0;
for (const modId of modules) {
  const modJsonPath = path.join(dataRoot, modId, 'module.json');
  if (!fs.existsSync(modJsonPath)) { console.error('[MISS] ' + modId); continue; }
  const modJson = JSON.parse(fs.readFileSync(modJsonPath, 'utf8'));

  for (const pack of modJson.packs || []) {
    declared += 1;
    // Same stale-`path` tolerance as dump_pack_keys.mjs: a module.json may still declare
    // the pre-v11 `packs/<name>.db` while the LevelDB directory is `packs/<name>`.
    const rels = [];
    if (pack.path) rels.push(pack.path.replace(/^\.\//, ''));
    rels.push('packs/' + pack.name);
    if (pack.path) rels.push(pack.path.replace(/^\.\//, '').replace(/\.db$/, ''));
    let packDir = null;
    for (const rel of rels) {
      const dir = path.join(dataRoot, modId, rel);
      if (fs.existsSync(path.join(dir, 'CURRENT'))) { packDir = dir; break; }
    }
    if (!packDir) { console.error('[MISS] ' + modId + '.' + pack.name + '; tried ' + rels.join(', ')); continue; }

    const loaded = await loadPack(packDir);
    const docs = loaded.docs;
    const packFolders = loaded.folders;
    if (loaded.orphanEmbeds.length) {
      console.error('[warn] ' + modId + '.' + pack.name + ': ' + loaded.orphanEmbeds.length + ' embedded docs with no parent');
    }

    // Load the pack's own mapping (from the shipped translation) before building.
    PACK_MAPPING = {};
    if (mappingFrom) {
      const shipped = path.join(mappingFrom, modId + '.' + pack.name + '.json');
      if (fs.existsSync(shipped)) {
        try { PACK_MAPPING = JSON.parse(fs.readFileSync(shipped, 'utf8')).mapping || {}; }
        catch (e) { /* leave empty */ }
      }
    }
    const usePf2e = !noOverride;
    const keys = allocateKeys(docs, pack.type);
    const entries = {};
    docs.forEach(function (doc, i) {
      const built2 = buildDoc(doc, pack.type, usePf2e);
      if (Object.keys(built2).length) entries[keys[i]] = built2;
    });

    const folders = {};
    for (const f of packFolders) if (f && typeof f.name === 'string' && f.name.trim()) folders[f.name] = f.name;

    const payload = { label: pack.label || pack.name, entries: entries };
    if (Object.keys(folders).length) payload.folders = folders;

    const outFile = path.join(outDir, modId + '.' + pack.name + '.json');
    fs.writeFileSync(outFile, JSON.stringify(payload, null, 1) + '\n', 'utf8');
    built += 1;
    console.log('[en] ' + (modId + '.' + pack.name).padEnd(58) + ' entries=' + String(Object.keys(entries).length).padStart(4)
      + ' packFolders=' + String(Object.keys(folders).length).padStart(3) + ' -> ' + path.basename(outFile));
  }
}
console.log('\nbaseline coverage: ' + built + '/' + declared + ' packs - missing ' + (declared - built));
console.log('leaf strings emitted: ' + stats.emitted + ' | skipped non-string: ' + stats.skippedNonString
  + ' | skipped empty: ' + stats.skippedEmpty);
if (built !== declared) process.exit(1);
