/**
 * dump_pack_keys.mjs - read installed Foundry v13/v14 LevelDB packs and emit
 *   (a) raw documents with `_key` injected (input for fotrp_update.load_adventures)
 *   (b) a key manifest carrying Babele's own export/match candidates per document
 *
 * Faithfully mirrors, for the installed Babele:
 *   - script/mapping/default-mappings.js              (collection traversal)
 *   - script/identity/document-identity.js            (export [name,_id,id] / match [_id,name,sourceId])
 *   - script/identity/identity-extractor-registry.js  (the four default extractors)
 *   - script/identity/export-key-allocator.js         (first unused candidate wins, per scope)
 * DO NOT EDIT the tables below by hand - re-derive them from the babele module.
 *
 * Foundry must be CLOSED: classic-level takes an exclusive lock.
 *
 * Usage:
 *   node dump_pack_keys.mjs --data-root <Data/modules> --modules a,b,c \
 *        --raw-out <dir> --keys-out <file.json>
 */
import { createRequire } from 'node:module';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';

const require = createRequire('C:/Users/Taka/Desktop/fvtt/package.json');
const { ClassicLevel } = require('classic-level');

// ---------------------------------------------------------------- mappings
// Sub-collections only: {outputKey: {path, documentType}}. Mirrors defaultMappings.
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

// nameCollection / textCollection sub-objects: keyed by the value, not by a document identity.
const VALUE_COLLECTIONS = {
  Adventure:    { folders: 'folders' },
  JournalEntry: { categories: 'categories' },
  Scene:        { notes: 'notes', drawings: 'drawings' },
};

// Per-type identity overrides taken from defaultMappings._identity
const IDENTITY = { TableResult: { export: ['range', '_id'], match: ['_id', 'range'] } };
const DEFAULT_IDENTITY = { export: ['name', '_id', 'id'], match: ['_id', 'name', 'sourceId'] };

// The four extractors babele registers by default. There is NO `range` extractor,
// so TableResult's export list degrades to ["_id"] - exactly as babele does at runtime.
function extract(doc, token) {
  switch (token) {
    case '_id':  return doc && doc._id ? [doc._id] : [];
    case 'id':   return doc && doc.id ? [doc.id] : [];
    case 'name': return doc && doc.name ? [doc.name] : [];
    case 'sourceId': {
      const uuid = (doc && doc.flags && doc.flags.core && doc.flags.core.sourceId)
        || (doc && doc._stats && doc._stats.compendiumSource);
      if (!uuid) return [];
      const id = String(uuid).split('.').pop();
      return id ? [id] : [];
    }
    default: return [];
  }
}

function candidates(doc, tokens) {
  const out = [];
  for (const t of tokens) {
    for (const k of extract(doc, t)) if (!out.includes(k)) out.push(k);
  }
  return out;
}

function identityFor(type) {
  return Object.assign({}, DEFAULT_IDENTITY, IDENTITY[type] || {});
}

/** Babele ExportKeys.keyFor: first unused candidate within scope, else `${prefix}-${n}`. */
function allocateKeys(docs, type, fallbackPrefix) {
  const prefix = fallbackPrefix || 'entry';
  const used = new Set();
  const out = [];
  for (const doc of docs || []) {
    const cands = candidates(doc, identityFor(type).export);
    let key = null;
    for (const c of cands) {
      if (!used.has(c)) { used.add(c); key = c; break; }
    }
    if (key === null) {
      let i = used.size;
      let fb = prefix + '-' + i;
      while (used.has(fb)) { i += 1; fb = prefix + '-' + i; }
      used.add(fb);
      key = fb;
    }
    out.push(key);
  }
  return out;
}

// ---------------------------------------------------------------- traversal
function describe(docs, type, nodePath, sink) {
  if (!Array.isArray(docs) || docs.length === 0) return;
  const keys = allocateKeys(docs, type);
  const entries = docs.map(function (doc, i) {
    return {
      key: keys[i],
      name: doc && doc.name !== undefined ? doc.name : null,
      _id: doc && doc._id !== undefined ? doc._id : null,
      exportCandidates: candidates(doc, identityFor(type).export),
      matchCandidates: candidates(doc, identityFor(type).match),
    };
  });
  sink.push({ path: nodePath, documentType: type, count: docs.length, entries: entries });

  const subs = COLLECTIONS[type] || {};
  for (const outKey of Object.keys(subs)) {
    const spec = subs[outKey];
    docs.forEach(function (doc, i) {
      const child = doc && doc[spec.path];
      if (Array.isArray(child) && child.length) {
        describe(child, spec.type, nodePath + '.' + keys[i] + '.' + outKey, sink);
      }
    });
  }

  const valueSubs = VALUE_COLLECTIONS[type] || {};
  for (const outKey of Object.keys(valueSubs)) {
    const srcPath = valueSubs[outKey];
    docs.forEach(function (doc, i) {
      const child = doc && doc[srcPath];
      if (!Array.isArray(child) || !child.length) return;
      const isText = (outKey === 'notes' || outKey === 'drawings');
      const vals = child
        .map(function (c) { return isText ? (c && c.text) : (c && c.name); })
        .filter(function (v) { return typeof v === 'string' && v.length; });
      const uniq = Array.from(new Set(vals));
      if (!uniq.length) return;
      sink.push({
        path: nodePath + '.' + keys[i] + '.' + outKey,
        documentType: outKey + '(valueCollection)',
        count: uniq.length,
        entries: uniq.map(function (v) {
          return { key: v, name: v, _id: null, exportCandidates: [v], matchCandidates: [v] };
        }),
      });
    });
  }
}

// ---------------------------------------------------------------- main
function arg(name, def) {
  const i = process.argv.indexOf('--' + name);
  return (i >= 0 && process.argv[i + 1]) ? process.argv[i + 1] : (def === undefined ? null : def);
}

const dataRoot = arg('data-root');
const modules = (arg('modules') || '').split(',').map(function (s) { return s.trim(); }).filter(Boolean);
const rawOut = arg('raw-out');
const keysOut = arg('keys-out');

if (!dataRoot || !modules.length || !rawOut || !keysOut) {
  console.error('usage: --data-root <dir> --modules a,b,c --raw-out <dir> --keys-out <file>');
  process.exit(2);
}

const manifest = { generatedBy: 'dump_pack_keys.mjs', dataRoot: dataRoot, packages: [], packs: {} };
let declaredPacks = 0;
let dumpedPacks = 0;

for (const modId of modules) {
  const modDir = path.join(dataRoot, modId);
  const modJsonPath = path.join(modDir, 'module.json');
  if (!fs.existsSync(modJsonPath)) { console.error('[MISS] ' + modId + ': no module.json'); continue; }
  const modJson = JSON.parse(fs.readFileSync(modJsonPath, 'utf8'));
  const version = modJson.version || 'unknown';
  manifest.packages.push({ id: modId, version: version, packCount: (modJson.packs || []).length });

  for (const pack of modJson.packs || []) {
    declaredPacks += 1;
    const rel = (pack.path || ('packs/' + pack.name)).replace(/^\.\//, '');
    const packDir = path.join(modDir, rel);
    if (!fs.existsSync(path.join(packDir, 'CURRENT'))) {
      console.error('[MISS] ' + modId + '.' + pack.name + ': no LevelDB at ' + packDir);
      continue;
    }

    const db = new ClassicLevel(packDir, { valueEncoding: 'json' });
    await db.open();
    const all = [];
    const rawDir = path.join(rawOut, modId + '@' + version, pack.name);
    fs.mkdirSync(rawDir, { recursive: true });
    let n = 0;
    for await (const [k, v] of db.iterator()) {
      const doc = Object.assign({}, v, { _key: k });   // LevelDB stores the key outside the value
      all.push(doc);
      const seg = k.split('!')[1] || 'doc';
      const safe = String(doc.name || doc._id || ('doc' + n)).replace(/[^A-Za-z0-9._-]+/g, '_').slice(0, 80);
      fs.writeFileSync(path.join(rawDir, safe + '__' + seg + '__' + (doc._id || n) + '.json'),
        JSON.stringify(doc, null, 2), 'utf8');
      n += 1;
    }
    await db.close();

    const packFolders = all.filter(function (d) { return String(d._key).startsWith('!folders!'); });
    const primaryDocs = all.filter(function (d) { return !String(d._key).startsWith('!folders!'); });

    const sink = [];
    describe(primaryDocs, pack.type, 'entries', sink);

    const collectionId = modId + '.' + pack.name;
    manifest.packs[collectionId] = {
      module: modId, version: version, pack: pack.name, type: pack.type, path: packDir,
      docCount: n,
      packFolders: packFolders.map(function (f) { return f.name; }),
      nodes: sink,
      sha256: crypto.createHash('sha256').update(JSON.stringify(sink)).digest('hex').slice(0, 16),
    };
    dumpedPacks += 1;
    const top = sink.find(function (s) { return s.path === 'entries'; });
    console.log('[ok] ' + collectionId.padEnd(58) + ' type=' + String(pack.type).padEnd(13)
      + ' docs=' + String(n).padStart(4) + ' nodes=' + String(sink.length).padStart(4)
      + ' top=' + (top ? top.count : 0));
  }
}

fs.mkdirSync(path.dirname(keysOut), { recursive: true });
fs.writeFileSync(keysOut, JSON.stringify(manifest, null, 1), 'utf8');
console.log('\nbaseline coverage: ' + dumpedPacks + '/' + declaredPacks + ' packs (declared '
  + declaredPacks + ' across ' + modules.length + ' module.json) - missing ' + (declaredPacks - dumpedPacks));
console.log('keys manifest -> ' + keysOut);
if (dumpedPacks !== declaredPacks) process.exit(1);
