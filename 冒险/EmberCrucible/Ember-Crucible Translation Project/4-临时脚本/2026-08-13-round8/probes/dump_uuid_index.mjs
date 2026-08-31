#!/usr/bin/env node
/**
 * Dump a *UUID resolution index* from a Foundry package's LevelDB packs.
 *
 * `dump_ids.mjs` only records ids that carry a `name`/`text`, and it flattens
 * every package into one bag. Neither is enough to answer "does
 * `@UUID[JournalEntry.a.JournalEntryPage.b]` resolve?": that needs (1) every
 * `_id` regardless of whether it has a name, (2) the *Document class* each id
 * belongs to (Actor vs Item vs JournalEntryPage), and (3) the parent->child
 * containment edges, so a real page id hung off the wrong journal id is still
 * a dead link.
 *
 * Output:
 * {
 *   "package": {id, type, version},
 *   "packs":   {"<packName>": "<top-level Document class>"},
 *   "ids":     {"<_id>": [{doc, name, type, pkg, pack, parent}]},   # doc = Document class
 *   "children":{"<parentId>": ["<childId>", ...]}
 * }
 *
 * Usage: node dump_uuid_index.mjs --package <foundry package dir> --out <file.json>
 */
import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';

const FVTT_NODE_ANCHOR = 'C:/Users/Taka/Desktop/fvtt/package.json';
const { ClassicLevel } = createRequire(FVTT_NODE_ANCHOR)('classic-level');

const argv = process.argv.slice(2);
const arg = (n, d) => { const i = argv.indexOf(n); return i >= 0 ? argv[i + 1] : d; };
const PACKAGE_DIR = arg('--package');
const OUT = arg('--out');

/**
 * JSON collection key -> Document class name, i.e. the token a UUID uses.
 * An Adventure's `actors`/`journal`/`scenes` are imported into the *world*, so
 * their children address as `Actor.x` / `JournalEntry.x` / `Scene.x`.
 * `levels` is Ember's own Scene sub-document (`Scene.x.Level.y` occurs in the
 * corpus), so it is listed here even though core Foundry has no such class.
 */
const COLLECTION_DOC = {
  actors: 'Actor', items: 'Item', journal: 'JournalEntry', pages: 'JournalEntryPage',
  scenes: 'Scene', tables: 'RollTable', results: 'TableResult', macros: 'Macro',
  playlists: 'Playlist', sounds: 'AmbientSound', cards: 'Cards', folders: 'Folder',
  combats: 'Combat', effects: 'ActiveEffect', notes: 'Note', tokens: 'Token',
  regions: 'Region', behaviors: 'RegionBehavior', lights: 'AmbientLight',
  drawings: 'Drawing', tiles: 'Tile', walls: 'Wall', templates: 'MeasuredTemplate',
  levels: 'Level', categories: 'JournalEntryCategory', combatants: 'Combatant',
  activities: 'Activity',
};

/** LevelDB key prefix (`!actors!...`) -> Document class for the top-level doc. */
const KEY_DOC = {
  actors: 'Actor', items: 'Item', journal: 'JournalEntry', scenes: 'Scene',
  tables: 'RollTable', macros: 'Macro', playlists: 'Playlist', cards: 'Cards',
  adventures: 'Adventure', folders: 'Folder', combats: 'Combat',
  effects: 'ActiveEffect',
};

const pkgFile = fs.existsSync(path.join(PACKAGE_DIR, 'system.json')) ? 'system.json' : 'module.json';
const manifest = JSON.parse(fs.readFileSync(path.join(PACKAGE_DIR, pkgFile), 'utf8'));
const PKG = manifest.id;

const ids = {};
const children = {};
const packs = {};

function record(id, doc, name, type, pack, parent) {
  if (!id || typeof id !== 'string') return;
  (ids[id] ??= []).push({ doc, name: name ?? null, type: type ?? null, pkg: PKG, pack, parent: parent ?? null });
  if (parent) (children[parent] ??= []).push(id);
}

/**
 * Walk a document tree. `doc` is the Document class of `node`; every recognised
 * child collection recurses with its own class and `node._id` as parent.
 * Unrecognised object fields are still walked (so nothing with an `_id` is
 * missed) but they inherit the parent's class as a `?`-suffixed label.
 */
function walk(node, docClass, pack, parentId) {
  if (Array.isArray(node)) { for (const n of node) walk(n, docClass, pack, parentId); return; }
  if (!node || typeof node !== 'object') return;
  let self = parentId;
  if (typeof node._id === 'string') {
    record(node._id, docClass, node.name, node.type, pack, parentId);
    self = node._id;
  }
  for (const [k, v] of Object.entries(node)) {
    if (!v || typeof v !== 'object') continue;
    const child = COLLECTION_DOC[k];
    walk(v, child ?? `${docClass}?${k}`, pack, self);
  }
}

const packsDir = path.join(PACKAGE_DIR, 'packs');
for (const name of fs.readdirSync(packsDir)) {
  const dir = path.join(packsDir, name);
  if (!fs.statSync(dir).isDirectory()) continue;
  // Register the pack even if its DB has no rows: "pack exists but is empty"
  // and "no such pack" are different verdicts for a link target.
  packs[name] = {};
  const db = new ClassicLevel(dir, { valueEncoding: 'json' });
  await db.open();
  for await (const [key, value] of db.iterator()) {
    // Keys look like `!items!<id>` or, for embedded rows, `!items.effects!<parentId>.<id>`.
    const [, collection, locator = ''] = String(key).split('!');
    const segs = (collection ?? '?').split('.');
    const docClass = KEY_DOC[collection] ?? COLLECTION_DOC[segs.at(-1)] ?? collection;
    const chain = locator.split('.');
    const parent = chain.length > 1 ? chain.at(-2) : null;
    packs[name][docClass] = (packs[name][docClass] ?? 0) + 1;
    walk(value, docClass, name, parent);
  }
  await db.close();
}

fs.writeFileSync(OUT, JSON.stringify({
  package: { id: PKG, type: pkgFile === 'system.json' ? 'system' : 'module', version: manifest.version },
  packs, ids, children,
}, null, 0), 'utf8');
console.log(`${PKG} v${manifest.version}: packs=${Object.keys(packs).length} ids=${Object.keys(ids).length} -> ${OUT}`);
