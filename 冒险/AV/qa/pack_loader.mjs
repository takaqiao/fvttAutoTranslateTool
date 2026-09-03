/**
 * pack_loader.mjs - read a Foundry v13/v14 LevelDB pack and hand back fully
 * reassembled primary documents.
 *
 * Foundry stores embedded documents as SEPARATE LevelDB keys, not inline:
 *
 *   !scenes!<sceneId>                                  primary
 *   !scenes.walls!<sceneId>.<wallId>                   embedded, depth 1
 *   !journal.pages!<journalId>.<pageId>                embedded, depth 1
 *   !scenes.regions.behaviors!<sceneId>.<regionId>.<behaviorId>   depth 2
 *   !folders!<folderId>                                pack-level folders
 *
 * The one exception is an Adventure pack: a single `!adventures!<id>` document
 * carries its whole world inline (journal[].pages[], actors[].items[], ...).
 *
 * Reading only `!<collection>!` keys and ignoring the dotted ones silently
 * produces scenes with no notes, journals with no pages, and (because the dotted
 * keys then look like top-level documents) wildly inflated document counts -
 * e.g. abomination-vaults-addons-scenes reads as 653 "scenes" when it holds 9.
 *
 * Foundry must be CLOSED: classic-level takes an exclusive lock.
 */
import { createRequire } from 'node:module';

const require = createRequire('C:/Users/Taka/Desktop/fvtt/package.json');
const { ClassicLevel } = require('classic-level');

/** `!journal.pages!aaa.bbb` -> {segments:['journal','pages'], ids:['aaa','bbb']} */
export function parseKey(key) {
  const parts = String(key).split('!');
  const segments = (parts[1] || '').split('.').filter(Boolean);
  const ids = (parts[2] || '').split('.').filter(Boolean);
  return { segments, ids };
}

/**
 * @param {string} packDir
 * @returns {Promise<{docs: object[], folders: object[], counts: Record<string, number>, orphanEmbeds: string[]}>}
 */
export async function loadPack(packDir) {
  const db = new ClassicLevel(packDir, { valueEncoding: 'json' });
  await db.open();
  const rows = [];
  const counts = {};
  for await (const [k, v] of db.iterator()) {
    const { segments, ids } = parseKey(k);
    const prefix = segments.join('.');
    counts[prefix] = (counts[prefix] || 0) + 1;
    rows.push({ key: k, segments, ids, doc: Object.assign({}, v, { _key: k }) });
  }
  await db.close();

  // Shallowest first so a parent always exists before its children are attached.
  rows.sort((a, b) => a.segments.length - b.segments.length);

  const primaries = new Map();   // collection -> Map(id -> doc)
  const folders = [];
  const orphanEmbeds = [];

  for (const row of rows) {
    if (row.segments.length !== 1) continue;
    const coll = row.segments[0];
    if (coll === 'folders') { folders.push(row.doc); continue; }
    if (!primaries.has(coll)) primaries.set(coll, new Map());
    primaries.get(coll).set(row.ids[0], row.doc);
  }

  for (const row of rows) {
    if (row.segments.length < 2) continue;
    const rootColl = row.segments[0];
    const root = primaries.get(rootColl) && primaries.get(rootColl).get(row.ids[0]);
    if (!root) { orphanEmbeds.push(row.key); continue; }

    // Walk down: segments[1..n-1] are intermediate array fields addressed by ids[1..],
    // segments[n] is the array the document itself belongs to.
    let cursor = root;
    let ok = true;
    for (let depth = 1; depth < row.segments.length - 1; depth += 1) {
      const field = row.segments[depth];
      const childId = row.ids[depth];
      const arr = Array.isArray(cursor[field]) ? cursor[field] : [];
      const next = arr.find((c) => c && c._id === childId);
      if (!next) { ok = false; break; }
      cursor = next;
    }
    if (!ok) { orphanEmbeds.push(row.key); continue; }

    const leafField = row.segments[row.segments.length - 1];
    if (!Array.isArray(cursor[leafField])) cursor[leafField] = [];
    cursor[leafField].push(row.doc);
  }

  // An Adventure pack inlines everything already; nothing above touches it.
  const docs = [];
  for (const byId of primaries.values()) for (const doc of byId.values()) docs.push(doc);

  return { docs, folders, counts, orphanEmbeds };
}
