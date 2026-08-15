#!/usr/bin/env node
/**
 * Dump `document _id -> English name` for every document (and embedded
 * document) in a Foundry package's LevelDB packs.
 *
 * Why: the extracted `compendium/en/*.json` are keyed by NAME, so a
 * `@UUID[...JournalEntryPage.4lTtZfLS961MjIST]` target cannot be resolved to a
 * page from the translation files alone. The ids only exist in the packs.
 * With this table the correct label for a link target is a fact, not a
 * majority vote.
 *
 * Usage: node dump_ids.mjs --package <dir> --out <file.json>
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

const out = {};   // id -> {name, type, pack, via}

function record(id, name, type, pack, via) {
  if (!id || typeof id !== 'string') return;
  if (!out[id]) out[id] = { name: name ?? null, type, pack, via };
  else if (out[id].name === null && name) out[id].name = name;
}

/** Recurse a document, recording every `_id`/`name` pair we meet. */
function scan(doc, pack, via) {
  if (Array.isArray(doc)) { for (const d of doc) scan(d, pack, via); return; }
  if (!doc || typeof doc !== 'object') return;
  if (typeof doc._id === 'string' && (typeof doc.name === 'string' || typeof doc.text === 'string')) {
    record(doc._id, doc.name ?? null, doc.type ?? null, pack, via);
  }
  for (const [k, v] of Object.entries(doc)) {
    if (v && typeof v === 'object') scan(v, pack, `${via}.${k}`);
  }
}

const packsDir = path.join(PACKAGE_DIR, 'packs');
for (const name of fs.readdirSync(packsDir)) {
  const dir = path.join(packsDir, name);
  if (!fs.statSync(dir).isDirectory()) continue;
  const db = new ClassicLevel(dir, { valueEncoding: 'json' });
  await db.open();
  for await (const [key, value] of db.iterator()) {
    scan(value, name, String(key).split('!')[1] ?? '?');
  }
  await db.close();
}

fs.writeFileSync(OUT, JSON.stringify(out, null, 1), 'utf8');
console.log(`ids=${Object.keys(out).length} -> ${OUT}`);
