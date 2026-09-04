/**
 * dump_pack_ids.mjs - collect every document id of every installed pack.
 *
 * A translation links out with `@UUID[Compendium.<scope>.<pack>.<Type>.<id>]`, and those
 * links break exactly like world-scoped ones do when a system or module reindexes. Checking
 * only world-scoped targets answers a much smaller question than "are the links good".
 *
 * Output: {"<scope>.<pack>": {"type": "...", "ids": [...]}}  - ids only, so the file stays
 * small enough to hold in memory while scanning a corpus.
 *
 * Usage:
 *   node dump_pack_ids.mjs --data-root <Data> --out ids.json [--packs a.b,c.d]
 */
import fs from 'fs';
import path from 'path';
import { ClassicLevel } from 'classic-level';

function arg(name, dflt) {
  const i = process.argv.indexOf('--' + name);
  return i === -1 ? dflt : process.argv[i + 1];
}

const dataRoot = arg('data-root');
const outPath = arg('out');
const only = (arg('packs', '') || '').split(',').map((s) => s.trim()).filter(Boolean);
if (!dataRoot || !outPath) {
  console.error('usage: --data-root <Data> --out ids.json [--packs a.b,c.d]');
  process.exit(2);
}

/** Every `!collection!id` primary key in a LevelDB pack. Embedded keys carry a dot. */
async function idsOf(dir) {
  const db = new ClassicLevel(dir, { valueEncoding: 'json' });
  await db.open();
  const ids = new Set();
  // name -> [id]. A link that lost its id can still be recovered from its label, but only
  // if the name is unique in the pack; the list keeps that checkable.
  const byName = {};
  for await (const [key, value] of db.iterator()) {
    const parts = String(key).split('!');
    if (parts.length < 3) continue;
    for (const seg of (parts[2] || '').split('.')) if (seg) ids.add(seg);
    const name = value && value.name;
    const id = value && value._id;
    if (name && id) (byName[name] = byName[name] || []).push(id);
  }
  await db.close();
  return { ids: [...ids], byName };
}

function packDirs(manifestPath, root, scope) {
  const json = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
  const out = [];
  for (const pack of json.packs || []) {
    const rels = [];
    if (pack.path) rels.push(pack.path.replace(/^\.\//, ''));
    rels.push('packs/' + pack.name);
    if (pack.path) rels.push(pack.path.replace(/^\.\//, '').replace(/\.db$/, ''));
    for (const rel of rels) {
      const dir = path.join(root, rel);
      if (fs.existsSync(path.join(dir, 'CURRENT'))) {
        out.push({ id: scope + '.' + pack.name, type: pack.type, dir });
        break;
      }
    }
  }
  return out;
}

const targets = [];
const sysDir = path.join(dataRoot, '..', 'Data', 'systems');
for (const base of [path.join(dataRoot, 'systems'), path.join(dataRoot, 'modules')]) {
  if (!fs.existsSync(base)) continue;
  for (const entry of fs.readdirSync(base)) {
    const root = path.join(base, entry);
    const manifest = ['system.json', 'module.json']
      .map((n) => path.join(root, n))
      .find((p) => fs.existsSync(p));
    if (!manifest) continue;
    const scope = JSON.parse(fs.readFileSync(manifest, 'utf8')).id || entry;
    targets.push(...packDirs(manifest, root, scope));
  }
}

const wanted = only.length ? targets.filter((t) => only.includes(t.id)) : targets;
console.error(`scanning ${wanted.length} packs of ${targets.length} installed`);

const result = {};
let done = 0;
for (const t of wanted) {
  try {
    const got = await idsOf(t.dir);
    result[t.id] = { type: t.type, ids: got.ids, byName: got.byName };
    done += 1;
    if (done % 25 === 0) console.error(`  ${done}/${wanted.length}`);
  } catch (e) {
    console.error(`[MISS] ${t.id}: ${e.message}`);
  }
}
fs.writeFileSync(outPath, JSON.stringify(result));
const total = Object.values(result).reduce((n, p) => n + p.ids.length, 0);
console.error(`\n${Object.keys(result).length} packs, ${total} ids -> ${outPath}`);
