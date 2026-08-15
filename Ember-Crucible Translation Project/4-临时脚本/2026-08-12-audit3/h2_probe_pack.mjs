/**
 * H2 probe: dump selected raw documents out of a Foundry LevelDB pack so the
 * FR-vs-ours extraction differences can be judged against ground truth.
 *
 * Usage:
 *   node h2_probe_pack.mjs --package <dir> --pack <name> [--name <docName>] [--jq <js expr on doc>]
 */
import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';

const { ClassicLevel } = createRequire('C:/Users/Taka/Desktop/fvtt/package.json')('classic-level');

const argv = process.argv.slice(2);
const arg = (n, d) => { const i = argv.indexOf(n); return i >= 0 ? argv[i + 1] : d; };

const PACKAGE_DIR = arg('--package');
const PACK = arg('--pack');
const NAME = arg('--name');
const FIELD = arg('--field');

const dir = path.join(PACKAGE_DIR, 'packs', PACK);
const db = new ClassicLevel(dir, { keyEncoding: 'utf8', valueEncoding: 'json' });
await db.open();

for await (const [key, doc] of db.iterator()) {
  if (NAME && doc?.name !== NAME) continue;
  if (FIELD) {
    const v = FIELD.split('.').reduce((a, k) => (a == null ? a : a[k]), doc);
    console.log(key, '|', doc?.name, '|', JSON.stringify(v));
  } else {
    console.log('=== ', key, doc?.name);
    console.log(JSON.stringify(doc, null, 1).slice(0, 6000));
  }
}
await db.close();
