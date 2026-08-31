/** Round-20 probe ③ helper: what do the pack docs actually look like? */
import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';
const { ClassicLevel } = createRequire('C:/Users/Taka/Desktop/fvtt/package.json')('classic-level');
const argv = process.argv.slice(2);
const arg = (n, d) => { const i = argv.indexOf(n); return i >= 0 ? argv[i + 1] : d; };
const PACKAGE_DIR = arg('--package');
const NEEDLE = arg('--needle', 'warlocks');

const packsDir = path.join(PACKAGE_DIR, 'packs');
let docsSeen = 0, hits = 0;
const keyPrefixes = new Map();
for (const p of fs.readdirSync(packsDir)) {
  const dir = path.join(packsDir, p);
  if (!fs.statSync(dir).isDirectory()) continue;
  const db = new ClassicLevel(dir, { keyEncoding: 'utf8', valueEncoding: 'json' });
  await db.open();
  for await (const [key, doc] of db.iterator()) {
    docsSeen++;
    const pre = `${p}:${key.split('!')[1] ?? '?'}`;
    keyPrefixes.set(pre, (keyPrefixes.get(pre) || 0) + 1);
    const s = JSON.stringify(doc);
    if (s.includes(NEEDLE)) {
      hits++;
      if (hits <= 3) console.log(`HIT ${p} ${key} name=${doc?.name} len=${s.length}`);
    }
  }
  await db.close();
}
console.log(`SCANNED docs=${docsSeen} needle=${JSON.stringify(NEEDLE)} hits=${hits}`);
console.log('key prefixes:');
for (const [k, v] of [...keyPrefixes].sort()) console.log(`  ${k} x${v}`);
