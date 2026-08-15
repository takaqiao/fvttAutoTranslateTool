/** Round-20 probe ③: where in the pack graph does a needle string appear (name-trail)? */
import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';
const { ClassicLevel } = createRequire('C:/Users/Taka/Desktop/fvtt/package.json')('classic-level');
const argv = process.argv.slice(2);
const arg = (n, d) => { const i = argv.indexOf(n); return i >= 0 ? argv[i + 1] : d; };
const PACKAGE_DIR = arg('--package');
const NEEDLE = arg('--needle');

let nodes = 0, hits = 0;
const seen = new Set();
function walk(node, pack, trail) {
  if (node == null || typeof node !== 'object') return;
  nodes++;
  const name = typeof node.name === 'string' ? node.name : null;
  const t2 = name ? `${trail}/${name}` : trail;
  for (const [k, v] of Object.entries(node)) {
    if (typeof v === 'string' && v.includes(NEEDLE)) {
      hits++;
      const line = `${pack}:${t2} .${k}`;
      if (!seen.has(line)) { seen.add(line); console.log(line, '::', v.length > 160 ? v.slice(0, 160) + '…' : v); }
    } else if (typeof v === 'object' && v !== null) walk(v, pack, t2);
  }
  if (typeof node.name === 'string' && node.name.includes(NEEDLE)) { /* counted above */ }
}
const packsDir = path.join(PACKAGE_DIR, 'packs');
let docs = 0;
for (const p of fs.readdirSync(packsDir)) {
  const dir = path.join(packsDir, p);
  if (!fs.statSync(dir).isDirectory()) continue;
  const db = new ClassicLevel(dir, { keyEncoding: 'utf8', valueEncoding: 'json' });
  await db.open();
  for await (const [, doc] of db.iterator()) { docs++; walk(doc, p, ''); }
  await db.close();
}
console.log(`SCANNED docs=${docs} nodes=${nodes} needle=${JSON.stringify(NEEDLE)} hits=${hits} uniqueLocations=${seen.size}`);
