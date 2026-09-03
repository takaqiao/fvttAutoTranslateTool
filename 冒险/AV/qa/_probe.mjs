import { createRequire } from 'node:module';
const require = createRequire('C:/Users/Taka/Desktop/fvtt/package.json');
const { ClassicLevel } = require('classic-level');
const dir = process.argv[2];
const db = new ClassicLevel(dir, { valueEncoding: 'json' });
await db.open();
let n = 0; const prefixes = new Map();
let sample = null;
for await (const [k, v] of db.iterator()) {
  n++;
  const p = k.split('!').slice(0, 2).join('!') + '!';
  prefixes.set(p, (prefixes.get(p) || 0) + 1);
  if (!sample && k.startsWith('!adventures!')) sample = [k, v];
}
console.log('total entries:', n);
console.log('prefixes:', JSON.stringify([...prefixes.entries()]));
if (sample) {
  const [k, v] = sample;
  console.log('adventure key:', k);
  console.log('value has _key?', '_key' in v, '| _id:', v._id, '| name:', v.name);
  console.log('top-level fields:', Object.keys(v).join(', '));
  for (const f of ['actors','items','journal','scenes','macros','tables','playlists','folders','cards']) {
    const val = v[f];
    if (Array.isArray(val)) console.log(`  ${f}: len=${val.length} firstType=${typeof val[0]}`);
    else if (val !== undefined) console.log(`  ${f}: ${typeof val}`);
  }
  const j = v.journal?.[0];
  if (j) console.log('  journal[0]:', Object.keys(j).slice(0, 12).join(','), '| pages:', Array.isArray(j.pages) ? j.pages.length : typeof j.pages);
  const a = v.actors?.[0];
  if (a) console.log('  actors[0]:', a.name, '| items:', Array.isArray(a.items) ? a.items.length : typeof a.items, '| _stats:', JSON.stringify(a._stats)?.slice(0,120));
}
await db.close();
