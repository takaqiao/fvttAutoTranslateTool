import { ClassicLevel } from 'classic-level';
import fs from 'fs';
const dir = process.argv[2];
const db = new ClassicLevel(dir, { valueEncoding: 'json' });
await db.open();
const out = [];
const keys = new Set();
for await (const [k, v] of db.iterator()) {
  keys.add(k.split('!')[1]);
  if (!Array.isArray(v.scenes)) continue;
  for (const s of v.scenes) {
    out.push({ id: s._id, name: s.name, flagKeys: Object.keys(s.flags||{}), emberFlags: s.flags?.ember ? Object.keys(s.flags.ember) : null, levels: s.flags?.ember?.levels ?? s.levels ?? null });
  }
}
console.log('doc types:', [...keys].join(','));
console.log('scenes:', out.length);
fs.writeFileSync(process.argv[3], JSON.stringify(out, null, 1), 'utf8');
await db.close();
