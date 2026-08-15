import { ClassicLevel } from 'classic-level';
import fs from 'fs';
const out = {};
for (const dir of process.argv.slice(3)) {
  const db = new ClassicLevel(dir, { valueEncoding: 'json' });
  await db.open();
  for await (const [k, v] of db.iterator()) {
    if (!Array.isArray(v.actors)) continue;
    for (const a of v.actors) out[a._id] = { name: a.name, token: a.prototypeToken?.name };
  }
  await db.close();
}
fs.writeFileSync(process.argv[2], JSON.stringify(out, null, 1), 'utf8');
console.log('actors', Object.keys(out).length);
