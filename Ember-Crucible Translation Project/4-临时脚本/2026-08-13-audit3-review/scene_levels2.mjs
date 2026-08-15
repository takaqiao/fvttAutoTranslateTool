import { ClassicLevel } from 'classic-level';
import fs from 'fs';
const out = {};
for (const dir of process.argv.slice(3)) {
  const db = new ClassicLevel(dir, { valueEncoding: 'json' });
  await db.open();
  for await (const [k, v] of db.iterator()) {
    if (!Array.isArray(v.scenes)) continue;
    for (const s of v.scenes) {
      const lv = s.flags?.ember?.levels ?? s.levels ?? [];
      out[s._id] = { name: s.name, levels: lv.map(l => l.name) };
    }
  }
  await db.close();
}
fs.writeFileSync(process.argv[2], JSON.stringify(out, null, 1), 'utf8');
console.log('scenes', Object.keys(out).length, 'levels', Object.values(out).reduce((a,b)=>a+b.levels.length,0));
