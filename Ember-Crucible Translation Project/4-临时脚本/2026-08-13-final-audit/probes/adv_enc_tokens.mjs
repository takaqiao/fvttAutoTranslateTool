import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';
const { ClassicLevel } = createRequire('C:/Users/Taka/Desktop/fvtt/package.json')('classic-level');

const EMBER = 'C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/packs';

async function readPack(name) {
  const db = new ClassicLevel(path.join(EMBER, name), { keyEncoding: 'utf8', valueEncoding: 'json' });
  await db.open();
  const out = [];
  for await (const [k, v] of db.iterator()) out.push([k, v]);
  await db.close();
  return out;
}

const res = {};
for (const pack of ['adventure', 'crucible-adventure']) {
  const rows = await readPack(pack);
  res[pack] = rows;
  console.log(pack, 'rows', rows.length, rows.map(r => r[0]).slice(0, 20).join(' | '));
}
fs.writeFileSync(process.argv[2] || 'raw_adv.json', JSON.stringify(res));
