// READ-ONLY: dump {name, type, system.identifier} from a Foundry LevelDB pack.
import { createRequire } from 'module';
const { ClassicLevel } = createRequire('C:/Users/Taka/Desktop/fvtt/package.json')('classic-level');
const dir = process.argv[2];
const db = new ClassicLevel(dir, { keyEncoding: 'utf8', valueEncoding: 'json' });
const out = [];
for await (const [k, v] of db.iterator()) {
  if (!k.startsWith('!items!') && !k.startsWith('!journal') && !k.startsWith('!actors!')) continue;
  out.push({ key: k, name: v?.name, type: v?.type, identifier: v?.system?.identifier });
}
await db.close();
out.sort((a, b) => String(a.type).localeCompare(String(b.type)) || String(a.name).localeCompare(String(b.name)));
for (const r of out) console.log(`${r.type}\t${r.identifier}\t${r.name}`);
console.log('TOTAL', out.length);
