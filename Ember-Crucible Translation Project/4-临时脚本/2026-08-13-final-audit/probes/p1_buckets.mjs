// Probe: list LevelDB bucket prefixes per pack, so we can see which embedded
// sibling buckets exist that extract_en.mjs's attachEmbedded() does NOT re-attach.
import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';
const { ClassicLevel } = createRequire('C:/Users/Taka/Desktop/fvtt/package.json')('classic-level');

const PKGS = [
  ['ember',    'C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember',   'module.json'],
  ['crucible', 'C:/Users/Taka/AppData/Local/FoundryVTT/Data/systems/crucible','system.json'],
];

for (const [id, dir, mf] of PKGS) {
  const manifest = JSON.parse(fs.readFileSync(path.join(dir, mf), 'utf8'));
  for (const p of manifest.packs ?? []) {
    const packDir = path.join(dir, p.path ?? `packs/${p.name}`);
    if (!fs.existsSync(packDir)) { console.log(`${id}.${p.name}: MISSING`); continue; }
    const db = new ClassicLevel(packDir, { createIfMissing: false });
    const counts = {};
    for await (const [k] of db.iterator({ values: false })) {
      const m = k.toString().match(/^!([^!]+)!/);
      if (m) counts[m[1]] = (counts[m[1]] ?? 0) + 1;
    }
    await db.close();
    console.log(`${id}.${p.name} [${p.type}]: ` + Object.entries(counts).map(([k,v])=>`${k}=${v}`).join(' '));
  }
}
