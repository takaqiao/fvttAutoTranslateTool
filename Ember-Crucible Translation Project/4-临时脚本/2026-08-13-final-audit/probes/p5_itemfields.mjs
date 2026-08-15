// Probe: string leaves under `system.*` on Item / Actor / ActiveEffect docs that
// the project's mapping layer does NOT reach. Same blind-spot class as
// Scene.levels: if the mapping never names the path, neither the extractor nor
// the runtime ever sees it.
import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';
const { ClassicLevel } = createRequire('C:/Users/Taka/Desktop/fvtt/package.json')('classic-level');

// paths the project mapping already reaches (normalized, [] for array index)
const REACHED = new Set([
  'system.description', 'system.description.public', 'system.description.private',
  'system.adjective',
  'system.actions[].name', 'system.actions[].description', 'system.actions[].condition',
  'system.actions[].effects[].name',
  'system.details.biography.name','system.details.biography.description','system.details.biography.public','system.details.biography.private','system.details.biography.appearance',
  'system.details.ancestry.name','system.details.ancestry.description','system.details.ancestry.public','system.details.ancestry.private','system.details.ancestry.appearance',
  'system.details.background.name','system.details.background.description','system.details.background.public','system.details.background.private','system.details.background.appearance',
  'system.details.archetype.name','system.details.archetype.description','system.details.archetype.public','system.details.archetype.private','system.details.archetype.appearance',
  'system.details.taxonomy.name','system.details.taxonomy.description','system.details.taxonomy.public','system.details.taxonomy.private','system.details.taxonomy.appearance',
]);

const looksLikeProse = (s) =>
  /[a-z]/.test(s) && /\s/.test(s) && s.length > 6 &&
  !/^[a-z0-9._/\-]+$/i.test(s) && !s.startsWith('modules/') && !s.startsWith('systems/') &&
  !/^Compendium\./.test(s) && !/^(Actor|Item|JournalEntry)\./.test(s);

const PKGS = [
  ['ember',    'C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember',   'module.json'],
  ['crucible', 'C:/Users/Taka/AppData/Local/FoundryVTT/Data/systems/crucible','system.json'],
];
const SKIP_PACKS = new Set(['ember.dnd5e-effects', 'ember.dnd5e-items', 'ember.adventure']);

const acc = new Map(); // path -> {n, prose, sample, packs:Set}
function scan(obj, prefix, packId, depth = 0) {
  if (obj === null || obj === undefined || depth > 5) return;
  if (typeof obj === 'string') {
    if (!obj.trim()) return;
    const e = acc.get(prefix) ?? { n: 0, prose: 0, sample: null, packs: new Set() };
    e.n += 1; e.packs.add(packId);
    if (looksLikeProse(obj)) { e.prose += 1; if (!e.sample) e.sample = obj.slice(0, 100); }
    acc.set(prefix, e); return;
  }
  if (Array.isArray(obj)) { for (const v of obj) scan(v, `${prefix}[]`, packId, depth + 1); return; }
  if (typeof obj === 'object') for (const [k, v] of Object.entries(obj)) scan(v, `${prefix}.${k}`, packId, depth + 1);
}

const DOC_BUCKETS = new Set(['items', 'items.effects', 'actors', 'actors.items', 'actors.effects', 'effects']);
for (const [id, dir, mf] of PKGS) {
  const manifest = JSON.parse(fs.readFileSync(path.join(dir, mf), 'utf8'));
  for (const p of manifest.packs ?? []) {
    const packId = `${id}.${p.name}`;
    if (SKIP_PACKS.has(packId)) continue;
    const packDir = path.join(dir, p.path ?? `packs/${p.name}`);
    if (!fs.existsSync(packDir)) continue;
    const db = new ClassicLevel(packDir, { createIfMissing: false });
    for await (const [k, v] of db.iterator()) {
      const m = k.toString().match(/^!([^!]+)!/); if (!m) continue;
      let d; try { d = JSON.parse(v.toString()); } catch { continue; }
      if (DOC_BUCKETS.has(m[1])) scan(d.system ?? {}, 'system', packId);
      if (m[1] === 'adventures') {
        for (const a of d.actors ?? []) { scan(a.system ?? {}, 'system', packId); for (const it of a.items ?? []) scan(it.system ?? {}, 'system', packId); for (const ef of a.effects ?? []) scan(ef.system ?? {}, 'system', packId); }
        for (const it of d.items ?? []) scan(it.system ?? {}, 'system', packId);
      }
    }
    await db.close();
  }
}

const rows = [...acc.entries()].filter(([p, e]) => e.prose > 0 && !REACHED.has(p)).sort((a, b) => b[1].prose - a[1].prose);
console.log('UNREACHED system.* paths carrying prose-looking strings:');
for (const [p, e] of rows) console.log(`  prose=${String(e.prose).padStart(5)} total=${String(e.n).padStart(5)}  ${p}\n        ${JSON.stringify(e.sample)}  packs=[${[...e.packs].join(',')}]`);
console.log(`\n(total distinct system.* string paths seen: ${acc.size})`);
