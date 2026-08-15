/**
 * H2-A probe: why does Crucible-FR extract `actions.*.effects.*.name` where we
 * extract nothing?  Dump the raw `system.actions[].effects[]` shape from the
 * source packs for a few of the disputed actions.
 *
 * Usage: node probe_action_effects.mjs [outfile]
 */
import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';

const { ClassicLevel } = createRequire('C:/Users/Taka/Desktop/fvtt/package.json')('classic-level');

const SYS = 'C:/Users/Taka/AppData/Local/FoundryVTT/Data/systems/crucible';

async function readPack(packDir) {
  const db = new ClassicLevel(packDir, { createIfMissing: false });
  const buckets = {};
  for await (const [k, v] of db.iterator()) {
    const m = k.toString().match(/^!([^!]+)!(.+)$/);
    if (!m) continue;
    let doc; try { doc = JSON.parse(v.toString()); } catch { continue; }
    (buckets[m[1]] ||= []).push({ idPart: m[2], doc });
  }
  await db.close();
  return buckets;
}

const WANT = new Set(['Burnout', 'Acid Spit', 'Dropping Strike', 'Adrenaline Surge', 'Grave Mark']);
const out = [];
const shapes = {};
let nActions = 0, nWithEffects = 0, nEffWithName = 0, nEffWithoutName = 0;

const manifest = JSON.parse(fs.readFileSync(path.join(SYS, 'system.json'), 'utf8'));
for (const pack of manifest.packs ?? []) {
  const dir = path.join(SYS, 'packs', path.basename(pack.path ?? pack.name));
  if (!fs.existsSync(dir)) continue;
  let buckets; try { buckets = await readPack(dir); } catch { continue; }
  const docs = [];
  for (const b of Object.values(buckets)) for (const { doc } of b) docs.push(doc);
  const visit = (o) => {
    if (Array.isArray(o)) return o.forEach(visit);
    if (!o || typeof o !== 'object') return;
    const acts = o?.system?.actions;
    if (Array.isArray(acts)) {
      for (const a of acts) {
        nActions++;
        if (!Array.isArray(a.effects) || !a.effects.length) continue;
        nWithEffects++;
        for (const e of a.effects) {
          const keys = Object.keys(e).sort().join(',');
          shapes[keys] = (shapes[keys] || 0) + 1;
          if (typeof e.name === 'string' && e.name.trim()) nEffWithName++;
          else nEffWithoutName++;
        }
        if (WANT.has(a.name)) {
          out.push({ pack: pack.name, doc: o.name, action: a.name, actionId: a.id,
                     effects: a.effects });
        }
      }
    }
    for (const v of Object.values(o)) if (v && typeof v === 'object') visit(v);
  };
  visit(docs);
}

console.log('actions', nActions, 'with effects', nWithEffects,
            'effect entries with name', nEffWithName, 'without name', nEffWithoutName);
console.log('effect-object key shapes:', JSON.stringify(shapes, null, 1));
console.log(JSON.stringify(out.slice(0, 8), null, 1));
if (process.argv[2]) fs.writeFileSync(process.argv[2], JSON.stringify({ shapes, samples: out }, null, 1), 'utf8');
