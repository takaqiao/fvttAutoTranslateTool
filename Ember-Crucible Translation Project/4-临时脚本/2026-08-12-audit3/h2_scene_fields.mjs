/**
 * H2-A: quantify the Scene sub-fields that Crucible-FR's
 * `ember_scene_levels_converter` translates and our Scene mapping does not:
 *   scene.navName, scene.levels[].name, scene.tokens[].name (delta tokens),
 *   scene.regions[].name / regions[].behaviors[].name (we DO have these).
 *
 * Usage: node h2_scene_fields.mjs --package <dir> [--pack <name>]
 */
import path from 'path';
import fs from 'fs';
import { createRequire } from 'module';

const { ClassicLevel } = createRequire('C:/Users/Taka/Desktop/fvtt/package.json')('classic-level');

const argv = process.argv.slice(2);
const arg = (n, d) => { const i = argv.indexOf(n); return i >= 0 ? argv[i + 1] : d; };
const PACKAGE_DIR = arg('--package');
const ONLY = arg('--pack');

const packsDir = path.join(PACKAGE_DIR, 'packs');
const packs = fs.readdirSync(packsDir).filter(p => !ONLY || p === ONLY);

const tally = { navName: [], levels: [], tokens: [], regions: [], behaviors: [], notes: [] };

function scanScene(scene, where) {
  if (scene.navName && scene.navName.trim() && scene.navName !== scene.name) {
    tally.navName.push([where, scene.name, scene.navName]);
  }
  for (const l of scene.levels ?? []) {
    if (l?.name?.trim()) tally.levels.push([where, scene.name, l.name]);
  }
  for (const t of scene.tokens ?? []) {
    if (t?.name?.trim()) tally.tokens.push([where, scene.name, t.name, !!t.actorLink]);
  }
  for (const r of scene.regions ?? []) {
    if (r?.name?.trim()) tally.regions.push([where, scene.name, r.name]);
    for (const b of r.behaviors ?? []) {
      if (b?.name?.trim()) tally.behaviors.push([where, scene.name, b.name]);
    }
  }
  for (const n of scene.notes ?? []) {
    if (n?.text?.trim()) tally.notes.push([where, scene.name, n.text]);
  }
}

for (const p of packs) {
  const dir = path.join(packsDir, p);
  if (!fs.statSync(dir).isDirectory()) continue;
  const db = new ClassicLevel(dir, { keyEncoding: 'utf8', valueEncoding: 'json' });
  try { await db.open(); } catch { continue; }
  for await (const [key, doc] of db.iterator()) {
    if (key.startsWith('!adventures!')) {
      for (const s of doc.scenes ?? []) scanScene(s, `${p}/${doc.name}`);
    } else if (key.startsWith('!scenes!')) {
      scanScene(doc, `${p}`);
    }
  }
  await db.close();
}

for (const [k, v] of Object.entries(tally)) {
  const uniq = new Set(v.map(x => x[2]));
  console.log(`${k}: ${v.length} occurrences / ${uniq.size} unique`);
  for (const row of v.slice(0, 12)) console.log('   ', JSON.stringify(row));
}
fs.writeFileSync(process.env.H2_OUT || 'h2_scene_fields.json',
  JSON.stringify(tally, null, 1), 'utf8');
