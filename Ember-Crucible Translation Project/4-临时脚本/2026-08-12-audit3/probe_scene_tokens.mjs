/**
 * H2-A probe: how much player-visible text lives in scene fields that neither
 * babele's default Scene mapping nor this project's layer extracts?
 *
 * Crucible-FR's `ember_scene_levels_converter` translates scene `levels[].name`
 * and `tokens[].name` (their "deltaTokens"); babele 2.9.1's default Scene
 * mapping has only name/drawings/notes/regions, and this project follows the
 * default. So neither we nor babele touch those two fields.
 *
 * Usage: node probe_scene_tokens.mjs
 */
import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';

const FVTT_NODE_ANCHOR = 'C:/Users/Taka/Desktop/fvtt/package.json';
const { ClassicLevel } = createRequire(FVTT_NODE_ANCHOR)('classic-level');

const TARGETS = [
  ['crucible', 'C:/Users/Taka/AppData/Local/FoundryVTT/Data/systems/crucible'],
  ['ember', 'C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember'],
];

const CJK = /[\u4e00-\u9fff]/;

async function readPack(packDir) {
  const db = new ClassicLevel(packDir, { createIfMissing: false });
  const buckets = {};
  for await (const [k, v] of db.iterator()) {
    const m = k.toString().match(/^!([^!]+)!(.+)$/);
    if (!m) continue;
    let doc;
    try { doc = JSON.parse(v.toString()); } catch { continue; }
    (buckets[m[1]] ||= []).push({ idPart: m[2], doc });
  }
  await db.close();
  return buckets;
}

const out = { scenes: [], totals: {} };
let totTokens = 0, totLevels = 0, totScenes = 0, totNav = 0;

for (const [label, dir] of TARGETS) {
  const manifest = JSON.parse(fs.readFileSync(
    path.join(dir, fs.existsSync(path.join(dir, 'system.json')) ? 'system.json' : 'module.json'), 'utf8'));
  for (const pack of manifest.packs ?? []) {
    const packDir = path.join(dir, 'packs', path.basename(pack.path ?? pack.name));
    if (!fs.existsSync(packDir)) continue;
    let buckets;
    try { buckets = await readPack(packDir); } catch (e) { continue; }

    // scenes can be standalone or nested inside an Adventure
    const scenes = [];
    for (const { doc } of (buckets.scenes || [])) scenes.push([pack.name, null, doc]);
    for (const { doc } of (buckets.adventures || [])) {
      for (const s of (doc.scenes || [])) scenes.push([pack.name, doc.name, s]);
    }
    for (const [packName, advName, s] of scenes) {
      totScenes++;
      const tokens = (s.tokens || []).map(t => t.name).filter(n => typeof n === 'string' && n.trim());
      const levels = (s.levels || []).map(l => l.name).filter(n => typeof n === 'string' && n.trim());
      const nav = (typeof s.navName === 'string' && s.navName.trim()) ? s.navName : null;
      if (nav) totNav++;
      totTokens += tokens.length;
      totLevels += levels.length;
      if (tokens.length || levels.length || nav) {
        out.scenes.push({
          module: label, pack: packName, adventure: advName, scene: s.name,
          n_tokens: tokens.length, n_levels: levels.length, navName: nav,
          token_names: [...new Set(tokens)].slice(0, 30),
          level_names: [...new Set(levels)].slice(0, 30),
        });
      }
    }
  }
}

out.totals = { scenes: totScenes, token_names: totTokens, level_names: totLevels, navNames: totNav };
console.log(JSON.stringify(out.totals));
const uniqTok = new Set(), uniqLvl = new Set();
for (const s of out.scenes) { s.token_names.forEach(n => uniqTok.add(n)); s.level_names.forEach(n => uniqLvl.add(n)); }
console.log('unique token names:', uniqTok.size, ' unique level names:', uniqLvl.size);
console.log('sample token names:', [...uniqTok].slice(0, 40).join(' | '));
console.log('sample level names:', [...uniqLvl].slice(0, 40).join(' | '));
fs.writeFileSync(process.argv[2] || 'scene_probe.json', JSON.stringify(out, null, 1), 'utf8');
console.log('wrote', process.argv[2]);
