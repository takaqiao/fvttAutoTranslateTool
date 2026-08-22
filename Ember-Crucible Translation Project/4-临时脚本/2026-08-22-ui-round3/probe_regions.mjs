/**
 * Probe: enumerate every RegionBehavior in upstream ember 0.6.1's two adventure
 * packs, grouped by `type`, and report which `system.*` string leaves are
 * non-empty.  Also enumerate Macro.command and Scene.drawings.
 *
 * SELF-PROOF (constraint 6) - TWO INDEPENDENT assertions, both must hold and
 * both are printed.  Assertion A (count) is explicitly NOT enough on its own:
 * it only proves we sliced the right NUMBER of things, not that we read the
 * right OBJECTS.  Hence B.
 *
 *   A. COUNT, cross-checked by a SECOND, structurally different route:
 *      the tree-walk count of behaviours must equal the number of behaviour
 *      `_id`s found by a flat bracket-scan of the serialised pack text, and
 *      also equal the number of DISTINCT behaviour ids.
 *      (Using our own EN baseline as the truth for A was tried first and
 *      FAILED - 438 vs 445 - which is itself a finding, reported separately.)
 *   B. IDENTITY: three hand-known behaviours must be found at the exact
 *      scene/region/name coordinates with the exact expected string value.
 */
import { createRequire } from 'module';
import fs from 'fs';
import path from 'path';

const require = createRequire('C:/Users/Taka/Desktop/fvtt/package.json');
const { ClassicLevel } = require('classic-level');

const PKG = 'C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember';
const PACKS = ['adventure', 'crucible-adventure'];

/** Hand-verified coordinates (read out of compendium/en, then located upstream). */
const IDENTITY = [
  ['The Bleak Archive', 'West Blade Trap Trigger', 'West Blade Trap Pressure Plate', 'system.message', 'Trap Triggered!'],
  ['Bastion Apex', 'Searing Light of Lantyr', 'Scrolling Text', 'system.text', 'Searing Light!'],
  ['Kadra Zann', 'Transition: Vespiary Hole', 'Teleport Token', 'system.dialog.revealed', 'Descend into tunnels?'],
];

/** For comparison only - NOT used as the A assertion (see header). */
const EN_BASELINE = { scenes: 99, regions: 378, behaviors: 438 };

const get = (o, p) => p.split('.').reduce((a, k) => (a == null ? a : a[k]), o);

async function readPack(name) {
  const db = new ClassicLevel(path.join(PKG, 'packs', name), { valueEncoding: 'json' });
  await db.open();
  const out = [];
  for await (const [k, v] of db.iterator()) out.push([k, v]);
  await db.close();
  return out;
}

function walkScenes(doc) {
  if (Array.isArray(doc?.scenes)) return doc.scenes;
  if (doc?.regions !== undefined && doc?.grid !== undefined) return [doc];
  return [];
}

const report = { packs: {}, selfProof: {}, enBaselineForComparison: EN_BASELINE };
let allOk = true;

for (const pack of PACKS) {
  const rows = await readPack(pack);
  const scenes = [];
  const macros = [];
  for (const [, v] of rows) {
    for (const s of walkScenes(v)) scenes.push(s);
    if (Array.isArray(v?.macros)) macros.push(...v.macros);
    if (v?.command !== undefined && v?.scope !== undefined) macros.push(v);
  }

  let regionN = 0, regionNamed = 0;
  const behaviors = [];
  const behIds = new Set();
  let drawingsN = 0, drawingsText = 0;
  const drawingSamples = [];
  for (const s of scenes) {
    for (const d of (s.drawings ?? [])) {
      drawingsN++;
      const t = d?.text;
      if (typeof t === 'string' && t.trim()) { drawingsText++; drawingSamples.push(t); }
    }
    for (const r of (s.regions ?? [])) {
      regionN++;
      if (typeof r.name === 'string' && r.name.trim()) regionNamed++;
      for (const b of (r.behaviors ?? [])) {
        behaviors.push({ scene: s.name, region: r.name, b });
        if (b._id) behIds.add(`${s._id}/${r._id}/${b._id}`);
      }
    }
  }
  const behNamed = behaviors.filter((x) => typeof x.b.name === 'string' && x.b.name.trim()).length;

  /* Why the EN baseline shows fewer: extract_en.mjs COLLAPSES same-key docs
   * whose extracted content is identical. Recompute that collapse here. */
  let regionKeys = 0, behaviorKeys = 0;
  for (const s of scenes) {
    regionKeys += new Set((s.regions ?? []).map((r) => r.name)).size;
    for (const r of (s.regions ?? [])) behaviorKeys += new Set((r.behaviors ?? []).map((b) => b.name)).size;
  }

  /* SECOND, structurally different route for assertion A: flat text scan. */
  const raw = JSON.stringify(rows.map(([, v]) => v));
  let flatBehIds = 0;
  const reBeh = /"behaviors":\[/g;
  let m;
  while ((m = reBeh.exec(raw))) {
    let i = m.index + m[0].length - 1, depth = 0;
    const start = i;
    for (; i < raw.length; i++) {
      const c = raw[i];
      if (c === '"') { i++; while (i < raw.length && !(raw[i] === '"' && raw[i - 1] !== '\\')) i++; continue; }
      if (c === '[' || c === '{') depth++;
      else if (c === ']' || c === '}') { depth--; if (depth === 0) break; }
    }
    const chunk = raw.slice(start, i + 1);
    // Count TOP-LEVEL OBJECT elements of the located array only.
    // Two refinements were forced by measurement, both recorded here because
    // each was a wrong answer this probe printed before it printed a right one:
    //   1. /"_id":"…"/g over the chunk overcounts (nested objects carry ids).
    //   2. `ember.trapTrigger` has its OWN `system.behaviors`, an array of
    //      UUID *strings* (the behaviours it switches on). Counting array
    //      length blind gives 447 vs 445 - the 2 extras are those two arrays.
    flatBehIds += JSON.parse(chunk).filter((x) => x && typeof x === 'object').length;
  }

  const A_ok = flatBehIds === behaviors.length && behIds.size === behaviors.length;

  const idResults = IDENTITY.map(([sc, rg, nm, p, val]) => {
    const hit = behaviors.find((x) => x.scene === sc && x.region === rg && x.b.name === nm);
    return {
      scene: sc, region: rg, name: nm, path: p, expected: val,
      found: hit ? get(hit.b, p) : null,
      ok: !!hit && get(hit.b, p) === val,
    };
  });
  const B_ok = idResults.every((r) => r.ok);
  if (!A_ok || !B_ok) allOk = false;

  const byType = {};
  for (const { scene, region, b } of behaviors) {
    const t = b.type ?? '<none>';
    byType[t] ??= { count: 0, fields: {} };
    byType[t].count++;
    const seen = [];
    (function rec(node, prefix) {
      if (node == null) return;
      if (typeof node === 'string') { if (node.trim()) seen.push([prefix, node]); return; }
      if (Array.isArray(node)) { node.forEach((x, i) => rec(x, `${prefix}[${i}]`)); return; }
      if (typeof node === 'object') for (const [k, v2] of Object.entries(node)) rec(v2, prefix ? `${prefix}.${k}` : k);
    })(b.system ?? {}, 'system');
    for (const [p, v2] of seen) {
      const norm = p.replace(/\[\d+\]/g, '[]');
      byType[t].fields[norm] ??= { n: 0, samples: [] };
      byType[t].fields[norm].n++;
      if (byType[t].fields[norm].samples.length < 4) {
        byType[t].fields[norm].samples.push({ scene, region, name: b.name, value: v2.slice(0, 260) });
      }
    }
  }

  report.selfProof[pack] = {
    A_treewalk_behaviors: behaviors.length,
    A_flatscan_behaviors: flatBehIds,
    A_distinct_behavior_ids: behIds.size,
    A_ok,
    B_identity: idResults,
    B_ok,
  };
  report.packs[pack] = {
    scenes: scenes.length, regions: regionN, regionsNamed: regionNamed,
    behaviors: behaviors.length, behaviorsNamed: behNamed,
    regionKeysAfterCollapse: regionKeys, behaviorKeysAfterCollapse: behaviorKeys,
    drawings: drawingsN, drawingsWithText: drawingsText, drawingSamples: drawingSamples.slice(0, 5),
    macros: macros.length,
    macroCommands: macros.map((mm) => ({ name: mm.name, command: mm.command })),
    byType,
  };
}

const OUT = 'C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/4-临时脚本/2026-08-22-ui-round3/probe_regions.json';
fs.writeFileSync(OUT, JSON.stringify(report, null, 2), 'utf8');
console.log(JSON.stringify(report.selfProof, null, 2));
for (const [p, v] of Object.entries(report.packs)) {
  console.log(`[${p}] scenes=${v.scenes} regions=${v.regions}(named ${v.regionsNamed}) behaviors=${v.behaviors}(named ${v.behaviorsNamed}) collapsedKeys r=${v.regionKeysAfterCollapse}/b=${v.behaviorKeysAfterCollapse} drawings=${v.drawings}(withText ${v.drawingsWithText}) macros=${v.macros}`);
  console.log('   types:', Object.entries(v.byType).map(([t, o]) => `${t}=${o.count}`).join(' '));
}
if (!allOk) { console.error('SELF-PROOF FAILED'); process.exit(2); }
console.log('SELF-PROOF OK (A: two independent counts agree; B: 3 identity objects read correctly) ->', OUT);
