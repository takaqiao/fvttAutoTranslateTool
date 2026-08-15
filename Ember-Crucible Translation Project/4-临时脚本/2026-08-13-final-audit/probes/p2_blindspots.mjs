// Probe: hunt for translatable fields that Babele 2.9.1's REAL defaults would
// look up but this project's BABELE_DEFAULTS mirror (mappings.mjs) omits.
import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';
const { ClassicLevel } = createRequire('C:/Users/Taka/Desktop/fvtt/package.json')('classic-level');

const PKGS = [
  ['ember',    'C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember',   'module.json'],
  ['crucible', 'C:/Users/Taka/AppData/Local/FoundryVTT/Data/systems/crucible','system.json'],
];

const out = {
  regionBehaviors: [],   // {pack, scene, type, text/dialog}
  macroCommands: [],
  cards: [],
  tableResultsWithUuid: [],
  tableResults: [],
  sceneTokens: [],
  sceneNav: [],
  sceneLevels: [],
  jepSrcWidthHeight: [],
  actorTokenName: [],
  playlistSounds: [],
};

function walkScene(pack, scene) {
  for (const r of scene.regions ?? []) {
    for (const b of r.behaviors ?? []) {
      out.regionBehaviors.push({ pack, scene: scene.name, region: r.name, type: b.type,
        text: b?.system?.text ?? null,
        revealed: b?.system?.dialog?.revealed ?? null,
        unrevealed: b?.system?.dialog?.unrevealed ?? null });
    }
  }
  for (const t of scene.tokens ?? []) out.sceneTokens.push({ pack, scene: scene.name, name: t.name });
  if (scene.navName) out.sceneNav.push({ pack, scene: scene.name, navName: scene.navName });
  for (const l of scene.levels ?? []) out.sceneLevels.push({ pack, scene: scene.name, name: l.name });
}

function walkTable(pack, tbl) {
  for (const r of tbl.results ?? []) {
    out.tableResults.push({ pack, table: tbl.name, name: r.name, range: r.range, docUuid: r.documentUuid ?? null, desc: r.description ?? null });
    if (r.documentUuid) out.tableResultsWithUuid.push({ pack, table: tbl.name, name: r.name, uuid: r.documentUuid });
  }
}

function walkActor(pack, a) {
  const tn = a?.prototypeToken?.name ?? null;
  out.actorTokenName.push({ pack, name: a.name, tokenName: tn, same: tn === a.name });
}

for (const [id, dir, mf] of PKGS) {
  const manifest = JSON.parse(fs.readFileSync(path.join(dir, mf), 'utf8'));
  for (const p of manifest.packs ?? []) {
    const packDir = path.join(dir, p.path ?? `packs/${p.name}`);
    if (!fs.existsSync(packDir)) continue;
    const packId = `${id}.${p.name}`;
    const db = new ClassicLevel(packDir, { createIfMissing: false });
    const buckets = {};
    for await (const [k, v] of db.iterator()) {
      const m = k.toString().match(/^!([^!]+)!(.+)$/);
      if (!m) continue;
      let d; try { d = JSON.parse(v.toString()); } catch { continue; }
      (buckets[m[1]] ||= []).push(d);
    }
    await db.close();

    for (const adv of buckets.adventures ?? []) {
      for (const s of adv.scenes ?? []) walkScene(packId, s);
      for (const t of adv.tables ?? []) walkTable(packId, t);
      for (const a of adv.actors ?? []) walkActor(packId, a);
      for (const m of adv.macros ?? []) out.macroCommands.push({ pack: packId, name: m.name, cmdLen: (m.command ?? '').length });
      for (const c of adv.cards ?? []) out.cards.push({ pack: packId, name: c.name });
      for (const pl of adv.playlists ?? []) for (const s of pl.sounds ?? []) out.playlistSounds.push({ pack: packId, playlist: pl.name, name: s.name, desc: s.description ?? null });
      for (const j of adv.journal ?? []) for (const pg of j.pages ?? []) {
        if (pg.src || pg?.video?.width || pg?.video?.height) out.jepSrcWidthHeight.push({ pack: packId, journal: j.name, page: pg.name, src: pg.src ?? null, w: pg?.video?.width ?? null, h: pg?.video?.height ?? null });
      }
    }
    for (const a of buckets.actors ?? []) walkActor(packId, a);
    for (const m of buckets.macros ?? []) out.macroCommands.push({ pack: packId, name: m.name, cmdLen: (m.command ?? '').length });
    for (const pg of buckets['journal.pages'] ?? []) {
      if (pg.src || pg?.video?.width || pg?.video?.height) out.jepSrcWidthHeight.push({ pack: packId, page: pg.name, src: pg.src ?? null, w: pg?.video?.width ?? null, h: pg?.video?.height ?? null });
    }
  }
}

const S = (k) => `${k}: ${out[k].length}`;
console.log(Object.keys(out).map(S).join('\n'));
fs.writeFileSync(process.argv[2] ?? 'p2.json', JSON.stringify(out, null, 1));
