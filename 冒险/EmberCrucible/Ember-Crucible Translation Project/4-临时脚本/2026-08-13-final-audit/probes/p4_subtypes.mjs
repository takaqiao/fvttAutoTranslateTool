// Probe: every JournalEntryPage `type` present in the shipped packs, and which
// of them the project's EMBER_PAGE_MAPPINGS covers. Any uncovered subtype falls
// back to Babele's base JournalEntryPage mapping (name/caption/text only), so
// prose living in `system.*` would be invisible to BOTH extractor and runtime.
import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';
const { ClassicLevel } = createRequire('C:/Users/Taka/Desktop/fvtt/package.json')('classic-level');
const { EMBER_PAGE_MAPPINGS } = await import('file:///C:/Users/Taka/Desktop/fvtt/Ember-Crucible%20Translation%20Project/3-%E5%B8%B8%E7%94%A8%E8%84%9A%E6%9C%AC/extract/mappings.mjs');

const covered = new Set(Object.keys(EMBER_PAGE_MAPPINGS).map((k) => k.replace(/^JournalEntryPage\./, '')));

const PKGS = [
  ['ember',    'C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember',   'module.json'],
  ['crucible', 'C:/Users/Taka/AppData/Local/FoundryVTT/Data/systems/crucible','system.json'],
];

const seen = {};   // type -> {count, systemStringPaths: Map<path, {n, sample}>}
function scanStrings(obj, prefix, sink, depth = 0) {
  if (obj === null || obj === undefined || depth > 4) return;
  if (typeof obj === 'string') {
    if (obj.trim().length > 1) {
      const e = sink.get(prefix) ?? { n: 0, sample: obj.slice(0, 90) };
      e.n += 1; sink.set(prefix, e);
    }
    return;
  }
  if (Array.isArray(obj)) { obj.forEach((v, i) => scanStrings(v, `${prefix}[]`, sink, depth + 1)); return; }
  if (typeof obj === 'object') { for (const [k, v] of Object.entries(obj)) scanStrings(v, prefix ? `${prefix}.${k}` : k, sink, depth + 1); }
}

function note(pg) {
  const t = pg.type ?? '(none)';
  const e = seen[t] ??= { count: 0, sink: new Map() };
  e.count += 1;
  scanStrings(pg.system ?? {}, 'system', e.sink);
}

for (const [id, dir, mf] of PKGS) {
  const manifest = JSON.parse(fs.readFileSync(path.join(dir, mf), 'utf8'));
  for (const p of manifest.packs ?? []) {
    const packDir = path.join(dir, p.path ?? `packs/${p.name}`);
    if (!fs.existsSync(packDir)) continue;
    const db = new ClassicLevel(packDir, { createIfMissing: false });
    for await (const [k, v] of db.iterator()) {
      const m = k.toString().match(/^!([^!]+)!/); if (!m) continue;
      let d; try { d = JSON.parse(v.toString()); } catch { continue; }
      if (m[1] === 'journal.pages') note(d);
      if (m[1] === 'adventures') for (const j of d.journal ?? []) for (const pg of j.pages ?? []) note(pg);
    }
    await db.close();
  }
}

for (const [t, e] of Object.entries(seen).sort((a, b) => b[1].count - a[1].count)) {
  const ok = covered.has(t) ? 'MAPPED  ' : 'UNMAPPED';
  console.log(`\n${ok} ${t}  (${e.count} pages)`);
  const paths = [...e.sink.entries()].sort((a, b) => b[1].n - a[1].n);
  for (const [p, s] of paths) console.log(`    ${String(s.n).padStart(5)}  ${p}   ${JSON.stringify(s.sample)}`);
}
console.log('\ncovered subtype keys never seen in data:', [...covered].filter((c) => !(c in seen)));
