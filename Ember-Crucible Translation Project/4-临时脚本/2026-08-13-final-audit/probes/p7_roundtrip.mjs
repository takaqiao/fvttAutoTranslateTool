/**
 * Round-trip probe (READ-ONLY).
 *
 * For every real document in the shipped ember 0.6.0 / crucible 0.10.1 packs:
 *   1. run extract_en.mjs's own extractDocument() to get the Babele entry
 *   2. mark every string leaf of that entry with U+2588
 *   3. feed the entry to the REAL babele 2.9.1 runtime (MappedCompendium.translate)
 *      configured with the repo's generated DOCUMENT_MAPPINGS + PROJECT_CONVERTERS
 *   4. count how many marks actually reached the translated document
 *
 * A leaf that the extractor emits but the runtime never consumes is a key a
 * translator can fill in for nothing.
 *
 * Known false-positive modes:
 *  - `document`-converter fields (items/effects/pages/...) rely on runtime pack
 *    visibility; here there is no runtime registry, so embedded fallback cannot
 *    fire. Their LOCAL payload should still land, which is what is measured.
 *  - foundry.utils.mergeObject is re-implemented (defaults only).
 */
import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';

const req = createRequire('C:/Users/Taka/Desktop/fvtt/package.json');
const { ClassicLevel } = req('classic-level');

/* ------------------------- foundry shim ------------------------- */
const isPlain = (v) => !!v && typeof v === 'object' && !Array.isArray(v);
function mergeObject(original, other = {}, opts = {}) {
  const inplace = opts.inplace !== false;
  const target = inplace ? original : structuredClone(original);
  for (const [k, v] of Object.entries(other ?? {})) {
    if (isPlain(target[k]) && isPlain(v)) mergeObject(target[k], v, { inplace: true });
    else target[k] = (isPlain(v) || Array.isArray(v)) ? structuredClone(v) : v;
  }
  return target;
}
globalThis.foundry = {
  utils: {
    mergeObject,
    getProperty: (o, p) => String(p).split('.').reduce((a, k) => (a == null ? a : a[k]), o),
    deepClone: (v) => (v == null ? v : structuredClone(v)),
    parseUuid: () => null,
    Collection: Map,
  },
};
globalThis.Hooks = { callAll() {}, on() {}, once() {} };
globalThis.game = { settings: { get: () => false } };
globalThis.CONFIG = { debug: {} };

const BB = 'file:///C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/babele/script';
const { DocumentMappings } = await import(BB + '/mapping/document-mappings.js');
const { ConverterRegistry } = await import(BB + '/converter/converter-registry.js');
const { IdentityExtractorRegistry } = await import(BB + '/identity/identity-extractor-registry.js');
const { Converters } = await import(BB + '/converter/converters.js');
const { DocumentConverter } = await import(BB + '/converter/document-converter.js');
const { StructuredDataConverter } = await import(BB + '/converter/structured-data-converter.js');
const { ReferencedDocumentFieldConverter } = await import(BB + '/converter/referenced-document-field-converter.js');
const { MappedCompendium } = await import(BB + '/compendium/mapped-compendium.js');

const PROJ = 'file:///C:/Users/Taka/Desktop/fvtt/Ember-Crucible%20Translation%20Project';
const emberRuntime = await import(PROJ + '/1-Ember%E6%B1%89%E5%8C%96%E6%8F%92%E4%BB%B6/babele-mappings.js');
const crucRuntime = await import(PROJ + '/2-Crucible%E6%B1%89%E5%8C%96%E6%8F%92%E4%BB%B6/babele-mappings.js');
const { effectiveMappings } = await import(PROJ + '/3-%E5%B8%B8%E7%94%A8%E8%84%9A%E6%9C%AC/extract/mappings.mjs');

/* extract_en.mjs runs main() on import, so lift its pure section instead */
const EXTRACT_FILE = 'C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/3-常用脚本/extract/extract_en.mjs';
const src = fs.readFileSync(EXTRACT_FILE, 'utf8');
const slice = src.slice(src.indexOf('const getPath ='), src.indexOf('/* --------------------------- pack reading'));
const { extractDocument } = await import(
  'data:text/javascript,' + encodeURIComponent(slice + '\nexport {extractDocument};')
);

function makeRuntime(layer, converters) {
  const converterRegistry = new ConverterRegistry({
    ...Converters.legacyRegistrations(),
    document: new DocumentConverter(),
    structured: new StructuredDataConverter(),
    referencedDocumentField: new ReferencedDocumentFieldConverter(),
    ...converters,
  });
  const identityExtractors = new IdentityExtractorRegistry({
    range: (d) => {
      const [s, e] = d?.range ?? [];
      return Number.isInteger(s) && Number.isInteger(e) ? s + '-' + e : null;
    },
  });
  const documentMappings = new DocumentMappings(undefined, {
    registeredMappings: [layer],
    identityExtractors,
    converterRegistry,
  });
  return { documentMappings, converterRegistry };
}

const MARK = '\u2588';
const mark = (v) => (typeof v === 'string' ? MARK + v
  : Array.isArray(v) ? v.map(mark)
  : isPlain(v) ? Object.fromEntries(Object.entries(v).map(([k, x]) => [k, mark(x)]))
  : v);
const nLeaves = (v) => (typeof v === 'string' ? 1
  : Array.isArray(v) ? v.reduce((n, x) => n + nLeaves(x), 0)
  : isPlain(v) ? Object.values(v).reduce((n, x) => n + nLeaves(x), 0) : 0);
const nMarks = (v) => (typeof v === 'string' ? (v.includes(MARK) ? 1 : 0)
  : Array.isArray(v) ? v.reduce((n, x) => n + nMarks(x), 0)
  : isPlain(v) ? Object.values(v).reduce((n, x) => n + nMarks(x), 0) : 0);

const BUCKET = {
  Item: 'items', Actor: 'actors', JournalEntry: 'journal', Adventure: 'adventures',
  ActiveEffect: 'effects', Macro: 'macros', RollTable: 'tables', Scene: 'scenes', Playlist: 'playlists',
};

const CASES = [
  ['crucible', crucRuntime, 'C:/Users/Taka/AppData/Local/FoundryVTT/Data/systems/crucible', 'system.json'],
  ['ember', emberRuntime, 'C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember', 'module.json'],
];

const lost = new Map();
let docs = 0, total = 0, landed = 0;

for (const [target, runtime, dir, mf] of CASES) {
  const { documentMappings, converterRegistry } = makeRuntime(runtime.DOCUMENT_MAPPINGS, runtime.PROJECT_CONVERTERS);
  const extMappings = effectiveMappings(target);
  const manifest = JSON.parse(fs.readFileSync(path.join(dir, mf), 'utf8'));

  for (const p of manifest.packs ?? []) {
    const packDir = path.join(dir, p.path ?? ('packs/' + p.name));
    if (!fs.existsSync(packDir)) continue;
    const db = new ClassicLevel(packDir, { createIfMissing: false });
    const buckets = {};
    for await (const [k, v] of db.iterator()) {
      const m = k.toString().match(/^!([^!]+)!(.+)$/);
      if (!m) continue;
      let d; try { d = JSON.parse(v.toString()); } catch { continue; }
      (buckets[m[1]] ||= []).push({ idPart: m[2], doc: d });
    }
    await db.close();

    const attach = (par, chi, f) => {
      const by = {};
      for (const e of (buckets[chi] || [])) (by[e.idPart.split('.')[0]] ||= []).push(e.doc);
      for (const e of (buckets[par] || [])) { const kids = by[e.doc._id]; if (kids?.length) e.doc[f] = kids; }
    };
    attach('actors', 'actors.items', 'items');
    attach('actors', 'actors.effects', 'effects');
    attach('items', 'items.effects', 'effects');
    attach('journal', 'journal.pages', 'pages');
    attach('journal', 'journal.categories', 'categories');
    attach('tables', 'tables.results', 'results');
    attach('scenes', 'scenes.regions', 'regions');

    const bucket = BUCKET[p.type];
    if (!bucket) continue;
    const meta = { id: manifest.id + '.' + p.name, name: p.name, type: p.type, packageName: manifest.id };

    for (const e of (buckets[bucket] || [])) {
      const doc = e.doc;
      const entry = extractDocument(doc, p.type, extMappings);
      if (!entry) continue;
      docs += 1;
      const key = doc.name || doc._id;

      for (const [field, val] of Object.entries(entry)) {
        const want = nLeaves(val);
        total += want;
        const pack = new MappedCompendium(meta, { entries: { [key]: { [field]: mark(val) } } },
          { documentMappings, converterRegistry, translationStrategies: [] });
        let out;
        try { out = pack.translate(structuredClone(doc)); } catch { out = doc; }
        const have = nMarks(out);
        landed += have;
        if (have < want) {
          const kk = p.type + '.' + field;
          const rec = lost.get(kk) ?? { docs: 0, missing: 0, sample: null };
          rec.docs += 1; rec.missing += (want - have);
          rec.sample ??= manifest.id + '.' + p.name + ' :: ' + doc.name;
          lost.set(kk, rec);
        }
      }
    }
  }
}

console.log('documents=' + docs + '  extracted leaves=' + total + '  landed=' + landed + '  lost=' + (total - landed));
console.log('\nfields whose extracted leaves do NOT reach the document:');
for (const [k, r] of [...lost.entries()].sort((a, b) => b[1].missing - a[1].missing)) {
  console.log('  ' + k.padEnd(30) + ' docs=' + String(r.docs).padStart(4) + '  lostLeaves=' + String(r.missing).padStart(5) + '   e.g. ' + r.sample);
}
