/**
 * TWO-SIDED MAPPING CONSISTENCY TEST (round-3 unit ③ acceptance item)
 *
 * Side EXTRACT : 3-常用脚本/extract/mappings.mjs  -> EMBER_LAYER / effectiveMappings
 * Side RUNTIME : 1-Ember汉化插件/babele-mappings.js -> DOCUMENT_MAPPINGS
 *                (what actually reaches `babele.registerMapping()`)
 *
 * Four checks, each measured, none assumed:
 *
 *  T1 shipped runtime file == regenerate(mappings.mjs)  [byte-identical]
 *      Regeneration goes into a STAGING tree that contains only the ember repo,
 *      so `generate_runtime.mjs` cannot touch the crucible repo (not ours).
 *
 *  T2 EMBER_LAYER  ==deep==  DOCUMENT_MAPPINGS
 *      Catches a hand-edited runtime file even if T1's generator were bypassed.
 *
 *  T3 Babele's REAL resolution, run here: feed builtin defaults + the shipped
 *      DOCUMENT_MAPPINGS through Babele 2.9.1's own `DocumentMappings` class and
 *      read back the effective field keys per document type. Then assert the
 *      keys the EXTRACTOR wrote into compendium/en are a subset of what the
 *      runtime resolves, per document type actually present in the baseline.
 *      -> this is the "抽得出来但运行时不翻 / 反之" check.
 *      Only `foundry.utils.mergeObject` is shimmed; the shim is itself asserted
 *      on three known inputs before use (S1..S3).
 *
 *  T4 drawings re-verification (unit ④): assert that registering our Scene layer
 *      ENRICHES rather than REPLACES Babele's Scene default, i.e. `drawings`,
 *      `notes`, `regions`, `name` all survive next to our levels/tokens/navName/
 *      sounds. Measured on the real class, not read off the source.
 */
import fs from 'fs';
import path from 'path';
import { execFileSync } from 'child_process';

const P = 'C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project';
const OUTDIR = `${P}/4-临时脚本/2026-08-22-ui-round3`;
const BABELE = 'C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/babele';

const results = [];
const rec = (id, ok, detail) => { results.push({ id, ok, detail }); console.log(`${ok ? 'PASS' : 'FAIL'} ${id}  ${detail}`); };

/* ---------------- shim: foundry.utils.mergeObject ---------------- */
function mergeObject(original, other = {}, { inplace = true } = {}) {
  const target = inplace ? original : structuredClone(original);
  for (const [k, v] of Object.entries(other)) {
    const cur = target[k];
    if (v && typeof v === 'object' && !Array.isArray(v)
        && cur && typeof cur === 'object' && !Array.isArray(cur)) {
      target[k] = mergeObject(cur, v, { inplace: false });
    } else {
      target[k] = v && typeof v === 'object' ? structuredClone(v) : v;
    }
  }
  return target;
}
globalThis.foundry = globalThis.foundry ?? {};
globalThis.foundry.utils = globalThis.foundry.utils ?? {};
globalThis.foundry.utils.mergeObject = mergeObject;
globalThis.foundry.utils.deepClone = (o) => structuredClone(o);
globalThis.foundry.utils.getProperty = (o, p) => String(p).split('.').reduce((a, k) => (a == null ? a : a[k]), o);
globalThis.foundry.utils.setProperty = (o, p, v) => {
  const ks = String(p).split('.'); const last = ks.pop();
  let cur = o; for (const k of ks) cur = (cur[k] ??= {});
  cur[last] = v; return true;
};

/* --- S1..S3: prove the shim before trusting anything built on it --- */
rec('S1-shim-insert', JSON.stringify(mergeObject({ a: 1 }, { b: 2 }, { inplace: false })) === '{"a":1,"b":2}',
  'mergeObject inserts new keys');
rec('S2-shim-overwrite', JSON.stringify(mergeObject({ a: 1 }, { a: 9 }, { inplace: false })) === '{"a":9}',
  'mergeObject overwrites scalars');
rec('S3-shim-recursive',
  JSON.stringify(mergeObject({ a: { x: 1, y: 2 } }, { a: { y: 9, z: 3 } }, { inplace: false })) === '{"a":{"x":1,"y":9,"z":3}}',
  'mergeObject recurses into plain objects');

/* ---------------- T1: regenerate into staging, byte-compare ---------------- */
const STAGE = `${OUTDIR}/stage`;
fs.rmSync(STAGE, { recursive: true, force: true });
fs.mkdirSync(`${STAGE}/1-Ember汉化插件`, { recursive: true });
const genOut = execFileSync(process.execPath,
  [`${P}/3-常用脚本/release/generate_runtime.mjs`, '--project', STAGE], { encoding: 'utf8' });
const shipped = fs.readFileSync(`${P}/1-Ember汉化插件/babele-mappings.js`);
const regen = fs.readFileSync(`${STAGE}/1-Ember汉化插件/babele-mappings.js`);
rec('T1-regen-byte-identical', Buffer.compare(shipped, regen) === 0,
  `shipped ${shipped.length} B vs regenerated ${regen.length} B; generator touched only the staged ember repo (${genOut.trim().split('\n').filter((l) => l.trim()).length} line(s) of output, crucible repo skipped)`);

/* ---------------- T2: deep-equal layer vs runtime export ---------------- */
const { EMBER_LAYER, effectiveMappings } = await import(`file:///${P.replace(/\\/g, '/')}/3-常用脚本/extract/mappings.mjs`);
const { DOCUMENT_MAPPINGS } = await import(`file:///${P.replace(/\\/g, '/')}/1-Ember汉化插件/babele-mappings.js`);
const canon = (o) => JSON.stringify(o, Object.keys(JSON.parse(JSON.stringify(o))).length ? undefined : undefined, 0);
const sortedJSON = (o) => JSON.stringify(o, (k, v) => (v && typeof v === 'object' && !Array.isArray(v)
  ? Object.fromEntries(Object.entries(v).sort(([a], [b]) => (a < b ? -1 : 1))) : v));
const layerTypes = Object.keys(EMBER_LAYER).sort();
const rtTypes = Object.keys(DOCUMENT_MAPPINGS).sort();
rec('T2-deep-equal', sortedJSON(EMBER_LAYER) === sortedJSON(DOCUMENT_MAPPINGS),
  `${layerTypes.length} document types both sides; type lists ${JSON.stringify(layerTypes) === JSON.stringify(rtTypes) ? 'identical' : 'DIFFER'}`);

/* also: no converter is a live function (JSON.stringify would silently drop it) */
const funcs = [];
(function walk(n, p) {
  if (typeof n === 'function') { funcs.push(p); return; }
  if (n && typeof n === 'object') for (const [k, v] of Object.entries(n)) walk(v, p ? `${p}.${k}` : k);
})(EMBER_LAYER, '');
rec('T2b-no-function-values', funcs.length === 0,
  `${funcs.length} function-valued entries in EMBER_LAYER (any >0 would be dropped by JSON.stringify in the generator)`);

/* ---------------- T3/T4: run Babele's real DocumentMappings ---------------- */
const { DocumentMappings } = await import(`file:///${BABELE}/script/mapping/document-mappings.js`);
// identityExtractors/converterRegistry are only consulted when materialising a
// DocumentMapping instance; #rebuild() (the merge under test) never touches them,
// so truthy placeholders are enough and keep the merge code itself unmodified.
const dm = new DocumentMappings(undefined, {
  registeredMappings: [DOCUMENT_MAPPINGS],
  identityExtractors: { get: () => undefined },
  converterRegistry: { get: () => undefined },
});
const eff = dm.current();

const keysOf = (def) => Object.keys(def ?? {}).filter((k) => !k.startsWith('_'));
const variantKeysOf = (def) => {
  const out = new Set();
  for (const v of (def?._variants ?? [])) for (const k of Object.keys(v)) if (k !== '_when') out.add(k);
  return out;
};

/* T4 — drawings survives */
const sceneKeys = keysOf(eff.Scene);
const need = ['name', 'drawings', 'notes', 'regions', 'levels', 'tokens', 'navName', 'sounds'];
const missing = need.filter((k) => !sceneKeys.includes(k));
rec('T4-scene-merge-not-replace', missing.length === 0,
  `effective Scene keys = [${sceneKeys.join(', ')}]; expected all of [${need.join(', ')}]; missing = [${missing.join(', ') || 'none'}]`);

/* T4b — our RegionBehavior subtype keys did not evict Babele's own variants */
const rbBase = keysOf(eff.RegionBehavior);
const rbVar = [...variantKeysOf(eff.RegionBehavior)];
const rbWhen = (eff.RegionBehavior?._variants ?? []).map((v) => JSON.stringify(v._when));
rec('T4b-regionbehavior-variants-concat',
  rbVar.includes('text') && rbVar.includes('revealedDialog') && rbVar.includes('unrevealedDialog')
  && rbVar.includes('message') && rbVar.includes('description') && rbVar.includes('effects'),
  `base=[${rbBase.join(', ')}] variantKeys=[${rbVar.join(', ')}] variants=${rbWhen.length} -> ${rbWhen.join(' | ')}`);

/* T3 — every key the extractor wrote must be resolvable at runtime */
const EN = `${P}/1-Ember汉化插件/compendium/en`;
const baseline = JSON.parse(fs.readFileSync(`${EN}/ember.adventure.json`, 'utf8'));
const runtimeKeysFor = (type) => new Set([...keysOf(eff[type]), ...variantKeysOf(eff[type])]);

const probes = [];
const ents = baseline.entries;
for (const ev of Object.values(ents)) {
  probes.push(['Adventure', ev]);
  for (const s of Object.values(ev.scenes ?? {})) {
    probes.push(['Scene', s]);
    for (const r of Object.values(s.regions ?? {})) {
      probes.push(['Region', r]);
      for (const b of Object.values(r.behaviors ?? {})) probes.push(['RegionBehavior', b]);
    }
  }
  for (const m of Object.values(ev.macros ?? {})) probes.push(['Macro', m]);
}
const unresolved = {};
for (const [type, doc] of probes) {
  const rk = runtimeKeysFor(type);
  for (const k of Object.keys(doc)) {
    if (!rk.has(k)) (unresolved[`${type}.${k}`] ??= 0), unresolved[`${type}.${k}`]++;
  }
}
rec('T3-extract-keys-resolvable-at-runtime', Object.keys(unresolved).length === 0,
  `${probes.length} baseline documents probed across ${new Set(probes.map((x) => x[0])).size} document types; unresolvable keys = ${JSON.stringify(unresolved)}`);

/* T3b — reverse direction, for the record: runtime keys the baseline never carries */
const seenByType = {};
for (const [type, doc] of probes) (seenByType[type] ??= new Set()), Object.keys(doc).forEach((k) => seenByType[type].add(k));
const runtimeOnly = {};
for (const [type, seen] of Object.entries(seenByType)) {
  const extra = [...runtimeKeysFor(type)].filter((k) => !seen.has(k));
  if (extra.length) runtimeOnly[type] = extra;
}
rec('T3b-runtime-only-keys (informational)', true, JSON.stringify(runtimeOnly));

fs.writeFileSync(`${OUTDIR}/consistency.json`, JSON.stringify({ results, runtimeOnly, effectiveScene: eff.Scene, effectiveRegionBehavior: eff.RegionBehavior }, null, 2), 'utf8');
const failed = results.filter((r) => !r.ok);
console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
if (failed.length) process.exit(1);
