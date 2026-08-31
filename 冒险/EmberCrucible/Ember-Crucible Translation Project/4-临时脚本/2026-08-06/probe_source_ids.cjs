/**
 * Does babele 2.9.1's document-converter source-pack fallback actually apply to
 * ember's adventure-embedded actor items?  It needs `_stats.compendiumSource`
 * or `flags.core.sourceId` on each embedded item.
 *
 * Usage: node probe_source_ids.cjs <packDir>
 */
const {createRequire} = require('module');
const req = createRequire('C:/Users/Taka/Desktop/fvtt/package.json');
const {ClassicLevel} = req('classic-level');

const strip = s => String(s).replace(/<[^>]+>/g, ' ');
const srcUuid = d => d?.flags?.core?.sourceId ?? d?._stats?.compendiumSource ?? null;
const coll = u => String(u ?? '').match(/^Compendium\.([^.]+\.[^.]+)\./)?.[1] ?? null;

(async () => {
  const db = new ClassicLevel(process.argv[2], {createIfMissing: false});
  const byPack = {};
  let withSrc = 0, withoutSrc = 0, chWith = 0, chWithout = 0;
  const noSrcSamples = [];

  for await (const [k, v] of db.iterator()) {
    if (!k.toString().startsWith('!adventures!')) continue;
    const adv = JSON.parse(v.toString());
    for (const a of (adv.actors || [])) {
      for (const it of (a.items || [])) {
        const d = it?.system?.description;
        const chars = strip(it?.name ?? '').length
          + strip(typeof d === 'string' ? d : (d?.public ?? '')).length
          + strip(typeof d === 'object' ? (d?.private ?? '') : '').length
          + (it?.system?.actions || []).reduce((s, x) =>
              s + strip(x?.name ?? '').length + strip(x?.description ?? '').length, 0);

        const c = coll(srcUuid(it));
        if (c) {
          withSrc++; chWith += chars;
          const b = byPack[c] ||= {n: 0, ch: 0};
          b.n++; b.ch += chars;
        } else {
          withoutSrc++; chWithout += chars;
          if (noSrcSamples.length < 12) noSrcSamples.push(`${a.name} / ${it?.name} (${it?.type})`);
        }
      }
    }
  }
  await db.close();

  console.log(`embedded items WITH source pack : ${withSrc}  (${chWith} chars)`);
  console.log(`embedded items WITHOUT source   : ${withoutSrc}  (${chWithout} chars)`);
  const tot = chWith + chWithout;
  console.log(`=> auto-translatable via source fallback: ${(100 * chWith / (tot || 1)).toFixed(1)}% of chars\n`);
  console.log('by source pack:');
  for (const [p, b] of Object.entries(byPack).sort((x, y) => y[1].ch - x[1].ch)) {
    console.log(`   ${p.padEnd(38)} ${String(b.n).padStart(5)} items  ${String(b.ch).padStart(8)} chars`);
  }
  if (noSrcSamples.length) {
    console.log('\nsample items with NO source pack (must be translated inline):');
    for (const s of noSrcSamples) console.log('   ' + s);
  }
})();
