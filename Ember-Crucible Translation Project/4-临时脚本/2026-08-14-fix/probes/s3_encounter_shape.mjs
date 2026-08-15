import { createRequire } from 'module';
const { ClassicLevel } = createRequire('C:/Users/Taka/Desktop/fvtt/package.json')('classic-level');
import path from 'path';

const EMBER = 'C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/packs';
const packs = process.argv.slice(2).length ? process.argv.slice(2) : ['crucible-adventure', 'adventure'];

for (const p of packs) {
  const db = new ClassicLevel(path.join(EMBER, p), { createIfMissing: false });
  const buckets = {};
  for await (const [k, v] of db.iterator()) {
    const m = k.toString().match(/^!([^!]+)!(.+)$/);
    if (!m) continue;
    let doc; try { doc = JSON.parse(v.toString()); } catch { continue; }
    (buckets[m[1]] ||= []).push({ id: m[2], doc });
  }
  await db.close();
  console.log('=== pack', p, 'buckets:', Object.keys(buckets).join(', '));

  let leaves = 0, uniq = new Set(), pages = 0, withEnc = 0, sample = null;
  const subtypes = {};
  const scan = (page) => {
    const enc = page?.system?.encounter;
    if (!enc) return;
    withEnc += 1;
    const toks = enc.tokens;
    if (!Array.isArray(toks)) { console.log('  !! tokens not array:', typeof toks, JSON.stringify(enc).slice(0,200)); return; }
    for (const t of toks) {
      for (const a of (t?.actors ?? [])) {
        const n = a?.tokenData?.name;
        if (typeof n === 'string' && n.trim()) { leaves++; uniq.add(n); if (!sample) sample = { page: page.name, type: page.type, t, a }; }
      }
    }
  };
  for (const { doc } of (buckets.adventures ?? [])) {
    for (const j of (doc.journal ?? [])) for (const pg of (j.pages ?? [])) { pages++; subtypes[pg.type] = (subtypes[pg.type]||0)+1; scan(pg); }
  }
  for (const { doc } of (buckets.journal ?? [])) for (const pg of (doc.pages ?? [])) { pages++; subtypes[pg.type]=(subtypes[pg.type]||0)+1; scan(pg); }
  for (const { doc } of (buckets.pages ?? [])) { pages++; subtypes[doc.type]=(subtypes[doc.type]||0)+1; scan(doc); }
  console.log('  pages:', pages, 'subtypes:', JSON.stringify(subtypes));
  console.log('  pages with system.encounter:', withEnc, ' tokenData.name leaves:', leaves, ' unique:', uniq.size);
  if (sample) {
    console.log('  sample page:', sample.page, '| type:', sample.type);
    console.log('  sample token entry keys:', Object.keys(sample.t));
    console.log('  sample token JSON:', JSON.stringify(sample.t).slice(0, 900));
  }
}
