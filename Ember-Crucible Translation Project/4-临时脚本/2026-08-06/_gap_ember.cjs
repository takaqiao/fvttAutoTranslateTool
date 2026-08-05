/**
 * True gap measurement: ember 0.6.0 adventure pack content vs ember_cn v1.0.15
 * translation, field-by-field, using v1.0.15's own mapping semantics.
 */
const fs = require('fs');
const {createRequire} = require('module');
const req = createRequire('C:/Users/Taka/Desktop/fvtt/package.json');
const {ClassicLevel} = req('classic-level');

const CJK = /[\u4e00-\u9fff]/;
const strip = s => String(s).replace(/<[^>]+>/g, ' ');
const get = (o, p) => p.split('.').reduce((a, k) => (a == null ? a : a[k]), o);

const CN = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const advCN = CN.entries[Object.keys(CN.entries)[0]] || {};

const S = {};                       // bucket -> {tot, cov, todo, todoCh, mapped}
const bump = (b, en, cn, mapped) => {
  const x = S[b] ||= {tot: 0, cov: 0, todo: 0, todoCh: 0, mapped};
  if (typeof en !== 'string' || !en.trim() || en.trim().length < 2) return;
  x.tot++;
  if (typeof cn === 'string' && CJK.test(cn)) x.cov++;
  else { x.todo++; x.todoCh += strip(en).length; }
};

(async () => {
  const db = new ClassicLevel(process.argv[2], {createIfMissing: false});
  for await (const [k, v] of db.iterator()) {
    if (!k.toString().startsWith('!adventures!')) continue;
    const adv = JSON.parse(v.toString());

    bump('adventure.name', adv.name, advCN.name, true);
    bump('adventure.description', adv.description, advCN.description, true);
    bump('adventure.caption', adv.caption, advCN.caption, true);

    // --- journals / pages ---
    for (const j of (adv.journal || [])) {
      const jc = advCN.journals?.[j.name];
      bump('journal.name', j.name, jc?.name, true);
      for (const p of (j.pages || [])) {
        const pc = jc?.pages?.[p.name];
        bump('page.name', p.name, pc?.name, true);
        bump('page.text.content', p.text?.content, pc?.text, true);
        bump('page.system.overview', get(p, 'system.overview'), pc?.soverview ?? pc?.overview, true);
        bump('page.system.exposition', get(p, 'system.exposition'), pc?.sexposition ?? pc?.exposition, true);
        bump('page.system.summary', get(p, 'system.summary'), pc?.ssummary ?? pc?.summary, true);
        bump('page.system.content.overview', get(p, 'system.content.overview'), pc?.scoverview ?? pc?.coverview, true);
        bump('page.system.content.gamemaster', get(p, 'system.content.gamemaster'), pc?.sgamemaster ?? pc?.gamemaster, true);
        // NOT in v1.0.15 mapping:
        bump('page.system.subtitle [UNMAPPED]', get(p, 'system.subtitle'), undefined, false);
        bump('page.system.pronunciation [UNMAPPED]', get(p, 'system.pronunciation'), undefined, false);
        bump('page.system.banner.caption [UNMAPPED]', get(p, 'system.banner.caption'), undefined, false);
        bump('page.system.terrain [UNMAPPED]', get(p, 'system.terrain'), undefined, false);
        for (const f of ['height', 'lifespan', 'origin'])
          bump('page.system.ancestryStats [UNMAPPED]', get(p, 'system.' + f), undefined, false);
        for (const o of (get(p, 'system.outcomes') || [])) {
          bump('page.system.outcomes.label [UNMAPPED]', o?.label, undefined, false);
          bump('page.system.outcomes.summary [UNMAPPED]', o?.summary, undefined, false);
        }
      }
    }

    // --- actors ---
    for (const a of (adv.actors || [])) {
      const ac = advCN.actors?.[a.name];
      bump('actor.name', a.name, ac?.name, true);
      bump('actor.prototypeToken', a.prototypeToken?.name, ac?.prototypeToken, true);
      bump('actor.bio.public', get(a, 'system.details.biography.public'), ac?.biographypublic, true);
      bump('actor.bio.private', get(a, 'system.details.biography.private'), ac?.biographyprivate, true);
      bump('actor.bio.appearance', get(a, 'system.details.biography.appearance'), ac?.biographyappearance, true);
      // ember_cn actors mapping has NO `items` key -> embedded actor items unmapped
      for (const it of (a.items || [])) {
        bump('actor.items.name [UNMAPPED]', it?.name, undefined, false);
        const d = get(it, 'system.description');
        bump('actor.items.desc [UNMAPPED]', typeof d === 'string' ? d : d?.public, undefined, false);
        bump('actor.items.desc [UNMAPPED]', typeof d === 'object' ? d?.private : undefined, undefined, false);
        for (const act of (get(it, 'system.actions') || [])) {
          bump('actor.items.action [UNMAPPED]', act?.name, undefined, false);
          bump('actor.items.action [UNMAPPED]', act?.description, undefined, false);
        }
      }
      for (const act of (get(a, 'system.actions') || [])) {
        bump('actor.actions [UNMAPPED]', act?.name, undefined, false);
        bump('actor.actions [UNMAPPED]', act?.description, undefined, false);
      }
    }

    // --- items ---
    for (const it of (adv.items || [])) {
      const ic = advCN.items?.[it.name];
      bump('item.name', it.name, ic?.name, true);
      const d = get(it, 'system.description');
      if (typeof d === 'string') bump('item.description', d, ic?.descriptionpublic, true);
      else {
        bump('item.description.public', d?.public, ic?.descriptionpublic, true);
        bump('item.description.private', d?.private, ic?.descriptionprivate, true);
      }
      const acts = get(it, 'system.actions') || [];
      const an = [].concat(ic?.actionname ?? []), ad = [].concat(ic?.actiondesc ?? []);
      acts.forEach((act, i) => {
        bump('item.action.name', act?.name, an[i], true);
        bump('item.action.description', act?.description, ad[i], true);
      });
    }

    // --- scenes / macros / folders / tables / playlists ---
    for (const s of (adv.scenes || [])) bump('scene.name', s?.name, advCN.scenes?.[s?.name]?.name ?? advCN.scenes?.[s?.name], true);
    for (const m of (adv.macros || [])) bump('macro.name', m?.name, advCN.macros?.[m?.name]?.name ?? advCN.macros?.[m?.name], true);
    for (const f of (adv.folders || [])) bump('folder.name', f?.name, advCN.folders?.[f?.name]?.name ?? advCN.folders?.[f?.name], true);
    for (const t of (adv.tables || [])) {
      const tc = advCN.tables?.[t?.name];
      bump('table.name', t?.name, tc?.name, true);
      bump('table.description', t?.description, tc?.description, true);
      for (const r of (t?.results || [])) bump('table.result [UNMAPPED?]', r?.description ?? r?.text, undefined, false);
    }
    for (const pl of (adv.playlists || [])) bump('playlist.name', pl?.name, advCN.playlists?.[pl?.name]?.name ?? advCN.playlists?.[pl?.name], true);
  }
  await db.close();

  const rows = Object.entries(S).sort((a, b) => b[1].todoCh - a[1].todoCh);
  console.log(`${'field'.padEnd(38)}${'EN'.padStart(7)}${'done'.padStart(7)}${'cov%'.padStart(6)}${'todo'.padStart(7)}${'todo chars'.padStart(12)}`);
  let T = [0, 0, 0, 0], U = [0, 0];
  for (const [k, x] of rows) {
    console.log(`${k.padEnd(38)}${String(x.tot).padStart(7)}${String(x.cov).padStart(7)}${String(Math.round(100 * x.cov / (x.tot || 1))).padStart(5)}%${String(x.todo).padStart(7)}${String(x.todoCh).padStart(12)}`);
    T = [T[0] + x.tot, T[1] + x.cov, T[2] + x.todo, T[3] + x.todoCh];
    if (!x.mapped) U = [U[0] + x.todo, U[1] + x.todoCh];
  }
  console.log(`${'TOTAL'.padEnd(38)}${String(T[0]).padStart(7)}${String(T[1]).padStart(7)}${String(Math.round(100 * T[1] / T[0])).padStart(5)}%${String(T[2]).padStart(7)}${String(T[3]).padStart(12)}`);
  console.log(`  of which UNMAPPED (needs converter work first): ${U[0]} strings, ${U[1]} chars`);
  console.log(`  of which mapped-but-untranslated              : ${T[2] - U[0]} strings, ${T[3] - U[1]} chars`);
})();
