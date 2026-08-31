import fs from 'fs';
const RAW = 'C:/Users/Taka/AppData/Local/Temp/claude/C--Users-Taka-Desktop-fvtt/892fdb28-d096-4415-94d1-99d12c38ef86/scratchpad/raw_adv.json';
const data = JSON.parse(fs.readFileSync(RAW, 'utf8'));

const report = {};
for (const [pack, rows] of Object.entries(data)) {
  const adv = rows[0][1];
  const journals = adv.journal || [];
  let pages = 0;
  const byType = {};
  const hits = [];           // {journal, page, type, tokenName, actorName, actorId, actorLink}
  for (const j of journals) {
    for (const p of (j.pages || [])) {
      pages++;
      byType[p.type] = (byType[p.type] || 0) + 1;
      const enc = p.system?.encounter;
      if (!enc) continue;
      const toks = enc.tokens;
      if (!Array.isArray(toks)) continue;
      for (const t of toks) {
        const actors = t.actors;
        if (!Array.isArray(actors)) continue;
        for (const a of actors) {
          hits.push({
            journal: j.name, page: p.name, type: p.type,
            tokenName: a.tokenData?.name ?? null,
            tokenDataKeys: a.tokenData ? Object.keys(a.tokenData) : null,
            actorRef: a.actor ?? null,
            actorEntry: Object.keys(a),
          });
        }
      }
    }
  }
  report[pack] = { journals: journals.length, pages, byType, hits };
}
fs.writeFileSync(process.argv[2], JSON.stringify(report, null, 1));
for (const [pack, r] of Object.entries(report)) {
  console.log('=== ', pack, 'journals', r.journals, 'pages', r.pages);
  console.log('  pageTypes:', JSON.stringify(r.byType));
  console.log('  encounter actor entries:', r.hits.length);
  const named = r.hits.filter(h => h.tokenName);
  console.log('  with tokenData.name:', named.length, 'distinct:', new Set(named.map(h=>h.tokenName)).size);
  const cnt = {};
  for (const h of named) cnt[h.tokenName] = (cnt[h.tokenName]||0)+1;
  console.log('  top:', Object.entries(cnt).sort((a,b)=>b[1]-a[1]).slice(0,15).map(x=>x[0]+':'+x[1]).join(', '));
  console.log('  page types carrying encounter:', JSON.stringify([...new Set(r.hits.map(h=>h.type))]));
  console.log('  sample tokenData keys:', JSON.stringify(r.hits.slice(0,3).map(h=>h.tokenDataKeys)));
  console.log('  sample actor entry keys:', JSON.stringify(r.hits.slice(0,3).map(h=>h.actorEntry)));
}
