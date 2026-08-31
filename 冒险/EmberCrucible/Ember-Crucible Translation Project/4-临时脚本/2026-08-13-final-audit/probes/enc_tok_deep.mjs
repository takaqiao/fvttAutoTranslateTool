import fs from 'fs';
const RAW = 'C:/Users/Taka/AppData/Local/Temp/claude/C--Users-Taka-Desktop-fvtt/892fdb28-d096-4415-94d1-99d12c38ef86/scratchpad/raw_adv.json';
const data = JSON.parse(fs.readFileSync(RAW, 'utf8'));

for (const pack of ['adventure','crucible-adventure']) {
  const adv = data[pack][0][1];
  console.log('=====', pack, 'top-level keys:', Object.keys(adv).join(','));
  const actors = adv.actors || [];
  console.log('  actors in adventure:', actors.length);
  const byId = new Map(), byUuid = new Map();
  for (const a of actors) { byId.set(a._id, a); }
  // build map of any uuid-ish
  const pagesWithNamed = new Set();
  const rows = [];
  for (const j of adv.journal||[]) for (const p of j.pages||[]) {
    const toks = p.system?.encounter?.tokens;
    if (!Array.isArray(toks)) continue;
    for (const t of toks) for (const a of (t.actors||[])) {
      const tn = a.tokenData?.name;
      if (!tn) continue;
      pagesWithNamed.add(j.name+' :: '+p.name);
      rows.push({ j:j.name, p:p.name, tn, ref:a.actor });
    }
  }
  console.log('  pages carrying named tokens:', pagesWithNamed.size);
  // resolve refs
  const refKinds = {};
  for (const r of rows) {
    const k = typeof r.ref === 'string' ? (r.ref.split('.')[0]) : typeof r.ref;
    refKinds[k] = (refKinds[k]||0)+1;
  }
  console.log('  actor ref forms:', JSON.stringify(refKinds));
  console.log('  sample refs:', JSON.stringify(rows.slice(0,5).map(r=>[r.tn, r.ref])));
  fs.writeFileSync(`C:/Users/Taka/AppData/Local/Temp/claude/C--Users-Taka-Desktop-fvtt/892fdb28-d096-4415-94d1-99d12c38ef86/scratchpad/rows_${pack}.json`, JSON.stringify(rows,null,1));
  // adventure actor names
  fs.writeFileSync(`C:/Users/Taka/AppData/Local/Temp/claude/C--Users-Taka-Desktop-fvtt/892fdb28-d096-4415-94d1-99d12c38ef86/scratchpad/actors_${pack}.json`, JSON.stringify(actors.map(a=>({_id:a._id,name:a.name,proto:a.prototypeToken?.name, link:a.prototypeToken?.actorLink})),null,1));
}
