import { createRequire } from 'module';
const { ClassicLevel } = createRequire('C:/Users/Taka/Desktop/fvtt/package.json')('classic-level');
import path from 'path';
const EMBER = 'C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/packs';
for (const p of ['crucible-adventure', 'adventure']) {
  const db = new ClassicLevel(path.join(EMBER, p), { createIfMissing: false });
  const advs = [];
  for await (const [k, v] of db.iterator()) {
    const m = k.toString().match(/^!([^!]+)!(.+)$/);
    if (!m || m[1] !== 'adventures') continue;
    advs.push(JSON.parse(v.toString()));
  }
  await db.close();
  const names = new Map(), deltaNames = new Map(), tdKeys = new Set(), otherNamePaths = new Map();
  let tokensArr = 0, actorsArr = 0, tdCount = 0;
  const subtypesWithEnc = {};
  for (const a of advs) for (const j of (a.journal ?? [])) for (const pg of (j.pages ?? [])) {
    const enc = pg?.system?.encounter; if (!enc) continue;
    subtypesWithEnc[pg.type] = (subtypesWithEnc[pg.type]||0)+1;
    if (Array.isArray(enc.tokens)) tokensArr++;
    for (const t of (enc.tokens ?? [])) {
      if (Array.isArray(t?.actors)) actorsArr++;
      // other string fields on token entry
      for (const [k2, v2] of Object.entries(t ?? {})) if (typeof v2 === 'string' && v2.trim()) otherNamePaths.set('tokens[].'+k2, (otherNamePaths.get('tokens[].'+k2)||0)+1);
      for (const ac of (t?.actors ?? [])) {
        const td = ac?.tokenData; if (!td) continue; tdCount++;
        for (const k3 of Object.keys(td)) tdKeys.add(k3);
        if (typeof td.name === 'string' && td.name.trim()) names.set(td.name, (names.get(td.name)||0)+1);
        for (const e of (td?.delta?.effects ?? [])) if (typeof e?.name === 'string' && e.name.trim()) deltaNames.set(e.name, (deltaNames.get(e.name)||0)+1);
      }
      for (const [k2,v2] of Object.entries(t?.combat ?? {})) if (typeof v2 === 'string' && v2.trim()) otherNamePaths.set('tokens[].combat.'+k2, (otherNamePaths.get('tokens[].combat.'+k2)||0)+1);
    }
  }
  console.log('=== ', p);
  console.log(' pages w/ encounter by subtype:', JSON.stringify(subtypesWithEnc));
  console.log(' tokens arrays:', tokensArr, ' actors arrays:', actorsArr, ' tokenData objs:', tdCount);
  console.log(' tokenData keys seen:', [...tdKeys].sort().join(','));
  console.log(' tokenData.name leaves:', [...names.values()].reduce((a,b)=>a+b,0), 'unique:', names.size);
  console.log(' delta.effects[].name leaves:', [...deltaNames.values()].reduce((a,b)=>a+b,0), 'unique:', [...deltaNames.keys()].join('|'));
  console.log(' other string fields on tokens[]:', JSON.stringify([...otherNamePaths.entries()]));
  if (p === 'crucible-adventure') {
    console.log(' --- all unique tokenData.name (count) ---');
    console.log(JSON.stringify([...names.entries()].sort((a,b)=>b[1]-a[1]), null, 0));
  }
}
