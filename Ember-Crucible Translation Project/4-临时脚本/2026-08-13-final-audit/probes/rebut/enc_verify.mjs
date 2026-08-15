// Independent verification of the encounter tokenData.name claim. READ ONLY.
import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';
const { ClassicLevel } = createRequire('C:/Users/Taka/Desktop/fvtt/package.json')('classic-level');
const PKG = 'C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/packs';

const out = {};
for (const pack of ['adventure', 'crucible-adventure']) {
  const db = new ClassicLevel(path.join(PKG, pack), { valueEncoding: 'json' });
  await db.open();
  let advCount = 0;
  const actorNames = new Map();
  const protoNames = new Map();
  const rows = [];
  const effectRows = [];
  let tokensArrays = 0, actorEntries = 0, noName = 0, emptyName = 0;
  const otherTokenDataKeys = new Map();
  const encounterKeys = new Map();
  for await (const [k, doc] of db.iterator()) {
    if (!k.toString().startsWith('!adventures!')) continue;
    advCount++;
    for (const a of doc?.actors ?? []) { actorNames.set(a._id, a.name); protoNames.set(a._id, a.prototypeToken?.name); }
    for (const j of doc?.journal ?? []) {
      for (const p of j.pages ?? []) {
        const enc = p?.system?.encounter;
        if (enc && typeof enc === 'object') for (const kk of Object.keys(enc)) encounterKeys.set(kk, (encounterKeys.get(kk)||0)+1);
        const toks = enc?.tokens;
        if (!Array.isArray(toks)) continue;
        tokensArrays++;
        for (const t of toks) {
          for (const a of t.actors ?? []) {
            actorEntries++;
            const td = a?.tokenData;
            if (td && typeof td === 'object') for (const kk of Object.keys(td)) otherTokenDataKeys.set(kk, (otherTokenDataKeys.get(kk)||0)+1);
            for (const ef of td?.delta?.effects ?? []) effectRows.push(ef?.name);
            const nm = td?.name;
            if (nm === undefined || nm === null) { noName++; continue; }
            if (typeof nm !== 'string' || !nm.trim()) { emptyName++; continue; }
            const id = String(a.actor ?? '').replace(/^Actor\./, '').split('.').pop();
            rows.push({ page: `${j.name} / ${p.name}`, jn: j.name, pn: p.name,
              tokenName: nm, actorName: actorNames.get(id), protoName: protoNames.get(id), actorRef: a.actor });
          }
        }
      }
    }
  }
  await db.close();
  out[pack] = { advCount, tokensArrays, actorEntries, withName: rows.length, noName, emptyName,
    encounterKeys: [...encounterKeys], tokenDataKeys: [...otherTokenDataKeys],
    effects: effectRows.length, effectUniq: [...new Set(effectRows)], rows };
}
fs.writeFileSync('enc_verify.json', JSON.stringify(out, null, 1));
for (const [pack, o] of Object.entries(out)) {
  console.log(`\n=== ${pack} : adventures=${o.advCount} pagesWithTokens=${o.tokensArrays} actorEntries=${o.actorEntries} withName=${o.withName} noName=${o.noName} empty=${o.emptyName}`);
  console.log('  encounter keys:', JSON.stringify(o.encounterKeys));
  console.log('  tokenData keys:', JSON.stringify(o.tokenDataKeys));
  console.log('  delta effects:', o.effects, JSON.stringify(o.effectUniq));
  const rows = o.rows;
  const eqActor = rows.filter(r => r.tokenName === r.actorName).length;
  const eqProto = rows.filter(r => r.tokenName === r.protoName).length;
  const missingActor = rows.filter(r => r.actorName === undefined).length;
  console.log(`  ==actor:${eqActor}  ==proto:${eqProto}  neither:${rows.filter(r=>r.tokenName!==r.actorName&&r.tokenName!==r.protoName).length}  actorNotFound:${missingActor}`);
  const uniq = new Set(rows.map(r => r.tokenName));
  console.log(`  distinct tokenName: ${uniq.size}`);
}
