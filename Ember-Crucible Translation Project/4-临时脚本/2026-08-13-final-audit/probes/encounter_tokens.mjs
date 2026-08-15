#!/usr/bin/env node
/**
 * For every `JournalEntryPage.system.encounter.tokens[].actors[]` entry in the
 * Ember adventure packs: print the referenced actor's English name, the literal
 * `tokenData.name` override, and whether the two agree.  Read-only.
 *
 * Why it matters: ember.mjs `_createActorToken(actor, tokenData)` ->
 * `actor.getTokenDocument(tokenData, {parent: this.scene})`, so `tokenData.name`
 * becomes the placed token's displayed name and overrides the (translated)
 * prototypeToken name.
 */
import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';
const { ClassicLevel } = createRequire('C:/Users/Taka/Desktop/fvtt/package.json')('classic-level');

const DIR = 'C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember';
const PACK = process.argv[2] ?? 'crucible-adventure';

const db = new ClassicLevel(path.join(DIR, 'packs', PACK), { createIfMissing: false });
const adventures = [];
for await (const [k, v] of db.iterator()) {
  const m = k.toString().match(/^!([^!]+)!(.+)$/);
  if (!m || m[1] !== 'adventures') continue;
  adventures.push(JSON.parse(v.toString()));
}
await db.close();

const rows = [];
for (const adv of adventures) {
  const actorById = new Map();
  for (const a of adv.actors ?? []) actorById.set(a._id, a);
  for (const j of adv.journal ?? []) {
    for (const p of j.pages ?? []) {
      const toks = p.system?.encounter?.tokens;
      if (!Array.isArray(toks)) continue;
      for (const t of toks) {
        for (const a of t.actors ?? []) {
          const name = a.tokenData?.name;
          if (!name) continue;
          const id = String(a.actor ?? '').split('.').pop();
          const src = actorById.get(id);
          rows.push({
            page: `${j.name} / ${p.name}`,
            actorName: src?.name ?? '(actor not in this adventure)',
            protoName: src?.prototypeToken?.name ?? '',
            tokenName: name,
          });
        }
      }
    }
  }
}

const same = rows.filter((r) => r.tokenName === r.actorName).length;
const sameProto = rows.filter((r) => r.tokenName === r.protoName).length;
console.log(`pack=${PACK}  encounter token overrides: ${rows.length}`);
console.log(`  tokenData.name === actor.name            : ${same}`);
console.log(`  tokenData.name === prototypeToken.name   : ${sameProto}`);
console.log(`  DIFFERENT from both (pure override)      : ${rows.filter((r) => r.tokenName !== r.actorName && r.tokenName !== r.protoName).length}`);
const uniq = new Map();
for (const r of rows) {
  const k = `${r.tokenName}\t${r.actorName}\t${r.protoName}`;
  uniq.set(k, (uniq.get(k) || 0) + 1);
}
console.log(`\n  ${uniq.size} distinct (tokenData.name, actor.name, prototypeToken.name) triples:`);
for (const [k, n] of [...uniq.entries()].sort((a, b) => b[1] - a[1])) {
  const [tn, an, pn] = k.split('\t');
  const flag = tn === an ? ' ' : (tn === pn ? '~' : '!');
  console.log(`  ${flag} ${String(n).padStart(3)}  token=${JSON.stringify(tn)}  actor=${JSON.stringify(an)}  proto=${JSON.stringify(pn)}`);
}
console.log('\n  sample pages:');
for (const r of rows.slice(0, 12)) console.log(`    ${r.page}  -> ${r.tokenName}`);
