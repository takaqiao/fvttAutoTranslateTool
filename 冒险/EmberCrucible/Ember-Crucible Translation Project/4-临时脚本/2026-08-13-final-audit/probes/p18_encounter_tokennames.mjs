#!/usr/bin/env node
/** P18: quantify encounter tokenData.name (unmapped) vs actor names. READ ONLY. */
import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';
const { ClassicLevel } = createRequire('C:/Users/Taka/Desktop/fvtt/package.json')('classic-level');
const PKG = 'C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/packs';
const EN = 'C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/1-Ember汉化插件/compendium/en';
const CN = 'C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/1-Ember汉化插件/compendium/cn';

const names = new Map();   // name -> {count, packs:Set, pages:Set}
const actorNames = new Map(); // Actor.<id> -> name

for (const pack of ['adventure', 'crucible-adventure']) {
  const db = new ClassicLevel(path.join(PKG, pack), { valueEncoding: 'json' });
  await db.open();
  for await (const [, doc] of db.iterator()) {
    for (const a of doc?.actors ?? []) actorNames.set(a._id, a.name);
    for (const j of doc?.journal ?? []) {
      for (const p of j.pages ?? []) {
        for (const t of p?.system?.encounter?.tokens ?? []) {
          for (const a of t.actors ?? []) {
            const nm = a?.tokenData?.name;
            if (typeof nm !== 'string' || !nm.trim()) continue;
            const rec = names.get(nm) ?? { count: 0, packs: new Set(), pages: new Set(), actor: a.actor };
            rec.count++; rec.packs.add(pack); rec.pages.add(`${j.name}/${p.name}`);
            names.set(nm, rec);
          }
        }
      }
    }
  }
  await db.close();
}

const enBlob = fs.readFileSync(path.join(EN, 'ember.adventure.json'), 'utf8')
  + fs.readFileSync(path.join(EN, 'ember.crucible-adventure.json'), 'utf8');
const cnBlob = fs.readFileSync(path.join(CN, 'ember.adventure.json'), 'utf8')
  + fs.readFileSync(path.join(CN, 'ember.crucible-adventure.json'), 'utf8');

let inEnAsSomethingElse = 0, notAnywhere = 0;
const rows = [];
for (const [nm, r] of [...names].sort((a, b) => b[1].count - a[1].count)) {
  const q = JSON.stringify(nm).slice(1, -1);
  const inEn = enBlob.includes(`"${q}"`);
  const actorNm = actorNames.get(String(r.actor).replace(/^Actor\./, ''));
  const sameAsActor = actorNm === nm;
  rows.push({ nm, count: r.count, inEnAsKey: inEn, actorNm, sameAsActor, page: [...r.pages][0] });
  if (inEn) inEnAsSomethingElse++; else notAnywhere++;
}
console.log(`distinct encounter tokenData.name values: ${rows.length}`);
console.log(`total occurrences: ${[...names.values()].reduce((a, b) => a + b.count, 0)}`);
console.log(`appears somewhere in the EN baseline as a quoted string (i.e. reachable via some other key): ${inEnAsSomethingElse}`);
console.log(`does NOT appear at all in the EN baseline: ${notAnywhere}`);
console.log(`\ndiffers from the linked actor's own name: ${rows.filter((r) => !r.sameAsActor).length}`);
for (const r of rows) {
  console.log(`  ${r.count.toString().padStart(3)}x  "${r.nm}"  actor="${r.actorNm}"  sameAsActor=${r.sameAsActor}  inEN=${r.inEnAsKey}   ${r.page}`);
}
console.log('\nAny of these strings already present in the CN pack?');
for (const r of rows.slice(0, 12)) {
  const q = JSON.stringify(r.nm).slice(1, -1);
  console.log(`  "${r.nm}" -> in CN blob: ${cnBlob.includes(`"${q}"`)}`);
}
