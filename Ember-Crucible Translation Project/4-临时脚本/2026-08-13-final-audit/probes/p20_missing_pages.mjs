#!/usr/bin/env node
/** P20: which live journal pages' text.content is absent from the EN baseline? READ ONLY. */
import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';
const { ClassicLevel } = createRequire('C:/Users/Taka/Desktop/fvtt/package.json')('classic-level');
const PKG = 'C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/packs';
const EN = 'C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/1-Ember汉化插件/compendium/en';
const CN = 'C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/1-Ember汉化插件/compendium/cn';

const known = new Set();
function collect(o) {
  if (o == null) return;
  if (Array.isArray(o)) { o.forEach(collect); return; }
  if (typeof o === 'object') { Object.values(o).forEach(collect); return; }
  if (typeof o === 'string') known.add(o.trim());
}
for (const f of fs.readdirSync(EN)) if (f.endsWith('.json')) collect(JSON.parse(fs.readFileSync(path.join(EN, f), 'utf8')));

const cnJson = {};
for (const f of fs.readdirSync(CN)) if (f.endsWith('.json')) cnJson[f] = JSON.parse(fs.readFileSync(path.join(CN, f), 'utf8'));
const enJson = {};
for (const f of fs.readdirSync(EN)) if (f.endsWith('.json')) enJson[f] = JSON.parse(fs.readFileSync(path.join(EN, f), 'utf8'));

const packFile = { adventure: 'ember.adventure.json', 'crucible-adventure': 'ember.crucible-adventure.json' };

for (const pack of ['adventure', 'crucible-adventure']) {
  const db = new ClassicLevel(path.join(PKG, pack), { valueEncoding: 'json' });
  await db.open();
  for await (const [, doc] of db.iterator()) {
    for (const j of doc?.journal ?? []) {
      for (const p of j.pages ?? []) {
        const t = p?.text?.content;
        if (typeof t !== 'string' || !t.trim()) continue;
        if (known.has(t.trim())) continue;
        const adv = enJson[packFile[pack]]?.entries ?? {};
        const advKey = Object.keys(adv)[0];
        const enPage = adv[advKey]?.journals?.[j.name]?.pages?.[p.name];
        const cnPage = cnJson[packFile[pack]]?.entries?.[advKey]?.journals?.[j.name]?.pages?.[p.name];
        console.log('=' .repeat(70));
        console.log(`${pack}  journal="${j.name}"  page="${p.name}"  type=${p.type}  liveChars=${t.replace(/<[^>]+>/g, '').length}`);
        console.log(`   EN baseline has this journal? ${!!adv[advKey]?.journals?.[j.name]}   has this page? ${!!enPage}`);
        console.log(`   EN baseline page keys: ${enPage ? Object.keys(enPage).join(',') : '-'}`);
        console.log(`   CN pack page present? ${!!cnPage}  keys: ${cnPage ? Object.keys(cnPage).join(',') : '-'}`);
        if (enPage?.text) {
          const a = enPage.text.trim(), b = t.trim();
          console.log(`   baseline text len=${a.length} live len=${b.length}  equal=${a === b}`);
          let i = 0; while (i < Math.min(a.length, b.length) && a[i] === b[i]) i++;
          console.log(`   first divergence at ${i}:`);
          console.log('     baseline:', JSON.stringify(a.slice(Math.max(0, i - 60), i + 120)));
          console.log('     live    :', JSON.stringify(b.slice(Math.max(0, i - 60), i + 120)));
        } else {
          console.log('   live text head:', t.slice(0, 200).replace(/\n/g, ' '));
        }
      }
    }
  }
  await db.close();
}
