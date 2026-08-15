#!/usr/bin/env node
/**
 * P21: dnd5e `system.details.biography.value` -- the GM/owner-side biography.
 *
 * mappings.mjs maps Actor.biography with `crucibleNested`, whose extractor
 * whitelists ['name','description','public','private','appearance'] only.
 * dnd5e's biography object is {value, public}: `public` is the players' blurb,
 * `value` is the full biography on the sheet's Biography tab. `value` is
 * therefore never extracted -> never in compendium/en -> outside the domain of
 * every gate the project runs.
 *
 * Quantify per pack, and check whether the same text is reachable via `public`.
 * READ ONLY.
 */
import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';
const { ClassicLevel } = createRequire('C:/Users/Taka/Desktop/fvtt/package.json')('classic-level');
const PKG = 'C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/packs';
const EN = 'C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/1-Ember汉化插件/compendium/en';

const known = new Set();
(function () {
  const collect = (o) => {
    if (o == null) return;
    if (Array.isArray(o)) return o.forEach(collect);
    if (typeof o === 'object') return Object.values(o).forEach(collect);
    if (typeof o === 'string') known.add(o.trim());
  };
  for (const f of fs.readdirSync(EN)) if (f.endsWith('.json')) collect(JSON.parse(fs.readFileSync(path.join(EN, f), 'utf8')));
})();

const plain = (s) => String(s).replace(/<[^>]+>/g, '').trim();
const per = {};
function look(pack, holder, actor) {
  const b = actor?.system?.details?.biography;
  if (!b || typeof b !== 'object') return;
  const v = typeof b.value === 'string' ? b.value : '';
  const pub = typeof b.public === 'string' ? b.public : '';
  const priv = typeof b.private === 'string' ? b.private : '';
  per[pack] ??= { actors: 0, withValue: 0, valueChars: 0, valueNotInBaseline: 0, notInChars: 0,
                  valueEqPublic: 0, withPublic: 0, withPrivate: 0, samples: [] };
  const r = per[pack];
  r.actors++;
  if (pub.trim()) r.withPublic++;
  if (priv.trim()) r.withPrivate++;
  if (!plain(v)) return;
  r.withValue++;
  r.valueChars += plain(v).length;
  if (v.trim() === pub.trim()) { r.valueEqPublic++; return; }
  if (!known.has(v.trim())) {
    r.valueNotInBaseline++;
    r.notInChars += plain(v).length;
    if (r.samples.length < 5) r.samples.push(`${holder}/${actor.name} (${plain(v).length} chars): ${plain(v).slice(0, 140)}`);
  }
}

for (const pack of fs.readdirSync(PKG).filter((d) => fs.statSync(path.join(PKG, d)).isDirectory())) {
  const db = new ClassicLevel(path.join(PKG, pack), { valueEncoding: 'json' });
  await db.open();
  for await (const [, doc] of db.iterator()) {
    if (Array.isArray(doc?.actors)) for (const a of doc.actors) look(pack, doc.name ?? '', a);
    if (doc?.system?.details?.biography) look(pack, '(top)', doc);
  }
  await db.close();
}
for (const [p, r] of Object.entries(per)) {
  console.log(`\n=== ${p}`);
  console.log(`  actors with a biography object: ${r.actors}  (public filled ${r.withPublic}, private filled ${r.withPrivate})`);
  console.log(`  biography.value filled: ${r.withValue}   total plain chars: ${r.valueChars}`);
  console.log(`  value identical to public (so already covered): ${r.valueEqPublic}`);
  console.log(`  value NOT reachable anywhere in the EN baseline: ${r.valueNotInBaseline}  (${r.notInChars} plain chars)`);
  for (const s of r.samples) console.log('    e.g.', s);
}
