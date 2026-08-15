#!/usr/bin/env node
/**
 * P14: which GM-only Ember fields never enter the translation pipeline at all?
 *
 * `extract_en.mjs` builds compendium/en by INTERPRETING mappings.mjs. So the
 * project's English baseline contains exactly the fields the mapping asks for.
 * Every existing gate (coverage, missing-key, drift) compares CN against that
 * baseline -- which means a field the mapping never lists is invisible to all
 * of them by construction. Ember's own module.json declares GM-shaped fields:
 *   htmlFields: [... "development.secrets"]   on 11 page subtypes
 *   gmOnlyFields: ["hooks"]                   on questEvent / standaloneEvent
 * Neither string appears anywhere in mappings.mjs.
 *
 * Read the live LevelDB packs and report how much prose actually sits there.
 * READ ONLY.
 */
import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';

const { ClassicLevel } = createRequire('C:/Users/Taka/Desktop/fvtt/package.json')('classic-level');

const PKG = 'C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/packs';
const packs = fs.readdirSync(PKG).filter((d) => fs.statSync(path.join(PKG, d)).isDirectory());

const strip = (s) => String(s).replace(/<[^>]+>/g, '').trim();

const report = {};
function note(bucket, pack, key, subtype, field, val) {
  const t = strip(val);
  if (!t) return;
  report[field] ??= { docs: 0, chars: 0, subtypes: {}, samples: [] };
  const r = report[field];
  r.docs++;
  r.chars += t.length;
  r.subtypes[subtype] = (r.subtypes[subtype] || 0) + 1;
  if (r.samples.length < 4) r.samples.push({ pack, key, subtype, text: String(val).slice(0, 300) });
}

function scanPage(pack, key, page) {
  const sub = page?.type ?? '?';
  const sys = page?.system ?? {};
  const dev = sys.development;
  if (dev && typeof dev === 'object') {
    for (const [k, v] of Object.entries(dev)) {
      if (typeof v === 'string') note('dev', pack, key, sub, `system.development.${k}`, v);
    }
  }
  if (typeof sys.hooks === 'string') note('hooks', pack, key, sub, 'system.hooks', sys.hooks);
  else if (Array.isArray(sys.hooks)) {
    sys.hooks.forEach((h, i) => {
      if (typeof h === 'string') note('hooks', pack, key, sub, 'system.hooks[]', h);
      else if (h && typeof h === 'object') {
        for (const [k, v] of Object.entries(h)) {
          if (typeof v === 'string' && v.trim()) note('hooks', pack, key, sub, `system.hooks[].${k}`, v);
        }
      }
    });
  } else if (sys.hooks && typeof sys.hooks === 'object') {
    for (const [k, v] of Object.entries(sys.hooks)) {
      if (typeof v === 'string') note('hooks', pack, key, sub, `system.hooks.${k}`, v);
    }
  }
  // any other system field whose name smells GM-only
  for (const [k, v] of Object.entries(sys)) {
    if (typeof v === 'string' && /secret|gm|gamemaster|hook|private/i.test(k)) {
      note('other', pack, key, sub, `system.${k}`, v);
    }
  }
}

function scanDoc(pack, key, doc) {
  if (Array.isArray(doc?.pages)) for (const p of doc.pages) scanPage(pack, key, p);
  if (doc?.type && doc?.system && doc?.text !== undefined) scanPage(pack, key, doc);
  // Adventure documents embed journals
  if (Array.isArray(doc?.journal)) {
    for (const j of doc.journal) if (Array.isArray(j.pages)) for (const p of j.pages) scanPage(pack, j.name + '/' + (p.name ?? ''), p);
  }
}

for (const pack of packs) {
  const db = new ClassicLevel(path.join(PKG, pack), { valueEncoding: 'json' });
  await db.open();
  for await (const [k, v] of db.iterator()) scanDoc(pack, k, v);
  await db.close();
}

for (const [field, r] of Object.entries(report).sort((a, b) => b[1].chars - a[1].chars)) {
  console.log(`\n### ${field}`);
  console.log(`   docs with content: ${r.docs}   total plain chars: ${r.chars}`);
  console.log(`   subtypes: ${JSON.stringify(r.subtypes)}`);
  for (const s of r.samples) console.log(`   e.g. [${s.subtype}] ${s.key}\n        ${s.text.replace(/\n/g, ' ')}`);
}
if (!Object.keys(report).length) console.log('no GM-shaped unmapped field carried content');
