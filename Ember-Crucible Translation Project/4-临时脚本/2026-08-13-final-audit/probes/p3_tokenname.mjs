// Probe: every translation entry carrying a `tokenName` key, and whether its
// value differs from the sibling `name` key.
//
// Why it matters: babele 2.9.1 maps Actor.tokenName as
//   {path:'prototypeToken.name', converter:'name'}
// and the built-in `name` converter is Converters.mappedField("name") =
//   (…) => contextCompendium.translateField("name", data, runtime)
// i.e. it returns the translation of the `name` FIELD and never reads the
// `tokenName` key from the entry.
//
// False-positive modes: (a) an object that has `name`/`tokenName` but is not an
// Actor entry — impossible here, `tokenName` only exists on the Actor mapping;
// (b) files hand-edited outside the generator.
import fs from 'fs';
import path from 'path';

const ROOTS = [
  ['ember',    'C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/1-Ember汉化插件/compendium'],
  ['crucible', 'C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/2-Crucible汉化插件/compendium'],
];

const rows = [];
function walk(node, file, trail) {
  if (!node || typeof node !== 'object') return;
  if (!Array.isArray(node) && typeof node.tokenName === 'string') {
    rows.push({ file, at: trail.join('/'), name: node.name ?? null, tokenName: node.tokenName });
  }
  for (const [k, v] of Object.entries(node)) walk(v, file, [...trail, k]);
}

for (const [id, root] of ROOTS) {
  for (const lang of ['en', 'cn']) {
    const dir = path.join(root, lang);
    if (!fs.existsSync(dir)) continue;
    for (const f of fs.readdirSync(dir).filter((x) => x.endsWith('.json') && x !== '_source.json')) {
      walk(JSON.parse(fs.readFileSync(path.join(dir, f), 'utf8')), `${id}/${lang}/${f}`, []);
    }
  }
}

const cn = rows.filter((r) => r.file.includes('/cn/'));
const en = rows.filter((r) => r.file.includes('/en/'));
const cnDiff = cn.filter((r) => r.tokenName !== r.name);
console.log(`tokenName keys: en=${en.length} cn=${cn.length}`);
console.log(`cn entries where tokenName !== name: ${cnDiff.length}`);
const byFile = {};
for (const r of cnDiff) byFile[r.file] = (byFile[r.file] ?? 0) + 1;
console.log(JSON.stringify(byFile, null, 1));
console.log('--- samples ---');
for (const r of cnDiff.slice(0, 25)) console.log(`${r.file} :: ${r.at}\n    name      = ${JSON.stringify(r.name)}\n    tokenName = ${JSON.stringify(r.tokenName)}`);
const enDiff = en.filter((r) => r.tokenName !== r.name);
console.log(`\nEN baseline entries where tokenName !== name: ${enDiff.length}`);
for (const r of enDiff.slice(0, 20)) console.log(`${r.file} :: ${r.at} | name=${JSON.stringify(r.name)} tokenName=${JSON.stringify(r.tokenName)}`);
