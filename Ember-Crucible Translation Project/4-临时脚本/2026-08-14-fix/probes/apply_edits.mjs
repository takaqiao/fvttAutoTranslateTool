import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = 'C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project/';
const CAND = {
  '3-常用脚本/extract/mappings.mjs': 'mappings.mjs',
  '3-常用脚本/release/runtime-converters.js': 'runtime-converters.js',
  '3-常用脚本/extract/extract_en.mjs': 'extract_en.mjs',
};

const edits = JSON.parse(fs.readFileSync(path.join(HERE, 'edits.json'), 'utf8'));

// 1) uniqueness of every `old` against the PRISTINE source
let ok = true;
for (const e of edits) {
  const orig = fs.readFileSync(ROOT + e.file, 'utf8');
  const n = orig.split(e.old).length - 1;
  if (n !== 1) ok = false;
  console.log(`${n === 1 ? 'OK  ' : 'FAIL'} count=${n}  [${path.basename(e.file)}] ${e.sig}`);
}
if (!ok) { console.error('anchors not unique'); process.exit(1); }

// 2) apply to candidate copies (fresh from source each run)
for (const [src, dst] of Object.entries(CAND)) {
  fs.copyFileSync(ROOT + src, path.join(HERE, 'cand', dst));
}
for (const e of edits) {
  const p = path.join(HERE, 'cand', CAND[e.file]);
  const s = fs.readFileSync(p, 'utf8');
  const n = s.split(e.old).length - 1;
  if (n !== 1) { console.error(`APPLY FAIL count=${n} ${e.sig}`); process.exit(1); }
  fs.writeFileSync(p, s.replace(e.old, e.new), 'utf8');
}
console.log('\nall edits applied to probes/cand/');
