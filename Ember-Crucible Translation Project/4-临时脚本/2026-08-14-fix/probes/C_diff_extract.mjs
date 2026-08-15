/** Leaf-level diff between the baseline and patched English extractions. */
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const flat = (node, trail, out) => {
  if (node === null || node === undefined) return;
  if (typeof node !== 'object') { out.set(trail.join(''), node); return; }
  for (const [k, v] of Object.entries(node)) flat(v, [...trail, k], out);
};

const added = new Map();   // leafKind -> count
const removed = new Map();
const changed = new Map();
let addedLeaves = 0; let removedLeaves = 0; let changedLeaves = 0;
const samples = { added: [], removed: [], changed: [] };

const kindOf = (k) => k.split('').filter((s) => !/^[A-Za-z0-9]{16}$/.test(s)).slice(-3).join('/');

for (const f of fs.readdirSync(path.join(HERE, 'out-base'))) {
  if (f === '_source.json') continue;
  const a = new Map(); const b = new Map();
  flat(JSON.parse(fs.readFileSync(path.join(HERE, 'out-base', f), 'utf8')), [], a);
  flat(JSON.parse(fs.readFileSync(path.join(HERE, 'out-cand', f), 'utf8')), [], b);
  for (const [k, v] of b) {
    if (!a.has(k)) {
      addedLeaves += 1;
      added.set(kindOf(k), (added.get(kindOf(k)) ?? 0) + 1);
      if (samples.added.length < 14) samples.added.push(`${f} :: ${k.replace(//g, '.')} = ${JSON.stringify(v)}`);
    } else if (a.get(k) !== v) {
      changedLeaves += 1;
      changed.set(kindOf(k), (changed.get(kindOf(k)) ?? 0) + 1);
      if (samples.changed.length < 10) samples.changed.push(`${f} :: ${k.replace(//g, '.')}`);
    }
  }
  for (const [k, v] of a) {
    if (!b.has(k)) {
      removedLeaves += 1;
      removed.set(kindOf(k), (removed.get(kindOf(k)) ?? 0) + 1);
      if (samples.removed.length < 20) samples.removed.push(`${f} :: ${k.replace(//g, '.')} = ${JSON.stringify(v)}`);
    }
  }
}

console.log(`ADDED   leaves: ${addedLeaves}`);
for (const [k, n] of [...added].sort((x, y) => y[1] - x[1])) console.log(`   ${String(n).padStart(5)}  …/${k}`);
console.log(`\nREMOVED leaves: ${removedLeaves}`);
for (const [k, n] of [...removed].sort((x, y) => y[1] - x[1])) console.log(`   ${String(n).padStart(5)}  …/${k}`);
console.log(`\nCHANGED leaves: ${changedLeaves}`);
for (const [k, n] of [...changed].sort((x, y) => y[1] - x[1])) console.log(`   ${String(n).padStart(5)}  …/${k}`);
console.log('\n--- added samples ---'); samples.added.forEach((s) => console.log('  ' + s));
if (samples.removed.length) { console.log('\n--- removed samples ---'); samples.removed.forEach((s) => console.log('  ' + s)); }
if (samples.changed.length) { console.log('\n--- changed samples ---'); samples.changed.forEach((s) => console.log('  ' + s)); }
