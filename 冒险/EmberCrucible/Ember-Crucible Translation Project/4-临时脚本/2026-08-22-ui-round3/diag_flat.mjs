/** Diagnose the 447 vs 445 gap: locate every `"behaviors":[` array in the raw
 *  serialised pack and report the elements the scene tree-walk did NOT see. */
import { createRequire } from 'module';
import path from 'path';
const require = createRequire('C:/Users/Taka/Desktop/fvtt/package.json');
const { ClassicLevel } = require('classic-level');
const PKG = 'C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember';

for (const pack of ['adventure']) {
  const db = new ClassicLevel(path.join(PKG, 'packs', pack), { valueEncoding: 'json' });
  await db.open();
  const rows = [];
  for await (const [k, v] of db.iterator()) rows.push([k, v]);
  await db.close();

  const seen = new Set();
  for (const [, v] of rows) {
    for (const s of (Array.isArray(v?.scenes) ? v.scenes : [])) {
      for (const r of (s.regions ?? [])) for (const b of (r.behaviors ?? [])) seen.add(b._id);
    }
  }

  const raw = JSON.stringify(rows.map(([, v]) => v));
  const re = /"behaviors":\[/g;
  let m, total = 0;
  const extra = [];
  while ((m = re.exec(raw))) {
    let i = m.index + m[0].length - 1, depth = 0;
    const start = i;
    for (; i < raw.length; i++) {
      const c = raw[i];
      if (c === '"') { i++; while (i < raw.length && !(raw[i] === '"' && raw[i - 1] !== '\\')) i++; continue; }
      if (c === '[' || c === '{') depth++;
      else if (c === ']' || c === '}') { depth--; if (depth === 0) break; }
    }
    const arr = JSON.parse(raw.slice(start, i + 1));
    total += arr.length;
    for (const b of arr) if (!seen.has(b?._id)) extra.push({ ctx: raw.slice(Math.max(0, m.index - 260), m.index), b });
  }
  console.log(pack, 'flat total', total, 'treewalk', seen.size, 'extra', extra.length);
  for (const e of extra) console.log('  EXTRA:', JSON.stringify(e.b).slice(0, 300), '\n    CTX...', e.ctx.slice(-260));
}
