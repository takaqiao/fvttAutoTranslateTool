/**
 * Round-22 probe: for every arrangement of every on-screen soundscape, dump
 *   <group label> | <arrangement label> | <first modules/ember/assets/... path found>
 * The asset path tells you what the track actually IS when the compendium prose
 * has no hit for the name (e.g. `Seven Sails` -> .../music/gravens-rest/...).
 *
 * Anti-空转: prints how many soundscapes resolved and how many arrangements were
 * walked; exits 2 if either is 0.
 */
import fs from 'fs';
import vm from 'node:vm';

const EMBER = 'C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/scripts/ember.mjs';
const src = fs.readFileSync(EMBER, 'utf8');

const reg = src.match(/var soundscapes=\/\*#__PURE__\*\/Object\.freeze\(\{__proto__:null,([^}]*)\}\)/);
if (!reg) { console.log('registry not found'); process.exit(2); }
const pairs = reg[1].split(',').map(s => s.trim()).filter(Boolean).map(s => {
  const [k, v] = s.split(':');
  return { k, v };
});

function blockAt(s, openIdx) {
  let d = 0, i = openIdx, q = null;
  for (; i < s.length; i++) {
    const c = s[i];
    if (q) { if (c === '\\') { i++; continue; } if (c === q) q = null; continue; }
    if (c === '"' || c === "'" || c === '`') { q = c; continue; }
    if (c === '{') d++;
    else if (c === '}') { d--; if (d === 0) return s.slice(openIdx, i + 1); }
  }
  return null;
}

let nSound = 0, nArr = 0;
const out = [];
for (const { k, v } of pairs) {
  const re = new RegExp(`(?:^|[;}])var ${v.replace(/\$/g, '\\$')} = \\{`, 'm');
  const m = re.exec(src);
  if (!m) continue;
  const body = blockAt(src, src.indexOf('{', m.index));
  if (!body) continue;
  let obj;
  try { obj = vm.runInNewContext('(' + body + ')'); } catch (e) { continue; }
  if (obj?.id !== k) continue;
  if (!['music', 'environment'].includes(obj.type)) continue;
  nSound++;
  for (const a of Object.values(obj.arrangements ?? {})) {
    nArr++;
    const j = JSON.stringify(a);
    const paths = [...j.matchAll(/"(modules\/ember\/assets\/[^"]+)"/g)].map(x => x[1]);
    out.push(`${obj.label} | ${a.label} | ${paths[0] ?? '(none)'}`);
  }
}
console.log(`soundscapes=${nSound} arrangements=${nArr}`);
if (!nSound || !nArr) process.exit(2);
fs.writeFileSync(new URL('./arr_src.txt', import.meta.url), out.join('\n') + '\n', 'utf8');
console.log('wrote arr_src.txt');
