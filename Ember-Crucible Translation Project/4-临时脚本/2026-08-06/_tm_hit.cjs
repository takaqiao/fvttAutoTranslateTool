/**
 * How much of the ember adventure's actor-embedded item text can be filled for
 * free from existing crucible-cn + ember_cn translations (value-level TM)?
 */
const fs = require('fs'), path = require('path');
const {createRequire} = require('module');
const req = createRequire('C:/Users/Taka/Desktop/fvtt/package.json');
const {ClassicLevel} = req('classic-level');

const CJK = /[\u4e00-\u9fff]/;
const strip = s => String(s).replace(/<[^>]+>/g, ' ');
const norm = s => String(s).replace(/\s+/g, ' ').trim();
const get = (o, p) => p.split('.').reduce((a, k) => (a == null ? a : a[k]), o);

// --- build TM from an EN extraction dir + parallel CN dir ---
const TM = new Map();
function walk(en, cn, cb) {
  if (en && typeof en === 'object' && !Array.isArray(en)) {
    for (const k of Object.keys(en)) walk(en[k], cn?.[k], cb);
  } else if (Array.isArray(en)) {
    en.forEach((v, i) => walk(v, Array.isArray(cn) ? cn[i] : undefined, cb));
  } else if (typeof en === 'string' && typeof cn === 'string') {
    cb(en, cn);
  }
}
function addPair(enDir, cnDir) {
  if (!fs.existsSync(enDir) || !fs.existsSync(cnDir)) return;
  for (const f of fs.readdirSync(enDir).filter(x => x.endsWith('.json'))) {
    const cf = path.join(cnDir, f);
    if (!fs.existsSync(cf)) continue;
    let E, C;
    try { E = JSON.parse(fs.readFileSync(path.join(enDir, f), 'utf8')); C = JSON.parse(fs.readFileSync(cf, 'utf8')); }
    catch { continue; }
    walk(E.entries, C.entries, (e, c) => {
      if (CJK.test(c) && !TM.has(norm(e))) TM.set(norm(e), c);
    });
  }
}
const SP = process.argv[2];
addPair(path.join(SP, 'repo_crucible_cn/compendium/en'), path.join(SP, 'repo_crucible_cn/compendium/cn'));
addPair(path.join(SP, 'en_crucible_0101'), path.join(SP, 'repo_crucible_cn/compendium/cn'));

// ember_cn: name-keyed entries, harvest name pairs "中文 English"
for (const f of ['ember.crucible-character.json', 'ember.crucible-adversary.json']) {
  const p = path.join(SP, 'repo_ember_cn/compendium/cn', f);
  if (!fs.existsSync(p)) continue;
  const C = JSON.parse(fs.readFileSync(p, 'utf8'));
  for (const [k, v] of Object.entries(C.entries || {})) {
    if (v?.name && CJK.test(v.name)) TM.set(norm(k), v.name);
    const d = v?.descriptionpublic ?? v?.description;
    if (typeof d === 'string' && CJK.test(d)) { /* no EN side available */ }
  }
}
console.log(`TM size: ${TM.size} EN->CN pairs`);

(async () => {
  const db = new ClassicLevel(process.argv[3], {createIfMissing: false});
  const B = {};
  const bump = (b, s) => {
    if (typeof s !== 'string' || s.trim().length < 2) return;
    const x = B[b] ||= {tot: 0, hit: 0, ch: 0, hitCh: 0};
    x.tot++; const L = strip(s).length; x.ch += L;
    if (TM.has(norm(s))) { x.hit++; x.hitCh += L; }
  };
  for await (const [k, v] of db.iterator()) {
    if (!k.toString().startsWith('!adventures!')) continue;
    const adv = JSON.parse(v.toString());
    for (const a of (adv.actors || [])) {
      for (const it of (a.items || [])) {
        bump('actor.items.name', it?.name);
        const d = get(it, 'system.description');
        bump('actor.items.desc', typeof d === 'string' ? d : d?.public);
        if (typeof d === 'object') bump('actor.items.desc', d?.private);
        for (const act of (get(it, 'system.actions') || [])) {
          bump('actor.items.action.name', act?.name);
          bump('actor.items.action.desc', act?.description);
        }
      }
      for (const act of (get(a, 'system.actions') || [])) {
        bump('actor.actions.name', act?.name);
        bump('actor.actions.desc', act?.description);
      }
    }
  }
  await db.close();
  console.log(`${'bucket'.padEnd(26)}${'strings'.padStart(9)}${'TM hit'.padStart(8)}${'hit%'.padStart(6)}${'chars'.padStart(10)}${'chars hit'.padStart(11)}${'hit%'.padStart(6)}`);
  let T = [0, 0, 0, 0];
  for (const [k, x] of Object.entries(B)) {
    console.log(`${k.padEnd(26)}${String(x.tot).padStart(9)}${String(x.hit).padStart(8)}${String(Math.round(100 * x.hit / (x.tot || 1))).padStart(5)}%${String(x.ch).padStart(10)}${String(x.hitCh).padStart(11)}${String(Math.round(100 * x.hitCh / (x.ch || 1))).padStart(5)}%`);
    T = [T[0] + x.tot, T[1] + x.hit, T[2] + x.ch, T[3] + x.hitCh];
  }
  console.log(`${'TOTAL'.padEnd(26)}${String(T[0]).padStart(9)}${String(T[1]).padStart(8)}${String(Math.round(100 * T[1] / T[0])).padStart(5)}%${String(T[2]).padStart(10)}${String(T[3]).padStart(11)}${String(Math.round(100 * T[3] / T[2])).padStart(5)}%`);
  console.log(`\nRESIDUAL after TM: ${T[0] - T[1]} strings, ${T[2] - T[3]} chars`);
})();
