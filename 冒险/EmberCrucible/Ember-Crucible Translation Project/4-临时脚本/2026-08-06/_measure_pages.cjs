const {createRequire} = require('module');
const req = createRequire('C:/Users/Taka/Desktop/fvtt/package.json');
const {ClassicLevel} = req('classic-level');

// Fields that hold real prose on ember's custom JournalEntryPage subtypes.
const PROSE = [
  'overview', 'exposition', 'summary', 'subtitle', 'pronunciation',
  'content.overview', 'content.gamemaster', 'development.secrets',
  'banner.caption', 'height', 'lifespan', 'origin', 'terrain'
];
const strip = s => String(s).replace(/<[^>]+>/g, ' ');
const get = (o, p) => p.split('.').reduce((a, k) => (a == null ? a : a[k]), o);

(async () => {
  const packs = process.argv.slice(2);
  for (const pack of packs) {
    const db = new ClassicLevel(pack, {createIfMissing: false});
    let textChars = 0, textCount = 0;
    const proseChars = {}, proseCount = {};
    let outcomeChars = 0, outcomeCount = 0;
    let dupSame = 0, dupDiff = 0;
    for await (const [k, v] of db.iterator()) {
      if (!k.toString().startsWith('!adventures!')) continue;
      const doc = JSON.parse(v.toString());
      for (const j of (doc.journal || [])) {
        for (const p of (j.pages || [])) {
          if (p.text?.content) { textChars += strip(p.text.content).length; textCount++; }
          for (const f of PROSE) {
            const val = get(p.system || {}, f);
            if (typeof val === 'string' && val.trim().length > 2) {
              proseChars[f] = (proseChars[f] || 0) + strip(val).length;
              proseCount[f] = (proseCount[f] || 0) + 1;
              if (f === 'overview' || f === 'content.overview') {
                if (p.text?.content && strip(p.text.content).trim() === strip(val).trim()) dupSame++;
                else dupDiff++;
              }
            }
          }
          for (const o of (p.system?.outcomes || [])) {
            for (const f of ['label', 'summary']) {
              if (typeof o?.[f] === 'string' && o[f].trim().length > 2) {
                outcomeChars += strip(o[f]).length; outcomeCount++;
              }
            }
          }
        }
      }
    }
    await db.close();
    console.log('==== ' + pack.split(/[\\/]/).pop() + ' ====');
    console.log(`text.content            : ${textCount} strings, ${textChars} chars  [CURRENTLY TRANSLATED]`);
    let tot = 0, cnt = 0;
    for (const f of PROSE) {
      if (!proseCount[f]) continue;
      console.log(`system.${f.padEnd(23)}: ${String(proseCount[f]).padStart(4)} strings, ${String(proseChars[f]).padStart(8)} chars  [NOT REACHABLE]`);
      tot += proseChars[f]; cnt += proseCount[f];
    }
    console.log(`system.outcomes[].*     : ${outcomeCount} strings, ${outcomeChars} chars  [NOT REACHABLE]`);
    tot += outcomeChars; cnt += outcomeCount;
    console.log(`--> MISSING TOTAL       : ${cnt} strings, ${tot} chars`);
    console.log(`overview vs text.content: identical=${dupSame} different=${dupDiff}`);
    console.log();
  }
})();
