"""Check whether ember's custom JournalEntryPage subtypes carry translatable text
that the current extractor (text.content only) is missing."""
import json, subprocess, sys, os, collections, re

# Use node to dump one adventure doc's journal pages structure.
NODE = r"""
const {createRequire} = require('module');
const req = createRequire('C:/Users/Taka/Desktop/fvtt/package.json');
const {ClassicLevel} = req('classic-level');
(async () => {
  const db = new ClassicLevel(process.argv[2], {createIfMissing:false});
  const typeFields = {};
  const typeCount = {};
  let advCount = 0;
  for await (const [k,v] of db.iterator()) {
    const key = k.toString();
    if (!key.startsWith('!adventures!')) continue;
    advCount++;
    const doc = JSON.parse(v.toString());
    for (const j of (doc.journal||[])) {
      for (const p of (j.pages||[])) {
        const t = p.type || '?';
        typeCount[t] = (typeCount[t]||0)+1;
        const acc = typeFields[t] ||= {};
        const walk = (o, pre) => {
          if (o && typeof o === 'object' && !Array.isArray(o)) {
            for (const [kk,vv] of Object.entries(o)) walk(vv, pre?pre+'.'+kk:kk);
          } else if (Array.isArray(o)) {
            if (o.length && typeof o[0] === 'string') { acc[pre+'[]str'] = (acc[pre+'[]str']||0)+o.length; }
            else o.slice(0,3).forEach(x => walk(x, pre+'[]'));
          } else if (typeof o === 'string' && o.trim().length > 2) {
            acc[pre] = (acc[pre]||0)+1;
          }
        };
        walk(p.system||{}, 'system');
        if (p.text?.content) acc['text.content'] = (acc['text.content']||0)+1;
        if (p.name) acc['name'] = (acc['name']||0)+1;
      }
    }
  }
  await db.close();
  console.log(JSON.stringify({advCount, typeCount, typeFields}, null, 1));
})();
"""
tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_dumppages.cjs')
open(tmp, 'w', encoding='utf-8').write(NODE)
for pack in [r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember\packs\crucible-adventure"]:
    print("="*70); print(pack); print("="*70)
    r = subprocess.run([r'node', tmp, pack, ], capture_output=True, text=True, encoding='utf-8')
    if r.returncode != 0:
        print(r.stderr[:3000]); continue
    d = json.loads(r.stdout)
    print("adventures:", d['advCount'])
    print("page types:", json.dumps(d['typeCount'], indent=1))
    for t, fields in d['typeFields'].items():
        interesting = {k: v for k, v in fields.items()
                       if not re.search(r'(_id|uuid|img|src|icon|\.id$|type$|sort$|ownership|flags|\.key$|Class$|css)', k, re.I)}
        print(f"\n--- {t} ({d['typeCount'][t]} pages) ---")
        for k, v in sorted(interesting.items(), key=lambda x: -x[1])[:28]:
            print(f"   {k:<52} {v}")
