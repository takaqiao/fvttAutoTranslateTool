# -*- coding: utf-8 -*-
"""补扫第一版 FIELDS 漏掉的展示字段名（header / heading / subtitle / prompt / note / warning）。"""
import io, re, sys, json, collections, bisect
sys.stdout.reconfigure(encoding='utf-8')
BS = chr(92)
STR = '"((?:[^"' + BS + BS + ']|' + BS + BS + '.)*)"'
RE = re.compile(r'\b(header|heading|subtitle|prompt|note|warning|blurb|desc)\s*:\s*' + STR)
RE_TOP = re.compile(r'^(?:export\s+)?(?:const|let|var|class|function|async function)\s+([A-Za-z_$][\w$]*)', re.M)
txt = io.open(r"C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/scripts/ember.mjs", encoding='utf-8').read()
tops = sorted((m.start(), m.group(1)) for m in RE_TOP.finditer(txt)); starts = [t[0] for t in tops]
own = lambda p: tops[bisect.bisect_right(starts, p) - 1][1]
RE_KEY = re.compile(r'^[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)+$')
out = collections.defaultdict(list)
seen = set()
for m in RE.finditer(txt):
    v = m.group(2).strip()
    if not v or v in seen or RE_KEY.match(v) or not re.search(r'[A-Za-z]', v): continue
    if re.match(r'^[a-z][A-Za-z0-9]*$', v): continue
    seen.add(v); out[own(m.start())].append(v)
tot = sum(len(x) for x in out.values())
print(f'补扫到 {tot} 条（{len(out)} 个块）\n')
for o, vs in sorted(out.items(), key=lambda kv: -len(kv[1]))[:18]:
    print(f'  [{len(vs):>3}] {o}')
    print(f'        {" · ".join(vs[:5])[:130]}')
io.open('extra_fields.json','w',encoding='utf-8').write(json.dumps({k: v for k, v in out.items()}, ensure_ascii=False, indent=1))
