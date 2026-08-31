# -*- coding: utf-8 -*-
"""把 ember 的硬编码展示串按**声明它的那个顶层结构**归类。

做法：每个命中位置往回找最近的顶层声明行（行首的 const/var/class/function），
拿那个名字当桶名。这样「资产目录里的贴图名」和「真 UI 文案」不会混在一个数里。
"""
import io, re, sys, json, collections
sys.stdout.reconfigure(encoding='utf-8')
BS = chr(92)
STR = '"((?:[^"' + BS + BS + ']|' + BS + BS + '.)*)"'
FIELDS = ["label", "title", "hint", "tooltip", "placeholder", "content", "text", "message",
          "legend", "caption", "summary"]
RE_FIELD = re.compile(r'\b(' + '|'.join(FIELDS) + r')' + r'\s*:\s*' + STR)
RE_TOP = re.compile(r'^(?:export\s+)?(?:const|let|var|class|function|async function)\s+([A-Za-z_$][\w$]*)', re.M)

path = r"C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/scripts/ember.mjs"
txt = io.open(path, encoding='utf-8').read()
tops = [(m.start(), m.group(1)) for m in RE_TOP.finditer(txt)]
tops.sort()
starts = [t[0] for t in tops]
import bisect
def owner(pos):
    i = bisect.bisect_right(starts, pos) - 1
    return tops[i][1] if i >= 0 else "(顶层之外)"

missing = set(json.load(io.open('coverage.json', encoding='utf-8'))["ember JS·展示字段"]["missing"])
buckets = collections.Counter()
examples = collections.defaultdict(list)
seen = set()
for m in RE_FIELD.finditer(txt):
    v = m.group(2)
    if v not in missing or v in seen: continue
    seen.add(v)
    o = owner(m.start())
    buckets[o] += 1
    if len(examples[o]) < 4: examples[o].append(v)

print(f"未覆盖的唯一串 {len(missing)}，归入 {len(buckets)} 个声明块\n")
tot = 0
for o, c in buckets.most_common(30):
    tot += c
    print(f"  {c:>5}  {o:<34}{' · '.join(x[:24] for x in examples[o])[:78]}")
print(f"\n  前 30 桶合计 {tot} / {len(missing)}")
io.open('partition.json', 'w', encoding='utf-8').write(json.dumps(
    {"buckets": dict(buckets.most_common()), "examples": {k: v for k, v in examples.items()}},
    ensure_ascii=False, indent=1))
