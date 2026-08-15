# -*- coding: utf-8 -*-
"""1:2 撞名检查：本文件里同一个中文串是否被两个不同英文键使用。

只看代码，不看注释（块注释里有举例，算进来就是假阳性）。
落盘运行，不走 -c，避免反斜杠被外壳吃掉（第十九轮踩过）。
自报扫了多少对；扫到 0 对 = 判据坏了，直接退出。
"""
import re
import collections
import json
import sys

HC = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\1-Ember汉化插件\scripts\ember-hardcoded-cn.mjs"
DATA = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\4-临时脚本\2026-08-15-round21\soundscapes.json"

src = open(HC, encoding="utf-8").read()
noc = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
noc = re.sub(r"//[^\n]*", "", noc)

PAIR = re.compile(r'"([^"\\]+)"\s*:\s*"([^"\\]+)"')
pairs = PAIR.findall(noc)
print("pairs scanned:", len(pairs))
if len(pairs) < 500:
    print("JUDGE BROKEN: too few pairs, regex is not matching the file")
    sys.exit(2)

# sanity: the regex must find a known pair
if ("Water Temple", "水之神殿") not in pairs:
    print("JUDGE BROKEN: known pair not found")
    sys.exit(2)

bycn = collections.defaultdict(set)
for en, cn in pairs:
    bycn[cn].add(en)

groups = json.load(open(DATA, encoding="utf-8"))["groupLabels"]
newcn = sorted({cn for en, cn in pairs if en in groups})
print("distinct CN values contributed by the 42 group names:", len(newcn))

bad = 0
for cn in newcn:
    ens = bycn[cn]
    if len(ens) > 1:
        bad += 1
        print("  COLLISION", cn, "<-", sorted(ens))
print("collisions:", bad)

# also: two group names must not share one CN string
g2 = collections.Counter(cn for en, cn in pairs if en in groups)
dupe = [c for c, n in g2.items() if n > 1]
print("CN strings used by more than one group name:", len(dupe), dupe)
sys.exit(1 if (bad or dupe) else 0)
