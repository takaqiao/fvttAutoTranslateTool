# -*- coding: utf-8 -*-
"""反向闸：42 条新组名译文里，有没有哪一条在合集里**已经**是另一个英文串的定稿译名？

对每个中文串 C，找出所有 CN 叶恰好等于 C 的位置，报出配对的 EN 叶。
若配对的 EN 与我们的键不同 → 1:2 撞名候选，人工判。
自报扫了多少叶；0 叶 = 语料没读进来。
"""
import json
import os
import sys

BASE = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\1-Ember汉化插件\compendium"
DATA = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\4-临时脚本\2026-08-15-round21\soundscapes.json"
HC = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\1-Ember汉化插件\scripts\ember-hardcoded-cn.mjs"


def leaves(o, path=()):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, path + (str(k),))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from leaves(v, path + (str(i),))
    elif isinstance(o, str):
        yield path, o


import re
src = open(HC, encoding="utf-8").read()
block = src[src.index("const SOUNDSCAPE_GROUPS = {"):]
block = block[:block.index("\n};")]
table = dict(re.findall(r'"([^"\\]+)"\s*:\s*"([^"\\]+)"', block))
print("table rows read from SOUNDSCAPE_GROUPS:", len(table))
if len(table) != 42:
    print("JUDGE BROKEN: expected 42 rows")
    sys.exit(2)

pairs = []
for fn in sorted(os.listdir(os.path.join(BASE, "cn"))):
    enp = os.path.join(BASE, "en", fn)
    if not fn.endswith(".json") or not os.path.exists(enp):
        continue
    cn = json.load(open(os.path.join(BASE, "cn", fn), encoding="utf-8"))
    en = json.load(open(enp, encoding="utf-8"))
    end = dict(leaves(en))
    for p, s in leaves(cn):
        pairs.append((s, end.get(p)))
print("cn leaves paired:", len(pairs))
if not pairs:
    print("NO-CORPUS")
    sys.exit(2)

# a leaf may be "中文 English" (bilingual tail) — strip the tail before comparing
def head(s):
    return s.split(" ")[0] if s else s

flag = 0
for k, v in sorted(table.items()):
    hits = {}
    for c, e in pairs:
        if e is None:
            continue
        c2 = c.strip()
        if c2 == v or c2 == f"{v} {e}":
            hits.setdefault(e, 0)
            hits[e] += 1
    others = {e: n for e, n in hits.items() if e != k}
    tag = ""
    if others:
        flag += 1
        tag = "  <<< OTHER EN USES THIS CN: " + json.dumps(others, ensure_ascii=False)
    print(f"  {k:<30} {v:<12} exact-leaf hits={hits.get(k, 0)}{tag}")
print("rows whose CN is already another EN string's rendering:", flag)
sys.exit(0)
