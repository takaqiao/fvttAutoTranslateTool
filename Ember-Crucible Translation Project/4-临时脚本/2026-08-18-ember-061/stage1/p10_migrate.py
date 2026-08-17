# -*- coding: utf-8 -*-
"""P10: 结构化迁移识别。
迁移候选 = gone 叶 g 与 added 叶 a 满足
  (S1) 路径同长且**恰好一个段不同**（改名/移位），或
  (S2) a == g + 一个尾段（结构下沉，如 description -> description/public），或
  (S3) g == a + 一个尾段（结构上提）
并且英文值 相等 或 difflib >= 0.90。
一对一贪心（先 S 强度、再值相似度）。
"""
import os, sys, difflib, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

delta = load_delta()
out = {}
for f in FILES:
    gone, added = delta[f]["gone"], delta[f]["added"]
    if not gone:
        out[f] = {"migrate": [], "delete": list(gone)}
        continue
    en060 = jload(os.path.join(EN060, f))
    en061 = jload(os.path.join(EN061, f))
    A = [(p, split_path(p), get_at(en061, split_path(p))) for p in added]
    by_len = collections.defaultdict(list)
    for p, sp, v in A:
        by_len[len(sp)].append((p, sp, v))

    cands = []
    for g in gone:
        gs = split_path(g)
        vg = get_at(en060, gs)
        if not isinstance(vg, str):
            continue
        # S1
        for p, sp, v in by_len.get(len(gs), []):
            diff = [i for i in range(len(gs)) if gs[i] != sp[i]]
            if len(diff) != 1:
                continue
            if not isinstance(v, str):
                continue
            r = 1.0 if v == vg else difflib.SequenceMatcher(None, vg, v).ratio()
            if r >= 0.90:
                cands.append((r, 'S1', g, p, diff[0]))
        # S2
        for p, sp, v in by_len.get(len(gs) + 1, []):
            if sp[:len(gs)] != gs or not isinstance(v, str):
                continue
            r = 1.0 if v == vg else difflib.SequenceMatcher(None, vg, v).ratio()
            if r >= 0.90:
                cands.append((r, 'S2', g, p, sp[-1]))
        # S3
        for p, sp, v in by_len.get(len(gs) - 1, []):
            if gs[:len(sp)] != sp or not isinstance(v, str):
                continue
            r = 1.0 if v == vg else difflib.SequenceMatcher(None, vg, v).ratio()
            if r >= 0.90:
                cands.append((r, 'S3', g, p, gs[-1]))
    cands.sort(key=lambda x: (-x[0], x[1]))
    ug, ua, mig = set(), set(), []
    for r, s, g, p, extra in cands:
        if g in ug or p in ua:
            continue
        ug.add(g); ua.add(p)
        mig.append({"src": g, "dst": p, "r": round(r, 4), "shape": s, "at": str(extra)})
    out[f] = {"migrate": mig, "delete": [g for g in gone if g not in ug]}
    print(f"\n##### {f}: 迁移 {len(mig)} / 真删 {len(out[f]['delete'])}")
    for m in sorted(mig, key=lambda x: x['src']):
        print(f"  [{m['shape']} r={m['r']}] {m['src'].split('/',3)[-1]}\n      -> {m['dst'].split('/',3)[-1]}")

t = sum(len(out[f]['migrate']) for f in FILES)
dd = sum(len(out[f]['delete']) for f in FILES)
print(f"\n合计 迁移 {t} / 真删 {dd} / 总 {t+dd}")
assert t + dd == 314
jdump(out, os.path.join(WORK, 'migrate.json'))
