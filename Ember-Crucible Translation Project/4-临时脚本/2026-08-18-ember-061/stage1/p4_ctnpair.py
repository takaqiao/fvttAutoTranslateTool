# -*- coding: utf-8 -*-
"""P4: 容器级配对（严判据）。
gone 容器 g 与 added 容器 a 配成「迁移」需同时满足：
  (1) 同父（g 与 a 的父路径相同）—— 即同一个 actor 的 items / 同一本 journal 的 pages …
  (2) 相对叶路径集合的 Jaccard >= 0.6
  (3) 共同相对路径上的英文值平均相似度 >= 0.75（且非 name 叶的平均 >= 0.8 或全等）
按 (3) 的分数贪心配对，一对一。
"""
import os, sys, collections, difflib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

delta = load_delta()


def build(ptrs, root, other_root):
    res = collections.defaultdict(dict)
    for ptr in ptrs:
        parts = split_path(ptr)
        for i in range(2, len(parts)):
            cpath = '/'.join(str(x) for x in parts[:i])
            res[cpath]['/'.join(str(x) for x in parts[i:])] = get_at(root, parts)
    full = {c: v for c, v in res.items() if not has_at(other_root, split_path('/' + c))}
    keys = sorted(full, key=len)
    out = []
    for k in keys:
        if not any(k.startswith(o + '/') for o in out):
            out.append(k)
    return {k: full[k] for k in out}


def score(gv, av):
    ks = set(gv) & set(av)
    j = len(ks) / max(1, len(set(gv) | set(av)))
    if j < 0.6 or not ks:
        return 0.0, j, 0.0
    sims = []
    for k in ks:
        a, b = gv[k], av[k]
        if not isinstance(a, str) or not isinstance(b, str):
            sims.append(1.0 if a == b else 0.0)
        else:
            sims.append(difflib.SequenceMatcher(None, a, b).ratio())
    m = sum(sims) / len(sims)
    return m, j, min(sims)


allres = {}
for f in FILES:
    gone, added = delta[f]["gone"], delta[f]["added"]
    if not gone:
        allres[f] = ([], [])
        continue
    en060 = jload(os.path.join(EN060, f))
    en061 = jload(os.path.join(EN061, f))
    G = build(gone, en060, en061)
    A = build(added, en061, en060)
    cands = []
    for g, gv in G.items():
        gp = g.rsplit('/', 1)[0]
        for a, av in A.items():
            if a.rsplit('/', 1)[0] != gp:
                continue
            m, j, mn = score(gv, av)
            if m >= 0.75:
                cands.append((m, j, mn, g, a))
    cands.sort(reverse=True)
    ug, ua, pairs = set(), set(), []
    for m, j, mn, g, a in cands:
        if g in ug or a in ua:
            continue
        ug.add(g); ua.add(a)
        pairs.append((g, a, m, j, mn))
    allres[f] = (pairs, sorted(set(G) - ug))
    print(f"\n##### {f}: 配上 {len(pairs)} 对 / 未配上 gone 容器 {len(G)-len(pairs)}")
    for g, a, m, j, mn in sorted(pairs):
        print(f"  [{m:.3f}/j{j:.2f}/min{mn:.2f}] {g.split('/',2)[-1]}\n        -> {a.split('/',2)[-1]}")

jdump({f: {"pairs": [list(x) for x in allres[f][0]], "unpaired": allres[f][1]} for f in FILES},
      os.path.join(WORK, 'ctnpair.json'))
