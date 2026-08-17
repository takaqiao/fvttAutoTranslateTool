# -*- coding: utf-8 -*-
"""P9: 不属于「整体消失容器」的那些 gone 叶（父容器还活着）。"""
import os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

delta = load_delta()
for f in FILES:
    gone, added = delta[f]["gone"], delta[f]["added"]
    if not gone:
        continue
    en060 = jload(os.path.join(EN060, f))
    en061 = jload(os.path.join(EN061, f))
    cn = jload(os.path.join(CN, f))

    def build(ptrs, root, other):
        res = collections.defaultdict(list)
        for ptr in ptrs:
            parts = split_path(ptr)
            for i in range(2, len(parts)):
                res['/'.join(str(x) for x in parts[:i])].append(ptr)
        full = {c: v for c, v in res.items() if not has_at(other, split_path('/' + c))}
        keys = sorted(full, key=len); out = []
        for k in keys:
            if not any(k.startswith(o + '/') for o in out):
                out.append(k)
        return {k: full[k] for k in out}

    G = build(gone, en060, en061)
    covered = set()
    for v in G.values():
        covered.update(v)
    orphans = [p for p in gone if p not in covered]
    print(f"\n##### {f}: 整体消失容器覆盖 {len(covered)} 叶 / 散叶 {len(orphans)} 叶 (合计 {len(gone)})")
    for p in sorted(orphans):
        parts = split_path(p)
        print(f"  {p}\n     EN060: {repr(get_at(en060, parts))[:300]}\n     CN   : {repr(get_at(cn, parts))[:300]}")
        # what else lives under the surviving parent in 061?
        par = parts[:-1]
        sib061 = get_at(en061, par)
        if isinstance(sib061, dict):
            print(f"     061 父下键: {sorted(sib061.keys())[:20]}")
