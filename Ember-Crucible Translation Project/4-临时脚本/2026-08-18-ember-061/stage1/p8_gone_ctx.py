# -*- coding: utf-8 -*-
"""P8: 每个 gone 容器 + 同父的 added 容器清单（供人工判迁移/真删）。"""
import os, sys, collections, difflib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

delta = load_delta()
for f in FILES:
    gone, added = delta[f]["gone"], delta[f]["added"]
    if not gone:
        continue
    en060 = jload(os.path.join(EN060, f))
    en061 = jload(os.path.join(EN061, f))

    def build(ptrs, root, other):
        res = collections.defaultdict(dict)
        for ptr in ptrs:
            parts = split_path(ptr)
            for i in range(2, len(parts)):
                res['/'.join(str(x) for x in parts[:i])]['/'.join(str(x) for x in parts[i:])] = get_at(root, parts)
        full = {c: v for c, v in res.items() if not has_at(other, split_path('/' + c))}
        keys = sorted(full, key=len); out = []
        for k in keys:
            if not any(k.startswith(o + '/') for o in out):
                out.append(k)
        return {k: full[k] for k in out}

    G = build(gone, en060, en061)
    A = build(added, en061, en060)
    byparent = collections.defaultdict(list)
    for a in A:
        byparent[a.rsplit('/', 1)[0]].append(a.rsplit('/', 1)[1])
    print(f"\n\n@@@@@@@@@@ {f} @@@@@@@@@@")
    for g in sorted(G):
        par = g.rsplit('/', 1)[0]
        print(f"\n=== GONE 容器: {g.split('/',2)[-1]}  ({len(G[g])} 叶)")
        for rel, v in sorted(G[g].items()):
            print(f"    .{rel} = {repr(v)[:400]}")
        sib = byparent.get(par, [])
        print(f"    -- 同父新增容器 ({len(sib)}): {sorted(sib)[:25]}")
