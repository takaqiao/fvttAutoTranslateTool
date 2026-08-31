# -*- coding: utf-8 -*-
"""P16: 用 EN061 当英文侧，找同英文不同中文的分叉组（same_en_split 口径的简化版）。"""
import os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

PATS = sys.argv[1:] or ['lashes out with an amorphous', 'revel in the thrill of combat', 'Shard Gods are mortal ascendants']
by_en = collections.defaultdict(dict)
for f in FILES:
    en = jload(os.path.join(EN061, f))
    cn = jload(os.path.join(CN, f))
    for parts, v in walk_leaves(en):
        if not isinstance(v, str) or not v.strip():
            continue
        c = get_at(cn, list(parts))
        if isinstance(c, str) and c.strip():
            by_en[v].setdefault(c, []).append((f, '/'.join(parts)))

for p in PATS:
    for envals, cnmap in by_en.items():
        if p in envals and len(cnmap) > 1:
            print('\n#### EN:', envals[:150])
            for c, locs in cnmap.items():
                print(f"  CN({len(locs)}): {c[:160]}")
                for f, pa in locs[:6]:
                    print(f"      [{f.split('.')[1]}] {pa[-90:]}")
