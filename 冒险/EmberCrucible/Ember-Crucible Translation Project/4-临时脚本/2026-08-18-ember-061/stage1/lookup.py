# -*- coding: utf-8 -*-
"""术语查询：给英文词，打出库内 EN 叶与对应 CN 叶（用于沿用既定译名）。"""
import os, sys, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

term = sys.argv[1]
lim = int(sys.argv[2]) if len(sys.argv) > 2 else 6
rx = re.compile(term if '\\' in term or '(' in term else r'\b' + re.escape(term) + r'\b')
ALL = ["ember.crucible-adventure.json", "ember.adventure.json", "ember.crucible-character.json",
       "ember.crucible-effects.json", "ember.crucible-adversary.json", "ember.crucible-affixes.json",
       "ember.character.json", "ember.dnd5e-effects.json"]
n = 0
for f in ALL:
    pe = os.path.join(EN060, f); pc = os.path.join(CN, f)
    if not (os.path.exists(pe) and os.path.exists(pc)):
        continue
    e = jload(pe); c = jload(pc)
    for parts, v in walk_leaves(e):
        if isinstance(v, str) and rx.search(v):
            cv = get_at(c, list(parts))
            if isinstance(cv, str):
                m = rx.search(v)
                print(f"[{f.split('.')[1]}] {'/'.join(parts)[-70:]}")
                print("  EN:", v[max(0, m.start() - 90):m.end() + 90])
                print("  CN:", cv[:260])
                n += 1
                if n >= lim:
                    sys.exit(0)
