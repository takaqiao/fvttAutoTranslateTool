# -*- coding: utf-8 -*-
"""P1: 删除桶盘点 —— 按容器分组，打出 EN060 值 + 现有 CN 译文。"""
import os, sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

delta = load_delta()
out = []
for f in FILES:
    d = delta[f]["gone"]
    if not d:
        continue
    en060 = jload(os.path.join(EN060, f))
    cn = jload(os.path.join(CN, f))
    out.append(f"\n########## {f}  ({len(d)} gone) ##########")
    # group by container = path minus last 1 or 2 segments
    groups = collections.defaultdict(list)
    for ptr in d:
        parts = split_path(ptr)
        # container: up to the entity level
        groups['/'.join(str(x) for x in parts[:4])].append(ptr)
    for g in sorted(groups):
        out.append(f"\n--- [{g}]  {len(groups[g])} ---")
        for ptr in sorted(groups[g]):
            parts = split_path(ptr)
            e = get_at(en060, parts)
            c = get_at(cn, parts)
            out.append(f"  {ptr}")
            out.append(f"     EN060: {repr(e)[:300]}")
            out.append(f"     CN   : {repr(c)[:300]}")

txt = '\n'.join(out)
open(os.path.join(WORK, 'gone_survey.txt'), 'w', encoding='utf-8').write(txt)
print(txt[:200])
print(f"\n(written {len(txt)} chars to gone_survey.txt)")
