# -*- coding: utf-8 -*-
"""P3: 用「容器整体搬家」的信号做迁移识别，而不是单叶等值。

容器 = 一个实体（journals/X/pages/Y、actors/A/items/I、scenes/S、items/I …）。
gone 容器：该容器下**全部**叶都在 gone 里（整体消失）；
added 容器：该容器下**全部**叶都在 added 里（整体新出现）。
再按「叶相对路径集合 + 值多重集」做相似度配对。
"""
import os, sys, collections, difflib, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

delta = load_delta()


def containers_of(ptrs, root):
    """把叶路径按所有前缀切成候选容器，返回 {容器: {相对路径: 值}}"""
    res = collections.defaultdict(dict)
    for ptr in ptrs:
        parts = split_path(ptr)
        for i in range(2, len(parts)):
            cpath = '/'.join(str(x) for x in parts[:i])
            rel = '/'.join(str(x) for x in parts[i:])
            res[cpath][rel] = get_at(root, parts)
    return res


for f in FILES:
    gone = delta[f]["gone"]
    added = delta[f]["added"]
    if not gone:
        continue
    en060 = jload(os.path.join(EN060, f))
    en061 = jload(os.path.join(EN061, f))
    gc = containers_of(gone, en060)
    ac = containers_of(added, en061)

    # 只保留「整体消失/整体新增」的容器：容器在对面根里完全不存在
    gfull = {c: v for c, v in gc.items() if not has_at(en061, split_path('/' + c))}
    afull = {c: v for c, v in ac.items() if not has_at(en060, split_path('/' + c))}
    # 取极大容器（不是别的 gfull 容器的子路径）
    def maximal(d):
        keys = sorted(d, key=len)
        out = []
        for k in keys:
            if not any(k.startswith(o + '/') for o in out):
                out.append(k)
        return {k: d[k] for k in out}
    gfull = maximal(gfull)
    afull = maximal(afull)
    print(f"\n##### {f}: gone叶 {len(gone)} / 整体消失容器 {len(gfull)} | added叶 {len(added)} / 整体新增容器 {len(afull)}")
    print("  -- 整体消失容器 --")
    for c in sorted(gfull):
        print(f"     {c}  ({len(gfull[c])} 叶)")
    print(f"  -- 整体新增容器 (前 60) --")
    for c in sorted(afull)[:60]:
        print(f"     {c}  ({len(afull[c])} 叶)")
    if len(afull) > 60:
        print(f"     ... 共 {len(afull)}")
