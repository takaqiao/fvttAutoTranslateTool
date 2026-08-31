# -*- coding: utf-8 -*-
"""查 pf2cn 中 Nagaji Venom / Titan Wrestler / Pinning Shot 的 canonical"""
import json, os

PF2CN_PATHS = [
    r"C:\Users\Taka\Desktop\fvtt\system\pf2_cn\zh_Hans\zh_Hans.json",
    r"C:\Users\Taka\Desktop\fvtt\system\sf2_cn\zh_Hans\zh_Hans.json",
]

queries = [
    ("nagaji venom", ["娜迦毒液", "娜迦裔毒液"]),
    ("titan wrestler", ["巨人摔角术", "巨人摔角手", "巨人摔跤"]),
    ("pinning shot", ["固定射击", "锚定射击", "钉刺射"]),
    ("ki adept", ["内家好手", "气功好手", "气功精通"]),
]

# 也查 锦标赛 vs 武道会
queries.append(("tournament", ["锦标赛", "武道会"]))

for path in PF2CN_PATHS:
    if not os.path.exists(path):
        print(f"!! {path} missing")
        continue
    print(f"\n=== {path} ===")
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    # query 用 string substr 找 hit count
    for en_query, candidates in queries:
        # 查英文 key 出现处
        en_count = raw.lower().count(en_query.lower())
        print(f"\n  Query: {en_query} (en hits: {en_count})")
        for cand in candidates:
            n = raw.count(cand)
            print(f"    {cand:18}: {n} hits")
