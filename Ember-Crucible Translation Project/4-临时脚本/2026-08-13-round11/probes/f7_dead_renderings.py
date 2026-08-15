# -*- coding: utf-8 -*-
"""Reverse scan: which glossary_ec keys still carry a Chinese rendering that has
ZERO occurrences anywhere in either pack's cn/ side?  Those are pure glossary
pollution -- apply_tm/fill_missing will re-inject a string the library never uses.
"""
import json, os, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from f7_gate import rows, ROOT  # noqa

DEAD = [
    "乌头", "海门", "希格纳", "法序议会", "奥术巨龙", "残酷巨龙", "卡达娜",
    "赫尔卡斯·格林", "离火之家", "碎齿", "碎牙者", "突变派", "嬗变师", "变异师",
    "变异学者", "突变剂师", "荆棘裔", "星兆", "维伦血统", "基瓦赫", "阿克登",
    "螯蛛以太兽", "印记者", "阿特西亚", "卡斯奇利安", "烬界", "安珀", "塞夫赫尔",
    "切夫赫尔", "阿纳克雷努姆", "阿纳克拉埃努姆", "鲁玛林", "丝珀特拉", "斯佩克特拉",
    "道岔", "轨道切换器", "突变药剂", "突变剂", "诱变剂",
]

R = rows()
cn_corpus_count = {t: 0 for t in DEAD}
for repo, fn, p, e, c in R:
    if not c:
        continue
    for t in DEAD:
        if t in c:
            cn_corpus_count[t] += 1

g = json.load(open(os.path.join(ROOT, "5-其他内容", "glossary",
                                "glossary_ec.json"), encoding="utf-8"))

print("rendering            pack_leaves  glossary_keys_carrying_it")
print("-" * 78)
for t in DEAD:
    keys = sorted(k for k, v in g.items() if t in v)
    print("%-18s %6d       %d" % (t, cn_corpus_count[t], len(keys)))
    for k in keys:
        print("        %-42r => %r" % (k, g[k]))
