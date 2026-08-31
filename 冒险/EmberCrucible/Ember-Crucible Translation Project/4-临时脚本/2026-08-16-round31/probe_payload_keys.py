# -*- coding: utf-8 -*-
"""探针：逐 kind 数出每条断言的载荷键与长度。

⚠ 硬约束 4：**先自证**。切出来的条数必须 == 已知真值（66 条 / 22 种 kind），
  否则直接 abort —— 不许在一个切错的清单上得结论。
"""
import json, os, sys, collections

sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
RULES = os.path.join(ROOT, "5-其他内容", "RESOLUTIONS.assertions.json")

A = json.load(open(RULES, encoding="utf-8"))["assertions"]

# ---- 前置自证 ----
TRUTH_N, TRUTH_K = 66, 22
kinds = sorted({r.get("kind") for r in A})
assert len(A) == TRUTH_N, f"条数 {len(A)} != 真值 {TRUTH_N}"
assert len(kinds) == TRUTH_K, f"kind 数 {len(kinds)} != 真值 {TRUTH_K}"
print(f"前置自证 ok：{len(A)} 条 / {len(kinds)} 种 kind\n")

# 非载荷的元数据键（不构成「判了几条规矩」的内容）
META = {"id", "kind", "title", "decision", "why", "note", "_why", "rule"}

by_kind = collections.defaultdict(list)
for r in A:
    by_kind[r["kind"]].append(r)

for k in kinds:
    rs = by_kind[k]
    print(f"=== {k}  ({len(rs)} 条)")
    # 统计每个键在这一类里的出现次数与长度分布
    keyinfo = collections.defaultdict(list)
    for r in rs:
        for key, v in r.items():
            if key in META:
                continue
            if isinstance(v, (list, dict)):
                keyinfo[key].append((r["id"], len(v)))
            else:
                keyinfo[key].append((r["id"], repr(v)))
    for key in sorted(keyinfo):
        vals = keyinfo[key]
        print(f"    {key:24s} 出现 {len(vals)}/{len(rs)} 条  " + ", ".join(
            f"{i}={n}" for i, n in vals))
    print()
