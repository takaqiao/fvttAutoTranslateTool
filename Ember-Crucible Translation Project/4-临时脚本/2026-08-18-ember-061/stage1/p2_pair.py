# -*- coding: utf-8 -*-
"""P2: 删除桶 vs 新增桶 内容相似度配对 —— 分「迁移」与「真删」。

配对判据（从严到宽）：
  T1 EXACT      : EN060(gone) == EN061(added)，且末段 key 相同
  T2 EXACT-anyk : EN060(gone) == EN061(added)，末段 key 不同（改结构）
  T3 NEAR       : 末段 key 相同 且 difflib ratio >= 0.86
  否则 真删
"""
import os, sys, collections, difflib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

NEAR = 0.86
delta = load_delta()
report = {}

for f in FILES:
    gone = delta[f]["gone"]
    added = delta[f]["added"]
    if not gone:
        report[f] = {"migrated": [], "deleted": []}
        continue
    en060 = jload(os.path.join(EN060, f))
    en061 = jload(os.path.join(EN061, f))

    # index added by value
    by_val = collections.defaultdict(list)
    add_items = []
    for ptr in added:
        v = get_at(en061, split_path(ptr))
        if isinstance(v, str):
            add_items.append((ptr, v))
            by_val[v].append(ptr)

    used = set()
    migrated, deleted = [], []
    for ptr in sorted(gone):
        parts = split_path(ptr)
        v = get_at(en060, parts)
        tail = str(parts[-1])
        if not isinstance(v, str):
            deleted.append((ptr, v, None, "non-str"))
            continue
        # T1
        cands = [p for p in by_val.get(v, []) if p not in used and str(split_path(p)[-1]) == tail]
        if cands:
            tgt = cands[0]
            used.add(tgt)
            migrated.append((ptr, v, tgt, "T1-EXACT"))
            continue
        # T2
        cands = [p for p in by_val.get(v, []) if p not in used]
        if cands:
            tgt = cands[0]
            used.add(tgt)
            migrated.append((ptr, v, tgt, "T2-EXACT-anykey"))
            continue
        # T3 near, restricted to same tail key + length band
        best, bs = None, 0.0
        L = len(v)
        for ap, av in add_items:
            if ap in used:
                continue
            if str(split_path(ap)[-1]) != tail:
                continue
            if not (0.55 * L <= len(av) <= 1.8 * L + 40):
                continue
            r = difflib.SequenceMatcher(None, v, av).ratio()
            if r > bs:
                bs, best = r, ap
        if best and bs >= NEAR:
            used.add(best)
            migrated.append((ptr, v, best, f"T3-NEAR {bs:.3f}"))
        else:
            deleted.append((ptr, v, best, f"best={bs:.3f}" if best else "no-cand"))
    report[f] = {"migrated": migrated, "deleted": deleted}

lines = []
for f in FILES:
    r = report[f]
    lines.append(f"\n########## {f}: 迁移 {len(r['migrated'])} / 真删 {len(r['deleted'])} ##########")
    lines.append("\n=== 迁移候选 ===")
    for src, v, tgt, why in r["migrated"]:
        lines.append(f"  [{why}]\n    src: {src}\n    dst: {tgt}\n    EN : {repr(v)[:200]}")
    lines.append("\n=== 真删候选 ===")
    for src, v, best, why in r["deleted"]:
        lines.append(f"  [{why}] {src}\n    EN : {repr(v)[:200]}" + (f"\n    近似最佳: {best}" if best else ""))
txt = '\n'.join(lines)
open(os.path.join(WORK, 'pair_report.txt'), 'w', encoding='utf-8').write(txt)

tm = sum(len(report[f]['migrated']) for f in FILES)
td = sum(len(report[f]['deleted']) for f in FILES)
print(f"合计: 迁移 {tm} / 真删 {td} / 总 {tm+td} (真值 314)")
assert tm + td == 314
jdump({f: {"migrated": [list(x) for x in report[f]['migrated']],
           "deleted": [list(x) for x in report[f]['deleted']]} for f in FILES},
      os.path.join(WORK, 'pair_report.json'))
for f in FILES:
    print(f"  {f:32s} 迁移 {len(report[f]['migrated']):3d} / 真删 {len(report[f]['deleted']):3d}")
