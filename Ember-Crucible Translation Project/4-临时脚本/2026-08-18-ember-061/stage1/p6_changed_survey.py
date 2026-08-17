# -*- coding: utf-8 -*-
"""P6: 改动桶盘点：按相似度 + 结构变化分档。"""
import os, sys, collections, difflib, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

delta = load_delta()
rows = []
for f in FILES:
    for ptr, (old, new) in delta[f]["changed"].items():
        if not isinstance(old, str) or not isinstance(new, str):
            rows.append((f, ptr, -1, 'NONSTR', len(str(old)), len(str(new))))
            continue
        r = difflib.SequenceMatcher(None, old, new).ratio()
        tag_d = collections.Counter(tag_multiset(new)) - collections.Counter(tag_multiset(old))
        tag_d2 = collections.Counter(tag_multiset(old)) - collections.Counter(tag_multiset(new))
        enh_d = collections.Counter(enh_multiset(new)) - collections.Counter(enh_multiset(old))
        enh_d2 = collections.Counter(enh_multiset(old)) - collections.Counter(enh_multiset(new))
        # 纯增补：old 是 new 的子序列块（difflib 只有 insert）
        sm = difflib.SequenceMatcher(None, old, new)
        ops = set(t[0] for t in sm.get_opcodes() if t[0] != 'equal')
        kind = 'PURE-ADD' if ops <= {'insert'} else ('PURE-DEL' if ops <= {'delete'} else 'REWRITE')
        rows.append((f, ptr, r, kind, len(old), len(new), dict(tag_d), dict(tag_d2), len(enh_d) + len(enh_d2)))

buckets = collections.Counter()
for row in rows:
    r = row[2]
    if r < 0:
        buckets['NONSTR'] += 1
    elif r >= 0.995:
        buckets['A 近乎相同 >=0.995'] += 1
    elif r >= 0.95:
        buckets['B 微调 0.95-0.995'] += 1
    elif r >= 0.80:
        buckets['C 中改 0.80-0.95'] += 1
    elif r >= 0.50:
        buckets['D 大改 0.50-0.80'] += 1
    else:
        buckets['E 重写 <0.50'] += 1
print("=== 相似度分档 (334) ===")
for k in sorted(buckets):
    print(f"  {k:24s} {buckets[k]}")
print("  合计", sum(buckets.values()))
kk = collections.Counter(r[3] for r in rows)
print("=== 编辑形态 ===", dict(kk))
print("=== 新增字符量 ===", sum(max(0, r[5] - r[4]) for r in rows if r[2] >= 0))

lines = []
for f, ptr, r, kind, lo, ln, *rest in sorted(rows, key=lambda x: x[2]):
    td, td2, ne = (rest + [{}, {}, 0])[:3]
    lines.append(f"{r:.4f} {kind:9s} len {lo}->{ln} tag+{td} tag-{td2} enhΔ{ne}  [{f.split('.')[1]}] {ptr}")
open(os.path.join(WORK, 'changed_index.txt'), 'w', encoding='utf-8').write('\n'.join(lines))
print("\n-> changed_index.txt")
