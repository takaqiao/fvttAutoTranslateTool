#!/usr/bin/env python3
"""生成本轮批次。

结论是 **0 条真缺陷**（14 条全部已跟上新英文），所以没有「改译文」的批次。
为了让这个 0 不是空转，这里产的是**恒等校验批次**：把 14 条叶**当前中文原样**写成
批次，跑 `apply_translations --force --dry`。它能证明三件事：
  1. 14 条路径在批次格式下全部解析得到（不是路径写错导致的静默 0）；
  2. 英文源与基线一致、标记签名一致 —— 零拒绝；
  3. 真去 apply 也是 no-op（写回去的就是原文），不会动库。
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(os.path.join(HERE, 'three_way_14.json'), encoding='utf-8'))
out = os.path.join(HERE, 'batches')
os.makedirs(out, exist_ok=True)

by_pack = {}
for r in rows:
    by_pack.setdefault(r['pack'], {})[r['path']] = r['cn']

for pack, d in by_pack.items():
    dst = os.path.join(out, f'noop_verify_{pack}')
    json.dump(d, open(dst, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f'{pack}: {len(d)} 条 -> {dst}')
print('合计', sum(len(d) for d in by_pack.values()), '条')
