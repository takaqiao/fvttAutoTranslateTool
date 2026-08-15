#!/usr/bin/env python3
"""用**标记指纹**机械判定「中文跟的是旧英文还是新英文」。

`scan_en_drift.py` 的 stale 桶判据是长度比值，假阳性很多（第 0 条已实证）。
但富文本标记（`@UUID[...]`、`[[/...]]`、`class="..."`、`@Condition[...]` 等）是**逐字节抄进译文**的，
所以它们是天然的版本指纹（阶段 20 的孤儿页面配对就是这么做的）：

  * 只存在于**旧**英文的标记出现在中文里  → 中文照旧英文写（STALE 硬证据）
  * 只存在于**新**英文的标记在中文里缺失  → 中文没跟上新增内容（STALE/PARTIAL 硬证据）
  * 两侧都干净                            → 标记层面已追平（大概率 OK，散文仍需抽读）

用法：
  python drift_marker_gate.py --drift <drift_*.json> --repo <repo> --baseline <旧基准目录>
                              [--bucket stale] [--start N] [--limit N] [--verbose]
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

MARKER = re.compile(
    r'@UUID\[[^\]]*\]'
    r'|@Embed\[[^\]]*\]'
    r'|@Action\[[^\]]*\]'
    r'|@Condition\[[^\]]*\]'
    r'|@CriticalSuccess\[[^\]]*\]'
    r'|@Advantage\[[^\]]*\]'
    r'|@ref\[[^\]]*\]'
    r'|&amp;[Rr]eference\[[^\]]*\]'
    r'|&[Rr]eference\[[^\]]*\]'
    r'|\[\[[^\]]*\]\]'
    r'|class="[^"]*"'
    r'|data-system="[^"]*"'
    r'|\[\[lookup [^\]]*\]\]'
)


def load_json(path):
    raw = open(path, encoding='utf-8-sig').read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(re.sub(r',(\s*[}\]])', r'\1', raw))


def leaves(node, path, out):
    if isinstance(node, dict):
        for k, v in node.items():
            leaves(v, path + [k], out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            leaves(v, path + [str(i)], out)
    elif isinstance(node, str) and node.strip():
        out['.'.join(path)] = node


def baseline_packs(bdir):
    out = {}
    for f in sorted(os.listdir(bdir)):
        if not f.endswith('.json') or f == '_source.json':
            continue
        key = ('ember.crucible-adventure.json' if f == '_repaired.json'
               else f.replace('-en.json', '.json'))
        out.setdefault(key, os.path.join(bdir, f))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--drift', required=True)
    ap.add_argument('--repo', required=True)
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--bucket', default='stale', choices=['stale', 'changed'])
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--limit', type=int, default=25)
    ap.add_argument('--verbose', action='store_true')
    a = ap.parse_args()

    d = load_json(a.drift)
    items = d['items'] if a.bucket == 'stale' else d['all_changed_with_cn']
    sel = items[a.start:a.start + a.limit]

    old_map = baseline_packs(a.baseline)
    cache = {}

    def pack_leaves(pack):
        if pack in cache:
            return cache[pack]
        o, n, c = {}, {}, {}
        if pack in old_map:
            leaves(load_json(old_map[pack]).get('entries', {}), [], o)
        pe = os.path.join(a.repo, 'compendium', 'en', pack)
        pc = os.path.join(a.repo, 'compendium', 'cn', pack)
        if os.path.exists(pe):
            leaves(load_json(pe).get('entries', {}), [], n)
        if os.path.exists(pc):
            leaves(load_json(pc).get('entries', {}), [], c)
        cache[pack] = (o, n, c)
        return cache[pack]

    for i, it in enumerate(sel, a.start):
        o, n, c = pack_leaves(it['pack'])
        p = it['path']
        old_en, new_en, cn = o.get(p) or '', n.get(p) or '', c.get(p) or ''
        mo, mn, mc = (Counter(MARKER.findall(x)) for x in (old_en, new_en, cn))
        # 只认「新英文里彻底没有」的标记，否则 old 2 份 / new 1 份会把正确译文误报成残留
        # （实测 177 The Villains、179 Ossuary 两条都是这样的假阳性）
        old_only = Counter({k: v for k, v in mo.items() if mn[k] == 0})
        new_only = Counter({k: v for k, v in mn.items() if mo[k] == 0})
        # 旧标记残留在中文里
        residue = old_only & mc
        # 新标记在中文里缺失
        missing = Counter({k: v for k, v in new_only.items() if mc[k] == 0})
        verdict = 'CLEAN' if not residue and not missing else 'DIRTY'
        print(f'[{i}] {verdict} | {p}')
        print(f'     旧独有 {sum(old_only.values())} / 新独有 {sum(new_only.values())} '
              f'| 中文残留旧标记 {sum(residue.values())} / 中文缺新标记 {sum(missing.values())}')
        if a.verbose or verdict == 'DIRTY':
            for k, v in residue.items():
                print(f'       残留旧×{v}: {k[:150]}')
            for k, v in missing.items():
                print(f'       缺新 ×{v}: {k[:150]}')


if __name__ == '__main__':
    main()
