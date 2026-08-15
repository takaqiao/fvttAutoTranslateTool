#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对比改 .mjs 前后 scan_cross_channel B 段的 MJS_LANG_DRIFT 明细，
把「DRIFT 31 -> 32 / 36 -> 37」那 +1 是哪一条指出来。反空转：先打印两侧条数。"""
import json, os, sys

G = os.path.dirname(os.path.abspath(__file__))


def drift_keys(path):
    d = json.load(open(path, encoding='utf-8'))
    out = []
    def walk(node):
        if isinstance(node, dict):
            if node.get('kind') == 'MJS_LANG_DRIFT' or node.get('verdict') == 'MJS_LANG_DRIFT' \
               or node.get('status') == 'MJS_LANG_DRIFT':
                n = {k: v for k, v in node.items() if k != 'file'}
                out.append(json.dumps(n, ensure_ascii=False, sort_keys=True))
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk(d)
    return out


for repo in ('ember', 'crucible'):
    a = drift_keys(os.path.join(G, f'cross_{repo}_BAK.json'))
    b = drift_keys(os.path.join(G, f'cross_{repo}.json'))
    print(f'=== {repo}: BAK DRIFT {len(a)} 条 · 现 DRIFT {len(b)} 条 ===')
    sa, sb = set(a), set(b)
    for x in sorted(sb - sa):
        print('  + 新增:', x[:400])
    for x in sorted(sa - sb):
        print('  - 消失:', x[:400])
