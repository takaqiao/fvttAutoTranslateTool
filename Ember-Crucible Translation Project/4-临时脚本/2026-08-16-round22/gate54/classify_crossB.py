#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 B 段 MJS_LANG_DRIFT 按 suggest 的形态分类，证明本轮新增那 1 条属于
**已有的同一个假阳性类**（「.mjs 内部同英文异译：X / X同调 X Attunement」），
不是新缺陷。反空转：每份报告都先打印总条数。"""
import json, os, re

G = os.path.dirname(os.path.abspath(__file__))


def drifts(path):
    d = json.load(open(path, encoding='utf-8'))
    out = []
    def walk(n):
        if isinstance(n, dict):
            if n.get('verdict') == 'MJS_LANG_DRIFT':
                out.append(n)
            for v in n.values():
                walk(v)
        elif isinstance(n, list):
            for v in n:
                walk(v)
    walk(d)
    return out


for tag in ('ember', 'crucible'):
    for which in ('BAK', 'NOW'):
        p = os.path.join(G, f'cross_{tag}{"_BAK" if which=="BAK" else ""}.json')
        ds = drifts(p)
        internal = [x for x in ds if str(x.get('suggest', '')).startswith('.mjs 内部同英文异译')]
        attune = [x for x in internal
                  if any('同调' in v for v in x.get('mjs_internal_variants', []))]
        print(f'{tag:9s} {which:3s}  DRIFT {len(ds):3d}  '
              f'其中「内部同英文异译」{len(internal):3d}  其中「X / X同调」族 {len(attune):3d}')
        if which == 'NOW':
            arr = [x for x in ds if x.get('table') == 'ARRANGEMENTS']
            print(f'{"":9s}      ARRANGEMENTS 表贡献的 DRIFT {len(arr)} 条: '
                  + ', '.join(f"{x['en']}->{x['mjs_cn']}" for x in arr))
