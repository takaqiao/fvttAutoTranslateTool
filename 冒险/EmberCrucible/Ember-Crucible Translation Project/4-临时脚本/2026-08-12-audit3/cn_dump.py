#!/usr/bin/env python3
"""按 drift 序号打印**现有中文全文**（可选同时打印新英文），用于逐条比对。

配合 drift_substantive.py 用：那边给出「上游改了什么」的 NEW 片段，
这边给中文全文，判「中文跟的是哪一版」。
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--drift', required=True)
    ap.add_argument('--repo', required=True)
    ap.add_argument('--bucket', default='stale')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--limit', type=int, default=5)
    ap.add_argument('--en', action='store_true')
    a = ap.parse_args()

    d = load_json(a.drift)
    items = (d['items'] if a.bucket == 'stale' else d['all_changed_with_cn'])[a.start:a.start + a.limit]
    cache = {}

    for i, it in enumerate(items, a.start):
        pack = it['pack']
        if pack not in cache:
            n, c = {}, {}
            pe = os.path.join(a.repo, 'compendium', 'en', pack)
            pc = os.path.join(a.repo, 'compendium', 'cn', pack)
            if os.path.exists(pe):
                leaves(load_json(pe).get('entries', {}), [], n)
            if os.path.exists(pc):
                leaves(load_json(pc).get('entries', {}), [], c)
            cache[pack] = (n, c)
        n, c = cache[pack]
        print(f'\n{"="*96}\n[{i}] {it["path"]}')
        if a.en:
            print('--- NEW EN ---')
            print(n.get(it['path'], '(缺)'))
        print('--- CN ---')
        print(c.get(it['path'], '(缺)'))


if __name__ == '__main__':
    main()
