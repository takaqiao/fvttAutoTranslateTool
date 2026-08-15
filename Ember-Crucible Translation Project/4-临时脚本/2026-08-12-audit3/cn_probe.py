#!/usr/bin/env python3
"""按 path + 锚点子串，从 compendium/cn（或 en / 旧基准）里切出局部上下文。

drift 复核时真正要读的只是「上游改动那几处」的中文，
整叶打印会把上下文吃光。锚点用标记（`@UUID[...]` 的 id、`[[/...]]` 命令体、数字）最稳，
因为标记在译文里是逐字节照抄的。

用法：
  python cn_probe.py --path <BATCH_PATH> --anchor <子串> [--anchor ...]
                     [--side cn|en|old] [--win 400]
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO = os.path.join(ROOT, '1-Ember汉化插件')
BASE = os.path.join(ROOT, '5-其他内容', 'english-baseline', 'ember-cn-v1.0.15-shipped-en')


def load(p):
    raw = open(p, encoding='utf-8-sig').read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(re.sub(r',(\s*[}\]])', r'\1', raw))


def leaves(n, path, out):
    if isinstance(n, dict):
        for k, v in n.items():
            leaves(v, path + [k], out)
    elif isinstance(n, list):
        for i, v in enumerate(n):
            leaves(v, path + [str(i)], out)
    elif isinstance(n, str) and n.strip():
        out['.'.join(path)] = n


_cache = {}


def get(side, pack):
    key = (side, pack)
    if key in _cache:
        return _cache[key]
    if side == 'old':
        f = os.path.join(BASE, '_repaired.json' if pack == 'ember.crucible-adventure.json'
                         else pack.replace('.json', '-en.json'))
    else:
        f = os.path.join(REPO, 'compendium', side, pack)
    m = {}
    if os.path.exists(f):
        leaves(load(f).get('entries', {}), [], m)
    _cache[key] = m
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack', default='ember.crucible-adventure.json')
    ap.add_argument('--path', required=True)
    ap.add_argument('--anchor', action='append', required=True)
    ap.add_argument('--side', default='cn')
    ap.add_argument('--win', type=int, default=400)
    a = ap.parse_args()

    for side in a.side.split(','):
        s = get(side, a.pack).get(a.path)
        print(f'##### side={side} len={len(s) if s else 0}')
        if not s:
            continue
        for anc in a.anchor:
            print(f'-- anchor {anc!r}')
            found = False
            for m in re.finditer(re.escape(anc), s):
                found = True
                lo = max(0, m.start() - a.win)
                hi = min(len(s), m.end() + a.win)
                print(f'   @{m.start()}: …{s[lo:hi]}…')
            if not found:
                print('   (未找到)')


if __name__ == '__main__':
    main()
