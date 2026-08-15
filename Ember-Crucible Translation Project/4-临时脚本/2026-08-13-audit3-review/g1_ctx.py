#!/usr/bin/env python3
"""按属性名 dump 上下文，用来判断某个属性名到底出现在什么语法结构里。

  python g1_ctx.py --repo <r> [--repo <r2>] --side en --attr apply --width 120 --max 12
"""
from __future__ import annotations
import argparse, json, os, re, sys
from collections import Counter
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

RAW_ATTR = re.compile(r'''(?<![\w:-])([A-Za-z_][\w:.-]*)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'<>\]]+))''')
INLINE_CMD = re.compile(r'\[\[/(?:[^\]"]|"[^"]*")*\]\]')


def walk(node, path, out):
    if isinstance(node, dict):
        for k, v in node.items():
            walk(v, path + [str(k)], out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            walk(v, path + [str(i)], out)
    elif isinstance(node, str):
        out.append(('.'.join(path), node))


def load(p):
    with open(p, encoding='utf-8') as fh:
        return json.load(fh).get('entries', {})


def packs(d):
    return sorted(f for f in os.listdir(d) if f.endswith('.json') and not f.startswith('_')) if os.path.isdir(d) else []


def mask_inline(t):
    o = list(t)
    for m in INLINE_CMD.finditer(t):
        for i in range(m.start(), m.end()):
            o[i] = ' '
    return ''.join(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', action='append', required=True)
    ap.add_argument('--side', default='en')
    ap.add_argument('--attr', action='append', required=True)
    ap.add_argument('--width', type=int, default=140)
    ap.add_argument('--max', type=int, default=10)
    ap.add_argument('--no-mask', action='store_true')
    a = ap.parse_args()
    want = {x.lower() for x in a.attr}
    shown = Counter()
    for repo in a.repo:
        d = os.path.join(repo, 'compendium', a.side)
        for f in packs(d):
            leaves = []
            walk(load(os.path.join(d, f)), [], leaves)
            for path, s in leaves:
                src = s if a.no_mask else mask_inline(s)
                for m in RAW_ATTR.finditer(src):
                    n = m.group(1).lower()
                    if n not in want or shown[n] >= a.max:
                        continue
                    shown[n] += 1
                    lo = max(0, m.start() - a.width)
                    hi = min(len(s), m.end() + a.width)
                    print(f'--- [{os.path.basename(repo)}/{f}] {n}  {path[:130]}')
                    print('   ' + s[lo:hi].replace('\n', ' '))
    print()
    for n, c in shown.most_common():
        print(f'{n}: shown {c}')


if __name__ == '__main__':
    main()
