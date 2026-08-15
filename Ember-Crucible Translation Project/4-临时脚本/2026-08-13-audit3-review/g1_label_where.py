#!/usr/bin/env python3
"""定位每一个 label= / readaloud= / data-tooltip* 出现在什么语法结构里（@Embed / HTML 标签 / 其它）。

  python g1_label_where.py --repo <r> [--repo <r2>] --side en
"""
from __future__ import annotations
import argparse, json, os, re, sys
from collections import Counter
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

RAW_ATTR = re.compile(r'''(?<![\w:-])([A-Za-z_][\w:.-]*)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'<>\]]+))''')
INLINE_CMD = re.compile(r'\[\[/(?:[^\]"]|"[^"]*")*\]\]')
EMBED = re.compile(r'@[Ee]mbed\[((?:[^\]"]|"[^"]*")*)\]')
REFERENCE = re.compile(r'&[A-Za-z]+\[((?:[^\]"]|"[^"]*")*)\]')
TAG = re.compile(r'<[a-zA-Z][\w:-]*((?:[^>"\']|"[^"]*"|\'[^\']*\')*?)/?>')
WANT = {'label', 'readaloud', 'data-tooltip', 'data-tooltip-text', 'data-tooltip-html',
        'title', 'alt', 'aria-label', 'placeholder', 'data-label', 'caption', 'summary'}


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


def spans(rx, t):
    return [(m.start(), m.end()) for m in rx.finditer(t)]


def inside(pos, sp):
    return any(a <= pos < b for a, b in sp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', action='append', required=True)
    ap.add_argument('--side', default='en')
    ap.add_argument('--show-other', action='store_true')
    a = ap.parse_args()
    kind = Counter()
    for repo in a.repo:
        d = os.path.join(repo, 'compendium', a.side)
        for f in packs(d):
            leaves = []
            walk(load(os.path.join(d, f)), [], leaves)
            for path, s in leaves:
                sp_i, sp_e, sp_r, sp_t = spans(INLINE_CMD, s), spans(EMBED, s), spans(REFERENCE, s), spans(TAG, s)
                for m in RAW_ATTR.finditer(s):
                    n = m.group(1).lower()
                    if n not in WANT:
                        continue
                    p = m.start()
                    if inside(p, sp_i):
                        k = 'inline'
                    elif inside(p, sp_e):
                        k = 'embed'
                    elif inside(p, sp_r):
                        k = 'reference'
                    elif inside(p, sp_t):
                        k = 'html'
                    else:
                        k = 'OTHER'
                    kind[(n, k)] += 1
                    if k in ('OTHER', 'reference', 'inline') or a.show_other:
                        lo, hi = max(0, p - 130), min(len(s), m.end() + 130)
                        print(f'--- [{os.path.basename(repo)}/{f}] {n} @{k}  {path[:130]}')
                        print('   ' + s[lo:hi].replace('\n', ' '))
    print()
    for (n, k), c in sorted(kind.items()):
        print(f'{n:20} {k:10} {c}')


if __name__ == '__main__':
    main()
