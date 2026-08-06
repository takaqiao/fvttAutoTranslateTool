#!/usr/bin/env python3
"""Normalise 「」 to “” across a plugin's Chinese packs.

The library settled on “” long ago (2618 vs 44). The 「」 minority came from a
later batch that followed a style note instead of the existing text. Quotes are
not markup, so the only thing to guard is that nothing else changed: the markup
signature of every rewritten string is compared before and after.

  python normalize_quotes.py --repo <repo> [--write]
"""
from __future__ import annotations
import argparse
import json
import os
import re
from collections import Counter

MARKUP = re.compile(r'@[A-Za-z]+\[[^\]]*\]')
INLINE_CMD = re.compile(r'\[\[[^\]]*\]\]')
TAGNAME = re.compile(r'<\s*(/?)([a-zA-Z][a-zA-Z0-9]*)')


def sig(s):
    return (Counter(MARKUP.findall(s)) + Counter(INLINE_CMD.findall(s))
            + Counter(f'<{a}{b.lower()}' for a, b in TAGNAME.findall(s)))


def convert(node, stats, samples):
    if isinstance(node, dict):
        return {k: convert(v, stats, samples) for k, v in node.items()}
    if isinstance(node, list):
        return [convert(v, stats, samples) for v in node]
    if isinstance(node, str) and ('「' in node or '」' in node):
        new = node.replace('「', '“').replace('」', '”')
        if sig(new) != sig(node):                      # 不可能发生，发生就是有别的东西被动了
            stats['refused'] += 1
            return node
        stats['changed'] += 1
        stats['pairs'] += node.count('「')
        if len(samples) < 6:
            i = node.index('「')
            samples.append(node[max(0, i - 30):i + 40])
        return new
    return node


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    ap.add_argument('--write', action='store_true')
    a = ap.parse_args()

    cn_dir = os.path.join(a.repo, 'compendium', 'cn')
    stats = Counter()
    samples = []
    for fn in sorted(f for f in os.listdir(cn_dir) if f.endswith('.json')):
        p = os.path.join(cn_dir, fn)
        with open(p, encoding='utf-8') as f:
            doc = json.load(f)
        before = stats['changed']
        out = convert(doc, stats, samples)
        if stats['changed'] > before:
            print(f'{fn:<38}{stats["changed"] - before:>5} 条')
            if a.write:
                with open(p, 'w', encoding='utf-8') as f:
                    json.dump(out, f, ensure_ascii=False, indent=2)
                    f.write('\n')

    print(f'\n条目 {stats["changed"]}，引号对 {stats["pairs"]}，拒绝 {stats["refused"]}')
    for s in samples:
        print('  …' + s.replace('\n', ' ') + '…')
    if not a.write:
        print('\n(未加 --write，什么都没写)')


main()
