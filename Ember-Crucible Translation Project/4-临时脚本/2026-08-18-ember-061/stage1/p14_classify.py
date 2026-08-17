# -*- coding: utf-8 -*-
"""P14: 把每个工作项的 diff ops 分成「纯标记」「纯空白」「文本」三类，统计有多少项是全机械的。"""
import os, sys, difflib, re, collections, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

TOK = re.compile(r'<[^>]+>|@\w+\[[^\]]*\](?:\{[^}]*\})?|\[\[[^\]]*\]\]|&\w+;|\w+|\s+|.')
ONLY_TAG = re.compile(r'^(?:<[^>]+>|\s)*$')


def wdiff(a, b):
    Aa = TOK.findall(a); Bb = TOK.findall(b)
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, Aa, Bb).get_opcodes():
        if tag == 'equal':
            continue
        out.append((''.join(Aa[max(0, i1 - 12):i1])[-140:], ''.join(Aa[i1:i2]),
                    ''.join(Bb[j1:j2]), ''.join(Aa[i2:i2 + 8])[:70]))
    return out


def cls(o, n):
    if o.strip() == n.strip():
        return 'WS'
    if ONLY_TAG.match(o) and ONLY_TAG.match(n):
        return 'TAG'
    return 'TXT'


items = json.load(open(os.path.join(WORK, 'items_scope.json'), encoding='utf-8'))
delta = load_delta()
kinds = collections.Counter()
tagmap = collections.Counter()
rows = []
for i, (scope, ptr) in enumerate(items, 1):
    f = 'ember.adventure.json' if scope in ('both', 'ember.adventure.json') else scope
    old, new = delta[f]['changed'][ptr]
    ds = wdiff(old, new)
    cs = [cls(o, n) for _, o, n, _ in ds]
    c = collections.Counter(cs)
    k = 'MECH' if c['TXT'] == 0 else ('MIXED' if c['TAG'] + c['WS'] else 'TEXT')
    kinds[k] += 1
    rows.append((i, k, dict(c), scope, ptr))
    for (_, o, n, _), cc in zip(ds, cs):
        if cc == 'TAG':
            tagmap[(o, n)] += 1
print(kinds)
print("\n最常见的标记替换:")
for (o, n), c in tagmap.most_common(25):
    print(f"  {c:4d}  {o!r} -> {n!r}")
open(os.path.join(WORK, 'classify.txt'), 'w', encoding='utf-8').write(
    '\n'.join(f"{i:03d} {k:5s} {d} {ptr}" for i, k, d, s, ptr in rows))
print("\nMECH 项号:", [r[0] for r in rows if r[1] == 'MECH'])
