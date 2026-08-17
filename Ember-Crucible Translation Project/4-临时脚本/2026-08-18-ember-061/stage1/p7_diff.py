# -*- coding: utf-8 -*-
"""P7: 逐条打 EN060->EN061 的词级 diff（可按相似度区间过滤）。"""
import os, sys, difflib, re, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

lo = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
hi = float(sys.argv[2]) if len(sys.argv) > 2 else 1.01
withcn = '--cn' in sys.argv
only = None
for a in sys.argv:
    if a.startswith('--only='):
        only = a.split('=', 1)[1]

TOK = re.compile(r'<[^>]+>|@\w+\[[^\]]*\](?:\{[^}]*\})?|\[\[[^\]]*\]\]|\w+|\s+|.')


def wdiff(a, b):
    A = TOK.findall(a); B = TOK.findall(b)
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, A, B).get_opcodes():
        if tag == 'equal':
            continue
        o = ''.join(A[i1:i2]); n = ''.join(B[j1:j2])
        ctx = ''.join(A[max(0, i1 - 8):i1])[-90:]
        out.append(f"      ctx…{ctx}\n      - {o[:1200]!r}\n      + {n[:1200]!r}")
    return out


delta = load_delta()
n = 0
for f in FILES:
    cn = jload(os.path.join(CN, f)) if withcn else None
    for ptr, (old, new) in sorted(delta[f]["changed"].items()):
        if only and only not in ptr:
            continue
        if not isinstance(old, str):
            r = -1
        else:
            r = difflib.SequenceMatcher(None, old, new).ratio()
        if not (lo <= r < hi):
            continue
        n += 1
        print(f"\n### [{f.split('.')[1]}] r={r:.4f} {ptr}")
        if isinstance(old, str):
            for d in wdiff(old, new):
                print(d)
        else:
            print('   OLD', repr(old)[:800]); print('   NEW', repr(new)[:800])
        if withcn:
            print("   CN:", repr(get_at(cn, split_path(ptr)))[:2500])
print(f"\n---- {n} leaves in [{lo},{hi}) ----", file=sys.stderr)
