# -*- coding: utf-8 -*-
"""P12: 210 条唯一工作项清单（含 EN 词级 diff + 现有 CN），分片输出便于逐条处理。"""
import os, sys, difflib, re, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

delta = load_delta()
A = delta['ember.adventure.json']['changed']
B = delta['ember.crucible-adventure.json']['changed']
C = delta['ember.crucible-adversary.json']['changed']
ca = jload(os.path.join(CN, 'ember.adventure.json'))
cb = jload(os.path.join(CN, 'ember.crucible-adventure.json'))
cc = jload(os.path.join(CN, 'ember.crucible-adversary.json'))
ea = jload(os.path.join(EN061, 'ember.adventure.json'))
eb = jload(os.path.join(EN061, 'ember.crucible-adventure.json'))

items = []
for p in sorted(set(A) | set(B)):
    sp = split_path(p)
    ina, inb = p in A, p in B
    if ina and inb:
        same = get_at(ca, sp) == get_at(cb, sp) and get_at(ea, sp) == get_at(eb, sp)
        scope = 'both' if same else 'SPLIT'
        old, new = A[p]
        cn = get_at(ca, sp)
    elif ina:
        scope, (old, new), cn = 'ember.adventure.json', A[p], get_at(ca, sp)
    else:
        scope, (old, new), cn = 'ember.crucible-adventure.json', B[p], get_at(cb, sp)
    items.append((scope, p, old, new, cn))
for p in sorted(C):
    old, new = C[p]
    items.append(('ember.crucible-adversary.json', p, old, new, get_at(cc, split_path(p))))

TOK = re.compile(r'<[^>]+>|@\w+\[[^\]]*\](?:\{[^}]*\})?|\[\[[^\]]*\]\]|&\w+;|\w+|\s+|.')


def wdiff(a, b):
    Aa = TOK.findall(a); Bb = TOK.findall(b)
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, Aa, Bb).get_opcodes():
        if tag == 'equal':
            continue
        o = ''.join(Aa[i1:i2]); n = ''.join(Bb[j1:j2])
        ctx = ''.join(Aa[max(0, i1 - 10):i1])[-110:]
        nxt = ''.join(Aa[i2:i2 + 6])[:50]
        out.append((ctx, o, n, nxt))
    return out


def is_ws_only(diffs):
    return all(o.strip() == n.strip() for _, o, n, _ in diffs)


rows = []
for scope, p, old, new, cn in items:
    if not isinstance(old, str) or not isinstance(new, str):
        rows.append((scope, p, old, new, cn, None, 'NONSTR'))
        continue
    ds = wdiff(old, new)
    kind = 'WS' if is_ws_only(ds) else 'EDIT'
    rows.append((scope, p, old, new, cn, ds, kind))

print('总工作项', len(rows), '| 纯空白', sum(1 for r in rows if r[6] == 'WS'),
      '| 需处理', sum(1 for r in rows if r[6] == 'EDIT'), '| 非串', sum(1 for r in rows if r[6] == 'NONSTR'))

# 输出分片
N = 30
todo = [r for r in rows if r[6] != 'WS']
os.makedirs(os.path.join(WORK, 'work'), exist_ok=True)
for i in range(0, len(todo), N):
    chunk = todo[i:i + N]
    out = []
    for scope, p, old, new, cn, ds, kind in chunk:
        out.append(f"\n\n=== #{todo.index((scope,p,old,new,cn,ds,kind))+1} [{scope}] {p}")
        if ds is None:
            out.append(f"  OLD: {old!r}\n  NEW: {new!r}")
        else:
            for ctx, o, n, nxt in ds:
                out.append(f"  …{ctx}|| - {o!r}\n{' '*(0)}       + {n!r}   ||{nxt}…")
        out.append(f"  --CN--: {cn!r}")
    open(os.path.join(WORK, 'work', f'w{i//N:02d}.txt'), 'w', encoding='utf-8').write('\n'.join(out))
print('分片', (len(todo) + N - 1) // N, '个 ->', os.path.join(WORK, 'work'))

# WS 清单
ws = [(r[0], r[1]) for r in rows if r[6] == 'WS']
jdump(ws, os.path.join(WORK, 'ws_only.json'))
jdump([[r[0], r[1]] for r in rows], os.path.join(WORK, 'worklist.json'))
