# -*- coding: utf-8 -*-
"""P13: 每个工作项写成单独文件 items/NNN.txt，并出一张索引表。"""
import os, sys, difflib, re, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

delta = load_delta()
A = delta['ember.adventure.json']['changed']
B = delta['ember.crucible-adventure.json']['changed']
C = delta['ember.crucible-adversary.json']['changed']
CNJ = {f: jload(os.path.join(CN, f)) for f in FILES}
ENJ = {f: jload(os.path.join(EN061, f)) for f in FILES}

items = []
for p in sorted(set(A) | set(B)):
    sp = split_path(p)
    ina, inb = p in A, p in B
    if ina and inb:
        same = (get_at(CNJ['ember.adventure.json'], sp) == get_at(CNJ['ember.crucible-adventure.json'], sp)
                and get_at(ENJ['ember.adventure.json'], sp) == get_at(ENJ['ember.crucible-adventure.json'], sp))
        if same:
            items.append(('both', p, A[p][0], A[p][1], get_at(CNJ['ember.adventure.json'], sp)))
        else:
            items.append(('ember.adventure.json', p, A[p][0], A[p][1], get_at(CNJ['ember.adventure.json'], sp)))
            items.append(('ember.crucible-adventure.json', p, B[p][0], B[p][1], get_at(CNJ['ember.crucible-adventure.json'], sp)))
    elif ina:
        items.append(('ember.adventure.json', p, A[p][0], A[p][1], get_at(CNJ['ember.adventure.json'], sp)))
    else:
        items.append(('ember.crucible-adventure.json', p, B[p][0], B[p][1], get_at(CNJ['ember.crucible-adventure.json'], sp)))
for p in sorted(C):
    items.append(('ember.crucible-adversary.json', p, C[p][0], C[p][1], get_at(CNJ['ember.crucible-adversary.json'], split_path(p))))

TOK = re.compile(r'<[^>]+>|@\w+\[[^\]]*\](?:\{[^}]*\})?|\[\[[^\]]*\]\]|&\w+;|\w+|\s+|.')


def wdiff(a, b):
    Aa = TOK.findall(a); Bb = TOK.findall(b)
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, Aa, Bb).get_opcodes():
        if tag == 'equal':
            continue
        out.append((''.join(Aa[max(0, i1 - 12):i1])[-140:], ''.join(Aa[i1:i2]),
                    ''.join(Bb[j1:j2]), ''.join(Aa[i2:i2 + 8])[:70]))
    return out


D = os.path.join(WORK, 'items')
os.makedirs(D, exist_ok=True)
idx = []
for i, (scope, p, old, new, cn) in enumerate(items, 1):
    ds = wdiff(old, new)
    ws = all(o.strip() == n.strip() for _, o, n, _ in ds)
    r = difflib.SequenceMatcher(None, old, new).ratio()
    out = [f"ITEM #{i}", f"scope: {scope}", f"ptr:   {p}", f"ratio: {r:.4f}  ops:{len(ds)}  ws_only:{ws}",
           f"len EN {len(old)}->{len(new)}  CN {len(cn) if cn else 0}", "", "--- EN diff ---"]
    for ctx, o, n, nxt in ds:
        out.append(f"  …{ctx}\n     - {o!r}\n     + {n!r}\n     …{nxt}")
    out += ["", "--- EN061 (new, full) ---", new, "", "--- CN (current, full) ---", str(cn)]
    open(os.path.join(D, f"{i:03d}.txt"), 'w', encoding='utf-8').write('\n'.join(out))
    idx.append(f"{i:03d} r={r:.4f} ops={len(ds):3d} ws={int(ws)} en{len(old)}->{len(new)} cn{len(cn) if cn else 0} [{ {'both':'BOTH','ember.adventure.json':'ADV','ember.crucible-adventure.json':'CRU','ember.crucible-adversary.json':'ADS'}[scope] }] {p.split('/',3)[-1]}")
open(os.path.join(WORK, 'items_index.txt'), 'w', encoding='utf-8').write('\n'.join(idx))
jdump([[s, p] for s, p, *_ in items], os.path.join(WORK, 'items_scope.json'))
print(len(items), 'items ->', D)
