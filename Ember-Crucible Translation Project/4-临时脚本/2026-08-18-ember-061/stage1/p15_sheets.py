# -*- coding: utf-8 -*-
"""P15: 生成翻译工作表（分片），每个槽给出：子级 diff + 旧 CN（PATCH）或全文（NEW）。"""
import os, sys, json, difflib, re, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from rebuild import rebuild, render, reattach_ids, Slot
from segs import seg

TOK = re.compile(r'@\w+\[[^\]]*\](?:\{[^}]*\})?|\[\[[^\]]*\]\]|&\w+;|\w+|\s+|.')


def sub_diff(a, b):
    A = TOK.findall(a); B = TOK.findall(b)
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, A, B).get_opcodes():
        if tag == 'equal':
            continue
        out.append((''.join(A[max(0, i1 - 12):i1])[-120:], ''.join(A[i1:i2]),
                    ''.join(B[j1:j2]), ''.join(A[i2:i2 + 8])[:60]))
    return out


items = json.load(open(os.path.join(WORK, 'items_scope.json'), encoding='utf-8'))
delta = load_delta()
CNJ = {f: jload(os.path.join(CN, f)) for f in FILES}
SH = os.path.join(WORK, 'sheets')
os.makedirs(SH, exist_ok=True)

blocks = []
auto = {}
for i, (scope, ptr) in enumerate(items, 1):
    f = 'ember.adventure.json' if scope in ('both', 'ember.adventure.json') else scope
    old, new = delta[f]['changed'][ptr]
    cn = get_at(CNJ[f], split_path(ptr))
    parts, slots = rebuild(old, new, cn)
    if not slots:
        auto[str(i)] = {}
        continue
    lines = [f"\n\n{'='*90}\nITEM {i}  [{scope}]  {ptr.split('/',3)[-1]}"]
    todo = 0
    for k, s in enumerate(slots):
        t0 = [t for kk, t in seg(s.en_old)if kk == 'T'] if s.en_old else None
        t1 = [t for kk, t in seg(s.en_new) if kk == 'T']
        m_only = (s.kind == 'PATCH' and t0 == t1)
        if m_only:
            continue
        todo += 1
        lines.append(f"\n--- slot {k} [{s.kind}] ---")
        if s.kind == 'PATCH':
            for ctx, o, n, nxt in sub_diff(s.en_old, s.en_new):
                lines.append(f"  …{ctx}\n     -EN {o!r}\n     +EN {n!r}\n     …{nxt}")
            lines.append(f"  CN旧: {s.cn_old}")
            lines.append(f"  EN新: {s.en_new}")
        else:
            lines.append(f"  EN新: {s.en_new}")
    if todo:
        blocks.append((i, '\n'.join(lines)))

# 分片
CHUNK = 24
for c in range(0, len(blocks), CHUNK):
    txt = ''.join(b for _, b in blocks[c:c + CHUNK])
    open(os.path.join(SH, f"s{c//CHUNK:02d}.txt"), 'w', encoding='utf-8').write(txt)
print("需人工的 item 数:", len(blocks), "分片", (len(blocks) + CHUNK - 1) // CHUNK)
print("全自动 item 数:", len(auto))
for fn in sorted(os.listdir(SH)):
    print(' ', fn, os.path.getsize(os.path.join(SH, fn)))
