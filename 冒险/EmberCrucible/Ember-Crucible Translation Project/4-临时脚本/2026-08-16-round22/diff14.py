#!/usr/bin/env python3
"""对 three_way_14.json 逐条做英文词级差分，并附中文全文，落盘成可读文本。"""
import json, os, re, sys, difflib

HERE = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(os.path.join(HERE, 'three_way_14.json'), encoding='utf-8'))

TOK = re.compile(r'<[^>]+>|[A-Za-z0-9一-鿿\'’-]+|[^\sA-Za-z0-9]')

def toks(s):
    return TOK.findall(s or '')

buf = []
for r in rows:
    a, b = toks(r['old_en']), toks(r['new_en'])
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    parts = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            seg = a[i1:i2]
            if len(seg) > 8:
                parts.append(' '.join(seg[:4]) + ' … ' + ' '.join(seg[-4:]))
            else:
                parts.append(' '.join(seg))
        else:
            if i1 != i2:
                parts.append('[-' + ' '.join(a[i1:i2]) + '-]')
            if j1 != j2:
                parts.append('{+' + ' '.join(b[j1:j2]) + '+}')
    buf.append('=' * 78)
    buf.append(f"### {r['pack']} :: {r['path']}")
    buf.append(f"old_en {len(r['old_en'])} chars / new_en {len(r['new_en'])} / cn {len(r['cn'])}")
    buf.append('--- EN diff (old -> new) ---')
    buf.append(' '.join(parts))
    buf.append('--- CN (full) ---')
    buf.append(r['cn'])
    buf.append('')

dst = os.path.join(HERE, 'diff14.txt')
open(dst, 'w', encoding='utf-8').write('\n'.join(buf))
print(f'写出 {len(rows)} 条差分 -> {dst}  ({sum(len(x) for x in buf)} chars)')
