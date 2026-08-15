# -*- coding: utf-8 -*-
"""Enumerate `]]`/`}` followed by space(s) then a full-width punctuation mark."""
import re
import f2_lib as L

FW = '，。：；、！？）《》“”'
PAT = re.compile(r'(\]\]|\})(\s+)([' + FW + r'])')

n = 0
for repo, pack in L.ALL:
    try:
        cn = L.cnmap(repo, pack)
    except FileNotFoundError:
        continue
    for p, v in cn.items():
        ms = list(PAT.finditer(v))
        if ms:
            print(f'{pack}|{p}  x{len(ms)}')
            for m in ms:
                n += 1
                print('    ', repr(v[max(0, m.start()-34):m.end()+18]))
print('TOTAL', n)
