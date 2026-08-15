# -*- coding: utf-8 -*-
"""Enumerate every CN leaf carrying &Reference[exhaustion], with EN wording,
branch context (dnd5e / crucible / none) and the exact numeral form used."""
import re
import f2_lib as L

REF = re.compile(r'&(?:amp;)?[Rr]eference\[[Ee]xhaustion\]')
SUB = re.compile(r'<sub data-system="(\w+)">|</sub>')


def branch_of(s, pos):
    """Which data-system sub-branch encloses pos, if any."""
    cur = None
    for m in SUB.finditer(s):
        if m.start() > pos:
            break
        cur = m.group(1)  # None when </sub>
    return cur


for repo, pack in L.ALL:
    try:
        cn = L.cnmap(repo, pack); en = L.enmap(repo, pack)
    except FileNotFoundError:
        continue
    for p, v in cn.items():
        if not REF.search(v):
            continue
        e = en.get(p, '')
        print(f'##### {pack} | {p}')
        for m in REF.finditer(v):
            print('   CN', repr(v[max(0, m.start()-24):m.end()+6]), 'branch=', branch_of(v, m.start()))
        for m in re.finditer(r'&(?:amp;)?[Rr]eference\[[Ee]xhaustion\]', e):
            print('   EN', repr(e[max(0, m.start()-34):m.end()+6]), 'branch=', branch_of(e, m.start()))
