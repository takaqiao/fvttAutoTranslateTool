# -*- coding: utf-8 -*-
"""Pair EN 'group (skill) check' wording against CN wording, leaf by leaf."""
import re
import f2_lib as L

ENPAT = re.compile(r'[Gg]roup\s+(?:[A-Za-z]+\s+)?[Cc]heck', re.S)
CNPAT = re.compile(r'(团队|群体|群组)(技能)?检定')

for repo, pack in L.ALL:
    try:
        cn = L.cnmap(repo, pack); en = L.enmap(repo, pack)
    except FileNotFoundError:
        continue
    for p, v in cn.items():
        e = en.get(p, '')
        ec = ENPAT.findall(e)
        cc = [m.group(0) for m in CNPAT.finditer(v)]
        if ec or cc:
            print(f'{pack}|{p}')
            print('   EN:', sorted(set(ec)), len(ec))
            print('   CN:', sorted(set(cc)), len(cc))
