# -*- coding: utf-8 -*-
"""Survey the F2 text-unification findings across both repos."""
import re, sys, json
import f2_lib as L

what = sys.argv[1] if len(sys.argv) > 1 else 'all'

TERMS = {
    'group': ['团队检定', '团队技能检定', '群体检定', '群体技能检定', '群组检定', '团队技能', '群体技能'],
    'hazard': ['坠落危害', '坠落危险'],
}


def scan(pred, label):
    print(f'###### {label}')
    tot = 0
    for repo, pack in L.ALL:
        try:
            cn = L.cnmap(repo, pack)
            en = L.enmap(repo, pack)
        except FileNotFoundError:
            continue
        for p, v in cn.items():
            hits = pred(v, en.get(p, ''))
            if hits:
                tot += len(hits) if isinstance(hits, list) else 1
                print(f'{repo.split("-")[0]}|{pack}|{p}')
                if isinstance(hits, list):
                    for h in hits:
                        print('    ', h)
    print('TOTAL', tot)


if what in ('group', 'all'):
    def f(v, e):
        return [t for t in TERMS['group'] if t in v] or None
    scan(f, 'group check terms')

if what in ('hazard', 'all'):
    def f(v, e):
        out = []
        for m in re.finditer('坠落危害|坠落危险', v):
            out.append(v[max(0, m.start()-40):m.end()+40])
        return out or None
    scan(f, 'falling hazard')
