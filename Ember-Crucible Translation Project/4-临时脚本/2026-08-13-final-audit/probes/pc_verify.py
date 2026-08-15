# -*- coding: utf-8 -*-
"""逐个核验：给英文名 + 若干中文写法，输出每处的 EN/CN 上下文与计数。"""
import os, re, sys
from collections import Counter
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_all
from pc_align import plain

ROWS = None

def run(en_pat, cn_terms, ctx=90, show=6):
    global ROWS
    if ROWS is None:
        ROWS = [r for r in load_all() if r[4]]
    rx = re.compile(en_pat)
    tot = Counter()
    print('=' * 100)
    print(f'EN /{en_pat}/   CN {cn_terms}')
    for repo, pack, path, en, cn in ROWS:
        pe, pc = plain(en), plain(cn)
        ne = len(rx.findall(pe))
        hits = {t: pc.count(t) for t in cn_terms}
        hits = {k: v for k, v in hits.items() if v}
        if not ne and not hits:
            continue
        tot['en'] += ne
        for k, v in hits.items():
            tot[k] += v
        if pack != 'ember.crucible-adventure.json' and pack != 'crucible.rules.json':
            continue
        print(f'  {pack[:16]} {path[-66:]}  EN={ne} {hits}')
        for t, v in hits.items():
            m = re.search(re.escape(t), pc)
            if m:
                print(f'     CN[{t}] …{pc[max(0,m.start()-ctx):m.end()+ctx]}…')
        m = rx.search(pe)
        if m:
            print(f'     EN     …{pe[max(0,m.start()-ctx):m.end()+ctx]}…')
    print('  TOTAL', dict(tot))


if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2].split(','))
