#!/usr/bin/env python3
"""只把「证据 token 周围那一小段」在 旧英文/新英文/中文 三边各打一遍，省得读整页。"""
import json, os, re, sys

BASE = os.path.dirname(os.path.abspath(__file__))
pat = sys.argv[1]
W = int(sys.argv[2]) if len(sys.argv) > 2 else 150
rows = []
for f in ('g4.ember.json', 'g4.crucible.json'):
    p = os.path.join(BASE, f)
    if os.path.exists(p):
        rows += json.load(open(p, encoding='utf-8'))['items']

NUMB = r'(?<![0-9A-Za-z.])%s(?![0-9A-Za-z])'


def ctx(s, needle, isnum):
    rx = re.compile(NUMB % re.escape(needle) if isnum else re.escape(needle))
    return [s[max(0, m.start() - W):m.start() + W] for m in rx.finditer(s)]


for r in rows:
    if pat not in r['path']:
        continue
    print('=' * 110)
    print(r['path'], '| len o/n/c', r['en_len_old'], r['en_len_new'], r['cn_len'])
    for kind, toks in r['ev'].items():
        isnum = kind.endswith('num')
        for t in toks:
            term = t.split('->')[0]
            print(f'  --- {kind}: {t}')
            for lbl, key in (('OLD', 'old_en'), ('NEW', 'new_en'), ('CN ', 'cn')):
                for c in ctx(r[key], term, isnum)[:3]:
                    print(f'     [{lbl}] …{c}…')
