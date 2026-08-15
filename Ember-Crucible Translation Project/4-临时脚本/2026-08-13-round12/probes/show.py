#!/usr/bin/env python3
"""按 path 子串把筛子命中项的 旧英文/新英文/中文 三份全文打出来。"""
import json, re, sys, os

BASE = os.path.dirname(os.path.abspath(__file__))
pat = sys.argv[1]
which = sys.argv[2] if len(sys.argv) > 2 else 'onc'
rows = []
for f in ('g4.ember.json', 'g4.crucible.json'):
    p = os.path.join(BASE, f)
    if os.path.exists(p):
        rows += json.load(open(p, encoding='utf-8'))['items']
hit = [r for r in rows if pat in r['path']]
for r in hit:
    print('=' * 110)
    print(r['pack'], '|', r['path'])
    print('EV', json.dumps(r['ev'], ensure_ascii=False))
    if 'o' in which:
        print('--- OLD EN ---'); print(r['old_en'])
    if 'n' in which:
        print('--- NEW EN ---'); print(r['new_en'])
    if 'c' in which:
        print('--- CN ---'); print(r['cn'])
print(f'({len(hit)} hit)')
