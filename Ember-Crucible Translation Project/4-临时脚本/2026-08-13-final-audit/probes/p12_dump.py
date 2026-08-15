#!/usr/bin/env python3
"""P12: dump the GM/secret regions of one path, EN beside CN."""
import sys, os, re, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gm_lens import REPOS, pairs
from htmlblocks import regions

HID = re.compile(r'gamemaster|secret', re.I)
want = sys.argv[1]
mode = sys.argv[2] if len(sys.argv) > 2 else 'gm'

for rname, repo in REPOS.items():
    for pack, prs in pairs(repo):
        for path, e, c in prs:
            if want not in path:
                continue
            print('#' * 78)
            print(f'{rname}/{pack}  {path}')
            if mode == 'full':
                print('--- EN ---'); print(e)
                print('--- CN ---'); print(c)
                continue
            for side, html in (('EN', e), ('CN', c or '')):
                print(f'----- {side} GM/secret regions -----')
                for cls, i, j, o, ee in regions(html, 'section'):
                    if HID.search(cls or ''):
                        print(f'  [{cls}] {html[o:ee]}')
            break
