# -*- coding: utf-8 -*-
"""Dump every folder-name leaf (top-level folders.* and entries.*.folders.*)
with EN source, CN value, and whether CN is bilingual (中文 English)."""
import re, json
import f2_lib as L

LAT = re.compile(r'[A-Za-z]')
rows = []
for repo, pack in L.ALL:
    try:
        cn = L.load(repo, 'cn', pack); en = L.load(repo, 'en', pack)
    except FileNotFoundError:
        continue
    # top-level folders
    for k, v in (cn.get('folders') or {}).items():
        e = (en.get('folders') or {}).get(k, k)
        rows.append((pack, 'top', k, e if isinstance(e, str) else k, v))
    # entries.*.folders.*
    for ek, ev in (cn.get('entries') or {}).items():
        if not isinstance(ev, dict):
            continue
        f = ev.get('folders')
        if isinstance(f, dict):
            ef = ((en.get('entries') or {}).get(ek) or {}).get('folders') or {}
            for k, v in f.items():
                rows.append((pack, ek, k, ef.get(k, k), v))

bi = [r for r in rows if LAT.search(r[4])]
bare = [r for r in rows if not LAT.search(r[4])]
print('total', len(rows), 'with-latin', len(bi), 'bare-cn', len(bare))
print()
for tag, group in (('BARE', bare), ('BILINGUAL', bi)):
    print('#####', tag)
    for pack, ent, k, e, v in group:
        print(f'  {pack:38s} | {ent[:22]:22s} | EN={e!r:34s} CN={v!r}')
    print()
