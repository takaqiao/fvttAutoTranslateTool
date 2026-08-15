# -*- coding: utf-8 -*-
"""抓所有「人物条」<dt>：形如  Name (Alignment, Ancestry Culture, pronouns)。只读。"""
import os, re, sys, json
from collections import Counter, defaultdict
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_all, plain

DT = re.compile(r'<dt\b[^>]*>(.*?)</dt>', re.S | re.I)
PRON = re.compile(r'\b(she/her|he/him|they/them|it/its|he/they|she/they|any pronouns|no pronouns)\b', re.I)

rows = []
for repo, pack, path, en, cn in load_all():
    if '<dt' not in en:
        continue
    edts = DT.findall(en)
    cdts = DT.findall(cn or '')
    if not edts:
        continue
    aligned = len(edts) == len(cdts)
    for i, e in enumerate(edts):
        pe = plain(e)
        if not PRON.search(pe):
            continue
        pc = plain(cdts[i]) if aligned and i < len(cdts) else None
        rows.append({'repo': repo, 'pack': pack, 'path': path, 'i': i,
                     'aligned': aligned, 'en': pe, 'cn': pc})

print('person-dt count:', len(rows), ' unaligned:', sum(1 for r in rows if not r['aligned']))
json.dump(rows, open('pc_dt.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
for r in rows[:25]:
    print(f"  EN: {r['en']}")
    print(f"  CN: {r['cn']}")
