# -*- coding: utf-8 -*-
"""检查 锦标赛 在 b1+b2 的上下文 — 看是否有需保留的固定术语"""
import os, re

ROOT = r"C:\Users\Taka\Desktop\fvtt\FotRP\需要翻译\NEW"
FILES = [
    ("book1_ch1",      os.path.join(ROOT, "第一本", "fvtt-JournalEntry-chapter-1-c0yHRsNbVDGXKaIu.json")),
    ("book1_back",     os.path.join(ROOT, "第一本", "fvtt-JournalEntry-back-matter-ti7ZfnFRd1IabwaA.json")),
    ("book2_ch1",      os.path.join(ROOT, "第二本", "fvtt-JournalEntry-chapter-1-RzfDjQH8KPxPJ2kK.json")),
    ("book2_ch2",      os.path.join(ROOT, "第二本", "fvtt-JournalEntry-chapter-2-FTJT1CRdfjQzy3Ek.json")),
    ("book2_ch3",      os.path.join(ROOT, "第二本", "fvtt-JournalEntry-chapter-3-SPXqld4nww21Orrg.json")),
    ("book2_back",     os.path.join(ROOT, "第二本", "fvtt-JournalEntry-back-matter-XLDMbpumIxhSEWd4.json")),
]

out = []
total = 0
for key, p in FILES:
    with open(p, encoding="utf-8") as f:
        c = f.read()
    for m in re.finditer("锦标赛", c):
        total += 1
        i = m.start()
        before = c[max(0, i-40):i].replace("\n", "\\n")
        after = c[i+3:i+43].replace("\n", "\\n")
        out.append(f"  [{key}] ...{before}【锦标赛】{after}...")

print(f"Total 锦标赛: {total}")
with open("tournament_contexts.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
