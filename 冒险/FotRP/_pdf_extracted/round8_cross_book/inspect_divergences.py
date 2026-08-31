# -*- coding: utf-8 -*-
"""检查疑似分歧的实际上下文 — 看 journal 用了什么中文"""
import json, re, os

ROOT = r"C:\Users\Taka\Desktop\fvtt\FotRP\需要翻译\NEW"

FILES = [
    ("book1_handouts", os.path.join(ROOT, "第一本", "fvtt-JournalEntry-handouts-FhOWhqywZr72iuml.json")),
    ("book1_ch1",      os.path.join(ROOT, "第一本", "fvtt-JournalEntry-chapter-1-c0yHRsNbVDGXKaIu.json")),
    ("book1_ch2",      os.path.join(ROOT, "第一本", "fvtt-JournalEntry-chapter-2-PpZKICROB7B08r53.json")),
    ("book1_ch3",      os.path.join(ROOT, "第一本", "fvtt-JournalEntry-chapter-3-jy3Vrm5yA3jrrG5W.json")),
    ("book1_back",     os.path.join(ROOT, "第一本", "fvtt-JournalEntry-back-matter-ti7ZfnFRd1IabwaA.json")),
    ("book2_handouts", os.path.join(ROOT, "第二本", "fvtt-JournalEntry-handouts-rXylkhhxGCEispwF.json")),
    ("book2_ch1",      os.path.join(ROOT, "第二本", "fvtt-JournalEntry-chapter-1-RzfDjQH8KPxPJ2kK.json")),
    ("book2_ch2",      os.path.join(ROOT, "第二本", "fvtt-JournalEntry-chapter-2-FTJT1CRdfjQzy3Ek.json")),
    ("book2_ch3",      os.path.join(ROOT, "第二本", "fvtt-JournalEntry-chapter-3-SPXqld4nww21Orrg.json")),
    ("book2_back",     os.path.join(ROOT, "第二本", "fvtt-JournalEntry-back-matter-XLDMbpumIxhSEWd4.json")),
    ("book3_ch1",      os.path.join(ROOT, "第三本", "fvtt-JournalEntry-chapter-1-xClvGtftweJDu3vX.json")),
    ("book3_ch3",      os.path.join(ROOT, "第三本", "fvtt-JournalEntry-chapter-3-VPgzvXimMH8NKzBk.json")),
]

file_contents = {}
for key, p in FILES:
    with open(p, encoding="utf-8") as f:
        file_contents[key] = f.read()

# 疑似分歧的英文形式
suspects = [
    "Arms of Balance",
    "Golarion's Finest",
    "Yabin the Just",
    "Yabin",
    "Artus Rodrivan",
    "Artus",
    "Lantondo",
    "Halspin",
    "Halspin the Stung",
    "Umbasi",
    "Ahmoza",
    "Ki Adept",
    "Ranya Shibhatesh", "Shibhatesh",
    "Jivati Rovat", "Rovat",
    "Pravan Majinapti", "Majinapti", "Pravan",
    "Usvani",
    "Brartork", "Krankkiss", "Numoriz", "Paunnima", "Rajna", "Mingyu",
]

out_lines = ["=== 疑似分歧的实际上下文 ===\n"]

for suspect in suspects:
    found_any = False
    for key, content in file_contents.items():
        for m in re.finditer(re.escape(suspect), content):
            i = m.start()
            before = content[max(0, i-60):i].replace("\n", "\\n")
            after = content[i+len(suspect):i+len(suspect)+60].replace("\n", "\\n")
            if not found_any:
                out_lines.append(f"\n=== {suspect} ===")
                found_any = True
            out_lines.append(f"  [{key}] ...{before}【{suspect}】{after}...")

with open("divergence_contexts.txt", "w", encoding="utf-8") as out:
    out.write("\n".join(out_lines))
print(f"Wrote divergence_contexts.txt, lines: {len(out_lines)}")
