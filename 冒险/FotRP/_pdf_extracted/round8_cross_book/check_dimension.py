# -*- coding: utf-8 -*-
"""检查 维度 上下文 在 b1_back/b3_ch2/b3_ch3 各处"""
import os, re

ROOT = r"C:\Users\Taka\Desktop\fvtt\FotRP\需要翻译\NEW"
FILES = [
    ("book1_back", os.path.join(ROOT, "第一本", "fvtt-JournalEntry-back-matter-ti7ZfnFRd1IabwaA.json")),
    ("book3_ch2",  os.path.join(ROOT, "第三本", "fvtt-JournalEntry-chapter-2-fwmZr935hxQLlBus.json")),
    ("book3_ch3",  os.path.join(ROOT, "第三本", "fvtt-JournalEntry-chapter-3-VPgzvXimMH8NKzBk.json")),
]

# 已知"固定能力名" — Syndara 等
fixed_terms = ["维度叠加", "维度吞噬", "维度连击", "维度抓握", "维度暗面之镜", "维度突围"]

out_lines = []
for key, p in FILES:
    with open(p, encoding="utf-8") as f:
        content = f.read()
    out_lines.append(f"\n=== {key} ===")
    for m in re.finditer("维度", content):
        i = m.start()
        before = content[max(0, i-30):i].replace("\n", "\\n")
        after = content[i+2:i+32].replace("\n", "\\n")
        # 看下一字符是 fixed_term 后缀
        nxt = content[i+2:i+8]
        is_fixed = any(nxt.startswith(suff[2:]) for suff in fixed_terms)
        marker = "FIXED" if is_fixed else "PROSE?"
        out_lines.append(f"  [{marker}] ...{before}【维度】{after}...")

with open("dimension_contexts.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out_lines))
print("Wrote dimension_contexts.txt")
