# -*- coding: utf-8 -*-
"""round24 ③-C 反向判据上下文核实：逐词打印每一处出现的行号 + 整行（截断）。
落盘再跑。"""
import os, re, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

EM = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember\scripts\ember.mjs"
CR = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\systems\crucible\crucible-compiled.mjs"

WORDS = ["Arcturian", "Human", "Kavir", "Keth", "Kivahr", "Lumek", "Oaken", "Wirrun",
         "Bejak", "Kessian", "Ordani", "Waerd"]

# 「像血统/文化标签」的判据：identifier 字段、裸 label、ancestry/culture 上下文
TAGGY = re.compile(r'(identifier\s*:\s*"[^"]*%s|label\s*:\s*"%s"|name\s*:\s*"%s"|Ancestry|Culture|ancestry|culture)')

for path, tag in ((EM, "ember.mjs"), (CR, "crucible-compiled.mjs")):
    lines = open(path, "r", encoding="utf-8", errors="replace").read().splitlines()
    for w in WORDS:
        hits = [(i + 1, l) for i, l in enumerate(lines) if w in l]
        if not hits:
            continue
        print("\n### %s @ %s  —— %d 行 / %d 次出现" % (w, tag, len(hits), sum(l.count(w) for _, l in hits)))
        # 先把「看起来像标签」的行挑出来
        taggy = [(n, l) for n, l in hits
                 if re.search(r'(identifier\s*:|^\s*label:\s*"%s"|name:\s*"%s"|[Aa]ncestry|[Cc]ulture)' % (w, w), l)]
        print("  -- taggy(%d):" % len(taggy))
        for n, l in taggy[:8]:
            print("     %6d | %s" % (n, l.strip()[:200]))
        print("  -- 其余样本:")
        rest = [(n, l) for n, l in hits if (n, l) not in taggy]
        for n, l in rest[:4]:
            print("     %6d | %s" % (n, l.strip()[:180]))
