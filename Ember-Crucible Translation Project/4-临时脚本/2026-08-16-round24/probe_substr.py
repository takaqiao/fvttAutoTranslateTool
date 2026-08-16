# -*- coding: utf-8 -*-
"""核实盲区：makeCorpus 用的是裸 includes（无词边界），短词会被更长的词吃进去。
逐词统计「独立成词的出现」vs「只是别的词的子串」。"""
import re, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

EM = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember\scripts\ember.mjs"
CR = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\systems\crucible\crucible-compiled.mjs"

WORDS = ["Abyss", "Heart", "Aura", "Cora", "Human", "Keth", "Kivahr", "Lumek",
         "Oaken", "Wirrun", "Bejak", "Kessian", "Ordani", "Waerd", "Arcturian", "Kavir"]

for path, tag in ((EM, "ember.mjs"), (CR, "crucible-compiled.mjs")):
    text = open(path, "r", encoding="utf-8", errors="replace").read()
    print("\n==== %s ====" % tag)
    for w in WORDS:
        total = text.count(w)
        if not total:
            print("  %-11s total=0" % w)
            continue
        # 独立成词：前后都不是字母/数字/下划线
        standalone = len(re.findall(r'(?<![A-Za-z0-9_])' + re.escape(w) + r'(?![A-Za-z0-9_])', text))
        print("  %-11s total=%-4d 独立成词=%-4d 仅子串=%d" % (w, total, standalone, total - standalone))
