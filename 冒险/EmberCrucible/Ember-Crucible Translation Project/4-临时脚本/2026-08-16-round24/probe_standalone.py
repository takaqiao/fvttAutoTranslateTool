# -*- coding: utf-8 -*-
"""列出「独立成词」的每一处，带行号 —— 反向判据的上下文核实靠这个定性。"""
import re, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

EM = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember\scripts\ember.mjs"
CR = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\systems\crucible\crucible-compiled.mjs"

WORDS = ["Human", "Keth", "Kivahr", "Lumek", "Oaken", "Wirrun",
         "Bejak", "Ordani", "Waerd", "Arcturian"]

for path, tag in ((EM, "ember.mjs"), (CR, "crucible-compiled.mjs")):
    lines = open(path, "r", encoding="utf-8", errors="replace").read().splitlines()
    for w in WORDS:
        pat = re.compile(r'(?<![A-Za-z0-9_])' + re.escape(w) + r'(?![A-Za-z0-9_])')
        hits = [(i + 1, l.strip()) for i, l in enumerate(lines) if pat.search(l)]
        if not hits:
            continue
        print("\n### %s @ %s —— 独立成词 %d 行" % (w, tag, len(hits)))
        for n, l in hits:
            print("   %7d | %s" % (n, l[:160]))
