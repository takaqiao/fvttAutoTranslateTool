# -*- coding: utf-8 -*-
import json, os, re, sys, collections
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
D = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\4-临时脚本\2026-08-13-round12\G10"
rows = json.load(open(os.path.join(D, "j02.ember.json"), encoding="utf-8"))["rows"]
PRE = "entries.Ember Early Access.journals.Arctus Plateau Gazetteer."
by = collections.OrderedDict()
misc = []
for x in rows:
    p = x["path"][len(PRE):]
    m = re.match(r"pages\.(.*?)\.(name|overview|exposition|text)$", p)
    if m:
        by.setdefault(m.group(1), {})[m.group(2)] = x
    else:
        misc.append((p, x))

order = ["name", "overview", "exposition", "text"]
pages = list(by)
CH = 7
chunks = [pages[i:i + CH] for i in range(0, len(pages), CH)]
for i, ch in enumerate(chunks, 1):
    out = []
    for pg in ch:
        out.append("\n" + "#" * 70 + f"\n# PAGE: {pg}\n" + "#" * 70)
        for f in order:
            r = by[pg].get(f)
            if not r:
                continue
            out.append(f"\n--- [{pg}] {f} EN ---\n{r['en']}\n--- [{pg}] {f} CN ---\n{r['cn']}")
    open(os.path.join(D, f"chunk{i:02d}.txt"), "w", encoding="utf-8").write("\n".join(out))
    print(f"chunk{i:02d}", ch, sum(len(o) for o in out))
out = ["MISC"]
for p, x in misc:
    out.append(f"\n--- {p} ---\nEN: {x['en']}\nCN: {x['cn']}")
open(os.path.join(D, "chunk00_misc.txt"), "w", encoding="utf-8").write("\n".join(out))
print("misc", len(misc))
