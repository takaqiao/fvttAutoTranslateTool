# -*- coding: utf-8 -*-
"""每个末段 id 在 ember.mjs 里是不是**字面量**（面板 D 档的口径：整串出现即可）。"""
import io, json, os, re
EMBER = r"C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember"
OUT = os.path.dirname(os.path.abspath(__file__))
src = io.open(os.path.join(EMBER, "scripts/ember.mjs"), encoding="utf-8").read()
d = json.load(io.open(os.path.join(OUT, "parts_universe2.json"), encoding="utf-8"))
segs = list(d["seg2layers"])
lit, non = [], []
for s in segs:
    (lit if s in src else non).append(s)
print("末段 id 共 %d：ember.mjs 里有字面量 %d / 没有 %d" % (len(segs), len(lit), len(non)))
json.dump({"literal": sorted(lit), "nonliteral": sorted(non)},
          io.open(os.path.join(OUT, "litcheck.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("没有字面量的前 40：", sorted(non)[:40])
