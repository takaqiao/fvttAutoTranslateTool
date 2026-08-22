# -*- coding: utf-8 -*-
"""上游**声明式** part id（`{id: "X"` 形态）全集，与图集互相印证。"""
import io, json, os, re, collections
EMBER = r"C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember"
OUT = os.path.dirname(os.path.abspath(__file__))
src = io.open(os.path.join(EMBER, "scripts/ember.mjs"), encoding="utf-8").read()
u = json.load(io.open(os.path.join(OUT, "parts_universe2.json"), encoding="utf-8"))
seg2l = u["seg2layers"]

decl = set()
for m in re.finditer(r'\bid:\s*"([A-Za-z0-9][A-Za-z0-9_./-]*)"', src):
    decl.add(m.group(1).split("/")[-1])
print("ember.mjs 里 `id: \"…\"` 声明的末段（全类型，含非部件）：%d" % len(decl))

atlas = set(seg2l)
shown = sorted(atlas & decl)
only_atlas = sorted(atlas - decl)
print("图集末段 %d ∩ 声明 = %d ；图集有而未声明 = %d" % (len(atlas), len(shown), len(only_atlas)))

# 运行时拼出来的三族
POSES = ["Backward", "Forward", "Neutral", "Sitting"]
legbases = collections.Counter()
composed = {}
for s in only_atlas:
    for p in POSES:
        if s.endswith(p) and s[:-len(p)] in decl:
            composed[s] = ("legpose", s[:-len(p)], p); legbases[s[:-len(p)]] += 1
    if s.startswith("Marbled") and s[len("Marbled"):] in decl:
        composed.setdefault(s, ("marbled", s[len("Marbled"):], None))
    if s.endswith("Lower") and s[:-5] in decl:
        composed.setdefault(s, ("lower", s[:-5], None))
print("图集有而未声明的 %d 条里，能由声明式 id 机械拼出的 %d 条" % (len(only_atlas), len(composed)))
rest = sorted(set(only_atlas) - set(composed))
print("既未声明、也拼不出的 %d 条（判为**上游未使用的图集残留**，不会上屏）：" % len(rest))
print("  ", rest[:60])
print("腿姿基名 %d 个：%s" % (len(legbases), sorted(legbases)))
json.dump({"shown_declared": shown, "composed": {k: list(v) for k, v in sorted(composed.items())},
           "leg_bases": sorted(legbases), "atlas_orphan_unused": rest},
          io.open(os.path.join(OUT, "declared.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
