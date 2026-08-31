# -*- coding: utf-8 -*-
"""按图层给部件显示名分组，并标出与现有 TOKEN_MAKER_UI / EMBER_WINDOW_UI 的键冲突。"""
import io, json, os, re, collections, subprocess

OUT = os.path.dirname(os.path.abspath(__file__))
ATL = r"C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules/ember/assets/tokens/maker"
frames = set()
for f in ["Character0.json", "Character1.json", "Monster0.json", "Party0.json"]:
    frames |= set(json.load(io.open(os.path.join(ATL, f), encoding="utf-8"))["frames"].keys())
p2l = collections.defaultdict(set)
for k in frames:
    seg = k.split("/")
    if len(seg) < 3: continue
    ns, ly, p = seg[0], seg[1], seg[-1]
    if p.endswith("Color"): p = p[:-5]
    p2l[p].add(ly)

d = json.load(io.open(os.path.join(OUT, "part_display_names.json"), encoding="utf-8"))
parts = d["parts"]

bylayer = collections.defaultdict(dict)
for pid, disp in parts.items():
    for ly in sorted(p2l.get(pid, {"(unknown)"})):
        bylayer[ly][pid] = disp

json.dump({k: dict(sorted(v.items())) for k, v in sorted(bylayer.items())},
          io.open(os.path.join(OUT, "parts_by_layer.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

order = sorted(bylayer, key=lambda k: -len(bylayer[k]))
for ly in order:
    print("%-16s %3d" % (ly, len(bylayer[ly])))
print("WROTE parts_by_layer.json")
