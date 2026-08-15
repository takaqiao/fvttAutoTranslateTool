# -*- coding: utf-8 -*-
"""G2 复核探针 2：
A) BY_DESIGN 档里「英文近乎相同」（去冠词/标点/大小写/空白后相等，或一方是另一方前缀）
   的行，看中文是否也跟着近乎相同。英文只差一个 the，中文却南辕北辙 = 真缺陷被藏进 BY_DESIGN。
B) 孪生包针脚标签比对（ember.adventure vs ember.crucible-adventure，同场景同英文 text）。
C) 全库表结果/针脚标签：同一英文标签在库内出现多次时中文是否统一（跨表/跨场景）。
"""
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "3-常用脚本", "qa"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import scan_name_binding as G

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
S = (r"C:\Users\Taka\AppData\Local\Temp\claude\C--Users-Taka-Desktop-fvtt"
     r"\e57b5596-8975-4155-bc4b-c3126ad4aad5\scratchpad")

probe = json.load(open(os.path.join(S, "rev_g2", "probe.json"), encoding="utf-8"))


def norm(s):
    s = (s or "").lower().strip()
    s = re.sub(r"^(the|a|an)\s+", "", s)
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


near = []
for r in probe["by_design"]:
    a, b = norm(r["en"]), norm(r["tgt_en"])
    if not a or not b:
        continue
    if a == b or a.startswith(b) or b.startswith(a):
        near.append(r)

print(f"BY_DESIGN 总 {len(probe['by_design'])}，其中英文近乎相同的 {len(near)}")
for r in near:
    cands = "/".join(r["cand"].keys()) if r["cand"] else "(无中文名)"
    print(f"  [{r['kind']}] {r['pack']} :: {r['path']}")
    print(f"     EN 标签 {r['en']!r} vs 目标 {r['tgt_en']!r}")
    print(f"     CN 标签 {r['cn']!r} vs 目标中文 {cands}")

# ---------- B) 孪生包针脚 ----------
repos = [os.path.join(P, "1-Ember汉化插件"), os.path.join(P, "2-Crucible汉化插件")]
packs = G.load_packs(repos)
notes_by = collections.defaultdict(dict)
labels = collections.defaultdict(lambda: collections.defaultdict(list))
for pk, pv in packs.items():
    en, cn = pv["en"], pv["cn"]
    for ent, ev in (en.get("entries") or {}).items():
        cv = (cn.get("entries") or {}).get(ent) or {}
        for sname, sv in (ev.get("scenes") or {}).items():
            csv_ = ((cv.get("scenes") or {}).get(sname) or {})
            for text, en_label in (sv.get("notes") or {}).items():
                if not isinstance(en_label, str):
                    continue
                cl = (csv_.get("notes") or {}).get(text)
                path = f"{ent}.scenes.{sname}.notes.{text}"
                if pv["file"] in ("ember.adventure.json", "ember.crucible-adventure.json"):
                    notes_by[(sname, text)][pv["file"]] = (cl, path)
                labels[en_label][cl].append(f"{pv['file']}::{path}")

conf = [(k, v) for k, v in notes_by.items()
        if len(v) == 2 and v["ember.adventure.json"][0] != v["ember.crucible-adventure.json"][0]]
print(f"\n孪生包针脚标签冲突：{len(conf)}")
for k, v in conf:
    print(f"  场景 {k[0]!r} 针脚 {k[1]!r}: adv {v['ember.adventure.json'][0]!r} vs "
          f"cru {v['ember.crucible-adventure.json'][0]!r}")

# ---------- C) 同英文标签库内多译 ----------
multi = {k: v for k, v in labels.items() if len({c for c in v if c}) > 1}
print(f"\n同一英文针脚标签库内出现多个中文的：{len(multi)}")
for k, v in sorted(multi.items())[:40]:
    print(f"  EN {k!r} -> {[(c, len(p)) for c, p in v.items()]}")

json.dump({"near_identical_by_design": near,
           "twin_note_conflicts": [{"scene": k[0], "text": k[1],
                                    "adv": v["ember.adventure.json"][0],
                                    "cru": v["ember.crucible-adventure.json"][0]}
                                   for k, v in conf],
           "multi_cn_per_en_label": {k: {c: p for c, p in v.items()} for k, v in multi.items()}},
          open(os.path.join(S, "rev_g2", "probe2.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
