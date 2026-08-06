"""Count competing renderings across the LIVE cn packs (orphan journals excluded,
since those are dead translations Babele never applies)."""
import json, os, re, sys

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
EN_DIR = os.path.join(P, "1-Ember汉化插件", "compendium", "en")
CN_DIR = os.path.join(P, "1-Ember汉化插件", "compendium", "cn")
PACK = "ember.crucible-adventure.json"

en = json.load(open(os.path.join(EN_DIR, PACK), encoding="utf-8"))
cn = json.load(open(os.path.join(CN_DIR, PACK), encoding="utf-8"))
EJ = set(en["entries"]["Ember Early Access"]["journals"])

live = []


def walk(o, path):
    if isinstance(o, dict):
        for k, v in o.items():
            walk(v, path + [str(k)])
    elif isinstance(o, list):
        for i, v in enumerate(o):
            walk(v, path + [str(i)])
    elif isinstance(o, str):
        live.append((".".join(path), o))


root = cn["entries"]["Ember Early Access"]
for section, node in root.items():
    if section == "journals":
        for jn, j in node.items():
            if jn in EJ:                      # 跳过孤儿卷
                walk(j, ["journals", jn])
    else:
        walk(node, [section])

blob = "\n".join(s for _, s in live)
DEFAULT = ["底层区", "矿渊", "调谐", "同调", "拉力之家", "聚归馆", "祖裔", "血统",
           "“", "「", "阿玛尔忒亚", "阿玛尔忒娅", "因卡罗", "印卡罗"]
for t in (sys.argv[1:] or DEFAULT):
    print(f"{t:<18}{blob.count(t):>6}")
