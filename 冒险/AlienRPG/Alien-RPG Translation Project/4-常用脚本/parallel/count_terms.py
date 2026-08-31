"""Count competing renderings across the LIVE cn packs (orphan journals excluded,
since those are dead translations Babele never applies)."""
import json, os, re, sys


def sole_entry(pack):
    """三个 Alien 包都是**单 Adventure 文档包**，顶层 entries 恰好一条，
    名字随包不同（Alien RPG System / Alien Evolved Starter Set /
    Alien Evolved Core Rules）。EC 版把 "Ember Early Access" 写死在调用处，
    换个包就静默返回空表。这里现取，取不到就当场喊。"""
    e = (pack or {}).get("entries") or {}
    if len(e) != 1:
        raise SystemExit(f"顶层 entries 有 {len(e)} 条，预期 1 条（单 Adventure 包）：{list(e)[:5]}")
    return next(iter(e.values()))


P = r"C:\Users\Taka\Desktop\fvtt\Alien-RPG Translation Project"
EN_DIR = os.path.join(P, "3-核心书汉化插件", "compendium", "en")
CN_DIR = os.path.join(P, "3-核心书汉化插件", "compendium", "cn")
PACK = "alien-evolved-corerules.alien-evolved-core-rules.json"

en = json.load(open(os.path.join(EN_DIR, PACK), encoding="utf-8"))
cn = json.load(open(os.path.join(CN_DIR, PACK), encoding="utf-8"))
EJ = set(sole_entry(en)["journals"])

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


root = sole_entry(cn)
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
