#!/usr/bin/env python3
"""How has this English term already been rendered in the campaign pack?

  python probe.py "Vorg" "Silver Beam"       # 在中文译文里找含该英文词的片段
  python probe.py --names "Lantern Roads"    # 列某卷的页名 EN -> CN
"""
import json, os, re, sys, collections

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
PACK = "ember.crucible-adventure.json"
CN = json.load(open(os.path.join(P, "1-Ember汉化插件", "compendium", "cn", PACK), encoding="utf-8"))
EN = json.load(open(os.path.join(P, "1-Ember汉化插件", "compendium", "en", PACK), encoding="utf-8"))

args = sys.argv[1:]
if args and args[0] == "--names":
    for jn in args[1:]:
        ej = EN["entries"]["Ember Early Access"]["journals"].get(jn, {})
        cj = CN["entries"]["Ember Early Access"]["journals"].get(jn, {})
        print(f"== {jn} -> {cj.get('name', '(未译)')}")
        for pn in (ej.get("pages") or {}):
            print(f"   {pn:<40} -> {(cj.get('pages') or {}).get(pn, {}).get('name', '')}")
        for k, v in (ej.get("categories") or {}).items():
            print(f"   [category] {k:<29} -> {(cj.get('categories') or {}).get(k, '')}")
    raise SystemExit

strings = []


def walk(o, path):
    if isinstance(o, dict):
        for k, v in o.items():
            walk(v, path + [str(k)])
    elif isinstance(o, list):
        for i, v in enumerate(o):
            walk(v, path + [str(i)])
    elif isinstance(o, str):
        strings.append((".".join(path), o))


walk(CN, [])
W = 34
for t in args:
    print(f"\n===== {t} =====")
    seen = collections.Counter()
    for _, s in strings:
        for m in re.finditer(re.escape(t), s):
            seen[s[max(0, m.start() - W):m.end() + W].replace("\n", " ")] += 1
    for frag, n in seen.most_common(10):
        print(f"  [{n}] …{frag}…")
    if not seen:
        print("  (中文译文里没有出现过这个英文词 —— 说明是本卷新出现的专名，需要你定名)")
