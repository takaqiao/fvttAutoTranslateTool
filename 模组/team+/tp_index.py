# -*- coding: utf-8 -*-
"""把汉化项目里所有「动作 id → 中文名」「英文名 → 中文名」索引出来。"""
import json, os, sys, io
sys.stdout.reconfigure(encoding="utf-8")
ROOT = "C:/Users/Taka/Desktop/fvtt/Ember-Crucible Translation Project"
DIRS = [f"{ROOT}/1-Ember汉化插件/compendium/cn", f"{ROOT}/2-Crucible汉化插件/compendium/cn"]

by_action = {}    # actionId -> set(中文名)
by_name    = {}   # 英文条目名 -> set(中文名)
by_effect  = {}   # 效果名 -> set(中文名)

def add(d, k, v):
    if not k or not v: return
    d.setdefault(k, set()).add(v)

def walk_entry(key, val, src):
    if not isinstance(val, dict): return
    add(by_name, key, val.get("name"))
    for ak, av in (val.get("actions") or {}).items():
        if isinstance(av, dict): add(by_action, ak, av.get("name"))
        elif isinstance(av, str): add(by_action, ak, av)
    for ek, ev in (val.get("effects") or {}).items():
        if isinstance(ev, dict): add(by_effect, ek, ev.get("name"))
        elif isinstance(ev, str): add(by_effect, ek, ev)
    # 冒险包 / actor 里还会再套一层 items
    for sub in ("items", "actors", "pages"):
        s = val.get(sub)
        if isinstance(s, dict):
            for k2, v2 in s.items(): walk_entry(k2, v2, src)

for d in DIRS:
    if not os.path.isdir(d): continue
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json"): continue
        try: data = json.load(open(f"{d}/{fn}", encoding="utf-8"))
        except Exception as e: print("skip", fn, e); continue
        for k, v in (data.get("entries") or {}).items():
            walk_entry(k, v, fn)

out = {
    "by_action": {k: sorted(v) for k, v in by_action.items()},
    "by_name":   {k: sorted(v) for k, v in by_name.items()},
    "by_effect": {k: sorted(v) for k, v in by_effect.items()},
}
json.dump(out, open("tp_index.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"动作 id {len(by_action)} 个 / 条目名 {len(by_name)} 个 / 效果名 {len(by_effect)} 个")
