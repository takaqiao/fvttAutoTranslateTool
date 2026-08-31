# -*- coding: utf-8 -*-
"""诊断 4: 进 entries 看 actor.name 格式"""
import json, re

p = r"C:\Users\Taka\Desktop\fvtt\FotRP\需要翻译\NEW\pf2e.fists-of-the-ruby-phoenix-bestiary.json"
with open(p, encoding="utf-8") as f:
    data = json.load(f)

entries = data.get("entries", {})
print(f"Entries type: {type(entries).__name__}")
if isinstance(entries, dict):
    print(f"  Keys count: {len(entries)}")
    keys = list(entries.keys())[:30]
    for k in keys:
        v = entries[k]
        if isinstance(v, dict):
            name = v.get("name", "?")
            print(f"  [{k}] name='{name}'")

# 看 mapping 字段 (i18n 映射)
print("\n--- mapping field ---")
mapping = data.get("mapping", {})
if isinstance(mapping, dict):
    for k, v in list(mapping.items())[:10]:
        print(f"  {k}: {v}")
