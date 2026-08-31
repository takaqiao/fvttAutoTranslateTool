# -*- coding: utf-8 -*-
"""诊断 3: 看 bestiary name 字段格式 + ItemSheet 一般化命名"""
import json, re

p = r"C:\Users\Taka\Desktop\fvtt\FotRP\需要翻译\NEW\pf2e.fists-of-the-ruby-phoenix-bestiary.json"
with open(p, encoding="utf-8") as f:
    data = json.load(f)

# bestiary 顶层结构 — 通常是一个 list of actors
print(f"Top-level type: {type(data).__name__}")
if isinstance(data, list):
    print(f"  List len: {len(data)}")
    for actor in data[:5]:
        if isinstance(actor, dict):
            print(f"  name: {actor.get('name', '?')}")
            # items - 看是否有 actions/equipment 等也有 bilingual name
            items = actor.get('items', [])
            for it in items[:3]:
                print(f"    item: {it.get('name', '?')}  (type: {it.get('type')})")
elif isinstance(data, dict):
    print(f"  Keys: {list(data.keys())[:20]}")
