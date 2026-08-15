#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""count_string_desc_blast.py 的加强版：把 Ember 自己的 crucible 侧包也算进名字池。"""
import json
import pathlib

R = pathlib.Path(r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project")
C = R / "2-Crucible汉化插件" / "compendium" / "cn"
E = R / "1-Ember汉化插件" / "compendium" / "cn"


def names(p):
    d = json.loads(p.read_text(encoding="utf-8"))
    e = d.get("entries", {})
    return set(e.keys()) if isinstance(e, dict) else set()


STR_PACKS = ["crucible.ancestry.json", "crucible.archetype.json", "crucible.background.json",
             "crucible.spell.json", "crucible.talent.json", "crucible.taxonomy.json",
             "crucible.adversary-talents.json"]
EMB_STR_PACKS = ["ember.crucible-character.json", "ember.crucible-adversary.json"]
OBJ_PACKS = ["crucible.equipment.json", "crucible.adversary-equipment.json",
             "crucible.crafting.json", "ember.crucible-items.json"]

s = set()
for f in STR_PACKS:
    p = C / f
    if p.exists():
        s |= names(p)
for f in EMB_STR_PACKS:
    p = E / f
    if p.exists():
        n = names(p)
        s |= n
        print(f"{f}: {len(n)}")

o = set()
for f in OBJ_PACKS:
    for base in (C, E):
        p = base / f
        if p.exists():
            o |= names(p)

d = json.loads((E / "ember.crucible-adventure.json").read_text(encoding="utf-8"))
ti = hs = ho = un = acts = hit_actors = 0
for a, ad in (d.get("entries") or {}).items():
    for an, aa in (ad.get("actors") or {}).items():
        acts += 1
        h = 0
        for iname in ((aa or {}).get("items") or {}):
            ti += 1
            if iname in s:
                hs += 1
                h += 1
            elif iname in o:
                ho += 1
            else:
                un += 1
        if h:
            hit_actors += 1

print(f"string-desc name pool {len(s)}   obj-desc pool {len(o)}")
print(f"actors {acts}   embedded item entries {ti}")
print(f"string-desc hits {hs}   obj-desc hits {ho}   unknown {un}   affected actors {hit_actors}")
