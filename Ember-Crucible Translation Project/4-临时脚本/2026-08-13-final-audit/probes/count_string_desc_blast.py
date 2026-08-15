#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
量化 normalizeDescriptionValue 形状假设错误的爆炸半径。

crucible 0.10.1 system.json documentTypes.Item：
  description 为**字符串**(HTMLField) 的类型 : ancestry archetype background spell talent taxonomy
  description 为**对象**{public,private} 的类型: accessory armor consumable loot schematic tool weapon

register.js 的 migrateLegacyDescriptionShape / sanitizeItemDataShape 把
「typeof description === 'string'」当作 legacy 特征，于是恰好命中前 6 类。

这里数：随 Ember 冒险包导入到世界里的 actor 内嵌 item 中，有多少个名字能在
crucible-cn 的 6 个「字符串描述」包里查到（＝会被写坏的那批）。
只读，不写库。
"""
import json
from pathlib import Path

ROOT = Path(r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project")
CRUC = ROOT / "2-Crucible汉化插件" / "compendium" / "cn"
EMBER = ROOT / "1-Ember汉化插件" / "compendium" / "cn"

STRING_DESC_PACKS = ["crucible.ancestry.json", "crucible.archetype.json",
                     "crucible.background.json", "crucible.spell.json",
                     "crucible.talent.json", "crucible.taxonomy.json",
                     "crucible.adversary-talents.json"]
OBJ_DESC_PACKS = ["crucible.equipment.json", "crucible.adversary-equipment.json",
                  "crucible.crafting.json"]


def load(p):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        return {"__err__": str(e)}


def names_of(pack):
    d = load(pack)
    e = d.get("entries", {})
    return set(e.keys()) if isinstance(e, dict) else set()


string_names, obj_names = set(), set()
print("== crucible-cn 包规模 ==")
for f in STRING_DESC_PACKS:
    p = CRUC / f
    if not p.exists():
        print(f"  (missing) {f}")
        continue
    n = names_of(p)
    string_names |= n
    print(f"  string-desc  {f:42s} {len(n):5d} entries")
for f in OBJ_DESC_PACKS:
    p = CRUC / f
    if not p.exists():
        continue
    n = names_of(p)
    obj_names |= n
    print(f"  object-desc  {f:42s} {len(n):5d} entries")

print(f"\n字符串描述型文档名合计 {len(string_names)}；对象描述型 {len(obj_names)}")

# ---- Ember 冒险包里的 actor 内嵌 item ----
adv = EMBER / "ember.crucible-adventure.json"
d = load(adv)
tot_actors = tot_items = hit_string = hit_obj = unknown = 0
per_actor_hit = []
for advname, advdata in (d.get("entries") or {}).items():
    actors = advdata.get("actors") or {}
    for aname, adata in actors.items():
        tot_actors += 1
        items = (adata or {}).get("items") or {}
        h = 0
        for iname in items:
            tot_items += 1
            if iname in string_names:
                hit_string += 1
                h += 1
            elif iname in obj_names:
                hit_obj += 1
            else:
                unknown += 1
        if h:
            per_actor_hit.append((aname, h))

print(f"\n== {adv.name} ==")
print(f"  actors                    {tot_actors}")
print(f"  actor 内嵌 item 名条目      {tot_items}")
print(f"  其中名字命中「字符串描述」包 {hit_string}   <- 会被 normalizeDescriptionValue 写成 [object Object]")
print(f"  其中名字命中「对象描述」包   {hit_obj}   <- 正确跳过")
print(f"  两边都查不到（多为独有物品）  {unknown}")
print(f"  受影响 actor 数            {len(per_actor_hit)} / {tot_actors}")
print("  受影响最多的 10 个 actor：")
for a, h in sorted(per_actor_hit, key=lambda x: -x[1])[:10]:
    print(f"    {h:4d}  {a}")
