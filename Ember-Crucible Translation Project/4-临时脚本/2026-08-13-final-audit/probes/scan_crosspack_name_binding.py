# -*- coding: utf-8 -*-
"""
scan_crosspack_name_binding.py —— 判据 P 的「按名字跨包绑定」形态

背景（上游事实，babele 2.9.1 源码）：
  script/converter/document-converter.js:_fallbackPolicySteps
    "owner-package-before-generic" = ["exact-source", "owner-package", "generic"]
  script/compendium/compendium-runtime.js:#packFromRegistry
    对同 documentType 的已翻译包**倒序**遍历，返回第一个 hasTranslation(data) 的包
  script/compendium/mapped-compendium.js:hasTranslation
    `if (this.types && !this.types.includes(data.type)) return false;`
    —— 只有翻译文件声明了 `types` 才会做**子类型**判据
  script/identity/document-identity.js
    默认 match 候选键 = ["_id", "name", "sourceId"]；我们的文件按 name 导出，
    所以实际按 **name** 命中。

于是：一个没有 `_stats.compendiumSource` 的 actor 内嵌 Item，
只要它的 name 在**任意**已翻译 Item 包里出现过，就会被那个包的译文绑定，
**不检查 item.type**。本探针找出「同名跨包且译文不同」的条目 —— 这些名字
就是可能被绑错的名字集合。

假阳性模式：
  1. 现实中该名字的内嵌物品都带 compendiumSource → 走 exact-source，绑不错；
  2. 两个包里同名条目译文相同 → 绑错也看不出来（本探针已排除这种）；
  3. adventure 包里我们**显式**给该 actor 的这个 item 写了译文 → currentPayload
     覆盖 fallbackPayload，绑不错。
所以本探针只给「名字集合」，不给结论；要坐实必须再看包里的 compendiumSource。

只读。
"""
import json
import glob
import os
from collections import defaultdict

ROOT = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"

# documentType 取自上游 system.json / module.json 的 packs[].type
# （babele 的 generic 回落先按 documentType 过滤，再按 name 命中）
PACK_TYPE = {
    "crucible.adversary-equipment": "Item", "crucible.adversary-talents": "Item",
    "crucible.affixes": "ActiveEffect", "crucible.ancestry": "Item",
    "crucible.archetype": "Item", "crucible.background": "Item",
    "crucible.crafting": "Item", "crucible.equipment": "Item",
    "crucible.macros": "Macro", "crucible.rules": "JournalEntry",
    "crucible.playtest": "Adventure", "crucible.pregens": "Actor",
    "crucible.spell": "Item", "crucible.summons": "Actor",
    "crucible.talent": "Item", "crucible.taxonomy": "Item",
    "ember.character": "Item", "ember.adventure": "Adventure",
    "ember.crucible-adversary": "Item", "ember.crucible-character": "Item",
    "ember.crucible-adventure": "Adventure", "ember.crucible-affixes": "ActiveEffect",
    "ember.crucible-effects": "ActiveEffect", "ember.dnd5e-effects": "ActiveEffect",
    "ember.crucible-items": "Item", "ember.dnd5e-items": "Item",
}
ITEM_PACKS = {k for k, v in PACK_TYPE.items() if v == "Item"}


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def main():
    packs = {}
    for repo in ("1-Ember汉化插件", "2-Crucible汉化插件"):
        for f in glob.glob(os.path.join(ROOT, repo, "compendium", "cn", "*.json")):
            pid = os.path.basename(f)[:-5]
            if pid not in ITEM_PACKS:
                continue
            d = load(f)
            ent = d.get("entries", {})
            if isinstance(ent, list):
                ent = {e.get("id") or e.get("name"): e for e in ent}
            packs[pid] = ent
            if "types" in d:
                print(f"  {pid}: declares types={d['types']}")

    print(f"扫描 Item 类翻译包 {len(packs)} 个，"
          f"条目合计 {sum(len(v) for v in packs.values())}")
    print("声明 types 的包数：",
          sum(1 for pid in packs if "types" in load(
              os.path.join(ROOT, "1-Ember汉化插件" if pid.startswith("ember") else "2-Crucible汉化插件",
                           "compendium", "cn", pid + ".json"))))

    byname = defaultdict(dict)
    for pid, ent in packs.items():
        for name, tr in ent.items():
            cn = tr.get("name") if isinstance(tr, dict) else None
            byname[name][pid] = cn

    collisions = {n: v for n, v in byname.items()
                  if len(v) > 1 and len({c for c in v.values() if c}) > 1}
    print(f"\n同名跨包 = {sum(1 for v in byname.values() if len(v) > 1)} 个名字；"
          f"其中译名不同 = {len(collisions)} 个")
    for n, v in sorted(collisions.items()):
        print(f"  [{n}]")
        for pid, cn in v.items():
            print(f"      {pid:34s} -> {cn}")


if __name__ == "__main__":
    main()
