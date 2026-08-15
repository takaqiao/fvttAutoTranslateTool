#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""删掉 ember 汉化包里被误译的 `system.terrain` 枚举值（审计 2026-08-12 第 1.3 条）。

背景
----
`JournalEntryPage` 的 `ember.location` / `ember.biome` 两个子类型里，`terrain` 是
**带 choices 的枚举 id**，不是正文：

    ember/scripts/ember.mjs:418   EmberBiome.defineSchema()
        terrain: new fields.StringField({choices: terrainChoices, initial: "normal", blank: false})
        // terrainChoices = () => ember.scenes.region.terrain   (ember.mjs:409)
    ember/scripts/ember.mjs:1752  EmberLocation.defineSchema()  = EmberBiome 的 schema，
                                  只把 terrain 改成 blank:true / initial:""
    ember/scripts/ember.mjs:575   EmberBiomePage.defineSchema()    = EmberBiome.defineSchema()
    ember/scripts/ember.mjs:1903  EmberLocationPage.defineSchema() = EmberLocation.defineSchema()

合法取值就是地形注册表的 id（ember.mjs:117746 起）：
    normal / difficult / extreme / impassable / bluffs / canyon / road / water

而 `extract/mappings.mjs` 把 `terrain` 列进了 `JournalEntryPage.ember.location`
与 `.biome` 的可译字段，于是两个 adventure 包各 52 条、共 104 条被写成了中文。
中文值不在 choices 里 → DataModel 校验失败；`EmberLocationPage.initializePage()`
会把这个值灌进 `ember.scenes.region.locations[...]`，随后
`ember.mjs:846-847` 的 `this.#region.terrain.get(terrain || …)` 取不到，直接
`throw new Error("Terrain 水域 not defined in Region configuration data.")`。
而且翻了也没有收益：界面显示的是注册表里的 `terrain.label`（ember.mjs:25512），
不是这个原始值。

`qa/apply_translations.py` 只能写值、不能删键，所以清理走这个一次性脚本。

删掉之后会怎样
--------------
babele 的 `PrimitiveConverter.translate()`（babele/script/converter/primitive-converter.js）
在 `translations[field]` 为 `undefined` 时返回 `undefined`，`FieldMapping.map()` 随即
`if (typeof value !== "undefined" && value !== null)` 跳过合并 —— 也就是**原样保留英文
包里的 id**。缺键不会报错、不会写空串，页面回落成 `normal` / `water` 等正确枚举值。

安全闸（PROJECT.md 第 8 节 2026-08-12「清理类操作前必须先验证死的判据本身没错」）
-------------------------------------------------------------------------------
只有同时满足下面三条的键才会被删，其余一律列出来但**不动**：
  1. 路径形如 `entries…pages.<页名>.terrain`（`pages` 必须是倒数第三段）；
  2. 中文侧该键的值是字符串；
  3. **英文基准同路径存在，且值是上面 8 个合法 terrain id 之一**。
只要出现一条不满足的，脚本拒绝写盘（除非显式 `--allow-anomalies`）。

用法
----
    python purge_terrain.py            # --dry 是默认行为，只打印
    python purge_terrain.py --write    # 真删（由主控执行）
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_REPO = os.path.join(PROJECT_ROOT, "1-Ember汉化插件")
DEFAULT_PACKS = ["ember.crucible-adventure.json", "ember.adventure.json"]

# ember.mjs:117746 起的 `terrain` 注册表；`ember.scenes.region.terrain` 的键就是这些 id。
TERRAIN_IDS = frozenset({
    "normal", "difficult", "extreme", "impassable",
    "bluffs", "canyon", "road", "water",
})

TARGET_KEY = "terrain"


# 与 qa/apply_translations.py 的 dump() 一致：项目既有格式。
def load(path):
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def dump(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def find_terrain_keys(node, path, hits):
    """收集所有叶子键名为 `terrain` 的路径。只下钻 dict / list，不改任何东西。"""
    if isinstance(node, dict):
        for k, v in node.items():
            if k == TARGET_KEY and not isinstance(v, (dict, list)):
                hits.append((path + [k], v))
            else:
                find_terrain_keys(v, path + [k], hits)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            find_terrain_keys(v, path + [str(i)], hits)


def get_at(root, parts):
    """按路径取值。返回 (exists, value)，区分「不存在」与「存在但是 None」。"""
    node = root
    for p in parts:
        if isinstance(node, dict):
            if p not in node:
                return False, None
            node = node[p]
        elif isinstance(node, list):
            try:
                node = node[int(p)]
            except (ValueError, IndexError):
                return False, None
        else:
            return False, None
    return True, node


def delete_at(root, parts):
    node = root
    for p in parts[:-1]:
        node = node[int(p)] if isinstance(node, list) else node[p]
    del node[parts[-1]]


def show(parts):
    return " | ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", default=DEFAULT_REPO, help="汉化仓库根目录（默认 1-Ember汉化插件）")
    ap.add_argument("--pack", action="append", dest="packs",
                    help="包文件名，可重复；默认两个 adventure 包")
    ap.add_argument("--write", action="store_true", help="真的落盘（不加就是 --dry）")
    ap.add_argument("--dry", action="store_true", help="只打印，不落盘（默认行为，写出来是为了显式）")
    ap.add_argument("--allow-anomalies", action="store_true",
                    help="即使存在不满足判据的 terrain 键，也允许写盘（默认拒绝）")
    a = ap.parse_args()

    packs = a.packs or DEFAULT_PACKS
    write = a.write and not a.dry

    total_del = 0
    total_anom = 0
    planned = []  # (pack, cn_obj, cn_path, [parts...])

    for pack in packs:
        cn_path = os.path.join(a.repo, "compendium", "cn", pack)
        en_path = os.path.join(a.repo, "compendium", "en", pack)
        if not os.path.exists(cn_path):
            print(f"{pack}\n  跳过：{cn_path} 不存在")
            continue
        if not os.path.exists(en_path):
            print(f"{pack}\n  跳过：英文基准 {en_path} 不存在，无法校验判据")
            total_anom += 1
            continue

        cn = load(cn_path)
        en = load(en_path)

        hits = []
        find_terrain_keys(cn, [], hits)

        to_delete = []
        anomalies = []
        for parts, cn_val in hits:
            reasons = []
            if len(parts) < 3 or parts[-3] != "pages":
                reasons.append("路径不是 …pages.<页名>.terrain")
            if not isinstance(cn_val, str):
                reasons.append(f"中文侧值不是字符串（{type(cn_val).__name__}）")
            en_exists, en_val = get_at(en, parts)
            if not en_exists:
                reasons.append("英文基准同路径不存在")
            elif not isinstance(en_val, str) or en_val not in TERRAIN_IDS:
                reasons.append(f"英文值 {en_val!r} 不是合法 terrain id")
            if reasons:
                anomalies.append((parts, cn_val, en_val if en_exists else None, reasons))
            else:
                to_delete.append((parts, cn_val, en_val))

        print(f"{pack}")
        print(f"  将删除 {len(to_delete)} 条 terrain 键"
              f"{'（dry run，未落盘）' if not write else ''}")
        for parts, cn_val, en_val in to_delete:
            print(f"    - {show(parts)}")
            print(f"        cn={cn_val!r}  ->  删除后回落英文 en={en_val!r}")
        if anomalies:
            print(f"  !! 不满足判据、保持不动 {len(anomalies)} 条：")
            for parts, cn_val, en_val, reasons in anomalies:
                print(f"    ? {show(parts)}")
                print(f"        cn={cn_val!r} en={en_val!r} 原因={'；'.join(reasons)}")

        total_del += len(to_delete)
        total_anom += len(anomalies)
        planned.append((pack, cn, cn_path, to_delete))

    print()
    print(f"合计：可删 {total_del} 条，异常 {total_anom} 条")

    if not write:
        print("(dry run: nothing written)")
        return 1 if total_anom else 0

    if total_anom and not a.allow_anomalies:
        print("拒绝写盘：存在不满足判据的 terrain 键。先人工确认，或加 --allow-anomalies。")
        return 2

    for pack, cn, cn_path, to_delete in planned:
        # 从深到浅删，避免同一父节点里的索引位移（本例没有数组，仍按此顺序保守处理）
        for parts, _cn_val, _en_val in sorted(to_delete, key=lambda x: -len(x[0])):
            delete_at(cn, parts)
        dump(cn_path, cn)
        print(f"已写入 {cn_path}（删除 {len(to_delete)} 条）")

    # 落盘后复读一遍，确认剩余 terrain 键数 == 异常数
    remaining = 0
    for pack, _cn, cn_path, _td in planned:
        again = load(cn_path)
        left = []
        find_terrain_keys(again, [], left)
        remaining += len(left)
    print(f"复读校验：落盘后剩余 terrain 键 {remaining} 条（应等于异常数 {total_anom}）")
    return 0 if remaining == total_anom else 3


if __name__ == "__main__":
    raise SystemExit(main())
