# -*- coding: utf-8 -*-
"""Y1-B：把逐块核出来的 3 处真缺陷出成批次（两包各一份）。

每条都是「旧串 → 新串 + 预期出现次数」的人裁表，次数对不上就整条拒绝，
不做静默替换（照 round16 `split_dives.py::apply_manual` 的口径）。
"""
from __future__ import annotations
import json
import os
import re
import sys

sys.path.insert(0, os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "3-常用脚本", "qa")))
from scan_dropped_terms import load_json, leaves  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
REPO = "1-Ember汉化插件"
PACKS = ["ember.adventure.json", "ember.crucible-adventure.json"]
OUT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "batches"))

P = "Ember Early Access.journals."
# 路径 -> [(旧串, 新串, 预期次数, 依据)]
FIXES = {
    P + "The Book Of Tales.pages.The Signborn's Secret.text": [(
        "恶臭的脓液从贾胡德的尸体中渗出",
        "恶臭的脓液从刺客的尸体中渗出", 1,
        "EN 旧 'from the Jahud corpses' → 新 'from the assassin corpses'；同叶 blk47 'These jahud'"
        "中文写「这些贾胡德」、blk59 'these assassins' 中文写「这些刺客」，两个词在本叶已分工，"
        "只有本块停在旧版"),
    ],
    P + "Disgraced House.pages.To Copy a Key.text": [(
        "如果队伍成功说服贾尼克斯帮助队伍，则每名参与的角色都会在事件结束时推进其",
        "任何帮助说服贾尼克斯的角色，都会在事件结束时推进其", 1,
        "EN 旧 'If the party manages to Persuade Janix to help the party, each character that "
        "participates advances their…' → 新 'Any character who helps persuade Janix advances "
        "their…'：条件从「队伍整体成功」改成「任何出手帮忙的角色」，中文停在旧版"),
        (
        "如果队伍成功为贾尼克斯完成一项差事，则每名角色都会在事件结束时推进其",
        "任何为贾尼克斯完成该差事的角色，都会在事件结束时推进其", 1,
        "EN 旧 'If the party successfully completes an errand for Janix, each character advances "
        "their…' → 新 'Any character who completes the errand for Janix advances their…'：同上"),
    ],
}


def main():
    os.makedirs(OUT, exist_ok=True)
    en_all, cn_all = {}, {}
    for pack in PACKS:
        d = {}
        leaves(load_json(os.path.join(REPO, "compendium", "en", pack)).get("entries", {}), [], d)
        en_all[pack] = d
        d = {}
        leaves(load_json(os.path.join(REPO, "compendium", "cn", pack)).get("entries", {}), [], d)
        cn_all[pack] = d

    n_leaf = n_rep = 0
    for pack in PACKS:
        batch = {}
        for path, ops in FIXES.items():
            en, cn = en_all[pack].get(path), cn_all[pack].get(path)
            if en is None or cn is None:
                print(f"  ✗ {pack} 缺路径 {path[-50:]}")
                continue
            # 孪生核：本条在两包的英文必须逐字节相同，才允许同一份改写落两包
            other = [p for p in PACKS if p != pack][0]
            if en_all[other].get(path) is not None and en_all[other][path] != en:
                print(f"  ⚠ {path[-50:]} 两包英文不同，本条不镜像")
                continue
            new = cn
            for old, rep, exp, _why in ops:
                got = new.count(old)
                if got != exp:
                    print(f"  ✗ 拒绝：{pack} {path[-46:]} 「{old[:16]}…」实测 {got} 处、表里写 {exp} 处")
                    new = None
                    break
                new = new.replace(old, rep)
                n_rep += 1
            if new and new != cn:
                batch[path] = new
                n_leaf += 1
        if batch:
            p = os.path.join(OUT, f"r18-drop.{pack}")
            json.dump(batch, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            print(f"{pack:32s} {len(batch)} 叶 -> {p}")
    print(f"共写 {n_leaf} 叶 · 替换 {n_rep} 处（两包合计）")


if __name__ == "__main__":
    main()
