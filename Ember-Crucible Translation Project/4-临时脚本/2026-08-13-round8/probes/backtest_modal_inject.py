# -*- coding: utf-8 -*-
"""scan_modal_strength.py 的灵敏度回测：往**临时副本**里注入已知的情态错配，
确认判据能报出来。绝不触碰 compendium/ 本体。

用法：
  python backtest_modal_inject.py --repo "<项目根>\\2-Crucible汉化插件" --work <临时目录>
"""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
SCAN = os.path.join(HERE, "scan_modal_strength.py")

# (pack, 中文原文片段, 替换成, 期望被哪个 mode 报出, 说明)
INJECTIONS = [
    ("crucible.rules.json",
     "游戏主持人可以在任何时候要求一次",
     "游戏主持人必须在任何时候要求一次",
     "conflict", "may -> 必须（GM 的自由裁量被写成强制）"),
    ("crucible.rules.json",
     "游戏主持人可以把这样的对手指定为",
     "游戏主持人不得把这样的对手指定为",
     "conflict", "may -> 不得（极性翻转）"),
    ("crucible.rules.json",
     "作为游戏主持人和故事讲述者，你应当与玩家合作",
     "作为游戏主持人和故事讲述者，你必须与玩家合作",
     "conflict", "should -> 必须（建议被写成强制）"),
    ("crucible.talent.json",
     "游戏主持人必须为你提供一个适合你探询方向的单词线索",
     "游戏主持人可以为你提供一个适合你探询方向的单词线索",
     "conflict", "must -> 可以（强制被写成自由裁量）"),
    ("crucible.talent.json",
     "你每 回合 可进入或更换架式一次",
     "你每 回合 必须进入或更换架式一次",
     "conflict", "may -> 必须（每回合可换架式变成必须换）"),
    ("crucible.equipment.json",
     "它必须用重型锁链固定在一个不可移动的物体上",
     "它可以用重型锁链固定在一个不可移动的物体上",
     "conflict", "must -> 可以（装备安装要求被弱化）"),
    ("crucible.rules.json",
     "游戏主持人应决定是把成功归于所有角色",
     "玩家应决定是把成功归于所有角色",
     "actor", "the Gamemaster should -> 玩家应（施动者错位）"),
    ("crucible.rules.json",
     "词缀 是可应用于 武器 、 护甲 或 配件 的魔法效果",
     "词缀 是必须应用于 武器 、 护甲 或 配件 的魔法效果",
     "conflict", "can be applied -> 必须应用"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--work", required=True)
    a = ap.parse_args()

    dst = os.path.join(a.work, "injected-repo")
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    shutil.copytree(os.path.join(a.repo, "compendium"),
                    os.path.join(dst, "compendium"))
    print(f"副本: {dst}")

    applied = []
    for pack, old, new, mode, note in INJECTIONS:
        p = os.path.join(dst, "compendium", "cn", pack)
        txt = open(p, encoding="utf-8-sig").read()
        n = txt.count(old)
        if n == 0:
            print(f"  [跳过] 锚点未命中: {pack} :: {old[:24]}")
            continue
        open(p, "w", encoding="utf-8").write(txt.replace(old, new))
        applied.append((pack, new, mode, note, n))
        print(f"  [注入] {mode:8} x{n}  {note}")

    results = {}
    for mode in ("conflict", "actor", "invented"):
        out = os.path.join(a.work, f"inject_{mode}.json")
        subprocess.run([sys.executable, SCAN, "--repo", dst, "--mode", mode,
                        "--out", out, "--show", "0"],
                       check=True, capture_output=True)
        results[mode] = json.load(open(out, encoding="utf-8"))["findings"]

    print("\n=== 灵敏度 ===")
    caught = 0
    for pack, new, mode, note, n in applied:
        key = new[:14]
        hit = [f for f in results[mode] if key in f["cn"] and f["pack"] == pack]
        if not hit:  # 允许被别的 mode 抓到，也算命中
            hit = [f for m in results for f in results[m]
                   if key in f["cn"] and f["pack"] == pack]
        ok = bool(hit)
        caught += ok
        tag = "命中" if ok else "**漏报**"
        extra = f"  <- {hit[0]['kind']}" if hit else ""
        print(f"  {tag}  [{mode}] {note}{extra}")
    print(f"\n注入 {len(applied)} 条，报出 {caught} 条")
    for mode in results:
        print(f"  {mode} 总报出 {len(results[mode])} 条"
              f"（两仓库干净基线 conflict=4 / actor=0 / invented=41；"
              f"本副本只含 2-Crucible，其干净基线三项全为 0，"
              f"所以这里报出的每一条都应当是注入项）")


if __name__ == "__main__":
    main()
