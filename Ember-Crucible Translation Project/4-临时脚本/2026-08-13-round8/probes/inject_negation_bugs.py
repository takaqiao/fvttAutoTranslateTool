# -*- coding: utf-8 -*-
"""灵敏度回测：把已知的否定类错误注入到一份**临时副本**，看判据报不报得出来。

绝不触碰 compendium/ —— 只往 --dest 写。

用法：
  python inject_negation_bugs.py --src "<2-Crucible汉化插件>" --dest "<临时目录>"
  python scan_negation_drift.py --repo "<临时目录>" ...
"""
from __future__ import annotations
import argparse, json, os, shutil, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# (pack, 中文叶路径, 原文片段, 注入后片段, 说明)
BUGS = [
    ("crucible.rules.json",
     ["Combat", "pages", "Movement", "text"],
     "生物不能穿过另一个生物所占据的空间",
     "生物可以穿过另一个生物所占据的空间",
     "长页第 49 块：`A creature may not move through the space occupied by another creature` 的否定被抹掉"),
    ("crucible.rules.json",
     ["Combat", "pages", "Movement", "text"],
     "强制移动不能把生物推穿墙壁或其他生物",
     "强制移动可以把生物推穿墙壁或其他生物",
     "长页第 59 块：`Forced Movement cannot push a creature through walls` 被反转"),
    ("crucible.rules.json",
     ["Conditions", "pages", "Dead", "text"],
     "且除非通过魔法手段，否则不能被复活",
     "且可以被复活",
     "整条 unless 从句 + cannot 一起被译丢（`cannot be revived except by magical means`）"),
    ("crucible.talent.json",
     ["Spellmute", "description"],
     "你不能执行任何带有",
     "你可以执行任何带有",
     "`You cannot perform any action with the Spell tag` 被反转"),
]


def dig(o, path):
    for k in path[:-1]:
        o = o[k]
    return o, path[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dest", required=True)
    a = ap.parse_args()
    assert "compendium" not in os.path.abspath(a.dest).replace("\\", "/").split("/")[-2:], \
        "dest 看起来指向真库，拒绝执行"

    if os.path.exists(a.dest):
        shutil.rmtree(a.dest)
    shutil.copytree(os.path.join(a.src, "compendium"),
                    os.path.join(a.dest, "compendium"))
    print(f"副本 -> {a.dest}")

    by_pack = {}
    for pack, path, old, new, why in BUGS:
        by_pack.setdefault(pack, []).append((path, old, new, why))

    manifest = []
    for pack, items in by_pack.items():
        p = os.path.join(a.dest, "compendium", "cn", pack)
        d = json.load(open(p, encoding="utf-8-sig"))
        for path, old, new, why in items:
            parent, key = dig(d["entries"], path)
            s = parent[key]
            assert old in s, f"注入点没找到：{pack} {path} {old!r}"
            parent[key] = s.replace(old, new, 1)
            manifest.append({"pack": pack, "path": "entries." + ".".join(path),
                             "old": old, "new": new, "why": why})
            print(f"  注入 {pack} entries.{'.'.join(path)}  「{old}」->「{new}」")
        json.dump(d, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump(manifest, open(os.path.join(a.dest, "_injected.json"), "w",
                             encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"共注入 {len(manifest)} 处")


if __name__ == "__main__":
    main()
