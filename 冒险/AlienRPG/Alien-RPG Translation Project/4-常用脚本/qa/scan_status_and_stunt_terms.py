#!/usr/bin/env python3
"""闸：状态菜单与技能特技必须使用业主确认的运行时译名。

这道闸只检查真正会交付给 Foundry 的语言文件、合集翻译和运行时映射。
它不会全局禁止「冷冻」：休眠/冷冻舱语境里的该词仍然可能正确；这里只钉死
Freezing 状态及其说明。相反，「炫技」在本项目的 Stunt 机制中是错义，所有
运行时表面都不得再出现。
"""
import io
import json
import os
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
HUB = os.path.join(ROOT, "1-系统汉化插件")
STARTER = os.path.join(ROOT, "2-新手包汉化插件")


def jload(path):
    with io.open(path, encoding="utf-8") as f:
        return json.load(f)


def flat(value, prefix=""):
    out = {}
    if isinstance(value, dict):
        for key, child in value.items():
            path = "%s.%s" % (prefix, key) if prefix else key
            out.update(flat(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            path = "%s[%d]" % (prefix, index)
            out.update(flat(child, path))
    else:
        out[prefix] = value
    return out


def main():
    cn_path = os.path.join(HUB, "lang", "cn.json")
    cn = flat(jload(cn_path))
    expected = {
        "ALIENRPG.Freezing": "受冻",
        "ALIENRPG.Encumbered": "超重",
        "ALIENRPG.gravitydyspraxia": "重力失调",
        "ALIENRPG.CriticalInjuries": "重伤",
        "ALIENRPG.Stunts": "特技",
        "ITEM.TypeSkill-stunts": "技能特技",
        "TYPES.Item.skill-stunts": "技能特技",
    }

    checked = 0
    failures = []
    for key, want in expected.items():
        checked += 1
        got = cn.get(key)
        if got != want:
            failures.append("%s：应为 %r，实为 %r" % (key, want, got))

    # Freezing 是角色受寒状态，不是把某个对象冷冻起来。
    checked += 1
    freezing_help = cn.get("ALIENRPG.TAH.freezing")
    if not isinstance(freezing_help, str) or "受冻" not in freezing_help:
        failures.append("ALIENRPG.TAH.freezing：说明没有使用状态名「受冻」")
    elif "冷冻有几个影响" in freezing_help or "冷冻状态" in freezing_help:
        failures.append("ALIENRPG.TAH.freezing：仍残留旧状态名「冷冻」")

    # Stunt 的所有用户可见运行时表面都应统一为「特技」。
    runtime_paths = [
        cn_path,
        os.path.join(HUB, "lang", "plugins", "evolved-stunts-cn.json"),
        os.path.join(HUB, "compendium", "cn", "alienrpg.alien-rpg-system.json"),
        os.path.join(HUB, "module.json"),
        os.path.join(HUB, "scripts", "alienrpg-hardcoded-cn.mjs"),
        os.path.join(STARTER, "compendium", "cn", "alien-evolved-starterset.alien-evolved-starter-set.json"),
    ]
    for path in runtime_paths:
        checked += 1
        with io.open(path, encoding="utf-8") as f:
            text = f.read()
        if "炫技" in text:
            failures.append("%s：仍包含错义词「炫技」" % os.path.relpath(path, ROOT))

    evolved = cn.get("ALIENRPG.EvolvedStunts")
    checked += 1
    if not isinstance(evolved, str) or "特技" not in evolved:
        failures.append("ALIENRPG.EvolvedStunts：规则说明没有采用「特技」")

    for failure in failures:
        print("  ✗ " + failure)
    print("checked=%d  violations=%d" % (checked, len(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
