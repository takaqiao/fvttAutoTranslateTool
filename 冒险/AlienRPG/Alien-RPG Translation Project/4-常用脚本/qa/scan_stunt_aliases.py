#!/usr/bin/env python3
"""闸：技能特技的**中文键别名**必须与技能名逐条对上。

这条闸在防什么
--------------
经典（非进化版）角色卡上的特技按钮走的是一条**按名字拼 key** 的路径：

    templates/actor/character-skills.hbs:15
        data-pmbut='{{skill.description}}'
    module/data/actor-character.mjs:468
        this.skills[skl].description = game.i18n.localize(CONFIG.ALIENRPG.skills[skl].name)
    module/sheets/character-sheet.mjs:1113-1116
        const newLangStr = langStr.replace(/\\s+/g, "");
        temp3 = game.i18n.localize("ALIENRPG." + newLangStr);

也就是说 key 是用**已经本地化的技能名**拼出来的。英文下 "Close Combat" → 去空格 →
`ALIENRPG.CloseCombat`，正好命中 alien-evolved-* 两个模块提供的特技键。
中文下技能名是「近战」，拼出来是 `ALIENRPG.近战` —— **上游没有这个键，永远落空**，
按钮只会显示「未录入特技」。这是系统自身的 i18n 缺陷，只在英文下成立。

我们的处理：在 evolved-stunts-cn.json 里额外提供 12 个**中文键别名**，
内容与对应的英文键逐字节相同。于是中文下那条查找也能命中。

⚠ 别名是跟着**技能名**走的。哪天把「近战」改成别的说法，别名就对不上了，
而且**不会报错**——按钮只是又变回「未录入特技」。这道闸就是那个警报。

    python scan_stunt_aliases.py
"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
HUB = os.path.join(ROOT, "1-系统汉化插件")

# CONFIG.ALIENRPG.skills 的 key → (技能名的 lang 键, 特技内容的 lang 键)
# 取自 systems/alienrpg/module/helpers/config.mjs:53-64
SKILLS = {
    "heavyMach": ("ALIENRPG.SkillheavyMach", "ALIENRPG.HeavyMachinery"),
    "closeCbt": ("ALIENRPG.SkillcloseCbt", "ALIENRPG.CloseCombat"),
    "stamina": ("ALIENRPG.Skillstamina", "ALIENRPG.Stamina"),
    "rangedCbt": ("ALIENRPG.SkillrangedCbt", "ALIENRPG.RangedCombat"),
    "mobility": ("ALIENRPG.Skillmobility", "ALIENRPG.Mobility"),
    "piloting": ("ALIENRPG.Skillpiloting", "ALIENRPG.Piloting"),
    "command": ("ALIENRPG.Skillcommand", "ALIENRPG.Command"),
    "manipulation": ("ALIENRPG.Skillmanipulation", "ALIENRPG.Manipulation"),
    "medicalAid": ("ALIENRPG.SkillmedicalAid", "ALIENRPG.MedicalAid"),
    "observation": ("ALIENRPG.Skillobservation", "ALIENRPG.Observation"),
    "survival": ("ALIENRPG.Skillsurvival", "ALIENRPG.Survival"),
    "comtech": ("ALIENRPG.Skillcomtech", "ALIENRPG.Comtech"),
}


def flat(d, p=""):
    out = {}
    for k, v in (d or {}).items():
        kk = "%s.%s" % (p, k) if p else k
        if isinstance(v, dict):
            out.update(flat(v, kk))
        else:
            out[kk] = v
    return out


def main():
    cn = flat(json.load(io.open(os.path.join(HUB, "lang", "cn.json"), encoding="utf-8")))
    stunts_path = os.path.join(HUB, "lang", "plugins", "evolved-stunts-cn.json")
    st = flat(json.load(io.open(stunts_path, encoding="utf-8")))

    checked = bad = 0
    for skl, (name_key, stunt_key) in SKILLS.items():
        checked += 1
        name = cn.get(name_key)
        if not name:
            print("  ✗ %s：技能名 %s 在 cn.json 里不存在" % (skl, name_key))
            bad += 1
            continue
        # 系统会去掉空白再拼 key
        alias = "ALIENRPG." + "".join(name.split())
        if stunt_key not in st:
            print("  ✗ %s：特技本体 %s 不在 evolved-stunts-cn.json 里" % (skl, stunt_key))
            bad += 1
            continue
        if alias not in st:
            print("  ✗ %s：技能名是「%s」，但缺少别名键 %s —— "
                  "经典角色卡的特技按钮会显示「未录入特技」" % (skl, name, alias))
            bad += 1
            continue
        if st[alias] != st[stunt_key]:
            print("  ✗ %s：别名 %s 的内容与 %s 不一致" % (skl, alias, stunt_key))
            bad += 1

    # 反向：别名不许多出来（技能名改过之后留下的孤儿）
    wanted = {"ALIENRPG." + "".join((cn.get(n) or "x").split()) for n, _ in SKILLS.values()}
    for k in st:
        tail = k.split(".", 1)[1]
        if any("一" <= c <= "鿿" for c in tail) and k not in wanted:
            print("  ✗ 多余的中文别名键 %s —— 没有任何技能名会拼出它" % k)
            bad += 1

    print("checked=%d  violations=%d" % (checked, bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
