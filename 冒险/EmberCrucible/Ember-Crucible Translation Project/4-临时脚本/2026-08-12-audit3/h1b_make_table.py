#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H1-b：把逐键判定表拼成 markdown（键 | 英文 | 现中文 | 判定 | 依据/建议）。

数据来源：
  h1_keys.json   .mjs 的表
  h1_gate.tsv    compendium 英文闸三桶
  lang           两个仓库的 cn.json（按英文值反查）
  人工裁决        VERDICT 覆盖表（只有这一部分是判断，其余是机械汇总）
只读。
"""
import json
import os
import sys

P = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
MOD = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\modules\ember"
SYS = r"C:\Users\Taka\AppData\Local\FoundryVTT\Data\systems\crucible"

# 人工裁决：键 -> (判定, 依据/建议)。未列出的默认「通过」。
VERDICT = {
    ("PREFIXED", "Ancestry"): ("⛔ 缺失", "enricher 只发 `Ancestry: {name}`（252 处），PREFIXED 里没有这一条 → 前缀永远是英文。补 `{en:\"Ancestry\", cn:\"血统\", table:{}}`"),
    ("PREFIXED", "Culture"): ("⛔ 缺失", "同上，222 处"),
    ("PREFIXED", "Path"): ("⛔ 缺失", "同上，214 处"),
    ("EXACT", "Attunement"): ("⛔ 缺失", "角色创建第 IV 步的 step.label（crucible-async.mjs:44 / ember.mjs:121888）是裸 `Attunement`；EXACT 没有它，PREFIXED 又要求带冒号 → 中文向导里夹一个英文步骤名"),
    ("EXACT", "Token"): ("⛔ 缺失", "同上，最后一步 `Token`。两个 lang 一致作**指示物**"),
    ("EXACT", "Class"): ("⛔ 缺失", "dnd5e 侧第 II 步 `Class` → 职业（附带项）"),
    ("EXACT", "-6 Banes"): ("⛔ 缺失", "`@Advantage[-6]` 实际出现 2 次，enrichAdvantage 渲染 `-6 Banes`，表里只有 -1/-2/-3 → 建议整组改用正则"),
    ("EXACT", "Ancestry"): ("✅ 通过（改注释）", "不是「增强器前缀单独出现」，而是角色创建 step.label（ember.mjs:121864）。键有用，注释写错了"),
    ("EXACT", "Culture"): ("✅ 通过（改注释）", "同上（crucible-async.mjs:25）"),
    ("EXACT", "Path"): ("✅ 通过（改注释）", "同上（crucible-async.mjs:34）"),
    ("EXACT", "Setting the Scene"): ("⚠ 跨通道不一致", "compendium 场景设定 8 : 设定场景 2；ember lang 的 3 个 `*.FIELDS.exposition.label` 却是**设定场景**。同一页「查看」与「编辑」两态同屏 → 改 lang 从多数"),
    ("EXACT", "Secret Lore"): ("⚠ 跨通道不一致", "compendium 秘辛 2:0、.mjs 秘辛，ember lang `EMBER.DEVELOPMENT.FIELDS.development.secrets.label` 却是**隐秘背景** → 改 lang"),
    ("EXACT", "Critical Success"): ("⚠ 库内不一致（非 .mjs 之过）", ".mjs 与 crucible lang 一致（大成功）；compendium 英文闸下 大成功 74 : 重大成功 15 : **严重成功 9**（「严重成功」不成词，已出批次）"),
    ("EXACT", "Critical Failure"): ("⚠ 库内不一致（非 .mjs 之过）", ".mjs 与 lang 一致（严重失败）；compendium 严重失败 7 : 大失败 6 : 重大失败 3 : 其它 3，需一次术语统一"),
    ("RESULTS", "Critical Success"): ("⚠ 同上", "同 EXACT 那条，同一个词"),
    ("RESULTS", "Critical Failure"): ("⚠ 同上", "同 EXACT 那条"),
    ("EXACT", "Begin Event"): ("⚠ 跨通道不一致", "GM 指南 6 处 `<span class=\"reference\">Begin Event</span>` 仍是英文，按钮已是中文 → 已出批次改 span"),
    ("EXACT", "Complete Event"): ("⚠ 跨通道不一致", "同上 2 处"),
    ("EXACT", "Award Attunements"): ("⚠ 跨通道不一致", "同上 2 处"),
    ("EXACT", "No Awarded Attunements"): ("⚠ 跨通道不一致", "同上 2 处"),
    ("ATTUNEMENTS", "Aura"): ("⚠ lang 未同步", "compendium name 是「奥拉 Aura」、.mjs 已是奥拉；但 ember lang `SPELL.INFLECTIONS.Aura`（图标 attunements/Aura.webp，与另 11 个月名同列）仍是**灵气** → 改 lang"),
    ("ATTUNEMENTS", "The Abyss"): ("ℹ 冗余", "`attunement.name` 取的是 Cosmos 页 `page.name`，已被 babele 译成「深渊 The Abyss」，本键在中文世界永不命中（英文世界才用得上），无害"),
    ("ATTUNEMENTS", "Heart of Ember"): ("ℹ 冗余", "同上，页名已是「余烬之心 Heart of Ember」"),
    ("KNOWLEDGE", "Outsiders"): ("ℹ crucible 侧已死", "ember `initialize()` 里 `delete crucible.CONFIG.knowledge.outsiders`；dnd5e 侧仍在用，保留"),
    ("PREFIXED", "Knowledge"): ("ℹ crucible 侧走不到", "crucible 自己注册了 `crucibleKnowledge` 增强器并用 i18n `ACTOR.KnowledgeSpecific`＝「知识：{knowledge}」，与本条同形；本条实际只在 dnd5e 侧生效"),
    ("PREFIXED", "Language"): ("✅ 通过", "crucible 的 `ACTOR.LanguageSpecific`＝「语言：{language}」与本条**逐字相同**，两条通道谁先命中都一样"),
    ("MOODS", "Calm"): ("✅ 通过", "`EmberSoundscape.MOODS` 实测只有 calm/tension；取 `EMBER.SoundscapeMoodCalm`"),
    ("MOODS", "Tension"): ("✅ 通过", "同上"),
    ("CALENDAR_DAY_ABBR", "Tues"): ("✅ 通过", "实测 ember 写的就是 `Tues` 不是 `Tue`（ember.mjs:3639），拼错就永不命中，此处对"),
}

TABLES_ORDER = ["ATTUNEMENTS", "LANGUAGES", "KNOWLEDGE", "MOODS", "EXACT", "RESULTS",
                "CALENDAR_MONTHS", "CALENDAR_DAYS", "CALENDAR_DAY_ABBR", "PREFIXED"]


def flat(o, pre="", out=None):
    if out is None:
        out = {}
    if isinstance(o, dict):
        for k, v in o.items():
            flat(v, f"{pre}.{k}" if pre else k, out)
    elif isinstance(o, str):
        out[pre] = o
    return out


def main():
    keys = json.load(open(sys.argv[1], encoding="utf-8"))
    gate = {}
    for i, line in enumerate(open(sys.argv[2], encoding="utf-8")):
        if i == 0:
            continue
        c = line.rstrip("\n").split("\t")
        if len(c) >= 7:
            gate[(c[0], c[1])] = (c[3], c[4], c[5], c[6])

    lang = {}
    for tag, endir, cndir in (("ember", MOD, os.path.join(P, "1-Ember汉化插件")),
                              ("cruc", SYS, os.path.join(P, "2-Crucible汉化插件"))):
        en = flat(json.load(open(os.path.join(endir, "lang", "en.json"), encoding="utf-8")))
        cn = flat(json.load(open(os.path.join(cndir, "lang", "cn.json"), encoding="utf-8")))
        for k, v in en.items():
            lang.setdefault(v.strip(), []).append(f"{tag}:{k}={cn.get(k, '<缺>')}")

    print("| 表 | 英文键 | 现中文 | 判定 | 依据/建议 |")
    print("|---|---|---|---|---|")
    for t in TABLES_ORDER:
        d = keys[t]
        items = [(e["en"], e["cn"]) for e in d] if t == "PREFIXED" else list(d.items())
        for en, cn in items:
            v = VERDICT.get((t, en))
            if v:
                verdict, why = v
            else:
                verdict = "✅ 通过"
                g = gate.get((t, en))
                bits = []
                if en in lang:
                    bits.append("lang " + " / ".join(lang[en]))
                if g:
                    bits.append(f"英文闸 叶{g[0]}/命中{g[1]}/异写{g[2]}")
                why = "；".join(bits) or "库内无英文出现，属纯 UI 串"
            print(f"| {t} | `{en}` | {cn} | {verdict} | {why} |")
    # 缺失键单列
    for k, (verdict, why) in VERDICT.items():
        if verdict.startswith("⛔ 缺失"):
            print(f"| {k[0]} | `{k[1]}` | **（无）** | {verdict} | {why} |")


if __name__ == "__main__":
    main()
