#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""H1: 生成逐键判定表 findings/H1.md。判定是人工裁的，这里只负责把它排成表。只读输入。"""
import json
from pathlib import Path

OUT = Path(r"C:/Users/Taka/AppData/Local/Temp/claude/C--Users-Taka-Desktop-fvtt/e57b5596-8975-4155-bc4b-c3126ad4aad5/scratchpad/audit3/findings/H1.md")
KEYS = Path(r"C:/Users/Taka/AppData/Local/Temp/claude/C--Users-Taka-Desktop-fvtt/e57b5596-8975-4155-bc4b-c3126ad4aad5/scratchpad/h1_keys.json")

# en -> (判定, 依据/建议)
V = {}

OK = "✅ 对"
FIX = "❌ 改"
DEAD = "⚪ 不生效"
ADD = "➕ 缺"

# ---- ATTUNEMENTS ----
for en, why in [
    ("The Abyss", "name 字段「深渊 The Abyss」"),
    ("Akon", "name 字段「阿肯 Akon」"),
    ("Cora", "name 字段「科拉 Cora」"),
    ("Heart of Ember", "name 字段「余烬之心 Heart of Ember」"),
    ("Luxarum", "name 字段「卢克萨鲁姆 Luxarum」"),
    ("Mayis", "name 字段「玛伊斯 Mayis」"),
    ("Orbis", "name 字段「奥比斯 Orbis」"),
    ("Primordis", "name 字段「普里莫迪斯 Primordis」"),
    ("Ragen", "name 字段「拉根 Ragen」"),
    ("Signara", "name 字段「西格纳拉 Signara」"),
]:
    V[("ATTUNEMENTS", en)] = (OK, why + "；但整张表当前不生效，见 M-2/M-3")
V[("ATTUNEMENTS", "Aura")] = (
    FIX,
    "**改「奥拉」**。§8 2026-08-12b 已裁：Aura 作月亮专名＝奥拉，只有手势 Gesture: Aura 才是灵气。"
    "Cosmos 页 name 字段就是「奥拉 Aura」；英文闸 \\bAura\\b 下 奥拉 117 叶 : 灵气 58 叶（后者全是 aura 通名/手势）。"
    "glossary_ec 里那条「灵气 Aura」是裁决前的旧值，已过期")

# ---- LANGUAGES ----
LANG = {
    "Common": (OK, "crucible lang `LANGUAGES.Common`＝通用语。注：crucible 侧该 label 是 i18n 键、i18nInit 已本地化，此键实际用不上"),
    "Sign": (OK, "crucible lang `LANGUAGES.Sign`＝手语。同上，实际用不上"),
    "Arcden": (OK, "§8 2026-08-06 裁「阿克登语→奥克登语」；glossary_ec＝奥克登语。**但 compendium 还剩 6 叶写「阿克登语」，是裁决没落干净，见 X-1**"),
    "Cascal": (FIX, "**改「卡斯卡尔语」**。英文闸下库内只有 卡斯卡尔语 4 处 : 卡斯卡语 0"),
    "Forest Speech": (FIX, "**改「森林语」**。英文闸 森林语 2 : 森语 0"),
    "Hardac": (OK, "glossary_ec＝哈达克语；英文闸 2:0"),
    "Imperial": (OK, "英文闸 帝国语 24 叶命中，无异写"),
    "Solical": ("⚠ 存疑", "库内 0 处，无任何证据。音译构词合理，暂留"),
    "Mithia": ("⚠ 存疑", "库内 0 处（compendium 该行整格没译）。暂留"),
    "Luma": ("⚠ 存疑", "库内 0 处。**compendium 的语言表把 Luma 一行译成了「龙语」（龙语＝Draconic），那是 compendium 的错，见 X-3**；本表 卢玛语 正确，保留"),
    "Kaziric": (FIX, "**改「卡兹里克语」**。英文闸 卡兹里克语 2 : 卡兹瑞克语 0"),
    "Scripta": ("⚠ 存疑", "库内 0 处。「书文语」是意译，Scripta 是专名，倾向音译，但无证据不动"),
    "Wyrdic": (OK, "glossary_ec＝维尔迪克语；英文闸 7:0"),
    "Pathward": (FIX, "**改「径道语」**。glossary_ec＝径道语；英文闸 径道语 46 处/26 叶 : 歧路语 0"),
    "Scor": ("⚠ 存疑", "库内 0 处，暂留"),
    "Towyr": (FIX, "**改「托维尔语」**。glossary_ec＝托维尔语；英文闸 托维尔语 55 处/27 叶 : 托威尔语 0"),
    "Windclaw": ("⚠ 存疑", "库内 0 处，暂留"),
    "Abyssal": (OK, "英文闸 深渊语 14 叶（另 528 叶的 Abyssal 是形容词，不相干）"),
    "Draconic": (OK, "英文闸 龙语 2 叶"),
    "Druidic": (OK, "英文闸 德鲁伊语 3 叶"),
    "Lunix": ("⚠ 存疑", "库内 0 处。「月语」是意译专名，同 Harmos 一类风险，但无证据不动"),
    "Caligon": ("⚠ 存疑", "库内 0 处，暂留"),
    "Eonic": (OK, "英文闸 永世语 2 叶"),
    "Harmos": (FIX, "**改「哈莫斯语」**。把专名 Harmos 当 harmony 意译成「和谐语」是机翻痕迹；英文闸 哈莫斯语 18 处 : 和谐语 0"),
    "Thieves' Cant": ("⚠ 存疑", "英文侧库内 0 处（该词只在模块 JS 里）。盗贼黑话是通行译法，保留"),
}
for k, v in LANG.items():
    V[("LANGUAGES", k)] = v

# ---- KNOWLEDGE ----
for en in ["Alchemy", "Ancients", "Artifacts", "Arts", "Beasts", "Celestials", "Cosmology", "Crafts",
           "Crime", "Dragons", "Elementals", "Fey", "Fiends", "Forensics", "Gods", "Intrigue",
           "Legends", "Machines", "Monsters", "Plants", "Politics", "Rituals", "Seafaring",
           "Souls", "Subterranea", "Tracking", "Trade", "Undeath", "Warfare", "Weather"]:
    V[("KNOWLEDGE", en)] = (OK, "与 crucible `lang/cn.json` 的 `KNOWLEDGE.*` 逐字相同（含 08-12 已修的 Crafts＝工艺 / Seafaring＝航海）")
V[("KNOWLEDGE", "Outsiders")] = (DEAD, "ember `initialize()` 里 `delete crucible.CONFIG.knowledge.outsiders`，装了 ember 就没有这一档。留着无害")
V[("KNOWLEDGE", "Abyssals")] = (FIX, "**改「深渊裔」**。glossary_ec＝深渊裔；英文闸 \\bAbyssals\\b 8 叶全部作「深渊裔」，0 叶作「深渊生物」（库内 60 处「深渊生物」译的是 abyssal creature/entity 一类通名）")
V[("KNOWLEDGE", "Aedir")] = (FIX, "**改「艾迪尔」**。glossary_ec＝艾迪尔 Aedir；英文闸 艾迪尔 394 叶 : 埃迪尔 2 叶（980 处 : 2 处）")
V[("KNOWLEDGE", "Leviathans")] = (OK, "glossary_ec＝利维坦；英文闸 139 叶全中")
V[("KNOWLEDGE", "Shent")] = (OK, "glossary_ec＝申特 Shent；英文闸 448 叶")

# ---- MOODS ----
for en in ["Combat", "Exploration", "Ambience", "Travel", "Rest"]:
    V[("MOODS", en)] = (FIX, "**整张表作废**。`EmberSoundscape.MOODS` 只有 `calm`/`tension` 两档（ember.mjs:15606），"
                             "这五个键在运行时永远匹配不到；实际出现的是 `Music Mood: Calm` / `Music Mood: Tension`")

# ---- RESULTS ----
for en, cn in [("Success", "成功"), ("Failure", "失败"), ("Critical Success", "大成功"), ("Critical Failure", "严重失败")]:
    V[("RESULTS", en)] = (OK, f"与 crucible lang `ACTION.EFFECT_RESULT_TYPES.*`＝{cn} 一致。"
                              "实际不会被触发：`Result of X` 的 X 只可能是 dnd5e 侧的数字（`Result of 15+`）")

# ---- CALENDAR ----
for en in ["Seeding", "Blooming", "Steading", "Gleaning", "Withering", "Stilling"]:
    V[("CALENDAR_MONTHS", en)] = (OK, "与 `lang/cn.json` 的 `EMBER.CALENDAR.SEASONS.*` 逐字一致。"
                                      "但整张表**不生效也不需要生效**，见 M-6：日期串走的是季节名 i18n，不是月名")
V[("CALENDAR_MONTHS", "Steading")] = (OK, "lang `EMBER.CALENDAR.SEASONS.STEADING`＝耕耘，一致（§8 2026-08-12b）。"
                                          "**但 compendium 那一路没跟上：英文闸下 14 叶全作「安居」、0 叶作「耕耘」，见 X-2**")
for en in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
    V[("CALENDAR_DAYS", en)] = (OK, "周一…周日，与缩写表一致；宽度 2 字，短于英文，撑不坏 UI。表本身当前无 UI 读取（M-6）")
for en in ["Mon", "Tues", "Wed", "Thu", "Fri", "Sat", "Sun"]:
    V[("CALENDAR_DAY_ABBR", en)] = (OK, "键与 `EMBER_CALENDAR_CONFIG.days.values[].abbreviation` 逐字对得上（Mon/Tues/Wed/Thu/Fri/Sat/Sun）")

# ---- EXACT ----
E = {
    "Ancestry": (OK, "ember lang `EMBER.ACTOR_FLAGS.FIELDS.character.ancestry.label`＝血统；英文闸 48/48"),
    "Culture": (OK, "同上＝文化；英文闸 125/125"),
    "Path": (OK, "ember lang `…character.path.label`＝道途。英文闸的 85 条 en_only 是 path 作「路径/道路」的通名，不相干"),
    "-3 Banes": (OK, "crucible 侧 `enrichAdvantage` 拼的就是 `${n} Banes`；§8 Bane＝祸骰"),
    "-2 Banes": (OK, "同上"),
    "-1 Banes": (OK, "同上"),
    "+1 Boons": (OK, "`+${n} Boons`；§8 2026-08-12 Boon＝**恩惠骰**，与 `DICE.Boons` 一致"),
    "+2 Boons": (OK, "同上"),
    "+3 Boons": (OK, "同上"),
    "Critical Success": (OK, "crucible lang `ACTION.EFFECT_RESULT_TYPES.CriticalSuccess`＝大成功（§8 2026-08-12b：.mjs 服从 lang）。"
                             "compendium 仍有 20 叶写「重大成功」（29 处），属 compendium 侧残留，见 X-4"),
    "Critical Failure": (OK, "crucible lang＝严重失败。compendium 5 叶写「重大失败」（9 处），同 X-4"),
    "Event Completed": (FIX, "中文本身没问题，但**永远不生效**：它写在 `data-tooltip-text` 上，而 `translateNode` 的属性白名单里没有这个属性。见 M-1（阻断）"),
    "Event Not Completed": (FIX, "同 M-1（阻断）"),
    "Event Outcome Completed": (FIX, "同 M-1（阻断）；且用词要跟着改：**事件结果已完成**（见下一行）"),
    "Event Outcome Not Completed": (FIX, "同 M-1（阻断）；**事件结果未完成**"),
    "Gamemaster Information": (FIX, "**改「游戏主持人信息」**。按 h 标签逐个对齐，compendium 4 处全作「游戏主持人信息」；"
                                    "全库 Gamemaster＝游戏主持人 1302 处，没有单说「主持人」的先例"),
    "Ancestry Details": (OK, "构词与 lang 的 血统 一致；库内无反例"),
    "Culture Details": (OK, "同上"),
    "Notable Inhabitants": (OK, "库内无对照，构词无问题"),
    "Secret Lore": (OK, "compendium h 标签对齐 2/2 作「秘辛」。`scan_cross_channel` 报的 lang「隐秘背景」是另一个键"
                        "（`EMBER.DEVELOPMENT.FIELDS.development.secrets.label`，编辑态字段名），不是同一处文案"),
    "At a Glance": (OK, "compendium h 标签对齐 84/84 作「概览」；glossary_ec＝概览"),
    "Setting the Scene": (FIX, "**改「场景设定」**。compendium 同一 h 标签 14 处作「场景设定」、0 处作「场景铺陈」。"
                               "（ember lang 的 `…exposition.label` 是「设定场景」——那是编辑态字段名，可另行统一）"),
    "Event Details": (OK, "compendium 4 处一致"),
    "Journal Summary": (OK, "lang `EMBER.CODEX.JOURNAL`＝日志；compendium 4 处一致"),
    "Event Outcomes": (FIX, "**改「事件结果」**。§8 2026-08-09 已裁 `Event Outcome`→事件结果；"
                            "h 标签对齐 18/18 作「事件结果」，全库 事件结果 158 : 事件结局 2"),
    "Quest Details": (OK, "构词一致，无反例"),
    "Involved Locations": (OK, "库内无对照。与同表的 `Related Locations`＝相关地点 刻意区分，保留"),
    "Event Summary": (OK, "构词与 事件/摘要 一致"),
    "Biome Details": (FIX, "**改「生物群系详情」**。见下一行"),
    "Locations": (OK, "h 标签对齐 2/2＝地点"),
    "Location Details": (OK, "构词一致"),
    "Biomes": (FIX, "**改「生物群系」**。glossary_ec `Biome/Biomes`＝生物群系；ember lang 17 处、compendium 460 处全是生物群系；"
                    "英文闸 \\bBiomes?\\b 80 叶 **80 叶命中生物群系、0 叶命中生态域**。「生态域」全库 0 次，是本文件独有的孤例"),
    "Related Locations": (OK, "compendium 4 处一致"),
    "Events": (OK, "h 标签对齐 2/2＝事件"),
    "Quest Overview": (FIX, "**改「任务概览」**。name 字段 `Main Quest Overview`＝「主线任务概览 Main Quest Overview」；"
                            "全库 任务概览 14 : 任务总览 0"),
    "Standalone Event": (OK, "glossary_ec＝独立事件；英文闸 21/21"),
    "Quest Event": (OK, "glossary_ec＝任务事件；英文闸 10 叶命中"),
    "Begin Event": (OK, "ember lang `EMBER.EventActionBegin`＝开始事件，一致"),
    "Reset Event": (OK, "构词与 重置发现 一族一致"),
    "Complete Event": (OK, "ember lang 侧一致；英文闸 4 叶命中"),
    "Mark as Discovered": (OK, "与 lang `EMBER.CODEX.DISCOVERY_RECORD`＝记录发现 属不同英文，无冲突"),
    "Reset Discovery": (FIX, "**改「重置发现」**。同一个英文串 `Reset Discovery` 在 ember lang `EMBER.CODEX.DISCOVERY_RESET` 里是「重置发现」，"
                             "本文件多了「状态」二字。§8：.mjs 服从 lang。`scan_cross_channel` B 段也报了这条（MJS_LANG_DRIFT）"),
    "Award Attunements": (OK, "§8 Attunement＝同调；英文闸 4/4"),
    "Attunements Awarded": (OK, "同上"),
    "No Awarded Attunements": (OK, "同上"),
    "Award Milestone": (OK, "Milestone＝里程碑（glossary_ec，库内 406 处）"),
    "Milestone Awarded": (OK, "同上"),
    "Granted attunement points require awarding.": (OK, "按钮浮窗，走 `aria-label`，属性白名单里有；行文通顺"),
    "All granted attunement points have been awarded.": (OK, "同上"),
    "No attunement points have been awarded.": (OK, "同上"),
    "Award a milestone point for the completion of this event.": (OK, "同上"),
    "The milestone point for this event has already been awarded.": (OK, "同上"),
    "Add to Party?": (DEAD, "中文没问题，但**永远不生效**：这是原生 `DialogV2` 的窗口标题，根元素 class 只有 `dialog`、类名是 `DialogV2`，"
                            "被 `patchRenderedApplications` 的 ember 闸挡掉。见 M-2（阻断）"),
    "Re-combine Caravans?": (DEAD, "同 M-2。Caravan＝商队（库内 848 处），用词本身对"),
    "Initiate Event": (DEAD, "同 M-2"),
    "Select Outcome": (FIX, "同 M-2 不生效；**且用词要改「选择结果」**——本文件内部 Outcome 一会儿「结局」一会儿「结果」"
                            "（`Result of`→结果：），而全库 事件结果 158 : 事件结局 2"),
    "Delete Saved Composition?": (FIX, "同 M-2 不生效；**且「编成」要改「构图」**。这是远景（Vista）构图，"
                                       "ember lang `EMBER.CONTROLS.VistaComposition`＝更改构图、glossary_ec＝构图、"
                                       "英文闸 \\bComposition\\b 8 叶全部作「构图」、0 叶作「编成」"),
    "Transition to Pathways?": (FIX, "同 M-2 不生效；**且「歧路」要改「通路」**。Cosmos 页 name＝「通路 Pathways」，"
                                     "glossary_ec＝通路 Pathways，英文闸 \\bPathways\\b 通路 408 叶 : 歧路 1 叶"),
    "Ring Alarm Bell?": (DEAD, "同 M-2。警钟用词对"),
    "Modify Flow Control Valve?": (DEAD, "同 M-2。控制阀用词对"),
    "Mine Cart Destination": (DEAD, "同 M-2。矿车＝库内 221 处"),
    "Install Junction Wheel": (DEAD, "同 M-2。道岔＝库内 28 处"),
    "Elevator Controls": (DEAD, "同 M-2。升降机＝库内 451 处"),
    "Elevator Destination": (DEAD, "同 M-2"),
    "Steam Cleansing Cutoff": (DEAD, "同 M-2。「切断」略生硬（原文是切断阀/断流装置），但不生效、且无库内对照，不动"),
    "Unspent Ability Points": (DEAD, "同 M-2。crucible lang 属性点，一致"),
    "Apply Soulbound Progression": (DEAD, "同 M-2。魂缚＝库内 298 处；Progression＝进程与 `EMBER.ATTUNEMENT.ProgressionTitle` 一致"),
}
for k, v in E.items():
    V[("EXACT", k)] = v


def main():
    raw = KEYS.read_text(encoding="utf-8")
    data = json.loads(raw[:raw.rindex("}") + 1])
    lines = []
    n_ok = n_fix = n_dead = n_doubt = 0
    for table in ["ATTUNEMENTS", "LANGUAGES", "KNOWLEDGE", "MOODS", "EXACT", "RESULTS",
                  "CALENDAR_MONTHS", "CALENDAR_DAYS", "CALENDAR_DAY_ABBR"]:
        d = data[table]
        lines.append(f"\n### `{table}`（{len(d)} 键）\n")
        lines.append("| 键（英文） | 现中文 | 判定 | 依据 / 建议 |")
        lines.append("|---|---|---|---|")
        for en, cn in d.items():
            verdict, why = V.get((table, en), ("⚠ 未判", ""))
            if verdict == OK:
                n_ok += 1
            elif verdict == FIX:
                n_fix += 1
            elif verdict == DEAD:
                n_dead += 1
            else:
                n_doubt += 1
            e = en.replace("|", "\\|")
            lines.append(f"| `{e}` | {cn} | {verdict} | {why} |")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"OK={n_ok} FIX={n_fix} DEAD={n_dead} DOUBT={n_doubt} total={n_ok+n_fix+n_dead+n_doubt}")


if __name__ == "__main__":
    main()
