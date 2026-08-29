# -*- coding: utf-8 -*-
"""Build glossary_alien.json v0 + provenance + disputes + pending.

Every entry below carries its evidence inline.  Counts written as "en-gated N"
were produced by 4-常用脚本/tm/term_gate.py in this session and are re-derivable
with the command recorded in _meta.gate_commands.  Bare Chinese counts are never
used as a reason to adopt a term.
"""
import io, json, os, sys, datetime

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = r"C:\Users\Taka\Desktop\fvtt\Alien-RPG Translation Project\7-其他内容\glossary"

CNJ = "C:/Users/Taka/AppData/Local/FoundryVTT/Data/systems/alienrpg/lang/cn.json"
SYS = "C:/Users/Taka/AppData/Local/FoundryVTT/Data/systems/alienrpg"
MOD = "C:/Users/Taka/AppData/Local/FoundryVTT/Data/modules"

# ---------------------------------------------------------------------------
# T = tier, C = category
# entry: (en, cn, tier, category, why, [candidates], aliases, dispute_id)
# candidate: (zh, source, count, evidence)
# ---------------------------------------------------------------------------
E = []


def add(en, cn, tier, cat, why, cands=(), aliases=(), dispute=None, note=None):
    E.append(dict(en=en, cn=cn, tier=tier, category=cat, why=why,
                  candidates=[dict(zh=c[0], source=c[1], count=c[2], evidence=c[3])
                              for c in cands],
                  aliases=list(aliases), dispute=dispute, note=note))


A = "cn.json stratum A (TI-130, human, 2020-11..2021-01)"
B = "cn.json stratum B (maintainer bulk MT pass, 2022-2026)"
W = "zh.wikipedia"
BD = "百度百科"
MG = "萌娘百科"
S = "CnSCG subtitle corpus (ONE vote — single fansub lineage)"
D = "derived from stratum-A convention"
STD = "standard modern Chinese usage"

# =========================================================== YZE MECHANICS ===
add("Stress", "压力", "T-PLAIN", "yze-mechanics",
    "Stratum A and B agree; en-gated 10 hits, no competitor.",
    [("压力", A + " + " + B, "en-gated 10 (regex \\bStress\\b, 22 en rows)",
      "cn.json:419 \"Stress\": \"压力\"; reinforced through the whole Panic7-15 table.")])
add("Stress Level", "压力水平", "T-PLAIN", "yze-mechanics",
    "en-gated 7 vs 2. Inside stratum A alone it is 5 (Panic9/10/11/12/13) vs 1 (Panic7).",
    [("压力水平", A, "en-gated 7 (5 stratum A + 2 stratum B)",
      "cn.json:301 Panic9, :293 Panic10, :294 Panic11, :295 Panic12, :296 Panic13; also TAH.dehydrated, TAH.starving."),
     ("压力等级", A + " + " + B, "en-gated 2 (1 stratum A + 1 stratum B)",
      "cn.json:299 Panic7 \"...你的压力等级...上升1\"; also TAH.freezing.")],
    aliases=["压力等级"], dispute="D3-stress-level")
add("Stress Die", "压力骰", "T-PLAIN", "yze-mechanics",
    "Compound of the settled Stress=压力 plus 骰; no standalone key exists to inherit.",
    [("压力骰", D, "no standalone en source",
      "cn.json ALIENRPG.HOWMANYSTRESS \"How Many Stress Dice?\" -> \"有多少压力骰子?\"; ALIENRPG.rollStress -> \"进行压力检定\".")])
add("Stress Dice", "压力骰", "T-PLAIN", "yze-mechanics", "Plural of Stress Die; Chinese has no plural.", [])
add("Base Die", "基础骰", "T-PLAIN", "yze-mechanics",
    "cn.json's own rendering is a hard MT failure; 基础 is nonetheless the attested rendering of Base.",
    [("基础骰", D, "en-gated 0 for the compound",
      "cn.json ALIENRPG.Base -> \"基础\", ALIENRPG.BaseMod -> \"基础修正\". BUT cn.json:205 HOWMANYDICE \"How Many Base Die?\" -> \"有多少基地死亡？\" (base->基地, die->死亡) — must be replaced.")])
add("Base Dice", "基础骰", "T-PLAIN", "yze-mechanics", "Plural of Base Die.", [])
add("Push", "孤注一掷", "T-PLAIN", "yze-mechanics",
    "Strongest single term in the file: en-gated 2/3, identical in zh-tw, no competitor.",
    [("孤注一掷", A, "en-gated 2 (regex \\bPush\\b, 3 en rows)",
      "cn.json:327 \"Push\": \"孤注一掷\"; ALIENRPG.followingPush -> \"选择孤注一掷后\".")])
add("Panic", "恐慌", "T-PLAIN", "yze-mechanics",
    "恐慌 is the noun everywhere in the UI keys; the competing 混乱 appears only in the phrase Panic Roll.",
    [("恐慌", A + " + " + B, "en-gated 1 direct + 11 sibling keys (Panic/Panicked/panic level)",
      "cn.json:303 \"Panicked\": \"恐慌\"; :302 \"PanicCondition\": \"恐慌状态\"; ALIENRPG.MorePanic -> \"更加恐慌\".")])
add("Panic Roll", "恐慌检定", "T-PLAIN", "yze-mechanics",
    "PROVISIONAL. Regularised on Panic=恐慌 + the stratum-A roll word 检定. The only attested "
    "rendering of the phrase is 混乱检定 (en-gated 3/7) but 混乱 means 'chaos', collides with nothing "
    "else in the file, and contradicts 恐慌状态/恐慌 in the same file. See dispute D1.",
    [("混乱检定", A, "en-gated 3 (regex '[Pp]anic [Rr]oll', 7 en rows)",
      "cn.json:295 Panic12, :296 Panic13, :297 Panic14 — all \"必须立刻进行混乱检定\"."),
     ("恐慌检定", D, "en-gated 0",
      "Coined here from 恐慌 (cn.json:303) + 检定 (the stratum-A roll word, en-gated 10 vs 滚动 9 which is all MT).")],
    aliases=["混乱检定"], dispute="D1-panic-roll")
add("Panic Condition", "恐慌状态", "T-PLAIN", "yze-mechanics", "Direct, unambiguous.",
    [("恐慌状态", A, "en-gated 1", "cn.json:302 \"PanicCondition\": \"恐慌状态\".")])
add("Panicked", "恐慌", "T-PLAIN", "yze-mechanics", "Direct.",
    [("恐慌", A, "en-gated 1", "cn.json:303 \"Panicked\": \"恐慌\".")])
add("Retreat Roll", "撤退检定", "T-PLAIN", "yze-mechanics",
    "en-gated 2/2, stratum A, page reference localised too.",
    [("撤退检定", A, "en-gated 2 (regex 'retreat roll')",
      "cn.json:294 Panic11 and :296 Panic13 \"你可以进行撤退检定（见P93）\".")])
add("Roll", "检定", "T-PLAIN", "yze-mechanics",
    "检定 is the project's roll word. en-gated 10 vs 滚动 9 / 掷骰 3 / 翻滚 2 — but every 滚动/翻滚 hit "
    "is stratum-B MT while every 检定 hit is human.",
    [("检定", A, "en-gated 10 (regex \\broll, 51 en rows)",
      "撤退检定 (Panic11/13), 混乱检定 (Panic12/13/14), 技能检定 (Panic8), 共情检定 (PermanantTrauma), 耐力检定."),
     ("滚动", B, "en-gated 9", "ALIENRPG.RollPanic -> \"点击滚动恐慌\"; TAH.freezing \"死亡滚动\" — all MT, reject."),
     ("掷骰", B, "en-gated 3", "TAH.starving \"进行掷骰\" — MT."),
     ("翻滚", B, "en-gated 2", "TAH.dehydrated \"死亡翻滚\" — MT, reads as a physical roll/tumble.")],
    note="cn.json ALIENRPG.Roll itself is \"投\" — a third rendering, single character, reject.")
add("Talent", "天赋", "T-BILINGUAL", "yze-mechanics",
    "en-gated 3, identical in zh-tw. Talents are compendium Item names, hence T-BILINGUAL on the name field.",
    [("天赋", A, "en-gated 3 (regex \\bTalent)",
      "cn.json:453 \"Talents\": \"天赋\"; ALIENRPG.GeneralTalent -> \"通用天赋\".")],
    note="cn.json ALIENRPG.Talent-Crit -> \"天赋/暴击\" leaks a D&D 暴击; Critical Injury is 重伤 here.")
add("Skill", "技能", "T-PLAIN", "yze-mechanics", "en-gated 3, no competitor.",
    [("技能", A, "en-gated 3", "cn.json:407 \"Skills\": \"技能\"; ALIENRPG.NoSkill -> \"没有技能\".")])
add("Stunts", "炫技", "T-PLAIN", "yze-mechanics",
    "Attested; kept rather than coined. 特技 is the more usual Chinese TRPG word for a stunt and reads "
    "better, but is unsourced — flagged for review, not adopted.",
    [("炫技", A, "en-gated 1", "cn.json ALIENRPG.Stunts -> \"炫技\"."),
     ("特技", STD, "unattested here", "Common Chinese TRPG rendering of 'stunt'. Not adopted without a source.")],
    aliases=["特技"])
add("Attribute", "属性", "T-PLAIN", "attribute", "en-gated 2/2.",
    [("属性", A, "en-gated 2", "cn.json:65 \"Attributes\": \"属性\"; ALIENRPG.NoAttribute -> \"无属性\".")])
add("Strength", "力量", "T-PLAIN", "attribute", "Stratum A attribute set, all four consistent with zh-tw.",
    [("力量", A, "en-gated 1", "cn.json:20 \"AbilityStr\": \"力量\".")],
    note="cn.json ALIENRPG.TAH.encumbered renders the same word 强度 mid-sentence (MT); and ALIENRPG.Pwr \"Pwr\" -> \"力量\" collides — Power is 动力.")
add("Agility", "敏捷", "T-PLAIN", "attribute", "Stratum A.",
    [("敏捷", A, "en-gated 1", "cn.json:18 \"AbilityAgl\": \"敏捷\"; also Panic8 \"所有使用敏捷...的技能检定\".")])
add("Wits", "机智", "T-PLAIN", "attribute", "Stratum A.",
    [("机智", A, "en-gated 1", "cn.json:21 \"AbilityWit\": \"机智\".")])
add("Empathy", "共情", "T-PLAIN", "attribute", "Stratum A, attested twice (UI key + prose).",
    [("共情", A, "en-gated 1", "cn.json:19 \"AbilityEmp\": \"共情\"; ALIENRPG.PermanantTrauma -> \"...进行一次共情检定。\"")])
add("Career", "职业", "T-PLAIN", "yze-mechanics", "Stratum A, identical in zh-tw.",
    [("职业", A, "en-gated 1", "cn.json:76 \"Career\": \"职业\".")])
add("Agenda", "目标", "T-PLAIN", "yze-mechanics",
    "en-gated 2. Kept, but it COLLIDES: cn.json renders 'Ad Hoc' as 目标 too. Ad Hoc is re-pointed to 临时 below.",
    [("目标", A, "en-gated 2 (regex \\bAgenda)",
      "cn.json:50 \"AgendaStory\" \"Agenda/Story Cards\" -> \"目标\" (the /Story Cards half dropped); :312 \"PersonalAgenda\": \"个人目标\"."),
     ("议程", STD, "unattested", "Literal but wrong register for a character drive.")],
    note="TYPES.Item.agenda is left as the English \"Agenda\" in cn.json — needs filling.")
add("Personal Agenda", "个人目标", "T-PLAIN", "yze-mechanics", "Stratum A, identical in zh-tw.",
    [("个人目标", A, "en-gated 1", "cn.json:312 \"PersonalAgenda\": \"个人目标\".")])
add("Ad Hoc", "临时", "T-PLAIN", "yze-mechanics",
    "RE-POINTED. cn.json:… ALIENRPG.AdHoc is \"目标\", which is the Agenda word; two unrelated game terms "
    "cannot share one string in a dropdown.",
    [("目标", B, "en-gated 1 — REJECTED", "cn.json ALIENRPG.AdHoc \"Ad Hoc\" -> \"目标\"."),
     ("临时", STD, "coined", "Standard rendering of ad hoc; removes the collision with Agenda.")])
add("Signature Item", "标志物品", "T-BILINGUAL", "yze-mechanics",
    "en-gated 1, identical in zh-tw. Reads slightly stiff; 象征物 would be smoother but is unsourced.",
    [("标志物品", A, "en-gated 1", "cn.json:390 \"SignatureItem\": \"标志物品\".")],
    aliases=["标志性物品"])
add("Signature Weapon", "标志武器", "T-BILINGUAL", "yze-mechanics",
    "Coined on the settled Signature Item=标志物品. Absent from en.json AND cn.json — nothing to inherit.",
    [("标志武器", D, "en-gated 0 (key does not exist)",
      "Not present in en.json or cn.json. NOTE cn.json ALIENRPG.SIGNATURE -> \"签名\" is a different term: the spacecraft SIGNATURE stat, mistranslated as an autograph; it should be 信号特征.")])
add("Buddy", "哥们", "T-PLAIN", "yze-mechanics", "Stratum A; colloquial but deliberate, matches zh-tw.",
    [("哥们", A, "en-gated 1", "cn.json:337 \"relOne\": \"哥们\".")])
add("Rival", "对头", "T-PLAIN", "yze-mechanics", "Stratum A, matches zh-tw.",
    [("对头", A, "en-gated 1", "cn.json:338 \"relTwo\": \"对头\".")])
add("Health", "生命", "T-PLAIN", "yze-mechanics",
    "PROVISIONAL. Three renderings coexist; 生命 is the only one on the canonical UI key, and both "
    "competitors occur only in stratum-B MT tooltips. See dispute D4.",
    [("生命", A, "en-gated 3 (1 canonical key + 2 MT tooltips)", "cn.json:193 \"Health\": \"生命\"."),
     ("健康", B, "en-gated 4 (all stratum B)", "ALIENRPG.healthDamage \" HEALTH Damage\" -> \" 健康损害\"; TAH.dehydrated / TAH.freezing / TAH.starving \"您无法恢复健康\"."),
     ("生命值", B, "en-gated 2 (all stratum B)", "TAH.freezing \"可以恢复生命值\"; TAH.radiation \"你就无法恢复生命值\".")],
    aliases=["生命值", "健康"], dispute="D4-health")
add("Radiation", "辐射", "T-PLAIN", "yze-mechanics", "en-gated 5, no competitor.",
    [("辐射", A + " + " + B, "en-gated 5 (regex \\bRadiation)",
      "cn.json:330 \"Radiation\": \"辐射\"; PermanentRadiationAdded -> \"添加永久辐射\".")])
add("Radiation Point", "辐射点", "T-PLAIN", "yze-mechanics",
    "Compound of the settled 辐射. cn.json's own MT gives 拉德 (the SI 'rad' unit) and leaves bare 'Rad' — both reject.",
    [("辐射点", D, "en-gated 0 for the compound",
      "cn.json ALIENRPG.TAH.radiation: \"每班可恢复 1 拉德\" and \"每次你要治愈 Rad 时\" — 拉德 is the physics unit rad, not the game's Radiation Point.")])
add("Rad", "辐射点", "T-PLAIN", "yze-mechanics", "Same game quantity as Radiation Point; not the SI unit.",
    [("拉德", B, "en-gated 2 — REJECTED", "TAH.radiation. 拉德 is the SI absorbed-dose unit.")])
add("Air Supply", "氧气供应", "T-PLAIN", "yze-mechanics", "en-gated 1, matches zh-tw.",
    [("氧气供应", A, "en-gated 1", "cn.json:53 \"AirSupply\": \"氧气供应\".")])
add("Air", "氧气", "T-PLAIN", "yze-mechanics", "Consumable track; the system means breathable oxygen, not 空气.",
    [("氧气", A, "en-gated 1", "cn.json:52 \"Air\": \"氧气\".")])
add("Food", "食物", "T-PLAIN", "yze-mechanics", "Consumable track.", [("食物", A, "en-gated 1", "cn.json ALIENRPG.Food.")])
add("Water", "水", "T-PLAIN", "yze-mechanics", "Consumable track.", [("水", A, "en-gated 1", "cn.json ALIENRPG.Water.")])
add("Power", "动力", "T-PLAIN", "yze-mechanics", "Consumable track.",
    [("动力", A, "en-gated 1", "cn.json:322 \"Power\": \"动力\".")],
    note="cn.json ALIENRPG.Pwr \"Pwr\" -> \"力量\" is a defect: that abbreviation is Power, not Strength.")
add("Consumables", "消耗品", "T-PLAIN", "yze-mechanics", "en-gated 1.",
    [("消耗品", A, "en-gated 1", "cn.json:101 \"Consumables\": \"消耗品\".")])
add("Supply", "补给", "T-PLAIN", "yze-mechanics",
    "The bare term. cn.json bundles the die into the key value (补给骰); split here so 'Supply Roll' can be regular.",
    [("补给", A, "en-gated 1 + 2 sibling keys",
      "ALIENRPG.supplyDecreases -> \"你的补给下降了\"; ALIENRPG.NoSupplys -> \"你的补给已耗尽。\""),
     ("补给骰", A, "en-gated 1", "cn.json:428 \"Supply\": \"补给骰\" — bundles 'die' into the noun.")],
    aliases=["补给骰"])
add("Supply Roll", "补给检定", "T-PLAIN", "yze-mechanics",
    "Regular on Supply=补给 + 检定.",
    [("补给检定", D, "en-gated 0 for the phrase", "Derived; cn.json has no 'Supply Roll' key.")])
add("Encumbrance", "负重", "T-PLAIN", "yze-mechanics", "en-gated 1, also used for the Capacity bar.",
    [("负重", A, "en-gated 1", "cn.json:156 \"Encumbrance\": \"负重\"; ALIENRPG.CapacityBarLabel \"Capacity\" -> \"负重\".")])
add("Encumbered", "受阻", "T-PLAIN", "yze-condition", "Condition form, attested.",
    [("受阻", A, "en-gated 1", "cn.json ALIENRPG.Encumbered -> \"受阻\"; the MT tooltip says 过度负担 instead.")])
add("Critical Injury", "重伤", "T-PLAIN", "yze-mechanics",
    "en-gated 1 on the canonical key; the competitors are D&D leakage.",
    [("重伤", A, "en-gated 1", "cn.json:113 \"CriticalInjuries\": \"重伤\"."),
     ("严重伤害", B, "en-gated 0", "ALIENRPG.NoCharCrit -> \"没有严重伤害表\"."),
     ("暴击", B, "REJECTED", "ALIENRPG.Talent-Crit -> \"天赋/暴击\"; ALIENRPG.RollCrit -> \"重击\". Both are D&D crit words, wrong game.")],
    note="TYPES.Item.critical-injury and ALIENRPG.CriticalInjuryfor are left English in cn.json.")
add("Death Roll", "死亡检定", "T-PLAIN", "yze-mechanics",
    "No usable rendering exists: cn.json gives FOUR, all stratum-B MT, all en-gated. Regularised on 检定.",
    [("死亡检定", D, "en-gated 0", "Coined on the stratum-A roll word 检定."),
     ("死亡掷骰", B, "en-gated 2", "TAH.dehydrated, TAH.radiation."),
     ("死亡翻滚", B, "en-gated 1", "TAH.dehydrated (same string as 死亡掷骰 — inconsistent within one value)."),
     ("死亡滚动", B, "en-gated 1", "TAH.freezing."),
     ("死亡卷", B, "en-gated 1", "TAH.starving — 卷 is nonsense here.")])
add("Broken", "濒死", "T-PLAIN", "yze-condition",
    "The one human rendering wins over three MT ones: 破碎 is en-gated 3 but every hit is a stratum-B tooltip.",
    [("濒死", A, "en-gated 1 (stratum A)", "cn.json:297 Panic14 \"直到目标濒死你才会停下\"."),
     ("破碎", B, "en-gated 3 (all stratum B)", "TAH.dehydrated, TAH.radiation, TAH.starving — 破碎 reads as 'shattered', wrong for a person."),
     ("受伤", B, "en-gated 1", "TAH.freezing — too weak, 受伤 is just 'injured'.")],
    aliases=["破碎"])
add("Fast Action", "快速动作", "T-PLAIN", "yze-mechanics",
    "Symmetric with the attested 慢速动作; the UI key itself is untranslated in cn.json.",
    [("快速动作", B, "en-gated 1", "cn.json ALIENRPG.TAH.overwatch opens \"作为一个快速动作，你可以...\". ALIENRPG.FastAction is left as the English \"Fast Action\".")])
add("Slow Action", "慢速动作", "T-PLAIN", "yze-mechanics",
    "Attested in stratum-A prose twice; the UI key ALIENRPG.SlowAction is left English.",
    [("慢速动作", A, "en-gated 2", "cn.json:293 Panic10 \"你失去下一个慢速动作\"; :295 Panic12 \"并失去下一个慢速动作\".")])
add("Round", "回合", "T-PLAIN", "yze-mechanics",
    "回合 on the UI key; stratum A's prose uses the bare 轮. Alien RPG's Round/Turn are inverted vs D&D, "
    "so the pair Round=回合 / Turn=轮次 is fixed deliberately.",
    [("回合", A, "en-gated 1", "cn.json:282 \"OneRound\": \"一回合\"."),
     ("轮", A, "in prose", "Panic10 \"呆滞一轮\", Panic11 \"一轮之后你可以如常行动\", Panic12 \"你尖叫一轮\".")],
    aliases=["轮"])
add("Turn", "轮次", "T-PLAIN", "yze-mechanics",
    "cn.json's own value is nonsense; coined to keep Round/Turn distinct.",
    [("轮次", D, "en-gated 0", "cn.json:285 \"OneTurn\": \"一斡\" — 斡 is an MT artifact, not a word for this.")])
add("Shift", "轮班", "T-PLAIN", "yze-mechanics",
    "The time unit. cn.json renders the same unit four ways; 轮班 is the one on the OneShift key.",
    [("轮班", B, "en-gated 1", "cn.json:284 \"OneShift\": \"一轮班\"."),
     ("转变", B, "en-gated 1 — REJECTED", "TAH.dehydrated \"每一次转变\"."),
     ("班次", B, "en-gated 1 — REJECTED", "TAH.exhausted \"睡一个班次\"."),
     ("每班", B, "en-gated 1", "TAH.radiation \"每班可恢复 1 拉德\".")],
    note="TRAP: the bare literal \"Shift\" in the Critical Injuries table's heal-time column is compared "
         "as ENGLISH at module/documents/actor.mjs:1888 (testArray[9] === \"Shift\"). That cell must stay "
         "English in the table content even though the lang key is Chinese. See _meta.lockstep_literals.")
add("Initiative", "先攻", "T-PLAIN", "yze-mechanics",
    "cn.json has no usable rendering — its MT read 'initiative' as a business initiative.",
    [("先攻", STD, "en-gated 0", "cn.json ALIENRPG.CombatantReroll \"ReRoll Initiative\" -> \"重新启动计划\"; TAH.encumbered ends \"...倡议</a>）\". Both wrong. 先攻 is the standard Chinese TRPG word.")])
add("Cover", "掩护", "T-PLAIN", "yze-mechanics", "Stratum A panic prose.",
    [("掩护", A, "en-gated 1", "cn.json:294 Panic11 \"SEEK COVER\" -> \"<b>寻找掩护：</b>\".")])
add("Full Cover", "全掩护", "T-PLAIN", "yze-mechanics", "Compound on the settled 掩护; key absent from cn.json.",
    [("全掩护", D, "en-gated 0", "en.json ALIENRPG.fullcover = \"Full Cover (-3)\"; cn.json has no such key.")])
add("Partial Cover", "半掩护", "T-PLAIN", "yze-mechanics", "Compound on the settled 掩护; key absent from cn.json.",
    [("半掩护", D, "en-gated 0", "en.json ALIENRPG.partialcover = \"Partial Cover (-2)\"; absent from cn.json.")])
add("No Cover", "无掩护", "T-PLAIN", "yze-mechanics", "Compound on the settled 掩护; key absent from cn.json.",
    [("无掩护", D, "en-gated 0", "en.json ALIENRPG.nocover = \"No Cover (0)\"; absent from cn.json.")])
add("Target Cover", "目标掩护", "T-PLAIN", "yze-mechanics", "Compound; key absent from cn.json.",
    [("目标掩护", D, "en-gated 0", "en.json ALIENRPG.targetCover = \"Target Cover\"; absent from cn.json.")])
add("Overwatch", "警戒", "T-PLAIN", "yze-condition",
    "cn.json's value is a hard error — it is the Chinese title of the Blizzard video game.",
    [("警戒", STD, "en-gated 0", "cn.json:290 \"Overwatch\": \"守望先锋\" — the Blizzard game. MUST be replaced."),
     ("监视位置", B, "en-gated 1", "TAH.overwatch uses 监视位置 and 守望位置 interchangeably in one value.")])
add("Campaign Mode", "战役模式", "T-PLAIN", "yze-mechanics",
    "cn.json read 'campaign' as a marketing campaign. 战役 is the standard Chinese TRPG rendering.",
    [("战役模式", STD, "en-gated 0", "cn.json:73 \"campaign\" \"CAMPAIGN\" -> \"活动\" — wrong sense.")])
add("Cinematic Mode", "剧本模式", "T-PLAIN", "yze-mechanics",
    "cn.json gives 电影 (the medium); zh-tw leaves the Latin 'CINEMA'. Neither is a mode name. "
    "Cinematic play in Alien RPG is scripted one-shot play, hence 剧本.",
    [("剧本模式", STD, "en-gated 0", "cn.json:79 \"cinematic\" \"CINEMATIC\" -> \"电影\"; zh-tw.json -> \"CINEMA\".")])
add("Story Points", "故事点", "T-PLAIN", "yze-mechanics", "en-gated 1, identical in zh-tw.",
    [("故事点", A, "en-gated 1", "cn.json:418 \"StoryPoints\": \"故事点\".")])
add("Game Mother", "游戏之母", "T-PLAIN", "yze-mechanics",
    "The Alien RPG name for the GM. Nothing to inherit — cn.json uses the bare Latin GM in prose.",
    [("游戏之母", D, "en-gated 0", "cn.json:301 Panic9 \"GM决定丢下了什么\" keeps GM in Latin.")])
add("GM Only", "仅限GM", "T-PLAIN", "yze-mechanics",
    "cn.json expanded GM to General Motors. Hard error.",
    [("仅限GM", D, "en-gated 0", "cn.json:186 \"GMONLY\": \"仅限通用汽车\" (= General Motors). MUST be replaced.")])
add("Select Firer", "选择射击者", "T-PLAIN", "yze-mechanics",
    "cn.json read 'firer' as firefighter.",
    [("选择射击者", D, "en-gated 0", "cn.json:370 \"SelectFirer\": \"选择消防员\" (= firefighter). MUST be replaced.")])
add("Armor Rating", "护甲等级", "T-PLAIN", "yze-mechanics", "en-gated 1.",
    [("护甲等级", A, "en-gated 1", "cn.json:61 \"ArmorRating\": \"护甲等级\"; ALIENRPG.Armor -> \"护甲\".")])
add("Armor", "护甲", "T-PLAIN", "yze-mechanics", "Attested.",
    [("护甲", A, "en-gated 1", "cn.json ALIENRPG.Armor -> \"护甲\".")],
    note="cn.json ALIENRPG.SHIP-ARMOR is \"盔甲\" instead — inconsistent, should be 护甲.")
add("Armor Piercing", "破甲", "T-PLAIN", "yze-mechanics",
    "As a damage keyword it is an effect, not a cartridge; cn.json's 穿甲弹 is the ammunition noun.",
    [("破甲", STD, "en-gated 0", "cn.json:60 \"ArmorPiercing\": \"穿甲弹\" (= AP ammunition, a noun).")],
    aliases=["穿甲弹"])
add("Resolve", "决心", "T-PLAIN", "yze-mechanics",
    "Alien Evolved Edition. No Chinese exists anywhere to inherit.",
    [("决心", STD, "en-gated 0 (key absent from cn.json)", "en.json ALIENRPG.Resolve = \"Resolve\"; cn.json has no such key.")])
add("Resolve Modifier", "决心调整值", "T-PLAIN", "yze-mechanics", "Compound on Resolve=决心.",
    [("决心调整值", D, "en-gated 0", "en.json ALIENRPG.ResolveMod = \"Resolve Modifier\"; absent from cn.json.")])
add("Stress Response", "压力反应", "T-PLAIN", "yze-mechanics", "Evolved. Compound on the settled Stress=压力.",
    [("压力反应", D, "en-gated 0", "en.json ALIENRPG.StressResponses = \"Stress Responses\"; absent from cn.json.")])
add("Panic Response", "恐慌反应", "T-PLAIN", "yze-mechanics", "Evolved. Compound on Panic=恐慌.",
    [("恐慌反应", D, "en-gated 0", "en.json ALIENRPG.PanicResponses = \"Panic Responses\"; absent from cn.json.")])
add("Response", "反应", "T-PLAIN", "yze-mechanics", "Evolved.",
    [("反应", D, "en-gated 0", "en.json ALIENRPG.Response = \"Response\"; absent from cn.json.")])

# ================================================ EVOLVED CONDITIONS (20) ===
# module/helpers/config.mjs ALIENRPG.conditions — verified 20 entries this session.
COND = [
    ("Jumpy", "惊跳", "stress", 1, "惊跳 keeps the startle sense; the condition adds +1 panic."),
    ("Tunnel Vision", "视野狭窄", "stress", 2, "Literal and standard; the condition is -2 Comtech/Observation/Survival."),
    ("Aggravated", "恼怒", "stress", 3, "恼怒 for the irritated-not-enraged sense; 发狂 is reserved for Frenzy."),
    ("Shakes", "颤抖", "stress", 4, "ANCHORED: stratum A renders the classic TREMBLE entry as 颤抖 (cn.json:300 Panic8)."),
    ("Frantic", "慌乱", "stress", 5, "慌乱 sits between 恐慌 (Panic) and 惊跳 (Jumpy)."),
    ("Deflated", "泄气", "stress", 6, "泄气 is the standard Chinese for morale collapse."),
    ("Mess Up", "搞砸", "stress", 7, "Colloquial register, matching the English."),
    ("Keeping Guard", "保持警戒", "stress", 0, "Built on Overwatch=警戒; NOT the same as Panic1 KEEPING IT TOGETHER=保持镇定."),
    ("Spooked", "受惊", "panic", 1, "受惊 is weaker than 恐慌, matching the table's first step."),
    ("Noisy", "喧闹", "panic", 2, "The condition makes the character audible."),
    ("Twitchy", "抽搐不安", "panic", 3, "Stratum A renders NERVOUS TWITCH as 精神紧绷 (cn.json:299 Panic7); the Evolved adjective needs its own form."),
    ("Lose Item", "失去物品", "panic", 4, "Stratum A renders DROP ITEM as 扔下物品 (cn.json:301 Panic9); Evolved says lose, not drop."),
    ("Paranoid", "疑神疑鬼", "panic", 5, "Idiomatic; 偏执 reads clinical."),
    ("Hesitant", "犹豫", "panic", 6, "Direct."),
    ("Freeze", "呆滞", "panic", 7, "ANCHORED: cn.json:293 Panic10 FREEZE -> 呆滞."),
    ("Seekcover", "寻找掩护", "panic", 8, "ANCHORED: cn.json:294 Panic11 SEEK COVER -> 寻找掩护. (en.json spells the key label 'Seekcover', one word.)"),
    ("Scream", "尖叫", "panic", 9, "ANCHORED: cn.json:295 Panic12 SCREAM -> 尖叫."),
    ("Flee", "逃跑", "panic", 10, "ANCHORED: cn.json:296 Panic13 FLEE -> 逃跑."),
    ("Frenzy", "发狂", "panic", 11, "ANCHORED: cn.json:297 Panic14 BERSERK -> 发狂. Evolved renamed BERSERK to Frenzy."),
    ("Catatonic", "失去知觉", "panic", 12, "ANCHORED: cn.json:298 Panic15 CATATONIC -> 失去知觉."),
]
for en, cn, resp, tn, why in COND:
    anchored = why.startswith("ANCHORED")
    add(en, cn, "T-PLAIN", "yze-condition-evolved",
        why + (" Not in cn.json (all 20 Evolved condition keys are missing)." if not anchored else ""),
        [(cn, A if anchored else D, "en-gated 0 (key absent from cn.json)",
          f"module/helpers/config.mjs ALIENRPG.conditions.{en.lower().replace(' ', '')}: resp=\"{resp}\", tableNumber={tn}. "
          + why)])

# ================================================== CLASSIC PANIC ENTRIES ===
PANIC = [
    ("KEEPING IT TOGETHER", "保持镇定", "cn.json:292 Panic1"),
    ("NERVOUS TWITCH", "精神紧绷", "cn.json:299 Panic7"),
    ("TREMBLE", "颤抖", "cn.json:300 Panic8"),
    ("DROP ITEM", "扔下物品", "cn.json:301 Panic9"),
    ("FREEZE", "呆滞", "cn.json:293 Panic10"),
    ("SEEK COVER", "寻找掩护", "cn.json:294 Panic11"),
    ("SCREAM", "尖叫", "cn.json:295 Panic12"),
    ("FLEE", "逃跑", "cn.json:296 Panic13"),
    ("BERSERK", "发狂", "cn.json:297 Panic14"),
    ("CATATONIC", "失去知觉", "cn.json:298 Panic15"),
]
for en, cn, cite in PANIC:
    add(en, cn, "T-PLAIN", "yze-panic-table",
        "Stratum A human translation of the d6+Stress panic table; the strongest block in the whole file.",
        [(cn, A, "en-gated 1", cite + " — re-derived line number this session.")])

# ============================================================== 12 SKILLS ===
# T-EXACT: the skill-stunts Item names must be byte-equal to these lang values.
SKILLS = [
    ("Heavy Machinery", "机械", "SkillheavyMach", 399, "Flattens 'heavy'; 重型机械 is more literal but unsourced."),
    ("Stamina", "耐力", "Skillstamina", 408, "Clean."),
    ("Close Combat", "肉搏", "SkillcloseCbt", 396, "Clean."),
    ("Mobility", "机动", "Skillmobility", 403, "Clean. The MT tooltip says 机动性 instead."),
    ("Ranged Combat", "射击", "SkillrangedCbt", 406, "Clean."),
    ("Piloting", "驾驶", "Skillpiloting", 405, "Clean."),
    ("Observation", "观察", "Skillobservation", 404, "Clean."),
    ("Survival", "求生", "Skillsurvival", 409, "Clean."),
    ("Comtech", "科技", "Skillcomtech", 398, "Drops the 'com' half; 通讯技术 is more accurate but unsourced and would break the T-EXACT byte-equality with the shipped item names if changed later."),
    ("Command", "指挥", "Skillcommand", 397, "Clean."),
    ("Manipulation", "操纵", "Skillmanipulation", 401, "Clean."),
    ("Medical Aid", "医疗", "SkillmedicalAid", 402, "Clean. The MT tooltips say 医疗援助 in prose."),
]
for en, cn, key, line, why in SKILLS:
    add(en, cn, "T-EXACT", "skill",
        "Stratum A skill set. T-EXACT: the matching skill-stunts Item name must be this string with NO English tail. " + why,
        [(cn, A, "en-gated 1", f"cn.json:{line} \"{key}\": \"{cn}\" — re-derived this session. zh-tw is the s2t conversion.")])

# ============================================================== 15 CAREERS ===
CAREERS = [
    ("Colonial Marine", "殖民陆战队", "ColonialMarine", 85, A, "en-gated 1", "Matches zh-tw 殖民陸戰隊. The subtitle corpus renders the loose 陆战队 (en-gated 7) far more often, but that is the generic short form."),
    ("Colonial Marshal", "殖民地执法官", "ColonialMarshal", 86, STD, "en-gated 0", "cn.json:86 says \"陆战队军官\" (= marine officer). A Colonial Marshal is law enforcement, not military. MUST be replaced."),
    ("Company Agent", "公司代理人", "CompanyAgent", None, A, "en-gated 1", "Clean."),
    ("Wildcatter", "独立勘探者", "Wildcatter", 479, STD, "en-gated 0", "cn.json:479 says \"野猫\" (a literal wildcat). A wildcatter is a speculative prospector/driller. MUST be replaced."),
    ("Roughneck", "钻井工", "Roughneck", 361, STD, "en-gated 0", "cn.json:361 says \"工人\" (generic worker) — all flavour lost. A roughneck is an oil-rig deck hand."),
    ("Kid", "孩童", "Kid", 222, A, "en-gated 1", "Clean."),
    ("Medic", "医护人员", "Medic", 239, A, "en-gated 1", "Matches zh-tw."),
    ("Officer", "长官", "Officer", 280, A, "en-gated 1", "Matches zh-tw."),
    ("Scientist", "科学家", "Scientist", 365, A, "en-gated 1", "Clean."),
    ("Entertainer", "艺人", "Entertainer", 162, A, "en-gated 1", "Clean."),
    ("Mercenary", "雇佣兵", "Mercenary", 245, A, "en-gated 1", "Matches zh-tw."),
    ("Pilot", "驾驶员", "Pilot", 314, A, "en-gated 1", "cn.json also has ALIENRPG.PILOT (the ship crew role) rendered 飞行员 — two renderings for one word; keep 驾驶员 for the career."),
]
for en, cn, key, line, src, cnt, why in CAREERS:
    cite = (f"cn.json:{line} \"{key}\"" if line else f"cn.json ALIENRPG.{key}") + " — re-derived this session."
    add(en, cn, "T-PLAIN", "career",
        why, [(cn, src, cnt, cite)],
        note="If this career also appears as a compendium Item name it takes T-BILINGUAL there.")
add("Synthetic", "生化人", "T-PLAIN", "career",
    "PROVISIONAL. The career/actor-type word. cn.json says 生化人; every Chinese film reference work says "
    "仿生人. The subtitle corpus does NOT support 生化人 for 'synthetic' — English-gated it renders "
    "'synthetic' exactly ONCE out of 5, while it renders android/droid 9 times out of 11. See dispute D5.",
    [("生化人", A, "en-gated 1 in subs (5 en rows); en-gated 1 in cn.json",
      "cn.json:436 \"Synthetic\": \"生化人\"; ALIENRPG.SynthDontNeed -> \"生化人无需空气，食物，水分或是休眠。\". "
      "In the subs 生化人 is the ANDROID word: en-gated 9/11 on \\bandroid|\\bdroid\\b, but only 1/5 on \\bsynthetic."),
     ("仿生人", W, "reference work, no count", "zh.wikipedia's 异形2 and 异形：夺命舰 articles use 仿生人 for Bishop and for the Romulus synthetic."),
     ("人造人", S, "en-gated 1", "ALIENS1986 1:40:23 'I may be synthetic, but I'm not stupid' -> 虽然我是人造人 我可不笨. Also the canonical Bishop line 'artificial person' -> 人造人 (0:32:07)."),
     ("机械人", S, "en-gated 1", "RESURRECTION 1:20:28 'I thought synthetics were supposed to be all logical' -> 和机械人讲逻辑."),
     ("机器人", S, "en-gated 1", "RESURRECTION 1:32:54 'this little synthetic bitch' -> 让机器人..."),
     ("合成人", S, "en-gated 1", "RESURRECTION 1:20:47 'revitalize the synthetic industry' -> 复兴合成人工业.")],
    aliases=["仿生人", "人造人"], dispute="D5-synthetic")
add("Artificial Person", "人造人", "T-PLAIN", "franchise-vocab",
    "The canonical in-universe euphemism; the subs render Bishop's line exactly this way.",
    [("人造人", S, "en-gated 2", "ALIENS1986 0:32:07 'I prefer the term \"artificial person\" myself' -> 我比较喜欢被称为\"人造人\"; 0:32:16.")])
add("Android", "生化人", "T-PLAIN", "franchise-vocab",
    "The subtitle corpus's strongest artificial-being term, English-gated: 9 of 11 android/droid rows.",
    [("生化人", S, "en-gated 9 (regex \\bandroid|\\bdroid\\b, 11 en rows; ALIEN1979+ALIENS1986+ALIEN3)",
      "ALIEN1979 1:21:38 'Android! He's an android!' -> 生化人 他是生化人!; ALIEN3 2:13:31 'I'm not the Bishop android' -> 我不是主教那种生化人.")])
add("Robot", "机器人", "T-PLAIN", "franchise-vocab",
    "English-gated 5 of 8 robot rows; a distinct word from android in the corpus.",
    [("机器人", S, "en-gated 5 (regex \\brobot, 8 en rows)",
      "RESURRECTION 1:20:41 'Robots designed by robots' -> 机器人造的机器人.")])

# ================================================================= RANGES ===
add("Engaged", "接战", "T-PLAIN", "range",
    "RESOLVED COLLISION. cn.json:157 gives Engaged=近战, but cn.json:476 gives Melee=近战 too — one string "
    "for two different game terms. 接战 is en-gated 2/3 on ENGAGED and 0 on Melee.",
    [("接战", A, "en-gated 2 (regex 'ENGAGED|Engaged', 3 en rows)",
      "cn.json:294 Panic11 \"如果你的接战范围内有敌人\"; :296 Panic13 same."),
     ("近战", B, "en-gated 1 — REJECTED (collides)",
      "cn.json:157 \"Engaged\": \"近战\" AND cn.json:476 \"WepTypeMelee\": \"近战\".")],
    aliases=["近战"], dispute="D2-engaged")
add("Melee", "近战", "T-PLAIN", "range",
    "Keeps 近战 for the weapon type, where it is correct and uncontested once Engaged moves to 接战.",
    [("近战", A, "en-gated 1 (regex \\bMelee\\b)", "cn.json:476 \"WepTypeMelee\": \"近战\".")])
add("Short", "短程", "T-PLAIN", "range",
    "UI key value; stratum-A prose says 短距范围 / 短距离 for the same range band.",
    [("短程", A, "en-gated 1", "cn.json:387 \"Short\": \"短程\"."),
     ("短距范围", A, "in prose", "cn.json:293 Panic10, :294 Panic11."),
     ("短距离", A, "in prose", "cn.json:299 Panic7.")],
    aliases=["短距范围"])
add("Medium", "中程", "T-PLAIN", "range", "UI key value.",
    [("中程", A, "en-gated 1", "cn.json:241 \"Medium\": \"中程\".")])
add("Long", "远程", "T-PLAIN", "range",
    "UI key value. Shares 远程 with the Ranged weapon type, but they are different dropdowns — harmless.",
    [("远程", A, "en-gated 1", "cn.json:229 \"Long\": \"远程\".")])
add("Extreme", "极远程", "T-PLAIN", "range", "UI key value, matches zh-tw.",
    [("极远程", A, "en-gated 1", "cn.json:166 \"Extreme\": \"极远程\".")])
add("Ranged", "远程", "T-PLAIN", "range", "Weapon type. Same word as the Long range band, different axis.",
    [("远程", A, "en-gated 1", "cn.json ALIENRPG.WepTypeRanged \"Ranged\" -> \"远程\".")])

# ==================================================== XENOMORPH / CREATURE ===
add("Alien", "异形", "T-BILINGUAL", "franchise-creature",
    "异形 is uniform: the subtitle corpus has 56 occurrences and ZERO of 异型, and zh.wikipedia's article "
    "title is 异形 (虚构生物). Owner decision: 异形 always, 异型 never.",
    [("异形", W + " + " + S, "en-gated: 56 bare hits across all 4 films; zh-wiki article title",
      "zh.wikipedia 异形 (虚构生物): 「異形（英語：Alien，又稱作Xenomorph XX121）是虛構的外星生物」. cn.json also uses it: ALIENRPG.DialTextXeno -> \"请输入对异形造成的伤害。\"")],
    note="异形 = the CREATURE noun only. The adjective 'alien' is 外星 — see the alien-* collocations.")
add("Xenomorph", "异形", "T-BILINGUAL", "franchise-creature",
    "Same word. The subs do not distinguish Alien from Xenomorph and neither does zh-wiki.",
    [("异形", W + " + " + S, "en-gated 3 (regex 'xenomorph', 3 en rows in subs)",
      "ALIENS1986 0:33:42 'A xenomorph.' -> 异形; ALIEN3 1:28:18 -> 异形怪物.")])
add("alien ship", "外星飞船", "T-PLAIN", "franchise-creature",
    "COLLOCATION, recorded so the pipeline cannot emit 异形船. The subs make the split explicitly.",
    [("外星船", S, "en-gated 1", "ALIENS1986 0:12:31 'It was an alien ship.' -> 是外星船.")])
add("alien life-form", "外星生命体", "T-PLAIN", "franchise-creature",
    "COLLOCATION. 外星 is the adjective.",
    [("外星生物", S, "en-gated 1", "ALIEN1979 1:18:24 on-screen text -> 调查外星生物.")])
add("extraterrestrial", "外星", "T-PLAIN", "franchise-creature", "The adjective sense of 'alien'.", [])
add("Alien Queen", "异形女王", "T-BILINGUAL", "franchise-creature",
    "zh.wikipedia names it directly; the subtitle corpus's 女王/王后 split is about the bare noun.",
    [("异形女王", W, "reference work", "zh.wikipedia 异形 (虚构生物): 「異形女王」（Alien Queen）是一個異形族群的領導者."),
     ("异形母后", W, "reference work, plot summary", "zh.wikipedia 异形2 plot summary uses 异形母后."),
     ("异形王后", S, "en-gated 0 (cn_only 1)", "RESURRECTION 0:11:27 'Her Majesty here is the real payoff' -> 异形王后才值回票价.")],
    aliases=["异形母后"], dispute="D6-queen")
add("Queen", "女王", "T-BILINGUAL", "franchise-creature",
    "PROVISIONAL. English-gated the subs give 女王 4 vs 王后 3 — much tighter than the bare counts suggested "
    "(7 vs 4), and one of the four 女王 hits is the bee-metaphor 女王蜂. zh.wikipedia settles it at 女王. See dispute D6.",
    [("女王", W + " + " + S, "en-gated 4 (regex \\bqueen\\b, 7 en rows) — ALIEN3 3, ALIENS1986 1",
      "ALIEN3 1:41:18 'I'm carrying the new queen' -> 我孕育新女王; 1:46:49 -> 它是个女王 会产卵. ALIENS1986 1:35:28 is 女王蜂 (bee metaphor). Plus zh-wiki 异形女王."),
     ("王后", S, "en-gated 3 — RESURRECTION only", "RESURRECTION 0:13:09 'It's a queen.' -> 是王后; 1:34:19 -> 王后产卵.")],
    aliases=["王后"], dispute="D6-queen")
add("Facehugger", "抱脸虫", "T-BILINGUAL", "franchise-creature",
    "OWNER'S DELIBERATE EXCEPTION to the wiki-formal spine: 百度百科 and 萌娘百科 both title their article 抱脸虫 "
    "and it is the dominant mainland form, while zh.wikipedia says 抱脸体. ZERO attestation in the subtitle corpus.",
    [("抱脸虫", BD + " + " + MG, "reference work, no count",
      "百度百科 has a dedicated 抱脸虫 entry (baike.baidu.com/item/抱脸虫/3022517); 萌娘百科 titles its article 抱脸虫."),
     ("抱脸体", W, "reference work", "zh.wikipedia 异形 (虚构生物): 「抱面體」或「抱臉體」（Facehugger）; 异形：夺命舰 uses 抱脸体."),
     ("抱面体", W, "reference work, variant", "Same zh-wiki sentence, alternate character."),
     ("抱脸者", B, "en-gated 2 IN cn.json — new find, not in the survey",
      "cn.json ALIENRPG.AutoPanicHint -> \"在投掷抱脸者后手动投掷恐慌\"; ALIENRPG.TAH.radiation also uses 抱脸者. A fourth candidate, stratum B, rejected."),
     ("(none)", S, "en-gated 0", "ZERO hits for facehugger/hugger in all 4 films.")],
    aliases=["抱脸体", "抱面体", "抱脸者"])
add("Chestburster", "破胸体", "T-BILINGUAL", "franchise-creature",
    "Wiki-formal spine, per the owner's decision. ZERO attestation in the subtitle corpus.",
    [("破胸体", W, "reference work", "zh.wikipedia 异形 (虚构生物): 「破胸體」（Chestburster）是由「抱面體」的生殖器從口腔把胚胎植入; 异形：夺命舰 uses 破胸体."),
     ("破胸者", BD, "reference work, colloquial", "百度百科 / mainland fan usage treats 破胸体 and 破胸者 as interchangeable."),
     ("(none)", S, "en-gated 0", "ZERO hits for chestburst/burster in all 4 films.")],
    aliases=["破胸者"])
add("Ovomorph", "异形卵", "T-BILINGUAL", "franchise-creature",
    "zh.wikipedia names it directly. ZERO attestation in the subtitle corpus.",
    [("异形卵", W, "reference work", "zh.wikipedia 异形 (虚构生物): 「異形卵」（Ovomorph）...是由在異形巢穴之中的「異形女王」，所生產出來的.")])
add("Egg", "卵", "T-PLAIN", "franchise-creature",
    "The bare noun. 卵 for the technical/collective sense, 蛋 only colloquially.",
    [("卵", S, "en-gated across 3 films", "ALIENS1986 0:13:09 'thousands of eggs' -> 上千个异形的卵; 1:35:05 'comes from an egg' -> 卵生的; ALIEN3 1:46:49 'egg layer' -> 会产卵."),
     ("蛋", S, "colloquial", "ALIEN1979 0:33:00; ALIENS1986 1:35:11.")])
add("Drone", "工蜂异形", "T-BILINGUAL", "franchise-creature",
    "zh.wikipedia offers 工蜂异形 or 人形异形; the first keeps the hive metaphor the game uses. ZERO subtitle attestation.",
    [("工蜂异形", W, "reference work", "zh.wikipedia 异形 (虚构生物): 「工蜂異形」或「人形異形」（Drone）."),
     ("人形异形", W, "reference work, alternate", "Same sentence."),
     ("(none)", S, "en-gated 0", "ZERO hits for drone as a caste in all 4 films.")],
    aliases=["人形异形"])
add("Warrior", "战士异形", "T-BILINGUAL", "franchise-creature",
    "zh.wikipedia offers 战斗异形 or 战士异形; 战士 matches the game's caste naming. ZERO subtitle attestation.",
    [("战士异形", W, "reference work", "zh.wikipedia 异形 (虚构生物): 「戰鬥異形」或「戰士異形」（Warrior）."),
     ("战斗异形", W, "reference work, alternate", "Same sentence."),
     ("(none)", S, "en-gated 0", "ZERO hits for warrior as a caste in all 4 films.")],
    aliases=["战斗异形"])
add("Acid Blood", "酸血", "T-PLAIN", "franchise-creature",
    "The game term, attested in cn.json; 强酸 is the subtitle corpus's word for the substance in prose.",
    [("酸血", A, "en-gated 1", "cn.json:23 \"AcidAttack\": \"酸血\". Also ALIENS1986 x2 in the subs."),
     ("强酸", S, "en-gated 3 (regex 'acid', 7 en rows), 3 films", "ALIEN1979 0:50:37 'This thing bled acid' -> 它的血都是强酸; ALIENS1986 0:12:50; ALIEN3 1:01:22."),
     ("硫酸", S, "en-gated 2 — REJECTED", "'molecular acid' -> 硫酸 (= sulfuric acid); a mistranslation.")],
    aliases=["强酸"],
    note="cn.json ALIENRPG.SkillAcidSplash \"Acid Splash\" -> \"酸性血液\" is wrong: splash is the effect, not the substance. Should be 酸血飞溅.")
add("Molecular Acid", "分子酸", "T-PLAIN", "franchise-creature",
    "Literal. The subtitle rendering 硫酸 (sulfuric acid) is simply wrong.",
    [("分子酸", STD, "en-gated 0", "Subs render it 硫酸 (ALIEN1979 0:42:15) and 酸血层 (ALIENS1986) — both reject.")])
add("Acid Splash", "酸血飞溅", "T-PLAIN", "franchise-creature", "Corrects cn.json's confusion of effect with substance.",
    [("酸血飞溅", D, "en-gated 0", "cn.json ALIENRPG.SkillAcidSplash -> \"酸性血液\" (= acidic blood, the substance).")])
add("Host", "宿主", "T-PLAIN", "franchise-creature",
    "en-gated 2 vs 寄主 1; 宿主 is also the standard biological term.",
    [("宿主", S + " + " + STD, "en-gated 2 (regex \\bhost\\b, 4 en rows)", "RESURRECTION 0:52:16 'She was the host' -> 她是这些异形的宿主; 1:34:35."),
     ("寄主", S, "en-gated 1", "RESURRECTION 0:05:56.")],
    aliases=["寄主"])
add("Hive", "巢穴", "T-PLAIN", "franchise-creature", "zh-wiki uses 異形巢穴; the subs agree.",
    [("巢穴", W + " + " + S, "en-gated 1 (regex \\bnest\\b|\\bhive\\b, 4 en rows)",
      "zh.wikipedia 异形 (虚构生物): 「在異形巢穴之中」. RESURRECTION 1:28:41 'we're near the nest' -> 接近异形巢穴了.")])
add("Nest", "巢穴", "T-PLAIN", "franchise-creature", "Same word as Hive.",
    [("筑巢", S, "en-gated 1, verb", "ALIEN3 1:14:34 'It'll nest in this area' -> 它会在这里筑巢.")])
add("Parasite", "寄生虫", "T-PLAIN", "franchise-creature", "Unambiguous, 2 films.",
    [("寄生虫", S, "en-gated 2", "ALIENS1986 0:34:01 'some kind of parasite' -> 类似寄生虫的东西; RESURRECTION 1:06:07.")])
add("Specimen", "标本", "T-PLAIN", "franchise-vocab", "en-gated 2, 2 films.",
    [("标本", S, "en-gated 2 (regex \\bspecimen, 3 en rows)", "ALIENS1986 1:35:36 'I want these specimens destroyed' -> 我要你把标本销毁; ALIEN3 2:16:26.")])
add("Organism", "生物", "T-PLAIN", "franchise-vocab", "The general noun; 有机体 only in the famous 'perfect organism' line.",
    [("生物", S, "en-gated, 3 films", "ALIEN1979 0:36:22 'An organism.' -> 一种生物; ALIENS1986 0:12:16."),
     ("有机体", S, "en-gated 1", "ALIEN1979 1:24:21 'A perfect organism' -> 完美的有机体.")],
    aliases=["有机体"])

# =========================================================== CORPORATIONS ===
add("Weyland-Yutani", "韦兰-尤坦尼集团", "T-BILINGUAL", "franchise-corp",
    "PROVISIONAL, per the owner's wiki-formal spine. English-gated the subtitle corpus casts exactly ONE "
    "vote (伟伦优达尼公司), it is Taiwan-flavoured, and neither zh-wiki's nor the fan corpus's form appears "
    "anywhere in it. See dispute D7.",
    [("韦兰-尤坦尼集团", W, "reference work; article title", "zh.wikipedia's article title and the 异形/异形2/异形：夺命舰 articles all use 韦兰-尤坦尼集团."),
     ("韦兰德-尤坦尼集团", W, "reference work; article lead", "The dedicated corp article's lead; traditional variant 韋蘭德-湯谷企業; alternate 韦兰德-汤谷公司."),
     ("维兰德-汤谷公司", "Chinese fan corpus (bilibili / douban / 机核)", "no count — corpus not machine-readable this session", "Overwhelmingly the fan form; short form 维汤."),
     ("伟伦优达尼公司", S, "en-gated 1 (regex 'Weyland|Yutani', 2 en rows)",
      "RESURRECTION 0:13:25 'Weyland-Yutani. Ripley 8's former employers.' -> 伟伦优达尼公司 蕾普丽8号生前的雇主. The ALIEN3 0:09:09 hit is on-screen text with an EMPTY English side, so it is cn_only, not a gated vote. ALIEN3 0:13:05 drops the name entirely.")],
    aliases=["韦兰德-尤坦尼集团", "维兰德-汤谷公司", "伟伦优达尼公司"], dispute="D7-weyland-yutani")
add("Weyland-Yutani Corporation", "韦兰-尤坦尼集团", "T-BILINGUAL", "franchise-corp",
    "集团 already carries 'Corporation'; do NOT append 公司.", [], dispute="D7-weyland-yutani")
add("Weyland Corporation", "韦兰德公司", "T-BILINGUAL", "franchise-corp", "zh-wiki's corp article.",
    [("韦兰德公司", W, "reference work", "zh.wikipedia 韦兰-尤坦尼集团 article: 韦兰德公司. The 异形系列 page uses 韦兰德工业 for Weyland Industries.")])
add("Yutani Corporation", "尤坦尼公司", "T-BILINGUAL", "franchise-corp", "zh-wiki's corp article.",
    [("尤坦尼公司", W, "reference work", "zh.wikipedia 韦兰-尤坦尼集团 article: 尤坦尼公司.")])
add("The Company", "公司", "T-PLAIN", "franchise-corp",
    "Both sources agree and the English-gated count is decisive.",
    [("公司", W + " + " + S, "en-gated 17 (regex \\bcompany\\b, 23 en rows), all 4 films",
      "zh.wikipedia 韦兰-尤坦尼集团: 「有时只被简称为公司（英语：The Company）」. ALIEN1979 1:22:35 'the company sent us a goddamn robot' -> 公司怎么会派生化人来."),
     ("机构", S, "en-gated 2", "RESURRECTION 0:13:56 — deliberate: in the U.S.M. era Weyland-Yutani no longer exists.")])
add("Building Better Worlds", "建造更好的世界", "T-BILINGUAL", "franchise-corp",
    "The Weyland-Yutani slogan. Subtitle rendering tidied (drops the stray 一个).",
    [("建造一个更好的世界", S, "en-gated 1", "ALIENS1986 0:22:46 'We're getting into a lot of terraforming now. \"Building Better Worlds\"' -> ...建造一个更好的世界.")])

# ============================================================ SHIPS / NAV ===
add("Nostromo", "诺史莫号", "T-BILINGUAL", "franchise-ship",
    "Wiki-formal spine per the owner's decision. The subtitle form 诺斯都罗莫号 is en-gated 5 and is the "
    "only one in the corpus, but zh-wiki's own articles say 诺史莫号.",
    [("诺史莫号", W, "reference work", "zh.wikipedia uses 诺史莫号 in the 异形 (电影) / 异形 (虚构生物) / 异形：夺命舰 articles. Search snippets of the same 异形 (电影) body also show 诺斯托罗莫号 — the wiki is internally inconsistent."),
     ("诺斯都罗莫号", S, "en-gated 5 (regex 'Nostromo', 6 en rows) — ALIEN1979 4, ALIEN3 1",
      "ALIEN1979 0:02:05 title card -> 宇宙货船诺斯都罗莫号; ALIEN3 2:18:50 -> 诺斯都罗莫号唯一生还者."),
     ("诺斯托罗莫号", "Chinese fan usage", "no count", "Common in fan writing; not the wiki's article form.")],
    aliases=["诺斯都罗莫号", "诺斯托罗莫号"],
    note="Ship naming pattern: <音译>号. Ship prefixes (USCSS / U.S.M. / USS) are dropped in Chinese — ZERO attestation in the subs.")
add("USCSS", "USCSS", "T-FROZEN", "franchise-ship",
    "Ship prefix. Kept Latin: the subtitle corpus never renders any prefix (en-gated 0), and zh-wiki drops them.",
    [("(none)", S, "en-gated 0", "ZERO hits for USCSS / USCM in all 4 subtitle files.")])
add("MU/TH/UR", "MU/TH/UR", "T-FROZEN", "franchise-computer",
    "Kept Latin: it is a stylised acronym, and zh-wiki prints it as \"MU/TH/UR 6000\" alongside the nickname.",
    [("MU/TH/UR 6000", W, "reference work", "zh.wikipedia 异形 (电影) prints the computer as 母亲 and MU/TH/UR 6000.")])
add("Mother", "母亲", "T-BILINGUAL", "franchise-computer",
    "The ship computer's nickname. zh-wiki says 母亲; the 1979 subs say 电脑老妈, which is a colloquialism, not a name.",
    [("母亲", W, "reference work", "zh.wikipedia 异形 (电影)."),
     ("电脑老妈", S, "en-gated ~8, ALIEN1979 only", "0:07:37 'Mother wants to talk to you' -> 电脑老妈找你说话.")],
    aliases=["电脑老妈"])
add("LV-426", "LV-426", "T-FROZEN", "franchise-place",
    "Latin designator, kept verbatim WITH the hyphen: zh-wiki writes LV-426, the subs drop the hyphen.",
    [("LV-426", W, "reference work", "zh.wikipedia 异形 (电影): 「LV-426号的行星」; 异形2: 「LV-426号行星」."),
     ("LV426", S, "en-gated 5 — hyphen dropped", "ALIENS1986 0:21:38 'the colony on LV-426' -> 我们与LV426失去联络.")])

# ============================================================== CHARACTERS ===
CHARS = [
    ("Ripley", "蕾普丽", "en-gated 75 (regex \\bRipley, 96 en rows) vs 蕾普莉 11",
     "蕾普丽 spans ALIENS1986/ALIEN3/RESURRECTION; 蕾普莉 is ALIEN1979 only. Bare counts in the survey were 82/12; English-gated they are 75/11."),
    ("Bishop", "主教", "en-gated 23 (27 en rows), ALIENS1986 + ALIEN3",
     "Rendered semantically (the chess/church rank), not phonetically — a deliberate, consistent choice. ALIEN3 2:13:15 'Bishop.' -> 主教."),
    ("Hicks", "希克斯", "en-gated 29 (32 en rows) vs 西克斯 1",
     "One stray variant at ALIENS1986 0:44:16."),
    ("Hudson", "哈德逊", "en-gated 32 (37 en rows) vs 韩森 1",
     "One stray variant at ALIENS1986 0:44:09."),
    ("Newt", "纽特", "en-gated 29 (35 en rows)",
     "ALIENS1986 1:02:22 'N-Newt.' -> 纽特. Her real name Rebecca is 丽贝卡 (1:02:32)."),
    ("Vasquez", "娃丝佳", "en-gated 13 (16 en rows)", "ALIENS1986 only."),
    ("Dallas", "达拉斯", "en-gated ~24, ALIEN1979 + ALIEN3", "ALIEN3 2:18:33 'Captain Dallas' -> 达拉斯船长."),
    ("Kane", "肯恩", "en-gated ~14, ALIEN1979 + ALIENS1986", "ALIEN1979 0:33:31, 0:36:16, 0:53:49."),
    ("Parker", "派克", "en-gated ~13, ALIEN1979 + ALIEN3", "ALIEN1979 x12, ALIEN3 x1."),
    ("Burke", "巴克", "en-gated 15, ALIENS1986", "1:36:27 'signed Burke, Carter J.' -> 巴克卡特签署. The self-introduction line 0:06:08 says 卡特巴特 — inconsistent, ignore."),
    ("Gorman", "高曼", "en-gated 11, ALIENS1986", "0:21:30 'This is Lieutenant Gorman' -> 高曼中尉."),
    ("Apone", "阿朋", "en-gated 10, ALIENS1986", "1:04:36 'Let's saddle up, Apone' -> 整装出发 阿朋."),
    ("Frost", "法斯", "en-gated 11, ALIENS1986", "1:12:36 'Frost, flamethrower. Kill it!' -> 法斯 发射喷火枪 杀了它!"),
    ("Drake", "垂克", "en-gated 8, ALIENS1986", "0:43:16."),
    ("Dietrich", "迪杰", "en-gated 6, ALIENS1986", "0:49:49."),
    ("Wierzbowski", "威伯斯基", "en-gated 5, ALIENS1986", "0:29:15."),
    ("Ferro", "弗洛", "en-gated 3, ALIENS1986", "0:45:04."),
    ("Dillon", "迪伦", "en-gated 18, ALIEN3", "0:09:36 'Here we go, Mr. Dillon' -> 可以了 迪伦先生."),
    ("Clemens", "克里蒙斯", "en-gated 4, ALIEN3", "0:12:59 'My name is Clemens.' -> 我叫克里蒙斯. 'Mr. Clemens' is habitually shortened to 克先生."),
    ("Andrews", "安德鲁", "en-gated 8, ALIEN3", "0:41:26 'Superintendent Andrews' -> 狱长安德鲁."),
    ("Aaron", "亚伦", "en-gated 9, ALIEN3", "0:33:58. The nickname '85' stays numeric."),
    ("Golic", "葛立", "en-gated 10, ALIEN3", "0:32:09."),
    ("Call", "柯儿", "en-gated 17, RESURRECTION", "0:16:05 'Vriess. Call.' -> 瑞斯 柯儿."),
    ("Elgyn", "艾金", "en-gated 12, RESURRECTION", "0:21:38."),
    ("Christie", "克里斯蒂", "en-gated 8, RESURRECTION", "0:38:20."),
    ("Vriess", "瑞斯", "en-gated 7, RESURRECTION", "0:18:12."),
    ("Johner", "强纳", "en-gated 4, RESURRECTION", "0:18:18."),
    ("Wren", "瑞温", "en-gated 4, RESURRECTION", "0:38:57 'Release Dr. Wren!' -> 放开瑞温博士!"),
]
for en, cn, cnt, ev in CHARS:
    add(en, cn, "T-BILINGUAL", "franchise-character",
        "Subtitle corpus, English-gated. The corpus is ONE vote, but for a phonetic transliteration with no "
        "reference-work competitor a single consistent lineage is the best source available.",
        [(cn, S, cnt, ev)])
add("Ash", "艾希", "T-BILINGUAL", "franchise-character",
    "en-gated 13 in ALIEN1979; the single 艾许 at ALIEN3 2:18:33 is a stray in a repeated log line.",
    [("艾希", S, "en-gated 13, ALIEN1979", "0:31:13, 1:09:15, 1:23:17."),
     ("艾许", S, "en-gated 1, ALIEN3", "2:18:33.")], aliases=["艾许"])
add("Lambert", "兰波特", "T-BILINGUAL", "franchise-character",
    "en-gated 13 vs 1 within the same file.",
    [("兰波特", S, "en-gated 13, ALIEN1979", "内部不一致：0:30:50 says 兰伯特 once.")], aliases=["兰伯特"])
add("Brett", "布雷特", "T-BILINGUAL", "franchise-character",
    "Three renderings in ONE file; 布雷特 leads 8 vs 2 vs 3.",
    [("布雷特", S, "en-gated 8, ALIEN1979", "布瑞特 2 (0:19:17), 布瑞 3 (0:07:02) — same file, same character.")],
    aliases=["布瑞特", "布瑞"])
add("Jones", "钟斯", "T-BILINGUAL", "franchise-character", "The cat. Thin but uncontested.",
    [("钟斯", S, "en-gated 2, ALIENS1986", "0:05:56 'Jonesy. Come here.' -> 钟斯 过来.")])
add("Rebecca Jorden", "丽贝卡·乔登", "T-BILINGUAL", "franchise-character",
    "Given name attested; the surname Jorden is NOT rendered anywhere in the corpus — 乔登 is the standard "
    "Chinese transliteration and is marked medium confidence.",
    [("丽贝卡", S, "en-gated 5, ALIENS1986", "1:02:32 'Nobody calls me Rebecca except my brother' -> 除了我哥哥 没人叫我丽贝卡."),
     ("乔登", STD, "en-gated 0", "Surname unattested in the corpus.")])

# ==================================================== HARDWARE / MARINE ===
add("Pulse Rifle", "脉冲步枪", "T-BILINGUAL", "hardware",
    "萌娘百科's article title is literally M41A脉冲步枪. The subtitle corpus's 电波枪 family is Taiwan-flavoured "
    "and internally inconsistent (three forms in one film) — rejected.",
    [("脉冲步枪", MG, "reference work; article title", "萌娘百科 zh.moegirl.org.cn/M41A脉冲步枪; MC百科 item page 'M41A脉冲步枪 (M41A Pulse Rifle)' mcmod.cn/item/226422.html; bilibili cv22321766 阿玛特防务M41A脉冲步枪."),
     ("电波枪", S, "en-gated 1 (regex 'pulse rifle', 4 en rows)", "ALIENS1986 1:08:25. Also 电波机关枪 1 (1:44:01) and 电波散弹枪 1 (1:26:31) — three renderings in one film.")],
    aliases=["电波枪"])
add("M41A Pulse Rifle", "M41A脉冲步枪", "T-BILINGUAL", "hardware",
    "Reference-work attested verbatim, including the model prefix.",
    [("M41A脉冲步枪", MG, "reference work; article title", "萌娘百科 article title; MC百科 item page.")])
add("Flamethrower", "喷火枪", "T-BILINGUAL", "hardware",
    "en-gated 5 across 2 films, and it also covers the film's 'incinerator unit'.",
    [("喷火枪", S, "en-gated 5 (regex 'flame ?thrower|flame unit|incinerator', 9 en rows)",
      "ALIEN1979 1:09:36 'three or four incinerator units?' -> 三 四支喷火枪吗; ALIENS1986 1:09:25 'Flame units only' -> 只准用喷火枪."),
     ("焚化炉", S, "en-gated 2 — REJECTED", "ALIENS1986 1:15:08/1:15:17 render 'incinerator' as 焚化炉, which reads as a crematorium and is used there for the A.P.C.")])
add("Incinerator Unit", "喷火枪", "T-BILINGUAL", "hardware", "Same device as Flamethrower in this franchise.", [])
add("Colonial Marines", "殖民陆战队", "T-BILINGUAL", "marine",
    "cn.json's career key and zh-tw agree; the subs' 陆战队 (en-gated 7) is the generic short form.",
    [("殖民陆战队", A, "en-gated 1 in cn.json; en-gated 1 in subs",
      "cn.json:85 \"ColonialMarine\": \"殖民陆战队\". ALIENS1986 0:22:07 'These Colonial Marines are very tough hombres' -> 这些殖民陆战队都是顶尖的."),
     ("陆战队", S, "en-gated 7 (regex \\bmarine, 10 en rows)", "The working short form; keep as an alias, not the spine."),
     ("海军陆战队", S, "en-gated 1", "ALIEN3 1:52:11 — wrong branch (naval infantry), reject.")],
    aliases=["陆战队"])
add("Marine", "陆战队员", "T-PLAIN", "marine", "The individual soldier / vocative.",
    [("陆战队员", S, "en-gated 2, ALIENS1986", "0:33:17 'Morning, marines' -> 早安 陆战队员.")])
add("Colonial Administration", "殖民地行政机构", "T-PLAIN", "marine",
    "The subs render only the person (殖民地行政主管); the institution needs 机构.",
    [("殖民地行政主管", S, "en-gated 1", "ALIENS1986 0:09:34 'Colonial Administration, insurance company guys' -> 殖民地行政主管 保险公司 — that is the officer, not the body.")])
add("United Systems Military", "联合星系军", "T-PLAIN", "marine",
    "Coined. The subs flatten it to the bare 军队 and drop the abbreviation entirely.",
    [("军队", S, "en-gated 1 — too generic", "RESURRECTION 0:13:56 'It's United Systems Military, not some greedy corporation' -> 这是军队 不是贪婪的机构. 'U.S.M. Auriga' -> 奥瑞戈舰, prefix dropped.")])
add("Bug Hunt", "除虫行动", "T-PLAIN", "marine",
    "en-gated 2, internally consistent, and exactly the flavour the RPG wants.",
    [("除虫行动", S, "en-gated 2, ALIENS1986", "0:33:31 'or another bug hunt?' -> 还是除虫行动?; 0:33:45 'It's a bug hunt.' -> 这是除虫行动.")])
add("Captain", "船长", "T-PLAIN", "marine", "Shipboard rank; 2 films on the same log line.",
    [("船长", S, "en-gated 2", "ALIEN1979 1:52:06 / ALIEN3 2:18:33 'Captain Dallas' -> 达拉斯船长.")])
add("Lieutenant", "中尉", "T-PLAIN", "marine", "en-gated 21 across 3 films — the strongest rank term.",
    [("中尉", S, "en-gated 21, 3 films", "ALIENS1986 0:22:15; ALIEN3 0:14:44 'Lieutenant Ripley' -> 蕾普丽中尉; RESURRECTION 0:35:30 'Lieutenant First Class' -> 一级中尉.")])
add("Corporal", "下士", "T-PLAIN", "marine", "en-gated 7 across 2 films.",
    [("下士", S, "en-gated 7", "ALIENS1986 1:22:29 'Corporal Hicks' -> 希克斯下士; ALIEN3 0:15:44.")])
add("Sergeant", "中士", "T-PLAIN", "marine",
    "CORRECTED. The corpus prefers 士官长 (4 vs 1) but 士官长 is the Taiwan rendering of Sergeant Major / "
    "Master Sergeant, a much higher rank; 中士 is the correct mainland rendering of Sergeant and also "
    "appears in the same film.",
    [("士官长", S, "en-gated 4 — REJECTED (Taiwan-flavoured, wrong rank)", "ALIENS1986 0:36:14 'Hey, Sarge' -> 士官长; 1:15:55; 1:20:53."),
     ("中士", S + " + " + STD, "en-gated 1", "ALIENS1986 1:15:34 'Sarge! Sarge!' -> 中士! — same film, contradicting itself.")],
    aliases=["士官长"])
add("Private", "二等兵", "T-PLAIN", "marine", "Single usable hit but standard and unambiguous.",
    [("二等兵", S, "en-gated 1", "ALIENS1986 0:35:16 'What is it, Private?' -> 什么事? 二等兵.")])
add("Warrant Officer", "准尉", "T-PLAIN", "marine",
    "CORRECTED. The subs flatten it to the generic 军官 (= officer), losing the rank; 准尉 is the standard term.",
    [("军官", S, "en-gated 1 — REJECTED", "ALIENS1986 0:13:35 'Warrant Officer E. Ripley' -> 蕾普丽军官."),
     ("准尉", STD, "en-gated 0", "Standard Chinese rendering of Warrant Officer.")])
add("Executive Officer", "大副", "T-PLAIN", "marine",
    "ZERO attestation in the corpus. 大副 is the standard merchant-marine rendering, which fits a "
    "commercial towing vehicle better than the naval 副长.",
    [("(none)", S, "en-gated 0", "ZERO hits across all 4 films.")])
add("Science Officer", "科学官", "T-PLAIN", "marine",
    "科学官 is the shipboard-role form; 科学研究员 (en-gated 2) is a job description, not a post.",
    [("科学官", S, "en-gated 1", "ALIEN1979 1:09:15 'Science department should be able to help us' -> 艾希 你是科学官."),
     ("科学研究员", S, "en-gated 2", "0:46:22, 0:50:51."),
     ("卫生署", S, "REJECTED", "0:45:52 'anything to do with the science division' -> 卫生署 — a mistranslation (health ministry).")],
    aliases=["科学研究员"])
add("Chief Medical Officer", "医疗主任", "T-PLAIN", "marine", "Single clean hit.",
    [("医疗主任", S, "en-gated 1", "ALIEN3 0:12:59 'I'm the chief medical officer here' -> 这里的医疗主任.")])

# ================================================ SHIPBOARD / COLONY OPS ===
add("Hypersleep", "长眠", "T-PLAIN", "ops", "en-gated 4, 2 films, no competitor.",
    [("长眠", S, "en-gated 4", "ALIENS1986 0:06:24 'such an unusually long hypersleep' -> 过度长眠; ALIEN3 0:13:40.")])
add("Cryo-tube", "低温舱", "T-PLAIN", "ops",
    "en-gated 7 vs 2. 冰温舱 (RESURRECTION only) is not standard Chinese.",
    [("低温舱", S, "en-gated 7 (regex 'cryo', 9 en rows), ALIEN3", "0:02:02 'Fire in cryogenic compartment' -> 低温舱发生火警; 0:15:52 'She drowned in her cryo-tube' -> 她溺死在低温舱内."),
     ("冰温舱", S, "en-gated 2, RESURRECTION", "1:04:49 'I was in cryo' -> 我在冰温舱.")],
    aliases=["冰温舱"])
add("Cryogenic Compartment", "低温舱", "T-PLAIN", "ops", "Same compartment as Cryo-tube in the corpus.", [])
add("Stasis", "休眠", "T-PLAIN", "ops",
    "CORRECTED. 静态平衡 is en-gated 2 but means 'static equilibrium' — a literal MT-grade rendering that "
    "makes no sense for suspended animation.",
    [("静态平衡", S, "en-gated 2 — REJECTED", "ALIEN3 0:02:02 & 0:57:42 'Stasis interrupted' -> 静态平衡受(到)干扰."),
     ("休眠", S, "en-gated 1", "RESURRECTION 0:25:02 'Stasis uninterrupted' -> 休眠中.")])
add("Air Lock", "气闸", "T-PLAIN", "ops",
    "CORRECTED. The corpus gives four renderings (空气舱 4 / 气舱 6 / 气密门 1), none of which is the standard "
    "Chinese aerospace term. 气闸 is; 气闸舱 for the compartment.",
    [("气舱", S, "en-gated 6", "ALIEN1979 1:09:50 主气舱室; ALIENS1986 0:12:06 气舱外."),
     ("空气舱", S, "en-gated 4", "ALIEN1979 1:00:12, 1:09:05, 1:10:37."),
     ("气密门", S, "en-gated 1", "RESURRECTION 1:39:02."),
     ("气闸", STD, "en-gated 0", "Standard Chinese aerospace term for an airlock.")],
    aliases=["气闸舱", "空气舱"])
add("Quarantine", "检疫", "T-PLAIN", "ops",
    "en-gated 4 vs 2 — the survey called these 'equally attested' on bare counts; English-gating separates them.",
    [("检疫", S, "en-gated 4 (regex 'quarantine', 8 en rows)", "ALIEN1979 0:45:52 'basic quarantine law' -> 检疫规定; ALIENS1986 1:36:08."),
     ("隔离", S, "en-gated 2", "ALIEN1979 0:36:28 'You know the quarantine procedure' -> 按照隔离规定来做.")],
    aliases=["隔离"],
    note="隔离 is the right word for isolating a PERSON; 检疫 for the legal/procedural quarantine.")
add("Decontamination", "净化除污", "T-PLAIN", "ops",
    "GAP in the corpus: 'decontamination' is folded into 隔离 and never rendered separately.",
    [("(none)", S, "en-gated 0 as a distinct term", "ALIEN1979 0:36:28 '24 hours for decontamination' -> 要隔离24小时.")])
add("Self-Destruct", "自毁", "T-PLAIN", "ops",
    "自动引爆 (en-gated 1) reads as an accidental auto-detonation; 自毁 is the standard term for a "
    "deliberate self-destruct system.",
    [("自动引爆", S, "en-gated 1", "ALIENS1986 0:11:43 'set for self-destruct by you' -> 却被你以不明原因自动引爆."),
     ("紧急毁灭系统", S, "cn_only 3", "ALIEN1979 1:34:04 — the ship's system, not the act.")],
    aliases=["自动引爆"])
add("Lifeboat", "救生船", "T-PLAIN", "ops", "en-gated 3, 2 films.",
    [("救生船", S, "en-gated 3 (regex 'lifeboat|life boat', 5 en rows)", "ALIENS1986 0:11:29 'The lifeboat's flight recorder' -> 救生船记录; RESURRECTION 0:43:50.")])
add("EEV", "逃生艇", "T-PLAIN", "ops", "Emergency Escape Vehicle. en-gated 6, ALIEN3, internally consistent.",
    [("逃生艇", S, "en-gated 6, ALIEN3", "0:06:23 'An E.E.V.'s come down' -> 一艘逃生艇坠落; 0:10:21 '337 model E.E.V.' -> 337型号紧急逃生艇.")],
    aliases=["紧急逃生艇"])
add("Escape Pod", "逃生舱", "T-PLAIN", "ops",
    "Distinct from the EEV; the ShipPanic14 'Abandon Ship' entry needs it and cn.json leaves that entry English.",
    [("逃生舱", STD, "en-gated 0", "en.json ALIENRPG.ShipPanic14 'You run to the nearest escape pod' — the whole ShipPanic7-15 block is untranslated in cn.json.")])
add("Flight Recorder", "航行记录器", "T-PLAIN", "ops", "en-gated 2, ALIEN3.",
    [("航行记录器", S, "en-gated 2", "ALIEN3 0:45:13, 0:56:59.")])
add("Hull", "船体", "T-PLAIN", "ops", "en-gated 3.",
    [("船体", S, "en-gated 3 (6 en rows)", "ALIEN1979 0:18:45 'Is the hull breached?' -> 船体破了吗.")])
add("Hatch", "舱门", "T-PLAIN", "ops", "Consistent across ALIEN1979.",
    [("舱门", S, "en-gated, ALIEN1979", "0:36:59 'Inner hatch opened' -> 内舱门开了; 1:11:35.")])
add("Air Duct", "通风道", "T-PLAIN", "ops", "The Alien franchise's signature location noun.",
    [("通风道", S, "en-gated, ALIEN1979", "1:08:34 'it's using the air ducts to move around' -> 利用通风道出没.")])
add("Infirmary", "医疗室", "T-PLAIN", "ops", "en-gated 6 across 2 films.",
    [("医疗室", S, "en-gated 6", "ALIEN1979 0:47:20; ALIEN3 0:11:40, 0:47:07, 1:02:35, 1:10:45."),
     ("医务室", S, "en-gated 1", "ALIEN1979 0:36:19.")])
add("Colony", "殖民地", "T-PLAIN", "ops", "en-gated 4.",
    [("殖民地", S, "en-gated 4 (regex '\\bcolony\\b|colonies', 7 en rows)", "ALIENS1986 0:22:40 'cofinanced that colony' -> 那个殖民地是我们出资支持的.")])
add("Colonist", "移民", "T-PLAIN", "ops", "en-gated 6. 殖民者 gets zero.",
    [("移民", S, "en-gated 6 (8 en rows)", "ALIENS1986 1:02:06 'Every colonist had one surgically implanted' -> 每个移民都有植入; 1:36:18 'the deaths of 157 colonists' -> 你害死157个移民."),
     ("殖民者", None, "en-gated 0", "Not used anywhere in the corpus.")])
add("Terraforming", "星球改造", "T-PLAIN", "ops",
    "The corpus splits 1-1 between 开垦计划 (reads as land reclamation) and 星球改造; the latter is both "
    "attested and the standard Chinese term.",
    [("星球改造", S + " + " + STD, "en-gated 1", "ALIENS1986 0:14:43 'Terraformers. Planet engineers.' -> 星球改造者 工程师."),
     ("开垦计划", S, "en-gated 1", "ALIENS1986 0:22:46 'a lot of terraforming' -> 广大的开垦计划.")])
add("Fusion Reactor", "核聚变反应堆", "T-PLAIN", "ops",
    "核子反应炉 (en-gated 1) is the Taiwan form; 核聚变反应堆 is the mainland standard and is more precise (fusion, not just nuclear).",
    [("核子反应炉", S, "en-gated 1 — Taiwan form", "ALIENS1986 1:08:49 'basically a big fusion reactor' -> 基地理论上是座核子反应炉.")])
add("Derelict", "废弃飞船", "T-PLAIN", "ops",
    "GAP: the subs collapse 'It was a derelict spacecraft. It was an alien ship.' into one clause and lose 'derelict'.",
    [("(none)", S, "en-gated 0", "ALIENS1986 0:12:31 -> 是外星船.")])
add("Cargo", "货物", "T-PLAIN", "ops", "en-gated across 2 films on the same log line.",
    [("货物", S, "en-gated 2", "ALIEN1979 1:52:10 / ALIEN3 2:18:36 'Cargo and ship destroyed' -> 货物和货船全都毁灭.")])
add("Salvage", "打捞", "T-PLAIN", "ops",
    "GAP: all three occurrences are erased or paraphrased in the corpus. 打捞 is the standard term for salvage at sea/in space.",
    [("(none)", S, "en-gated 0", "ALIENS1986 0:05:07 'there goes our salvage' -> 只找到她 各位 (erased); 0:07:11 'a deep-salvage team' -> dropped.")])

# =============================================================== T-FROZEN ===
FROZEN_TABLES = [
    ("Panic Table", "module/documents/actor.mjs:554 game.tables.getName(\"Panic Table\")"),
    ("Stress Response Table", "module/documents/actor.mjs:845"),
    ("Panic Response Table", "module/documents/actor.mjs:1067"),
    ("EV - Critical Injuries", "module/documents/actor.mjs:1812 — tries game.i18n.localize(\"ALIENRPG.EVCriticalInjuries\") first, then this literal"),
    ("Critical Injuries", "module/documents/actor.mjs:1816 (and :1815 via ALIENRPG.CriticalInjuries). ALSO module/helpers/rollTableData.mjs:27 filters folder children by name.startsWith(\"Critical Injuries\") with NO localize fallback — every child table must keep this English prefix"),
    ("Critical injuries", "module/documents/actor.mjs:1817 — lowercase-i variant, looked up separately"),
    ("Critical Injuries on Synthetics", "module/documents/actor.mjs:1830"),
    ("critical injuries on synthetics", "module/documents/actor.mjs:1830 — all-lowercase variant"),
    ("Spaceship Minor Component Damage", "module/documents/actor.mjs:1848"),
    ("Spaceship Major Component Damage", "module/documents/actor.mjs:1855"),
]
for name, cite in FROZEN_TABLES:
    add(name, name, "T-FROZEN", "frozen-rolltable",
        "RollTable looked up by English name at runtime. Renaming it in the Babele mapping makes the lookup "
        "return undefined and the roll silently does nothing.",
        [(name, "system source", "n/a", cite + " — re-derived this session.")])

FROZEN_FOLDERS = [
    ("Alien Creature Tables", "module/helpers/rollTableData.mjs:7 game.folders.contents.find(x => x.name === \"Alien Creature Tables\"); a rename makes `folder` undefined and the creature sheet throws on folder.contents"),
    ("Alien Mother Tables", "module/helpers/rollTableData.mjs:24 — same pattern"),
]
for name, cite in FROZEN_FOLDERS:
    add(name, name, "T-FROZEN", "frozen-folder",
        "Folder found by exact English name with no fallback. Renaming it throws.",
        [(name, "system source", "n/a", cite + " — re-derived this session.")])
add("Alien Tables", "Alien Tables", "T-FROZEN", "frozen-folder",
    "Conditionally frozen. module/apps/init.mjs:49 guards first-time setup with `!game.settings.get(moduleKey,\"imported\") "
    "&& game.user.isGM && !game.folders.getName(\"Alien Tables\")`. Once `imported` is true the folder check is "
    "short-circuited, so a rename is harmless on an established world — but it re-triggers the import on a fresh one. "
    "Kept frozen because the cost of being wrong is a duplicate full import.",
    [("Alien Tables", "system source", "n/a", "module/apps/init.mjs:49 — re-derived this session.")])

FROZEN_IMPORT = [
    ("Alien RPG System", "Adventure document name. systems/alienrpg/module/apps/init.mjs:9 `adventurePackName`, consumed at :77 `pack.getName(adventurePackName)` and by `pack.index.find(a => a.name === adventurePackName)` in ModuleImport/ReImport"),
    ("MU/TH/ER Instructions.", "JournalEntry name, INCLUDING the trailing period. systems/alienrpg/module/apps/init.mjs:14 `welcomeJournalEntry`, consumed at :81 `game.journal.getName(welcomeJournalEntry).show()`"),
    ("Alien Evolved Core Rules", "BOTH the Adventure name (modules/alien-evolved-corerules/module/init.js:9) AND the Scene name (:12 `sceneToActivate`, consumed at :86 and :114 `game.scenes.getName(sceneToActivate).activate()`)"),
    ("CORE RULES - HOW TO USE THIS MODULE", "JournalEntry name. modules/alien-evolved-corerules/module/init.js:11, consumed at :87 and :115"),
    ("Alien Evolved Starter Set", "BOTH the Adventure name (modules/alien-evolved-starterset/module/init.js:9) AND the Scene name (:12, consumed at :119 and :151)"),
    ("STARTER SET - HOW TO USE THIS MODULE", "JournalEntry name. modules/alien-evolved-starterset/module/init.js:11, consumed at :121 and :152"),
]
for name, cite in FROZEN_IMPORT:
    add(name, name, "T-FROZEN", "frozen-import-lookup",
        "First-run import lookup name. Translating it makes getName() return undefined and the import chain "
        "throws before the welcome journal opens.",
        [(name, "module source", "n/a", cite + " — re-derived this session.")])

add("PACK MULE", "PACK MULE", "T-FROZEN", "frozen-item-name",
    "Talent Item name matched in English at runtime; translating it silently disables the mechanic "
    "(Pack Mule doubles encumbrance).",
    [("PACK MULE", "system source", "n/a",
      "module/sheets/character-sheet.mjs:491 `if (i.name.toUpperCase() === \"PACK MULE\")`; also colony-sheet.mjs:339, synthetic-sheet.mjs:478.")])
add("TAKE CONTROL", "TAKE CONTROL", "T-FROZEN", "frozen-item-name",
    "Same pattern as PACK MULE.",
    [("TAKE CONTROL", "system source", "n/a",
      "module/data/actor-character.mjs:440 and module/data/actor-synthetic.mjs:430.")])
add("None", "None", "T-FROZEN", "frozen-sentinel",
    "The sentinel row of the creature-table and mother-table dropdowns is a hard-coded English literal, and "
    "ALIENRPG.None also participates in the crit-table heal-time equality switch. cn.json currently renders "
    "it 莫, which is both meaningless and breaks that switch.",
    [("None", "system source", "n/a",
      "module/helpers/rollTableData.mjs:12 and :30 `lTables[0] = { key: \"None\", label: \"None\" }`; the switch is at module/documents/actor.mjs:1922-1941. cn.json:266 \"None\": \"莫\" — re-derived this session.")])

# ============================================================== FILM TITLES ===
add("Alien (1979 film)", "异形", "T-BILINGUAL", "franchise-title", "zh-wiki 异形系列.",
    [("异形", W, "reference work", "zh.wikipedia 异形系列: 《异形》. Traditional-script regions write 異形.")])
add("Aliens (1986 film)", "异形2", "T-BILINGUAL", "franchise-title", "zh-wiki article title; HK uses 异形续集.",
    [("异形2", W, "reference work", "zh.wikipedia 异形2; the variant table shows 香港 异形续集.")])
add("Alien 3 (1992 film)", "异形3", "T-BILINGUAL", "franchise-title", "Series numbering, consistent with 异形2.", [])
add("Alien Resurrection (1997 film)", "异形4：浴火重生", "T-BILINGUAL", "franchise-title", "zh-wiki 异形系列 list.",
    [("异形4：浴火重生", W, "reference work", "zh.wikipedia 异形系列 simplified-Chinese title list.")])
add("Prometheus (2012 film)", "普罗米修斯", "T-BILINGUAL", "franchise-title", "zh-wiki 异形系列 list.", [])
add("Alien: Covenant (2017 film)", "异形：圣约", "T-BILINGUAL", "franchise-title", "zh-wiki 异形系列 list.", [])
add("Alien: Romulus (2024 film)", "异形：夺命舰", "T-BILINGUAL", "franchise-title",
    "The clearest documented three-way regional split in the franchise; 大陆 form chosen.",
    [("异形：夺命舰", W, "reference work", "zh.wikipedia 异形：夺命舰 lists 大陆 异形：夺命舰 / 台湾 异形：罗穆路斯 / 香港 异形：罗穆卢斯.")],
    aliases=["异形：罗穆路斯", "异形：罗穆卢斯"])
add("Alien: Earth (2025 series)", "异形：地球", "T-BILINGUAL", "franchise-title", "zh-wiki 异形系列 list.", [])
add("Special Order 937", "937号特别指令", "T-BILINGUAL", "franchise-vocab",
    "DIRECTLY ATTESTED on zh.wikipedia, contrary to the task brief's low-confidence flag. See pending.resolved_since_brief.",
    [("937号特别指令", W, "reference work; verbatim", "zh.wikipedia 异形 (电影): 「937号特别指令」."),
     ("特别指令", S, "en-gated 1, number dropped", "ALIEN1979 1:23:39 'What was your special order?' -> 你的特别指令是什么? The number 937 never appears in any of the 4 subtitle files.")])

# ---------------------------------------------------------------------------
def bilingual(en, cn):
    return f"{cn} {en}"


gloss = {}
prov = {}
collisions = {}
for e in E:
    en, cn = e["en"], e["cn"]
    if en in gloss and gloss[en] != cn:
        raise SystemExit(f"DUPLICATE KEY with different value: {en!r}")
    gloss[en] = cn
    p = {"cn": cn, "tier": e["tier"], "category": e["category"], "why_this_won": e["why"]}
    if e["candidates"]:
        p["candidates"] = e["candidates"]
    if e["aliases"]:
        p["aliases"] = e["aliases"]
    if e["dispute"]:
        p["dispute"] = e["dispute"]
    if e["note"]:
        p["note"] = e["note"]
    if e["tier"] == "T-BILINGUAL":
        p["bilingual_name"] = bilingual(en, cn)
    prov[en] = p
    collisions.setdefault(cn, []).append(en)

MULTI = {cn: ens for cn, ens in collisions.items() if len(ens) > 1}

TIERCOUNT = {}
CATCOUNT = {}
for e in E:
    TIERCOUNT[e["tier"]] = TIERCOUNT.get(e["tier"], 0) + 1
    CATCOUNT[e["category"]] = CATCOUNT.get(e["category"], 0) + 1

gloss = dict(sorted(gloss.items(), key=lambda kv: (kv[0].lower(), kv[0])))
prov_sorted = {k: prov[k] for k in gloss}

STAMP = "2026-08-29"

provenance = {
    "_meta": {
        "artifact": "glossary_alien.provenance.json",
        "version": "v0",
        "generated": STAMP,
        "count": len(gloss),
        "count_is_authoritative": True,
        "tier_counts": dict(sorted(TIERCOUNT.items())),
        "category_counts": dict(sorted(CATCOUNT.items())),
        "companion_files": {
            "glossary": "glossary_alien.json (flat {en: cn} — the only shape consumers read)",
            "disputes": "glossary_alien.disputes.json",
            "pending": "glossary_alien.pending.json",
        },
        "value_shape": (
            "glossary_alien.json values are BARE Chinese terminology, never a bilingual tail — EXCEPT "
            "T-FROZEN entries, whose value is the byte-exact English because that surface must not be "
            "translated at all. For T-BILINGUAL entries the ready-made name-field string is in "
            "provenance.terms[<en>].bilingual_name ('中文 English', ONE ASCII space, no parentheses). "
            "This differs from glossary_ec.json, which bakes the tail into the value because it was "
            "harvested from shipped name fields; this file is an authored term spine, not a harvest."
        ),
        "tiers": {
            "T-FROZEN": "English, byte-exact. Import lookup names, hard-coded RollTable and Folder names, "
                        "PACK MULE / TAKE CONTROL, the 'None' sentinel. Translating these breaks mechanics.",
            "T-EXACT": "Pure Chinese, byte-equal to a lang key. The 12 skill-stunts Item names must equal "
                       "lang/cn.json's ALIENRPG.Skill<key> exactly, with NO English tail.",
            "T-BILINGUAL": "'中文 English' separated by ONE ASCII space (no parentheses) on name / page-name "
                           "fields: proper nouns, gear, talents, planets, tables.",
            "T-PLAIN": "Bare Chinese in prose and inside {label} text.",
        },
        "source_priority": [
            "1. Chinese reference work (zh.wikipedia > 百度百科 > 萌娘百科) — highest for proper nouns.",
            "2. lang/cn.json STRATUM A (TI-130, human, 2020-11..2021-01), English-gated. Attributes, "
            "skills, careers, ranges, the whole Panic7-15 table.",
            "3. Standard modern Chinese technical / TRPG usage — only for COMMON nouns, never for a proper noun.",
            "4. CnSCG subtitle corpus — ONE vote, and only where 1-3 are silent AND the rendering is neither "
            "Taiwan-flavoured nor an outright error.",
            "5. Otherwise -> glossary_alien.pending.json. Never guess a proper noun.",
        ],
        "hard_rules": [
            "The subtitle corpus is ONE vote, not four. All four films are CnSCG (圣城家园) releases from a "
            "single translation lineage; cross-film agreement is shared provenance, not corroboration.",
            "STRATUM B of cn.json (maintainer bulk MT passes 2022-2026) is never a source. Where a term is "
            "attested only in stratum B it is recorded as a candidate and marked REJECTED or superseded.",
            "No conclusion rests on a bare Chinese frequency count. Every count written 'en-gated N' was "
            "produced by term_gate.py, which splits the count by what the paired English at the same "
            "position actually says.",
            "EC terminology was NOT merged in. Different world; EC's own hard constraint forbids "
            "cross-project glossary merging. Only the SHAPE of glossary_ec.* was consulted.",
        ],
        "gate_tool": r"C:\Users\Taka\Desktop\fvtt\Alien-RPG Translation Project\4-常用脚本\tm\term_gate.py",
        "gate_commands": [
            r'python term_gate.py --mode lang --src "C:/Users/Taka/AppData/Local/FoundryVTT/Data/systems/alienrpg/lang" --en "<regex>" --cn "<zh1,zh2>"',
            r'python term_gate.py --mode subs --src "C:/Users/Taka/Desktop/fvtt/AlienRPG" --en "<regex>" --cn "<zh1,zh2>" --ignore-case --by-source',
            r'python term_gate.py --mode compendium --src "<repoDir>" --en "<regex>" --cn "<zh1,zh2>"   # once the packs are extracted',
        ],
        "corpora": {
            "lang": "C:/Users/Taka/AppData/Local/FoundryVTT/Data/systems/alienrpg/lang/{en,cn}.json — 590 en leaves, 511 cn keys.",
            "subs": "C:/Users/Taka/Desktop/fvtt/AlienRPG/*.ssa — 4,665 deduped bilingual Dialogue events across 4 films.",
        },
        "lockstep_literals": {
            "_why": "module/documents/actor.mjs parses Critical-Injury RollTable output by STRING EQUALITY "
                    "against localized values. These are not glossary terms; they are content/lang pairs that "
                    "must move together or healTime silently resolves to 0 on Chinese play.",
            "Shift": "actor.mjs:1888 `testArray[9] === \"Shift\"` — a BARE English literal with no localize() "
                     "call. The heal-time cell must stay English in the table content, even though the lang "
                     "key ALIENRPG.Shift may be Chinese. Today cn.json leaves ALIENRPG.Shift English, so this happens to work.",
            "Yes": "actor.mjs:1906 and :1912 `case game.i18n.localize(\"ALIENRPG.Yes\") + \", –1 \":` — note the "
                   "separator uses U+2013 EN DASH, not ASCII hyphen-minus. Verified by hexdump. Any lockstep "
                   "table built with an ASCII hyphen can never match and silently loses the cFatal=true branches.",
            "Permanent": "actor.mjs:1888 `testArray[9] !== game.i18n.localize(\"ALIENRPG.Permanent\")`.",
            "None / One Round / One Turn / One Shift / One Day": "actor.mjs:1922-1941 switch on "
                    "localize('ALIENRPG.None'|'OneRound'|'OneTurn'|'OneShift'|'OneDay') + ' '. cn.json today "
                    "translates these four while leaving Yes/Permanent/Shift English — an inconsistent state "
                    "that already breaks healTime.",
        },
        "corrections_to_the_survey": [
            "terminology-web.json:88 records Facehugger candidates 抱脸体/抱面体/抱脸虫 but MISSES a fourth that "
            "is already in the file: cn.json renders facehugger as 抱脸者 (en-gated 2 — ALIENRPG.AutoPanicHint "
            "and ALIENRPG.TAH.radiation), while ALIENRPG.hideChatBGImageNote leaves the English word untranslated.",
            "The '13 hits' cited for Synthetic=生化人 is a BARE Chinese count. English-gated, 生化人 renders "
            "'synthetic' 1 time out of 5 and android/droid 9 times out of 11 — it is the corpus's android word, "
            "not its synthetic word.",
            "Queen: the survey's 女王 7 / 王后 4 are bare counts. English-gated they are 4 / 3, and one of the "
            "four 女王 hits is the bee-metaphor 女王蜂 (ALIENS1986 1:35:28).",
            "Weyland-Yutani: the survey says 2 subtitle occurrences. English-gated the corpus casts ONE vote — "
            "the ALIEN3 0:09:09 hit is on-screen text whose English side is empty, so it is cn_only.",
            "Ripley: survey says 蕾普丽 82 / 蕾普莉 12 (bare). English-gated: 75 / 11.",
            "Quarantine: the survey calls 隔离 and 检疫 'equally attested'. English-gated it is 检疫 4 vs 隔离 2.",
            "The Evolved condition list in terminology-web.json's Q1 note includes 'fatigued'. The live list at "
            "module/helpers/config.mjs ALIENRPG.conditions has 20 entries and 'fatigued' is NOT one of them — "
            "the 20th is 'messup' (Mess Up). ALIENRPG.fatigued exists as a key but is not a condition.",
            "system-i18n.json hardcoded_sites[49] says renaming the folder 'Alien Tables' re-triggers the "
            "first-time import. On an established world it does not: init.mjs:49 short-circuits on the "
            "`imported` setting before the folder check. verify-mapping.json is right that only 'Alien Creature "
            "Tables' and 'Alien Mother Tables' are unconditionally frozen. 'Alien Tables' is kept frozen here "
            "anyway, because the failure mode on a fresh world is a duplicate full import.",
            "All cn.json line numbers quoted in terminology-web.json were re-derived this session and every one "
            "checked out (419 Stress, 157 Engaged, 186 GMONLY, 290 Overwatch, 205 HOWMANYDICE, 370 SelectFirer, "
            "479 Wildcatter, 86 ColonialMarshal, 23 AcidAttack, 436 Synthetic, 428 Supply, 390 SignatureItem, "
            "337/338 relOne/relTwo, 418 StoryPoints, 327 Push, 76 Career, 21/19 AbilityWit/AbilityEmp, "
            "302/303 PanicCondition/Panicked, 293-301 Panic7-15).",
        ],
        "chinese_value_collisions": {
            "_why": "Two English terms mapping to one Chinese string. Harmless when the two live on different "
                    "axes (e.g. Long range vs Ranged weapon type); dangerous inside one dropdown.",
            "collisions": {cn: ens for cn, ens in sorted(MULTI.items())},
        },
        "frozen_count_reconciliation": {
            "_why": "The owner's settled decision 4 names '9 first-run import lookup names, 11 hard-coded "
                    "RollTable names, 2 hard-coded Folder names'. Re-deriving from source this session gives "
                    "slightly different DISTINCT-STRING counts. Recorded rather than silently reconciled.",
            "import_lookup": "9 lookup SITES over 6 DISTINCT strings, because 'Alien Evolved Core Rules' and "
                             "'Alien Evolved Starter Set' each name BOTH an Adventure and a Scene. Sites: "
                             "systems/alienrpg/module/apps/init.mjs:9 (Adventure), :14 (Journal), :49 (Folder "
                             "'Alien Tables'); modules/alien-evolved-corerules/module/init.js:9 (Adventure), "
                             ":11 (Journal), :12 (Scene); modules/alien-evolved-starterset/module/init.js:9, "
                             ":11, :12. The folder 'Alien Tables' is filed under frozen-folder here, so this "
                             "file's frozen-import-lookup category holds 6 entries.",
            "rolltable": "10 distinct string literals, not 11 — module/documents/actor.mjs :554, :845, :1067, "
                         ":1812, :1816, :1817, :1830 (x2 case variants), :1848, :1855. The 11th in the survey is "
                         "most likely 'Space Combat Panic Roll' at module/actor/old-actor.js:397, which is NOT "
                         "in the live import graph (verify-i18n.json establishes the graph is 62 files, of which "
                         "the only two .js are apps/update.js and devmsg.js). It is therefore dead and excluded.",
            "folder": "3 recorded, of which 2 are unconditionally frozen ('Alien Creature Tables', 'Alien Mother "
                      "Tables' — rollTableData.mjs:7 and :24 use .find() with no fallback and throw on a rename). "
                      "'Alien Tables' is the third and is only conditionally frozen; see its entry.",
        },
        "known_gaps": [
            "The 3 content packs are not yet extracted (compendium/en is empty in all three modules), so no "
            "compendium-mode gate could be run. Every gate in this file is lang-mode or subs-mode. Re-run the "
            "disputed terms in compendium mode once the packs are dumped.",
            "純美蘋果園 topic=121082.0 (a 2021 Chinese translation of the Alien RPG stealth rules) is behind a "
            "login wall and was not read. It is the only other Chinese Alien RPG text known to exist and could "
            "overturn several mechanics choices.",
            "bilibili (cv18100844, cv2014315) and 萌娘百科 both block automated fetch, so the fan-corpus side of "
            "dispute D7 rests on the survey's own reading, not on a re-derived citation.",
        ],
    },
    "terms": prov_sorted,
}

# ------------------------------------------------------------------ disputes
disputes = {
    "_meta": {
        "artifact": "glossary_alien.disputes.json",
        "version": "v0",
        "generated": STAMP,
        "count": 7,
        "count_is_authoritative": True,
        "note": "Genuine open conflicts. glossary_alien.json carries a PROVISIONAL winner for each so the "
                "file stays usable; that provisional value is NOT a decision. Each entry names what would "
                "settle it. Resolve before the first public release — every one of these is a term players "
                "will see on every session.",
        "provisional_values_live_in": "glossary_alien.json; provenance.terms[<en>].dispute points back here.",
    },
    "disputes": {
        "D1-panic-roll": {
            "en": "Panic Roll",
            "provisional_cn": "恐慌检定",
            "tier": "T-PLAIN",
            "why_open": "cn.json disagrees with itself: its UI keys use 恐慌 for Panic while its human-translated "
                        "panic-table prose renders the phrase 'Panic Roll' as 混乱检定. One of the two has to give.",
            "sides": [
                {
                    "cn": "恐慌检定",
                    "argument": "Regular. Panic = 恐慌 is settled by ALIENRPG.Panicked and ALIENRPG.PanicCondition, "
                                "and 检定 is the stratum-A roll word. Keeps Panic / Panicked / Panic Condition / "
                                "Panic Roll / Panic Response one family. 混乱 means 'chaos, disorder' and does not "
                                "mean panic.",
                    "sources": ["derived from stratum A (恐慌 + 检定)"],
                    "gated_count": "en-gated 0 for the phrase; 恐慌 en-gated 1 + 11 sibling keys; 检定 en-gated 10",
                    "evidence": "cn.json:303 \"Panicked\": \"恐慌\"; cn.json:302 \"PanicCondition\": \"恐慌状态\"; "
                                "ALIENRPG.RollPanic -> \"点击滚动恐慌\"; ALIENRPG.MorePanic -> \"更加恐慌\".",
                },
                {
                    "cn": "混乱检定",
                    "argument": "It is the ONLY attested rendering of the phrase, and it is in stratum A — the good "
                                "human work the project is otherwise told to mine. Changing it edits the best "
                                "translator's deliberate choice on the basis of a rule, not a source.",
                    "sources": ["cn.json stratum A"],
                    "gated_count": "en-gated 3 of 7 rows matching '[Pp]anic [Rr]oll'",
                    "evidence": "cn.json:295 Panic12 \"...必须立刻进行混乱检定。\"; cn.json:296 Panic13 same; "
                                "cn.json:297 Panic14 same. Also cn.json:300 Panic8 \"直到混乱结束\" for 'until your panic stops'.",
                },
            ],
            "what_would_settle_it": "The 純美蘋果園 2021 Alien RPG stealth-rules translation (topic=121082.0) is by "
                                    "a different hand and would show whether 混乱 was a house choice or an outlier. "
                                    "Failing that: an owner ruling. Note that whichever wins must also be applied "
                                    "to Panic8's \"直到混乱结束\".",
            "blast_radius": "Panic12, Panic13, Panic14, Panic8 prose + every Evolved Panic Response string.",
        },
        "D2-engaged": {
            "en": "Engaged",
            "provisional_cn": "接战",
            "tier": "T-PLAIN",
            "why_open": "cn.json uses ONE string, 近战, for two different game terms: the Engaged range band and "
                        "the Melee weapon type. In the weapon-type dropdown and the range dropdown the player "
                        "sees the same word for two unrelated things.",
            "sides": [
                {
                    "cn": "接战",
                    "argument": "Removes the collision and is the stratum-A prose choice. 接战 ('in contact') is "
                                "also the better description of the band: Engaged is not about melee weapons.",
                    "sources": ["cn.json stratum A (panic prose)"],
                    "gated_count": "en-gated 2 of 3 rows matching 'ENGAGED|Engaged'; en-gated 0 on '\\bMelee\\b'",
                    "evidence": "cn.json:294 Panic11 \"如果你的接战范围内有敌人，你可以进行撤退检定（见P93）\"; "
                                "cn.json:296 Panic13 \"若逃跑时你的接战范围内有敌人\".",
                },
                {
                    "cn": "近战",
                    "argument": "It is what the UI dropdown says today, so it is what any existing Chinese-speaking "
                                "table already reads. Changing it is a visible break.",
                    "sources": ["cn.json (UI key)"],
                    "gated_count": "en-gated 1 on 'ENGAGED|Engaged' AND en-gated 1 on '\\bMelee\\b' — the collision itself",
                    "evidence": "cn.json:157 \"Engaged\": \"近战\" and cn.json:476 \"WepTypeMelee\": \"近战\" — "
                                "both re-derived this session.",
                },
            ],
            "what_would_settle_it": "Nearly settled: 接战 wins on evidence. Left open only because it changes a "
                                    "dropdown label a player may already know. Owner ruling.",
            "blast_radius": "ALIENRPG.Engaged, weapon_range_list, evolved_weapon_range_list, vehicle range lists, "
                            "and all range prose.",
        },
        "D3-stress-level": {
            "en": "Stress Level",
            "provisional_cn": "压力水平",
            "tier": "T-PLAIN",
            "why_open": "Both renderings are in cn.json, both in stratum A, and both in stratum B. There is no "
                        "clean stratum split to arbitrate on — only a count.",
            "sides": [
                {
                    "cn": "压力水平",
                    "argument": "Majority in both strata. Reads as a continuous level, which matches a track that "
                                "goes up and down by one.",
                    "sources": ["cn.json stratum A", "cn.json stratum B"],
                    "gated_count": "en-gated 7 of 14 rows matching 'STRESS LEVEL|[Ss]tress [Ll]evel' — 5 stratum A + 2 stratum B",
                    "evidence": "Stratum A: cn.json:301 Panic9, :293 Panic10, :294 Panic11, :295 Panic12, :296 "
                                "Panic13. Stratum B: TAH.dehydrated, TAH.starving.",
                },
                {
                    "cn": "压力等级",
                    "argument": "Stress Level is a discrete integer step scale (0-10), and 等级 is the Chinese word "
                                "for a discrete grade. cn.json already uses 等级 for the sibling quantity: "
                                "ALIENRPG.PCPanicLevel renders 'Panic level' as 恐慌等级. Choosing 水平 here makes "
                                "the two sibling tracks inconsistent with each other.",
                    "sources": ["cn.json stratum A (Panic7)", "cn.json stratum B (TAH.freezing)"],
                    "gated_count": "en-gated 2 of 14 — 1 stratum A + 1 stratum B",
                    "evidence": "cn.json:299 Panic7 \"你的压力等级，以及短距离内所有友善角色的压力等级上升1。\"; "
                                "TAH.freezing \"你的压力等级会增加一级\". Sibling: ALIENRPG.PCPanicLevel -> "
                                "\"角色的恐慌等级已经提升了一级\".",
                },
            ],
            "what_would_settle_it": "A ruling on whether Stress Level and Panic Level must use the same head noun. "
                                    "If yes, 压力等级 wins on consistency despite losing 7-2 on count.",
            "blast_radius": "Panic7 and Panic9-13, the 5 untranslated ShipPanic entries, and every Evolved Stress "
                            "Response string.",
        },
        "D4-health": {
            "en": "Health",
            "provisional_cn": "生命",
            "tier": "T-PLAIN",
            "why_open": "Three renderings coexist in one file. 健康 actually has the higher English-gated count, "
                        "so the provisional winner is chosen on stratum, not on count.",
            "sides": [
                {
                    "cn": "生命",
                    "argument": "The only rendering on the canonical UI key. In a game where Health is a "
                                "damage track that reaching zero means Broken, 生命 (life) is the TRPG-conventional "
                                "head noun.",
                    "sources": ["cn.json ALIENRPG.Health"],
                    "gated_count": "en-gated 3 of 6 rows matching '\\b[Hh]ealth\\b|HEALTH' — 1 canonical key + 2 stratum-B tooltips",
                    "evidence": "cn.json:193 \"Health\": \"生命\" — re-derived this session.",
                },
                {
                    "cn": "健康",
                    "argument": "Highest gated count, and it is what the damage message says: ALIENRPG.healthDamage "
                                "' HEALTH Damage' -> ' 健康损害' is a UI key, not a tooltip.",
                    "sources": ["cn.json stratum B"],
                    "gated_count": "en-gated 4 of 6 — ALL stratum B",
                    "evidence": "ALIENRPG.healthDamage; TAH.dehydrated / TAH.freezing / TAH.starving \"您无法恢复健康\".",
                },
                {
                    "cn": "生命值",
                    "argument": "Unambiguous as a numeric stat; the most common Chinese CRPG/TRPG form.",
                    "sources": ["cn.json stratum B"],
                    "gated_count": "en-gated 2 of 6 — both stratum-B tooltips",
                    "evidence": "TAH.freezing \"可以恢复生命值\"; TAH.radiation \"你就无法恢复生命值\".",
                },
            ],
            "what_would_settle_it": "Owner ruling. Note the choice must also fix ALIENRPG.healthDamage, which is a "
                                    "UI key and is the one place a player reads the word during play.",
            "blast_radius": "ALIENRPG.Health, ALIENRPG.healthDamage, the 4 TAH condition tooltips, character and "
                            "synthetic sheet headers.",
        },
        "D5-synthetic": {
            "en": "Synthetic",
            "provisional_cn": "生化人",
            "tier": "T-PLAIN",
            "why_open": "The system file and the film reference works disagree, and the subtitle corpus — cited by "
                        "the survey as supporting 生化人 with 13 hits — does not actually support it once the count "
                        "is English-gated.",
            "sides": [
                {
                    "cn": "生化人",
                    "argument": "It is what cn.json ships, including a full sentence of stratum-A prose, so every "
                                "existing Chinese Alien RPG table reads it. Continuity argument.",
                    "sources": ["cn.json", "CnSCG subs (as the ANDROID word)"],
                    "gated_count": "cn.json: en-gated 1. Subs: en-gated 1 of 5 rows matching '\\bsynthetic' — but "
                                   "en-gated 9 of 11 rows matching '\\bandroid|\\bdroid\\b'.",
                    "evidence": "cn.json:436 \"Synthetic\": \"生化人\"; ALIENRPG.SynthDontNeed -> "
                                "\"生化人无需空气，食物，水分或是休眠。\". Subs gated hit: ALIENS1986 0:32:03 'We always "
                                "have a synthetic on board' -> 船上一向都有生化人. The other 11 occurrences of 生化人 "
                                "in the corpus pair with android / droid / robot, NOT with synthetic. "
                                "Literally: 生化人 = 'bio-chemical human', which is closer to a cyborg than to an "
                                "artificial person.",
                },
                {
                    "cn": "仿生人",
                    "argument": "Every Chinese film reference work uses it, for Bishop and for the Romulus synthetic. "
                                "It is the wiki-formal spine the project has otherwise committed to, and it is "
                                "literally right (仿生 = biomimetic).",
                    "sources": ["zh.wikipedia 异形2, 异形：夺命舰"],
                    "gated_count": "reference work, no count. en-gated 0 in the subs and 0 in cn.json.",
                    "evidence": "terminology-web.json:50 records zh.wikipedia's Alien film articles using 仿生人 "
                                "consistently. NOT re-derivable this session: WebFetch was not available.",
                },
                {
                    "cn": "人造人",
                    "argument": "The in-universe euphemism Bishop himself uses ('I prefer the term artificial "
                                "person'), and the subs render BOTH that line and one 'synthetic' line this way. "
                                "Adopting it would make Synthetic and Artificial Person the same word, which is "
                                "arguably what the fiction intends.",
                    "sources": ["CnSCG subs"],
                    "gated_count": "en-gated 1 of 5 on '\\bsynthetic'; en-gated 2 on 'artificial person'",
                    "evidence": "ALIENS1986 1:40:23 'I may be synthetic, but I'm not stupid' -> 虽然我是人造人 我可不笨; "
                                "0:32:07 'I prefer the term \"artificial person\" myself' -> 我比较喜欢被称为\"人造人\".",
                },
                {
                    "cn": "合成人",
                    "argument": "The most literal rendering of 'synthetic' and the one the corpus uses for the "
                                "industry term.",
                    "sources": ["CnSCG subs"],
                    "gated_count": "en-gated 1 of 5",
                    "evidence": "RESURRECTION 1:20:47 'they were supposed to revitalize the synthetic industry' -> "
                                "本来预期复兴合成人工业.",
                },
            ],
            "what_would_settle_it": "A ruling on whether the wiki-formal spine outranks cn.json continuity for a "
                                    "term that is BOTH a franchise noun and a playable Career. The Facehugger "
                                    "exception shows the owner is willing to break the spine for the dominant "
                                    "mainland form — here the dominant mainland form (仿生人) is the spine, and it "
                                    "is cn.json that is the outlier.",
            "blast_radius": "The Synthetic Career, the synthetic Actor type and both its sheets, ALIENRPG.Synthetic, "
                            "ALIENRPG.SynthDontNeed, ALIENRPG.SynthStress, ALIENRPG.NoSynCrit, the 'Critical "
                            "Injuries on Synthetics' table's page prose, and every Bishop/Ash/Call NPC.",
        },
        "D6-queen": {
            "en": "Queen",
            "provisional_cn": "女王",
            "tier": "T-BILINGUAL",
            "why_open": "The corpus splits by film, and English-gating narrows the margin to 4-3. zh.wikipedia "
                        "breaks the tie for the compound (异形女王) but its own 异形2 article uses a third form.",
            "sides": [
                {
                    "cn": "女王",
                    "argument": "Wins on the reference work and on the gated count, and it is the form in the "
                                "compound the RPG actually needs (Alien Queen -> 异形女王).",
                    "sources": ["zh.wikipedia 异形 (虚构生物)", "CnSCG subs (ALIEN3 + ALIENS1986)"],
                    "gated_count": "en-gated 4 of 7 rows matching '\\bqueen\\b' — ALIEN3 3, ALIENS1986 1. One of "
                                   "those 4 is the bee metaphor 女王蜂, so the plain-noun count is really 3.",
                    "evidence": "zh.wikipedia 异形 (虚构生物): 「異形女王」（Alien Queen）是一個異形族群的領導者. "
                                "ALIEN3 1:41:18 'I'm carrying the new queen' -> 我孕育新女王; 1:46:49 'It's a queen- "
                                "an egg layer' -> 它是个女王 会产卵. ALIENS1986 1:35:28 -> 女王蜂 (bee-queen metaphor).",
                },
                {
                    "cn": "王后",
                    "argument": "Uniform across the whole of Alien Resurrection, which is the film with the most "
                                "Queen dialogue; and the corpus also produces 异形王后 for 'Her Majesty'. If the "
                                "project ever wanted the most recent film's register, this is it.",
                    "sources": ["CnSCG subs (RESURRECTION)"],
                    "gated_count": "en-gated 3 of 7 — all RESURRECTION",
                    "evidence": "RESURRECTION 0:13:09 'It's a queen.' -> 是王后; 1:29:58 -> 是王后; 1:34:19 'The "
                                "queen laid her eggs' -> 王后产卵. Plus cn_only 0:11:27 'Her Majesty here is the "
                                "real payoff' -> 异形王后才值回票价.",
                },
            ],
            "what_would_settle_it": "Effectively settled by zh.wikipedia for the compound. Left open because "
                                    "the bare noun appears in creature stat blocks and a mixed 异形女王 / 王后 file "
                                    "would be worse than either choice consistently applied.",
            "blast_radius": "The Queen Actor(s) in the core-rules pack, Alien Queen, and hive prose.",
        },
        "D7-weyland-yutani": {
            "en": "Weyland-Yutani",
            "provisional_cn": "韦兰-尤坦尼集团",
            "tier": "T-BILINGUAL",
            "why_open": "Three renderings from three registers, and the only one attested in this project's own "
                        "corpus is the one nobody proposes.",
            "sides": [
                {
                    "cn": "韦兰-尤坦尼集团",
                    "argument": "zh.wikipedia's article title and the form used in the 异形 / 异形2 / 异形：夺命舰 "
                                "articles. It is the owner's declared wiki-formal spine. 集团 already carries "
                                "'Corporation', so 韦兰-尤坦尼集团公司 would be wrong.",
                    "sources": ["zh.wikipedia (article title)"],
                    "gated_count": "reference work, no count. en-gated 0 in the subtitle corpus.",
                    "evidence": "terminology-web.json:95. The same article's LEAD says 韦兰德-尤坦尼集团, with "
                                "traditional variant 韋蘭德-湯谷企業 and alternate 韦兰德-汤谷公司 — zh-wiki is "
                                "internally inconsistent between its title and its lead. NOT re-derivable this "
                                "session: WebFetch was not available.",
                },
                {
                    "cn": "维兰德-汤谷公司",
                    "argument": "Overwhelmingly the Chinese fan-corpus form (bilibili, douban, 机核), with the short "
                                "form 维汤. If the glossary targets players rather than encyclopaedia readers, this "
                                "is the string they will recognise.",
                    "sources": ["Chinese fan corpus"],
                    "gated_count": "no count — bilibili serves a captcha to curl and an empty SPA shell to "
                                   "WebFetch, so the fan corpus could not be counted this session.",
                    "evidence": "terminology-web.json's Q3 register note. Unverified count is the weakest point of "
                                "this side.",
                },
                {
                    "cn": "伟伦优达尼公司",
                    "argument": "The ONLY form attested anywhere in this project's own 4,665-line bilingual corpus.",
                    "sources": ["CnSCG subs"],
                    "gated_count": "en-gated 1 of 2 rows matching 'Weyland|Yutani'",
                    "evidence": "RESURRECTION 0:13:25 'Weyland-Yutani. Ripley 8's former employers.' -> "
                                "伟伦优达尼公司 蕾普丽8号生前的雇主. The ALIEN3 0:09:09 occurrence is on-screen comms "
                                "text with an EMPTY English side, so it does not gate. ALIEN3 0:13:05, which does "
                                "have the English, drops the name entirely (-> 银河系最偏远的劳动监狱). "
                                "伟伦优达尼 is Taiwan-flavoured phonetics and is not the mainland-common form.",
                },
            ],
            "what_would_settle_it": "Read bilibili cv18100844 / cv2014315 through a browser to put a real number on "
                                    "the fan-corpus side, and re-read the zh-wiki corp article to decide between "
                                    "its own title (韦兰-尤坦尼) and its lead (韦兰德-尤坦尼). Until then the owner's "
                                    "declared spine stands.",
            "blast_radius": "Every 'The Company' journal page, the Company Agent career, ship registries, the "
                            "Building Better Worlds slogan, and most starter-set NPC backgrounds.",
        },
    },
}

# ------------------------------------------------------------------- pending
pending = {
    "_meta": {
        "artifact": "glossary_alien.pending.json",
        "version": "v0",
        "generated": STAMP,
        "count": 12,
        "count_is_authoritative": True,
        "actionable_count": 12,
        "resolved_since_brief_count": 1,
        "note": "Terms that could NOT be sourced to a reference work and MUST NOT be guessed. They are "
                "deliberately ABSENT from glossary_alien.json: an absent key makes a translator stop and ask, "
                "whereas a plausible-looking guess propagates silently. Every proper noun here needs a human "
                "decision or a reachable source before it enters the glossary.",
        "rule": "Standard modern Chinese usage may settle a COMMON noun (that is why Air Lock and Salvage are "
                "in the glossary and not here). It can never settle a PROPER noun — a place, a ship, or a "
                "model designation.",
    },
    "terms": {
        "Acheron": {
            "kind": "proper noun — planet name (LV-426's colonial name)",
            "candidates": [
                {"zh": "阿克伦", "source": "survey proposal", "count": "en-gated 0",
                 "note": "Could not be found in zh.wikipedia's Alien articles or in cn.json."},
                {"zh": "阿刻戎 / 阿克戎", "source": "standard Chinese for the Greek river Acheron", "count": "n/a",
                 "note": "The conventional rendering of the mythological river, which argues AGAINST 阿克伦."},
            ],
            "corpus": "en-gated 0 in the subtitle corpus — the planet is only ever called LV426 there.",
            "why_unresolved": "Two plausible transliterations and no Alien-specific attestation for either.",
            "next_step": "Check whether the 异形2 zh-wiki article names the colony's planet at all; otherwise "
                         "decide between the mythological 阿刻戎 and the phonetic 阿克伦.",
        },
        "Hadley's Hope": {
            "kind": "proper noun — colony name",
            "candidates": [
                {"zh": "哈德利希望镇", "source": "Chinese fan writing", "count": "unverified",
                 "note": "Could not be confirmed against a reference work."},
                {"zh": "希望之地", "source": "zh.wikipedia 异形2 plot summary", "count": "n/a",
                 "note": "A loose paraphrase, not a transliteration. Wrong as a place name — it drops Hadley entirely."},
            ],
            "corpus": "en-gated 0 — the subs refer to it generically as 殖民地 / 移民.",
            "why_unresolved": "The only reference-work rendering is a paraphrase that loses the proper noun.",
            "next_step": "Owner ruling on a transliteration pattern for possessive place names (哈德利希望 vs "
                         "哈德利希望镇 vs 哈德利之希望), then apply it consistently.",
        },
        "Sulaco": {
            "kind": "proper noun — ship name",
            "candidates": [
                {"zh": "苏拉可号", "source": "CnSCG subs", "count": "en-gated 2 of 3 rows matching 'Sulaco', ALIEN3 only",
                 "note": "ALIEN3 0:57:35 'What happened on the Sulaco?' -> 苏拉可号发生什么事?; 0:58:24. ALIENS1986 1:39:28 drops the name."},
                {"zh": "苏拉克号", "source": "zh.wikipedia 异形2 plot summary", "count": "n/a", "note": "The reference work's form."},
                {"zh": "苏拉科号", "source": "Chinese fan writing", "count": "unverified",
                 "note": "The most common fan form; could not be sourced to a reference work this session."},
            ],
            "corpus": "en-gated 2 for 苏拉可号; 苏拉克号 and 苏拉科号 both en-gated 0.",
            "why_unresolved": "Three transliterations, one per register, and the reference work and the corpus "
                              "disagree. Same shape as the Nostromo problem, which the owner settled by decree "
                              "(诺史莫号) — Sulaco has no such decree.",
            "next_step": "Owner ruling, ideally the same way Nostromo was settled: take zh-wiki's 苏拉克号 for "
                         "consistency with the 诺史莫号 decision.",
        },
        "Smartgun": {
            "kind": "proper noun — weapon model (M56 Smartgun)",
            "candidates": [
                {"zh": "智能枪", "source": "survey proposal", "count": "en-gated 0", "note": "Plausible but no Chinese reference-work entry was found."},
                {"zh": "机关枪", "source": "CnSCG subs", "count": "en-gated 1 — REJECTED",
                 "note": "ALIEN3 1:25:34 'go in there with smart guns and kill it' -> 拿出机关枪 轰死它. A flattening: 'smart' is lost entirely."},
            ],
            "corpus": "One occurrence, badly rendered.",
            "why_unresolved": "萌娘百科 has an M41A article but 403s to automated fetch, so the sibling M56 entry "
                              "(if it exists) could not be read — and that is exactly the source that settled Pulse Rifle.",
            "next_step": "Open zh.moegirl.org.cn in a browser and search for M56; if 萌娘百科 names it, adopt that "
                         "form the way M41A脉冲步枪 was adopted.",
        },
        "Motion Tracker": {
            "kind": "gear name",
            "candidates": [
                {"zh": "行动追踪器", "source": "CnSCG subs", "count": "en-gated 1 of 6 rows matching 'motion tracker|\\btracker'",
                 "note": "ALIENS1986 0:49:05 'use your motion trackers' -> 用你们的行动追踪器. The short form 追踪器 is en-gated 3 across 2 films."},
                {"zh": "运动探测器", "source": "survey proposal", "count": "en-gated 0", "note": "Could not be pinned to an exact zh-wiki string."},
            ],
            "corpus": "The corpus DOES attest 行动追踪器, but only once and in the Taiwan-flavoured ALIENS1986 track.",
            "why_unresolved": "A single subtitle hit is one vote from one lineage; the Motion Tracker is the "
                              "franchise's second-most-recognisable prop and deserves a reference-work check first.",
            "next_step": "Check 萌娘百科 / 百度百科 for a Motion Tracker entry. If nothing exists, 运动探测器 "
                         "(technical) and 行动追踪器 (attested) are both defensible — owner ruling.",
        },
        "Atmosphere Processor": {
            "kind": "proper noun — installation type",
            "candidates": [
                {"zh": "大气处理厂", "source": "survey proposal", "count": "en-gated 0", "note": "Unattested in any reachable source."},
                {"zh": "大气处理机", "source": "CnSCG subs", "count": "en-gated 1 of 4",
                 "note": "ALIENS1986 0:44:45 'That's the atmosphere processor?' -> 那就是大气处理机? But the SAME film also says 大气工程 (0:14:46), 加工厂 (1:04:25) and 处理室 (1:28:41)."},
                {"zh": "核融合发电厂", "source": "zh.wikipedia 异形2 plot summary", "count": "n/a — REJECTED",
                 "note": "= fusion power plant. A plot-summary paraphrase, and factually a different building."},
            ],
            "corpus": "Four renderings inside one film — the corpus cannot vote.",
            "why_unresolved": "No source is both consistent and correct.",
            "next_step": "Owner ruling between 大气处理厂 (installation) and 大气处理机 (machine). The core-rules "
                         "pack will decide which sense the text actually needs.",
        },
        "Power Loader": {
            "kind": "proper noun — vehicle model (P-5000 Power Loader)",
            "candidates": [
                {"zh": "动力装载机", "source": "Chinese fan writing", "count": "unverified", "note": "Common fan rendering; unsourced to a reference work."},
                {"zh": "动力服", "source": "zh.wikipedia 异形2 plot summary", "count": "n/a — REJECTED", "note": "= powered suit. Loses the 'loader' sense entirely."},
                {"zh": "起重机械人", "source": "CnSCG subs", "count": "en-gated 1", "note": "ALIENS1986 0:37:02 'I can drive that loader' -> 我能操作起重机械人. Reads as a crane robot."},
            ],
            "corpus": "One hit, and it renames the machine.",
            "why_unresolved": "Same shape as Smartgun: no reference work reachable.",
            "next_step": "Check 萌娘百科 for P-5000. Note the English canon itself says 'Powered Work Loader' "
                         "(Caterpillar P-5000), which argues for 动力装载机.",
        },
        "Dropship": {
            "kind": "vehicle class (UD-4L Cheyenne)",
            "candidates": [
                {"zh": "(none)", "source": "CnSCG subs", "count": "en-gated 0 — the term is ERASED",
                 "note": "ALIENS1986 1:39:28 'get the other drop ship from the Sulaco' -> 我们必须派一艘船来救我们. The only occurrence."},
            ],
            "corpus": "Zero usable renderings.",
            "why_unresolved": "No source at all. 空降艇 / 登陆艇 are both plausible and neither is attested.",
            "next_step": "Owner ruling; the starter set will need this on the first read-through.",
        },
        "APC": {
            "kind": "vehicle class (M577 Armored Personnel Carrier)",
            "candidates": [
                {"zh": "(none)", "source": "CnSCG subs", "count": "en-gated 0 — MISTRANSLATED",
                 "note": "ALIENS1986 1:15:17 'fall back by squads to the A.P.C.' -> 离开焚化炉 (= leave the crematorium — badly wrong); 1:26:28 'salvage out of the A.P.C. wreckage' -> 失事船上能用的武器."},
            ],
            "corpus": "Both occurrences are wrong.",
            "why_unresolved": "装甲运兵车 is the standard Chinese military term and is almost certainly right, but "
                              "'APC' is also a model-bearing proper noun in this franchise (M577) and the project "
                              "rule sends proper nouns here rather than to a convention call.",
            "next_step": "Almost certainly 装甲运兵车 / M577装甲运兵车. Needs only an owner nod, not research.",
        },
        "Remote Sentry Unit": {
            "kind": "gear name (UA 571-C sentry gun)",
            "candidates": [
                {"zh": "机械感应器", "source": "CnSCG subs", "count": "en-gated 1 — REJECTED",
                 "note": "ALIENS1986 1:26:53 'we got four of these robot sentries' -> 我们有四套这种机械感应器 (= mechanical sensor). At 1:29:07 and 1:29:30 'sentry units' is dropped entirely."},
                {"zh": "自动哨戒炮 / 遥控哨戒单元", "source": "standard Chinese gaming usage", "count": "en-gated 0", "note": "Plausible; unattested for this franchise."},
            ],
            "corpus": "One hit and it turns a gun into a sensor.",
            "why_unresolved": "No source; and the device is a weapon, so getting it wrong changes what players think it does.",
            "next_step": "Owner ruling.",
        },
        "Fury 161": {
            "kind": "proper noun — planet name (a corruption of Fiorina 161)",
            "candidates": [
                {"zh": "复仇161星", "source": "CnSCG subs", "count": "en-gated 2, ALIEN3 only",
                 "note": "0:13:02 'Fury 161' -> 复仇161星; 0:37:13. RENDERS 'Fury' SEMANTICALLY ('vengeance') and silently drops that Fury is a corruption of Fiorina."},
            ],
            "corpus": "Consistent within one film, but the rendering destroys the etymology the name depends on.",
            "why_unresolved": "A phonetic 富里161 or 菲奥里纳161 would preserve the joke; neither is attested.",
            "next_step": "Owner ruling on semantic vs phonetic for franchise place names — the same decision "
                         "governs Acheron and Hadley's Hope, so settle all three together.",
        },
        "Ellen Ripley": {
            "kind": "proper noun — given name",
            "candidates": [
                {"zh": "海伦·蕾普丽", "source": "CnSCG subs", "count": "en-gated 2, RESURRECTION — REJECTED",
                 "note": "0:10:47 'Ellen Ripley died trying to wipe this species out' -> 海伦·蕾普丽死于致力消灭异形; 0:35:30. 海伦 is the standard Chinese rendering of HELEN, not Ellen — the corpus mistranslates the given name."},
                {"zh": "埃伦·蕾普丽 / 艾伦·蕾普丽", "source": "standard Chinese for 'Ellen'", "count": "en-gated 0", "note": "Two conventional forms; no Alien-specific attestation."},
            ],
            "corpus": "The only attestation is a mistranslation. The SURNAME is settled (蕾普丽, en-gated 75).",
            "why_unresolved": "Cannot pick between 埃伦 and 艾伦 without a reference work, and must not keep 海伦.",
            "next_step": "Read zh.wikipedia 异形 (电影) for the character's full Chinese name.",
        },
    },
    "resolved_since_brief": {
        "_note": "The task brief listed these as unsourced and not to be guessed. On re-checking the research "
                 "they ARE sourced, so they were entered into glossary_alien.json rather than left pending. "
                 "Flagged here so the discrepancy is visible rather than silently overridden.",
        "Special Order 937": {
            "cn": "937号特别指令",
            "source": "zh.wikipedia 异形 (电影) — verbatim",
            "evidence": "terminology-web.json:105 records the string 「937号特别指令」 directly attested, at "
                        "confidence 'high'. The brief's low-confidence flag does not match the research it cites. "
                        "The subtitle corpus separately attests the bare 特别指令 (ALIEN1979 1:23:39) with the "
                        "number dropped, which is consistent, not contradictory.",
            "caveat": "The zh-wiki citation is from terminology-web.json and was NOT re-derived this session "
                      "(WebFetch was unavailable). If that turns out to be wrong, move it back to terms.",
        },
    },
}

os.makedirs(OUT, exist_ok=True)


def w(name, obj):
    p = os.path.join(OUT, name)
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"{os.path.getsize(p):>8}  {p}")


w("glossary_alien.json", gloss)
w("glossary_alien.provenance.json", provenance)
w("glossary_alien.disputes.json", disputes)
w("glossary_alien.pending.json", pending)

print()
print("terms:", len(gloss))
print("tiers:", dict(sorted(TIERCOUNT.items())))
print("categories:", dict(sorted(CATCOUNT.items())))
print("chinese-value collisions:", len(MULTI))
for cn, ens in sorted(MULTI.items()):
    print("   ", cn, "<-", ens)
