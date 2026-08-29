#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
glossary_additions.py — terms the NEW sources supply that the Phase-0 spine had no key
for, plus the frozen literals the Phase-0 spine missed, plus the terms that must LEAVE
the spine because the new ladder gives them no usable evidence.

Imported by build_glossary_v1.py.  Same contract as glossary_adjudication.py: every
count is read out of a miner artifact at build time.
"""


def build(ctx):
    fanc = ctx["fanc"]; subc = ctx["subc"]; isoc = ctx["isoc"]
    langc = ctx["langc"]; refc = ctx["refc"]; convc = ctx["convc"]; cand = ctx["cand"]
    CNSCG = "CnSCG(异形1/2/3/4·同源)"

    def A(en, cn, level, why, cands, **kw):
        d = {"cn": cn, "source_level": level, "why_this_won": why, "candidates": cands}
        d.update(kw)
        return en, d

    AD = {}

    def add(*args, **kw):
        en, d = A(*args, **kw)
        AD[en] = d

    # ---------------------------------------------------------------- §1.4 --
    add("Stretch", "节", 1,
        "Level 1 ch.1's time-unit table, glossed inline: 节 (Stretch) / 5-10分 / 潜行. Level 4 has "
        "no Stretch key at all, so level 1 is uncontested. Completes the 轮 / 节 / 班 set.",
        [fanc("Stretch", zh="节")], tier="T-PLAIN", category="yze-mechanics",
        time_unit_set=["Round", "Stretch", "Shift"])

    add("Frontier", "边境", 1,
        "Level 1 ch.1, glossed inline. No competitor at any other level.",
        [fanc("the Frontier", zh="边境")], tier="T-PLAIN", category="franchise-place")
    add("Outer Veil", "外层帷幕", 1,
        "Level 1 ch.1, glossed inline. The old 1e fan file (level 5 register) says 外面纱 and is "
        "explicitly rejected: its own filename declares it is the FIRST edition, not Evolved.",
        [fanc("the Outer Veil", zh="外层帷幕"),
         cand("外面纱", 5, "1st-edition Chinese fan file (level-5 register, NOT this book)",
              "occurrence; the file's own name says 这是旧版的 非新版Evolved", "occurrence_in_1e_file",
              "mined_fan_translation.json terms[en='Outer Veil (1e file)']")],
        tier="T-BILINGUAL", category="franchise-place")
    add("Core Systems", "核心星系", 1, "Level 1 ch.1, glossed inline.",
        [fanc("the Core Systems", zh="核心星系")], tier="T-BILINGUAL", category="franchise-place")
    add("Outer Rim", "外环", 1, "Level 1 ch.1, glossed inline.",
        [fanc("the Outer Rim", zh="外环")], tier="T-BILINGUAL", category="franchise-place")
    add("Evolved Edition", "进化版", 1,
        "Level 1 ch.1, and the owner's own working directories use 进化版 for this edition.",
        [fanc("Evolved Edition", zh="进化版")], tier="T-PLAIN", category="franchise-title")

    # ------------------------------------------------- polities and bodies --
    add("Three World Empire", "三界帝国", 1,
        "Level 1 ch.1, glossed inline; the level-5 1e file independently agrees "
        "(三界帝国（Three World Empire，缩写3WE）).",
        [fanc("Three World Empire", zh="三界帝国")], tier="T-BILINGUAL", category="franchise-corp")
    add("United Americas", "联合美洲", 1,
        "Level 1 ch.1, glossed inline. The 1e file says 美洲联邦 and is rejected as a different "
        "edition.",
        [fanc("United Americas", zh="联合美洲")], tier="T-BILINGUAL", category="franchise-corp")
    add("Union of Progressive Peoples (UPP)", "人类进步联盟", 1,
        "Level 1 ch.1, glossed inline. The 1e file says 进步人民联盟 and is rejected.",
        [fanc("Union of Progressive Peoples (UPP)", zh="人类进步联盟")],
        tier="T-BILINGUAL", category="franchise-corp")
    add("Independent Core System Colonies (ICSC)", "独立核心星系殖民地", 1,
        "Level 1 ch.1, glossed inline (the fan gloss itself misspells COLONIES as 'COLONISE'; the "
        "Chinese is unaffected).",
        [fanc("Independent Core System Colonies (ICSC)", zh="独立核心星系殖民地")],
        tier="T-BILINGUAL", category="franchise-corp")
    add("Interstellar Commerce Commission (ICC)", "星际商务委员会", 1,
        "Level 1 ch.2, glossed inline inside the ICC licence name.",
        [fanc("Interstellar Commerce Commission (ICC)", zh="星际商务委员会")],
        tier="T-BILINGUAL", category="franchise-corp")
    add("Colonial Navy", "殖民海军", 1,
        "Level 1 ch.1. ⚠ 殖民海军 is a strict PREFIX of 殖民海军陆战队 (Colonial Marines). Any "
        "search-and-replace over this string will corrupt every Colonial Marines string in the "
        "corpus. Anchor on the longer form first.",
        [fanc("Colonial Navy", zh="殖民海军")], tier="T-BILINGUAL", category="marine",
        prefix_hazard="殖民海军陆战队")
    add("Seegson", "西格森", 1,
        "Level 1 ch.1, glossed inline. ⚠ Level 3 (Isolation, 217 keys) NEVER transliterates it — it "
        "keeps the Latin 'Seegson' and suffixes a Chinese common noun with no separating space "
        "(Seegson公司, Seegson通信中心). That is the official localisation's house style, not a "
        "terminology decision, and it conflicts with this project's T-BILINGUAL rule (Chinese "
        "first, ONE ASCII space). Level 1 wins; level 3 is recorded so the divergence is visible.",
        [fanc("Seegson", zh="西格森"), isoc("Seegson", zh="Seegson")],
        tier="T-BILINGUAL", category="franchise-corp")

    # ---------------------------------------------------- places and ships --
    add("Anchorpoint", "锚点", 1, "Level 1 ch.1, glossed inline.",
        [fanc("Anchorpoint", zh="锚点")], tier="T-BILINGUAL", category="franchise-place")
    add("Anchorpoint Station", "锚点站", 1, "Level 1 ch.1, glossed inline.",
        [fanc("Anchorpoint Station", zh="锚点站")], tier="T-BILINGUAL",
        category="franchise-place", derived_from="Anchorpoint")
    add("Sevastopol Station", "塞瓦斯托波尔站", 1,
        "Level 1 ch.1, glossed inline. Level 3 (Isolation, 228 keys) keeps the Latin name and "
        "appends 空间站; same house-style divergence as Seegson. Level 1 wins.",
        [fanc("Sevastopol Station", zh="塞瓦斯托波尔站"), isoc("Sevastopol Station", zh="Sevastopol空间站")],
        tier="T-BILINGUAL", category="franchise-place")
    add("Thedus", "特杜斯", 1, "Level 1 ch.1, glossed inline. The 1e file says 西德斯; rejected.",
        [fanc("Thedus", zh="特杜斯")], tier="T-BILINGUAL", category="franchise-place")
    add("Torin Prime", "托林主星", 1,
        "Level 1 ch.1, glossed inline. The 1e file transliterates 'Prime' as 普莱姆; level 1 reads "
        "it correctly as the primary world.",
        [fanc("Torin Prime", zh="托林主星")], tier="T-BILINGUAL", category="franchise-place")
    add("Hadley's Hope", "哈德利的希望", 1,
        "Level 1 ch.1, glossed inline. WAS PENDING: the subtitle corpus has zero hits for this name "
        "across all 11,088 pairs, which is why Phase 0 could not place it. Level 1 reaches it.",
        [fanc("Hadley's Hope", zh="哈德利的希望"),
         cand(None, 6, "CnSCG Alien 1/2/3/4 subtitles (level 6)",
              "en-gated 0 — zero hits corpus-wide", "not_attested",
              "mined_subtitles_vote.json zero_hit_terms includes \"Hadley's Hope\"")],
        tier="T-BILINGUAL", category="franchise-place", was_pending=True)
    add("Fiorina 161", "菲奥莉娜161", 1,
        "Level 1 ch.1, majority reading (2 of 3 occurrences). ⚠ Level 1 is internally inconsistent: "
        "the ONE inline-glossed instance uses the minority spelling 费奥莉娜 161 (with a space). The "
        "majority spelling is adopted and the glossed minority is recorded, because a glossary that "
        "silently picks the glossed instance would import the typo too.",
        [fanc("Fiorina 161", zh="菲奥莉娜161"),
         fanc("Fiorina 161 (glossed, minority)", zh="费奥莉娜161")],
        tier="T-BILINGUAL", category="franchise-place", aliases=["费奥莉娜161", "复仇161星"],
        note="Level 6 renders the same planet 复仇161星, which is Taiwan-flavoured and on the "
             "project's known-bad register list. 'Fury 161' is the same world under its other "
             "English name and stays in pending until level 1 or 2 reaches that spelling.")
    add("Sulaco", "萨拉科号", 1,
        "Level 1 ch.1, glossed inline as USS萨拉科号. WAS PENDING. Level 6's 苏拉可号 is thin (2 lines, "
        "one lineage) and Taiwan-flavoured; zh.wikipedia's 苏拉克号 was the Phase-0 candidate and is "
        "level 7.",
        [fanc("USS Sulaco", zh="USS萨拉科号"), subc("Sulaco", CNSCG, zh="苏拉可"),
         refc("苏拉克号", "zh.wikipedia (level 7)", "The Phase-0 candidate value.")],
        tier="T-BILINGUAL", category="franchise-ship", was_pending=True,
        aliases=["苏拉可号", "苏拉克号"])
    add("Prometheus", "普罗米修斯号", 1,
        "Level 1 ch.1 (USCSS普罗米修斯号) with levels 2 and 5 agreeing on 普罗米修斯 across three "
        "lineages. The ship, as distinct from the film title.",
        [fanc("USCSS Prometheus", zh="USCSS普罗米修斯号"),
         subc("Prometheus", "Romulus2024", zh="普罗米修斯"),
         subc("Prometheus", "Prometheus2012", zh="普罗米修斯"),
         subc("Prometheus", "Covenant2017", zh="普罗米修斯")],
        tier="T-BILINGUAL", category="franchise-ship")
    add("Covenant", "契约号", 1,
        "Level 1 ch.1 (USCSS 契约号) and level 5 (Covenant 2017, en-gated 14 rows) agree.",
        [fanc("USCSS Covenant", zh="USCSS 契约号"), subc("Covenant", "Covenant2017", zh="契约号")],
        tier="T-BILINGUAL", category="franchise-ship")
    add("USS", "USS", 1,
        "Latin ship prefix. Level 1 keeps every prefix (USCSS / USS / UAS) in Latin letters and "
        "glues it to the front of the Chinese hull name. Not mechanics-critical — see the T-LATIN "
        "tier note.",
        [fanc("USS Sulaco", zh="USS萨拉科号")], tier="T-LATIN", category="franchise-ship")
    add("UAS", "UAS", 1, "Latin ship prefix; level 1 ch.1 (UAS大天使号).",
        [cand("UAS", 1, "《异形RPG：进化版》中文连载 ch1 (level 1)",
              "occurrence; the prefix is left in Latin", "occurrence_in_translated_book",
              "ch1 'UAS大天使号' / 「大天使号」(UAS Archangel)")],
        tier="T-LATIN", category="franchise-ship")

    # --------------------------------------------------- people --------------
    add("Peter Weyland", "彼得·韦兰德", 1,
        "Level 1 ch.1, glossed inline. Surname locked to 韦兰德 — see the Weyland surname family.",
        [fanc("Peter Weyland", zh="彼得·韦兰德"), subc("Peter Weyland", "Prometheus2012", zh="维兰德")],
        tier="T-BILINGUAL", category="franchise-character",
        surname_crossref=["Weyland-Yutani", "Weyland Corporation"])
    add("David", "大卫", 1,
        "Level 1 ch.1 glossed inline, with levels 5 and 6 agreeing across three lineages "
        "(en-gated 51 rows).",
        [fanc("David (android series)", zh="大卫"), subc("David", "Covenant2017", zh="大卫"),
         subc("David", "Prometheus2012", zh="大卫"), subc("David", CNSCG, zh="大卫")],
        tier="T-BILINGUAL", category="franchise-character")
    add("Meredith Vickers", "梅雷迪思·维克尔斯", 1,
        "Level 1 ch.1, glossed inline. Level 5 (Prometheus 2012) says 维克斯 for the surname alone; "
        "the 1e file says 梅雷迪斯·维克斯 and is rejected.",
        [fanc("Meredith Vickers", zh="梅雷迪思·维克尔斯"),
         subc("Meredith Vickers", "Prometheus2012", zh="维克斯")],
        tier="T-BILINGUAL", category="franchise-character")

    # ------------------------------------------------ tech and hardware ------
    add("FTL Drive", "超光速引擎", 1,
        "Level 1 ch.1, glossed inline. Level 1's careers section drifts to 超光速驱动器 once; the "
        "glossed instance wins.",
        [fanc("FTL drive", zh="超光速引擎")], tier="T-BILINGUAL", category="hardware",
        aliases=["超光速驱动器"])
    add("FTL Travel", "超光速飞行", 1, "Level 1 ch.1.",
        [fanc("FTL travel", zh="超光速飞行")], tier="T-PLAIN", category="ops")
    add("Hypersleep Pod", "深度睡眠舱", 1, "Level 1 ch.1, glossed inline.",
        [fanc("hypersleep pod", zh="深度睡眠舱")], tier="T-BILINGUAL", category="hardware",
        derived_from="Hypersleep")
    add("Atmosphere Processor", "大气处理器", 1,
        "Level 1 ch.1, glossed inline. WAS PENDING, and the Phase-0 survey's 大气处理厂 ('atmosphere "
        "processing PLANT') is explicitly what §1.4 corrects. Level 6 only ever produces the bare "
        "大气.",
        [fanc("atmospheric processor", zh="大气处理器"),
         subc("atmosphere processor", CNSCG, zh="大气")],
        tier="T-BILINGUAL", category="hardware", was_pending=True, aliases=["大气处理厂"])
    add("Motion Tracker", "运动追踪器", 1,
        "Level 1 ch.2 (M314运动追踪器) AND level 3's hardest gate — GLOBAL.TXT [TEXT_MOTIONTRACKER] "
        "-> 运动追踪器 plus 17 key-gated keys. WAS PENDING. Level 6's 行动追踪器 is a single thin line.",
        [fanc("M314 Motion Tracker", zh="M314运动追踪器"), isoc("motion tracker", zh="运动追踪器"),
         subc("motion tracker", CNSCG, zh="行动追踪器")],
        tier="T-BILINGUAL", category="hardware", was_pending=True)
    add("Smart Gun", "智能炮", 1,
        "Level 1 ch.2 (M56A2智能炮). WAS PENDING. Level 6's 机关枪 ('machine gun') is on the "
        "project's explicit known-bad list and is never eligible.",
        [fanc("M56A2 Smart Gun", zh="M56A2智能炮"), subc("smartgun", CNSCG, zh="机关枪")],
        tier="T-BILINGUAL", category="hardware", was_pending=True,
        known_bad_rejected=["机关枪"])
    add("Dropship", "着陆舰", 1,
        "Level 1 ch.2 ('From starfighters to dropships, freighters to frigates'). WAS PENDING "
        "because level 6 ERASES the word — its only two hits drop 'dropship' from the Chinese line "
        "entirely, which is on the known-bad list.",
        [fanc("dropship", zh="着陆舰"), subc("dropship", CNSCG, zh="一艘船")],
        tier="T-BILINGUAL", category="hardware", was_pending=True,
        known_bad_rejected=["(erased)"])
    add("Starfighter", "星际战机", 1, "Level 1 ch.2, same sentence as Dropship.",
        [fanc("starfighter", zh="星际战机")], tier="T-BILINGUAL", category="hardware")
    add("Bolt Gun", "气矢枪", 3,
        "⚠ LADDER DEVIATION, declared. Level 1 renders the Watatsumi DV-303 Bolt Gun 栓动枪, which "
        "means BOLT-ACTION — a rifle mechanism the DV-303 does not have. Level 3 key-gates "
        "AI_UI_ITEM_BOLTGUN -> 气矢枪 ('air-dart gun'), which describes the pneumatic captive-bolt "
        "weapon the fiction actually has. Logged in _meta.ladder_deviations.",
        [fanc("Watatsumi DV-303 Bolt Gun", zh="海神DV-303栓动枪"), isoc("bolt gun", zh="气矢枪")],
        tier="T-BILINGUAL", category="hardware", beats_a_higher_level=1)
    add("M5A3 RPG Launcher", "M5A3 RPG发射器", "C",
        "⚠ KEYED FOR A RUNTIME REASON, not a terminology one, and it is the MIRROR IMAGE of Pack "
        "Mule. character-sheet.mjs:428 and synthetic-sheet.mjs:415 decide this weapon's per-round "
        "ammo weight (0.5 kg vs 0.25 kg) from its NAME: the item qualifies if "
        "name.includes(' RPG ') OR name.startsWith('RPG') OR name.endsWith('RPG'). The data leg "
        "(system.attributes.class.value === 'RPG') is measured DEAD in every pack, so the name "
        "test is the only live path. Here the T-BILINGUAL tail is what KEEPS IT WORKING — "
        "'M5A3 RPG发射器 M5A3 RPG Launcher' still contains ' RPG ' — whereas a pure-Chinese rename "
        "drops it and the encumbrance silently halves. Pack Mule says 'no tail'; this one says "
        "'tail required'. Same file, opposite rules.",
        [convc("M5A3 RPG发射器", "model designator kept in Latin, as with M41A脉冲步枪 (level 1); "
                                 "发射器 follows Grenade Launcher 榴弹发射器")],
        tier="T-BILINGUAL", category="hardware",
        substring_tests=[" RPG ", "RPG (startsWith)", "RPG (endsWith)"],
        substring_semantics="at_least_one",
        breaks_if_translated="ammoweight silently falls from 0.5 to 0.25 kg per round. The sheet "
                             "shows a lighter load, the Encumbered threshold moves, and nothing "
                             "errors.")
    add("Keycard", "钥匙卡", 3,
        "Level 3, key-gated: TEXT_KEYCARD -> 钥匙卡 and the natural-language key 'Use Key Card' -> "
        "使用钥匙卡. No level 1 or 2 competitor.",
        [isoc("keycard", zh="钥匙卡")], tier="T-BILINGUAL", category="hardware")
    add("MedPod", "医疗舱", 1, "Level 1 ch.2.", [fanc("MedPod", zh="医疗舱")],
        tier="T-BILINGUAL", category="hardware")

    # ------------------------------------------------- creatures -------------
    add("Neomorph", "新变种", 1,
        "Level 1 ch.1. WAS PENDING: the subtitle corpus has zero hits for 'neomorph' across all "
        "11,088 pairs. The corerules pack ships a Neomorph creature and an 'EV - Neomorph Attacks' "
        "table, so an absent key here was a live gap.",
        [fanc("Neomorph", zh="新变种"),
         cand(None, 6, "CnSCG Alien 1/2/3/4 subtitles (level 6)",
              "en-gated 0 — zero hits corpus-wide", "not_attested",
              "mined_subtitles_vote.json zero_hit_terms includes 'neomorph'")],
        tier="T-BILINGUAL", category="franchise-creature", was_pending=True)

    # ------------------------------------------ play structure and money -----
    add("Act", "幕", 1, "Level 1 ch.2 — the act of a cinematic adventure.",
        [fanc("act (of a cinematic adventure)", zh="幕")], tier="T-PLAIN", category="yze-mechanics")
    add("Game Session", "游戏会话", 1,
        "Level 1 ch.2; it drifts to 游戏环节 once and the majority form is adopted.",
        [fanc("game session", zh="游戏会话")], tier="T-PLAIN", category="yze-mechanics")
    add("Character Sheet", "角色卡", 1, "Level 1 ch.2.", [fanc("character sheet", zh="角色卡")],
        tier="T-PLAIN", category="yze-mechanics")
    add("Player Character", "玩家角色", 1, "Level 1 ch.1, glossed inline.",
        [fanc("player character (PC)", zh="玩家角色")], tier="T-PLAIN", category="yze-mechanics")
    add("Non-Player Character", "非玩家角色", 1, "Level 1 ch.1.",
        [fanc("non-player character (NPC)", zh="非玩家角色")],
        tier="T-PLAIN", category="yze-mechanics")
    add("Experience Points", "经验值", 1,
        "Level 1 ch.2, glossed inline as 经验值(XP); it writes 经验点 once and the glossed form wins.",
        [fanc("Experience Points (XP)", zh="经验值")], tier="T-PLAIN", category="yze-mechanics")
    add("Countdown", "倒计时", 1, "Level 1 ch.1.", [fanc("countdown", zh="倒计时")],
        tier="T-PLAIN", category="yze-mechanics")
    add("Zone", "区域", 1, "Level 1 ch.2.", [fanc("zone", zh="区域")],
        tier="T-PLAIN", category="range")
    add("Stealth Mode", "潜行模式", 1, "Level 1 ch.2.", [fanc("stealth mode", zh="潜行模式")],
        tier="T-PLAIN", category="yze-mechanics")
    add("Over-Encumbered", "超重", 1, "Level 1 ch.2.", [fanc("Over-Encumbered", zh="超重")],
        tier="T-PLAIN", category="yze-condition")
    add("W-Y", "韦汤", 1,
        "Level 1's clipped form of Weyland-Yutani, used 11 times in ch.1 and glossed inline in the "
        "currency name 韦汤币 (W-Y dollar). This is the SHORT FORM the owner named in §1.4.",
        [fanc("Weyland-Yutani", zh="韦汤")], tier="T-PLAIN", category="franchise-corp",
        derived_from="Weyland-Yutani",
        surname_crossref=["Weyland-Yutani", "Weyland Corporation", "Peter Weyland"])
    add("W-Y Dollars", "韦汤币", 1,
        "Level 1 ch.2, glossed inline as 韦汤币 (W-Y dollar). The career blocks write the bare W-Y币 "
        "nine times; the glossed form is the term.",
        [fanc("W-Y dollars", zh="韦汤币")], tier="T-PLAIN", category="franchise-corp",
        derived_from="W-Y", aliases=["W-Y币"])
    add("Space Truckers", "太空卡车司机", 1,
        "Level 1 ch.1 — one of the three campaign frameworks.",
        [fanc("Space Truckers (framework)", zh="太空卡车司机")],
        tier="T-BILINGUAL", category="yze-mechanics")
    add("Frontier Colonists", "边境殖民者", 1,
        "Level 1 ch.1 — one of the three campaign frameworks; composed of Frontier 边境 and "
        "Colonist 殖民者, both level 1.",
        [fanc("Frontier Colonists (framework)", zh="边境殖民者")],
        tier="T-BILINGUAL", category="yze-mechanics")
    add("Campaign Framework", "战役框架", 1, "Level 1 ch.1.",
        [fanc("campaign framework", zh="战役框架")], tier="T-PLAIN", category="yze-mechanics")

    # =====================================================================
    # FROZEN LITERALS the Phase-0 spine missed.
    # Every one of these is a REAL lookup in DO-NOT-TRANSLATE.json.
    # =====================================================================
    _frozen_note = ("Added in v1.0. The Phase-0 glossary carried 24 of the register's "
                    "mechanics-critical literals and missed these five, which means five ways to "
                    "break the game silently had no glossary key telling a translator to leave them "
                    "alone.")
    for _lit, _sec, _sites, _effect, _live in [
        ("EV - 48a. LS - DANGER EVENT DETAIL", "rolltable_names",
         "alien-evolved-corerules pack :: Macro 'Roll on Danger Event Detail Table' "
         "(_id OHnG8YcGCm4txlR9) .command",
         "UNGUARDED: `table.formula` is read one line after getName(). Translating the table name "
         "throws a TypeError and the only Macro the Core Rules pack ships stops working.", True),
        ("HARDENED", "item_names",
         "module/data/actor-character.mjs:449, module/data/actor-synthetic.mjs:439",
         "attrMod.health += 1 in BOTH modes. Silent: the character just loses 1 max Health. "
         "⚠⚠ LIVE — 'Hardened' ships as a talent in corerules AND embedded in the Actor "
         "'EV - MINING WILDCATTER'. ⚠⚠ AND level 1 renders this talent 硬汉, so the fan translation "
         "would break it. The NAME field must stay English; 硬汉 may be used in the talent's PROSE.",
         True),
        ("STOIC", "item_names",
         "module/data/actor-character.mjs:452, module/data/actor-synthetic.mjs:442",
         "this.skills.stamina.ability = 'wit' when WIT > STR, BOTH modes. Silent: every Stamina "
         "roll's modifier changes. LIVE — 'Stoic' ships as a talent in corerules.", True),
        ("TOUGH", "item_names",
         "module/data/actor-character.mjs:444, module/data/actor-synthetic.mjs:434",
         "attrMod.health += 2, classic mode only. Silent. Dead-but-reserved: no document with this "
         "name ships today.", False),
        ("NERVES OF STEEL", "item_names",
         "module/data/actor-character.mjs:436, module/data/actor-synthetic.mjs:426",
         "attrMod.stress -= 2, classic mode only. Silent: the character panics sooner. "
         "Dead-but-reserved.", False),
    ]:
        AD[_lit] = {
            "cn": _lit, "source_level": "FROZEN", "candidates": [],
            "why_this_won": "Not a terminology decision. This is a byte-exact string the running "
                            "code compares against; a translation of any kind breaks it.",
            "tier": "T-FROZEN",
            "category": "frozen-rolltable" if _sec == "rolltable_names" else "frozen-item-name",
            "frozen_section": _sec, "call_sites": _sites, "breaks_if_translated": _effect,
            "live_in_packs": _live, "note": _frozen_note,
        }
    AD["HARDENED"]["comparison_is_uppercased"] = True
    AD["HARDENED"]["shipped_document_name"] = "Hardened"
    AD["HARDENED"]["level_1_collision"] = (
        "《异形RPG：进化版》中文连载 ch.2 renders the talent Hardened as 硬汉. Applying that to the "
        "ITEM NAME silently costs every affected character 1 max Health, because "
        "'硬汉'.toUpperCase() is not 'HARDENED'. This is the single clearest case in the project "
        "of a level-1 rendering that is correct as prose and fatal as a name.")
    AD["STOIC"]["comparison_is_uppercased"] = True
    AD["STOIC"]["shipped_document_name"] = "Stoic"
    AD["TOUGH"]["comparison_is_uppercased"] = True
    AD["NERVES OF STEEL"]["comparison_is_uppercased"] = True

    # =====================================================================
    # LEAVING THE SPINE
    # =====================================================================
    TO_PENDING = {
        "Chestburster": {
            "phase0_cn": "破胸体",
            "why": "Zero English-gated hits across all 11,088 subtitle pairs, zero in the Alien: "
                   "Isolation localisation, and level 1 does not reach the Xenomorph life cycle — "
                   "that is ch.10 and only ch.1 and ch.2 exist. The Phase-0 value came from "
                   "zh.wikipedia, which the new ladder demotes to level 7, the floor. A level-7 "
                   "value for a creature caste name that the packs use as an Actor name is exactly "
                   "the kind of plausible guess that propagates silently.",
            "sources_checked": ["level 1: not reached (life cycle is ch.10)",
                                "level 2: en-gated 0", "level 3: not attested",
                                "level 5: en-gated 0", "level 6: en-gated 0",
                                "level 7: zh.wikipedia 破胸体 / 百度百科 破胸者"],
            "what_would_settle_it": "ch.10 of the fan serial, if it is ever written. The owner has "
                                    "confirmed only ch.1 and ch.2 exist, so plan for this staying "
                                    "pending.",
            "pack_impact": "corerules ships Chestburster Actors and an 'EV - Chestburster Attacks' "
                           "RollTable. The table NAME is separately frozen by actor.mjs's "
                           "system.rTables lookup, so the pending term blocks only the prose and "
                           "the Actor name.",
        },
        "Ovomorph": {
            "phase0_cn": "异形卵",
            "why": "Same evidence state as Chestburster: en-gated 0 everywhere, level 1 does not "
                   "reach it. 异形卵 is a compound Phase 0 built, not a rendering anyone attested.",
            "sources_checked": ["level 1: not reached", "level 2: en-gated 0",
                                "level 3: not attested", "level 6: en-gated 0"],
            "what_would_settle_it": "ch.10 of the fan serial.",
            "note": "The bare noun Egg IS settled (卵, level 3). Ovomorph is the caste designation "
                    "and is a different word in the English.",
        },
        "Drone": {
            "phase0_cn": "工蜂异形",
            "why": "en-gated 0 across every level. 工蜂异形 ('worker-bee alien') is a Phase-0 "
                   "coinage from the wiki's bee metaphor.",
            "sources_checked": ["level 1: not reached", "level 2: en-gated 0",
                                "level 3: not attested", "level 6: en-gated 0"],
            "what_would_settle_it": "ch.10 of the fan serial, or the Evolved bestiary's own Chinese "
                                    "release.",
        },
        "Warrior": {
            "phase0_cn": "战士异形",
            "why": "The corpus's only hit is a single Alien: Earth 2025 line rendering 战士模式 "
                   "('warrior mode'), which is not the caste noun. Grade 'thin', one line, one "
                   "lineage. Not enough to name an Actor.",
            "sources_checked": ["level 1: not reached",
                                "level 2: en-gated 1 line, 战士模式, grade thin",
                                "level 3: not attested", "level 6: en-gated 0"],
            "what_would_settle_it": "ch.10 of the fan serial.",
        },
        "United Systems Military": {
            "phase0_cn": "联合星系军",
            "why": "RULE VIOLATION, carried over from Phase 0 and now resolved by removal. This is "
                   "a proper noun — a military branch — and its Phase-0 why_this_won said literally "
                   "'Coined.' Its only recorded candidate was 军队 at en-gated 1, dismissed as too "
                   "generic. The glossary's own rule (and pending._meta.rule) is that convention "
                   "may settle a COMMON noun and never a proper noun. Its sibling organisation "
                   "USCM was already pending for exactly this reason; two military branches in the "
                   "same evidence state were getting opposite treatment.",
            "sources_checked": ["level 1: not reached", "level 2: en-gated 0",
                                "level 3: not attested", "level 6: en-gated 0", "level 7: silent"],
            "what_would_settle_it": "bilibili cv18100844 (named in PROJECT.md §7.2), or ch.3+ of "
                                    "the fan serial.",
            "was_rule_violation": True,
        },
        "Rebecca Jorden": {
            "phase0_cn": "丽贝卡·乔登",
            "why": "RULE VIOLATION, carried over from Phase 0 and now resolved by splitting. The "
                   "GIVEN name 丽贝卡 is en-gated 5 (ALIENS1986 1:02:32) and is sound. The SURNAME "
                   "乔登 was sourced to 'standard modern Chinese usage' at en-gated 0 — a proper "
                   "noun settled by convention, which the rule forbids. The usual Chinese form of "
                   "Jorden/Jordan is 乔丹, so 乔登 is a choice presented as a standard. The character "
                   "IS reachable under her nickname: Newt = 纽特 stays in the glossary at level 6, "
                   "en-gated 35.",
            "sources_checked": ["given name 丽贝卡: level 6, en-gated 5",
                                "surname 乔登: en-gated 0 at every level",
                                "nickname Newt 纽特: level 6, en-gated 35 — IN the glossary"],
            "what_would_settle_it": "Any level 1-3 source that spells the surname. Until then use "
                                    "the nickname, which is what the fiction uses anyway.",
            "was_rule_violation": True,
        },
    }

    return AD, TO_PENDING
