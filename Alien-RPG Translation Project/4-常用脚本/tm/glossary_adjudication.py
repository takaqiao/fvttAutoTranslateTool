#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
glossary_adjudication.py — the AUTHORED half of the v1.0 Alien-RPG glossary rebuild.

Imported by build_glossary_v1.py, which supplies the candidate constructors
(fanc / subc / isoc / langc / refc / convc) so that every count in here is READ OUT
of a miner artifact at build time and never transcribed by hand.

OVERRIDES  — a Phase-0 term whose winner CHANGES under the new ladder, or whose
             evidence is upgraded (same winner, stronger source).
ADDITIONS  — a term the new sources supply that the Phase-0 spine had no key for.
TO_PENDING — a term that must LEAVE the spine: no usable evidence under the ladder.
"""


def build(ctx):
    fanc = ctx["fanc"]; subc = ctx["subc"]; isoc = ctx["isoc"]
    langc = ctx["langc"]; refc = ctx["refc"]; convc = ctx["convc"]; cand = ctx["cand"]
    CNSCG = "CnSCG(异形1/2/3/4·同源)"

    def O(cn, level, why, cands, **kw):
        d = {"cn": cn, "source_level": level, "why_this_won": why, "candidates": cands}
        d.update(kw)
        return d

    OV = {}
    AD = {}

    # =======================================================================
    # 1. THE OWNER'S NAMED CHANGES  (PROJECT.md §1.4)
    # =======================================================================
    def wy_cands():
        return [
            fanc("Weyland-Yutani (full form)", zh="韦兰德-汤谷公司"),
            fanc("Weyland-Yutani", zh="韦汤"),
            subc("Weyland-Yutani", "AlienEarth2025", zh="威兰汤谷"),
            subc("Weyland-Yutani", "Romulus2024", zh="威兰"),
            isoc("Weyland-Yutani", zh="Weyland-Yutani"),
            refc("韦兰-尤坦尼集团", "zh.wikipedia (level 7) — article title",
                 "The Phase-0 spine. The same article's LEAD says 韦兰德-尤坦尼集团, its traditional "
                 "variant is 韋蘭德-湯谷企業 and its alternate is 韦兰德-汤谷公司 — zh-wiki is "
                 "internally inconsistent between its own title and its own lead."),
            cand("伟伦优达尼公司", 6,
                 "CnSCG Alien 1/2/3/4 subtitles (level 6 — ALL FOUR ARE ONE VOTE)",
                 "en-gated 1 of 2 rows matching 'Weyland|Yutani' in the CnSCG lineage; the rebuilt "
                 "11,088-pair corpus returns no CnSCG per-lineage top for this term at all",
                 "english_gated_subtitle_vote",
                 "RESURRECTION 0:13:25 'Weyland-Yutani. Ripley 8s former employers.' -> "
                 "伟伦优达尼公司 蕾普丽8号生前的雇主.  Taiwan-flavoured phonetics. The ALIEN3 0:09:09 "
                 "hit is on-screen text with an EMPTY English side, so it is cn_only, not a vote."),
        ]

    WY_WHY = (
        "LEVEL 1 wins outright. The Chinese translation of THIS book spells it 韦兰德-汤谷公司 in "
        "full and clips it to 韦汤(公司) 11 times in ch.1. Level 2 (Romulus 2024 / Alien: Earth "
        "2025) independently agrees on the SEMANTIC treatment of Yutani — 汤谷, the Japanese "
        "surname 湯谷 — and differs only on the phonetics of Weyland (威兰 vs 韦兰德). Level 3 "
        "(Isolation) never transliterates the name at all. Only level 7 (zh.wikipedia) produced "
        "尤坦尼, and level 7 is the ladder's floor. The whole 尤坦尼 branch is therefore abandoned, "
        "and with it the Phase-0 spine value.")

    OV["Weyland-Yutani"] = O(
        "韦兰德-汤谷公司", 1, WY_WHY, wy_cands(),
        tier="T-BILINGUAL", category="franchise-corp",
        short_form="韦汤",
        aliases=["韦汤", "韦汤公司", "威兰汤谷", "韦兰-尤坦尼集团", "伟伦优达尼公司"],
        surname_crossref=["Weyland Corporation", "Weyland-Yutani Corporation", "Peter Weyland",
                          "Yutani Corporation", "W-Y"],
        note="⚠ SURNAME LOCK. Weyland is 韦兰德 in EVERY key that carries it — Weyland Corporation "
             "韦兰德公司, Weyland-Yutani 韦兰德-汤谷公司, Peter Weyland 彼得·韦兰德. The Phase-0 file "
             "spelled the same surname two ways (韦兰德公司 vs 韦兰-尤坦尼集团) with no cross-reference "
             "between them, which is exactly how a consistency sweep 'unifies' a family the wrong "
             "way round. All five keys now carry surname_crossref so the sweep can see they are one "
             "family and which spelling is the family's.",
        dispute=None)

    OV["Weyland-Yutani Corporation"] = O(
        "韦兰德-汤谷公司", 1,
        "Same string as Weyland-Yutani: level 1's 公司 already carries 'Corporation', so appending "
        "a second 公司 would be wrong. Kept as its own key only because the English source uses "
        "both spellings.",
        wy_cands(), tier="T-BILINGUAL", category="franchise-corp",
        derived_from="Weyland-Yutani",
        surname_crossref=["Weyland-Yutani", "Weyland Corporation", "Peter Weyland"],
        dispute=None)

    OV["Weyland Corporation"] = O(
        "韦兰德公司", 1,
        "Level 1 ch.1 names the pre-merger company 韦兰德公司 outright. Now CONSISTENT with "
        "Weyland-Yutani 韦兰德-汤谷公司 — under the Phase-0 wiki spine these two spelled one surname "
        "two ways.",
        [fanc("Weyland Corp / Weyland Industries", zh="韦兰德公司"),
         subc("Weyland Corp", "Covenant2017", zh="维兰德公司"),
         subc("Weyland Corp", "Prometheus2012", zh="维兰德工业"),
         subc("Weyland (bare)", "AlienEarth2025", zh="威兰")],
        tier="T-BILINGUAL", category="franchise-corp",
        aliases=["韦兰德工业", "维兰德公司", "威兰"],
        surname_crossref=["Weyland-Yutani", "Weyland-Yutani Corporation", "Peter Weyland"],
        note="Level 1 ch.1 also writes 韦兰德工业 for Weyland Industries — same company, second "
             "English name. Levels 2 and 5 both say 维兰德/威兰; level 1 wins.")

    OV["Yutani Corporation"] = O(
        "汤谷公司", 1,
        "Level 1. Yutani is the Japanese surname 湯谷, and levels 1 and 2 both render it "
        "semantically as 汤谷 instead of transliterating it. 尤坦尼 exists only at level 7.",
        [fanc("Yutani Corporation", zh="汤谷公司"),
         subc("Yutani", "AlienEarth2025", zh="汤谷"),
         refc("尤坦尼公司", "zh.wikipedia (level 7)", "The Phase-0 spine value, now superseded.")],
        tier="T-BILINGUAL", category="franchise-corp",
        surname_crossref=["Weyland-Yutani"])

    OV["Nostromo"] = O(
        "诺斯特罗莫号", 1,
        "Level 1 writes USCSS诺斯特罗莫号 with the English gloss spliced inside the Chinese name, so "
        "the pairing is explicit rather than inferred. ⚠ Level 2 (Romulus 2024) says 诺史莫 — the "
        "same form as zh.wikipedia — and level 1 still wins: the ladder is not a popularity contest, "
        "level 1 is the translation of this book. Logged in _meta.ladder_deviations so the choice "
        "is visible rather than buried.",
        [fanc("USCSS Nostromo", zh="USCSS诺斯特罗莫号"),
         subc("Nostromo", "Romulus2024", zh="诺史莫"),
         isoc("Nostromo", zh="Nostromo号"),
         subc("Nostromo", CNSCG, zh="诺斯都罗莫"),
         refc("诺史莫号", "zh.wikipedia (level 7)", "The Phase-0 spine value, now superseded.")],
        tier="T-BILINGUAL", category="franchise-ship",
        aliases=["诺史莫号", "诺斯都罗莫号", "诺斯托罗莫号"],
        note="Ship-name pattern: <音译>号. Level 1 glues the Latin prefix (USCSS / USS / UAS) to the "
             "front of the Chinese name; this glossary keeps the prefix out of the term and freezes "
             "it as its own key. Level 3 (Isolation) keeps the whole hull name in Latin and appends "
             "号 — a third pattern, not adopted.")

    OV["Round"] = O(
        "轮", 1,
        "Level 1 ch.1's time-unit table glosses it inline: 轮 (Round) / 5-10秒 / 战斗. Level 4 has "
        "no bare 'Round' key — only ALIENRPG.OneRound '一回合' — so the level-4 competitor is 回合, "
        "and level 1 beats it. ⚠ RUNTIME: ALIENRPG.OneRound is a case in the healTime switch; 一回合 "
        "must become 一轮 in the SAME commit as the RollTable cell. See _meta.lockstep_literals.",
        [fanc("Round", zh="轮"), fanc("combat round", zh="战斗轮"), langc("OneRound", zh="一回合")],
        tier="T-PLAIN", category="yze-mechanics", aliases=["回合"],
        lockstep="ALIENRPG.OneRound")

    OV["Shift"] = O(
        "班", 1,
        "Level 1 ch.1's time-unit table glosses it inline: 班 (Shift) / 5-10小时 / 恢复. Level 4 "
        "offers 一轮班 (ALIENRPG.OneShift) and stratum-B prose offers 班次; level 1 beats both, and "
        "that CLOSES dispute D8-shift. ⚠⚠ TWO DIFFERENT THINGS SHARE THIS NAME: the RollTable "
        "heal-time CELL must stay the bare English literal 'Shift' (actor.mjs:1890) or the crit roll "
        "throws, while the lang key ALIENRPG.Shift is an OUTPUT and SHOULD become 班 "
        "(actor.mjs:1891). The per-field split is in glossary_alien.byrole.json.",
        [fanc("Shift", zh="班"), fanc("shift (body-text variant)", zh="班次"),
         langc("OneShift", zh="一轮班")],
        tier="T-PLAIN", category="yze-mechanics", aliases=["班次", "轮班"],
        lockstep="ALIENRPG.Shift / ALIENRPG.OneShift", dispute=None)

    OV["Synthetic"] = O(
        "合成人", 1,
        "Levels 1 AND 2 agree, which is as strong as this project's evidence gets. Level 1 ch.1: "
        "'合成人扮演上帝'; level 2 Alien: Earth 2025 renders synthetic 合成人, and English-gating puts "
        "合成人 in 3 lineages. Level 4's 生化人 ('bio-chemical human') is the corpus's ANDROID word, "
        "not its synthetic word: English-gated, 生化人 pairs with android/droid 9 times out of 11 "
        "and with 'synthetic' exactly once. CLOSES dispute D5-synthetic.",
        [fanc("Synthetic", zh="合成人"), fanc("synthetic people (ch1 prose)", zh="合成人"),
         subc("synthetic", "AlienEarth2025", zh="合成人"),
         subc("synthetic", "Romulus2024", zh="合成人"),
         subc("synthetic", CNSCG, zh="合成人"),
         langc("Synthetic", zh="生化人"),
         subc("artificial person", CNSCG, zh="人造人")],
        tier="T-PLAIN", category="career", aliases=["生化人"], dispute=None,
        three_way_split="Synthetic 合成人 / Android 仿生人 / Robot 机器人 — three English words, "
                        "three Chinese words. Never collapse them.")

    OV["Android"] = O(
        "仿生人", 3,
        "The owner's §1.4 three-way split, and level 3 supplies the HARD evidence: Alien: "
        "Isolation's official localisation key-gates ANDROID -> 仿生人 on 11 keys of the form "
        "*_KILLED_BY_ANDROID -> 被仿生人所杀, with 220 keys carrying the value. ⚠ Level 1 renders "
        "Android 机器人 (ch.2 ×18) — but then level 1 has NO word left for Robot, which the English "
        "does distinguish. Deviating from level 1 here IS the split, and it is the owner's own "
        "§1.4 wording.",
        [isoc("android / synthetic", zh="仿生人"), fanc("Android", zh="机器人"),
         fanc("android (David-series, stray)", zh="仿生人"),
         subc("android", CNSCG, zh="生化人")],
        tier="T-PLAIN", category="franchise-vocab", aliases=["机器人", "生化人"],
        beats_a_higher_level=1,
        three_way_split="Synthetic 合成人 / Android 仿生人 / Robot 机器人.",
        note="Level 1 is internally inconsistent here and the miner says so: two timeline lines "
             "apart, the same David series is 机器人 in one and 仿生人 in the other. The Isolation "
             "key gate is the only HARD English-side evidence anywhere in the corpus for this word.")

    OV["Robot"] = O(
        "机器人", 2,
        "Level 2 (Alien: Earth 2025) plus levels 5 and 6, en-gated 14 rows over 3 lineages, all "
        "机器人. Level 1 spends 机器人 on Android instead, which is precisely the collapse the §1.4 "
        "split forbids.",
        [subc("robot", "AlienEarth2025", zh="机器人"), subc("robot", "Prometheus2012", zh="机器人"),
         subc("robot", CNSCG, zh="机器人")],
        tier="T-PLAIN", category="franchise-vocab",
        three_way_split="Synthetic 合成人 / Android 仿生人 / Robot 机器人.")

    OV["Artificial Person"] = O(
        "人造人", 2,
        "Level 2 (Romulus 2024) and level 6 agree, en-gated 4 rows over 2 lineages. It is the "
        "in-universe euphemism Bishop uses and stays distinct from all three of Synthetic / Android "
        "/ Robot.",
        [subc("artificial person", "Romulus2024", zh="人造人"),
         subc("artificial person", CNSCG, zh="人造人")],
        tier="T-PLAIN", category="franchise-vocab")

    OV["Cinematic Mode"] = O(
        "电影模式", 1,
        "Level 1 ch.1. Level 4 is silent. The Phase-0 剧本模式 had no source at all.",
        [fanc("cinematic play", zh="电影模式")],
        tier="T-PLAIN", category="yze-mechanics", aliases=["剧情模式"],
        note="Level 1 ch.2 once writes 剧情模式 for the same mode ('在剧情模式中每幕一次'). The ch.1 "
             "definitional use wins; 剧情模式 is recorded as an alias so a sweep does not 'fix' it "
             "into a second term.")
    OV["Campaign Mode"] = O(
        "战役模式", 1, "Level 1 ch.1 — same table as Cinematic Mode. Unchanged in value, upgraded "
        "from an unsourced Phase-0 guess to a level-1 attestation.",
        [fanc("campaign play", zh="战役模式")],
        tier="T-PLAIN", category="yze-mechanics")

    # =======================================================================
    # 2. TERMS THE NEW LADDER RE-DECIDES
    # =======================================================================
    OV["Wits"] = O(
        "机智", 4,
        "OWNER-PINNED EXCEPTION (PROJECT.md §1.4): level 4 beats level 1 here. Level 1 says 智力, "
        "and level 1's own gloss for Wits contains 智力 in its ordinary sense ('感官感知、智力和理智'), "
        "so adopting it would make the term and its definition the same word. 机智 is what every "
        "character sheet in Foundry shows today.",
        [langc("AbilityWit", zh="机智"), fanc("Wits", zh="智力")],
        tier="T-PLAIN", category="attribute", owner_pinned=True, beats_a_higher_level=1)
    OV["Empathy"] = O(
        "共情", 4,
        "OWNER-PINNED EXCEPTION (PROJECT.md §1.4): level 4 beats level 1 here. Level 1 says 同理心, "
        "which likewise appears inside its own gloss. Both are two characters; 共情 is what players "
        "see in Foundry today.",
        [langc("AbilityEmp", zh="共情"), fanc("Empathy", zh="同理心")],
        tier="T-PLAIN", category="attribute", owner_pinned=True, beats_a_higher_level=1)

    OV["Strength"] = O("力量", 1, "Levels 1 and 4 agree.",
                       [fanc("Strength", zh="力量"), langc("AbilityStr", zh="力量")],
                       tier="T-PLAIN", category="attribute")
    OV["Agility"] = O("敏捷", 1, "Levels 1 and 4 agree.",
                      [fanc("Agility", zh="敏捷"), langc("AbilityAgl", zh="敏捷")],
                      tier="T-PLAIN", category="attribute")

    # --- skills (T-EXACT: the Item name and the lang value are ONE string) ---
    OV["Heavy Machinery"] = O(
        "重型机械", 1,
        "Level 1 beats level 4's bare 机械. ⚠ T-EXACT: adopting this REQUIRES writing 重型机械 into "
        "ALIENRPG.SkillheavyMach as well, or the Item name and the lang value stop being one string.",
        [fanc("Heavy Machinery", zh="重型机械"), langc("SkillheavyMach", zh="机械")],
        tier="T-EXACT", category="skill", lang_key="ALIENRPG.SkillheavyMach")
    OV["Close Combat"] = O(
        "近战", 1,
        "Level 1 beats level 4's 肉搏. This also DISSOLVES dispute D2-engaged from the other side: "
        "level 4 spent 近战 on BOTH the Engaged range band and the Melee weapon type, and once "
        "近战 belongs to the Close Combat skill neither of the other two can keep it. Engaged stays "
        "接战 and Melee takes the freed 肉搏.",
        [fanc("Close Combat", zh="近战"), langc("SkillcloseCbt", zh="肉搏")],
        tier="T-EXACT", category="skill", lang_key="ALIENRPG.SkillcloseCbt",
        collision_split=["Engaged", "Melee"])
    OV["Melee"] = O(
        "肉搏", 4,
        "Freed by Close Combat taking 近战. 肉搏 is level 4's own word for hand-to-hand and is a "
        "correct rendering of the Melee WEAPON TYPE (ALIENRPG.WepTypeMelee), which is what this key "
        "actually labels.",
        [langc("WepTypeMelee", zh="近战"), langc("SkillcloseCbt", zh="肉搏")],
        tier="T-PLAIN", category="range", collision_split=["Close Combat", "Engaged"],
        note="⚠ The lang value ALIENRPG.WepTypeMelee must be rewritten from 近战 to 肉搏 in the same "
             "commit, or the weapon-type dropdown and the skill list both read 近战.")
    OV["Engaged"] = O(
        "接战", 4,
        "Level-4 stratum-A prose (Panic11, Panic13). CLOSES dispute D2-engaged: the dispute existed "
        "only because 近战 was doing double duty for Engaged and Melee, and Close Combat has now "
        "taken 近战 outright.",
        [langc("Engaged", zh="近战"),
         cand("接战", 4, "system lang/cn.json stratum A (TI-130 human) — panic-table prose",
              "en-gated 2 of 3 rows matching 'ENGAGED|Engaged'", "lang_key_pair",
              "cn.json Panic11 '如果你的接战范围内有敌人，你可以进行撤退检定（见P93）'; Panic13 "
              "'若逃跑时你的接战范围内有敌人'")],
        tier="T-PLAIN", category="range", dispute=None, collision_split=["Close Combat", "Melee"])
    OV["Observation"] = O(
        "侦察", 1, "Level 1 beats level 4's 观察. ⚠ T-EXACT: ALIENRPG.Skillobservation must be "
                   "rewritten to match.",
        [fanc("Observation", zh="侦察"), langc("Skillobservation", zh="观察")],
        tier="T-EXACT", category="skill", lang_key="ALIENRPG.Skillobservation", aliases=["观察"])
    OV["Survival"] = O(
        "生存", 1, "Level 1 beats level 4's 求生. ⚠ T-EXACT: ALIENRPG.Skillsurvival must be "
                   "rewritten to match.",
        [fanc("Survival", zh="生存"), langc("Skillsurvival", zh="求生")],
        tier="T-EXACT", category="skill", lang_key="ALIENRPG.Skillsurvival", aliases=["求生"])
    OV["Ranged Combat"] = O(
        "远程战斗", 1,
        "Level 1 beats level 4's 射击. 射击 is 'shooting' and under-covers a skill that also runs "
        "thrown weapons. ⚠ T-EXACT: ALIENRPG.SkillrangedCbt must be rewritten to match.",
        [fanc("Ranged Combat", zh="远程战斗"), langc("SkillrangedCbt", zh="射击")],
        tier="T-EXACT", category="skill", lang_key="ALIENRPG.SkillrangedCbt", aliases=["射击"])
    OV["Manipulation"] = O(
        "操控", 1, "Level 1 beats level 4's 操纵 — same sound, and 操控 is the ordinary modern form. "
                   "⚠ T-EXACT: ALIENRPG.Skillmanipulation must be rewritten to match.",
        [fanc("Manipulation", zh="操控"), langc("Skillmanipulation", zh="操纵")],
        tier="T-EXACT", category="skill", lang_key="ALIENRPG.Skillmanipulation", aliases=["操纵"])
    for _sk, _key in [("Stamina", "Skillstamina"), ("Medical Aid", "SkillmedicalAid"),
                      ("Mobility", "Skillmobility"), ("Command", "Skillcommand"),
                      ("Piloting", "Skillpiloting")]:
        OV[_sk] = O(ctx["LANG_CN"]["ALIENRPG." + _key], 1,
                    "Levels 1 and 4 agree; nothing to decide. Upgraded from a level-4-only "
                    "attestation to a level-1 + level-4 concurrence.",
                    [fanc(_sk), langc(_key)],
                    tier="T-EXACT", category="skill", lang_key="ALIENRPG." + _key)

    # --- Comtech: the one skill where level 1 is demonstrably narrower ---
    OV["Comtech"] = O(
        "科技", 4,
        "PROVISIONAL, and the only skill where this rebuild does NOT take level 1. Level 1 renders "
        "Comtech 计算机科学 ('computer science'), which excludes the comms and electronics half of a "
        "skill whose English name is COMmunications + TECHnology — the miner flags the same thing. "
        "Level 4's 科技 is vague but not wrong. Adopting a demonstrably narrower rendering because "
        "it sits one rung higher would be following the ladder off a cliff, so this goes to "
        "disputes instead of into the spine. See D10-comtech.",
        [langc("Skillcomtech", zh="科技"), fanc("Comtech", zh="计算机科学")],
        tier="T-EXACT", category="skill", lang_key="ALIENRPG.Skillcomtech",
        dispute="D10-comtech", beats_a_higher_level=1)

    # --- YZE mechanics ---
    OV["Push"] = O(
        "追骰", 1,
        "Level 1 uses 追骰 11 times in ch.2 and never once writes 推骰 or 加骰. Level 4's 孤注一掷 "
        "('stake everything on one throw') is an idiom, not a term, and cannot take a modifier the "
        "way 追骰 can.",
        [fanc("push (a roll)", zh="追骰"), langc("Push", zh="孤注一掷")],
        tier="T-PLAIN", category="yze-mechanics", aliases=["孤注一掷"])
    OV["Health"] = O(
        "生命值", 1,
        "Level 1. CLOSES dispute D4-health, which had 生命 / 健康 / 生命值 with the count favouring a "
        "stratum-B rendering. Level 1 picks 生命值 and outranks every side of the old argument.",
        [fanc("Health", zh="生命值"), langc("Health", zh="生命"),
         langc("healthDamage", stratum="B")],
        tier="T-PLAIN", category="yze-mechanics", dispute=None, aliases=["生命", "健康"])
    OV["Resolve"] = O(
        "精神强度", 1,
        "Level 1. Level 4 leaves ALIENRPG.Resolve NULL, so there is no level-4 competitor at all — "
        "the Phase-0 决心 was an unsourced coinage.",
        [fanc("Resolve", zh="精神强度")],
        tier="T-PLAIN", category="yze-mechanics", lang_key="ALIENRPG.Resolve")
    OV["Resolve Modifier"] = O(
        "精神强度修正", "C", "Derived from Resolve 精神强度 + 修正, level 4's own modifier word "
                             "(ALIENRPG.StressMod 压力修正, ALIENRPG.BaseMod 基础修正). "
                             "ALIENRPG.ResolveMod is NULL in cn.json.",
        [convc("精神强度修正", "Resolve=精神强度 (level 1) + 修正 (level 4 ALIENRPG.BaseMod 基础修正)")],
        tier="T-PLAIN", category="yze-mechanics", derived_from="Resolve")
    OV["Stress Level"] = O(
        "压力等级", 1,
        "Level 1. CLOSES dispute D3-stress-level, which was 压力水平 7 vs 压力等级 2 on a level-4 "
        "count. Level 1 settles it for 压力等级, and that also makes Stress Level and Panic Level "
        "share the head noun 等级 — the consistency argument the dispute could not resolve on count.",
        [fanc("stress level", zh="压力等级")],
        tier="T-PLAIN", category="yze-mechanics", dispute=None, aliases=["压力水平"])
    OV["Panic Roll"] = O(
        "恐慌检定", 1,
        "Level 1. CLOSES dispute D1-panic-roll: level 4's panic-table prose renders the phrase "
        "混乱检定 ('chaos check'), which is not what panic means, and level 1 confirms the regular "
        "form. Whichever side won, level 4's Panic8 '直到混乱结束' had to be rewritten; it still does.",
        [fanc("panic roll", zh="恐慌检定"), langc("Panicked", zh="恐慌")],
        tier="T-PLAIN", category="yze-mechanics", dispute=None,
        note="Applying this also requires rewriting cn.json Panic12/13/14 (混乱检定) and Panic8 "
             "(直到混乱结束).")
    OV["Base Dice"] = O(
        "基础骰子", 1, "Level 1 ch.1. Level 4 has no Base Dice key (ALIENRPG.Base is the bare word "
                       "'Base'), so level 1 is uncontested.",
        [fanc("base dice", zh="基础骰子")], tier="T-PLAIN", category="yze-mechanics",
        aliases=["基础骰"])
    OV["Base Die"] = O("基础骰子", 1, "Singular of Base Dice; one Chinese string covers both.",
                       [fanc("base dice", zh="基础骰子")], tier="T-PLAIN",
                       category="yze-mechanics", derived_from="Base Dice")
    OV["Stress Dice"] = O("压力骰子", 1, "Level 1 ch.1. No level-4 key exists.",
                          [fanc("stress dice", zh="压力骰子")], tier="T-PLAIN",
                          category="yze-mechanics", aliases=["压力骰"])
    OV["Stress Die"] = O("压力骰子", 1, "Singular of Stress Dice.",
                         [fanc("stress dice", zh="压力骰子")], tier="T-PLAIN",
                         category="yze-mechanics", derived_from="Stress Dice")
    OV["Buddy"] = O(
        "朋友", 1, "Level 1 beats level 4's colloquial 哥们. Level 1's Buddy/Rival pair (朋友/敌人) "
                   "is parallel and gender-neutral; level 4's (哥们/对头) is neither.",
        [fanc("Buddy", zh="朋友"), langc("relOne")], tier="T-PLAIN",
        category="yze-mechanics", aliases=["哥们"])
    OV["Rival"] = O(
        "敌人", 1, "Level 1, and the other half of the Buddy/Rival pair. 敌人 is broader than "
                   "'rival', but level 1 chose the pair deliberately and level 4's 对头 is regional "
                   "colloquial.",
        [fanc("Rival", zh="敌人"), langc("relTwo")], tier="T-PLAIN",
        category="yze-mechanics", aliases=["对头"])
    OV["Personal Agenda"] = O(
        "个人任务", 1, "Level 1 beats level 4's 个人目标.",
        [fanc("Personal Agenda", zh="个人任务"), langc("PersonalAgenda", zh="个人目标")],
        tier="T-PLAIN", category="yze-mechanics", aliases=["个人目标"])
    OV["Agenda"] = O(
        "任务", "C", "Derived from Personal Agenda 个人任务 (level 1) so the bare noun and the "
                     "compound stay one family. Level 4's ALIENRPG.AgendaStory says 目标, which "
                     "would split them.",
        [convc("任务", "head noun of Personal Agenda 个人任务 (level 1)"),
         langc("AgendaStory", zh="目标")],
        tier="T-PLAIN", category="yze-mechanics", derived_from="Personal Agenda")
    OV["Signature Item"] = O(
        "标志性物品", 1, "Level 1 beats level 4's 标志物品 (no 性).",
        [fanc("Signature Item", zh="标志性物品"), langc("SignatureItem", zh="标志物品")],
        tier="T-BILINGUAL", category="yze-mechanics", aliases=["标志物品"])
    OV["Signature Weapon"] = O(
        "标志性武器", "C", "Derived from Signature Item 标志性物品 (level 1).",
        [convc("标志性武器", "Signature Item=标志性物品 (level 1) with 物品->武器")],
        tier="T-BILINGUAL", category="yze-mechanics", derived_from="Signature Item")
    OV["Supply"] = O(
        "余量", 1, "Level 1 renders the supply rating 余量 ('remaining amount'), which is what the "
                   "supply dial actually tracks. Level 4's 补给 is the noun 'supplies', not the "
                   "rating.",
        [fanc("supply rating", zh="余量"), langc("Supply", zh="补给骰")],
        tier="T-PLAIN", category="yze-mechanics", aliases=["补给"])
    OV["Supply Roll"] = O(
        "余量检定", 1, "Level 1. Level 4 says 补给骰, which names the die rather than the roll.",
        [fanc("Supply Roll", zh="余量检定"), langc("Supply", zh="补给骰")],
        tier="T-PLAIN", category="yze-mechanics", derived_from="Supply", aliases=["补给检定"])
    OV["Air"] = O(
        "空气", 1, "Level 1's consumables list is 空气 / 弹药 / 电力. Level 4 says 氧气 ('oxygen'), "
                   "which is narrower than the English and wrong for a pressure suit's air supply.",
        [fanc("Air (consumable)", zh="空气"), langc("Air", zh="氧气")],
        tier="T-PLAIN", category="yze-mechanics", aliases=["氧气"])
    OV["Air Supply"] = O(
        "空气供应", "C", "Derived from Air 空气 (level 1) + level 4's own 供应. Level 4 ships "
                          "氧气供应; the head noun must follow Air.",
        [convc("空气供应", "Air=空气 (level 1) + 供应 (level 4 ALIENRPG.AirSupply 氧气供应)"),
         langc("AirSupply", zh="氧气供应")],
        tier="T-PLAIN", category="yze-mechanics", derived_from="Air")
    OV["Power"] = O(
        "电力", 1, "Level 1's consumables list. Level 4's 动力 is 'motive power' and does not read "
                   "as a consumable charge.",
        [fanc("Power (consumable)", zh="电力"), langc("Power", zh="动力")],
        tier="T-PLAIN", category="yze-mechanics", aliases=["动力"])
    OV["Broken"] = O("濒死", 1, "Levels 1 and 4 agree; upgraded to a level-1 attestation.",
                     [fanc("Broken", zh="濒死")], tier="T-PLAIN", category="yze-condition")
    OV["Critical Injury"] = O("重伤", 1, "Level 1 ch.2.",
                              [fanc("critical injury", zh="重伤")],
                              tier="T-PLAIN", category="yze-mechanics")
    OV["Death Roll"] = O("死亡检定", 1, "Level 1 ch.2.", [fanc("death roll", zh="死亡检定")],
                         tier="T-PLAIN", category="yze-mechanics")
    OV["Initiative"] = O("先攻", 1, "Level 1 ch.1.", [fanc("initiative", zh="先攻")],
                         tier="T-PLAIN", category="yze-mechanics")
    OV["Encumbrance"] = O("负重", 1, "Level 1 ch.2.", [fanc("Encumbrance", zh="负重")],
                          tier="T-PLAIN", category="yze-mechanics")
    OV["Panic"] = O("恐慌", 1, "Levels 1 and 4 agree.",
                    [fanc("Panic", zh="恐慌"), langc("Panicked", zh="恐慌")],
                    tier="T-PLAIN", category="yze-mechanics")
    OV["Stress"] = O("压力", 1, "Levels 1 and 4 agree.",
                     [fanc("Stress", zh="压力"), langc("Stress", zh="压力")],
                     tier="T-PLAIN", category="yze-mechanics")
    OV["Story Points"] = O("故事点", 1, "Levels 1 and 4 agree.", [fanc("Story Points", zh="故事点")],
                           tier="T-PLAIN", category="yze-mechanics")
    OV["Career"] = O("职业", 1, "Levels 1 and 4 agree.", [fanc("Career", zh="职业")],
                     tier="T-PLAIN", category="yze-mechanics")
    OV["Consumables"] = O("消耗品", 1, "Levels 1 and 4 agree.",
                          [fanc("Consumables", zh="消耗品"), langc("Consumables", zh="消耗品")],
                          tier="T-PLAIN", category="yze-mechanics")
    OV["Talent"] = O("天赋", 1, "Levels 1 and 4 agree.",
                     [fanc("Talents", zh="天赋"), langc("Talents", zh="天赋")],
                     tier="T-BILINGUAL", category="yze-mechanics")
    OV["Talents (Career)"] = O(
        "职业天赋", 1, "Level 1 ch.2. Replaces the Phase-0 parenthesised 天赋（职业）, which was a "
                       "shape invented for the folder listing rather than a term.",
        [fanc("career talent", zh="职业天赋")], tier="T-PLAIN", category="pack-structure")
    OV["Talents (General)"] = O(
        "通用天赋", 1, "Levels 1 and 4 agree (ALIENRPG.GeneralTalent 通用天赋).",
        [fanc("general talent", zh="通用天赋"), langc("GeneralTalent", zh="通用天赋")],
        tier="T-PLAIN", category="pack-structure")

    # --- careers ---
    OV["Colonial Marine"] = O(
        "殖民海军陆战队", 1,
        "Level 1, ch.1 ×6 and ch.2 ×2 (against 殖民地海军陆战队 ×2 in the same file). Level 2 "
        "(Romulus 2024) says 殖民地陆战队 and level 6 says 殖民陆战队; level 1 wins.",
        [fanc("Colonial Marine (career)", zh="殖民海军陆战队"),
         subc("Colonial Marine", "Romulus2024", zh="殖民地陆战队"),
         subc("Colonial Marine", CNSCG, zh="殖民陆战队")],
        tier="T-PLAIN", category="career", aliases=["殖民陆战队", "殖民地海军陆战队"],
        note="⚠ 殖民海军 (Colonial NAVY) and 殖民海军陆战队 (Colonial MARINES) differ by two "
             "characters and one is a prefix of the other. Any search-and-replace over 殖民海军 will "
             "corrupt every Colonial Marines string. Always anchor on the longer form first.")
    OV["Colonial Marines"] = O(
        "殖民海军陆战队", 1, "Same string as Colonial Marine; the English source uses both.",
        [fanc("Colonial Marines / USCMC", zh="殖民海军陆战队")],
        tier="T-BILINGUAL", category="marine", derived_from="Colonial Marine",
        note="See Colonial Marine for the 殖民海军 prefix hazard.")
    OV["Colonial Marshal"] = O(
        "殖民地执法官", 1, "Levels 1 and 3 agree independently — the fan translation and the "
                           "official Isolation localisation reach the same string.",
        [fanc("Colonial Marshal (career)", zh="殖民地执法官"),
         isoc("Colonial Marshal", zh="殖民地执法官")],
        tier="T-PLAIN", category="career")
    OV["Company Agent"] = O("公司代理人", 1, "Level 1 ch.2 career list.",
                            [fanc("Company Agent (career)", zh="公司代理人")],
                            tier="T-PLAIN", category="career", aliases=["公司代表"])
    OV["Kid"] = O("儿童", 1, "Level 1 ch.2 career list; replaces the Phase-0 孩童.",
                  [fanc("Kid (career)", zh="儿童")], tier="T-PLAIN", category="career")
    OV["Medic"] = O("医生", 1, "Level 1 ch.2 career list; replaces the Phase-0 医护人员.",
                    [fanc("Medic (career)", zh="医生")], tier="T-PLAIN", category="career",
                    aliases=["医务官"])
    OV["Officer"] = O(
        "飞行官", 1,
        "Level 1 ch.2 career list. The English is bare 'Officer' and 飞行官 imports a 'flight' the "
        "English does not have — but level 1 is internally motivated: the same chapter renders the "
        "career's own licence, Commercial Flight Officer License, as ICC商业飞行员执照. Recorded as "
        "an over-specification rather than silently smoothed away.",
        [fanc("Officer (career)", zh="飞行官")], tier="T-PLAIN", category="career",
        aliases=["长官"])
    OV["Pilot"] = O(
        "驾驶员", 1, "Level 1 ch.2 career list. Note the deliberate contrast level 1 draws: the "
                     "SKILL Piloting is 驾驶 and the CAREER Pilot is 驾驶员.",
        [fanc("Pilot (career)", zh="驾驶员")], tier="T-PLAIN", category="career")
    OV["Roughneck"] = O("杂工", 1, "Level 1 ch.2 career list; replaces the Phase-0 钻井工.",
                        [fanc("Roughneck (career)", zh="杂工")], tier="T-PLAIN", category="career")
    OV["Scientist"] = O("科学家", 1, "Level 1 ch.2 career list.",
                        [fanc("Scientist (career)", zh="科学家")], tier="T-PLAIN", category="career")

    # --- ranks, roles, ops ---
    OV["Captain"] = O(
        "舰长", 2,
        "Level 2 (Alien: Earth 2025) says 舰长, and level 5 splits — Covenant 2017 舰长, Prometheus "
        "2012 船长. Level 6 says 船长. Level 4 has no Captain key at all, so the highest rung that "
        "speaks is level 2. 船长 is kept as an alias because it is the better register for a "
        "commercial hauler, which is what most Alien RPG ships are.",
        [subc("Captain", "AlienEarth2025", zh="舰长"), subc("Captain", "Covenant2017", zh="舰长"),
         subc("Captain", "Prometheus2012", zh="船长"), subc("Captain", CNSCG, zh="船长")],
        tier="T-PLAIN", category="marine", aliases=["船长"])
    OV["Colonist"] = O(
        "殖民者", 1, "Level 1 ch.1, with level 5 (Covenant 2017) agreeing. Level 6's 移民 "
                     "('immigrant') was the Phase-0 value and loses two rungs.",
        [fanc("colonist", zh="殖民者"), subc("colonist", "Covenant2017", zh="殖民者"),
         subc("colonist", CNSCG, zh="移民")],
        tier="T-PLAIN", category="ops", aliases=["移民"])
    OV["Colonial Administration"] = O(
        "殖民管理局", 1, "Level 1 ch.1; replaces the Phase-0 coinage 殖民地行政机构.",
        [fanc("Colonial Administration", zh="殖民管理局")], tier="T-PLAIN", category="marine")
    OV["Sergeant"] = O(
        "中士", 2, "Level 2 (Alien: Earth 2025) and level 5 (Covenant 2017) both say 中士. Level 6's "
                   "士官长 is Taiwan-flavoured and is on the project's known-bad list of register "
                   "problems.",
        [subc("Sergeant", "AlienEarth2025", zh="中士"), subc("Sergeant", "Covenant2017", zh="中士"),
         subc("Sergeant", CNSCG, zh="士官长")], tier="T-PLAIN", category="marine")
    OV["Science Officer"] = O(
        "科学官", 2, "Level 2 (Romulus 2024). Level 5's 首席科学家 and level 6's 科学研究员 both "
                     "over-translate a shipboard post as a job title.",
        [subc("Science Officer", "Romulus2024", zh="科学官"),
         subc("Science Officer", "Covenant2017", zh="首席科学家"),
         subc("Science Officer", CNSCG, zh="科学研究员")],
        tier="T-PLAIN", category="marine")
    OV["Quarantine"] = O(
        "隔离", 2,
        "Level 2 (Alien: Earth 2025) and level 5 (Covenant 2017, Prometheus 2012) all say 隔离, "
        "en-gated across 4 lineages. The Phase-0 检疫 rested on level 6 alone. Level 3 (Isolation) "
        "has the compound 检疫隔离 but only in ungated terminal prose.",
        [subc("quarantine", "AlienEarth2025", zh="隔离"),
         subc("quarantine", "Covenant2017", zh="隔离"),
         subc("quarantine", "Prometheus2012", zh="隔离"),
         isoc("quarantine", zh="检疫隔离"), subc("quarantine", CNSCG, zh="检疫")],
        tier="T-PLAIN", category="ops", aliases=["检疫"])
    OV["Terraforming"] = O(
        "地球化", 1,
        "Level 1 is INTERNALLY SPLIT — the ch.1 body says 地形改造 ('landform modification', which "
        "is not what terraforming means) while the ch.1 timeline says 地球化 for the same Hadley's "
        "Hope colony. Level 2 breaks the tie: Romulus 2024 and Prometheus 2012 both say 地球化. So "
        "this is not a ladder deviation — it is choosing between two level-1 readings with level 2 "
        "as the tiebreak.",
        [fanc("terraforming", zh="地形改造"),
         subc("terraforming", "Romulus2024", zh="地球化"),
         subc("terraforming", "Prometheus2012", zh="地球化"),
         subc("terraforming", "Covenant2017", zh="整地"),
         subc("terraforming", CNSCG, zh="改造")],
        tier="T-PLAIN", category="ops", aliases=["地形改造", "星球改造"])
    OV["Hypersleep"] = O(
        "深度睡眠", 1, "Level 1 ch.1, glossed inline via hypersleep pod 深度睡眠舱. Level 3's 深度休眠 "
                       "is the nearest independent form and is ungated; levels 5 and 6 give 冬眠 "
                       "and 长眠.",
        [fanc("hypersleep / stasis", zh="深度睡眠"), isoc("hypersleep / cryo", zh="深度休眠"),
         subc("hypersleep", "Covenant2017", zh="冬眠"), subc("hypersleep", CNSCG, zh="长眠")],
        tier="T-PLAIN", category="ops", aliases=["休眠", "冷冻睡眠", "长眠"],
        note="Level 1 uses three renderings inside ch.1 alone — 深度睡眠 / 休眠 / 冷冻. The glossed "
             "one wins; 休眠 is kept for Stasis so the two English words stay two Chinese words.")
    OV["Stasis"] = O(
        "休眠", 1, "Level 1's secondary rendering, kept deliberately distinct from Hypersleep "
                   "深度睡眠. Level 6's 静态平衡 is on the project's known-bad list and is not "
                   "eligible.",
        [fanc("hypersleep / stasis", zh="深度睡眠"),
         cand("休眠", 1, "《异形RPG：进化版》中文连载 ch1 (level 1)",
              "occurrence, en-gated by construction",
              "occurrence_in_translated_book",
              "ch1 '不必在休眠状态中花费太多时间' and '补偿他们在休眠中失去的时间'"),
         subc("stasis", CNSCG, zh="静态平衡")],
        tier="T-PLAIN", category="ops", known_bad_rejected=["静态平衡"])
    OV["Lifeboat"] = O(
        "救生艇", 5, "Level 5 (Prometheus 2012). Level 6's 太空梭 is Taiwan-flavoured ('space "
                     "shuttle') and is not a lifeboat.",
        [subc("lifeboat", "Prometheus2012", zh="救生艇"), subc("lifeboat", CNSCG, zh="太空梭")],
        tier="T-PLAIN", category="ops")
    OV["EEV"] = O(
        "紧急逃生舱", 1,
        "Level 1 glosses EEV as 逃生舱; level 2 (Romulus 2024) supplies the 紧急 that the E in EEV "
        "stands for. Composed so that EEV / Escape Vehicle / Escape Pod are three distinct strings "
        "— the Phase-0 file had Escape Pod and Escape Vehicle BOTH at 逃生舱.",
        [fanc("EEV (Emergency Escape Vehicle)", zh="逃生舱"),
         subc("EEV", "Romulus2024", zh="紧急逃生艇"), subc("EEV", CNSCG, zh="逃生艇")],
        tier="T-PLAIN", category="ops", collision_split=["Escape Vehicle", "Escape Pod"])
    OV["Escape Vehicle"] = O(
        "逃生舱", 1, "Level 1's own gloss of the Emergency Escape Vehicle.",
        [fanc("EEV (Emergency Escape Vehicle)", zh="逃生舱")],
        tier="T-BILINGUAL", category="hardware", collision_split=["EEV", "Escape Pod"])
    OV["Escape Pod"] = O(
        "逃生艇", 6, "Level 6, en-gated. Split off 逃生舱 so that the three escape-craft keys are "
                     "three strings.",
        [subc("EEV", CNSCG, zh="逃生艇")], tier="T-PLAIN", category="ops",
        collision_split=["EEV", "Escape Vehicle"])
    OV["Airlock"] = O(
        "气闸", 1, "Level 1 ch.1. Level 3 (Isolation) has 气闸舱, gated by the natural-language key "
                   "[TEXT_Get to the Airlock]; the bare 气闸 is the term and 气闸舱 the compartment.",
        [fanc("airlock", zh="气闸"), isoc("airlock", zh="气闸舱"),
         subc("airlock", "Prometheus2012", zh="气闸舱"),
         subc("airlock", "Romulus2024", zh="气锁室")],
        tier="T-PLAIN", category="ops", aliases=["气闸舱"])
    OV["Air Lock"] = O("气闸", 1, "Spelling variant of Airlock; one Chinese string.",
                       [fanc("airlock", zh="气闸")], tier="T-PLAIN", category="ops",
                       derived_from="Airlock")
    OV["Reactor"] = O("反应堆", 3, "Level 3, key-gated (AI_UI_REACTOR_CORE -> 反应堆堆芯).",
                      [isoc("reactor", zh="反应堆")], tier="T-BILINGUAL", category="hardware")
    OV["Flamethrower"] = O(
        "火焰喷射器", 3,
        "Level 3, key-gated on 15 keys (TEXT_FLAMETHROWER -> 火焰喷射器). Level 6's 喷火枪 was the "
        "Phase-0 value; keeping the two apart also keeps Flamethrower distinct from Incinerator "
        "Unit 焚烧器.",
        [isoc("flamethrower", zh="火焰喷射器"), subc("flamethrower", CNSCG, zh="喷火枪")],
        tier="T-BILINGUAL", category="hardware", aliases=["喷火器", "喷火枪"],
        collision_split=["Incinerator Unit"])
    OV["Stun Baton"] = O(
        "电棍", 1, "Level 1 ch.2 gear list. Level 3's 电击棍 is key-gated and more precise; kept as "
                   "an alias.",
        [fanc("Stun baton", zh="电棍"), isoc("cattle prod / stun baton", zh="电击棍")],
        tier="T-BILINGUAL", category="hardware", aliases=["电击棍"])
    OV["Cutting Torch"] = O(
        "切割炬", 1, "Level 1 ch.2 gear list; level 3 independently uses 割炬 as the head noun for "
                     "all three of its torch tiers (气体/等离子/离子割炬).",
        [fanc("Cutting torch", zh="切割炬"), isoc("cutting tool", zh="切割工具")],
        tier="T-BILINGUAL", category="hardware")
    OV["Comm Unit"] = O("通信单元", 1, "Level 1 ch.2 gear list; also 通信 not 通讯, the mainland form.",
                        [fanc("Comm unit", zh="通信单元")], tier="T-BILINGUAL", category="hardware")
    OV["Personal Medkit"] = O("个人医疗包", 1, "Level 1 ch.2 gear list; level 3 gates med-kit 医疗包.",
                              [fanc("Personal medkit", zh="个人医疗包"), isoc("med-kit", zh="医疗包")],
                              tier="T-BILINGUAL", category="hardware")
    OV["Surgical Kit"] = O("外科手术包", 1, "Level 1 ch.2 gear list.",
                           [fanc("Surgical kit", zh="外科手术包")],
                           tier="T-BILINGUAL", category="hardware")
    OV["Frigate"] = O("护卫舰", 1, "Level 1 ch.2.", [fanc("frigate", zh="护卫舰")],
                      tier="T-BILINGUAL", category="hardware")
    OV["Pulse Rifle"] = O(
        "脉冲步枪", 2, "Level 2 (Romulus 2024), en-gated. Level 6's 电波枪 is on the known-bad list.",
        [subc("pulse rifle", "Romulus2024", zh="脉冲步枪"), subc("pulse rifle", CNSCG, zh="电波")],
        tier="T-BILINGUAL", category="hardware", known_bad_rejected=["电波枪"])
    OV["M41A Pulse Rifle"] = O("M41A脉冲步枪", 1, "Level 1 ch.2 gear list.",
                               [fanc("M41A Pulse Rifle", zh="M41A脉冲步枪")],
                               tier="T-BILINGUAL", category="hardware")

    OV["Turn"] = O(
        "节", "C",
        "⚠ CORRECTED IN v1.0. Phase 0 had 轮次, which is one character away from Round 轮 and names "
        "no unit the game has. Turn is the CLASSIC-mode name for the middle time unit; Evolved "
        "renamed it Stretch. Two independent confirmations: level 1 ch.1's table gives the middle "
        "unit as 节 (Stretch) 5-10分, and the healTime switch orders its cases None(0) < "
        "OneRound(1) < OneTurn(2) < OneShift(3) < OneDay(4), which puts Turn between Round and "
        "Shift — exactly where Stretch sits. So Turn and Stretch are one duration under two "
        "editions' names and must be one Chinese word. ⚠ RUNTIME: ALIENRPG.OneTurn currently reads "
        "一斡 — a stratum-B MT artefact; 斡 is not a Chinese counter for anything — and it is a "
        "case in the healTime switch. It becomes 一节, in lockstep with its RollTable cell.",
        [convc("节", "Stretch=节 (level 1 ch.1 time-unit table, glossed inline); Turn is the "
                     "classic-mode name for the same duration",
               "Level 1 ch.1: '轮 (Round) 5-10秒 战斗 / 节 (Stretch) 5-10分 潜行 / 班 (Shift) "
               "5-10小时 恢复'. The healTime switch's case order independently places OneTurn "
               "between OneRound and OneShift."),
         langc("OneTurn", zh="一斡", stratum="B")],
        tier="T-PLAIN", category="yze-mechanics", derived_from="Stretch",
        lockstep="ALIENRPG.OneTurn", aliases=["轮次"])

    OV["LV-426"] = O(
        "LV-426", 1,
        "Level 1 ch.1 keeps the designator in Latin letters WITH the hyphen. Level 6 drops the "
        "hyphen (LV426). T-LATIN, not T-FROZEN: no code looks this up, so renaming it would be "
        "wrong rather than fatal.",
        [fanc("LV-426", zh="LV-426"), subc("LV-426", CNSCG, zh="LV426"),
         refc("LV-426", "zh.wikipedia (level 7)", "Writes LV-426 with the hyphen.")],
        tier="T-LATIN", category="franchise-place")
    OV["Armor Piercing"] = O(
        "破甲", "C",
        "PROVISIONAL and explicitly a CONVENTION, not an attestation — en-gated 0 at every level "
        "of the ladder. Phase 0 filed it at the reference-work level because the web survey cited "
        "zh.wikipedia, but the survey's own confidence was 'low' and no article was re-derived. "
        "Level C is the honest label. See dispute D9-armor-piercing.",
        [convc("破甲", "TRPG convention: Armor Piercing is a weapon PROPERTY that attaches to "
                       "knives, bolt guns and acid attacks, not only to bullets",
               "cn.json's 穿甲弹 means 'armour-piercing ROUND' and cannot attach to a Combat "
               "Knife or to Acid Splash."),
         langc("ArmorPiercing", zh="穿甲弹")],
        tier="T-PLAIN", category="yze-mechanics", dispute="D9-armor-piercing")

    # --- range bands: level 1 uses the 距离 pattern, which also splits a collision --
    OV["Short"] = O(
        "近距离", 1,
        "Level 1 ch.2 writes 近距离 for the Short range band. Adopting level 1's 距离 pattern for "
        "the whole band set ALSO splits a Phase-0 collision: Long and Ranged were both 远程, so a "
        "player reading a weapon's range could not tell the band from the weapon class.",
        [fanc("Short range", zh="近距离")], tier="T-PLAIN", category="range",
        aliases=["短程"], collision_split=["Long", "Ranged"])
    OV["Medium"] = O(
        "中距离", "C", "Derived from Short 近距离 (level 1) so the four bands are one set.",
        [convc("中距离", "Short=近距离 (level 1 ch.2); the band set follows one pattern")],
        tier="T-PLAIN", category="range", derived_from="Short", aliases=["中程"])
    OV["Long"] = O(
        "远距离", "C",
        "Derived from Short 近距离 (level 1). Also the collision fix: Phase 0 had Long AND Ranged "
        "both at 远程. 远程 now belongs to Ranged alone.",
        [convc("远距离", "Short=近距离 (level 1 ch.2); the band set follows one pattern")],
        tier="T-PLAIN", category="range", derived_from="Short", aliases=["远程"],
        collision_split=["Ranged"])
    OV["Extreme"] = O(
        "极远距离", "C", "Derived from Short 近距离 (level 1) so the four bands are one set.",
        [convc("极远距离", "Short=近距离 (level 1 ch.2); the band set follows one pattern")],
        tier="T-PLAIN", category="range", derived_from="Short", aliases=["极远程"])
    OV["Ranged"] = O(
        "远程", "C",
        "Keeps 远程 for the WEAPON CLASS now that the range BAND has moved to 远距离. This is the "
        "same word Ranged Combat 远程战斗 (level 1) uses, so the weapon class and the skill stay "
        "one family.",
        [convc("远程", "head word of Ranged Combat 远程战斗 (level 1)")],
        tier="T-PLAIN", category="range", derived_from="Ranged Combat",
        collision_split=["Long"])

    # --- franchise vocabulary and creatures ---
    OV["Building Better Worlds"] = O(
        "建造更美好世界", 1,
        "Level 1 ch.1 renders the Weyland-Yutani motto 《建造更美好世界》. The Phase-0 建造更好的世界 "
        "was not among its own recorded candidates — it was a smoothing of the level-7 "
        "建造一个更好的世界, which is why the rebuild's proper-noun assertion caught it.",
        [fanc("Building Better Worlds", zh="《建造更美好世界》"),
         refc("建造一个更好的世界", "zh.wikipedia (level 7)",
              "The Phase-0 candidate; the adopted Phase-0 value was neither this nor anything "
              "else in its candidate array.")],
        tier="T-BILINGUAL", category="franchise-corp",
        note="The book-title brackets 《》 belong to the fan text's typography, not to the motto. "
             "The glossary value is the bare slogan; a page NAME that is the motto may carry them.")

    OV["Alien"] = O(
        "异形", 3,
        "Level 3 hard gate: AI_EXTERN_360_PRESENCE_KILLED_BY_ALIEN -> 被异形所杀, 72 key-gated keys. "
        "Level 2 agrees for the creature sense. The owner's settled decision 4 (异形 = the creature, "
        "外星 = the adjective) is unchanged and is now attested at level 3 rather than level 7.",
        [isoc("Alien (the creature)", zh="异形"), subc("xenomorph", CNSCG, zh="异形"),
         subc("xenomorph", "AlienEarth2025", zh="异形"),
         subc("xenomorph", "Romulus2024", zh="异形")],
        tier="T-BILINGUAL", category="franchise-creature")
    OV["Xenomorph"] = O(
        "异形", 2, "Level 2, three lineages en-gated. ⚠ Level 3 has NO distinct rendering for "
                   "'xenomorph' at all — Isolation collapses it into 异形 — so Xenomorph and Alien "
                   "are one Chinese word by every source that speaks.",
        [subc("xenomorph", "AlienEarth2025", zh="异形"), subc("xenomorph", "Romulus2024", zh="异形"),
         subc("xenomorph", CNSCG, zh="异形"), fanc("Xenomorph / the Alien", zh="异形")],
        tier="T-BILINGUAL", category="franchise-creature")
    OV["extraterrestrial"] = O(
        "外星", 1, "Level 1 ch.1, and level 2 for the adjectival sense. The 异形/外星 split is the "
                   "owner's settled decision 4: 外星飞船, never 异形飞船.",
        [fanc("alien (adjective) / extraterrestrial", zh="外星"),
         subc("alien (any)", "AlienEarth2025", zh="外星"),
         subc("alien (any)", "Prometheus2012", zh="外星")],
        tier="T-PLAIN", category="franchise-creature")
    OV["alien ship"] = O("外星飞船", 1, "The owner's settled decision 4 made concrete.",
                         [fanc("alien (adjective) / extraterrestrial", zh="外星")],
                         tier="T-PLAIN", category="franchise-creature", derived_from="extraterrestrial")
    OV["alien life-form"] = O("外星生命体", 1, "Same split; 生命体 from level 5 (Covenant 2017).",
                              [fanc("alien (adjective) / extraterrestrial", zh="外星"),
                               subc("organism", "Covenant2017", zh="生命体")],
                              tier="T-PLAIN", category="franchise-creature",
                              derived_from="extraterrestrial")
    OV["Facehugger"] = O(
        "抱脸虫", 3,
        "PROMOTED FROM A GUESS TO AN ATTESTATION. Phase 0 adopted 抱脸虫 on the owner's ruling with "
        "only 百度百科 behind it and zero corpus hits. Level 3 now HARD-gates it: three keys of the "
        "form *_KILLED_BY_FACEHUGGER -> 被抱脸虫所杀 in the official Simplified Chinese localisation. "
        "The subtitle corpus is still zero across all 11,088 pairs.",
        [isoc("facehugger", zh="抱脸虫"),
         refc("抱脸体", "zh.wikipedia (level 7)", "The wiki form; superseded by the level-3 gate."),
         langc("AutoPanicHint", stratum="B")],
        tier="T-BILINGUAL", category="franchise-creature", aliases=["抱脸体", "抱脸者"])
    OV["Egg"] = O(
        "卵", 3,
        "Level 3 and level 6 both say 卵. ⚠ LADDER DEVIATION, declared: level 2 (Alien: Earth 2025) "
        "renders it 蛋, but its en-gated top n-gram is 颗蛋 — a measure-word phrase in spoken "
        "dialogue, not a term. A rules noun needs the biological word, and the official localisation "
        "supplies it. Logged in _meta.ladder_deviations.",
        [isoc("egg", zh="卵"), subc("egg", CNSCG, zh="卵"), subc("egg", "AlienEarth2025", zh="蛋")],
        tier="T-PLAIN", category="franchise-creature", beats_a_higher_level=2)
    OV["Nest"] = O("巢穴", 6, "Level 6, en-gated; level 3 agrees in ungated prose (巢穴, 13 keys).",
                   [subc("nest", CNSCG, zh="巢穴"), isoc("nest / hive", zh="巢穴")],
                   tier="T-PLAIN", category="franchise-creature", collision_split=["Hive"])
    OV["Hive"] = O(
        "虫巢", "C",
        "COLLISION SPLIT, common noun. Phase 0 had Hive and Nest both at 巢穴. Nest keeps 巢穴 "
        "(level 6, en-gated); Hive takes 虫巢, the ordinary Chinese for an insectoid colony "
        "structure. Level 6's only Hive hit is the simile 像蚂蚁窝 (one line) and is not a term.",
        [convc("虫巢", "collision split from Nest 巢穴; standard modern Chinese for an insectoid hive"),
         subc("hive", CNSCG, zh="蚂蚁窝")],
        tier="T-PLAIN", category="franchise-creature", collision_split=["Nest"])
    OV["Organism"] = O(
        "生物体", "C",
        "COLLISION SPLIT, common noun. 生物 is already the pack folder 'Creatures' and the level-2 "
        "rendering of 'creature'. Level 5 (Covenant 2017) gives 生命体 for organism; 生物体 keeps the "
        "level-2/6 head noun 生物 while staying distinct from both.",
        [subc("organism", "Romulus2024", zh="生物"), subc("organism", "Covenant2017", zh="生命体"),
         subc("organism", CNSCG, zh="生物"),
         convc("生物体", "collision split from Creatures 生物 and creature 生物")],
        tier="T-PLAIN", category="franchise-vocab", collision_split=["Creatures"])
    OV["Specimen"] = O(
        "样本", 2,
        "Level 2 is SPLIT — Alien: Earth 2025 says 样本, Romulus 2024 says 标本 — so the tie drops "
        "to level 5, where Prometheus 2012 says 样本. 标本 is a mounted museum specimen; a live "
        "captive is 样本.",
        [subc("specimen", "AlienEarth2025", zh="样本"), subc("specimen", "Romulus2024", zh="标本"),
         subc("specimen", "Prometheus2012", zh="样本"), subc("specimen", CNSCG, zh="标本")],
        tier="T-PLAIN", category="franchise-vocab", aliases=["标本"])
    OV["Host"] = O("宿主", 2, "Four lineages including level 2, en-gated, unanimous.",
                   [subc("host", "AlienEarth2025", zh="宿主"), subc("host", "Romulus2024", zh="宿主"),
                    subc("host", "Covenant2017", zh="宿主"), subc("host", CNSCG, zh="宿主")],
                   tier="T-PLAIN", category="franchise-creature")
    OV["Parasite"] = O("寄生虫", 2, "Four lineages including level 2, en-gated.",
                       [subc("parasite", "AlienEarth2025", zh="寄生虫"),
                        subc("parasite", "Romulus2024", zh="寄生物"),
                        subc("parasite", "Covenant2017", zh="寄生虫"),
                        subc("parasite", CNSCG, zh="寄生虫")],
                       tier="T-PLAIN", category="franchise-vocab", aliases=["寄生物"])
    OV["The Company"] = O("公司", 2, "Four lineages including level 2, en-gated 23 rows, unanimous.",
                          [subc("the Company", "AlienEarth2025", zh="公司"),
                           subc("the Company", "Romulus2024", zh="公司"),
                           subc("the Company", "Covenant2017", zh="公司"),
                           subc("the Company", CNSCG, zh="公司")],
                          tier="T-PLAIN", category="franchise-corp")
    OV["Xenomorphology"] = O(
        "异形生物学", 1, "Level 1 ch.2 talent list, glossed inline; replaces the Phase-0 异形学.",
        [fanc("Xenomorphology", zh="异形生物学")],
        tier="T-BILINGUAL", category="franchise-creature")
    OV["Mother"] = O(
        "老妈", 1,
        "Level 1 renders the SHIP AI 老妈 in all four ch.1 log entries, and levels 5 and 6 agree "
        "(15 lines). Level 2 says 母亲. ⚠ Three different things share this English word: the ship "
        "AI (老妈), the Game Mother role (游戏管理员), and the computer's own name MU/TH/UR, which "
        "level 1 leaves in Latin.",
        [fanc("MOTHER (ship AI)", zh="老妈"),
         subc("MU/TH/UR (Mother)", "Covenant2017", zh="老妈"),
         subc("MU/TH/UR (Mother)", CNSCG, zh="老妈"),
         subc("MU/TH/UR (Mother)", "AlienEarth2025", zh="母亲"),
         subc("MU/TH/UR (Mother)", "Romulus2024", zh="母亲")],
        tier="T-BILINGUAL", category="franchise-computer", aliases=["母亲"],
        crossref=["MU/TH/UR", "Game Mother"])
    OV["Game Mother"] = O(
        "游戏管理员", 1,
        "Level 1 ch.1 heads the section 游戏管理员 with the English glossed inline, and uses 老妈 "
        "colloquially in the same chapter. The formal form is the term; 老妈 is the register "
        "variant and is the SHIP AI's word.",
        [fanc("Game Mother", zh="游戏管理员"), fanc("Game Mother (2nd rendering)", zh="老妈"),
         fanc("GM / Game Master", zh="游戏主持人")],
        tier="T-PLAIN", category="yze-mechanics", aliases=["老妈", "游戏主持人"],
        crossref=["Mother", "MU/TH/UR"])

    # --- named people and places that the ladder now reaches ---
    OV["Newt"] = O("纽特", 6, "Level 6, en-gated 35 rows. No higher rung reaches this name.",
                   [subc("Newt", CNSCG, zh="纽特")], tier="T-BILINGUAL", category="franchise-character")
    OV["Ripley"] = O(
        "蕾普丽", 6, "Level 6, en-gated 75 rows — the single best-attested name in the corpus. No "
                     "higher rung reaches it. Recorded as level 6 so nobody mistakes it for a "
                     "mainland-standard transliteration; 雷普利 is the commoner mainland form and has "
                     "zero attestation here.",
        [subc("Ripley", CNSCG, zh="蕾普丽")], tier="T-BILINGUAL", category="franchise-character",
        note="Level 6 is a Taiwan-influenced lineage. This name is adopted on weight of attestation, "
             "not on register; if a level 1-3 source ever reaches it, it should be revisited.")

    return OV, AD
