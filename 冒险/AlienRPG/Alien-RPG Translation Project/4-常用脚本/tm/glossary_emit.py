#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
glossary_emit.py — disputes, pending, byrole and the invariant gate for the v1.0 rebuild.
Imported by build_glossary_v1.py.
"""

# ===========================================================================
# DISPUTES — genuine OPEN conflicts only.  A conflict the ladder settles is not
# a dispute; it is a decision, and it goes to disputes._meta.closed_in_v1_0.
# ===========================================================================

def disputes(ctx):
    subc = ctx["subc"]; langc = ctx["langc"]; fanc = ctx["fanc"]
    CNSCG = "CnSCG(异形1/2/3/4·同源)"
    SUB = ctx["SUB_BY_KEY"]

    q = SUB["queen"]
    D = {}

    D["D6-queen"] = {
        "en": "Queen",
        "provisional_cn": "女王",
        "tier": "T-BILINGUAL",
        "governs_terms": ["Queen", "Alien Queen"],
        "why_open": (
            "ONE VOTE ARGUING WITH ITSELF. Every hit is CnSCG (level 6), and the project's hard "
            "rule folds all four films into a single vote precisely because cross-film agreement "
            "there is shared provenance rather than corroboration — so cross-film DISAGREEMENT is "
            "one translator team changing its mind, not two sources in conflict. Levels 1, 2, 3 "
            "and 5 are all silent: the fan serial does not reach the Queen (she is a ch.10 "
            "creature), Alien: Isolation has no Queen, and neither Romulus 2024 nor Alien: Earth "
            "2025 names her. There is nothing on the ladder above level 6 to break the tie, and "
            "level 6 cannot break it because it IS the tie."),
        "vote_shape": {
            "english_regex": q["english_regex"],
            "total_en_gated_hits": q["total_hits"],
            "hits_by_lineage": q["hits_by_lineage"],
            "distinct_lineages_with_any_hit": len(q["hits_by_lineage"]),
            "lineage_count_of_the_winning_candidate": (q["candidates"][0] or {}).get("lineage_count"),
        },
        "sides": [
            {"cn": "女王", "source_level": 6,
             "argument": "The form in the compound the RPG actually needs (Alien Queen -> 异形女王), "
                         "and the form zh.wikipedia's 异形 (虚构生物) article uses. Wins the "
                         "line count 3-2 once the bee-metaphor hit is discounted.",
             "gated_count": "en-gated 4 of 7 rows matching \\bqueen\\b — ALIEN3 3, ALIENS1986 1. "
                            "One of the 4 is the simile 女王蜂 (ALIENS1986 1:35:28), so the "
                            "plain-noun count is 3.",
             "evidence": "ALIEN3 1:41:18 'I'm carrying the new queen' -> 我孕育新女王; "
                         "ALIEN3 1:46:49 'It's a queen- an egg layer' -> 它是个女王 会产卵.",
             "lineage": "CnSCG — the SAME lineage as the other side"},
            {"cn": "王后", "source_level": 6,
             "argument": "Uniform across the whole of Alien Resurrection, the film with the most "
                         "Queen dialogue, and the same film produces 异形王后 for 'Her Majesty'.",
             "gated_count": "en-gated 3 of 7 — all RESURRECTION",
             "evidence": "RESURRECTION 0:13:09 'It's a queen.' -> 是王后; 1:29:58 -> 是王后; "
                         "1:34:19 'The queen laid her eggs' -> 王后产卵. Plus the cn_only "
                         "0:11:27 -> 异形王后才值回票价.",
             "lineage": "CnSCG — the SAME lineage as the other side"},
        ],
        "what_would_settle_it": (
            "ch.10 of the fan serial (level 1) would settle it in one line, and the owner has "
            "confirmed it does not exist. Failing that, an owner ruling. Do NOT settle it by "
            "counting CnSCG lines: that is a bare count inside one vote, which is the exact "
            "failure mode the project's hard rules forbid."),
        "blast_radius": "The Queen Actor(s) in the core-rules pack, the compound Alien Queen, and "
                        "all hive prose.",
    }

    D["D9-armor-piercing"] = {
        "en": "Armor Piercing",
        "provisional_cn": "破甲",
        "tier": "T-PLAIN",
        "governs_terms": ["Armor Piercing"],
        "why_open": (
            "The provisional value overrides a shipped lang value that is not wrong, only wrong in "
            "part of speech. Overriding a string players already read is an owner call. Nothing on "
            "the new ladder reaches this term at all: levels 1, 2, 3, 5 and 6 are all silent, so "
            "the rebuild could not close it either."),
        "sides": [
            {"cn": "破甲", "source_level": "C",
             "argument": "Armor Piercing is a weapon PROPERTY that attaches to knives, bolt guns "
                         "and acid attacks as well as to bullets. 破甲 is adjectival in Chinese and "
                         "is the TRPG convention.",
             "gated_count": "en-gated 0 — a convention call, not an attestation",
             "evidence": "Recorded at confidence 'low' in the Phase-0 web survey."},
            {"cn": "穿甲弹", "source_level": 4,
             "argument": "It is what the shipped file says, and for the pulse-rifle case it is "
                         "exactly right.",
             "gated_count": "en-gated 1 (a lang key/value pair)",
             "evidence": "ALIENRPG.ArmorPiercing 'Armor Piercing' -> 穿甲弹. 弹 means "
                         "'round/projectile', so the string cannot attach to a Combat Knife or to "
                         "Acid Splash."},
            {"cn": "穿甲", "source_level": "C",
             "argument": "The minimal fix: drop only the 弹 from the shipped value, keeping the "
                         "literal reading and making it a property rather than a noun.",
             "gated_count": "en-gated 0",
             "evidence": "Smallest edit that repairs the part of speech."},
        ],
        "what_would_settle_it": "Owner ruling between the convention (破甲) and the minimal fix "
                                "(穿甲). 穿甲弹 is not defensible as-is because of the non-firearm "
                                "carriers.",
        "blast_radius": "ALIENRPG.ArmorPiercing plus every weapon Item that carries the feature.",
    }

    D["D10-comtech"] = {
        "en": "Comtech",
        "provisional_cn": "科技",
        "tier": "T-EXACT",
        "governs_terms": ["Comtech"],
        "opened_in": "v1.0",
        "why_open": (
            "NEW IN v1.0, and the only place in this rebuild where level 1 was not taken. Comtech "
            "is COMmunications + TECHnology: the skill covers comms, electronics and computer "
            "intrusion. Level 1 renders it 计算机科学, 'computer science', which drops the comms and "
            "electronics half — the miner flags the same over-narrowing. Level 4's 科技 is vague "
            "but not wrong. Taking the higher rung here would import a demonstrably narrower "
            "reading, so the term is parked instead of decided."),
        "sides": [
            {"cn": "科技", "source_level": 4,
             "argument": "Broad enough to cover the whole skill, and it is what every Chinese "
                         "Foundry table reads today. Vague, but not narrower than the English.",
             "gated_count": "en-gated 1 (lang key/value pair ALIENRPG.Skillcomtech)",
             "evidence": "lang/en.json ALIENRPG.Skillcomtech 'Comtech' -> lang/cn.json '科技'."},
            {"cn": "计算机科学", "source_level": 1,
             "argument": "It is the reading in the Chinese translation of this very book, which is "
                         "the top of the ladder, and it is unambiguous where 科技 is vague.",
             "gated_count": "en-gated by construction — the fan text translates the known English "
                            "skill list position by position",
             "evidence": "ch.2 '我们给她3级重型机械，2级耐力和近战，以及1级计算机科学、侦察和医疗'. "
                         "The miner's own note: 'over-narrows a skill that also covers comms and "
                         "electronics'."},
            {"cn": "通信科技", "source_level": "C",
             "argument": "Splits the difference: keeps level 4's head noun 科技 and restores the "
                         "COM half the English name carries. Four characters, same as 计算机科学.",
             "gated_count": "en-gated 0",
             "evidence": "Composition, offered so the owner has a third option that is neither "
                         "vague nor narrow."},
        ],
        "what_would_settle_it": "Owner ruling. ⚠ T-EXACT: whichever wins must be written into "
                                "ALIENRPG.Skillcomtech AND onto the skill-stunts Item name in the "
                                "same commit, or character-sheet.mjs's getName() lookup misses.",
        "blast_radius": "ALIENRPG.Skillcomtech, the Comtech skill-stunts Item, every career's key "
                        "skill list, and every talent that keys off the skill.",
    }

    D["D11-captain"] = {
        "en": "Captain",
        "provisional_cn": "舰长",
        "tier": "T-PLAIN",
        "governs_terms": ["Captain"],
        "opened_in": "v1.0",
        "why_open": (
            "NEW IN v1.0. The ladder picks 舰长 because level 2 is the highest rung that speaks and "
            "Alien: Earth 2025 says 舰长 — but level 2 itself does not speak with one voice at level "
            "5 below it (Covenant 2017 舰长, Prometheus 2012 船长), and 舰长 is the word for a "
            "WARSHIP's commander. Most Alien RPG ships are commercial haulers with a crew of five, "
            "which is 船长 territory. The ladder result and the register are pulling opposite ways "
            "and neither is obviously wrong."),
        "sides": [
            {"cn": "舰长", "source_level": 2,
             "argument": "The highest rung that speaks. Level 2 (Alien: Earth 2025) and level 5 "
                         "(Covenant 2017) both use it, and it is the form a player who watched the "
                         "2024-25 releases will recognise.",
             "gated_count": "en-gated: 'Captain' matched 63 rows over 4 lineages; the per-lineage "
                            "top is 舰长 for AlienEarth2025 and Covenant2017",
             "evidence": "mined_subtitles_vote.json terms[key='Captain'].per_lineage_top"},
            {"cn": "船长", "source_level": 5,
             "argument": "The right register for a commercial freighter, which is what the Space "
                         "Truckers framework and most published Alien RPG ships are. Level 5 "
                         "(Prometheus 2012) and level 6 both use it.",
             "gated_count": "en-gated: per-lineage top 船长 for Prometheus2012 and CnSCG",
             "evidence": "mined_subtitles_vote.json terms[key='Captain'].per_lineage_top"},
        ],
        "what_would_settle_it": "ch.3+ of the fan serial would settle it at level 1; it does not "
                                "exist. Otherwise an owner ruling on whether ship ranks follow "
                                "naval register (舰长) or merchant-marine register (船长). Note "
                                "level 4 has NO Captain key, so nothing is at stake in the UI.",
        "blast_radius": "Every ship-crew NPC in all three packs, the Officer career's description, "
                        "and all voyage-log prose.",
    }
    return D


# ===========================================================================
# BYROLE — "does this name carry the English tail" is a PER-FIELD decision.
# ===========================================================================

ROLE_DOC = {
    "document.name": "Actor / Item / JournalEntry / RollTable / Scene / Macro `name`.",
    "page.name": "JournalEntryPage `name` — also the anchor slug source; see the id= remedy.",
    "folder.name": "Folder `name`.",
    "lang.value": "A value in lang/cn.json.",
    "prose": "Running HTML text inside a page or a description field.",
    "enricher.label": "The {label} inside @UUID[...]{...} / @TEXTDRAW[...]{...} / @DRAW[...]{...}.",
    "actor.table_ref": "Actor system.rTables / system.cTables — a literal table NAME stored as data.",
    "table.result_cell": "A cell of a RollTable result that the code parses by string equality.",
    "macro.command": "Macro.command — executable JavaScript. NEVER emitted, NEVER translated.",
}

CATEGORY_ROLES = {
    "frozen-import-lookup": ["document.name", "folder.name", "prose"],
    "frozen-rolltable": ["document.name", "actor.table_ref", "enricher.label", "macro.command", "prose"],
    "frozen-folder": ["folder.name", "macro.command", "prose"],
    "frozen-item-name": ["document.name", "prose"],
    "frozen-sentinel": ["actor.table_ref", "lang.value", "prose"],
    "skill": ["document.name", "lang.value", "prose"],
    "attribute": ["lang.value", "prose"],
    "range": ["lang.value", "prose"],
    "yze-mechanics": ["lang.value", "prose"],
    "yze-condition": ["lang.value", "prose"],
    "yze-condition-evolved": ["lang.value", "prose"],
    "yze-panic-table": ["lang.value", "table.result_cell", "prose"],
    "career": ["document.name", "lang.value", "prose"],
    "marine": ["lang.value", "prose"],
    "ops": ["lang.value", "prose"],
    "hardware": ["document.name", "enricher.label", "prose"],
    "pack-structure": ["folder.name", "page.name", "prose"],
    "franchise-creature": ["document.name", "page.name", "enricher.label", "prose"],
    "franchise-character": ["document.name", "page.name", "enricher.label", "prose"],
    "franchise-ship": ["document.name", "page.name", "enricher.label", "prose"],
    "franchise-place": ["document.name", "page.name", "enricher.label", "prose"],
    "franchise-corp": ["document.name", "page.name", "enricher.label", "prose"],
    "franchise-title": ["page.name", "enricher.label", "prose"],
    "franchise-computer": ["document.name", "page.name", "enricher.label", "prose"],
    "franchise-vocab": ["lang.value", "prose"],
}

# Terms whose per-field answer is NOT derivable from the tier.  These are the reason
# the byrole file exists at all.
BYROLE_OVERRIDES = {
    "Shift": {
        "_why": "The single most dangerous term in the project: one English word, three fields, "
                "three different answers.",
        "table.result_cell": {
            "tier": "T-FROZEN", "emit": "Shift",
            "why": "actor.mjs:1890 `if (testArray[9] === \"Shift\")` compares a BARE English "
                   "literal with no localize() call. Translate the cell and control falls to "
                   ":1893 `testArray[9].match(/^\\[\\[([0-9]d[0-9]+)]/)[1]`, the regex misses, and "
                   "null[1] throws a TypeError that aborts the whole Critical-Injury roll. "
                   "THIS IS A CRASH, NOT A SILENT ZERO.",
            "produced_by": "RollTable 'EV - Critical Injuries', rows 14-14 and 15-15",
        },
        "lang.value": {
            "tier": "T-PLAIN", "emit": "班",
            "why": "ALIENRPG.Shift is an OUTPUT key: actor.mjs:1891 assigns "
                   "`testArray[9] = game.i18n.localize(\"ALIENRPG.Shift\")` immediately AFTER the "
                   "comparison succeeds. It SHOULD be Chinese. cn.json leaves it as the English "
                   "word 'Shift' today, which is why the path happens to work by accident.",
        },
        "prose": {"tier": "T-PLAIN", "emit": "班",
                  "why": "The time unit as level 1 names it."},
    },
    "HARDENED": {
        "_why": "Correct as prose, fatal as a name.",
        "document.name": {
            "tier": "T-FROZEN", "emit": "Hardened",
            "why": "actor-character.mjs:449 and actor-synthetic.mjs:439 compare "
                   "`Attrib.name.toUpperCase() === \"HARDENED\"`. '硬汉'.toUpperCase() is '硬汉'. "
                   "The talent silently stops granting +1 max Health, in BOTH modes, on documents "
                   "that ship today.",
        },
        "prose": {"tier": "T-PLAIN", "emit": "硬汉",
                  "why": "Level 1 ch.2 renders the talent 硬汉. That is fine everywhere except the "
                         "name field."},
    },
    "Pack Mule": {
        "_why": "Uppercased before comparison, so case may change but the LETTERS may not.",
        "document.name": {"tier": "T-FROZEN", "emit": "Pack Mule",
                          "why": "character-sheet.mjs:491 / synthetic-sheet.mjs:478 / "
                                 "colony-sheet.mjs:339 test `i.name.toUpperCase() === \"PACK MULE\"`. "
                                 "'驮马 Pack Mule'.toUpperCase() is '驮马 PACK MULE' — NOT equal. "
                                 "A bilingual tail is fatal here."},
        "prose": {"tier": "T-PLAIN", "emit": "驮马",
                  "why": "Safe in the talent's description text."},
    },
    "Take Control": {
        "document.name": {"tier": "T-FROZEN", "emit": "Take Control",
                          "why": "actor-character.mjs:440 / actor-synthetic.mjs:430 test "
                                 "`Attrib.name.toUpperCase() === \"TAKE CONTROL\"`. Same "
                                 "bilingual-tail prohibition as Pack Mule."},
        "prose": {"tier": "T-PLAIN", "emit": "接管",
                  "why": "Safe in description text."},
    },
    "STOIC": {
        "document.name": {"tier": "T-FROZEN", "emit": "Stoic",
                          "why": "actor-character.mjs:452 / actor-synthetic.mjs:442. Silently "
                                 "reverts Stamina's key attribute from WIT to STR."},
        "prose": {"tier": "T-PLAIN", "emit": "坚忍", "why": "Safe in description text."},
    },
    "Alien Sub-Tables": {
        "_why": "Looks frozen, is not.  Its two sibling folders ARE frozen.",
        "folder.name": {"tier": "T-PLAIN", "emit": "异形子表",
                        "why": "The literal 'Alien Sub-Tables' appears ONLY at "
                               "module/apps/migratefolders.js:29, and migratefolders is "
                               "import-commented at module/apps/init.mjs:2. Folder#contents is "
                               "non-recursive, so rollTableData.mjs never walks into it. This "
                               "folder is fully translatable — unlike 'Alien Creature Tables' and "
                               "'Alien Mother Tables', which throw on rename."},
    },
    "None": {
        "_why": "A sentinel, not a word.  Two fields must agree or the dropdown desyncs.",
        "actor.table_ref": {"tier": "T-FROZEN", "emit": "None",
                            "why": "rollTableData.mjs:12 and :30 write the option KEY and the "
                                   "option LABEL from the same literal, and actor.mjs:2493 "
                                   "compares the stored value against it. 26 Actors store the "
                                   "literal 'None' today."},
        "lang.value": {"tier": "T-FROZEN", "emit": "None",
                       "why": "⚠ ALIENRPG.None is a case in the healTime switch "
                              "(actor.mjs:1924). cn.json currently renders it 莫, a stratum-B MT "
                              "artefact. The owner's decision 4 keeps the sentinel English."},
        "prose": {"tier": "T-PLAIN", "emit": "无",
                  "why": "⚠ THE SENTINEL IS NOT THE WORD. An English 'None' in running prose — "
                         "'None' in an armour column, 'no effect' — is ordinary vocabulary and "
                         "translates to 无. Only the DROPDOWN OPTION and ALIENRPG.None are frozen. "
                         "Freezing the prose too would leave English scattered through the rules "
                         "text for no mechanical reason."},
    },
    "Round": {
        "lang.value": {"tier": "T-PLAIN", "emit": "一轮",
                       "why": "⚠ ALIENRPG.OneRound is a case in the healTime switch "
                              "(actor.mjs:1927). Changing 一回合 -> 一轮 REQUIRES rewriting the "
                              "matching RollTable cell in the same commit, or healTime falls to "
                              "the default at :1939 and silently becomes 0."},
        "table.result_cell": {"tier": "T-PLAIN", "emit": "一轮",
                              "why": "Must be byte-equal to localize('ALIENRPG.OneRound') + one "
                                     "trailing ASCII space."},
        "prose": {"tier": "T-PLAIN", "emit": "轮", "why": "The bare time unit."},
    },
    "M5A3 RPG Launcher": {
        "_why": "A SUBSTRING test, not an equality test — the opposite conclusion from Pack Mule.",
        "document.name": {
            "tier": "T-BILINGUAL", "emit": "M5A3 RPG发射器 M5A3 RPG Launcher",
            "why": "character-sheet.mjs:428 / synthetic-sheet.mjs:415 accept the item if ANY of "
                   "`name.includes(' RPG ')`, `name.startsWith('RPG')`, `name.endsWith('RPG')` "
                   "holds. The T-BILINGUAL form KEEPS ' RPG ' intact, so the bilingual tail is "
                   "what SAVES this one — whereas a pure-Chinese rename drops it and ammoweight "
                   "silently falls from 0.5 kg to 0.25 kg per round. The data leg "
                   "(system.attributes.class.value === 'RPG') is measured DEAD, so the name test "
                   "is the only live path.",
        },
    },
    "Xenomorph": {
        "document.name": {"tier": "T-BILINGUAL", "emit": "异形 Xenomorph",
                          "why": "Creature Actor names carry the English tail."},
        "enricher.label": {"tier": "T-BILINGUAL", "emit": "异形 Xenomorph",
                           "why": "⚠ An explicit {Label} inside @UUID does NOT update when the "
                                  "target document is renamed. 42 links in the corerules pack "
                                  "carry an explicit label and 12 of the 39 Item links point into "
                                  "the STARTERSET pack — a hard sync point across two git repos."},
        "prose": {"tier": "T-PLAIN", "emit": "异形", "why": "Bare Chinese in running text."},
    },
    "Weyland-Yutani": {
        "document.name": {"tier": "T-BILINGUAL", "emit": "韦兰德-汤谷公司 Weyland-Yutani"},
        "prose": {"tier": "T-PLAIN", "emit": "韦兰德-汤谷公司",
                  "why": "Level 1 clips it to 韦汤(公司) in running text after first mention; both "
                         "are correct prose, and 韦汤 is keyed separately as W-Y."},
    },
}


def byrole(terms, meta_tiers):
    """Emit the per-(English, field role) tier table."""
    out = {}
    for en in sorted(terms):
        t = terms[en]
        tier = t["tier"]
        cat = t.get("category", "")
        roles = list(CATEGORY_ROLES.get(cat, ["prose"]))
        rec = {"cn": t["cn"], "term_tier": tier, "category": cat, "roles": {}}
        for r in roles:
            if r == "macro.command":
                rec["roles"][r] = {
                    "tier": "T-FROZEN", "emit": "(never emitted)",
                    "why": "Macro.command is executable JavaScript and Babele would translate it "
                           "by default (babele default-mappings.js:163). The extractor's refusal to "
                           "emit `command` is load-bearing, not incidental."}
                continue
            if tier == "T-FROZEN":
                rec["roles"][r] = {"tier": "T-FROZEN", "emit": t["cn"],
                                   "why": "Byte-exact English; the running code compares against it."}
            elif tier == "T-LATIN":
                rec["roles"][r] = {"tier": "T-LATIN", "emit": t["cn"],
                                   "why": "A Latin designator that stays Latin. NOT mechanics-"
                                          "critical: renaming it would not break code, it would "
                                          "just be wrong."}
            elif tier == "T-EXACT":
                if r in ("document.name", "lang.value"):
                    rec["roles"][r] = {
                        "tier": "T-EXACT", "emit": t["cn"],
                        "why": "The Item name and the lang value must be BYTE-EQUAL. "
                               "character-skills.hbs:15 emits data-pmbut='{{skill.description}}', "
                               "system.skills.<skl>.description is overwritten every "
                               "prepareDerivedData by localize('ALIENRPG.Skill<key>'), and "
                               "character-sheet.mjs then feeds it to game.items.getName(). "
                               "NO English tail is permitted."}
                else:
                    rec["roles"][r] = {"tier": "T-PLAIN", "emit": t["cn"],
                                       "why": "Bare Chinese in running text."}
            elif tier == "T-BILINGUAL":
                if r in ("document.name", "page.name", "folder.name", "enricher.label"):
                    rec["roles"][r] = {
                        "tier": "T-BILINGUAL", "emit": t.get("bilingual_name", t["cn"]),
                        "why": ("中文 + ONE ASCII space + English, no parentheses."
                                + (" ⚠ An explicit {Label} does NOT follow a document rename; it "
                                   "must be edited in lockstep with the name."
                                   if r == "enricher.label" else ""))}
                else:
                    rec["roles"][r] = {"tier": "T-PLAIN", "emit": t["cn"],
                                       "why": "Bare Chinese in running text."}
            else:
                rec["roles"][r] = {"tier": "T-PLAIN", "emit": t["cn"],
                                   "why": "Bare Chinese in prose and inside {label} text."}
        # authored per-field exceptions win
        ov = BYROLE_OVERRIDES.get(en)
        if ov:
            for k, v in ov.items():
                if k == "_why":
                    rec["_why"] = v
                else:
                    rec["roles"][k] = v
                    rec.setdefault("authored_exception", []).append(k)
        out[en] = rec
    return out
