#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""glossary_metablocks.py — the _meta prose for the v1.0 rebuild."""

LADDER = [
    {"level": 1, "source": "《异形RPG：进化版》中文连载 — ch.1 宇宙是地狱 + ch.2 你的角色",
     "what": "The Chinese translation of THIS book, this edition. 25,615 characters, forum serial, "
             "still edited 2026-04-28.",
     "reach": "⚠ ONLY these two chapters exist — the owner has confirmed there is no more. Anything "
              "they do not touch falls through to level 2. The Xenomorph life cycle is ch.10 and is "
              "therefore permanently out of reach.",
     "miner": "mined_fan_translation.json (285 records, 98 inline-glossed)"},
    {"level": 2, "source": "Alien: Romulus 2024 · Alien: Earth 2025 subtitles",
     "what": "The newest mainland-facing screen translations; Romulus had a theatrical release.",
     "miner": "mined_subtitles_vote.json lineages Romulus2024 (1,150 pairs) and "
              "AlienEarth2025 (2,266 pairs)"},
    {"level": 3, "source": "Alien: Isolation official Simplified Chinese localisation",
     "what": "An OFFICIAL localisation, and its entries are UI-length — closer in shape to RPG "
             "terminology than film dialogue is.",
     "caveat": "13,565 unique keys, not the 162,804 quoted elsewhere: the 768 files on disk collapse "
               "to 64 md5 hashes (12 identical copies). And there is NO English side anywhere — only "
               "19.5% of live keys can be English-gated at all, via key slugs and natural-language "
               "keys. An ungated Isolation reading is a Chinese-only reading and is marked as such.",
     "miner": "mined_alien_isolation.json (44 terms)"},
    {"level": 4, "source": "system lang/cn.json A-stratum (TI-130 / Tian#7972, 2020-11..2021-01)",
     "what": "Human translation, and it is what players see in Foundry today.",
     "caveat": "B-stratum (maintainer bulk MT 2022-2026) lives in the SAME FILE and is NEVER a "
               "source. Its signature errors: GM Only -> 仅限通用汽车, Overwatch -> 守望先锋, "
               "One Turn -> 一斡, None -> 莫.",
     "miner": "read directly from the installed system at build time"},
    {"level": 5, "source": "Alien: Covenant 2017 · Prometheus 2012 subtitles",
     "what": "Independent lineages, older register.",
     "miner": "mined_subtitles_vote.json"},
    {"level": 6, "source": "CnSCG Alien 1/2/3/4 subtitles",
     "what": "The classic trilogy plus Resurrection.",
     "caveat": "⚠ ALL FOUR COUNT AS ONE VOTE. They are a single fansub lineage, so cross-film "
               "agreement is shared provenance, not corroboration — and cross-film disagreement is "
               "one team changing its mind. Contains Taiwan-flavoured renderings (士官长, 太空梭, "
               "电波枪, 复仇161星, 伟伦优达尼) and outright errors (APC -> 焚化炉, dropship erased, "
               "stasis -> 静态平衡, Ellen -> 海伦, smart gun -> 机关枪, science division -> 卫生署).",
     "miner": "mined_subtitles_vote.json lineage CnSCG(异形1/2/3/4·同源) (4,591 pairs)"},
    {"level": 7, "source": "zh.wikipedia",
     "what": "A general encyclopedia about the FILMS, not about this RPG.",
     "caveat": "Demoted to last resort. It was the Phase-0 SPINE, which is the single biggest "
               "difference between v0.2 and v1.0.",
     "miner": "not re-derivable this session (no WebFetch); carried from the Phase-0 survey"},
]

NOT_A_LEVEL = {
    "C": "Convention, or derivation from terms already adopted at a real level. LEGAL for a common "
         "noun. ILLEGAL for a proper noun — see rule_proper_nouns. Every C term either names a "
         "derived_from parent or is a common noun.",
    "B-rejected": "cn.json stratum B. Recorded as a candidate so the reader can see it was "
                  "considered and refused, never as a winner.",
    "off-ladder": "The Chinese fan corpus (bilibili / douban / 机核). Named in the Phase-0 survey, "
                  "uncountable this session (bilibili serves a captcha to curl and an empty SPA "
                  "shell to WebFetch). Not on the owner's ladder.",
    "FROZEN": "Not a terminology decision at all. The value is a byte-exact English string the "
              "running code compares against.",
}

HARD_RULES = [
    "NEVER adopt a term from a bare Chinese frequency count. Every count in this file is either "
    "English-gated (the paired English at the same position was checked), a key/value pair, an "
    "inline gloss, or an explicit derivation. The Phase-0 build caught three bare counts that had "
    "already reached the survey's conclusions: Synthetic 生化人 '13 hits' was really 1 synthetic "
    "against 9 android/droid; Queen 女王 7 / 王后 4 was really 4 / 3; Ripley 蕾普丽 82 was really 75.",
    "The CnSCG corpus is ONE vote, not four. term_vote.py folds it; do not undo that.",
    "cn.json stratum B is never a source.",
    "EC and PF2E terminology were NOT merged in. Different world; both projects' own hard "
    "constraints forbid it. Only the SHAPE of glossary_ec.* was consulted.",
    "A term with ZERO English-gated hits is not evidence. It goes to pending, not into the spine.",
]

RULE_PROPER_NOUNS = (
    "Standard modern Chinese usage may settle a COMMON noun. It can NEVER settle a proper noun — a "
    "place, a ship, a person, an organisation or a model designation. Phase 0 broke this rule twice "
    "(United Systems Military 联合星系军, whose why_this_won said literally 'Coined.', and Rebecca "
    "Jorden 丽贝卡·乔登, whose surname was sourced to 'standard modern Chinese usage' at en-gated 0). "
    "Both are now in glossary_alien.pending.json, and assertion #7 in the builder makes the rule "
    "mechanical: any franchise-* term at level C without a derived_from parent fails the build.")

TIERS = {
    "T-FROZEN": "English, byte-exact. NARROWED IN v1.0 to mean exactly one thing: this string is "
                "compared by the running code, and translating it breaks the game. Every T-FROZEN "
                "entry is traceable to a named lookup in 7-其他内容/DO-NOT-TRANSLATE.json, and "
                "assertion #5 fails the build if one is not.",
    "T-LATIN": "NEW IN v1.0. A Latin designator that stays in Latin letters because that is what it "
               "is — LV-426, LV-223, MU/TH/UR, USCSS, USS, UAS — and NOT because any code looks it "
               "up. Safe to leave English; renaming would be wrong, not fatal.",
    "T-EXACT": "Pure Chinese, byte-equal to a lang key value, NO English tail. The 12 skill-stunts "
               "Item names must equal ALIENRPG.Skill<key> exactly.",
    "T-BILINGUAL": "'中文 English' — ONE ASCII space (U+0020), no parentheses — on name and "
                   "page-name fields.",
    "T-PLAIN": "Bare Chinese in prose and inside {label} text.",
}

TIER_SCOPE_DECISION = {
    "defect": "The Phase-0 file applied T-FROZEN to four entries that no code looks up: LV-426, "
              "LV-223, MU/TH/UR and USCSS. PROJECT.md §3.4 defines the tier by its mechanism "
              "('9 import lookup names, 11 hard-coded RollTable names, 2 folder names, PACK MULE, "
              "TAKE CONTROL, the None sentinel'), so the file was using the tier in a second, wider "
              "sense — 'emit verbatim' — without saying so.",
    "what_was_done": "NARROWED THE TIER, and added T-LATIN for the wider sense.",
    "why_narrow_rather_than_widen": (
        "Because T-FROZEN is load-bearing as a PROOF, not just as a label. Three QA gates "
        "(scan_name_lookup_traps, scan_crit_lockstep, scan_table_ref_sync) exist to catch a "
        "translated literal breaking the game, and the only mechanical way to check them is "
        "'every T-FROZEN entry resolves to a lookup in DO-NOT-TRANSLATE.json'. Widening the "
        "definition to cover Latin designators makes that assertion unwriteable: it would have to "
        "carry a hand-maintained exception list, and a gate with an exception list is a gate that "
        "quietly stops gating. Narrowing costs one new tier name and buys a build-time assertion. "
        "The cost of getting it wrong is asymmetric too — reading 'T-FROZEN' as 'renaming this "
        "breaks code' when it does not is a false alarm, but the reverse would be a missed one, "
        "and a wider tier produces both."),
    "consequence_for_PROJECT_md": "§3.4 needs a fifth row for T-LATIN. Until it has one, this file "
                                  "is ahead of the doc and says so here rather than pretending the "
                                  "doc already agrees.",
    "count_moved": 4,
    "entries_moved": ["LV-426", "LV-223", "MU/TH/UR", "USCSS"],
    "entries_added_to_the_narrowed_tier": ["EV - 48a. LS - DANGER EVENT DETAIL", "HARDENED",
                                           "Hardened", "STOIC", "Stoic", "TOUGH",
                                           "NERVES OF STEEL"],
}

BILINGUAL_CONTRACT = (
    "terms[<en>].bilingual_name is ALWAYS terms[<en>].cn + ONE ASCII space (U+0020) + the English "
    "NAME, where the NAME is the glossary key minus any trailing ' (...)' disambiguator. No "
    "parentheses around the English, no full-width space, no en space. Asserted on every "
    "T-BILINGUAL entry; the build fails if any entry deviates. The disambiguator is preserved "
    "separately as terms[<en>].key_disambiguator so that 'Alien 3 (1992 film)' yields "
    "'异形3 Alien 3' and never writes '(1992 film)' onto a page-name field.")

VALUE_SHAPE = (
    "glossary_alien.json values are BARE Chinese, never a bilingual tail — EXCEPT T-FROZEN and "
    "T-LATIN entries, whose value is the byte-exact Latin string. For T-BILINGUAL entries the "
    "ready-made name-field string is provenance.terms[<en>].bilingual_name. Which form a given "
    "FIELD takes is a separate question and is answered per-field in glossary_alien.byrole.json — "
    "that file exists because 'does this name carry the English tail' has different answers for "
    "the same English word depending on where it lands.")

WHAT_CHANGED = {
    "headline": "The spine was re-cut. Phase 0 adjudicated with zh.wikipedia at the top because "
                "that and one fansub lineage were all the evidence the owner had supplied. Three "
                "stronger sources have since arrived, one of them the Chinese translation of this "
                "very book, and zh.wikipedia has moved from rung 1 to rung 7.",
    "the_seven_named_changes": {
        "Weyland-Yutani": "韦兰-尤坦尼集团 -> 韦兰德-汤谷公司 (short form 韦汤)",
        "Nostromo": "诺史莫号 -> 诺斯特罗莫号",
        "Sulaco": "was PENDING -> 萨拉科号",
        "Synthetic": "生化人 -> 合成人, and split three ways: Android -> 仿生人, Robot -> 机器人",
        "atmosphere processor": "was PENDING -> 大气处理器 (never 大气处理厂)",
        "Hadley's Hope": "was PENDING -> 哈德利的希望",
        "new sets": "Round/Stretch/Shift = 轮/节/班 · Cinematic/Campaign Mode = 电影模式/战役模式 · "
                    "Frontier/Outer Veil/Core Systems/Outer Rim = 边境/外层帷幕/核心星系/外环 · "
                    "Evolved Edition = 进化版",
    },
    "kept_unchanged": {
        "Wits": "机智 — owner-pinned, level 4 beats level 1 (which says 智力)",
        "Empathy": "共情 — owner-pinned, level 4 beats level 1 (which says 同理心)",
        "Facehugger": "抱脸虫 — value unchanged but PROMOTED from a guess to a level-3 hard gate: "
                      "Alien: Isolation key-gates *_KILLED_BY_FACEHUGGER -> 被抱脸虫所杀 on 3 keys. "
                      "Phase 0 had zero corpus hits for this word.",
        "异形/外星": "the owner's settled decision 4, now attested at level 3 (72 key-gated keys "
                     "for 异形) instead of level 7",
    },
}

PHASE0_DEFECTS = [
    {"id": 1, "what": "terms['Bug Hunt'] carried a second candidate whose zh duplicated the first, "
                      "whose count said 1 when the re-derived gate says 2, and whose evidence text "
                      "described a --en '\\bwarrior' gate run pasted from an unrelated term.",
     "status": "FIXED. Verified this build: no term in the file has two candidates with the same "
               "zh (assertion #1), and Bug Hunt carries one candidate at en-gated 2.",
     "found_by": "Phase-0 verifier; repaired in v0.2.1 and re-asserted here."},
    {"id": 2, "what": "Two entries were glossed against the glossary's own proper-noun rule: "
                      "United Systems Military 联合星系军 ('Coined.') and Rebecca Jorden 丽贝卡·乔登 "
                      "(surname at en-gated 0, sourced to 'standard modern Chinese usage').",
     "status": "FIXED BY REMOVAL. Both are now in glossary_alien.pending.json with their evidence "
               "state written out. The rule is now mechanical (assertion #7), so it cannot be "
               "broken again without failing the build. Rebecca Jorden's character remains "
               "reachable: Newt = 纽特 stays in the spine at level 6, en-gated 35.",
     "reconciliation_note": "United Systems Military's sibling USCM was ALREADY pending for exactly "
                            "this reason. Two military branches in identical evidence states were "
                            "getting opposite treatment; they now match."},
    {"id": 3, "what": "_meta.lockstep_literals carried three wrong actor.mjs anchors: Shift cited "
                      "at :1888 (two lines off, and colliding with the Permanent comparison that "
                      "genuinely lives there), and the healTime switch given as :1922-1941 (off by "
                      "one at both ends — :1922 is blank, :1941 is the default arm's break). It "
                      "also described ONE switch where the source has two.",
     "status": "FIXED, AND MADE UNBREAKABLE. lockstep_literals is no longer transcribed: "
               "derive_lockstep() locates every anchor BY PATTERN in the installed actor.mjs at "
               "build time. If upstream moves a line, the next build reports the new number "
               "instead of lying. The two switches are now named separately: switch(testArray[3]) "
               "drives cFatal, switch(testArray[5]) drives healTime."},
    {"id": 4, "what": "_meta.lockstep_literals._why said these 'must move together or healTime "
                      "silently resolves to 0'. Half true, and the wrong half was the dangerous one.",
     "status": "FIXED. Both failure modes are now stated, and they are different in kind: the "
               "testArray[5] switch DOES fall to default and set healTime = 0 with no log. The "
               "Shift branch does NOT — a non-'Shift', non-empty cell reaches "
               "testArray[9].match(...)[1], the regex misses, and null[1] throws an uncaught "
               "TypeError that aborts the whole Critical-Injury roll. ⚠ The QA gate for that path "
               "must look for a CRASH, not for a zero. A gate written against 'healTime == 0' "
               "would pass while the roll dies."},
    {"id": 5, "what": "Weyland Corporation 韦兰德公司 and Weyland-Yutani 韦兰-尤坦尼集团 spelled the "
                      "same surname two ways with no cross-reference between them.",
     "status": "FIXED TWICE OVER. Under the new ladder both become 韦兰德-, so the inconsistency "
               "dissolves — but the cross-reference was added anyway, because a consistency sweep "
               "that finds two spellings of one surname will unify them, and without the "
               "cross-link it has no way to know which direction is right. Five keys now carry "
               "surname_crossref (Weyland-Yutani, Weyland-Yutani Corporation, Weyland Corporation, "
               "Peter Weyland, W-Y) and assertion #15 fails the build if any Weyland key drops "
               "韦兰德 or loses its cross-reference."},
    {"id": 6, "what": "Dispute D8-shift had no backlink from the term it governed, although "
                      "_meta.provisional_values_live_in promised one.",
     "status": "FIXED AND GENERALISED. Backlinks are now GENERATED from each dispute's "
               "governs_terms list rather than maintained by hand, and assertion #10 checks both "
               "directions — every governed term backlinks to its dispute, and no term points at a "
               "dispute that does not exist. Audited: all nine Phase-0 disputes had backlinks by "
               "the end of v0.2.1; the gap was real when reported and the generation makes it "
               "unrepeatable.",
     "also_fixed": "D8's do_not_confuse_with cited actor.mjs:1888 for the bare 'Shift' literal — "
                   "the same wrong line as defect 3, still uncorrected in the disputes file after "
                   "the provenance file had been fixed. D8 is now CLOSED by level 1, and the "
                   "runtime hazard it warned about moved to byrole['Shift'], where the anchor is "
                   "re-derived rather than quoted."},
    {"id": 7, "what": "T-FROZEN was applied more broadly than PROJECT.md §3.4 defines it — four "
                      "entries carried the tier that no code looks up.",
     "status": "FIXED BY NARROWING THE TIER. See _meta.tier_scope_decision for which was chosen "
               "and why."},
    {"id": "extra", "what": "FOUND THIS BUILD, not in the brief: the T-FROZEN key "
                            "'CORE RULES - HOW TO USE THIS MODULE' was spelled with ASCII spaces. "
                            "The literal in alien-evolved-corerules/module/init.js:11 uses "
                            "NO-BREAK SPACE U+00A0 between every word. Its sibling "
                            "'STARTER SET - HOW TO USE THIS MODULE' really does use ASCII spaces, "
                            "so they are NOT symmetric and must not be normalised to match.",
     "status": "FIXED. The key is now byte-exact and carries its codepoint dump. A T-FROZEN key "
               "that is not byte-exact is worse than no key at all: a translator searching for it "
               "never finds the real string, and a QA gate built on it reports clean forever.",
     "sibling_check": "All 9 name_lookups literals were codepoint-dumped this build; this is the "
                      "only one with non-ASCII whitespace."},
]

KNOWN_GAPS = [
    "⚠ THE BUILDER SEEDS ITSELF. build_glossary_v1.py READS glossary_alien.provenance.json and "
    "glossary_alien.pending.json and then WRITES them. That is verified idempotent — two "
    "consecutive runs produce byte-identical output on all five files — but it means the build is "
    "not reproducible from a clean checkout with these two files deleted: it carries 276 Phase-0 "
    "terms forward rather than re-deriving them. The Phase-0 dispute set WAS archived in time "
    "(glossary_alien.disputes.phase0.json) and the Phase-0 research blocks were recovered into "
    "_meta.phase0_research_carried_forward, but the Phase-0 provenance file itself was consumed in "
    "place by the first v1.0 run and is not recoverable from disk. Anyone rebuilding from scratch "
    "must treat the current provenance file as the baseline, not as a build product.",
    "The 3 content packs are still not extracted to compendium/en, so no compendium-mode gate could "
    "be run against pack PROSE. Every disputed term should be re-gated in compendium mode once the "
    "packs are dumped.",
    "Level 1 stops after ch.2. The whole of the Xenomorph life cycle (ch.10), the bestiary, "
    "spaceship combat and the GM chapters are out of its reach and always will be — the owner has "
    "confirmed no more chapters exist. Five caste names stay pending permanently unless another "
    "source appears.",
    "Level 3 has NO English side. 80.5% of its live keys cannot be English-gated by any mechanical "
    "means. Every ungated Isolation reading in this file is marked gate_method: ungated_prose or "
    "external_knowledge+value and must not be cited as attestation.",
    "純美蘋果園 topic=121082.0 (a 2021 Chinese translation of the Alien RPG stealth rules) is behind "
    "a login wall and was not read. It is the only other Chinese Alien RPG text known to exist.",
    "bilibili (cv18100844, cv2014315) and 萌娘百科 block automated fetch, so the off-ladder fan "
    "corpus could not be counted. Nothing in this file rests on it.",
    "58 Talent Item names and ~40 star-system designations are in the packs and are NOT in this "
    "glossary. That is deliberate — they are content for the pack pass, not terminology spine — "
    "with three exceptions now keyed because the code compares against them: HARDENED, STOIC and "
    "the dead-but-reserved TOUGH / NERVES OF STEEL.",
]

PHASE0_RESEARCH_CARRIED_FORWARD = {
    "_why": "These were re-derived from source during Phase 0 and cost real work. The v1.0 build "
            "re-cut the spine above them, but the FACTS below are about the corpora and the code, "
            "not about the ladder, so they survive the re-cut. Carried here verbatim except where "
            "v1.0 supersedes them, which is marked inline. ⚠ They were nearly lost: the first v1.0 "
            "build run overwrote glossary_alien.provenance.json, which was also its own input.",
    "english_gating_caught_these_bare_counts": [
        "Synthetic = 生化人 was cited at '13 hits'. That is a bare Chinese count. English-gated, "
        "生化人 renders 'synthetic' 1 time out of 5 and android/droid 9 times out of 11 — it is the "
        "corpus's ANDROID word. v1.0 acts on this: Synthetic is 合成人 and Android is 仿生人.",
        "Queen 女王 7 / 王后 4 were bare counts. English-gated they are 4 / 3, and one of the four "
        "女王 hits is the bee simile 女王蜂 (ALIENS1986 1:35:28), so the plain-noun count is 3 / 3.",
        "Ripley 蕾普丽 82 / 蕾普莉 12 were bare. English-gated: 75 / 11.",
        "Quarantine: the survey called 隔离 and 检疫 'equally attested'. English-gated in the CnSCG "
        "lineage it is 检疫 4 vs 隔离 2. ⚠ SUPERSEDED BY EVIDENCE, not by the ladder: the rebuilt "
        "11,088-pair corpus adds levels 2 and 5, where 隔离 wins in three lineages. v1.0 adopts 隔离.",
        "Weyland-Yutani: the survey said 2 subtitle occurrences. English-gated the CnSCG corpus "
        "casts ONE vote — the ALIEN3 0:09:09 hit is on-screen text whose English side is empty, so "
        "it is cn_only.",
        "Bug Hunt = 除虫行动 is en-gated 2 (ALIENS1986 0:33:31 and 0:33:45), not 1. The Phase-0 "
        "file's second candidate for this term was a paste error and was removed.",
    ],
    "cn_json_facts_re_derived_and_still_true": [
        "ALIENRPG.conditions at module/helpers/config.mjs:161 has TWENTY-ONE entries, not 20 and "
        "not 19, and 'fatigued' IS one of them (resp:\"\", tableNumber:0). The 21 split 7 / 12 / 2: "
        "seven stress responses (jumpy, tunnelvision, aggravated, shakes, frantic, deflated, "
        "messup), twelve panic responses (spooked, noisy, twitchy, loseitem, paranoid, hesitant, "
        "freeze, seekcover, scream, flee, frenzy, catatonic), and two on no response table "
        "(fatigued, keepingguard).",
        "ALL 21 condition keys are UNTRANSLATED in cn.json: 20 are absent outright and "
        "ALIENRPG.messup holds the English 'Mess Up'. So every condition rendering in this "
        "glossary is an AUTHORED proposal and none can be defended by 'cn.json already says so'. "
        "Same for ALIENRPG.Resolve, ResolveMod, FastAction, SlowAction and Panic.",
        "ARMOR: cn.json disagrees with itself. ALIENRPG.Armor / ArmorRating / InventoryArmorHeader "
        "all say 护甲 but ALIENRPG.SHIP-ARMOR ('ARMOR') says 盔甲. 盔甲 is helmet-and-plate body "
        "armour and is wrong for a spaceship. The spine is 护甲.",
        "ALIENRPG.None renders as 莫 and ALIENRPG.OneTurn as 一斡. Both are stratum-B MT artefacts "
        "— 斡 is not a Chinese counter for anything. v1.0 keeps None frozen English and corrects "
        "OneTurn to 一节 (see the Turn entry).",
    ],
    "frozen_count_reconciliation": {
        "_why": "The owner's decision 4 names '9 first-run import lookup names, 11 hard-coded "
                "RollTable names, 2 folder names'. Re-deriving from source gives different "
                "DISTINCT-STRING counts. Recorded rather than silently reconciled.",
        "import_lookup": "9 lookup SITES over 6 DISTINCT strings, because 'Alien Evolved Core "
                         "Rules' and 'Alien Evolved Starter Set' each name BOTH an Adventure and a "
                         "Scene. The folder 'Alien Tables' is filed under frozen-folder.",
        "rolltable": "10 distinct literals in module JS, not 11 — actor.mjs :554, :845, :1067, "
                     ":1812, :1816, :1817, :1830 (x2 case variants), :1848, :1855. The 11th is "
                     "'EV - 48a. LS - DANGER EVENT DETAIL', which lives in the corerules pack's "
                     "own Macro.command, not in module JS. v1.0 keys it; Phase 0 did not.",
        "folder": "3 recorded, of which 2 are UNCONDITIONALLY frozen ('Alien Creature Tables', "
                  "'Alien Mother Tables' — rollTableData.mjs:7 and :24 use .find() with no "
                  "fallback and throw on rename). 'Alien Tables' is only conditionally frozen: "
                  "init.mjs:49 short-circuits on the `imported` setting first, so on an "
                  "established world a rename is harmless and on a fresh one it causes a duplicate "
                  "full import.",
        "citation_path": "The frozen folder lookups are in module/helpers/rollTableData.mjs (:7, "
                         ":24) — the LIVE file. module/actor/old-rollTableData.js:7/:21 is the dead "
                         "V1 copy. 'Alien Sub-Tables' appears ONLY at "
                         "module/apps/migratefolders.js:29, and migratefolders is import-commented "
                         "at module/apps/init.mjs:2, so it is NOT frozen and may be translated.",
        "item_names": "6 talent-name comparisons, not 2. The first register carried only PACK MULE "
                      "and TAKE CONTROL and missed NERVES OF STEEL, TOUGH, HARDENED and STOIC. "
                      "HARDENED and STOIC are LIVE — documents with those names ship today.",
    },
    "deliberately_out_of_scope": "58 Talent Item names and ~40 star-system designations live in the "
                                 "packs and are NOT terminology spine. They are content for the "
                                 "pack pass. The exceptions are the four talent names the code "
                                 "compares against, which v1.0 keys as T-FROZEN.",
}

LADDER_DEVIATIONS_INTRO = (
    "Places where this build did NOT take the highest rung that spoke. Each one is here because a "
    "higher-rung reading was demonstrably wrong for this text, not because it was merely "
    "unfamiliar. The ladder decides ties; it does not license importing an error.")
