# -*- coding: utf-8 -*-
"""Revise glossary_alien.* from v0 -> v0.2.

v0 (produced earlier the same day) is kept as the base: its English-gated counts
were re-derived this session and every cn.json / init.mjs / actor.mjs citation in
it checked out.  This pass does four things and nothing else:

  1. FIX one factual error in _meta.corrections_to_the_survey (the condition
     count) plus three citation paths, and correct the two frozen item-name keys
     to the surface the packs actually ship.
  2. ADD the coverage the brief asks for that v0 missed: the Xenomorph life-cycle
     STAGE vocabulary, the pack's structural folder labels, and the hardware /
     Colonial-Marine common-noun set.  Every addition is grounded in a name
     re-derived from 6-工作区/raw-dumps/*.json or in a key re-derived from
     systems/alienrpg/lang/{en,cn}.json this session.
  3. MOVE the Evolved caste ladder and the named-planet set to pending, because
     they are coined designations with no Chinese source and the project rule
     forbids guessing one.
  4. REGENERATE glossary_alien.json from provenance so the two can never drift,
     and recompute every count in _meta from the data.

Run:  python build_glossary_v0_2.py
"""
import json, collections, io, os, sys

G = r"C:/Users/Taka/Desktop/fvtt/Alien-RPG Translation Project/7-其他内容/glossary"
LANG = r"C:/Users/Taka/AppData/Local/FoundryVTT/Data/systems/alienrpg/lang"

SUBS = "CnSCG subtitle corpus (ONE vote — single fansub lineage)"
STRA = "cn.json stratum A (TI-130, human, 2020-11..2021-01)"
STRB = "cn.json stratum B (maintainer bulk MT, 2022-2026) — NOT a source"
PACK = "pack content, re-derived from 6-工作区/raw-dumps this session"
WEB = "terminology-web.json (zh.wikipedia / 百度百科 / community)"
STD = "standard modern Chinese technical usage (COMMON nouns only)"
DERV = "derived from stratum-A convention"


def load(name):
    with io.open(os.path.join(G, name), encoding="utf-8") as f:
        return json.load(f)


def dump(name, obj):
    p = os.path.join(G, name)
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.write("\n")
    return p, os.path.getsize(p)


def T(cn, tier, cat, why, cands, **kw):
    d = {"cn": cn, "tier": tier, "category": cat, "why_this_won": why, "candidates": cands}
    d.update(kw)
    return d


def C(zh, source, count, evidence):
    return {"zh": zh, "source": source, "count": count, "evidence": evidence}


prov = load("glossary_alien.provenance.json")
disp = load("glossary_alien.disputes.json")
pend = load("glossary_alien.pending.json")
terms = prov["terms"]

# ---------------------------------------------------------------- 1. FIXES ---
meta = prov["_meta"]

# 1a. The condition count. v0 asserted "20 entries and 'fatigued' is NOT one of
#     them".  Re-derived from source this session: 21 entries and fatigued IS
#     one of them.  Both the survey and v0 were wrong, in opposite directions.
corr = [c for c in meta["corrections_to_the_survey"] if "fatigued" not in c]
corr.append(
    "CONDITIONS — v0's own correction was itself wrong and is retracted. Re-derived "
    "this session by brace-matching the object literal at module/helpers/config.mjs:161: "
    "ALIENRPG.conditions has TWENTY-ONE entries, not 20 and not 19, and 'fatigued' IS "
    "one of them (resp:\"\", tableNumber:0). The 21 split 7 / 12 / 2: stress-response "
    "tableNumber 1-7 = jumpy, tunnelvision, aggravated, shakes, frantic, deflated, messup; "
    "panic-response tableNumber 1-12 = spooked, noisy, twitchy, loseitem, paranoid, "
    "hesitant, freeze, seekcover, scream, flee, frenzy, catatonic; and two that sit on no "
    "response table = fatigued and keepingguard. The brief's '20 condition words' is the "
    "21 minus fatigued. 'Fatigued' has been added to the glossary as a result."
)
corr.append(
    "CONDITION LANG COVERAGE — all 21 condition keys are UNTRANSLATED in cn.json: 20 are "
    "absent outright and ALIENRPG.messup is present but holds the English 'Mess Up'. So "
    "every condition rendering in this glossary is an AUTHORED proposal, not an attestation. "
    "None of them can be defended by 'cn.json already says so'. Same for ALIENRPG.Resolve, "
    "ALIENRPG.ResolveMod, ALIENRPG.FastAction, ALIENRPG.SlowAction and ALIENRPG.Panic, all "
    "of which cn.json leaves English."
)
corr.append(
    "CITATION PATH — v0 cited the frozen folder lookups as 'rollTableData.mjs:7 and :24'. "
    "The lines are right but the path is ambiguous: there are two files. The LIVE one is "
    "module/helpers/rollTableData.mjs (:7 'Alien Creature Tables', :24 'Alien Mother Tables'). "
    "module/actor/old-rollTableData.js:7/:21 is the dead V1 copy. A third literal, "
    "'Alien Sub-Tables', appears ONLY at module/apps/migratefolders.js:29, and migratefolders "
    "is import-commented at module/apps/init.mjs:2 — so 'Alien Sub-Tables' is a real folder in "
    "all three packs but is NOT frozen and may be translated."
)
corr.append(
    "FROZEN ITEM NAMES — the code does not compare against the literal 'PACK MULE'. All five "
    "sites uppercase FIRST: module/sheets/character-sheet.mjs:491, module/sheets/colony-sheet.mjs:339, "
    "module/sheets/synthetic-sheet.mjs:478 (i.name.toUpperCase() === \"PACK MULE\") and "
    "module/data/actor-character.mjs:440, module/data/actor-synthetic.mjs:430 "
    "(Attrib.name.toUpperCase() === \"TAKE CONTROL\"). The item as SHIPPED is 'Pack Mule' in title "
    "case (raw-dumps/corerules.json items). Case therefore does not matter; the Latin string "
    "surviving intact does. A bilingual name breaks it: '驮马 Pack Mule'.toUpperCase() is "
    "'驮马 PACK MULE', which is not equal to 'PACK MULE'. Both case forms are keyed here."
)
corr.append(
    "ALIENRPG.None — cn.json renders the 'None' sentinel as 莫, and ALIENRPG.OneTurn as 一斡. "
    "Both are stratum-B MT artefacts (斡 is not a Chinese counter for a game turn). Because "
    "actor.mjs:1922-1941 switches on localize('ALIENRPG.None') + ' ', the current file happens "
    "to be self-consistent, but 'None' is a T-FROZEN sentinel by owner decision and this "
    "glossary keeps it English. Re-derived from lang/cn.json this session."
)
corr.append(
    "ARMOR — cn.json disagrees with itself: ALIENRPG.Armor / ArmorRating / InventoryArmorHeader "
    "all say 护甲, but ALIENRPG.SHIP-ARMOR ('ARMOR') says 盔甲. 盔甲 is body armour of the "
    "helmet-and-plate kind and is wrong for a spaceship. The spine is 护甲; 盔甲 is recorded as "
    "REJECTED. Re-derived this session."
)
meta["corrections_to_the_survey"] = corr

# 1b. The 12 skill names were nearly rejected as MT until the pattern showed up.
meta["skill_scheme_finding"] = {
    "_why": "The T-EXACT tier freezes the 12 skill-stunt Item names to cn.json's ALIENRPG.Skill* "
            "values. Two of those values (Comtech -> 科技, Medical Aid -> 医疗) look like MT "
            "flattening and were candidates for correction.",
    "finding": "They are not MT. Re-derived from lang/cn.json this session, ALL TWELVE skill values "
               "are EXACTLY two Han characters: 肉搏 指挥 科技 机械 操纵 医疗 机动 观察 驾驶 射击 耐力 求生. "
               "A uniform 2-character width across twelve unrelated English words of 6 to 16 letters "
               "cannot arise from machine translation; it is a deliberate house style, chosen so the "
               "skill column of the character sheet aligns. This is stratum-A human work.",
    "consequence": "Do NOT 'fix' 科技 to 通讯技术 or 医疗 to 医疗救护. Widening any one of them breaks "
                   "the scheme and the sheet layout. The looser readings are recorded as aliases only.",
    "note": "ALIENRPG.SkillheavyMachAbb ('Heavy Mach.') also resolves to 机械, i.e. the abbreviation "
            "and the full name are the same string in Chinese. Harmless — 机械 is already at the "
            "2-character floor.",
}

# 1c. frozen item-name keys
terms.pop("PACK MULE", None)
terms.pop("TAKE CONTROL", None)
FIX_FROZEN = {
    "Pack Mule": ("Pack Mule", "the surface the pack actually ships"),
    "PACK MULE": ("PACK MULE", "the uppercased form the code compares against"),
    "Take Control": ("Take Control", "title-case form, for the Item name field"),
    "TAKE CONTROL": ("TAKE CONTROL", "the uppercased form the code compares against"),
}
for k, (v, note) in FIX_FROZEN.items():
    terms[k] = T(
        v, "T-FROZEN", "frozen-item-name",
        "Talent Item name read by string equality after .toUpperCase(). Any Chinese in the name "
        "field — bare or bilingual — makes the comparison fail and the talent silently stops working.",
        [C(v, "system source, re-derived this session",
           "n/a — frozen",
           "character-sheet.mjs:491 / colony-sheet.mjs:339 / synthetic-sheet.mjs:478 for PACK MULE; "
           "actor-character.mjs:440 / actor-synthetic.mjs:430 for TAKE CONTROL. Shipped surface "
           "'Pack Mule' re-derived from raw-dumps/corerules.json.")],
        note="Keyed in both cases because " + note + "; the comparison is case-insensitive but the "
             "Latin string must survive. Put the Chinese in the description, never in the name.",
    )

# 1d. split the two collisions that are actually harmful
terms["Flamethrower"] = T(
    "喷火器", "T-BILINGUAL", "hardware",
    "Split from Incinerator Unit. v0 mapped both to 喷火枪, but 'M240 Incinerator Unit' is a "
    "distinct shipped Item and the two would be indistinguishable in the weapon list. 喷火器 is "
    "the standard Chinese for a flamethrower; -枪 wrongly implies a sidearm.",
    [C("喷火器", STD, "en-gated 0", "No occurrence in either corpus; a common noun settled by convention."),
     C("喷火枪", "v0 (superseded)", "n/a", "Collided with Incinerator Unit.")],
    bilingual_name="喷火器 Flamethrower",
)
terms["Incinerator Unit"] = T(
    "焚烧器", "T-BILINGUAL", "hardware",
    "Split from Flamethrower. The M240 is a purpose-built incinerator, not a flamethrower, and "
    "the packs ship both. 焚烧 preserves 'incinerate'.",
    [C("焚烧器", STD, "en-gated 0", "Item 'M240 Incinerator Unit', raw-dumps/corerules.json."),
     C("喷火枪", "v0 (superseded)", "n/a", "Collided with Flamethrower.")],
    bilingual_name="焚烧器 Incinerator Unit",
    note="Sibling Item 'Multi-Directional Flame Unit' should follow this word, not Flamethrower's.",
)

# 1e. Armor Piercing — cn.json's value is a noun, the game term is a property
terms["Armor Piercing"] = T(
    "破甲", "T-PLAIN", "yze-mechanics",
    "Weapon PROPERTY, not a munition. cn.json's 穿甲弹 means 'armour-piercing ROUND' and cannot "
    "attach to a knife, a bolt gun or acid. Open as D9 because it overrides a lang value.",
    [C("破甲", WEB + " / TRPG convention", "en-gated 0",
       "terminology-web.json lists 破甲 at confidence low. Reads as a property in Chinese."),
     C("穿甲", STD, "en-gated 0", "The literal reading; also a property. Acceptable second choice."),
     C("穿甲弹", "cn.json ALIENRPG.ArmorPiercing", "en-gated 1",
       "Re-derived this session. A noun; wrong part of speech for a weapon feature.")],
    dispute="D9-armor-piercing",
)

# ------------------------------------------------------------------ 2. ADDS ---
ADD = {}

# --- Xenomorph life cycle: the STAGE vocabulary (the castes themselves go pending)
for roman, num, zh in [("I", 1, "一"), ("II", 2, "二"), ("III", 3, "三"),
                       ("IV", 4, "四"), ("V", 5, "五"), ("VI", 6, "六")]:
    ADD["Stage %s" % roman] = T(
        "第%s阶段" % zh, "T-PLAIN", "franchise-creature",
        "Every Evolved creature Actor is named '<caste> (Stage <roman> Xenomorph)'. The stage "
        "label is a plain ordinal and is the one part of that name string that needs no research.",
        [C("第%s阶段" % zh, STD, "en-gated 0",
           "Actor names re-derived from raw-dumps/{corerules,starterset}.json, e.g. "
           "'EV - Ovomorph (Stage I Xenomorph)', 'EV - Queen (Stage VI Xenomorph)'.")],
        note="Roman numeral -> Chinese ordinal. Do NOT keep the Roman numeral: 第I阶段 mixes scripts.",
    )
ADD["Life Cycle"] = T(
    "生命周期", "T-PLAIN", "franchise-creature",
    "Standard biological term; the six Stage labels are its steps.",
    [C("生命周期", STD, "en-gated 0", "Universal in Chinese biology writing.")],
)
ADD["Caste"] = T(
    "品级", "T-PLAIN", "franchise-creature",
    "品级 is the established Chinese term for a eusocial-insect caste, which is exactly the "
    "metaphor the Xenomorph ladder runs on. 种姓 is the human/Indian sense and is wrong here.",
    [C("品级", STD, "en-gated 0", "Entomological standard (蚁群品级 / 蜂群品级)."),
     C("种姓", STD, "en-gated 0 — REJECTED", "Human social caste; wrong register.")],
)
ADD["Xenomorphology"] = T(
    "异形学", "T-BILINGUAL", "franchise-creature",
    "Talent Item name. Built on the settled 异形 spine with the standard -学 suffix.",
    [C("异形学", DERV, "en-gated 0", "Item 'Xenomorphology', raw-dumps/corerules.json.")],
    bilingual_name="异形学 Xenomorphology",
)
ADD["Fatigued"] = T(
    "疲劳", "T-PLAIN", "yze-condition",
    "The 21st entry of ALIENRPG.conditions, missed by v0 because v0 wrongly believed the list "
    "had 20 and excluded fatigued. Distinct from Exhausted: fatigued is a status effect, "
    "Exhausted is the Trials-and-Hazards state.",
    [C("疲劳", STD, "en-gated 0 — key untranslated in cn.json",
       "config.mjs:161 block, entry 8 of 21, resp:\"\" tableNumber:0. ALIENRPG.fatigued has no "
       "cn.json value; re-derived this session."),
     C("精疲力竭", STRB, "en-gated 1 — REJECTED for this key",
       "cn.json TAH.exhausted renders 'Exhausted' as 精疲力竭. Reserve that string for Exhausted "
       "so the two states stay distinguishable.")],
)

# --- Trials and Hazards states (the four TAH keys with a cn value)
for en_, zh, alt, ev in [
    ("Exhausted", "精疲力竭", None, "cn.json TAH.exhausted, re-derived this session."),
    ("Starving", "挨饿", None, "cn.json TAH.starving '一天没有食物后，您就会挨饿'."),
    ("Dehydrated", "脱水", None, "cn.json TAH.dehydrated '脱水有多种影响'."),
    ("Freezing", "冷冻", "受冻",
     "cn.json TAH.freezing '冷冻有几个影响'. 冷冻 is transitive-ish (to freeze something); "
     "受冻 is the state a character is in. Kept as attested, alias recorded."),
]:
    cands = [C(zh, STRB, "en-gated 1", ev)]
    if alt:
        cands.append(C(alt, STD, "en-gated 0", "Better register for a character state."))
    ADD[en_] = T(zh, "T-PLAIN", "yze-mechanics",
                 "Trials-and-Hazards state. Attested in cn.json; stratum B, but these four are "
                 "plain vocabulary where MT and a human land in the same place.",
                 cands, **({"aliases": [alt]} if alt else {}))

# --- structural folder / type labels the Babele folders mapping needs
FOLDERS = [
    ("Creatures", "生物", "cn.json ALIENRPG.CreatureSkill '该生物没有此技能' and ROLLONCREATURETABLE '绘制外星生物表' both use 生物.", "T-PLAIN"),
    # NOT T-FROZEN: nothing in the code reads this string. It is a deliberate
    # non-translation, which is a T-PLAIN value that happens to be Latin.
    ("NPCs", "NPC", "cn.json ALIENRPG.NPC keeps 'NPC' untranslated; the Latin acronym is universal at Chinese tables. Singular because Chinese does not inflect for number.", "T-PLAIN"),
    ("Weapons", "武器", "cn.json ALIENRPG.InventoryWeaponsHeader 'Weapons' -> 武器.", "T-PLAIN"),
    ("Equipment", "装备", "Standard; no competing value in cn.json.", "T-PLAIN"),
    # NOTE: the 'Armor' folder label reuses the existing 'Armor' term (护甲) and is
    # therefore not re-added here; the 盔甲 rejection is appended to that term below.
    ("Spaceships", "星际飞船", "cn.json ALIENRPG.Hull/HULL -> 船体 fixes the ship word family; 星际飞船 disambiguates from 载具.", "T-PLAIN"),
    ("Vehicles", "载具", "cn.json ALIENRPG.fullCrew renders 'The Vehicle is full' as 车辆已满, but 车辆 excludes the VTOL gyrocar and the dropship that the Vehicles folder actually holds. 载具 is the TRPG-standard superset.", "T-PLAIN"),
    ("Vehicle Weapons", "载具武器", "Compound of the two above.", "T-PLAIN"),
    ("Spaceship Weapons", "星际飞船武器", "cn.json ALIENRPG.SpacecraftWeapons is left English; compound of the settled parts.", "T-PLAIN"),
    ("Spaceship Modules/Upgrades", "星际飞船模块/升级", "cn.json ALIENRPG.MODULES-UPGRADES -> 模块/升级, prefixed.", "T-PLAIN"),
    ("Talents (Career)", "天赋（职业）", "cn.json ALIENRPG.Talents -> 天赋, ALIENRPG.Career -> 职业. Full-width parentheses because the whole label is Chinese.", "T-PLAIN"),
    ("Talents (General)", "天赋（通用）", "cn.json ALIENRPG.GeneralTalent 'General Talent' -> 通用天赋.", "T-PLAIN"),
    ("Careers", "职业", "cn.json:76.", "T-PLAIN"),
    ("Skill-Stunts", "技能炫技", "cn.json ALIENRPG.Skills -> 技能 and ALIENRPG.Stunts -> 炫技.", "T-PLAIN"),
    ("Alien Sub-Tables", "异形子表", "NOT frozen — see the citation-path correction. 子表 is the standard word for a nested RollTable.", "T-PLAIN"),
    # The one bilingual folder label: the others are generic category words, this one
    # is a proper-noun book title, and the owner's T-BILINGUAL tier covers proper nouns.
    ("The Art of Alien", "异形艺术设定集", "Journal folder of concept art. 设定集 is the Chinese trade term for an art book. Bilingual because it is a book TITLE, unlike the generic category folders around it.", "T-BILINGUAL"),
]
for en_, zh, ev, tier in FOLDERS:
    kw = {}
    if tier == "T-BILINGUAL":
        kw["bilingual_name"] = zh + " " + en_
    ADD[en_] = T(zh, tier, "pack-structure",
                 "Folder label shipped by the packs; re-derived from raw-dumps this session. "
                 "Needed by the Babele folders mapping.",
                 [C(zh, STRA if "cn.json" in ev else STD, "en-gated 0", ev)], **kw)

# --- hardware / ops common nouns the packs actually need
HARDWARE = [
    ("Frigate", "护卫舰", "hardware", STD, "en-gated 0 in subs (0 English rows).",
     "Actor 'CONESTOGA-CLASS FRIGATE'. 护卫舰 is the PRC naval standard; 巡防舰 is the Taiwan form and is rejected under the register rule."),
    ("Shuttle", "穿梭机", "hardware", STD, "en-gated 4 for 太空梭 — REJECTED as Taiwan register.",
     "Actor 'EV - STARCUB SHUTTLE'. The corpus says 太空梭 (ALIEN1979 1:15:15, 1:15:22), which the brief names as one of the Taiwan-flavoured renderings. 穿梭机 is the mainland form."),
    ("Turret", "炮塔", "hardware", STD, "en-gated 0.",
     "Items 'Light/Medium/Heavy Railgun Turret', 'Phased Plasma Pulse Cannon Turret'."),
    ("Hardpoint", "挂点", "hardware", STD, "en-gated 0 — ALIENRPG.hardpoint left English in cn.json.",
     "Items 'Added Hardpoint, size I..III'. 挂点 is the standard aviation term for a weapon station."),
    ("Bulkhead", "舱壁", "hardware", STD, "en-gated 0.", "Item 'Armored Bulkheads'."),
    ("Pressure Suit", "增压服", "hardware", STD, "en-gated 0.",
     "Item 'IRC Mk.35 Pressure Suit'. Distinct Item from the Compression Suit; both ship."),
    ("Compression Suit", "压缩服", "hardware", STD, "en-gated 0.",
     "Item 'IRC Mk.50 Compression Suit'; also named in ALIENRPG.ShipPanic14 (untranslated). "
     "Kept distinct from Pressure Suit; the T-BILINGUAL tail disambiguates on the name field."),
    ("Cutting Torch", "切割炬", "hardware", STD, "en-gated 0.",
     "Items 'Cutting Torch' and 'Mechanical Cutting Torch*'."),
    ("Grenade Launcher", "榴弹发射器", "hardware", STD, "en-gated 0.",
     "Item 'Armat U1 Grenade Launcher'. 榴弹发射器 is the PLA standard term."),
    ("Grenade", "手榴弹", "hardware", STD, "en-gated 0.", "Items 'M40 HEDP Grenade', 'G2 Electroshock Grenade'."),
    ("Combat Knife", "战斗刀", "hardware", STD, "en-gated 0.", "Item 'Combat Knife'."),
    ("Stun Baton", "电击棍", "hardware", STD, "en-gated 0.",
     "Item 'Stun Baton'. NOTE the brief flags 电波枪 as a Taiwan-flavoured corpus rendering; it is not used here."),
    ("Service Pistol", "制式手枪", "hardware", STD, "en-gated 0.",
     "Items 'M4A3 Service Pistol', 'VP-70MA6 Service Pistol'."),
    ("Optical Scope", "光学瞄具", "hardware", STD, "en-gated 0.", "Item 'Optical Scope'."),
    ("Binoculars", "双筒望远镜", "hardware", STD, "en-gated 0.", "Item 'Binoculars'."),
    ("Flashlight", "手电筒", "hardware", STD, "en-gated 0.", "Item 'Flashlight'."),
    ("Personal Medkit", "个人医疗包", "hardware", STD, "en-gated 0.", "Item 'Personal Medkit'."),
    ("Surgical Kit", "手术包", "hardware", STD, "en-gated 0.", "Item 'Surgical Kit*'."),
    ("Power Cell", "电池组", "hardware", STD, "en-gated 0.", "Item 'Power Cell'."),
    ("Comm Unit", "通讯装置", "hardware", STD, "en-gated 0.",
     "Item 'Comm Unit'. Uses 通讯, the same head word the Comtech skill's alias uses."),
    ("Tactical Nuke", "战术核弹", "hardware", STD, "en-gated 0.", "Item 'Tactical Nuke'."),
    ("Sensor Drone", "传感器无人机", "hardware", STD, "en-gated 0.",
     "Item 'Sensor Drones'. 无人机 here is the UAV sense and does NOT collide with the Xenomorph "
     "Drone caste, which is never rendered 无人机."),
    ("Docking Umbilical", "对接脐带管", "hardware", STD, "en-gated 0.",
     "Item 'Docking Umbilical'. 脐带管 is the standard aerospace term for an umbilical."),
    ("Cryo Deck", "低温舱室", "hardware", STD, "en-gated 0.",
     "Items 'Cryo Deck I..V'. Distinct from Cryo-tube (低温舱, the individual pod) — the deck is the compartment."),
    ("Science Lab", "科学实验室", "hardware", STD, "en-gated 0.", "Item 'Science Lab'."),
    ("Medlab", "医疗实验室", "hardware", STD, "en-gated 0.",
     "Item 'Medlab'. Distinct from Infirmary (医疗室), which is also a shipped surface."),
    ("Galley", "厨舱", "hardware", STD, "en-gated 0.",
     "Items 'Galley I..V'; also named in ALIENRPG.ShipPanic13 (untranslated). Nautical 厨舱, not 厨房."),
    ("Cargo Bay", "货舱", "hardware", STD, "en-gated 0.", "Items 'Cargo Bay I..V'."),
    ("Hangar", "机库", "hardware", STD, "en-gated 0.", "Items 'Hangar I..V'."),
    ("Vehicle Bay", "载具舱", "hardware", STD, "en-gated 0.", "Items 'Vehicle Bay I..V'."),
    ("Air Scrubber", "空气净化器", "hardware", STD, "en-gated 0.", "Items 'Air Scrubbers I..IV'."),
    ("Reactor", "反应堆", "hardware", STD, "en-gated 0.",
     "ALIENRPG.ShipPanic15 'Reactor breach' (untranslated). Consistent with the settled Fusion Reactor 核聚变反应堆."),
    ("Escape Vehicle", "逃生舱", "hardware", STD, "en-gated 0.",
     "Items 'Emergency Escape Vehicle I..III'. Same Chinese as Escape Pod, deliberately: the packs use EEV and escape pod for one thing."),
    ("Zero-G", "零重力", "ops", STD, "en-gated 0.", "Talent 'Zero-G Training'."),
    ("EVA", "舱外活动", "ops", STD, "en-gated 0.",
     "Talent 'EVA Specialist', Item 'Rexim RXF-M5A3 EVA Pistol'. Spell it out; the Latin acronym is not current in Chinese TRPG text."),
]
for en_, zh, cat, src, cnt, ev in HARDWARE:
    kw = {}
    if cat == "hardware":
        kw["bilingual_name"] = zh + " " + en_
    ADD[en_] = T(zh, "T-BILINGUAL" if cat == "hardware" else "T-PLAIN", cat,
                 "Common noun. Settled by standard modern Chinese technical usage, which the "
                 "project rule permits for common nouns but never for a proper noun or a model "
                 "designation — those go to pending.",
                 [C(zh, src, cnt, ev)], **kw)

# --- Colonial-Marine vocabulary
MARINE = [
    ("Squad", "小队", "Talent 'Squad Leader' and Actor 'EV - SQUAD LEADER'. 小队 not 班: the "
     "Colonial Marine squad in Aliens is a mixed 12-person unit, not an infantry section.",
     "en-gated 0 for 小队; the corpus renders 'squad' only inside 'fall back by squads', which it mistranslates."),
    ("Squad Leader", "小队长", "Compound of Squad.", "en-gated 0."),
    ("Corps", "陆战队", "cn.json has no key. Corpus ALIENS1986 0:28:59 'I love the corps!' -> 我爱死陆战队, "
     "which is cn_only under a /marine/ gate but exactly right under a /corps/ one.",
     "en-gated 1 under '\\bcorps\\b'."),
    ("Marines", "陆战队员", "Plural of the settled Marine. en-gated 2 in the corpus "
     "(ALIENS1986 0:33:17 'Morning, marines' -> 早安 陆战队员; 1:09:54).",
     "en-gated 2 of 10 rows matching '\\bmarine'."),
]
for en_, zh, why, cnt in MARINE:
    ADD[en_] = T(zh, "T-PLAIN", "marine",
                 why,
                 [C(zh, SUBS if "corpus" in why or "ALIENS" in why else STD, cnt,
                    "Gate re-run this session with 4-常用脚本/tm/term_gate.py --mode subs.")])
ADD["Marines"]["candidates"].append(
    C("海军陆战队", SUBS, "en-gated 1 — REJECTED",
      "ALIEN3 1:52:11 'they sent in marines' -> 上次派去的海军陆战队. 海军 = navy. The United States "
      "Colonial Marine Corps is not a naval service in this setting, and the same corpus renders the "
      "adjectival form correctly as 殖民陆战队 (ALIENS1986 0:22:07). A back-formation from the "
      "real-world USMC."))

# --- designations that stay Latin
ADD["LV-223"] = T(
    "LV-223", "T-FROZEN", "franchise-place",
    "Catalogue designation, same treatment as the already-settled LV-426: Latin+digits, "
    "untranslated and untransliterated.",
    [C("LV-223", PACK, "n/a", "Item 'LV-223', raw-dumps/corerules.json. Sibling of Item 'LV-426'.")],
)

for k, v in ADD.items():
    if k in terms:
        raise SystemExit("ADD collides with an existing key: %r" % k)
    terms[k] = v

# --- enrich two terms that already existed and that this pass re-derived
ENRICHED = []
terms["Armor"]["candidates"].append(
    C("盔甲", STRB, "en-gated 1 — REJECTED",
      "cn.json ALIENRPG.SHIP-ARMOR ('ARMOR') -> 盔甲, re-derived this session. cn.json is "
      "internally inconsistent: ALIENRPG.Armor, ALIENRPG.ArmorRating and "
      "ALIENRPG.InventoryArmorHeader all say 护甲. 盔甲 is helmet-and-plate body armour and is "
      "wrong on a spaceship hull. 护甲 is the spine; the SHIP-ARMOR key must be corrected to match."))
terms["Armor"]["note"] = ("Doubles as the Babele label for the 'Armor' Item folder shipped by the "
                          "core-rules pack (raw-dumps/corerules.json folders).")
ENRICHED.append("Armor")
terms["Bug Hunt"]["candidates"].append(
    C("除虫行动", SUBS, "en-gated 1 — re-derived this session",
      "ALIENS1986 0:33:31 'Is this gonna be a stand-up fight, sir, or another bug hunt?' -> "
      "这是一场战斗还是除虫行动? Gate re-run with term_gate.py --mode subs --en '\\bwarrior' "
      "surfaced this row as cn_only on 战斗, which is how the rendering was located. One of the very "
      "few corpus renderings that is both idiomatic and unambiguous, so the ONE-vote discount does "
      "not sink it."))
ENRICHED.append("Bug Hunt")

# ----------------------------------------------------- 2b. SCHEMA REPAIRS ---
# (i) Nine v0 entries carry no `candidates` at all, so the brief's "every candidate
#     rendering, its source, its count" is unmet for them.  Each is a term DERIVED
#     from a sibling rather than sourced on its own; that is a legitimate answer,
#     but it has to be stated, not left as a hole.
DERIVED_FROM = {
    "Base Dice": ("Base Die", "English plural of an already-settled term. Chinese has no plural "
                              "inflection, so the two share one rendering by construction."),
    "Stress Dice": ("Stress Die", "English plural of an already-settled term."),
    "Cryogenic Compartment": ("Cryo-tube", "The packs and the films use both surfaces for one thing."),
    "extraterrestrial": ("Alien", "The ADJECTIVE half of the owner's 异形 / 外星 split: 异形 is the "
                                  "creature noun, 外星 is the adjective. Never interchangeable."),
    "Alien 3 (1992 film)": ("Aliens (1986 film)", "Series numbering, following 异形2."),
    "Alien: Covenant (2017 film)": ("Alien (1979 film)", "zh.wikipedia 异形系列 title list."),
    "Alien: Earth (2025 series)": ("Alien (1979 film)", "zh.wikipedia 异形系列 title list."),
    "Prometheus (2012 film)": ("Alien (1979 film)", "zh.wikipedia 异形系列 title list."),
    "Weyland-Yutani Corporation": ("Weyland-Yutani", "The long form of the same proper noun. 集团 "
                                                     "already carries 'Corporation'; do NOT append 公司."),
}
REPAIRED_CANDIDATES = []
for k, (parent, why) in DERIVED_FROM.items():
    v = terms[k]
    if "candidates" in v:
        continue
    v["derived_from"] = parent
    v["candidates"] = [C(v["cn"], "derived from the entry for %r — see its candidates" % parent,
                         "inherited", why)]
    REPAIRED_CANDIDATES.append(k)

# (ii) Nine keys carry a parenthetical DISAMBIGUATOR that is part of the glossary key,
#      not part of the name.  v0 baked it into bilingual_name, which would have put
#      "异形3 Alien 3 (1992 film)" on a page-name field.  Strip it.
REPAIRED_BILINGUAL = []
for k, v in terms.items():
    b = v.get("bilingual_name")
    if not b:
        continue
    base = k.split(" (")[0] if (" (" in k and k.endswith(")")) else k
    want = v["cn"] + " " + base
    if b != want:
        v["bilingual_name"] = want
        v["key_disambiguator"] = k[len(base):].strip() or None
        REPAIRED_BILINGUAL.append(k)

# ------------------------------------------------------------- 3. TO PENDING ---
def P(kind, cands, corpus, why, nxt, **kw):
    d = {"kind": kind, "candidates": cands, "corpus": corpus,
         "why_unresolved": why, "next_step": nxt}
    d.update(kw)
    return d


CASTE_PATTERN = (
    "The only two caste renderings with any source at all are terminology-web.json's "
    "Drone -> 工蜂异形 and Warrior -> 战士异形, both at confidence 'medium' and both from "
    "community writing rather than a reference work. Re-gated this session against the "
    "subtitle corpus: '\\bdrone' matches 0 English rows, '\\bwarrior' matches 0, "
    "'\\bpraetorian' 0, '\\bneomorph' 0. The corpus cannot vote on ANY caste name. "
    "Extending 工蜂/战士异形 into a nine-rung ladder would be coining a taxonomy from a "
    "two-point sample, which is exactly what the no-guessing rule forbids."
)
CASTES = [
    ("Praetorian", "Stage VI caste, Actor 'EV - Praetorian (Stage VI Xenomorph)'",
     [("禁卫异形", "禁卫 is the standard Chinese for the Roman Praetorian Guard; + the 异形 pattern."),
      ("近卫异形", "近卫 is the alternative rendering of Praetorian Guard.")]),
    ("Scout", "Stage IV caste, Actor 'EV - Scout (Stage IV Xenomorph)'",
     [("侦察异形", "侦察 = scout/recon.")]),
    ("Stalker", "Stage IV caste, Actor 'EV - Stalker (Stage IV Xenomorph)'",
     [("潜行异形", "潜行 = stalk/prowl."), ("猎杀异形", "Emphasises the hunt rather than the stealth.")]),
    ("Sentry", "Stage V caste, Actor 'EV - Sentry (Stage V Xenomorph)'",
     [("哨戒异形", "哨戒 = sentry duty. Collides with the proposed reading of the UA 571-C Sentry Gun.")]),
    ("Soldier", "Stage V caste, Actor 'EV - Soldier (Stage V Xenomorph)'",
     [("士兵异形", "Plain."), ("战士异形", "REJECTED here — already spoken for by Warrior.")]),
    ("Worker", "Stage V caste, Actor 'EV - Worker (Stage V Xenomorph)'",
     [("劳工异形", "Avoids 工蜂, which Drone already holds."), ("工兵异形", "Reads as a combat engineer.")]),
    ("Charger / Crusher", "Stage VI caste, Actor 'EV - Charger/Crusher (Stage VI Xenomorph)'",
     [("冲撞异形 / 碾压异形", "A slash-joined double name; needs a ruling on whether to keep the slash.")]),
    ("Neomorph", "Alien: Covenant species; Actors 'EV - Adult Neomorph (Stage V)', 'EV - Neophyte (Stage IV Neomorph)'",
     [("新形异形", "Parallel to 异形 on the -morph suffix."),
      ("白异形", "The common fan handle, from the creature's colour. Not a translation.")]),
    ("Neomorphic Bloodburster", "Stage III Neomorph, Actor 'EV - Neomorphic Bloodburster (Stage III Neomorph)'",
     [("破血体", "Built on the settled Chestburster -> 破胸体 pattern (破X体).")]),
    ("Neophyte", "Stage IV Neomorph", [("新生体", "Literal; unsourced.")]),
    ("Bambi Burster", "Stage III Xenomorph, the Alien-3 runner burster",
     [("(none)", "An in-joke English nickname. No Chinese source and no obvious calque.")]),
    ("Imp", "Stage III Xenomorph", [("小鬼异形", "Unsourced.")]),
    ("Queenburster", "Stage III Xenomorph",
     [("女王破胸体", "Compound of the settled Queen and Chestburster — but Queen itself is dispute D6, "
       "so this cannot be settled before D6 is.")]),
    ("Royal Facehugger", "Stage II Xenomorph",
     [("皇家抱脸虫", "Compound of the owner-decreed 抱脸虫.")]),
    ("Praeto-Facehugger", "Stage II Xenomorph",
     [("禁卫抱脸虫", "Depends on Praetorian, which is itself pending.")]),
    ("Harvester", "Non-Xenomorph creature; Actors 'EV - Harvester', 'EV - Harvester Juvenile'",
     [("收割者", "Literal. No Alien-specific attestation.")]),
    ("Lion Worm", "Non-Xenomorph creature, Actor 'EV - Lion Worm'",
     [("狮蠕虫", "Literal compound; unsourced.")]),
    ("XX121", "The Weyland-Yutani species designation; Actor 'EV -  XX121 - Stage IV Xenomorph'",
     [("XX121", "Almost certainly stays Latin, like LV-426 and LV-223 — but it is a species "
       "designation rather than a catalogue number, so it needs the nod rather than the rule.")]),
]
for en_, where, cands in CASTES:
    pend["terms"][en_] = P(
        "proper noun — Xenomorph/creature caste designation",
        [{"zh": z, "source": "composed this session from the attested pattern, NOT a source",
          "count": "en-gated 0", "note": n} for z, n in cands],
        "en-gated 0. Gate re-run this session: no caste word has a single English-side hit "
        "in the 4,665-row corpus.",
        CASTE_PATTERN,
        "Read the Evolved core-rules bestiary text once the packs are extracted — it may gloss "
        "the ladder well enough to translate as a set. Failing that, an owner ruling on the "
        "whole ladder AT ONCE; deciding these one at a time guarantees an inconsistent taxonomy.",
        shipped_as=where,
    )
pend["terms"]["United States Colonial Marine Corps (USCM)"] = P(
    "proper noun — military organisation",
    [{"zh": "美国殖民陆战队", "source": "composed from the settled Colonial Marine -> 殖民陆战队",
      "count": "en-gated 0", "note": "The adjective is settled; the 'United States' half is not."},
     {"zh": "USCM", "source": "keep the acronym", "count": "en-gated 0",
      "note": "Chinese military writing routinely keeps foreign service acronyms."}],
    "en-gated 0 — the brief already flags the corpus as having zero USCM attestation, and the "
    "re-run confirms it.",
    "The settled 殖民陆战队 covers the adjectival use ('a Colonial Marine'). The full service name "
    "and its acronym are a separate decision, and the setting's United States is not the "
    "present-day one.",
    "Owner ruling, together with United Systems Military (联合星系军), which is already in the "
    "glossary and sets the pattern for a service name.",
)
pend["terms"]["Named planets and colonies (Evolved core rules)"] = P(
    "proper nouns — a SET, to be ruled on as a set",
    [{"zh": "(per-name transliteration)", "source": "none", "count": "en-gated 0",
      "note": "Calpamos, Thedus, Torin Prime, Volcus, Tanaka 5, Linna 349, Fiorina 161 and "
              "'GJ1187 “SHĀNMÉN”'. Re-derived from raw-dumps/corerules.json items."}],
    "en-gated 0 for every one of them.",
    "Same class as Acheron: transliteration with no Chinese attestation. Note GJ1187 “SHĀNMÉN” "
    "is already pinyin for 山门 and should probably be RESTORED to 山门 rather than transliterated "
    "back — that one is a Chinese name that made a round trip through English.",
    "One owner ruling fixing a transliteration policy, then apply it mechanically. The pure "
    "catalogue designations (KOI-*, GJ*, GL*, ALPHA BOÖTIS, 70 OPHIUCHI, ...) need no ruling: "
    "they stay Latin, like LV-426.",
    count_note="~40 designations; only the pronounceable names need a decision.",
)
pend["_meta"]["count"] = len(pend["terms"])
pend["_meta"]["actionable_count"] = len(pend["terms"])
pend["_meta"]["version"] = "v0.2"
pend["_meta"]["generated"] = "2026-08-29"
pend["_meta"]["added_in_v0_2"] = (
    "The Evolved Xenomorph caste ladder, the Neomorph line, the non-Xenomorph creatures, USCM and "
    "the named-planet set. v0 missed them because it worked from the survey's term lists rather "
    "than from the pack inventory; this pass re-derived every Actor and Item name from "
    "6-工作区/raw-dumps/*.json and found 18 creature designations the glossary had no entry for. "
    "They are pending rather than glossed because the subtitle corpus has ZERO English-side hits "
    "for any caste word and no reference work covers them."
)

# --------------------------------------------------------------- 4. DISPUTES ---
disp["disputes"]["D8-shift"] = {
    "en": "Shift",
    "provisional_cn": "轮班",
    "tier": "T-PLAIN",
    "why_open": "Shift is BOTH a time unit the rules count in AND a bare English literal the "
                "Critical-Injury heal-time parser compares against. cn.json renders the time unit "
                "two different ways in two different places while leaving the parser literal English.",
    "sides": [
        {"cn": "轮班",
         "argument": "The form on the enumerated time-unit key, which is the one the heal-time "
                     "table actually feeds. Keeps One Shift / One Turn / One Round / One Day a set.",
         "sources": ["cn.json ALIENRPG.OneShift"],
         "gated_count": "en-gated 1 on '\\bOne Shift\\b'",
         "evidence": "cn.json ALIENRPG.OneShift 'One Shift' -> 一轮班, re-derived this session."},
        {"cn": "班次",
         "argument": "The form in the running prose a player actually reads, and it is the more "
                     "natural Chinese noun for a work shift. 轮班 is really the verb (to work shifts).",
         "sources": [STRB],
         "gated_count": "en-gated 2 inside TAH prose",
         "evidence": "cn.json TAH.exhausted '您会倒下并睡一个班次' and '一旦您睡了至少一个班次'; "
                     "TAH.freezing uses 轮班 in the same paragraph position — the file is not even "
                     "internally consistent."},
    ],
    "what_would_settle_it": "Owner ruling. Whichever wins must be applied to BOTH TAH.exhausted and "
                            "ALIENRPG.OneShift, or the heal-time table and the hazard prose will "
                            "describe the same duration with two different words.",
    "blast_radius": "ALIENRPG.OneShift, TAH.exhausted, TAH.freezing, TAH.starving, TAH.dehydrated, "
                    "and every Critical-Injury heal-time cell.",
    "do_not_confuse_with": "actor.mjs:1888 compares testArray[9] === \"Shift\" against a BARE English "
                           "literal with no localize() call. That comparison is unaffected by this "
                           "dispute and the table cell must stay English either way. See "
                           "provenance._meta.lockstep_literals.",
}
disp["disputes"]["D9-armor-piercing"] = {
    "en": "Armor Piercing",
    "provisional_cn": "破甲",
    "tier": "T-PLAIN",
    "why_open": "The provisional value overrides a lang value that is not obviously wrong, only "
                "wrong in part of speech. Any override of a shipped string is an owner call.",
    "sides": [
        {"cn": "破甲",
         "argument": "Armor Piercing is a weapon PROPERTY that attaches to knives, bolt guns and "
                     "acid attacks as well as to bullets. 破甲 is an adjectival property in Chinese "
                     "and is the TRPG convention.",
         "sources": [WEB],
         "gated_count": "en-gated 0",
         "evidence": "terminology-web.json lists 破甲 at confidence 'low' — it is a convention "
                     "call, not an attestation, and is recorded as such."},
        {"cn": "穿甲弹",
         "argument": "It is what the shipped file says, and for the pulse-rifle case it is exactly right.",
         "sources": ["cn.json ALIENRPG.ArmorPiercing"],
         "gated_count": "en-gated 1",
         "evidence": "cn.json ALIENRPG.ArmorPiercing 'Armor Piercing' -> 穿甲弹, re-derived this "
                     "session. 弹 means 'round/projectile'; the string cannot attach to a "
                     "Combat Knife or to Acid Splash."},
        {"cn": "穿甲",
         "argument": "Splits the difference: the literal reading, and a property rather than a noun.",
         "sources": [STD],
         "gated_count": "en-gated 0",
         "evidence": "Drops only the 弹 from the shipped value, so it is the smallest edit that "
                     "fixes the part of speech."},
    ],
    "what_would_settle_it": "Owner ruling between the convention (破甲) and the minimal fix (穿甲). "
                            "穿甲弹 is not defensible as-is because of the non-firearm carriers.",
    "blast_radius": "ALIENRPG.ArmorPiercing plus every weapon Item that carries the feature.",
}
disp["_meta"]["count"] = len(disp["disputes"])
disp["_meta"]["version"] = "v0.2"
disp["_meta"]["generated"] = "2026-08-29"
disp["_meta"]["added_in_v0_2"] = (
    "D8-shift and D9-armor-piercing, both found by re-deriving lang/cn.json this session rather "
    "than by reading the survey. The seven disputes the brief required are D1-D7 and are unchanged "
    "except that their citations were re-derived and all of them held."
)

# ------------------------------------------------------- 5. REGEN + RECOUNT ---
glossary = {k: v["cn"] for k, v in sorted(terms.items(), key=lambda kv: kv[0].lower())}

rev = collections.defaultdict(list)
for k, v in terms.items():
    rev[v["cn"]].append(k)
collisions = {zh: sorted(ks) for zh, ks in sorted(rev.items()) if len(ks) > 1}

meta["version"] = "v0.2"
meta["generated"] = "2026-08-29"
meta["count"] = len(terms)
meta["count_is_authoritative"] = True
meta["tier_counts"] = dict(sorted(collections.Counter(v["tier"] for v in terms.values()).items()))
meta["category_counts"] = dict(sorted(collections.Counter(v["category"] for v in terms.values()).items()))
meta["chinese_value_collisions"]["collisions"] = collisions
# v0 dumped the collision list raw, which makes a reader check all of them by hand.
# Classify instead: only one class can actually hurt.
CASE_VARIANT = {"发狂", "呆滞", "失去知觉", "寻找掩护", "尖叫", "逃跑", "颤抖"}
INFLECTION = {"压力骰", "基础骰", "殖民陆战队", "陆战队员", "职业", "恐慌"}
SAME_REFERENT = {"低温舱", "异形", "辐射点", "逃生舱", "韦兰-尤坦尼集团", "巢穴", "生物"}
DIFFERENT_AXIS = {"远程"}
UNDER_DISPUTE = {"生化人"}
_cls = {}
for zh in collisions:
    _cls[zh] = ("case-variant — the SHOUTING panic-table label and the lower-case condition "
                "key are the same word; one Chinese string is correct for both"
                if zh in CASE_VARIANT else
                "english-inflection — singular/plural or noun/participle of one term; Chinese "
                "does not inflect, so one string is correct for both"
                if zh in INFLECTION else
                "same-referent — two English surfaces the packs use for one thing"
                if zh in SAME_REFERENT else
                "DIFFERENT AXIS — the two never appear in one dropdown, but they DO appear on "
                "one character sheet. 远程 is both the Long range band and the Ranged weapon "
                "type. Watch this one; it is the same shape as the Engaged/Melee collision that "
                "became dispute D2."
                if zh in DIFFERENT_AXIS else
                "UNDER DISPUTE — see D5-synthetic. Android and Synthetic are different things in "
                "the fiction and collapsing them is a decision, not a coincidence."
                if zh in UNDER_DISPUTE else
                "unclassified — review")
meta["chinese_value_collisions"]["classification"] = {
    "_why": "A collision is only a bug when a player can see both English terms rendered as the "
            "same Chinese in ONE control. Classified so a reader does not have to re-derive all "
            "%d by hand." % len(collisions),
    "harmful": sorted(zh for zh in collisions if _cls[zh].startswith(("DIFFERENT", "UNDER", "unclassified"))),
    "benign": sorted(zh for zh in collisions if not _cls[zh].startswith(("DIFFERENT", "UNDER", "unclassified"))),
    "per_value": {zh: _cls[zh] for zh in sorted(collisions)},
}
meta["revision_history"] = [
    {"version": "v0", "what": "First build. 275 terms from seed_candidates + terminology-{subs,web} "
                              "+ lang/cn.json stratum A."},
    {"version": "v0.2", "what": "Adversarial re-derivation pass. Every cn.json line number, every "
                                "init.mjs / actor.mjs / rollTableData.mjs anchor and the Queen gate "
                                "were re-derived from source and HELD. One factual error was found "
                                "and retracted (the condition count), four citations were made "
                                "precise, two harmful Chinese-value collisions were split, and the "
                                "pack inventory was mined for the first time — which is where all "
                                "the new coverage and all the new pending entries came from."},
]
meta["what_v0_2_changed"] = {
    "retracted": "provenance._meta.corrections_to_the_survey said ALIENRPG.conditions has 20 entries "
                 "and excludes 'fatigued'. It has 21 and includes it. See the replacement entry.",
    "added_terms": None,   # filled below
    "split_collisions": ["Flamethrower / Incinerator Unit (both were 喷火枪)",
                         "Cryo Deck / Cryo-tube (低温舱室 vs 低温舱)"],
    "rekeyed": "'PACK MULE' / 'TAKE CONTROL' are now keyed in BOTH cases, because the code "
               "uppercases before comparing and the shipped Item is title-case 'Pack Mule'.",
    "moved_to_pending": 18 + 2,
    "new_disputes": ["D8-shift", "D9-armor-piercing"],
    "enriched_in_place": ENRICHED,
    "schema_repairs": {
        "candidates_backfilled": {
            "_why": "v0 left these nine with no candidates array, so there was no record of what "
                    "was considered or rejected. Each is derived from a sibling entry; that is now "
                    "stated explicitly via terms[<en>].derived_from.",
            "keys": sorted(REPAIRED_CANDIDATES)},
        "bilingual_name_disambiguator_stripped": {
            "_why": "The glossary KEY for a film title carries a disambiguator — 'Alien 3 (1992 film)' "
                    "— so that it does not collide with 'Alien'. v0 concatenated the whole key into "
                    "bilingual_name, which would have written '异形3 Alien 3 (1992 film)' onto a "
                    "page-name field. bilingual_name now uses the name only; the disambiguator is "
                    "preserved separately as terms[<en>].key_disambiguator.",
            "keys": sorted(REPAIRED_BILINGUAL)},
    },
}
meta["bilingual_name_contract"] = (
    "terms[<en>].bilingual_name is ALWAYS exactly terms[<en>].cn + one ASCII space (U+0020) + the "
    "English NAME — which is the glossary key minus any trailing ' (...)' disambiguator. No "
    "parentheses around the English, no full-width space, no en space. Asserted on every "
    "T-BILINGUAL entry by the validator; the build fails if any entry deviates.")
meta["known_gaps"] = [g for g in meta["known_gaps"] if "not yet extracted" not in g]
meta["known_gaps"].insert(0,
    "The 3 content packs are still not extracted to compendium/en, so no compendium-mode gate "
    "could be run. This pass worked around that by mining 6-工作区/raw-dumps/*.json for Actor, Item "
    "and Folder NAMES, which is enough to establish COVERAGE but not enough to gate a rendering "
    "against pack PROSE. Re-run every disputed term in compendium mode once the packs are dumped.")
meta["known_gaps"].append(
    "58 Talent Item names (Analysis, Authority, Banter, ... Zero-G Training) and ~40 star-system "
    "designations are in the packs and are NOT in this glossary. That is deliberate: they are "
    "content to be translated in the pack pass, not terminology spine. They are listed here so "
    "their absence reads as a decision rather than an oversight.")

new_keys = sorted(set(ADD) | set(FIX_FROZEN))
meta["what_v0_2_changed"]["added_terms"] = {"count": len(new_keys), "keys": new_keys}

paths = []
paths.append(dump("glossary_alien.json", glossary))
paths.append(dump("glossary_alien.provenance.json", prov))
paths.append(dump("glossary_alien.disputes.json", disp))
paths.append(dump("glossary_alien.pending.json", pend))

sys.stdout.reconfigure(encoding="utf-8")
print("terms      %d  (v0 was 275)" % len(terms))
print("tiers     ", meta["tier_counts"])
print("categories", len(meta["category_counts"]))
print("disputes   %d" % len(disp["disputes"]))
print("pending    %d" % len(pend["terms"]))
print("collisions %d" % len(collisions))
for p, n in paths:
    print("%9d  %s" % (n, p))
