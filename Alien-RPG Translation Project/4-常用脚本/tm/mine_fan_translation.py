# -*- coding: utf-8 -*-
"""Mine the 《异形RPG：进化版》 Chinese fan translation into a machine-readable term table.
Every quote is re-derived from the source file at build time; nothing is hand-typed."""
import io, json, os, re, sys

BASE = r"C:/Users/Taka/Desktop/fvtt/Alien-RPG Translation Project/7-其他内容/reference/cn-fan-translation"
OUT  = r"C:/Users/Taka/Desktop/fvtt/Alien-RPG Translation Project/7-其他内容/glossary/mined_fan_translation.json"

FILES = {
    "ch1":   "异形RPG：进化版-第一章：宇宙是地狱.txt",
    "ch2":   "异形RPG：进化版-第二章：你的角色.txt",
    "old1e": "《异形RPG》版异形宇宙设定翻译·时间线·第一部分-这是旧版的 非新版Evolved.txt",
}
TXT = {k: open(os.path.join(BASE, v), encoding="utf-8").read().replace("\r", "").replace("\xa0", " ")
       for k, v in FILES.items()}

ENDERS = "。！？\n"
def quote(chap, needle, before=0, after=0, maxlen=170):
    """Return the sentence containing `needle`, expanded by `before`/`after` sentences."""
    t = TXT[chap]
    i = t.find(needle)
    if i < 0:
        raise SystemExit("NEEDLE NOT FOUND in %s: %r" % (chap, needle))
    # walk back
    s = i
    for _ in range(before + 1):
        j = max(t.rfind(c, 0, s) for c in ENDERS)
        s = j if j >= 0 else -1
        if s < 0:
            s = -1
            break
    s += 1
    # walk forward
    e = i + len(needle)
    for _ in range(after + 1):
        cands = [t.find(c, e) for c in ENDERS]
        cands = [c for c in cands if c >= 0]
        e = (min(cands) + 1) if cands else len(t)
    # a one-clause quote is useless for judging register: keep extending forward
    while len(re.sub(r"\s+", " ", t[s:e].strip())) < 46 and e < len(t):
        cands = [t.find(c, e) for c in ENDERS]
        cands = [c for c in cands if c >= 0]
        if not cands:
            e = len(t); break
        e = min(cands) + 1
    # still short (start of file / start of a run of tiny table rows): extend backward
    while len(re.sub(r"\s+", " ", t[s:e].strip())) < 46 and s > 0:
        j = max(t.rfind(c, 0, max(0, s - 1)) for c in ENDERS)
        if j < 0:
            s = 0; break
        s = j + 1
    q = t[s:e].strip()
    q = re.sub(r"\s+", " ", q)
    if len(q) > maxlen:
        k = q.find(needle)
        lo = max(0, k - maxlen // 2)
        q = ("…" if lo else "") + q[lo:lo + maxlen] + "…"
    return q

TERMS = []
def T(en, zh, tier, chap, needle, glossed=False, conflict=None, conf="high",
      before=0, after=0, note=None):
    d = {
        "en": en, "zh": zh, "tier_hint": tier, "chapter": chap,
        "quote": quote(chap, needle, before, after),
        "inline_glossed": glossed,
        "internal_conflict": conflict,
        "confidence": conf,
    }
    if note:
        d["note"] = note
    TERMS.append(d)

# ── stable conflict strings ────────────────────────────────────────────────────
C_FIORINA  = "菲奥莉娜161 (ch1 ×2) vs 费奥莉娜161 (ch1 ×1); the inline-glossed instance is 费奥莉娜 but the document's own majority reading is 菲奥莉娜 — adopt 菲奥莉娜161"
C_ANDROID  = "机器人 (ch1 ×1 glossed 'androids', ch2 ×18) vs 仿生人 (ch1 ×1, 'David 7 series') vs 合成人 (ch1 ×1 'synthetic people', ch2 ×1 'Synthetic characters'); the document's dominant reading is 机器人=Android and 合成人=Synthetic, with 仿生人 a one-off stray"
C_SYNTH    = "合成人 renders 'synthetic' in both chapters, but ch2 uses it as an apposition to 机器人 in the same sentence ('机器人：合成人角色…'), so the doc treats Android and Synthetic as one referent under two labels"
C_MOTHER   = "Game Mother rendered 游戏管理员 (heading, glossed) / 老妈 (heading '成为老妈 (Game MOTHER)') / 游戏主持人 (body, glossed 'GM'); the ship AI MOTHER is ALSO 老妈, so 老妈 is doubly loaded"
C_WITS     = "fan uses 智力 for Wits, but 智力 also appears inside the Wits gloss itself ('感官感知、智力和理智' = 'Intelligence, alertness…'), so 智力 is doubly loaded; system lang cn.json A-stratum has 机智 and the owner has pinned 机智"
C_EMPATHY  = "fan uses 同理心 for Empathy, but 同理心 also appears inside its own gloss ('个人魅力、同理心以及操控他人的能力'); system lang cn.json A-stratum has 共情 and the owner has pinned 共情"
C_FREELEAG = "自由联盟 (ch1) vs untranslated 'Free League' (ch2), for the same publisher in the same sentence role (download character sheets)"
C_MARINES  = "殖民海军陆战队 (ch1 ×6, ch2 ×2) vs 殖民地海军陆战队 (ch1 ×2); 殖民海军陆战队 is the majority and matches the campaign-framework list"
C_PRESSUIT = "IRC MK.35 'Pressure Suit' and IRC MK.50 'Compression Suit' are BOTH rendered 压力服 — two distinct armor items in corerules.json collapse to one Chinese string"
C_WYD      = "韦汤币 (glossed 'W-Y dollar') vs bare W-Y币 (all nine career cash lines) vs '300刀W-Y币' (worked example)"

# ══════════════════════════════════════════════════════════════════════════════
# 1. ATTRIBUTES  (inline-glossed, ch2)
# ══════════════════════════════════════════════════════════════════════════════
T("Strength", "力量", "T-EXACT", "ch2", "力量 (Strength)", True, None, "high")
T("Agility", "敏捷", "T-EXACT", "ch2", "敏捷 (Agility)", True, None, "high")
T("Wits", "智力", "T-EXACT", "ch2", "智力 (Wits)", True, C_WITS, "high",
  note="REJECT for the Foundry build: owner-pinned exception, level-4 cn.json 机智 wins")
T("Empathy", "同理心", "T-EXACT", "ch2", "同理心 (Empathy)", True, C_EMPATHY, "high",
  note="REJECT for the Foundry build: owner-pinned exception, level-4 cn.json 共情 wins")
T("Attributes", "属性", "T-EXACT", "ch2", "属性你的角色有四个属性", False, None, "high")
T("key attribute", "关键属性", "T-PLAIN", "ch2", "列为\"关键属性\"的属性", False, None, "high")

# ══════════════════════════════════════════════════════════════════════════════
# 2. THE TWELVE SKILLS  (not inline-glossed; English fixed by career key-skill lists)
# ══════════════════════════════════════════════════════════════════════════════
SK_QUOTE = "内特的关键技能是重型机械、耐力和近战"
T("Skills", "技能", "T-EXACT", "ch2", "游戏中共有十二种核心技能", False, None, "high")
T("Heavy Machinery", "重型机械", "T-EXACT", "ch2", SK_QUOTE, False,
  "system lang cn.json A-stratum has 机械 (bare)", "high")
T("Stamina", "耐力", "T-EXACT", "ch2", SK_QUOTE, False, None, "high",
  note="agrees with system lang cn.json A-stratum 耐力")
T("Close Combat", "近战", "T-EXACT", "ch2", SK_QUOTE, False,
  "system lang cn.json A-stratum has 肉搏", "high")
T("Comtech", "计算机科学", "T-EXACT", "ch2", "1级计算机科学、侦察和医疗", False,
  "system lang cn.json A-stratum has 科技; fan's 计算机科学 over-narrows a skill that also covers comms and electronics", "high")
T("Observation", "侦察", "T-EXACT", "ch2", "1级计算机科学、侦察和医疗", False,
  "system lang cn.json A-stratum has 观察; the fan doc itself uses 观察 once in a non-skill sense", "high")
T("Medical Aid", "医疗", "T-EXACT", "ch2", "1级计算机科学、侦察和医疗", False, None, "high",
  note="agrees with system lang cn.json A-stratum 医疗")
T("Mobility", "机动", "T-EXACT", "ch2", "关键技能：机动、生存、侦察", False, None, "high",
  note="agrees with system lang cn.json A-stratum 机动")
T("Survival", "生存", "T-EXACT", "ch2", "关键技能：机动、生存、侦察", False,
  "system lang cn.json A-stratum has 求生", "high")
T("Ranged Combat", "远程战斗", "T-EXACT", "ch2", "关键技能：近战、耐力、远程战斗", False,
  "system lang cn.json A-stratum has 射击", "high")
T("Command", "指挥", "T-EXACT", "ch2", "关键技能：侦察、远程战斗、指挥", False, None, "high",
  note="agrees with system lang cn.json A-stratum 指挥")
T("Manipulation", "操控", "T-EXACT", "ch2", "关键技能：计算机科学、侦察、操控", False,
  "system lang cn.json A-stratum has 操纵 (same sound, different second character)", "high")
T("Piloting", "驾驶", "T-EXACT", "ch2", "关键技能：驾驶、远程战斗、计算机科学", False, None, "high",
  note="agrees with system lang cn.json A-stratum 驾驶")
T("key skills", "关键技能", "T-PLAIN", "ch2", "关键技能：近战、耐力、远程战斗", False, None, "high")
T("skill level", "技能等级", "T-PLAIN", "ch2", "在游戏过程中，你可以提升你的技能等级", False, None, "high")
T("skill roll", "技能检定", "T-PLAIN", "ch2", "机器人不能追骰技能检定", False, None, "high")

# ══════════════════════════════════════════════════════════════════════════════
# 3. DICE / STRESS / PANIC ENGINE
# ══════════════════════════════════════════════════════════════════════════════
T("base dice", "基础骰子", "T-PLAIN", "ch1", "基础骰子（黑色）和压力骰子（黄色）", False, None, "high")
T("stress dice", "压力骰子", "T-PLAIN", "ch1", "基础骰子（黑色）和压力骰子（黄色）", False, None, "high")
T("Stress", "压力", "T-EXACT", "ch2", "压力太空生活是致命的", False, None, "high",
  note="agrees with system lang cn.json A-stratum 压力")
T("stress level", "压力等级", "T-PLAIN", "ch2", "这种不断增加的紧张感通过你的压力等级来体现", False, None, "high")
T("push (a roll)", "追骰", "T-PLAIN", "ch2", "会因追骰（见第43页）或经历令人害怕或紧张的情况而增加", False, None, "high",
  note="load-bearing: the fan doc never once uses 推骰/加骰; 追骰 is its single stable rendering of Push, used 11× in ch2")
T("Panic", "恐慌", "T-EXACT", "ch2", "机器人永远不会进行恐慌检定", False, None, "high",
  note="agrees with system lang cn.json A-stratum 恐慌")
T("panic roll", "恐慌检定", "T-PLAIN", "ch2", "机器人永远不会进行恐慌检定", False, None, "high")
T("Resolve", "精神强度", "T-EXACT", "ch2", "你需要精神强度——一个数值评分", False,
  "system lang cn.json A-stratum leaves ALIENRPG.Resolve untranslated (null), so there is no level-4 competitor", "high")
T("Health", "生命值", "T-EXACT", "ch2", "生命值即使你保持冷静", False,
  "system lang cn.json A-stratum has bare 生命 for ALIENRPG.Health", "high")
T("Story Points", "故事点", "T-EXACT", "ch2", "你将获得一个故事点", False, None, "high",
  note="agrees with system lang cn.json A-stratum 故事点")
T("Experience Points (XP)", "经验值", "T-PLAIN", "ch2", "用经验值(XP)来衡量", True, None, "high",
  note="ch2 also uses 经验点 once ('1点经验点奖励') for the same thing")
T("D66", "D66", "T-FROZEN", "ch1", "另一种掷骰类型是D66", False, None, "high")

# ══════════════════════════════════════════════════════════════════════════════
# 4. TIME UNITS  ⚠ Shift is the crit-parse lockstep literal
# ══════════════════════════════════════════════════════════════════════════════
T("Round", "轮", "T-EXACT", "ch1", "轮 (Round)", True, None, "high",
  note="table row: 轮 / 5-10秒 / 战斗. Body prose also writes 战斗轮 and 每轮战斗. System lang cn.json A-stratum has 一回合 for ALIENRPG.OneRound, i.e. 回合 not 轮 — a real level-1/level-4 clash")
T("Stretch", "节", "T-EXACT", "ch1", "节 (Stretch)", True, None, "high",
  note="table row: 节 / 5-10分 / 潜行. Used in ch2 body as 每节时间之后 and 1节潜行模式. No level-4 competitor: cn.json has no Stretch key")
T("Shift", "班", "T-EXACT", "ch1", "班 (Shift)", True, None, "high",
  note="⚠ RUNTIME: Shift is the crit-parse lockstep literal. Table row 班 / 5-10小时 / 恢复. ch2 body drifts to 班次 twice ('至少接受老师一个班次的指导', '每天的每个班次'). System lang cn.json leaves ALIENRPG.Shift as the English word 'Shift' and renders ALIENRPG.OneShift as 一轮班")
T("shift (body-text variant)", "班次", "T-PLAIN", "ch2", "至少接受老师一个班次的指导", False,
  "班 (table, glossed) vs 班次 (body ×2) for the same unit", "high")
T("combat round", "战斗轮", "T-PLAIN", "ch1", "战斗轮、节和班的准确持续时间会根据情况变化", False, None, "high")
T("countdown", "倒计时", "T-PLAIN", "ch1", "倒计时：在某些灾难性事件的倒计时期间", False, None, "high")

# ══════════════════════════════════════════════════════════════════════════════
# 5. GAME MODES / FRAMEWORKS / GM
# ══════════════════════════════════════════════════════════════════════════════
T("cinematic play", "电影模式", "T-PLAIN", "ch1", "可以以两种不同的模式进行：电影模式和战役模式", False,
  "ch2 once writes 剧情模式 for the same mode ('在剧情模式中每幕一次')", "high")
T("campaign play", "战役模式", "T-PLAIN", "ch1", "可以以两种不同的模式进行：电影模式和战役模式", False, None, "high")
T("campaign framework", "战役框架", "T-PLAIN", "ch1", "核心规则书支持三种不同的战役框架", False, None, "high")
T("Space Truckers (framework)", "太空卡车司机", "T-PLAIN", "ch1", "1.4边境的职业太空卡车司机", False, None, "high",
  note="one of the three campaign frameworks; also the ch1 career-overview heading")
T("Colonial Marines (framework)", "殖民海军陆战队", "T-PLAIN", "ch1", "殖民海军陆战队美国殖民海军陆战队代表了有史以来最精锐的作战力量", False, C_MARINES, "high")
T("Frontier Colonists (framework)", "边境殖民者", "T-PLAIN", "ch1", "边境殖民者对大多数人来说，成为殖民者意味着", False, None, "high")
T("Game Mother", "游戏管理员", "T-PLAIN", "ch1", "游戏管理员 (the Game MOTHER)", True, C_MOTHER, "medium")
T("Game Mother (2nd rendering)", "老妈", "T-PLAIN", "ch1", "成为老妈 (Game MOTHER)", True, C_MOTHER, "medium")
T("GM / Game Master", "游戏主持人", "T-PLAIN", "ch1", "最后一名玩家是游戏主持人，简称GM", True, C_MOTHER, "high")
T("MOTHER (ship AI)", "老妈", "T-PLAIN", "ch1", "老妈妥协了", False, C_MOTHER, "high",
  note="ship-AI MOTHER = 老妈 in all four log entries (老妈妥协了 / 老妈禁止打开气闸 / 老妈把我们的通讯发送到了HQ公司 / 老妈不肯帮我们锁上门)")
T("MOTHER (section title, untranslated)", "MOTHER", "T-FROZEN", "ch1", "1.1 MOTHER，发生了什么", False, C_MOTHER, "high",
  note="the ch1 §1.1 heading 'WHAT'S THE STORY MOTHER?' keeps MOTHER in Latin letters")
T("player character (PC)", "玩家角色", "T-PLAIN", "ch1", "都扮演一个玩家角色 (PC)", True, None, "high")
T("non-player character (NPC)", "非玩家角色", "T-PLAIN", "ch1", "由GM 控制的角色称为非玩家角色，简称NPC", False, None, "high")
T("Player versus Player (PvP)", "PVP", "T-FROZEN", "ch2", "PVP在《异形》角色扮演游戏中", False, None, "high")
T("act (of a cinematic adventure)", "幕", "T-PLAIN", "ch2", "在冒险的三幕的开始", False, None, "high")
T("game session", "游戏会话", "T-PLAIN", "ch2", "在每场游戏会话结束时", False,
  "ch2 also writes 游戏环节 once ('你参加了游戏环节吗？') for the same session", "high")
T("character sheet", "角色卡", "T-PLAIN", "ch2", "你需要一张角色卡", False, None, "high")
T("pre-generated character", "预设角色", "T-PLAIN", "ch1", "所有电影模式冒险均包含预设角色", False,
  "ch2 uses 预制 ('在预制电影模式冒险中') for the same thing", "high")
T("Free League (publisher)", "自由联盟", "T-PLAIN", "ch1", "你也可以从自由联盟网站下载空白角色卡", False, C_FREELEAG, "high")
T("Free League (untranslated)", "Free League", "T-FROZEN", "ch2", "从Free League网站下载并打印", False, C_FREELEAG, "high")
T("safety tools", "安全道具", "T-PLAIN", "ch1", "我们建议您在游戏前、游戏中和游戏后使用安全道具", False, None, "high")
T("life path method", "生命历程方法", "T-PLAIN", "ch2", "你可以用本书第282页列出的生命历程方法", False, None, "high")

# ══════════════════════════════════════════════════════════════════════════════
# 6. CHARACTER-CREATION / PERSONAL STUFF
# ══════════════════════════════════════════════════════════════════════════════
T("Career", "职业", "T-EXACT", "ch2", "你角色的第一个选择是你的职业", False, None, "high",
  note="agrees with system lang cn.json A-stratum 职业")
T("Talents", "天赋", "T-EXACT", "ch2", "天赋天赋是一些技巧、动作和小能力", False, None, "high",
  note="agrees with system lang cn.json A-stratum 天赋")
T("career talent", "职业天赋", "T-PLAIN", "ch2", "内特的职业天赋是坚韧", False, None, "high")
T("general talent", "通用天赋", "T-PLAIN", "ch2", "从你的职业天赋和通用天赋中自由选择一项新天赋", False, None, "high",
  note="agrees with system lang cn.json A-stratum 通用天赋")
T("Personal Agenda", "个人任务", "T-EXACT", "ch2", "个人任务的运作方式在电影模式和战役模式中有所不同", False,
  "system lang cn.json A-stratum has 个人目标 for ALIENRPG.PersonalAgenda and 目标 for ALIENRPG.AgendaStory; 任务 collides with 'mission'", "high")
T("Buddy", "朋友", "T-EXACT", "ch2", "你的PC可以在其他PC中拥有一个朋友和一个敌人", False,
  "system lang cn.json A-stratum has 哥们 for ALIENRPG.relOne", "high")
T("Rival", "敌人", "T-EXACT", "ch2", "你的PC可以在其他PC中拥有一个朋友和一个敌人", False,
  "system lang cn.json A-stratum has 对头 for ALIENRPG.relTwo; 敌人 collides with generic 'enemy' used throughout the talent text", "high")
T("Signature Item", "标志性物品", "T-EXACT", "ch2", "你还有一个标志性物品", False,
  "system lang cn.json A-stratum has 标志物品 (no 性)", "high")
T("Appearance", "外貌", "T-PLAIN", "ch2", "外貌用几句话描述你玩家角色的外貌", False,
  "ch2 career blocks head the same field 外观 in the 10-step list ('6.决定你的外观')", "high")
T("Cash", "现金", "T-PLAIN", "ch2", "现金：D6x100 W-Y币", False, None, "high")
T("W-Y dollars", "韦汤币", "T-PLAIN", "ch2", "你还会获得韦汤币（W-Y dollar）", True, C_WYD, "high")
T("Gear", "装备", "T-PLAIN", "ch2", "2.3 你的装备为了在异形的世界中生存", False, None, "high")
T("starting gear", "初始装备", "T-PLAIN", "ch2", "初始装备：在电影模式中", False, None, "high")
T("Encumbrance", "负重", "T-PLAIN", "ch2", "负重你可以轻松携带数量相当于你力量等级两倍的常规大小物品", False, None, "high")
T("Over-Encumbered", "超重", "T-PLAIN", "ch2", "超重：你可以暂时携带最多2倍于正常负重上限的物品", False, None, "high")
T("Heavy (item weight)", "重", "T-PLAIN", "ch2", "重（Heavy）&轻（Light）物品", True, None, "high")
T("Light (item weight)", "轻", "T-PLAIN", "ch2", "重（Heavy）&轻（Light）物品", True, None, "high")
T("Tiny (item weight)", "小", "T-PLAIN", "ch2", "小（Tiny）物品", True, None, "high")
T("regular item", "常规物品", "T-PLAIN", "ch2", "常规物品（在第5章的装备列表中重量为1）", False, None, "high")
T("Consumables", "消耗品", "T-PLAIN", "ch2", "这些称为消耗品", False, None, "high")
T("supply rating", "余量", "T-PLAIN", "ch2", "总的来说，这些被称为余量", False,
  "system lang cn.json A-stratum has 补给 (ALIENRPG.NoSupplys 补给已耗尽 / supplyDecreases 补给下降)", "high")
T("Supply Roll", "余量检定", "T-EXACT", "ch2", "余量检定：每隔固定的时间", False,
  "system lang cn.json A-stratum has 补给骰 for ALIENRPG.Supply", "high")
T("supply dial", "余量计数器", "T-PLAIN", "ch2", "余量计数器《希望的最后一天入门套装》包含一个有用的余量计数器", False, None, "high")
T("Air (consumable)", "空气", "T-PLAIN", "ch2", "空气：每节时间之后", False,
  "system lang cn.json A-stratum has 氧气供应 for ALIENRPG.AirSupply", "high")
T("Ammo (consumable)", "弹药", "T-PLAIN", "ch2", "弹药：大多数热武器都有弹匣来提供弹药", False, None, "high")
T("Power (consumable)", "电力", "T-PLAIN", "ch2", "电力：一些物品需要电力才能工作", False, None, "high")

# ══════════════════════════════════════════════════════════════════════════════
# 7. COMBAT / STATE VOCABULARY (not glossed, English fixed by the EN rulebook)
# ══════════════════════════════════════════════════════════════════════════════
T("Broken", "濒死", "T-PLAIN", "ch2", "激励濒死角色", False, None, "high",
  note="EN 'rally a broken character' → 激励濒死角色; also 效果持续到你或视线范围内所有敌人都濒死为止 = 'until you or all enemies in sight are broken'")
T("critical injury", "重伤", "T-PLAIN", "ch2", "你可以重骰任何死亡检定或重伤检定一次", False, None, "high")
T("death roll", "死亡检定", "T-PLAIN", "ch2", "你可以重骰任何死亡检定或重伤检定一次", False, None, "high")
T("zone", "区域", "T-PLAIN", "ch2", "当你在一个区域（见第57页）主动寻找隐藏的东西时", False, None, "high")
T("Short range", "近距离", "T-PLAIN", "ch2", "你和近距离内的所有人压力等级-2", False, None, "high")
T("safe place", "安全地点", "T-PLAIN", "ch2", "每在一个安全地点（见第73页）度过1节", False, None, "high")
T("quick action", "快动作", "T-PLAIN", "ch2", "作为快动作而不是完整动作", False, None, "high")
T("full action", "完整动作", "T-PLAIN", "ch2", "作为快动作而不是完整动作", False, None, "high")
T("stealth mode", "潜行模式", "T-PLAIN", "ch2", "你每次进行1节潜行模式（见第55页）", False, None, "high")
T("grapple", "擒抱", "T-PLAIN", "ch2", "当你尝试在近战（见第62页）中擒抱住对手时", False, None, "high")
T("first aid", "急救", "T-PLAIN", "ch2", "当你为了急救（见第68页）而进行医疗检定时", False, None, "high")
T("dodge", "躲避", "T-PLAIN", "ch2", "当你躲避攻击（见第65页）进行机动检定时", False, None, "high")
T("take cover", "寻找掩护", "T-PLAIN", "ch2", "不能寻找掩护、躲避或防御攻击", False, None, "high")
T("initiative", "先攻", "T-PLAIN", "ch1", "用于战斗中抽取先攻（见第5章）", False, None, "high")
T("Vacuum / world of hurt", "充满伤害的世界", "T-PLAIN", "ch2", "你就会进入一个充满伤害的世界", False, None, "low",
  note="literal calque of 'entering a world of hurt'; the EN cross-ref is to the Vacuum rules, so this is prose, not a term")

# ══════════════════════════════════════════════════════════════════════════════
# 8. SYNTHETICS
# ══════════════════════════════════════════════════════════════════════════════
T("Android", "机器人", "T-EXACT", "ch2", "扮演机器人机器人是《异形》角色扮演游戏中一个重要的组成部分", False, C_ANDROID, "high",
  note="18 hits in ch2; the ch1 timeline glosses it explicitly as '系列机器人 (androids)'")
T("Synthetic", "合成人", "T-EXACT", "ch2", "机器人：合成人角色在分配完14点属性后", False, C_SYNTH, "high",
  note="the single sentence where both labels meet: 机器人 heads the rule, 合成人 names the character type")
T("synthetic people (ch1 prose)", "合成人", "T-PLAIN", "ch1", "合成人扮演上帝", False, C_SYNTH, "high")
T("android (David-series, stray)", "仿生人", "T-PLAIN", "ch1", "韦兰德公司的大卫7代系列仿生人在劳动中变得很常见", False, C_ANDROID, "low",
  note="ONE-OFF. Two timeline lines apart, the same David series is 机器人 in 2023 and 仿生人 in 2066. Drop 仿生人")
T("secret android", "秘密机器人", "T-PLAIN", "ch2", "秘密机器人：多数机器人看起来像人类", False, None, "high")

# ══════════════════════════════════════════════════════════════════════════════
# 9. THE NINE CAREERS (headings; English fixed by corerules LIFE PATH POSTINGS tables)
# ══════════════════════════════════════════════════════════════════════════════
T("Colonial Marine (career)", "殖民海军陆战队", "T-EXACT", "ch2", "2.5 职业：殖民海军陆战队", False, C_MARINES, "high")
T("Colonial Marshal (career)", "殖民地执法官", "T-EXACT", "ch2", "2.6 职业：殖民地执法官", False,
  "ch1 uses bare 执法官 for the same office ('选举执法官来监督日常的执法工作', '我给殖民地执法官发了消息')", "high")
T("Company Agent (career)", "公司代理人", "T-EXACT", "ch2", "2.7 职业：公司代理人", False,
  "ch1 renders the same archetype 公司代表 ('公司代表在边境，企业的权力甚至超过政府') — that is EN 'Company Reps', a different heading, so both are defensible", "high")
T("Kid (career)", "儿童", "T-EXACT", "ch2", "2.8 职业：儿童", False, None, "high")
T("Medic (career)", "医生", "T-EXACT", "ch2", "2.9 职业：医生", False,
  "ch1 log entries call the same crew role 医务官 ('唯一的新人是医务官海斯'); ch2's worked example also uses 医务官海斯", "high")
T("Officer (career)", "飞行官", "T-EXACT", "ch2", "2.10 职业：飞行官", False,
  "EN is plain 'Officer'; 飞行官 imports 'flight' that the English does not have, and the same word is reused for the ICC Commercial Flight Officer licence", "medium")
T("Pilot (career)", "驾驶员", "T-EXACT", "ch2", "2.11 职业：驾驶员", False,
  "the SKILL Piloting is 驾驶 and the CAREER Pilot is 驾驶员 — distinct, but one character apart", "high")
T("Roughneck (career)", "杂工", "T-EXACT", "ch2", "2.12 职业：杂工", False, None, "high",
  note="ch1 log uses the same word for the crew: '内特和里德是杂工'")
T("Scientist (career)", "科学家", "T-EXACT", "ch2", "2.13 职业：科学家", False, None, "high")

# ══════════════════════════════════════════════════════════════════════════════
# 10. CAREER TALENTS (all inline-glossed in ch2)
# ══════════════════════════════════════════════════════════════════════════════
T("Banter", "玩笑", "T-EXACT", "ch2", "1.玩笑（banter）", True, None, "high")
T("Overkill", "滥杀", "T-EXACT", "ch2", "2.滥杀（overkill）", True, None, "high")
T("Precise Shooter", "精确射手", "T-EXACT", "ch2", "3.精确射手（precise shooter）", True, None, "high")
T("Authority", "权威", "T-EXACT", "ch2", "1. 权威（authority）", True, None, "high")
T("Investigator", "调查员", "T-EXACT", "ch2", "2. 调查员（investigator）", True, None, "high")
T("Subdue", "征服", "T-EXACT", "ch2", "3. 征服（subdue）", True, None, "medium",
  note="EN 'Subdue' is the grapple talent; 征服 ('conquer') over-reaches — 制伏 would match the mechanic")
T("Company Resources", "公司资源", "T-EXACT", "ch2", "1.公司资源（company resources）", True, None, "high")
T("Cunning", "欺诈", "T-EXACT", "ch2", "2.欺诈（cunning）", True, None, "medium",
  note="EN 'Cunning' is Wits-for-Empathy on Manipulation; 欺诈 ('fraud') is narrower than the English")
T("Personal Safety", "人身安全", "T-EXACT", "ch2", "3.人身安全（personal safety）", True, None, "high")
T("Hard to Catch", "抓不到我", "T-EXACT", "ch2", "1.抓不到我（hard to catch）", True, None, "high")
T("Lucky", "幸运", "T-EXACT", "ch2", "2.幸运（lucky）", True, None, "high")
T("Nimble", "敏锐", "T-EXACT", "ch2", "3.敏锐（nimble）", True, None, "medium",
  note="EN 'Nimble' is an Agility-push talent; 敏锐 reads as acuity/sharpness (a Wits word), 敏捷 is taken by the attribute — 灵巧 would fit better")
T("Field Medic", "野战外科", "T-EXACT", "ch2", "1.野战外科（field medic）", True, None, "medium",
  note="外科 = surgery, which collides with the separate Surgeon talent in the same career block")
T("Nurse", "护理", "T-EXACT", "ch2", "2.护理（nurse）", True, None, "high")
T("Surgeon", "外科医生", "T-EXACT", "ch2", "3.外科医生（surgeon）", True, None, "high")
T("Field Commander", "战场指挥", "T-EXACT", "ch2", "1.战场指挥（field commander）", True, None, "high")
T("Frontline Leader", "前线领导", "T-EXACT", "ch2", "2.前线领导（frontline leader）", True, None, "high")
T("Pull Rank", "官大一级", "T-EXACT", "ch2", "3.官大一级（pull rank）", True, None, "high")
T("Full Throttle", "全速前进", "T-EXACT", "ch2", "1.全速前进（full throttle）", True, None, "high")
T("Like the Back of Your Hand", "了如指掌", "T-EXACT", "ch2", "2.了如指掌（like the back of your hand）", True, None, "high")
T("Reckless", "鲁莽", "T-EXACT", "ch2", "3.鲁莽（reckless）", True, None, "high")
T("Resilient", "坚韧", "T-EXACT", "ch2", "1.坚韧（resillient）", True, None, "high",
  note="fan doc misspells the English twice: 'Resllient' in the worked example, 'resillient' in the Roughneck block. Correct headword is Resilient")
T("Steady Hands", "稳定的双手", "T-EXACT", "ch2", "2.稳定的双手（Steady hand）", True, None, "high",
  note="fan doc writes 'Steady Hands' in the worked example and 'Steady hand' (singular, lowercase) in the Roughneck block")
T("True Grit", "真正的勇士", "T-EXACT", "ch2", "3.真正的勇士（True Grit）", True, None, "high")
T("Analysis", "解析", "T-EXACT", "ch2", "1.解析（anglysis）", True, None, "high",
  note="fan doc misspells the English as 'anglysis'; the corerules talent is Analysis")
T("Inquisitive", "保持好奇", "T-EXACT", "ch2", "2.保持好奇（inquisitive）", True, None, "high")
T("Xenomorphology", "异形生物学", "T-EXACT", "ch2", "3.异形生物学（Xenomorphology）", True, None, "high")
T("Seen It All", "见多识广", "T-EXACT", "ch2", "见多识广 (Seen ItAll)", True, None, "high",
  note="fan doc writes 'Seen ItAll' with the space in the wrong place; corerules talent is 'Seen It All'")
T("Hardened", "硬汉", "T-EXACT", "ch2", "但还可以通过硬汉 (Hardened)天赋增加", True, None, "high")

# ══════════════════════════════════════════════════════════════════════════════
# 11. GOVERNMENTS, CORPORATIONS, ORGANISATIONS
# ══════════════════════════════════════════════════════════════════════════════
T("Three World Empire", "三界帝国", "T-BILINGUAL", "ch1", "三界帝国 (THE THREE WORLD EMPIRE)", True, None, "high",
  note="the old-1e file (level 5) agrees: 三界帝国（Three World Empire，缩写3WE）")
T("United Americas", "联合美洲", "T-BILINGUAL", "ch1", "联合美洲 (THE UNITED AMERICAS)", True,
  "the old-1e file (level 5) says 美洲联邦 for the same body — REJECT 美洲联邦, level 1 wins", "high")
T("Union of Progressive Peoples (UPP)", "人类进步联盟", "T-BILINGUAL", "ch1", "人类进步联盟(THE UNION OF PROGRESSIVE PEOPLES)", True,
  "the old-1e file (level 5) says 进步人民联盟 — REJECT, level 1 wins", "high")
T("Independent Core System Colonies (ICSC)", "独立核心星系殖民地", "T-BILINGUAL", "ch1", "独立核心星系殖民地 (THE INDEPENDENT CORE SYSTEM COLONISE)", True, None, "high",
  note="the fan doc's own English gloss misspells COLONIES as 'COLONISE'")
T("Weyland-Yutani", "韦汤", "T-BILINGUAL", "ch1", "韦汤 (Weyland-Yutani) 公司", True,
  "the old-1e file (level 5) says 维兰德-汤谷 — REJECT, level 1 wins", "high",
  note="the fan doc uses the clipped 韦汤公司 11× in ch1 and spells the full name 韦兰德-汤谷公司 only once, at the merger")
T("Weyland-Yutani (full form)", "韦兰德-汤谷公司", "T-BILINGUAL", "ch1", "组建成韦兰德-汤谷公司", False,
  "韦汤 (clipped, ×11) vs 韦兰德-汤谷 (full, ×1) — both are the fan doc's own", "high")
T("Weyland Corp / Weyland Industries", "韦兰德公司", "T-BILINGUAL", "ch1", "韦兰德公司陷入财务困境", False,
  "韦兰德公司 and 韦兰德工业 both appear for Weyland Corp / Weyland Industries; the old-1e file (level 5) says 维兰德公司", "high")
T("Yutani Corporation", "汤谷公司", "T-BILINGUAL", "ch1", "日本汤谷公司合并后创建的", False, None, "high")
T("Lasalle Bionational", "生物国家", "T-BILINGUAL", "ch1", "生物国家 (BioNational) 公司", True,
  "the fan doc's own English gloss is 'BioNational'; the Evolved corerules name is 'Lasalle Bionational', so the fan rendering drops Lasalle", "low",
  note="REJECT as a headword: 生物国家 ('bio-nation') is a calque of a truncated name")
T("Seegson", "西格森", "T-BILINGUAL", "ch1", "公司和西格森 (Seegson) 公司", True, None, "high")
T("Colonial Marines / USCMC", "殖民海军陆战队", "T-BILINGUAL", "ch1", "殖民海军陆战队美国殖民海军陆战队代表了有史以来最精锐的作战力量", False, C_MARINES, "high")
T("Colonial Navy", "殖民海军", "T-BILINGUAL", "ch1", "外环被广泛殖民，殖民海军在此活动", False, None, "high",
  note="⚠ 殖民海军 (Colonial Navy) and 殖民海军陆战队 (Colonial Marines) differ by two characters and the shorter is a prefix of the longer — a substring-matching hazard for any TM apply pass")
T("United Americas Outer Rim Defense Fleet", "联合美洲外环防卫舰队", "T-BILINGUAL", "ch1", "联合美洲外环防卫舰队的成立", False,
  "ch1 later writes 联合美洲外环防卫舰队 in the careers section as 联合美洲外环防卫舰队 too; the old-1e file (level 5) says 联合美州联邦外环防系统 / 外环防御舰队", "high")
T("USCMC (untranslated acronym)", "USCMC", "T-FROZEN", "ch2", "你就报名加入了USCMC", False, None, "high")
T("Colonial Administration", "殖民管理局", "T-BILINGUAL", "ch1", "但是老妈把我们的通讯发送到了HQ公司而不是殖民管理局", False, None, "medium",
  note="EN log line reads 'Corporate HQ, not Colonial Control'; the fan renders Colonial Control as 殖民管理局, which is the Evolved book's Colonial Administration")
T("Interstellar Commerce Commission (ICC)", "星际商务委员会", "T-BILINGUAL", "ch2", "3.ICC（星际商务委员会）商业飞行员执照", True, None, "high")
T("Practitioners of the Holy Immolation", "神圣焚祭信徒组织", "T-BILINGUAL", "ch1", "\"神圣焚祭信徒组织\"", False, None, "high")
T("Church of Immaculate Incubation", "无瑕孕育教会", "T-BILINGUAL", "ch1", "\"无瑕孕育教会\"", False, None, "high")
T("the Earth Savers", "地球守护者", "T-BILINGUAL", "ch1", "名为地球守护者（the Earth Savers）", True,
  "the old-1e file (level 5) says 地球拯救者", "high")

# ══════════════════════════════════════════════════════════════════════════════
# 12. PLACES, SHIPS, STATIONS
# ══════════════════════════════════════════════════════════════════════════════
T("the Frontier", "边境", "T-PLAIN", "ch1", "1.2边境 (Frontier) 的生活", True, None, "high")
T("the Outer Veil", "外层帷幕", "T-BILINGUAL", "ch1", "边境始于外层帷幕 (theOuter Veil)的前端", True,
  "the old-1e file (level 5) says 外面纱 — REJECT, level 1 wins", "high",
  note="the fan doc's own English gloss is run together as 'theOuter Veil'")
T("the Core Systems", "核心星系", "T-BILINGUAL", "ch1", "外层帷幕位于核心星系(the Core Systems)", True, None, "high")
T("the Outer Rim", "外环", "T-BILINGUAL", "ch1", "与外环 (the Outer Rim)之间", True, None, "high")
T("Anchorpoint", "锚点", "T-BILINGUAL", "ch1", "像锚点 (Anchorpoint)这样的空间站", True, None, "high")
T("Anchorpoint Station", "锚点站", "T-BILINGUAL", "ch1", "第一个锚点站（Anchorpoint Station）", True, None, "high")
T("Thedus", "特杜斯", "T-BILINGUAL", "ch1", "在一次对特杜斯（Thedus）的维和任务中", True,
  "the old-1e file (level 5) says 西德斯 — REJECT, level 1 wins", "high")
T("Torin Prime", "托林主星", "T-BILINGUAL", "ch1", "2106托林主星（Torin Prime）", True,
  "the old-1e file (level 5) says 托林普莱姆 (transliterated 'Prime') — REJECT, level 1's 托林主星 is correct", "high")
T("the Solomons", "所罗门星系", "T-BILINGUAL", "ch1", "公司要求将其拖往所罗门星系", False, None, "medium",
  note="EN 'the Solomons' are seven colonised MOONS of Alpha Caeli V, not a star system; 星系 is factually loose")
T("LV-426", "LV-426", "T-FROZEN", "ch1", "距离LV-426上的哈德利的希望殖民地", False, None, "high")
T("Hadley's Hope", "哈德利的希望", "T-BILINGUAL", "ch1", "哈德利的希望殖民地 (the Hadley's Hope colony)", True, None, "high")
T("Fiorina 161", "菲奥莉娜161", "T-BILINGUAL", "ch1", "只涉及菲奥莉娜161的事后情况", False, C_FIORINA, "high",
  note="MAJORITY reading (2 of 3 hits). The one glossed instance uses the minority spelling 费奥莉娜")
T("Fiorina 161 (glossed, minority)", "费奥莉娜161", "T-BILINGUAL", "ch1", "费奥莉娜 161 (Fiorina 161)", True, C_FIORINA, "high",
  note="the source writes this one with a space before the number (费奥莉娜 161) while the majority spelling is closed up (菲奥莉娜161)")
T("USS Sulaco", "USS萨拉科号", "T-BILINGUAL", "ch1", "USS萨拉科号 (USS Sulaco)", True, None, "high")
T("USCSS Nostromo", "USCSS诺斯特罗莫号", "T-BILINGUAL", "ch1", "商业拖运飞船USCSS诺斯特罗莫（Nostromo）号", True, None, "high",
  note="in the source the English gloss sits INSIDE the Chinese name (USCSS诺斯特罗莫（Nostromo）号); the headword here is the reconstructed clean form")
T("USCSS Prometheus", "USCSS普罗米修斯号", "T-BILINGUAL", "ch1", "臭名昭著的USCSS普罗米修斯 (Prometheus) 号", True, None, "high",
  note="in the source the English gloss sits INSIDE the Chinese name (USCSS普罗米修斯 (Prometheus) 号); the headword here is the reconstructed clean form")
T("USCSS Covenant", "USCSS 契约号", "T-BILINGUAL", "ch1", "USCSS 契约（Covenant）号任务公布", True, None, "high",
  note="in the source the English gloss sits INSIDE the Chinese name (USCSS 契约（Covenant）号); the headword here is the reconstructed clean form")
T("USCSS Miranda", "USCSS 米兰达号", "T-BILINGUAL", "ch1", "USCSS 米兰达号船长", False, None, "high",
  note="the ship of the ch1 voyage-log framing device; the EN corerules worked examples use the same ship")
T("the Heliades", "希利亚德号", "T-BILINGUAL", "ch1", "希利亚德号 (the Heliades)", True, None, "medium",
  note="the Evolved rulebook says 'a Heliades class Space Exploration Vehicle' — a CLASS. The fan doc names it as a single ship (号). Check before adopting 号")
T("UAS Archangel", "大天使号", "T-BILINGUAL", "ch1", "UAS部队的运兵舰\"大天使（Archangel）\"", True,
  "the old-1e file (level 5) writes UAS大天使号", "high",
  note="in the source the English gloss sits INSIDE the quoted Chinese name 「大天使（Archangel）」; the headword here is the reconstructed clean form")
T("Sevastopol Station", "塞瓦斯托波尔站", "T-BILINGUAL", "ch1", "塞瓦斯托波尔站（Sevastopol Station)", True, None, "high",
  note="matches the Alien: Isolation official 简中 lineage (level 3)")
T("Mendel Station", "门德尔站", "T-BILINGUAL", "ch1", "位于外环的门德尔站（Mendel Station）", True, None, "high")
T("Seegson Station LV 44-40", "西格森站", "T-BILINGUAL", "ch1", "与西格森站（Seegson Station）LV 44-40", True, None, "high")
T("Wright-Aberra Waystation", "赖特-阿贝拉中转站", "T-BILINGUAL", "ch1", "以及赖特-阿贝拉中转站（Wright-Aberra Waystation）", True, None, "medium",
  note="no hit for 'Wright-Aberra' anywhere in the Evolved corerules dump — this name comes from the printed timeline only")
T("8 Eta Bootis A III", "牧夫座伊塔8 A3", "T-BILINGUAL", "ch1", "在边境世界牧夫座伊塔8 A3（8 Eta Boötis A Ⅲ）上", True, None, "low",
  note="word order is mangled — the '8' belongs in front (8 牧夫座η A III). Also mixes Arabic '3' in the Chinese with Roman 'Ⅲ' in the gloss")
T("The Tientsin Campaign", "天津战役", "T-BILINGUAL", "ch1", "爆发了天津战役（The Tientsin Campaign)", True, None, "high",
  note="correct: Tientsin is the Wade-Giles form of 天津")
T("HR-2429", "HR-2429星系", "T-FROZEN", "ch1", "朝着HR-2429星系", False, None, "high")
T("Origae-6", "Origae-6", "T-FROZEN", "ch1", "前往远方第87区的行星Origae-6", False, None, "high",
  note="left in Latin letters by the fan translator; 'Sector 87' becomes 第87区")
T("Luna", "露娜", "T-BILINGUAL", "ch1", "2031对地球的月球露娜进行地形改造作业", False, None, "medium",
  note="transliteration where 月球 is already used in the same clause; the Evolved corerules call it LUNA, humanity's first off-world colony")
T("Sevastopol / KG348", "行星KG348", "T-FROZEN", "ch1", "并坠入行星KG348 的大气层", False, None, "high")
T("GJ667CC", "系外行星GJ667CC", "T-FROZEN", "ch1", "成功在系外行星GJ667CC上创造了可呼吸的大气", False, None, "high")
T("HD85512 B", "HD85512 B", "T-FROZEN", "ch1", "2042HD85512 B，地球的第一个球外监狱", False, None, "high",
  note="'球外监狱' is a calque of 'off-world prison'; 地外/外星 would be the idiomatic choice")
T("Space Beast (banned book)", "《太空野兽》", "T-BILINGUAL", "ch1", "这本书名为《太空野兽》", False, None, "high")
T("Robert Morse", "罗伯特·莫尔斯", "T-BILINGUAL", "ch1", "是囚犯罗伯特·莫尔斯 (Robert Morse)", True, None, "high")
T("Peter Weyland", "彼得·韦兰德", "T-BILINGUAL", "ch1", "在彼得·韦兰德 (Peter Weyland) 2023年臭名昭著的TED演讲之后", True,
  "the old-1e file (level 5) says 彼得·维兰德爵士 — REJECT, level 1 wins", "high")
T("Meredith Vickers", "梅雷迪思·维克尔斯", "T-BILINGUAL", "ch1", "首席执行官梅雷迪思·维克尔斯 (Meredith Vickers)", True,
  "the old-1e file (level 5) says 梅雷迪斯·维克斯 — REJECT, level 1 wins", "high")
T("David (android series)", "大卫", "T-BILINGUAL", "ch1", "韦兰德生产了第一代大卫 (David) 系列机器人", True, C_ANDROID, "high")

# ══════════════════════════════════════════════════════════════════════════════
# 13. TECH / SETTING NOUNS
# ══════════════════════════════════════════════════════════════════════════════
T("FTL drive", "超光速引擎", "T-PLAIN", "ch1", "超光速引擎 (FTL drives)", True,
  "ch1 careers section says 超光速驱动器 ('新型更快速的超光速驱动器的出现') for the same device", "high")
T("FTL travel", "超光速飞行", "T-PLAIN", "ch1", "随着超光速飞行成为现实", False, None, "high")
T("hypersleep pod", "深度睡眠舱", "T-PLAIN", "ch1", "以及深度睡眠舱 (hypersleep pods)", True, None, "high")
T("hypersleep / stasis", "深度睡眠", "T-PLAIN", "ch1", "正在准备进入深度睡眠", False,
  "ch1 also uses 休眠 ('不必在休眠状态中花费太多时间', '补偿他们在休眠中失去的时间') and 冷冻 ('没有他我们没法进行冷冻') for the same state", "high",
  note="⚠ three renderings of hypersleep/stasis inside ch1 alone: 深度睡眠 / 休眠 / 冷冻")
T("atmospheric processor", "大气处理器", "T-PLAIN", "ch1", "引进了大气处理器 (atmospheric processors)", True, None, "high",
  note="the crude first-pass regex captured '引进了大气处理器'; the noun is 大气处理器")
T("terraforming", "地形改造", "T-PLAIN", "ch1", "2031对地球的月球露娜进行地形改造作业", False,
  "ch1 timeline also says 地球化 ('哈德利的希望的地球化殖民地在LV-426上建立') for the same process", "high")
T("EEV (Emergency Escape Vehicle)", "逃生舱", "T-PLAIN", "ch1", "我们在逃生舱 (EEV)舱门区建立了防御屏障", True, None, "high")
T("airlock", "气闸", "T-PLAIN", "ch1", "关闭你的风暴百叶窗并封住气闸", False, None, "high")
T("Xenomorph / the Alien", "异形", "T-EXACT", "ch1", "从新变种到异形生物", False, None, "high",
  note="matches the settled project decision Xenomorph = 异形")
T("Neomorph", "新变种", "T-PLAIN", "ch1", "从新变种到异形生物", False, None, "medium",
  note="corerules has a Neomorph creature and 'EV - Neomorph Attacks' table; 新变种 is a semantic rendering, not the usual 新形态")
T("alien (adjective) / extraterrestrial", "外星", "T-PLAIN", "ch1", "外星生命。这是《异形》角色扮演游戏", False, None, "high",
  note="matches the settled project decision alien(adj) = 外星; the doc keeps 外星生命 / 外星生物 distinct from 异形")
T("dropship", "着陆舰", "T-PLAIN", "ch2", "从星际战机到着陆舰，从货船到护卫舰", False, None, "low",
  note="EN 'From starfighters to dropships, freighters to frigates'. 着陆舰 is idiosyncratic; the known-bad CnSCG lineage erases dropship entirely, so this at least renders it")
T("starfighter", "星际战机", "T-PLAIN", "ch2", "从星际战机到着陆舰，从货船到护卫舰", False, None, "medium")
T("frigate", "护卫舰", "T-PLAIN", "ch2", "从星际战机到着陆舰，从货船到护卫舰", False, None, "high")
T("MedPod", "医疗舱", "T-PLAIN", "ch2", "如果有人能够坚持到达到医疗舱", False, None, "medium")
T("space trucker", "太空卡车司机", "T-PLAIN", "ch1", "1.4边境的职业太空卡车司机", False, None, "high")
T("colonist", "殖民者", "T-PLAIN", "ch1", "殖民者是人类的生命之血", False, None, "high")

# ══════════════════════════════════════════════════════════════════════════════
# 14. GEAR (career equipment tables; English fixed by corerules item names)
# ══════════════════════════════════════════════════════════════════════════════
G = "装备选择2项或投掷2次D6：1.M41A脉冲步枪"
T("M41A Pulse Rifle", "M41A脉冲步枪", "T-BILINGUAL", "ch2", "1.M41A脉冲步枪", False, None, "high")
T("M56A2 Smart Gun", "M56A2智能炮", "T-BILINGUAL", "ch2", "2.M56A2智能炮", False, None, "high",
  note="AVOIDS the known-bad CnSCG rendering smart gun→机关枪")
T("M314 Motion Tracker", "M314运动追踪器", "T-BILINGUAL", "ch2", "3.M314运动追踪器", False, None, "high",
  note="ch2 prose spaces it as 'M314 运动追踪器'; the gear tables close it up as 'M314运动追踪器'")
T("G2 Electroshock Grenade", "G2电击手榴弹", "T-BILINGUAL", "ch2", "4.2枚G2电击手榴弹", False, None, "high")
T("IRC Mk.35 Pressure Suit", "IRC MK.35压力服", "T-BILINGUAL", "ch2", "5.IRC MK.35压力服", False, C_PRESSUIT, "high")
T("IRC Mk.50 Compression Suit", "IRC MK.50压力服", "T-BILINGUAL", "ch2", "2.IRC MK.50 压力服", False, C_PRESSUIT, "medium",
  note="should be 加压服/抗压服 to keep it distinct from the Mk.35 Pressure Suit")
T("M3 Personnel Armor", "M3单兵防弹衣", "T-BILINGUAL", "ch2", "6.M3单兵防弹衣", False, None, "high")
T(".357 Magnum Revolver", ".357马格南左轮手枪", "T-BILINGUAL", "ch2", "1. .357马格南左轮手枪", False, None, "high")
T("Armat Model 37A2 12-Gauge Pump-Action", "军用型37A2 12号口径霰弹枪", "T-BILINGUAL", "ch2", "2. 军用型37A2 12号口径霰弹枪", False, None, "low",
  note="ERROR: 'Armat' is the manufacturer's name, not the adjective 'armed/military'. 军用型 is a mistranslation — should transliterate Armat (阿玛特)")
T("Watatsumi DV-303 Bolt Gun", "海神DV-303栓动枪", "T-BILINGUAL", "ch2", "2.海神DV-303栓动枪", False, None, "low",
  note="Watatsumi is the Japanese sea deity; the fan translator rendered it semantically as 海神 instead of transliterating a corporate name. corerules spells it both 'Watatsumi' and 'Watsumi'")
T("Rexim RXF-M5 EVA Pistol", "雷克希姆RXF-M5 EVA手枪", "T-BILINGUAL", "ch2", "2.雷克希姆RXF-M5 EVA手枪", False, None, "high")
T("M4A3 Service Pistol", "M4A3制式手枪", "T-BILINGUAL", "ch2", "1.M4A3制式手枪", False, None, "high")
T("VP-70MA6 Service Pistol", "VP-70MA6制式手枪", "T-BILINGUAL", "ch2", "5.VP-70MA6制式手枪", False, None, "high")
T("Stun baton", "电棍", "T-PLAIN", "ch2", "5. 电棍", False, None, "high")
T("Hi-beam flashlight", "强光手电筒", "T-PLAIN", "ch2", "3. 强光手电筒", False, None, "high")
T("Personal medkit", "个人医疗包", "T-PLAIN", "ch2", "4. 个人医疗包", False, None, "high")
T("Surgical kit", "外科手术包", "T-PLAIN", "ch2", "1.外科手术包", False, None, "high")
T("Maintenance jack", "维修千斤顶", "T-PLAIN", "ch2", "4.维修千斤顶", False, None, "high")
T("Cutting torch", "切割炬", "T-PLAIN", "ch2", "1.切割炬", False, None, "high")
T("Comm unit", "通信单元", "T-PLAIN", "ch2", "2.通信单元", False, None, "high")
T("Hand radio", "手持无线电", "T-PLAIN", "ch2", "3.手持无线电", False, None, "high")
T("Samani E-Series Watch", "萨马尼E系列手表", "T-BILINGUAL", "ch2", "3.萨马尼E系列手表", False, None, "high")
T("Seegson P-DAT", "西格森个人数据平板", "T-BILINGUAL", "ch2", "6.西格森个人数据平板（P-DAT）", True, None, "high")
T("Seegson System Diagnostic Device", "西格森系统诊断设备", "T-BILINGUAL", "ch2", "5.西格森系统诊断设备", False, None, "high")
T("PR-PUT Uplink Terminal", "PR-PUT上行终端", "T-BILINGUAL", "ch2", "2.PR-PUT上行终端", False, None, "high")
T("Personal Data Transmitter", "个人数据传输器", "T-PLAIN", "ch2", "4.个人数据传输器", False, None, "high")
T("Data transmitter card with corporate clearance", "具有公司许可级别的数据传输卡", "T-PLAIN", "ch2", "4.具有公司许可级别的数据传输卡", False, None, "high")
T("Digital video camera", "数字摄像机", "T-PLAIN", "ch2", "1.数字摄像机", False, None, "high")
T("Personal locator beacon", "个人定位装置", "T-PLAIN", "ch2", "5.个人定位装置", False, None, "high")
T("Leather briefcase", "皮革公文包", "T-PLAIN", "ch2", "1.皮革公文包", False, None, "high")
T("Naproleve", "镇痛剂", "T-PLAIN", "ch2", "6.D6剂镇痛剂", False, None, "low",
  note="Naproleve is a brand name in the corerules gear list; 镇痛剂 renders it as the generic category 'analgesic', losing the name")
T("Neversleep pills", "不眠药", "T-PLAIN", "ch2", "4.D6剂不眠药", False, None, "medium")
T("Hydr8tion", "8号药剂", "T-PLAIN", "ch2", "3.D6剂8号药剂", False, None, "low",
  note="ERROR: 'Hydr8tion' is leetspeak for 'hydration'. The fan translator parsed the embedded 8 as a product number and produced '8号药剂' ('agent no. 8')")
T("Aspen beer", "阿斯彭啤酒", "T-BILINGUAL", "ch2", "要是能有一罐阿斯彭啤酒（Aspen beer）", True, None, "high")
T("Souta Dry", "索塔干啤", "T-BILINGUAL", "ch2", "或者甚至是索塔干啤（Souta Dry）", True, None, "high")
T("Laser pointer", "激光笔", "T-PLAIN", "ch2", "1.激光笔", False, None, "high")
T("Radio-controlled car", "遥控车", "T-PLAIN", "ch2", "2.遥控车", False, None, "high")
T("Yo-yo", "溜溜球", "T-PLAIN", "ch2", "3.溜溜球", False, None, "high")
T("Electronic handheld game", "掌上游戏机", "T-PLAIN", "ch2", "4.掌上游戏机", False, None, "high")
T("Coloring pens", "彩笔", "T-PLAIN", "ch2", "6.彩笔", False, None, "high")
T("ICC Commercial Flight Officer license", "ICC商业飞行员执照", "T-BILINGUAL", "ch2", "3.ICC（星际商务委员会）商业飞行员执照", True, None, "high",
  note="in the source the ICC expansion （星际商务委员会） is spliced into the middle of the item name; the headword here is the clean form")
T("Albert Einstein Award", "阿尔伯特爱因斯坦奖", "T-BILINGUAL", "ch2", "1.阿尔伯特爱因斯坦奖", False, None, "high",
  note="the fan doc omits the interpunct: 阿尔伯特爱因斯坦, not 阿尔伯特·爱因斯坦")

# ══════════════════════════════════════════════════════════════════════════════
# 15. PUBLICATIONS
# ══════════════════════════════════════════════════════════════════════════════
T("Rapture Protocol", "《狂喜协议》", "T-BILINGUAL", "ch1", "《狂喜协议》(Rapture Protocol)", True, None, "high")
T("Hope's Last Day Starter Set", "《希望的最后一天入门套装》", "T-BILINGUAL", "ch1", "《希望的最后一天入门套装》(Hope's Last Day Starter Set)", True,
  "ch2 expands the same title to 《哈德利的希望的最后一天入门套装》, importing 'Hadley' that the English title does not have", "high")
T("the Draconis Saga trilogy", "《德拉科尼斯传奇》三部曲", "T-BILINGUAL", "ch1", "《德拉科尼斯传奇》三部曲(the Draconis Saga trilogy)", True, None, "high")
T("Colonial Marines Operations Guide", "《殖民海军行动指南》", "T-BILINGUAL", "ch1", "《殖民海军行动指南》(the Colonial Marines Operations Guide)", True, None, "medium",
  note="EN is Colonial MARINES; 殖民海军 is the Colonial NAVY. Should be 《殖民海军陆战队行动指南》")
T("Building Better Worlds", "《建造更美好世界》", "T-BILINGUAL", "ch1", "《建造更美好世界》", False, None, "high")
T("ALIEN roleplaying game", "《异形》角色扮演游戏", "T-BILINGUAL", "ch1", "这是《异形》角色扮演游戏", False, None, "high")
T("Evolved Edition", "进化版", "T-PLAIN", "ch1", "进化版这是《异形》角色扮演游戏的进化版", False, None, "high")

# ══════════════════════════════════════════════════════════════════════════════
# 16. OLD-1e FILE — LEVEL 5 EVIDENCE ONLY, NEVER LEVEL 1
# ══════════════════════════════════════════════════════════════════════════════
OLD = "⚠ LEVEL 5 ONLY — this file's own filename says 这是旧版的 非新版Evolved (1st-edition material). Do NOT count it as level-1 evidence."
T("Weyland-Yutani (1e file)", "维兰德-汤谷", "T-BILINGUAL", "old1e", "维兰德-汤谷，阿尔法科技", False,
  "conflicts with level-1 韦汤 / 韦兰德-汤谷", "low", note=OLD)
T("United Americas (1e file)", "美洲联邦", "T-BILINGUAL", "old1e", "组成一个名为美洲联邦的超级集团", False,
  "conflicts with level-1 联合美洲", "low", note=OLD)
T("Union of Progressive Peoples (1e file)", "进步人民联盟", "T-BILINGUAL", "old1e", "进步人民联盟(Union of Progressive Peoples ，简称UPP)", True,
  "conflicts with level-1 人类进步联盟", "low", note=OLD)
T("Three World Empire (1e file)", "三界帝国", "T-BILINGUAL", "old1e", "三界帝国（Three World Empire，缩写3WE）", True,
  None, "medium", note=OLD + " Agrees with level 1, so it corroborates rather than conflicts.")
T("Outer Veil (1e file)", "外面纱", "T-BILINGUAL", "old1e", "保护外面纱边缘的新系外行星", False,
  "conflicts with level-1 外层帷幕", "low", note=OLD)
T("Thedus (1e file)", "西德斯", "T-BILINGUAL", "old1e", "在西德斯引发了一场大规模的劳资纠纷", False,
  "conflicts with level-1 特杜斯", "low", note=OLD)
T("Torin Prime (1e file)", "托林普莱姆", "T-BILINGUAL", "old1e", "巴拉圭人的殖民地托林普莱姆", False,
  "conflicts with level-1 托林主星", "low", note=OLD)
T("USCMC (1e file)", "美国殖民海军陆战队集团军", "T-BILINGUAL", "old1e", "美国殖民海军陆战队集团军(USCMC)", True,
  "the same file also writes 美国殖民地陆战队集团军 and 美国殖民地陆战队大连 (a garbled 大队); conflicts with level-1 殖民海军陆战队", "low", note=OLD)
T("Meredith Vickers (1e file)", "梅雷迪斯·维克斯", "T-BILINGUAL", "old1e", "首席执行官梅雷迪斯·维克斯", False,
  "conflicts with level-1 梅雷迪思·维克尔斯", "low", note=OLD)
T("Peter Weyland (1e file)", "彼得·维兰德爵士", "T-BILINGUAL", "old1e", "彼得·维兰德爵士", False,
  "conflicts with level-1 彼得·韦兰德", "low", note=OLD)
T("Earth Savers (1e file)", "地球拯救者", "T-BILINGUAL", "old1e", "这个狂热的“地球拯救者”", False,
  "conflicts with level-1 地球守护者", "low", note=OLD)
T("UAS Archangel (1e file)", "UAS大天使号", "T-BILINGUAL", "old1e", "运兵船UAS大天使号在执行维和任务时被摧毁", False,
  None, "medium", note=OLD + " Adds the 号 that level 1 omits.")
T("AlphaTech (1e file)", "阿尔法科技", "T-BILINGUAL", "old1e", "阿尔法科技（AlphaTech）", True, None, "low", note=OLD)
T("Newcomer Grant Corp (1e file)", "新科莫·格兰特公司", "T-BILINGUAL", "old1e", "新科莫·格兰特公司（Newcomer Grant Corp）", True, None, "low", note=OLD)
T("UNIC / UN Interstellar Corps (1e file)", "联合国行星际部队", "T-BILINGUAL", "old1e", "联合国行星际部队(UNIC)", True, None, "low", note=OLD)
T("CANC (1e file)", "中国/亚洲国家合作组织", "T-BILINGUAL", "old1e", "中国/亚洲国家合作组织", False, None, "low", note=OLD)
T("J'Har rebels (1e file)", "贾哈尔叛军", "T-BILINGUAL", "old1e", "那里的贾哈尔叛军发布了一份恐怖宣言", False, None, "low",
  note=OLD + " The Evolved corerules do carry 'J'Har rebels' (Torin Prime, 2106 uprising), so the referent is real even though the file is 1e.")

# ══════════════════════════════════════════════════════════════════════════════
INCONSISTENCIES = [
  {"id": "fiorina", "term": "Fiorina 161",
   "readings": ["菲奥莉娜161 (ch1 ×2)", "费奥莉娜161 (ch1 ×1)"],
   "which_the_document_supports": "菲奥莉娜161",
   "why": "2 of 3 occurrences, including both narrative mentions (成员的牺牲 sidebar and the 2179 timeline entry). The single 费 spelling is the one that carries the English gloss, which makes it look authoritative, but it is the minority.",
   "quotes": [quote("ch1", "只涉及菲奥莉娜161的事后情况"),
              quote("ch1", "但她的一艘逃生舱坠落在高级安保行星菲奥莉娜161"),
              quote("ch1", "费奥莉娜 161 (Fiorina 161)")]},

  {"id": "android", "term": "Android / Synthetic",
   "readings": ["机器人 (ch1 ×1 glossed 'androids', ch2 ×18)",
                "合成人 (ch1 ×1 'synthetic people', ch2 ×1 'Synthetic characters')",
                "仿生人 (ch1 ×1, David 7 series)"],
   "which_the_document_supports": "机器人 for Android; 合成人 for Synthetic; DROP 仿生人",
   "why": "机器人 carries 19 of the 21 hits and is the heading of the ch2 rules block 扮演机器人 (= PLAYING AN ANDROID). 合成人 appears only where the English says 'synthetic', including the one sentence that uses both labels for the same referent (机器人：合成人角色…). 仿生人 is a single stray, two timeline lines away from 机器人 for the SAME David series — a drafting slip, not a distinction.",
   "quotes": [quote("ch1", "韦兰德生产了第一代大卫 (David) 系列机器人"),
              quote("ch1", "韦兰德公司的大卫7代系列仿生人在劳动中变得很常见"),
              quote("ch2", "机器人：合成人角色在分配完14点属性后"),
              quote("ch1", "合成人扮演上帝")]},

  {"id": "mother", "term": "Game Mother / MOTHER (ship AI) / GM",
   "readings": ["游戏管理员 (ch1, glossed 'the Game MOTHER')",
                "老妈 (ch1, glossed 'Game MOTHER' in the 成为老妈 sidebar)",
                "游戏主持人 (ch1 ×3, glossed 'GM')",
                "老妈 (ch1 ×4, the USCSS Miranda's MU/TH/UR AI)",
                "MOTHER (ch1 §1.1 heading, untranslated)"],
   "which_the_document_supports": "SPLIT, deliberately: 游戏管理员/游戏主持人 for the human GM, 老妈 for the ship AI — but the 成为老妈 (Game MOTHER) sidebar breaks the split by using 老妈 for the GM.",
   "why": "The English pun is the whole point: the GM is called the Game Mother because the ship AI is called MOTHER. The fan doc splits the pun apart for the body text and then re-fuses it for one sidebar heading. Any Foundry build has to pick one policy and hold it, because the ship-AI 老妈 lines are in-fiction log entries the player reads and the GM 老妈 line is a rules cross-reference.",
   "quotes": [quote("ch1", "游戏管理员 (the Game MOTHER)"),
              quote("ch1", "成为老妈 (Game MOTHER)"),
              quote("ch1", "老妈妥协了"),
              quote("ch1", "1.1 MOTHER，发生了什么")]},

  {"id": "hypersleep", "term": "hypersleep / stasis",
   "readings": ["深度睡眠 (ch1)", "休眠 (ch1 ×2)", "冷冻 (ch1)", "深度睡眠舱 (ch1, glossed 'hypersleep pods')"],
   "which_the_document_supports": "深度睡眠 / 深度睡眠舱 — it is the only reading that carries an English gloss.",
   "why": "休眠 and 冷冻 appear in adjacent paragraphs of the same chapter for the same act. The known-bad CnSCG lineage renders stasis as 静态平衡, so any of the fan's three beats that, but the doc does not settle on one.",
   "quotes": [quote("ch1", "以及深度睡眠舱 (hypersleep pods)"),
              quote("ch1", "正在准备进入深度睡眠"),
              quote("ch1", "不必在休眠状态中花费太多时间"),
              quote("ch1", "没有他我们没法进行冷冻")]},

  {"id": "marines", "term": "Colonial Marines",
   "readings": ["殖民海军陆战队 (ch1 ×6, ch2 ×2)", "殖民地海军陆战队 (ch1 ×2)"],
   "which_the_document_supports": "殖民海军陆战队",
   "why": "8 of 10 hits, and it is the form used in the campaign-framework list and the ch2 career heading. 殖民地海军陆战队 appears twice inside the same careers paragraph as 殖民海军陆战队, so it is drafting noise.",
   "quotes": [quote("ch1", "通常需要殖民海军陆战队介入以恢复秩序"),
              quote("ch1", "殖民地海军陆战队能够在几乎任何环境下独立作战")]},

  {"id": "freeleague", "term": "Free League (publisher)",
   "readings": ["自由联盟 (ch1)", "Free League (ch2, untranslated)"],
   "which_the_document_supports": "no majority — one hit each, in the same sentence role",
   "why": "Both sentences tell the reader to download a blank character sheet from the publisher's site. ch1 localises the name, ch2 leaves it in Latin letters. Nothing in the document breaks the tie.",
   "quotes": [quote("ch1", "你也可以从自由联盟网站下载空白角色卡"),
              quote("ch2", "从Free League网站下载并打印")]},

  {"id": "shift", "term": "Shift (time unit)",
   "readings": ["班 (ch1 table, glossed 'Shift')", "班次 (ch2 ×2, body prose)"],
   "which_the_document_supports": "班 — it is the glossed table entry and the parallel member of 轮/节/班",
   "why": "⚠ RUNTIME CONSEQUENCE. Shift is the crit-parse lockstep literal, so whatever string the build adopts has to be the SAME string everywhere the parser sees it. The fan doc's own body text already drifts to 班次 twice ('接受老师一个班次的指导', '每天的每个班次'), which would break a literal match. Note also that system lang cn.json leaves ALIENRPG.Shift as the English word 'Shift' and renders ALIENRPG.OneShift as 一轮班 — three candidate strings across two sources.",
   "quotes": [quote("ch1", "班 (Shift)"),
              quote("ch2", "至少接受老师一个班次的指导"),
              quote("ch2", "每天的每个班次，你可以照料的病人数等于你的医疗技能等级")]},

  {"id": "round", "term": "Round (time unit)",
   "readings": ["轮 (ch1 table, glossed 'Round')", "战斗轮 (ch1 body)", "回合 (ch2, once: '你必须每回合都攻击你的敌人')"],
   "which_the_document_supports": "轮",
   "why": "轮 is the glossed table entry and the head of the 轮/节/班 series; 战斗轮 is just 轮 with its domain attached. But the Overkill talent slips into 回合, which is the mainstream Chinese TRPG word for Round and is also what system lang cn.json uses (ALIENRPG.OneRound = 一回合). The level-1 and level-4 sources disagree here.",
   "quotes": [quote("ch1", "轮 (Round)"),
              quote("ch1", "战斗轮、节和班的准确持续时间会根据情况变化"),
              quote("ch2", "你必须每回合都攻击你的敌人")]},

  {"id": "wy_dollar", "term": "W-Y dollars",
   "readings": ["韦汤币 (ch2, glossed 'W-Y dollar')", "W-Y币 (ch2 ×9, every career cash line)", "刀W-Y币 (ch2, worked example: '300刀W-Y币')"],
   "which_the_document_supports": "W-Y币 by count, 韦汤币 by authority",
   "why": "The glossed form 韦汤币 appears once, in the rules explanation. Every career block then uses the mixed-script W-Y币. The worked example adds a third form by prefixing the colloquial 刀 for 'dollar' to the already-complete W-Y币.",
   "quotes": [quote("ch2", "你还会获得韦汤币（W-Y dollar）"),
              quote("ch2", "现金：D6x100 W-Y币"),
              quote("ch2", "一套压力服和300刀W-Y币")]},

  {"id": "pressure_suit", "term": "Pressure Suit vs Compression Suit",
   "readings": ["IRC MK.35压力服 (ch2)", "IRC MK.50压力服 (ch2)"],
   "which_the_document_supports": "neither — the document collapses two distinct corerules armor items into one Chinese string",
   "why": "corerules.json carries 'IRC Mk.35 Pressure Suit' and 'IRC Mk.50 Compression Suit' as two separate armor documents. The fan doc renders both 压力服, so the only thing distinguishing them in Chinese is the model number. A Foundry build needs two strings.",
   "quotes": [quote("ch2", "5.IRC MK.35压力服"),
              quote("ch2", "2.IRC MK.50 压力服")]},

  {"id": "cinematic", "term": "cinematic play",
   "readings": ["电影模式 (ch1 ×several, ch2)", "剧情模式 (ch2, once)"],
   "which_the_document_supports": "电影模式",
   "why": "电影模式 is the defined term introduced in ch1 §1.6 and reused throughout. The Company Resources talent text slips to 剧情模式 for the same mode, in a sentence that contrasts it with 战役模式.",
   "quotes": [quote("ch1", "可以以两种不同的模式进行：电影模式和战役模式"),
              quote("ch2", "在剧情模式中每幕一次，或在战役模式中每次会话一次")]},

  {"id": "hopes_last_day", "term": "Hope's Last Day Starter Set",
   "readings": ["《希望的最后一天入门套装》 (ch1, glossed)", "《哈德利的希望的最后一天入门套装》 (ch2)"],
   "which_the_document_supports": "《希望的最后一天入门套装》 — it is the glossed form and matches the English title exactly",
   "why": "ch2 imports 哈德利 (Hadley) into the title. The English product is 'Hope's Last Day Starter Set'; the in-fiction colony is Hadley's Hope, so the expansion is an explanatory addition the title does not license.",
   "quotes": [quote("ch1", "《希望的最后一天入门套装》(Hope's Last Day Starter Set)"),
              quote("ch2", "如《狂喜协议》和《哈德利的希望的最后一天入门套装》")]},

  {"id": "session", "term": "game session",
   "readings": ["游戏会话 (ch2 ×several)", "游戏环节 (ch2, once, in the XP checklist)"],
   "which_the_document_supports": "游戏会话",
   "why": "游戏会话 is used in the agenda rules, the XP rules and the buddy/rival rules. 游戏环节 appears once, in the very first XP question, for the same thing.",
   "quotes": [quote("ch2", "在每场游戏会话结束时"),
              quote("ch2", "你参加了游戏环节吗？")]},

  {"id": "pregen", "term": "pre-generated characters",
   "readings": ["预设角色 (ch1, ch2)", "预制 (ch2)"],
   "which_the_document_supports": "预设角色",
   "why": "预设 is used for both the characters and their agendas ('在电影模式冒险中，选择是预设的'). 预制 appears once, as a modifier on 电影模式冒险.",
   "quotes": [quote("ch1", "所有电影模式冒险均包含预设角色"),
              quote("ch2", "在预制电影模式冒险中")]},

  {"id": "date_2183", "term": "the present date of the setting",
   "readings": ["2183年 (ch1 真相在此)", "2180s (ch1 timeline heading 2180-至今 and '2180s是一个艰难的时代')"],
   "which_the_document_supports": "both, but the English does not license 2183",
   "why": "The Evolved English reads 'It's the 2180s – only a few years have passed since…'. The fan translation pins it to 现在是2183年…不过三年多的时间. 2183 is defensible from the corerules art page titled 'City of 2183', but it is an addition, not a translation.",
   "quotes": [quote("ch1", "真相在此现在是2183年"),
              quote("ch1", "2180s是一个艰难的时代")]},

  {"id": "ftl_drive", "term": "FTL drive",
   "readings": ["超光速引擎 (ch1 timeline, glossed 'FTL drives')", "超光速驱动器 (ch1 careers)"],
   "which_the_document_supports": "超光速引擎",
   "why": "It carries the English gloss. 超光速驱动器 appears once, in the Space Truckers career blurb, for the same device.",
   "quotes": [quote("ch1", "超光速引擎 (FTL drives)"),
              quote("ch1", "新型更快速的超光速驱动器的出现显著缩短了星际旅行的时间")]},

  {"id": "terraform", "term": "terraforming",
   "readings": ["地形改造 (ch1, 2031 entry)", "地球化 (ch1, 2157 entry)"],
   "which_the_document_supports": "no majority — one hit each, both in the same timeline",
   "why": "2031 for Luna is 地形改造作业; 2157 Hadley's Hope is 地球化殖民地. Same process, two words, 126 in-fiction years and about eight printed lines apart.",
   "quotes": [quote("ch1", "2031对地球的月球露娜进行地形改造作业"),
              quote("ch1", "2157哈德利的希望的地球化殖民地在LV-426上建立")]},

  {"id": "medic", "term": "Medic",
   "readings": ["医生 (ch2 career heading)", "医务官 (ch1 log entries, ch2 worked example)"],
   "which_the_document_supports": "SPLIT and defensible: 医生 for the career, 医务官 for the shipboard billet",
   "why": "Both render EN 'medic', but in different registers — 医务官 appears only where Hayes is introduced as a crew member. The English uses the one word for both, so a Foundry build that keys on the career name will not find the log lines.",
   "quotes": [quote("ch1", "唯一的新人是医务官海斯"),
              quote("ch2", "2.9 职业：医生"),
              quote("ch2", "以及医务官海斯")]},

  {"id": "career_agent", "term": "Company Agent / Company Reps",
   "readings": ["公司代理人 (ch2 career heading)", "公司代表 (ch1 careers-on-the-Frontier section)"],
   "which_the_document_supports": "SPLIT and correct: the English headings genuinely differ (Company Agent vs Company Reps)",
   "why": "Listed here so a reviewer does not 'fix' it. ch1 §1.4 renders the setting section 'Company Reps' as 公司代表; ch2 §2.7 renders the career 'COMPANY AGENT' as 公司代理人. Both are right.",
   "quotes": [quote("ch1", "公司代表在边境，企业的权力甚至超过政府"),
              quote("ch2", "2.7 职业：公司代理人")]},

  {"id": "marshal", "term": "Colonial Marshal",
   "readings": ["殖民地执法官 (ch2 career heading)", "执法官 (ch1 ×3, bare)", "殖民地执法官 (ch1, log entry)"],
   "which_the_document_supports": "殖民地执法官",
   "why": "The bare 执法官 in ch1 is elliptical prose ('选举执法官来监督日常的执法工作'), not a competing term. Recorded so a TM apply pass does not treat the bare form as a separate headword.",
   "quotes": [quote("ch1", "选举执法官来监督日常的执法工作"),
              quote("ch1", "我给殖民地执法官发了消息"),
              quote("ch2", "2.6 职业：殖民地执法官")]},
]

NOTES = [
  "SOURCE PROVENANCE. ch1 = 异形RPG：进化版-第一章：宇宙是地狱.txt (11,380 chars, last edited 2026-04-28 by 鹿鹿); ch2 = 异形RPG：进化版-第二章：你的角色.txt (14,233 chars, 2026-04-28). Both are LEVEL 1 under the owner's settled priority: they are chapters 1 and 2 of the very book being translated.",
  "THE THIRD FILE IS NOT LEVEL 1. 《异形RPG》版异形宇宙设定翻译·时间线·第一部分-这是旧版的 非新版Evolved.txt (3,896 chars, posted 2022-08-15 by 维兰德时空策略部官方) is 1st-edition material by its own filename and by its content. Every term mined from it is tagged chapter='old1e' and carries a note saying LEVEL 5 ONLY. It disagrees with level 1 on ten headwords (维兰德-汤谷 / 美洲联邦 / 进步人民联盟 / 外面纱 / 西德斯 / 托林普莱姆 / 美国殖民地陆战队 / 梅雷迪斯·维克斯 / 彼得·维兰德 / 地球拯救者). In every one of those ten, level 1 wins.",
  "ENGLISH ANCHORING. Every non-glossed entry was anchored against two artifacts, not guessed: C:/Users/Taka/AppData/Local/FoundryVTT/Data/systems/alienrpg/lang/en.json (590 flattened keys) and C:/Users/Taka/Desktop/fvtt/Alien-RPG Translation Project/6-工作区/raw-dumps/corerules.json (adventure name 'Alien Evolved Core Rules'; 283 items incl. 57 talents, 123 rolltables, 61 actors, 7 journals). The journal pages 'Alien Evolved Player Guide / 1. SPACE IS HELL' and '/ 2. YOUR CHARACTER' are the English of exactly these two fan-translated chapters, so the career blocks, talent names, gear lists and the Round/Stretch/Shift table were matched line for line rather than inferred.",
  "INLINE GLOSS HARVEST. A cleaned regex pass over the three files found 56 distinct 中文 (English) pairs in ch1, 30 in ch2 and 4 in old1e. The earlier first-pass count of 35 was low AND dirty: its captures ran leading context into the headword (e.g. '引进了大气处理器' for atmospheric processors, '韦兰德生产了第一代大卫' for David, '距离LV-426上的哈德利的希望殖民地' for Hadley's Hope colony). Every headword in this file is the trimmed noun; the untrimmed sentence is preserved in `quote` so a reviewer can see where the boundary was drawn.",
  "THE FAN DOC'S ENGLISH IS ITSELF TYPO-BEARING. Do not key any matcher on the parenthesised English. Observed: 'theOuter Veil' (missing space), 'Seen ItAll' (space in the wrong place), 'Resllient' and 'resillient' (both for Resilient), 'anglysis' (for Analysis), 'Steady hand' vs 'Steady Hands', 'THE INDEPENDENT CORE SYSTEM COLONISE' (for COLONIES), 'BioNational' (for Lasalle Bionational).",
  "FOUR OUTRIGHT MISTRANSLATIONS worth quarantining. (1) Hydr8tion → 8号药剂: the leetspeak 8 in 'hydration' was read as a product number. (2) Armat Model 37A2 → 军用型37A2: the manufacturer Armat was read as the adjective 'armed/military'. (3) Watatsumi DV-303 → 海神DV-303: a Japanese corporate name rendered semantically ('sea god') instead of transliterated. (4) Lasalle Bionational → 生物国家 (glossed 'BioNational'): the company's first name is dropped and the rest calqued.",
  "TWO OWNER-PINNED REJECTIONS ARE CONFIRMED PRESENT IN THE SOURCE. The fan translation uses 智力 for Wits and 同理心 for Empathy. Both entries are recorded with tier T-EXACT and a note saying REJECT, because level-4 cn.json (机智 / 共情) is what players already see in Foundry. Independently of the pin, both fan renderings are self-colliding: 智力 also appears INSIDE the Wits gloss ('感官感知、智力和理智') and 同理心 also appears INSIDE the Empathy gloss ('个人魅力、同理心以及操控他人的能力'), so neither can serve as a unique key.",
  "THE TWELVE SKILLS ARE COMPLETE AND CONSISTENT IN THE FAN DOC, and they disagree with level-4 cn.json on 7 of 12. Agreeing: 耐力 Stamina, 医疗 Medical Aid, 机动 Mobility, 指挥 Command, 驾驶 Piloting. Disagreeing: 近战/肉搏 Close Combat, 计算机科学/科技 Comtech, 侦察/观察 Observation, 远程战斗/射击 Ranged Combat, 重型机械/机械 Heavy Machinery, 操控/操纵 Manipulation, 生存/求生 Survival. The fan doc never once uses the cn.json form for any of the seven, so this is a clean two-source disagreement, not internal noise.",
  "⚠ SHIFT HAS RUNTIME CONSEQUENCES. Three candidate strings exist across the two sources for one lockstep literal: 班 (fan, glossed table entry), 班次 (fan body prose, twice), and — in system cn.json — the untranslated English 'Shift' for ALIENRPG.Shift plus 一轮班 for ALIENRPG.OneShift. Whatever the build picks has to be applied to every one of those sites at once.",
  "SUBSTRING HAZARD FOR ANY TM APPLY PASS. 殖民海军 (Colonial Navy) is a strict prefix of 殖民海军陆战队 (Colonial Marines), and both occur 11 and 8 times respectively in ch1 alone. Longest-match ordering is mandatory. Same shape: 驾驶 (Piloting, skill) is a prefix of 驾驶员 (Pilot, career); 机器人 (Android) is a substring of 秘密机器人 (secret android); 余量 (supply rating) is a prefix of 余量检定 (supply roll) and 余量计数器 (supply dial).",
  "PUSH. The fan doc's rendering of Push is 追骰, used 11 times in ch2 and never varied. system cn.json has no Push key at all (ALIENRPG.SynthStress is the nearest and is garbled: 'Human Panic, Push, ect.' → '模仿人类的恐慌和按钮', which mis-reads Push as a button). So level 1 is the only real evidence for this term and 追骰 should be adopted.",
  "RESOLVE. 精神强度. system cn.json leaves ALIENRPG.Resolve and ALIENRPG.ResolveMod untranslated (null), so again level 1 is the only evidence.",
  "TERMS THE FAN DOC DOES NOT COVER. Neither chapter contains the panic-response table, the stress-response table, critical injuries, spaceship attributes, the bestiary, or chapters 3-14. Anything about ranges beyond 近距离 (Short), about initiative beyond the word 先攻, or about creature names beyond 异形 / 新变种 / 抱脸虫-class terms is simply absent — do not synthesise it from these two chapters.",
  "CHECKED AND FOUND CLEAN. The doc keeps 异形 (Xenomorph/the Alien) and 外星 (alien, adjective) apart exactly as the project has already settled: '外星生命', '外星生物学', '奇怪外星工艺品或生物' vs '异形的生命周期', '异形生物学'. It also avoids three known-bad CnSCG renderings: smart gun is 智能炮 not 机关枪, the dropship is rendered (as 着陆舰) rather than erased, and stasis is 深度睡眠/休眠/冷冻 rather than 静态平衡.",
]

payload = {
    "_meta": {
        "source_priority_level": 1,
        "source_dir": BASE,
        "source_files": {k: {"file": v, "chars": len(TXT[k])} for k, v in FILES.items()},
        "anchors": [
            "C:/Users/Taka/AppData/Local/FoundryVTT/Data/systems/alienrpg/lang/en.json",
            "C:/Users/Taka/AppData/Local/FoundryVTT/Data/systems/alienrpg/lang/cn.json",
            "C:/Users/Taka/Desktop/fvtt/Alien-RPG Translation Project/6-工作区/raw-dumps/corerules.json",
        ],
        "tier_legend": {
            "T-FROZEN": "keep English bytes exactly",
            "T-EXACT": "pure Chinese, must be byte-equal to a lang key value",
            "T-BILINGUAL": "中文 English, ONE ASCII space, no parentheses",
            "T-PLAIN": "bare Chinese in prose and {labels}",
        },
        "term_count": len(TERMS),
        "inline_glossed_count": sum(1 for t in TERMS if t["inline_glossed"]),
        "conflicted_count": sum(1 for t in TERMS if t["internal_conflict"]),
    },
    "terms": TERMS,
    "internal_inconsistencies": INCONSISTENCIES,
    "notes": NOTES,
}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
print("wrote", OUT)
print("terms", len(TERMS), "glossed", payload["_meta"]["inline_glossed_count"],
      "conflicted", payload["_meta"]["conflicted_count"],
      "inconsistencies", len(INCONSISTENCIES), "notes", len(NOTES))
print("bytes", os.path.getsize(OUT))
