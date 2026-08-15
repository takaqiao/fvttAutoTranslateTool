# -*- coding: utf-8 -*-
"""Round-22: build the ARRANGEMENTS table body from the *probed* 212 unique
arrangement labels, so that nothing can silently fall through.

Rules enforced here (not by eyeball):
  * every one of the 212 labels must be classified as exactly one of
    DUAL (lives in SOUNDSCAPE_GROUPS) / KEEP_EN (deliberately untranslated) /
    translated-here. Anything left over => exit 2 and print it.
  * no key written here may also be a SOUNDSCAPE_GROUPS key (that would let
    ARRANGEMENTS shadow a group name inside MOOD_PANEL's spread).

Anti-空转: prints the probe count it read, the size of each bucket, and the
final key count; refuses to emit anything if the probe file is missing/short.
"""
import json, io, os, re, sys

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project"
MJS = os.path.join(BASE, r"1-Ember汉化插件\scripts\ember-hardcoded-cn.mjs")

labels = json.load(io.open(os.path.join(HERE, "soundscapes_r22.json"), encoding="utf-8"))["arrLabels"]
print(f"probe labels read: {len(labels)}")
if len(labels) != 212:
    print("UNEXPECTED LABEL COUNT")
    sys.exit(2)

# --- read the real SOUNDSCAPE_GROUPS keys out of the live file (no hand copy) ---
src = io.open(MJS, encoding="utf-8").read()
m = re.search(r"const SOUNDSCAPE_GROUPS = \{(.*?)\n\};", src, re.S)
if not m:
    print("SOUNDSCAPE_GROUPS not found")
    sys.exit(2)
GROUPS = re.findall(r'^\s*"([^"]+)":', m.group(1), re.M)
print(f"SOUNDSCAPE_GROUPS keys read from file: {len(GROUPS)}")
if len(GROUPS) != 42:
    print("UNEXPECTED GROUP COUNT")
    sys.exit(2)

DUAL = [l for l in labels if l in GROUPS]
print(f"dual-tier (defined in SOUNDSCAPE_GROUPS, must NOT be repeated here): {len(DUAL)} {DUAL}")

KEEP_EN = {
    # 唯一一条查不到任何依据的：`Seven Sails` 在两个仓库 43115 条英文叶里 en-hits=0，
    # `Sails` 单词也 0 命中，模块自带语料里也找不到这个名字的所指（酒馆？船？曲名？）。
    "Seven Sails": "两仓库英文闸 en-hits=0（`Sails` 亦 0），指代物不明，按「宁可留英」留英",
}

T = {}

# ---------- 结构词统一译法（先定表，再逐条套） ----------
TIME = {"Day": "白天", "Night": "夜晚"}
MOOD = {"Calm": "平静", "Tension": "紧张", "Quiet": "静谧", "Chaos": "混乱",
        "Sad": "哀伤", "Intense": "激烈", "Relaxed": "舒缓", "test": "测试"}
FORM = {"Main": "主段", "Interlude": "间奏", "Interval": "间歇", "Rises": "渐强",
        "Verse": "主歌", "Chorus": "副歌", "Bridge": "桥段", "Melody": "旋律",
        "Rhythm": "节律", "Heroic": "英勇", "Atonal": "无调性", "Spooky": "阴森",
        "Weird": "诡谲", "Dramatic": "戏剧性"}
SECTION = {"1": "第一段", "2": "第二段", "3": "第三段"}

# ---------- 地名 / 专名（全部来自英文闸或 glossary_ec，见表内注） ----------
PLACE = {
    "Amerasp Grove": "阿梅拉斯普林地", "Arcturel": "阿克图瑞尔",
    "Bleak Archive": "黯淡秘库", "Blood Woods": "血色森林",
    "Bloodletter Cave": "放血者洞穴", "Bluffs": "峭壁",
    "Broken Tower": "破碎之塔", "Burial Grounds": "墓地",
    "Camp Vista": "营地远景", "Cauldron": "坩埚湖",
    "Cindaric Temple": "辛达里克神殿", "Clockwork Dungeon": "发条地下城",
    "Corpin Sanctuary": "科尔平庇护所", "Dripstones": "滴石笋",
    "Dungeon": "地下城", "Ember Cosmos": "余烬寰宇",
    "Ember's Bounty": "余烬的恩赐", "Fogbound Caverns": "雾缚洞窟",
    "Forest of Stone": "石之森林", "Golden Flats": "金色平原",
    "Golden Flats Water": "金色平原水域", "Graven's Rest": "格雷文之憩",
    "Helkas": "赫尔卡斯", "Inkaro Pools": "因卡罗水潭",
    "Jungle": "丛林", "Kaleidoscope Caverns": "万花筒洞窟",
    "Kaleidoscope Grave": "万花筒之墓", "Lady Stonecraft": "石艺女士",
    "Lower Arcturel": "下层阿克图瑞尔", "Marlstone Gala": "马尔石晚会",
    "Mountains": "群山", "Mutagist Laboratory": "突变学派实验室",
    "Mycelian Expanse": "菌丝旷野", "Nain": "奈因",
    "Noxious Cave": "剧毒洞穴", "Ocean": "海洋",
    "Ooze Farm": "软泥农场", "Ordain Docks": "奥尔丹船坞",
    "Ordain Flats": "奥尔丹平原", "Ordain Interior": "奥尔丹室内",
    "Ordain Spires": "奥尔丹尖塔区", "Ordain Temple": "奥尔丹神殿",
    "Pathways": "通路", "Primordial Bastion": "原初堡垒",
    "Raiders' Hideout": "劫掠者藏身处", "Redrak Fields": "雷德拉克原野",
    "Rock Spires": "岩石尖塔", "Rustvar Valley": "鲁斯特瓦尔山谷",
    "Rustvar Valleys": "鲁斯特瓦尔山谷", "Sarin Strand": "萨林海滨",
    "Scrapyard": "废料场", "Seawall": "海堤", "Seydiri": "塞迪里",
    "Shent Water Temple": "申特水之神殿", "Shipwreck": "沉船",
    "Shrine of Nite": "奈特圣祠", "Signara": "西格纳拉",
    "Skybrush": "天刷镇", "Spellbreaker Tower": "破法者之塔",
    "Splinter Canyons": "碎裂峡谷", "Stadium Underworks": "竞技场地下工事",
    "Steed's Point": "斯蒂德角", "Sunken Rejarh": "沉没的雷贾尔",
    "Teeth": "卡迪索斯之牙", "The Ballad of Dereth Erekos": "德雷斯·埃雷科斯之歌",
    "The Teeth": "卡迪索斯之牙", "Tidal Pools": "潮汐池",
    "Upper Arcturel": "上层阿克图瑞尔", "Verdant Paths": "翠绿径",
    "Waterworks": "水务工程", "Wedgelands": "楔地",
    "Yakoshta": "雅科什塔", "Yakoshta Mine": "雅科什塔矿井",
    "Ancient Giants": "远古巨人", "Ancient Ruins Magic Depths": "远古遗迹 · 魔法深处",
    "Aedir Garrison Exploration": "艾迪尔驻军营探索",
    "Helkas Festival": "赫尔卡斯庆典", "Ordain Folk": "奥尔丹民谣",
    "Arcane Theme": "奥术主题", "Ocean Ship": "海洋 · 船上",
    "Helkas Attack": "赫尔卡斯袭击",
    "Helkas Attack (Drakes)": "赫尔卡斯袭击（龙兽）",
    "Helkas Attack (Raiders)": "赫尔卡斯袭击（劫掠者）",
    # 第二十一轮就已在表里的 4 条（原样保留，只是改由本生成器一并排版）
    "Shent Ruins": "申特遗迹", "The Pit Trap": "陷坑",
}
# `X Fight` / `X Combat` 与组名表同调：战斗类前缀
FIGHT = {
    "Abyssal Fight": "深渊战斗", "Beast Fight": "野兽战斗",
    "Bandit Fight": "强盗战斗", "Illusory Fight": "幻象战斗",
    "Mutagenic Fight": "诱变战斗", "Ooze Fight": "软泥怪战斗",
    "Raider Fight": "劫掠者战斗", "Undead Fight": "不死生物战斗",
    "Celestial Combat": "天界生物战斗", "Construct Combat": "构装体战斗",
    "Monstrosity Combat": "畸怪战斗", "Earth Elemental Combat": "土元素战斗",
    "Fire Elemental Combat": "火元素战斗", "Frost Elemental Combat": "冰霜元素战斗",
}

unresolved = []
for lab in labels:
    if lab in DUAL or lab in KEEP_EN:
        continue
    if lab in PLACE:
        T[lab] = PLACE[lab]
        continue
    # `<Fight/Combat prefix> Section N`
    m2 = re.match(r"^(.*) Section ([123])$", lab)
    if m2 and m2.group(1) in FIGHT:
        T[lab] = f"{FIGHT[m2.group(1)]} · {SECTION[m2.group(2)]}"
        continue
    # `<Fight prefix> <form word>`  and  `<Fight prefix> - <form word>`
    m2 = re.match(r"^(.*?)(?: -)? (\w+)$", lab)
    if m2 and m2.group(1) in FIGHT and (m2.group(2) in FORM or m2.group(2) in MOOD):
        tail = m2.group(2)
        T[lab] = f"{FIGHT[m2.group(1)]} · {FORM.get(tail) or MOOD[tail]}"
        continue
    # `<place> <Day|Night>` / `<place> <mood>`
    m2 = re.match(r"^(.*?)(?: -)? ([A-Za-z]+)$", lab)
    if m2 and m2.group(1) in PLACE:
        tail = m2.group(2)
        if tail in TIME:
            T[lab] = f"{PLACE[m2.group(1)]} · {TIME[tail]}"
            continue
        if tail in MOOD:
            T[lab] = f"{PLACE[m2.group(1)]} · {MOOD[tail]}"
            continue
    unresolved.append(lab)

print(f"translated here: {len(T)}   kept-EN: {len(KEEP_EN)}   dual: {len(DUAL)}")
if unresolved:
    print(f"UNRESOLVED ({len(unresolved)}):")
    for u in unresolved:
        print("   ", u)
    sys.exit(2)

assert len(T) + len(KEEP_EN) + len(DUAL) == 212, (len(T), len(KEEP_EN), len(DUAL))
bad = [k for k in T if k in GROUPS]
if bad:
    print("SHADOWS A GROUP NAME:", bad)
    sys.exit(2)

# collision report: same CN for different EN keys (expected only for the two
# upstream spelling variants Rustvar Valley/Valleys and Teeth/The Teeth)
rev = {}
for k, v in T.items():
    rev.setdefault(v, []).append(k)
dups = {v: ks for v, ks in rev.items() if len(ks) > 1}
print(f"\nCN values shared by >1 EN key: {len(dups)}")
for v, ks in sorted(dups.items()):
    print(f"   {v}  <-  {ks}")

w = max(len(k) for k in T) + 3
lines = []
for k in sorted(T):
    lines.append(f'  {(chr(34)+k+chr(34)+":").ljust(w)} "{T[k]}",')
io.open(os.path.join(HERE, "arrangements_body.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
print(f"\nwrote arrangements_body.txt ({len(lines)} lines)")
