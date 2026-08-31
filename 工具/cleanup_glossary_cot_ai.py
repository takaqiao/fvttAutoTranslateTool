# -*- coding: utf-8 -*-
"""
清理 glossary_cot_ai.json：
1. 弯引号 → ASCII 引号统一（删除弯引号版本，保留源文佐证的翻译）
2. 单复数冲突对齐（以源文为准统一翻译，删除冗余复数条目）
"""

import json, re, sys, io, copy
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

INPUT  = "glossary_cot_ai.json"
OUTPUT = "glossary_cot_ai.json"  # 原地覆盖

d = json.loads(open(INPUT, encoding="utf-8-sig").read())
original_count = len(d)

changes = []

# ─────────────────────────────────────────────
# 1. 弯引号 key 清理
#    策略：删除弯引号 key，ASCII key 保留源文佐证的翻译
# ─────────────────────────────────────────────

# 源文佐证的正确翻译（ASCII key → 正确中文）
curly_overrides = {
    # 源文匹配：阿拉兹尼的热情（Arazni's Fervor）
    "Arazni's Fervor":               "阿拉兹尼的热情",
    # 源文匹配：承流学会统一法论（Cascade Bearer's Spellcasting）
    "Cascade Bearer's Spellcasting": "承流学会统一法论",
    # 源文匹配：死墓骑士诅咒 Graveknight's Curse
    "Graveknight's Curse":           "死墓骑士诅咒",
    # 源文匹配：警醒城围栏（Vigil's Palisades）
    "Vigil's Palisades":             "警醒城围栏",
    # 阿拉兹妮→阿拉兹尼（全文统一用阿拉兹尼）
    "Arazni's Divine Intercessions": "阿拉兹尼的神力干预",
    "Arazni's Freedom":              "阿拉兹尼的自由",
    # "Rest"→"憩" 比 "眠" 更贴原意
    "Iomedae's Rest":                "艾奥梅黛之憩",
    # "lament"→"哀歌" 标准译法
    "thieves' lament":               "盗贼哀歌",
    # PF2E 标准用语
    "explorer's clothing":           "探索者服装",
    # "厨师匠心" 出现在合并术语表中（occ=1）
    "A Chef's Touch":                "厨师匠心",
}

# 收集所有弯引号 key
curly_keys = [k for k in d if "\u2019" in k or "\u2018" in k]
for ck in curly_keys:
    ascii_k = ck.replace("\u2019", "'").replace("\u2018", "'")
    # 如有 ASCII 对应版，先确保 ASCII 版本用正确翻译
    if ascii_k in curly_overrides:
        old_val = d.get(ascii_k, "")
        new_val = curly_overrides[ascii_k]
        if old_val != new_val:
            d[ascii_k] = new_val
            changes.append(f"curly-fix: [{ascii_k}] {old_val} → {new_val}")
    elif ascii_k not in d:
        # ASCII 版本不存在，把弯引号的值移过去
        d[ascii_k] = d[ck]
        changes.append(f"curly-move: [{ck}] → [{ascii_k}] = {d[ck]}")
    # 删除弯引号 key
    del d[ck]
    changes.append(f"curly-del: [{ck}]")

# ─────────────────────────────────────────────
# 2. 单复数冲突对齐
#    策略：以源文为准统一翻译，保留单数形式
# ─────────────────────────────────────────────

# 基于源文 + 上下文判定的标准翻译
plural_fixes = {
    # 源文 x18 "觅宝精"。复数 "米米塞特" 是音译错误
    ("Mirmicette", "mirmicettes"):       ("觅宝精", "觅宝精"),
    # 源文 "爪抓" x5 用于 claw melee，"爪击" 用于 Claw Feint 技能名
    # claw 作为通用近战攻击 = 爪击，保留两个但统一
    ("claw", "claws"):                   ("爪击", "爪击"),
    # PF2E "zombie" = "丧尸"（参考 glossary.json），统一
    ("plague zombie", "PLAGUE ZOMBIES"):  ("疫病丧尸", "疫病丧尸"),
    ("zombie shambler", "ZOMBIE SHAMBLERS"): ("蹒跚丧尸", "蹒跚丧尸"),
    ("Zombie Desecrator", "Zombie Desecrators"): ("亵渎丧尸", "亵渎丧尸"),
    # 绯红 = regex 已确立的标准，复数同步
    ("Crimson Keepers", "Crimson Keeper"): ("绯红守护者", "绯红守护者"),
    # "报丧女妖" 是 PF2E banshee 的标准译名
    ("banshee", "banshees"):             ("报丧女妖", "报丧女妖"),
    # "卡诺卜罐" 更标准的音译
    ("canopic jar", "canopic jars"):     ("卡诺卜罐", "卡诺卜罐"),
    # 统一为 "幽灵箭"
    ("ghost arrow", "ghost arrows"):     ("幽灵箭", "幽灵箭"),
    # 统一为 "潜猎食尸鬼"
    ("ghoul stalker", "GHOUL STALKERS"): ("潜猎食尸鬼", "潜猎食尸鬼"),
    # PF2E "次级" 是标准前缀
    ("lesser healing potion", "lesser healing potions"): ("次级治疗药水", "次级治疗药水"),
    # 统一 "高等"
    ("major acid flask", "major acid flasks"): ("高等强酸瓶", "高等强酸瓶"),
    # "规避点" 简洁
    ("Evasion Point", "Evasion Points"):  ("规避点", "规避点数"),
    # Princastle 和 Princastles 含义不同，保留两个
    ("Princastle", "Princastles"):        (None, None),  # skip
}

for (sing, plur), (sing_zh, plur_zh) in plural_fixes.items():
    if sing_zh is None:
        continue  # skip pairs where both should be kept as-is

    # Fix singular translation
    if sing in d and d[sing] != sing_zh:
        old = d[sing]
        d[sing] = sing_zh
        changes.append(f"plural-fix: [{sing}] {old} → {sing_zh}")

    # Fix or delete plural
    if plur in d:
        if sing_zh == plur_zh:
            # Same translation → delete plural, keep singular
            del d[plur]
            changes.append(f"plural-del: [{plur}] (redundant with [{sing}])")
        else:
            # Different (e.g., Evasion Point vs Points)
            if d[plur] != plur_zh:
                old = d[plur]
                d[plur] = plur_zh
                changes.append(f"plural-fix: [{plur}] {old} → {plur_zh}")

# ─────────────────────────────────────────────
# 3. 阿拉兹妮 → 阿拉兹尼 全局统一
# ─────────────────────────────────────────────
for k in list(d.keys()):
    if "阿拉兹妮" in d[k]:
        old = d[k]
        d[k] = d[k].replace("阿拉兹妮", "阿拉兹尼")
        changes.append(f"typo-fix: [{k}] {old} → {d[k]}")

# ─────────────────────────────────────────────
# 4. 僵尸 → 丧尸 统一（PF2E 标准）
#    仅修改明确是 zombie 相关的条目
# ─────────────────────────────────────────────
zombie_keys = [k for k in d if "zombie" in k.lower()]
for k in zombie_keys:
    if "僵尸" in d[k]:
        old = d[k]
        d[k] = d[k].replace("僵尸", "丧尸")
        changes.append(f"zombie-fix: [{k}] {old} → {d[k]}")

# ─────────────────────────────────────────────
# 5. SunlightPowerlessness / WispForm 等无空格 key 清理
# ─────────────────────────────────────────────
junk_keys = []
for k in d:
    # CamelCase without spaces that duplicates a spaced version
    if re.match(r'^[A-Z][a-z]+[A-Z]', k) and ' ' not in k:
        # Check if spaced version exists
        spaced = re.sub(r'([a-z])([A-Z])', r'\1 \2', k)
        if spaced in d:
            junk_keys.append(k)

for k in junk_keys:
    spaced = re.sub(r'([a-z])([A-Z])', r'\1 \2', k)
    del d[k]
    changes.append(f"camel-del: [{k}] (duplicate of [{spaced}])")

# Also clean TarBaphon → already have Tar-Baphon
if "TarBaphon" in d and "Tar-Baphon" in d:
    del d["TarBaphon"]
    changes.append("camel-del: [TarBaphon] (duplicate of [Tar-Baphon])")

# ClaimCorpse → Claim Corpse
if "ClaimCorpse" in d and "Claim Corpse" in d:
    del d["ClaimCorpse"]
    changes.append("camel-del: [ClaimCorpse] (duplicate of [Claim Corpse])")

# ─────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────
# Sort by key
d_sorted = dict(sorted(d.items(), key=lambda x: x[0].lower()))

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(d_sorted, f, ensure_ascii=False, indent=2)

print(f"原始条目: {original_count}")
print(f"清理后: {len(d_sorted)}")
print(f"变更: {len(changes)}")
print()
for c in changes:
    print(f"  {c}")
