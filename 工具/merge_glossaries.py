#!/usr/bin/env python3
"""Merge glossary_crusible.json + glossary_adaptive_crucible.json into a clean glossary.

Strategy:
  1. crusible is the base (user confirmed "还可以")
  2. Add 230 new terms from adaptive
  3. For entries where adaptive added bad variants, revert to crusible original
  4. For genuinely polysemous terms, keep multi-value
  5. Output merged glossary
"""

import json
from pathlib import Path

BASE = Path("C:/Users/Taka/Desktop/fvtt")
CRUSIBLE_PATH = BASE / "glossary_crusible.json"
ADAPTIVE_PATH = BASE / "glossary_adaptive_crucible.json"
OUTPUT_PATH = BASE / "glossary_crucible_merged.json"

# --- Rules for the 43 entries where crusible had single value and adaptive added variants ---
# True  = keep adaptive multi-value (genuine polysemy)
# False = revert to crusible single value (noise/error)
# Custom list = manually curated values
VARIANT_RULES: dict[str, bool | list[str]] = {
    # Game mechanics - genuine polysemy in different contexts
    "Action":       ["动作", "行动"],          # game action vs narrative action
    "Common":       ["通用语", "普通"],          # language name vs adjective
    "Fire":         ["火焰", "火"],             # 火元素 is wrong for most contexts; 火焰/火 are the real variants
    "Earth":        ["大地", "土"],             # proper noun vs element
    "Light":        True,                       # already multi in crusible: 光/光亮术/轻型
    "Passive":      ["被动效果", "被动"],        # full form vs short form, both legit
    "Health":       ["生命值", "生命"],          # both used
    "Society":      ["社会", "社交"],            # different meanings

    # Weapons/items - short forms are OK in ability names
    "Axe":          "斧",                       # 斧 is more standard than 斧头
    "Hammer":       "锤",                       # 锤 is more standard
    "Bite":         ["啃咬", "噬咬"],           # both OK; drop bare 咬
    "Shield":       ["盾牌", "护盾术", "盾", "护盾"],  # all legitimate in different contexts

    # Conditions/states - pick the correct one
    "Blinded":      "目盲",                     # 目眩 means dazzled, wrong
    "Exhaustion":   "力竭",                     # 疲惫 is a different concept
    "restrained":   "受缚",                     # 受拘束 is verbose synonym
    "Fortitude":    "坚韧",                     # 强韧 is just a synonym, keep consistent
    "Steadfast":    "坚定不移",                  # 坚定 is a shortening, lose precision
    "Willpower":    ["意志", "意志力"],          # both legit: stat name vs description

    # Game terms - keep original for consistency
    "Attunement":   "调谐",                     # 同调 is a different system's term
    "Beacon":       "烽火",                     # 信标 is generic, 烽火 is the in-world term
    "Gesture":      "手势",                     # 姿势 is wrong (posture vs gesture)
    "Spellcraft":   "施法",                     # 法术 means "spell" not "spellcraft"
    "boon":         "恩惠骰",                   # 恩惠 loses the dice meaning
    "Travel pace":  "旅行步调",                  # keep original
    "Armor":        "护甲",                     # 铠甲 is a synonym but 护甲 is standard
    "Adept":        "行家",                     # 专家/老练者 are noise
    "Frost":        "霜冻",                     # 寒霜 is a synonym
    "Undead":       "不死生物",                  # 亡灵 is a different flavor

    # Proper nouns - must be consistent, revert to original
    "House Cevher Mausoleum":  "杰夫赫尔家族陵墓",
    "House Cevher Signet Ring": "杰夫赫尔家族印戒",
    "Hulg'run Lineage":        "赫尔格伦血统",
    "Inkaro pearl":             "因卡罗珍珠",
    "Ken Crystals":             "肯水晶",
    "Thornling":                "荆棘裔",
    "agrimage":                 "农法师",
    "agrimagic":                "农法",
    "arcden":                   "奥克登语",
    "jurtak":                   "尤塔克",
    "kaleidoscope crystal":     "万花筒水晶",
    "kithil":                   "奇希尔",
    "quickwater":               "激流水",
    "wind raiders":             "风掠者",
    "wingsuit":                 "飞行服",
    "drake":                    "龙兽",         # crusible already has Drake:["龙兽","幼龙"]
    "Reliquary":                "圣物室",
}

# --- Post-merge fixups for inconsistencies found during review ---
POST_FIXUPS: dict[str, str | list[str]] = {
    # Aberin: 阿伯林 not 阿贝林
    "Aberin's Folly":           "阿伯林的愚举",
    # Arcturian: standardize to 阿克图里安 (not 阿克图瑞安)
    "Arcturian Automatons":     "阿克图里安自动机工坊",
    "Arcturian Liquor":         "阿克图里安烈酒",
    # Inkaro: standardize to 因卡罗 (not 印卡罗)
    "Inkaro Pearl":             "因卡罗珍珠",
    # jurtak lowercase: align with Jurtak = 尤尔塔克
    "jurtak":                   "尤尔塔克",
    # kithil lowercase: align with Kithil
    "Kithil":                   "奇希尔",     # align Kithil to match kithil=奇希尔
    # thornling lowercase: align with Thornling = 荆棘裔
    "thornling":                "荆棘裔",
    "Afflicted Thornling":      "受折磨的荆棘裔",
    # Ordain: clean up — place name 奥尔丹 + verb 授命
    "Ordain":                   ["奥尔丹", "授命"],
    "Ordain Gazetteer":         "奥尔丹地志",
    "Overhead in Ordain":       "《奥尔丹传闻》",
    # Fire: include 火元素 as it was the crusible original
    "Fire":                     ["火元素", "火焰", "火"],
    # Case-variant pairs: align to the crusible original (lowercase key)
    "Agrimage":                 "农法师",
    "Agrimagic":                "农法",
    "Arcden":                   "奥克登语",
    "attunement":               "调谐",
    "Boon":                     "恩惠骰",
    "Quickwater":               "激流水",
    "reliquary":                "圣物室",
    "Restrained":               "受缚",
    "Travel Pace":              "旅行步调",
    "Wind Raiders":             "风掠者",
    "Wingsuit":                 "飞行服",
}

# --- Rules for 15 new multi-value terms (only in adaptive) ---
NEW_MULTI_RULES: dict[str, str | list[str]] = {
    "Blast":          ["爆破", "爆炸"],          # both legit
    "Body":           "身体",                   # 躯体 is synonym noise
    "Chamber":        ["密室", "厅室"],          # 之室 is weird; use 密室/厅室
    "Critical Hit":   "暴击",                   # 重击 is a different concept in TTRPG
    "Presence":       ["存在", "气场"],          # genuine polysemy
    "Shadow":         ["暗影", "阴影"],          # both legit
    "Ward":           ["防护", "护卫"],          # both legit
    "Wisdom":         "感知",                   # 智慧 is D&D's term; Crucible uses 感知
    "blade":          "刀刃",                   # 刃 is too short
    "level":          ["级", "等级"],            # both used
    "saving throw":   ["豁免", "豁免检定"],      # full/short form
    "spellbreaker":   "破法者",                  # 破咒者 is variant, keep consistent
    "spirit":         ["灵体", "灵魂"],          # both legit
    "staff":          ["木杖", "法杖"],          # different item subtypes
    "sword":          "剑",                     # 长剑 is too specific
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def normalize_value(val):
    """Ensure consistent output format: string for single, list for multi."""
    if isinstance(val, list):
        if len(val) == 1:
            return val[0]
        return val
    return val


def merge() -> dict:
    crusible = load_json(CRUSIBLE_PATH)
    adaptive = load_json(ADAPTIVE_PATH)

    merged = dict(crusible)  # start from crusible base

    # 1. Add new terms from adaptive (not in crusible)
    new_count = 0
    for key, val in adaptive.items():
        if key not in merged:
            # Check if it has a custom rule for new multi-value
            if key in NEW_MULTI_RULES:
                merged[key] = normalize_value(NEW_MULTI_RULES[key])
            elif isinstance(val, list):
                # New multi-value without explicit rule: take first value
                merged[key] = val[0]
            else:
                merged[key] = val
            new_count += 1

    # 2. Apply variant rules for existing entries
    reverted = 0
    kept_multi = 0
    for key, rule in VARIANT_RULES.items():
        if key not in merged:
            continue
        if rule is True:
            # Keep adaptive multi-value as-is
            merged[key] = adaptive[key]
            kept_multi += 1
        elif isinstance(rule, list):
            merged[key] = rule if len(rule) > 1 else rule[0]
            kept_multi += 1
        elif isinstance(rule, str):
            merged[key] = rule
            reverted += 1
        else:  # False
            # Revert to crusible original
            reverted += 1

    print(f"Base (crusible): {len(crusible)} entries")
    print(f"New terms added: {new_count}")
    print(f"Variant rules applied: reverted={reverted}, kept_multi={kept_multi}")
    print(f"Total merged: {len(merged)} entries")

    # Count final multi-value
    multi_count = sum(1 for v in merged.values() if isinstance(v, list))
    print(f"Multi-value entries in output: {multi_count}")

    return merged


def main():
    merged = merge()

    # Apply post-merge fixups
    fixup_count = 0
    for key, val in POST_FIXUPS.items():
        if key in merged:
            merged[key] = normalize_value(val) if isinstance(val, list) else val
            fixup_count += 1
    print(f"Post-merge fixups applied: {fixup_count}")

    # Sort by key for readability
    sorted_merged = dict(sorted(merged.items(), key=lambda x: x[0].lower()))

    text = json.dumps(sorted_merged, ensure_ascii=False, indent=2)
    OUTPUT_PATH.write_text(text + "\n", encoding="utf-8")
    print(f"\nOutput: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
