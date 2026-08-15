# -*- coding: utf-8 -*-
"""精确抽 EXACT 表，核对若干候选串是否被覆盖。只读。"""
import re

HC = r"C:\Users\Taka\Desktop\fvtt\Ember-Crucible Translation Project\1-Ember汉化插件\scripts\ember-hardcoded-cn.mjs"
hc = open(HC, encoding="utf-8").read()
blk = re.search(r"const EXACT = \{(.*?)\n\};", hc, re.S).group(1)
pair = re.compile(r'"([^"]+)"\s*:\s*"([^"]*)"')
EXACT = dict(pair.findall(blk))
print("EXACT entries:", len(EXACT))

TESTS = [
    "Cosmological Attunements", "Make Active", "Active",
    "Ability Scores", "Points", "Base Ability Scores", "Points Remaining",
    "Spend 9 points across 6 ability scores, allocating up to 3 points per ability.",
    "Attunement", "Token", "Culture", "Path", "Ancestry",
    "Abyss", "Exit", "Quest", "Entry Date",
    "Select a quest from the left menu.",
    "Ancient Languages", "Obscure Languages",
    "-6 Banes", "-4 Banes", "+4 Boons",
    "Aster Progression", "Soulbound Progression",
    "Age of Creation", "Age of Beasts", "Age of the Tower", "After Shattering",
    "Years Ago", "Years From Now", "Current Year",
    "Talent", "Spell tooltips are still TO-DO.",
]
for t in TESTS:
    print("  %-80s in_EXACT=%s" % (repr(t), t in EXACT))

# PREFIXED 前缀
print("PREFIXED 前缀:", re.findall(r'\{\s*en:\s*"([^"]+)"', hc))
