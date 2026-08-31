# -*- coding: utf-8 -*-
import json, re, sys
sys.stdout.reconfigure(encoding="utf-8")
idx = json.load(open("tp_index.json", encoding="utf-8"))
BA, BN, BE = idx["by_action"], idx["by_name"], idx["by_effect"]

# 本模块真正打补丁的 action id（从 tempfix.mjs 抓）
src = open("C:/Users/Taka/Desktop/fvtt/ember-crucible-tempfix/scripts/tempfix.mjs", encoding="utf-8").read()
ids = set()
for m in re.finditer(r'ACTION_PATCHES\.(\w+)\s*=', src): ids.add(m.group(1))
for m in re.finditer(r'actionIds:\s*\[([^\]]*)\]', src):
    ids |= set(re.findall(r'"(\w+)"', m.group(1)))
for m in re.finditer(r'ACTION_PATCHES\["?(\w+)"?\]\s*=', src): ids.add(m.group(1))
# 提示语/文档里点名的其它 id
for extra in ["throwWeapon","wildStrike","swallow","regurgitate","tumble","dawnBeacon",
              "repugnantPustules","abyssalRemains","noxiousSpray","selfDestruct","devourThoughts",
              "formidableStamina","implacableHunter","sentinelKick","sentinelShielding",
              "tyraphicTransformation","heartSparkOfEmber","bewilderingGaze","antigravityStone",
              "darkflameCirclet","abyssMarkUnmaking","suddenBite","offhandStrike",
              "mayisRestorativeRedirection","crystalizeWounds","extremeMetabolism","livingStone",
              "regulatedRhythm","thornbark","inscrutableVisage","menacingVisage","beguilingVisage",
              "mindFlay","eldritchEmanation","ferociousHowl","pestilentLash","shieldBash","steamVent",
              "alchemicalGrenade","frostFlask","electroAmpoule","net","waterAversion"]:
    ids.add(extra)

print(f"{'action id':<32} {'汉化项目里的中文名'}")
print("-" * 80)
missing = []
for i in sorted(ids):
    v = BA.get(i)
    if v: print(f"{i:<32} {' | '.join(v)}")
    else: missing.append(i)
print("\n索引里查不到的：", ", ".join(missing) if missing else "（无）")
