# -*- coding: utf-8 -*-
"""查 glossary 是否定义了这些分歧 NPC 名"""
import json, re, os

GLOSSARY_PATHS = [
    r"C:\Users\Taka\Desktop\fvtt\glossary.json",
    r"C:\Users\Taka\Desktop\fvtt\pf2wiki-scraper\out\glossary_wiki.json",
]

terms_to_check = [
    "Umbasi", "Lantondo", "Artus Rodrivan", "Halspin",
    "Ahmoza", "Krankkiss", "Numoriz", "Paunnima", "Rajna", "Jun ", "Mingyu",
    "Yoh Souran", "Hwanggot", "Brartork",
]

for gpath in GLOSSARY_PATHS:
    if not os.path.exists(gpath):
        print(f"!! {gpath} not exists")
        continue
    print(f"\n========== {os.path.basename(gpath)} ==========")
    try:
        with open(gpath, encoding="utf-8") as f:
            g = json.load(f)
    except Exception as e:
        print(f"!! load failed: {e}")
        continue

    # glossary 可能是 dict 或 list of {en, zh}
    if isinstance(g, dict):
        for term in terms_to_check:
            hits = []
            for k, v in g.items():
                if not isinstance(v, (str, list, dict)): continue
                # 看 key 是否包含 term, 或者 value 包含 term
                if term.lower() in k.lower():
                    hits.append(f"key={k!r} → val={v!r}")
                elif isinstance(v, str) and term.lower() in v.lower():
                    hits.append(f"key={k!r} → val={v!r}")
            for h in hits[:3]:
                print(f"  {term:25} → {h}")
    elif isinstance(g, list):
        for term in terms_to_check:
            hits = []
            for entry in g:
                if not isinstance(entry, dict): continue
                en_field = entry.get("en") or entry.get("english") or entry.get("Origin") or ""
                zh_field = entry.get("zh") or entry.get("chinese") or entry.get("Translation") or ""
                if term.lower() in str(en_field).lower():
                    hits.append(f"en={en_field!r} → zh={zh_field!r}")
            for h in hits[:3]:
                print(f"  {term:25} → {h}")
