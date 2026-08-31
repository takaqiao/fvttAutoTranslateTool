# -*- coding: utf-8 -*-
"""诊断 2: 试 handouts / back-matter / bestiary"""
import json, re, os

FILES = [
    ("book1_handouts", r"C:\Users\Taka\Desktop\fvtt\FotRP\需要翻译\NEW\第一本\fvtt-JournalEntry-handouts-FhOWhqywZr72iuml.json"),
    ("book1_back", r"C:\Users\Taka\Desktop\fvtt\FotRP\需要翻译\NEW\第一本\fvtt-JournalEntry-back-matter-ti7ZfnFRd1IabwaA.json"),
    ("book2_handouts", r"C:\Users\Taka\Desktop\fvtt\FotRP\需要翻译\NEW\第二本\fvtt-JournalEntry-handouts-rXylkhhxGCEispwF.json"),
    ("book2_back", r"C:\Users\Taka\Desktop\fvtt\FotRP\需要翻译\NEW\第二本\fvtt-JournalEntry-back-matter-XLDMbpumIxhSEWd4.json"),
    ("book3_handouts", r"C:\Users\Taka\Desktop\fvtt\FotRP\需要翻译\NEW\第三本\fvtt-JournalEntry-handouts-diQf6WXrCu8AguYk.json"),
    ("book3_back", r"C:\Users\Taka\Desktop\fvtt\FotRP\需要翻译\NEW\第三本\fvtt-JournalEntry-back-matter-1ylYqjGKvevX3BgC.json"),
    ("bestiary", r"C:\Users\Taka\Desktop\fvtt\FotRP\需要翻译\NEW\pf2e.fists-of-the-ruby-phoenix-bestiary.json"),
]

patt_zh_en = re.compile(r"([一-鿿·•\-]{2,8})\s*\(\s*([A-Z][a-zA-Z\s\'\-\d]{2,40})\s*\)")
patt_en_zh = re.compile(r"([A-Z][a-zA-Z\s\'\-\d]{2,40}?)\s*\(\s*([一-鿿·•\-]{1,12})\s*\)")

# 直接搜某些关键术语 count
key_terms = ["Yoh", "Souran", "Hwanggot", "Lighthouse", "Sun Wing", "Sunwing", "Aldanar",
             "Phoenix", "Hao Jin", "Mogaru", "Tournament", "Onmyodo",
             "Goka", "Kaifen", "Iron Mountain", "Flying Mountain",
             "Xhai Zhia", "Ostovites", "Tian Xia", "Vudra"]

for name, path in FILES:
    try:
        with open(path, encoding="utf-8") as f:
            raw = f.read()
    except Exception as e:
        print(f"!! {name}: {e}")
        continue
    print(f"\n=== {name} ({len(raw)} chars) ===")
    hits = patt_zh_en.findall(raw)
    hits2 = patt_en_zh.findall(raw)
    print(f"  ZH(EN): {len(hits)}, EN(ZH): {len(hits2)}")
    if hits:
        sample = list(set([f"{zh}({en})" for zh,en in hits]))[:8]
        print(f"  ZH(EN) samples: {sample}")
    if hits2:
        sample2 = list(set([f"{en}({zh})" for en,zh in hits2]))[:8]
        print(f"  EN(ZH) samples: {sample2}")
    # term counts
    counts = {t: raw.count(t) for t in key_terms if raw.count(t)>0}
    if counts:
        print(f"  key terms: {counts}")
