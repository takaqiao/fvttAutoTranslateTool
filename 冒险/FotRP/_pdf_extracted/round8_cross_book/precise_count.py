# -*- coding: utf-8 -*-
"""精确计数 Round 8 候选 (含 字串 context-aware)"""
import os, re, json

ROOT = r"C:\Users\Taka\Desktop\fvtt\FotRP\需要翻译\NEW"

FILES = {
    "book1_handouts": os.path.join(ROOT, "第一本", "fvtt-JournalEntry-handouts-FhOWhqywZr72iuml.json"),
    "book1_ch1":      os.path.join(ROOT, "第一本", "fvtt-JournalEntry-chapter-1-c0yHRsNbVDGXKaIu.json"),
    "book1_ch2":      os.path.join(ROOT, "第一本", "fvtt-JournalEntry-chapter-2-PpZKICROB7B08r53.json"),
    "book1_ch3":      os.path.join(ROOT, "第一本", "fvtt-JournalEntry-chapter-3-jy3Vrm5yA3jrrG5W.json"),
    "book1_back":     os.path.join(ROOT, "第一本", "fvtt-JournalEntry-back-matter-ti7ZfnFRd1IabwaA.json"),
    "book2_handouts": os.path.join(ROOT, "第二本", "fvtt-JournalEntry-handouts-rXylkhhxGCEispwF.json"),
    "book2_ch1":      os.path.join(ROOT, "第二本", "fvtt-JournalEntry-chapter-1-RzfDjQH8KPxPJ2kK.json"),
    "book2_ch2":      os.path.join(ROOT, "第二本", "fvtt-JournalEntry-chapter-2-FTJT1CRdfjQzy3Ek.json"),
    "book2_ch3":      os.path.join(ROOT, "第二本", "fvtt-JournalEntry-chapter-3-SPXqld4nww21Orrg.json"),
    "book2_back":     os.path.join(ROOT, "第二本", "fvtt-JournalEntry-back-matter-XLDMbpumIxhSEWd4.json"),
    "book3_handouts": os.path.join(ROOT, "第三本", "fvtt-JournalEntry-handouts-diQf6WXrCu8AguYk.json"),
    "book3_ch1":      os.path.join(ROOT, "第三本", "fvtt-JournalEntry-chapter-1-xClvGtftweJDu3vX.json"),
    "book3_ch2":      os.path.join(ROOT, "第三本", "fvtt-JournalEntry-chapter-2-fwmZr935hxQLlBus.json"),
    "book3_ch3":      os.path.join(ROOT, "第三本", "fvtt-JournalEntry-chapter-3-VPgzvXimMH8NKzBk.json"),
    "book3_back":     os.path.join(ROOT, "第三本", "fvtt-JournalEntry-back-matter-1ylYqjGKvevX3BgC.json"),
    "bestiary":       os.path.join(ROOT, "pf2e.fists-of-the-ruby-phoenix-bestiary.json"),
}

contents = {}
for k, p in FILES.items():
    with open(p, encoding="utf-8") as f:
        contents[k] = f.read()

# 候选替换 (按精确字串)
# (old, new, src_note, allow_files_regex_or_None)
candidates = [
    # A. NPC name unification (book1/book2 → bestiary canonical)
    ("蜂蛰哈尔斯丁", "哈尔斯丁·刺痛",                       "Halspin: bestiary canonical"),
    ("乌姆巴西",     "乌木巴西",                             "Umbasi: bestiary canonical"),
    ("兰通多",       "兰托多",                               "Lantondo: bestiary canonical"),
    ("阿托斯·罗德里万", "阿图斯·罗德里万",                   "Artus full name"),
    ("阿托斯",       "阿图斯",                               "Artus short (after full match)"),
    ("阿姆扎双胞胎", "阿莫扎双子",                           "Ahmoza Twins"),
    ("阿姆扎",       "阿莫扎",                               "Ahmoza short"),
    ("克兰基斯",     "库拉克丝",                             "Krankkiss"),
    ("努莫里兹",     "路莫里兹",                             "Numoriz"),
    ("波尼玛",       "保宁玛",                               "Paunnima"),
    ("拉吉娜",       "拉贾娜",                               "Rajna"),
    ("君（Jun",      "俊（Jun",                              "Jun (precise bilingual context)"),
    # B. Wave drift
    ("锦标赛",       "武道会",                               "Wave 7 N tournament canonical (b1+b2 leftover from Round 5)"),
    ("跨维度场景",   "跨位面场景",                           "Wave 8 BK dimension→plane (b1_back prose)"),
    # C. Bestiary internal
    ("娜迦毒液",     "娜迦裔毒液",                           "pf2cn trait 娜迦裔 canonical"),
    ("锚定射击",     "固定射击",                             "Pinning Shot: majority + b1_ch1 consistent"),
    # 巨人摔角术 vs 巨人摔角手 — both are feat-related; 巨人摔角手 is more common in PF2 翻译 (people doing it)
    # bestiary 2:1 → 巨人摔角手
    ("巨人摔角术",   "巨人摔角手",                           "Titan Wrestler: 巨人摔角手 (feat name canonical)"),
]

print("=== Round 8 — 候选替换 precise count ===\n")
total_changes = 0
for old, new, note in candidates:
    file_counts = {}
    total = 0
    for k, content in contents.items():
        n = content.count(old)
        if n > 0:
            file_counts[k] = n
            total += n
    total_changes += total
    if total > 0:
        print(f"{old:18} → {new:18} : {total:4}  ({note})")
        for k in sorted(file_counts.keys()):
            print(f"    {k:18}: {file_counts[k]}")

print(f"\nTotal: {total_changes} changes")

# 也检查 new 字串是否已存在 (避免双重 replace)
print("\n=== Sanity: 目标新字串是否已存在 (确认不会重叠) ===")
for old, new, note in candidates:
    co = sum(contents[k].count(old) for k in contents)
    cn = sum(contents[k].count(new) for k in contents)
    if co > 0:
        print(f"  '{old}' (old)={co} | '{new}' (new)={cn}  {'⚠ 双重 ' if cn > 0 and old in new else ''}")
