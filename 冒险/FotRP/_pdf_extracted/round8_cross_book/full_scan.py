# -*- coding: utf-8 -*-
"""
Round 8 全扫描 — 三大检查:
1. Wave 历史 canonical drift 扫描 (在 书 1 / 书 2 上, 因 Round 5 只跑了书 3)
2. Bestiary canonical 名跨书一致性 (将 bestiary 中名作 canonical, 检查 journals 是否一致)
3. 3 个 bestiary 内部分歧的 prose-level 出现频次
"""
import json, re, os, sys
from collections import defaultdict, Counter

ROOT = r"C:\Users\Taka\Desktop\fvtt\FotRP\需要翻译\NEW"

FILES = [
    ("book1_handouts", os.path.join(ROOT, "第一本", "fvtt-JournalEntry-handouts-FhOWhqywZr72iuml.json")),
    ("book1_ch1",      os.path.join(ROOT, "第一本", "fvtt-JournalEntry-chapter-1-c0yHRsNbVDGXKaIu.json")),
    ("book1_ch2",      os.path.join(ROOT, "第一本", "fvtt-JournalEntry-chapter-2-PpZKICROB7B08r53.json")),
    ("book1_ch3",      os.path.join(ROOT, "第一本", "fvtt-JournalEntry-chapter-3-jy3Vrm5yA3jrrG5W.json")),
    ("book1_back",     os.path.join(ROOT, "第一本", "fvtt-JournalEntry-back-matter-ti7ZfnFRd1IabwaA.json")),
    ("book2_handouts", os.path.join(ROOT, "第二本", "fvtt-JournalEntry-handouts-rXylkhhxGCEispwF.json")),
    ("book2_ch1",      os.path.join(ROOT, "第二本", "fvtt-JournalEntry-chapter-1-RzfDjQH8KPxPJ2kK.json")),
    ("book2_ch2",      os.path.join(ROOT, "第二本", "fvtt-JournalEntry-chapter-2-FTJT1CRdfjQzy3Ek.json")),
    ("book2_ch3",      os.path.join(ROOT, "第二本", "fvtt-JournalEntry-chapter-3-SPXqld4nww21Orrg.json")),
    ("book2_back",     os.path.join(ROOT, "第二本", "fvtt-JournalEntry-back-matter-XLDMbpumIxhSEWd4.json")),
    ("book3_handouts", os.path.join(ROOT, "第三本", "fvtt-JournalEntry-handouts-diQf6WXrCu8AguYk.json")),
    ("book3_ch1",      os.path.join(ROOT, "第三本", "fvtt-JournalEntry-chapter-1-xClvGtftweJDu3vX.json")),
    ("book3_ch2",      os.path.join(ROOT, "第三本", "fvtt-JournalEntry-chapter-2-fwmZr935hxQLlBus.json")),
    ("book3_ch3",      os.path.join(ROOT, "第三本", "fvtt-JournalEntry-chapter-3-VPgzvXimMH8NKzBk.json")),
    ("book3_back",     os.path.join(ROOT, "第三本", "fvtt-JournalEntry-back-matter-1ylYqjGKvevX3BgC.json")),
    ("bestiary",       os.path.join(ROOT, "pf2e.fists-of-the-ruby-phoenix-bestiary.json")),
]

# 加载所有文件
file_contents = {}
for key, p in FILES:
    with open(p, encoding="utf-8") as f:
        file_contents[key] = f.read()

# ==== 检查 1: Wave 历史 canonical drift (整库扫描, 不只第三本) ====
# 来自 wave_drift_scan.txt 历史数据
wave_drift_pairs = [
    # (old, new, source, prose_only_warning)
    # Wave 8 BK
    ("维度", "位面",       "Wave 8 BK (dimension→plane)", True),  # 但 维度叠加/吞噬/连击 等 Syndara 能力名固定, 需上下文
    ("选手", "参赛者",     "Wave 8 BK (player→contestant)", True),  # 但 晋级选手 是固定术语
    # Wave 7 N
    ("锦标赛", "武道会",   "Wave 7 N (Tournament canonical 244 处)", False),
    # Wave 6 C2
    ("位面之门", "次元门", "Wave 6 C2 (Dimension Door spell)", False),
    # Ki→Qi Remaster 区分
    ("真气", "斗气",       "Wave 6 (Qi Remaster — careful: pf2cn 法术名仍是 真气)", True),
    # Wave 4
    ("阳舰二号", "太阳剑·二式", "Wave 4 (Sunwing II)", False),
    # GM 旁白 您→你 (Wave 7-8 manual)
    ("您",       "你",     "Wave 7-8 GM 旁白 (除引号内)", True),
]

drift_report = ["=== Round 8 — Wave 历史 canonical 跨书 drift 扫描 ===\n"]

for key, content in file_contents.items():
    file_drifts = []
    for old, new, src, prose_only in wave_drift_pairs:
        n = content.count(old)
        if n > 0:
            new_n = content.count(new) if new not in old else 0
            warning = " ⚠ prose_only" if prose_only else ""
            file_drifts.append(f"  {old}={n} (→ {new}, {src}){warning}")
    if file_drifts:
        drift_report.append(f"\n--- {key} ---")
        drift_report.extend(file_drifts)

with open("drift_scan.txt", "w", encoding="utf-8") as out:
    out.write("\n".join(drift_report))

# ==== 检查 2: Bestiary canonical 名跨书一致性 ====
# 加载 bestiary actor 名映射
bestiary_path = file_contents["bestiary"]
bestiary_obj = json.loads(bestiary_path)
canonical_names = []  # [(english_key, chinese_canonical)]
for en_key, actor in bestiary_obj.get("entries", {}).items():
    if not isinstance(actor, dict): continue
    full_name = actor.get("name", "")
    m = re.search(r"([A-Z][A-Za-z\(\)\d\s\-\'’]*)\s*$", full_name)
    if m:
        zh_part = full_name[:m.start()].strip()
        if zh_part:
            # 去掉 （X级） 部分, 保留 base 名
            base_zh = re.sub(r"[（(]\s*\d+\s*级\s*[)）]", "", zh_part).strip()
            canonical_names.append((en_key, base_zh))

# 对每个 canonical_zh, 跨 journal 文件数它出现的次数
# 同时也看 English 短形式出现 (识别用了不同中文)
consistency_report = ["=== Round 8 — Bestiary canonical 名跨 journal 一致性 ===\n"]
consistency_report.append("# 找出: bestiary canonical 中名 = X, 但 journal 中 X 罕见或不存在\n")
consistency_report.append("# (说明 journal 可能用了不同中文渲染)\n\n")

for en_key, zh_canon in canonical_names:
    # 只看明显的 NPC 名 (中文 >= 2 字)
    if len(zh_canon) < 2: continue
    # 去 levels variant 重复
    journal_keys = [k for k in file_contents if k != "bestiary"]
    journal_counts = {}
    for k in journal_keys:
        n = file_contents[k].count(zh_canon)
        if n > 0:
            journal_counts[k] = n
    # 也看 English 短形式 (e.g., "Yoh Souran" → take first word "Yoh", or full)
    en_short = en_key.split("(")[0].strip()  # 去 (Level X)
    en_first_word = en_short.split()[0] if en_short else ""
    en_appearances = {}
    if en_short and len(en_short) > 3:
        for k in journal_keys:
            n = file_contents[k].count(en_short)
            if n > 0:
                en_appearances[k] = n
    # 报告: en 在 journal 出现 但 zh_canon 不出现 → 嫌疑分歧
    suspect = []
    for jk in journal_keys:
        en_n = en_appearances.get(jk, 0)
        zh_n = journal_counts.get(jk, 0)
        if en_n > 0 and zh_n == 0:
            suspect.append(f"    {jk}: en={en_n}, zh canonical=0 ⚠")
    if suspect or len(journal_counts) > 0:
        consistency_report.append(f"\n• {en_short:35} canonical 中文 = '{zh_canon}'")
        if journal_counts:
            consistency_report.append(f"    中文分布: {dict(journal_counts)}")
        if en_appearances:
            consistency_report.append(f"    EN短形式分布: {dict(en_appearances)}")
        for s in suspect:
            consistency_report.append(s)

with open("cross_book_consistency.txt", "w", encoding="utf-8") as out:
    out.write("\n".join(consistency_report))

# ==== 检查 3: 内部 bestiary 分歧 — 在所有文件中各形式 frequency ====
internal_pairs = [
    ("Nagaji Venom", "娜迦毒液", "娜迦裔毒液"),
    ("Titan Wrestler", "巨人摔角术", "巨人摔角手"),
    ("Pinning Shot", "固定射击", "锚定射击"),
]
internal_report = ["=== Round 8 — Bestiary 内部分歧的 prose 分布 ===\n"]
for en, opt1, opt2 in internal_pairs:
    internal_report.append(f"\n{en}: '{opt1}' vs '{opt2}'")
    for key, content in file_contents.items():
        n1 = content.count(opt1)
        n2 = content.count(opt2)
        if n1 > 0 or n2 > 0:
            internal_report.append(f"  {key:18}: {opt1}={n1}, {opt2}={n2}")

with open("internal_inconsistencies.txt", "w", encoding="utf-8") as out:
    out.write("\n".join(internal_report))

print("Generated:")
print("  drift_scan.txt")
print("  cross_book_consistency.txt")
print("  internal_inconsistencies.txt")
