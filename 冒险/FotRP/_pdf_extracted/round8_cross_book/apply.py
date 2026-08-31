# -*- coding: utf-8 -*-
"""
Round 8 Apply — 备份 + 替换
执行顺序: 长字串先 (避免短匹配先吃掉长字串).
"""
import os, shutil, json, sys, re

ROOT = r"C:\Users\Taka\Desktop\fvtt\FotRP\需要翻译\NEW"
BACKUP_BASE = r"C:\Users\Taka\Desktop\fvtt\FotRP\_backup\20260519_round8_cross_book"

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

# 替换执行顺序: 长字串先, 短字串后. 同一字串多渲染先对全长再对部分.
REPLACEMENTS = [
    # 跨书 NPC 全名 (长字串先)
    ("阿托斯·罗德里万", "阿图斯·罗德里万"),
    ("阿姆扎双胞胎",     "阿莫扎双子"),
    ("蜂蛰哈尔斯丁",     "哈尔斯丁·刺痛"),
    # 跨书 NPC 短名 (在长字串之后)
    ("阿托斯",          "阿图斯"),
    ("阿姆扎",          "阿莫扎"),
    ("乌姆巴西",        "乌木巴西"),
    ("兰通多",          "兰托多"),
    ("克兰基斯",        "库拉克丝"),
    ("努莫里兹",        "路莫里兹"),
    ("波尼玛",          "保宁玛"),
    ("拉吉娜",          "拉贾娜"),
    # 精确 bilingual context — 仅替换 "君（Jun" 而不是裸 "君"
    ("君（Jun",         "俊（Jun"),
    # Wave drift
    ("锦标赛",          "武道会"),
    ("跨维度场景",      "跨位面场景"),
    # Bestiary internal
    ("娜迦毒液",        "娜迦裔毒液"),
    ("锚定射击",        "固定射击"),
    ("巨人摔角术",      "巨人摔角手"),
]

# Step 1: 备份
print("Step 1: 备份")
if os.path.exists(BACKUP_BASE):
    print(f"  备份目录已存在 (rerun): {BACKUP_BASE}")
else:
    os.makedirs(BACKUP_BASE)
    for k, p in FILES.items():
        rel = os.path.relpath(p, ROOT)
        dest = os.path.join(BACKUP_BASE, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(p, dest)
        print(f"    backed up: {rel}")

# Step 2: 替换
print("\nStep 2: 替换")
total_changes = 0
report = []
for k, p in FILES.items():
    with open(p, encoding="utf-8") as f:
        content = f.read()
    original = content
    file_changes = 0
    file_log = []
    for old, new in REPLACEMENTS:
        n_before = content.count(old)
        if n_before > 0:
            content = content.replace(old, new)
            file_changes += n_before
            file_log.append(f"    {old} → {new}: {n_before}")
    if file_changes > 0:
        # JSON 解析验证
        try:
            json.loads(content)
        except json.JSONDecodeError as e:
            print(f"  !! {k} JSON 解析失败 — 不写入: {e}")
            report.append(f"  !! {k} JSON ERROR: {e}")
            continue
        # 写入
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  {k}: {file_changes} 处")
        for line in file_log:
            print(line)
        report.append(f"\n{k}: {file_changes} 处")
        report.extend(file_log)
        total_changes += file_changes

print(f"\nTotal: {total_changes} 处替换")

# Step 3: 写报告
with open("apply_report.txt", "w", encoding="utf-8") as f:
    f.write(f"=== Round 8 Apply Report ===\n")
    f.write(f"Total: {total_changes} 处\n\n")
    f.write("\n".join(report))
print("Wrote: apply_report.txt")
