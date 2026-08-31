# -*- coding: utf-8 -*-
"""
Round 8 QA:
1. JSON 全部解析有效
2. HTML 标签平衡
3. 残留扫描 — 之前的旧字串是否还在 (确认换干净)
"""
import os, json, re
from html.parser import HTMLParser
from collections import Counter

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

results = []

# ==== QA 1: JSON valid ====
print("=== QA 1: JSON Valid ===")
all_valid = True
for k, p in FILES.items():
    try:
        with open(p, encoding="utf-8") as f:
            json.load(f)
        print(f"  OK {k}")
    except Exception as e:
        print(f"  FAIL {k}: {e}")
        all_valid = False
        results.append(f"JSON ERR in {k}: {e}")
print("ALL JSON VALID" if all_valid else "JSON ISSUES")

# ==== QA 2: HTML balance ====
print("\n=== QA 2: HTML Balance ===")
SELF_CLOSING = {"br", "hr", "img", "input", "link", "meta", "wbr", "source", "track", "col"}

class TagBalance(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack = []
        self.imbalances = []
    def handle_starttag(self, tag, attrs):
        if tag not in SELF_CLOSING:
            self.stack.append(tag)
    def handle_startendtag(self, tag, attrs):
        # XHTML self-closing <hr/> — 不入栈也不报错
        pass
    def handle_endtag(self, tag):
        # 自闭合标签的 </> 应忽略
        if tag in SELF_CLOSING:
            return
        if not self.stack:
            self.imbalances.append(f"unmatched </{tag}>")
        elif self.stack[-1] == tag:
            self.stack.pop()
        else:
            self.imbalances.append(f"mismatched </{tag}> at stack top={self.stack[-1]}")
    def get_unclosed(self):
        return list(self.stack)

def extract_html_strings(obj, acc):
    if isinstance(obj, dict):
        for k, v in obj.items():
            extract_html_strings(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            extract_html_strings(v, acc)
    elif isinstance(obj, str):
        if "<" in obj and ">" in obj:
            acc.append(obj)

html_ok = True
for k, p in FILES.items():
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    strs = []
    extract_html_strings(data, strs)
    file_issues = []
    for idx, s in enumerate(strs):
        parser = TagBalance()
        try:
            parser.feed(s)
        except Exception:
            pass
        unclosed = parser.get_unclosed()
        if unclosed or parser.imbalances:
            sample = s[:80].replace("\n", "\\n")
            file_issues.append(f"  string#{idx} unclosed={unclosed} imbal={parser.imbalances[:3]} sample='{sample}'")
    if file_issues:
        print(f"  FAIL {k}: {len(file_issues)} imbalance(s)")
        for fi in file_issues[:3]:
            print(fi)
        html_ok = False
        results.append(f"HTML imbalance in {k}: {len(file_issues)}")
    else:
        print(f"  OK {k}")

# ==== QA 3: 残留扫描 — 旧字串应=0 ====
print("\n=== QA 3: 残留 (旧字串 count) ===")
old_strings = [
    "蜂蛰哈尔斯丁", "乌姆巴西", "兰通多", "阿托斯·罗德里万", "阿姆扎双胞胎",
    "克兰基斯", "努莫里兹", "波尼玛", "拉吉娜", "君（Jun",
    "锦标赛", "跨维度场景", "娜迦毒液", "锚定射击", "巨人摔角术",
    # 阿托斯 / 阿姆扎 / 阿姆扎 短形式应该 0 但可能在地名/其他场景出现, 加 warn
]
soft_check = ["阿托斯", "阿姆扎"]

residual = Counter()
soft_residual = Counter()
for k, p in FILES.items():
    with open(p, encoding="utf-8") as f:
        content = f.read()
    for old in old_strings:
        n = content.count(old)
        if n > 0:
            residual[old] += n
            print(f"  WARN {k}: {old} 残留 {n} 次")
    for s in soft_check:
        n = content.count(s)
        if n > 0:
            soft_residual[s] += n

print("\n--- Hard 残留汇总 ---")
for old, n in residual.most_common():
    print(f"  {old}: {n}")
if not residual:
    print("  无残留 OK")

print("\n--- Soft 残留 (短形式, 检查可疑) ---")
for s, n in soft_residual.most_common():
    print(f"  {s}: {n}")

# ==== QA 4: 新字串成功 inflated ====
print("\n=== QA 4: 新字串验证 ===")
new_strings = [
    "哈尔斯丁·刺痛", "乌木巴西", "兰托多", "阿图斯·罗德里万", "阿莫扎双子",
    "库拉克丝", "路莫里兹", "保宁玛", "拉贾娜", "俊（Jun",
    "武道会", "跨位面场景", "娜迦裔毒液", "固定射击", "巨人摔角手",
]
new_counts = Counter()
for k, p in FILES.items():
    with open(p, encoding="utf-8") as f:
        content = f.read()
    for new in new_strings:
        new_counts[new] += content.count(new)
for new, n in new_counts.most_common():
    print(f"  {new}: 总 {n} 次")

# 写 final report
with open("qa_report.txt", "w", encoding="utf-8") as f:
    f.write("=== Round 8 QA Report ===\n\n")
    f.write("JSON 有效性: " + ("ALL VALID" if all_valid else "ISSUES") + "\n")
    f.write("HTML 平衡: " + ("ALL OK" if html_ok else "ISSUES") + "\n")
    f.write(f"Hard 残留: {dict(residual)}\n")
    f.write(f"Soft 残留 (期望 0): {dict(soft_residual)}\n")
    f.write(f"新字串汇总: {dict(new_counts)}\n")
print("\nWrote qa_report.txt")
