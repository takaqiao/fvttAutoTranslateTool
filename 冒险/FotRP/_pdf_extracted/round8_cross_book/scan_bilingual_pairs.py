# -*- coding: utf-8 -*-
"""
Round 8 / A1 — 跨书一致性扫描

策略:
1. 加载第一/二/三本所有 journal + bestiary
2. 用正则抽取 中文(English) / English(中文) 双语对照
3. 按 English 聚合, 检查是否有同一 English 对应多个 Chinese 渲染
4. 输出按"分歧大小"排序的候选清单

输入文件路径都是绝对路径, 输出到当前目录.
"""
import os, re, json, glob
from collections import defaultdict, Counter

ROOT = r"C:\Users\Taka\Desktop\fvtt\FotRP\需要翻译\NEW"

FILES = {
    "book1_handouts":  os.path.join(ROOT, "第一本", "fvtt-JournalEntry-handouts-FhOWhqywZr72iuml.json"),
    "book1_ch1":       os.path.join(ROOT, "第一本", "fvtt-JournalEntry-chapter-1-c0yHRsNbVDGXKaIu.json"),
    "book1_ch2":       os.path.join(ROOT, "第一本", "fvtt-JournalEntry-chapter-2-PpZKICROB7B08r53.json"),
    "book1_ch3":       os.path.join(ROOT, "第一本", "fvtt-JournalEntry-chapter-3-jy3Vrm5yA3jrrG5W.json"),
    "book1_back":      os.path.join(ROOT, "第一本", "fvtt-JournalEntry-back-matter-ti7ZfnFRd1IabwaA.json"),
    "book2_handouts":  os.path.join(ROOT, "第二本", "fvtt-JournalEntry-handouts-rXylkhhxGCEispwF.json"),
    "book2_ch1":       os.path.join(ROOT, "第二本", "fvtt-JournalEntry-chapter-1-RzfDjQH8KPxPJ2kK.json"),
    "book2_ch2":       os.path.join(ROOT, "第二本", "fvtt-JournalEntry-chapter-2-FTJT1CRdfjQzy3Ek.json"),
    "book2_ch3":       os.path.join(ROOT, "第二本", "fvtt-JournalEntry-chapter-3-SPXqld4nww21Orrg.json"),
    "book2_back":      os.path.join(ROOT, "第二本", "fvtt-JournalEntry-back-matter-XLDMbpumIxhSEWd4.json"),
    "book3_handouts":  os.path.join(ROOT, "第三本", "fvtt-JournalEntry-handouts-diQf6WXrCu8AguYk.json"),
    "book3_ch1":       os.path.join(ROOT, "第三本", "fvtt-JournalEntry-chapter-1-xClvGtftweJDu3vX.json"),
    "book3_ch2":       os.path.join(ROOT, "第三本", "fvtt-JournalEntry-chapter-2-fwmZr935hxQLlBus.json"),
    "book3_ch3":       os.path.join(ROOT, "第三本", "fvtt-JournalEntry-chapter-3-VPgzvXimMH8NKzBk.json"),
    "book3_back":      os.path.join(ROOT, "第三本", "fvtt-JournalEntry-back-matter-1ylYqjGKvevX3BgC.json"),
    "bestiary":        os.path.join(ROOT, "pf2e.fists-of-the-ruby-phoenix-bestiary.json"),
}

# 抽取双语对照: 中文(English) — 中文 1-12 字, English 大写开头
PATT_ZH_EN = re.compile(
    r"([一-鿿·‧･·\-]{1,12})\s*\(([A-Z][a-zA-Z\s\'\-’\d]{2,40}?)\)"
)
# English(中文) — English 大写开头, 中文 1-12 字
PATT_EN_ZH = re.compile(
    r"([A-Z][a-zA-Z\s\'\-’\d]{2,40}?)\s*\(([一-鿿·‧･·\-]{1,12})\)"
)

# 排除噪声 (HTML/JSON 元数据)
EN_BLACKLIST = {"Hide", "HP", "AC", "Saves", "Will", "Fort", "Ref", "DC", "Speed", "Str", "Dex", "Con",
                "Int", "Wis", "Cha", "OF", "BY", "ID", "API", "URL", "AP", "GM",
                "PCs", "PC", "NPC", "NPCs", "GMs", "DM", "DMs", "TC", "FC",
                "And", "But", "The", "An", "Or", "If", "When",
                "II", "III", "IV", "VI", "VII"}

# 中文 noise 词
ZH_BLACKLIST = {"诅咒", "状态", "免疫", "弱点", "抗力", "速度", "击败", "命中", "伤害", "防御"}


def strip_html(s):
    """简单去 HTML 标签"""
    s = re.sub(r"<[^>]+>", " ", s)
    return s


def extract_text(obj, acc):
    """递归收集所有字符串值"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            extract_text(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            extract_text(v, acc)
    elif isinstance(obj, str):
        if len(obj) > 10:  # 跳过短键值
            acc.append(obj)


# eng_to_zh_per_file[english_name][file_key] = Counter({zh1: n1, zh2: n2})
eng_to_zh_per_file = defaultdict(lambda: defaultdict(Counter))
# zh_to_eng_per_file 反向 (有时 PDF 标 English(中文), JSON 也用此格)
all_eng_seen = Counter()
all_zh_seen = Counter()

for key, path in FILES.items():
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"!! {key} load failed: {e}")
        continue
    strs = []
    extract_text(data, strs)
    text_all = "\n".join(strs)
    text_all = strip_html(text_all)

    # 中文(English)
    for m in PATT_ZH_EN.finditer(text_all):
        zh, en = m.group(1).strip(), m.group(2).strip()
        if en in EN_BLACKLIST: continue
        if zh in ZH_BLACKLIST: continue
        if len(zh) < 1 or len(en) < 3: continue
        # 跳过纯数字/版本号
        if re.fullmatch(r"[\dIVX]+", en): continue
        eng_to_zh_per_file[en][key][zh] += 1
        all_eng_seen[en] += 1

    # English(中文)
    for m in PATT_EN_ZH.finditer(text_all):
        en, zh = m.group(1).strip(), m.group(2).strip()
        if en in EN_BLACKLIST: continue
        if zh in ZH_BLACKLIST: continue
        if len(zh) < 1 or len(en) < 3: continue
        if re.fullmatch(r"[\dIVX]+", en): continue
        eng_to_zh_per_file[en][key][zh] += 1
        all_eng_seen[en] += 1

print(f"Total unique English terms found: {len(eng_to_zh_per_file)}")

# 找出"跨书分歧": 同一 English 在多个文件出现, 且中文渲染不同
divergent = []
for en, per_file in eng_to_zh_per_file.items():
    # 汇总所有中文渲染
    all_zh = Counter()
    files_seen = set()
    for fkey, zh_counter in per_file.items():
        files_seen.add(fkey)
        for zh, n in zh_counter.items():
            all_zh[zh] += n
    if len(all_zh) < 2:
        continue  # 只有一种渲染, 跳过
    # 但是, 「忽略 PDF-vs-JSON 这种字内分歧」 — 我们要的是 跨书 分歧
    # 检查是否真的跨书 (是否在多本书都出现)
    books = set()
    for fkey in files_seen:
        for prefix in ("book1", "book2", "book3", "bestiary"):
            if fkey.startswith(prefix):
                books.add(prefix)
                break
    if len(books) < 2:
        continue  # 单一书内部, 跳过 (单书内部一致性问题已在 Round 1-6 处理)
    divergent.append((en, all_zh, per_file, books))

# 按总出现次数排序 (优先处理高频)
divergent.sort(key=lambda x: -sum(x[1].values()))

with open("scan_divergent_terms.txt", "w", encoding="utf-8") as out:
    out.write(f"# Round 8 — 跨书一致性扫描结果\n\n")
    out.write(f"共找到 {len(divergent)} 个跨书分歧的 English 术语\n\n")
    out.write("=" * 80 + "\n\n")
    for en, all_zh, per_file, books in divergent:
        total = sum(all_zh.values())
        out.write(f"## {en}  (总 {total} 处, 跨 {len(books)} 本书)\n\n")
        out.write(f"中文渲染分布:\n")
        for zh, n in all_zh.most_common():
            out.write(f"  {zh:20} = {n}\n")
        out.write(f"\n按文件分布:\n")
        for fkey in sorted(per_file.keys()):
            zh_c = per_file[fkey]
            renderings = ", ".join(f"{zh}×{n}" for zh, n in zh_c.most_common())
            out.write(f"  {fkey:18} : {renderings}\n")
        out.write("\n" + "-" * 80 + "\n\n")

print(f"Output: scan_divergent_terms.txt — {len(divergent)} divergent terms")
