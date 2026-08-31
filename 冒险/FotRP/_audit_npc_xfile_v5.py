#!/usr/bin/env python3
"""Cross-file NPC bilingual audit — Wave 8 v5.

Strategy:
1. Find all interpunct names in all canonical files.
2. Determine canonical English gloss for each name (search any file).
3. For each file, find names that appear but aren't glossed in that file.
"""
import json
import re
from pathlib import Path
from collections import defaultdict

NEW_DIR = Path("C:/Users/Taka/Desktop/fvtt/FotRP/需要翻译/NEW")

WAVE7_NAMES = {
    "千野", "千野·董", "千野·童", "董",
    "苏塔奴", "苏塔奴大师", "塔奴",
    "白谷真野", "白谷", "真野",
    "捣达", "捣达大师",
    "蓝蝮蛇",
    "鹰虎",
    "纪幽", "纪幽大师",
    "亚彬", "正义的亚彬",
    "拉祖", "拉祖大师", "巨星拉祖",
    "辛达拉", "雕匠辛达拉",
    "郝金", "赤凰郝金", "郝金挂毯",
    "奈燕妃", "奈燕妃女皇", "奈燕妃女士",
    "优·苏然", "优·苏然大师", "苏然",
    "卡熙莱",
    "祖建",
    "多贺台·艾米", "多贺台", "艾米",
    "雅滕贝",
    "真慧",
}

# Names that aren't NPCs
NON_NPC_INTERPUNCT = {
    "太阳剑·一式", "太阳剑·二式",
    "凯登·凯连",  # deity
    "义洛理",
}

# Regex for canonical English gloss after Chinese name
GLOSS_AFTER = re.compile(
    r'\s*[（(]\s*([A-Za-z][A-Za-z\s\'\-\.]+?)(?=[，,；;）)、]|$)'
)
BARE_ENGLISH_AFTER = re.compile(
    r'\s+([A-Z][a-z]+(?:\s+[A-Z][a-z\'\-]+){1,4})(?=\s|，|$)'
)

CJK = r"一-龥"
INTERPUNCT_PATTERN = re.compile(
    rf'(?<![{CJK}·])([{CJK}]{{2,8}}(?:·[{CJK}]{{1,8}}){{1,3}})(?![{CJK}·])'
)


def strip_html(text: str) -> str:
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'@\w+\[[^\]]+\](?:\{[^}]*\})?', ' ', text)
    text = re.sub(r'@\w+\[[^\]]+\]', ' ', text)
    text = re.sub(r'\[\[[^\]]+\]\]', ' ', text)
    return text


def load_file(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    page_contents = []
    for p in data.get('pages', []):
        page_name = p.get('name', '')
        content = p.get('text', {}).get('content', '') or ''
        page_contents.append((page_name, strip_html(content)))
    return page_contents


def find_canonical_gloss(name: str, all_text: str) -> str | None:
    """Find any (English) gloss after `name` in all_text. Return first match."""
    # Search all occurrences
    for m in re.finditer(re.escape(name), all_text):
        after = all_text[m.end():m.end() + 60]
        # Paren-enclosed
        g = GLOSS_AFTER.match(after)
        if g:
            eng = g.group(1).strip()
            # Must have at least 1 space — usually multi-word
            # Or single capitalized word ≥ 4 chars
            if (' ' in eng and len(eng) >= 4) or (len(eng) >= 4 and eng[0].isupper()):
                return eng
        # Bare English (no paren)
        g = BARE_ENGLISH_AFTER.match(after)
        if g:
            return g.group(1).strip()
    return None


def find_name_with_simple_followup(name: str, after_text: str) -> bool:
    """Check if name is immediately followed by (English) or bare English."""
    g = GLOSS_AFTER.match(after_text)
    if g:
        eng = g.group(1).strip()
        if (' ' in eng and len(eng) >= 4) or (len(eng) >= 4 and eng[0].isupper()):
            return True
    g = BARE_ENGLISH_AFTER.match(after_text)
    if g:
        return True
    return False


def main():
    files = []
    for book in ['第一本', '第二本', '第三本']:
        book_dir = NEW_DIR / book
        for f in sorted(book_dir.glob('*.json')):
            files.append(f)

    # Load all
    file_data = {}
    all_pages_combined = ""
    for f in files:
        pages = load_file(f)
        file_data[f.name] = pages
        for _, content in pages:
            all_pages_combined += content + "\n"

    # Find ALL unique interpunct names across all files
    all_names = set()
    for f in files:
        for _, content in file_data[f.name]:
            for m in INTERPUNCT_PATTERN.finditer(content):
                name = m.group(1)
                # Filter overlapping patterns by capturing the LONGEST form
                # E.g. "兰雅·什瓦纳特丝" not "兰雅·什瓦纳特"
                all_names.add(name)

    # For each unique name, find its canonical English gloss (from any file)
    name_canonical = {}
    for name in all_names:
        if name in WAVE7_NAMES:
            continue
        if name in NON_NPC_INTERPUNCT:
            continue
        if name.startswith("太阳剑") or name.startswith("凯登·凯连"):
            continue
        # If name starts with a known prefix that's a phrase artifact, skip
        artifacts_prefix = ['可对', '虽然', '溃了', '人库比亚', '看到', '了阿玛拉',
                            '溃了', '撞见', '但阿基拉', '认为普希', '人阿达纳',
                            '可辨认为', '与阿达纳', '与纳瓦里', '叫多贺台',
                            '仪多贺台', '阿达纳·乌马尔挨', '但布肯',
                            '然玛莱卡', '玛莱卡·陶为他们', '玛莱卡·陶会',
                            '对玛莱卡', '玛莱卡·陶与阿达',
                            '长阿玛拉', '时阿玛拉', '得阿玛拉', '把琴·禅古罗']
        if any(name.startswith(p) for p in artifacts_prefix):
            continue
        # Check if name overlaps any artifact substring
        if any(p in name for p in ['可对', '撞见', '看到', '虽然']):
            continue

        gloss = find_canonical_gloss(name, all_pages_combined)
        if gloss:
            name_canonical[name] = gloss
        else:
            # Track even ungiossed - might be a real NPC
            name_canonical[name] = None

    # For each file, find each name's first occurrence, check if gloss is local
    report_per_file = defaultdict(list)
    for fname, pages in file_data.items():
        for name, canonical_eng in name_canonical.items():
            # Where does this name first appear in this file?
            first_pos = None
            first_page_name = None
            for i, (pname, content) in enumerate(pages):
                idx = content.find(name)
                if idx != -1:
                    first_pos = (i, idx)
                    first_page_name = pname
                    full_content = content
                    break
            if first_pos is None:
                continue
            i, idx = first_pos
            full_content = pages[i][1]
            after_text = full_content[idx + len(name):idx + len(name) + 60]
            # Check if local first occurrence has gloss
            has_local_gloss = find_name_with_simple_followup(name, after_text)
            if not has_local_gloss:
                # Get context
                start = max(0, idx - 30)
                end = min(len(full_content), idx + len(name) + 60)
                context = full_content[start:end].replace('\n', ' ').replace('  ', ' ')
                # Count total occurrences in this file
                count = 0
                for _, c in pages:
                    count += c.count(name)
                report_per_file[fname].append({
                    'name': name,
                    'count_in_file': count,
                    'first_page': first_page_name,
                    'context': context,
                    'canonical_eng': canonical_eng,
                })

    # Write report
    out_path = Path("C:/Users/Taka/Desktop/fvtt/FotRP/_audit_npc_bilingual_wave8_v5_report.txt")
    with open(out_path, 'w', encoding='utf-8') as out:
        out.write("=" * 80 + "\n")
        out.write("FotRP Cross-File NPC Bilingual Audit — Wave 8 v5\n")
        out.write("=" * 80 + "\n")
        out.write("For each interpunct name X·Y, check if its first-mention in each\n")
        out.write("file has an inline English gloss. If canonical English is known\n")
        out.write("(found elsewhere), suggest adding it.\n\n")

        total = 0
        for fname in sorted(report_per_file.keys()):
            items = report_per_file[fname]
            if not items:
                continue
            out.write(f"\n## {fname}\n")
            out.write(f"   {len(items)} name(s) lacking inline bilingual on first mention:\n")
            for item in items:
                total += 1
                out.write(f"     - {item['name']} (×{item['count_in_file']} in file)\n")
                out.write(f"       page: '{item['first_page'][:80] if item['first_page'] else '?'}'\n")
                eng_str = f"YES, canonical: ({item['canonical_eng']})" if item['canonical_eng'] else "NO canonical English found in corpus"
                out.write(f"       Canonical EN gloss known? {eng_str}\n")
                out.write(f"       context: \"{item['context']}\"\n")

        out.write(f"\n\n## TOTAL: {total} (name, file) pairs lacking local first-mention bilingual\n")
        out.write(f"## Canonical glosses found in corpus: {len([n for n, e in name_canonical.items() if e])}\n")
    print(f"Report: {out_path}")


if __name__ == '__main__':
    main()
