#!/usr/bin/env python3
"""Audit NPC bilingual gloss — Wave 8 v4.

For each potential NPC name found via heuristic context, check if there's
an English gloss within a small character window after the name.
This catches false positives where the name IS glossed inline.
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

CJK = r"一-龥"


def strip_html(text: str) -> str:
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'@\w+\[[^\]]+\](?:\{[^}]*\})?', ' ', text)
    text = re.sub(r'@\w+\[[^\]]+\]', ' ', text)
    text = re.sub(r'\[\[[^\]]+\]\]', ' ', text)
    return text


# Find all interpunct names with possible glossing
# An interpunct name is at least 2 Han chars + · + 1+ Han chars
# After it, within 30 chars, check for (English) or （English）
NAME_AND_OPTIONAL_GLOSS = re.compile(
    rf'([{CJK}]{{2,6}}(?:·[{CJK}]{{1,6}})+(?:[丝兹齐][^{CJK}·])?)'
    r'(.{0,30})',
    re.DOTALL
)


def find_all_interpunct_names(text: str):
    """Return list of (name, context_after) for interpunct names."""
    text = strip_html(text)
    pattern = re.compile(rf'(?:[^{CJK}·]|^)([{CJK}]{{2,6}}·[{CJK}]{{1,6}}(?:·[{CJK}]{{1,6}})?)(?=[^{CJK}·]|$)')
    results = []
    for m in pattern.finditer(text):
        name = m.group(1)
        after = text[m.end():m.end() + 40]
        results.append((name, after, m.start()))
    return results


def has_english_gloss_after(after_text: str, max_distance: int = 40) -> str | None:
    """Check if there's an English gloss within max_distance chars.

    Returns the matched English name if found, None otherwise.
    """
    # Try common patterns
    # 1. （English ...） — find English at start of parenthetical
    # Closing may be ）, comma, , or other punctuation
    m = re.match(r'\s*[（(]([A-Za-z][A-Za-z\s\'\-\.]+?)(?=[，,；;）)]|$)', after_text)
    if m:
        eng = m.group(1).strip()
        if len(eng) >= 3:
            return eng
    # 2. /English (some files use slash)
    m = re.match(r'\s*/\s*([A-Za-z][A-Za-z\s\'\-\.]+?)(?=[，。\s）)])', after_text)
    if m:
        return m.group(1).strip()
    # 3. Plain English with leading space (no paren): " Urnak Lostwind"
    m = re.match(r'\s+([A-Z][a-z]+(?:\s+[A-Z][a-z\']+)+)', after_text)
    if m:
        eng = m.group(1).strip()
        if len(eng) >= 4:
            return eng
    return None


def audit_file(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    page_contents = []
    for p in data.get('pages', []):
        page_name = p.get('name', '')
        content = p.get('text', {}).get('content', '') or ''
        page_contents.append((page_name, content))

    # Build full text, but ALSO keep page boundaries
    full_text = "\n".join(c for _, c in page_contents)

    # Find all interpunct names in file
    all_interpunct_occurrences = find_all_interpunct_names(full_text)

    # Group by name; first occurrence's gloss status determines coverage
    by_name = defaultdict(list)
    for name, after, pos in all_interpunct_occurrences:
        by_name[name].append((after, pos))

    # For each unique name, check if it has gloss anywhere
    name_status = {}  # name -> (count, has_gloss_anywhere, first_after_no_gloss)
    for name, occs in by_name.items():
        if name in WAVE7_NAMES:
            continue
        # Skip if name's bare form (without trailing single char) is in WAVE7
        if any(name.startswith(w) for w in WAVE7_NAMES):
            continue
        if name == "太阳剑·二式" or name == "太阳剑·一式" or name.startswith("太阳剑"):
            continue
        if name == "凯登·凯连" or name.startswith("凯登·凯连"):
            continue
        has_gloss = False
        first_glossed = None
        first_no_gloss_after = None
        for after, pos in occs:
            gloss = has_english_gloss_after(after)
            if gloss:
                has_gloss = True
                first_glossed = gloss
                break
            elif first_no_gloss_after is None:
                first_no_gloss_after = after[:30]
        name_status[name] = {
            'count': len(occs),
            'has_gloss': has_gloss,
            'gloss': first_glossed,
            'first_after': first_no_gloss_after,
        }

    # Now find which page each name first appears on
    first_page = {}
    first_context = {}
    for name in name_status:
        for i, (pname, content) in enumerate(page_contents):
            ct = strip_html(content)
            if name in ct:
                first_page[name] = pname
                idx = ct.find(name)
                start = max(0, idx - 30)
                end = min(len(ct), idx + len(name) + 60)
                first_context[name] = ct[start:end].replace('\n', ' ')
                break

    return {
        'file': path.name,
        'name_status': name_status,
        'first_page': first_page,
        'first_context': first_context,
    }


def main():
    files = []
    for book in ['第一本', '第二本', '第三本']:
        book_dir = NEW_DIR / book
        for f in sorted(book_dir.glob('*.json')):
            files.append(f)

    report = []
    for f in files:
        try:
            res = audit_file(f)
            report.append(res)
        except Exception as e:
            print(f"ERROR on {f.name}: {e}")
            import traceback; traceback.print_exc()

    out_path = Path("C:/Users/Taka/Desktop/fvtt/FotRP/_audit_npc_bilingual_wave8_v4_report.txt")
    with open(out_path, 'w', encoding='utf-8') as out:
        out.write("=" * 80 + "\n")
        out.write("FotRP Interpunct NPC Bilingual Audit — Wave 8 v4\n")
        out.write("=" * 80 + "\n")
        out.write("Checks: does name X·Y have a (English) gloss anywhere in file?\n\n")

        total_missing = 0
        total_glossed = 0
        for r in report:
            missing = []
            glossed = []
            for name, status in sorted(r['name_status'].items()):
                if status['has_gloss']:
                    glossed.append((name, status))
                else:
                    missing.append((name, status))

            total_missing += len(missing)
            total_glossed += len(glossed)

            if not missing:
                continue
            out.write(f"\n## {r['file']}\n")
            out.write(f"   Names already glossed: {len(glossed)}\n")
            out.write(f"   Names lacking ANY gloss in file: {len(missing)}\n")
            for name, status in missing:
                out.write(f"     - {name} (×{status['count']})\n")
                out.write(f"       page: '{r['first_page'].get(name, '?')[:80]}'\n")
                out.write(f"       context: \"{r['first_context'].get(name, '?')}\"\n")
                out.write(f"       after first occurrence: \"{status['first_after']}\"\n")

        out.write(f"\n\n## TOTAL\n")
        out.write(f"   Glossed: {total_glossed} unique interpunct names\n")
        out.write(f"   Missing: {total_missing} unique interpunct names\n")
    print(f"Report: {out_path}")


if __name__ == '__main__':
    main()
