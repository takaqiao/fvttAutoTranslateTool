#!/usr/bin/env python3
"""Audit NPC bilingual gloss coverage — Wave 8 v2.

Strict heuristic targeting personal names only:
1. Names with · (interpunct, e.g. 多贺台·艾米) — strong transliteration signal
2. Names preceded by NPC titles (大师 / 大将军 / 女皇 / 女士 etc.)
3. Names following NPC introduction verbs (名为/称为/叫做/即/是)
4. Names already glossed in bilingual pairs (used as confirmation set)
"""
import json
import re
from pathlib import Path
from collections import defaultdict

NEW_DIR = Path("C:/Users/Taka/Desktop/fvtt/FotRP/需要翻译/NEW")

# Wave 7 already-bilingualised NPCs — skip these
WAVE7_NAMES = {
    "千野", "千野·董", "千野·童",
    "苏塔奴", "苏塔奴大师",
    "白谷真野", "白谷",
    "捣达", "捣达大师",
    "蓝蝮蛇",
    "鹰虎",
    "纪幽", "纪幽大师",
    "亚彬", "正义的亚彬",
    "拉祖", "拉祖大师", "巨星拉祖",
    "辛达拉", "雕匠辛达拉",
    "郝金", "赤凰郝金",
    "奈燕妃", "奈燕妃女皇", "奈燕妃女士",
    "优·苏然", "优·苏然大师",
    "卡熙莱",
    "祖建",
    "多贺台·艾米", "多贺台",
    "雅滕贝",
    "真慧",
}

# Common Chinese honorifics/titles to strip
TITLES = ["大将军", "大师", "大人", "夫人", "女士", "勋爵", "爵士",
          "女皇", "皇后", "天皇", "陛下", "殿下", "公主", "王子", "皇女",
          "教主", "教士", "祭司", "主教", "大祭司", "大主教",
          "院长", "校长", "掌柜", "大掌柜",
          "队长", "首领", "首脑", "长老",
          "将军", "上将", "中将", "少将", "大将", "元帅",
          "天龙",  # 天龙 is suffix used sometimes
          "雕匠", "巨星", "赤凰",
          "老衲", "圣者", "贤者", "智者", "法师", "大法师",
          "教练", "司仪", "祭主",
          "·董", "·童",
          "御医",  # imperial physician
          ]

CJK = r"一-龥"


def strip_html(text: str) -> str:
    """Strip HTML tags and enricher braces."""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'@\w+\[[^\]]+\](?:\{[^}]*\})?', ' ', text)
    text = re.sub(r'@\w+\[[^\]]+\]', ' ', text)
    text = re.sub(r'\[\[[^\]]+\]\]', ' ', text)
    return text


# Patterns capturing potential NPC names
# 1. Names with interpunct ·
INTERPUNCT_NAME = re.compile(rf'[{CJK}]{{2,4}}·[{CJK}]{{1,4}}(?:·[{CJK}]{{1,4}})?')

# 2. Title + name: 大师X, 师傅X, X大师, X大将军 etc.
# Preceding title — title comes before name: "大将军飞田卡索"
PRE_TITLE_NAMES = re.compile(rf'(?:大将军|大祭司|大主教|大法师|大师|师傅|师父|师尊|将军|教主|院长|校长|长老|大掌柜|掌柜)([{CJK}]{{2,5}})(?:[^{CJK}·]|$)')
# Following title — title comes after: "X大师" "X女士" "X女皇" etc
POST_TITLE_NAMES = re.compile(rf'(?<![{CJK}])([{CJK}]{{2,5}})(?:大师|师父|师傅|师尊|大人|大将军|将军|女皇|女士|夫人|爵士|勋爵|皇后|公主|王子|大祭司|教主|院长|校长|大掌柜|掌柜|长老|爵爷)')

# 3. Names following intro verbs: "名为X" "叫X" "称X" "即X"
INTRO_NAMES = re.compile(rf'(?:名为|名叫|称之为|称为|被称为|名号为|叫做|号称|又称|绰号|外号)\s*([{CJK}]{{2,6}})')

# 4. @Actor / Foundry references containing Chinese names: {中文 - 生物X}
# We already exclude these via strip_html so no need

# 5. Names in 「邀请X前来」 etc. patterns (NPC actors of common verbs)
ACTION_NAMES = re.compile(rf'(?:[^{CJK}]|^)([{CJK}]{{2,4}})(?:邀请|宣称|宣布|声称|表示|说道|笑道|告诉|提议|建议|要求|警告|哀求)')


def extract_potential_npc_names(content: str):
    """Extract names that look like NPCs based on context patterns."""
    text = strip_html(content)
    names = defaultdict(int)

    for m in INTERPUNCT_NAME.findall(text):
        names[m] += 1
    for m in PRE_TITLE_NAMES.findall(text):
        names[m] += 1
    for m in POST_TITLE_NAMES.findall(text):
        names[m] += 1
    for m in INTRO_NAMES.findall(text):
        names[m] += 1
    for m in ACTION_NAMES.findall(text):
        names[m] += 1

    return names


def extract_bilingual_pairs(content: str):
    """Find (zh_name, en_name) pairs in 中文（English） format."""
    text = strip_html(content)
    pattern = re.compile(rf'([{CJK}]+)[（(](?:\s)?([A-Za-z][A-Za-z\s\'\-\.]+?)[）)]')
    return pattern.findall(text)


def is_likely_npc_name(zh: str) -> bool:
    """Heuristic — is zh a likely NPC name (not a place/concept/common noun)?"""
    if len(zh) < 2 or len(zh) > 8:
        return False
    # Strip title suffix
    name = zh
    for t in TITLES:
        if name.endswith(t):
            name = name[:-len(t)]
            break
    if len(name) < 2:
        return False
    if name in WAVE7_NAMES:
        return False
    # Skip if it ends in common-noun suffixes
    common_suffixes = ['们', '中', '内', '外', '下', '上', '前', '后', '里', '间',
                       '王国', '帝国', '城市', '城堡', '神庙', '寺院',
                       '学院', '大学', '学派', '组织', '组', '团', '会', '盟', '帮',
                       '族', '部', '宗派', '派别', '门', '宗', '教']
    for s in common_suffixes:
        if name.endswith(s):
            return False
    # Skip if entirely common words
    return True


def audit_file(path: Path):
    """Audit one journal file."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    page_contents = []
    for p in data.get('pages', []):
        page_name = p.get('name', '')
        content = p.get('text', {}).get('content', '') or ''
        page_contents.append((page_name, content))

    full_text = "\n".join(c for _, c in page_contents)

    # Build set of glossed name fragments
    pairs = extract_bilingual_pairs(full_text)
    glossed_fragments = set()
    glossed_full = []
    for zh, en in pairs:
        glossed_full.append((zh, en))
        # Pull last 2-6 chars of zh — these are typically the name root
        for n in range(2, 7):
            if len(zh) >= n:
                glossed_fragments.add(zh[-n:])
        glossed_fragments.add(zh)

    # Find NPC name candidates
    npc_candidates = extract_potential_npc_names(full_text)

    # Filter: keep names that aren't already glossed and look like NPC names
    missing = []
    for name, cnt in sorted(npc_candidates.items(), key=lambda x: -x[1]):
        if name in WAVE7_NAMES:
            continue
        if not is_likely_npc_name(name):
            continue
        # Is any fragment of this name in glossed_fragments?
        # Check if name overlaps with glossed (e.g., name "X" appears as suffix of a glossed zh)
        is_glossed = False
        if name in glossed_fragments:
            is_glossed = True
        # Also check: if the name (or its prefix/suffix subset) is in glossed
        for frag in [name, name[1:], name[:-1]]:
            if len(frag) >= 2 and frag in glossed_fragments:
                is_glossed = True
                break
        if is_glossed:
            continue
        # Find first page
        first_page = None
        first_page_idx = None
        for i, (pname, content) in enumerate(page_contents):
            if name in strip_html(content):
                first_page = pname
                first_page_idx = i
                break
        missing.append({
            'name': name,
            'count': cnt,
            'first_page': first_page or '',
            'first_page_idx': first_page_idx,
        })

    return {
        'file': path.name,
        'glossed_pairs_count': len(pairs),
        'glossed_examples': glossed_full[:20],
        'missing': missing,
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

    out_path = Path("C:/Users/Taka/Desktop/fvtt/FotRP/_audit_npc_bilingual_wave8_v2_report.txt")
    with open(out_path, 'w', encoding='utf-8') as out:
        out.write("=" * 80 + "\n")
        out.write("FotRP NPC Bilingual Gloss Audit — Wave 8 v2 (strict NPC heuristic)\n")
        out.write("=" * 80 + "\n\n")
        out.write("Detection patterns:\n")
        out.write("  1. Names with · interpunct (e.g. 多贺台·艾米)\n")
        out.write("  2. Names preceded by title (大师X, 师傅X, 大将军X)\n")
        out.write("  3. Names followed by title (X大师, X女士, X夫人)\n")
        out.write("  4. Names after intro verb (名为X, 叫X, 称为X)\n")
        out.write("  5. Names with NPC action verbs (X邀请, X说道, X宣称)\n\n")

        total_missing = 0
        for r in report:
            if not r['missing']:
                continue
            out.write(f"\n## {r['file']}\n")
            out.write(f"   Bilingual pairs already in file: {r['glossed_pairs_count']}\n")
            out.write(f"   Glossed examples: {[zh + '/' + en for zh, en in r['glossed_examples'][:8]]}\n")
            out.write(f"   Missing bilingual (≥1 occurrence):\n")
            for m in r['missing']:
                total_missing += 1
                out.write(f"     - {m['name']} (×{m['count']})  first page: '{m['first_page'][:80]}'\n")

        out.write(f"\n\n## TOTAL: {total_missing} candidates across all files\n")
    print(f"Report: {out_path}")


if __name__ == '__main__':
    main()
