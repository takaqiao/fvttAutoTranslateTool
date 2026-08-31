#!/usr/bin/env python3
"""Audit NPC bilingual gloss — Wave 8 v3.

For each candidate name, also pull the context so user can verify.
"""
import json
import re
from pathlib import Path
from collections import defaultdict

NEW_DIR = Path("C:/Users/Taka/Desktop/fvtt/FotRP/需要翻译/NEW")

# Wave 7 already-bilingualised NPCs — skip these (also include common variants)
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

# Strongly-suggesting items that aren't NPCs (weapons, titles, etc.)
NON_NPC = {
    "太阳剑·一式", "太阳剑·二式",  # weapons
    "凯登·凯连",  # deity
    "义洛理", "阿巴达尔", "阿斯莫迪斯",  # deities
}

# Item / phrase blacklist  -- regex captured partial sentence fragments
PHRASE_FRAGMENTS = {
    # Trailing fragments often captured by regex
    "而是多位", "过峡爱戴的", "回到她作为", "是首位",
    "我是修诺", "或者用郝金", "正是修诺",
    "若队伍", "挑战一旦", "接受", "有权", "来源",
    "民那塔的群岛", "邦木的小岛", "邦木岛",
    "石桌集市", "活动来", "向执行者", "不听", "是当年",
    "人类武器", "武器", "执行者", "活动",
    "把琴·禅古罗的",
    "被奈燕妃", "尽管奈燕妃", "这是角色给", "整队在",
    "可以追溯", "角色不仅在", "食人魔法师", "直到郝金", "郝金便",
    "课程而闻名", "在她所谓", "尽管", "过峡夫人与", "奈珠朝",
    "自奈燕妃", "龙兽", "在这位", "别了我的", "这位",
    "莱特拉的瓦瑞", "相崎的黑风女", "魔加鲁王或终",
    "前去", "过峡人", "附近居民",
    "玛莱卡·陶与阿达",  # incomplete
    "阿达纳·乌马尔勋",  # truncated
    "飞田卡索大",  # truncated
    "大掌柜命·玛丽",  # this is 大掌柜 + 命·玛丽, captured incorrectly
    "布肯·特古拉爵",  # truncated
    "叫多贺台·艾米", "纳瓦里·万家乐夫", "然玛莱卡·陶",
    "玛莱卡·陶会以三", "对玛莱卡·陶", "仪多贺台·艾米",
    "阿达纳·乌马尔挨", "但布肯·特古拉爵", "之袍",
    "介绍", "弓匠打造了", "的彩排表演", "一位", "的口味",
    "的吟游诗人", "过峡的大", "商业领袖与", "飞田", "能给这位",
    "乃至过峡", "除了为", "与过峡", "这位", "飞田大", "当大",
    "人所知的术士", "琉璃光屋", "冰牙巢的庞大", "鹏的圣鸟之灵",
    "莎琳之梳", "伊阿加拉的古", "玛莱卡还", "这场展示", "然后悄悄",
    "继续", "便愉快地", "她接着", "准备",
    "溃了千野·董", "人库比亚·骨柱", "骨柱的",
    "邱美莎", "奈羽妃", "燕轻爵", "乌马尔", "夏之芽", "特古拉",
    "万家乐",
    "看到千野·董", "了阿玛拉·李",
    "凯登·凯连的传", "凯登·凯连那狂", "千野·董亲自向",
    "太阳剑·二式", "太阳剑·一式",
    "所定的规程", "视作家园", "这位剑术", "林冠", "当奈燕妃",
    "艾森格里之造", "炅裕的人类女", "便会被", "以所",
    "你可以", "立彦", "认为普希·努瓦",
    "所造", "这群元素", "辛达拉的奇匠", "缠绕长蛇", "托拉洛阿",
    "两姐妹", "如果角色", "你能", "真慧点头", "充分",
    "施展自己的", "乐器", "这位邪恶的", "陶玛塔", "高等",
    "的岛屿要塞", "救世主之时", "狸缗灶", "但我", "这只狸的", "但他",
    "郝金即将", "坎迪维西安",
    # already glossed elsewhere
}

CJK = r"一-龥"


def strip_html(text: str) -> str:
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'@\w+\[[^\]]+\](?:\{[^}]*\})?', ' ', text)
    text = re.sub(r'@\w+\[[^\]]+\]', ' ', text)
    text = re.sub(r'\[\[[^\]]+\]\]', ' ', text)
    return text


def get_context(text: str, name: str, length: int = 60) -> str:
    """Get the surrounding context of name (first occurrence)."""
    idx = text.find(name)
    if idx == -1:
        return ""
    start = max(0, idx - length // 2)
    end = min(len(text), idx + len(name) + length // 2)
    snippet = text[start:end].replace('\n', ' ')
    return snippet


# Patterns capturing potential NPC names
INTERPUNCT_NAME = re.compile(rf'[{CJK}]{{2,4}}·[{CJK}]{{1,4}}(?:·[{CJK}]{{1,4}})?')
PRE_TITLE_NAMES = re.compile(rf'(?:大将军|大祭司|大主教|大法师|大师|师傅|师父|师尊|将军|教主|院长|校长|长老|大掌柜|掌柜)([{CJK}]{{2,5}})(?:[^{CJK}·]|$)')
POST_TITLE_NAMES = re.compile(rf'(?<![{CJK}])([{CJK}]{{2,5}})(?:大师|师父|师傅|师尊|大人|大将军|将军|女皇|女士|夫人|爵士|勋爵|皇后|公主|王子|大祭司|教主|院长|校长|大掌柜|掌柜|长老|爵爷)')
INTRO_NAMES = re.compile(rf'(?:名为|名叫|称之为|称为|被称为|名号为|叫做|号称|又称|绰号|外号)\s*([{CJK}]{{2,6}})')
ACTION_NAMES = re.compile(rf'(?:[^{CJK}]|^)([{CJK}]{{2,4}})(?:邀请|宣称|宣布|声称|表示|说道|笑道|告诉|提议|建议|要求|警告|哀求)')


def extract_potential_npc_names(content: str):
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
    text = strip_html(content)
    pattern = re.compile(rf'([{CJK}]+)[（(](?:\s)?([A-Za-z][A-Za-z\s\'\-\.]+?)[）)]')
    return pattern.findall(text)


def audit_file(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    page_contents = []
    for p in data.get('pages', []):
        page_name = p.get('name', '')
        content = p.get('text', {}).get('content', '') or ''
        page_contents.append((page_name, content))

    full_text = "\n".join(c for _, c in page_contents)
    full_text_stripped = strip_html(full_text)

    # Build set of glossed name fragments
    pairs = extract_bilingual_pairs(full_text)
    glossed_fragments = set()
    glossed_full = []
    for zh, en in pairs:
        glossed_full.append((zh, en))
        for n in range(2, 8):
            if len(zh) >= n:
                glossed_fragments.add(zh[-n:])
        glossed_fragments.add(zh)

    # Find NPC candidates
    npc_candidates = extract_potential_npc_names(full_text)

    missing = []
    for name, cnt in sorted(npc_candidates.items(), key=lambda x: -x[1]):
        # Filter out obvious non-names
        if name in WAVE7_NAMES:
            continue
        if name in NON_NPC:
            continue
        if name in PHRASE_FRAGMENTS:
            continue
        # Strip title from name
        stripped = name
        skip = False
        for t in ["大将军", "大师", "大人", "夫人", "女士", "勋爵", "爵士",
                  "女皇", "皇后", "陛下", "殿下", "公主", "王子",
                  "教主", "祭司", "主教", "院长", "校长", "掌柜",
                  "队长", "首领", "长老", "将军", "上将", "中将", "少将", "大将", "元帅"]:
            if stripped.endswith(t):
                stripped = stripped[:-len(t)]
                break
        if stripped in WAVE7_NAMES or stripped in NON_NPC:
            continue
        if len(stripped) < 2:
            continue
        # Glossed check
        is_glossed = False
        for frag in [name, stripped]:
            if frag in glossed_fragments:
                is_glossed = True
                break
        # Also check substring: name appears as suffix of a glossed entry
        if not is_glossed:
            for gf in glossed_fragments:
                if (name in gf or stripped in gf) and len(name) >= 2:
                    is_glossed = True
                    break
        if is_glossed:
            continue

        # Find first page + context
        first_page = None
        context = None
        for i, (pname, content) in enumerate(page_contents):
            ct = strip_html(content)
            if name in ct:
                first_page = pname
                context = get_context(ct, name, length=80)
                break
        missing.append({
            'name': name,
            'count': cnt,
            'first_page': first_page or '',
            'context': context or '',
        })

    return {
        'file': path.name,
        'glossed_pairs_count': len(pairs),
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

    out_path = Path("C:/Users/Taka/Desktop/fvtt/FotRP/_audit_npc_bilingual_wave8_v3_report.txt")
    with open(out_path, 'w', encoding='utf-8') as out:
        out.write("=" * 80 + "\n")
        out.write("FotRP NPC Bilingual Gloss Audit — Wave 8 v3 (with context)\n")
        out.write("=" * 80 + "\n\n")

        total_missing = 0
        for r in report:
            if not r['missing']:
                continue
            out.write(f"\n## {r['file']}\n")
            out.write(f"   Bilingual pairs already in file: {r['glossed_pairs_count']}\n")
            out.write(f"   NPC candidates lacking bilingual gloss:\n")
            for m in r['missing']:
                total_missing += 1
                out.write(f"     - {m['name']} (×{m['count']})\n")
                out.write(f"       page: '{m['first_page'][:80]}'\n")
                out.write(f"       context: \"{m['context']}\"\n")

        out.write(f"\n\n## TOTAL: {total_missing} candidates across all files\n")
    print(f"Report: {out_path}")


if __name__ == '__main__':
    main()
