#!/usr/bin/env python3
"""Audit NPC bilingual gloss coverage in FotRP canonical files.

Goal: find Chinese names that appear in prose but never receive
a (English) bilingual gloss anywhere in the same file, prioritising
names that occur multiple times (i.e. plausibly NPC-grade rather than
one-off cosmetics).
"""
import json
import re
from pathlib import Path
from collections import defaultdict

# Canonical file root
NEW_DIR = Path("C:/Users/Taka/Desktop/fvtt/FotRP/需要翻译/NEW")

# Known NPCs from Wave 7 (already bilingualised) — skip these
WAVE7_NAMES = {
    "千野", "千野·董", "千野·童",
    "苏塔奴", "苏塔奴的", "苏塔奴大师",
    "白谷真野", "白谷",
    "捣达", "捣达大师",
    "蓝蝮蛇",
    "鹰虎",
    "纪幽",
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
          "女皇", "皇后", "天皇", "陛下", "殿下", "公主", "王子",
          "教主", "教士", "祭司", "主教", "大祭司", "大主教",
          "院长", "校长", "掌柜", "大掌柜",
          "队长", "队员", "首领", "首脑", "长老",
          "将军", "上将", "中将", "少将", "大将", "元帅",
          "天龙", "神", "雕匠", "巨星", "赤凰", "黑曜",
          "老衲", "圣者", "贤者", "智者", "法师", "大法师",
          "教练", "司仪", "祭主",
          "·董", "·童", ]

CJK = r"一-龥"
NAME_PATTERN_PAREN = re.compile(
    rf'([{CJK}]+)[（(](?:\s)?([A-Za-z][A-Za-z\s\'\-\.]+?)[）)]'
)


def strip_html(text: str) -> str:
    """Strip HTML tags and enricher braces, keep prose."""
    text = re.sub(r'<[^>]+>', ' ', text)
    # Strip @Actor[id]{...}, @UUID[...]{...} etc.
    text = re.sub(r'@\w+\[[^\]]+\](?:\{[^}]*\})?', ' ', text)
    # Strip @Check[...] @Damage[...] expressions
    text = re.sub(r'@\w+\[[^\]]+\]', ' ', text)
    # Strip [[/r ...]] roll expressions
    text = re.sub(r'\[\[[^\]]+\]\]', ' ', text)
    return text


def extract_bilingual_pairs(content: str):
    """Return list of (zh_name, en_name) pairs found in form 中文（English）."""
    text = strip_html(content)
    return NAME_PATTERN_PAREN.findall(text)


def find_chinese_name_candidates(content: str):
    """Find Chinese names that look like person names.

    Heuristic: a sequence of 2-6 CJK chars that:
    - Appears as a standalone token (next char not CJK)
    - Doesn't start with a common verb / generic word
    - Contains 2-6 chars total
    """
    text = strip_html(content)
    # Match 2-6 CJK chars sequences, possibly with · between syllables
    pattern = re.compile(rf'[{CJK}]+(?:·[{CJK}]+)?')
    return pattern.findall(text)


# Words that are NOT person names (places, items, common nouns, abstract)
NON_NAMES = {
    # Geographic
    "崖庭", "崖庭区", "冰牙巢", "五柱", "五柱院", "灯笼小舍", "莎琳之梳",
    "无尽市场", "赤玉庄", "财神爷大钱庄", "旭鸿湾", "天境之墙", "过峡帝国大学",
    "七龙桥", "琉璃光屋", "过峡", "过峡寺", "明凯", "明海", "天夏",
    "天舒", "义洛理", "义洛理神庙", "格拉利昂", "格拉里昂", "邦木", "邦木岛",
    "民那塔", "苏迈蛮林", "浩瀚众界", "天夏语", "通用语",
    "灰日之下", "灰日之下队", "灰日", "盛阳", "盛阳漫步", "盛阳漫步队",
    "黄金联盟", "黄金盟", "黄金", "金联", "金盟",
    "本卷", "下一卷", "上一卷", "前一卷",
    "本章", "下一章", "上一章", "前一章",
    "本节", "下一节", "上一节", "前一节",
    "本书", "下一本", "上一本", "前一本",
    "玛甘比", "玛甘比学院", "扎摩诔", "伏陀罗", "艾欧巴瑞亚",
    "奥斯里昂", "珂莱士", "阿维斯坦", "芒吉莽原", "岚阳", "因拜罗大洋",
    "阿斯莫迪斯", "阿巴达尔", "阿斯莫", "阿巴达", "义洛理",
    "扎摩诔", "天舒族", "天垠族", "天明族",
    "凤凰宫", "赤凰宫", "赤凰之翼", "赤凰武道会", "赤凰格斗扇", "赤凰斗扇",
    "赤凰格斗", "赤凰复生", "凤凰挑战", "赤凰挑战",
    "天境", "天境之墙",
    "邦木岛", "邦木", "蛮林", "苏迈蛮林",
    "无境", "蜂鸟", "无境·蜂鸟",  # this IS a name actually
    # Common nouns
    "玩家", "玩家们", "玩家角色", "角色", "角色们", "队伍", "队员",
    "对手", "敌人", "盟友", "队员", "首领", "教练", "选手", "战士",
    "战团", "组织", "联盟", "公会", "工会", "门派", "派系",
    "门徒", "门人", "学徒", "弟子", "学生", "学员",
    "导师", "老师", "师父", "师傅", "教师",
    "战士", "法师", "牧师", "诗人", "盗贼", "游侠", "祭司",
    "修士", "武僧", "侠客", "战僧", "刺客", "决斗者",
    "怪物", "野兽", "异兽", "魔兽", "妖怪", "鬼怪", "鬼魂", "亡灵",
    "神祇", "神明", "神灵", "神圣", "神器",
    "动物", "植物", "生物", "造物", "造物者",
    "队长", "副队长", "首领", "副手", "头目",
    "公民", "市民", "村民", "镇民", "国民", "百姓",
    "商人", "工匠", "学者", "贵族", "平民", "农民", "渔民", "猎人",
    "卫兵", "守卫", "护卫", "侍卫", "门卫", "巡逻",
    "客人", "宾客", "游客", "旅客", "过客",
    "群众", "众人", "他们", "她们", "它们", "我们", "你们", "诸位",
    "彼此", "对方", "本人", "本身", "本场",
    "时间", "空间", "位面", "次元", "维度",
    "时代", "时期", "时辰", "时光",
    "故事", "传说", "传奇", "历史", "记录", "纪录", "回忆",
    "战斗", "战役", "战争", "决斗", "搏斗", "格斗", "比赛", "竞赛",
    "锦标赛", "武道会", "巡回赛", "争霸赛",
    "总决赛", "决赛", "半决赛", "四分之一决赛",
    "事件", "情节", "状况", "情况", "局势",
    "魔法", "法术", "咒语", "祷文", "圣咒",
    "武器", "护甲", "盾牌", "工具", "道具", "装备",
    "宝物", "财宝", "珍宝", "圣物", "神器",
    "信使", "信徒", "信众", "教徒", "教众",
    "凡人", "凡夫", "凡躯",
    "刺客", "杀手", "佣兵", "雇佣兵",
    "巫师", "女巫", "巫女", "巫祝",
    "村民", "村人", "村妇",
    "天空", "海洋", "陆地", "土地",
    "城市", "都市", "城邦", "城镇", "村庄", "村落",
    # Pronouns / common stuff
    "他", "她", "它", "你", "我",
    "这", "那", "哪", "什", "怎", "为",
    "其他", "其余", "其它", "另外", "另一", "余下",
    # Time, sequence
    "如今", "今日", "今天", "昨日", "昨天", "明日", "明天",
    "如此", "因此", "所以", "因为", "由于",
    "首先", "其次", "再者", "此外", "另外",
    # Abstract concepts
    "和平", "战争", "正义", "邪恶", "善良", "残忍",
    "勇气", "胆怯", "忠诚", "背叛",
    "智慧", "愚蠢", "聪明", "迟钝",
    "力量", "弱小", "强大", "渺小",
    # Race / class names
    "人类", "精灵", "矮人", "半身人", "侏儒", "兽人", "半兽人",
    "妖精", "仙灵", "天使", "魔鬼", "恶魔",
    "龙类", "龙人", "龙兽",
    "天狗", "鬼神", "妖怪", "恶鬼", "饿鬼", "饿狼",
    "无境", "蛇龙",
    # Misc
    "天罚", "天命", "天意", "天道", "天理", "天罗",
    "魔王", "邪王", "妖王", "鬼王", "鬼帝",
    "皇帝", "皇后", "皇族", "皇室", "王族", "王室",
    "封号", "称号", "外号", "绰号", "名号",
    # PDF artifacts
    "图标", "图像", "图片", "图表", "插图",
    "本卷", "本页", "本章", "本节",
    "页码", "页面",
    # Sentence connectives that look like names
    "可是", "但是", "然而", "不过", "只是",
    "因为", "由于", "所以", "因此",
    # Numerals
    "一", "二", "三", "四", "五", "六", "七", "八", "九", "十",
    "百", "千", "万", "亿",
    "第一", "第二", "第三", "第四", "第五", "第六", "第七",
}


def is_likely_name(zh: str) -> bool:
    """Return True if zh looks like a person name (not a place/concept)."""
    if len(zh) < 2 or len(zh) > 8:
        return False
    if zh in NON_NAMES:
        return False
    if zh in WAVE7_NAMES:
        return False
    # Strip trailing title and re-check
    stripped = zh
    for t in TITLES:
        if stripped.endswith(t):
            stripped = stripped[:-len(t)]
            break
    if stripped in WAVE7_NAMES or stripped in NON_NAMES:
        return False
    if len(stripped) < 2:
        return False
    return True


def audit_file(path: Path):
    """Return audit data for one file."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Concatenate all prose content
    all_content = []
    page_contents = []
    for p in data.get('pages', []):
        page_name = p.get('name', '')
        content = p.get('text', {}).get('content', '') or ''
        page_contents.append((page_name, content))
        all_content.append(content)

    full_text = "\n".join(all_content)

    # 1. Extract all bilingual pairs already in file
    pairs = extract_bilingual_pairs(full_text)
    glossed_zh = set()
    for zh, en in pairs:
        # Take last 4 chars before paren as potential name root
        # Actually use full zh part — the regex captures whatever CJK is before (
        # but it may include verb/etc. Strip to last few chars.
        # Look at the last 5 chars to extract name
        candidates = [zh, zh[-2:], zh[-3:], zh[-4:], zh[-5:], zh[-6:]]
        for c in candidates:
            if 2 <= len(c) <= 6:
                glossed_zh.add(c)

    # 2. Find all CJK name candidates in prose
    text_only = strip_html(full_text)
    candidates = find_chinese_name_candidates(text_only)

    # Count occurrences
    name_counts = defaultdict(int)
    for c in candidates:
        if is_likely_name(c):
            name_counts[c] += 1

    # Filter: only report names that are NOT glossed and appear >= 2 times
    missing = []
    for name, cnt in sorted(name_counts.items(), key=lambda x: -x[1]):
        if cnt < 2:
            continue
        # Check if any version of this name has been glossed
        is_glossed = False
        for c in [name, name[-2:], name[-3:], name[-4:]]:
            if c in glossed_zh:
                is_glossed = True
                break
        if is_glossed:
            continue
        # Final check: is this name probably a name?
        # Find first page where it appears
        first_page = None
        first_page_idx = None
        for i, (pname, content) in enumerate(page_contents):
            content_text = strip_html(content)
            if name in content_text:
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
        'glossed_count': len(pairs),
        'glossed_examples': pairs[:10],
        'missing': missing,
    }


def main():
    # All canonical journal files (3 books × 5 files each + bestiary + addons + maps)
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

    # Write findings to file
    out_path = Path("C:/Users/Taka/Desktop/fvtt/FotRP/_audit_npc_bilingual_wave8_report.txt")
    with open(out_path, 'w', encoding='utf-8') as out:
        out.write("=" * 80 + "\n")
        out.write("FotRP NPC Bilingual Gloss Audit — Wave 8\n")
        out.write("=" * 80 + "\n")

        total_missing = 0
        for r in report:
            if not r['missing']:
                continue
            out.write(f"\n## {r['file']}\n")
            out.write(f"   Total glossed bilingual pairs in file: {r['glossed_count']}\n")
            out.write(f"   Candidates lacking bilingual gloss (≥2 occurrences):\n")
            for m in r['missing']:
                total_missing += 1
                out.write(f"     - {m['name']} (×{m['count']})  first page: '{m['first_page'][:80]}'\n")

        out.write(f"\n\n## TOTAL: {total_missing} candidates across all files\n")
    print(f"Report written to {out_path}")


if __name__ == '__main__':
    main()
