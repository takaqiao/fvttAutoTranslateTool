"""
COT (Claws of the Tyrant / 暴君之爪) 术语提取工具

从 COT 文件夹中的 txt/docx/pdf 文件提取中英术语对，
交叉参照 glossary.json 标记已有术语，输出冒险专属术语表。

提取模式:
  1. 中文（English）或 中文(English)  — 括号内英文
  2. 中文 ENGLISH CAPS                — 标题行全大写英文
  3. 中文·中文 English Name           — 专名并列
  4. 中英对照行                       — 同行或相邻行配对
  5. 机制术语（技能、法术、专长等）    — 从规则文本中抽取

用法: python extract_cot_glossary.py [--folder COT] [--glossary glossary.json] [--output glossary_cot.json]
"""

import argparse
import json
import re
import sys
import io
from pathlib import Path

# 修复 Windows 控制台编码
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("gbk"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# 读取文件
# ---------------------------------------------------------------------------

def read_txt(path: Path) -> str:
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            return path.read_text(encoding=enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return ""


def read_docx(path: Path) -> str:
    try:
        import docx
        doc = docx.Document(str(path))
        parts = []
        for para in doc.paragraphs:
            parts.append(para.text)
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    parts.append(cell.text)
        return "\n".join(parts)
    except Exception as e:
        print(f"  ⚠ 读取 docx 失败: {path.name} | {e}")
        return ""


def read_pdf(path: Path) -> str:
    try:
        import fitz
        text_parts = []
        with fitz.open(str(path)) as doc:
            for page in doc:
                text_parts.append(page.get_text("text") or "")
        return "\n".join(text_parts)
    except Exception as e:
        print(f"  ⚠ 读取 PDF 失败: {path.name} | {e}")
        return ""


def load_all_texts(folder: Path) -> list[tuple[str, str]]:
    """返回 [(filename, text), ...]"""
    results = []
    for p in sorted(folder.iterdir()):
        if p.is_dir():
            continue
        ext = p.suffix.lower()
        if ext == ".txt":
            results.append((p.name, read_txt(p)))
        elif ext == ".docx":
            results.append((p.name, read_docx(p)))
        elif ext == ".pdf":
            results.append((p.name, read_pdf(p)))
    return results

# ---------------------------------------------------------------------------
# 正则提取
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _contains_zh(s: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", s or ""))


def _contains_en(s: str) -> bool:
    return bool(re.search(r"[A-Za-z]{2,}", s or ""))


def _clean_zh(s: str) -> str:
    s = _norm(s)
    # 去掉尾部标点
    s = re.sub(r"[，。、；：！？,.:;!?\s]+$", "", s)
    s = re.sub(r"^[，。、；：！？,.:;!?\s]+", "", s)
    return s.strip()


def _clean_en(s: str) -> str:
    s = _norm(s)
    s = re.sub(r"[，。、；：！？\s]+$", "", s)
    s = re.sub(r"^[，。、；：！？\s]+", "", s)
    return s.strip()


# 分类标签 — 用于输出时标注术语类型
CATEGORY_PATTERNS = {
    "creature": re.compile(r"生物\s*\d+|creature", re.I),
    "spell": re.compile(r"法术|戏法|spell|cantrip", re.I),
    "feat": re.compile(r"专长\s*\d+|feat", re.I),
    "artifact": re.compile(r"神器|artifact", re.I),
    "archetype": re.compile(r"变体|archetype", re.I),
    "skill": re.compile(r"技能|skill", re.I),
    "action": re.compile(r"动作|行动|action|activity", re.I),
    "item": re.compile(r"物品|道具|item|equipment", re.I),
    "location": re.compile(r"城|要塞|村|镇|堡|fort|city|town|village", re.I),
    "npc": re.compile(r"NPC|生物\s*\d+", re.I),
}


def _is_sentence(zh: str) -> bool:
    """判断中文文本是否像句子而非术语"""
    # 包含明显的句子结构词
    sentence_markers = re.compile(
        r"可以|能够|进行|使用|获得|承受|必须|如果|若是|持续|造成|对抗|"
        r"需要|试图|触发|处于|具有|你的|你会|你是|它们|该生物|"
        r"可为|也能|即便|就会|还会|不会|无法|应该|能让|可能|"
        r"一个|一次|一种|并且|或者|以及|因此|然而|尽管|"
        r"在.*时|在.*中|对.*的|为.*的|从.*中|"
        r"不采取|不足以|检定|成功|通过|已经|仍然|如今|"
        r"命令|冒险者|这些|那些|曾经|每个|装着|图案|效果如同|尝试征服"
    )
    if sentence_markers.search(zh):
        return True
    # 中文字符数超过12个基本是句子
    zh_chars = len(re.findall(r"[\u4e00-\u9fff]", zh))
    if zh_chars > 12:
        return True
    return False


# 全局切断模式 — 用于从提取的中文文本中去除句子前缀
_CUT_CHARS_HARD = set("的了着被把将让若当该此则")
_CUT_CHARS_SOFT = set("在于从与和或而也就是为以如同对向给跟比用到由")
_CUT_VERBS = re.compile(
    r"(?:施放|使用|获得|具有|成为|陷入|选择|作出|来到|称为|前往|视为|如同|"
    r"名为|身为|知晓|采取|遭受|首次|尝试征服|尝试|征服|命令|说的|试图|"
    r"战马|效果如同|效果|法术来施放|法术来|术来|他的|她的|它的|你的|我的|此处|这里|该|"
    r"指挥官|骑士团|牧师|官员|副官|手下|受到|今则由|"
    r"后者|前者|著名|祭司|属于|名叫|一只|骸骨属于|骸骨|大家庭|两具|愤恨)"
)


def _trim_zh_context(zh: str) -> str:
    """从提取的中文文本中去除句子/上下文前缀，保留术语核心"""
    # 迭代切断（最多3轮，防止无限循环）
    for _ in range(3):
        prev = zh
        # 用动词模式切断
        vm = _CUT_VERBS.search(zh)
        if vm:
            rest = zh[vm.end():]
            if rest and _contains_zh(rest) and len(rest) >= 2:
                zh = rest
        # 硬切
        best_cut = 0
        for i, ch in enumerate(zh):
            if ch in _CUT_CHARS_HARD and i < len(zh) - 1:
                best_cut = i + 1
        if best_cut > 0:
            zh = zh[best_cut:]
        # 软切开头
        while zh and zh[0] in _CUT_CHARS_SOFT and len(zh) > 2:
            zh = zh[1:]
        # 如果没变化，退出
        if zh == prev:
            break
    # "次X" → 只有 "次级" 是有效术语前缀
    if zh and zh[0] == "次" and (len(zh) < 2 or zh[1] != "级"):
        zh = zh[1:]
    return zh


def extract_pairs(text: str) -> list[dict]:
    """从文本中提取所有中英术语对"""
    pairs = []
    seen = set()

    def _add(zh: str, en: str, pattern_name: str, context: str = ""):
        zh = _clean_zh(zh)
        en = _clean_en(en)
        if not zh or not en:
            return
        if not _contains_zh(zh) or not _contains_en(en):
            return
        if len(zh) > 40 or len(en) > 100:
            return
        if len(en) < 2 or len(zh) < 2:
            return
        # 过滤句子：中文部分不应该是句子
        if _is_sentence(zh):
            return
        # 过滤纯数字/标点的英文
        if not re.search(r"[A-Za-z]{2,}", en):
            return
        # 纯大写缩写2字母（如 AP, DC）—— 通常不是独立术语
        if re.match(r"^[A-Z]{1,2}$", en):
            return
        # 英文不应包含明显的句子结构
        if re.search(r"\b(the|is|are|was|were|has|have|this|that|these|those|with|from|into|onto|also|then|each|every|must|should|would|could|while|during|after|before|between|against|through)\b", en, re.I):
            return
        key = (en.lower(), zh)
        if key in seen:
            return
        seen.add(key)
        pairs.append({
            "english": en,
            "chinese": zh,
            "pattern": pattern_name,
            "context": context[:120] if context else "",
        })

    # ── Pattern A: 行首标题格式 中文名（含·）（English）　　　神器/专长/生物 XX ──
    # 例: "阿拉兹尼的心血石（Heart Bloodstones of Arazni）　　　　神器 11"
    # 例: "破碎圣礼（Shattered Sacrament）　　　　专长 14"
    pat_header_typed = re.compile(
        r"^([\u4e00-\u9fff][\u4e00-\u9fff·・'''\-\s]{0,30}?)\s*[（(]\s*"
        r"([A-Za-z][A-Za-z0-9''\-\s]{1,60}?)\s*[）)]\s*"
        r"(?:　+\s*)?(?:神器|专长|生物|物品|法术|危害|背景|戏法|仪式|变体)?\s*\d*\s*$",
        re.MULTILINE
    )
    for m in pat_header_typed.finditer(text):
        zh = m.group(1).strip()
        en = m.group(2).strip()
        en = re.sub(r"\s*[;；].*$", "", en)
        _add(zh, en, "header_typed", text[max(0, m.start()-10):m.end()+10])

    # ── Pattern B: 启动——中文名（English）格式 ──
    # 例: "启动——举起罐子（Raise Jar）[单动]"
    # 例: "启动——信心倍增（Embolden）[反应]"
    pat_activate = re.compile(
        r"(?:启动|特殊|频率|触发)[——\-\s]*\s*([\u4e00-\u9fff]{2,15})\s*[（(]\s*"
        r"([A-Z][A-Za-z''\-\s]{1,50}?)\s*[）)]"
    )
    for m in pat_activate.finditer(text):
        _add(m.group(1), m.group(2), "activate", text[max(0, m.start()-10):m.end()+10])

    # ── Pattern C: 能力名 中文 English (行内，无括号) ──
    # 例: "萤火虫之忆 Firefly's Remembrance 萤火灵体现着..."
    # 例: "分散 Disperse [AA]"
    # 例: "垂死之息 Dying Breath [AA]"
    # 例: "怀疑之种 Seed of Doubt"
    pat_ability_inline = re.compile(
        r"(?:^|[\n])\s*([\u4e00-\u9fff][\u4e00-\u9fff·・'']{1,15})\s+"
        r"([A-Z][a-z]+(?:['\u2019]s)?\s+(?:[A-Z][a-z]+|[Oo]f|[Aa]nd|[Tt]he|[Ff]or|[Ii]n)(?:\s+[A-Za-z][a-z']+){0,5})"
        r"(?:\s+[\[（(]|\s+[\u4e00-\u9fff]|\s*$)"
    )
    for m in pat_ability_inline.finditer(text):
        _add(m.group(1), m.group(2), "ability_inline", text[max(0, m.start()-10):m.end()+40])

    # ── Pattern D: 单词能力名 中文 English [动作] ──
    # 例: "分散 Disperse [AA]"
    pat_ability_single = re.compile(
        r"(?:^|[\n])\s*([\u4e00-\u9fff]{2,8})\s+"
        r"([A-Z][a-z]{2,20})\s*\[",
    )
    for m in pat_ability_single.finditer(text):
        _add(m.group(1), m.group(2), "ability_single", text[max(0, m.start()-10):m.end()+10])

    # ── Pattern E: 中文术语（English）— 直接正则匹配 ──
    # E1: "XX的YY（English）" — 所有格名称（如 阿拉兹尼的血石）
    pat_e1 = re.compile(
        r"([\u4e00-\u9fff·\u30fb]{2,10}的[\u4e00-\u9fff·\u30fb]{1,10})\s*[（(]\s*"
        r"([A-Z][A-Za-z0-9''\-\s]{1,60}?)\s*(?:[;；][^）)]*)?[）)]"
    )
    for m in pat_e1.finditer(text):
        _add(m.group(1), m.group(2), "paren_possessive", text[max(0, m.start()-10):m.end()+10])

    # E2: "中文术语（English）" — 使用前向匹配 + 后处理
    # 匹配 "任意前缀 + 中文（English）" 然后剥离前缀
    pat_e2 = re.compile(
        r"([\u4e00-\u9fff·\u30fb\-]{2,16})\s*[（(]\s*"
        r"([A-Z][A-Za-z0-9''\u2019\-\s]{1,60}?)\s*(?:[;；][^）)]*)?[）)]"
    )
    # （切断逻辑使用全局 _trim_zh_context 函数）
    for m in pat_e2.finditer(text):
        zh = _trim_zh_context(m.group(1))
        if zh and _contains_zh(zh) and len(zh) >= 2:
            _add(zh, m.group(2), "paren_compact", text[max(0, m.start()-10):m.end()+10])

    # ── Pattern F: 全大写英文标题行 ──
    # 例: "萤火灵 ENGERRA  生物 9"
    # 例: "信念灵卫 GARRHOLDION"
    pat_caps_header = re.compile(
        r"^([\u4e00-\u9fff][\u4e00-\u9fff·・]{0,20})\s+"
        r"([A-Z][A-Z''\s\-]{2,50}?)\s*(?:生物\s*\d+)?\s*$",
        re.MULTILINE
    )
    for m in pat_caps_header.finditer(text):
        zh = _trim_zh_context(m.group(1).strip())
        en = m.group(2).strip()
        _add(zh, en.title(), "caps_header", text[max(0, m.start()-10):m.end()+10])

    # ── Pattern G: 中文专名 + English Name ──
    # 例: "塞尔德格·贝德利斯 Seldeg Bhedlis"
    pat_name = re.compile(
        r"^([\u4e00-\u9fff][\u4e00-\u9fff·・\-]{1,25})\s+"
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z']+){0,5})\s*$",
        re.MULTILINE
    )
    for m in pat_name.finditer(text):
        _add(m.group(1), m.group(2), "name_side", text[max(0, m.start()-10):m.end()+10])

    # ── Pattern H: 中文名（别名）ENGLISH CAPS + 生物 XX ──
    # 例: "萤火灵（萤火虫福使） ENGERRA (FIREFLY AGATHION)"
    pat_creature_full = re.compile(
        r"^([\u4e00-\u9fff][\u4e00-\u9fff·・（）\s]{0,30}?)\s+"
        r"([A-Z][A-Z''\s\-]{2,30})\s*"
        r"(?:\([A-Z][A-Z\s]+\))?\s*$",
        re.MULTILINE
    )
    for m in pat_creature_full.finditer(text):
        zh = re.sub(r"[（(].*?[）)]", "", m.group(1)).strip()
        en = m.group(2).strip()
        _add(zh, en.title(), "creature_full", text[max(0, m.start()-10):m.end()+10])

    # ── Pattern I: 额外学识/分布地点 等小节标题 ──
    pat_sidebar = re.compile(
        r"(?:额外学识|建议和规则|分布地点|战役作用)[：:]\s*([\u4e00-\u9fff][\u4e00-\u9fff\s]{1,15})\s+"
        r"([A-Z][A-Z''\s\-]{2,40})",
        re.MULTILINE
    )
    for m in pat_sidebar.finditer(text):
        _add(m.group(1).strip(), m.group(2).strip().title(), "sidebar", text[max(0, m.start()-10):m.end()+10])

    # ── Pattern J: 先决条件/许可中的术语 ──
    # 例: "先决条件　调和语者入门（Halcyon Speaker Dedication；角色指南 104页）"
    # 例: "许可　你是维鲁米斯学者（Vellumis Scholars）的成员"
    pat_prereq = re.compile(
        r"(?:先决条件|许可)[　\s]+"
        r".*?([\u4e00-\u9fff]{2,15})\s*[（(]\s*"
        r"([A-Z][A-Za-z''\-\s]{2,50}?)\s*(?:[;；][^）)]*)?[）)]",
    )
    for m in pat_prereq.finditer(text):
        _add(m.group(1), m.group(2), "prereq", text[max(0, m.start()-10):m.end()+10])

    # ── Pattern K: 地名/人名 中文·中文（English Title）──
    # 例: "奥泽姆骑士团（Knights of Ozem）"
    # 例: "死坟末土（Gravelands）"
    pat_proper = re.compile(
        r"([\u4e00-\u9fff·\u30fb\-]{2,15})\s*[（(]\s*"
        r"([A-Z][a-z]+(?:[\s\-]+(?:of|the|and|in|on|at|de|le|la|von|van|du|des))?(?:\s+[A-Z][a-z']+){0,5})\s*[）)]"
    )
    for m in pat_proper.finditer(text):
        zh = _trim_zh_context(m.group(1))
        if zh and _contains_zh(zh) and len(zh) >= 2:
            _add(zh, m.group(2), "proper_name", text[max(0, m.start()-20):m.end()+10])

    # ── Pattern L: 施放/如同 + 法术名（English）──
    # 例: "施放祝福术（Bless）", "如同毒云术（Toxic Cloud；DC 28）"
    # 只取最后的法术名词，不取前面的动词
    pat_spell_like = re.compile(
        r"(?:如同|施放|视为|类似)\s*([\u4e00-\u9fff]{2,8})\s*[（(]\s*"
        r"([A-Z][A-Za-z''\s]{1,40}?)\s*(?:[;；][^）)]*)?[）)]"
    )
    for m in pat_spell_like.finditer(text):
        zh = m.group(1)
        # 去掉前面可能粘连的非法术名字符
        # "法术解除结界" → "解除结界", "肝血石施放祝福术" → "祝福术" (但施放已被regex排除)
        _add(zh, m.group(2), "spell_like", text[max(0, m.start()-10):m.end()+10])

    # ── Pattern M: 战役作用 Campaign Role 等章节标题 ──
    pat_chapter = re.compile(
        r"^([\u4e00-\u9fff]{2,10})\s+"
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4})\s*$",
        re.MULTILINE
    )
    for m in pat_chapter.finditer(text):
        _add(m.group(1), m.group(2), "chapter_title", text[max(0, m.start()-10):m.end()+10])

    # ── Pattern N: 内在法术列表中的法术 ──
    # 例: "5环 唤起魂灵；3环 信念崩塌，灼热圣光；1环 衰弱术，灌注命能"
    # 这些在文中通常没有英文对照，跳过

    # ── Pattern O: 冒险章节名 ──
    # 例: "死坟末土的幸存者（Gravelands Survivors）"
    # 已被 pat_paren_smart 覆盖

    return pairs


def guess_category(pair: dict, context_text: str) -> str:
    """猜测术语类别"""
    combined = pair.get("context", "") + " " + pair.get("chinese", "")
    for cat, pat in CATEGORY_PATTERNS.items():
        if pat.search(combined):
            return cat
    return "term"

# ---------------------------------------------------------------------------
# 交叉参照
# ---------------------------------------------------------------------------

def cross_reference(pairs: list[dict], glossary: dict) -> list[dict]:
    """标记每个术语对是否已在 glossary.json 中"""
    lower_glossary = {}
    for en, zh in glossary.items():
        lower_glossary[en.lower().strip()] = {"en": en, "zh": zh}

    for pair in pairs:
        en_lower = pair["english"].lower().strip()
        if en_lower in lower_glossary:
            existing = lower_glossary[en_lower]
            pair["in_glossary"] = True
            pair["glossary_zh"] = existing["zh"] if isinstance(existing["zh"], str) else ", ".join(existing["zh"])
            if isinstance(existing["zh"], str) and existing["zh"] != pair["chinese"]:
                pair["zh_differs"] = True
            else:
                pair["zh_differs"] = False
        else:
            pair["in_glossary"] = False
            pair["glossary_zh"] = ""
            pair["zh_differs"] = False

    return pairs

# ---------------------------------------------------------------------------
# 去重 & 合并
# ---------------------------------------------------------------------------

def merge_pairs(all_pairs: list[dict]) -> list[dict]:
    """合并重复术语，保留出现次数最多的中文翻译"""
    bucket: dict[str, dict] = {}

    for pair in all_pairs:
        key = pair["english"].lower().strip()
        if key not in bucket:
            bucket[key] = {
                "english": pair["english"],
                "zh_variants": {},
                "patterns": set(),
                "contexts": [],
                "in_glossary": pair.get("in_glossary", False),
                "glossary_zh": pair.get("glossary_zh", ""),
                "zh_differs": pair.get("zh_differs", False),
            }
        entry = bucket[key]
        zh = pair["chinese"]
        entry["zh_variants"][zh] = entry["zh_variants"].get(zh, 0) + 1
        entry["patterns"].add(pair["pattern"])
        if pair.get("context") and len(entry["contexts"]) < 3:
            entry["contexts"].append(pair["context"])
        # 保留更完整的英文拼写
        if len(pair["english"]) > len(entry["english"]):
            entry["english"] = pair["english"]

    results = []
    for key, entry in sorted(bucket.items()):
        sorted_variants = sorted(entry["zh_variants"].items(), key=lambda x: x[1], reverse=True)
        preferred_zh = sorted_variants[0][0]
        results.append({
            "english": entry["english"],
            "chinese": preferred_zh,
            "zh_variants": {k: v for k, v in sorted_variants} if len(sorted_variants) > 1 else {},
            "occurrences": sum(v for v in entry["zh_variants"].values()),
            "patterns": sorted(entry["patterns"]),
            "in_glossary": entry["in_glossary"],
            "glossary_zh": entry["glossary_zh"],
            "zh_differs": entry["zh_differs"],
        })

    results.sort(key=lambda x: (-x["occurrences"], x["english"].lower()))
    return results

# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------

def build_output(merged: list[dict], full_text: str) -> dict:
    """构建输出数据"""
    # 简洁 glossary dict
    glossary_dict = {}
    for item in sorted(merged, key=lambda x: x["english"].lower()):
        glossary_dict[item["english"]] = item["chinese"]

    # 分类统计
    new_terms = [x for x in merged if not x["in_glossary"]]
    existing_terms = [x for x in merged if x["in_glossary"]]
    differing = [x for x in merged if x["zh_differs"]]

    # 分类
    categorized = {}
    for item in merged:
        cat = guess_category(item, full_text)
        item["category"] = cat
        categorized.setdefault(cat, []).append(item)

    payload = {
        "meta": {
            "adventure": "Claws of the Tyrant / 暴君之爪",
            "total_terms": len(merged),
            "new_terms": len(new_terms),
            "existing_in_glossary": len(existing_terms),
            "translation_differs": len(differing),
            "categories": {cat: len(items) for cat, items in sorted(categorized.items())},
        },
        "glossary": glossary_dict,
        "terms_detail": merged,
        "new_terms_only": {x["english"]: x["chinese"] for x in new_terms},
        "differing_translations": [
            {
                "english": x["english"],
                "cot_zh": x["chinese"],
                "glossary_zh": x["glossary_zh"],
            }
            for x in differing
        ],
    }
    return payload


def main():
    parser = argparse.ArgumentParser(description="COT 冒险术语提取工具")
    parser.add_argument("--folder", default="COT", help="COT 文件夹路径")
    parser.add_argument("--glossary", default="glossary.json", help="参照术语表路径")
    parser.add_argument("--output", default="glossary_cot.json", help="输出术语表路径")
    parser.add_argument("--output-detail", default="glossary_cot_detail.json", help="输出详细报告路径")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        print(f"❌ 文件夹不存在: {folder}")
        return

    # 加载参照术语表
    glossary_path = Path(args.glossary)
    glossary = {}
    if glossary_path.exists():
        try:
            glossary = json.loads(glossary_path.read_text(encoding="utf-8-sig"))
            if not isinstance(glossary, dict):
                glossary = {}
            print(f"已加载参照术语表: {glossary_path} ({len(glossary)} 条)")
        except Exception as e:
            print(f"⚠ 读取术语表失败: {e}")
    else:
        print(f"⚠ 未找到参照术语表: {glossary_path}")

    # 读取所有文件
    print(f"\n扫描文件夹: {folder}")
    file_texts = load_all_texts(folder)
    if not file_texts:
        print("❌ 未找到可处理的文件")
        return

    full_text = ""
    all_pairs = []
    for filename, text in file_texts:
        if not text.strip():
            print(f"  跳过空文件: {filename}")
            continue
        print(f"  处理: {filename} ({len(text)} 字符)")
        full_text += "\n" + text
        pairs = extract_pairs(text)
        print(f"    提取到 {len(pairs)} 个术语对")
        all_pairs.extend(pairs)

    if not all_pairs:
        print("❌ 未提取到任何术语对")
        return

    print(f"\n总计提取: {len(all_pairs)} 个原始术语对")

    # 交叉参照
    all_pairs = cross_reference(all_pairs, glossary)

    # 合并去重
    merged = merge_pairs(all_pairs)
    print(f"去重合并后: {len(merged)} 个唯一术语")

    # 构建输出
    payload = build_output(merged, full_text)

    # 写入简洁术语表
    output_path = Path(args.output)
    output_path.write_text(
        json.dumps(payload["glossary"], ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"\n✅ 术语表已输出: {output_path} ({len(payload['glossary'])} 条)")

    # 写入详细报告
    detail_path = Path(args.output_detail)
    detail_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"✅ 详细报告已输出: {detail_path}")

    # 打印摘要
    meta = payload["meta"]
    print(f"\n{'='*50}")
    print(f"冒险: {meta['adventure']}")
    print(f"总术语数: {meta['total_terms']}")
    print(f"  新术语 (不在 glossary.json 中): {meta['new_terms']}")
    print(f"  已有术语: {meta['existing_in_glossary']}")
    print(f"  译名不同: {meta['translation_differs']}")
    print(f"分类统计:")
    for cat, count in sorted(meta["categories"].items()):
        print(f"  {cat}: {count}")
    print(f"{'='*50}")

    # 打印一些新术语示例
    new_terms = payload.get("new_terms_only", {})
    if new_terms:
        print(f"\n新术语示例 (前 30):")
        for i, (en, zh) in enumerate(list(new_terms.items())[:30]):
            print(f"  {en}: {zh}")

    # 打印译名差异
    diffs = payload.get("differing_translations", [])
    if diffs:
        print(f"\n译名差异 (前 20):")
        for d in diffs[:20]:
            print(f"  {d['english']}: COT={d['cot_zh']} vs glossary={d['glossary_zh']}")


if __name__ == "__main__":
    main()
