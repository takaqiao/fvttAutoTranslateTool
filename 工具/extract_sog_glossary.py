"""
SoG (Season of Ghosts / 肆季鬼志) 综合术语提取工具

三阶段流水线:
  Phase 1 — 纯脚本: 从 adventure JSON + 中文 PDF 正则提取双语术语对
  Phase 2 — AI 双语对齐: 英文 PDF + 中文 PDF 对齐送 AI 交叉比对
  Phase 3 — AI 增量: 纯英文 PDF / 纯中文材料送 AI 提取增量术语

用法:
  python extract_sog_glossary.py [--config glossary_extract_config_sog.json] [--phase 1|2|3|all]
  python extract_sog_glossary.py --phase 1          # 仅跑脚本提取（无需 API）
  python extract_sog_glossary.py --phase all         # 跑全部（默认）
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import sys
import io
import time
from pathlib import Path
from threading import Lock
from html.parser import HTMLParser

from openai import OpenAI
from tqdm import tqdm

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import docx as python_docx
except ImportError:
    python_docx = None

# Windows 控制台编码修复
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("gbk"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

CONFIG_PATH = Path("glossary_extract_config_sog.json")

_log_lock = Lock()
_cache_lock = Lock()
_stats_lock = Lock()


# ===========================================================================
# 通用工具函数
# ===========================================================================

def _write_log(log_path: Path, msg: str):
    ts = time.strftime("%H:%M:%S", time.localtime())
    with _log_lock:
        try:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(f"[{ts}] {msg}\n")
        except Exception:
            pass


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _hash_text(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8", errors="ignore")).hexdigest()


def _contains_zh(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def _contains_en(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]{2,}", text or ""))


class _HTMLStripper(HTMLParser):
    """简单 HTML 标签剥离器"""
    def __init__(self):
        super().__init__()
        self.parts = []
    def handle_data(self, data):
        self.parts.append(data)
    def get_text(self):
        return "".join(self.parts)


def strip_html(html_text: str) -> str:
    s = _HTMLStripper()
    s.feed(html_text or "")
    return s.get_text()


def _clean_zh(s: str) -> str:
    s = _norm(s)
    s = re.sub(r"[，。、；：！？,.:;!?\s]+$", "", s)
    s = re.sub(r"^[，。、；：！？,.:;!?\s]+", "", s)
    # 去除开头的助词/结构词
    s = re.sub(r"^[的了着]", "", s)
    return s.strip()


def _clean_en(s: str) -> str:
    s = _norm(s)
    s = re.sub(r"[，。、；：！？\s]+$", "", s)
    s = re.sub(r"^[，。、；：！？\s]+", "", s)
    return s.strip()


def _is_sentence(zh: str) -> bool:
    sentence_markers = re.compile(
        r"可以|能够|进行|使用|获得|承受|必须|如果|若是|持续|造成|对抗|"
        r"需要|试图|触发|处于|具有|你的|你会|你是|它们|该生物|"
        r"可为|也能|即便|就会|还会|不会|无法|应该|能让|可能|"
        r"一个|一次|一种|并且|或者|以及|因此|然而|尽管|"
        r"包括|字均|望为|但是|因为|所以|然后|检定以|"
        r"在.*时|在.*中|对.*的|为.*的|从.*中"
    )
    if sentence_markers.search(zh):
        return True
    zh_chars = len(re.findall(r"[\u4e00-\u9fff]", zh))
    if zh_chars > 14:
        return True
    return False


_CUT_VERBS = re.compile(
    r"(?:施放|使用|获得|具有|成为|陷入|选择|来到|称为|前往|视为|如同|"
    r"名为|身为|知晓|采取|遭受|尝试|命令|试图|效果如同|效果|"
    r"他的|她的|它的|你的|我的|该|受到|属于|名叫)"
)


def _trim_zh_context(zh: str) -> str:
    for _ in range(3):
        prev = zh
        vm = _CUT_VERBS.search(zh)
        if vm:
            rest = zh[vm.end():]
            if rest and _contains_zh(rest) and len(rest) >= 2:
                zh = rest
        if zh == prev:
            break
    return zh


# ===========================================================================
# 文件读取
# ===========================================================================

def read_pdf_pages(pdf_path: Path, page_start: int = 1, page_end: int | None = None) -> list[tuple[int, str]]:
    if fitz is None:
        raise RuntimeError("需要 PyMuPDF: pip install pymupdf")
    pages = []
    with fitz.open(str(pdf_path)) as doc:
        total = len(doc)
        start = max(1, page_start)
        end = total if page_end is None else min(total, page_end)
        for pn in range(start, end + 1):
            text = doc[pn - 1].get_text("text") or ""
            text = text.replace("\x00", "")
            pages.append((pn, text))
    return pages


def read_pdf_text(pdf_path: Path) -> str:
    if fitz is None:
        raise RuntimeError("需要 PyMuPDF: pip install pymupdf")
    text_parts = []
    with fitz.open(str(pdf_path)) as doc:
        for page in doc:
            text_parts.append(page.get_text("text") or "")
    return "\n".join(text_parts)


def read_txt(path: Path) -> str:
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            return path.read_text(encoding=enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return ""


def read_docx(path: Path) -> str:
    if python_docx is None:
        print(f"  ⚠ 需要 python-docx: pip install python-docx")
        return ""
    try:
        doc = python_docx.Document(str(path))
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


def read_extra_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".txt":
        return read_txt(path)
    elif ext == ".docx":
        return read_docx(path)
    elif ext == ".pdf":
        return read_pdf_text(path)
    return ""


# ===========================================================================
# Phase 1: 纯脚本提取
# ===========================================================================

def extract_from_adventure_json(json_path: Path) -> list[dict]:
    """从 FVTT adventure JSON 提取双语术语对。
    
    name 字段格式: "中文名 English Name" 或 "中文名\nEnglish Name"
    """
    if not json_path.exists():
        print(f"  ⚠ adventure JSON 不存在: {json_path}")
        return []

    data = json.loads(json_path.read_text(encoding="utf-8-sig"))
    pairs = []
    seen = set()

    def _add(zh: str, en: str, source: str):
        zh = _clean_zh(zh)
        en = _clean_en(en)
        if not zh or not en:
            return
        if not _contains_zh(zh) or not _contains_en(en):
            return
        if len(zh) > 40 or len(en) > 120:
            return
        if len(en) < 2 or len(zh) < 1:
            return
        key = (en.lower(), zh)
        if key in seen:
            return
        seen.add(key)
        pairs.append({"english": en, "chinese": zh, "category": "term", "source": source})

    def _split_bilingual(val: str, source_label: str):
        """尝试从双语字段分离中英文。"""
        if not val or not isinstance(val, str):
            return
        val = val.strip()
        if not _contains_zh(val) or not _contains_en(val):
            return

        # 模式1: "中文\n英文"
        if "\n" in val:
            parts = val.split("\n", 1)
            zh_part = parts[0].strip()
            en_part = parts[1].strip()
            if _contains_zh(zh_part) and _contains_en(en_part) and not _contains_zh(en_part):
                _add(zh_part, en_part, source_label)
                return
            elif _contains_en(zh_part) and _contains_zh(en_part) and not _contains_en(en_part):
                _add(en_part, zh_part, source_label)
                return

        # 模式2: "中文 English Name" — 在第一个拉丁字符大写之前分割
        # 典型: "柳岸镇 Willowshore", "D3. 危险的小路 D3. Treacherous Trail"
        m = re.match(
            r"^(?:(?:[A-Z]\d+\.\s*)?([\u4e00-\u9fff][\u4e00-\u9fff·・\-\s]*?))\s+"
            r"(?:[A-Z]\d+\.\s*)?([A-Z][A-Za-z0-9''\-\s,]+)$",
            val
        )
        if m:
            zh_part = m.group(1).strip()
            en_part = m.group(2).strip()
            if _contains_zh(zh_part) and not _is_sentence(zh_part):
                _add(zh_part, en_part, source_label)
                return

        # 模式3: 更宽松的匹配 — 找到中英文的分界点
        # "觉醒树 Awakened Tree"  
        splits = re.split(r'(?<=[\u4e00-\u9fff·・])\s+(?=[A-Z])', val, maxsplit=1)
        if len(splits) == 2:
            zh_part = splits[0].strip()
            en_part = splits[1].strip()
            if _contains_zh(zh_part) and _contains_en(en_part) and not _is_sentence(zh_part):
                _add(zh_part, en_part, source_label)
                return

    def _walk(obj, path: str = ""):
        """递归遍历 JSON 结构，提取所有 name 字段"""
        if isinstance(obj, dict):
            # 提取 name 字段的双语对
            name_val = obj.get("name")
            if isinstance(name_val, str) and name_val.strip():
                _split_bilingual(name_val, f"json:{path}/name")

            # 提取 description 中的括号术语
            desc = obj.get("description", "")
            if isinstance(desc, str) and _contains_zh(desc) and _contains_en(desc):
                plain = strip_html(desc)
                _extract_regex_pairs_desc(plain, pairs, seen, f"json:{path}/desc")

            # 递归子对象
            for key, val in obj.items():
                if key in ("description",):  # 已处理
                    continue
                if isinstance(val, str) and key in ("name", "label"):
                    _split_bilingual(val, f"json:{path}/{key}")
                elif isinstance(val, (dict, list)):
                    _walk(val, f"{path}/{key}")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                _walk(item, f"{path}[{i}]")

    _walk(data)
    return pairs


def _extract_regex_pairs_desc(text: str, pairs: list, seen: set, source: str):
    """从描述文本中用正则提取括号术语对。"""

    def _add(zh: str, en: str, pattern_name: str):
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
        if _is_sentence(zh):
            return
        if not re.search(r"[A-Za-z]{2,}", en):
            return
        # 过滤句子结构英文
        if re.search(r"\b(the|is|are|was|were|has|have|this|that|these|those|with|from|into|while|during|after|before|between|against|through)\b", en, re.I):
            return
        key = (en.lower(), zh)
        if key in seen:
            return
        seen.add(key)
        pairs.append({"english": en, "chinese": zh, "category": "term", "source": f"{source}:{pattern_name}"})

    # 中文（English）
    pat_paren = re.compile(
        r"([\u4e00-\u9fff·\u30fb\-]{2,16})\s*[（(]\s*"
        r"([A-Z][A-Za-z0-9''\u2019\-\s]{1,60}?)\s*(?:[;；][^）)]*)?[）)]"
    )
    for m in pat_paren.finditer(text):
        zh = _trim_zh_context(m.group(1))
        if zh and _contains_zh(zh) and len(zh) >= 2:
            _add(zh, m.group(2), "paren")

    # 中文 English (行内标题)
    pat_inline = re.compile(
        r"(?:^|[\n])\s*([\u4e00-\u9fff][\u4e00-\u9fff·・'']{1,15})\s+"
        r"([A-Z][a-z]+(?:['\u2019]s)?\s+(?:[A-Z][a-z]+|[Oo]f|[Aa]nd|[Tt]he|[Ff]or|[Ii]n)(?:\s+[A-Za-z][a-z']+){0,5})"
        r"(?:\s+[\[（(]|\s+[\u4e00-\u9fff]|\s*$)"
    )
    for m in pat_inline.finditer(text):
        _add(m.group(1), m.group(2), "inline")


def extract_regex_from_pdf(pdf_path: Path) -> list[dict]:
    """从中文 PDF 正则提取术语（不需要 AI）"""
    text = read_extra_file(pdf_path)
    if not text:
        return []
    pairs = []
    seen = set()
    _extract_regex_pairs_desc(text, pairs, seen, f"pdf:{pdf_path.name}")

    # 更多 PDF 特有模式

    def _add(zh: str, en: str, pattern_name: str):
        zh = _clean_zh(zh)
        en = _clean_en(en)
        if not zh or not en or not _contains_zh(zh) or not _contains_en(en):
            return
        if len(zh) > 40 or len(en) > 100 or len(en) < 2 or len(zh) < 2:
            return
        if _is_sentence(zh):
            return
        key = (en.lower(), zh)
        if key in seen:
            return
        seen.add(key)
        pairs.append({"english": en, "chinese": zh, "category": "term", "source": f"pdf:{pdf_path.name}:{pattern_name}"})

    # 全大写英文标题: "中文 ENGLISH CAPS"
    pat_caps = re.compile(
        r"^([\u4e00-\u9fff][\u4e00-\u9fff·・]{0,20})\s+"
        r"([A-Z][A-Z''\s\-]{2,50}?)\s*(?:生物\s*\d+)?\s*$",
        re.MULTILINE
    )
    for m in pat_caps.finditer(text):
        zh = _trim_zh_context(m.group(1).strip())
        en = m.group(2).strip()
        _add(zh, en.title(), "caps_header")

    # 中文专名 + English Name (行内)
    pat_name = re.compile(
        r"^([\u4e00-\u9fff][\u4e00-\u9fff·・\-]{1,25})\s+"
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z']+){0,5})\s*$",
        re.MULTILINE
    )
    for m in pat_name.finditer(text):
        _add(m.group(1), m.group(2), "name_side")

    # 启动——中文名（English）
    pat_activate = re.compile(
        r"(?:启动|特殊|频率|触发)[——\-\s]*\s*([\u4e00-\u9fff]{2,15})\s*[（(]\s*"
        r"([A-Z][A-Za-z''\-\s]{1,50}?)\s*[）)]"
    )
    for m in pat_activate.finditer(text):
        _add(m.group(1), m.group(2), "activate")

    # 中文名 English [动作]
    pat_ability = re.compile(
        r"(?:^|[\n])\s*([\u4e00-\u9fff]{2,8})\s+"
        r"([A-Z][a-z]{2,20})\s*\[",
    )
    for m in pat_ability.finditer(text):
        _add(m.group(1), m.group(2), "ability_single")

    return pairs


# ===========================================================================
# Phase 2 & 3: AI 提取
# ===========================================================================

SYSTEM_PROMPT_DUAL = """\
你是 PF2E TRPG 术语提取专家。你会收到英文原文和对应的中文翻译文本。
这些文本来自《肆季鬼志》(Season of Ghosts) 冒险路径。

你的任务：
1. 从英文原文中识别所有 **专有名词、游戏术语、地名、人名、怪物名、法术名、专长名、物品名、\
技能名、动作名、特征名、变体名、组织名、章节名、建筑名、神祇名** 等术语。
2. 从中文翻译中找到这些术语的对应翻译。
3. 同时提取中文文本中以 中文（English）格式明确标注的术语对。
4. 每个术语只需输出一次，选最准确的翻译。

特别注意：
- 肆季鬼志背景设定在天夏（Tian Xia），涉及大量东亚风格命名
- 注意提取所有 NPC 姓名（往往是日本/中国风格名字的音译）
- 地名、建筑名、节日名、组织名也是重点
- 怪物/生物的名称务必提取
- 法术、专长、物品等游戏机制术语也要提取

输出要求：
- 严格 JSON 数组，每项格式：{"english":"...","chinese":"...","category":"..."}
- category 可选值：creature, spell, feat, item, action, skill, location, npc, organization, \
chapter, trait, archetype, hazard, artifact, term, deity, building, festival, ancestry, heritage
- 不要猜测或编造不存在于文本中的术语
- 英文保持原文大小写
- 中文只输出术语名，不要输出整句话
- 若无可提取术语，输出 []

示例输出：
[
  {"english":"Willowshore","chinese":"柳岸镇","category":"location"},
  {"english":"Kugaptee","chinese":"枯鸦亭","category":"npc"},
  {"english":"Season of Ghosts","chinese":"鬼季","category":"term"}
]\
"""

SYSTEM_PROMPT_ZH_ONLY = """\
你是 PF2E TRPG 术语提取专家。你会收到中文翻译文本（可能包含括号标注的英文术语）。
这些文本来自《肆季鬼志》(Season of Ghosts) 冒险路径相关材料。

你的任务：
1. 提取文本中所有以 中文（English）或 中文 English 格式出现的术语对。
2. 识别所有专有名词（人名、地名、怪物名、法术名、专长名、物品名等）。
3. 对于只有中文没有英文的专有名词，若你确信知道其英文原文（比如 PF2E 通用规则术语），也可以输出。

输出要求：
- 严格 JSON 数组，每项格式：{"english":"...","chinese":"...","category":"..."}
- category 可选值同上
- 中文只输出术语名，不要输出整句话
- 若无可提取术语，输出 []
"""

SYSTEM_PROMPT_EN_ONLY = """\
你是 PF2E TRPG 术语提取专家。你会收到英文原文页面。
这些文本来自《Season of Ghosts》(肆季鬼志) 冒险路径 AP3 或 AP4。

你的任务：
1. 从英文原文中识别所有 **专有名词**：NPC 姓名、地点名、生物名、法术名、专长名、物品名、\
组织名、神祇名、章节名、建筑名等。
2. 根据你对PF2E官方中文翻译惯例的理解，给出最合理的中文翻译。
3. 优先参考提供的参考术语表保持翻译一致。

注意：
- Season of Ghosts 背景在 Tian Xia（天夏），NPC 名字常为东亚风格
- 不要翻译你不确定的通用英文单词
- 优先提取有实际翻译价值的专有名词

输出要求：
- 严格 JSON 数组，每项格式：{"english":"...","chinese":"...","category":"...","confidence":"high|medium|low"}
- 用 confidence 标注你对翻译准确性的信心
- 若无可提取术语，输出 []
"""


class TermExtractorAI:
    def __init__(self, cfg: dict):
        api_key = cfg.get("openai_api_key") or os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("未设置 openai_api_key 且环境变量 OPENAI_API_KEY 为空")
        self.client = OpenAI(
            api_key=api_key,
            base_url=cfg.get("openai_base_url", "https://api.openai.com/v1"),
        )
        self.model = cfg.get("model", "gpt-4.1")
        self.max_retries = max(1, int(cfg.get("max_retries", 3)))

    def extract(self, batch: dict, ref_terms_text: str = "") -> list[dict]:
        en_text = batch.get("en_text", "")
        zh_text = batch.get("zh_text", "")
        mode = batch.get("mode", "dual")  # dual / zh_only / en_only

        if mode == "dual":
            instructions = SYSTEM_PROMPT_DUAL
            user_input = "[英文原文]\n" + en_text + "\n\n[中文翻译]\n" + zh_text
        elif mode == "zh_only":
            instructions = SYSTEM_PROMPT_ZH_ONLY
            user_input = "[中文文本]\n" + zh_text
        elif mode == "en_only":
            instructions = SYSTEM_PROMPT_EN_ONLY
            user_input = "[英文原文]\n" + en_text
        else:
            return []

        if ref_terms_text:
            user_input += "\n\n[参考术语表（已有翻译，优先保持一致）]\n" + ref_terms_text

        user_input += "\n\n请提取所有术语，返回 JSON 数组。"

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.responses.create(
                    model=self.model,
                    instructions=instructions,
                    input=user_input,
                )
                raw = (resp.output_text or "").strip()
                return self._parse_response(raw)
            except Exception as e:
                if attempt == self.max_retries:
                    raise RuntimeError(f"AI 提取失败: {batch.get('label', '?')} | {e}") from e
                time.sleep(attempt * 2)
        return []

    def _parse_response(self, raw: str) -> list[dict]:
        if not raw:
            return []
        fence = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", raw, re.I)
        if fence:
            text = fence.group(1)
        else:
            start = raw.find("[")
            end = raw.rfind("]")
            if start != -1 and end > start:
                text = raw[start:end + 1]
            else:
                return []
        try:
            arr = json.loads(text)
            if not isinstance(arr, list):
                return []
        except json.JSONDecodeError:
            return []

        results = []
        for item in arr:
            if not isinstance(item, dict):
                continue
            en = _norm(str(item.get("english", "")))
            zh = _norm(str(item.get("chinese", "")))
            cat = str(item.get("category", "term")).strip().lower()
            if not en or not zh or not _contains_en(en) or not _contains_zh(zh):
                continue
            if len(zh) > 40 or len(en) > 100:
                continue
            entry = {"english": en, "chinese": zh, "category": cat}
            conf = item.get("confidence")
            if conf:
                entry["confidence"] = conf
            results.append(entry)
        return results


# ===========================================================================
# 限速 & 缓存
# ===========================================================================

class RateLimiter:
    def __init__(self, rpm: int):
        self.rpm = max(1, rpm)
        self.min_interval = 60.0 / self.rpm
        self.last_request = 0.0
        self.lock = Lock()

    def wait(self):
        with self.lock:
            now = time.time()
            gap = self.last_request + self.min_interval - now
            if gap > 0:
                time.sleep(gap)
            self.last_request = time.time()


def _load_cache(path: Path, enabled: bool) -> dict:
    if not enabled or not path.exists():
        return {"batches": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and isinstance(data.get("batches"), dict):
            return data
    except Exception:
        pass
    return {"batches": {}}


def _save_cache(path: Path, cache: dict, enabled: bool):
    if not enabled:
        return
    try:
        path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


# ===========================================================================
# 批次构建
# ===========================================================================

def build_aligned_batches(
    en_pages: list[tuple[int, str]],
    zh_pages: list[tuple[int, str]],
    pages_per_batch: int,
    max_chars: int,
    page_offset: int = 0,
) -> list[dict]:
    en_dict = {pn: text for pn, text in en_pages}
    zh_dict = {pn: text for pn, text in zh_pages}
    all_en_pages = sorted(en_dict.keys())
    if not all_en_pages:
        return []

    batches = []
    i = 0
    while i < len(all_en_pages):
        batch_en_parts = []
        batch_zh_parts = []
        batch_en_pns = []
        batch_zh_pns = []
        chars = 0

        while i < len(all_en_pages):
            en_pn = all_en_pages[i]
            zh_pn = en_pn + page_offset
            en_text = en_dict.get(en_pn, "")
            zh_text = zh_dict.get(zh_pn, "")
            part_len = len(en_text) + len(zh_text) + 60

            if batch_en_pns and (len(batch_en_pns) >= pages_per_batch or chars + part_len > max_chars):
                break

            if en_text.strip():
                batch_en_parts.append(f"=== EN Page {en_pn} ===\n{en_text}")
                batch_en_pns.append(en_pn)
            if zh_text.strip():
                batch_zh_parts.append(f"=== ZH Page {zh_pn} ===\n{zh_text}")
                batch_zh_pns.append(zh_pn)
            chars += part_len
            i += 1

        if batch_en_parts or batch_zh_parts:
            batches.append({
                "en_text": "\n\n".join(batch_en_parts),
                "zh_text": "\n\n".join(batch_zh_parts),
                "mode": "dual",
                "en_pages": batch_en_pns,
                "zh_pages": batch_zh_pns,
                "label": f"EN p{batch_en_pns[0]}-{batch_en_pns[-1]}" if batch_en_pns else "extra",
            })
    return batches


def build_text_batches(texts: list[tuple[str, str]], max_chars: int, mode: str) -> list[dict]:
    """将文本文件分批。"""
    batches = []
    for filename, text in texts:
        if not text.strip():
            continue
        chunks = []
        start = 0
        while start < len(text):
            end = start + max_chars
            chunks.append(text[start:end])
            start = end
        for ci, chunk in enumerate(chunks):
            b = {
                "en_text": chunk if mode == "en_only" else "",
                "zh_text": chunk if mode in ("zh_only", "dual") else "",
                "mode": mode,
                "en_pages": [],
                "zh_pages": [],
                "label": f"{filename} part{ci+1}",
            }
            batches.append(b)
    return batches


def build_en_only_batches(
    pages: list[tuple[int, str]],
    pages_per_batch: int,
    max_chars: int,
    label_prefix: str,
) -> list[dict]:
    batches = []
    i = 0
    while i < len(pages):
        batch_parts = []
        batch_pns = []
        chars = 0

        while i < len(pages):
            pn, text = pages[i]
            part_len = len(text) + 30
            if batch_pns and (len(batch_pns) >= pages_per_batch or chars + part_len > max_chars):
                break
            if text.strip():
                batch_parts.append(f"=== Page {pn} ===\n{text}")
                batch_pns.append(pn)
            chars += part_len
            i += 1

        if batch_parts:
            batches.append({
                "en_text": "\n\n".join(batch_parts),
                "zh_text": "",
                "mode": "en_only",
                "en_pages": batch_pns,
                "zh_pages": [],
                "label": f"{label_prefix} p{batch_pns[0]}-{batch_pns[-1]}",
            })
    return batches


# ===========================================================================
# 合并 & 输出
# ===========================================================================

def merge_all_terms(all_terms: list[dict]) -> list[dict]:
    bucket: dict[str, dict] = {}
    for item in all_terms:
        key = item["english"].lower().strip()
        if key not in bucket:
            bucket[key] = {
                "english": item["english"],
                "zh_variants": {},
                "categories": {},
                "sources": set(),
                "confidence_scores": [],
            }
        entry = bucket[key]
        zh = item["chinese"]
        cat = item.get("category", "term")
        src = item.get("source", "ai")
        entry["zh_variants"][zh] = entry["zh_variants"].get(zh, 0) + 1
        entry["categories"][cat] = entry["categories"].get(cat, 0) + 1
        entry["sources"].add(src)
        if item.get("confidence"):
            entry["confidence_scores"].append(item["confidence"])
        if len(item["english"]) > len(entry["english"]):
            entry["english"] = item["english"]

    results = []
    for entry in sorted(bucket.values(), key=lambda x: x["english"].lower()):
        sorted_zh = sorted(entry["zh_variants"].items(), key=lambda x: x[1], reverse=True)
        sorted_cat = sorted(entry["categories"].items(), key=lambda x: x[1], reverse=True)
        best_zh = sorted_zh[0][0]

        # 后合并质量过滤
        if _is_bad_term(entry["english"], best_zh):
            continue

        r = {
            "english": entry["english"],
            "chinese": best_zh,
            "category": sorted_cat[0][0],
            "occurrences": sum(v for v in entry["zh_variants"].values()),
            "sources": sorted(entry["sources"]),
        }
        if len(sorted_zh) > 1:
            r["zh_variants"] = {k: v for k, v in sorted_zh}
        if entry["confidence_scores"]:
            # en_only 提取的置信度
            r["min_confidence"] = min(entry["confidence_scores"])
        results.append(r)
    return results


def _is_bad_term(en: str, zh: str) -> bool:
    """后合并质量过滤：剔除明显不是术语的条目"""
    # 中文以 的/了/着 开头（助词残留）
    if zh and zh[0] in "的了着":
        return True
    # 中文包含 "检定以"（句子碎片，如 "生存检定以追踪"）
    if "检定以" in zh:
        return True
    # 中文包含明显句子词（不应出现在术语中）
    bad_zh_patterns = re.compile(
        r"字均|望为|包括|但是|因为|所以|然后|来自|而且|不过|"
        r"的语言是|不死生物的|的运动或|灵魂将会|没有离开|当时一队|板宿是在|"
        r"你在.{1,6}技能|落入了|正在寻找|透过反常|代表柳岸|表中日期|"
        r"大成功或大失败|食物储备|当时适逢|运动或航海"
    )
    if bad_zh_patterns.search(zh):
        return True
    # 英文有拼写错误的常见模式
    if en.lower() in ("forec open",):
        return True
    return False


def cross_reference_glossary(terms: list[dict], glossary: dict) -> list[dict]:
    lower_glossary = {}
    for en, zh in glossary.items():
        lower_glossary[en.lower().strip()] = {"en": en, "zh": zh}

    for term in terms:
        key = term["english"].lower().strip()
        if key in lower_glossary:
            existing = lower_glossary[key]
            term["in_glossary"] = True
            term["glossary_zh"] = existing["zh"] if isinstance(existing["zh"], str) else ", ".join(existing["zh"])
            term["zh_differs"] = term["chinese"] != term["glossary_zh"]
        else:
            term["in_glossary"] = False
            term["glossary_zh"] = ""
            term["zh_differs"] = False
    return terms


def build_ref_terms_text(glossary: dict, text: str, max_terms: int = 60) -> str:
    if not glossary or not text:
        return ""
    text_lower = text.lower()
    relevant = []
    for en, zh in glossary.items():
        if en.lower() in text_lower:
            zh_str = zh if isinstance(zh, str) else ", ".join(zh)
            relevant.append(f"{en}: {zh_str}")
    relevant = relevant[:max_terms]
    return "\n".join(relevant) if relevant else ""


# ===========================================================================
# 配置
# ===========================================================================

def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        sys.exit(1)
    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(cfg, dict):
            raise ValueError("配置文件应为 JSON 对象")
        return cfg
    except Exception as e:
        print(f"❌ 配置文件读取失败: {e}")
        sys.exit(1)


# ===========================================================================
# 主流程
# ===========================================================================

def run_phase1(cfg: dict) -> list[dict]:
    """Phase 1: 纯脚本提取（无需 API）"""
    print("\n" + "=" * 60)
    print("Phase 1: 纯脚本提取")
    print("=" * 60)

    all_terms = []

    # 1a: 从 adventure JSON 提取
    json_path = Path(cfg.get("adventure_json", ""))
    if json_path.exists():
        print(f"\n📄 从 adventure JSON 提取: {json_path}")
        terms = extract_from_adventure_json(json_path)
        print(f"  → 提取到 {len(terms)} 个术语对")
        all_terms.extend(terms)
    else:
        print(f"  ⚠ adventure JSON 不存在: {json_path}")

    # 1b: 从中文 PDF/文件正则提取
    zh_files = cfg.get("zh_only_files", [])
    for fp in zh_files:
        p = Path(fp)
        if p.exists():
            print(f"\n📄 正则扫描: {p.name}")
            terms = extract_regex_from_pdf(p)
            print(f"  → 提取到 {len(terms)} 个术语对")
            all_terms.extend(terms)
        else:
            print(f"  ⚠ 文件不存在: {fp}")

    # 1c: 从中文 PDF（汉化组翻译）正则提取
    for pair_cfg in cfg.get("dual_pdf_pairs", []):
        zh_pdf = Path(pair_cfg.get("zh_pdf", ""))
        if zh_pdf.exists():
            print(f"\n📄 正则扫描中文 AP: {zh_pdf.name}")
            terms = extract_regex_from_pdf(zh_pdf)
            print(f"  → 提取到 {len(terms)} 个术语对")
            all_terms.extend(terms)

    print(f"\nPhase 1 总计: {len(all_terms)} 个原始术语对")
    return all_terms


def run_phase2(cfg: dict, combined_ref: dict, cache: dict, log_path: Path, stats: dict) -> list[dict]:
    """Phase 2: AI 双语对齐提取"""
    print("\n" + "=" * 60)
    print("Phase 2: AI 双语对齐提取")
    print("=" * 60)

    if fitz is None:
        print("❌ 需要 PyMuPDF: pip install pymupdf")
        return []

    pages_per_batch = int(cfg.get("pages_per_batch", 6))
    max_chars = int(cfg.get("max_chars_per_batch", 24000))
    all_batches = []

    for pair_cfg in cfg.get("dual_pdf_pairs", []):
        label = pair_cfg.get("label", "Unknown")
        en_pdf = Path(pair_cfg.get("en_pdf", ""))
        zh_pdf = Path(pair_cfg.get("zh_pdf", ""))

        if not en_pdf.exists() or not zh_pdf.exists():
            print(f"  ⚠ 跳过 {label}: PDF 不存在")
            continue

        en_start = int(pair_cfg.get("en_page_start", 1))
        en_end = pair_cfg.get("en_page_end")
        zh_start = int(pair_cfg.get("zh_page_start", 1))
        zh_end = pair_cfg.get("zh_page_end")
        page_offset = int(pair_cfg.get("page_offset", 0))

        print(f"\n📖 {label}:")
        en_pages = read_pdf_pages(en_pdf, en_start, en_end)
        print(f"  英文: {len(en_pages)} 页")
        zh_pages = read_pdf_pages(zh_pdf, zh_start, zh_end)
        print(f"  中文: {len(zh_pages)} 页")

        batches = build_aligned_batches(en_pages, zh_pages, pages_per_batch, max_chars, page_offset)
        for b in batches:
            b["label"] = f"{label} {b['label']}"
        all_batches.extend(batches)

    # 中文独立文件也加入 AI 提取
    zh_files = cfg.get("zh_only_files", [])
    zh_texts = []
    for fp in zh_files:
        p = Path(fp)
        if p.exists():
            text = read_extra_file(p)
            if text.strip():
                zh_texts.append((p.name, text))
                print(f"  中文文件: {p.name} ({len(text)} 字符)")
    zh_batches = build_text_batches(zh_texts, max_chars, "zh_only")
    all_batches.extend(zh_batches)

    if not all_batches:
        print("  无双语批次")
        return []

    print(f"\n总批次: {len(all_batches)}")
    stats["phase2_batches"] = len(all_batches)
    return _run_ai_extraction(cfg, all_batches, combined_ref, cache, log_path, stats, "phase2")


def run_phase3(cfg: dict, combined_ref: dict, cache: dict, log_path: Path, stats: dict) -> list[dict]:
    """Phase 3: AI 纯英文 PDF 提取"""
    print("\n" + "=" * 60)
    print("Phase 3: AI 纯英文 PDF 增量提取")
    print("=" * 60)

    if fitz is None:
        print("❌ 需要 PyMuPDF: pip install pymupdf")
        return []

    pages_per_batch = int(cfg.get("pages_per_batch", 6))
    max_chars = int(cfg.get("max_chars_per_batch", 24000))
    all_batches = []

    for en_cfg in cfg.get("en_only_pdfs", []):
        label = en_cfg.get("label", "Unknown")
        en_pdf = Path(en_cfg.get("pdf", ""))
        if not en_pdf.exists():
            print(f"  ⚠ 跳过 {label}: {en_pdf} 不存在")
            continue

        page_start = int(en_cfg.get("page_start", 1))
        page_end = en_cfg.get("page_end")

        print(f"\n📖 {label}: {en_pdf.name}")
        pages = read_pdf_pages(en_pdf, page_start, page_end)
        print(f"  {len(pages)} 页")

        batches = build_en_only_batches(pages, pages_per_batch, max_chars, label)
        all_batches.extend(batches)

    if not all_batches:
        print("  无纯英文批次")
        return []

    print(f"\n总批次: {len(all_batches)}")
    stats["phase3_batches"] = len(all_batches)
    return _run_ai_extraction(cfg, all_batches, combined_ref, cache, log_path, stats, "phase3")


def _run_ai_extraction(
    cfg: dict,
    batches: list[dict],
    combined_ref: dict,
    cache: dict,
    log_path: Path,
    stats: dict,
    phase_label: str,
) -> list[dict]:
    """公共 AI 批量提取逻辑"""
    extractor = TermExtractorAI(cfg)
    limiter = RateLimiter(int(cfg.get("target_rpm", 60)))
    max_workers = max(1, int(cfg.get("max_workers", 4)))
    cache_enabled = bool(cfg.get("cache_enabled", True))
    model = cfg.get("model", "gpt-4.1")

    all_terms = []
    phase_stats = {"cache_hit": 0, "ai_ok": 0, "ai_fail": 0}

    def process_batch(idx: int, batch: dict) -> list[dict]:
        label = batch["label"]
        batch_hash = _hash_text(model + "|" + batch["en_text"] + batch["zh_text"])
        cache_key = f"{phase_label}:batch:{idx}:{label}"

        with _cache_lock:
            cached = cache["batches"].get(cache_key)
            if (
                cache_enabled
                and isinstance(cached, dict)
                and cached.get("hash") == batch_hash
                and isinstance(cached.get("terms"), list)
            ):
                phase_stats["cache_hit"] += 1
                return cached["terms"]

        ref_text = build_ref_terms_text(combined_ref, batch["en_text"] + " " + batch["zh_text"], 80)

        try:
            limiter.wait()
            terms = extractor.extract(batch, ref_text)
            phase_stats["ai_ok"] += 1

            if cache_enabled:
                with _cache_lock:
                    cache["batches"][cache_key] = {"hash": batch_hash, "terms": terms}
            return terms
        except Exception as e:
            _write_log(log_path, f"ERROR {label} | {e}")
            phase_stats["ai_fail"] += 1
            return []

    print(f"\n开始 AI 术语提取 ({phase_label}, model={cfg.get('model')}, workers={max_workers})...")
    _write_log(log_path, f"START {phase_label} batches={len(batches)} model={cfg.get('model')}")

    if max_workers <= 1:
        for idx, batch in enumerate(tqdm(batches, desc=f"{phase_label} 术语提取", dynamic_ncols=True)):
            terms = process_batch(idx, batch)
            all_terms.extend(terms)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_batch, idx, batch): idx for idx, batch in enumerate(batches)}
            for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"{phase_label} 术语提取", dynamic_ncols=True):
                terms = fut.result()
                all_terms.extend(terms)

    stats[f"{phase_label}_cache_hit"] = phase_stats["cache_hit"]
    stats[f"{phase_label}_ai_ok"] = phase_stats["ai_ok"]
    stats[f"{phase_label}_ai_fail"] = phase_stats["ai_fail"]

    print(f"  {phase_label}: 缓存={phase_stats['cache_hit']} AI成功={phase_stats['ai_ok']} AI失败={phase_stats['ai_fail']}")
    print(f"  原始提取: {len(all_terms)} 个术语对")
    return all_terms


def main():
    parser = argparse.ArgumentParser(description="SoG (肆季鬼志) 综合术语提取工具")
    parser.add_argument("--config", default=str(CONFIG_PATH), help="配置文件路径")
    parser.add_argument("--phase", default="all", choices=["1", "2", "3", "all"],
                        help="执行阶段: 1=纯脚本, 2=AI双语, 3=AI纯英文, all=全部")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    log_path = Path(cfg.get("log_path", "glossary_extract_sog.log"))
    run_phases = args.phase

    print(f"🎴 SoG (肆季鬼志) 术语提取工具")
    print(f"   配置: {args.config}")
    print(f"   阶段: {run_phases}")

    # ── 加载参考术语表 ──
    ref_glossary = {}
    ref_path = cfg.get("reference_glossary", "")
    if ref_path:
        p = Path(ref_path)
        if p.exists():
            try:
                ref_glossary = json.loads(p.read_text(encoding="utf-8-sig"))
                if not isinstance(ref_glossary, dict):
                    ref_glossary = {}
                print(f"参考术语表: {p} ({len(ref_glossary)} 条)")
            except Exception as e:
                print(f"⚠ 参考术语表读取失败: {e}")

    # ── 加载已有 SoG 术语表（增量基础）──
    existing_glossary = {}
    existing_path = cfg.get("existing_glossary", "")
    if existing_path:
        p = Path(existing_path)
        if p.exists():
            try:
                existing_glossary = json.loads(p.read_text(encoding="utf-8-sig"))
                if not isinstance(existing_glossary, dict):
                    existing_glossary = {}
                print(f"增量基础术语表: {p} ({len(existing_glossary)} 条)")
            except Exception as e:
                print(f"⚠ 增量基础术语表读取失败: {e}")

    combined_ref = dict(ref_glossary)
    combined_ref.update(existing_glossary)
    print(f"合并参考: {len(combined_ref)} 条")

    # ── 缓存 ──
    cache_enabled = bool(cfg.get("cache_enabled", True))
    cache_path = Path(cfg.get("cache_path", "glossary_extract_cache_sog.json"))
    cache = _load_cache(cache_path, cache_enabled)

    stats = {}
    all_terms = []

    # ── Phase 1 ──
    if run_phases in ("1", "all"):
        phase1_terms = run_phase1(cfg)
        all_terms.extend(phase1_terms)
        stats["phase1_count"] = len(phase1_terms)

    # ── Phase 2 ──
    if run_phases in ("2", "all"):
        phase2_terms = run_phase2(cfg, combined_ref, cache, log_path, stats)
        all_terms.extend(phase2_terms)
        stats["phase2_count"] = len(phase2_terms)
        _save_cache(cache_path, cache, cache_enabled)

    # ── Phase 3 ──
    if run_phases in ("3", "all"):
        # 把 Phase 1+2 已提取的术语加入参考，让 Phase 3 保持一致
        phase12_glossary = {}
        for t in all_terms:
            lk = t["english"].lower()
            if lk not in phase12_glossary:
                phase12_glossary[t["english"]] = t["chinese"]
        augmented_ref = dict(combined_ref)
        augmented_ref.update(phase12_glossary)

        phase3_terms = run_phase3(cfg, augmented_ref, cache, log_path, stats)
        all_terms.extend(phase3_terms)
        stats["phase3_count"] = len(phase3_terms)
        _save_cache(cache_path, cache, cache_enabled)

    print(f"\n{'='*60}")
    print(f"总原始提取: {len(all_terms)} 个术语对")

    # ── 合并去重 ──
    merged = merge_all_terms(all_terms)
    print(f"去重合并: {len(merged)} 个唯一术语")

    # ── 交叉参照 ──
    if ref_glossary:
        merged = cross_reference_glossary(merged, ref_glossary)

    # ── 输出 ──
    output_json = Path(cfg.get("output_json", "glossary_sog.json"))
    output_candidates = Path(cfg.get("output_candidates_json", "glossary_sog_candidates.json"))
    output_conflicts = Path(cfg.get("output_conflicts_json", "glossary_sog_conflicts.json"))

    # 构建最终术语表（增量合并，支持一词多义）
    glossary_dict = dict(existing_glossary)
    ai_added = 0
    ai_conflicts = []
    lower_existing = {k.lower().strip(): k for k in glossary_dict}

    # 多义阈值：variant 出现次数 >= 主译名次数 * 0.25 且 >= 2 次才列入
    MULTI_RATIO = 0.25
    MULTI_MIN_OCC = 2

    for item in merged:
        en = item["english"]
        zh = item["chinese"]
        lk = en.lower().strip()

        # 构建多义值：主译名 + 符合阈值的变体
        multi_zh = [zh]
        if item.get("zh_variants"):
            main_occ = item["zh_variants"].get(zh, 1)
            threshold = max(MULTI_MIN_OCC, int(main_occ * MULTI_RATIO))
            for alt_zh, alt_occ in sorted(item["zh_variants"].items(), key=lambda x: -x[1]):
                if alt_zh == zh:
                    continue
                # 跳过明显是句子碎片的变体
                if _is_bad_term(en, alt_zh) or _is_sentence(alt_zh):
                    continue
                if alt_occ >= threshold:
                    multi_zh.append(alt_zh)

        # 最终值：单个字符串或列表
        val = multi_zh[0] if len(multi_zh) == 1 else multi_zh

        if lk in lower_existing:
            existing_key = lower_existing[lk]
            old_zh = glossary_dict[existing_key]
            if old_zh != val:
                ai_conflicts.append({"english": en, "existing_zh": old_zh, "new_zh": val,
                                     "occurrences": item.get("occurrences", 1)})
        else:
            glossary_dict[en] = val
            lower_existing[lk] = en
            ai_added += 1

    print(f"最终术语表: {len(glossary_dict)} 条 (新增 {ai_added}, 冲突 {len(ai_conflicts)})")

    # 写入简洁术语表
    sorted_glossary = dict(sorted(glossary_dict.items(), key=lambda x: x[0].lower()))
    output_json.write_text(json.dumps(sorted_glossary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 分类统计
    new_terms = [x for x in merged if not x.get("in_glossary", False)]
    existing_terms = [x for x in merged if x.get("in_glossary", False)]
    differing = [x for x in merged if x.get("zh_differs", False)]
    categories = {}
    for item in merged:
        cat = item.get("category", "term")
        categories[cat] = categories.get(cat, 0) + 1

    # 详细报告
    payload = {
        "meta": {
            "adventure": cfg.get("adventure_name", "Season of Ghosts / 肆季鬼志"),
            "total_unique_terms": len(merged),
            "new_vs_ref_glossary": len(new_terms),
            "existing_in_ref_glossary": len(existing_terms),
            "translation_differs_vs_ref": len(differing),
            "incremental_base": len(existing_glossary),
            "incremental_added": ai_added,
            "incremental_conflicts": len(ai_conflicts),
            "final_glossary_size": len(glossary_dict),
            "categories": dict(sorted(categories.items())),
            "stats": stats,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "glossary": sorted_glossary,
        "terms_detail": merged,
        "new_terms_only": {x["english"]: x["chinese"] for x in new_terms},
        "differing_vs_ref_glossary": [
            {"english": x["english"], "sog_zh": x["chinese"], "glossary_zh": x["glossary_zh"]}
            for x in differing
        ],
        "incremental_conflicts": ai_conflicts,
    }
    output_candidates.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # 冲突报告
    conflict_payload = {
        "meta": {
            "vs_ref_glossary": len(differing),
            "vs_existing_sog": len(ai_conflicts),
        },
        "vs_ref_glossary": payload["differing_vs_ref_glossary"],
        "vs_existing_sog": ai_conflicts,
    }
    output_conflicts.write_text(json.dumps(conflict_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 打印摘要 ──
    print(f"\n{'='*60}")
    print(f"冒险: {cfg.get('adventure_name', 'Season of Ghosts')}")
    print(f"提取唯一术语: {len(merged)}")
    print(f"  vs 参考术语表: 新={len(new_terms)} 已有={len(existing_terms)} 差异={len(differing)}")
    if existing_glossary:
        print(f"增量合并: 基础={len(existing_glossary)} + 新增={ai_added} = {len(glossary_dict)} (冲突={len(ai_conflicts)})")
    print(f"分类:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")
    if stats:
        print(f"统计: {json.dumps(stats, ensure_ascii=False)}")
    print(f"{'='*60}")

    # 新术语示例
    if new_terms:
        print(f"\n新术语示例 (前 40):")
        for item in sorted(new_terms, key=lambda x: -x.get("occurrences", 1))[:40]:
            conf = f" [{item['min_confidence']}]" if item.get("min_confidence") else ""
            src = f" ({', '.join(item.get('sources', [])[:2])})" if item.get("sources") else ""
            print(f"  {item['english']}: {item['chinese']}{conf}{src}")

    # 译名差异
    if differing:
        print(f"\n译名差异 (前 20):")
        for d in differing[:20]:
            print(f"  {d['english']}: SoG={d['chinese']} vs glossary={d['glossary_zh']}")

    print(f"\n✅ 术语表: {output_json} ({len(glossary_dict)} 条)")
    print(f"✅ 详细报告: {output_candidates}")
    print(f"✅ 差异报告: {output_conflicts}")

    _save_cache(cache_path, cache, cache_enabled)
    _write_log(log_path, f"DONE unique={len(merged)} new={len(new_terms)} differs={len(differing)}")


if __name__ == "__main__":
    main()
