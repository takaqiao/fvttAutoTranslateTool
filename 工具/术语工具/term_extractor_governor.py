import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import time
from pathlib import Path
from threading import Lock

from openai import OpenAI
from tqdm import tqdm

CONFIG_PATH = Path("术语工具/term_extractor_governor_config.json")

DEFAULT_CONFIG = {
    "openai_api_key": "",
    "openai_base_url": "https://api.openai.com/v1",
    "model": "gpt-5.4",
    "target_folder": ".",
    "target_file_glob": "*.json",
    "recursive": True,
    "include_path_keywords": ["en.json", "/lang/", "/i18n/", "crucible"],
    "exclude_path_keywords": ["cache", "node_modules", ".mapping", "uuid-fixed", "merged"],
    "max_items_per_chunk": 1400,
    "max_chars_per_chunk": 320000,
    "max_workers": 4,
    "quality_first_serial": True,
    "target_rpm": 240,
    "max_retries": 3,
    "request_timeout_seconds": 240,
    "cache_enabled": True,
    "cache_path": "term_extractor_governor_cache.json",
    "state_path": "term_extractor_governor_state.json",
    "output_glossary_path": "glossary_governed.json",
    "output_terms_payload_path": "glossary_governed_terms_payload.json",
    "output_overlap_report_path": "glossary_governed_overlap_report.json",
    "keep_variants_for_polysemy": True,
    "variant_max_count": 4,
    "overlap_context_max_terms": 80,
    "overlap_min_token_len": 4,
    "min_en_len": 2,
    "max_en_len": 64,
    "extract_english_from_bilingual": True,
    "extract_pairs_from_bilingual": True,
    "skip_key_suffixes": [
        "img",
        "image",
        "icon",
        "thumb",
        "tokenimg",
        "tokenimage",
        "uuid",
        "id",
        "_id"
    ],
    "log_path": "term_extractor_governor.log"
}

_LOG_LOCK = Lock()
_CACHE_LOCK = Lock()
_STATS_LOCK = Lock()


def _write_log(log_path: Path, msg: str):
    ts = time.strftime("%H:%M:%S", time.localtime())
    line = f"[{ts}] {msg}\\n"
    with _LOG_LOCK:
        try:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass


def _load_json(path: Path, fallback):
    if not path.exists():
        return fallback
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data
    except Exception:
        return fallback


def _save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _norm_en(text: str) -> str:
    t = _normalize_ws(text)
    t = re.sub(r"\s*([,;/:&\-])\s*", r"\1", t)
    return t


def _norm_zh(text: str) -> str:
    return _normalize_ws(text)


def _contains_en(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", text or ""))


def _contains_zh(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def _extract_last_key(path_str: str) -> str:
    if not path_str:
        return ""
    cleaned = re.sub(r"\[\d+\]", "", path_str)
    return cleaned.split(".")[-1] if "." in cleaned else cleaned


def _is_media_path(value: str) -> bool:
    if not value:
        return False
    if re.search(r"^(https?://|modules/|systems/|icons/|assets/)", value, flags=re.I):
        return True
    return bool(re.search(r"\.(png|jpg|jpeg|webp|gif|svg|mp3|ogg|wav|m4a|webm)(\?|$)", value, flags=re.I))


def _strip_markup(text: str) -> str:
    t = text or ""
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"&[a-zA-Z0-9#]+;", " ", t)
    return _normalize_ws(t)


def _extract_english_candidate(text: str, extract_bilingual: bool = True) -> str:
    if not isinstance(text, str):
        return ""
    raw = text.strip()
    if not raw:
        return ""

    cleaned = _strip_markup(raw)
    has_en = _contains_en(cleaned)
    has_zh = _contains_zh(cleaned)

    if has_en and not has_zh:
        return cleaned

    if not (extract_bilingual and has_en and has_zh):
        return ""

    # Prefer EN-only lines in bilingual content (common format: ZH line + EN line).
    lines = [
        _strip_markup(line)
        for line in re.split(r"[\r\n]+", raw)
    ]
    en_lines = [line for line in lines if line and _contains_en(line) and not _contains_zh(line)]
    if en_lines:
        return "\n".join(en_lines)

    # Fallback for mixed single-line labels like "中文 English Name".
    mixed = _strip_markup(raw)
    match = re.search(r"([A-Za-z][A-Za-z0-9'’:,/&() -]{2,})$", mixed)
    if match:
        tail = _normalize_ws(match.group(1))
        if _contains_en(tail) and not _contains_zh(tail):
            return tail

    return ""


def _extract_bilingual_pairs(text: str) -> list[tuple[str, str]]:
    if not isinstance(text, str):
        return []
    raw = text.strip()
    if not raw:
        return []

    # Preserve paragraph boundaries before stripping markup.
    pre = re.sub(r"(?i)</p>", "\n", raw)
    pre = re.sub(r"(?i)<br\s*/?>", "\n", pre)
    lines = [_strip_markup(line) for line in re.split(r"[\r\n]+", pre)]
    lines = [line for line in lines if line]

    pairs = []

    # Case 1: paired lines (ZH line + EN line).
    zh_lines = [line for line in lines if _contains_zh(line) and not _contains_en(line)]
    en_lines = [line for line in lines if _contains_en(line) and not _contains_zh(line)]
    for idx in range(min(len(zh_lines), len(en_lines))):
        en = _norm_en(en_lines[idx])
        zh = _norm_zh(zh_lines[idx])
        if en and zh:
            pairs.append((en, zh))

    # Case 2: mixed same line, e.g. "绑定武装 Bind Armament".
    for line in lines:
        if not (_contains_zh(line) and _contains_en(line)):
            continue
        m = re.search(r"^(.*?[\u4e00-\u9fff].*?)\s+([A-Za-z][A-Za-z0-9'’:,/&() -]{2,})$", line)
        if not m:
            continue
        zh = _norm_zh(m.group(1))
        en = _norm_en(m.group(2))
        if en and zh and _contains_en(en) and _contains_zh(zh):
            pairs.append((en, zh))

    # Deduplicate while preserving order.
    dedup = []
    seen = set()
    for en, zh in pairs:
        key = f"{en.casefold()}|||{zh}"
        if key in seen:
            continue
        seen.add(key)
        dedup.append((en, zh))
    return dedup


def _iter_json_strings(node, parts=None, out=None):
    if parts is None:
        parts = []
    if out is None:
        out = []
    if isinstance(node, dict):
        for key, value in node.items():
            _iter_json_strings(value, parts + [str(key)], out)
    elif isinstance(node, list):
        for idx, value in enumerate(node):
            _iter_json_strings(value, parts + [f"[{idx}]"], out)
    elif isinstance(node, str):
        out.append((".".join(parts), node))
    return out


def _should_pick_pair(path_str: str, english: str, chinese: str, skip_key_suffixes: set[str], min_en_len: int, max_en_len: int) -> bool:
    en = (english or "").strip()
    zh = (chinese or "").strip()
    if not en or not zh:
        return False
    if not _contains_en(en) or not _contains_zh(zh):
        return False
    if len(en) < min_en_len or len(en) > 300:
        return False
    if _is_media_path(en):
        return False
    last_key = _extract_last_key(path_str).casefold()
    if last_key in skip_key_suffixes:
        return False
    if len(re.sub(r"[^A-Za-z]", "", en)) > max_en_len * 8:
        return False
    return True


def _should_pick(path_str: str, value: str, skip_key_suffixes: set[str], min_en_len: int, max_en_len: int) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text:
        return False
    if _contains_zh(text):
        return False
    if not _contains_en(text):
        return False
    if len(text) < min_en_len or len(text) > 3000:
        return False
    if _is_media_path(text):
        return False
    last_key = _extract_last_key(path_str).casefold()
    if last_key in skip_key_suffixes:
        return False
    if len(re.sub(r"[^A-Za-z]", "", text)) > max_en_len * 8:
        return False
    return True


def _load_config(config_path: Path) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if config_path.exists():
        user_cfg = _load_json(config_path, {})
        if isinstance(user_cfg, dict):
            cfg.update(user_cfg)
    else:
        _save_json(config_path, cfg)
        print(f"已生成默认配置: {config_path}")
    return cfg


def _hash_text(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8", errors="ignore")).hexdigest()


def _parse_json_object(raw: str):
    if not raw:
        return {}
    text = raw.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", text, flags=re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


class RateLimiter:
    def __init__(self, rpm: int):
        self.rpm = max(int(rpm), 1)
        self.capacity = self.rpm
        self.tokens = self.capacity
        self.last_refill = time.time()
        self.min_interval = 60.0 / self.rpm
        self.last_request = 0.0
        self.lock = Lock()

    def wait(self):
        with self.lock:
            while True:
                now = time.time()
                elapsed = now - self.last_refill
                if elapsed >= 60.0:
                    refill = int(elapsed // 60.0) * self.capacity
                    self.tokens = min(self.capacity, self.tokens + refill)
                    self.last_refill = now

                if self.tokens <= 0:
                    time.sleep(max(0.5, 60.0 - (now - self.last_refill)))
                    continue

                spacing = self.last_request + self.min_interval - now
                if spacing > 0:
                    time.sleep(spacing)
                    continue

                self.tokens -= 1
                self.last_request = time.time()
                break


class TermExtractor:
    def __init__(self, cfg: dict):
        api_key = cfg.get("openai_api_key") or os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("未设置 openai_api_key 且环境变量 OPENAI_API_KEY 为空")
        self.client = OpenAI(api_key=api_key, base_url=cfg.get("openai_base_url", "https://api.openai.com/v1"))
        self.model = str(cfg.get("model", "gpt-5.4") or "gpt-5.4")
        self.max_retries = max(1, int(cfg.get("max_retries", 3)))
        self.timeout = float(cfg.get("request_timeout_seconds", 240))

    def extract(self, chunk_payload: str, related_terms_text: str) -> dict:
        instructions = (
            "你是TRPG术语治理专家。你会收到中英候选配对。请判断哪些是真正应入术语表的术语，并校对中英对应关系。"
            "输出必须是严格JSON对象，格式: {\"terms\":[{...}]}。"
            "每个term字段: english, preferred_zh, alt_zh, polysemy, reason, confidence, decision。"
            "规则: 1) 优先统一术语。2) 仅在明确多义且语境冲突时才保留多元。"
            "3) alt_zh最多3个，且不得与preferred_zh重复。4) 不要杜撰文本中不存在的术语。"
            "5) 如果提供了已有术语候选，优先保持其译名，除非当前语境明确更优。"
            "6) 英文术语应为短语/词组，不要返回整句。"
            "7) 保留大小写自然形式，但不要重复同义拼写。"
            "8) confidence 为 0~1 小数。"
            "9) decision 只能是 keep 或 drop。"
        )
        user_input = (
            "[当前分段候选配对(含来源路径)]\\n"
            + chunk_payload
            + "\\n\\n[已有关联术语候选]\n"
            + (related_terms_text or "(none)")
            + "\\n\\n请直接返回 JSON 对象。"
        )

        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.responses.create(
                    model=self.model,
                    instructions=instructions,
                    input=user_input,
                    timeout=self.timeout,
                )
                raw = (resp.output_text or "").strip()
                parsed = _parse_json_object(raw)
                if isinstance(parsed.get("terms"), list):
                    return parsed
                last_err = RuntimeError(f"非JSON或缺少terms: {raw[:240]}")
            except Exception as exc:
                last_err = exc
            time.sleep(attempt)

        raise RuntimeError(f"AI术语提取失败: {last_err}")


def _find_target_files(cfg: dict) -> list[Path]:
    root = Path(str(cfg.get("target_folder", ".")).strip() or ".")
    glob_pat = str(cfg.get("target_file_glob", "*.json")).strip() or "*.json"
    recursive = bool(cfg.get("recursive", True))
    include_keywords = [str(x).strip().casefold() for x in cfg.get("include_path_keywords", []) if str(x).strip()]
    exclude_keywords = [str(x).strip().casefold() for x in cfg.get("exclude_path_keywords", []) if str(x).strip()]

    it = root.rglob(glob_pat) if recursive else root.glob(glob_pat)
    out = []
    for p in it:
        if not p.is_file() or p.suffix.lower() != ".json":
            continue
        tag = str(p).replace("\\\\", "/").casefold()
        if include_keywords and not any(k in tag for k in include_keywords):
            continue
        if exclude_keywords and any(k in tag for k in exclude_keywords):
            continue
        out.append(p)
    out.sort()
    return out


def _collect_text_items(file_paths: list[Path], cfg: dict):
    skip_key_suffixes = {str(x).strip().casefold() for x in cfg.get("skip_key_suffixes", []) if str(x).strip()}
    min_en_len = max(1, int(cfg.get("min_en_len", 2)))
    max_en_len = max(8, int(cfg.get("max_en_len", 64)))
    extract_bilingual = bool(cfg.get("extract_english_from_bilingual", True))
    extract_pairs = bool(cfg.get("extract_pairs_from_bilingual", True))

    items = []
    for path in file_paths:
        try:
            data = _load_json(path, None)
        except Exception:
            data = None
        if data is None:
            continue

        entries = _iter_json_strings(data)
        for rel_path, value in entries:
            source = f"{path.as_posix()}::{rel_path}"
            if extract_pairs:
                pairs = _extract_bilingual_pairs(value)
                for en, zh in pairs:
                    if not _should_pick_pair(rel_path, en, zh, skip_key_suffixes, min_en_len, max_en_len):
                        continue
                    items.append({"source": source, "english": en, "chinese": zh})
            else:
                candidate = _extract_english_candidate(value, extract_bilingual=extract_bilingual)
                if not _should_pick(rel_path, candidate, skip_key_suffixes, min_en_len, max_en_len):
                    continue
                items.append({"source": source, "english": candidate, "chinese": ""})

    return items


def _build_chunks(items: list[dict], cfg: dict):
    max_items = max(50, int(cfg.get("max_items_per_chunk", 1400)))
    max_chars = max(4000, int(cfg.get("max_chars_per_chunk", 320000)))

    chunks = []
    cur = []
    cur_chars = 0

    for item in items:
        line = f"- {item['source']}\\n  EN: {item['english']}\\n  ZH: {item.get('chinese', '')}\\n"
        ln = len(line)

        if cur and (len(cur) >= max_items or cur_chars + ln > max_chars):
            chunks.append(cur)
            cur = []
            cur_chars = 0

        cur.append(item)
        cur_chars += ln

    if cur:
        chunks.append(cur)

    return chunks


def _tokenize_english(text: str, min_len: int) -> set[str]:
    tokens = set()
    for tk in re.findall(r"[A-Za-z][A-Za-z0-9'_-]{1,}", text or ""):
        t = tk.casefold()
        if len(t) >= min_len:
            tokens.add(t)
    return tokens


def _build_related_terms_for_chunk(chunk: list[dict], term_state: dict, cfg: dict):
    max_terms = max(8, int(cfg.get("overlap_context_max_terms", 80)))
    min_tok = max(2, int(cfg.get("overlap_min_token_len", 4)))

    chunk_text = "\n".join(item.get("english", "") for item in chunk)
    chunk_tokens = _tokenize_english(chunk_text, min_tok)

    rows = []
    for english, term in term_state.get("terms", {}).items():
        en_norm = str(english).strip()
        if not en_norm:
            continue
        en_tokens = _tokenize_english(en_norm, min_tok)
        token_hit = bool(en_tokens and en_tokens.intersection(chunk_tokens))
        phrase_hit = en_norm.casefold() in chunk_text.casefold()
        if not token_hit and not phrase_hit:
            continue

        preferred = term.get("preferred_zh", "")
        variants = term.get("zh_variants", {})
        alt = [k for k in variants.keys() if k != preferred][:3]
        rows.append({
            "english": en_norm,
            "preferred_zh": preferred,
            "alt_zh": alt,
            "polysemy": bool(term.get("polysemy", False)),
            "occurrences": int(term.get("total_occurrences", 0)),
        })

    rows.sort(key=lambda x: x["occurrences"], reverse=True)
    rows = rows[:max_terms]
    return json.dumps(rows, ensure_ascii=False, indent=2)


def _normalize_term_row(row: dict):
    if not isinstance(row, dict):
        return None
    decision = str(row.get("decision", "keep")).strip().lower()
    if decision and decision not in {"keep", "drop"}:
        decision = "keep"
    if decision == "drop":
        return None
    en = _norm_en(str(row.get("english", "")))
    zh = _norm_zh(str(row.get("preferred_zh", "")))
    if not en or not zh or not _contains_en(en) or not _contains_zh(zh):
        return None

    alt_raw = row.get("alt_zh", [])
    alt = []
    if isinstance(alt_raw, list):
        for item in alt_raw:
            z = _norm_zh(str(item))
            if z and z != zh and z not in alt:
                alt.append(z)

    polysemy = bool(row.get("polysemy", False))
    reason = _normalize_ws(str(row.get("reason", "")))

    conf = row.get("confidence", 0.7)
    try:
        conf = float(conf)
    except Exception:
        conf = 0.7
    conf = max(0.0, min(1.0, conf))

    return {
        "english": en,
        "preferred_zh": zh,
        "alt_zh": alt,
        "polysemy": polysemy,
        "reason": reason,
        "confidence": conf,
    }


def _upsert_term(term_state: dict, row: dict, cfg: dict):
    keep_variants = bool(cfg.get("keep_variants_for_polysemy", True))
    variant_max_count = max(1, int(cfg.get("variant_max_count", 4)))

    en = row["english"]
    zh = row["preferred_zh"]
    alt = row["alt_zh"]
    polysemy = bool(row["polysemy"])
    reason = row["reason"]
    confidence = row["confidence"]

    terms = term_state.setdefault("terms", {})
    entry = terms.get(en)

    if not entry:
        variants = {zh: 1}
        for item in alt[: max(0, variant_max_count - 1)]:
            variants[item] = variants.get(item, 0) + 1
        terms[en] = {
            "preferred_zh": zh,
            "zh_variants": variants,
            "polysemy": polysemy,
            "total_occurrences": 1,
            "reasons": [reason] if reason else [],
            "avg_confidence": confidence,
        }
        return 1, 0

    changed = 0
    overlap = 1
    entry["total_occurrences"] = int(entry.get("total_occurrences", 0)) + 1

    variants = entry.setdefault("zh_variants", {})
    variants[zh] = int(variants.get(zh, 0)) + 1
    for item in alt:
        if len(variants) >= variant_max_count and item not in variants:
            break
        variants[item] = int(variants.get(item, 0)) + 1

    if polysemy:
        entry["polysemy"] = True

    reasons = entry.setdefault("reasons", [])
    if reason and reason not in reasons:
        reasons.append(reason)
        if len(reasons) > 6:
            del reasons[0 : len(reasons) - 6]

    prev_conf = float(entry.get("avg_confidence", 0.7))
    count = int(entry.get("total_occurrences", 1))
    entry["avg_confidence"] = round((prev_conf * (count - 1) + confidence) / max(count, 1), 4)

    if entry.get("polysemy") and keep_variants:
        sorted_vars = sorted(variants.items(), key=lambda kv: kv[1], reverse=True)
        entry["preferred_zh"] = sorted_vars[0][0]
    else:
        old_pref = entry.get("preferred_zh", "")
        if variants.get(zh, 0) >= variants.get(old_pref, 0):
            entry["preferred_zh"] = zh
            if zh != old_pref:
                changed += 1

    return changed, overlap


def _build_chunk_payload(chunk: list[dict]) -> str:
    lines = []
    for item in chunk:
        lines.append(f"- {item['source']}\\n  EN: {item.get('english', '')}\\n  ZH: {item.get('chinese', '')}")
    return "\n".join(lines)


def _dict_glossary_from_state(term_state: dict):
    out = {}
    for en, entry in sorted(term_state.get("terms", {}).items(), key=lambda kv: kv[0].casefold()):
        preferred = entry.get("preferred_zh", "")
        variants = entry.get("zh_variants", {})
        polysemy = bool(entry.get("polysemy", False))
        sorted_vars = [k for k, _ in sorted(variants.items(), key=lambda kv: kv[1], reverse=True)]
        if polysemy and len(sorted_vars) > 1:
            out[en] = sorted_vars
        else:
            out[en] = preferred or (sorted_vars[0] if sorted_vars else "")
    return out


def _payload_from_state(term_state: dict):
    rows = []
    for en, entry in sorted(term_state.get("terms", {}).items(), key=lambda kv: kv[0].casefold()):
        vars_sorted = sorted(entry.get("zh_variants", {}).items(), key=lambda kv: kv[1], reverse=True)
        rows.append(
            {
                "english": en,
                "preferred_zh": entry.get("preferred_zh", ""),
                "zh_variants": {k: v for k, v in vars_sorted},
                "polysemy": bool(entry.get("polysemy", False)),
                "total_occurrences": int(entry.get("total_occurrences", 0)),
                "avg_confidence": float(entry.get("avg_confidence", 0.7)),
                "reasons": entry.get("reasons", []),
            }
        )
    return {
        "meta": {
            "total_terms": len(rows),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "note": "quality-first glossary; keep variants only for confirmed polysemy",
        },
        "terms": rows,
    }


def main():
    config_path_default = CONFIG_PATH
    parser = argparse.ArgumentParser(description="Large-context glossary extractor with overlap-aware governance")
    parser.add_argument("--config", default=str(config_path_default), help="Path to config JSON")
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = _load_config(config_path)

    log_path = Path(str(cfg.get("log_path", "term_extractor_governor.log")))
    cache_enabled = bool(cfg.get("cache_enabled", True))
    cache_path = Path(str(cfg.get("cache_path", "term_extractor_governor_cache.json")))
    state_path = Path(str(cfg.get("state_path", "term_extractor_governor_state.json")))

    output_glossary_path = Path(str(cfg.get("output_glossary_path", "glossary_governed.json")))
    output_terms_payload_path = Path(str(cfg.get("output_terms_payload_path", "glossary_governed_terms_payload.json")))
    output_overlap_report_path = Path(str(cfg.get("output_overlap_report_path", "glossary_governed_overlap_report.json")))

    cache = _load_json(cache_path, {"chunks": {}}) if cache_enabled else {"chunks": {}}
    if not isinstance(cache, dict) or not isinstance(cache.get("chunks"), dict):
        cache = {"chunks": {}}

    term_state = _load_json(state_path, {"terms": {}, "meta": {}})
    if not isinstance(term_state, dict):
        term_state = {"terms": {}, "meta": {}}
    if not isinstance(term_state.get("terms"), dict):
        term_state["terms"] = {}

    files = _find_target_files(cfg)
    if not files:
        print("❌ 未找到目标文件，请检查 target_folder/target_file_glob/include_path_keywords")
        return

    print(f"目标文件: {len(files)}")
    items = _collect_text_items(files, cfg)
    if not items:
        print("❌ 未提取到可处理中英候选配对")
        return

    chunks = _build_chunks(items, cfg)
    print(f"候选中英对: {len(items)} | 分段数: {len(chunks)}")

    extractor = TermExtractor(cfg)
    limiter = RateLimiter(int(cfg.get("target_rpm", 240)))

    stats = {
        "chunks": len(chunks),
        "cache_hit": 0,
        "ai_ok": 0,
        "ai_fail": 0,
        "terms_changed": 0,
        "overlap_hits": 0,
    }
    overlap_report = []

    def worker(chunk_idx: int, chunk: list[dict]):
        payload = _build_chunk_payload(chunk)
        related_text = _build_related_terms_for_chunk(chunk, term_state, cfg)

        chunk_hash = _hash_text(payload + "\\n" + related_text)
        cache_key = f"chunk:{chunk_idx}"

        with _CACHE_LOCK:
            item = cache["chunks"].get(cache_key)
            if (
                cache_enabled
                and isinstance(item, dict)
                and item.get("hash") == chunk_hash
                and isinstance(item.get("terms"), list)
            ):
                with _STATS_LOCK:
                    stats["cache_hit"] += 1
                return item["terms"], len(json.loads(related_text)) if related_text and related_text != "(none)" else 0, True

        limiter.wait()
        parsed = extractor.extract(payload, related_text)
        raw_terms = parsed.get("terms", [])
        normalized = []
        for row in raw_terms:
            nr = _normalize_term_row(row)
            if nr:
                normalized.append(nr)

        if cache_enabled:
            with _CACHE_LOCK:
                cache["chunks"][cache_key] = {"hash": chunk_hash, "terms": normalized}

        rel_count = 0
        try:
            if related_text and related_text != "(none)":
                rel_count = len(json.loads(related_text))
        except Exception:
            rel_count = 0
        return normalized, rel_count, False

    quality_first_serial = bool(cfg.get("quality_first_serial", True))
    max_workers = max(1, int(cfg.get("max_workers", 4)))

    if quality_first_serial or max_workers <= 1:
        for idx, chunk in enumerate(tqdm(chunks, total=len(chunks), desc="术语治理提取", dynamic_ncols=True), start=1):
            try:
                terms, rel_count, from_cache = worker(idx, chunk)
            except Exception as e:
                with _STATS_LOCK:
                    stats["ai_fail"] += 1
                _write_log(log_path, f"ERROR worker failed | {e}")
                continue

            with _STATS_LOCK:
                if not from_cache:
                    stats["ai_ok"] += 1
                stats["overlap_hits"] += rel_count

            local_changed = 0
            local_overlap = 0
            for row in terms:
                changed, overlap = _upsert_term(term_state, row, cfg)
                local_changed += changed
                local_overlap += overlap

            with _STATS_LOCK:
                stats["terms_changed"] += local_changed

            overlap_report.append({
                "related_terms_in_prompt": rel_count,
                "returned_terms": len(terms),
                "merged_overlap_updates": local_overlap,
            })
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(worker, idx + 1, chunk) for idx, chunk in enumerate(chunks)]
            for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="术语治理提取", dynamic_ncols=True):
                try:
                    terms, rel_count, from_cache = fut.result()
                except Exception as e:
                    with _STATS_LOCK:
                        stats["ai_fail"] += 1
                    _write_log(log_path, f"ERROR worker failed | {e}")
                    continue

                with _STATS_LOCK:
                    if not from_cache:
                        stats["ai_ok"] += 1
                    stats["overlap_hits"] += rel_count

                local_changed = 0
                local_overlap = 0
                for row in terms:
                    changed, overlap = _upsert_term(term_state, row, cfg)
                    local_changed += changed
                    local_overlap += overlap

                with _STATS_LOCK:
                    stats["terms_changed"] += local_changed

                overlap_report.append({
                    "related_terms_in_prompt": rel_count,
                    "returned_terms": len(terms),
                    "merged_overlap_updates": local_overlap,
                })

    term_state["meta"] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "stats": stats,
    }

    glossary_dict = _dict_glossary_from_state(term_state)
    terms_payload = _payload_from_state(term_state)

    _save_json(state_path, term_state)
    _save_json(output_glossary_path, glossary_dict)
    _save_json(output_terms_payload_path, terms_payload)
    _save_json(output_overlap_report_path, {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "chunks": len(chunks),
            "stats": stats,
        },
        "items": overlap_report,
    })
    if cache_enabled:
        _save_json(cache_path, cache)

    print("✅ 完成：术语治理提取结束")
    print(f"术语表: {output_glossary_path} | 条目数: {len(glossary_dict)}")
    print(f"术语明细: {output_terms_payload_path}")
    print(f"重叠报告: {output_overlap_report_path}")
    print(
        "统计: "
        f"chunks={stats['chunks']} "
        f"cache_hit={stats['cache_hit']} "
        f"ai_ok={stats['ai_ok']} "
        f"ai_fail={stats['ai_fail']} "
        f"overlap_prompt_terms={stats['overlap_hits']}"
    )


if __name__ == "__main__":
    main()
