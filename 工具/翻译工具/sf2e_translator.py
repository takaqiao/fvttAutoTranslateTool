import concurrent.futures
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from threading import Lock

from tqdm import tqdm
from openai import OpenAI

# ================= 简化模式（仅术语替换 + GPT-5.2 翻译） =================

SIMPLE_CONFIG_PATH = Path("sf2e_simple_config.json")
SIMPLE_DEFAULT_CONFIG = {
    "openai_api_key": "",
    "openai_base_url": "https://api.openai.com/v1",
    "model": "gpt-5.2",
    "fallback_on_failure_enabled": True,
    "fallback_model_on_failure": "gpt-5-mini",
    "fallback_timeout_seconds": 45,
    "target_json_path": "system/sf2e/en.json",
    "target_folder_path": "",
    "target_file_glob": "*.json",
    "target_folder_recursive": False,
    "adaptive_glossary_enabled": False,
    "adaptive_glossary_mode": "file",
    "adaptive_glossary_entry_batch_size": 5,
    "adaptive_glossary_output_path": "glossary_adaptive_sf2e.json",
    "adaptive_glossary_max_terms_per_batch": 8,
    "adaptive_glossary_overwrite_existing": False,
    "glossary_path": "glossary.json",
    "extra_glossary_path": "",
    "prompt_profile": "default",
    "system_lang_mode": "auto",
    "system_lang_keep_original": False,
    "max_workers": 12,
    "target_rpm": 1200,
    "max_retries": 2,
    "request_timeout_seconds": 90,
    "length_aware_timeout_enabled": True,
    "long_text_char_threshold": 2500,
    "timeout_seconds_per_1k_chars": 35,
    "timeout_overhead_seconds": 20,
    "timeout_max_seconds": 240,
    "retry_delay_base_seconds": 0.5,
    "retry_delay_max_seconds": 2.0,
    "log_path": "sf2e_simple_run.log",
    "cache_path": "sf2e_simple_cache.json",
    "cache_enabled": True,
    "top_n_slowest": 20
}


def _simple_load_config():
    cfg = dict(SIMPLE_DEFAULT_CONFIG)
    if SIMPLE_CONFIG_PATH.exists():
        try:
            user_cfg = json.loads(SIMPLE_CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(user_cfg, dict):
                cfg.update(user_cfg)
        except Exception:
            pass
    return cfg


SIMPLE_CONFIG = _simple_load_config()
SIMPLE_OPENAI_API_KEY = SIMPLE_CONFIG.get("openai_api_key") or os.getenv("OPENAI_API_KEY", "")
SIMPLE_OPENAI_BASE_URL = SIMPLE_CONFIG.get("openai_base_url", "https://api.openai.com/v1")
SIMPLE_MODEL_ID = SIMPLE_CONFIG.get("model", "gpt-5.2")
SIMPLE_FALLBACK_ON_FAILURE_ENABLED = bool(SIMPLE_CONFIG.get("fallback_on_failure_enabled", True))
SIMPLE_FALLBACK_MODEL_ID = str(SIMPLE_CONFIG.get("fallback_model_on_failure", "gpt-5-mini") or "").strip()

SIMPLE_TARGET_JSON_PATH = Path(SIMPLE_CONFIG.get("target_json_path", "system/sf2e/en.json"))
SIMPLE_TARGET_FOLDER_PATH_RAW = str(SIMPLE_CONFIG.get("target_folder_path", "")).strip()
SIMPLE_TARGET_FOLDER_PATH = Path(SIMPLE_TARGET_FOLDER_PATH_RAW) if SIMPLE_TARGET_FOLDER_PATH_RAW else None
SIMPLE_TARGET_FILE_GLOB = str(SIMPLE_CONFIG.get("target_file_glob", "*.json")).strip() or "*.json"
SIMPLE_TARGET_FOLDER_RECURSIVE = bool(SIMPLE_CONFIG.get("target_folder_recursive", False))
SIMPLE_ADAPTIVE_GLOSSARY_ENABLED = bool(SIMPLE_CONFIG.get("adaptive_glossary_enabled", False))
SIMPLE_ADAPTIVE_GLOSSARY_MODE = str(SIMPLE_CONFIG.get("adaptive_glossary_mode", "file")).strip().lower()
if SIMPLE_ADAPTIVE_GLOSSARY_MODE not in {"file", "entry", "entry_batch"}:
    SIMPLE_ADAPTIVE_GLOSSARY_MODE = "file"
SIMPLE_ADAPTIVE_GLOSSARY_ENTRY_BATCH_SIZE = max(1, int(SIMPLE_CONFIG.get("adaptive_glossary_entry_batch_size", 5)))
SIMPLE_ADAPTIVE_GLOSSARY_OUTPUT_PATH_RAW = str(SIMPLE_CONFIG.get("adaptive_glossary_output_path", "glossary_adaptive_sf2e.json")).strip()
SIMPLE_ADAPTIVE_GLOSSARY_OUTPUT_PATH = Path(SIMPLE_ADAPTIVE_GLOSSARY_OUTPUT_PATH_RAW) if SIMPLE_ADAPTIVE_GLOSSARY_OUTPUT_PATH_RAW else Path("glossary_adaptive_sf2e.json")
SIMPLE_ADAPTIVE_GLOSSARY_MAX_TERMS_PER_BATCH = max(1, int(SIMPLE_CONFIG.get("adaptive_glossary_max_terms_per_batch", 8)))
SIMPLE_ADAPTIVE_GLOSSARY_OVERWRITE_EXISTING = bool(SIMPLE_CONFIG.get("adaptive_glossary_overwrite_existing", False))
SIMPLE_GLOSSARY_PATH = Path(SIMPLE_CONFIG.get("glossary_path", "glossary.json"))
SIMPLE_EXTRA_GLOSSARY_PATH_RAW = str(SIMPLE_CONFIG.get("extra_glossary_path", "")).strip()
SIMPLE_EXTRA_GLOSSARY_PATH = Path(SIMPLE_EXTRA_GLOSSARY_PATH_RAW) if SIMPLE_EXTRA_GLOSSARY_PATH_RAW else None
SIMPLE_EFFECTIVE_EXTRA_GLOSSARY_PATH = SIMPLE_EXTRA_GLOSSARY_PATH
if SIMPLE_ADAPTIVE_GLOSSARY_ENABLED and SIMPLE_EFFECTIVE_EXTRA_GLOSSARY_PATH is None:
    SIMPLE_EFFECTIVE_EXTRA_GLOSSARY_PATH = SIMPLE_ADAPTIVE_GLOSSARY_OUTPUT_PATH
SIMPLE_PROMPT_PROFILE = str(SIMPLE_CONFIG.get("prompt_profile", "default")).strip().lower()
SIMPLE_SYSTEM_LANG_MODE = str(SIMPLE_CONFIG.get("system_lang_mode", "auto")).strip().lower()
if SIMPLE_SYSTEM_LANG_MODE not in {"auto", "on", "off"}:
    SIMPLE_SYSTEM_LANG_MODE = "auto"
SIMPLE_SYSTEM_LANG_KEEP_ORIGINAL = bool(SIMPLE_CONFIG.get("system_lang_keep_original", False))

SIMPLE_MAX_WORKERS = int(SIMPLE_CONFIG.get("max_workers", 8))
SIMPLE_TARGET_RPM = int(SIMPLE_CONFIG.get("target_rpm", 450))
SIMPLE_MAX_RETRIES = max(1, int(SIMPLE_CONFIG.get("max_retries", 1)))
SIMPLE_REQUEST_TIMEOUT_SECONDS = float(SIMPLE_CONFIG.get("request_timeout_seconds", 45))
SIMPLE_FALLBACK_TIMEOUT_SECONDS = float(SIMPLE_CONFIG.get("fallback_timeout_seconds", SIMPLE_REQUEST_TIMEOUT_SECONDS))
SIMPLE_LENGTH_AWARE_TIMEOUT_ENABLED = bool(SIMPLE_CONFIG.get("length_aware_timeout_enabled", True))
SIMPLE_LONG_TEXT_CHAR_THRESHOLD = max(1, int(SIMPLE_CONFIG.get("long_text_char_threshold", 2500)))
SIMPLE_TIMEOUT_SECONDS_PER_1K_CHARS = max(0.0, float(SIMPLE_CONFIG.get("timeout_seconds_per_1k_chars", 35)))
SIMPLE_TIMEOUT_OVERHEAD_SECONDS = max(0.0, float(SIMPLE_CONFIG.get("timeout_overhead_seconds", 20)))
SIMPLE_TIMEOUT_MAX_SECONDS = max(
    SIMPLE_REQUEST_TIMEOUT_SECONDS,
    SIMPLE_FALLBACK_TIMEOUT_SECONDS,
    float(SIMPLE_CONFIG.get("timeout_max_seconds", 240))
)
SIMPLE_RETRY_DELAY_BASE_SECONDS = float(SIMPLE_CONFIG.get("retry_delay_base_seconds", 0.5))
SIMPLE_RETRY_DELAY_MAX_SECONDS = float(SIMPLE_CONFIG.get("retry_delay_max_seconds", 2.0))
SIMPLE_LOG_PATH = Path(SIMPLE_CONFIG.get("log_path", "sf2e_simple_run.log"))
SIMPLE_CACHE_PATH = Path(SIMPLE_CONFIG.get("cache_path", "sf2e_simple_cache.json"))
SIMPLE_CACHE_ENABLED = bool(SIMPLE_CONFIG.get("cache_enabled", True))
SIMPLE_TOP_N_SLOWEST = int(SIMPLE_CONFIG.get("top_n_slowest", 20))

_simple_log_lock = Lock()


def _simple_write_log(msg: str):
    ts = time.strftime("%H:%M:%S", time.localtime())
    line = f"[{ts}] {msg}\n"
    with _simple_log_lock:
        try:
            with SIMPLE_LOG_PATH.open("a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass


def _simple_snip(text: str, limit: int = 80) -> str:
    if not text:
        return ""
    t = re.sub(r"\s+", " ", text)
    return t[:limit]


def _simple_extract_original(text: str) -> str:
    if not isinstance(text, str) or "\n" not in text:
        return text
    first, rest = text.split("\n", 1)
    if _simple_contains_english(rest) and not _simple_contains_english(first):
        return rest
    return text


def _simple_hash(text: str) -> str:
    if text is None:
        text = ""
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def _simple_load_cache(path: Path):
    if not SIMPLE_CACHE_ENABLED or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _simple_save_cache(path: Path, cache: dict):
    if not SIMPLE_CACHE_ENABLED:
        return
    try:
        path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

# === 简化模式筛选规则 ===
SIMPLE_SKIP_PATH_SUBSTRINGS = {".mapping"}
SIMPLE_SKIP_KEYS = {
    "img", "portrait", "image", "icon", "thumb", "thumbnail",
    "token", "tokenImg", "tokenImage"
}
SIMPLE_ALLOW_KEYS = {
    "name", "label", "caption", "description", "text", "content",
    "publicNotes", "privateNotes", "publicnotes", "privatenotes",
    "gm_notes", "gm_description",
    "blurb", "reset", "routine", "stealth", "stealthDetails", "stealthdetails",
    "disable", "hazarddescription", "tokenName", "prototypeToken",
    "notes", "title", "header", "navName", "gmnote"
}

SIMPLE_INLINE_MERGE_KEYS = {"name", "tokenName", "prototypeToken"}

SIMPLE_SKIP_KEYS_NORM = {k.casefold() for k in SIMPLE_SKIP_KEYS}
SIMPLE_ALLOW_KEYS_NORM = {k.casefold() for k in SIMPLE_ALLOW_KEYS}
SIMPLE_INLINE_MERGE_KEYS_NORM = {k.casefold() for k in SIMPLE_INLINE_MERGE_KEYS}


def _simple_strip_codes_for_lang_detect(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'@UUID\[[^\]]+\]', ' ', text)
    text = re.sub(r'@Compendium\[[^\]]+\]', ' ', text)
    text = re.sub(r'@Localize\[[^\]]+\]', ' ', text)
    text = re.sub(r'\[\[.*?\]\]', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'&[a-zA-Z0-9#]+;', ' ', text)
    return text


def _simple_contains_english(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", _simple_strip_codes_for_lang_detect(text)))


def _simple_contains_chinese(text: str) -> bool:
    if not text:
        return False
    return bool(re.search(r"[\u4e00-\u9fff]", _simple_strip_codes_for_lang_detect(text)))


def _simple_extract_last_key(path_str: str) -> str:
    if not path_str:
        return ""
    cleaned = re.sub(r"\[\d+\]", "", path_str)
    return cleaned.split(".")[-1] if "." in cleaned else cleaned


def _simple_is_media_path(text: str) -> bool:
    if not isinstance(text, str) or not text:
        return False
    if re.search(r"^(https?://|modules/|systems/|icons/|assets/)", text, flags=re.I):
        return True
    return bool(re.search(r"\.(png|jpg|jpeg|webp|gif|svg|mp3|ogg|wav|m4a|webm)(\?|$)", text, flags=re.I))


def _simple_is_localize_only(text: str) -> bool:
    if not isinstance(text, str) or not text:
        return False
    return bool(re.fullmatch(r"\s*@Localize\[[^\]]+\]\s*", text))


def _simple_is_table_result_path(path_str: str) -> bool:
    if not isinstance(path_str, str) or not path_str:
        return False
    # Matches paths like root.entries.<table_name>.results.<range_key>
    # where <range_key> is often 1-1, 2-2, etc.
    return bool(re.search(r"\.entries\..+\.results\.[^.\[\]]+$", path_str))


def _simple_merge_translation(path_str: str, cn: str, original: str, keep_original: bool = True) -> str:
    if not keep_original:
        return (cn or "").strip()

    last_key_norm = _simple_extract_last_key(path_str).casefold()
    if last_key_norm in SIMPLE_INLINE_MERGE_KEYS_NORM:
        cn_text = (cn or "").replace("\r\n", " ").replace("\n", " ")
        original_text = (original or "").replace("\r\n", " ").replace("\n", " ")
        cn_norm = re.sub(r"\s{2,}", " ", cn_text).strip()
        original_norm = re.sub(r"\s{2,}", " ", original_text).strip()

        # For inline keys (especially name), avoid ballooning duplicated text.
        if not cn_norm:
            return original_norm
        if cn_norm.casefold() == original_norm.casefold():
            return original_norm
        if not _simple_contains_chinese(cn_norm):
            return original_norm

        if original_norm and cn_norm.endswith(original_norm):
            return cn_norm

        merged = f"{cn_norm} {original_norm}".strip()
        merged = re.sub(r"\s{2,}", " ", merged)
        return merged
    return f"{cn}\n{original}"


def _simple_is_index_like_page_name(path_str: str, value: str) -> bool:
    if not isinstance(value, str):
        return False
    if ".pages." not in path_str or not path_str.endswith(".name"):
        return False

    words = re.findall(r"[A-Za-z]+", value)
    if not words:
        return True
    if any((not w.isupper()) or len(w) > 2 for w in words):
        return False

    # Bucket headers like A/B/C, Q, NPC index labels should not be retried forever.
    total_letters = sum(len(w) for w in words)
    return len(words) <= 12 and total_letters <= 24


def _simple_is_system_lang_file(file_path: Path) -> bool:
    if SIMPLE_SYSTEM_LANG_MODE == "on":
        return True
    if SIMPLE_SYSTEM_LANG_MODE == "off":
        return False

    path_norm = str(file_path).replace("\\", "/").casefold()
    name = file_path.name.casefold()
    if not (name == "zh_hans.json" or name.endswith("-zh_hans.json")):
        return False

    return any(
        marker in path_norm
        for marker in (
            "/system/sf2_cn/",
            "/system/sf2e/",
            "/system/pf2_cn/",
            "/system/pf2e/",
        )
    )


def _simple_should_translate(path_str: str, value: str, is_system_lang_file: bool = False) -> bool:
    if not isinstance(value, str):
        return False
    if _simple_contains_chinese(value):
        return False
    if _simple_is_index_like_page_name(path_str, value):
        return False
    if SIMPLE_PROMPT_PROFILE == "dscryb" and re.search(r"\.(Image|Text)\.name$", path_str):
        return False
    if any(s in path_str for s in SIMPLE_SKIP_PATH_SUBSTRINGS):
        return False
    path_norm = path_str.casefold()
    if ".macros." in path_norm and path_norm.endswith(".command"):
        return False
    if ".folders." in path_norm and isinstance(value, str):
        if _simple_is_media_path(value) or _simple_is_localize_only(value):
            return False
        return _simple_contains_english(value)
    if _simple_is_table_result_path(path_str):
        if _simple_is_media_path(value) or _simple_is_localize_only(value):
            return False
        return _simple_contains_english(value)
    last_key_norm = _simple_extract_last_key(path_str).casefold()
    if (not is_system_lang_file) and SIMPLE_ALLOW_KEYS_NORM and last_key_norm not in SIMPLE_ALLOW_KEYS_NORM:
        return False
    if last_key_norm in SIMPLE_SKIP_KEYS_NORM:
        return False
    if _simple_is_media_path(value):
        return False
    if _simple_is_localize_only(value):
        return False
    if not _simple_contains_english(value):
        return False
    return True


def _simple_load_glossary(path: Path, glossary_data=None):
    data = glossary_data if isinstance(glossary_data, dict) else _simple_load_glossary_json(path)
    if not isinstance(data, dict):
        return {}, 0

    terms = []
    for en_raw, cn_raw in data.items():
        en = str(en_raw).strip()
        letters = re.sub(r"[^A-Za-z]", "", en)
        if not letters:
            continue

        if isinstance(cn_raw, list):
            cns = [str(x).strip() for x in cn_raw if str(x).strip()]
        else:
            cn = str(cn_raw).strip()
            cns = [cn] if cn else []

        if not cns:
            continue

        words = re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?", en)
        if not words:
            continue

        token_patterns = []
        for word in words:
            normalized = re.sub(r"(?:['’]s|s')$", "", word)
            if not normalized:
                continue
            token_base = re.escape(normalized)
            if len(normalized) >= 3:
                token_patterns.append(rf"{token_base}(?:['’]s|s'|s)?")
            else:
                token_patterns.append(token_base)

        if not token_patterns:
            continue

        separator = r"(?:[\s\-_.,:;!?/\\()\[\]{}'’\"`]+)?"
        pattern = r"(?<![A-Za-z0-9])" + separator.join(token_patterns) + r"(?![A-Za-z0-9])"
        terms.append((len(letters), re.compile(pattern, flags=re.IGNORECASE), cns, en))

    terms.sort(key=lambda x: x[0], reverse=True)

    buckets = {}
    for length, pattern, cns, en in terms:
        letters = re.sub(r"[^A-Za-z]", "", en)
        if not letters:
            continue
        first = letters[0].lower()
        length_bucket = (length // 5) * 5
        buckets.setdefault(first, {}).setdefault(length_bucket, []).append((length, pattern, cns, en))

    return buckets, len(terms)


def _simple_load_glossary_json(path: Path) -> dict:
    if not path or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return data if isinstance(data, dict) else {}


def _simple_save_glossary_json(path: Path, data: dict):
    try:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        _simple_write_log(f"ERROR save glossary failed | {path} | {e}")


def _simple_ensure_glossary_file(path: Path):
    try:
        if path.exists():
            return
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    except Exception as e:
        _simple_write_log(f"ERROR ensure glossary file failed | {path} | {e}")


def _simple_to_cn_list(value) -> list:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    return [text] if text else []


def _simple_merge_cn_values(old_value, new_value):
    merged = []
    seen = set()
    for item in _simple_to_cn_list(old_value) + _simple_to_cn_list(new_value):
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    if not merged:
        return ""
    if len(merged) == 1:
        return merged[0]
    return merged


def _simple_build_merged_glossary_data(base_path: Path, extra_path: Path | None):
    base_data = _simple_load_glossary_json(base_path)
    extra_data = _simple_load_glossary_json(extra_path) if extra_path else {}
    merged_data = dict(base_data)
    lower_to_key = {str(k).strip().casefold(): k for k in merged_data.keys() if str(k).strip()}
    conflict_terms = 0

    for en_raw, cn_raw in extra_data.items():
        en = str(en_raw).strip()
        if not en:
            continue
        lk = en.casefold()
        if lk in lower_to_key:
            existing_key = lower_to_key[lk]
            merged_data[existing_key] = _simple_merge_cn_values(merged_data.get(existing_key), cn_raw)
            conflict_terms += 1
        else:
            merged_data[en] = cn_raw
            lower_to_key[lk] = en

    return merged_data, len(base_data), len(extra_data), len(merged_data), conflict_terms


def _simple_apply_glossary(text: str, glossary_buckets):
    if not text or not glossary_buckets:
        return text, []
    letters = {w[0].lower() for w in re.findall(r"[A-Za-z]+", text)}
    if not letters:
        return text, []
    matched_ranked = []
    seen = set()
    for first in sorted(letters):
        length_map = glossary_buckets.get(first)
        if not length_map:
            continue
        for bucket_len in sorted(length_map.keys(), reverse=True):
            bucket_terms = length_map[bucket_len]
            for term_len, pattern, cns, en in bucket_terms:
                if pattern.search(text):
                    key = en.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    if len(cns) == 1:
                        matched_ranked.append((term_len, en, cns[0]))
                    else:
                        choices = [f"{chr(65 + i)}:{cn}" for i, cn in enumerate(cns[:5])]
                        matched_ranked.append((term_len, en, " | ".join(choices)))

    # Prefer longer terms in prompt hints to reduce over-triggering by ambiguous short words.
    matched_ranked.sort(key=lambda item: (-item[0], item[1].casefold()))
    matched = [(en, cn) for _, en, cn in matched_ranked]
    return text, matched


def _simple_compute_timeout_seconds(text: str, base_timeout: float) -> float:
    timeout = max(1.0, float(base_timeout))
    if not SIMPLE_LENGTH_AWARE_TIMEOUT_ENABLED:
        return timeout

    text_len = len(text or "")
    if text_len < SIMPLE_LONG_TEXT_CHAR_THRESHOLD:
        return timeout

    # Keep single-request translation, but allow very long entries more time.
    dynamic_timeout = (
        SIMPLE_TIMEOUT_OVERHEAD_SECONDS
        + (text_len / 1000.0) * SIMPLE_TIMEOUT_SECONDS_PER_1K_CHARS
    )
    return min(SIMPLE_TIMEOUT_MAX_SECONDS, max(timeout, dynamic_timeout))


class _SimpleRateLimiter:
    def __init__(self, rpm: int):
        self.rpm = max(rpm, 1)
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
                    sleep_time = max(0.5, 60.0 - (now - self.last_refill))
                    time.sleep(sleep_time)
                    continue

                spacing = self.last_request + self.min_interval - now
                if spacing > 0:
                    time.sleep(spacing)
                    continue

                self.tokens -= 1
                self.last_request = time.time()
                break


class _SimpleTranslator:
    def __init__(self):
        if not SIMPLE_OPENAI_API_KEY:
            raise RuntimeError("未设置 OPENAI_API_KEY")
        self.client = OpenAI(
            api_key=SIMPLE_OPENAI_API_KEY,
            base_url=SIMPLE_OPENAI_BASE_URL,
            timeout=SIMPLE_TIMEOUT_MAX_SECONDS,
            max_retries=0,
        )

    def _build_translation_instructions(self, glossary_terms=None) -> str:
        glossary_hint = ""
        if glossary_terms:
            pairs = [f"{e} -> {c}" for e, c in glossary_terms[:24]]
            glossary_hint = (
                " Contextual glossary candidates (apply only when the source uses the same sense): "
                + "; ".join(pairs)
                + "."
            )
        if SIMPLE_PROMPT_PROFILE == "dscryb":
            profile_prompt = (
                "You are a professional TRPG scene translator. "
                "Translate to Simplified Chinese only. "
                "Use vivid but concise imagery, natural Chinese flow, "
                "and maintain an immersive tone consistent across entries. "
                "Use glossary terms only when they match the source context."
            )
        else:
            profile_prompt = (
                "You are a professional Starfinder 2e (SF2E) translator. "
                "Translate to Simplified Chinese only. "
                "Use glossary terms only when they match the source context."
            )
        return (
            profile_prompt
            + glossary_hint +
            "Priority order: (1) preserve original meaning, (2) natural Chinese, (3) glossary preference. "
            "Never change sentence meaning just to force a glossary term. "
            "For polysemous words, choose by local context. "
            "If a word is everyday language, translate naturally even if a game-term glossary entry exists "
            "(example: 'I hope you are well' should use '希望', not a spell term). "
            "Use trait/spell/item specific glossary labels only when the source is clearly referring to game mechanics "
            "(for example, stat block traits, actions, feats, spells, item names, or rules text). "
            "If uncertain, prefer literal meaning and fluency over glossary injection. "
            "If a glossary term provides A/B/C candidate translations, choose the most context-appropriate option "
            "and keep that choice consistent within the same entry. "
            "Apply glossary terms with fuzzy recognition: treat whitespace, hyphen/punctuation, "
            "and apostrophe variants as equivalent (e.g., term, term's, terms, s' forms). "
            "Preserve all code-like tokens and formatting exactly as-is, "
            "including @UUID[...] , @Compendium[...] , [[...]] , HTML tags, "
            "punctuation, numbers, and line breaks. "
            "Do not add explanations or extra text. Output only the translation."
        )

    def _translate_once(self, model_id: str, text: str, instructions: str, timeout_seconds: float) -> str:
        response = self.client.responses.create(
            model=model_id,
            timeout=timeout_seconds,
            instructions=instructions,
            input=text
        )
        return (response.output_text or "").strip()

    def translate(self, text: str, path_str: str, glossary_terms=None) -> str:
        instructions = self._build_translation_instructions(glossary_terms)
        last_error = None
        primary_timeout_seconds = _simple_compute_timeout_seconds(
            text,
            SIMPLE_REQUEST_TIMEOUT_SECONDS,
        )

        if primary_timeout_seconds > SIMPLE_REQUEST_TIMEOUT_SECONDS:
            _simple_write_log(
                f"LONG_TEXT_TIMEOUT primary={primary_timeout_seconds:.1f}s len={len(text)} | {path_str}"
            )

        for attempt in range(1, SIMPLE_MAX_RETRIES + 1):
            try:
                return self._translate_once(
                    SIMPLE_MODEL_ID,
                    text,
                    instructions,
                    primary_timeout_seconds,
                )
            except Exception as e:
                last_error = e
                if attempt == SIMPLE_MAX_RETRIES:
                    break
                _simple_write_log(
                    f"RETRY {attempt}/{SIMPLE_MAX_RETRIES} | timeout={primary_timeout_seconds:.1f}s | {path_str} | {e}"
                )
                delay = min(
                    SIMPLE_RETRY_DELAY_MAX_SECONDS,
                    SIMPLE_RETRY_DELAY_BASE_SECONDS * (2 ** (attempt - 1))
                )
                time.sleep(max(0.0, delay))

        can_fallback = (
            SIMPLE_FALLBACK_ON_FAILURE_ENABLED
            and bool(SIMPLE_FALLBACK_MODEL_ID)
            and SIMPLE_FALLBACK_MODEL_ID != SIMPLE_MODEL_ID
        )
        if can_fallback:
            fallback_timeout_seconds = _simple_compute_timeout_seconds(
                text,
                SIMPLE_FALLBACK_TIMEOUT_SECONDS,
            )
            _simple_write_log(
                f"FALLBACK model={SIMPLE_FALLBACK_MODEL_ID} timeout={fallback_timeout_seconds:.1f}s | {path_str} | primary_error={last_error}"
            )
            try:
                fallback_cn = self._translate_once(
                    SIMPLE_FALLBACK_MODEL_ID,
                    text,
                    instructions,
                    fallback_timeout_seconds,
                )
                if fallback_cn:
                    _simple_write_log(f"FALLBACK_OK model={SIMPLE_FALLBACK_MODEL_ID} | {path_str}")
                    return fallback_cn
            except Exception as fallback_error:
                last_error = fallback_error

        print(f"⚠️ 翻译失败: {path_str} | {last_error}")
        _simple_write_log(
            f"ERROR translate failed | {path_str} | {last_error} | text={_simple_snip(text)}"
        )
        return ""

    def extract_basic_glossary(self, pair_lines: list[str], max_terms: int = 8) -> dict:
        if not pair_lines:
            return {}
        max_terms = max(1, int(max_terms))
        joined = "\n".join(pair_lines[:120])
        glossary_timeout_seconds = _simple_compute_timeout_seconds(
            joined,
            SIMPLE_REQUEST_TIMEOUT_SECONDS,
        )
        try:
            response = self.client.responses.create(
                model=SIMPLE_MODEL_ID,
                timeout=glossary_timeout_seconds,
                instructions=(
                    "Extract a minimal, foundational glossary for SF2E/science-fantasy TRPG translation. "
                    "Use only very basic reusable terms. "
                    "Avoid proper nouns, names, place names, item IDs, and long phrases. "
                    "Prefer 1-2 English words, at most 3 words. "
                    "Return ONLY a valid JSON object: {\"English term\":\"简体中文\"}. "
                    f"Return at most {max_terms} terms."
                ),
                input=(
                    "Bilingual pairs (EN => ZH):\n"
                    f"{joined}\n"
                ),
            )
            raw = (response.output_text or "").strip()
            if not raw:
                return {}
            parsed = None
            try:
                parsed = json.loads(raw)
            except Exception:
                m = re.search(r"\{[\s\S]*\}", raw)
                if m:
                    try:
                        parsed = json.loads(m.group(0))
                    except Exception:
                        parsed = None
            if not isinstance(parsed, dict):
                return {}
            return parsed
        except Exception as e:
            _simple_write_log(f"ERROR extract glossary failed | {e}")
            return {}


def _simple_format_path(parts) -> str:
    out = "root"
    for p in parts:
        if isinstance(p, int):
            out += f"[{p}]"
        else:
            out += f".{p}"
    return out


def _simple_collect_string_tasks(node, parts=None, tasks=None):
    if tasks is None:
        tasks = []
    if parts is None:
        parts = []
    if isinstance(node, dict):
        for k, v in node.items():
            _simple_collect_string_tasks(v, parts + [k], tasks)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _simple_collect_string_tasks(v, parts + [i], tasks)
    elif isinstance(node, str):
        tasks.append((parts, _simple_format_path(parts), node))
    return tasks


def _simple_set_value_at_path(data, parts, value):
    if not parts:
        return
    cur = data
    try:
        for p in parts[:-1]:
            cur = cur[p]
        cur[parts[-1]] = value
    except Exception as e:
        _simple_write_log(f"ERROR set_value failed | path={_simple_format_path(parts)} | {e}")
        return


def _simple_normalize_term_key(text: str) -> str:
    t = re.sub(r"\s+", " ", str(text or "").strip())
    return t


def _simple_is_basic_term(term: str, cn: str) -> bool:
    en = _simple_normalize_term_key(term)
    zh = str(cn or "").strip()
    if not en or not zh:
        return False
    if not re.fullmatch(r"[A-Za-z][A-Za-z\s\-']*[A-Za-z]", en):
        return False
    words = re.findall(r"[A-Za-z]+", en)
    if not words or len(words) > 3:
        return False
    if len("".join(words)) < 3:
        return False
    if any(len(w) > 20 for w in words):
        return False
    if re.search(r"[0-9@\[\]{}_/\\]", en):
        return False
    return True


def _simple_upsert_adaptive_glossary(glossary_data: dict, extracted_terms: dict):
    if not isinstance(glossary_data, dict) or not isinstance(extracted_terms, dict):
        return 0

    lower_to_key = {str(k).strip().casefold(): k for k in glossary_data.keys() if str(k).strip()}
    changed = 0

    for en_raw, cn_raw in extracted_terms.items():
        en = _simple_normalize_term_key(en_raw)
        cns = _simple_to_cn_list(cn_raw)
        if not cns:
            continue
        cn = cns[0]
        if not _simple_is_basic_term(en, cn):
            continue

        lk = en.casefold()
        if lk in lower_to_key:
            key = lower_to_key[lk]
            if SIMPLE_ADAPTIVE_GLOSSARY_OVERWRITE_EXISTING:
                merged = cn
            else:
                merged = _simple_merge_cn_values(glossary_data.get(key), cn)
            if merged != glossary_data.get(key):
                glossary_data[key] = merged
                changed += 1
        else:
            glossary_data[en] = cn
            lower_to_key[lk] = en
            changed += 1

    return changed


def _simple_refresh_runtime_glossary(glossary_data: dict):
    buckets, count = _simple_load_glossary(SIMPLE_GLOSSARY_PATH, glossary_data=glossary_data)
    return buckets, count


def _simple_collect_target_files():
    if SIMPLE_TARGET_FOLDER_PATH:
        if not SIMPLE_TARGET_FOLDER_PATH.exists() or not SIMPLE_TARGET_FOLDER_PATH.is_dir():
            print(f"❌ 批量目录不存在或不是文件夹: {SIMPLE_TARGET_FOLDER_PATH}")
            return []
        if SIMPLE_TARGET_FOLDER_RECURSIVE:
            files = sorted([p for p in SIMPLE_TARGET_FOLDER_PATH.rglob(SIMPLE_TARGET_FILE_GLOB) if p.is_file()])
        else:
            files = sorted([p for p in SIMPLE_TARGET_FOLDER_PATH.glob(SIMPLE_TARGET_FILE_GLOB) if p.is_file()])
        return files

    if SIMPLE_TARGET_JSON_PATH.exists() and SIMPLE_TARGET_JSON_PATH.is_file():
        return [SIMPLE_TARGET_JSON_PATH]

    print(f"❌ 目标文件不存在: {SIMPLE_TARGET_JSON_PATH}")
    return []


def _simple_translate_one_file(file_path: Path, translator, limiter, glossary_data: dict, glossary_buckets, cache: dict):
    file_stats = {
        "translated": 0,
        "skipped": 0,
        "cached": 0,
        "failed": 0,
        "candidates": 0,
        "error": False,
    }

    try:
        with file_path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except Exception as e:
        file_stats["error"] = True
        print(f"⚠️ 跳过无法读取的文件: {file_path} | {e}")
        _simple_write_log(f"ERROR load json failed | {file_path} | {e}")
        return file_stats, []

    is_system_lang_file = _simple_is_system_lang_file(file_path)
    keep_original = (not is_system_lang_file) or SIMPLE_SYSTEM_LANG_KEEP_ORIGINAL

    if is_system_lang_file:
        print("检测到系统 lang 文件：放宽键筛选，默认输出纯中文。")

    tasks = _simple_collect_string_tasks(data)
    candidates = []
    for parts, path_str, original in tasks:
        if _simple_contains_chinese(original):
            continue
        source_text = _simple_extract_original(original)
        if _simple_should_translate(path_str, source_text, is_system_lang_file=is_system_lang_file):
            candidates.append((parts, path_str, original))

    file_stats["candidates"] = len(candidates)
    print(f"待处理文本: {len(candidates)} 条")
    _simple_write_log(f"START file={file_path} tasks={len(candidates)} cache={len(cache)}")

    results = {}
    stats_lock = Lock()
    slowest = []
    translated_pairs = []
    file_prefix = str(file_path.resolve()).replace("\\", "/")

    adaptive_mode = SIMPLE_ADAPTIVE_GLOSSARY_MODE if SIMPLE_ADAPTIVE_GLOSSARY_ENABLED else "off"

    if adaptive_mode in {"entry", "entry_batch"}:
        entry_glossary_batch_size = 1 if adaptive_mode == "entry" else SIMPLE_ADAPTIVE_GLOSSARY_ENTRY_BATCH_SIZE
        entry_glossary_pair_lines = []
        entry_glossary_last_path = ""

        def flush_entry_glossary_batch(force: bool = False):
            nonlocal glossary_buckets, entry_glossary_pair_lines, entry_glossary_last_path
            if not entry_glossary_pair_lines:
                return
            if (not force) and len(entry_glossary_pair_lines) < entry_glossary_batch_size:
                return

            extracted = translator.extract_basic_glossary(
                entry_glossary_pair_lines,
                max_terms=SIMPLE_ADAPTIVE_GLOSSARY_MAX_TERMS_PER_BATCH,
            )
            changed = _simple_upsert_adaptive_glossary(glossary_data, extracted)
            if changed > 0:
                _simple_save_glossary_json(SIMPLE_ADAPTIVE_GLOSSARY_OUTPUT_PATH, glossary_data)
                glossary_buckets, _ = _simple_refresh_runtime_glossary(glossary_data)
                _simple_write_log(
                    f"GLOSSARY_UPDATE mode={adaptive_mode} file={file_path.name} entries={len(entry_glossary_pair_lines)} "
                    f"last_path={entry_glossary_last_path} changed={changed}"
                )

            entry_glossary_pair_lines = []
            entry_glossary_last_path = ""

        for item in tqdm(candidates, total=len(candidates), desc="翻译", position=0, leave=True, dynamic_ncols=True):
            parts, path_str, original = item
            source_text = _simple_extract_original(original)
            source_hash = _simple_hash(source_text)
            cache_key = f"{file_prefix}::{path_str}"
            if SIMPLE_CACHE_ENABLED and cache.get(cache_key) == source_hash:
                file_stats["cached"] += 1
                continue

            limiter.wait()
            pre, matched = _simple_apply_glossary(source_text, glossary_buckets)
            t0 = time.time()
            cn = translator.translate(pre, path_str, matched)
            t1 = time.time()
            slowest.append((t1 - t0, f"{file_path.name} | {path_str}"))

            if not cn:
                file_stats["failed"] += 1
                continue

            if SIMPLE_CACHE_ENABLED:
                cache[cache_key] = source_hash

            file_stats["translated"] += 1
            results[tuple(parts)] = _simple_merge_translation(path_str, cn, original, keep_original=keep_original)

            entry_glossary_pair_lines.append(f"EN: {source_text}\nZH: {cn}")
            entry_glossary_last_path = path_str
            flush_entry_glossary_batch()

        flush_entry_glossary_batch(force=True)

        for parts, value in results.items():
            _simple_set_value_at_path(data, list(parts), value)

        if results:
            with file_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        return file_stats, slowest, glossary_buckets

    def worker(item):
        try:
            parts, path_str, original = item
            if _simple_contains_chinese(original):
                with stats_lock:
                    file_stats["skipped"] += 1
                return None
            source_text = _simple_extract_original(original)
            if not _simple_should_translate(path_str, source_text, is_system_lang_file=is_system_lang_file):
                with stats_lock:
                    file_stats["skipped"] += 1
                return None
            source_hash = _simple_hash(source_text)
            cache_key = f"{file_prefix}::{path_str}"
            if SIMPLE_CACHE_ENABLED and cache.get(cache_key) == source_hash:
                with stats_lock:
                    file_stats["cached"] += 1
                return None
            limiter.wait()
            pre, matched = _simple_apply_glossary(source_text, glossary_buckets)
            t0 = time.time()
            cn = translator.translate(pre, path_str, matched)
            t1 = time.time()
            with stats_lock:
                slowest.append((t1 - t0, f"{file_path.name} | {path_str}"))
            if not cn:
                with stats_lock:
                    file_stats["failed"] += 1
                return None
            if SIMPLE_CACHE_ENABLED:
                cache[cache_key] = source_hash
            with stats_lock:
                file_stats["translated"] += 1
                translated_pairs.append((source_text, cn))
            return parts, _simple_merge_translation(path_str, cn, original, keep_original=keep_original)
        except Exception as e:
            _simple_write_log(
                f"ERROR worker failed | {file_path} | {path_str} | {e} | text={_simple_snip(original)}"
            )
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=SIMPLE_MAX_WORKERS) as exe:
        for res in tqdm(exe.map(worker, candidates), total=len(candidates), desc="翻译", position=0, leave=True, dynamic_ncols=True):
            if not res:
                continue
            parts, value = res
            results[tuple(parts)] = value

    for parts, value in results.items():
        _simple_set_value_at_path(data, list(parts), value)

    if results:
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    if SIMPLE_ADAPTIVE_GLOSSARY_ENABLED and SIMPLE_ADAPTIVE_GLOSSARY_MODE == "file" and translated_pairs:
        pair_lines = [f"EN: {en}\nZH: {zh}" for en, zh in translated_pairs[:120]]
        extracted = translator.extract_basic_glossary(
            pair_lines,
            max_terms=SIMPLE_ADAPTIVE_GLOSSARY_MAX_TERMS_PER_BATCH,
        )
        changed = _simple_upsert_adaptive_glossary(glossary_data, extracted)
        if changed > 0:
            _simple_save_glossary_json(SIMPLE_ADAPTIVE_GLOSSARY_OUTPUT_PATH, glossary_data)
            glossary_buckets, _ = _simple_refresh_runtime_glossary(glossary_data)
            _simple_write_log(f"GLOSSARY_UPDATE mode=file file={file_path.name} changed={changed}")

    return file_stats, slowest, glossary_buckets


def main():
    if SIMPLE_ADAPTIVE_GLOSSARY_ENABLED:
        _simple_ensure_glossary_file(SIMPLE_ADAPTIVE_GLOSSARY_OUTPUT_PATH)

    target_files = _simple_collect_target_files()
    if not target_files:
        return

    glossary_data, base_glossary_count, extra_glossary_count, merged_glossary_count, glossary_conflict_count = _simple_build_merged_glossary_data(
        SIMPLE_GLOSSARY_PATH,
        SIMPLE_EFFECTIVE_EXTRA_GLOSSARY_PATH
    )
    glossary_buckets, glossary_count = _simple_load_glossary(
        SIMPLE_GLOSSARY_PATH,
        glossary_data=glossary_data
    )

    print(
        f"术语表加载: 基础={base_glossary_count} 冒险专用={extra_glossary_count} "
        f"合并后键数={merged_glossary_count} 冲突候选={glossary_conflict_count} 有效术语={glossary_count}"
    )
    if SIMPLE_EFFECTIVE_EXTRA_GLOSSARY_PATH and not SIMPLE_EFFECTIVE_EXTRA_GLOSSARY_PATH.exists():
        print(f"⚠️ 冒险专用术语表不存在，已忽略: {SIMPLE_EFFECTIVE_EXTRA_GLOSSARY_PATH}")

    if SIMPLE_ADAPTIVE_GLOSSARY_ENABLED:
        print(
            f"术语增量学习: 已启用 模式={SIMPLE_ADAPTIVE_GLOSSARY_MODE} "
            f"每批最多={SIMPLE_ADAPTIVE_GLOSSARY_MAX_TERMS_PER_BATCH} 输出={SIMPLE_ADAPTIVE_GLOSSARY_OUTPUT_PATH}"
        )
        if SIMPLE_ADAPTIVE_GLOSSARY_MODE == "entry_batch":
            print(f"entry_batch 提取粒度: 每 {SIMPLE_ADAPTIVE_GLOSSARY_ENTRY_BATCH_SIZE} 条 entry 提取一次")
        if SIMPLE_EXTRA_GLOSSARY_PATH is None:
            print(f"增量术语将作为额外术语表参与合并: {SIMPLE_EFFECTIVE_EXTRA_GLOSSARY_PATH}")
        if SIMPLE_ADAPTIVE_GLOSSARY_MODE in {"entry", "entry_batch"} and SIMPLE_MAX_WORKERS > 1:
            print("⚠️ entry/entry_batch 模式将按串行处理词条，以确保新术语能用于后续词条。")

    print(f"目标文件数: {len(target_files)}")
    if SIMPLE_TARGET_FOLDER_PATH:
        mode = "递归" if SIMPLE_TARGET_FOLDER_RECURSIVE else "当前目录"
        print(f"批量模式: 文件夹={SIMPLE_TARGET_FOLDER_PATH} 匹配={SIMPLE_TARGET_FILE_GLOB} 范围={mode}")

    print(
        f"系统lang模式: {SIMPLE_SYSTEM_LANG_MODE} "
        f"(lang保留原文={SIMPLE_SYSTEM_LANG_KEEP_ORIGINAL})"
    )

    cache = _simple_load_cache(SIMPLE_CACHE_PATH)

    translator = _SimpleTranslator()
    limiter = _SimpleRateLimiter(SIMPLE_TARGET_RPM)

    total_stats = {
        "translated": 0,
        "skipped": 0,
        "cached": 0,
        "failed": 0,
        "candidates": 0,
        "file_errors": 0,
    }
    all_slowest = []

    for i, file_path in enumerate(target_files, start=1):
        print(f"\n[{i}/{len(target_files)}] 处理文件: {file_path}")
        file_stats, file_slowest, glossary_buckets = _simple_translate_one_file(
            file_path,
            translator,
            limiter,
            glossary_data,
            glossary_buckets,
            cache,
        )
        total_stats["translated"] += file_stats["translated"]
        total_stats["skipped"] += file_stats["skipped"]
        total_stats["cached"] += file_stats["cached"]
        total_stats["failed"] += file_stats["failed"]
        total_stats["candidates"] += file_stats["candidates"]
        if file_stats["error"]:
            total_stats["file_errors"] += 1
        all_slowest.extend(file_slowest)

        print(
            f"文件统计：候选={file_stats['candidates']} 翻译={file_stats['translated']} "
            f"缓存跳过={file_stats['cached']} 跳过={file_stats['skipped']} 失败={file_stats['failed']}"
        )

    _simple_save_cache(SIMPLE_CACHE_PATH, cache)

    all_slowest.sort(key=lambda x: x[0], reverse=True)
    top_n = all_slowest[:max(SIMPLE_TOP_N_SLOWEST, 0)] if SIMPLE_TOP_N_SLOWEST else []

    print("✅ 完成：批量处理结束")
    print(
        f"统计：文件={len(target_files)} 文件错误={total_stats['file_errors']} 候选={total_stats['candidates']} "
        f"翻译={total_stats['translated']} 跳过={total_stats['skipped']} "
        f"缓存跳过={total_stats['cached']} 失败={total_stats['failed']}"
    )
    if top_n:
        print("最慢条目：")
        for sec, path_str in top_n:
            print(f"  {sec:.2f}s | {path_str}")
    _simple_write_log(
        f"DONE files={len(target_files)} file_errors={total_stats['file_errors']} "
        f"translated={total_stats['translated']} skipped={total_stats['skipped']} "
        f"cached={total_stats['cached']} failed={total_stats['failed']}"
    )


def print_help() -> None:
    print("SF2E translator")
    print("Usage:")
    print("  python 翻译工具/sf2e_translator.py")
    print("")
    print("This script is config-driven and has no CLI options yet.")
    print(f"Config file: {SIMPLE_CONFIG_PATH}")
    print("Key config for system lang files:")
    print("  system_lang_mode: auto | on | off")
    print("  system_lang_keep_original: false (default, output Chinese only in lang mode)")
    print("Edit config first, then run without extra arguments.")

if __name__ == "__main__":
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print_help()
        raise SystemExit(0)
    main()