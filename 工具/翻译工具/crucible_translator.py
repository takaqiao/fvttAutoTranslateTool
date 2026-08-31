import concurrent.futures
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from threading import Event, Lock
from typing import Any

from tqdm import tqdm
from openai import OpenAI

# ================= 简化模式（仅术语替换 + GPT-5.2 翻译） =================

SIMPLE_CONFIG_PATH = Path("crucible_simple_config.json")
SIMPLE_DEFAULT_CONFIG = {
    "openai_api_key": "",
    "openai_base_url": "https://api.openai.com/v1",
    "model": "gpt-5.2",
    "fallback_on_failure_enabled": True,
    "fallback_model_on_failure": "gpt-5-mini",
    "fallback_timeout_seconds": 45,
    "run_mode": "translate",
    "repair_report_path": "untranslated-english-report.json",
    "repair_context_enabled": True,
    "repair_context_max_chars": 12000,
    "repair_force_cn_only": True,
    "repair_max_retries_per_item": 2,
    "repair_entry_batch_size": 12,
    "repair_include_en_preview": True,
    "repair_glossary_expand_enabled": True,
    "repair_glossary_extract_model": "gpt-5-mini",
    "repair_glossary_max_terms_per_batch": 12,
    "optimize_batch_enabled": True,
    "optimize_batch_size": 8,
    "optimize_model_enabled": True,
    "optimize_glossary_patch_only": False,
    "optimize_max_items_per_file": 0,
    "optimize_include_keys_only": [],
    "quality_report_path": "quality_review_report.json",
    "quality_second_pass_enabled": True,
    "quality_strict_semantic_guard": True,
    "quality_require_numeric_consistency": True,
    "quality_fail_on_markup_risk": True,
    "quality_context_scope": "entry",
    "quality_max_items_per_file": 0,
    "quality_include_keys_only": ["name", "description", "public", "private", "condition"],
    "quality_use_bilingual_backup_source": True,
    "quality_glossary_review_enabled": True,
    "quality_glossary_review_apply": False,
    "quality_glossary_review_max_terms_per_run": 0,
    "quality_glossary_review_report_path": "quality_glossary_review_report.json",
    "quality_glossary_review_with_context_enabled": True,
    "quality_glossary_review_parallel_enabled": True,
    "quality_glossary_review_workers": 128,
    "quality_glossary_review_max_examples_per_term": 6,
    "quality_glossary_review_context_max_chars": 1800,
    "quality_glossary_review_context_scan_keys": [
        "name",
        "label",
        "title",
        "caption",
        "description",
        "public",
        "private",
        "text",
        "content",
        "biographyprivate",
        "biographypublic",
        "biographyappearance"
    ],
    "pipeline_target_cn_dirs": [],
    "pipeline_target_file_glob": "*.json",
    "pipeline_target_recursive": False,
    "pipeline_glossary_iterations": 2,
    "pipeline_repair_iterations": 1,
    "pipeline_enable_glossary_phase": True,
    "pipeline_enable_repair_phase": True,
    "pipeline_glossary_include_keys_only": [
        "name",
        "label",
        "title",
        "caption",
        "tokenname",
        "prototypetoken",
        "actionname"
    ],
    "pipeline_glossary_max_chars_per_side": 260,
    "pipeline_glossary_dedup_enabled": True,
    "target_json_path": "system/crucible/en.json",
    "target_folder_path": "",
    "target_file_glob": "*.json",
    "target_folder_recursive": False,
    "adaptive_glossary_enabled": False,
    "adaptive_glossary_during_translation_enabled": True,
    "adaptive_glossary_mode": "file",
    "adaptive_glossary_entry_batch_size": 5,
    "adaptive_glossary_entry_parallel_enabled": False,
    "adaptive_glossary_entry_non_blocking_enabled": True,
    "adaptive_glossary_extract_model": "gpt-5-mini",
    "adaptive_glossary_extract_max_pending_batches": 64,
    "adaptive_glossary_output_path": "glossary_adaptive_crucible.json",
    "adaptive_glossary_max_terms_per_batch": 8,
    "adaptive_glossary_overwrite_existing": False,
    "extract_only_pair_batch_size": 120,
    "extract_only_max_pairs_per_file": 1200,
    "extract_only_parallel_enabled": True,
    "extract_only_max_workers": 16,
    "extract_only_target_rpm": 1200,
    "extract_only_progress_cache_enabled": True,
    "extract_only_progress_cache_path": "crucible_extract_only_progress.json",
    "glossary_path": "glossary.json",
    "extra_glossary_path": "",
    "prompt_profile": "default",
    "system_lang_mode": "auto",
    "system_lang_keep_original": False,
    "lang_file_path_markers": ["/lang/", "/i18n/"],
    "lang_file_names": ["en.json", "cn.json", "zh_hans.json", "zh-hans.json", "zh_cn.json", "zh-cn.json"],
    "allow_keys_only": [],
    "allow_keys_extra": [],
    "force_keep_original_keys": [],
    "force_cn_only_keys": [],
    "sync_translation_enabled": False,
    "sync_translation_source_key": "actionname",
    "sync_translation_target_key": "actioneffectname",
    "sync_translation_only_reuse": True,
    "translate_path_include_substrings": [],
    "translate_path_skip_substrings": [".mapping"],
    "strict_coverage_enabled": False,
    "strict_coverage_max_rounds": 2,
    "strict_coverage_relaxed_allow_keys": True,
    "strict_coverage_max_items_per_round": 0,
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
    "log_path": "crucible_simple_run.log",
    "cache_path": "crucible_simple_cache.json",
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

SIMPLE_TARGET_JSON_PATH = Path(SIMPLE_CONFIG.get("target_json_path", "system/crucible/en.json"))
SIMPLE_TARGET_FOLDER_PATH_RAW = str(SIMPLE_CONFIG.get("target_folder_path", "")).strip()
SIMPLE_TARGET_FOLDER_PATH = Path(SIMPLE_TARGET_FOLDER_PATH_RAW) if SIMPLE_TARGET_FOLDER_PATH_RAW else None
SIMPLE_TARGET_FILE_GLOB = str(SIMPLE_CONFIG.get("target_file_glob", "*.json")).strip() or "*.json"
SIMPLE_TARGET_FOLDER_RECURSIVE = bool(SIMPLE_CONFIG.get("target_folder_recursive", False))
SIMPLE_RUN_MODE = str(SIMPLE_CONFIG.get("run_mode", "translate")).strip().lower()
if SIMPLE_RUN_MODE in {"extract", "extract_only", "glossary_only", "glossary_extract_only"}:
    SIMPLE_RUN_MODE = "extract_glossary_only"
if SIMPLE_RUN_MODE in {"repair", "repair_from_report", "repair_context"}:
    SIMPLE_RUN_MODE = "repair_from_report"
if SIMPLE_RUN_MODE in {"optimize", "optimize_existing", "post_optimize", "tune"}:
    SIMPLE_RUN_MODE = "optimize_existing"
if SIMPLE_RUN_MODE in {"quality", "quality_review", "quality_repair", "quality_review_and_repair"}:
    SIMPLE_RUN_MODE = "quality_review_and_repair"
if SIMPLE_RUN_MODE in {"full", "full_pipeline", "quality_full", "quality_full_pipeline"}:
    SIMPLE_RUN_MODE = "quality_full_pipeline"
if SIMPLE_RUN_MODE not in {
    "translate",
    "extract_glossary_only",
    "repair_from_report",
    "optimize_existing",
    "quality_review_and_repair",
    "quality_full_pipeline",
}:
    SIMPLE_RUN_MODE = "translate"
SIMPLE_REPAIR_REPORT_PATH_RAW = str(SIMPLE_CONFIG.get("repair_report_path", "untranslated-english-report.json")).strip()
SIMPLE_REPAIR_REPORT_PATH = Path(SIMPLE_REPAIR_REPORT_PATH_RAW) if SIMPLE_REPAIR_REPORT_PATH_RAW else Path("untranslated-english-report.json")
SIMPLE_REPAIR_CONTEXT_ENABLED = bool(SIMPLE_CONFIG.get("repair_context_enabled", True))
SIMPLE_REPAIR_CONTEXT_MAX_CHARS = max(500, int(SIMPLE_CONFIG.get("repair_context_max_chars", 12000)))
SIMPLE_REPAIR_FORCE_CN_ONLY = bool(SIMPLE_CONFIG.get("repair_force_cn_only", True))
SIMPLE_REPAIR_MAX_RETRIES_PER_ITEM = max(1, int(SIMPLE_CONFIG.get("repair_max_retries_per_item", 2)))
SIMPLE_REPAIR_ENTRY_BATCH_SIZE = max(1, int(SIMPLE_CONFIG.get("repair_entry_batch_size", 12)))
SIMPLE_REPAIR_INCLUDE_EN_PREVIEW = bool(SIMPLE_CONFIG.get("repair_include_en_preview", True))
SIMPLE_REPAIR_GLOSSARY_EXPAND_ENABLED = bool(SIMPLE_CONFIG.get("repair_glossary_expand_enabled", True))
SIMPLE_REPAIR_GLOSSARY_EXTRACT_MODEL_ID = str(
    SIMPLE_CONFIG.get("repair_glossary_extract_model", SIMPLE_FALLBACK_MODEL_ID or SIMPLE_MODEL_ID) or ""
).strip() or SIMPLE_MODEL_ID
SIMPLE_REPAIR_GLOSSARY_MAX_TERMS_PER_BATCH = max(
    1, int(SIMPLE_CONFIG.get("repair_glossary_max_terms_per_batch", 12))
)
SIMPLE_OPTIMIZE_BATCH_ENABLED = bool(SIMPLE_CONFIG.get("optimize_batch_enabled", True))
SIMPLE_OPTIMIZE_BATCH_SIZE = max(1, int(SIMPLE_CONFIG.get("optimize_batch_size", 8)))
SIMPLE_OPTIMIZE_MODEL_ENABLED = bool(SIMPLE_CONFIG.get("optimize_model_enabled", True))
SIMPLE_OPTIMIZE_GLOSSARY_PATCH_ONLY = bool(SIMPLE_CONFIG.get("optimize_glossary_patch_only", False))
SIMPLE_OPTIMIZE_MAX_ITEMS_PER_FILE = max(0, int(SIMPLE_CONFIG.get("optimize_max_items_per_file", 0)))
SIMPLE_OPTIMIZE_INCLUDE_KEYS_ONLY = {
    str(x).strip().casefold()
    for x in SIMPLE_CONFIG.get("optimize_include_keys_only", [])
    if str(x).strip()
}
SIMPLE_QUALITY_REPORT_PATH_RAW = str(SIMPLE_CONFIG.get("quality_report_path", "quality_review_report.json")).strip()
SIMPLE_QUALITY_REPORT_PATH = (
    Path(SIMPLE_QUALITY_REPORT_PATH_RAW) if SIMPLE_QUALITY_REPORT_PATH_RAW else Path("quality_review_report.json")
)
SIMPLE_QUALITY_SECOND_PASS_ENABLED = bool(SIMPLE_CONFIG.get("quality_second_pass_enabled", True))
SIMPLE_QUALITY_STRICT_SEMANTIC_GUARD = bool(SIMPLE_CONFIG.get("quality_strict_semantic_guard", True))
SIMPLE_QUALITY_REQUIRE_NUMERIC_CONSISTENCY = bool(
    SIMPLE_CONFIG.get("quality_require_numeric_consistency", True)
)
SIMPLE_QUALITY_FAIL_ON_MARKUP_RISK = bool(SIMPLE_CONFIG.get("quality_fail_on_markup_risk", True))
SIMPLE_QUALITY_CONTEXT_SCOPE = str(SIMPLE_CONFIG.get("quality_context_scope", "entry")).strip().lower()
if SIMPLE_QUALITY_CONTEXT_SCOPE not in {"entry", "file"}:
    SIMPLE_QUALITY_CONTEXT_SCOPE = "entry"
SIMPLE_QUALITY_MAX_ITEMS_PER_FILE = max(0, int(SIMPLE_CONFIG.get("quality_max_items_per_file", 0)))
SIMPLE_QUALITY_INCLUDE_KEYS_ONLY = {
    str(x).strip().casefold()
    for x in SIMPLE_CONFIG.get("quality_include_keys_only", ["name", "description", "public", "private", "condition"])
    if str(x).strip()
}
SIMPLE_QUALITY_USE_BILINGUAL_BACKUP_SOURCE = bool(
    SIMPLE_CONFIG.get("quality_use_bilingual_backup_source", True)
)
SIMPLE_QUALITY_GLOSSARY_REVIEW_ENABLED = bool(
    SIMPLE_CONFIG.get("quality_glossary_review_enabled", True)
)
SIMPLE_QUALITY_GLOSSARY_REVIEW_APPLY = bool(
    SIMPLE_CONFIG.get("quality_glossary_review_apply", False)
)
SIMPLE_QUALITY_GLOSSARY_REVIEW_MAX_TERMS_PER_RUN = max(
    0, int(SIMPLE_CONFIG.get("quality_glossary_review_max_terms_per_run", 0))
)
SIMPLE_QUALITY_GLOSSARY_REVIEW_REPORT_PATH_RAW = str(
    SIMPLE_CONFIG.get("quality_glossary_review_report_path", "quality_glossary_review_report.json")
).strip()
SIMPLE_QUALITY_GLOSSARY_REVIEW_REPORT_PATH = (
    Path(SIMPLE_QUALITY_GLOSSARY_REVIEW_REPORT_PATH_RAW)
    if SIMPLE_QUALITY_GLOSSARY_REVIEW_REPORT_PATH_RAW
    else Path("quality_glossary_review_report.json")
)
SIMPLE_QUALITY_GLOSSARY_REVIEW_WITH_CONTEXT_ENABLED = bool(
    SIMPLE_CONFIG.get("quality_glossary_review_with_context_enabled", True)
)
SIMPLE_QUALITY_GLOSSARY_REVIEW_PARALLEL_ENABLED = bool(
    SIMPLE_CONFIG.get("quality_glossary_review_parallel_enabled", True)
)
SIMPLE_QUALITY_GLOSSARY_REVIEW_WORKERS = max(
    1,
    int(SIMPLE_CONFIG.get("quality_glossary_review_workers", 16)),
)
SIMPLE_QUALITY_GLOSSARY_REVIEW_MAX_EXAMPLES_PER_TERM = max(
    1,
    min(8, int(SIMPLE_CONFIG.get("quality_glossary_review_max_examples_per_term", 8))),
)
SIMPLE_QUALITY_GLOSSARY_REVIEW_CONTEXT_MAX_CHARS = max(
    300,
    int(SIMPLE_CONFIG.get("quality_glossary_review_context_max_chars", 1800)),
)
SIMPLE_QUALITY_GLOSSARY_REVIEW_CONTEXT_SCAN_KEYS = {
    str(x).strip().casefold()
    for x in SIMPLE_CONFIG.get(
        "quality_glossary_review_context_scan_keys",
        [
            "name",
            "label",
            "title",
            "caption",
            "description",
            "public",
            "private",
            "text",
            "content",
            "biographyprivate",
            "biographypublic",
            "biographyappearance",
        ],
    )
    if str(x).strip()
}
SIMPLE_PIPELINE_TARGET_CN_DIRS = [
    str(x).strip()
    for x in SIMPLE_CONFIG.get("pipeline_target_cn_dirs", [])
    if str(x).strip()
]
SIMPLE_PIPELINE_TARGET_FILE_GLOB = str(SIMPLE_CONFIG.get("pipeline_target_file_glob", "*.json")).strip() or "*.json"
SIMPLE_PIPELINE_TARGET_RECURSIVE = bool(SIMPLE_CONFIG.get("pipeline_target_recursive", False))
SIMPLE_PIPELINE_GLOSSARY_ITERATIONS = max(1, int(SIMPLE_CONFIG.get("pipeline_glossary_iterations", 2)))
SIMPLE_PIPELINE_REPAIR_ITERATIONS = max(1, int(SIMPLE_CONFIG.get("pipeline_repair_iterations", 1)))
SIMPLE_PIPELINE_ENABLE_GLOSSARY_PHASE = bool(SIMPLE_CONFIG.get("pipeline_enable_glossary_phase", True))
SIMPLE_PIPELINE_ENABLE_REPAIR_PHASE = bool(SIMPLE_CONFIG.get("pipeline_enable_repair_phase", True))
SIMPLE_PIPELINE_GLOSSARY_INCLUDE_KEYS_ONLY = {
    str(x).strip().casefold()
    for x in SIMPLE_CONFIG.get(
        "pipeline_glossary_include_keys_only",
        ["name", "label", "title", "caption", "tokenname", "prototypetoken", "actionname"],
    )
    if str(x).strip()
}
SIMPLE_PIPELINE_GLOSSARY_MAX_CHARS_PER_SIDE = max(
    0, int(SIMPLE_CONFIG.get("pipeline_glossary_max_chars_per_side", 260))
)
SIMPLE_PIPELINE_GLOSSARY_DEDUP_ENABLED = bool(
    SIMPLE_CONFIG.get("pipeline_glossary_dedup_enabled", True)
)
SIMPLE_ADAPTIVE_GLOSSARY_ENABLED = bool(SIMPLE_CONFIG.get("adaptive_glossary_enabled", False))
SIMPLE_ADAPTIVE_GLOSSARY_DURING_TRANSLATION_ENABLED = bool(
    SIMPLE_CONFIG.get("adaptive_glossary_during_translation_enabled", True)
)
SIMPLE_ADAPTIVE_GLOSSARY_MODE = str(SIMPLE_CONFIG.get("adaptive_glossary_mode", "file")).strip().lower()
if SIMPLE_ADAPTIVE_GLOSSARY_MODE == "entry_patch":
    SIMPLE_ADAPTIVE_GLOSSARY_MODE = "entry_batch"
if SIMPLE_ADAPTIVE_GLOSSARY_MODE not in {"file", "entry", "entry_batch"}:
    SIMPLE_ADAPTIVE_GLOSSARY_MODE = "file"
SIMPLE_ADAPTIVE_GLOSSARY_ENTRY_BATCH_SIZE = max(1, int(SIMPLE_CONFIG.get("adaptive_glossary_entry_batch_size", 5)))
SIMPLE_ADAPTIVE_GLOSSARY_ENTRY_PARALLEL_ENABLED = bool(
    SIMPLE_CONFIG.get("adaptive_glossary_entry_parallel_enabled", False)
)
SIMPLE_ADAPTIVE_GLOSSARY_ENTRY_NON_BLOCKING_ENABLED = bool(
    SIMPLE_CONFIG.get("adaptive_glossary_entry_non_blocking_enabled", True)
)
SIMPLE_ADAPTIVE_GLOSSARY_EXTRACT_MODEL_ID = str(
    SIMPLE_CONFIG.get(
        "adaptive_glossary_extract_model",
        SIMPLE_FALLBACK_MODEL_ID or SIMPLE_MODEL_ID,
    )
    or ""
).strip() or SIMPLE_MODEL_ID
SIMPLE_ADAPTIVE_GLOSSARY_EXTRACT_MAX_PENDING_BATCHES = max(
    1,
    int(SIMPLE_CONFIG.get("adaptive_glossary_extract_max_pending_batches", 64)),
)
SIMPLE_ADAPTIVE_GLOSSARY_OUTPUT_PATH_RAW = str(SIMPLE_CONFIG.get("adaptive_glossary_output_path", "glossary_adaptive_crucible.json")).strip()
SIMPLE_ADAPTIVE_GLOSSARY_OUTPUT_PATH = Path(SIMPLE_ADAPTIVE_GLOSSARY_OUTPUT_PATH_RAW) if SIMPLE_ADAPTIVE_GLOSSARY_OUTPUT_PATH_RAW else Path("glossary_adaptive_crucible.json")
SIMPLE_ADAPTIVE_GLOSSARY_MAX_TERMS_PER_BATCH = max(1, int(SIMPLE_CONFIG.get("adaptive_glossary_max_terms_per_batch", 8)))
SIMPLE_ADAPTIVE_GLOSSARY_OVERWRITE_EXISTING = bool(SIMPLE_CONFIG.get("adaptive_glossary_overwrite_existing", False))
SIMPLE_EXTRACT_ONLY_PAIR_BATCH_SIZE = max(1, int(SIMPLE_CONFIG.get("extract_only_pair_batch_size", 120)))
SIMPLE_EXTRACT_ONLY_MAX_PAIRS_PER_FILE = int(SIMPLE_CONFIG.get("extract_only_max_pairs_per_file", 1200))
if SIMPLE_EXTRACT_ONLY_MAX_PAIRS_PER_FILE < 0:
    SIMPLE_EXTRACT_ONLY_MAX_PAIRS_PER_FILE = 0
SIMPLE_EXTRACT_ONLY_PARALLEL_ENABLED = bool(SIMPLE_CONFIG.get("extract_only_parallel_enabled", True))
SIMPLE_EXTRACT_ONLY_MAX_WORKERS = max(1, int(SIMPLE_CONFIG.get("extract_only_max_workers", 16)))
SIMPLE_EXTRACT_ONLY_TARGET_RPM = max(1, int(SIMPLE_CONFIG.get("extract_only_target_rpm", 1200)))
SIMPLE_EXTRACT_ONLY_PROGRESS_CACHE_ENABLED = bool(
    SIMPLE_CONFIG.get("extract_only_progress_cache_enabled", True)
)
SIMPLE_EXTRACT_ONLY_PROGRESS_CACHE_PATH_RAW = str(
    SIMPLE_CONFIG.get("extract_only_progress_cache_path", "crucible_extract_only_progress.json")
).strip()
SIMPLE_EXTRACT_ONLY_PROGRESS_CACHE_PATH = (
    Path(SIMPLE_EXTRACT_ONLY_PROGRESS_CACHE_PATH_RAW)
    if SIMPLE_EXTRACT_ONLY_PROGRESS_CACHE_PATH_RAW
    else Path("crucible_extract_only_progress.json")
)
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
SIMPLE_LANG_FILE_PATH_MARKERS = [
    str(x).strip().casefold() for x in SIMPLE_CONFIG.get("lang_file_path_markers", ["/lang/", "/i18n/"]) if str(x).strip()
]
SIMPLE_LANG_FILE_NAMES = {
    str(x).strip().casefold() for x in SIMPLE_CONFIG.get(
        "lang_file_names",
        ["en.json", "cn.json", "zh_hans.json", "zh-hans.json", "zh_cn.json", "zh-cn.json"],
    ) if str(x).strip()
}
SIMPLE_TRANSLATE_PATH_INCLUDE_SUBSTRINGS = [
    str(x).strip().casefold() for x in SIMPLE_CONFIG.get("translate_path_include_substrings", []) if str(x).strip()
]
SIMPLE_TRANSLATE_PATH_SKIP_SUBSTRINGS = [
    str(x).strip().casefold() for x in SIMPLE_CONFIG.get("translate_path_skip_substrings", [".mapping"]) if str(x).strip()
]
SIMPLE_ALLOW_KEYS_EXTRA = {
    str(x).strip().casefold() for x in SIMPLE_CONFIG.get("allow_keys_extra", []) if str(x).strip()
}
SIMPLE_ALLOW_KEYS_ONLY = {
    str(x).strip().casefold() for x in SIMPLE_CONFIG.get("allow_keys_only", []) if str(x).strip()
}
SIMPLE_FORCE_KEEP_ORIGINAL_KEYS = {
    str(x).strip().casefold() for x in SIMPLE_CONFIG.get("force_keep_original_keys", []) if str(x).strip()
}
SIMPLE_FORCE_CN_ONLY_KEYS = {
    str(x).strip().casefold() for x in SIMPLE_CONFIG.get("force_cn_only_keys", []) if str(x).strip()
}
SIMPLE_SYNC_TRANSLATION_ENABLED = bool(SIMPLE_CONFIG.get("sync_translation_enabled", False))
SIMPLE_SYNC_TRANSLATION_SOURCE_KEY = str(SIMPLE_CONFIG.get("sync_translation_source_key", "actionname")).strip().casefold()
SIMPLE_SYNC_TRANSLATION_TARGET_KEY = str(SIMPLE_CONFIG.get("sync_translation_target_key", "actioneffectname")).strip().casefold()
SIMPLE_SYNC_TRANSLATION_ONLY_REUSE = bool(SIMPLE_CONFIG.get("sync_translation_only_reuse", True))
SIMPLE_STRICT_COVERAGE_ENABLED = bool(SIMPLE_CONFIG.get("strict_coverage_enabled", True))
SIMPLE_STRICT_COVERAGE_MAX_ROUNDS = max(1, int(SIMPLE_CONFIG.get("strict_coverage_max_rounds", 2)))
SIMPLE_STRICT_COVERAGE_RELAXED_ALLOW_KEYS = bool(
    SIMPLE_CONFIG.get("strict_coverage_relaxed_allow_keys", True)
)
SIMPLE_STRICT_COVERAGE_MAX_ITEMS_PER_ROUND = int(
    SIMPLE_CONFIG.get("strict_coverage_max_items_per_round", 0)
)
if SIMPLE_STRICT_COVERAGE_MAX_ITEMS_PER_ROUND < 0:
    SIMPLE_STRICT_COVERAGE_MAX_ITEMS_PER_ROUND = 0

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
SIMPLE_LOG_PATH = Path(SIMPLE_CONFIG.get("log_path", "crucible_simple_run.log"))
SIMPLE_CACHE_PATH = Path(SIMPLE_CONFIG.get("cache_path", "crucible_simple_cache.json"))
SIMPLE_CACHE_ENABLED = bool(SIMPLE_CONFIG.get("cache_enabled", True))
SIMPLE_TOP_N_SLOWEST = int(SIMPLE_CONFIG.get("top_n_slowest", 20))
SIMPLE_RISKY_MAX_WORKERS = 32
SIMPLE_RISKY_TARGET_RPM = 3000

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


def _simple_load_extract_only_progress(path: Path):
    if not SIMPLE_EXTRACT_ONLY_PROGRESS_CACHE_ENABLED or not path.exists():
        return {"done_batches": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            done = data.get("done_batches")
            if isinstance(done, dict):
                return data
    except Exception:
        pass
    return {"done_batches": {}}


def _simple_save_extract_only_progress(path: Path, progress: dict):
    if not SIMPLE_EXTRACT_ONLY_PROGRESS_CACHE_ENABLED:
        return
    try:
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        _simple_write_log(f"ERROR save extract-only progress failed | {path} | {e}")


class _SimpleFatalAPIError(RuntimeError):
    """Unrecoverable API error; abort current run to avoid wasteful retries."""


def _simple_error_text(err: Exception) -> str:
    if err is None:
        return ""
    text = str(err)
    body = getattr(err, "body", None)
    if body:
        try:
            body_text = body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)
        except Exception:
            body_text = repr(body)
        if body_text and body_text not in text:
            text = f"{text} | {body_text}"
    return text


def _simple_extract_error_status_code(err: Exception) -> int | None:
    status_code = getattr(err, "status_code", None)
    if isinstance(status_code, int):
        return status_code

    response = getattr(err, "response", None)
    if response is not None:
        response_code = getattr(response, "status_code", None)
        if isinstance(response_code, int):
            return response_code

    text = _simple_error_text(err)
    match = re.search(r"Error code:\s*(\d{3})", text, flags=re.I)
    if match:
        return int(match.group(1))
    return None


def _simple_extract_error_code(err: Exception) -> str:
    for attr in ("code", "error_code"):
        value = getattr(err, attr, None)
        if value:
            return str(value).strip().casefold()

    body = getattr(err, "body", None)
    if isinstance(body, dict):
        error_obj = body.get("error") if isinstance(body.get("error"), dict) else body
        for key in ("code", "type"):
            value = error_obj.get(key) if isinstance(error_obj, dict) else None
            if value:
                return str(value).strip().casefold()

    text = _simple_error_text(err)
    patterns = (
        r"'code':\s*'([^']+)'",
        r'"code":\s*"([^\"]+)"',
        r"'type':\s*'([^']+)'",
        r'"type":\s*"([^\"]+)"',
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip().casefold()
    return ""


def _simple_is_insufficient_quota_error(err: Exception) -> bool:
    code = _simple_extract_error_code(err)
    if code in {"insufficient_quota", "billing_hard_limit_reached"}:
        return True

    text = _simple_error_text(err).casefold()
    return (
        "insufficient_quota" in text
        or "exceeded your current quota" in text
        or "billing_hard_limit_reached" in text
        or ("quota" in text and "billing details" in text)
    )


def _simple_is_fatal_api_error(err: Exception) -> bool:
    if _simple_is_insufficient_quota_error(err):
        return True

    code = _simple_extract_error_code(err)
    if code in {
        "invalid_api_key",
        "incorrect_api_key_provided",
        "authentication_error",
        "account_deactivated",
        "organization_deactivated",
    }:
        return True

    text = _simple_error_text(err).casefold()
    fatal_markers = (
        "invalid api key",
        "incorrect api key",
        "authentication",
        "account deactivated",
        "organization deactivated",
    )
    return any(marker in text for marker in fatal_markers)


def _simple_is_retriable_api_error(err: Exception) -> bool:
    if _simple_is_fatal_api_error(err):
        return False

    status_code = _simple_extract_error_status_code(err)
    if status_code is not None:
        if status_code in {408, 409, 429}:
            return True
        return status_code >= 500

    text = _simple_error_text(err).casefold()
    retriable_markers = (
        "connection error",
        "timed out",
        "timeout",
        "rate limit",
        "server_error",
        "temporar",
    )
    return any(marker in text for marker in retriable_markers)


def _simple_should_try_fallback(last_error: Exception | None) -> bool:
    if last_error is None:
        return True
    if _simple_is_fatal_api_error(last_error):
        return False

    text = _simple_error_text(last_error).casefold()
    status_code = _simple_extract_error_status_code(last_error)
    if status_code is not None and 400 <= status_code < 500 and status_code not in {408, 409, 429}:
        # Missing primary model can still be salvaged by fallback model.
        if "model" in text and ("not found" in text or "does not exist" in text):
            return True
        code = _simple_extract_error_code(last_error)
        if code in {"model_not_found", "invalid_model"}:
            return True
        return False

    return True


def _simple_print_risk_warnings():
    if SIMPLE_MAX_WORKERS > SIMPLE_RISKY_MAX_WORKERS:
        print(
            f"⚠️ 风险配置：max_workers={SIMPLE_MAX_WORKERS} 偏高，"
            "当配额不足或网络抖动时会放大失败请求。"
        )
    if SIMPLE_TARGET_RPM > SIMPLE_RISKY_TARGET_RPM:
        print(
            f"⚠️ 风险配置：target_rpm={SIMPLE_TARGET_RPM} 偏高，"
            "建议先降速排查，避免短时间触发大量失败。"
        )
    if not SIMPLE_CACHE_ENABLED:
        print("⚠️ 风险配置：cache_enabled=false，重跑会重复请求已翻译文本。")

# === 简化模式筛选规则 ===
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
    "notes", "title", "header", "navName", "gmnote", "descriptionprivate", "descriptionpublic", "biographyprivate",
    "hint", "summary", "subtitle", "message", "tooltip", "placeholder",
    "instruction", "instructions", "intro", "outro", "footer", "tagline", "warning","public","private","condition",
    "hint"
}

SIMPLE_INLINE_MERGE_KEYS = {"name", "tokenName", "prototypeToken"}

SIMPLE_SKIP_KEYS_NORM = {k.casefold() for k in SIMPLE_SKIP_KEYS}
SIMPLE_ALLOW_KEYS_NORM = {k.casefold() for k in SIMPLE_ALLOW_KEYS} | SIMPLE_ALLOW_KEYS_EXTRA
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


def _simple_plain_text_for_glossary(text: str) -> str:
    cleaned = _simple_strip_codes_for_lang_detect(str(text or ""))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


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


def _simple_resolve_keep_original(path_str: str, default_keep_original: bool) -> bool:
    last_key_norm = _simple_extract_last_key(path_str).casefold()
    if last_key_norm in SIMPLE_FORCE_KEEP_ORIGINAL_KEYS:
        return True
    if last_key_norm in SIMPLE_FORCE_CN_ONLY_KEYS:
        return False
    return default_keep_original


def _simple_sync_key(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().casefold()


def _simple_is_nochange_translation(source_text: str, merged_value: str, is_system_lang_file: bool) -> bool:
    src = re.sub(r"\s+", " ", str(source_text or "")).strip().casefold()
    dst = re.sub(r"\s+", " ", str(merged_value or "")).strip().casefold()
    if not dst:
        return True
    if src == dst:
        return True
    # For system lang files we expect Chinese output by default.
    if is_system_lang_file and not _simple_contains_chinese(merged_value or ""):
        return True
    return False


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
    if name in SIMPLE_LANG_FILE_NAMES and any(marker in path_norm for marker in SIMPLE_LANG_FILE_PATH_MARKERS):
        return True
    if name == "zh_hans.json" or name.endswith("-zh_hans.json"):
        return True

    return False


def _simple_should_translate(
    path_str: str,
    value: str,
    is_system_lang_file: bool = False,
    allow_key_filter: bool = True,
) -> bool:
    if not isinstance(value, str):
        return False
    if _simple_contains_chinese(value):
        return False
    if _simple_is_index_like_page_name(path_str, value):
        return False
    if SIMPLE_PROMPT_PROFILE == "dscryb" and re.search(r"\.(Image|Text)\.name$", path_str):
        return False
    path_norm = path_str.casefold()
    if SIMPLE_TRANSLATE_PATH_INCLUDE_SUBSTRINGS and not any(
        s in path_norm for s in SIMPLE_TRANSLATE_PATH_INCLUDE_SUBSTRINGS
    ):
        return False
    if any(s in path_norm for s in SIMPLE_TRANSLATE_PATH_SKIP_SUBSTRINGS):
        return False
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
    # Hard whitelist: when configured, only these keys are translatable.
    if SIMPLE_ALLOW_KEYS_ONLY and last_key_norm not in SIMPLE_ALLOW_KEYS_ONLY:
        return False
    if (
        allow_key_filter
        and (not is_system_lang_file)
        and SIMPLE_ALLOW_KEYS_NORM
        and last_key_norm not in SIMPLE_ALLOW_KEYS_NORM
    ):
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
            pairs = [f"{e} -> {c}" for e, c in glossary_terms]
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
                "You are a professional Crucible TRPG translator. "
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
                error_text = _simple_error_text(e)
                if _simple_is_fatal_api_error(e):
                    _simple_write_log(f"FATAL translate abort | {path_str} | {error_text}")
                    raise _SimpleFatalAPIError(f"{path_str} | {error_text}") from e

                if not _simple_is_retriable_api_error(e):
                    _simple_write_log(
                        f"NO_RETRY non-retriable | timeout={primary_timeout_seconds:.1f}s | {path_str} | {error_text}"
                    )
                    break

                if attempt == SIMPLE_MAX_RETRIES:
                    break
                _simple_write_log(
                    f"RETRY {attempt}/{SIMPLE_MAX_RETRIES} | timeout={primary_timeout_seconds:.1f}s | {path_str} | {error_text}"
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
            and _simple_should_try_fallback(last_error)
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
                if _simple_is_fatal_api_error(fallback_error):
                    error_text = _simple_error_text(fallback_error)
                    _simple_write_log(f"FATAL fallback abort | {path_str} | {error_text}")
                    raise _SimpleFatalAPIError(f"{path_str} | {error_text}") from fallback_error

        final_error_text = _simple_error_text(last_error) if last_error else "unknown error"
        print(f"⚠️ 翻译失败: {path_str} | {final_error_text}")
        _simple_write_log(
            f"ERROR translate failed | {path_str} | {final_error_text} | text={_simple_snip(text)}"
        )
        return ""

    def repair_mixed_text(
        self,
        current_cn_text: str,
        path_str: str,
        glossary_terms=None,
        en_preview: str = "",
        english_words: list[str] | None = None,
        entry_context_json: str = "",
    ) -> str:
        glossary_hint = ""
        if glossary_terms:
            pairs = [f"{e} -> {c}" for e, c in glossary_terms]
            glossary_hint = (
                " Contextual glossary candidates (apply only with matching sense): "
                + "; ".join(pairs)
                + "."
            )

        words_hint = ", ".join([str(w).strip() for w in (english_words or []) if str(w).strip()][:30])
        entry_context_json = str(entry_context_json or "").strip()
        en_preview = str(en_preview or "").strip()

        instructions = (
            "You are a professional Chinese localization post-editor for Crucible TRPG. "
            "Task: repair mixed CN/EN text by translating only the remaining English fragments into Simplified Chinese. "
            "Keep existing Chinese wording and sentence structure as much as possible. "
            "Do not rewrite the whole paragraph unless required for local coherence. "
            "Do not add or remove gameplay facts. "
            "Preserve all code-like tokens and formatting exactly as-is, including @UUID[...] , @Compendium[...] , @Condition[...] , [[...]] , HTML tags/attributes, punctuation, numbers, whitespace, and line breaks. "
            "Output only the repaired full text string. "
            + glossary_hint
        )

        input_parts = [
            f"PATH: {path_str}",
            "CURRENT_CN_TEXT:",
            current_cn_text,
        ]
        if SIMPLE_REPAIR_INCLUDE_EN_PREVIEW and en_preview:
            input_parts.extend(["SOURCE_EN_PREVIEW:", en_preview])
        if words_hint:
            input_parts.append(f"ENGLISH_HINT_WORDS: {words_hint}")
        if SIMPLE_REPAIR_CONTEXT_ENABLED and entry_context_json:
            input_parts.extend(["ENTRY_CONTEXT_JSON:", entry_context_json])

        request_text = "\n".join(input_parts)
        timeout_seconds = _simple_compute_timeout_seconds(
            request_text,
            SIMPLE_REQUEST_TIMEOUT_SECONDS,
        )

        last_error = None
        for attempt in range(1, SIMPLE_REPAIR_MAX_RETRIES_PER_ITEM + 1):
            try:
                return self._translate_once(
                    SIMPLE_MODEL_ID,
                    request_text,
                    instructions,
                    timeout_seconds,
                )
            except Exception as e:
                last_error = e
                if _simple_is_fatal_api_error(e):
                    error_text = _simple_error_text(e)
                    _simple_write_log(f"FATAL repair abort | {path_str} | {error_text}")
                    raise _SimpleFatalAPIError(f"{path_str} | {error_text}") from e

                if not _simple_is_retriable_api_error(e):
                    _simple_write_log(
                        f"NO_RETRY repair non-retriable | timeout={timeout_seconds:.1f}s | {path_str} | {_simple_error_text(e)}"
                    )
                    break

                if attempt == SIMPLE_REPAIR_MAX_RETRIES_PER_ITEM:
                    break

                delay = min(
                    SIMPLE_RETRY_DELAY_MAX_SECONDS,
                    SIMPLE_RETRY_DELAY_BASE_SECONDS * (2 ** (attempt - 1)),
                )
                time.sleep(max(0.0, delay))

        can_fallback = (
            SIMPLE_FALLBACK_ON_FAILURE_ENABLED
            and bool(SIMPLE_FALLBACK_MODEL_ID)
            and SIMPLE_FALLBACK_MODEL_ID != SIMPLE_MODEL_ID
            and _simple_should_try_fallback(last_error)
        )
        if can_fallback:
            fallback_timeout_seconds = _simple_compute_timeout_seconds(
                request_text,
                SIMPLE_FALLBACK_TIMEOUT_SECONDS,
            )
            try:
                fallback_cn = self._translate_once(
                    SIMPLE_FALLBACK_MODEL_ID,
                    request_text,
                    instructions,
                    fallback_timeout_seconds,
                )
                if fallback_cn:
                    return fallback_cn
            except Exception as fallback_error:
                if _simple_is_fatal_api_error(fallback_error):
                    error_text = _simple_error_text(fallback_error)
                    raise _SimpleFatalAPIError(f"{path_str} | {error_text}") from fallback_error

        error_text = _simple_error_text(last_error) if last_error else "unknown error"
        _simple_write_log(f"ERROR repair failed | {path_str} | {error_text}")
        return ""

    def repair_mixed_batch(self, items: list[dict[str, Any]]) -> dict[str, str]:
        if not items:
            return {}

        normalized_items = []
        for item in items:
            if not isinstance(item, dict):
                continue
            path_str = str(item.get("path", "")).strip()
            text = str(item.get("text", "") or "")
            if not path_str or not text:
                continue
            normalized_items.append(
                {
                    "path": path_str,
                    "text": text,
                    "en_preview": str(item.get("en_preview", "") or ""),
                    "english_words": list(item.get("english_words", []) or []),
                    "entry_context_json": str(item.get("entry_context_json", "") or ""),
                    "glossary_terms": list(item.get("glossary_terms", []) or []),
                }
            )

        if not normalized_items:
            return {}

        payload_items = []
        for it in normalized_items:
            payload_item = {
                "path": it["path"],
                "current_cn_text": it["text"],
            }
            words = [str(w).strip() for w in it["english_words"] if str(w).strip()][:30]
            if SIMPLE_REPAIR_INCLUDE_EN_PREVIEW and it["en_preview"]:
                payload_item["source_en_preview"] = it["en_preview"]
            if words:
                payload_item["english_hint_words"] = words
            if SIMPLE_REPAIR_CONTEXT_ENABLED and it["entry_context_json"]:
                payload_item["entry_context_json"] = it["entry_context_json"]
            if it["glossary_terms"]:
                payload_item["glossary_terms"] = [
                    {"en": e, "zh": c} for e, c in it["glossary_terms"]
                ]
            payload_items.append(payload_item)

        instructions = (
            "You are a professional Chinese localization post-editor for Crucible TRPG. "
            "For each item, translate only residual English fragments into Simplified Chinese while preserving existing Chinese structure as much as possible. "
            "Do not rewrite full paragraphs unless needed for coherence. "
            "Do not add or remove facts. "
            "Preserve all code-like tokens and formatting exactly as-is, including @UUID[...] , @Compendium[...] , @Condition[...] , [[...]] , HTML tags/attributes, punctuation, numbers, whitespace, and line breaks. "
            "Return ONLY JSON object: {\"items\":[{\"path\":\"...\",\"repaired\":\"...\"}, ...]} with one entry per input path."
        )

        request_text = json.dumps(
            {"items": payload_items}, ensure_ascii=False, indent=2
        )
        timeout_seconds = _simple_compute_timeout_seconds(
            request_text,
            SIMPLE_REQUEST_TIMEOUT_SECONDS,
        )

        def _parse_output(raw_text: str) -> dict[str, str]:
            raw = str(raw_text or "").strip()
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
            items_data = parsed.get("items")
            if not isinstance(items_data, list):
                return {}
            out = {}
            for obj in items_data:
                if not isinstance(obj, dict):
                    continue
                p = str(obj.get("path", "")).strip()
                repaired = str(obj.get("repaired", "") or "")
                if p and repaired:
                    out[p] = repaired
            return out

        last_error = None
        for attempt in range(1, SIMPLE_REPAIR_MAX_RETRIES_PER_ITEM + 1):
            try:
                response = self.client.responses.create(
                    model=SIMPLE_MODEL_ID,
                    timeout=timeout_seconds,
                    instructions=instructions,
                    input=request_text,
                )
                out = _parse_output(response.output_text or "")
                if out:
                    return out
            except Exception as e:
                last_error = e
                if _simple_is_fatal_api_error(e):
                    raise _SimpleFatalAPIError(_simple_error_text(e)) from e
                if not _simple_is_retriable_api_error(e):
                    break
                if attempt == SIMPLE_REPAIR_MAX_RETRIES_PER_ITEM:
                    break
                delay = min(
                    SIMPLE_RETRY_DELAY_MAX_SECONDS,
                    SIMPLE_RETRY_DELAY_BASE_SECONDS * (2 ** (attempt - 1)),
                )
                time.sleep(max(0.0, delay))

        can_fallback = (
            SIMPLE_FALLBACK_ON_FAILURE_ENABLED
            and bool(SIMPLE_FALLBACK_MODEL_ID)
            and SIMPLE_FALLBACK_MODEL_ID != SIMPLE_MODEL_ID
            and _simple_should_try_fallback(last_error)
        )
        if can_fallback:
            fallback_timeout_seconds = _simple_compute_timeout_seconds(
                request_text,
                SIMPLE_FALLBACK_TIMEOUT_SECONDS,
            )
            try:
                response = self.client.responses.create(
                    model=SIMPLE_FALLBACK_MODEL_ID,
                    timeout=fallback_timeout_seconds,
                    instructions=instructions,
                    input=request_text,
                )
                out = _parse_output(response.output_text or "")
                if out:
                    return out
            except Exception as fallback_error:
                if _simple_is_fatal_api_error(fallback_error):
                    raise _SimpleFatalAPIError(_simple_error_text(fallback_error)) from fallback_error

        if last_error:
            _simple_write_log(f"ERROR repair batch failed | {_simple_error_text(last_error)}")
        return {}

    def quality_review_and_repair_field(
        self,
        path_str: str,
        current_cn_text: str,
        source_en_text: str,
        source_en_backup_text: str = "",
        entry_context_json: str = "",
        glossary_terms=None,
        force_repair: bool = False,
    ) -> dict[str, Any]:
        glossary_hint = ""
        if glossary_terms:
            pairs = [f"{e} -> {c}" for e, c in glossary_terms]
            glossary_hint = " Context glossary candidates: " + "; ".join(pairs) + "."

        instructions = (
            "You are a senior TRPG localization reviewer for Crucible. "
            "Compare source English and current Chinese translation under provided context, focusing on semantic accuracy, rules correctness, and wording naturalness. "
            "Detect if repair is needed. If needed, provide repaired Chinese text only for this field. "
            "Preserve all code-like tokens and formatting exactly as-is, including @UUID[...] , @Compendium[...] , @Condition[...] , [[...]] , HTML tags/attributes, punctuation, numbers, whitespace, and line breaks. "
            "Do not add or remove mechanics facts. "
            f"Return ONLY JSON with keys: should_repair(bool), repaired_text(str), issues(list[str]), confidence(number 0-1), risk_flags(list[str]).{glossary_hint}"
        )
        if force_repair:
            instructions += " Force repairing this field if any quality risk is detected. "

        payload = {
            "path": path_str,
            "source_en_text": source_en_text,
            "source_en_backup_text": source_en_backup_text,
            "current_cn_text": current_cn_text,
            "entry_context_json": entry_context_json if SIMPLE_REPAIR_CONTEXT_ENABLED else "",
        }
        request_text = json.dumps(payload, ensure_ascii=False, indent=2)
        timeout_seconds = _simple_compute_timeout_seconds(
            request_text,
            SIMPLE_REQUEST_TIMEOUT_SECONDS,
        )

        def _parse(raw_text: str) -> dict[str, Any]:
            raw = str(raw_text or "").strip()
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
            return {
                "should_repair": bool(parsed.get("should_repair", False)),
                "repaired_text": str(parsed.get("repaired_text", "") or ""),
                "issues": [str(x) for x in parsed.get("issues", []) if str(x).strip()],
                "confidence": float(parsed.get("confidence", 0.0) or 0.0),
                "risk_flags": [str(x) for x in parsed.get("risk_flags", []) if str(x).strip()],
            }

        last_error = None
        for attempt in range(1, SIMPLE_REPAIR_MAX_RETRIES_PER_ITEM + 1):
            try:
                response = self.client.responses.create(
                    model=SIMPLE_MODEL_ID,
                    timeout=timeout_seconds,
                    instructions=instructions,
                    input=request_text,
                )
                out = _parse(response.output_text or "")
                if out:
                    return out
            except Exception as e:
                last_error = e
                if _simple_is_fatal_api_error(e):
                    raise _SimpleFatalAPIError(_simple_error_text(e)) from e
                if not _simple_is_retriable_api_error(e):
                    break
                if attempt == SIMPLE_REPAIR_MAX_RETRIES_PER_ITEM:
                    break
                delay = min(
                    SIMPLE_RETRY_DELAY_MAX_SECONDS,
                    SIMPLE_RETRY_DELAY_BASE_SECONDS * (2 ** (attempt - 1)),
                )
                time.sleep(max(0.0, delay))

        can_fallback = (
            SIMPLE_FALLBACK_ON_FAILURE_ENABLED
            and bool(SIMPLE_FALLBACK_MODEL_ID)
            and SIMPLE_FALLBACK_MODEL_ID != SIMPLE_MODEL_ID
            and _simple_should_try_fallback(last_error)
        )
        if can_fallback:
            try:
                response = self.client.responses.create(
                    model=SIMPLE_FALLBACK_MODEL_ID,
                    timeout=_simple_compute_timeout_seconds(request_text, SIMPLE_FALLBACK_TIMEOUT_SECONDS),
                    instructions=instructions,
                    input=request_text,
                )
                out = _parse(response.output_text or "")
                if out:
                    return out
            except Exception as fallback_error:
                if _simple_is_fatal_api_error(fallback_error):
                    raise _SimpleFatalAPIError(_simple_error_text(fallback_error)) from fallback_error

        if last_error:
            _simple_write_log(f"ERROR quality review failed | {path_str} | {_simple_error_text(last_error)}")
        return {
            "should_repair": False,
            "repaired_text": "",
            "issues": ["quality-review-failed"],
            "confidence": 0.0,
            "risk_flags": ["review_failed"],
        }

    def review_glossary_term(self, en_term: str, zh_value: str, context_examples: list[dict[str, str]] | None = None) -> dict[str, Any]:
        en = str(en_term or "").strip()
        zh = str(zh_value or "").strip()
        if not en or not zh:
            return {
                "keep": True,
                "suggested_zh": zh,
                "reason": "empty-term",
                "confidence": 0.0,
            }

        instructions = (
            "You are a senior TRPG localization terminologist for Crucible. "
            "Review whether Chinese glossary wording is accurate, natural, and domain-appropriate for this English term. "
            "Use the provided EN/CN corpus contexts as the primary semantic evidence (they may contain multi-sentence paragraphs). "
            "Do not judge from isolated terms only. "
            "If current Chinese wording is good, keep it. If not, suggest a better concise Chinese term. "
            "Return ONLY JSON: {\"keep\":bool,\"suggested_zh\":str,\"reason\":str,\"confidence\":number}."
        )
        request_text = json.dumps(
            {
                "en_term": en,
                "zh_value": zh,
                "context_examples": [
                    {
                        "source_en_context": str(x.get("source_en_context", "") or "").strip(),
                        "current_zh_context": str(x.get("current_zh_context", "") or "").strip(),
                    }
                    for x in list(context_examples or [])
                    if str(x.get("source_en_context", "") or "").strip()
                    and str(x.get("current_zh_context", "") or "").strip()
                ],
            },
            ensure_ascii=False,
        )
        timeout_seconds = _simple_compute_timeout_seconds(request_text, SIMPLE_REQUEST_TIMEOUT_SECONDS)

        def _parse(raw_text: str) -> dict[str, Any]:
            raw = str(raw_text or "").strip()
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
            return {
                "keep": bool(parsed.get("keep", True)),
                "suggested_zh": str(parsed.get("suggested_zh", "") or ""),
                "reason": str(parsed.get("reason", "") or ""),
                "confidence": float(parsed.get("confidence", 0.0) or 0.0),
            }

        last_error = None
        for attempt in range(1, SIMPLE_REPAIR_MAX_RETRIES_PER_ITEM + 1):
            try:
                response = self.client.responses.create(
                    model=SIMPLE_MODEL_ID,
                    timeout=timeout_seconds,
                    instructions=instructions,
                    input=request_text,
                )
                out = _parse(response.output_text or "")
                if out:
                    return out
            except Exception as e:
                last_error = e
                if _simple_is_fatal_api_error(e):
                    raise _SimpleFatalAPIError(_simple_error_text(e)) from e
                if not _simple_is_retriable_api_error(e):
                    break
                if attempt == SIMPLE_REPAIR_MAX_RETRIES_PER_ITEM:
                    break
                delay = min(
                    SIMPLE_RETRY_DELAY_MAX_SECONDS,
                    SIMPLE_RETRY_DELAY_BASE_SECONDS * (2 ** (attempt - 1)),
                )
                time.sleep(max(0.0, delay))

        if SIMPLE_FALLBACK_ON_FAILURE_ENABLED and SIMPLE_FALLBACK_MODEL_ID and SIMPLE_FALLBACK_MODEL_ID != SIMPLE_MODEL_ID:
            try:
                response = self.client.responses.create(
                    model=SIMPLE_FALLBACK_MODEL_ID,
                    timeout=_simple_compute_timeout_seconds(request_text, SIMPLE_FALLBACK_TIMEOUT_SECONDS),
                    instructions=instructions,
                    input=request_text,
                )
                out = _parse(response.output_text or "")
                if out:
                    return out
            except Exception:
                pass

        if last_error:
            _simple_write_log(f"WARN glossary term review failed | {en} | {_simple_error_text(last_error)}")
        return {
            "keep": True,
            "suggested_zh": zh,
            "reason": "review-failed",
            "confidence": 0.0,
        }

    def extract_basic_glossary(self, pair_lines: list[str], max_terms: int = 8, model_id: str | None = None) -> dict:
        if not pair_lines:
            return {}
        max_terms = max(1, int(max_terms))
        model_to_use = str(model_id or SIMPLE_MODEL_ID).strip() or SIMPLE_MODEL_ID
        joined = "\n".join(pair_lines[:120])
        glossary_timeout_seconds = _simple_compute_timeout_seconds(
            joined,
            SIMPLE_REQUEST_TIMEOUT_SECONDS,
        )
        last_error = None

        for attempt in range(1, SIMPLE_MAX_RETRIES + 1):
            try:
                response = self.client.responses.create(
                    model=model_to_use,
                    timeout=glossary_timeout_seconds,
                    instructions=(
                        "Extract a minimal, foundational glossary for Crucible/fantasy TRPG translation. "
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
                    _simple_write_log(
                        f"WARN extract glossary non-json response | model={model_to_use} | raw={_simple_snip(raw, 200)}"
                    )
                    return {}
                return parsed
            except Exception as e:
                last_error = e
                if _simple_is_fatal_api_error(e):
                    raise _SimpleFatalAPIError(_simple_error_text(e)) from e

                if not _simple_is_retriable_api_error(e):
                    _simple_write_log(
                        f"NO_RETRY extract glossary non-retriable | model={model_to_use} | {e}"
                    )
                    break

                if attempt == SIMPLE_MAX_RETRIES:
                    break

                delay = min(
                    SIMPLE_RETRY_DELAY_MAX_SECONDS,
                    SIMPLE_RETRY_DELAY_BASE_SECONDS * (2 ** (attempt - 1))
                )
                _simple_write_log(
                    f"RETRY_EXTRACT {attempt}/{SIMPLE_MAX_RETRIES} | model={model_to_use} "
                    f"timeout={glossary_timeout_seconds:.1f}s | {e}"
                )
                time.sleep(max(0.0, delay))

        error_text = _simple_error_text(last_error) if last_error else "unknown error"
        _simple_write_log(
            f"ERROR extract glossary failed | model={model_to_use} | timeout={glossary_timeout_seconds:.1f}s | {error_text}"
        )
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


def _simple_iter_chunks(items: list[str], chunk_size: int):
    for idx in range(0, len(items), chunk_size):
        yield items[idx:idx + chunk_size]


def _simple_extract_bilingual_pair(value: str):
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or "\n" not in text:
        return None

    lines = text.splitlines()
    if len(lines) < 2:
        return None

    for split_idx in range(1, len(lines)):
        zh = "\n".join(lines[:split_idx]).strip()
        en = "\n".join(lines[split_idx:]).strip()
        if not zh or not en:
            continue
        if not _simple_contains_chinese(zh):
            continue
        if not _simple_contains_english(en):
            continue
        if _simple_contains_chinese(en):
            continue
        return en, zh

    first, rest = text.split("\n", 1)
    if _simple_contains_chinese(first) and _simple_contains_english(rest):
        return rest.strip(), first.strip()
    return None


def _simple_collect_pair_lines_for_extract_only(file_path: Path):
    try:
        with file_path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except Exception as e:
        _simple_write_log(f"ERROR load json failed (extract only) | {file_path} | {e}")
        return [], str(e)

    is_system_lang_file = _simple_is_system_lang_file(file_path)
    pair_lines = []
    seen = set()

    for _, path_str, value in _simple_collect_string_tasks(data):
        pair = _simple_extract_bilingual_pair(value)
        if not pair:
            continue
        en, zh = pair
        if not _simple_should_translate(path_str, en, is_system_lang_file=is_system_lang_file):
            continue

        pair_key = _simple_hash(f"{en}\n{zh}")
        if pair_key in seen:
            continue
        seen.add(pair_key)
        pair_lines.append(f"EN: {en}\nZH: {zh}")

        if (
            SIMPLE_EXTRACT_ONLY_MAX_PAIRS_PER_FILE > 0
            and len(pair_lines) >= SIMPLE_EXTRACT_ONLY_MAX_PAIRS_PER_FILE
        ):
            break

    return pair_lines, ""


def _simple_extract_glossary_only(target_files, translator, glossary_data: dict, glossary_buckets):
    total_stats = {
        "files": 0,
        "file_errors": 0,
        "pairs": 0,
        "batches": 0,
        "batch_cached": 0,
        "batch_processed": 0,
        "changed": 0,
    }
    aborted = False
    abort_reason = ""
    progress = _simple_load_extract_only_progress(SIMPLE_EXTRACT_ONLY_PROGRESS_CACHE_PATH)
    done_batches = progress.get("done_batches") if isinstance(progress, dict) else None
    if not isinstance(done_batches, dict):
        done_batches = {}
        progress = {"done_batches": done_batches}

    limiter = _SimpleRateLimiter(SIMPLE_EXTRACT_ONLY_TARGET_RPM)

    def _make_batch_key(file_path: Path, batch_lines: list[str]) -> str:
        file_norm = str(file_path.resolve()).replace("\\", "/")
        payload_hash = _simple_hash("\n".join(batch_lines))
        return f"{file_norm}::{payload_hash}"

    def _mark_batch_done(batch_key: str):
        done_batches[batch_key] = 1
        _simple_save_extract_only_progress(SIMPLE_EXTRACT_ONLY_PROGRESS_CACHE_PATH, progress)

    for i, file_path in enumerate(target_files, start=1):
        print(f"\n[{i}/{len(target_files)}] 提炼文件: {file_path}")
        total_stats["files"] += 1
        pair_lines, error = _simple_collect_pair_lines_for_extract_only(file_path)
        if error:
            total_stats["file_errors"] += 1
            print(f"⚠️ 跳过无法读取的文件: {file_path} | {error}")
            continue

        pair_count = len(pair_lines)
        total_stats["pairs"] += pair_count
        print(f"可提炼双语条目: {pair_count}")
        if pair_count == 0:
            continue

        all_batches = list(_simple_iter_chunks(pair_lines, SIMPLE_EXTRACT_ONLY_PAIR_BATCH_SIZE))
        batch_count = len(all_batches)
        batch_jobs = []
        for batch_idx, batch_lines in enumerate(all_batches, start=1):
            batch_key = _make_batch_key(file_path, batch_lines)
            if batch_key in done_batches:
                total_stats["batch_cached"] += 1
                continue
            batch_jobs.append((batch_idx, batch_lines, batch_key))

        total_stats["batches"] += len(batch_jobs)
        changed_this_file = 0
        if not batch_jobs:
            print(f"文件提炼统计：批次={batch_count} 缓存跳过={batch_count} 新增/更新术语={changed_this_file}")
            continue

        use_parallel = (
            SIMPLE_EXTRACT_ONLY_PARALLEL_ENABLED
            and SIMPLE_EXTRACT_ONLY_MAX_WORKERS > 1
            and len(batch_jobs) > 1
        )

        def _extract_batch(job):
            batch_idx, batch_lines, batch_key = job
            limiter.wait()
            try:
                extracted = translator.extract_basic_glossary(
                    batch_lines,
                    max_terms=SIMPLE_ADAPTIVE_GLOSSARY_MAX_TERMS_PER_BATCH,
                    model_id=SIMPLE_ADAPTIVE_GLOSSARY_EXTRACT_MODEL_ID,
                )
                return batch_idx, batch_key, batch_lines, extracted, False, ""
            except _SimpleFatalAPIError as fatal_error:
                return batch_idx, batch_key, batch_lines, {}, True, str(fatal_error).strip()

        if use_parallel:
            print(
                f"并行提炼：workers={SIMPLE_EXTRACT_ONLY_MAX_WORKERS} "
                f"rpm={SIMPLE_EXTRACT_ONLY_TARGET_RPM} 待处理批次={len(batch_jobs)}"
            )
            with concurrent.futures.ThreadPoolExecutor(max_workers=SIMPLE_EXTRACT_ONLY_MAX_WORKERS) as exe:
                future_map = {exe.submit(_extract_batch, job): job for job in batch_jobs}
                for fut in concurrent.futures.as_completed(future_map):
                    try:
                        batch_idx, batch_key, batch_lines, extracted, is_fatal, fatal_reason = fut.result()
                    except Exception as e:
                        _simple_write_log(f"ERROR extract-only worker failed | file={file_path} | {e}")
                        continue

                    total_stats["batch_processed"] += 1

                    if is_fatal:
                        aborted = True
                        abort_reason = fatal_reason or "fatal API error"
                        _simple_write_log(
                            f"ABORT extract-only file={file_path} batch={batch_idx}/{batch_count} | {abort_reason}"
                        )
                        print(f"⛔ 术语提炼提前停止：{abort_reason}")
                        for pending in future_map:
                            if not pending.done():
                                pending.cancel()
                        break

                    changed = _simple_upsert_adaptive_glossary(glossary_data, extracted)
                    changed_this_file += changed
                    total_stats["changed"] += changed
                    if changed > 0:
                        _simple_save_glossary_json(SIMPLE_ADAPTIVE_GLOSSARY_OUTPUT_PATH, glossary_data)
                        glossary_buckets, _ = _simple_refresh_runtime_glossary(glossary_data)

                    _mark_batch_done(batch_key)
                    _simple_write_log(
                        f"GLOSSARY_EXTRACT_ONLY file={file_path.name} batch={batch_idx}/{batch_count} "
                        f"pairs={len(batch_lines)} changed={changed} parallel={use_parallel}"
                    )

                    if aborted:
                        break
        else:
            for batch_idx, batch_lines, batch_key in batch_jobs:
                total_stats["batch_processed"] += 1
                batch_idx, batch_key, batch_lines, extracted, is_fatal, fatal_reason = _extract_batch(
                    (batch_idx, batch_lines, batch_key)
                )
                if is_fatal:
                    aborted = True
                    abort_reason = fatal_reason or "fatal API error"
                    _simple_write_log(
                        f"ABORT extract-only file={file_path} batch={batch_idx}/{batch_count} | {abort_reason}"
                    )
                    print(f"⛔ 术语提炼提前停止：{abort_reason}")
                    break

                changed = _simple_upsert_adaptive_glossary(glossary_data, extracted)
                changed_this_file += changed
                total_stats["changed"] += changed
                if changed > 0:
                    _simple_save_glossary_json(SIMPLE_ADAPTIVE_GLOSSARY_OUTPUT_PATH, glossary_data)
                    glossary_buckets, _ = _simple_refresh_runtime_glossary(glossary_data)

                _mark_batch_done(batch_key)
                _simple_write_log(
                    f"GLOSSARY_EXTRACT_ONLY file={file_path.name} batch={batch_idx}/{batch_count} "
                    f"pairs={len(batch_lines)} changed={changed} parallel={use_parallel}"
                )

        print(
            f"文件提炼统计：批次={batch_count} 缓存跳过={batch_count - len(batch_jobs)} "
            f"已处理={len(batch_jobs)} 新增/更新术语={changed_this_file}"
        )
        if aborted:
            break

    _simple_save_glossary_json(SIMPLE_ADAPTIVE_GLOSSARY_OUTPUT_PATH, glossary_data)
    _simple_save_extract_only_progress(SIMPLE_EXTRACT_ONLY_PROGRESS_CACHE_PATH, progress)

    if aborted:
        print("⚠️ 术语提炼提前终止（致命API错误）")
    else:
        print("✅ 完成：术语提炼结束")
    print(
        f"统计：文件={total_stats['files']}/{len(target_files)} 文件错误={total_stats['file_errors']} "
        f"双语条目={total_stats['pairs']} 批次待处理={total_stats['batches']} "
        f"批次已处理={total_stats['batch_processed']} 批次缓存跳过={total_stats['batch_cached']} "
        f"新增/更新术语={total_stats['changed']}"
    )
    _simple_write_log(
        f"EXTRACT_ONLY_DONE files={total_stats['files']}/{len(target_files)} aborted={aborted} "
        f"file_errors={total_stats['file_errors']} pairs={total_stats['pairs']} batches={total_stats['batches']} "
        f"batch_processed={total_stats['batch_processed']} batch_cached={total_stats['batch_cached']} "
        f"changed={total_stats['changed']}"
    )

    return aborted, abort_reason, glossary_buckets


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


def _simple_get_value_at_path(data, parts):
    cur = data
    try:
        for p in parts:
            cur = cur[p]
    except Exception:
        return None
    return cur


def _simple_parse_report_path(path_text: str):
    raw = str(path_text or "").strip()
    if not raw:
        return []
    if raw.startswith("root."):
        raw = raw[5:]
    elif raw == "root":
        return []

    parts = []
    for token in raw.split("."):
        tk = token.strip()
        if not tk:
            continue
        if re.fullmatch(r"\d+", tk):
            parts.append(int(tk))
        else:
            parts.append(tk)
    return parts


def _simple_is_significant_english_leak(text: str) -> bool:
    if not isinstance(text, str):
        return False
    cleaned = _simple_strip_codes_for_lang_detect(text)
    words = re.findall(r"[A-Za-z][A-Za-z0-9'/-]{2,}", cleaned)
    if not words:
        return False
    if not _simple_contains_chinese(cleaned):
        return True
    meaningful = [w for w in words if len(w) >= 3]
    return len(meaningful) >= 2


def _simple_resolve_report_file_path(file_path_text: str) -> Path:
    raw = str(file_path_text or "").strip()
    if not raw:
        return Path("")

    candidates = [
        Path(raw),
        Path(raw.replace("\\", "/")),
        Path.cwd() / raw,
        Path.cwd() / raw.replace("\\", "/"),
    ]

    seen = set()
    for cand in candidates:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        if cand.exists():
            return cand

    return Path(raw.replace("\\", "/"))


def _simple_find_entry_context_parts(parts):
    if not parts:
        return []
    for idx, part in enumerate(parts[:-1]):
        if isinstance(part, str) and part.casefold() == "entries" and idx + 1 < len(parts):
            return parts[: idx + 2]
    if len(parts) >= 2:
        return parts[:-1]
    return []


def _simple_trim_json_text(text: str, max_chars: int) -> str:
    s = str(text or "")
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 120] + "\n...\n[TRUNCATED]"


def _simple_extract_english_words_for_repair(text: str, max_items: int = 20) -> list[str]:
    cleaned = _simple_strip_codes_for_lang_detect(str(text or ""))
    words = re.findall(r"[A-Za-z][A-Za-z0-9'/-]{2,}", cleaned)
    out = []
    seen = set()
    for w in words:
        lw = w.casefold()
        if lw in seen:
            continue
        seen.add(lw)
        out.append(w)
        if len(out) >= max_items:
            break
    return out


def _simple_build_repair_context_json(data, parts) -> str:
    if not SIMPLE_REPAIR_CONTEXT_ENABLED:
        return ""
    context_parts = _simple_find_entry_context_parts(parts)
    context_obj = _simple_get_value_at_path(data, context_parts) if context_parts else None
    if not isinstance(context_obj, (dict, list)):
        return ""
    try:
        return _simple_trim_json_text(
            json.dumps(context_obj, ensure_ascii=False, indent=2),
            SIMPLE_REPAIR_CONTEXT_MAX_CHARS,
        )
    except Exception:
        return ""


def _simple_collect_optimize_findings_for_file(file_path: Path, data) -> list[dict[str, Any]]:
    is_system_lang_file = _simple_is_system_lang_file(file_path)
    findings: list[dict[str, Any]] = []

    for parts, path_str, value in _simple_collect_string_tasks(data):
        if not isinstance(value, str):
            continue
        if not _simple_contains_english(value):
            continue

        last_key_norm = _simple_extract_last_key(path_str).casefold()
        if SIMPLE_OPTIMIZE_INCLUDE_KEYS_ONLY and last_key_norm not in SIMPLE_OPTIMIZE_INCLUDE_KEYS_ONLY:
            continue
        if last_key_norm in SIMPLE_SKIP_KEYS_NORM:
            continue
        if any(s in path_str.casefold() for s in SIMPLE_TRANSLATE_PATH_SKIP_SUBSTRINGS):
            continue

        should_use = _simple_is_significant_english_leak(value)
        if not should_use:
            # Also allow rows that still look translatable by allow rules.
            source_text = _simple_extract_original(value)
            should_use = _simple_should_translate(
                path_str,
                source_text,
                is_system_lang_file=is_system_lang_file,
                allow_key_filter=False,
            )
        if not should_use:
            continue

        en_preview = ""
        pair = _simple_extract_bilingual_pair(value)
        if pair:
            en_preview = pair[0]

        findings.append(
            {
                "path": path_str,
                "reason": "optimize_scan",
                "english_words": _simple_extract_english_words_for_repair(value),
                "en_preview": (en_preview or "")[:300],
            }
        )

        if SIMPLE_OPTIMIZE_MAX_ITEMS_PER_FILE > 0 and len(findings) >= SIMPLE_OPTIMIZE_MAX_ITEMS_PER_FILE:
            break

    return findings


def _simple_extract_numbers_for_guard(text: str) -> list[str]:
    return re.findall(r"\d+(?:\.\d+)?", str(text or ""))


def _simple_extract_protected_tokens(text: str) -> list[str]:
    s = str(text or "")
    tokens = []
    tokens.extend(re.findall(r"@\w+\[[^\]]*\](?:\{[^}]*\})?", s))
    tokens.extend(re.findall(r"\[\[[^\]]+\]\]", s))
    tokens.extend(re.findall(r"<[^>]+>", s))
    return sorted(tokens)


def _simple_is_markup_safe(original_text: str, repaired_text: str) -> bool:
    return _simple_extract_protected_tokens(original_text) == _simple_extract_protected_tokens(repaired_text)


def _simple_is_numeric_consistent(source_en: str, repaired_text: str) -> bool:
    return _simple_extract_numbers_for_guard(source_en) == _simple_extract_numbers_for_guard(repaired_text)


def _simple_resolve_source_en_file_for_cn(cn_file_path: Path) -> Path | None:
    p = Path(cn_file_path)
    parts = list(p.parts)
    for idx, part in enumerate(parts):
        low = part.casefold()
        replacement: str | None = None
        if low == "cn":
            replacement = "en"
        elif low.startswith("cn_") or low.startswith("cn-"):
            replacement = "en" + part[2:]
        if replacement is not None:
            new_parts = list(parts)
            new_parts[idx] = replacement
            en_path = Path(*new_parts)
            if en_path.exists() and en_path.is_file():
                return en_path
            break
    if p.name.casefold() == "cn.json":
        en_path = p.with_name("en.json")
        if en_path.exists() and en_path.is_file():
            return en_path
    return None


def _simple_load_source_en_data(cn_file_path: Path):
    en_path = _simple_resolve_source_en_file_for_cn(cn_file_path)
    if not en_path:
        return None, None
    try:
        with en_path.open("r", encoding="utf-8-sig") as f:
            return json.load(f), en_path
    except Exception as e:
        _simple_write_log(f"WARN source en load failed | cn={cn_file_path} | en={en_path} | {e}")
        return None, en_path


def _simple_collect_quality_candidates(file_path: Path, data, en_data) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for parts, path_str, value in _simple_collect_string_tasks(data):
        if not isinstance(value, str):
            continue
        last_key_norm = _simple_extract_last_key(path_str).casefold()
        if SIMPLE_QUALITY_INCLUDE_KEYS_ONLY and last_key_norm not in SIMPLE_QUALITY_INCLUDE_KEYS_ONLY:
            continue
        if last_key_norm in SIMPLE_SKIP_KEYS_NORM:
            continue
        if any(s in path_str.casefold() for s in SIMPLE_TRANSLATE_PATH_SKIP_SUBSTRINGS):
            continue

        source_en = _simple_get_value_at_path(en_data, parts) if en_data is not None else None
        if not isinstance(source_en, str) or not source_en.strip():
            continue

        # Focus quality passes on already localized content or mixed content.
        if (not _simple_contains_chinese(value)) and (not _simple_is_significant_english_leak(value)):
            continue

        candidates.append(
            {
                "parts": parts,
                "path": path_str,
                "current_cn": value,
                "source_en": source_en,
                "english_words": _simple_extract_english_words_for_repair(value),
            }
        )
        if SIMPLE_QUALITY_MAX_ITEMS_PER_FILE > 0 and len(candidates) >= SIMPLE_QUALITY_MAX_ITEMS_PER_FILE:
            break
    return candidates


def _simple_write_quality_report(report_rows: list[dict[str, Any]]):
    try:
        SIMPLE_QUALITY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with SIMPLE_QUALITY_REPORT_PATH.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(report_rows, f, ensure_ascii=False, indent=2)
            f.write("\n")
    except Exception as e:
        _simple_write_log(f"WARN quality report write failed | {SIMPLE_QUALITY_REPORT_PATH} | {e}")


def _simple_write_json_report(path: Path, rows):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
            f.write("\n")
    except Exception as e:
        _simple_write_log(f"WARN json report write failed | {path} | {e}")


def _simple_trim_text_for_context(text: str, max_chars: int) -> str:
    t = str(text or "").strip()
    if max_chars <= 0 or len(t) <= max_chars:
        return t
    return t[: max(0, max_chars)] + "..."


def _simple_term_lookup_key(term: str) -> str:
    words = [_simple_normalize_en_token(w) for w in re.findall(r"[A-Za-z]+", str(term or "").casefold())]
    words = [w for w in words if w]
    return " ".join(words)


def _simple_normalize_en_token(token: str) -> str:
    t = str(token or "").strip().casefold()
    if len(t) > 3 and t.endswith("'s"):
        t = t[:-2]
    elif len(t) > 3 and t.endswith("s") and not t.endswith(("ss", "us", "is")):
        t = t[:-1]
    return t


def _simple_term_tokens(term: str) -> list[str]:
    out = []
    seen = set()
    for w in re.findall(r"[A-Za-z]+", str(term or "").casefold()):
        n = _simple_normalize_en_token(w)
        if not n or n in seen:
            continue
        seen.add(n)
        out.append(n)
    return out


def _simple_build_term_phrase_regex(en_term: str):
    words = re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?", str(en_term or ""))
    if not words:
        return None

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
        return None

    separator = r"(?:[\s\-_.,:;!?/\\()\[\]{}'’\"`]+)?"
    pattern = r"(?<![A-Za-z0-9])" + separator.join(token_patterns) + r"(?![A-Za-z0-9])"
    try:
        return re.compile(pattern, flags=re.IGNORECASE)
    except Exception:
        return None


def _simple_build_glossary_context_index(target_files) -> tuple[dict[str, list[dict[str, str]]], dict[str, list[dict[str, str]]], list[dict[str, str]], int]:
    if not target_files:
        return {}, {}, [], 0

    phrase_index: dict[str, list[dict[str, str]]] = {}
    token_index: dict[str, list[dict[str, str]]] = {}
    global_examples: list[dict[str, str]] = []
    global_seen = set()
    total_pairs = 0

    for file_path in target_files:
        try:
            with file_path.open("r", encoding="utf-8-sig") as f:
                cn_data = json.load(f)
        except Exception:
            continue

        en_data, _ = _simple_load_source_en_data(file_path)
        if en_data is None:
            continue

        for parts, path_str, cn_value in _simple_collect_string_tasks(cn_data):
            if not isinstance(cn_value, str):
                continue
            last_key_norm = _simple_extract_last_key(path_str).casefold()
            if SIMPLE_QUALITY_GLOSSARY_REVIEW_CONTEXT_SCAN_KEYS and last_key_norm not in SIMPLE_QUALITY_GLOSSARY_REVIEW_CONTEXT_SCAN_KEYS:
                continue

            en_value = _simple_get_value_at_path(en_data, parts)
            if not isinstance(en_value, str):
                continue
            if (not _simple_contains_english(en_value)) or (not _simple_contains_chinese(cn_value)):
                continue

            pair = _simple_extract_bilingual_pair(cn_value)
            zh_context = pair[1] if pair else cn_value
            en_context = en_value
            en_plain = _simple_plain_text_for_glossary(en_context)
            if not en_plain:
                continue

            total_pairs += 1
            words = [_simple_normalize_en_token(w) for w in re.findall(r"[A-Za-z]+", en_plain.casefold())]
            words = [w for w in words if w]
            if not words:
                continue

            example = {
                "file": str(file_path),
                "path": path_str,
                "source_en_context": _simple_trim_text_for_context(en_context, SIMPLE_QUALITY_GLOSSARY_REVIEW_CONTEXT_MAX_CHARS),
                "current_zh_context": _simple_trim_text_for_context(zh_context, SIMPLE_QUALITY_GLOSSARY_REVIEW_CONTEXT_MAX_CHARS),
            }
            ex_key = (example["file"], example["path"])
            if ex_key not in global_seen:
                global_seen.add(ex_key)
                if len(global_examples) < 5000:
                    global_examples.append(example)

            max_ngram = min(4, len(words))
            seen_keys = set()
            for n in range(1, max_ngram + 1):
                for i in range(0, len(words) - n + 1):
                    key = " ".join(words[i : i + n])
                    if len(key) < 3 or key in seen_keys:
                        continue
                    seen_keys.add(key)
                    bucket = phrase_index.setdefault(key, [])
                    if len(bucket) >= max(4, SIMPLE_QUALITY_GLOSSARY_REVIEW_MAX_EXAMPLES_PER_TERM * 3):
                        continue
                    bucket.append(example)

            for tk in set(words):
                if len(tk) < 3:
                    continue
                tk_bucket = token_index.setdefault(tk, [])
                if len(tk_bucket) >= max(8, SIMPLE_QUALITY_GLOSSARY_REVIEW_MAX_EXAMPLES_PER_TERM * 8):
                    continue
                tk_bucket.append(example)

    return phrase_index, token_index, global_examples, total_pairs


def _simple_collect_context_examples_for_term(
    en_term: str,
    phrase_index: dict[str, list[dict[str, str]]],
    token_index: dict[str, list[dict[str, str]]],
    global_examples: list[dict[str, str]],
    max_examples: int,
) -> list[dict[str, str]]:
    if max_examples <= 0:
        return []

    out: list[dict[str, str]] = []
    seen = set()

    def _push(ex: dict[str, str]):
        key = (str(ex.get("file", "")), str(ex.get("path", "")))
        if key in seen:
            return
        seen.add(key)
        out.append(ex)

    # Step 1: preprocess term and perform direct full-text search over EN value content.
    exact_pattern = _simple_build_term_phrase_regex(str(en_term))
    if exact_pattern is not None and global_examples:
        for ex in global_examples:
            en_ctx = _simple_plain_text_for_glossary(str(ex.get("source_en_context", "")))
            if not en_ctx:
                continue
            if exact_pattern.search(en_ctx):
                _push(ex)
                if len(out) >= max_examples:
                    return out

    tokens = _simple_term_tokens(str(en_term))
    if not tokens:
        return out

    # Step 2: fallback with token-overlap ranking over EN value content.
    score_map: dict[tuple[str, str], dict[str, Any]] = {}
    token_set = set(tokens)
    for ex in global_examples:
        en_ctx = _simple_plain_text_for_glossary(str(ex.get("source_en_context", "")))
        if not en_ctx:
            continue
        ctx_words = [_simple_normalize_en_token(w) for w in re.findall(r"[A-Za-z]+", en_ctx.casefold())]
        ctx_set = {w for w in ctx_words if w}
        if not ctx_set:
            continue
        overlap = len(token_set & ctx_set)
        if overlap <= 0:
            continue
        ex_key = (str(ex.get("file", "")), str(ex.get("path", "")))
        score_map[ex_key] = {"score": overlap, "example": ex}

    ranked = sorted(
        score_map.values(),
        key=lambda x: (
            -int(x.get("score", 0)),
            len(str(x.get("example", {}).get("source_en_context", ""))),
        ),
    )
    token_target = max(1, len(tokens))
    if token_target >= 4:
        min_overlap = 3
    elif token_target == 3:
        min_overlap = 2
    else:
        min_overlap = 1

    for item in ranked:
        overlap = int(item.get("score", 0) or 0)
        if overlap < min_overlap:
            continue
        _push(item.get("example", {}))
        if len(out) >= max_examples:
            break
    return out


def _simple_quality_review_glossary(translator, glossary_data: dict, target_files=None):
    if not SIMPLE_QUALITY_GLOSSARY_REVIEW_ENABLED:
        return 0
    if not isinstance(glossary_data, dict) or not glossary_data:
        return 0

    reviewed = 0
    changed = 0
    report_rows = []
    items = list(glossary_data.items())
    if SIMPLE_QUALITY_GLOSSARY_REVIEW_MAX_TERMS_PER_RUN > 0:
        items = items[:SIMPLE_QUALITY_GLOSSARY_REVIEW_MAX_TERMS_PER_RUN]

    phrase_context_index = {}
    token_context_index = {}
    global_context_examples = []
    indexed_pairs = 0
    if SIMPLE_QUALITY_GLOSSARY_REVIEW_WITH_CONTEXT_ENABLED:
        phrase_context_index, token_context_index, global_context_examples, indexed_pairs = _simple_build_glossary_context_index(target_files or [])

    prepared = []
    for en_term, zh_raw in items:
        zh_value = ""
        if isinstance(zh_raw, list):
            zh_value = str(zh_raw[0] if zh_raw else "").strip()
        else:
            zh_value = str(zh_raw or "").strip()
        if not en_term or not zh_value:
            continue

        contexts = []
        if phrase_context_index or token_context_index or global_context_examples:
            contexts = _simple_collect_context_examples_for_term(
                str(en_term),
                phrase_context_index,
                token_context_index,
                global_context_examples,
                SIMPLE_QUALITY_GLOSSARY_REVIEW_MAX_EXAMPLES_PER_TERM,
            )

        prepared.append(
            {
                "en_term": str(en_term),
                "zh_raw": zh_raw,
                "zh_value": zh_value,
                "contexts": contexts,
            }
        )

    workers = max(1, int(SIMPLE_QUALITY_GLOSSARY_REVIEW_WORKERS))
    parallel_enabled = SIMPLE_QUALITY_GLOSSARY_REVIEW_PARALLEL_ENABLED and workers > 1

    def _review_one(payload: dict[str, Any]) -> dict[str, Any]:
        en_term = payload["en_term"]
        zh_value = payload["zh_value"]
        contexts = payload.get("contexts", [])
        result = translator.review_glossary_term(en_term, zh_value, context_examples=contexts)
        return {
            "en_term": en_term,
            "zh_raw": payload["zh_raw"],
            "zh_value": zh_value,
            "contexts": contexts,
            "result": result,
        }

    review_rows = []
    aborted = False
    progress = tqdm(prepared, total=len(prepared), desc="术语审校", position=0, leave=True, dynamic_ncols=True)
    if parallel_enabled:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as exe:
            futures = [exe.submit(_review_one, p) for p in prepared]
            for fut in concurrent.futures.as_completed(futures):
                progress.update(1)
                try:
                    review_rows.append(fut.result())
                except _SimpleFatalAPIError as fatal_error:
                    _simple_write_log(f"ABORT glossary review fatal | {fatal_error}")
                    aborted = True
                    break
                except Exception as e:
                    _simple_write_log(f"WARN glossary review worker failed | {e}")
    else:
        for p in prepared:
            progress.update(1)
            try:
                review_rows.append(_review_one(p))
            except _SimpleFatalAPIError as fatal_error:
                _simple_write_log(f"ABORT glossary review fatal | {fatal_error}")
                aborted = True
                break
            except Exception as e:
                _simple_write_log(f"WARN glossary review worker failed | {e}")
    progress.close()

    for row in review_rows:
        en_term = row["en_term"]
        zh_raw = row["zh_raw"]
        zh_value = row["zh_value"]
        result = row["result"]
        contexts = row.get("contexts", [])

        reviewed += 1
        keep = bool(result.get("keep", True))
        suggested = str(result.get("suggested_zh", "") or "").strip()
        reason = str(result.get("reason", "") or "")
        confidence = float(result.get("confidence", 0.0) or 0.0)

        applied = False
        if (not keep) and suggested and SIMPLE_QUALITY_GLOSSARY_REVIEW_APPLY:
            if isinstance(zh_raw, list):
                new_list = list(zh_raw)
                if new_list:
                    new_list[0] = suggested
                else:
                    new_list = [suggested]
                if new_list != zh_raw:
                    glossary_data[en_term] = new_list
                    applied = True
                    changed += 1
            else:
                if suggested != zh_value:
                    glossary_data[en_term] = suggested
                    applied = True
                    changed += 1

        report_rows.append(
            {
                "en_term": str(en_term),
                "current_zh": zh_value,
                "keep": keep,
                "suggested_zh": suggested,
                "reason": reason,
                "confidence": confidence,
                "context_example_count": len(contexts),
                "applied": applied,
            }
        )

    _simple_write_json_report(SIMPLE_QUALITY_GLOSSARY_REVIEW_REPORT_PATH, report_rows)
    if changed > 0 and SIMPLE_QUALITY_GLOSSARY_REVIEW_APPLY:
        _simple_save_glossary_json(SIMPLE_ADAPTIVE_GLOSSARY_OUTPUT_PATH, glossary_data)

    print(
        f"术语审校：reviewed={reviewed} changed={changed} apply={SIMPLE_QUALITY_GLOSSARY_REVIEW_APPLY} "
        f"parallel={parallel_enabled} workers={workers} context={SIMPLE_QUALITY_GLOSSARY_REVIEW_WITH_CONTEXT_ENABLED} "
        f"indexed_pairs={indexed_pairs} report={SIMPLE_QUALITY_GLOSSARY_REVIEW_REPORT_PATH}"
    )
    _simple_write_log(
        f"GLOSSARY_QUALITY_REVIEW reviewed={reviewed} changed={changed} apply={SIMPLE_QUALITY_GLOSSARY_REVIEW_APPLY} "
        f"parallel={parallel_enabled} workers={workers} context={SIMPLE_QUALITY_GLOSSARY_REVIEW_WITH_CONTEXT_ENABLED} "
        f"indexed_pairs={indexed_pairs} aborted={aborted} report={SIMPLE_QUALITY_GLOSSARY_REVIEW_REPORT_PATH}"
    )
    return changed


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


def _simple_collect_target_files_from_dirs(dir_paths: list[str], file_glob: str, recursive: bool) -> list[Path]:
    out: list[Path] = []
    seen = set()
    for raw in dir_paths:
        p = Path(raw)
        if not p.exists() or not p.is_dir():
            print(f"⚠️ 目录不存在，已跳过: {p}")
            continue
        files = p.rglob(file_glob) if recursive else p.glob(file_glob)
        for f in files:
            if not f.is_file():
                continue
            key = str(f.resolve()).replace("\\", "/")
            if key in seen:
                continue
            seen.add(key)
            out.append(f)
    out.sort(key=lambda x: str(x).casefold())
    return out


def _simple_should_include_glossary_key(path_str: str) -> bool:
    if not SIMPLE_PIPELINE_GLOSSARY_INCLUDE_KEYS_ONLY:
        return True
    return _simple_extract_last_key(path_str).casefold() in SIMPLE_PIPELINE_GLOSSARY_INCLUDE_KEYS_ONLY


def _simple_build_glossary_pair(en_value: str, cn_value: str) -> tuple[str, str] | None:
    if not isinstance(en_value, str) or not isinstance(cn_value, str):
        return None

    zh_candidate = cn_value
    pair = _simple_extract_bilingual_pair(cn_value)
    if pair:
        zh_candidate = pair[1]

    en_norm = _simple_plain_text_for_glossary(en_value)
    zh_norm = _simple_plain_text_for_glossary(zh_candidate)

    if not en_norm or not zh_norm:
        return None
    if _simple_is_media_path(en_norm) or _simple_is_media_path(zh_norm):
        return None
    if not _simple_contains_english(en_norm):
        return None
    if not _simple_contains_chinese(zh_norm):
        return None

    if SIMPLE_PIPELINE_GLOSSARY_MAX_CHARS_PER_SIDE > 0:
        if len(en_norm) > SIMPLE_PIPELINE_GLOSSARY_MAX_CHARS_PER_SIDE:
            return None
        if len(zh_norm) > SIMPLE_PIPELINE_GLOSSARY_MAX_CHARS_PER_SIDE:
            return None

    return en_norm, zh_norm


def _simple_extract_glossary_from_corpus(target_files, translator, glossary_data: dict, glossary_buckets):
    total_pairs = 0
    total_changed = 0
    seen_pairs: set[str] = set()

    for file_path in target_files:
        try:
            with file_path.open("r", encoding="utf-8-sig") as f:
                cn_data = json.load(f)
        except Exception as e:
            _simple_write_log(f"WARN corpus glossary skip cn load failed | {file_path} | {e}")
            continue

        en_data, _ = _simple_load_source_en_data(file_path)
        if en_data is None:
            continue

        pair_lines = []
        for parts, path_str, cn_value in _simple_collect_string_tasks(cn_data):
            if not isinstance(cn_value, str):
                continue
            if not _simple_contains_chinese(cn_value):
                continue
            if not _simple_should_include_glossary_key(path_str):
                continue
            en_value = _simple_get_value_at_path(en_data, parts)
            if not isinstance(en_value, str) or not _simple_contains_english(en_value):
                pair = _simple_extract_bilingual_pair(cn_value)
                if pair:
                    en_value = pair[0]
                else:
                    continue

            built_pair = _simple_build_glossary_pair(en_value, cn_value)
            if not built_pair:
                continue

            en_norm, zh_norm = built_pair
            pair_key = f"{en_norm.casefold()}\t{zh_norm}"
            if SIMPLE_PIPELINE_GLOSSARY_DEDUP_ENABLED and pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            pair_lines.append(f"EN: {en_norm}\nZH: {zh_norm}")
            if len(pair_lines) >= 1200:
                break

        if not pair_lines:
            continue

        total_pairs += len(pair_lines)
        for batch in _simple_iter_chunks(pair_lines, SIMPLE_EXTRACT_ONLY_PAIR_BATCH_SIZE):
            try:
                extracted = translator.extract_basic_glossary(
                    batch,
                    max_terms=SIMPLE_REPAIR_GLOSSARY_MAX_TERMS_PER_BATCH,
                    model_id=SIMPLE_REPAIR_GLOSSARY_EXTRACT_MODEL_ID,
                )
            except _SimpleFatalAPIError as fatal_error:
                _simple_write_log(f"ABORT corpus glossary extract | {fatal_error}")
                return total_pairs, total_changed, glossary_buckets, True, str(fatal_error)

            changed = _simple_upsert_adaptive_glossary(glossary_data, extracted)
            if changed > 0:
                total_changed += changed
                _simple_save_glossary_json(SIMPLE_ADAPTIVE_GLOSSARY_OUTPUT_PATH, glossary_data)
                glossary_buckets, _ = _simple_refresh_runtime_glossary(glossary_data)

    return total_pairs, total_changed, glossary_buckets, False, ""


def _simple_translate_one_file(file_path: Path, translator, limiter, glossary_data: dict, glossary_buckets, cache: dict):
    file_stats = {
        "translated": 0,
        "nochange": 0,
        "skipped": 0,
        "cached": 0,
        "failed": 0,
        "candidates": 0,
        "strict_rounds": 0,
        "strict_translated": 0,
        "strict_remaining": 0,
        "error": False,
        "aborted": False,
        "abort_reason": "",
    }

    try:
        with file_path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except Exception as e:
        file_stats["error"] = True
        print(f"⚠️ 跳过无法读取的文件: {file_path} | {e}")
        _simple_write_log(f"ERROR load json failed | {file_path} | {e}")
        return file_stats, [], glossary_buckets

    is_system_lang_file = _simple_is_system_lang_file(file_path)
    keep_original = (not is_system_lang_file) or SIMPLE_SYSTEM_LANG_KEEP_ORIGINAL

    if is_system_lang_file:
        print("检测到系统 lang 文件：放宽键筛选，默认输出纯中文。")

    tasks = _simple_collect_string_tasks(data)

    sync_enabled = (
        SIMPLE_SYNC_TRANSLATION_ENABLED
        and bool(SIMPLE_SYNC_TRANSLATION_SOURCE_KEY)
        and bool(SIMPLE_SYNC_TRANSLATION_TARGET_KEY)
        and SIMPLE_SYNC_TRANSLATION_SOURCE_KEY != SIMPLE_SYNC_TRANSLATION_TARGET_KEY
    )
    sync_map = {}
    sync_map_lock = Lock()
    sync_target_candidates = []

    def _is_sync_source(path_str: str) -> bool:
        return _simple_extract_last_key(path_str).casefold() == SIMPLE_SYNC_TRANSLATION_SOURCE_KEY

    def _is_sync_target(path_str: str) -> bool:
        return _simple_extract_last_key(path_str).casefold() == SIMPLE_SYNC_TRANSLATION_TARGET_KEY

    def _sync_put(en_text: str, cn_text: str):
        key = _simple_sync_key(en_text)
        val = str(cn_text or "").strip()
        if not key or not val:
            return
        with sync_map_lock:
            sync_map[key] = val

    def _sync_get(en_text: str) -> str:
        with sync_map_lock:
            return sync_map.get(_simple_sync_key(en_text), "")

    if sync_enabled:
        for parts, path_str, original in tasks:
            if not _is_sync_source(path_str):
                continue
            pair = _simple_extract_bilingual_pair(original)
            if pair:
                en_existing, zh_existing = pair
                _sync_put(en_existing, zh_existing)

    candidates = []
    for parts, path_str, original in tasks:
        if _simple_contains_chinese(original):
            continue
        source_text = _simple_extract_original(original)
        if _simple_should_translate(path_str, source_text, is_system_lang_file=is_system_lang_file):
            if sync_enabled and _is_sync_target(path_str):
                sync_target_candidates.append((parts, path_str, original, source_text))
                continue
            candidates.append((parts, path_str, original))

    file_stats["candidates"] = len(candidates)
    print(f"待处理文本: {len(candidates)} 条")
    _simple_write_log(f"START file={file_path} tasks={len(candidates)} cache={len(cache)}")

    results = {}
    stats_lock = Lock()
    abort_event = Event()
    abort_state = {"reason": ""}

    def _mark_abort(reason: str):
        message = str(reason or "unknown fatal API error").strip()
        with stats_lock:
            if abort_event.is_set():
                return
            abort_state["reason"] = message
            file_stats["aborted"] = True
            file_stats["abort_reason"] = message
            abort_event.set()
        _simple_write_log(f"ABORT file={file_path} | {message}")
        print(f"⛔ 停止当前文件：{file_path.name} | {message}")

    slowest = []
    translated_pairs = []
    file_prefix = str(file_path.resolve()).replace("\\", "/")

    def _run_strict_coverage_pass():
        if not SIMPLE_STRICT_COVERAGE_ENABLED or abort_event.is_set():
            return

        allow_key_filter = not SIMPLE_STRICT_COVERAGE_RELAXED_ALLOW_KEYS

        def _collect_unresolved():
            unresolved = []
            for parts, path_str, _ in tasks:
                key = tuple(parts)
                if key in results:
                    continue

                current_value = _simple_get_value_at_path(data, parts)
                if not isinstance(current_value, str):
                    continue
                if _simple_contains_chinese(current_value):
                    continue

                source_text = _simple_extract_original(current_value)
                if sync_enabled and _is_sync_target(path_str):
                    continue
                if not _simple_should_translate(
                    path_str,
                    source_text,
                    is_system_lang_file=is_system_lang_file,
                    allow_key_filter=allow_key_filter,
                ):
                    continue

                unresolved.append((parts, path_str, current_value, source_text))
            return unresolved

        for round_idx in range(1, SIMPLE_STRICT_COVERAGE_MAX_ROUNDS + 1):
            if abort_event.is_set():
                break

            unresolved = _collect_unresolved()
            if not unresolved:
                break

            if SIMPLE_STRICT_COVERAGE_MAX_ITEMS_PER_ROUND > 0:
                unresolved = unresolved[:SIMPLE_STRICT_COVERAGE_MAX_ITEMS_PER_ROUND]

            _simple_write_log(
                f"STRICT_COVERAGE round={round_idx} file={file_path.name} unresolved={len(unresolved)}"
            )
            file_stats["strict_rounds"] = round_idx
            progress = 0

            for parts, path_str, original, source_text in tqdm(
                unresolved,
                total=len(unresolved),
                desc=f"严格补翻 R{round_idx}",
                position=0,
                leave=True,
                dynamic_ncols=True,
            ):
                if abort_event.is_set():
                    break

                source_hash = _simple_hash(source_text)
                cache_key = f"{file_prefix}::{path_str}"
                if SIMPLE_CACHE_ENABLED and cache.get(cache_key) == source_hash:
                    file_stats["cached"] += 1
                    continue

                limiter.wait()
                if abort_event.is_set():
                    break

                pre, matched = _simple_apply_glossary(source_text, glossary_buckets)
                t0 = time.time()
                try:
                    cn = translator.translate(pre, path_str, matched)
                except _SimpleFatalAPIError as fatal_error:
                    _mark_abort(str(fatal_error))
                    break
                t1 = time.time()
                slowest.append((t1 - t0, f"{file_path.name} | {path_str} | strict"))

                if not cn:
                    file_stats["failed"] += 1
                    continue

                if SIMPLE_CACHE_ENABLED:
                    cache[cache_key] = source_hash

                merged_value = _simple_merge_translation(
                    path_str,
                    cn,
                    original,
                    keep_original=_simple_resolve_keep_original(path_str, keep_original),
                )
                if _simple_is_nochange_translation(source_text, merged_value, is_system_lang_file):
                    file_stats["nochange"] += 1
                    continue

                file_stats["translated"] += 1
                file_stats["strict_translated"] += 1
                progress += 1
                results[tuple(parts)] = merged_value
                if sync_enabled and _is_sync_source(path_str):
                    _sync_put(source_text, cn)

            if progress <= 0:
                break

        if not abort_event.is_set():
            file_stats["strict_remaining"] = len(_collect_unresolved())
            if file_stats["strict_rounds"] > 0:
                _simple_write_log(
                    f"STRICT_COVERAGE_DONE file={file_path.name} rounds={file_stats['strict_rounds']} "
                    f"extra_translated={file_stats['strict_translated']} remaining={file_stats['strict_remaining']}"
                )

    adaptive_enabled_for_translation = (
        SIMPLE_ADAPTIVE_GLOSSARY_ENABLED
        and SIMPLE_ADAPTIVE_GLOSSARY_DURING_TRANSLATION_ENABLED
    )
    adaptive_mode = SIMPLE_ADAPTIVE_GLOSSARY_MODE if adaptive_enabled_for_translation else "off"

    if adaptive_mode in {"entry", "entry_batch"}:
        entry_glossary_batch_size = 1 if adaptive_mode == "entry" else SIMPLE_ADAPTIVE_GLOSSARY_ENTRY_BATCH_SIZE
        entry_glossary_pair_lines = []
        entry_glossary_last_path = ""
        entry_parallel_enabled = (
            SIMPLE_ADAPTIVE_GLOSSARY_ENTRY_PARALLEL_ENABLED
            and SIMPLE_MAX_WORKERS > 1
            and len(candidates) > 1
        )
        entry_non_blocking_enabled = (
            entry_parallel_enabled
            and SIMPLE_ADAPTIVE_GLOSSARY_ENTRY_NON_BLOCKING_ENABLED
        )
        glossary_update_lock = Lock()
        entry_glossary_changed_total = 0
        entry_glossary_dirty = False
        glossary_extract_futures = []
        glossary_extractor = (
            concurrent.futures.ThreadPoolExecutor(max_workers=1)
            if entry_non_blocking_enabled
            else None
        )

        def _apply_entry_glossary_batch(batch_lines: list[str], batch_last_path: str):
            nonlocal glossary_buckets, entry_glossary_changed_total, entry_glossary_dirty
            if abort_event.is_set():
                return
            try:
                extracted = translator.extract_basic_glossary(
                    batch_lines,
                    max_terms=SIMPLE_ADAPTIVE_GLOSSARY_MAX_TERMS_PER_BATCH,
                    model_id=SIMPLE_ADAPTIVE_GLOSSARY_EXTRACT_MODEL_ID,
                )
            except _SimpleFatalAPIError as fatal_error:
                _mark_abort(str(fatal_error))
                return
            if not extracted:
                return

            with glossary_update_lock:
                changed = _simple_upsert_adaptive_glossary(glossary_data, extracted)
                if changed <= 0:
                    return

                entry_glossary_changed_total += changed
                entry_glossary_dirty = True
                if not entry_non_blocking_enabled:
                    _simple_save_glossary_json(SIMPLE_ADAPTIVE_GLOSSARY_OUTPUT_PATH, glossary_data)
                    glossary_buckets, _ = _simple_refresh_runtime_glossary(glossary_data)

                _simple_write_log(
                    f"GLOSSARY_UPDATE mode={adaptive_mode} file={file_path.name} entries={len(batch_lines)} "
                    f"last_path={batch_last_path} changed={changed} async={entry_non_blocking_enabled}"
                )

        def _drain_glossary_futures(wait_all: bool = False):
            nonlocal glossary_extract_futures
            if not glossary_extract_futures:
                return

            if abort_event.is_set():
                for fut in glossary_extract_futures:
                    fut.cancel()
                glossary_extract_futures = []
                return

            if wait_all:
                for fut in glossary_extract_futures:
                    try:
                        fut.result()
                    except Exception as e:
                        _simple_write_log(f"ERROR async glossary worker failed | {e}")
                glossary_extract_futures = []
                return

            if len(glossary_extract_futures) < SIMPLE_ADAPTIVE_GLOSSARY_EXTRACT_MAX_PENDING_BATCHES:
                return

            done, not_done = concurrent.futures.wait(
                glossary_extract_futures,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for fut in done:
                try:
                    fut.result()
                except Exception as e:
                    _simple_write_log(f"ERROR async glossary worker failed | {e}")
            glossary_extract_futures = list(not_done)

        def flush_entry_glossary_batch(force: bool = False):
            nonlocal entry_glossary_pair_lines, entry_glossary_last_path
            if not entry_glossary_pair_lines:
                return
            if abort_event.is_set():
                entry_glossary_pair_lines = []
                entry_glossary_last_path = ""
                return
            if (not force) and len(entry_glossary_pair_lines) < entry_glossary_batch_size:
                return

            batch_lines = list(entry_glossary_pair_lines)
            batch_last_path = entry_glossary_last_path
            entry_glossary_pair_lines = []
            entry_glossary_last_path = ""

            if entry_non_blocking_enabled and glossary_extractor is not None:
                future = glossary_extractor.submit(_apply_entry_glossary_batch, batch_lines, batch_last_path)
                glossary_extract_futures.append(future)
                _drain_glossary_futures(wait_all=False)
            else:
                _apply_entry_glossary_batch(batch_lines, batch_last_path)

        if entry_parallel_enabled:
            # Parallel mode favors throughput; some new terms may apply only to later queued entries.
            def entry_worker(item):
                path_str = ""
                original = ""
                try:
                    if abort_event.is_set():
                        return None
                    parts, path_str, original = item
                    source_text = _simple_extract_original(original)
                    source_hash = _simple_hash(source_text)
                    cache_key = f"{file_prefix}::{path_str}"
                    if SIMPLE_CACHE_ENABLED and cache.get(cache_key) == source_hash:
                        with stats_lock:
                            file_stats["cached"] += 1
                        return None

                    limiter.wait()
                    if abort_event.is_set():
                        return None
                    pre, matched = _simple_apply_glossary(source_text, glossary_buckets)
                    t0 = time.time()
                    try:
                        cn = translator.translate(pre, path_str, matched)
                    except _SimpleFatalAPIError as fatal_error:
                        _mark_abort(str(fatal_error))
                        return None
                    t1 = time.time()
                    with stats_lock:
                        slowest.append((t1 - t0, f"{file_path.name} | {path_str}"))

                    if not cn:
                        with stats_lock:
                            file_stats["failed"] += 1
                        return None

                    merged_value = _simple_merge_translation(
                        path_str,
                        cn,
                        original,
                        keep_original=_simple_resolve_keep_original(path_str, keep_original),
                    )
                    with stats_lock:
                        if SIMPLE_CACHE_ENABLED:
                            cache[cache_key] = source_hash
                        if _simple_is_nochange_translation(source_text, merged_value, is_system_lang_file):
                            file_stats["nochange"] += 1
                            return None
                        file_stats["translated"] += 1
                        if sync_enabled and _is_sync_source(path_str):
                            _sync_put(source_text, cn)
                    return parts, path_str, source_text, cn, merged_value
                except Exception as e:
                    _simple_write_log(
                        f"ERROR entry worker failed | {file_path} | {path_str} | {e} | text={_simple_snip(original)}"
                    )
                    return None

            with concurrent.futures.ThreadPoolExecutor(max_workers=SIMPLE_MAX_WORKERS) as exe:
                for res in tqdm(exe.map(entry_worker, candidates), total=len(candidates), desc="翻译", position=0, leave=True, dynamic_ncols=True):
                    if abort_event.is_set():
                        break
                    if not res:
                        continue
                    parts, path_str, source_text, cn, merged_value = res
                    results[tuple(parts)] = merged_value
                    entry_glossary_pair_lines.append(f"EN: {source_text}\nZH: {cn}")
                    entry_glossary_last_path = path_str
                    flush_entry_glossary_batch()
        else:
            for item in tqdm(candidates, total=len(candidates), desc="翻译", position=0, leave=True, dynamic_ncols=True):
                if abort_event.is_set():
                    break
                parts, path_str, original = item
                source_text = _simple_extract_original(original)
                source_hash = _simple_hash(source_text)
                cache_key = f"{file_prefix}::{path_str}"
                if SIMPLE_CACHE_ENABLED and cache.get(cache_key) == source_hash:
                    file_stats["cached"] += 1
                    continue

                limiter.wait()
                if abort_event.is_set():
                    break
                pre, matched = _simple_apply_glossary(source_text, glossary_buckets)
                t0 = time.time()
                try:
                    cn = translator.translate(pre, path_str, matched)
                except _SimpleFatalAPIError as fatal_error:
                    _mark_abort(str(fatal_error))
                    break
                t1 = time.time()
                slowest.append((t1 - t0, f"{file_path.name} | {path_str}"))

                if not cn:
                    file_stats["failed"] += 1
                    continue

                if SIMPLE_CACHE_ENABLED:
                    cache[cache_key] = source_hash

                merged_value = _simple_merge_translation(
                    path_str,
                    cn,
                    original,
                    keep_original=_simple_resolve_keep_original(path_str, keep_original),
                )
                if _simple_is_nochange_translation(source_text, merged_value, is_system_lang_file):
                    file_stats["nochange"] += 1
                    continue

                file_stats["translated"] += 1
                results[tuple(parts)] = merged_value
                if sync_enabled and _is_sync_source(path_str):
                    _sync_put(source_text, cn)

                entry_glossary_pair_lines.append(f"EN: {source_text}\nZH: {cn}")
                entry_glossary_last_path = path_str
                flush_entry_glossary_batch()

        flush_entry_glossary_batch(force=True)
        if entry_non_blocking_enabled:
            _drain_glossary_futures(wait_all=True)
            if glossary_extractor is not None:
                glossary_extractor.shutdown(wait=True, cancel_futures=True)
            if entry_glossary_dirty and not abort_event.is_set():
                _simple_save_glossary_json(SIMPLE_ADAPTIVE_GLOSSARY_OUTPUT_PATH, glossary_data)
                glossary_buckets, _ = _simple_refresh_runtime_glossary(glossary_data)
                _simple_write_log(
                    f"GLOSSARY_UPDATE_SUMMARY mode={adaptive_mode} file={file_path.name} changed_total={entry_glossary_changed_total}"
                )

        if sync_enabled and not abort_event.is_set() and sync_target_candidates:
            sync_applied = 0
            sync_missed = 0
            for parts, path_str, original, source_text in sync_target_candidates:
                cn = _sync_get(source_text)
                if not cn:
                    sync_missed += 1
                    if not SIMPLE_SYNC_TRANSLATION_ONLY_REUSE:
                        limiter.wait()
                        if abort_event.is_set():
                            break
                        pre, matched = _simple_apply_glossary(source_text, glossary_buckets)
                        t0 = time.time()
                        try:
                            cn = translator.translate(pre, path_str, matched)
                        except _SimpleFatalAPIError as fatal_error:
                            _mark_abort(str(fatal_error))
                            break
                        t1 = time.time()
                        slowest.append((t1 - t0, f"{file_path.name} | {path_str} | sync-fallback"))
                        if not cn:
                            file_stats["failed"] += 1
                            continue

                merged_value = _simple_merge_translation(
                    path_str,
                    cn,
                    original,
                    keep_original=_simple_resolve_keep_original(path_str, keep_original),
                )
                if _simple_is_nochange_translation(source_text, merged_value, is_system_lang_file):
                    file_stats["nochange"] += 1
                    continue

                results[tuple(parts)] = merged_value
                file_stats["translated"] += 1
                sync_applied += 1

            _simple_write_log(
                f"SYNC_TRANSLATION file={file_path.name} source={SIMPLE_SYNC_TRANSLATION_SOURCE_KEY} "
                f"target={SIMPLE_SYNC_TRANSLATION_TARGET_KEY} applied={sync_applied} missed={sync_missed} "
                f"only_reuse={SIMPLE_SYNC_TRANSLATION_ONLY_REUSE}"
            )

        _run_strict_coverage_pass()

        for parts, value in results.items():
            _simple_set_value_at_path(data, list(parts), value)

        if results:
            with file_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        if abort_event.is_set() and not file_stats["abort_reason"]:
            file_stats["aborted"] = True
            file_stats["abort_reason"] = abort_state["reason"] or "fatal API error"

        return file_stats, slowest, glossary_buckets

    def worker(item):
        try:
            if abort_event.is_set():
                return None
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
            if abort_event.is_set():
                return None
            pre, matched = _simple_apply_glossary(source_text, glossary_buckets)
            t0 = time.time()
            try:
                cn = translator.translate(pre, path_str, matched)
            except _SimpleFatalAPIError as fatal_error:
                _mark_abort(str(fatal_error))
                return None
            t1 = time.time()
            with stats_lock:
                slowest.append((t1 - t0, f"{file_path.name} | {path_str}"))
            if not cn:
                with stats_lock:
                    file_stats["failed"] += 1
                return None
            if SIMPLE_CACHE_ENABLED:
                cache[cache_key] = source_hash
            merged_value = _simple_merge_translation(
                path_str,
                cn,
                original,
                keep_original=_simple_resolve_keep_original(path_str, keep_original),
            )
            with stats_lock:
                if _simple_is_nochange_translation(source_text, merged_value, is_system_lang_file):
                    file_stats["nochange"] += 1
                    return None
                file_stats["translated"] += 1
                translated_pairs.append((source_text, cn))
                if sync_enabled and _is_sync_source(path_str):
                    _sync_put(source_text, cn)
            return parts, merged_value
        except Exception as e:
            _simple_write_log(
                f"ERROR worker failed | {file_path} | {path_str} | {e} | text={_simple_snip(original)}"
            )
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=SIMPLE_MAX_WORKERS) as exe:
        for res in tqdm(exe.map(worker, candidates), total=len(candidates), desc="翻译", position=0, leave=True, dynamic_ncols=True):
            if abort_event.is_set():
                break
            if not res:
                continue
            parts, value = res
            results[tuple(parts)] = value

    if sync_enabled and not abort_event.is_set() and sync_target_candidates:
        sync_applied = 0
        sync_missed = 0
        for parts, path_str, original, source_text in sync_target_candidates:
            cn = _sync_get(source_text)
            if not cn:
                sync_missed += 1
                if not SIMPLE_SYNC_TRANSLATION_ONLY_REUSE:
                    limiter.wait()
                    if abort_event.is_set():
                        break
                    pre, matched = _simple_apply_glossary(source_text, glossary_buckets)
                    t0 = time.time()
                    try:
                        cn = translator.translate(pre, path_str, matched)
                    except _SimpleFatalAPIError as fatal_error:
                        _mark_abort(str(fatal_error))
                        break
                    t1 = time.time()
                    slowest.append((t1 - t0, f"{file_path.name} | {path_str} | sync-fallback"))
                    if not cn:
                        file_stats["failed"] += 1
                        continue

            merged_value = _simple_merge_translation(
                path_str,
                cn,
                original,
                keep_original=_simple_resolve_keep_original(path_str, keep_original),
            )
            if _simple_is_nochange_translation(source_text, merged_value, is_system_lang_file):
                file_stats["nochange"] += 1
                continue

            results[tuple(parts)] = merged_value
            file_stats["translated"] += 1
            sync_applied += 1

        _simple_write_log(
            f"SYNC_TRANSLATION file={file_path.name} source={SIMPLE_SYNC_TRANSLATION_SOURCE_KEY} "
            f"target={SIMPLE_SYNC_TRANSLATION_TARGET_KEY} applied={sync_applied} missed={sync_missed} "
            f"only_reuse={SIMPLE_SYNC_TRANSLATION_ONLY_REUSE}"
        )

    _run_strict_coverage_pass()

    for parts, value in results.items():
        _simple_set_value_at_path(data, list(parts), value)

    if results:
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    if adaptive_enabled_for_translation and SIMPLE_ADAPTIVE_GLOSSARY_MODE == "file" and translated_pairs:
        pair_lines = [f"EN: {en}\nZH: {zh}" for en, zh in translated_pairs[:120]]
        try:
            extracted = translator.extract_basic_glossary(
                pair_lines,
                max_terms=SIMPLE_ADAPTIVE_GLOSSARY_MAX_TERMS_PER_BATCH,
            )
        except _SimpleFatalAPIError as fatal_error:
            _mark_abort(str(fatal_error))
            extracted = {}
        changed = _simple_upsert_adaptive_glossary(glossary_data, extracted)
        if changed > 0:
            _simple_save_glossary_json(SIMPLE_ADAPTIVE_GLOSSARY_OUTPUT_PATH, glossary_data)
            glossary_buckets, _ = _simple_refresh_runtime_glossary(glossary_data)
            _simple_write_log(f"GLOSSARY_UPDATE mode=file file={file_path.name} changed={changed}")

    if abort_event.is_set() and not file_stats["abort_reason"]:
        file_stats["aborted"] = True
        file_stats["abort_reason"] = abort_state["reason"] or "fatal API error"

    return file_stats, slowest, glossary_buckets


def _simple_repair_from_report(translator, limiter, glossary_data: dict, glossary_buckets, cache: dict):
    if not SIMPLE_REPAIR_REPORT_PATH.exists():
        print(f"❌ 修复报告不存在: {SIMPLE_REPAIR_REPORT_PATH}")
        _simple_write_log(f"ERROR repair report missing | {SIMPLE_REPAIR_REPORT_PATH}")
        return {
            "files": 0,
            "items": 0,
            "repaired": 0,
            "nochange": 0,
            "skipped": 0,
            "cached": 0,
            "failed": 0,
            "file_errors": 0,
            "aborted": False,
            "abort_reason": "",
        }, glossary_buckets

    try:
        report_data = json.loads(SIMPLE_REPAIR_REPORT_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"❌ 修复报告读取失败: {SIMPLE_REPAIR_REPORT_PATH} | {e}")
        _simple_write_log(f"ERROR repair report load failed | {SIMPLE_REPAIR_REPORT_PATH} | {e}")
        return {
            "files": 0,
            "items": 0,
            "repaired": 0,
            "nochange": 0,
            "skipped": 0,
            "cached": 0,
            "failed": 0,
            "file_errors": 1,
            "aborted": False,
            "abort_reason": "",
        }, glossary_buckets

    if not isinstance(report_data, dict):
        print("❌ 修复报告格式错误，根对象必须是字典")
        _simple_write_log("ERROR repair report invalid root type")
        return {
            "files": 0,
            "items": 0,
            "repaired": 0,
            "nochange": 0,
            "skipped": 0,
            "cached": 0,
            "failed": 0,
            "file_errors": 1,
            "aborted": False,
            "abort_reason": "",
        }, glossary_buckets

    stats = {
        "files": 0,
        "items": 0,
        "repaired": 0,
        "nochange": 0,
        "skipped": 0,
        "cached": 0,
        "failed": 0,
        "file_errors": 0,
        "aborted": False,
        "abort_reason": "",
    }

    glossary_pair_lines = []

    def _flush_repair_glossary_batch(force: bool = False):
        nonlocal glossary_pair_lines, glossary_buckets
        if not SIMPLE_REPAIR_GLOSSARY_EXPAND_ENABLED:
            return
        if not glossary_pair_lines:
            return
        if (not force) and len(glossary_pair_lines) < SIMPLE_REPAIR_ENTRY_BATCH_SIZE:
            return

        batch_lines = list(glossary_pair_lines)
        glossary_pair_lines = []
        try:
            extracted = translator.extract_basic_glossary(
                batch_lines,
                max_terms=SIMPLE_REPAIR_GLOSSARY_MAX_TERMS_PER_BATCH,
                model_id=SIMPLE_REPAIR_GLOSSARY_EXTRACT_MODEL_ID,
            )
        except _SimpleFatalAPIError as fatal_error:
            stats["aborted"] = True
            stats["abort_reason"] = str(fatal_error)
            _simple_write_log(f"ABORT repair glossary extract | {fatal_error}")
            return

        changed = _simple_upsert_adaptive_glossary(glossary_data, extracted)
        if changed > 0:
            _simple_save_glossary_json(SIMPLE_ADAPTIVE_GLOSSARY_OUTPUT_PATH, glossary_data)
            glossary_buckets, _ = _simple_refresh_runtime_glossary(glossary_data)
            _simple_write_log(f"REPAIR_GLOSSARY_UPDATE changed={changed} lines={len(batch_lines)}")

    file_items = list(report_data.items())
    for idx, (file_key, findings) in enumerate(file_items, start=1):
        if stats["aborted"]:
            break
        if not isinstance(findings, list) or not findings:
            continue

        target_file = _simple_resolve_report_file_path(file_key)
        if not target_file.exists() or not target_file.is_file():
            stats["file_errors"] += 1
            print(f"⚠️ 报告文件路径不存在，跳过: {file_key}")
            _simple_write_log(f"WARN repair missing target file | {file_key} -> {target_file}")
            continue

        try:
            with target_file.open("r", encoding="utf-8-sig") as f:
                data = json.load(f)
        except Exception as e:
            stats["file_errors"] += 1
            print(f"⚠️ 读取失败，跳过: {target_file} | {e}")
            _simple_write_log(f"ERROR repair load target failed | {target_file} | {e}")
            continue

        stats["files"] += 1
        file_changed = 0
        file_prefix = str(target_file.resolve()).replace("\\", "/")
        print(f"\n[{idx}/{len(file_items)}] 修复文件: {target_file} | 条目={len(findings)}")

        for item in tqdm(findings, total=len(findings), desc="修复漏词", position=0, leave=True, dynamic_ncols=True):
            if stats["aborted"]:
                break
            if not isinstance(item, dict):
                stats["skipped"] += 1
                continue

            path_str = str(item.get("path", "")).strip()
            if not path_str:
                stats["skipped"] += 1
                continue

            parts = _simple_parse_report_path(path_str)
            if not parts:
                stats["skipped"] += 1
                continue

            stats["items"] += 1
            current_value = _simple_get_value_at_path(data, parts)
            if not isinstance(current_value, str):
                stats["skipped"] += 1
                continue
            if not _simple_contains_english(current_value):
                stats["skipped"] += 1
                continue
            if not _simple_is_significant_english_leak(current_value):
                stats["skipped"] += 1
                continue

            source_hash = _simple_hash(current_value)
            cache_key = f"{file_prefix}::repair::{path_str}"
            if SIMPLE_CACHE_ENABLED and cache.get(cache_key) == source_hash:
                stats["cached"] += 1
                continue

            context_json = ""
            if SIMPLE_REPAIR_CONTEXT_ENABLED:
                context_parts = _simple_find_entry_context_parts(parts)
                context_obj = _simple_get_value_at_path(data, context_parts) if context_parts else None
                if isinstance(context_obj, (dict, list)):
                    try:
                        context_json = _simple_trim_json_text(
                            json.dumps(context_obj, ensure_ascii=False, indent=2),
                            SIMPLE_REPAIR_CONTEXT_MAX_CHARS,
                        )
                    except Exception:
                        context_json = ""

            english_words = item.get("english_words", [])
            if not isinstance(english_words, list):
                english_words = []
            en_preview = str(item.get("en_preview", "") or "")

            limiter.wait()
            pre, matched = _simple_apply_glossary(current_value, glossary_buckets)
            try:
                repaired = translator.repair_mixed_text(
                    pre,
                    path_str,
                    glossary_terms=matched,
                    en_preview=en_preview,
                    english_words=english_words,
                    entry_context_json=context_json,
                )
            except _SimpleFatalAPIError as fatal_error:
                stats["aborted"] = True
                stats["abort_reason"] = str(fatal_error)
                _simple_write_log(f"ABORT repair file={target_file} path={path_str} | {fatal_error}")
                break

            if not repaired:
                stats["failed"] += 1
                continue

            repaired_norm = re.sub(r"\s+", " ", repaired).strip()
            current_norm = re.sub(r"\s+", " ", current_value).strip()
            if not repaired_norm or repaired_norm.casefold() == current_norm.casefold():
                stats["nochange"] += 1
                continue

            if _simple_is_significant_english_leak(repaired):
                stats["failed"] += 1
                _simple_write_log(
                    f"WARN repair still has significant english | {target_file} | {path_str} | {repaired[:160]}"
                )
                continue

            final_value = repaired if SIMPLE_REPAIR_FORCE_CN_ONLY else _simple_merge_translation(
                path_str,
                repaired,
                current_value,
                keep_original=_simple_resolve_keep_original(path_str, True),
            )

            _simple_set_value_at_path(data, parts, final_value)
            file_changed += 1
            stats["repaired"] += 1
            if SIMPLE_CACHE_ENABLED:
                cache[cache_key] = source_hash

            if SIMPLE_REPAIR_GLOSSARY_EXPAND_ENABLED and en_preview and _simple_contains_chinese(final_value):
                glossary_pair_lines.append(f"EN: {en_preview}\nZH: {final_value}")
                _flush_repair_glossary_batch(force=False)

        if file_changed > 0:
            with target_file.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"文件修复写回: {target_file.name} | 修复={file_changed}")

        _flush_repair_glossary_batch(force=False)

    _flush_repair_glossary_batch(force=True)

    if stats["aborted"]:
        print(f"⚠️ 上下文修复提前终止: {stats['abort_reason']}")
    else:
        print("✅ 完成：report 上下文修复结束")

    print(
        f"修复统计：文件={stats['files']} 条目={stats['items']} 修复={stats['repaired']} "
        f"未变化={stats['nochange']} 跳过={stats['skipped']} 缓存跳过={stats['cached']} "
        f"失败={stats['failed']} 文件错误={stats['file_errors']}"
    )
    _simple_write_log(
        f"REPAIR_DONE files={stats['files']} items={stats['items']} repaired={stats['repaired']} "
        f"nochange={stats['nochange']} skipped={stats['skipped']} cached={stats['cached']} "
        f"failed={stats['failed']} file_errors={stats['file_errors']} aborted={stats['aborted']} "
        f"reason={stats['abort_reason']}"
    )
    return stats, glossary_buckets


def _simple_optimize_existing_files(target_files, translator, limiter, glossary_data: dict, glossary_buckets, cache: dict):
    stats = {
        "files": 0,
        "items": 0,
        "optimized": 0,
        "nochange": 0,
        "skipped": 0,
        "cached": 0,
        "failed": 0,
        "file_errors": 0,
        "aborted": False,
        "abort_reason": "",
    }

    glossary_pair_lines = []

    def _flush_glossary_batch(force: bool = False):
        nonlocal glossary_pair_lines, glossary_buckets
        if not SIMPLE_REPAIR_GLOSSARY_EXPAND_ENABLED:
            return
        if not glossary_pair_lines:
            return
        if (not force) and len(glossary_pair_lines) < SIMPLE_REPAIR_ENTRY_BATCH_SIZE:
            return

        lines = list(glossary_pair_lines)
        glossary_pair_lines = []
        try:
            extracted = translator.extract_basic_glossary(
                lines,
                max_terms=SIMPLE_REPAIR_GLOSSARY_MAX_TERMS_PER_BATCH,
                model_id=SIMPLE_REPAIR_GLOSSARY_EXTRACT_MODEL_ID,
            )
        except _SimpleFatalAPIError as fatal_error:
            stats["aborted"] = True
            stats["abort_reason"] = str(fatal_error)
            _simple_write_log(f"ABORT optimize glossary extract | {fatal_error}")
            return

        changed = _simple_upsert_adaptive_glossary(glossary_data, extracted)
        if changed > 0:
            _simple_save_glossary_json(SIMPLE_ADAPTIVE_GLOSSARY_OUTPUT_PATH, glossary_data)
            glossary_buckets, _ = _simple_refresh_runtime_glossary(glossary_data)
            _simple_write_log(f"OPTIMIZE_GLOSSARY_UPDATE changed={changed} lines={len(lines)}")

    for idx, file_path in enumerate(target_files, start=1):
        if stats["aborted"]:
            break

        try:
            with file_path.open("r", encoding="utf-8-sig") as f:
                data = json.load(f)
        except Exception as e:
            stats["file_errors"] += 1
            print(f"⚠️ 读取失败，跳过: {file_path} | {e}")
            _simple_write_log(f"ERROR optimize load failed | {file_path} | {e}")
            continue

        findings = _simple_collect_optimize_findings_for_file(file_path, data)
        if not findings:
            continue

        stats["files"] += 1
        file_prefix = str(file_path.resolve()).replace("\\", "/")
        file_changed = 0
        print(f"\n[{idx}/{len(target_files)}] 优化文件: {file_path} | 候选={len(findings)}")

        prepared = []
        for item in findings:
            path_str = str(item.get("path", "")).strip()
            parts = _simple_parse_report_path(path_str)
            if not path_str or not parts:
                stats["skipped"] += 1
                continue

            current_value = _simple_get_value_at_path(data, parts)
            if not isinstance(current_value, str):
                stats["skipped"] += 1
                continue
            if not _simple_contains_english(current_value):
                stats["skipped"] += 1
                continue

            stats["items"] += 1
            source_hash = _simple_hash(current_value)
            cache_key = f"{file_prefix}::optimize::{path_str}"
            if SIMPLE_CACHE_ENABLED and cache.get(cache_key) == source_hash:
                stats["cached"] += 1
                continue

            pre, matched = _simple_apply_glossary(current_value, glossary_buckets)
            context_json = _simple_build_repair_context_json(data, parts)
            prepared.append(
                {
                    "parts": parts,
                    "path": path_str,
                    "source_hash": source_hash,
                    "cache_key": cache_key,
                    "current": current_value,
                    "pre": pre,
                    "glossary_terms": matched,
                    "english_words": item.get("english_words", []),
                    "en_preview": item.get("en_preview", ""),
                    "entry_context_json": context_json,
                }
            )

        if not prepared:
            continue

        progress = tqdm(prepared, total=len(prepared), desc="优化修复", position=0, leave=True, dynamic_ncols=True)
        pending = list(prepared)

        while pending and not stats["aborted"]:
            if SIMPLE_OPTIMIZE_GLOSSARY_PATCH_ONLY or (not SIMPLE_OPTIMIZE_MODEL_ENABLED):
                item = pending.pop(0)
                progress.update(1)
                repaired = item["pre"]
                repaired_norm = re.sub(r"\s+", " ", repaired).strip()
                current_norm = re.sub(r"\s+", " ", item["current"]).strip()
                if not repaired_norm or repaired_norm.casefold() == current_norm.casefold():
                    stats["nochange"] += 1
                    continue

                _simple_set_value_at_path(data, item["parts"], repaired)
                file_changed += 1
                stats["optimized"] += 1
                if SIMPLE_CACHE_ENABLED:
                    cache[item["cache_key"]] = item["source_hash"]
                if SIMPLE_REPAIR_GLOSSARY_EXPAND_ENABLED and _simple_contains_chinese(repaired):
                    en_preview = item.get("en_preview") or _simple_extract_original(item["current"])
                    glossary_pair_lines.append(f"EN: {en_preview}\nZH: {repaired}")
                    _flush_glossary_batch(force=False)
                continue

            if SIMPLE_OPTIMIZE_BATCH_ENABLED:
                batch = pending[:SIMPLE_OPTIMIZE_BATCH_SIZE]
                pending = pending[SIMPLE_OPTIMIZE_BATCH_SIZE:]
                progress.update(len(batch))
                limiter.wait()
                try:
                    outputs = translator.repair_mixed_batch(
                        [
                            {
                                "path": it["path"],
                                "text": it["pre"],
                                "glossary_terms": it["glossary_terms"],
                                "en_preview": it["en_preview"],
                                "english_words": it["english_words"],
                                "entry_context_json": it["entry_context_json"],
                            }
                            for it in batch
                        ]
                    )
                except _SimpleFatalAPIError as fatal_error:
                    stats["aborted"] = True
                    stats["abort_reason"] = str(fatal_error)
                    break

                for it in batch:
                    repaired = outputs.get(it["path"], "")
                    if not repaired:
                        stats["failed"] += 1
                        continue

                    repaired_norm = re.sub(r"\s+", " ", repaired).strip()
                    current_norm = re.sub(r"\s+", " ", it["current"]).strip()
                    if (not repaired_norm) or repaired_norm.casefold() == current_norm.casefold():
                        stats["nochange"] += 1
                        continue
                    if _simple_is_significant_english_leak(repaired):
                        stats["failed"] += 1
                        continue

                    _simple_set_value_at_path(data, it["parts"], repaired)
                    file_changed += 1
                    stats["optimized"] += 1
                    if SIMPLE_CACHE_ENABLED:
                        cache[it["cache_key"]] = it["source_hash"]
                    if SIMPLE_REPAIR_GLOSSARY_EXPAND_ENABLED and _simple_contains_chinese(repaired):
                        en_preview = it.get("en_preview") or _simple_extract_original(it["current"])
                        glossary_pair_lines.append(f"EN: {en_preview}\nZH: {repaired}")
                        _flush_glossary_batch(force=False)
            else:
                item = pending.pop(0)
                progress.update(1)
                limiter.wait()
                try:
                    repaired = translator.repair_mixed_text(
                        item["pre"],
                        item["path"],
                        glossary_terms=item["glossary_terms"],
                        en_preview=item["en_preview"],
                        english_words=item["english_words"],
                        entry_context_json=item["entry_context_json"],
                    )
                except _SimpleFatalAPIError as fatal_error:
                    stats["aborted"] = True
                    stats["abort_reason"] = str(fatal_error)
                    break

                if not repaired:
                    stats["failed"] += 1
                    continue
                repaired_norm = re.sub(r"\s+", " ", repaired).strip()
                current_norm = re.sub(r"\s+", " ", item["current"]).strip()
                if (not repaired_norm) or repaired_norm.casefold() == current_norm.casefold():
                    stats["nochange"] += 1
                    continue
                if _simple_is_significant_english_leak(repaired):
                    stats["failed"] += 1
                    continue

                _simple_set_value_at_path(data, item["parts"], repaired)
                file_changed += 1
                stats["optimized"] += 1
                if SIMPLE_CACHE_ENABLED:
                    cache[item["cache_key"]] = item["source_hash"]
                if SIMPLE_REPAIR_GLOSSARY_EXPAND_ENABLED and _simple_contains_chinese(repaired):
                    en_preview = item.get("en_preview") or _simple_extract_original(item["current"])
                    glossary_pair_lines.append(f"EN: {en_preview}\nZH: {repaired}")
                    _flush_glossary_batch(force=False)

        progress.close()

        if file_changed > 0:
            with file_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"文件优化写回: {file_path.name} | 优化={file_changed}")

        _flush_glossary_batch(force=False)

    _flush_glossary_batch(force=True)

    if stats["aborted"]:
        print(f"⚠️ 优化提前终止: {stats['abort_reason']}")
    else:
        print("✅ 完成：全量优化结束")
    print(
        f"优化统计：文件={stats['files']} 条目={stats['items']} 优化={stats['optimized']} "
        f"未变化={stats['nochange']} 跳过={stats['skipped']} 缓存跳过={stats['cached']} "
        f"失败={stats['failed']} 文件错误={stats['file_errors']}"
    )
    _simple_write_log(
        f"OPTIMIZE_DONE files={stats['files']} items={stats['items']} optimized={stats['optimized']} "
        f"nochange={stats['nochange']} skipped={stats['skipped']} cached={stats['cached']} "
        f"failed={stats['failed']} file_errors={stats['file_errors']} aborted={stats['aborted']} "
        f"reason={stats['abort_reason']}"
    )

    return stats, glossary_buckets


def _simple_quality_review_and_repair_files(target_files, translator, limiter, glossary_data: dict, glossary_buckets, cache: dict):
    stats = {
        "files": 0,
        "items": 0,
        "reviewed": 0,
        "repaired": 0,
        "nochange": 0,
        "skipped": 0,
        "cached": 0,
        "failed": 0,
        "file_errors": 0,
        "aborted": False,
        "abort_reason": "",
    }
    quality_rows: list[dict[str, Any]] = []
    glossary_pair_lines = []

    glossary_changed = _simple_quality_review_glossary(translator, glossary_data, target_files)
    if glossary_changed > 0:
        glossary_buckets, _ = _simple_refresh_runtime_glossary(glossary_data)

    def _flush_glossary_batch(force: bool = False):
        nonlocal glossary_pair_lines, glossary_buckets
        if not SIMPLE_REPAIR_GLOSSARY_EXPAND_ENABLED:
            return
        if not glossary_pair_lines:
            return
        if (not force) and len(glossary_pair_lines) < SIMPLE_REPAIR_ENTRY_BATCH_SIZE:
            return
        lines = list(glossary_pair_lines)
        glossary_pair_lines = []
        try:
            extracted = translator.extract_basic_glossary(
                lines,
                max_terms=SIMPLE_REPAIR_GLOSSARY_MAX_TERMS_PER_BATCH,
                model_id=SIMPLE_REPAIR_GLOSSARY_EXTRACT_MODEL_ID,
            )
        except _SimpleFatalAPIError as fatal_error:
            stats["aborted"] = True
            stats["abort_reason"] = str(fatal_error)
            return
        changed = _simple_upsert_adaptive_glossary(glossary_data, extracted)
        if changed > 0:
            _simple_save_glossary_json(SIMPLE_ADAPTIVE_GLOSSARY_OUTPUT_PATH, glossary_data)
            glossary_buckets, _ = _simple_refresh_runtime_glossary(glossary_data)

    for idx, file_path in enumerate(target_files, start=1):
        if stats["aborted"]:
            break
        try:
            with file_path.open("r", encoding="utf-8-sig") as f:
                data = json.load(f)
        except Exception as e:
            stats["file_errors"] += 1
            _simple_write_log(f"ERROR quality load cn failed | {file_path} | {e}")
            continue

        en_data, en_file = _simple_load_source_en_data(file_path)
        if en_data is None:
            stats["skipped"] += 1
            _simple_write_log(f"WARN quality missing source en | cn={file_path} | en={en_file}")
            continue

        candidates = _simple_collect_quality_candidates(file_path, data, en_data)
        if not candidates:
            continue

        stats["files"] += 1
        file_prefix = str(file_path.resolve()).replace("\\", "/")
        file_changed = 0
        print(f"\n[{idx}/{len(target_files)}] 质量审校: {file_path} | 候选={len(candidates)}")

        for item in tqdm(candidates, total=len(candidates), desc="质量修复", position=0, leave=True, dynamic_ncols=True):
            if stats["aborted"]:
                break
            stats["items"] += 1
            path_str = item["path"]
            current_cn = item["current_cn"]
            source_en = item["source_en"]
            source_en_backup = ""
            if SIMPLE_QUALITY_USE_BILINGUAL_BACKUP_SOURCE:
                pair = _simple_extract_bilingual_pair(current_cn)
                if pair:
                    source_en_backup = str(pair[0] or "").strip()
            parts = item["parts"]
            cache_key = f"{file_prefix}::quality::{path_str}"
            source_hash = _simple_hash(f"{source_en}\n{current_cn}")

            if SIMPLE_CACHE_ENABLED and cache.get(cache_key) == source_hash:
                stats["cached"] += 1
                continue

            context_json = ""
            if SIMPLE_QUALITY_CONTEXT_SCOPE == "entry":
                context_json = _simple_build_repair_context_json(data, parts)
            elif SIMPLE_QUALITY_CONTEXT_SCOPE == "file":
                try:
                    context_json = _simple_trim_json_text(
                        json.dumps(data, ensure_ascii=False, indent=2),
                        SIMPLE_REPAIR_CONTEXT_MAX_CHARS,
                    )
                except Exception:
                    context_json = ""

            _, matched = _simple_apply_glossary(source_en, glossary_buckets)
            limiter.wait()
            try:
                reviewed = translator.quality_review_and_repair_field(
                    path_str=path_str,
                    current_cn_text=current_cn,
                    source_en_text=source_en,
                    source_en_backup_text=source_en_backup,
                    entry_context_json=context_json,
                    glossary_terms=matched,
                    force_repair=False,
                )
            except _SimpleFatalAPIError as fatal_error:
                stats["aborted"] = True
                stats["abort_reason"] = str(fatal_error)
                break

            stats["reviewed"] += 1
            should_repair = bool(reviewed.get("should_repair", False))
            repaired_text = str(reviewed.get("repaired_text", "") or "")
            issues = reviewed.get("issues", [])
            confidence = float(reviewed.get("confidence", 0.0) or 0.0)
            risk_flags = reviewed.get("risk_flags", [])

            if SIMPLE_QUALITY_STRICT_SEMANTIC_GUARD and (confidence < 0.35) and should_repair:
                should_repair = False
                risk_flags = list(risk_flags) + ["low_confidence_guard"]

            guard_failed = False
            if should_repair and repaired_text:
                if SIMPLE_QUALITY_FAIL_ON_MARKUP_RISK and (not _simple_is_markup_safe(current_cn, repaired_text)):
                    guard_failed = True
                    risk_flags = list(risk_flags) + ["markup_guard_failed"]
                if SIMPLE_QUALITY_REQUIRE_NUMERIC_CONSISTENCY and (not _simple_is_numeric_consistent(source_en, repaired_text)):
                    guard_failed = True
                    risk_flags = list(risk_flags) + ["numeric_guard_failed"]

            if should_repair and guard_failed and SIMPLE_QUALITY_SECOND_PASS_ENABLED:
                try:
                    limiter.wait()
                    reviewed2 = translator.quality_review_and_repair_field(
                        path_str=path_str,
                        current_cn_text=current_cn,
                        source_en_text=source_en,
                        source_en_backup_text=source_en_backup,
                        entry_context_json=context_json,
                        glossary_terms=matched,
                        force_repair=True,
                    )
                    repaired2 = str(reviewed2.get("repaired_text", "") or "")
                    if repaired2:
                        second_guard_failed = False
                        if SIMPLE_QUALITY_FAIL_ON_MARKUP_RISK and (not _simple_is_markup_safe(current_cn, repaired2)):
                            second_guard_failed = True
                        if SIMPLE_QUALITY_REQUIRE_NUMERIC_CONSISTENCY and (not _simple_is_numeric_consistent(source_en, repaired2)):
                            second_guard_failed = True
                        if not second_guard_failed:
                            repaired_text = repaired2
                            guard_failed = False
                            risk_flags = [x for x in risk_flags if x not in {"markup_guard_failed", "numeric_guard_failed"}]
                            risk_flags = list(risk_flags) + ["second_pass_applied"]
                except _SimpleFatalAPIError as fatal_error:
                    stats["aborted"] = True
                    stats["abort_reason"] = str(fatal_error)
                    break

            final_applied = False
            if should_repair and repaired_text and (not guard_failed):
                repaired_norm = re.sub(r"\s+", " ", repaired_text).strip()
                current_norm = re.sub(r"\s+", " ", current_cn).strip()
                if repaired_norm and repaired_norm.casefold() != current_norm.casefold():
                    _simple_set_value_at_path(data, parts, repaired_text)
                    file_changed += 1
                    stats["repaired"] += 1
                    final_applied = True
                    if SIMPLE_CACHE_ENABLED:
                        cache[cache_key] = source_hash
                    if SIMPLE_REPAIR_GLOSSARY_EXPAND_ENABLED and _simple_contains_chinese(repaired_text):
                        glossary_pair_lines.append(f"EN: {source_en}\nZH: {repaired_text}")
                        _flush_glossary_batch(force=False)
                else:
                    stats["nochange"] += 1
            elif should_repair and guard_failed:
                stats["failed"] += 1
            else:
                stats["skipped"] += 1
                if SIMPLE_CACHE_ENABLED:
                    cache[cache_key] = source_hash

            quality_rows.append(
                {
                    "file": str(file_path),
                    "path": path_str,
                    "issues": issues,
                    "confidence": confidence,
                    "risk_flags": risk_flags,
                    "applied": final_applied,
                }
            )

        if file_changed > 0:
            with file_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"文件质量修复写回: {file_path.name} | 修复={file_changed}")

    _flush_glossary_batch(force=True)
    _simple_write_quality_report(quality_rows)

    if stats["aborted"]:
        print(f"⚠️ 质量修复提前终止: {stats['abort_reason']}")
    else:
        print("✅ 完成：质量审校修复结束")
    print(
        f"质量统计：文件={stats['files']} 条目={stats['items']} 审校={stats['reviewed']} 修复={stats['repaired']} "
        f"未变化={stats['nochange']} 跳过={stats['skipped']} 缓存跳过={stats['cached']} 失败={stats['failed']}"
    )

    _simple_write_log(
        f"QUALITY_DONE files={stats['files']} items={stats['items']} reviewed={stats['reviewed']} repaired={stats['repaired']} "
        f"nochange={stats['nochange']} skipped={stats['skipped']} cached={stats['cached']} failed={stats['failed']} "
        f"aborted={stats['aborted']} reason={stats['abort_reason']} report={SIMPLE_QUALITY_REPORT_PATH}"
    )

    return stats, glossary_buckets


def _simple_quality_full_pipeline(target_files, translator, limiter, glossary_data: dict, glossary_buckets, cache: dict):
    aborted = False
    abort_reason = ""

    if SIMPLE_PIPELINE_ENABLE_GLOSSARY_PHASE:
        for round_idx in range(1, SIMPLE_PIPELINE_GLOSSARY_ITERATIONS + 1):
            print(f"\n=== 阶段1 术语统一迭代 {round_idx}/{SIMPLE_PIPELINE_GLOSSARY_ITERATIONS} ===")
            pairs, changed, glossary_buckets, is_aborted, reason = _simple_extract_glossary_from_corpus(
                target_files,
                translator,
                glossary_data,
                glossary_buckets,
            )
            if is_aborted:
                aborted = True
                abort_reason = reason or "fatal glossary extraction error"
                break

            if SIMPLE_QUALITY_GLOSSARY_REVIEW_ENABLED:
                changed_by_review = _simple_quality_review_glossary(translator, glossary_data, target_files)
                if changed_by_review > 0:
                    glossary_buckets, _ = _simple_refresh_runtime_glossary(glossary_data)
                    changed += changed_by_review

            print(f"术语迭代结果：pairs={pairs} changed={changed}")
            if changed <= 0:
                print("术语迭代已收敛，提前结束术语阶段。")
                break

    if (not aborted) and SIMPLE_PIPELINE_ENABLE_REPAIR_PHASE:
        for round_idx in range(1, SIMPLE_PIPELINE_REPAIR_ITERATIONS + 1):
            print(f"\n=== 阶段2 质量修复迭代 {round_idx}/{SIMPLE_PIPELINE_REPAIR_ITERATIONS} ===")
            stats, glossary_buckets = _simple_quality_review_and_repair_files(
                target_files,
                translator,
                limiter,
                glossary_data,
                glossary_buckets,
                cache,
            )
            if stats.get("aborted"):
                aborted = True
                abort_reason = stats.get("abort_reason") or "fatal quality repair error"
                break
            if stats.get("repaired", 0) <= 0:
                print("质量修复阶段已收敛，提前结束修复阶段。")
                break

    if aborted:
        print(f"⚠️ 复合流水线提前终止: {abort_reason}")
        _simple_write_log(f"PIPELINE_ABORT reason={abort_reason}")
    else:
        print("✅ 完成：复合质量流水线结束")
        _simple_write_log("PIPELINE_DONE")

    return not aborted


def main():
    glossary_extract_enabled = SIMPLE_ADAPTIVE_GLOSSARY_ENABLED or SIMPLE_RUN_MODE == "extract_glossary_only"
    if glossary_extract_enabled:
        _simple_ensure_glossary_file(SIMPLE_ADAPTIVE_GLOSSARY_OUTPUT_PATH)

    target_files = []
    if SIMPLE_RUN_MODE in {
        "translate",
        "extract_glossary_only",
        "optimize_existing",
        "quality_review_and_repair",
        "quality_full_pipeline",
    }:
        target_files = _simple_collect_target_files()
        if not target_files:
            return

    if SIMPLE_RUN_MODE == "quality_full_pipeline" and SIMPLE_PIPELINE_TARGET_CN_DIRS:
        target_files = _simple_collect_target_files_from_dirs(
            SIMPLE_PIPELINE_TARGET_CN_DIRS,
            SIMPLE_PIPELINE_TARGET_FILE_GLOB,
            SIMPLE_PIPELINE_TARGET_RECURSIVE,
        )
        if not target_files:
            print("❌ 复合流水线未找到目标文件，请检查 pipeline_target_cn_dirs")
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

    print(f"运行模式: {SIMPLE_RUN_MODE}")

    if SIMPLE_ADAPTIVE_GLOSSARY_ENABLED:
        print(
            f"术语增量学习: 已启用 模式={SIMPLE_ADAPTIVE_GLOSSARY_MODE} "
            f"每批最多={SIMPLE_ADAPTIVE_GLOSSARY_MAX_TERMS_PER_BATCH} 输出={SIMPLE_ADAPTIVE_GLOSSARY_OUTPUT_PATH}"
        )
        if SIMPLE_ADAPTIVE_GLOSSARY_DURING_TRANSLATION_ENABLED:
            if SIMPLE_ADAPTIVE_GLOSSARY_MODE == "entry_batch":
                print(f"entry_batch 提取粒度: 每 {SIMPLE_ADAPTIVE_GLOSSARY_ENTRY_BATCH_SIZE} 条 entry 提取一次")
        else:
            print("翻译阶段提词: 已关闭（可改 run_mode=extract_glossary_only 单独提炼）")
        if SIMPLE_EXTRA_GLOSSARY_PATH is None:
            print(f"增量术语将作为额外术语表参与合并: {SIMPLE_EFFECTIVE_EXTRA_GLOSSARY_PATH}")
        if (
            SIMPLE_ADAPTIVE_GLOSSARY_DURING_TRANSLATION_ENABLED
            and SIMPLE_ADAPTIVE_GLOSSARY_MODE in {"entry", "entry_batch"}
            and SIMPLE_MAX_WORKERS > 1
        ):
            if SIMPLE_ADAPTIVE_GLOSSARY_ENTRY_PARALLEL_ENABLED:
                if SIMPLE_ADAPTIVE_GLOSSARY_ENTRY_NON_BLOCKING_ENABLED:
                    print("entry/entry_batch 并行提词流水线: 已启用（翻译不等待每批提词）")
                else:
                    print("entry/entry_batch 并行模式: 已启用（更快；新术语可能不会立刻命中当前批次）")
            else:
                print("⚠️ entry/entry_batch 模式将按串行处理词条，以确保新术语能用于后续词条。")

    if SIMPLE_RUN_MODE == "extract_glossary_only":
        max_pairs_text = (
            str(SIMPLE_EXTRACT_ONLY_MAX_PAIRS_PER_FILE)
            if SIMPLE_EXTRACT_ONLY_MAX_PAIRS_PER_FILE > 0
            else "不限制"
        )
        print(
            f"仅提炼模式: 批次大小={SIMPLE_EXTRACT_ONLY_PAIR_BATCH_SIZE} "
            f"每文件最多双语条目={max_pairs_text} "
            f"并行={SIMPLE_EXTRACT_ONLY_PARALLEL_ENABLED} workers={SIMPLE_EXTRACT_ONLY_MAX_WORKERS} "
            f"rpm={SIMPLE_EXTRACT_ONLY_TARGET_RPM}"
        )
        if SIMPLE_EXTRACT_ONLY_PROGRESS_CACHE_ENABLED:
            print(f"仅提炼进度缓存: {SIMPLE_EXTRACT_ONLY_PROGRESS_CACHE_PATH}")
    elif SIMPLE_RUN_MODE == "repair_from_report":
        print(
            f"上下文修复模式: report={SIMPLE_REPAIR_REPORT_PATH} context={SIMPLE_REPAIR_CONTEXT_ENABLED} "
            f"context_max_chars={SIMPLE_REPAIR_CONTEXT_MAX_CHARS} force_cn_only={SIMPLE_REPAIR_FORCE_CN_ONLY}"
        )
        print(
            f"修复增强: include_en_preview={SIMPLE_REPAIR_INCLUDE_EN_PREVIEW} "
            f"glossary_expand={SIMPLE_REPAIR_GLOSSARY_EXPAND_ENABLED} "
            f"batch_size={SIMPLE_REPAIR_ENTRY_BATCH_SIZE} terms_per_batch={SIMPLE_REPAIR_GLOSSARY_MAX_TERMS_PER_BATCH}"
        )
    elif SIMPLE_RUN_MODE == "optimize_existing":
        print(
            f"全量优化模式: batch_enabled={SIMPLE_OPTIMIZE_BATCH_ENABLED} batch_size={SIMPLE_OPTIMIZE_BATCH_SIZE} "
            f"model_enabled={SIMPLE_OPTIMIZE_MODEL_ENABLED} glossary_patch_only={SIMPLE_OPTIMIZE_GLOSSARY_PATCH_ONLY}"
        )
        print(
            f"优化筛选: max_items_per_file={SIMPLE_OPTIMIZE_MAX_ITEMS_PER_FILE or '不限制'} "
            f"include_keys_only={sorted(SIMPLE_OPTIMIZE_INCLUDE_KEYS_ONLY) or ['*']}"
        )
    elif SIMPLE_RUN_MODE == "quality_review_and_repair":
        print(
            f"质量审校模式: report={SIMPLE_QUALITY_REPORT_PATH} context_scope={SIMPLE_QUALITY_CONTEXT_SCOPE} "
            f"second_pass={SIMPLE_QUALITY_SECOND_PASS_ENABLED} strict_semantic_guard={SIMPLE_QUALITY_STRICT_SEMANTIC_GUARD}"
        )
        print(
            f"质量门禁: numeric={SIMPLE_QUALITY_REQUIRE_NUMERIC_CONSISTENCY} markup={SIMPLE_QUALITY_FAIL_ON_MARKUP_RISK} "
            f"max_items_per_file={SIMPLE_QUALITY_MAX_ITEMS_PER_FILE or '不限制'} "
            f"include_keys_only={sorted(SIMPLE_QUALITY_INCLUDE_KEYS_ONLY) or ['*']}"
        )
        print(
            f"术语审校: context={SIMPLE_QUALITY_GLOSSARY_REVIEW_WITH_CONTEXT_ENABLED} "
            f"parallel={SIMPLE_QUALITY_GLOSSARY_REVIEW_PARALLEL_ENABLED} workers={SIMPLE_QUALITY_GLOSSARY_REVIEW_WORKERS} "
            f"examples_per_term={SIMPLE_QUALITY_GLOSSARY_REVIEW_MAX_EXAMPLES_PER_TERM}"
        )
    elif SIMPLE_RUN_MODE == "quality_full_pipeline":
        print(
            f"复合质量流水线: glossary_phase={SIMPLE_PIPELINE_ENABLE_GLOSSARY_PHASE} "
            f"glossary_iterations={SIMPLE_PIPELINE_GLOSSARY_ITERATIONS} "
            f"repair_phase={SIMPLE_PIPELINE_ENABLE_REPAIR_PHASE} repair_iterations={SIMPLE_PIPELINE_REPAIR_ITERATIONS}"
        )
        print(
            f"pipeline dirs={SIMPLE_PIPELINE_TARGET_CN_DIRS} glob={SIMPLE_PIPELINE_TARGET_FILE_GLOB} "
            f"recursive={SIMPLE_PIPELINE_TARGET_RECURSIVE}"
        )
        print(
            f"pipeline glossary filter: keys={sorted(SIMPLE_PIPELINE_GLOSSARY_INCLUDE_KEYS_ONLY) or ['*']} "
            f"max_chars_per_side={SIMPLE_PIPELINE_GLOSSARY_MAX_CHARS_PER_SIDE or '不限制'} "
            f"dedup={SIMPLE_PIPELINE_GLOSSARY_DEDUP_ENABLED}"
        )

    if SIMPLE_RUN_MODE in {
        "translate",
        "extract_glossary_only",
        "optimize_existing",
        "quality_review_and_repair",
        "quality_full_pipeline",
    }:
        print(f"目标文件数: {len(target_files)}")
        if SIMPLE_TARGET_FOLDER_PATH:
            mode = "递归" if SIMPLE_TARGET_FOLDER_RECURSIVE else "当前目录"
            print(f"批量模式: 文件夹={SIMPLE_TARGET_FOLDER_PATH} 匹配={SIMPLE_TARGET_FILE_GLOB} 范围={mode}")

    print(
        f"系统lang模式: {SIMPLE_SYSTEM_LANG_MODE} "
        f"(lang保留原文={SIMPLE_SYSTEM_LANG_KEEP_ORIGINAL})"
    )
    print(
        f"lang识别: markers={SIMPLE_LANG_FILE_PATH_MARKERS} names={sorted(SIMPLE_LANG_FILE_NAMES)}"
    )
    print(
        f"路径筛选: include={SIMPLE_TRANSLATE_PATH_INCLUDE_SUBSTRINGS or ['*']} "
        f"skip={SIMPLE_TRANSLATE_PATH_SKIP_SUBSTRINGS}"
    )
    print(
        f"按键输出覆盖: keep_original_keys={sorted(SIMPLE_FORCE_KEEP_ORIGINAL_KEYS)} "
        f"cn_only_keys={sorted(SIMPLE_FORCE_CN_ONLY_KEYS)}"
    )
    if SIMPLE_SYNC_TRANSLATION_ENABLED:
        print(
            f"键映射复用: enabled source={SIMPLE_SYNC_TRANSLATION_SOURCE_KEY} "
            f"target={SIMPLE_SYNC_TRANSLATION_TARGET_KEY} only_reuse={SIMPLE_SYNC_TRANSLATION_ONLY_REUSE}"
        )
    if SIMPLE_STRICT_COVERAGE_ENABLED:
        print(
            f"严格覆盖补翻: {SIMPLE_STRICT_COVERAGE_ENABLED} "
            f"(max_rounds={SIMPLE_STRICT_COVERAGE_MAX_ROUNDS}, "
            f"relaxed_allow_keys={SIMPLE_STRICT_COVERAGE_RELAXED_ALLOW_KEYS}, "
            f"max_items_per_round={SIMPLE_STRICT_COVERAGE_MAX_ITEMS_PER_ROUND or '不限制'})"
        )

    _simple_print_risk_warnings()

    translator = _SimpleTranslator()

    if SIMPLE_RUN_MODE == "extract_glossary_only":
        _simple_extract_glossary_only(target_files, translator, glossary_data, glossary_buckets)
        return

    if SIMPLE_RUN_MODE == "repair_from_report":
        cache = _simple_load_cache(SIMPLE_CACHE_PATH)
        limiter = _SimpleRateLimiter(SIMPLE_TARGET_RPM)
        _simple_repair_from_report(
            translator,
            limiter,
            glossary_data,
            glossary_buckets,
            cache,
        )
        _simple_save_cache(SIMPLE_CACHE_PATH, cache)
        return

    if SIMPLE_RUN_MODE == "optimize_existing":
        cache = _simple_load_cache(SIMPLE_CACHE_PATH)
        limiter = _SimpleRateLimiter(SIMPLE_TARGET_RPM)
        _simple_optimize_existing_files(
            target_files,
            translator,
            limiter,
            glossary_data,
            glossary_buckets,
            cache,
        )
        _simple_save_cache(SIMPLE_CACHE_PATH, cache)
        return

    if SIMPLE_RUN_MODE == "quality_review_and_repair":
        cache = _simple_load_cache(SIMPLE_CACHE_PATH)
        limiter = _SimpleRateLimiter(SIMPLE_TARGET_RPM)
        _simple_quality_review_and_repair_files(
            target_files,
            translator,
            limiter,
            glossary_data,
            glossary_buckets,
            cache,
        )
        _simple_save_cache(SIMPLE_CACHE_PATH, cache)
        return

    if SIMPLE_RUN_MODE == "quality_full_pipeline":
        cache = _simple_load_cache(SIMPLE_CACHE_PATH)
        limiter = _SimpleRateLimiter(SIMPLE_TARGET_RPM)
        _simple_quality_full_pipeline(
            target_files,
            translator,
            limiter,
            glossary_data,
            glossary_buckets,
            cache,
        )
        _simple_save_cache(SIMPLE_CACHE_PATH, cache)
        return

    cache = _simple_load_cache(SIMPLE_CACHE_PATH)

    limiter = _SimpleRateLimiter(SIMPLE_TARGET_RPM)

    total_stats = {
        "translated": 0,
        "nochange": 0,
        "skipped": 0,
        "cached": 0,
        "failed": 0,
        "candidates": 0,
        "file_errors": 0,
        "strict_translated": 0,
        "strict_remaining": 0,
    }
    aborted_run = False
    processed_files = 0
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
        processed_files += 1
        total_stats["translated"] += file_stats["translated"]
        total_stats["nochange"] += file_stats.get("nochange", 0)
        total_stats["skipped"] += file_stats["skipped"]
        total_stats["cached"] += file_stats["cached"]
        total_stats["failed"] += file_stats["failed"]
        total_stats["candidates"] += file_stats["candidates"]
        total_stats["strict_translated"] += file_stats.get("strict_translated", 0)
        total_stats["strict_remaining"] += file_stats.get("strict_remaining", 0)
        if file_stats["error"]:
            total_stats["file_errors"] += 1
        all_slowest.extend(file_slowest)

        print(
            f"文件统计：候选={file_stats['candidates']} 翻译={file_stats['translated']} "
            f"未变化={file_stats.get('nochange', 0)} 缓存跳过={file_stats['cached']} "
            f"跳过={file_stats['skipped']} 失败={file_stats['failed']} "
            f"严格补翻={file_stats.get('strict_translated', 0)} 剩余未翻={file_stats.get('strict_remaining', 0)}"
        )
        if file_stats.get("aborted"):
            aborted_run = True
            reason = file_stats.get("abort_reason") or "fatal API error"
            print(f"⛔ 运行提前停止：{reason}")
            _simple_write_log(f"ABORT run stopped at file={file_path} | {reason}")
            break

    _simple_save_cache(SIMPLE_CACHE_PATH, cache)

    all_slowest.sort(key=lambda x: x[0], reverse=True)
    top_n = all_slowest[:max(SIMPLE_TOP_N_SLOWEST, 0)] if SIMPLE_TOP_N_SLOWEST else []

    if aborted_run:
        print("⚠️ 批量处理提前终止（致命API错误）")
    else:
        print("✅ 完成：批量处理结束")
    print(
        f"统计：文件={processed_files}/{len(target_files)} 文件错误={total_stats['file_errors']} 候选={total_stats['candidates']} "
        f"翻译={total_stats['translated']} 未变化={total_stats['nochange']} 跳过={total_stats['skipped']} "
        f"缓存跳过={total_stats['cached']} 失败={total_stats['failed']} "
        f"严格补翻={total_stats['strict_translated']} 剩余未翻={total_stats['strict_remaining']}"
    )
    if top_n:
        print("最慢条目：")
        for sec, path_str in top_n:
            print(f"  {sec:.2f}s | {path_str}")
    _simple_write_log(
        f"DONE files={processed_files}/{len(target_files)} aborted={aborted_run} file_errors={total_stats['file_errors']} "
        f"translated={total_stats['translated']} nochange={total_stats['nochange']} skipped={total_stats['skipped']} "
        f"cached={total_stats['cached']} failed={total_stats['failed']} "
        f"strict_translated={total_stats['strict_translated']} strict_remaining={total_stats['strict_remaining']}"
    )


def print_help() -> None:
    print("Crucible translator")
    print("Usage:")
    print("  python 翻译工具/crucible_translator.py")
    print("")
    print("This script is config-driven and has no CLI options yet.")
    print(f"Config file: {SIMPLE_CONFIG_PATH}")
    print("Run mode options:")
    print("  run_mode: translate | extract_glossary_only | repair_from_report | optimize_existing | quality_review_and_repair | quality_full_pipeline")
    print("  adaptive_glossary_during_translation_enabled: true/false")
    print("Repair mode options:")
    print("  repair_report_path: untranslated-english-report.json")
    print("  repair_context_enabled: true/false")
    print("  repair_context_max_chars: 12000")
    print("  repair_force_cn_only: true (default: overwrite with repaired CN text only)")
    print("  repair_max_retries_per_item: 2")
    print("  repair_entry_batch_size: 12")
    print("  repair_include_en_preview: true")
    print("  repair_glossary_expand_enabled: true")
    print("  repair_glossary_extract_model: gpt-5.4")
    print("  repair_glossary_max_terms_per_batch: 12")
    print("Optimize mode options:")
    print("  optimize_batch_enabled: true/false")
    print("  optimize_batch_size: 8")
    print("  optimize_model_enabled: true/false")
    print("  optimize_glossary_patch_only: true/false")
    print("  optimize_max_items_per_file: 0 (0 means unlimited)")
    print("  optimize_include_keys_only: []")
    print("Quality mode options:")
    print("  quality_report_path: quality_review_report.json")
    print("  quality_second_pass_enabled: true/false")
    print("  quality_strict_semantic_guard: true/false")
    print("  quality_require_numeric_consistency: true/false")
    print("  quality_fail_on_markup_risk: true/false")
    print("  quality_context_scope: entry | file")
    print("  quality_max_items_per_file: 0 (0 means unlimited)")
    print("  quality_include_keys_only: ['name', 'description', 'public', 'private', 'condition']")
    print("  quality_use_bilingual_backup_source: true/false")
    print("  quality_glossary_review_enabled: true/false")
    print("  quality_glossary_review_apply: true/false")
    print("  quality_glossary_review_max_terms_per_run: 0 (0 means unlimited)")
    print("  quality_glossary_review_report_path: quality_glossary_review_report.json")
    print("  quality_glossary_review_with_context_enabled: true/false")
    print("  quality_glossary_review_parallel_enabled: true/false")
    print("  quality_glossary_review_workers: 128")
    print("  quality_glossary_review_max_examples_per_term: 6")
    print("  quality_glossary_review_context_max_chars: 1800")
    print("  quality_glossary_review_context_scan_keys: ['name', 'description', 'public', ...]")
    print("Full pipeline options:")
    print("  pipeline_target_cn_dirs: ['.../crucible-cn/compendium/cn', '.../ember_cn_unofficial/compendium/cn']")
    print("  pipeline_target_file_glob: *.json")
    print("  pipeline_target_recursive: false")
    print("  pipeline_glossary_iterations: 2")
    print("  pipeline_repair_iterations: 1")
    print("  pipeline_enable_glossary_phase: true")
    print("  pipeline_enable_repair_phase: true")
    print("  pipeline_glossary_include_keys_only: ['name', 'label', 'title', ...]")
    print("  pipeline_glossary_max_chars_per_side: 260 (0 means unlimited)")
    print("  pipeline_glossary_dedup_enabled: true/false")
    print("Adaptive glossary options:")
    print("  adaptive_glossary_mode: file | entry | entry_batch (entry_patch alias is accepted)")
    print("  adaptive_glossary_entry_parallel_enabled: false (set true for rough-pass extraction)")
    print("  adaptive_glossary_entry_non_blocking_enabled: true (do not wait glossary extraction during translation)")
    print("  adaptive_glossary_extract_model: gpt-5.4")
    print("  adaptive_glossary_extract_max_pending_batches: 64 (backpressure for async extraction)")
    print("  extract_only_pair_batch_size: 120 (pairs sent per extraction request)")
    print("  extract_only_max_pairs_per_file: 1200 (0 means unlimited scan per file)")
    print("  extract_only_parallel_enabled: true/false")
    print("  extract_only_max_workers: 16")
    print("  extract_only_target_rpm: 1200")
    print("  extract_only_progress_cache_enabled: true/false")
    print("  extract_only_progress_cache_path: crucible_extract_only_progress.json")
    print("Key config for system lang files:")
    print("  system_lang_mode: auto | on | off")
    print("  system_lang_keep_original: false (default, output Chinese only in lang mode)")
    print("  lang_file_path_markers: ['/lang/', '/i18n/']")
    print("  lang_file_names: ['en.json', 'cn.json', 'zh_hans.json', ...]")
    print("  allow_keys_only: [] (hard whitelist, non-empty means translate only these keys)")
    print("  allow_keys_extra: [] (append more key names to capture)")
    print("  force_keep_original_keys: [] (force bilingual output for these key names)")
    print("  force_cn_only_keys: [] (force Chinese-only output for these key names)")
    print("  sync_translation_enabled: false (reuse translation from source key to target key)")
    print("  sync_translation_source_key: actionname")
    print("  sync_translation_target_key: actioneffectname")
    print("  sync_translation_only_reuse: true (do not request API for missing target mapping)")
    print("Path filter options:")
    print("  translate_path_include_substrings: [] (empty means no include restriction)")
    print("  translate_path_skip_substrings: ['.mapping']")
    print("Strict coverage options:")
    print("  strict_coverage_enabled: true/false")
    print("  strict_coverage_max_rounds: 2")
    print("  strict_coverage_relaxed_allow_keys: true (bypass allow-key filter in fallback pass)")
    print("  strict_coverage_max_items_per_round: 0 (0 means unlimited)")
    print("Edit config first, then run without extra arguments.")

if __name__ == "__main__":
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print_help()
        raise SystemExit(0)
    main()