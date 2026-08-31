#!/usr/bin/env python3
"""Batch-calibrate existing Crucible translations against glossary + EN source.

Groups related fields (same actor / journal page / item) into batches,
sends each batch to an LLM with glossary context, and applies corrections.

Usage:
    # Preview mode (no file writes)
    python 翻译工具/crucible_calibrator.py

    # Apply corrections in-place
    python 翻译工具/crucible_calibrator.py --apply

    # Custom config
    python 翻译工具/crucible_calibrator.py --config my_config.json --apply
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_PATH = Path("crucible_calibrate_config.json")

DEFAULT_CONFIG = {
    "openai_api_key": "",
    "openai_base_url": "https://api.openai.com/v1",
    "model": "gpt-4.1",
    "fallback_model": "",
    "max_retries": 2,
    "request_timeout_seconds": 90,
    "long_text_timeout_seconds": 180,
    "long_text_char_threshold": 8000,
    "target_rpm": 500,
    "max_workers": 8,
    "glossary_path": "glossary_crucible_merged.json",
    "files": [
        # Mixed files (bilingual format): read AND written (if --apply)
    ],
    "file_pairs": [
        # Optional: [en_path, cn_path] pairs for separate EN/CN files
    ],
    "batch_max_chars": 8000,
    "batch_max_fields": 40,
    "group_depth": 3,
    "skip_untranslated": True,
    "skip_markup_only": True,
    "skip_keys": ["command", "formula", "img", "src", "flags"],
    "include_keys_only": [],
    "report_path": "calibration_report.json",
    "dry_run_sample_batches": 3,
    "extra_glossary_path": "",
    "system_name": "Crucible",
    "system_prompt_override": "",
    "use_responses_api": False,
}


def _load_config(path: Path) -> dict:
    if path.exists():
        with open(path, encoding="utf-8-sig") as f:
            user = json.load(f)
        merged = dict(DEFAULT_CONFIG)
        merged.update(user)
    else:
        merged = dict(DEFAULT_CONFIG)
    # Fallback to env var if api_key is empty
    if not merged.get("openai_api_key"):
        merged["openai_api_key"] = os.getenv("OPENAI_API_KEY", "")
    return merged


# ---------------------------------------------------------------------------
# Glossary
# ---------------------------------------------------------------------------

_GLOSSARY_SEP_RE = re.compile(r"(?:[\s\-_.,:;!?/\\()\[\]{}''\"`]+)?")
_GLOSSARY_LETTERS_RE = re.compile(r"[A-Za-z]")
_GLOSSARY_WORD_RE = re.compile(r"[A-Za-z]+")


def _load_glossary(path: Path) -> dict[str, str | list[str]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _build_glossary_index(glossary: dict) -> list[tuple[re.Pattern, str, str | list[str]]]:
    """Build sorted (longest-first) regex index for glossary matching."""
    entries: list[tuple[int, re.Pattern, str, str | list[str]]] = []

    for en_term, zh_val in glossary.items():
        letters = _GLOSSARY_LETTERS_RE.findall(en_term)
        if not letters:
            continue
        letter_count = len(letters)

        # Build flexible regex: allow optional separators between words
        words = _GLOSSARY_WORD_RE.findall(en_term)
        if not words:
            continue

        # Handle possessives and plurals
        word_patterns = []
        for w in words:
            if len(w) >= 3:
                word_patterns.append(f"{re.escape(w)}(?:'?s)?")
            else:
                word_patterns.append(re.escape(w))

        sep = _GLOSSARY_SEP_RE.pattern
        inner = sep.join(word_patterns)
        pattern = re.compile(
            rf"(?<![A-Za-z0-9]){inner}(?![A-Za-z0-9])",
            re.IGNORECASE,
        )
        entries.append((letter_count, pattern, en_term, zh_val))

    # Sort by length descending (longer matches first)
    entries.sort(key=lambda x: -x[0])
    return [(pat, en, zh) for _, pat, en, zh in entries]


def _match_glossary(text: str, index: list) -> list[tuple[str, str]]:
    """Return list of (en_term, zh_hint) that match in text."""
    matched: list[tuple[str, str]] = []
    seen_lower: set[str] = set()

    for pattern, en_term, zh_val in index:
        lk = en_term.lower()
        if lk in seen_lower:
            continue
        if pattern.search(text):
            seen_lower.add(lk)
            if isinstance(zh_val, list):
                hint = " | ".join(zh_val)
            else:
                hint = zh_val
            matched.append((en_term, hint))

    return matched


def _format_glossary_hint(terms: list[tuple[str, str]]) -> str:
    if not terms:
        return ""
    lines = [f"  {en} → {zh}" for en, zh in terms]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON traversal
# ---------------------------------------------------------------------------

PathParts = list[str | int]

_CODE_LIKE_RE = re.compile(
    r"@(?:UUID|Check|Damage|Template|Localize|RollTable|Embed|Actor|Item)\["
    r"|@(?:abilities|attributes|skills?|saves?|item|actor)\."
    r"|\[\[/"
    r"|Compendium\."
)
_HAS_CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
_HAS_ENGLISH_RE = re.compile(r"[A-Za-z]{2,}")
_MARKUP_ONLY_RE = re.compile(
    r"^[\s<>/\[\]@{}()|&;#=\"\'\-_.,:!?0-9\\%+*^~`\n\r\t]*$"
)


def _extract_bilingual(value: str) -> tuple[str, str] | None:
    """Extract (cn_part, en_part) from bilingual string.

    Handles two formats:
    - Inline: "中文名 English Name" (single line, CN first then EN)
    - Multiline: "中文\\nEnglish" (CN on first lines, EN on last lines)

    Returns None if not bilingual.
    """
    if not isinstance(value, str) or not value.strip():
        return None

    text = value.strip()

    # Try multiline split first: CN lines then EN lines
    if "\n" in text:
        lines = text.splitlines()
        for split_idx in range(1, len(lines)):
            zh = "\n".join(lines[:split_idx]).strip()
            en = "\n".join(lines[split_idx:]).strip()
            if not zh or not en:
                continue
            if not _HAS_CHINESE_RE.search(zh):
                continue
            if not _HAS_ENGLISH_RE.search(en):
                continue
            if _HAS_CHINESE_RE.search(en):
                continue
            return zh, en

    # Single-line bilingual: find the boundary between CN and EN
    # Pattern: CN chars end, then space(s), then EN chars start
    # e.g., "阿克图斯高原 The Arctus Plateau"
    if not _HAS_CHINESE_RE.search(text) or not _HAS_ENGLISH_RE.search(text):
        return None

    # Skip if it's HTML content with embedded EN tags/attributes
    if "<" in text and ">" in text:
        return None

    # Find the last Chinese char, then split after it
    last_cn_idx = -1
    for i, c in enumerate(text):
        if "\u4e00" <= c <= "\u9fff" or "\u3400" <= c <= "\u4dbf":
            last_cn_idx = i

    if last_cn_idx < 0 or last_cn_idx >= len(text) - 2:
        return None

    # The split point should be a space after the CN part
    rest = text[last_cn_idx + 1:]
    space_idx = 0
    while space_idx < len(rest) and rest[space_idx] in " \t":
        space_idx += 1

    if space_idx == 0:
        return None  # No space separator

    cn_part = text[: last_cn_idx + 1].strip()
    en_part = rest[space_idx:].strip()

    if not cn_part or not en_part:
        return None
    if not _HAS_ENGLISH_RE.search(en_part):
        return None

    return cn_part, en_part


def _path_to_en_context(parts: PathParts) -> str:
    """Extract English context from path keys (actor/item/scene names)."""
    context_parts = []
    for p in parts:
        if isinstance(p, str) and _HAS_ENGLISH_RE.search(p):
            context_parts.append(p)
    return " > ".join(context_parts) if context_parts else ""


def _iter_string_leaves(
    node: Any, parts: PathParts | None = None
) -> list[tuple[PathParts, str]]:
    if parts is None:
        parts = []
    results: list[tuple[PathParts, str]] = []
    if isinstance(node, dict):
        for key, val in node.items():
            results.extend(_iter_string_leaves(val, parts + [key]))
    elif isinstance(node, list):
        for idx, val in enumerate(node):
            results.extend(_iter_string_leaves(val, parts + [idx]))
    elif isinstance(node, str):
        results.append((list(parts), node))
    return results


def _get_at_path(node: Any, parts: PathParts) -> Any:
    cur = node
    for p in parts:
        try:
            cur = cur[p]
        except (KeyError, IndexError, TypeError):
            return None
    return cur


def _set_at_path(node: Any, parts: PathParts, value: Any) -> bool:
    cur = node
    for p in parts[:-1]:
        try:
            cur = cur[p]
        except (KeyError, IndexError, TypeError):
            return False
    try:
        cur[parts[-1]] = value
        return True
    except (KeyError, IndexError, TypeError):
        return False


def _path_to_str(parts: PathParts) -> str:
    out = "root"
    for p in parts:
        if isinstance(p, int):
            out += f"[{p}]"
        else:
            out += f".{p}"
    return out


# ---------------------------------------------------------------------------
# Field collection & filtering
# ---------------------------------------------------------------------------

@dataclass
class FieldPair:
    parts: PathParts
    path_str: str
    en_text: str          # EN source (may be empty for pure CN fields)
    cn_text: str           # CN translation to calibrate
    original_value: str    # Original value in mixed file (for restoring bilingual format)
    is_bilingual: bool     # Whether the original was bilingual format


@dataclass
class Batch:
    group_key: str
    fields: list[FieldPair]
    total_chars: int = 0


def _collect_field_pairs(
    en_data: Any,
    cn_data: Any,
    *,
    skip_untranslated: bool,
    skip_markup_only: bool,
    skip_keys: set[str],
    include_keys_only: set[str],
) -> list[FieldPair]:
    """Collect all EN/CN string pairs suitable for calibration (separate EN/CN files)."""
    pairs: list[FieldPair] = []
    for parts, en_text in _iter_string_leaves(en_data):
        # Skip by key name
        if parts:
            leaf_key = str(parts[-1]).lower()
            if leaf_key in skip_keys:
                continue
            if include_keys_only and leaf_key not in include_keys_only:
                continue

        cn_text = _get_at_path(cn_data, parts)
        if not isinstance(cn_text, str):
            continue

        # Skip untranslated (CN == EN)
        if skip_untranslated and cn_text == en_text:
            continue

        # Skip empty
        if not cn_text.strip() or not en_text.strip():
            continue

        # Skip markup-only strings
        if skip_markup_only and _MARKUP_ONLY_RE.match(en_text):
            continue

        pairs.append(FieldPair(
            parts=parts,
            path_str=_path_to_str(parts),
            en_text=en_text,
            cn_text=cn_text,
            original_value=cn_text,
            is_bilingual=False,
        ))

    return pairs


def _collect_fields_from_mixed(
    data: Any,
    *,
    skip_markup_only: bool,
    skip_keys: set[str],
    include_keys_only: set[str],
) -> list[FieldPair]:
    """Collect calibration fields from a single mixed (bilingual) file."""
    pairs: list[FieldPair] = []

    for parts, value in _iter_string_leaves(data):
        if not value.strip():
            continue

        # Skip by key name
        if parts:
            leaf_key = str(parts[-1]).lower()
            if leaf_key in skip_keys:
                continue
            if include_keys_only and leaf_key not in include_keys_only:
                continue

        # Must contain Chinese to be a translated field
        if not _HAS_CHINESE_RE.search(value):
            continue

        # Try to extract bilingual pair
        bilingual = _extract_bilingual(value)
        if bilingual:
            cn_text, en_text = bilingual
            is_bilingual = True
        else:
            # Pure CN field — use path keys as EN context
            cn_text = value
            en_text = _path_to_en_context(parts)
            is_bilingual = False

        # Skip markup-only
        if skip_markup_only and not _HAS_CHINESE_RE.search(cn_text):
            continue

        pairs.append(FieldPair(
            parts=parts,
            path_str=_path_to_str(parts),
            en_text=en_text,
            cn_text=cn_text,
            original_value=value,
            is_bilingual=is_bilingual,
        ))

    return pairs


def _group_key(parts: PathParts, depth: int) -> str:
    """Extract group key from path at given depth."""
    key_parts = []
    for i, p in enumerate(parts):
        if i >= depth:
            break
        key_parts.append(str(p))
    return ".".join(key_parts) if key_parts else "root"


def _build_batches(
    pairs: list[FieldPair],
    *,
    group_depth: int,
    max_chars: int,
    max_fields: int,
) -> list[Batch]:
    """Group field pairs into batches by semantic proximity."""
    # Step 1: group by key
    groups: dict[str, list[FieldPair]] = {}
    for fp in pairs:
        gk = _group_key(fp.parts, group_depth)
        groups.setdefault(gk, []).append(fp)

    # Step 2: build batches — small groups merge, large groups stay alone
    batches: list[Batch] = []
    pending_fields: list[FieldPair] = []
    pending_chars = 0
    pending_key_parts: list[str] = []

    def _flush():
        nonlocal pending_fields, pending_chars, pending_key_parts
        if pending_fields:
            gk = " + ".join(pending_key_parts) if pending_key_parts else "mixed"
            batches.append(Batch(
                group_key=gk,
                fields=list(pending_fields),
                total_chars=pending_chars,
            ))
        pending_fields = []
        pending_chars = 0
        pending_key_parts = []

    for gk in sorted(groups.keys()):
        group_fields = groups[gk]
        group_chars = sum(len(f.cn_text) + len(f.en_text) for f in group_fields)

        # If this group alone exceeds limits, make it its own batch
        if group_chars > max_chars or len(group_fields) > max_fields:
            _flush()
            # Split large groups into sub-batches
            sub_fields: list[FieldPair] = []
            sub_chars = 0
            for fp in group_fields:
                fc = len(fp.en_text) + len(fp.cn_text)
                if sub_fields and (sub_chars + fc > max_chars or len(sub_fields) >= max_fields):
                    batches.append(Batch(
                        group_key=gk,
                        fields=list(sub_fields),
                        total_chars=sub_chars,
                    ))
                    sub_fields = []
                    sub_chars = 0
                sub_fields.append(fp)
                sub_chars += fc
            if sub_fields:
                batches.append(Batch(
                    group_key=gk,
                    fields=list(sub_fields),
                    total_chars=sub_chars,
                ))
            continue

        # Try to merge with pending
        if pending_fields and (
            pending_chars + group_chars > max_chars
            or len(pending_fields) + len(group_fields) > max_fields
        ):
            _flush()

        pending_fields.extend(group_fields)
        pending_chars += group_chars
        pending_key_parts.append(gk)

    _flush()
    return batches


# ---------------------------------------------------------------------------
# API / LLM
# ---------------------------------------------------------------------------

_DEFAULT_SYSTEM_PROMPT = """\
You are a senior TRPG localization calibrator for {system_name} (a Foundry VTT game system).

Task: Review Chinese translations. Fix ONLY these issues:
1. **Glossary misalignment** — use the provided glossary terms when the source uses that term in the matching sense
2. **Inconsistency** — same English term should have the same Chinese translation within this batch
3. **Semantic errors** — meaning deviates from English source (when EN is available)
4. **Naturalness** — awkward Chinese phrasing that can be improved without changing meaning

Do NOT change:
- Markup tokens: @UUID[...], @Compendium[...], @Condition[...], @ref[...], [[...]], HTML tags — copy them EXACTLY
- Numbers, punctuation style, line breaks
- Correct translations that just use different valid wording

For fields marked [bilingual], return ONLY the corrected CN part (not the EN part).
For fields marked [cn-only], return the corrected CN text.
For polysemous glossary terms (A | B | C), choose by context. Do not force a glossary term where it doesn't fit.

Return a JSON object mapping field numbers (as strings) to corrected Chinese text.
Include ONLY fields that need changes. If nothing needs fixing, return {{}}.
Do NOT include unchanged fields.\
"""


def _get_system_prompt(config: dict) -> str:
    override = str(config.get("system_prompt_override", "")).strip()
    if override:
        return override
    name = config.get("system_name", "Crucible")
    return _DEFAULT_SYSTEM_PROMPT.format(system_name=name)


def _build_user_message(batch: Batch, glossary_terms: list[tuple[str, str]]) -> str:
    parts: list[str] = []

    if glossary_terms:
        parts.append("【术语表】")
        parts.append(_format_glossary_hint(glossary_terms))
        parts.append("")

    parts.append(f"【校准批次: {batch.group_key}】({len(batch.fields)} fields)")
    parts.append("")

    for idx, fp in enumerate(batch.fields, 1):
        if fp.is_bilingual:
            parts.append(f"#{idx} [bilingual] [{fp.path_str}]")
            parts.append(f"EN: {fp.en_text}")
            parts.append(f"CN: {fp.cn_text}")
        elif not fp.is_bilingual and fp.original_value == fp.cn_text:
            # cn-only field from mixed file — path gives EN context
            parts.append(f"#{idx} [cn-only] [{fp.path_str}]")
            parts.append(f"CN: {fp.cn_text}")
        elif fp.en_text:
            # Has actual EN source (from separate EN/CN file pair)
            parts.append(f"#{idx} [{fp.path_str}]")
            parts.append(f"EN: {fp.en_text}")
            parts.append(f"CN: {fp.cn_text}")
        else:
            parts.append(f"#{idx} [cn-only] [{fp.path_str}]")
            parts.append(f"CN: {fp.cn_text}")
        parts.append("")

    return "\n".join(parts)


def _parse_response(text: str, batch: Batch) -> dict[int, str]:
    """Parse LLM response into {field_index: corrected_cn}."""
    # Try direct JSON parse
    text = text.strip()
    # Remove markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return {int(k): str(v) for k, v in result.items() if str(k).isdigit()}
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: extract JSON object via regex
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, dict):
                return {int(k): str(v) for k, v in result.items() if str(k).isdigit()}
        except (json.JSONDecodeError, ValueError):
            pass

    return {}


# ---------------------------------------------------------------------------
# Safety guards
# ---------------------------------------------------------------------------

_UUID_RE = re.compile(r"@UUID\[[^\]]+\]")
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _is_markup_safe(original: str, revised: str) -> bool:
    """Check that markup tokens are preserved."""
    orig_uuids = set(_UUID_RE.findall(original))
    rev_uuids = set(_UUID_RE.findall(revised))
    if orig_uuids != rev_uuids:
        return False

    # Check HTML tags are preserved (allow reordering but not removal)
    orig_tags = sorted(_HTML_TAG_RE.findall(original))
    rev_tags = sorted(_HTML_TAG_RE.findall(revised))
    if orig_tags != rev_tags:
        return False

    return True


def _has_meaningful_change(original: str, revised: str) -> bool:
    """Check that the revision actually changed something meaningful."""
    if original == revised:
        return False
    # Ignore whitespace-only changes
    if original.split() == revised.split():
        return False
    return True


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class _RateLimiter:
    def __init__(self, rpm: int):
        self.rpm = max(rpm, 1)
        self.min_interval = 60.0 / self.rpm
        self._lock = Lock()
        self._last = 0.0

    def wait(self):
        with self._lock:
            now = time.time()
            elapsed = now - self._last
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self._last = time.time()


# ---------------------------------------------------------------------------
# Main calibration engine
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    batch_key: str
    changes: list[dict[str, str]]
    errors: list[str] = field(default_factory=list)


def _calibrate_batch(
    batch: Batch,
    glossary_index: list,
    config: dict,
    rate_limiter: _RateLimiter,
    stats: dict,
    stats_lock: Lock,
) -> CalibrationResult:
    """Calibrate one batch via API call."""
    from openai import OpenAI

    result = CalibrationResult(batch_key=batch.group_key, changes=[])

    # Collect glossary terms from all EN + CN texts in this batch
    combined_en = " ".join(f.en_text for f in batch.fields if f.en_text)
    # Also scan CN text for English words embedded in descriptions
    combined_cn_en = " ".join(
        f.cn_text for f in batch.fields if _HAS_ENGLISH_RE.search(f.cn_text)
    )
    glossary_terms = _match_glossary(
        combined_en + " " + combined_cn_en, glossary_index
    )

    user_msg = _build_user_message(batch, glossary_terms)

    # Compute timeout
    total_chars = batch.total_chars
    if total_chars > config["long_text_char_threshold"]:
        timeout = config["long_text_timeout_seconds"]
    else:
        timeout = config["request_timeout_seconds"]

    system_prompt = _get_system_prompt(config)
    use_responses = config.get("use_responses_api", False)

    client = OpenAI(
        api_key=config["openai_api_key"],
        base_url=config["openai_base_url"],
        timeout=timeout,
        max_retries=0,
    )

    model = config["model"]
    max_retries = config["max_retries"]

    def _call_api(model_id: str) -> str:
        if use_responses:
            resp = client.responses.create(
                model=model_id,
                instructions=system_prompt,
                input=user_msg,
                timeout=timeout,
            )
            return (resp.output_text or "").strip()
        else:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.3,
                timeout=timeout,
            )
            return resp.choices[0].message.content or ""

    for attempt in range(max_retries + 1):
        rate_limiter.wait()
        try:
            raw_text = _call_api(model)
            break
        except Exception as e:
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            # Try fallback model
            fallback = config.get("fallback_model", "")
            if fallback and fallback != model:
                try:
                    rate_limiter.wait()
                    raw_text = _call_api(fallback)
                    break
                except Exception:
                    pass
            result.errors.append(f"API error after {max_retries+1} attempts: {e}")
            with stats_lock:
                stats["api_errors"] += 1
            return result

    # Parse response
    corrections = _parse_response(raw_text, batch)

    with stats_lock:
        stats["api_calls"] += 1

    # Validate and collect changes
    for field_idx, corrected_cn in corrections.items():
        if field_idx < 1 or field_idx > len(batch.fields):
            continue
        fp = batch.fields[field_idx - 1]

        if not _has_meaningful_change(fp.cn_text, corrected_cn):
            continue

        if not _is_markup_safe(fp.cn_text, corrected_cn):
            with stats_lock:
                stats["markup_rejected"] += 1
            result.errors.append(
                f"Markup guard rejected #{field_idx} ({fp.path_str})"
            )
            continue

        # For bilingual fields, reconstruct "CN EN" format
        if fp.is_bilingual and fp.en_text:
            final_value = f"{corrected_cn} {fp.en_text}"
        else:
            final_value = corrected_cn

        result.changes.append({
            "path_str": fp.path_str,
            "parts": fp.parts,
            "en": fp.en_text,
            "old_cn": fp.cn_text,
            "new_cn": corrected_cn,
            "final_value": final_value,
        })
        with stats_lock:
            stats["fields_changed"] += 1

    return result


# ---------------------------------------------------------------------------
# File processing
# ---------------------------------------------------------------------------

def _process_file_pair(
    en_path: Path,
    cn_path: Path,
    glossary_index: list,
    config: dict,
    apply: bool,
) -> tuple[list[CalibrationResult], dict]:
    """Process one EN/CN file pair."""

    en_data = json.loads(en_path.read_text(encoding="utf-8-sig"))
    cn_data = json.loads(cn_path.read_text(encoding="utf-8-sig"))

    skip_keys = {k.lower() for k in config.get("skip_keys", [])}
    include_keys = {k.lower() for k in config.get("include_keys_only", [])}

    pairs = _collect_field_pairs(
        en_data, cn_data,
        skip_untranslated=config["skip_untranslated"],
        skip_markup_only=config["skip_markup_only"],
        skip_keys=skip_keys,
        include_keys_only=include_keys,
    )

    batches = _build_batches(
        pairs,
        group_depth=config["group_depth"],
        max_chars=config["batch_max_chars"],
        max_fields=config["batch_max_fields"],
    )

    print(f"  Fields: {len(pairs)}, Batches: {len(batches)}")

    stats = {
        "api_calls": 0,
        "api_errors": 0,
        "fields_changed": 0,
        "markup_rejected": 0,
    }
    stats_lock = Lock()
    rate_limiter = _RateLimiter(config["target_rpm"])
    all_results: list[CalibrationResult] = []

    max_workers = config["max_workers"]

    def _process_one(batch: Batch) -> CalibrationResult:
        return _calibrate_batch(
            batch, glossary_index, config, rate_limiter, stats, stats_lock,
        )

    first_error_printed = False

    if max_workers <= 1:
        for i, batch in enumerate(tqdm(batches, desc="  Calibrating", unit="batch")):
            res = _process_one(batch)
            all_results.append(res)
            if res.errors and not first_error_printed:
                tqdm.write(f"    First error: {res.errors[0]}")
                first_error_printed = True
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_process_one, batch): (i, batch)
                for i, batch in enumerate(batches)
            }
            pbar = tqdm(total=len(batches), desc="  Calibrating", unit="batch")
            for future in concurrent.futures.as_completed(futures):
                i, batch = futures[future]
                try:
                    res = future.result()
                    all_results.append(res)
                    if res.errors and not first_error_printed:
                        tqdm.write(f"    First error: {res.errors[0]}")
                        first_error_printed = True
                except Exception as e:
                    tqdm.write(f"    Batch {i+1} error: {e}")
                    traceback.print_exc()
                pbar.update(1)
            pbar.close()

    # Apply changes to cn_data
    total_applied = 0
    if apply:
        for res in all_results:
            for change in res.changes:
                value = change.get("final_value", change["new_cn"])
                ok = _set_at_path(cn_data, change["parts"], value)
                if ok:
                    total_applied += 1

        text = json.dumps(cn_data, ensure_ascii=False, indent=4)
        cn_path.write_text(text + "\n", encoding="utf-8")
        print(f"  Written: {cn_path} ({total_applied} fields updated)")
    else:
        total_applied = sum(len(r.changes) for r in all_results)
        print(f"  Preview: {total_applied} fields would change (add --apply to write)")

    stats["total_applied"] = total_applied
    return all_results, stats


def _process_mixed_file(
    file_path: Path,
    glossary_index: list,
    config: dict,
    apply: bool,
) -> tuple[list[CalibrationResult], dict]:
    """Process a single mixed (bilingual) file."""

    data = json.loads(file_path.read_text(encoding="utf-8-sig"))

    skip_keys = {k.lower() for k in config.get("skip_keys", [])}
    include_keys = {k.lower() for k in config.get("include_keys_only", [])}

    pairs = _collect_fields_from_mixed(
        data,
        skip_markup_only=config["skip_markup_only"],
        skip_keys=skip_keys,
        include_keys_only=include_keys,
    )

    batches = _build_batches(
        pairs,
        group_depth=config["group_depth"],
        max_chars=config["batch_max_chars"],
        max_fields=config["batch_max_fields"],
    )

    bilingual_count = sum(1 for p in pairs if p.is_bilingual)
    print(f"  Fields: {len(pairs)} (bilingual: {bilingual_count}, "
          f"cn-only: {len(pairs) - bilingual_count}), Batches: {len(batches)}")

    stats = {
        "api_calls": 0,
        "api_errors": 0,
        "fields_changed": 0,
        "markup_rejected": 0,
    }
    stats_lock = Lock()
    rate_limiter = _RateLimiter(config["target_rpm"])
    all_results: list[CalibrationResult] = []

    max_workers = config["max_workers"]

    def _process_one(batch: Batch) -> CalibrationResult:
        return _calibrate_batch(
            batch, glossary_index, config, rate_limiter, stats, stats_lock,
        )

    first_error_printed = False

    if max_workers <= 1:
        for i, batch in enumerate(tqdm(batches, desc="  Calibrating", unit="batch")):
            res = _process_one(batch)
            all_results.append(res)
            if res.errors and not first_error_printed:
                tqdm.write(f"    First error: {res.errors[0]}")
                first_error_printed = True
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_process_one, batch): (i, batch)
                for i, batch in enumerate(batches)
            }
            pbar = tqdm(total=len(batches), desc="  Calibrating", unit="batch")
            for future in concurrent.futures.as_completed(futures):
                i, batch = futures[future]
                try:
                    res = future.result()
                    all_results.append(res)
                    if res.errors and not first_error_printed:
                        tqdm.write(f"    First error: {res.errors[0]}")
                        first_error_printed = True
                except Exception as e:
                    tqdm.write(f"    Batch {i+1} error: {e}")
                    traceback.print_exc()
                pbar.update(1)
            pbar.close()

    # Apply changes to data
    total_applied = 0
    if apply:
        for res in all_results:
            for change in res.changes:
                value = change.get("final_value", change["new_cn"])
                ok = _set_at_path(data, change["parts"], value)
                if ok:
                    total_applied += 1

        text = json.dumps(data, ensure_ascii=False, indent=4)
        file_path.write_text(text + "\n", encoding="utf-8")
        print(f"  Written: {file_path} ({total_applied} fields updated)")
    else:
        total_applied = sum(len(r.changes) for r in all_results)
        print(f"  Preview: {total_applied} fields would change (add --apply to write)")

    stats["total_applied"] = total_applied
    return all_results, stats


# ---------------------------------------------------------------------------
# Dry run (preview batch construction without API)
# ---------------------------------------------------------------------------

def _dry_run_mixed(
    file_path: Path,
    glossary_index: list,
    config: dict,
    sample_count: int = 3,
):
    """Show sample batches from a mixed file without calling API."""
    data = json.loads(file_path.read_text(encoding="utf-8-sig"))

    skip_keys = {k.lower() for k in config.get("skip_keys", [])}
    include_keys = {k.lower() for k in config.get("include_keys_only", [])}

    pairs = _collect_fields_from_mixed(
        data,
        skip_markup_only=config["skip_markup_only"],
        skip_keys=skip_keys,
        include_keys_only=include_keys,
    )

    batches = _build_batches(
        pairs,
        group_depth=config["group_depth"],
        max_chars=config["batch_max_chars"],
        max_fields=config["batch_max_fields"],
    )

    bilingual_count = sum(1 for p in pairs if p.is_bilingual)

    print(f"\n{'='*60}")
    print(f"File: {file_path.name}")
    print(f"Total fields: {len(pairs)} (bilingual: {bilingual_count}, "
          f"cn-only: {len(pairs) - bilingual_count})")
    print(f"Total batches: {len(batches)}")

    _print_batch_stats(batches, glossary_index, sample_count)


def _dry_run_pair(
    en_path: Path,
    cn_path: Path,
    glossary_index: list,
    config: dict,
    sample_count: int = 3,
):
    """Show sample batches from separate EN/CN files without calling API."""
    en_data = json.loads(en_path.read_text(encoding="utf-8-sig"))
    cn_data = json.loads(cn_path.read_text(encoding="utf-8-sig"))

    skip_keys = {k.lower() for k in config.get("skip_keys", [])}
    include_keys = {k.lower() for k in config.get("include_keys_only", [])}

    pairs = _collect_field_pairs(
        en_data, cn_data,
        skip_untranslated=config["skip_untranslated"],
        skip_markup_only=config["skip_markup_only"],
        skip_keys=skip_keys,
        include_keys_only=include_keys,
    )

    batches = _build_batches(
        pairs,
        group_depth=config["group_depth"],
        max_chars=config["batch_max_chars"],
        max_fields=config["batch_max_fields"],
    )

    print(f"\n{'='*60}")
    print(f"File: {en_path.name} / {cn_path.name}")
    print(f"Total fields: {len(pairs)}")
    print(f"Total batches: {len(batches)}")

    _print_batch_stats(batches, glossary_index, sample_count)


def _print_batch_stats(
    batches: list[Batch],
    glossary_index: list,
    sample_count: int,
):
    """Print batch size stats and sample batch previews."""
    sizes = [b.total_chars for b in batches]
    fields_per = [len(b.fields) for b in batches]
    if sizes:
        sizes.sort()
        fields_per.sort()
        print(f"Chars/batch: min={sizes[0]}, median={sizes[len(sizes)//2]}, "
              f"max={sizes[-1]}")
        print(f"Fields/batch: min={fields_per[0]}, "
              f"median={fields_per[len(fields_per)//2]}, max={fields_per[-1]}")

    for i, batch in enumerate(batches[:sample_count]):
        combined_en = " ".join(f.en_text for f in batch.fields if f.en_text)
        combined_cn_en = " ".join(
            f.cn_text for f in batch.fields if _HAS_ENGLISH_RE.search(f.cn_text)
        )
        terms = _match_glossary(combined_en + " " + combined_cn_en, glossary_index)

        msg = _build_user_message(batch, terms)
        print(f"\n--- Sample Batch {i+1}: {batch.group_key} ---")
        print(f"Fields: {len(batch.fields)}, Chars: {batch.total_chars}, "
              f"Glossary terms: {len(terms)}")
        # Show first ~800 chars of the message
        preview = msg[:800]
        if len(msg) > 800:
            preview += f"\n... ({len(msg) - 800} more chars)"
        print(preview)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _write_report(
    report_path: Path,
    all_results: list[CalibrationResult],
    all_stats: list[dict],
    file_pairs: list[tuple[str, str]],
):
    report = {
        "file_pairs": [[str(a), str(b)] for a, b in file_pairs],
        "summary": {
            "total_api_calls": sum(s["api_calls"] for s in all_stats),
            "total_api_errors": sum(s["api_errors"] for s in all_stats),
            "total_fields_changed": sum(s["fields_changed"] for s in all_stats),
            "total_markup_rejected": sum(s["markup_rejected"] for s in all_stats),
        },
        "changes": [],
    }
    for res in all_results:
        for change in res.changes:
            report["changes"].append({
                "batch": res.batch_key,
                "path": change["path_str"],
                "en": change["en"][:200],
                "old_cn": change["old_cn"][:200],
                "new_cn": change["new_cn"][:200],
            })

    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nReport: {report_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch-calibrate Crucible translations against glossary + EN source."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH),
                        help="Config JSON path")
    parser.add_argument("--apply", action="store_true",
                        help="Write corrections to files")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run",
                        help="Show sample batches without calling API")
    parser.add_argument("--dry-run-samples", type=int, default=3,
                        help="Number of sample batches to show in dry run")
    args = parser.parse_args()

    config = _load_config(Path(args.config))

    mixed_files = config.get("files", [])
    file_pairs = config.get("file_pairs", [])

    if not mixed_files and not file_pairs:
        print("No files or file_pairs configured. Add them to your config file.")
        print(f"Config: {args.config}")
        sys.exit(1)

    # Load glossary
    glossary_path = Path(config["glossary_path"])
    glossary = _load_glossary(glossary_path)
    print(f"Glossary: {len(glossary)} terms from {glossary_path}")

    # Load extra glossary (merge)
    extra_path_raw = str(config.get("extra_glossary_path", "")).strip()
    if extra_path_raw:
        extra_path = Path(extra_path_raw)
        if extra_path.exists():
            extra = _load_glossary(extra_path)
            glossary.update(extra)
            print(f"Extra glossary: +{len(extra)} terms from {extra_path} (total {len(glossary)})")

    glossary_index = _build_glossary_index(glossary)

    # Resolve mixed files
    resolved_mixed: list[Path] = []
    for fp in mixed_files:
        p = Path(fp).resolve()
        if not p.is_file():
            print(f"File not found: {p}")
            continue
        resolved_mixed.append(p)

    # Resolve file pairs
    resolved_pairs: list[tuple[Path, Path]] = []
    for pair in file_pairs:
        en_p = Path(pair[0]).resolve()
        cn_p = Path(pair[1]).resolve()
        if not en_p.is_file():
            print(f"EN file not found: {en_p}")
            continue
        if not cn_p.is_file():
            print(f"CN file not found: {cn_p}")
            continue
        resolved_pairs.append((en_p, cn_p))

    if not resolved_mixed and not resolved_pairs:
        print("No valid files found.")
        sys.exit(1)

    if resolved_mixed:
        print(f"Mixed files: {len(resolved_mixed)}")
    if resolved_pairs:
        print(f"File pairs: {len(resolved_pairs)}")

    # Dry run mode
    if args.dry_run:
        for fp in resolved_mixed:
            _dry_run_mixed(fp, glossary_index, config,
                           sample_count=args.dry_run_samples)
        for en_p, cn_p in resolved_pairs:
            _dry_run_pair(en_p, cn_p, glossary_index, config,
                          sample_count=args.dry_run_samples)
        return

    # Calibration mode
    all_results: list[CalibrationResult] = []
    all_stats: list[dict] = []
    all_file_labels: list[tuple[str, str]] = []

    for fp in resolved_mixed:
        print(f"\n{'='*60}")
        print(f"Mixed: {fp}")
        results, stats = _process_mixed_file(
            fp, glossary_index, config, apply=args.apply,
        )
        all_results.extend(results)
        all_stats.append(stats)
        all_file_labels.append((str(fp), str(fp)))

    for en_p, cn_p in resolved_pairs:
        print(f"\n{'='*60}")
        print(f"EN: {en_p}")
        print(f"CN: {cn_p}")
        results, stats = _process_file_pair(
            en_p, cn_p, glossary_index, config, apply=args.apply,
        )
        all_results.extend(results)
        all_stats.append(stats)
        all_file_labels.append((str(en_p), str(cn_p)))

    # Summary
    total_changed = sum(s["fields_changed"] for s in all_stats)
    total_errors = sum(s["api_errors"] for s in all_stats)
    total_calls = sum(s["api_calls"] for s in all_stats)
    print(f"\n{'='*60}")
    print(f"Done. API calls: {total_calls}, Changes: {total_changed}, "
          f"Errors: {total_errors}")

    # Write report
    report_path = Path(config["report_path"]).resolve()
    _write_report(report_path, all_results, all_stats, all_file_labels)


if __name__ == "__main__":
    main()
