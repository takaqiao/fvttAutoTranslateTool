#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import copy
from difflib import SequenceMatcher
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
LATIN_RE = re.compile(r"[A-Za-z]")
WS_RE = re.compile(r"\s+")
TAG_RE = re.compile(r"<[^>]+>")
UUID_LABEL_RE = re.compile(r"(@(?:UUID|JournalEntry|Actor|Item|Compendium)\[[^\]]+\]\{)([^}]*)(\})")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
HTML_TAG_RE = re.compile(r"<[^>]+>")
FVTT_REF_NODE_RE = re.compile(r"^\s*(?:@[A-Za-z0-9_.:-]+\[[^\]]+\]\{[^}]*\}\s*)+$")
FVTT_REF_RE = re.compile(r"@([A-Za-z0-9_.:-]+)\[[^\]]+\](?:\{([^}]*)\})?")
FVTT_REF_NODE_FLEX_RE = re.compile(r"^\s*(?:@[A-Za-z0-9_.:-]+\[[^\]]+\](?:\{[^}]*\})?\s*)+$")
PathToken = Union[str, int]
TmCandidate = Tuple[List[PathToken], str]


def contains_cjk(text: str) -> bool:
    return bool(CJK_RE.search(text))


def contains_latin(text: str) -> bool:
    return bool(LATIN_RE.search(text))


def cjk_char_count(text: str) -> int:
    return len(CJK_RE.findall(text))


def normalize_for_match(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = html_to_text(text)

    def repl_ref(m: re.Match[str]) -> str:
        return f" @{m.group(1)} "

    text = FVTT_REF_RE.sub(repl_ref, text)
    text = UUID_LABEL_RE.sub(r"\1\3", text)
    text = CJK_RE.sub("", text)
    text = text.lower()
    text = NON_ALNUM_RE.sub(" ", text)
    text = WS_RE.sub(" ", text)
    return text.strip()


def normalize_for_equivalence(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    if looks_like_html(text):
        parts = re.split(r"(<[^>]+>)", text)
        filtered_parts: List[str] = []
        for i, part in enumerate(parts):
            if i % 2 != 0:
                filtered_parts.append(part)
                continue
            if contains_cjk(part):
                filtered_parts.append("")
            else:
                filtered_parts.append(part)
        text = "".join(filtered_parts)
    else:
        text = CJK_RE.sub("", text)

    text = TAG_RE.sub(" ", text)

    def repl_ref(m: re.Match[str]) -> str:
        label = m.group(2)
        if label:
            return f" {label} "
        return f" {m.group(1)} "

    text = FVTT_REF_RE.sub(repl_ref, text)
    text = CJK_RE.sub("", text)
    text = text.lower()
    text = NON_ALNUM_RE.sub(" ", text)
    text = WS_RE.sub(" ", text)
    return text.strip()


def html_to_text(text: str) -> str:
    return TAG_RE.sub("", text)


def looks_like_html(text: str) -> bool:
    return bool(HTML_TAG_RE.search(text))


def extract_text_nodes(html: str) -> List[str]:
    parts = re.split(r"(<[^>]+>)", html)
    nodes: List[str] = []
    for i, part in enumerate(parts):
        if i % 2 == 0 and part:
            nodes.append(part)
    return nodes


def _tag_signature(tag: str) -> str:
    m = re.match(r"<\s*(/)?\s*([a-zA-Z0-9:-]+)", tag)
    if not m:
        return ""
    closing = bool(m.group(1))
    name = m.group(2).lower()
    return f"/{name}" if closing else name


def harmonize_html_shell(old_html: str, new_html: str) -> str:
    old_tags = HTML_TAG_RE.findall(old_html)
    new_tags = HTML_TAG_RE.findall(new_html)
    if not old_tags or len(old_tags) != len(new_tags):
        return old_html

    old_sigs = [_tag_signature(x) for x in old_tags]
    new_sigs = [_tag_signature(x) for x in new_tags]
    if old_sigs != new_sigs:
        return old_html

    i = 0

    def repl(_: re.Match[str]) -> str:
        nonlocal i
        tag = new_tags[i]
        i += 1
        return tag

    return HTML_TAG_RE.sub(repl, old_html)


def harmonize_html_class_tokens(old_html: str, new_html: str) -> str:
    if "narrative-block" in new_html and "narrative-block" not in old_html:
        def repl_class(m: re.Match[str]) -> str:
            quote = m.group(1)
            classes = m.group(2)
            tokens = classes.split()
            updated = ["narrative-block" if t == "narrative" else t for t in tokens]
            return f'class={quote}{" ".join(updated)}{quote}'

        old_html = re.sub(r'class=("|\')([^"\']*)(\1)', repl_class, old_html)

    return old_html


def trim_trailing_english_duplicate_html(old_html: str) -> str:
    parts = re.split(r"(<[^>]+>)", old_html)
    seen_cjk_nodes = 0
    cut_from: int | None = None

    for i, part in enumerate(parts):
        if i % 2 != 0:
            continue
        text = part.strip()
        if not text:
            continue

        if contains_cjk(text):
            seen_cjk_nodes += 1
            continue

        if seen_cjk_nodes >= 3 and contains_latin(text):
            if FVTT_REF_NODE_RE.fullmatch(text):
                continue
            norm_len = len(normalize_for_match(text))
            if norm_len >= 40:
                cut_from = max(i - 1, 0)
                break

    if cut_from is not None:
        return "".join(parts[:cut_from]).strip()

    return old_html


def strip_english_text_nodes_preserving_html(old_html: str) -> str:
    parts = re.split(r"(<[^>]+>)", old_html)
    for i, part in enumerate(parts):
        if i % 2 != 0:
            continue

        text = part
        if not text or not text.strip():
            continue

        if contains_cjk(text):
            continue

        if contains_latin(text):
            if FVTT_REF_NODE_RE.fullmatch(text):
                continue
            parts[i] = ""

    return cleanup_empty_html_fragments("".join(parts))


def cleanup_empty_html_fragments(html: str) -> str:
    prev = None
    cur = html

    while cur != prev:
        prev = cur

        cur = re.sub(
            r"<(h[1-6]|p|em|strong|li|figcaption)(\s[^>]*)?>\s*</\1>",
            "",
            cur,
            flags=re.IGNORECASE,
        )

        cur = re.sub(
            r"<(article|aside|section|header)(\s[^>]*)?>\s*</\1>",
            "",
            cur,
            flags=re.IGNORECASE,
        )

        cur = re.sub(
            r"<ul(\s[^>]*)?>\s*</ul>",
            "",
            cur,
            flags=re.IGNORECASE,
        )

    return cur


def build_bilingual_html(old_html: str, new_html: str) -> str:
    raise NotImplementedError("Use build_bilingual_html_with_new_shell")


def build_local_phrase_memory_from_html(old_html: str, min_key_len: int) -> Dict[str, str]:
    phrase_tm: Dict[str, str] = {}

    def add_phrase(english_text: str, chinese_text: str) -> None:
        key = normalize_for_match(english_text)
        if len(key) < min_key_len:
            return
        existing = phrase_tm.get(key)
        phrase_tm[key] = keep_better_translation(existing, chinese_text)

    text_nodes = [x.strip() for x in extract_text_nodes(old_html) if x.strip()]
    text_nodes = [x for x in text_nodes if not FVTT_REF_NODE_FLEX_RE.fullmatch(x)]
    cjk_nodes = [x for x in text_nodes if contains_cjk(x)]
    eng_nodes = [x for x in text_nodes if contains_latin(x) and not contains_cjk(x)]

    if cjk_nodes and eng_nodes and len(cjk_nodes) == len(eng_nodes):
        for zh, en in zip(cjk_nodes, eng_nodes):
            add_phrase(en, zh)
        return phrase_tm

    if len(text_nodes) >= 4:
        mid = len(text_nodes) // 2
        zh_half = [x for x in text_nodes[:mid] if contains_cjk(x)]
        en_half = [x for x in text_nodes[mid:] if contains_latin(x) and not contains_cjk(x)]
        if zh_half and en_half and len(zh_half) == len(en_half):
            for zh, en in zip(zh_half, en_half):
                add_phrase(en, zh)

    return phrase_tm


def build_bilingual_html_with_new_shell(
    old_html: str,
    new_html: str,
    min_key_len: int,
) -> str:
    local_tm = build_local_phrase_memory_from_html(old_html, min_key_len)
    zh_part, replaced_count, candidate_count = replace_html_text_nodes(new_html, local_tm, min_key_len)

    if candidate_count == 0:
        return new_html

    if replaced_count != candidate_count:
        localized_old = strip_english_text_nodes_preserving_html(old_html)
        if contains_cjk(localized_old):
            fallback_zh = harmonize_html_shell(localized_old, new_html)
            fallback_zh = harmonize_html_class_tokens(fallback_zh, new_html)
            if contains_cjk(fallback_zh):
                return f"{fallback_zh}\n{new_html}"
        return new_html

    if not contains_cjk(zh_part):
        return new_html

    return f"{zh_part}\n{new_html}"


def extract_localized_plain_text(old_text: str) -> str:
    text = old_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return ""

    if not contains_cjk(text):
        return ""

    if not contains_latin(text):
        return text

    lines = [line.strip() for line in text.split("\n")]
    cjk_lines = [line for line in lines if line and contains_cjk(line)]
    if cjk_lines:
        return "\n".join(cjk_lines).strip()

    return text


def extract_page_field_key(path_tokens: List[PathToken]) -> Tuple[str, str] | None:
    if len(path_tokens) < 3:
        return None
    if not isinstance(path_tokens[-1], str):
        return None

    for i, token in enumerate(path_tokens):
        if token == "pages" and i + 1 < len(path_tokens):
            page_token = path_tokens[i + 1]
            if isinstance(page_token, str):
                return (page_token, path_tokens[-1])
    return None


def build_page_field_index(old_root: Dict[str, Any], scope: str) -> Dict[Tuple[str, str], List[TmCandidate]]:
    index: Dict[Tuple[str, str], List[TmCandidate]] = {}

    for path_tokens, old_value in collect_strings(old_root):
        if not in_scope(path_tokens, scope):
            continue
        if not (contains_cjk(old_value) and contains_latin(old_value)):
            continue

        key = extract_page_field_key(path_tokens)
        if key is None:
            continue

        index.setdefault(key, []).append((path_tokens, old_value))

    return index


def pick_equivalent_candidate(
    candidates: List[TmCandidate],
    target_path: List[PathToken],
    new_value: str,
    min_ratio: float = 0.99,
    min_coverage: float = 0.99,
) -> str | None:
    new_eq = normalize_for_equivalence(new_value)
    if not new_eq:
        return None

    strict_mode = min_ratio >= 1.0 and min_coverage >= 1.0

    new_tokens = [x for x in new_eq.split(" ") if x]
    new_token_set = set(new_tokens)
    if not new_token_set:
        return None

    best_value: str | None = None
    best_score = (-1.0, -1.0, -1, -1, -1, -1)

    for cand_path, cand_value in candidates:
        cand_eq = normalize_for_equivalence(cand_value)
        if not cand_eq:
            continue

        if strict_mode and cand_eq != new_eq:
            continue

        ratio = SequenceMatcher(None, new_eq, cand_eq).ratio()

        cand_tokens = set(x for x in cand_eq.split(" ") if x)
        if not cand_tokens:
            continue
        coverage = len(new_token_set & cand_tokens) / len(new_token_set)

        if strict_mode:
            if ratio < min_ratio or coverage < min_coverage:
                continue
        else:
            if ratio < min_ratio and coverage < min_coverage:
                continue

        suffix, prefix = path_similarity_score(target_path, cand_path)
        score = (coverage, ratio, suffix, prefix, cjk_char_count(cand_value), len(cand_value))
        if score > best_score:
            best_score = score
            best_value = cand_value

    return best_value


def replace_html_text_nodes(new_html: str, phrase_tm: Dict[str, str], min_key_len: int) -> Tuple[str, int, int]:
    def transplant_new_refs(translated_text: str, new_text: str) -> str:
        translated_refs = list(FVTT_REF_RE.finditer(translated_text))
        if not translated_refs:
            return translated_text

        new_refs = [m.group(0) for m in FVTT_REF_RE.finditer(new_text)]
        if len(new_refs) != len(translated_refs):
            return translated_text

        rebuilt: List[str] = []
        last = 0
        for i, m in enumerate(translated_refs):
            rebuilt.append(translated_text[last:m.start()])
            rebuilt.append(new_refs[i])
            last = m.end()
        rebuilt.append(translated_text[last:])
        return "".join(rebuilt)

    parts = re.split(r"(<[^>]+>)", new_html)
    replaced_count = 0
    candidate_count = 0

    for i, part in enumerate(parts):
        if i % 2 != 0:
            continue

        text = part
        if not text or not text.strip():
            continue
        if contains_cjk(text):
            continue
        if not contains_latin(text):
            continue

        key = normalize_for_match(text)
        if len(key) < min_key_len:
            continue
        candidate_count += 1

        translated = phrase_tm.get(key)
        if translated:
            translated_ref_matches_before = list(FVTT_REF_RE.finditer(translated))
            new_ref_matches = list(FVTT_REF_RE.finditer(text))
            if len(translated_ref_matches_before) != len(new_ref_matches):
                continue

            translated = transplant_new_refs(translated, text)
            new_ref_types = set(x.group(1) for x in FVTT_REF_RE.finditer(text))
            translated_ref_types = set(x.group(1) for x in FVTT_REF_RE.finditer(translated))
            if not new_ref_types.issubset(translated_ref_types):
                continue
            parts[i] = translated
            replaced_count += 1

    return "".join(parts), replaced_count, candidate_count


def split_candidate_segments(text: str) -> List[str]:
    segments = [text]

    by_newline = [x.strip() for x in re.split(r"\n+", text) if x.strip()]
    segments.extend(by_newline)

    by_para = [x.strip() for x in re.split(r"</p>|<br\s*/?>", text, flags=re.IGNORECASE) if x.strip()]
    segments.extend(by_para)

    clean_segments: List[str] = []
    seen = set()
    for seg in segments:
        seg_text = html_to_text(seg).strip()
        if not seg_text:
            continue
        if seg_text in seen:
            continue
        seen.add(seg_text)
        clean_segments.append(seg)
    return clean_segments


def format_path(path_tokens: List[PathToken]) -> str:
    parts: List[str] = []
    for token in path_tokens:
        if isinstance(token, int):
            if parts:
                parts[-1] = f"{parts[-1]}[{token}]"
            else:
                parts.append(f"[{token}]")
        else:
            parts.append(token)
    return ".".join(parts)


def collect_strings(node: Any, path_tokens: List[PathToken] | None = None) -> List[Tuple[List[PathToken], str]]:
    if path_tokens is None:
        path_tokens = []

    found: List[Tuple[List[PathToken], str]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            found.extend(collect_strings(value, path_tokens + [key]))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            found.extend(collect_strings(value, path_tokens + [i]))
    elif isinstance(node, str):
        found.append((path_tokens, node))
    return found


def get_by_path(root: Any, path_tokens: List[PathToken]) -> Any:
    cur = root
    for token in path_tokens:
        if isinstance(token, int):
            cur = cur[token]
        else:
            cur = cur[token]
    return cur


def get_by_path_safe(root: Any, path_tokens: List[PathToken]) -> Tuple[bool, Any]:
    try:
        return True, get_by_path(root, path_tokens)
    except (KeyError, IndexError, TypeError):
        return False, None


def set_by_path(root: Any, path_tokens: List[PathToken], value: Any) -> None:
    cur = root
    for i, token in enumerate(path_tokens):
        is_last = i == len(path_tokens) - 1
        if isinstance(token, int):
            if is_last:
                cur[token] = value
            else:
                cur = cur[token]
        else:
            if is_last:
                cur[token] = value
            else:
                cur = cur[token]


def is_journal_path(path_tokens: List[PathToken]) -> bool:
    return any(isinstance(x, str) and x == "journals" for x in path_tokens)


def in_scope(path_tokens: List[PathToken], scope: str) -> bool:
    if scope == "all":
        return True
    return is_journal_path(path_tokens)


def keep_better_translation(existing: str | None, candidate: str) -> str:
    if existing is None:
        return candidate
    existing_score = (cjk_char_count(existing), len(existing))
    candidate_score = (cjk_char_count(candidate), len(candidate))
    if candidate_score > existing_score:
        return candidate
    return existing


def path_similarity_score(target: List[PathToken], candidate: List[PathToken]) -> Tuple[int, int]:
    suffix = 0
    i = len(target) - 1
    j = len(candidate) - 1
    while i >= 0 and j >= 0 and target[i] == candidate[j]:
        suffix += 1
        i -= 1
        j -= 1

    prefix = 0
    max_prefix = min(len(target), len(candidate))
    while prefix < max_prefix and target[prefix] == candidate[prefix]:
        prefix += 1

    return suffix, prefix


def choose_best_candidate(candidates: List[TmCandidate], target_path: List[PathToken]) -> str:
    best_value = candidates[0][1]
    best_score = (-1, -1, -1, -1)

    for cand_path, cand_value in candidates:
        suffix, prefix = path_similarity_score(target_path, cand_path)
        score = (suffix, prefix, cjk_char_count(cand_value), len(cand_value))
        if score > best_score:
            best_score = score
            best_value = cand_value

    return best_value


def expand_match_keys(key: str) -> List[str]:
    keys = [key]
    tokens = key.split(" ") if key else []

    while tokens and tokens[0].isdigit():
        tokens = tokens[1:]
    if tokens:
        trimmed_leading_numbers = " ".join(tokens)
        if trimmed_leading_numbers not in keys:
            keys.append(trimmed_leading_numbers)

    if tokens and len(tokens) % 2 == 0:
        half = len(tokens) // 2
        if tokens[:half] == tokens[half:]:
            dedup_halves = " ".join(tokens[:half])
            if dedup_halves and dedup_halves not in keys:
                keys.append(dedup_halves)

    if tokens and len(tokens) >= 3 and tokens[0] == tokens[-1]:
        drop_first = " ".join(tokens[1:])
        drop_last = " ".join(tokens[:-1])
        if drop_first and drop_first not in keys:
            keys.append(drop_first)
        if drop_last and drop_last not in keys:
            keys.append(drop_last)

    if len(tokens) >= 2 and tokens[0] == tokens[1]:
        drop_repeated_prefix = " ".join(tokens[1:])
        if drop_repeated_prefix and drop_repeated_prefix not in keys:
            keys.append(drop_repeated_prefix)

    return keys


def build_translation_memory(old_root: Dict[str, Any], scope: str, min_key_len: int) -> Dict[str, List[TmCandidate]]:
    tm: Dict[str, List[TmCandidate]] = {}

    def add_candidate(key: str, path_tokens: List[PathToken], value: str) -> None:
        bucket = tm.setdefault(key, [])
        for i, (old_path, old_value) in enumerate(bucket):
            if old_path == path_tokens:
                bucket[i] = (path_tokens, keep_better_translation(old_value, value))
                return
        bucket.append((path_tokens, value))

    for path_tokens, old_value in collect_strings(old_root):
        if not in_scope(path_tokens, scope):
            continue
        if not contains_cjk(old_value):
            continue
        if not contains_latin(old_value):
            continue

        candidates = split_candidate_segments(old_value)
        for cand in candidates:
            if contains_cjk(cand):
                continue
            if not contains_latin(cand):
                continue
            key = normalize_for_match(cand)
            if len(key) < min_key_len:
                continue
            for expanded_key in expand_match_keys(key):
                if len(expanded_key) < min_key_len:
                    continue
                add_candidate(expanded_key, path_tokens, old_value)

        whole_key = normalize_for_match(old_value)
        if len(whole_key) >= min_key_len:
            for expanded_key in expand_match_keys(whole_key):
                if len(expanded_key) < min_key_len:
                    continue
                add_candidate(expanded_key, path_tokens, old_value)

    return tm


def merge_journals(
    old_root: Dict[str, Any],
    new_root: Dict[str, Any],
    scope: str,
    min_key_len: int,
    min_ratio: float,
    min_coverage: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    tm = build_translation_memory(old_root, scope, min_key_len)
    page_field_index = build_page_field_index(old_root, scope)
    merged = copy.deepcopy(new_root)
    report: List[Dict[str, Any]] = []

    for path_tokens, new_value in collect_strings(new_root):
        if not in_scope(path_tokens, scope):
            continue
        if contains_cjk(new_value):
            continue
        if not contains_latin(new_value):
            continue

        key = normalize_for_match(new_value)
        if len(key) < min_key_len:
            report.append({"path": format_path(path_tokens), "status": "skipped_short", "new_len": len(new_value)})
            continue

        old_full = None

        found_same_path, same_path_value = get_by_path_safe(old_root, path_tokens)
        if found_same_path and isinstance(same_path_value, str):
            if contains_cjk(same_path_value) and contains_latin(same_path_value):
                if pick_equivalent_candidate(
                    [(path_tokens, same_path_value)],
                    path_tokens,
                    new_value,
                    min_ratio=min_ratio,
                    min_coverage=min_coverage,
                ):
                    old_full = same_path_value

        if old_full is None:
            page_key = extract_page_field_key(path_tokens)
            if page_key is not None:
                page_candidates = page_field_index.get(page_key)
                if page_candidates:
                    old_full = pick_equivalent_candidate(
                        page_candidates,
                        path_tokens,
                        new_value,
                        min_ratio=min_ratio,
                        min_coverage=min_coverage,
                    )

        if old_full is None:
            candidates = tm.get(key)
            if candidates:
                old_full = pick_equivalent_candidate(
                    candidates,
                    path_tokens,
                    new_value,
                    min_ratio=min_ratio,
                    min_coverage=min_coverage,
                )

        if old_full is not None:
            replacement_value = old_full
            if looks_like_html(new_value):
                if looks_like_html(old_full):
                    localized_old = strip_english_text_nodes_preserving_html(old_full)
                    if contains_cjk(localized_old):
                        replacement_value = build_bilingual_html_with_new_shell(localized_old, new_value, min_key_len)
                    else:
                        replacement_value = new_value
                else:
                    replacement_value = new_value
            else:
                localized_text = extract_localized_plain_text(old_full)
                if localized_text:
                    if normalize_for_equivalence(localized_text) == normalize_for_equivalence(new_value):
                        replacement_value = localized_text
                    else:
                        replacement_value = f"{localized_text}\n{new_value}"
                else:
                    replacement_value = new_value

            if replacement_value != new_value:
                set_by_path(merged, path_tokens, replacement_value)
                report.append({"path": format_path(path_tokens), "status": "replaced", "new_len": len(new_value), "old_len": len(old_full)})
            else:
                report.append({"path": format_path(path_tokens), "status": "unmatched", "new_len": len(new_value)})
        else:
            report.append({"path": format_path(path_tokens), "status": "unmatched", "new_len": len(new_value)})

    return merged, report


def main() -> None:
    parser = argparse.ArgumentParser(description="FVTT PF2E 文本汉化回填工具（按英文匹配）")
    parser.add_argument("--old", required=True, help="旧版已汉化 JSON")
    parser.add_argument("--new", required=True, help="新版官方 JSON")
    parser.add_argument("--out", required=True, help="输出 JSON")
    parser.add_argument("--report", default="merge_report.json", help="匹配报告 JSON")
    parser.add_argument("--scope", choices=["all", "journals"], default="all", help="替换范围：all=全量字符串值，journals=仅 journals")
    parser.add_argument("--min-key-len", type=int, default=4, help="匹配键最小长度（默认 4，避免过短噪声）")
    parser.add_argument("--min-ratio", type=float, default=1.0, help="文本等价最小相似度（默认 1.0）")
    parser.add_argument("--min-coverage", type=float, default=1.0, help="文本token覆盖率阈值（默认 1.0）")
    args = parser.parse_args()

    old_path = Path(args.old)
    new_path = Path(args.new)
    out_path = Path(args.out)
    report_path = Path(args.report)

    with old_path.open("r", encoding="utf-8") as f:
        old_root = json.load(f)
    with new_path.open("r", encoding="utf-8") as f:
        new_root = json.load(f)

    merged, report = merge_journals(
        old_root,
        new_root,
        args.scope,
        args.min_key_len,
        args.min_ratio,
        args.min_coverage,
    )

    out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    replaced = sum(1 for x in report if x["status"] == "replaced")
    unmatched = sum(1 for x in report if x["status"] == "unmatched")

    print(
        f"完成：scope={args.scope}, min_key_len={args.min_key_len}, "
        f"min_ratio={args.min_ratio}, min_coverage={args.min_coverage}, "
        f"replaced={replaced}, unmatched={unmatched}"
    )
    print(f"输出文件: {out_path}")
    print(f"报告文件: {report_path}")


if __name__ == "__main__":
    main()
