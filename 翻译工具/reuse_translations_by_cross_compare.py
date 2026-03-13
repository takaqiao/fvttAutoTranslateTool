#!/usr/bin/env python3
"""Cross-compare language packs and reuse translated entries.

Typical use case:
- Build translation memory from PF2E EN/ZH file pairs.
- Apply it to SF2E EN/ZH pairs.
- Replace only untranslated target entries by default.

Examples:
    # Preview only (no file is written)
    python 翻译工具/reuse_translations_by_cross_compare.py \
      --ref path/to/pf2e-en.json path/to/pf2e-zh.json \
      --target sf2e/en.json sf2e/zh_Hans.json

    # Write side-by-side output files with suffix (.reused.json)
    python 翻译工具/reuse_translations_by_cross_compare.py \
      --ref path/to/pf2e-en.json path/to/pf2e-zh.json \
      --target sf2e/en.json sf2e/zh_Hans.json \
      --apply

    # Overwrite target zh files in place
    python 翻译工具/reuse_translations_by_cross_compare.py \
      --ref path/to/pf2e-en.json path/to/pf2e-zh.json \
      --target sf2e/en.json sf2e/zh_Hans.json \
      --apply --in-place

        # Direct mapping mode: SF2 EN + PF2 ZH -> SF2 ZH
        python 翻译工具/reuse_translations_by_cross_compare.py \
            --map sf2e/en.json pf2e/zh_Hans.json sf2e/zh_Hans.json \
            --map sf2e/action-en.json pf2e/action-zh_Hans.json sf2e/action-zh_Hans.json \
            --apply
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


PathToken = str | int
PathTuple = tuple[PathToken, ...]

COMPENDIUM_SYSTEM_RE = re.compile(r"(Compendium\.)([A-Za-z0-9_-]+)(\.)")
COMPENDIUM_BRACKET_RE = re.compile(r"(@Compendium\[)([A-Za-z0-9_-]+)(\.[^\]]+\])")
PLACEHOLDER_RE = re.compile(r"\{[A-Za-z0-9_.-]+\}")
INLINE_LINK_LABEL_RE = re.compile(
    r"(?P<head>(?:"
    r"@(?:UUID|Compendium|Actor|Item|JournalEntry|RollTable|Macro|Check|Damage|Template|Localize|Embed)\[[^\]]+\]"
    r"|\[\[/[^\]]+\]\]"
    r"))\{[^\}]*\}"
)
WHITESPACE_RE = re.compile(r"\s+", flags=re.UNICODE)
TOKEN_RE = re.compile(r"[a-z0-9]+")
# Foundry-like inline command markers that should be preserved from the target text.
CODE_LIKE_RE = re.compile(
    r"@(?:UUID|Check|Damage|Template|Localize|RollTable|Embed|Actor|Item)\["
    r"|@(?:abilities|attributes|skills?|saves?|item|actor)\."
    r"|\[\[/"
    r"|Compendium\."
)
UUID_LINK_RE = re.compile(r"@UUID\[(?P<link>[^\]]+)\](?:\{(?P<label>[^\}]*)\})?")
COMPENDIUM_UUID_LINK_RE = re.compile(
    r"^(Compendium\.)(?P<system>[A-Za-z0-9_-]+)\.(?P<pack>[A-Za-z0-9_-]+)\.(?P<rest>.+)$"
)
PACK_ALIAS_MAP = {
    "actionspf2e": "actions",
    "conditionitems": "conditions",
    "spells-srd": "spells",
}
_MISSING = object()


@dataclass(frozen=True)
class Candidate:
    zh: str
    source_en: str
    source_path: PathTuple
    source_label: str


@dataclass
class FileStats:
    target_en: str
    target_zh: str
    total_strings: int = 0
    eligible_strings: int = 0
    replaced_path_match: int = 0
    replaced_value_match: int = 0
    replaced_path_only: int = 0
    replaced_uuid_path_match: int = 0
    replaced_uuid_value_match: int = 0
    skipped_existing: int = 0
    skipped_no_match: int = 0
    skipped_ambiguous: int = 0
    skipped_placeholder: int = 0
    skipped_uuid_mismatch: int = 0
    skipped_code_like: int = 0
    unchanged_same_text: int = 0
    output_path: str = ""
    samples: list[dict[str, str]] = field(default_factory=list)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def dump_json(path: Path, data: Any) -> None:
    text = json.dumps(data, ensure_ascii=False, indent=4)
    path.write_text(text + "\n", encoding="utf-8")


def iter_string_leaves(node: Any, base_path: PathTuple = ()) -> Iterable[tuple[PathTuple, str]]:
    if isinstance(node, dict):
        for key, value in node.items():
            yield from iter_string_leaves(value, base_path + (key,))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from iter_string_leaves(value, base_path + (index,))
    elif isinstance(node, str):
        yield base_path, node


def get_at_path(node: Any, path: PathTuple) -> Any:
    current = node
    for token in path:
        if isinstance(token, int):
            if not isinstance(current, list) or token >= len(current):
                return _MISSING
        else:
            if not isinstance(current, dict) or token not in current:
                return _MISSING
        current = current[token]
    return current


def set_at_path(node: Any, path: PathTuple, value: Any) -> bool:
    current = node
    for idx, token in enumerate(path):
        is_last = idx == len(path) - 1
        next_token = None if is_last else path[idx + 1]

        if isinstance(token, str):
            if not isinstance(current, dict):
                return False
            if is_last:
                current[token] = value
                return True
            if token not in current or not isinstance(current[token], (dict, list)):
                current[token] = [] if isinstance(next_token, int) else {}
            current = current[token]
            continue

        if not isinstance(current, list):
            return False

        while len(current) <= token:
            current.append([] if isinstance(next_token, int) else {})

        if is_last:
            current[token] = value
            return True

        if not isinstance(current[token], (dict, list)):
            current[token] = [] if isinstance(next_token, int) else {}
        current = current[token]

    return False


def path_to_text(path: PathTuple) -> str:
    out = "root"
    for token in path:
        if isinstance(token, int):
            out += f"[{token}]"
        else:
            out += f".{token}"
    return out


def pack_name_from_filename(file_name: str) -> str:
    stem = Path(file_name).stem.lower()
    if "." in stem:
        return stem.split(".", 1)[1]
    return stem


def stem_tokens(name: str) -> set[str]:
    stem = Path(name).stem.lower()
    return set(TOKEN_RE.findall(stem))


def normalize_source_text(text: str) -> str:
    normalized = COMPENDIUM_SYSTEM_RE.sub(r"\1__SYS__\3", text)
    normalized = COMPENDIUM_BRACKET_RE.sub(r"\1__SYS__\3", normalized)
    # Ignore whitespace differences when comparing English source strings.
    return WHITESPACE_RE.sub("", normalized)


def has_uuid_link(text: str) -> bool:
    return "@UUID[" in text


def has_system_link_like(text: str) -> bool:
    # These patterns often embed system-specific identifiers.
    return "@UUID[" in text or "Compendium." in text or "@Compendium[" in text


def has_compendium_link(text: str) -> bool:
    return "Compendium." in text or "@Compendium[" in text


def normalize_uuid_source_text(text: str) -> str:
    # Keep visible UUID labels and surrounding content, but ignore the UUID target itself.
    def repl(match: re.Match[str]) -> str:
        label = match.group("label")
        if label is None:
            return "@UUID[__UUID__]"
        return f"@UUID[__UUID__]{{{label}}}"

    return normalize_source_text(UUID_LINK_RE.sub(repl, text))


def normalize_uuid_label(label: str | None) -> str:
    if not label:
        return ""
    return WHITESPACE_RE.sub(" ", label).strip().lower()


def uuid_link_identifier(link: str) -> str:
    link = link.strip()
    if not link:
        return ""
    return link.rsplit(".", 1)[-1].lower()


def rewrite_compendium_pack_alias_in_uuid_link(link: str) -> str:
    match = COMPENDIUM_UUID_LINK_RE.match(link)
    if not match:
        return link

    pack = match.group("pack")
    mapped_pack = PACK_ALIAS_MAP.get(pack, pack)
    return (
        f"{match.group(1)}{match.group('system')}."
        f"{mapped_pack}.{match.group('rest')}"
    )


def rewrite_uuid_targets_from_target_en(translation: str, target_en: str) -> str | None:
    trans_matches = list(UUID_LINK_RE.finditer(translation))
    target_matches = list(UUID_LINK_RE.finditer(target_en))

    target_index_by_trans_index: dict[int, int] = {}

    if len(trans_matches) == len(target_matches):
        target_index_by_trans_index = {idx: idx for idx in range(len(trans_matches))}
    elif len(trans_matches) > len(target_matches):
        # Conservatively keep target UUID targets and demote unmatched extra UUID links to plain labels.
        matched_trans_indexes: list[int] = []
        search_start = 0

        for target_match in target_matches:
            target_id = uuid_link_identifier(target_match.group("link"))
            target_label = normalize_uuid_label(target_match.group("label"))
            chosen_index: int | None = None

            if target_id:
                for idx in range(search_start, len(trans_matches)):
                    trans_id = uuid_link_identifier(trans_matches[idx].group("link"))
                    if trans_id == target_id:
                        chosen_index = idx
                        break

            if chosen_index is None and target_label:
                for idx in range(search_start, len(trans_matches)):
                    trans_label = normalize_uuid_label(trans_matches[idx].group("label"))
                    if trans_label == target_label:
                        chosen_index = idx
                        break

            if chosen_index is None:
                return None

            matched_trans_indexes.append(chosen_index)
            search_start = chosen_index + 1

        matched_set = set(matched_trans_indexes)
        if len(matched_set) != len(matched_trans_indexes):
            return None

        target_index_by_trans_index = {
            trans_index: target_index
            for target_index, trans_index in enumerate(matched_trans_indexes)
        }
    elif len(trans_matches) < len(target_matches):
        # Conservatively allow source text to have fewer UUID links: rewrite links that exist,
        # and keep missing links absent rather than forcing insertion.
        target_index_by_id: dict[str, int] = {}
        target_index_by_label: dict[str, int] = {}
        for idx, target_match in enumerate(target_matches):
            target_id = uuid_link_identifier(target_match.group("link"))
            if target_id and target_id not in target_index_by_id:
                target_index_by_id[target_id] = idx

            target_label = normalize_uuid_label(target_match.group("label"))
            if target_label and target_label not in target_index_by_label:
                target_index_by_label[target_label] = idx

        search_start = 0

        for trans_index, trans_match in enumerate(trans_matches):
            trans_id = uuid_link_identifier(trans_match.group("link"))
            trans_label = normalize_uuid_label(trans_match.group("label"))
            chosen_index: int | None = None

            if trans_id:
                chosen_index = target_index_by_id.get(trans_id)

            if chosen_index is None and trans_label:
                chosen_index = target_index_by_label.get(trans_label)

            if chosen_index is None:
                if search_start >= len(target_matches):
                    return None
                chosen_index = search_start

            target_index_by_trans_index[trans_index] = chosen_index
            if chosen_index >= search_start:
                search_start = chosen_index + 1
    else:
        return None

    rebuilt: list[str] = []
    cursor = 0
    for trans_index, trans_match in enumerate(trans_matches):
        rebuilt.append(translation[cursor : trans_match.start()])
        target_index = target_index_by_trans_index.get(trans_index)
        if target_index is None:
            label = trans_match.group("label")
            if label is not None:
                rebuilt.append(label)
            else:
                extra_link = rewrite_compendium_pack_alias_in_uuid_link(trans_match.group("link"))
                rebuilt.append(f"@UUID[{extra_link}]")
        else:
            target_match = target_matches[target_index]
            target_link = target_match.group("link")
            trans_label = trans_match.group("label")
            target_label = target_match.group("label")
            if trans_label is not None:
                rebuilt.append(f"@UUID[{target_link}]{{{trans_label}}}")
            elif target_label is not None:
                rebuilt.append(f"@UUID[{target_link}]{{{target_label}}}")
            else:
                rebuilt.append(f"@UUID[{target_link}]")

        cursor = trans_match.end()

    rebuilt.append(translation[cursor:])
    return "".join(rebuilt)


def is_code_like_text(text: str) -> bool:
    return bool(CODE_LIKE_RE.search(text))


def placeholder_tokens(text: str) -> set[str]:
    # Ignore braces used as display labels in inline links, e.g. @UUID[...]{Label}.
    sanitized = INLINE_LINK_LABEL_RE.sub(r"\g<head>", text)
    return set(PLACEHOLDER_RE.findall(sanitized))


def is_untranslated(current_zh: Any, en_text: str) -> bool:
    if not isinstance(current_zh, str):
        return True
    if current_zh.strip() == "":
        return True
    return current_zh == en_text


def extract_compendium_systems(text: str) -> list[str]:
    systems: list[str] = []
    for match in COMPENDIUM_SYSTEM_RE.finditer(text):
        system = match.group(2)
        if system not in systems:
            systems.append(system)
    for match in COMPENDIUM_BRACKET_RE.finditer(text):
        system = match.group(2)
        if system not in systems:
            systems.append(system)
    return systems


def rewrite_compendium_systems(translation: str, source_en: str, target_en: str) -> str:
    source_systems = extract_compendium_systems(source_en)
    target_systems = extract_compendium_systems(target_en)
    if not source_systems or not target_systems:
        return translation

    mapping: dict[str, str]
    if len(source_systems) == len(target_systems):
        mapping = dict(zip(source_systems, target_systems))
    elif len(target_systems) == 1:
        mapping = {source: target_systems[0] for source in source_systems}
    else:
        return translation

    def repl(match: re.Match[str]) -> str:
        prefix, system, suffix = match.group(1), match.group(2), match.group(3)
        return f"{prefix}{mapping.get(system, system)}{suffix}"

    rewritten = COMPENDIUM_SYSTEM_RE.sub(repl, translation)
    rewritten = COMPENDIUM_BRACKET_RE.sub(repl, rewritten)
    return rewritten


def pick_unique(
    candidates: list[Candidate],
    *,
    target_name_tokens: set[str] | None = None,
    preferred_source_files: set[str] | None = None,
) -> tuple[Candidate | None, bool]:
    if not candidates:
        return None, False
    uniq_by_zh: dict[str, Candidate] = {}
    for candidate in candidates:
        uniq_by_zh.setdefault(candidate.zh, candidate)
    if len(uniq_by_zh) == 1:
        return next(iter(uniq_by_zh.values())), False

    # Conservative tie-breaker: if one source filename is clearly closest to target filename,
    # use it; otherwise keep as ambiguous.
    if target_name_tokens:
        scored: list[tuple[int, Candidate]] = []
        for candidate in uniq_by_zh.values():
            source_en_name = candidate.source_label.split("|", 1)[0]
            score = len(target_name_tokens & stem_tokens(source_en_name))
            scored.append((score, candidate))

        best_score = max((score for score, _ in scored), default=0)
        if best_score > 0:
            best_candidates = [candidate for score, candidate in scored if score == best_score]
            best_uniq_by_zh: dict[str, Candidate] = {}
            for candidate in best_candidates:
                best_uniq_by_zh.setdefault(candidate.zh, candidate)
            if len(best_uniq_by_zh) == 1:
                return next(iter(best_uniq_by_zh.values())), False

    # Conservative pack affinity tie-breaker: prefer source file(s) that map directly
    # to the current target pack name.
    if preferred_source_files:
        preferred_candidates = [
            candidate
            for candidate in uniq_by_zh.values()
            if candidate.source_label.split("|", 1)[0] in preferred_source_files
        ]
        if preferred_candidates:
            preferred_uniq_by_zh: dict[str, Candidate] = {}
            for candidate in preferred_candidates:
                preferred_uniq_by_zh.setdefault(candidate.zh, candidate)
            if len(preferred_uniq_by_zh) == 1:
                return next(iter(preferred_uniq_by_zh.values())), False

    return None, True


def preferred_source_files_for_target(
    target_en: Path,
    ref_pairs: list[tuple[Path, Path]],
) -> set[str]:
    target_pack = pack_name_from_filename(target_en.name)
    preferred: set[str] = set()
    for ref_en, _ in ref_pairs:
        source_pack = pack_name_from_filename(ref_en.name)
        if source_pack == target_pack or PACK_ALIAS_MAP.get(source_pack) == target_pack:
            preferred.add(ref_en.name)
    return preferred


def build_reference_maps(
    refs: list[tuple[Path, Path]],
    *,
    skip_code_like: bool,
    enable_uuid_safe_match: bool,
) -> tuple[
    dict[tuple[PathTuple, str], list[Candidate]],
    dict[str, list[Candidate]],
    dict[PathTuple, list[Candidate]],
    dict[tuple[PathTuple, str], list[Candidate]],
    dict[str, list[Candidate]],
]:
    path_map: dict[tuple[PathTuple, str], list[Candidate]] = {}
    value_map: dict[str, list[Candidate]] = {}
    path_only_map: dict[PathTuple, list[Candidate]] = {}
    uuid_path_map: dict[tuple[PathTuple, str], list[Candidate]] = {}
    uuid_value_map: dict[str, list[Candidate]] = {}

    for ref_en, ref_zh in refs:
        ref_en_data = load_json(ref_en)
        ref_zh_data = load_json(ref_zh)
        source_label = f"{ref_en.name}|{ref_zh.name}"

        for path, en_text in iter_string_leaves(ref_en_data):
            zh_text = get_at_path(ref_zh_data, path)
            if not isinstance(zh_text, str):
                continue
            if zh_text.strip() == "" or zh_text == en_text:
                continue

            is_code_like = is_code_like_text(en_text)
            is_system_link_like = has_system_link_like(en_text)
            if (
                skip_code_like
                and is_code_like
                and is_system_link_like
                and has_uuid_link(en_text)
                and not enable_uuid_safe_match
            ):
                # Keep conservative default for system-linked markup in normal mapping.
                pass
            else:
                normalized = normalize_source_text(en_text).lower()
                candidate = Candidate(
                    zh=zh_text,
                    source_en=en_text,
                    source_path=path,
                    source_label=source_label,
                )
                path_map.setdefault((path, normalized), []).append(candidate)
                value_map.setdefault(normalized, []).append(candidate)

            if enable_uuid_safe_match and has_uuid_link(en_text):
                uuid_normalized = normalize_uuid_source_text(en_text).lower()
                candidate = Candidate(
                    zh=zh_text,
                    source_en=en_text,
                    source_path=path,
                    source_label=source_label,
                )
                uuid_path_map.setdefault((path, uuid_normalized), []).append(candidate)
                uuid_value_map.setdefault(uuid_normalized, []).append(candidate)

    return path_map, value_map, path_only_map, uuid_path_map, uuid_value_map


def add_path_only_references(
    path_only_map: dict[PathTuple, list[Candidate]],
    refs_zh_only: list[Path],
) -> None:
    for ref_zh in refs_zh_only:
        ref_zh_data = load_json(ref_zh)
        source_label = f"zh-only:{ref_zh.name}"
        for path, zh_text in iter_string_leaves(ref_zh_data):
            if zh_text.strip() == "":
                continue
            candidate = Candidate(
                zh=zh_text,
                source_en="",
                source_path=path,
                source_label=source_label,
            )
            path_only_map.setdefault(path, []).append(candidate)


def process_target_pair(
    target_en: Path,
    target_zh: Path,
    *,
    path_map: dict[tuple[PathTuple, str], list[Candidate]],
    value_map: dict[str, list[Candidate]],
    path_only_map: dict[PathTuple, list[Candidate]],
    uuid_path_map: dict[tuple[PathTuple, str], list[Candidate]],
    uuid_value_map: dict[str, list[Candidate]],
    overwrite_existing: bool,
    use_path_match: bool,
    use_value_match: bool,
    use_path_only_match: bool,
    use_uuid_safe_match: bool,
    check_placeholders: bool,
    rewrite_compendium: bool,
    skip_code_like: bool,
    preferred_source_files: set[str],
    max_samples: int,
) -> tuple[Any, FileStats]:
    target_en_data = load_json(target_en)
    # If target zh file is missing, start from EN so every key remains present.
    target_zh_data = load_json(target_zh) if target_zh.exists() else copy.deepcopy(target_en_data)
    target_zh_out = copy.deepcopy(target_zh_data)

    stats = FileStats(target_en=str(target_en), target_zh=str(target_zh))
    target_name_tokens = stem_tokens(target_en.name)

    for path, en_text in iter_string_leaves(target_en_data):
        stats.total_strings += 1
        current_zh = get_at_path(target_zh_out, path)

        is_code_like = is_code_like_text(en_text)
        has_uuid = has_uuid_link(en_text)

        if skip_code_like and is_code_like:
            is_system_link_like = has_system_link_like(en_text)
            allow_uuid_safe = use_uuid_safe_match and has_uuid
            allow_compendium_safe = rewrite_compendium and has_compendium_link(en_text)
            if is_system_link_like and not (allow_uuid_safe or allow_compendium_safe):
                stats.skipped_code_like += 1
                continue

        if not overwrite_existing and not is_untranslated(current_zh, en_text):
            stats.skipped_existing += 1
            continue

        stats.eligible_strings += 1
        normalized_en = normalize_source_text(en_text).lower()

        candidate: Candidate | None = None
        method = ""
        found_ambiguous = False

        if use_uuid_safe_match and has_uuid:
            uuid_normalized_en = normalize_uuid_source_text(en_text).lower()

            if use_path_match:
                uuid_path_candidates = uuid_path_map.get((path, uuid_normalized_en), [])
                picked, ambiguous = pick_unique(
                    uuid_path_candidates,
                    target_name_tokens=target_name_tokens,
                    preferred_source_files=preferred_source_files,
                )
                if picked is not None:
                    candidate = picked
                    method = "uuid-path+masked-en"
                found_ambiguous = found_ambiguous or ambiguous

            if candidate is None and use_value_match:
                uuid_value_candidates = uuid_value_map.get(uuid_normalized_en, [])
                picked, ambiguous = pick_unique(
                    uuid_value_candidates,
                    target_name_tokens=target_name_tokens,
                    preferred_source_files=preferred_source_files,
                )
                if picked is not None:
                    candidate = picked
                    method = "uuid-masked-en"
                found_ambiguous = found_ambiguous or ambiguous

        if candidate is None and use_path_match:
            path_candidates = path_map.get((path, normalized_en), [])
            picked, ambiguous = pick_unique(
                path_candidates,
                target_name_tokens=target_name_tokens,
                preferred_source_files=preferred_source_files,
            )
            if picked is not None:
                candidate = picked
                method = "path+normalized-en"
            found_ambiguous = found_ambiguous or ambiguous

        if candidate is None and use_value_match:
            value_candidates = value_map.get(normalized_en, [])
            picked, ambiguous = pick_unique(
                value_candidates,
                target_name_tokens=target_name_tokens,
                preferred_source_files=preferred_source_files,
            )
            if picked is not None:
                candidate = picked
                method = "normalized-en"
            found_ambiguous = found_ambiguous or ambiguous

        if candidate is None and use_path_only_match:
            path_only_candidates = path_only_map.get(path, [])
            picked, ambiguous = pick_unique(
                path_only_candidates,
                target_name_tokens=target_name_tokens,
                preferred_source_files=preferred_source_files,
            )
            if picked is not None:
                candidate = picked
                method = "path-only-zh"
            found_ambiguous = found_ambiguous or ambiguous

        if candidate is None:
            if found_ambiguous:
                stats.skipped_ambiguous += 1
            else:
                stats.skipped_no_match += 1
            continue

        replaced_zh = candidate.zh
        if rewrite_compendium and candidate.source_en:
            replaced_zh = rewrite_compendium_systems(replaced_zh, candidate.source_en, en_text)

        if method.startswith("uuid-"):
            rewritten = rewrite_uuid_targets_from_target_en(replaced_zh, en_text)
            if rewritten is None:
                stats.skipped_uuid_mismatch += 1
                continue
            replaced_zh = rewritten

        if check_placeholders and placeholder_tokens(replaced_zh) != placeholder_tokens(en_text):
            stats.skipped_placeholder += 1
            continue

        if isinstance(current_zh, str) and current_zh == replaced_zh:
            stats.unchanged_same_text += 1
            continue

        ok = set_at_path(target_zh_out, path, replaced_zh)
        if not ok:
            # Target zh root structure is unexpected (not dict/list). Skip safely.
            stats.skipped_no_match += 1
            continue

        if method in {"path+normalized-en", "uuid-path+masked-en"}:
            stats.replaced_path_match += 1
            if method == "uuid-path+masked-en":
                stats.replaced_uuid_path_match += 1
        elif method == "path-only-zh":
            stats.replaced_path_only += 1
        else:
            stats.replaced_value_match += 1
            if method == "uuid-masked-en":
                stats.replaced_uuid_value_match += 1

        if len(stats.samples) < max_samples:
            stats.samples.append(
                {
                    "path": path_to_text(path),
                    "method": method,
                    "source": candidate.source_label,
                }
            )

    return target_zh_out, stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cross-compare EN/ZH JSON pairs and reuse translations into target files."
    )
    parser.add_argument(
        "--ref",
        action="append",
        nargs=2,
        metavar=("REF_EN", "REF_ZH"),
        default=[],
        help="Reference EN/ZH pair (repeatable)",
    )
    parser.add_argument(
        "--ref-zh-only",
        action="append",
        default=[],
        metavar="REF_ZH",
        help="Reference ZH-only file for path-only fallback matching (repeatable)",
    )
    parser.add_argument(
        "--target",
        action="append",
        nargs=2,
        metavar=("TARGET_EN", "TARGET_ZH"),
        default=[],
        help="Target EN/ZH pair to update (repeatable)",
    )
    parser.add_argument(
        "--map",
        action="append",
        nargs=3,
        default=[],
        metavar=("SF2_EN", "PF2_ZH", "SF2_ZH"),
        help=(
            "Direct mapping triple: use SF2 EN + PF2 ZH as reference and write into SF2 ZH "
            "(repeatable)"
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write outputs. Without this flag, script runs in preview mode.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite target zh files directly (requires --apply)",
    )
    parser.add_argument(
        "--suffix",
        default=".reused",
        help="Output suffix when --apply is used without --in-place (default: .reused)",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Also replace already translated target strings",
    )
    parser.add_argument(
        "--disable-path-match",
        action="store_true",
        help="Disable path+normalized-en matching",
    )
    parser.add_argument(
        "--disable-value-match",
        action="store_true",
        help="Disable normalized-en-only matching",
    )
    parser.add_argument(
        "--no-placeholder-check",
        action="store_true",
        help="Do not validate {placeholders} between target EN and replacement ZH",
    )
    parser.add_argument(
        "--no-rewrite-compendium",
        action="store_true",
        help="Do not auto-rewrite Compendium.<system>. prefixes from source to target",
    )
    parser.add_argument(
        "--allow-code-like-match",
        action="store_true",
        help=(
            "Allow matching/replacing strings containing code-like markers (e.g. @UUID, [[/act], "
            "Compendium.). By default, these entries are preserved from target content."
        ),
    )
    parser.add_argument(
        "--uuid-safe-match",
        action="store_true",
        help=(
            "Enable conservative @UUID handling: match by text with UUID targets masked, then keep "
            "target @UUID[...] identifiers while applying translated text."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=30,
        help="Max replacement samples to print per target file (default: 30)",
    )
    parser.add_argument(
        "--report",
        default="",
        help="Optional report JSON path",
    )
    return parser


def resolve_pairs(raw_pairs: list[list[str]]) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for pair in raw_pairs:
        en_path = Path(pair[0]).resolve()
        zh_path = Path(pair[1]).resolve()
        if not en_path.is_file():
            raise FileNotFoundError(f"Reference/target EN file not found: {en_path}")
        pairs.append((en_path, zh_path))
    return pairs


def resolve_zh_only_refs(raw_paths: list[str]) -> list[Path]:
    refs: list[Path] = []
    for raw in raw_paths:
        path = Path(raw).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"ZH-only reference file not found: {path}")
        refs.append(path)
    return refs


def resolve_map_triples(raw_maps: list[list[str]]) -> list[tuple[Path, Path, Path]]:
    triples: list[tuple[Path, Path, Path]] = []
    for raw in raw_maps:
        sf2_en = Path(raw[0]).resolve()
        pf2_zh = Path(raw[1]).resolve()
        sf2_zh = Path(raw[2]).resolve()

        if not sf2_en.is_file():
            raise FileNotFoundError(f"SF2 EN file not found: {sf2_en}")
        if not pf2_zh.is_file():
            raise FileNotFoundError(f"PF2 ZH file not found: {pf2_zh}")

        triples.append((sf2_en, pf2_zh, sf2_zh))
    return triples


def output_path_for_target(target_zh: Path, suffix: str) -> Path:
    return target_zh.with_name(f"{target_zh.stem}{suffix}{target_zh.suffix}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.in_place and not args.apply:
        parser.error("--in-place requires --apply")

    if args.disable_path_match and args.disable_value_match:
        parser.error("At least one matching strategy must be enabled")

    ref_pairs = resolve_pairs(args.ref)
    refs_zh_only = resolve_zh_only_refs(args.ref_zh_only)
    target_pairs = resolve_pairs(args.target)
    mapped_triples = resolve_map_triples(args.map)

    if mapped_triples:
        ref_pairs.extend((sf2_en, pf2_zh) for sf2_en, pf2_zh, _ in mapped_triples)
        target_pairs.extend((sf2_en, sf2_zh) for sf2_en, _, sf2_zh in mapped_triples)

    if not target_pairs:
        parser.error("Provide at least one --target pair or --map triple")
    if not ref_pairs and not refs_zh_only:
        parser.error("Provide at least one --ref pair, --ref-zh-only file, or --map triple")

    print(f"Reference pairs: {len(ref_pairs)}")
    print(f"ZH-only references: {len(refs_zh_only)}")
    print(f"Mapped triples: {len(mapped_triples)}")
    print(f"Target pairs: {len(target_pairs)}")

    path_map, value_map, path_only_map, uuid_path_map, uuid_value_map = build_reference_maps(
        ref_pairs,
        skip_code_like=not bool(args.allow_code_like_match),
        enable_uuid_safe_match=bool(args.uuid_safe_match),
    )
    add_path_only_references(path_only_map, refs_zh_only)
    print(f"Reference entries (path+normalized-en): {len(path_map)}")
    print(f"Reference entries (normalized-en): {len(value_map)}")
    print(f"Reference entries (path-only-zh): {len(path_only_map)}")
    print(f"Reference entries (uuid path+masked-en): {len(uuid_path_map)}")
    print(f"Reference entries (uuid masked-en): {len(uuid_value_map)}")

    all_stats: list[FileStats] = []

    for target_en, target_zh in target_pairs:
        preferred_source_files = preferred_source_files_for_target(target_en, ref_pairs)
        out_data, stats = process_target_pair(
            target_en,
            target_zh,
            path_map=path_map,
            value_map=value_map,
            path_only_map=path_only_map,
            uuid_path_map=uuid_path_map,
            uuid_value_map=uuid_value_map,
            overwrite_existing=bool(args.overwrite_existing),
            use_path_match=not bool(args.disable_path_match),
            use_value_match=not bool(args.disable_value_match),
            use_path_only_match=bool(path_only_map),
            use_uuid_safe_match=bool(args.uuid_safe_match),
            check_placeholders=not bool(args.no_placeholder_check),
            rewrite_compendium=not bool(args.no_rewrite_compendium),
            skip_code_like=not bool(args.allow_code_like_match),
            preferred_source_files=preferred_source_files,
            max_samples=max(0, int(args.max_samples)),
        )

        if args.apply:
            out_path = target_zh if args.in_place else output_path_for_target(target_zh, args.suffix)
            dump_json(out_path, out_data)
            stats.output_path = str(out_path)

        all_stats.append(stats)

        print("-" * 80)
        print(f"Target EN: {target_en}")
        print(f"Target ZH: {target_zh}")
        print(
            "Stats: "
            f"total={stats.total_strings}, "
            f"eligible={stats.eligible_strings}, "
            f"replaced(path)={stats.replaced_path_match}, "
            f"replaced(value)={stats.replaced_value_match}, "
            f"replaced(path-only)={stats.replaced_path_only}, "
            f"replaced(uuid-path)={stats.replaced_uuid_path_match}, "
            f"replaced(uuid-value)={stats.replaced_uuid_value_match}, "
            f"skipped_existing={stats.skipped_existing}, "
            f"skipped_no_match={stats.skipped_no_match}, "
            f"skipped_ambiguous={stats.skipped_ambiguous}, "
            f"skipped_placeholder={stats.skipped_placeholder}, "
            f"skipped_uuid_mismatch={stats.skipped_uuid_mismatch}, "
            f"skipped_code_like={stats.skipped_code_like}, "
            f"unchanged={stats.unchanged_same_text}"
        )
        if stats.output_path:
            print(f"Output: {stats.output_path}")
        for sample in stats.samples:
            print(
                "  - "
                f"{sample['path']} "
                f"[{sample['method']}] "
                f"from {sample['source']}"
            )

    replaced_total = sum(
        s.replaced_path_match + s.replaced_value_match + s.replaced_path_only for s in all_stats
    )
    print("=" * 80)
    print(
        "Summary: "
        f"targets={len(all_stats)}, replaced_total={replaced_total}, "
        f"mode={'apply' if args.apply else 'preview'}"
    )
    if not args.apply:
        print("Preview mode only. Add --apply to write output files.")

    if args.report:
        report_path = Path(args.report).resolve()
        report = {
            "reference_pairs": [[str(en), str(zh)] for en, zh in ref_pairs],
            "reference_zh_only": [str(path) for path in refs_zh_only],
            "mapped_triples": [
                [str(sf2_en), str(pf2_zh), str(sf2_zh)]
                for sf2_en, pf2_zh, sf2_zh in mapped_triples
            ],
            "target_pairs": [[str(en), str(zh)] for en, zh in target_pairs],
            "settings": {
                "apply": bool(args.apply),
                "in_place": bool(args.in_place),
                "suffix": args.suffix,
                "overwrite_existing": bool(args.overwrite_existing),
                "use_path_match": not bool(args.disable_path_match),
                "use_value_match": not bool(args.disable_value_match),
                "check_placeholders": not bool(args.no_placeholder_check),
                "rewrite_compendium": not bool(args.no_rewrite_compendium),
                "skip_code_like": not bool(args.allow_code_like_match),
                "use_uuid_safe_match": bool(args.uuid_safe_match),
            },
            "stats": [
                {
                    "target_en": s.target_en,
                    "target_zh": s.target_zh,
                    "total_strings": s.total_strings,
                    "eligible_strings": s.eligible_strings,
                    "replaced_path_match": s.replaced_path_match,
                    "replaced_value_match": s.replaced_value_match,
                    "replaced_path_only": s.replaced_path_only,
                    "replaced_uuid_path_match": s.replaced_uuid_path_match,
                    "replaced_uuid_value_match": s.replaced_uuid_value_match,
                    "skipped_existing": s.skipped_existing,
                    "skipped_no_match": s.skipped_no_match,
                    "skipped_ambiguous": s.skipped_ambiguous,
                    "skipped_placeholder": s.skipped_placeholder,
                    "skipped_uuid_mismatch": s.skipped_uuid_mismatch,
                    "skipped_code_like": s.skipped_code_like,
                    "unchanged_same_text": s.unchanged_same_text,
                    "output_path": s.output_path,
                    "samples": s.samples,
                }
                for s in all_stats
            ],
        }
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
