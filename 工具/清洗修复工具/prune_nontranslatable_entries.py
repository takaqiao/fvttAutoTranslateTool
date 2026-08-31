#!/usr/bin/env python3
"""Prune non-translatable placeholder/path entries from JSON files.

Default behavior removes two kinds of dict entries:
1) Keys and values that are both "invisible" text (control/format chars + whitespace),
   e.g. "\u200e": "\u200e".
2) Keys and values that are identical and look like folder/file path pointers,
   e.g. "modules/pf2e/...": "modules/pf2e/...".

Use dry-run (default) first, then apply with --in-place.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PATH_PREFIX_RE = re.compile(
    r"^(?:[A-Za-z]:\\|\\\\|\./|\.\./|~?/|modules/|systems/|icons/|assets/)",
    flags=re.IGNORECASE,
)
FILE_EXT_RE = re.compile(
    r"\.(?:png|jpg|jpeg|webp|gif|svg|mp3|ogg|wav|m4a|webm|json|txt|md)$",
    flags=re.IGNORECASE,
)
SEPARATOR_RE = re.compile(r"[\\/]+")


@dataclass
class FileResult:
    file_path: Path
    removed_control_only: int = 0
    removed_path_like: int = 0
    changed: bool = False
    error: str = ""

    @property
    def removed_total(self) -> int:
        return self.removed_control_only + self.removed_path_like


def _strip_control_format(text: str) -> str:
    # Cc=control, Cf=format (includes U+200E/U+200F and zero-width markers).
    return "".join(ch for ch in text if unicodedata.category(ch) not in {"Cc", "Cf"})


def _is_invisible_only(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return _strip_control_format(text).strip() == ""


def _is_path_like(text: str, path_min_segments: int) -> bool:
    if not isinstance(text, str):
        return False

    s = text.strip()
    if not s:
        return False

    if PATH_PREFIX_RE.search(s):
        return True
    if FILE_EXT_RE.search(s):
        return True

    if not re.search(r"[\\/]", s):
        return False

    # Keep heuristic conservative: avoid deleting short labels like "A/B".
    if re.search(r"\s", s):
        return False
    segments = [seg for seg in SEPARATOR_RE.split(s) if seg]
    return len(segments) >= max(2, int(path_min_segments))


def _format_path(parts: list[str | int], key: str) -> str:
    out = "root"
    for p in parts:
        if isinstance(p, int):
            out += f"[{p}]"
        else:
            out += f".{p}"
    return f"{out}.{key}"


def _prune_node(
    node: Any,
    parts: list[str | int],
    *,
    remove_control_only: bool,
    remove_path_like_pairs: bool,
    path_min_segments: int,
    samples: list[str],
    sample_limit: int,
    stats: dict[str, int],
) -> None:
    if isinstance(node, dict):
        for key in list(node.keys()):
            value = node.get(key)

            removed_reason = ""
            if isinstance(key, str) and isinstance(value, str):
                if remove_control_only and _is_invisible_only(key) and _is_invisible_only(value):
                    removed_reason = "control_only"
                elif (
                    remove_path_like_pairs
                    and key == value
                    and _is_path_like(key, path_min_segments=path_min_segments)
                ):
                    removed_reason = "path_like_pair"

            if removed_reason:
                del node[key]
                stats[removed_reason] += 1
                if len(samples) < sample_limit:
                    samples.append(f"{removed_reason}: {_format_path(parts, key)} => {value!r}")
                continue

            _prune_node(
                value,
                parts + [key],
                remove_control_only=remove_control_only,
                remove_path_like_pairs=remove_path_like_pairs,
                path_min_segments=path_min_segments,
                samples=samples,
                sample_limit=sample_limit,
                stats=stats,
            )

    elif isinstance(node, list):
        for idx, item in enumerate(node):
            _prune_node(
                item,
                parts + [idx],
                remove_control_only=remove_control_only,
                remove_path_like_pairs=remove_path_like_pairs,
                path_min_segments=path_min_segments,
                samples=samples,
                sample_limit=sample_limit,
                stats=stats,
            )


def _iter_json_files(paths: list[str], glob_pattern: str) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            files.extend(f for f in p.glob(glob_pattern) if f.is_file())
        elif p.is_file():
            files.append(p)
    unique = sorted({f.resolve() for f in files})
    return [Path(x) for x in unique]


def _load_json(path: Path) -> Any:
    # utf-8-sig allows reading files with or without BOM.
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def process_file(
    file_path: Path,
    *,
    in_place: bool,
    remove_control_only: bool,
    remove_path_like_pairs: bool,
    path_min_segments: int,
    sample_limit: int,
) -> tuple[FileResult, list[str]]:
    result = FileResult(file_path=file_path)
    samples: list[str] = []

    try:
        data = _load_json(file_path)
    except Exception as exc:  # noqa: BLE001
        result.error = str(exc)
        return result, samples

    stats = {"control_only": 0, "path_like_pair": 0}
    _prune_node(
        data,
        [],
        remove_control_only=remove_control_only,
        remove_path_like_pairs=remove_path_like_pairs,
        path_min_segments=path_min_segments,
        samples=samples,
        sample_limit=sample_limit,
        stats=stats,
    )

    result.removed_control_only = stats["control_only"]
    result.removed_path_like = stats["path_like_pair"]
    result.changed = result.removed_total > 0

    if result.changed and in_place:
        _write_json(file_path, data)

    return result, samples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Remove non-translatable placeholder/path entries from JSON files."
    )
    parser.add_argument("paths", nargs="+", help="JSON file(s) or directory(ies) to process")
    parser.add_argument(
        "-g",
        "--glob",
        default="**/*.json",
        help="Glob for directory input (default: **/*.json)",
    )
    parser.add_argument(
        "-i",
        "--in-place",
        action="store_true",
        help="Write changes back to file (default: dry-run)",
    )
    parser.add_argument(
        "--no-control-only",
        action="store_true",
        help="Do not remove entries where key/value are invisible-only control/format chars",
    )
    parser.add_argument(
        "--no-path-like",
        action="store_true",
        help="Do not remove path-like equal pairs (key == value)",
    )
    parser.add_argument(
        "--path-min-segments",
        type=int,
        default=3,
        help="Minimum slash-separated segments for generic path heuristic (default: 3)",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=20,
        help="Max removal samples to print per file (default: 20)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    remove_control_only = not bool(args.no_control_only)
    remove_path_like_pairs = not bool(args.no_path_like)

    files = _iter_json_files(args.paths, args.glob)
    if not files:
        print("No JSON files found.")
        return

    print(f"Files to process: {len(files)}")
    print(f"Mode: {'in-place' if args.in_place else 'dry-run'}")
    print(
        "Rules: "
        f"control_only={'on' if remove_control_only else 'off'}, "
        f"path_like_pair={'on' if remove_path_like_pairs else 'off'}"
    )

    total_control = 0
    total_path_like = 0
    total_changed = 0
    total_errors = 0

    for file_path in files:
        result, samples = process_file(
            file_path,
            in_place=args.in_place,
            remove_control_only=remove_control_only,
            remove_path_like_pairs=remove_path_like_pairs,
            path_min_segments=args.path_min_segments,
            sample_limit=args.sample_limit,
        )

        if result.error:
            total_errors += 1
            print(f"ERROR {file_path}: {result.error}")
            continue

        total_control += result.removed_control_only
        total_path_like += result.removed_path_like
        if result.changed:
            total_changed += 1

        print(
            f"{file_path}: removed={result.removed_total} "
            f"(control_only={result.removed_control_only}, path_like={result.removed_path_like})"
        )
        for line in samples:
            print(f"  - {line}")

    print("Done.")
    print(
        f"Summary: files={len(files)}, changed={total_changed}, errors={total_errors}, "
        f"removed_total={total_control + total_path_like}, "
        f"control_only={total_control}, path_like={total_path_like}"
    )
    if not args.in_place:
        print("Dry-run only. Add --in-place to write changes.")


if __name__ == "__main__":
    main()
