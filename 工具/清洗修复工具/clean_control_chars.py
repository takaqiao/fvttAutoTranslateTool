#!/usr/bin/env python3
"""Remove hidden control characters from text files.

Removes C0/C1 control characters except for tab, LF, and CR by default.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

# C0: 0x00-0x1F, C1: 0x7F-0x9F
# Keep \t (0x09), \n (0x0A), \r (0x0D) by default.
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]")


def clean_text(text: str) -> str:
    return CONTROL_CHARS_RE.sub("", text)


def process_file(path: Path, in_place: bool) -> bool:
    original = path.read_text(encoding="utf-8")
    cleaned = clean_text(original)
    if cleaned == original:
        return False
    if in_place:
        path.write_text(cleaned, encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove hidden control characters from files.")
    parser.add_argument("paths", nargs="+", help="File(s) or directory(ies) to scan")
    parser.add_argument("-i", "--in-place", action="store_true", help="Modify files in place")
    parser.add_argument(
        "-g",
        "--glob",
        default="**/*.json",
        help="Glob pattern used when a directory is provided (default: **/*.json)",
    )
    args = parser.parse_args()

    changed_files: list[Path] = []
    for p in map(Path, args.paths):
        if p.is_dir():
            for file_path in p.glob(args.glob):
                if file_path.is_file() and process_file(file_path, args.in_place):
                    changed_files.append(file_path)
        elif p.is_file():
            if process_file(p, args.in_place):
                changed_files.append(p)

    if changed_files:
        print("Cleaned:")
        for f in changed_files:
            print(f"- {f}")
    else:
        print("No control characters found.")


if __name__ == "__main__":
    main()
