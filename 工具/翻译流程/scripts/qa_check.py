"""Final QA on a directory of FVTT zh-CN JSON files: JSON validity + enricher counts.

Usage:
  python qa_check.py <target_dir> [old_dir]

If <old_dir> is given, compares NEW (target_dir) vs OLD (old_dir) for the same filenames
and prints diff in UUID / enricher / HTML tag / total-string counts. Useful after merges
to confirm enricher integrity preserved.

Without [old_dir], just validates JSON and reports per-file counts.
"""
import json
import os
import re
import sys
from pathlib import Path

SKIP_DIR_TOKENS = ("_backup", "_tmp", "_cache", "_qa_reports", "_pdf", "_zh_synthetic", "NEW")


def discover_json_files(target_path):
    files = []
    for root, dirs, names in os.walk(target_path):
        if any(tok in root for tok in SKIP_DIR_TOKENS):
            continue
        for n in names:
            if n.endswith(".json"):
                files.append(Path(root) / n)
    return sorted(files)


def count_uuids(text):
    return len(re.findall(r'@UUID\[[^\]]+\]', text))


def count_checks(text):
    return len(re.findall(r'@(Check|Damage|Template|Localize)\[[^\]]+\]', text))


def count_html_tags(text):
    return len(re.findall(r'<[^>]+>', text))


def walk(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk(v)
    elif isinstance(obj, str):
        yield obj


def aggregate(file_path):
    data = json.load(open(file_path, encoding="utf-8"))
    uuids = checks = html = total = 0
    for s in walk(data):
        uuids += count_uuids(s)
        checks += count_checks(s)
        html += count_html_tags(s)
        total += 1
    return uuids, checks, html, total


def main():
    if len(sys.argv) < 2:
        print("Usage: python qa_check.py <target_dir> [old_dir]")
        sys.exit(1)
    target_path = Path(sys.argv[1])
    old_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    if not target_path.is_dir():
        print(f"Target dir not found: {target_path}")
        sys.exit(1)

    json_files = discover_json_files(target_path)
    if not json_files:
        print(f"No .json files found under {target_path}")
        sys.exit(0)

    print("=" * 70)
    print(f"JSON validity + enricher count QA: {target_path}")
    if old_path:
        print(f"Comparing against OLD: {old_path}")
    print("=" * 70)

    for path in json_files:
        fname = str(path.relative_to(target_path))
        try:
            json.load(open(path, encoding="utf-8"))
            valid = "OK"
        except Exception as e:
            valid = f"FAIL: {e}"
        n_uuids, n_checks, n_html, n_total = aggregate(path)
        print(f"\n{fname}")
        print(f"  JSON valid: {valid}")
        if old_path:
            old_file = old_path / fname
            if old_file.exists():
                o_uuids, o_checks, o_html, o_total = aggregate(old_file)
                print(f"  UUIDs:    NEW {n_uuids}  vs OLD {o_uuids}  (diff: {n_uuids-o_uuids:+d})")
                print(f"  Enrichers: NEW {n_checks}  vs OLD {o_checks}  (diff: {n_checks-o_checks:+d})")
                print(f"  HTML tags: NEW {n_html}  vs OLD {o_html}  (diff: {n_html-o_html:+d})")
                print(f"  Total strings: NEW {n_total}  vs OLD {o_total}  (diff: {n_total-o_total:+d})")
            else:
                print(f"  (no OLD counterpart at {old_file})")
                print(f"  UUIDs={n_uuids}  Enrichers={n_checks}  HTML={n_html}  Total={n_total}")
        else:
            print(f"  UUIDs={n_uuids}  Enrichers={n_checks}  HTML={n_html}  Total={n_total}")


if __name__ == "__main__":
    main()
