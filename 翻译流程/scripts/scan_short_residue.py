"""Tighter scan to catch short English item names previously missed (en >= 3 threshold).

Usage: python scan_short_residue.py <target_dir>
       e.g. python scan_short_residue.py gracye/
"""
import json
import os
import re
import sys
from pathlib import Path

SKIP_DIR_TOKENS = ("_backup", "_tmp", "_cache", "_qa_reports", "_pdf", "_zh_synthetic", "NEW")

SKIP_PATH_PATTERNS = (
    '/_id', '/_stats', '/src', '/img', '/icon', '/sort',
    '/flags/pdftofoundry', '/flags/babele', '/flags/core',
    '/ownership', '/sourceId', '/system/source',
    '/macro', '/command',
    '/mapping/',
    '/data/placeHolder',
    '/folderId',
)


def discover_json_files(target_path):
    files = []
    for root, dirs, names in os.walk(target_path):
        if any(tok in root for tok in SKIP_DIR_TOKENS):
            continue
        if os.path.basename(root) == 'en':  # skip Babele source-language dir
            continue
        for n in names:
            if n.endswith(".json"):
                files.append(Path(root) / n)
    return sorted(files)


def walk(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, f"{path}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, f"{path}[{i}]")
    elif isinstance(obj, str):
        yield (path, obj)


def should_skip(p):
    return any(pat in p for pat in SKIP_PATH_PATTERNS)


def is_short_english_only(v):
    """True if string is short, has English letters, no Chinese, AFTER stripping FVTT tokens."""
    if not isinstance(v, str) or not v.strip():
        return False
    if any('一' <= c <= '鿿' for c in v):
        return False
    # Strip FVTT runtime tokens (@Localize/@Check/@Damage all auto-render)
    stripped = re.sub(r'@(Localize|Check|Damage|Template|Compendium|UUID)\[[^\]]+\](\{[^}]*\})?', '', v)
    stripped = re.sub(r'\[\[/[^\]]+\]\](\{[^}]*\})?', '', stripped)
    stripped = re.sub(r'<[^>]+>', '', stripped)
    stripped = re.sub(r'&\w+;', '', stripped).strip()
    # Skip pure DC/stat strings
    if re.match(r'^(DC|AC|HP)\s*\d+\s*$', stripped):
        return False
    # Skip if no real letters left
    has_eng = any(c.isalpha() and ord(c) < 128 for c in stripped)
    if not has_eng:
        return False
    # Need at least 2 alpha chars
    alpha_count = sum(1 for c in stripped if c.isalpha() and ord(c) < 128)
    return alpha_count >= 2


def main():
    if len(sys.argv) < 2:
        print("Usage: python scan_short_residue.py <target_dir>")
        sys.exit(1)
    target_path = Path(sys.argv[1])
    if not target_path.is_dir():
        print(f"Target dir not found: {target_path}")
        sys.exit(1)

    json_files = discover_json_files(target_path)
    if not json_files:
        print(f"No .json files found under {target_path}")
        sys.exit(0)

    out = []
    total = 0
    for path in json_files:
        fname = str(path.relative_to(target_path))
        try:
            data = json.load(open(path, encoding="utf-8"))
        except Exception as e:
            out.append(f"  parse fail: {fname}: {e}")
            continue
        misses = []
        for p, value in walk(data):
            if should_skip(p):
                continue
            if not is_short_english_only(value):
                continue
            # Only flag if path looks user-visible
            if any(x in p for x in [
                "/items/", "/actors/", "/folders/", "/blurb",
                "/publicNotes", "/privateNotes", "/disable", "/routine",
                "/ac", "/hp", "/speed", "/allSaves", "/stealthdetails",
                "/hazardDescription", "/scenes/", "/journals/", "/notes/", "/pages/",
                "/drawings/", "/entries/",
            ]):
                # exclude pure numeric / IDs
                if re.match(r'^[A-Z0-9_\-\.]+$', value) and len(value) < 30:
                    continue
                misses.append((p, value))
        out.append("=" * 78)
        out.append(f"{fname}: {len(misses)} short English residues")
        total += len(misses)
        for p, v in misses[:30]:
            out.append(f"  {p[:80]}: {v[:80]}")
        if len(misses) > 30:
            out.append(f"  ... {len(misses)-30} more")
    out.append("")
    out.append(f"TOTAL: {total}")
    report_path = Path("_tmp_tight_scan_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"Total short English residues: {total}")
    print(f"Detail: {report_path}")


if __name__ == "__main__":
    main()
