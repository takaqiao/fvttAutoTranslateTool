"""Scan a directory of FVTT zh-CN JSON files for REAL English residues.

Filters out:
- Empty / metadata strings
- FVTT runtime tokens (@UUID/@Check/@Damage/@Template/@Localize that auto-render)
- Bilingual short strings (≥1 Chinese char + English in <60-char field is OK)
- IDs/paths/flags/mapping schema

Usage: python scan_residue.py <target_dir>
       e.g. python scan_residue.py gracye/
            python scan_residue.py FotRP/需要翻译/
"""
import json
import re
import sys
import os
from pathlib import Path

SKIP_PATH_PATTERNS = (
    '/_id', '/_stats', '/src', '/img', '/icon', '/sort',
    '/flags/pdftofoundry', '/flags/babele', '/flags/core',
    '/ownership', '/sourceId', '/system/source',
    '/macro', '/command',
    '/mapping/',
    '/data/placeHolder',
    '/folderId',
)
# Suffix-anchored skip patterns (must be exact path-end, not substring)
SKIP_PATH_SUFFIXES = (
    '/folder',  # folder ID reference (singular), not folder name
)
# Value patterns that are FVTT internal IDs (not translatable)
FVTT_ID_PATTERN = re.compile(r'^[A-Za-z0-9]{16}$')  # 16-char alphanumeric IDs

STRIP_PATTERNS = [
    re.compile(r'@(UUID|Check|Damage|Template|Localize|Compendium)\[[^\]]+\](\{[^}]*\})?'),
    re.compile(r'\[\[/[^\]]+\]\](\{[^}]*\})?'),
    re.compile(r'<[^>]+>'),
    re.compile(r'&[a-z]+;'),
]


def strip_for_count(s):
    for pat in STRIP_PATTERNS:
        s = pat.sub(' ', s)
    return s


def count_en(s):
    return sum(1 for c in s if c.isalpha() and ord(c) < 128)


def count_zh(s):
    return sum(1 for c in s if '一' <= c <= '鿿')


def walk(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, f"{path}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk(v, f"{path}[{i}]")
    elif isinstance(obj, str):
        yield (path, obj)


def should_skip(path):
    return any(pat in path for pat in SKIP_PATH_PATTERNS)


def is_real_miss(path, value):
    if should_skip(path):
        return False
    # Suffix-anchored skip
    for suf in SKIP_PATH_SUFFIXES:
        if path.endswith(suf):
            return False
    if not value.strip():
        return False
    # FVTT internal ID values
    if FVTT_ID_PATTERN.match(value.strip()):
        return False
    stripped = strip_for_count(value)
    en = count_en(stripped)
    zh = count_zh(stripped)
    # Bilingual short strings (≥1 Chinese char + any English) are FINE - skip
    if len(stripped.strip()) < 60 and zh >= 1:
        return False
    # Substantial English with no Chinese
    if en > 5 and zh < 2:
        return True
    # Long block with little Chinese
    if en > 200 and zh < 20:
        return True
    return False


def main():
    out = []
    by_type = {}  # category -> count
    total = 0
    real_misses_by_file = {}
    if len(sys.argv) < 2:
        print("Usage: python scan_residue.py <target_dir>")
        sys.exit(1)
    target_dir = sys.argv[1]
    target_path = Path(target_dir)
    if not target_path.is_dir():
        print(f"Target dir not found: {target_dir}")
        sys.exit(1)
    # Collect all .json files (skip backup/cache)
    json_files = []
    for root, dirs, files in os.walk(target_path):
        if any(s in root for s in ["NEW", "_backup", "_pdf", "_zh_synthetic", "_qa_reports"]):
            continue
        if os.path.basename(root) == 'en':  # skip Babele source-language dir
            continue
        for f in files:
            if f.endswith(".json"):
                json_files.append(Path(root) / f)
    for path in sorted(json_files):
        fname = str(path.relative_to(target_path))
        try:
            data = json.load(open(path, encoding="utf-8"))
        except Exception as e:
            out.append(f"  parse fail: {fname}: {e}")
            continue
        residues = []
        for p, value in walk(data):
            if is_real_miss(p, value):
                residues.append((p, value))
        out.append("=" * 78)
        out.append(f"{fname}: {len(residues)} residues")
        total += len(residues)
        # Categorize
        cats = {"name_field": 0, "description_field": 0, "folder": 0, "label": 0, "other": 0}
        for p, v in residues:
            if p.endswith("/name"):
                cats["name_field"] += 1
            elif p.endswith("/description"):
                cats["description_field"] += 1
            elif "/folders/" in p:
                cats["folder"] += 1
            elif "/label" in p or p == "/label":
                cats["label"] += 1
            else:
                cats["other"] += 1
        for k, v in cats.items():
            if v:
                out.append(f"  {k}: {v}")
        # Sample misses
        if residues:
            for p, v in residues[:30]:
                preview = v[:100].replace("\n", " ")
                out.append(f"  {p[:80]}")
                out.append(f"    {preview}")
            if len(residues) > 30:
                out.append(f"  ... {len(residues)-30} more")
        real_misses_by_file[fname] = residues

    out.append("")
    out.append(f"TOTAL real English residues: {total}")

    report_path = Path("_tmp_residue_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))

    # Also dump full miss list as JSON for next stage
    miss_dump = {fn: [(p, v) for p, v in lst] for fn, lst in real_misses_by_file.items()}
    json_path = Path("_tmp_residues.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(miss_dump, f, ensure_ascii=False, indent=2)
    print(f"Total real residues: {total}")
    print(f"Detail: {report_path}")
    print(f"Data: {json_path}")


if __name__ == "__main__":
    main()
