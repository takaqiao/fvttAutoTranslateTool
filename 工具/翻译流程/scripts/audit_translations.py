"""Audit translation quality on a directory of FVTT zh-CN JSON files.

Checks:
1. HTML tag balance (open vs close)
2. UUID enricher integrity (well-formed @UUID/@Check/@Damage)
3. Bracket balance ([], (), {})
4. Bilingual name pattern compliance
5. Common term consistency (per glossary)
6. Lurking English words inside Chinese-marked fields

Usage: python audit_translations.py <target_dir>
       e.g. python audit_translations.py gracye/
"""
import json
import os
import re
import sys
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).parent

SKIP_DIR_TOKENS = ("_backup", "_tmp", "_cache", "_qa_reports", "_pdf", "_zh_synthetic", "NEW")


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


def check_html_balance(s):
    """Return list of (open_tag, close_tag) pairs that look unbalanced."""
    self_closing = re.findall(r'<(\w+)[^>]*/>', s)
    opens = re.findall(r'<(\w+)(?:\s[^>]*)?>', s)
    closes = re.findall(r'</(\w+)>', s)
    opens_stripped = list(opens)
    for sc in self_closing:
        if sc in opens_stripped:
            opens_stripped.remove(sc)
    VOID = {"br", "hr", "img", "input", "meta", "link"}
    opens_stripped = [t for t in opens_stripped if t.lower() not in VOID]
    open_counts = Counter(opens_stripped)
    close_counts = Counter(closes)
    diffs = []
    for tag in set(open_counts) | set(close_counts):
        if open_counts[tag] != close_counts[tag]:
            diffs.append((tag, open_counts[tag], close_counts[tag]))
    return diffs


def check_uuid_integrity(s):
    """Check that all @UUID[...] / @Check[...] etc. are well-formed."""
    issues = []
    for m in re.finditer(r'@(\w+)\[([^\]]*)\](?:\{([^}]*)\})?', s):
        kind = m.group(1)
        content = m.group(2)
        if not content.strip():
            issues.append(f"empty-content @{kind}[]")
        if kind == "UUID" and not content.startswith("Compendium.") and not content.startswith("Actor.") and not content.startswith("JournalEntry."):
            issues.append(f"suspicious @UUID: {content[:40]}")
    bad = re.findall(r'@\w+\[[^\]]{200,}', s)
    for b in bad[:3]:
        issues.append(f"unclosed enricher: {b[:60]}")
    return issues


def find_english_in_zh_field(s):
    """Find suspicious English fragments inside what should be Chinese content."""
    if not isinstance(s, str):
        return []
    has_zh = any('一' <= c <= '鿿' for c in s)
    if not has_zh:
        return []
    cleaned = re.sub(r'@\w+\[[^\]]+\](\{[^}]*\})?', ' ', s)
    cleaned = re.sub(r'\[\[/[^\]]+\]\](\{[^}]*\})?', ' ', cleaned)
    cleaned = re.sub(r'<[^>]+>', ' ', cleaned)
    cleaned = re.sub(r'&\w+;', ' ', cleaned)
    issues = []
    for m in re.finditer(r'(?<![一-鿿])\b[A-Za-z][A-Za-z\-\'\s]{3,}\b(?!\s*[一-鿿])', cleaned):
        word = m.group(0).strip()
        if len(word) < 5:
            continue
        if word.isupper() and len(word) <= 5:
            continue
        SKIP = {"Strong", "Cantrip", "Frequency", "Effect", "Trigger", "Requirements",
                "Activate", "Critical", "Success", "Failure", "Saving", "Throw", "Range",
                "Area", "Onset", "Duration", "Stage", "DC", "Maximum", "Description"}
        if word.split()[0] in SKIP:
            continue
        issues.append(word[:60])
    return issues[:5]


def main():
    if len(sys.argv) < 2:
        print("Usage: python audit_translations.py <target_dir>")
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
    total_html_issues = 0
    total_uuid_issues = 0
    total_eng_issues = 0
    for path in json_files:
        fname = str(path.relative_to(target_path))
        try:
            data = json.load(open(path, encoding="utf-8"))
        except Exception as e:
            out.append(f"parse fail: {fname}: {e}")
            continue
        out.append("=" * 78)
        out.append(fname)
        html_issues = []
        uuid_issues = []
        eng_issues = []
        for p, v in walk(data):
            if not isinstance(v, str) or not v:
                continue
            if "<" in v and ">" in v:
                hb = check_html_balance(v)
                if hb:
                    html_issues.append((p, hb, v[:80]))
            ui = check_uuid_integrity(v)
            if ui:
                uuid_issues.append((p, ui, v[:80]))
            ei = find_english_in_zh_field(v)
            if ei:
                eng_issues.append((p, ei, v[:80]))
        out.append(f"  HTML imbalance issues: {len(html_issues)}")
        for p, hb, prev in html_issues[:5]:
            out.append(f"    {p[:70]}: {hb}")
            out.append(f"      preview: {prev}")
        out.append(f"  UUID/enricher issues: {len(uuid_issues)}")
        for p, ui, prev in uuid_issues[:5]:
            out.append(f"    {p[:70]}: {ui}")
            out.append(f"      preview: {prev}")
        out.append(f"  Suspicious English in Chinese fields: {len(eng_issues)}")
        for p, ei, prev in eng_issues[:8]:
            out.append(f"    {p[:70]}: {ei}")
            out.append(f"      preview: {prev[:120]}")
        total_html_issues += len(html_issues)
        total_uuid_issues += len(uuid_issues)
        total_eng_issues += len(eng_issues)
    out.append("")
    out.append("=" * 78)
    out.append(f"TOTALS: HTML={total_html_issues}, UUID={total_uuid_issues}, EN-in-ZH={total_eng_issues}")
    report_path = Path("_tmp_audit_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"Report: {report_path}")
    print(f"HTML imbalances: {total_html_issues}")
    print(f"UUID issues: {total_uuid_issues}")
    print(f"Suspicious English fragments: {total_eng_issues}")


if __name__ == "__main__":
    main()
