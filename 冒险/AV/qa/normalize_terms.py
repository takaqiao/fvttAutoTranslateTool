"""normalize_terms.py - collapse Chinese term variants onto the canonical rendering.

Both sides of every rule are Chinese, so a plain substring replacement cannot reach the
ASCII machine parts of a Babele file: `@UUID[...]` bracket bodies, HTML tags/attributes
and the English half of a `中文 English` name are all ASCII and stay untouched by
construction.  A variant sitting inside a `{label}` or inside a name's Chinese half is
exactly what we DO want fixed, so no path filtering is needed.

Default mode is a report.  Nothing is written without --write.

Handles .json (every string leaf) and .md / .txt (whole file).

Usage:
  python normalize_terms.py --terms <terms.json> --target <path> [--target <path> ...]
                            [--write] [--report out.json]
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import Counter
from pathlib import Path

SKIP_DIRS = {"_backup", "_cache", "_qa_reports", "reports", "units", "units_out", "en", "_源"}


def load_terms(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def apply_to_string(text, terms, counter, where):
    for wrong, right in terms.items():
        if wrong in text:
            counter[wrong] += text.count(wrong)
            text = text.replace(wrong, right)
    return text


def walk_json(node, terms, counter, where):
    if isinstance(node, dict):
        return {k: walk_json(v, terms, counter, where) for k, v in node.items()}
    if isinstance(node, list):
        return [walk_json(v, terms, counter, where) for v in node]
    if isinstance(node, str):
        return apply_to_string(node, terms, counter, where)
    return node


def iter_targets(targets):
    for target in targets:
        target = Path(target)
        if target.is_file():
            yield target
        elif target.is_dir():
            for path in sorted(target.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in {".json", ".md", ".txt"}:
                    continue
                if any(part in SKIP_DIRS for part in path.parts):
                    continue
                yield path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--terms", required=True, type=Path)
    parser.add_argument("--target", action="append", required=True)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--backup-dir", type=Path)
    args = parser.parse_args(argv)

    terms = load_terms(args.terms)
    print(f"{len(terms)} rules: " + ", ".join(f"{k}->{v}" for k, v in list(terms.items())[:6])
          + (" ..." if len(terms) > 6 else "") + "\n")

    grand = Counter()
    per_file = {}
    stamp = time.strftime("%Y%m%d_%H%M%S")

    for path in iter_targets(args.target):
        counter = Counter()
        try:
            if path.suffix.lower() == ".json":
                data = json.loads(path.read_text(encoding="utf-8"))
                out = walk_json(data, terms, counter, path.name)
                new_text = json.dumps(out, ensure_ascii=False, indent=2) + "\n"
            else:
                original = path.read_text(encoding="utf-8")
                new_text = apply_to_string(original, terms, counter, path.name)
        except Exception as exc:
            print(f"  skip {path.name}: {exc}")
            continue
        if not counter:
            continue
        grand.update(counter)
        per_file[str(path)] = dict(counter)
        print(f"[{'write' if args.write else 'dry  '}] {path.name[:60]:<60} {dict(counter)}")
        if args.write:
            backup_root = args.backup_dir or (path.parent / "_backup" / f"terms_{stamp}")
            backup_root.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup_root / path.name)
            path.write_text(new_text, encoding="utf-8", newline="\n")

    print(f"\ntotal replacements: {sum(grand.values())} across {len(per_file)} files")
    for wrong, n in grand.most_common():
        print(f"    {wrong} -> {terms[wrong]}   {n}")
    if not args.write:
        print("\n(dry run; pass --write)")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"terms": terms, "totals": dict(grand), "files": per_file},
                                          ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
