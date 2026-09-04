"""clear_leaves.py - blank the leaves a scan condemned, so the units pipeline re-emits them.

A translation that already exists is not automatically a translation worth keeping. Three
kinds show up every time an old corpus is re-run against the current standard:

  * the appended-English kind that no stripper could split
  * the "has some Chinese so it counts as done" kind - an English paragraph whose only
    Chinese is inside an enricher label
  * the wrong-item kind - filled by name from a compendium that had a different item under
    that name, which the bracket-count mismatch against the baseline exposes

`emit_units.py` decides what needs a translator by looking for EMPTY leaves, so the way to
put these back on the work list is to empty them. That is a destructive edit, so it only
ever runs from an explicit list of paths (a report produced by a scan), never from a
predicate evaluated here - the decision and the deletion stay separate steps.

Every cleared leaf is written to the report with its old value, so the edit is reversible
even after the `.bak` files are gone.

Usage:
  python clear_leaves.py --cn-dir <dir> --list <worklist.json> [--why gap,bilingual]
                         [--write] [--report cleared.json]

  <worklist.json> is a list of {"file": "<basename, no .json>", "path": "<dotted>",
  "why": "<reason>"}. --why filters to those reasons (substring match, comma separated).
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import Counter
from pathlib import Path


def get_path(entries, dotted):
    node = entries
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node if isinstance(node, str) else None


def set_path(entries, dotted, value):
    node = entries
    parts = dotted.split(".")
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    if not isinstance(node, dict) or parts[-1] not in node:
        return False
    node[parts[-1]] = value
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cn-dir", type=Path, required=True)
    parser.add_argument("--list", type=Path, required=True)
    parser.add_argument("--why", default="",
                        help="comma separated reason substrings; empty = every row")
    parser.add_argument("--skip-file", action="append", default=[],
                        help="basename (no .json) to leave alone; repeatable")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    wanted = [w for w in args.why.split(",") if w]
    rows = json.loads(args.list.read_text(encoding="utf-8"))
    skip = set(args.skip_file)

    by_file = {}
    for row in rows:
        if row["file"] in skip:
            continue
        if wanted and not any(w in row["why"] for w in wanted):
            continue
        by_file.setdefault(row["file"], []).append(row)

    stats = Counter()
    cleared = []
    for stem, todo in sorted(by_file.items()):
        cn_file = args.cn_dir / f"{stem}.json"
        if not cn_file.exists():
            stats["no-file"] += len(todo)
            continue
        data = json.loads(cn_file.read_text(encoding="utf-8"))
        entries = data.get("entries", {})
        touched = 0
        for row in todo:
            old = get_path(entries, row["path"])
            if old is None:
                stats["not-present"] += 1
                continue
            if not old.strip():
                stats["already-empty"] += 1
                continue
            if set_path(entries, row["path"], ""):
                cleared.append({"file": stem, "path": row["path"],
                                "why": row["why"], "old": old})
                touched += 1
            else:
                stats["path-blocked"] += 1
        stats["cleared"] += touched
        print(f"[{'write' if args.write else 'dry  '}] {stem:52s} cleared={touched}")
        if touched and args.write:
            shutil.copy2(cn_file,
                         cn_file.with_suffix(f".json.bak-{time.strftime('%Y%m%d-%H%M%S')}"))
            cn_file.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")

    print(f"\ncleared {stats['cleared']} leaves"
          f"{'' if args.write else ' (dry run; pass --write)'}")
    for key, count in stats.most_common():
        if key != "cleared":
            print(f"    {key:20s} {count}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(cleared, ensure_ascii=False, indent=2),
                               encoding="utf-8")
        print(f"\nold values ({len(cleared)}) -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
