"""apply_path_patches.py - rewrite one leaf, addressed by its exact path.

`normalize_terms.py` is a plain substring replace, which is the right tool while the
wrong rendering is a string that is wrong *everywhere*.  It cannot express a homograph:
`支撑物 Support` is correct on a shop table (the piece of equipment) and wrong on a
druid's animal companion (the Support action).  Both leaves hold byte-identical text,
so only the path tells them apart.

Each patch names the file, the dotted path from `entries`, the exact expected current
value and the replacement.  A patch whose path is missing, or whose current value is
not what it claims, is an ERROR - never a silent skip: a stale patch that quietly does
nothing is indistinguishable from one that worked.  Already-applied patches (value
already equals `to`) are reported as satisfied and are not errors, so the file is
re-runnable.

Usage:
  python apply_path_patches.py --cn-dir <dir> --patches _path_patches.json [--write]
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path


def get_path(node, parts):
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            return None, False
        node = node[part]
    return node, True


def set_path(node, parts, value):
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cn-dir", required=True, type=Path)
    parser.add_argument("--patches", required=True, type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)

    raw = json.loads(args.patches.read_text(encoding="utf-8"))
    patches = [p for p in raw.get("patches", []) if not str(p.get("file", "")).startswith("_")]

    by_file = {}
    for patch in patches:
        by_file.setdefault(patch["file"], []).append(patch)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    errors, applied, satisfied = [], 0, 0

    for filename, items in sorted(by_file.items()):
        path = args.cn_dir / filename
        if not path.exists():
            errors.append(f"{filename}: file not found")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        changed = 0
        for patch in items:
            parts = patch["path"].split(".") if isinstance(patch["path"], str) else patch["path"]
            current, found = get_path(data.get("entries", {}), parts)
            if not found:
                errors.append(f"{filename}: path not found: {patch['path']}")
                continue
            if current == patch["to"]:
                satisfied += 1
                print(f"  [ok  ] {filename[:34]:<34} {patch['path'][-46:]}  already {patch['to'][:20]}")
                continue
            if current != patch["from"]:
                errors.append(f"{filename}: {patch['path']}\n"
                              f"        expected {patch['from']!r}\n"
                              f"        found    {current!r}")
                continue
            if args.write:
                set_path(data["entries"], parts, patch["to"])
            changed += 1
            applied += 1
            print(f"  [{'write' if args.write else 'dry  '}] {filename[:34]:<34} "
                  f"{patch['path'][-46:]}  {patch['from'][:18]} -> {patch['to'][:18]}")
        if args.write and changed:
            backup = path.parent.parent / "_backup" / f"pathpatch_{stamp}"
            backup.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup / path.name)
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8", newline="\n")

    print(f"\n{len(patches)} patches: {applied} {'applied' if args.write else 'would apply'}, "
          f"{satisfied} already satisfied, {len(errors)} errors")
    for err in errors:
        print(f"  [ERR] {err}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
