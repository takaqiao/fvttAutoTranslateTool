"""repair_split_keys.py - rejoin translation keys that a tool split on '.'.

Foundry document names contain dots: `02. Drop into Darkness`, `A02 - Drawbridge Collapse`,
`Dr. Quagmire`. A tool that addresses leaves by a dotted path and then rebuilds the tree by
splitting that path turns one key into two nested ones:

    notes: { "02": { " Drop into Darkness": {...} } }      instead of
    notes: { "02. Drop into Darkness": {...} }

Babele matches keys against document names, so the split form matches nothing. It is not a
visible error - the affected notes simply render in English, and the file still looks full.

The English baseline decides: a split branch is only rejoined when the rejoined key exists
there, which proves the dot belonged inside the name rather than between two real keys.

Usage:
  python repair_split_keys.py --cn-dir <dir> --en-dir <dir> [--write] [--report out.json]
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path


def walk_keys(node, path=()):
    if isinstance(node, dict):
        yield path, node
        for key, value in node.items():
            yield from walk_keys(value, path + (key,))


def english_keysets(en_data):
    """path-of-parent -> set of keys the English has there."""
    out = {}
    for path, node in walk_keys(en_data.get("entries", {})):
        out[path] = set(node.keys())
    return out


def repair(node, path, en_sets, fixes):
    """Depth-first; rejoin `A` + `.` + `B` when the English has `A.B` at this level."""
    if not isinstance(node, dict):
        return node
    node = {k: repair(v, path + (k,), en_sets, fixes) for k, v in node.items()}
    here = en_sets.get(path)
    if not here:
        return node
    out = dict(node)
    for key, value in list(node.items()):
        if key in here or not isinstance(value, dict):
            continue
        for sub_key, sub_value in list(value.items()):
            joined = f"{key}.{sub_key}"
            if joined not in here:
                continue
            # The dot belonged inside the name. Move the branch to the joined key,
            # without overwriting a correct key that already exists there.
            target = out.setdefault(joined, {})
            if isinstance(target, dict) and isinstance(sub_value, dict):
                for k2, v2 in sub_value.items():
                    target.setdefault(k2, v2)
            elif not target:
                out[joined] = sub_value
            del out[key][sub_key]
            fixes.append({"path": ".".join(path), "from": f"{key} > {sub_key}", "to": joined})
        if isinstance(out.get(key), dict) and not out[key]:
            del out[key]
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cn-dir", required=True, type=Path)
    parser.add_argument("--en-dir", required=True, type=Path)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    all_fixes = []
    for cn_path in sorted(args.cn_dir.glob("*.json")):
        en_path = args.en_dir / cn_path.name
        if not en_path.exists():
            continue
        cn_data = json.loads(cn_path.read_text(encoding="utf-8"))
        en_sets = english_keysets(json.loads(en_path.read_text(encoding="utf-8")))
        fixes = []
        entries = repair(cn_data.get("entries", {}), (), en_sets, fixes)
        if not fixes:
            continue
        for f in fixes:
            f["file"] = cn_path.name
        all_fixes.extend(fixes)
        print(f"[{'write' if args.write else 'dry  '}] {cn_path.name[:52]:<52} rejoined {len(fixes)}")
        for f in fixes[:6]:
            print(f"        {f['from'][:44]:<44} -> {f['to'][:40]}")
        if args.write:
            backup = cn_path.parent.parent / "_backup" / f"splitkeys_{stamp}"
            backup.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cn_path, backup / cn_path.name)
            cn_data["entries"] = entries
            cn_path.write_text(json.dumps(cn_data, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8", newline="\n")

    print(f"\nrejoined {len(all_fixes)} split keys")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(all_fixes, ensure_ascii=False, indent=1),
                               encoding="utf-8")
        print(f"report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
