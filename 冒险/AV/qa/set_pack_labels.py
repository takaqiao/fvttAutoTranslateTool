"""set_pack_labels.py - set each pack file's top-level `label` from a mapping.

The label is what Babele shows for the pack in the compendium sidebar, and
`regen-labels-titles.py` copies it straight into `labels.json`. A pack file created
mid-pipeline inherits the English label from the baseline, so this runs last.

Usage:
  python set_pack_labels.py --labels _pack_labels.json --cn-dir <dir> [--write]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--cn-dir", required=True, type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)

    mapping = {k: v for k, v in json.loads(args.labels.read_text(encoding="utf-8")).items()
               if not k.startswith("_")}
    changed = missing = same = 0
    for collection_id, label in mapping.items():
        path = args.cn_dir / f"{collection_id}.json"
        if not path.exists():
            print(f"  MISSING {collection_id}")
            missing += 1
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("label") == label:
            same += 1
            continue
        print(f"  {collection_id[:56]:<56} {data.get('label')!r} -> {label!r}")
        data["label"] = label
        changed += 1
        if args.write:
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8", newline="\n")
    print(f"\nchanged {changed}, already correct {same}, missing {missing}")
    if not args.write:
        print("(dry run; pass --write)")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
