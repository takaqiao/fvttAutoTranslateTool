"""normalize_name_format.py - enforce `中文 English` on name-class leaves.

The project's first hard constraint is that a name-class leaf (name / tokenName /
prototypeToken, anything under `folders`, and scene note keys) reads

    中文 English            one ASCII space, no parentheses, no newline

The corpus inherited an older convention, `中文\\nEnglish`, from before that rule -
114 leaves, almost all folder labels.  Foundry renders a folder label on one line, so
the newline shows up as a stray gap or is swallowed entirely depending on the sheet.

Only the separator is touched.  Parentheses that belong to the English name itself
(`Religious Symbol (Wooden)`) and spaces inside the Chinese half (`原木1 上行至原木2`)
are not separators and are left alone - which is why this matches the tail rather than
trying to parse the name.

Usage:
  python normalize_name_format.py --cn-dir <dir> [--write] [--report out.json]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("nb", HERE / "normalize_bilingual.py")
nb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nb)

CJK = re.compile(r"[一-鿿]")
# head (contains CJK) \n tail (pure ASCII, no CJK) at end of string
NEWLINE_SPLIT = re.compile(r"^(.*[一-鿿].*?)\s*\n\s*([\x20-\x7E‘’–—]+)$", re.S)


def fix_value(value):
    m = NEWLINE_SPLIT.match(value)
    if not m:
        return None
    head, tail = m.group(1).strip(), m.group(2).strip()
    if not head or not tail or CJK.search(tail) or "\n" in head:
        return None
    return f"{head} {tail}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cn-dir", required=True, type=Path)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    grand, rows = Counter(), []
    for cn_path in sorted(args.cn_dir.glob("*.json")):
        data = json.loads(cn_path.read_text(encoding="utf-8"))
        changed = 0

        def fix(node, path=()):
            nonlocal changed
            if isinstance(node, dict):
                return {k: fix(v, path + (k,)) for k, v in node.items()}
            if not isinstance(node, str) or not path:
                return node
            if not nb.should_keep_bilingual(path, path[-1]):
                return node
            new = fix_value(node)
            if new is None or new == node:
                return node
            changed += 1
            rows.append({"file": cn_path.name, "path": ".".join(path[-3:]),
                         "from": node, "to": new})
            return new

        entries = fix(data.get("entries", {}), ("entries",))
        if not changed:
            continue
        grand[cn_path.name] = changed
        print(f"[{'write' if args.write else 'dry  '}] {cn_path.name[:52]:<52} {changed}")
        if args.write:
            backup = cn_path.parent.parent / "_backup" / f"nameformat_{stamp}"
            backup.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cn_path, backup / cn_path.name)
            data["entries"] = entries
            cn_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8", newline="\n")

    print(f"\nnewline separators replaced with a space: {sum(grand.values())}")
    for row in rows[:8]:
        print(f"    {row['file'][:30]}:{row['path'][-40:]}  {row['to'][:56]}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"per_file": dict(grand), "rows": rows},
                                          ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
