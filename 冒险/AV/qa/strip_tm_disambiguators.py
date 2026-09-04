"""strip_tm_disambiguators.py - remove the wiki's page-title disambiguators from prose.

The Chinese PF2 wiki disambiguates same-named pages by appending a category in full-width
parentheses: `幽灵（特征）` is the *trait* Ghost, as distinct from the creature. That suffix
belongs to a page TITLE. It is not part of the term, and it must never reach running text.

`build_3source_tm.py` harvests those titles, so every autofill and every translator who
looked a term up carries the suffix through:

    CN  典型的幽灵（特征）只能离开它被杀害之处…        EN  A typical ghost can stray only…
    CN  （幽灵（特征））束缚之地 (Ghost) Site Bound      EN  (Ghost) Site Bound
    CN  该生物的惊惧（状态）增加1                      EN  the creature's frightened value increases by 1

None of those have an English counterpart - the English simply says "ghost" and
"frightened". So the suffix is deletable on sight, and the English baseline proves it:
a leaf is only touched when its English does NOT carry a matching `(trait)` / `(condition)`
parenthetical of its own.

Usage:
  python strip_tm_disambiguators.py --cn-dir <dir> [--en-dir <dir>] [--write] [--report r.json]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from collections import Counter
from pathlib import Path

# The categories the wiki actually uses as title disambiguators.
CATEGORIES = ("特征", "状态", "信仰", "职业", "法术", "物品", "动作", "背景",
              "族裔", "变体", "专长", "生物", "装备", "近战", "远程", "仪式")
DISAMB = re.compile("（(?:" + "|".join(CATEGORIES) + ")）")
# The English equivalents, so a leaf that legitimately says "(Trait)" is left alone.
EN_DISAMB = re.compile(r"\((?:trait|condition|deity|class|spell|item|action|background|"
                       r"ancestry|archetype|feat|creature|equipment|ritual)\)", re.I)


def walk(node, path=()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from walk(value, path + (key,))
    elif isinstance(node, str):
        yield path, node


def clean(text):
    """Drop the suffix, then repair the double parentheses it leaves behind.

    `（幽灵（特征））束缚之地` -> `（幽灵）束缚之地`, not `（幽灵））束缚之地`.
    """
    out = DISAMB.sub("", text)
    out = re.sub(r"（\s*）", "", out)
    out = re.sub(r"\(\s*\)", "", out)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cn-dir", required=True, type=Path)
    parser.add_argument("--en-dir", type=Path,
                        help="when given, a leaf whose English carries its own (trait)-style "
                             "parenthetical is skipped and reported")
    parser.add_argument("--also", action="append", default=[],
                        help="extra JSON files that are not Babele packs (a module's own "
                             "lang file, for instance) - they hold prose too")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    grand, rows = Counter(), []
    targets = sorted(args.cn_dir.glob("*.json")) + [Path(a) for a in args.also]

    for cn_path in targets:
        if not cn_path.exists():
            continue
        data = json.loads(cn_path.read_text(encoding="utf-8"))
        root_key = "entries" if "entries" in data else None
        en_flat = {}
        if args.en_dir:
            en_path = args.en_dir / cn_path.name
            if en_path.exists():
                en_data = json.loads(en_path.read_text(encoding="utf-8"))
                en_flat = {".".join(p): v for p, v in
                           walk(en_data.get("entries", en_data))}
        changed = 0

        def fix(node, path=()):
            nonlocal changed
            if isinstance(node, dict):
                return {k: fix(v, path + (k,)) for k, v in node.items()}
            if not isinstance(node, str) or not DISAMB.search(node):
                return node
            english = en_flat.get(".".join(path))
            if english and EN_DISAMB.search(english):
                grand["skip-english-has-one"] += 1
                rows.append({"file": cn_path.name, "path": ".".join(path[-3:]),
                             "why": "english-has-its-own", "cn": node[:160]})
                return node
            new = clean(node)
            if new == node:
                return node
            changed += 1
            grand["cleaned"] += len(DISAMB.findall(node))
            rows.append({"file": cn_path.name, "path": ".".join(path[-3:]),
                         "before": node[:120], "after": new[:120]})
            return new

        # Both sides are walked from *inside* `entries`, so the flattened paths line up.
        root = data[root_key] if root_key else data
        fixed = fix(root)
        if not changed:
            continue
        print(f"[{'write' if args.write else 'dry  '}] {cn_path.name[:52]:<52} leaves {changed}")
        if args.write:
            backup = cn_path.parent.parent / "_backup" / f"disamb_{stamp}"
            backup.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cn_path, backup / cn_path.name)
            if root_key:
                data[root_key] = fixed
            else:
                data = fixed
            cn_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8", newline="\n")

    print(f"\nsuffixes removed: {grand['cleaned']}   skipped (English has one too): "
          f"{grand['skip-english-has-one']}")
    for row in rows[:8]:
        if "before" in row:
            print(f"  {row['file'][:26]:<26} {row['before'][:56]}")
            print(f"  {'':<26} -> {row['after'][:56]}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"stats": dict(grand), "rows": rows},
                                          ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
