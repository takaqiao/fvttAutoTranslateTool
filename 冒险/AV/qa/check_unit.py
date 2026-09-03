"""check_unit.py - validate ONE translated unit result against its English source.

Same checks apply_units.py enforces, but scoped to a single file so a translator can
self-correct before its output ever reaches the packs.

  C1 zh contains Chinese
  C2 HTML tag-name sequence identical to the English
  C3 every @X[...] / [[...]] bracket body identical, in order (only {label} may differ)
  C4 bilingual leaves end with " <english>"; prose leaves carry no long English run
  C5 zh is not a copy of en
  C6 every leaf in the unit has a translation

Exit code 0 only when the unit is complete and clean.

Usage:
  python check_unit.py --unit <units/<pack>/<NN>.json> --result <units_out/<pack>/<NN>.json>
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("apply_units", HERE / "apply_units.py")
au = importlib.util.module_from_spec(spec)
spec.loader.exec_module(au)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--unit", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--max-show", type=int, default=12)
    args = parser.parse_args(argv)

    unit = json.loads(args.unit.read_text(encoding="utf-8"))
    by_path = {tuple(leaf["path"]): leaf for leaf in unit["items"]}

    if not args.result.exists():
        print(f"FAIL: result file missing: {args.result}")
        return 1
    try:
        result = json.loads(args.result.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"FAIL: result file is not valid JSON: {exc}")
        return 1

    if result.get("pack") != unit["pack"]:
        print(f"FAIL: pack mismatch: result={result.get('pack')!r} unit={unit['pack']!r}")
        return 1

    seen = set()
    bad = []
    for item in result.get("translations", []):
        path = tuple(item.get("path", []))
        leaf = by_path.get(path)
        if leaf is None:
            bad.append((".".join(map(str, path))[:90], ["unknown-path"], "", ""))
            continue
        zh = item.get("zh")
        if not isinstance(zh, str) or not zh.strip():
            bad.append((".".join(path[-3:])[:90], ["empty"], leaf["en"][:160], ""))
            continue
        seen.add(path)
        problems = au.check(leaf, zh)
        if problems:
            bad.append((".".join(path[-3:])[:90], problems, leaf["en"][:220], zh[:220]))

    missing = [p for p in by_path if p not in seen]

    print(f"unit {unit['pack']} #{unit['unit']}: {len(by_path)} leaves, "
          f"{len(seen)} translated, {len(missing)} missing, {len(bad)} invalid")
    for path, problems, en, zh in bad[:args.max_show]:
        print(f"  ! {path}\n      problems: {problems}\n      EN: {en}\n      ZH: {zh}")
    if len(bad) > args.max_show:
        print(f"  ... and {len(bad) - args.max_show} more")
    for path in missing[:args.max_show]:
        leaf = by_path[path]
        print(f"  - MISSING {'.'.join(path[-3:])[:90]}  [{leaf['style']}] {leaf['en'][:110]}")
    if len(missing) > args.max_show:
        print(f"  ... and {len(missing) - args.max_show} more missing")

    if bad or missing:
        return 1
    print("OK: unit is complete and valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
