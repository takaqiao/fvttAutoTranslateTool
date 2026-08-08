#!/usr/bin/env python3
"""Assemble a realign unit's edited page files back into a batch.json.

  python collect_realign.py <单元目录> [--quiet]

Reads `<unit>/index.json` + `<unit>/pages/<i>.cn.html` and writes
`<unit>/batch.json` in the flat `{"<path>": "<中文>"}` form that
`apply_translations.py` consumes.

It deliberately does NOT re-implement the markup gate. `apply_translations.py`
is the gate; duplicating its regexes here would just create a second source of
truth that drifts. What this adds is the one signal the gate cannot give:
**which pages are still byte-identical to the original**, i.e. the ones the
agent has not actually done yet. A unit is only finished when UNTOUCHED is 0.

Self-check after collecting (--force because 8c/8j targets already hold Chinese,
which is the whole definition of the defect; --dry so nothing is written):

  python "$P\\3-常用脚本\\qa\\apply_translations.py" --repo "$P\\1-Ember汉化插件" ^
    --pack ember.crucible-adventure.json --batch "<unit>\\batch.json" --dry --force
"""
from __future__ import annotations
import json
import os
import sys


def read(path):
    # written with newline='' and no BOM, but tolerate an editor adding one
    with open(path, encoding="utf-8-sig", newline="") as f:
        return f.read()


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    quiet = "--quiet" in sys.argv
    if not args:
        raise SystemExit("usage: collect_realign.py <单元目录> [--only <id>] [--quiet]")
    unit = args[0]
    # `--only 7` writes batch.only.json with just page 7, so the gate's report is
    # about one page instead of 41. The gate prints at most 25 problems; without
    # this you cannot see the tail of a unit's failures.
    only = args[1] if len(args) > 1 else None

    index = json.load(open(os.path.join(unit, "index.json"), encoding="utf-8"))
    original = json.load(open(os.path.join(unit, "_original_cn.json"), encoding="utf-8"))

    batch, untouched, missing = {}, [], []
    for page in index["pages"]:
        if only is not None and str(page["id"]) != str(only):
            continue
        i = str(page["id"])
        f = os.path.join(unit, "pages", f"{i}.cn.html")
        if not os.path.exists(f):
            missing.append(page["path"])
            continue
        text = read(f)
        # Files are written without a trailing newline; strip one back off in
        # case an editor added it. Nothing else is normalised -- the point of
        # this format is that untouched bytes stay untouched.
        if text.endswith("\n"):
            text = text[:-1]
        if text == original.get(i):
            untouched.append(page["path"])
        batch[page["path"]] = text

    name = "batch.json" if only is None else "batch.only.json"
    out = os.path.join(unit, name)
    json.dump(batch, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print(f"{index['unit']}: {len(batch)} 页 -> {name}")
    print(f"  UNTOUCHED {len(untouched)}   （必须为 0 才算做完）")
    if missing:
        print(f"  页面文件不见了 {len(missing)}: {missing[:5]}")
    if untouched and not quiet:
        for p in untouched[:20]:
            print(f"    未动: {p}")
    raise SystemExit(1 if (untouched or missing) else 0)


main()
