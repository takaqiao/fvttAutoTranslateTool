"""apply_units.py - validate translated units and write them into the CN packs.

Every translation is checked against its English source before it can land.  A unit
that fails is reported and skipped whole-file-safe: nothing is written unless the run
is clean, or --partial is given (which writes only the leaves that passed).

Checks, per leaf:
  C1 has Chinese                zh contains CJK
  C2 markup preserved           HTML tag-name sequence of zh == that of en
  C3 enricher machinery intact  every `@X[...]` / `[[...]]` bracket body of zh equals
                                en's, in the same order - only the `{label}` may differ
  C4 style honoured             bilingual leaves end with " <english>"; prose leaves
                                carry no >=5-word English run in their visible text
  C5 not a copy                 zh != en

Result-file shape (one per unit):
  {"pack": "<collection-id>", "unit": "07",
   "translations": [{"path": ["entries", ...], "zh": "..."}]}

Usage:
  python apply_units.py --units-dir <dir> --results-dir <dir> --cn-dir <dir>
                        --en-dir <dir> [--write] [--partial] [--report out.json]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from collections import Counter
from pathlib import Path

CJK = re.compile(r"[㐀-鿿]")
TAG = re.compile(r"<\s*(/?)\s*([A-Za-z][A-Za-z0-9]*)\b[^>]*>")
ENRICHER_BODY = re.compile(r"@[A-Za-z]+\[([^\]]*)\]|\[\[([^\]]*)\]\]")
ENRICHER_ANY = re.compile(r"@[A-Za-z]+\[[^\]]*\](?:\{[^{}]*\})?|\[\[[^\]]*\]\](?:\{[^{}]*\})?")
WORD_RUN = re.compile(r"(?:\b[A-Za-z][A-Za-z'’\-]*\b[ ,]+){4,}\b[A-Za-z][A-Za-z'’\-]*\b")


def tag_seq(text):
    return [(bool(m.group(1)), m.group(2).lower()) for m in TAG.finditer(text)]


def bracket_bodies(text):
    return [a or b for a, b in ENRICHER_BODY.findall(text)]


def visible(text):
    text = ENRICHER_ANY.sub(" ", text)
    text = TAG.sub(" ", text)
    return re.sub(r"\s+", " ", re.sub(r"&[a-zA-Z#0-9]+;", " ", text)).strip()


def check(leaf, zh):
    en = leaf["en"]
    problems = []
    if not CJK.search(zh):
        problems.append("C1-no-chinese")
    if tag_seq(zh) != tag_seq(en):
        problems.append("C2-markup-drift")
    if bracket_bodies(zh) != bracket_bodies(en):
        problems.append("C3-enricher-changed")
    if leaf["style"] == "bilingual":
        if not zh.endswith(" " + en):
            problems.append("C4-not-bilingual")
    elif WORD_RUN.search(visible(zh)):
        problems.append("C4-english-left")
    if zh.strip() == en.strip():
        problems.append("C5-untranslated")
    return problems


def set_at(node, path, value):
    for seg in path[:-1]:
        nxt = node.get(seg)
        if not isinstance(nxt, dict):
            nxt = {}
            node[seg] = nxt
        node = nxt
    node[path[-1]] = value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--units-dir", required=True, type=Path)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--cn-dir", required=True, type=Path)
    parser.add_argument("--en-dir", type=Path,
                        help="English baselines; supplies `label` when a pack file is created")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--partial", action="store_true",
                        help="write the leaves that passed even when others failed")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    # index every emitted leaf by (pack, path) so a result can be validated against source
    index = {}
    for unit_file in sorted(args.units_dir.glob("*/*.json")):
        unit = json.loads(unit_file.read_text(encoding="utf-8"))
        for leaf in unit["items"]:
            index[(unit["pack"], tuple(leaf["path"]))] = leaf

    accepted = {}
    problems = []
    stats = Counter()

    for result_file in sorted(args.results_dir.glob("*/*.json")):
        try:
            result = json.loads(result_file.read_text(encoding="utf-8"))
        except Exception as exc:
            problems.append({"file": str(result_file), "error": f"unreadable: {exc}"})
            stats["unreadable"] += 1
            continue
        pack = result.get("pack")
        for item in result.get("translations", []):
            key = (pack, tuple(item.get("path", [])))
            leaf = index.get(key)
            if leaf is None:
                stats["unknown-path"] += 1
                problems.append({"file": result_file.name, "path": item.get("path"),
                                 "problems": ["unknown-path"]})
                continue
            zh = item.get("zh")
            if not isinstance(zh, str) or not zh.strip():
                stats["empty"] += 1
                continue
            found = check(leaf, zh)
            if found:
                stats["rejected"] += 1
                for code in found:
                    stats[code] += 1
                problems.append({"file": result_file.name, "pack": pack,
                                 "path": item["path"], "problems": found,
                                 "en": leaf["en"][:300], "zh": zh[:300]})
                continue
            accepted.setdefault(pack, []).append((tuple(item["path"]), zh))
            stats["accepted"] += 1

    print(f"accepted {stats['accepted']}  rejected {stats['rejected']}  "
          f"empty {stats['empty']}  unknown-path {stats['unknown-path']}")
    for code in ("C1-no-chinese", "C2-markup-drift", "C3-enricher-changed",
                 "C4-not-bilingual", "C4-english-left", "C5-untranslated"):
        if stats[code]:
            print(f"    {code:<22} {stats[code]}")
    for item in problems[:8]:
        print(f"  ! {item.get('path', [''])[-1] if item.get('path') else item.get('file')}"
              f" {item.get('problems')}")

    if args.write and accepted and (not stats["rejected"] or args.partial):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        for pack, items in accepted.items():
            cn_path = args.cn_dir / f"{pack}.json"
            if cn_path.exists():
                data = json.loads(cn_path.read_text(encoding="utf-8"))
            else:
                # A brand-new pack file still needs a `label`, or Babele shows the pack
                # name in English in the sidebar and check_pack_targets fails it.
                label = pack
                if args.en_dir:
                    en_path = args.en_dir / f"{pack}.json"
                    if en_path.exists():
                        label = json.loads(en_path.read_text(encoding="utf-8")).get("label", pack)
                data = {"label": label, "entries": {}}
                print(f"  created {cn_path.name} with label {label!r}")
            data.setdefault("entries", {})
            for path, zh in items:
                set_at(data, list(path), zh)     # path[0] is already "entries"
            backup = cn_path.parent.parent / "_backup" / f"units_{stamp}"
            backup.mkdir(parents=True, exist_ok=True)
            if cn_path.exists():
                shutil.copy2(cn_path, backup / cn_path.name)
            cn_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8", newline="\n")
            print(f"  wrote {len(items):>5} leaves -> {cn_path.name}")
    elif args.write:
        print("\nNOT written: fix the rejected leaves, or pass --partial")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"stats": dict(stats), "problems": problems},
                                          ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"report -> {args.report}")
    return 1 if stats["rejected"] and not args.partial else 0


if __name__ == "__main__":
    raise SystemExit(main())
