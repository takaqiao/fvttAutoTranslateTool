"""repair_orphan_enrichers.py - put back the `@` an enricher lost.

`Check[reflex|dc:28]` is not an enricher. Foundry only recognises the form with its
sigil, so the leaf renders the literal text `Check[reflex|dc:28]` where a save button
should be - and nothing in the pipeline notices, because every check that looks at
enrichers finds them by matching `@X[...]` in the first place. What is not matched is
not inspected.

These are usually upstream typos: the English baseline carries the same broken text, and
`repair_bracket_bodies.py` faithfully preserves it because bracket bodies are byte-copied.
Babele rewrites the rendered text, so the Chinese file is the only place it can be fixed -
the same reasoning that lets `repair_dead_links.py` fix links that are dead in English.

Only a well-formed body is repaired. The bracket must balance, and the body must look
like the enricher it claims to be:

    Check      a `|`-separated parameter list, or a bare check type
    Damage     a formula, so it must contain a digit
    Template   a `|`-separated parameter list
    UUID       a dotted document path
    Localize   a dotted i18n key

Anything else is reported and left alone: `Act[` at the start of an English sentence is a
word, not a lost enricher.

Usage:
  python repair_orphan_enrichers.py --cn-dir <dir> [--en-dir <dir>] [--write] [--report r.json]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from collections import Counter
from pathlib import Path

ORPHAN = re.compile(r"(?<![@A-Za-z])(UUID|Check|Damage|Template|Localize|Compendium|Embed)\[")
SHAPE = {
    "Check": re.compile(r"^[a-z]+(\|[^|]+)*$"),
    "Template": re.compile(r"^[a-z]+:[^|]+(\|[^|]+)*$"),
    "Damage": re.compile(r"\d"),
    "UUID": re.compile(r"^[A-Za-z][\w\-]*(\.[\w\-]+)+"),
    "Compendium": re.compile(r"^[A-Za-z][\w\-]*(\.[\w\-]+)+"),
    "Localize": re.compile(r"^[A-Za-z][\w\-]*(\.[\w\-]+)+$"),
    "Embed": re.compile(r"^[A-Za-z][\w\-]*(\.[\w\-]+)+"),
}


def body_of(text, start):
    """The bracket body starting at `start` (the index of `[`), or None if unbalanced."""
    depth, i = 0, start
    while i < len(text):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i + 1
        i += 1
    return None, None


def walk(node, path=()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from walk(value, path + (key,))
    elif isinstance(node, str):
        yield path, node


def repair(text, stats, found):
    out, i = [], 0
    while True:
        match = ORPHAN.search(text, i)
        if not match:
            out.append(text[i:])
            break
        kind = match.group(1)
        body, end = body_of(text, match.end() - 1)
        if body is None:
            stats["unbalanced"] += 1
            out.append(text[i:match.end()])
            i = match.end()
            continue
        if not SHAPE[kind].search(body):
            stats["shape-rejected"] += 1
            found.append({"kind": kind, "body": body, "verdict": "left-alone"})
            out.append(text[i:end])
            i = end
            continue
        stats[f"repaired:{kind}"] += 1
        found.append({"kind": kind, "body": body, "verdict": "repaired"})
        out.append(text[i:match.start()])
        out.append("@" + text[match.start():end])
        i = end
    return "".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cn-dir", type=Path, required=True)
    parser.add_argument("--en-dir", type=Path,
                        help="only to report whether the defect is upstream's or ours")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    stats, found = Counter(), []
    for cn_file in sorted(args.cn_dir.glob("*.json")):
        data = json.loads(cn_file.read_text(encoding="utf-8"))
        entries = data.get("entries", {})
        touched = 0
        for path, value in list(walk(entries)):
            if not ORPHAN.search(value):
                continue
            new = repair(value, stats, found)
            if new == value:
                continue
            node = entries
            for key in path[:-1]:
                node = node[key]
            node[path[-1]] = new
            found[-1]["file"] = cn_file.stem
            found[-1]["path"] = ".".join(path)
            touched += 1
        if touched:
            print(f"[{'write' if args.write else 'dry  '}] {cn_file.stem:52s} leaves={touched}")
            if args.write:
                shutil.copy2(cn_file,
                             cn_file.with_suffix(f".json.bak-{time.strftime('%Y%m%d-%H%M%S')}"))
                cn_file.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                                   encoding="utf-8")

    if args.en_dir:
        upstream = sum(len(ORPHAN.findall(v)) for f in args.en_dir.glob("*.json")
                       for _, v in walk(json.loads(f.read_text(encoding="utf-8")).get("entries", {})))
        print(f"\nthe English baseline has {upstream} - "
              + ("the same defect, so it is upstream's" if upstream else "so this is ours"))

    print(f"\n{'' if args.write else '(dry run; pass --write)  '}"
          + "  ".join(f"{k}={v}" for k, v in sorted(stats.items())))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(found, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
