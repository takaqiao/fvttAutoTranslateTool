"""seed_from_existing.py - start a new pack's translation from whatever already exists.

A module is rarely translated from nothing. There is usually an older working copy, an
upstream community translation, or both - each covering a different, partly stale slice of
the current pack. This merges them onto the CURRENT English baseline, which is the only
thing that defines what keys legitimately exist.

Two rules make it safe:

  * A key is only seeded if the baseline has it. Copying a key the pack no longer has
    creates an ORPHAN: Babele will never apply it, and the binding gate will fail. Dead
    keys are counted and reported per source rather than carried along.
  * Sources are tried in the order given, first match wins, and every seeded leaf records
    which source it came from - so a later pass can re-examine everything that came from
    the source you trust least.

Frozen fields are never seeded or translated: `src` is an asset path, and a script macro's
`command` is code. Babele falls back to the original when a key is absent, so omitting them
is lossless.

Usage:
  python seed_from_existing.py --en-dir <baseline> --out-dir <workspace>
                               --source <name>=<file-or-dir> [--source ...] [--write]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

CJK = re.compile(r"[一-鿿]")
# Never translated: an asset path, and macro code. See PROJECT.md "冻结字段".
FROZEN_KEYS = {"src", "img", "texture", "sound", "path", "command"}


def walk(node, path=()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from walk(value, path + (key,))
    elif isinstance(node, str):
        yield path, node


def flat(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {".".join(p): v for p, v in walk(data.get("entries", {}))}, data


def set_path(entries, dotted, value):
    parts = dotted.split(".")
    node = entries
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            return False
    node[parts[-1]] = value
    return True


def resolve_source(spec, pack_name):
    """`name=path` where path is a file or a directory holding <pack_name>."""
    label, _, raw = spec.partition("=")
    p = Path(raw)
    if p.is_dir():
        p = p / pack_name
    return label or p.stem, p


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--en-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--source", action="append", required=True,
                        help="name=path ; earlier sources win")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {}

    for en_path in sorted(args.en_dir.glob("*.json")):
        en, en_data = flat(en_path)
        sources = []
        for spec in args.source:
            label, path = resolve_source(spec, en_path.name)
            if path.exists():
                sources.append((label, *flat(path)))
            else:
                print(f"  [skip] {label}: no {path.name}")

        seeded, provenance, frozen = {}, Counter(), 0
        for key, en_value in en.items():
            leaf = key.split(".")[-1]
            if leaf in FROZEN_KEYS:
                frozen += 1
                continue
            for label, src, _raw in sources:
                value = src.get(key)
                if value and CJK.search(value):
                    seeded[key] = value
                    provenance[label] += 1
                    break

        gap = [k for k in en if k not in seeded and k.split(".")[-1] not in FROZEN_KEYS]
        dead = {label: len([k for k in src if k not in en]) for label, src, _ in sources}

        out = {"label": en_data.get("label"), "entries": {}}
        for key, value in seeded.items():
            set_path(out["entries"], key, value)
        # Keys with no translation yet stay ABSENT rather than English: Babele falls back
        # to the original, and an English value would look like a finished translation to
        # every downstream scan.
        target = args.out_dir / en_path.name
        if args.write:
            target.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                              encoding="utf-8", newline="\n")

        print(f"[{'write' if args.write else 'dry  '}] {en_path.name[:52]:<52}")
        print(f"        baseline {len(en):>5}  frozen {frozen:>4}  seeded {len(seeded):>5} "
              f"({len(seeded) / max(1, len(en) - frozen) * 100:.1f}%)  gap {len(gap):>5} "
              f"({sum(len(en[k]) for k in gap)} chars)")
        print(f"        by source: {dict(provenance)}   dead keys per source: {dead}")
        report[en_path.name] = {"baseline": len(en), "frozen": frozen,
                                "seeded": len(seeded), "gap": len(gap),
                                "gap_chars": sum(len(en[k]) for k in gap),
                                "provenance": dict(provenance), "dead": dead,
                                "gap_keys": gap}

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nreport -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
