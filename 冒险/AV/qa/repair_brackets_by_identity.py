"""repair_brackets_by_identity.py - fix one drifted bracket without touching the prose.

`repair_bracket_bodies.py --from-baseline` merges positionally, so it refuses a leaf whose
bracket KIND sequence differs from the baseline's - and that refusal is correct, because
position is not a safe key (AV PROJECT.md, pitfall 10). But Chinese word order routinely
moves enrichers around, so a leaf can be perfectly good prose with one stale bracket and
still be skipped forever:

    CN  @Check[reflex|dc:@actor.flags.pf2e.aspectDC|basic:true]           (module's old syntax)
    EN  @Check[reflex|dc:resolve(@actor.flags.pf2e.aspectDC)|basic:true]  (module updated)

    CN  [[/r 1d8 #额外效果]]        (flavor label translated; upstream keeps these English
    EN  [[/r 1d8 #additional effect]]  - 930 ASCII vs 10 CJK across pf2e_compendium_chn)

Both render wrong and neither is a translation problem: the surrounding Chinese is fine.

So this ignores order entirely and compares the two bracket MULTISETS. It acts only when
the difference is a single token on each side, of the same kind - then there is exactly one
possible reading and no guess is involved. Anything larger (a dropped enricher, a `[[/r]]`
turned into an `@Damage[]`, an upstream description with a different link set) is a content
difference, not a drifted id, and is left for a translator.

ORDERING: this must run BEFORE `repair_dead_links.py`, never after. A repaired dead link
is deliberately NOT the baseline's token - that is the whole point of repairing it - so on
a second pass this tool sees "one token differs, same kind" and dutifully puts the dead
target back. Pass `--rulings` to make that impossible: every target a ruling produces is
treated as protected and never rewritten.

Usage:
  python repair_brackets_by_identity.py --cn-dir <dir> --en-dir <dir> [--write]
                                        [--rulings _link_rulings.json] [--report out.json]
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import Counter
from pathlib import Path

import re

BRACKET = re.compile(r"@[A-Za-z]+\[[^\]]*\]|\[\[[^\]]*\]\]")


def kind(token: str) -> str:
    return "[[" if token.startswith("[[") else token[: token.index("[") + 1]


def walk(node, path=()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from walk(value, path + (key,))
    elif isinstance(node, str):
        yield path, node


def flat(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return {".".join(p): v for p, v in walk(data.get("entries", {}))}, data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cn-dir", type=Path, required=True)
    parser.add_argument("--en-dir", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--rulings", type=Path,
                        help="_link_rulings.json - its repaired targets are protected "
                             "from being reverted to the baseline's dead ones")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    protected = set()
    if args.rulings and args.rulings.exists():
        raw = json.loads(args.rulings.read_text(encoding="utf-8"))
        for key, value in raw.items():
            if not key.startswith("_") and isinstance(value, str):
                protected.add(value)

    stats = Counter()
    fixed, left = [], []

    for cn_file in sorted(args.cn_dir.glob("*.json")):
        en_file = args.en_dir / cn_file.name
        if not en_file.exists():
            continue
        en_flat, _ = flat(en_file)
        cn_data = json.loads(cn_file.read_text(encoding="utf-8"))
        entries = cn_data.get("entries", {})
        touched = 0

        for path, value in list(walk(entries)):
            key = ".".join(path)
            baseline = en_flat.get(key)
            if not baseline or not value.strip():
                continue
            cn_tokens = BRACKET.findall(value)
            en_tokens = BRACKET.findall(baseline)
            if cn_tokens == en_tokens:
                continue
            extra = Counter(cn_tokens) - Counter(en_tokens)
            missing = Counter(en_tokens) - Counter(cn_tokens)
            if not extra and not missing:
                stats["reorder-only"] += 1        # same tokens, different order: fine
                continue
            if sum(extra.values()) != 1 or sum(missing.values()) != 1:
                stats["content-differs"] += 1
                left.append({"file": cn_file.stem, "path": key,
                             "extra": list(extra), "missing": list(missing)})
                continue
            old = next(iter(extra))
            new = next(iter(missing))
            if any(target in old for target in protected):
                stats["protected-by-ruling"] += 1
                continue
            if kind(old) != kind(new):
                stats["kind-differs"] += 1
                left.append({"file": cn_file.stem, "path": key,
                             "extra": [old], "missing": [new]})
                continue
            node = entries
            for part in path[:-1]:
                node = node[part]
            node[path[-1]] = value.replace(old, new)
            fixed.append({"file": cn_file.stem, "path": key, "old": old, "new": new})
            stats["repaired"] += 1
            touched += 1

        if touched:
            print(f"[{'write' if args.write else 'dry  '}] {cn_file.stem:52s} repaired={touched}")
            if args.write:
                shutil.copy2(cn_file,
                             cn_file.with_suffix(f".json.bak-{time.strftime('%Y%m%d-%H%M%S')}"))
                cn_file.write_text(json.dumps(cn_data, ensure_ascii=False, indent=2) + "\n",
                                   encoding="utf-8")

    print(f"\nrepaired {stats['repaired']} leaves"
          f"{'' if args.write else ' (dry run; pass --write)'}")
    for key, count in stats.most_common():
        if key != "repaired":
            print(f"    {key:20s} {count}")
    for row in fixed[:6]:
        print(f"  {row['file']}:{row['path']}\n      {row['old'][:90]}\n   -> {row['new'][:90]}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"repaired": fixed, "left_for_a_translator": left},
                                          ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nreport -> {args.report}  (left for a translator: {len(left)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
