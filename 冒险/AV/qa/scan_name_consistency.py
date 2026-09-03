"""scan_name_consistency.py - one English name, one Chinese rendering.

Units are translated independently, so the same document can be named two ways in two
packs (`Also Crisp and Fresh` came out as 也是鲜脆果蔬 in one and 又见爽脆鲜货 in another).
Nothing else in the pipeline notices: each file is internally valid.

This groups every bilingual name leaf by its ENGLISH half and reports the English names
that carry more than one Chinese head, so a single rendering can be chosen.

`--fix` rewrites the minority renderings to the winner, chosen by:
  1. an explicit ruling in --rulings
  2. the merged translation memory, when it has that exact English name
  3. otherwise the most frequent rendering; ties are reported, never guessed

Usage:
  python scan_name_consistency.py --cn-dir <dir> [--tm <tm.json>] [--rulings f.json]
                                  [--fix] [--report out.json]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

CJK = re.compile(r"[㐀-鿿]")
NAME_KEYS = {"name", "tokenName", "prototypeToken"}
LATIN_TAIL = re.compile(r"\s+[\x20-\x7E‘’–—]+$")


def split_bilingual(value):
    """`焦皮地精 Charhide Goblin` -> ('焦皮地精', 'Charhide Goblin')."""
    if not isinstance(value, str) or not CJK.search(value):
        return None, None
    m = LATIN_TAIL.search(value.strip())
    if not m:
        return None, None
    head = value.strip()[: m.start()].strip()
    tail = value.strip()[m.start():].strip()
    if not head or not tail or CJK.search(tail):
        return None, None
    return head, tail


def walk(node, path=()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from walk(value, path + (key,))
    elif isinstance(node, str):
        yield path, node


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cn-dir", required=True, type=Path)
    parser.add_argument("--tm", type=Path)
    parser.add_argument("--rulings", type=Path)
    parser.add_argument("--fix", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    tm = json.loads(args.tm.read_text(encoding="utf-8")) if args.tm and args.tm.exists() else {}
    rulings = {}
    if args.rulings and args.rulings.exists():
        rulings = {k: v for k, v in json.loads(args.rulings.read_text(encoding="utf-8")).items()
                   if not k.startswith("_")}

    renderings = defaultdict(Counter)
    where = defaultdict(set)
    # Which renderings the AV main pack uses - it is the authoritative layer for this
    # family, so it breaks ties.
    authoritative = defaultdict(set)
    AUTHORITY = "pf2e-abomination-vaults.av.json"
    for cn_path in sorted(args.cn_dir.glob("*.json")):
        data = json.loads(cn_path.read_text(encoding="utf-8"))
        for path, value in walk(data.get("entries", {}), ("entries",)):
            if path[-1] not in NAME_KEYS:
                continue
            head, tail = split_bilingual(value)
            if head:
                renderings[tail][head] += 1
                where[tail].add(cn_path.name)
                if cn_path.name == AUTHORITY:
                    authoritative[tail].add(head)

    conflicts = {en: c for en, c in renderings.items() if len(c) > 1}
    print(f"bilingual names: {len(renderings)} distinct English, "
          f"{sum(sum(c.values()) for c in renderings.values())} occurrences")
    print(f"names with more than one Chinese rendering: {len(conflicts)}\n")

    decisions = {}
    unresolved = []
    for en, counter in sorted(conflicts.items(), key=lambda kv: -sum(kv[1].values())):
        source = None
        if en in rulings:
            winner, source = rulings[en], "ruling"
        else:
            tm_entry = tm.get(en)
            tm_head = None
            if tm_entry:
                th, tt = split_bilingual(tm_entry.get("name", ""))
                tm_head = th
            # Only ever pick a rendering that is ALREADY in use. The TM's flat index is
            # first-file-wins across 84 packs, so it happily returns the wrong homograph
            # (`Bolts` -> 破阵弓矢, a magic item, not crossbow bolts; `Invisibility` ->
            # 隐形符文, the rune, not the spell). Corroboration by the corpus is the guard.
            if tm_head and tm_head in counter:
                winner, source = tm_head, "tm"
            else:
                top = counter.most_common()
                if len(top) > 1 and top[0][1] == top[1][1]:
                    # A tie between renderings that are all already in use. Break it
                    # deterministically rather than leaving the corpus inconsistent:
                    # the AV main pack wins, else the most frequent then lexicographic
                    # order - and record the choice so it is auditable.
                    tied = [h for h, n in top if n == top[0][1]]
                    from_authority = [h for h in tied if h in authoritative.get(en, set())]
                    winner = sorted(from_authority)[0] if from_authority else sorted(tied)[0]
                    source = "tie-break-authority" if from_authority else "tie-break-stable"
                else:
                    winner, source = top[0][0], "majority"
        decisions[en] = {"winner": winner, "source": source, "counts": dict(counter),
                         "files": sorted(where[en])}
        print(f"  {en[:44]:<44} -> {winner[:20]:<20} [{source}]  {dict(counter)}")

    if unresolved:
        print(f"\ntied, needs a ruling ({len(unresolved)}):")
        for en, counts in unresolved:
            print(f"  {en[:44]:<44} {counts}")

    if args.fix and decisions:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        changed_total = 0
        for cn_path in sorted(args.cn_dir.glob("*.json")):
            data = json.loads(cn_path.read_text(encoding="utf-8"))
            changed = 0

            def fix(node, path=()):
                nonlocal changed
                if isinstance(node, dict):
                    return {k: fix(v, path + (k,)) for k, v in node.items()}
                if not isinstance(node, str) or (path and path[-1] not in NAME_KEYS):
                    return node
                head, tail = split_bilingual(node)
                if not head or tail not in decisions:
                    return node
                winner = decisions[tail]["winner"]
                if head == winner:
                    return node
                changed += 1
                return f"{winner} {tail}"

            data["entries"] = fix(data.get("entries", {}), ("entries",))
            if changed:
                backup = cn_path.parent.parent / "_backup" / f"names_{stamp}"
                backup.mkdir(parents=True, exist_ok=True)
                shutil.copy2(cn_path, backup / cn_path.name)
                cn_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                                   encoding="utf-8", newline="\n")
                print(f"  fixed {changed:>4} name leaves in {cn_path.name}")
                changed_total += changed
        print(f"\nunified {changed_total} name leaves")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"decisions": decisions,
                                           "unresolved": [{"en": e, "counts": c} for e, c in unresolved]},
                                          ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"report -> {args.report}")
    return 1 if unresolved else 0


if __name__ == "__main__":
    raise SystemExit(main())
