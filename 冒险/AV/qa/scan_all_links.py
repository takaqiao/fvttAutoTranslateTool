"""scan_all_links.py - verify EVERY link form, not just the world-scoped ones.

A corpus links out six ways, and only one of them was ever being checked:

    @UUID[Compendium.<scope>.<pack>.<Type>.<id>]   cross-pack, the overwhelming majority
    @Compendium[<scope>.<pack>.<id>]               the older syntax, same thing
    @UUID[JournalEntry.<id>.JournalEntryPage.<id>] world-scoped, resolves after import
    @UUID[.<id>]                                   relative to the containing document
    @Actor[<id>]                                   v9 form, world lookup by id
    @Actor[Chafkhem]                               v9 form, world lookup BY NAME

`repair_dead_links.py` only looked at the third form, so "24 broken links" answered a much
smaller question than it sounded like. The last two were worse than unchecked: `classify()`
returned no ids for them and the caller skipped anything with no ids, so 71 links were
counted as inspected and never looked at. A name-based link is dead by construction in a
translated world - `collection.getName("Chafkhem")` cannot match an actor we renamed to
`查夫肯姆 Chafkhem` - so it is reported as a defect, not merely resolved.

This checks all six:

  * compendium ids against `pack-ids.json` (every id of every installed pack)
  * world ids against the family's own pack dump
  * relative ids against the ids inside the containing top-level document
  * a reference to a pack that is not installed is reported separately - that is a missing
    dependency, not a typo, and the fix is different

A module's own i18n file is not a Babele pack and does not sit in the pack directory, but
it holds prose and therefore links - AV:E keeps 647 of them there. `--also` brings those
files into the same scan; leaving them out is how the AV:E prose went unchecked entirely.

Usage:
  python scan_all_links.py --cn-dir <dir> --pack-ids <pack-ids.json> --keys <pack-keys.json>
                           [--also <file.json>] [--report out.json]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from pack_index import load_pack_ids

LINK = re.compile(r"@([A-Za-z]+)\[([^\]]+)\](?:\{([^{}]*)\})?")
DOC_TYPES = {"Actor", "Item", "JournalEntry", "JournalEntryPage", "Scene", "RollTable",
             "TableResult", "Macro", "Playlist", "PlaylistSound", "Adventure", "Folder",
             "ActiveEffect", "Cards", "Card", "Region", "Note"}
ID = re.compile(r"^[A-Za-z0-9]{16}$")


def walk(node, path=()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from walk(value, path + (key,))
    elif isinstance(node, str):
        yield path, node


def strip_anchor(target):
    """A `#heading` anchor is not part of any id.

    Without this the last id of an anchored link is compared as
    `OmLsmbwPtNMw7csF#Aesephna-menhemes`, matches nothing, and ten perfectly live AV:E
    links get reported dead - a false positive that costs more trust than a miss.
    """
    return target.partition("#")[0]


def classify(scheme, target):
    """-> (kind, pack, [ids])

    `scheme` is the word before the bracket. Only UUID, Compendium and the document types
    are links; @Check, @Damage, @Localize and friends share the syntax and are not.
    """
    if scheme in DOC_TYPES:
        # v9 form: `@Actor[<id>]` looks the target up in the world by id,
        # `@Actor[Chafkhem]` by name. Only the first can survive translation.
        head = target.split("#")[0]
        return ("legacy-id", None, [head]) if ID.match(head) else ("named", None, [])
    if scheme not in {"UUID", "Compendium"}:
        return "not-a-link", None, []
    if target.startswith("."):
        return "relative", None, [s for s in target.lstrip(".").split(".") if ID.match(s)]
    segs = target.split(".")
    if segs[0] == "Compendium":
        if len(segs) < 3:
            return "malformed", None, []
        pack = f"{segs[1]}.{segs[2]}"
        rest = segs[3:]
        # Take every segment that looks like an id and is not a type word. Pairing off
        # `Type.id` breaks on the common `…<pack>.<parentId>.JournalEntryPage.<pageId>`
        # form, where the parent id comes first and the pairing lands on nothing; the old
        # fallback then swept the literal word `JournalEntryPage` in as an id - and it is
        # exactly 16 alphanumerics, so `ID` matched it and every journal-page link in the
        # corpus was reported dead.
        ids = [s for s in rest if ID.match(s) and s not in DOC_TYPES]
        return "compendium", pack, ids
    if segs[0] in DOC_TYPES:
        return "world", None, [s for s in segs if ID.match(s) and s not in DOC_TYPES]
    # legacy @Compendium[scope.pack.id]
    if len(segs) >= 3:
        return "compendium", f"{segs[0]}.{segs[1]}", [s for s in segs[2:] if ID.match(s)]
    return "malformed", None, []


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cn-dir", required=True, type=Path)
    parser.add_argument("--pack-ids", required=True, type=Path)
    parser.add_argument("--keys", required=True, type=Path)
    parser.add_argument("--also", action="append", default=[],
                        help="a module's own i18n file - prose, therefore links")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--rulings", type=Path,
                        help="_link_rulings.json - its `_known_unresolvable` targets are "
                             "counted separately instead of as defects")
    args = parser.parse_args(argv)

    pack_ids = {k: set(v["ids"])
                for k, v in load_pack_ids(args.pack_ids, args.keys).items()}
    manifest = json.loads(args.keys.read_text(encoding="utf-8"))
    world_ids = {e["_id"] for p in manifest["packs"].values()
                 for n in p["nodes"] for e in n["entries"] if e.get("_id")}
    # ids grouped by the top-level document they live in, for relative links
    by_root = defaultdict(set)
    for coll, pack in manifest["packs"].items():
        for node in pack["nodes"]:
            root = node["path"].split(".")[1] if node["path"].count(".") >= 1 else node["path"]
            for e in node["entries"]:
                if e.get("_id"):
                    by_root[(coll, root)].add(e["_id"])

    # Targets that are dead upstream and cannot be re-pointed to anything that exists.
    # They are still reported, but under their own heading and outside the DEAD count -
    # otherwise a corpus can never be clean and the number stops meaning anything.
    accepted = set()
    if args.rulings and args.rulings.exists():
        raw = json.loads(args.rulings.read_text(encoding="utf-8"))
        accepted = {row["target"] for row in raw.get("_known_unresolvable", [])
                    if row.get("target")}

    stats = Counter()
    bad = []
    scanned = sorted(args.cn_dir.glob("*.json")) + [Path(a) for a in args.also]
    for cn_path in scanned:
        if not cn_path.exists():
            continue
        coll = cn_path.stem
        data = json.loads(cn_path.read_text(encoding="utf-8"))
        for path, value in walk(data.get("entries", data), ("entries",)):
            root = path[1] if len(path) > 1 else ""
            for m in LINK.finditer(value):
                target, label = m.group(2).strip(), (m.group(3) or "").strip()
                kind, pack, ids = classify(m.group(1), strip_anchor(target))
                if kind == "not-a-link":
                    continue
                stats[kind] += 1
                if kind == "named":
                    stats["named:dead-after-rename"] += 1
                    bad.append({"file": cn_path.name, "path": ".".join(path[-3:]),
                                "target": target, "label": label,
                                "why": "name-based-link"})
                    continue
                if not ids:
                    stats[f"{kind}:no-id"] += 1
                    bad.append({"file": cn_path.name, "path": ".".join(path[-3:]),
                                "target": target, "label": label,
                                "why": f"{kind}-no-id"})
                    continue
                if kind == "compendium":
                    if pack not in pack_ids:
                        stats["compendium:pack-not-installed"] += 1
                        bad.append({"file": cn_path.name, "path": ".".join(path[-3:]),
                                    "target": target, "label": label,
                                    "why": "pack-not-installed", "pack": pack})
                        continue
                    missing = [i for i in ids if i not in pack_ids[pack]]
                    if missing:
                        stats["compendium:dead"] += 1
                        bad.append({"file": cn_path.name, "path": ".".join(path[-3:]),
                                    "target": target, "label": label,
                                    "why": "compendium-id-gone", "pack": pack,
                                    "missing": missing})
                    else:
                        stats["compendium:ok"] += 1
                elif kind in ("world", "legacy-id"):
                    missing = [i for i in ids if i not in world_ids]
                    if missing:
                        stats[f"{kind}:dead"] += 1
                        bad.append({"file": cn_path.name, "path": ".".join(path[-3:]),
                                    "target": target, "label": label,
                                    "why": "world-id-gone", "missing": missing})
                    else:
                        stats[f"{kind}:ok"] += 1
                elif kind == "relative":
                    pool = by_root.get((coll, root), set())
                    missing = [i for i in ids if i not in pool and i not in world_ids]
                    if missing:
                        stats["relative:dead"] += 1
                        bad.append({"file": cn_path.name, "path": ".".join(path[-3:]),
                                    "target": target, "label": label,
                                    "why": "relative-id-gone", "missing": missing})
                    else:
                        stats["relative:ok"] += 1

    print("=== 引用总览 ===")
    for k in sorted(stats):
        print(f"  {k:<34} {stats[k]}")
    dead_rows = [b for b in bad if b["why"] not in
                 {"pack-not-installed", "compendium-no-id", "accepted-upstream-defect"}]
    for row in dead_rows:
        if row["target"] in accepted:
            row["why"] = "accepted-upstream-defect"
    dead = (stats["compendium:dead"] + stats["world:dead"] + stats["relative:dead"]
            + stats["legacy-id:dead"] + stats["named:dead-after-rename"])
    excused = sum(1 for b in bad if b["why"] == "accepted-upstream-defect")
    dead -= excused
    notinst = stats["compendium:pack-not-installed"]
    print(f"\nscanned {len(scanned)} files")
    print(f"DEAD LINKS: {dead}   PACK NOT INSTALLED: {notinst}"
          + (f"   ACCEPTED UPSTREAM DEFECTS: {excused}" if excused else ""))

    by_reason = Counter(b["why"] for b in bad)
    for why, n in by_reason.most_common():
        print(f"\n--- {why}: {n} ---")
        seen = set()
        for b in bad:
            if b["why"] != why:
                continue
            key = b["target"]
            if key in seen:
                continue
            seen.add(key)
            print(f"  {b['file'].split('.')[0][:22]:<22} {{{b['label'][:18]:<18}}} {b['target'][:66]}")
            if len(seen) >= 10:
                print(f"  … 另有 {n - 10} 条")
                break

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"stats": dict(stats), "bad": bad},
                                          ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nreport -> {args.report}")
    return 1 if dead else 0


if __name__ == "__main__":
    raise SystemExit(main())
