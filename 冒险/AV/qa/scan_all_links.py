"""scan_all_links.py - verify EVERY link form, not just the world-scoped ones.

A corpus links out four ways, and only one of them was ever being checked:

    @UUID[Compendium.<scope>.<pack>.<Type>.<id>]   cross-pack, the overwhelming majority
    @Compendium[<scope>.<pack>.<id>]               the older syntax, same thing
    @UUID[JournalEntry.<id>.JournalEntryPage.<id>] world-scoped, resolves after import
    @UUID[.<id>]                                   relative to the containing document

`repair_dead_links.py` only looked at the third form, so "24 broken links" answered a much
smaller question than it sounded like. This checks all four:

  * compendium ids against `pack-ids.json` (every id of every installed pack)
  * world ids against the family's own pack dump
  * relative ids against the ids inside the containing top-level document
  * a reference to a pack that is not installed is reported separately - that is a missing
    dependency, not a typo, and the fix is different

Usage:
  python scan_all_links.py --cn-dir <dir> --pack-ids <pack-ids.json> --keys <pack-keys.json>
                           [--report out.json]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

LINK = re.compile(r"@(UUID|Compendium)\[([^\]]+)\](?:\{([^{}]*)\})?")
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


def classify(target):
    """-> (kind, pack, [ids])"""
    if target.startswith("."):
        return "relative", None, [s for s in target.lstrip(".").split(".") if ID.match(s)]
    segs = target.split(".")
    if segs[0] == "Compendium":
        if len(segs) < 3:
            return "malformed", None, []
        pack = f"{segs[1]}.{segs[2]}"
        rest = segs[3:]
        ids = [rest[i + 1] for i in range(0, len(rest) - 1, 2) if rest[i] in DOC_TYPES]
        if not ids:
            ids = [s for s in rest if ID.match(s)]
        return "compendium", pack, ids
    if segs[0] in DOC_TYPES:
        ids = [segs[i + 1] for i in range(0, len(segs) - 1, 2) if segs[i] in DOC_TYPES]
        return "world", None, ids
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
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    pack_ids = {k: set(v["ids"]) for k, v in
                json.loads(args.pack_ids.read_text(encoding="utf-8")).items()}
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

    stats = Counter()
    bad = []
    for cn_path in sorted(args.cn_dir.glob("*.json")):
        coll = cn_path.stem
        data = json.loads(cn_path.read_text(encoding="utf-8"))
        for path, value in walk(data.get("entries", {}), ("entries",)):
            root = path[1] if len(path) > 1 else ""
            for m in LINK.finditer(value):
                target, label = m.group(2).strip(), (m.group(3) or "").strip()
                kind, pack, ids = classify(target)
                stats[kind] += 1
                if not ids:
                    stats[f"{kind}:no-id"] += 1
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
                elif kind == "world":
                    missing = [i for i in ids if i not in world_ids]
                    if missing:
                        stats["world:dead"] += 1
                        bad.append({"file": cn_path.name, "path": ".".join(path[-3:]),
                                    "target": target, "label": label,
                                    "why": "world-id-gone", "missing": missing})
                    else:
                        stats["world:ok"] += 1
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
    dead = stats["compendium:dead"] + stats["world:dead"] + stats["relative:dead"]
    notinst = stats["compendium:pack-not-installed"]
    print(f"\nDEAD LINKS: {dead}   PACK NOT INSTALLED: {notinst}")

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
