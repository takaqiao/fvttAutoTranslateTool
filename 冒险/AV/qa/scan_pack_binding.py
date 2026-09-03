"""scan_pack_binding.py - does a Babele translation file actually bind to the installed pack?

Resolves every installed document against the translation dict using Babele's own
match order (``match: ["_id", "name", "sourceId"]``, per
``babele/script/identity/document-identity.js``), and reports three verdicts:

  BOUND    a match candidate is a key in the translation collection
  UNBOUND  no candidate matches -> that document renders in English (not yet translated)
  ORPHAN   a translation key matches no installed document -> dead translation,
           Babele will never apply it.  This is the key-drift signal.

UNBOUND is a coverage number.  ORPHAN is a defect: it has exactly one cause.

Input is ``pack-keys.json`` from ``dump_pack_keys.mjs``.

Usage:
  python scan_pack_binding.py --keys <pack-keys.json> --translation <file.json> --collection <mod.pack>
  python scan_pack_binding.py --keys <pack-keys.json> --dir <dir-of-translations> [--report out.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def index_nodes(pack):
    return {node["path"]: node for node in pack["nodes"]}


def walk(node_path, trans_dict, nodes, results, collection_id):
    """Resolve one manifest node against the translation dict living at the same path."""
    node = nodes.get(node_path)
    if node is None or not isinstance(trans_dict, dict):
        return

    consumed = set()
    for entry in node["entries"]:
        matched = None
        for cand in entry["matchCandidates"]:
            if cand in trans_dict:
                matched = cand
                break
        if matched is None:
            results["unbound"].append(
                {
                    "collection": collection_id,
                    "path": node_path,
                    "type": node["documentType"],
                    "name": entry["name"],
                    "_id": entry["_id"],
                }
            )
            continue

        consumed.add(matched)
        results["bound"].append({"collection": collection_id, "path": node_path})

        child_value = trans_dict[matched]
        if not isinstance(child_value, dict):
            continue
        prefix = node_path + "." + entry["key"] + "."
        for child_path in nodes:
            if child_path.startswith(prefix) and "." not in child_path[len(prefix):]:
                out_key = child_path[len(prefix):]
                walk(child_path, child_value.get(out_key), nodes, results, collection_id)

    for key in trans_dict:
        if key not in consumed:
            results["orphan"].append(
                {
                    "collection": collection_id,
                    "path": node_path,
                    "type": node["documentType"],
                    "key": key,
                }
            )


def scan_one(pack, translation, collection_id):
    nodes = index_nodes(pack)
    results = {"bound": [], "unbound": [], "orphan": []}
    walk("entries", translation.get("entries"), nodes, results, collection_id)
    return results


def per_path_stats(results):
    stats = {}
    for verdict in ("bound", "unbound", "orphan"):
        for row in results[verdict]:
            slot = stats.setdefault(row["path"], {"bound": 0, "unbound": 0, "orphan": 0})
            slot[verdict] += 1
    return stats


def collapse(path):
    """entries.<adv>.actors.<name>.items -> entries.*.actors.*.items"""
    parts = path.split(".")
    return ".".join(p if i % 2 == 0 else "*" for i, p in enumerate(parts))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--keys", required=True, type=Path)
    parser.add_argument("--translation", type=Path)
    parser.add_argument("--collection", type=str)
    parser.add_argument("--dir", type=Path, help="scan every <mod>.<pack>.json in this directory")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--min-bound", type=float, default=0.98)
    parser.add_argument("--max-orphan", type=int, default=0)
    parser.add_argument("--quiet-unbound", action="store_true")
    args = parser.parse_args(argv)

    manifest = load(args.keys)
    jobs = []
    if args.dir:
        for path in sorted(args.dir.glob("*.json")):
            if path.name in {"labels.json", "titles.json"}:
                continue
            collection_id = path.stem
            if collection_id in manifest["packs"]:
                jobs.append((collection_id, path))
    if args.translation:
        collection_id = args.collection or args.translation.stem
        jobs.append((collection_id, args.translation))
    if not jobs:
        print("no translation files matched a dumped pack", file=sys.stderr)
        return 2

    report = {}
    failed = False
    for collection_id, path in jobs:
        pack = manifest["packs"].get(collection_id)
        if pack is None:
            print(f"[SKIP] {collection_id}: not in the key manifest")
            continue
        results = scan_one(pack, load(path), collection_id)
        n_bound, n_unbound, n_orphan = (len(results[k]) for k in ("bound", "unbound", "orphan"))
        total = n_bound + n_unbound
        rate = n_bound / total if total else 1.0
        verdict = "PASS"
        if n_orphan > args.max_orphan or rate < args.min_bound:
            verdict = "FAIL"
            failed = True
        print(
            f"[{verdict}] {collection_id:<58} bound={n_bound:>5} unbound={n_unbound:>5} "
            f"orphan={n_orphan:>5} rate={rate:6.2%}  <- {path.name}"
        )

        stats = per_path_stats(results)
        rolled = {}
        for node_path, slot in stats.items():
            key = collapse(node_path)
            acc = rolled.setdefault(key, {"bound": 0, "unbound": 0, "orphan": 0})
            for verdict_name in acc:
                acc[verdict_name] += slot[verdict_name]
        for key in sorted(rolled):
            slot = rolled[key]
            if slot["orphan"] or (slot["unbound"] and not args.quiet_unbound):
                tot = slot["bound"] + slot["unbound"]
                pct = slot["bound"] / tot if tot else 1.0
                print(
                    f"        {key:<50} bound={slot['bound']:>5} unbound={slot['unbound']:>5} "
                    f"orphan={slot['orphan']:>5} ({pct:.0%})"
                )
        report[collection_id] = {
            "file": str(path),
            "bound": n_bound,
            "unbound": n_unbound,
            "orphan": n_orphan,
            "rate": rate,
            "verdict": verdict,
            "by_path": rolled,
            "orphans": results["orphan"][:2000],
            "unbounds": results["unbound"][:2000],
        }

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nreport -> {args.report}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
