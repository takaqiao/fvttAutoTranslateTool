"""repair_dead_links.py - re-point @UUID links whose target id no longer exists.

When Abomination Vaults was rebuilt for Foundry v14 its journal ids changed. The add-on
modules (addons, gauntlight-extras) still ship links to the OLD ids, so those links are
dead - and they are dead in the English too, so this is an upstream defect, not a
translation defect. Babele rewrites the rendered text, which makes the Chinese the only
place we can actually fix it.

The target is recovered from the link's own LABEL, not guessed: the English baseline's
label at the same position names the document, and the pack dump says which id currently
carries that name. A link is only rewritten when that name resolves to EXACTLY ONE
document of the right type - an ambiguous name is reported, never guessed.

The bracket body is machine data everywhere else in this toolchain; here it is the thing
being repaired, so every rewrite is recorded in the report with its evidence.

Usage:
  python repair_dead_links.py --cn-dir <dir> --en-dir <dir> --keys <pack-keys.json>
                              [--write] [--report out.json]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

UUID = re.compile(r"@(?:UUID|Compendium)\[([^\]]+)\](?:\{([^{}]*)\})?")
DOC_TYPES = {"Actor", "Item", "JournalEntry", "JournalEntryPage", "Scene", "RollTable",
             "TableResult", "Macro", "Playlist", "PlaylistSound", "Adventure", "Folder"}


def walk(node, path=()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from walk(value, path + (key,))
    elif isinstance(node, str):
        yield path, node


def parse_pairs(target):
    """`JournalEntry.a.JournalEntryPage.b` -> [('JournalEntry','a'), ('JournalEntryPage','b')].

    Returns [] for compendium-scoped and relative targets, which are not our problem:
    a Compendium uuid is resolved by pack, and a relative `.id` by the containing document.
    """
    if target.startswith(".") or target.split(".")[0] == "Compendium":
        return []
    segs = target.split(".")
    pairs, i = [], 0
    while i + 1 < len(segs):
        if segs[i] in DOC_TYPES:
            pairs.append((segs[i], segs[i + 1]))
            i += 2
        else:
            i += 1
    return pairs


def build_index(keys_path):
    """id set, name -> candidates, and journal_id -> {page name: page id}."""
    manifest = json.loads(Path(keys_path).read_text(encoding="utf-8"))
    real = set()
    by_name = defaultdict(list)
    parent_of = {}
    journal_pages = {}
    for coll, pack in manifest["packs"].items():
        nodes = {n["path"]: n for n in pack["nodes"]}
        for path, node in nodes.items():
            doc_type = node["documentType"]
            # A journal key can itself contain dots (`02. The Forgotten Dungeon`), so the
            # parent cannot be found by splitting on the last dot - that silently truncates
            # the key and the parent lookup fails. Match by longest node-path prefix instead.
            parent_id = parent_name = None
            if path.endswith(".pages"):
                stem = path[: -len(".pages")]
                candidates = [p for p in nodes if stem.startswith(p + ".")]
                if candidates:
                    parent_node = max(candidates, key=len)
                    key = stem[len(parent_node) + 1:]
                    for e in nodes[parent_node]["entries"]:
                        if e.get("key") == key or e.get("name") == key:
                            parent_id, parent_name = e.get("_id"), e.get("name")
                            break
            if parent_id:
                journal_pages[parent_id] = {
                    "name": parent_name, "coll": coll,
                    "pages": {e["name"]: e["_id"] for e in node["entries"]
                              if e.get("name") and e.get("_id")},
                }
            for e in node["entries"]:
                if not e.get("_id"):
                    continue
                real.add(e["_id"])
                if e.get("name"):
                    by_name[(doc_type, e["name"])].append((e["_id"], parent_id, coll))
                if parent_id:
                    parent_of[e["_id"]] = parent_id
    return real, by_name, parent_of, journal_pages


def resolve_old_parents(clusters, journal_pages, min_hits=2):
    """old journal id -> new journal id, decided by which current journal holds the same pages.

    A page name like `The Roseguard` exists in three packs, so name alone is ambiguous. But
    every dead link that shared an old parent pointed into ONE journal, and the journal that
    still holds the most of those page names is that journal's successor. Requires a strict
    winner with at least `min_hits` shared pages - a 1-page cluster proves nothing.
    """
    mapping, notes = {}, {}
    for old_parent, labels in clusters.items():
        scored = []
        for jid, j in journal_pages.items():
            hits = len(labels & set(j["pages"]))
            if hits:
                scored.append((hits, jid, j))
        scored.sort(key=lambda t: -t[0])
        if not scored or scored[0][0] < min_hits:
            notes[old_parent] = f"no journal holds >={min_hits} of {sorted(labels)[:4]}"
            continue
        if len(scored) > 1 and scored[1][0] == scored[0][0]:
            notes[old_parent] = (f"tie between {scored[0][2]['name']!r} and "
                                 f"{scored[1][2]['name']!r} at {scored[0][0]} pages")
            continue
        mapping[old_parent] = scored[0][1]
        notes[old_parent] = (f"-> {scored[0][2]['name']!r} ({scored[0][2]['coll']}) "
                             f"covering {scored[0][0]}/{len(labels)} pages")
    return mapping, notes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cn-dir", required=True, type=Path)
    parser.add_argument("--en-dir", required=True, type=Path)
    parser.add_argument("--keys", required=True, type=Path)
    parser.add_argument("--rulings", type=Path,
                        help="hand decisions for targets the packs cannot resolve")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    rulings = {}
    if args.rulings and args.rulings.exists():
        rulings = {k: v for k, v in json.loads(args.rulings.read_text(encoding="utf-8")).items()
                   if not k.startswith("_")}
    real, by_name, parent_of, journal_pages = build_index(args.keys)
    print(f"pack index: {len(real)} ids, {len(by_name)} (type,name) pairs, "
          f"{len(journal_pages)} journals")

    # ---- pass A: which old journal id did each dead page link point into? ----
    clusters = defaultdict(set)
    for cn_path in sorted(args.cn_dir.glob("*.json")):
        en_path = args.en_dir / cn_path.name
        if not en_path.exists():
            continue
        en_flat = {".".join(p): v for p, v in
                   walk(json.loads(en_path.read_text(encoding="utf-8")).get("entries", {}),
                        ("entries",))}
        for path, value in walk(json.loads(cn_path.read_text(encoding="utf-8")).get("entries", {}),
                                ("entries",)):
            en_text = en_flat.get(".".join(path), "")
            en_by_target = {m.group(1).strip(): (m.group(2) or "").strip()
                            for m in UUID.finditer(en_text)}
            for m in UUID.finditer(value):
                target = m.group(1).strip()
                pairs = parse_pairs(target)
                if len(pairs) != 2 or pairs[1][0] != "JournalEntryPage":
                    continue
                if pairs[0][1] in real:
                    continue
                label = en_by_target.get(target)
                if label:
                    clusters[pairs[0][1]].add(label)

    old_to_new, cluster_notes = resolve_old_parents(clusters, journal_pages)
    print(f"\n=== 旧父日志 -> 新日志（按页名重叠推定） ===")
    for old, note in sorted(cluster_notes.items()):
        mark = "ok " if old in old_to_new else "?? "
        print(f"  [{mark}] {old}  {note}")
    print()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    stats = Counter()
    rows = []

    for cn_path in sorted(args.cn_dir.glob("*.json")):
        en_path = args.en_dir / cn_path.name
        en_flat = {}
        if en_path.exists():
            # Same key basis as the Chinese walk below, which starts at ("entries",) -
            # otherwise every lookup misses by exactly one path segment.
            en_flat = {".".join(p): v for p, v in
                       walk(json.loads(en_path.read_text(encoding="utf-8")).get("entries", {}),
                            ("entries",))}
        data = json.loads(cn_path.read_text(encoding="utf-8"))
        changed = 0

        def fix(node, path=()):
            nonlocal changed
            if isinstance(node, dict):
                return {k: fix(v, path + (k,)) for k, v in node.items()}
            if not isinstance(node, str) or "@" not in node:
                return node
            en_text = en_flat.get(".".join(path), "")
            en_labels = [(m.group(1).strip(), (m.group(2) or "").strip())
                         for m in UUID.finditer(en_text)]
            en_by_target = {t: l for t, l in en_labels}

            def sub(m):
                nonlocal changed
                target, label = m.group(1).strip(), (m.group(2) or "").strip()
                pairs = parse_pairs(target)
                if not pairs:
                    return m.group(0)
                missing = [(t, i) for t, i in pairs if i not in real]
                if not missing:
                    stats["ok"] += 1
                    return m.group(0)
                stats["broken"] += 1

                if target in rulings:
                    changed += 1
                    stats["repaired-by-ruling"] += 1
                    rows.append({"file": cn_path.name, "path": ".".join(path[-3:]),
                                 "target": target, "new_target": rulings[target],
                                 "label": label, "verdict": "repaired", "how": "ruling"})
                    tail = f"{{{label}}}" if m.group(2) is not None else ""
                    return f"@UUID[{rulings[target]}]{tail}"

                # Strategy 0, strongest evidence: the PAGE id still exists and only its
                # parent went stale. The page is the thing being linked to, so its current
                # parent is authoritative - no name lookup, no cluster inference needed.
                if (len(pairs) == 2 and pairs[1][0] == "JournalEntryPage"
                        and pairs[1][1] in real and pairs[0][1] not in real):
                    live_parent = parent_of.get(pairs[1][1])
                    if live_parent:
                        new_target = f"JournalEntry.{live_parent}.JournalEntryPage.{pairs[1][1]}"
                        if new_target != target:
                            changed += 1
                            stats["repaired-by-live-page"] += 1
                            rows.append({"file": cn_path.name, "path": ".".join(path[-3:]),
                                         "target": target, "new_target": new_target,
                                         "label": label, "verdict": "repaired",
                                         "how": "live-page"})
                            tail = f"{{{label}}}" if m.group(2) is not None else ""
                            return f"@UUID[{new_target}]{tail}"
                        return m.group(0)

                en_label = en_by_target.get(target)
                if not en_label:
                    stats["no-en-label"] += 1
                    rows.append({"file": cn_path.name, "path": ".".join(path[-3:]),
                                 "target": target, "label": label,
                                 "verdict": "no-en-label"})
                    return m.group(0)
                doc_type = pairs[-1][0]
                # A dead page link is resolved inside the journal its cluster identified,
                # which is what makes a name like `The Roseguard` unambiguous.
                if doc_type == "JournalEntryPage" and len(pairs) == 2:
                    new_parent = old_to_new.get(pairs[0][1])
                    if new_parent:
                        page_id = journal_pages[new_parent]["pages"].get(en_label)
                        if page_id:
                            new_target = f"JournalEntry.{new_parent}.JournalEntryPage.{page_id}"
                            if new_target != target:
                                changed += 1
                                stats["repaired-by-cluster"] += 1
                                rows.append({"file": cn_path.name, "path": ".".join(path[-3:]),
                                             "target": target, "new_target": new_target,
                                             "label": label, "en_label": en_label,
                                             "verdict": "repaired"})
                                tail = f"{{{label}}}" if m.group(2) is not None else ""
                                return f"@UUID[{new_target}]{tail}"
                            return m.group(0)
                cands = by_name.get((doc_type, en_label), [])
                if len(cands) != 1:
                    stats[f"unresolved-{'ambiguous' if cands else 'no-match'}"] += 1
                    rows.append({"file": cn_path.name, "path": ".".join(path[-3:]),
                                 "target": target, "label": label, "en_label": en_label,
                                 "doc_type": doc_type, "candidates": len(cands),
                                 "verdict": "ambiguous" if cands else "no-match"})
                    return m.group(0)
                new_id, parent_id, _coll = cands[0]
                if doc_type == "JournalEntryPage" and parent_id:
                    new_target = f"JournalEntry.{parent_id}.JournalEntryPage.{new_id}"
                elif doc_type == "JournalEntryPage":
                    stats["unresolved-no-parent"] += 1
                    return m.group(0)
                else:
                    new_target = f"{doc_type}.{new_id}"
                if new_target == target:
                    return m.group(0)
                changed += 1
                stats["repaired"] += 1
                rows.append({"file": cn_path.name, "path": ".".join(path[-3:]),
                             "target": target, "new_target": new_target,
                             "label": label, "en_label": en_label, "verdict": "repaired"})
                tail = f"{{{label}}}" if m.group(2) is not None else ""
                return f"@UUID[{new_target}]{tail}"

            return UUID.sub(sub, node)

        entries = fix(data.get("entries", {}), ("entries",))
        if not changed:
            continue
        print(f"[{'write' if args.write else 'dry  '}] {cn_path.name[:52]:<52} repaired {changed}")
        if args.write:
            backup = cn_path.parent.parent / "_backup" / f"deadlinks_{stamp}"
            backup.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cn_path, backup / cn_path.name)
            data["entries"] = entries
            cn_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8", newline="\n")

    remaining = stats["broken"] - sum(v for k, v in stats.items() if k.startswith("repaired"))
    # Always emitted, even when zero: a Counter omits a zero key, so a caller grepping for
    # 'broken' cannot tell "none left" apart from "my pattern did not match".
    print(f"\nDEAD LINKS REMAINING: {remaining}")
    print(f"{dict(stats)}")
    for row in rows:
        if row["verdict"] == "repaired":
            print(f"  [fix] {row['file'].split('.')[0][:22]:<22} {{{row['label'][:18]:<18}}} "
                  f"{row['target'][:46]}  ->  {row['new_target'][:46]}")
    for row in rows:
        if row["verdict"] != "repaired":
            print(f"  [{row['verdict']:<12}] {row['file'].split('.')[0][:22]:<22} "
                  f"{{{row['label'][:18]:<18}}} en={row.get('en_label', '')[:26]} "
                  f"type={row.get('doc_type', '')}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"stats": dict(stats), "rows": rows},
                                          ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
