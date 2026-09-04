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

from pack_index import load_pack_ids

UUID = re.compile(r"@(?:UUID|Compendium)\[([^\]]+)\](?:\{([^{}]*)\})?")
DOC_TYPES = {"Actor", "Item", "JournalEntry", "JournalEntryPage", "Scene", "RollTable",
             "TableResult", "Macro", "Playlist", "PlaylistSound", "Adventure", "Folder"}


def walk(node, path=()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from walk(value, path + (key,))
    elif isinstance(node, str):
        yield path, node


def split_anchor(target):
    """-> (target without `#heading`, "#heading" or "").

    A link into a heading carries the anchor on the last segment, so the id compared
    against the pack was `OmLsmbwPtNMw7csF#Aesephna-menhemes` - never equal to any real
    id. Every anchored link therefore looked dead and no repair strategy could see the
    live page underneath it; the ten AV:E ones fell through to a name lookup that had
    nothing to match.
    """
    head, sep, anchor = target.partition("#")
    return head, (sep + anchor) if sep else ""


def parse_pairs(target):
    """`JournalEntry.a.JournalEntryPage.b` -> [('JournalEntry','a'), ('JournalEntryPage','b')].

    Returns [] for compendium-scoped and relative targets, which are not our problem:
    a Compendium uuid is resolved by pack, and a relative `.id` by the containing document.
    """
    target, _anchor = split_anchor(target)
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


GRADE = re.compile(r"\s*\((?:Lesser|Moderate|Greater|Major|True|Minor)\)\s*$")
PAREN = re.compile(r"\s*\([^()]*\)\s*$")


_FOLDED_NAMES = {}


def _folded_names(pack_index):
    """casefolded name -> [(pack, id, pack type)], built once for the whole run."""
    key = id(pack_index)
    if key not in _FOLDED_NAMES:
        table = defaultdict(list)
        for pack_key, entry in pack_index.items():
            for name, ids in (entry.get("byName") or {}).items():
                for doc_id in ids:
                    table[name.casefold()].append((pack_key, doc_id, entry.get("type")))
        _FOLDED_NAMES[key] = table
    return _FOLDED_NAMES[key]


def resolve_by_name(pack_index, preferred_pack, name, doc_type=None, successor=None,
                    name_moves=None):
    """Find `name` in the packs. Returns (pack, id, how) or None.

    Upstream does three things to a linked document over time, and each needs a different
    lookup: it renumbers it (same pack, same name), it MOVES it (Remaster shifted bestiary
    content into pathfinder-monster-core), or it drops a grade suffix (`Effect: X (Moderate)`
    became just `Effect: X` when graded effects were merged). Only an unambiguous single
    match counts - a creature name that exists in three bestiaries is not resolvable this way.

    `doc_type` is what makes most of the moved cases resolvable at all: `Bounty Hunter` is
    an Actor in pathfinder-npc-core AND an Item in backgrounds, so without the type from the
    link itself the name is ambiguous and nothing gets repaired.
    """
    if not name:
        return None
    # Upstream also drops qualifiers and singularises group entries: `Chimera (Primal)`
    # became `Chimera`, `Lesser Deaths` became `Lesser Death`, `Acrobats` became `Acrobat`.
    # An explicit Remaster ruling comes first and ignores the type filter: a rename can move
    # a document across types (Changelings was a bestiary Actor, Changeling is a heritage Item).
    renamed = (name_moves or {}).get(name)
    if renamed:
        # `pack:Name` pins the pack when the new name is ambiguous across packs. Only a
        # prefix that actually looks like a pack id counts - `Spell Effect: Bless` is a
        # NAME containing a colon, and splitting it would erase every effect rename.
        want_pack = ""
        head, sep, tail = renamed.partition(":")
        if sep and "." in head and " " not in head:
            want_pack, renamed = head, tail.strip()
        hits = [(pk, cid) for pk, entry in pack_index.items()
                if not want_pack or pk == want_pack
                for cid in entry.get("byName", {}).get(renamed, [])]
        if len(hits) == 1:
            return hits[0][0], hits[0][1], f"renamed:{renamed!r}"
        if hits:
            same = ([h for h in hits if h[0] == preferred_pack]
                    or [h for h in hits if h[0] == (successor or "")])
            if len(same) == 1:
                return same[0][0], same[0][1], f"renamed:{renamed!r}"
    variants = [name, GRADE.sub("", name), PAREN.sub("", name).strip()]
    for base in list(variants):
        if base.endswith("s") and not base.endswith("ss"):
            variants.append(base[:-1])
    seen_variants = set()
    for candidate in [v for v in variants
                      if v and not (v in seen_variants or seen_variants.add(v))]:
        hits = []
        for pk, entry in pack_index.items():
            if doc_type and entry.get("type") and entry["type"] != doc_type:
                continue
            for cid in entry.get("byName", {}).get(candidate, []):
                hits.append((pk, cid))
        if not hits:
            continue
        # prefer the pack the link already named; otherwise require a unique hit
        same = [h for h in hits if h[0] == preferred_pack]
        if not same and successor:
            same = [h for h in hits if h[0] == successor]
        if len(same) == 1:
            how = "same-pack" if candidate == name else f"same-pack:{candidate!r}"
            return same[0][0], same[0][1], how
        if len(hits) == 1:
            how = "moved-pack" if candidate == name else f"moved:{candidate!r}"
            return hits[0][0], hits[0][1], how

    # Last resort: upstream re-cases names between releases - `Mage For Hire` is now
    # `Mage for Hire` - and an exact-match lookup reports that as a deleted document.
    folded = _folded_names(pack_index)
    for candidate in variants:
        hits = [(pk, cid) for pk, cid, ptype in folded.get(candidate.casefold(), [])
                if not (doc_type and ptype and ptype != doc_type)]
        same = [h for h in hits if h[0] in (preferred_pack, successor)]
        if len(same) == 1:
            return same[0][0], same[0][1], f"recased:{candidate!r}"
        if len(hits) == 1:
            return hits[0][0], hits[0][1], f"recased:{candidate!r}"
    return None

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cn-dir", required=True, type=Path)
    parser.add_argument("--also", action="append", default=[],
                        help="a module's own i18n file. AV:E keeps its journal prose there "
                             "rather than in the Babele pack, so 647 links sat outside "
                             "every scan until this existed")
    parser.add_argument("--en-dir", required=True, type=Path)
    parser.add_argument("--keys", required=True, type=Path)
    parser.add_argument("--pack-ids", type=Path,
                        help="pack-ids.json from dump_pack_ids.mjs; enables repair of "
                             "cross-pack Compendium links, which are the majority")
    parser.add_argument("--rulings", type=Path,
                        help="hand decisions for targets the packs cannot resolve")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    rulings, pack_moves, name_moves = {}, {}, {}
    if args.rulings and args.rulings.exists():
        raw = json.loads(args.rulings.read_text(encoding="utf-8"))
        rulings = {k: v for k, v in raw.items() if not k.startswith("_")}
        pack_moves = {k: v for k, v in (raw.get("_pack_moves") or {}).items()
                      if not k.startswith("_")}
        name_moves = {k: v for k, v in (raw.get("_name_moves") or {}).items()
                      if not k.startswith("_")}
    pack_index = load_pack_ids(args.pack_ids, args.keys)
    if pack_index:
        print(f"pack ids: {len(pack_index)} packs, "
              f"{sum(len(p['ids']) for p in pack_index.values())} ids")
    targets = sorted(args.cn_dir.glob("*.json")) + [Path(a) for a in args.also]
    real, by_name, parent_of, journal_pages = build_index(args.keys)
    print(f"pack index: {len(real)} ids, {len(by_name)} (type,name) pairs, "
          f"{len(journal_pages)} journals")

    # ---- pass A: which old journal id did each dead page link point into? ----
    clusters = defaultdict(set)
    for cn_path in targets:
        en_path = args.en_dir / cn_path.name
        if not en_path.exists() or not cn_path.exists():
            continue
        en_data = json.loads(en_path.read_text(encoding="utf-8"))
        en_flat = {".".join(p): v for p, v in
                   walk(en_data.get("entries", en_data), ("entries",))}
        cn_data = json.loads(cn_path.read_text(encoding="utf-8"))
        for path, value in walk(cn_data.get("entries", cn_data), ("entries",)):
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

    for cn_path in targets:
        if not cn_path.exists():
            continue
        en_path = args.en_dir / cn_path.name
        en_flat = {}
        if en_path.exists():
            # Same key basis as the Chinese walk below, which starts at ("entries",) -
            # otherwise every lookup misses by exactly one path segment.
            en_raw = json.loads(en_path.read_text(encoding="utf-8"))
            en_flat = {".".join(p): v for p, v in
                       walk(en_raw.get("entries", en_raw), ("entries",))}
        data = json.loads(cn_path.read_text(encoding="utf-8"))
        root_key = "entries" if "entries" in data else None
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
                # Cross-pack links are checked FIRST: parse_pairs() deliberately returns []
                # for them, so anything placed after the `if not pairs` guard below never
                # runs on the form that makes up five sixths of the corpus.
                # A cross-pack link: `Compendium.<scope>.<pack>.<Type>.<id>`. The system
                # renumbers its own packs between releases, so these die exactly like
                # world ids do - and they outnumber them five to one.
                # Two syntaxes name the same thing: `Compendium.<scope>.<pack>.<Type>.<id>`
                # and the legacy `<scope>.<pack>.<id>`. Matching only the first skipped the
                # 549 legacy links entirely - and those are where most of the equipment and
                # bestiary references live.
                is_compendium = target.startswith("Compendium.")
                is_legacy = (not is_compendium and not target.startswith(".")
                             and target.split(".")[0] not in DOC_TYPES
                             and len(target.split(".")) >= 3)
                if pack_index and (is_compendium or is_legacy):
                    segs = target.split(".")
                    if True:
                        pk = f"{segs[1]}.{segs[2]}" if is_compendium else f"{segs[0]}.{segs[1]}"
                        entry = pack_index.get(pk)
                        if entry:
                            pids = set(entry["ids"])
                            rest = segs[3:] if is_compendium else segs[2:]
                            tail = [s for s in rest if len(s) == 16 and s.isalnum()
                                    and s not in DOC_TYPES]
                            if tail and any(i not in pids for i in tail):
                                stats["broken"] += 1
                                en_label = en_by_target.get(target)
                                # A document can keep its id and change packs - the
                                # system moved `Hellknight Armiger` out of npc-gallery
                                # into lost-omens-bestiary without renumbering it.
                                # Looking only at the pack the link names calls that
                                # dead; looking for the id anywhere finds it, and the
                                # pack segment is the only thing that has to change.
                                elsewhere = [k for k, v in pack_index.items()
                                             if k != pk and tail[-1] in set(v["ids"])]
                                if len(elsewhere) == 1:
                                    new_target = target.replace(pk, elsewhere[0])
                                    changed += 1
                                    stats["repaired-compendium-same-id-moved"] += 1
                                    rows.append({"file": cn_path.name,
                                                 "path": ".".join(path[-3:]),
                                                 "target": target,
                                                 "new_target": new_target,
                                                 "label": label, "en_label": en_label,
                                                 "verdict": "repaired",
                                                 "how": "same-id-moved-pack"})
                                    tl = f"{{{label}}}" if m.group(2) is not None else ""
                                    return f"@UUID[{new_target}]{tl}"
                                link_type = next((x for x in rest if x in DOC_TYPES), None)
                                # The legacy @Compendium[scope.pack.id] form carries no Type
                                # segment; the pack's own declared type supplies it.
                                if not link_type:
                                    link_type = entry.get("type")
                                found = resolve_by_name(pack_index, pk, en_label, link_type,
                                                        pack_moves.get(pk), name_moves)
                                if found:
                                    new_pack, new_id, how = found
                                    new_target = (target.replace(tail[-1], new_id)
                                                  if new_pack == pk
                                                  else target.replace(pk, new_pack).replace(tail[-1], new_id))
                                    changed += 1
                                    stats[f"repaired-compendium-{how}"] += 1
                                    rows.append({"file": cn_path.name,
                                                 "path": ".".join(path[-3:]),
                                                 "target": target, "new_target": new_target,
                                                 "label": label, "en_label": en_label,
                                                 "verdict": "repaired", "how": how})
                                    tl = f"{{{label}}}" if m.group(2) is not None else ""
                                    return f"@UUID[{new_target}]{tl}"
                                stats["compendium-unresolved"] += 1
                                rows.append({"file": cn_path.name, "path": ".".join(path[-3:]),
                                             "target": target, "label": label,
                                             "en_label": en_label, "pack": pk,
                                             "verdict": "no-match"})
                            else:
                                stats["ok"] += 1
                        else:
                            stats["pack-not-installed"] += 1
                    return m.group(0)

                pairs = parse_pairs(target)
                if not pairs:
                    return m.group(0)
                _bare, anchor = split_anchor(target)
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
                        new_target = (f"JournalEntry.{live_parent}"
                                      f".JournalEntryPage.{pairs[1][1]}{anchor}")
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
                            new_target = (f"JournalEntry.{new_parent}"
                                          f".JournalEntryPage.{page_id}{anchor}")
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
                    new_target = f"JournalEntry.{parent_id}.JournalEntryPage.{new_id}{anchor}"
                elif doc_type == "JournalEntryPage":
                    stats["unresolved-no-parent"] += 1
                    return m.group(0)
                else:
                    new_target = f"{doc_type}.{new_id}{anchor}"
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

        entries = fix(data[root_key] if root_key else data, ("entries",))
        if not changed:
            continue
        print(f"[{'write' if args.write else 'dry  '}] {cn_path.name[:52]:<52} repaired {changed}")
        if args.write:
            backup = cn_path.parent / "_backup" / f"deadlinks_{stamp}"
            backup.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cn_path, backup / cn_path.name)
            if root_key:
                data[root_key] = entries
            else:
                data = entries
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
