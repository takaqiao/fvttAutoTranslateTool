"""repair_named_links.py - fix the v9-era `@Type[Name]` links our own renaming broke.

Foundry still resolves the old form (`foundry.mjs:35782`, `collection.getName(target)`),
but it resolves it BY NAME against the world. Once the adventure is imported with our
translated names, a world actor is called `查夫肯姆 Chafkhem`, so `@Actor[Chafkhem]` finds
nothing and renders as a broken link. In an English world the same text works - which is why
this class never showed up in a diff against upstream, and why the corpus scanners missed it:
`classify()` saw no id and filed it under "malformed".

Three further facts decide the rewrite:

  * `_createContentLink` (foundry.mjs:35606) only attaches `dataset.hash` inside `if (doc)`,
    and the legacy branch never sets `doc`. So `@JournalEntry[X#Y]` opens X and silently
    ignores Y. Converting to a real page UUID is not just a repair, it restores the anchor.
  * AV:E ships four links that already target world ids (`@Actor[4q9EP8Kio0Li5tlZ]`), so the
    author is relying on Adventure import preserving `_id`. Rewriting to world-scoped
    `@UUID[Actor.<id>]` keeps the author's semantics - the link lands on the token's actor,
    not on a compendium copy - while being immune to any rename.
  * AV:E was written against an AV whose journals were one per dungeon level (`E: Arena`,
    `Otari Locations`, `Character Biographies`). Installed AV 4.1.3 has ten numbered chapters
    plus one `Adventure Toolbox`. So the journal half of these links is dead upstream too,
    and only the page half still identifies anything.

Nothing here is hand-tabulated. The letter map is derived from the data - journal
`09. On the Hunt` is the one whose pages are all `I##. …`, so `I: Hunting Grounds` resolves
to it - and a journal name that means nothing any more is discarded in favour of finding the
anchor as a page anywhere in the module.

Usage:
  python repair_named_links.py --keys qa/reports/pack-keys.json --cn-dir 工作区
      [--also 工作区/lang/x.json] [--pack-ids qa/reports/pack-ids-all.json]
      [--rulings qa/_named_rulings.json] [--write] [--report r.json]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

# The legacy form, with the `#anchor` split out the way Foundry's own regex splits it.
NAMED = re.compile(r"@(Actor|Item|JournalEntry|Scene|RollTable|Macro)"
                   r"\[([^\]#]+)(?:#([^\]]+))?\](\{([^{}]*)\})?")
ID = re.compile(r"^[A-Za-z0-9]{16}$")
LEVEL_CODE = re.compile(r"^([A-J])\d")
# `E: Arena`, `D: Belcora's Retreat` - the letter is the only part that survived the
# upstream restructure, so it is the only part worth matching on.
LETTER_JOURNAL = re.compile(r"^([A-J]):\s")
COLL_TYPE = {"actors": "Actor", "items": "Item", "journals": "JournalEntry",
             "scenes": "Scene", "tables": "RollTable", "macros": "Macro"}


def fold(name):
    """Match names the way a reader does, not the way a byte comparator does.

    `Otari Locations#Wrin's Wonders` never resolved because the page is `Wrin’s Wonders`
    with U+2019. Curly quotes, NBSP and case are all noise here; nothing else is touched,
    so two genuinely different names stay different.
    """
    s = unicodedata.normalize("NFKC", name)
    s = s.replace("’", "'").replace("‘", "'").replace("`", "'")
    s = s.replace("“", '"').replace("”", '"').replace(" ", " ")
    return re.sub(r"\s+", " ", s).strip().casefold()


def walk(node, path=()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from walk(value, path + (key,))
    elif isinstance(node, str):
        yield path, node


class Index:
    """Everything needed to turn a name into an id, kept per pack so that a link can prefer
    a target in its own module before reaching into a sibling's copy of the same name."""

    def __init__(self, manifest):
        self.docs = defaultdict(list)          # (type, folded name) -> [(pack, id)]
        self.journal_id = {}                   # (pack, folded name) -> id
        self.pages = defaultdict(dict)         # (pack, folded journal) -> {folded page: id}
        self.pages_any = defaultdict(list)     # (pack, folded page) -> [(journal id, id)]
        self.live = set()                      # every id the family actually ships
        self.letter = defaultdict(Counter)     # (pack, "E") -> Counter of journal names
        self.jletters = defaultdict(Counter)   # (pack, journal) -> Counter of letters
        for pack_key, pack in manifest["packs"].items():
            for node in pack["nodes"]:
                segs = node["path"].split(".")
                coll = segs[-1]
                doc_type = COLL_TYPE.get(coll)
                # …journals.<journal name>.pages - the journal name can contain dots
                # ("02. The Forgotten Dungeon"), so take everything between the marker
                # and the trailing "pages" rather than a fixed offset.
                journal = (".".join(segs[segs.index("journals") + 1:-1])
                           if coll == "pages" and "journals" in segs else None)
                for entry in node["entries"]:
                    name, doc_id = entry.get("name"), entry.get("_id")
                    if not name or not doc_id:
                        continue
                    key = fold(name)
                    self.live.add(doc_id)
                    if doc_type:
                        self.docs[(doc_type, key)].append((pack_key, doc_id))
                    if coll == "journals":
                        self.journal_id[(pack_key, key)] = doc_id
                    elif journal is not None:
                        self.pages[(pack_key, fold(journal))][key] = doc_id
                        self.pages_any[(pack_key, key)].append((journal, doc_id))
                        m = LEVEL_CODE.match(name)
                        if m:
                            self.letter[(pack_key, m.group(1))][journal] += 1
                            self.jletters[(pack_key, journal)][m.group(1)] += 1
        # A journal owns a letter when most of its pages carry that letter's room codes AND
        # it carries almost nothing else. Without the second half, AV:E's single
        # `Changes to the Vaults` - which touches every level from B to I - would claim all
        # ten letters, and `E: Arena`, `I: Hunting Grounds` and `B: Servants' Quarters`
        # would every one of them resolve to that same journal.
        resolved = {}
        for key, counts in self.letter.items():
            journal, hits = counts.most_common(1)[0]
            own = self.jletters[(key[0], journal)]
            if hits / max(1, sum(own.values())) >= 0.8:
                resolved[key] = journal
        self.letter = resolved

    def pick(self, candidates, order):
        """Prefer the link's own module, then the adventure everything else extends."""
        for pack in order:
            hits = [c for c in candidates if c[0] == pack]
            if len(hits) == 1:
                return hits[0]
            if len(hits) > 1:
                return None
        return candidates[0] if len(candidates) == 1 else None


class SystemIndex:
    """`pack-ids-all.json` - the installed system/module compendia, for names the family
    does not own at all (`@Actor[Shadow]` is a Monster Core creature)."""

    def __init__(self, path):
        self.by = defaultdict(list)
        if not path or not path.exists():
            return
        for pack_key, pack in json.loads(path.read_text(encoding="utf-8")).items():
            doc_type = pack.get("type")
            for name, ids in (pack.get("byName") or {}).items():
                for doc_id in ids:
                    self.by[(doc_type, fold(name))].append((pack_key, doc_id))

    def get(self, doc_type, name):
        hits = self.by.get((doc_type, fold(name)), [])
        return hits[0] if len(hits) == 1 else None


def resolve(index, system, doc_type, target, anchor, order, rulings, primary):
    """-> (uuid, why) or (None, why-it-failed)"""
    ruling = rulings.get(f"@{doc_type}[{target}" + (f"#{anchor}" if anchor else "") + "]")
    if ruling:
        return (ruling, "ruling") if ruling != "SKIP" else (None, "ruled-unresolvable")

    if doc_type == "JournalEntry":
        for pack in order:
            key = fold(target)
            journal = key if (pack, key) in index.journal_id else None
            if journal is None:
                m = LETTER_JOURNAL.match(target)
                if m and (pack, m.group(1)) in index.letter:
                    journal = fold(index.letter[(pack, m.group(1))])
            if journal is None:
                continue
            jid = index.journal_id.get((pack, journal))
            if jid is None:
                continue
            if not anchor:
                return f"JournalEntry.{jid}", ("journal" if journal == fold(target)
                                               else "journal-via-letter")
            pid = index.pages[(pack, journal)].get(fold(anchor))
            if pid:
                return f"JournalEntry.{jid}.JournalEntryPage.{pid}", "page"
        # The journal half no longer names anything (the upstream restructure merged the
        # per-level journals). The anchor still does, so look for it module-wide.
        # A journal name that resolves nowhere is a reference to the restructured AV, not to
        # the module's own content: had the anchor lived in one of the module's own journals,
        # that journal's name would still be current and step 1 would have found it. So the
        # primary adventure is searched first here, not last. With no anchor the target
        # itself is the name to look for - `@JournalEntry[<stale id>]{Odd Stories}` is a
        # page, and pages outnumber journals fifty to one.
        wanted = anchor or target
        if wanted:
            for pack in [primary] + [p for p in order if p != primary]:
                hits = index.pages_any.get((pack, fold(wanted)), [])
                if len(hits) == 1:
                    journal, pid = hits[0]
                    jid = index.journal_id.get((pack, fold(journal)))
                    if jid:
                        return (f"JournalEntry.{jid}.JournalEntryPage.{pid}",
                                "page-anywhere")
        return None, ("page-not-found" if anchor else "journal-not-found")

    hits = index.docs.get((doc_type, fold(target)), [])
    if hits:
        chosen = index.pick(hits, order)
        if chosen:
            return f"{doc_type}.{chosen[1]}", "name"
        return None, f"ambiguous({len(hits)})"
    outside = system.get(doc_type, target)
    if outside:
        return f"Compendium.{outside[0]}.{doc_type}.{outside[1]}", "system-compendium"
    return None, "name-not-found"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--keys", required=True, type=Path)
    parser.add_argument("--cn-dir", required=True, type=Path)
    parser.add_argument("--also", action="append", default=[])
    parser.add_argument("--pack-ids", type=Path)
    parser.add_argument("--en-dir", type=Path,
                        help="English baseline. A legacy `@JournalEntry[<id>]` whose id no "
                             "longer exists carries no name at all, so its English label is "
                             "the only remaining evidence of what it pointed at")
    parser.add_argument("--rulings", type=Path)
    parser.add_argument("--primary", default="pf2e-abomination-vaults.av",
                        help="the adventure every other pack in the family extends")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    manifest = json.loads(args.keys.read_text(encoding="utf-8"))
    index = Index(manifest)
    system = SystemIndex(args.pack_ids)
    rulings = {}
    if args.rulings and args.rulings.exists():
        rulings = {k: v for k, v in
                   json.loads(args.rulings.read_text(encoding="utf-8")).items()
                   if not k.startswith("_")}
    all_packs = list(manifest["packs"])
    stamp = time.strftime("%Y%m%d_%H%M%S")
    stats, rows = Counter(), []
    targets = sorted(args.cn_dir.glob("*.json")) + [Path(a) for a in args.also]

    for cn_path in targets:
        if not cn_path.exists():
            continue
        # A module's own i18n file is named after the module, a Babele file after the pack;
        # both start with the module id, which is what decides "my own module".
        module_id = cn_path.stem.split(".")[0]
        home = [p for p in all_packs if p.split(".")[0] == module_id]
        order = home + [args.primary] + [p for p in all_packs
                                         if p not in home and p != args.primary]
        data = json.loads(cn_path.read_text(encoding="utf-8"))
        root_key = "entries" if "entries" in data else None
        en_flat = {}
        if args.en_dir and (args.en_dir / cn_path.name).exists():
            en_raw = json.loads((args.en_dir / cn_path.name).read_text(encoding="utf-8"))
            en_flat = {".".join(p): v for p, v in walk(en_raw.get("entries", en_raw))}
        changed = 0

        def fix(node, path=()):
            nonlocal changed
            if isinstance(node, dict):
                return {k: fix(v, path + (k,)) for k, v in node.items()}
            if not isinstance(node, str) or "@" not in node:
                return node
            english = en_flat.get(".".join(path), "")
            en_labels = {mm.group(2).strip(): (mm.group(5) or "").strip()
                         for mm in NAMED.finditer(english)}

            def sub(m):
                nonlocal changed
                doc_type, target, anchor, braces, label = m.groups()
                if ID.match(target):
                    if target in index.live:
                        stats["already-id"] += 1
                        return m.group(0)
                    # Upstream ships this one dead too: AV 4.1.3's `Valuable Books` still
                    # points at a page id retired several versions ago. The label is the
                    # only surviving description of the target, so resolve through it.
                    english_label = en_labels.get(target)
                    if not english_label:
                        stats["unresolved:stale-id-no-label"] += 1
                        return m.group(0)
                    target, anchor = english_label, None
                uuid, why = resolve(index, system, doc_type, target, anchor, order,
                                    rulings, args.primary)
                row = {"file": cn_path.name, "path": ".".join(path[-2:]),
                       "from": m.group(0)[:110], "type": doc_type, "target": target,
                       "anchor": anchor, "label": label, "why": why}
                if not uuid:
                    stats[f"unresolved:{why}"] += 1
                    rows.append(row)
                    return m.group(0)
                stats[f"fixed:{why}"] += 1
                changed += 1
                # A label-less legacy link displayed the raw target name. A UUID link with no
                # label displays the document's name, which after Babele is the translated
                # one - so dropping the braces is what keeps the prose Chinese.
                new = f"@UUID[{uuid}]" + (braces or "")
                row["to"] = new
                rows.append(row)
                return new

            return NAMED.sub(sub, node)

        root = data[root_key] if root_key else data
        fixed = fix(root)
        if not changed:
            continue
        print(f"[{'write' if args.write else 'dry  '}] {cn_path.name[:52]:<52} {changed}")
        if args.write:
            backup = cn_path.parent / "_backup" / f"namedlinks_{stamp}"
            backup.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cn_path, backup / cn_path.name)
            if root_key:
                data[root_key] = fixed
            else:
                data = fixed
            cn_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8", newline="\n")

    print("\nletter -> journal (derived from page room codes):")
    for (pack, ch), journal in sorted(index.letter.items()):
        if pack == args.primary:
            print(f"  {ch} -> {journal}")
    for key in sorted(stats):
        print(f"  {key:<34} {stats[key]}")
    unresolved = sum(v for k, v in stats.items()
                     if k.startswith("unresolved") and k != "unresolved:ruled-unresolvable")
    print(f"\nUNRESOLVED (excluding ruled): {unresolved}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"stats": dict(stats), "rows": rows},
                                          ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"report -> {args.report}")
    return 1 if unresolved else 0


if __name__ == "__main__":
    raise SystemExit(main())
