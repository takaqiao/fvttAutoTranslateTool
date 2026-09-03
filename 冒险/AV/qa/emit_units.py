"""emit_units.py - slice untranslated leaves into reviewable translation units.

A unit is a coherent chunk of one pack (one folder / journal / actor and its children)
small enough to translate and review in one pass.  Each leaf carries the style it must
be written in, so a translator never has to re-derive the convention:

  bilingual  `name` / `tokenName` / `prototypeToken`, folder leaves, scene map notes
             -> `中文 English`, ONE ascii space, no parentheses
  prose      everything else -> pure Chinese
  frozen     `command` / `src` / `width` / `height` -> never emitted

Paths are emitted as arrays, not dotted strings: document keys routinely contain dots
(`01. A Light in the Fog`), so a dotted path cannot be split back apart.

Names are emitted before prose across the whole pack, because prose and enricher labels
both cite names - translating prose first means retranslating it.

Usage:
  python emit_units.py --en-dir <dir> --cn-dir <dir> --out-dir <dir>
                       [--pack <collection-id>] [--max-leaves 120] [--max-chars 24000]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

CJK = re.compile(r"[㐀-鿿]")
WORD = re.compile(r"[A-Za-z]{2,}")
ENRICHER = re.compile(r"@[A-Za-z]+\[[^\]]*\](?:\{[^{}]*\})?|\[\[[^\]]*\]\](?:\{[^{}]*\})?")
TAG = re.compile(r"<[^>]+>")


def visible(text: str) -> str:
    """Text a player actually reads: no markup, no enricher machinery."""
    text = ENRICHER.sub(" ", text)
    text = TAG.sub(" ", text)
    text = re.sub(r"&[a-zA-Z#0-9]+;", " ", text)
    return re.sub(r"\s+", " ", text).strip()
NAME_KEYS = {"name", "tokenName", "prototypeToken"}
PROSE_UNDER_NOTES = {"label", "summary", "text", "description", "content",
                     "public", "private", "gamemaster", "caption", "subtitle"}
FROZEN_KEYS = {"command", "src", "width", "height"}


def style_for(path, key):
    if key in NAME_KEYS:
        return "bilingual"
    if len(path) >= 2 and path[-2] == "notes" and key not in PROSE_UNDER_NOTES:
        return "bilingual"
    if "folders" in path:
        return "bilingual"
    return "prose"


def walk(node, path=()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from walk(value, path + (key,))
    elif isinstance(node, str):
        yield path, node


def get_at(node, path):
    for seg in path:
        if not isinstance(node, dict) or seg not in node:
            return None
        node = node[seg]
    return node


def unit_group(path):
    """entries.<adv>.<collection>.<doc>.… -> a stable grouping key."""
    return ".".join(path[:4]) if len(path) >= 4 else ".".join(path[:2])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--en-dir", required=True, type=Path)
    parser.add_argument("--cn-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--pack", action="append", default=None)
    parser.add_argument("--max-leaves", type=int, default=120)
    parser.add_argument("--max-chars", type=int, default=24000)
    parser.add_argument("--max-leaf-chars", type=int, default=20000,
                        help="leaves bigger than this are carved out for manual handling; "
                             "they are almost always credit/attribution lists, not prose")
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    for en_path in sorted(args.en_dir.glob("*.json")):
        collection_id = en_path.stem
        if args.pack and collection_id not in args.pack:
            continue
        en_data = json.loads(en_path.read_text(encoding="utf-8"))
        cn_path = args.cn_dir / en_path.name
        cn_data = json.loads(cn_path.read_text(encoding="utf-8")) if cn_path.exists() else {}

        todo = []
        oversize = []
        stats = Counter()
        for path, english in walk(en_data.get("entries", {}), ("entries",)):
            key = path[-1]
            if key in FROZEN_KEYS:
                stats["frozen"] += 1
                continue
            current = get_at(cn_data.get("entries", {}), path[1:]) if cn_data else None
            if isinstance(current, str) and CJK.search(current):
                stats["done"] += 1
                continue
            # `<p>@Localize[PF2E.NPC.Abilities.Glossary.Darkvision]</p>` and image-only
            # leaves carry English *inside markup only*: the PF2e system resolves the
            # @Localize key through pf2_cn, so translating them would be wrong.
            if not WORD.search(visible(english)):
                stats["markup-only"] += 1
                continue
            if len(english) > args.max_leaf_chars:
                # e.g. AV's `Audio Credits` page: 125k chars, 1,588 Syrinscape entries of
                # `"track" by "artist"`. Those are attribution and stay in English; only
                # the short intro is prose. Handle such leaves by hand, not in bulk.
                stats["oversize"] += 1
                oversize.append({"path": list(path), "chars": len(english),
                                 "head": english[:400]})
                continue
            style = style_for(path, key)
            todo.append({"path": list(path), "style": style, "field": key,
                         "en": english, "current": current})
            stats["todo"] += 1

        if oversize:
            (args.out_dir / f"_oversize.{collection_id}.json").write_text(
                json.dumps(oversize, ensure_ascii=False, indent=1), encoding="utf-8")
        if not todo:
            summary[collection_id] = dict(stats)
            continue

        # Names first, then prose; keep each document's leaves together.
        todo.sort(key=lambda leaf: (0 if leaf["style"] == "bilingual" else 1,
                                    unit_group(leaf["path"]), len(leaf["path"])))

        pack_dir = args.out_dir / collection_id
        pack_dir.mkdir(parents=True, exist_ok=True)
        for stale in pack_dir.glob("*.json"):
            stale.unlink()

        # Prefer to break on a document boundary, but never let a unit run away: a
        # single journal can hold 130k characters, which is far too much for one pass.
        hard_chars = int(args.max_chars * 1.5)
        units, current_unit, chars = [], [], 0
        last_group = None
        for leaf in todo:
            group = unit_group(leaf["path"])
            over = (len(current_unit) >= args.max_leaves or chars >= args.max_chars)
            if current_unit and ((over and group != last_group) or chars >= hard_chars):
                units.append(current_unit)
                current_unit, chars = [], 0
            current_unit.append(leaf)
            chars += len(leaf["en"])
            last_group = group
        if current_unit:
            units.append(current_unit)

        for i, unit in enumerate(units, 1):
            styles = Counter(leaf["style"] for leaf in unit)
            payload = {
                "pack": collection_id,
                "unit": f"{i:02d}",
                "leaves": len(unit),
                "chars": sum(len(leaf["en"]) for leaf in unit),
                "styles": dict(styles),
                "convention": {
                    "bilingual": "中文 English - one ASCII space, no parentheses",
                    "prose": "pure Chinese; keep every @X[...] / [[...]] bracket byte-identical, "
                             "translate only the {label} tail; keep HTML tags and their order",
                },
                "items": unit,
            }
            (pack_dir / f"{i:02d}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

        summary[collection_id] = dict(stats) | {"units": len(units)}
        print(f"{collection_id[:56]:<56} todo={stats['todo']:>5} done={stats['done']:>5} "
              f"units={len(units):>3} chars={sum(len(l['en']) for l in todo):>7}")

    (args.out_dir / "_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    total = sum(v.get("todo", 0) for v in summary.values())
    print(f"\ntotal leaves to translate: {total}")
    print(f"units -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
