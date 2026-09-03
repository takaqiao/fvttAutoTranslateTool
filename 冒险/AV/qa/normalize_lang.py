"""normalize_lang.py - bring a module's own `languages/cn.json` up to the project standard.

`abomination-vaults-expanded` keeps its journal prose in its own i18n file rather than in
the Babele pack (the pack's page bodies are just `@Localize[...]` pointers).  That file was
written to the OLD convention: every value is `中文\\nEnglish`.

Two different leaf kinds live in it and they need opposite treatment:

  *.title    page / journal titles.  Name-like, so they stay bilingual - but as
             `中文 English` with ONE ASCII SPACE, not a newline.  These titles duplicate
             names the Babele pack already translates, and the two layers had drifted
             (忠诚的狗 vs 忠犬, 憎恶秘库 vs 憎恨魔窟), so the pack's rendering wins: it is
             what Babele actually shows in the compendium.
  everything the journal body itself.  Prose, so the English half is removed outright.
  else

`I18N.LANGUAGE` is the language's own label and is left untouched.

Usage:
  python normalize_lang.py --en <en.json> --cn <cn.json> --pack <babele pack.json>
                           --out <out.json> [--write]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("nb", HERE / "normalize_bilingual.py")
nb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nb)

CJK = re.compile(r"[㐀-鿿]")
LEAVE_ALONE = {("I18N.LANGUAGE",)}


def flatten(node, path=()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from flatten(value, path + (key,))
    elif isinstance(node, str):
        yield path, node


def set_at(node, path, value):
    for seg in path[:-1]:
        node = node.setdefault(seg, {})
    node[path[-1]] = value


def pack_name_index(pack_path):
    """{English name: bilingual Chinese name} from every level of the Babele pack."""
    index = {}
    if not pack_path or not Path(pack_path).exists():
        return index
    data = json.loads(Path(pack_path).read_text(encoding="utf-8"))

    def descend(node):
        if not isinstance(node, dict):
            return
        for key, value in node.items():
            if isinstance(value, dict):
                name = value.get("name")
                if isinstance(name, str) and CJK.search(name):
                    index.setdefault(key, name)
                descend(value)

    descend(data.get("entries", {}))
    return index


def to_space_bilingual(cn_value, english):
    """`中文\\nEnglish` (or any tail form) -> `中文 English`, one ASCII space."""
    head = cn_value
    if english and english in cn_value:
        head = cn_value.replace(english, "")
    head = head.strip().strip("\n").strip()
    head = nb.LATIN_TAIL.sub("", head) if hasattr(nb, "LATIN_TAIL") else head
    head = re.sub(r"\s+$", "", head)
    if not head or not CJK.search(head):
        return None
    return f"{head} {english}" if english else head


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--en", required=True, type=Path)
    parser.add_argument("--cn", required=True, type=Path)
    parser.add_argument("--pack", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--overrides", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    overrides = {}
    fragments = {}
    if args.overrides and args.overrides.exists():
        raw = json.loads(args.overrides.read_text(encoding="utf-8"))
        overrides = {k: v for k, v in raw.items() if not k.startswith("_")}
        fragments = raw.get("_fragments", {})
    en_data = json.loads(args.en.read_text(encoding="utf-8"))
    cn_data = json.loads(args.cn.read_text(encoding="utf-8"))
    pack_names = pack_name_index(args.pack)
    print(f"babele pack supplies {len(pack_names)} translated names\n")

    en_flat = dict(flatten(en_data))
    cn_flat = dict(flatten(cn_data))
    out = {}
    stats = Counter()
    leftovers = []

    for path, english in en_flat.items():
        dotted = ".".join(path)
        if dotted in overrides:
            set_at(out, path, overrides[dotted])
            stats["override"] += 1
            continue
        current = cn_flat.get(path)
        if path in LEAVE_ALONE or not isinstance(current, str):
            set_at(out, path, current if isinstance(current, str) else english)
            stats["untouched"] += 1
            continue

        if path[-1] == "title":
            from_pack = pack_names.get(english)
            if from_pack:
                set_at(out, path, from_pack)
                stats["title-from-pack"] += 1
                continue
            rebuilt = to_space_bilingual(current, english)
            if rebuilt:
                set_at(out, path, rebuilt)
                stats["title-respaced"] += 1
            else:
                set_at(out, path, current)
                stats["title-untranslated"] += 1
                leftovers.append({"path": list(path), "kind": "title", "en": english, "cn": current})
            continue

        if not CJK.search(current):
            set_at(out, path, current)
            stats["prose-untranslated"] += 1
            leftovers.append({"path": list(path), "kind": "prose", "en": english[:600], "cn": current[:600]})
            continue

        if current.strip() == english.strip():
            set_at(out, path, current)
            stats["identical-preserved"] += 1
            continue

        head = None
        for name, fn in nb.STRATEGIES:
            head, _ = fn(current, english, 0.90)
            if head is not None:
                stats[f"prose-{name}"] += 1
                break
        if head is None:
            if nb.has_english(current, 4):
                set_at(out, path, current)
                stats["prose-manual"] += 1
                leftovers.append({"path": list(path), "kind": "prose-manual",
                                  "en": english[:600], "cn": current[:600]})
            else:
                set_at(out, path, current)
                stats["prose-already-clean"] += 1
            continue
        set_at(out, path, head.strip())

    # Sentence-level patches run last, over whatever the whole-leaf rules produced.
    for dotted, pairs in fragments.items():
        path = tuple(dotted.split("."))
        node, ok = out, True
        for seg in path[:-1]:
            node = node.get(seg) if isinstance(node, dict) else None
            if node is None:
                ok = False
                break
        if not ok or not isinstance(node, dict) or path[-1] not in node:
            print(f"    !! fragment path not found: {dotted}")
            continue
        value = node[path[-1]]
        for find, replace in pairs:
            if find in value:
                value = value.replace(find, replace)
                stats["fragment"] += 1
            else:
                print(f"    !! fragment text not found in {path[-1]}: {find[:60]}")
        node[path[-1]] = value

    for key in sorted(stats):
        print(f"    {key:<24} {stats[key]}")

    missing = set(en_flat) - set(flatten_keys(out))
    if missing:
        print(f"\n!! {len(missing)} keys missing from output")
    else:
        print(f"\nkey parity OK: {len(en_flat)} keys in, {len(list(flatten(out)))} out")

    if args.write:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8", newline="\n")
        print(f"written -> {args.out}")
    else:
        print("(dry run; pass --write)")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"stats": dict(stats), "leftovers": leftovers},
                                          ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"report -> {args.report}  ({len(leftovers)} leaves need a human)")
    return 0


def flatten_keys(node, path=()):
    for p, _ in flatten(node, path):
        yield p


if __name__ == "__main__":
    raise SystemExit(main())
