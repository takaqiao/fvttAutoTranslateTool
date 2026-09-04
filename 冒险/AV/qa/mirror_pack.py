"""mirror_pack.py - carry a finished translation across to a pack that is the same text
with different compendium ids.

Several PF2e "+" modules ship each pack twice: once for Pathfinder 2e and once for
Starfinder 2e (`items` / `sf2e-items`, `pf2e-misc` / `sf2e-misc`, ...). The two are the
same prose, document for document; what differs is machine-only:

    @UUID[Compendium.pf2e.conditionitems.Item.x]{Off-Guard}       (pf2e pack)
    @UUID[Compendium.sf2e.conditions.Item.x]{Off-Guard}           (sf2e pack)

Translating both would be the same work twice and would guarantee drift between them.
A global search/replace is not available either: `Compendium.pf2e.actionspf2e` maps to
three different sf2e packs depending on the entry, so the rewrite is only well defined
per occurrence.

So this transplants POSITIONALLY, which the shape of the data makes exact:

  * proof first - with every bracket body masked, the two English leaves must be equal.
    That is what makes them the same text; a pair that fails is reported, never guessed.
  * the rewrite table is read off the two baselines: pairing their bracket tokens by
    index is sound because masked equality already proved the skeletons identical. Each
    `@X[...]` / `[[...]]` token in the Chinese is then translated THROUGH THAT TABLE, by
    identity rather than by index. Position is the wrong key: Chinese word order moves
    enrichers around (`@UUID{擒拿}…@Damage[…]` for `@Damage[…]…@UUID[…]`), and a
    positional copy would hand `@Damage`'s id to a `@UUID` - the silent corruption of
    AV PROJECT.md pitfall 10. Only the bracket body changes; the `{label}` after it is
    prose and is left as the translator wrote it.
  * a token the table does not contain, or one whose source appears twice with two
    different destinations, means the Chinese and its own baseline have diverged. Those
    are reported, never guessed.
  * `<hr>` vs `<hr />` is likewise taken from the destination, positionally, so the
    mirrored file matches its own baseline byte for byte outside the prose.

Usage:
  python mirror_pack.py --cn-dir <dir> --en-dir <dir> --pair <from>=<to> [--pair ...]
                        [--overwrite] [--write] [--report out.json]

  <from>/<to> are file basenames without `.json`, e.g.
      --pair pf2e-team-plus-magic.items=pf2e-team-plus-magic.sf2e-items

Without --overwrite a destination leaf that already holds text is left alone, so this is
safe to re-run after the destination has been hand-corrected.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from collections import Counter
from pathlib import Path

BRACKET = re.compile(r"@[A-Za-z]+\[[^\]]*\]|\[\[[^\]]*\]\]")
LABEL = re.compile(r"\{[^{}]*\}")
HR = re.compile(r"<hr\s*/?>")
CJK = re.compile(r"[一-鿿]")
# Never mirrored: an asset path, and macro code - the sf2e copy of a macro points at
# sf2e packs in its own source, and translating code is never right anyway.
FROZEN_KEYS = {"src", "img", "texture", "sound", "path", "command"}


def walk(node, path=()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from walk(value, path + (key,))
    elif isinstance(node, str):
        yield path, node


def flat(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    return {".".join(p): v for p, v in walk(data.get("entries", {}))}, data


def kinds(text: str):
    """The bracket kind sequence: what makes two leaves the same enricher skeleton."""
    out = []
    for token in BRACKET.findall(text):
        out.append("[[" if token.startswith("[[") else token[: token.index("[") + 1])
    return out


def masked(text: str) -> str:
    """Everything the sameness proof is allowed to ignore: bracket bodies, hr form,
    whitespace runs. Label text is NOT masked - it is prose and must match too."""
    text = BRACKET.sub("\x00", text)
    text = HR.sub("<hr>", text)
    return re.sub(r"\s+", " ", text).strip()


def rewrite_table(src_en: str, dst_en: str):
    """{source token -> destination token}, or None if a source token would need two
    different destinations (then nothing here can decide which one this leaf meant)."""
    table = {}
    for src, dst in zip(BRACKET.findall(src_en), BRACKET.findall(dst_en)):
        if table.setdefault(src, dst) != dst:
            return None
    return table


def transplant(cn: str, table, dst_en: str):
    """Map the Chinese leaf's machine tokens through the table. Returns None if the leaf
    uses a token its own baseline never had."""
    missing = [t for t in BRACKET.findall(cn) if t not in table]
    if missing:
        return None
    out = BRACKET.sub(lambda m: table[m.group(0)], cn)
    hrs = iter(HR.findall(dst_en))
    return HR.sub(lambda _: next(hrs), out)


def set_path(entries, dotted, value):
    node = entries
    parts = dotted.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            return False
    node[parts[-1]] = value
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cn-dir", type=Path, required=True)
    parser.add_argument("--en-dir", type=Path, required=True)
    parser.add_argument("--pair", action="append", required=True,
                        help="<from-basename>=<to-basename>, no .json")
    parser.add_argument("--overwrite", action="store_true",
                        help="also replace destination leaves that already hold text")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    grand = Counter()
    problems = []

    for pair in args.pair:
        src_name, dst_name = pair.split("=", 1)
        src_en, _ = flat(args.en_dir / f"{src_name}.json")
        dst_en, _ = flat(args.en_dir / f"{dst_name}.json")
        src_cn, _ = flat(args.cn_dir / f"{src_name}.json")
        dst_file = args.cn_dir / f"{dst_name}.json"
        dst_cn, dst_data = (flat(dst_file) if dst_file.exists()
                            else ({}, {"label": dst_name, "entries": {}}))
        entries = dst_data.setdefault("entries", {})

        stats = Counter()
        for key, dst_text in dst_en.items():
            leaf = key.split(".")[-1]
            if leaf in FROZEN_KEYS:
                stats["frozen"] += 1
                continue
            if key not in src_en:
                stats["no-source-key"] += 1
                continue
            if masked(src_en[key]) != masked(dst_text):
                stats["text-differs"] += 1
                problems.append({"pair": pair, "path": key, "why": "english text differs"})
                continue
            cn = src_cn.get(key)
            if not cn or not cn.strip():
                stats["source-untranslated"] += 1
                continue
            if dst_cn.get(key, "").strip() and not args.overwrite:
                stats["kept-existing"] += 1
                continue
            table = rewrite_table(src_en[key], dst_text)
            if table is None:
                stats["ambiguous-rewrite"] += 1
                problems.append({"pair": pair, "path": key,
                                 "why": "one source token needs two destinations"})
                continue
            if len(HR.findall(cn)) != len(HR.findall(dst_text)):
                stats["hr-count-mismatch"] += 1
                problems.append({"pair": pair, "path": key, "why": "hr count differs"})
                continue
            mirrored = transplant(cn, table, dst_text)
            if mirrored is None:
                stats["cn-has-unknown-token"] += 1
                problems.append({"pair": pair, "path": key,
                                 "why": "Chinese uses a bracket its own baseline lacks",
                                 "cn": kinds(cn), "src": kinds(src_en[key])})
                continue
            if set_path(entries, key, mirrored):
                stats["mirrored"] += 1
            else:
                stats["path-blocked"] += 1

        print(f"[{'write' if args.write else 'dry  '}] {dst_name:52s} "
              + "  ".join(f"{k}={v}" for k, v in sorted(stats.items())))
        grand.update(stats)

        if args.write and stats["mirrored"]:
            if dst_file.exists():
                shutil.copy2(dst_file,
                             dst_file.with_suffix(f".json.bak-{time.strftime('%Y%m%d-%H%M%S')}"))
            dst_file.write_text(json.dumps(dst_data, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")

    print(f"\nmirrored {grand['mirrored']} leaves"
          f"{'' if args.write else ' (dry run; pass --write)'}")
    for key, count in grand.most_common():
        if key != "mirrored":
            print(f"    {key:24s} {count}")
    for problem in problems[:5]:
        print(f"  ! {problem['pair']} {problem['path']}: {problem['why']}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(problems, ensure_ascii=False, indent=2),
                               encoding="utf-8")
        print(f"\nproblems ({len(problems)}) -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
