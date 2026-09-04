"""autofill_srd_by_name.py - fill SRD item descriptions the compendiumSource route missed.

`autofill_from_tm.py` fills an item's description by following `_stats.compendiumSource`,
which is exact. Some modules do not carry that field - a shop module builds its stock by
copying item data, so the link back to the SRD item is gone and only the NAME survives.

Matching by name is weaker, so three guards stand in for the missing id:

  * one candidate only. A name that resolves to two different SRD descriptions is skipped -
    that is the homograph case (the Slither feat vs the Slither spell) and guessing it is
    how a translation acquires a confidently wrong paragraph.
  * the enricher skeleton must match. Every `@X[...]` bracket body and `[[...]]` roll in the
    candidate Chinese must also appear in this item's English description. A rules item's
    brackets are its identity; if they differ, the two items are not the same item.
  * plausible length. Chinese runs roughly 0.25-0.95x the English character count; outside
    that band the candidate is describing something else.

Everything rejected is reported with its reason, so the residue is a work list rather than
a silent gap.

Usage:
  python autofill_srd_by_name.py --en-dir <dir> --cn-dir <dir> --compendium-dir <chn>
                                 [--packs pf2e.equipment-srd,...] [--write] [--report r.json]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

CJK = re.compile(r"[一-鿿]")
BRACKET = re.compile(r"@[A-Za-z]+\[([^\]]*)\]|\[\[([^\]]*)\]\]")
MIN_RATIO, MAX_RATIO = 0.25, 0.95


def walk(node, path=()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from walk(value, path + (key,))
    elif isinstance(node, str):
        yield path, node


def flat(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {".".join(p): v for p, v in walk(data.get("entries", {}))}, data


# Presentation-only segments upstream has added over time. Their absence in an older
# translation is drift in FORM, not in meaning, so they must not decide the comparison.
NOISE_SEGMENT = re.compile(r"^(showDC|options|traits|name|basic|overrideTraits)(:|$)")


def normalize_bracket(body):
    """Reduce a bracket body to what actually identifies it.

    Upstream reformats enrichers without changing meaning: `[[/r X]]` becomes `@Damage[X]`,
    `will|dc:28` gains `type:` and `showDC:all`. Comparing raw bodies rejects those as
    different items. What must NOT be normalised away is a document id - a translation that
    points at a different spell than the current English is genuinely stale, not reformatted.
    """
    body = body.strip()
    body = re.sub(r"^/[a-z]+\s+", "", body)          # [[/r 1d6]] -> 1d6
    if body.startswith("Compendium.") or re.match(r"^[A-Za-z]+\.[A-Za-z0-9]{16}", body):
        return "id:" + body.split(".")[-1]            # compare the id, ignore the path shape
    segs = [s for s in body.split("|") if not NOISE_SEGMENT.match(s)]
    segs = [re.sub(r"^type:", "", s) for s in segs]
    return "|".join(sorted(s for s in segs if s))


def brackets(text):
    return Counter(normalize_bracket(m.group(1) or m.group(2) or "")
                   for m in BRACKET.finditer(text))


def set_path(entries, dotted, value):
    parts = dotted.split(".")
    node = entries
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            return False
    node[parts[-1]] = value
    return True


def build_index(compendium_dir, packs):
    index = defaultdict(list)
    for path in sorted(Path(compendium_dir).glob("pf2e.*.json")):
        pack = path.stem[len("pf2e."):]
        if packs and pack not in packs:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        entries = data.get("entries")
        if not isinstance(entries, dict):
            continue
        for en_name, value in entries.items():
            if isinstance(value, dict) and isinstance(value.get("description"), str) \
                    and value["description"].strip():
                index[en_name].append((pack, value["description"]))
    return index


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--en-dir", required=True, type=Path)
    parser.add_argument("--cn-dir", required=True, type=Path)
    parser.add_argument("--compendium-dir", required=True, type=Path)
    parser.add_argument("--packs", default="equipment-srd",
                        help="comma-separated chn pack stems to trust; empty = all pf2e.*")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    packs = {p.strip() for p in args.packs.split(",") if p.strip()}
    index = build_index(args.compendium_dir, packs)
    print(f"index: {len(index)} English names with a Chinese description "
          f"from {sorted(packs) if packs else 'all pf2e.* packs'}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    grand, rejects = Counter(), []

    for en_path in sorted(args.en_dir.glob("*.json")):
        cn_path = args.cn_dir / en_path.name
        if not cn_path.exists():
            continue
        en, _ = flat(en_path)
        cn, cn_data = flat(cn_path)
        filled = 0
        for key, en_text in en.items():
            if key.split(".")[-1] != "description":
                continue
            if cn.get(key) and CJK.search(cn[key]):
                continue
            item_name = key.split(".")[-2]
            cands = index.get(item_name, [])
            uniq = {d for _p, d in cands}
            if not cands:
                grand["no-match"] += 1
                continue
            if len(uniq) > 1:
                grand["ambiguous"] += 1
                rejects.append({"file": en_path.name, "item": item_name, "why": "ambiguous",
                                "candidates": len(uniq)})
                continue
            zh = cands[0][1]
            en_br, zh_br = brackets(en_text), brackets(zh)
            if zh_br - en_br:
                grand["bracket-mismatch"] += 1
                rejects.append({"file": en_path.name, "item": item_name,
                                "why": "bracket-mismatch",
                                "extra": list((zh_br - en_br).keys())[:3]})
                continue
            ratio = len(zh) / max(1, len(en_text))
            if not (MIN_RATIO <= ratio <= MAX_RATIO):
                grand["length-band"] += 1
                rejects.append({"file": en_path.name, "item": item_name, "why": "length-band",
                                "ratio": round(ratio, 2)})
                continue
            set_path(cn_data.setdefault("entries", {}), key, zh)
            filled += 1
            grand["filled"] += 1

        if not filled:
            continue
        print(f"[{'write' if args.write else 'dry  '}] {en_path.name[:52]:<52} filled {filled}")
        if args.write:
            backup = cn_path.parent.parent / "_backup" / f"srdname_{stamp}"
            backup.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cn_path, backup / cn_path.name)
            cn_path.write_text(json.dumps(cn_data, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8", newline="\n")

    print(f"\n{dict(grand)}")
    by_reason = Counter(r["why"] for r in rejects)
    for reason, n in by_reason.most_common():
        print(f"  rejected {reason}: {n}")
        for r in [x for x in rejects if x["why"] == reason][:4]:
            print(f"      {r['item'][:44]}  {({k: v for k, v in r.items() if k not in ('file', 'item', 'why')})}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"stats": dict(grand), "rejects": rejects},
                                          ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
