"""strip_baseline_gloss.py - drop the English gloss a translator left inline.

Some community translations render headings and link labels as `中文 English`:

    <h2>遭遇 Encounter</h2>            @UUID[…]{延后 Delay}
    <h3>经验值 Experience Points</h3>   @UUID[…]{6 狗头人战士 Kobold Warriors}

Only name-class leaves may carry English in this project; a heading and a label are prose.

The obvious test - "the English run also appears in the English text" - is useless here,
because so does everything else. In the beginner box it fires on `Pathfinder`, `Foundry`,
`Paizo`, `NPC`, `PDF`, `DC 20` and `Ctrl`, none of which are glosses and all of which the
Chinese sentence genuinely uses. 198 hits, of which roughly half are wrong.

What separates a gloss from a borrowed word is position: the gloss is the ENTIRE English
heading (or label) that stands at the same index in the baseline. So the two sides are
lined up first, and a tail is dropped only when the Chinese ends with exactly its
counterpart. `巨鼠 (4)` against `Giant Rats (4)` keeps its `(4)`; `基础DC` against
`Simple DCs` keeps its `DC`; `Foundry VTT 建议` is not a tail at all.

Usage:
  python strip_baseline_gloss.py --cn-dir <dir> --en-dir <dir> [--also f.json]
                                 [--write] [--report out.json]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from collections import Counter
from pathlib import Path

CJK = re.compile(r"[一-鿿]")
WS = re.compile(r"\s+")
TAG = re.compile(r"<[^>]+>")
HEADING = re.compile(r"(<(h[1-6])\b[^>]*>)(.*?)(</\2>)", re.S | re.I)
# The braces are optional on purpose. Matching only labelled links made the two sides
# disagree on how many there were whenever one side had dropped a label, and 28 leaves -
# every one holding the spell list - were skipped as "unaligned" rather than compared.
LABEL = re.compile(r"@[A-Za-z]+\[[^\]]+\](?:(\{)([^{}]*)(\}))?")
LINK_WITH_LABEL = re.compile(r"@[A-Za-z]+\[([^\]]+)\]\{([^{}]*)\}")
NAME_KEYS = {"name", "tokenName", "prototypeToken"}
# What may sit between the Chinese and its gloss, and must go with it.
SEPARATOR = re.compile(r"[\s　\-–—:：/（(\[]+$")


def walk(node, path=()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from walk(value, path + (key,))
    elif isinstance(node, str):
        yield path, node


def norm(text):
    return WS.sub(" ", TAG.sub("", text)).strip()


ASCII_TAIL = re.compile(r"[A-Za-z][A-Za-z0-9 ,.'’\-()/]*$")
WORD3 = re.compile(r"[A-Za-z]{3}")
ALL_PAREN = re.compile(r"^\(.*\)$")


def strip_tail(cn_inner, en_inner):
    """-> the Chinese with its English gloss removed, or None if there is no gloss."""
    cn_flat, en_flat = norm(cn_inner), norm(en_inner)
    if not en_flat or not CJK.search(cn_flat) or cn_flat == en_flat:
        return None
    if cn_flat.endswith(en_flat):
        head = cn_flat[: -len(en_flat)]
    else:
        # The counterpart often carries a count the Chinese renders with a measure word:
        # `6 狗头人战士 Kobold Warriors` against `6 Kobold Warriors`. Whole-string equality
        # misses every one of those, so accept a WORD-ALIGNED suffix of the counterpart -
        # and only that, so `基础DC` is not shortened by `Simple DCs`.
        m = ASCII_TAIL.search(cn_flat)
        if not m:
            return None
        tail = m.group(0).strip()
        if not WORD3.search(tail) or ALL_PAREN.match(tail):
            return None      # `(NPC)`, `(4)`, a bare `DC`: an abbreviation, not a gloss
        if not (en_flat == tail or en_flat.endswith(" " + tail)):
            return None
        # What the counterpart has LEFT after the tail must carry no word of its own.
        # `6 Kobold Warriors` leaves `6`, so `Kobold Warriors` really is a duplicate of
        # what the Chinese already says. `Encounter Budget` leaves `Encounter`, so
        # `Budget` is the untranslated half of the term - dropping it would delete the
        # meaning, turning `遭遇 Budget` into a wrong translation rather than a clean one.
        if re.search(r"[A-Za-z]", en_flat[: len(en_flat) - len(tail)]):
            return None
        head = cn_flat[: m.start()]
    head = SEPARATOR.sub("", head)
    # A gloss is an addition, not the whole thing; whatever is left must still read as
    # the translation.
    if not head or not CJK.search(head):
        return None
    return head


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cn-dir", required=True, type=Path)
    parser.add_argument("--en-dir", required=True, type=Path)
    parser.add_argument("--also", action="append", default=[])
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    stats, rows = Counter(), []
    targets = sorted(args.cn_dir.glob("*.json")) + [Path(a) for a in args.also]

    # A leaf whose enricher counts differ from the baseline cannot be aligned by position -
    # SRD descriptions filled in from a compendium have more links than this pack's own
    # English. The link TARGET still identifies the label, and the same target carries the
    # same English label everywhere, so learn that map once across the whole corpus.
    label_of = {}
    for en_path in sorted(args.en_dir.glob("*.json")):
        raw = json.loads(en_path.read_text(encoding="utf-8"))
        for _path, value in walk(raw.get("entries", raw)):
            for m in LINK_WITH_LABEL.finditer(value):
                target, label = m.group(1).strip(), (m.group(2) or "").strip()
                if label and label_of.setdefault(target, label) != label:
                    label_of[target] = ""      # ambiguous: never use it
    label_of = {k: v for k, v in label_of.items() if v}

    for cn_path in targets:
        if not cn_path.exists():
            continue
        en_path = args.en_dir / cn_path.name
        if not en_path.exists():
            stats["no-baseline"] += 1
            continue
        en_raw = json.loads(en_path.read_text(encoding="utf-8"))
        en_flat = {".".join(p): v for p, v in walk(en_raw.get("entries", en_raw))}
        data = json.loads(cn_path.read_text(encoding="utf-8"))
        root_key = "entries" if "entries" in data else None
        changed = 0

        def fix(node, path=()):
            nonlocal changed
            if isinstance(node, dict):
                return {k: fix(v, path + (k,)) for k, v in node.items()}
            if not isinstance(node, str) or path[-1:] and path[-1] in NAME_KEYS:
                return node
            english = en_flat.get(".".join(path), "")
            if not english:
                return node

            def by_target(text, leaf_path):
                """Same rule, but the counterpart is found by link target, not position."""
                nonlocal changed

                def one(m):
                    nonlocal changed
                    head = strip_tail(m.group(2), label_of.get(m.group(1).strip(), ""))
                    if head is None:
                        return m.group(0)
                    changed += 1
                    stats["label:stripped-by-target"] += 1
                    rows.append({"file": cn_path.name,
                                 "path": ".".join(leaf_path[-3:]), "kind": "label",
                                 "before": norm(m.group(2))[:80], "after": head[:80],
                                 "english": label_of.get(m.group(1).strip(), "")[:60]})
                    return m.group(0).replace("{" + m.group(2) + "}", "{" + head + "}")

                return LINK_WITH_LABEL.sub(one, text)

            out = node
            for kind, pattern in (("heading", HEADING), ("label", LABEL)):
                cn_hits = list(pattern.finditer(out))
                en_hits = list(pattern.finditer(english))
                if not cn_hits:
                    continue
                if len(cn_hits) != len(en_hits):
                    if kind == "label":
                        out = by_target(out, path)
                    stats[f"{kind}:unaligned"] += 1
                    continue
                pieces, last = [], 0
                for cn_m, en_m in zip(cn_hits, en_hits):
                    inner_group = 3 if kind == "heading" else 2
                    cn_inner = cn_m.group(inner_group)
                    if cn_inner is None or en_m.group(inner_group) is None:
                        continue      # one side carries no label at this position
                    head = strip_tail(cn_inner, en_m.group(inner_group))
                    if head is None:
                        continue
                    start, end = cn_m.start(inner_group), cn_m.end(inner_group)
                    pieces.append(out[last:start])
                    pieces.append(head)
                    last = end
                    changed += 1
                    stats[f"{kind}:stripped"] += 1
                    rows.append({"file": cn_path.name, "path": ".".join(path[-3:]),
                                 "kind": kind, "before": norm(cn_inner)[:80],
                                 "after": head[:80],
                                 "english": norm(en_m.group(inner_group))[:60]})
                if pieces:
                    pieces.append(out[last:])
                    out = "".join(pieces)
            return out

        root = data[root_key] if root_key else data
        fixed = fix(root)
        if not changed:
            continue
        print(f"[{'write' if args.write else 'dry  '}] {cn_path.name[:52]:<52} {changed}")
        if args.write:
            backup = cn_path.parent / "_backup" / f"gloss_{stamp}"
            backup.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cn_path, backup / cn_path.name)
            if root_key:
                data[root_key] = fixed
            else:
                data = fixed
            cn_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8", newline="\n")

    for key in sorted(stats):
        print(f"  {key:<28} {stats[key]}")
    for row in rows[:12]:
        print(f"  {row['kind']:<8} {row['before'][:44]:<44} -> {row['after'][:30]}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"stats": dict(stats), "rows": rows},
                                          ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
