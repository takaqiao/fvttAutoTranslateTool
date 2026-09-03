"""repair_bracket_bodies.py - put back the machine words a translator changed.

Everything inside `@X[...]` and `[[...]]` is machine input: damage types, trait slugs,
skill names, DCs.  Foundry matches them against English keys, so a translated one does
not render in Chinese - it fails to resolve, and the enricher renders as literal text or
drops its trait chips.  Upstream (`pf2e_compendium_chn`) never translates them either:
its own files carry `@Damage[4d8[cold]]`, not `[冰寒]`.

The one exception is `name:` - that is the *label* the player reads on a @Check, and it
should be Chinese.  So this walks each bracket body segment by segment and only repairs
the machine ones.

The proof it worked is not "no CJK left": it is that every bracket body is now
byte-identical to the English baseline's body at the same path.  `--verify-against`
does that comparison and is the real gate.

Usage:
  python repair_bracket_bodies.py --cn-dir <dir> [--en-dir <dir>] [--write]
  python repair_bracket_bodies.py --cn-dir <dir> --verify-against <en-dir>
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
BRACKET = re.compile(r"(@[A-Za-z]+\[)([^\]]*(?:\[[^\]]*\][^\]]*)*)(\])|(\[\[)([^\]]*(?:\[[^\]]*\][^\]]*)*)(\]\])")
# Segments whose value is shown to the player, not matched against a key.
LABEL_KEYS = {"name", "text", "label"}

# Machine words seen translated in this corpus. Keys are exact tokens, so a
# label that merely contains the word is untouched.
MACHINE = {
    "治疗": "healing",
    "虚空": "void",
    "钝击": "bludgeoning",
    "穿刺": "piercing",
    "挥砍": "slashing",
    "火焰": "fire",
    "冰寒": "cold",
    "强酸": "acid",
    "电击": "electricity",
    "音波": "sonic",
    "精神": "mental",
    "精魂": "spirit",
    "流血": "bleed",
    "毒素": "poison",
    "机械": "mechanical",
    # Save types. Only ever reached inside a bracket body outside a label segment.
    "意志": "will",
    "强韧": "fortitude",
    "反射": "reflex",
    # @Localize glossary keys. Translating the KEY does not translate the text - it
    # breaks the lookup, and PF2e then prints the raw key. pf2_cn supplies the Chinese.
    "紧勒": "Constrict",
    "气势凶猛": "FrightfulPresence",
    "恶臭": "Stench",
    "陷阱": "trap",
    "魔法": "magical",
    "环境": "environmental",
}


def repair_body(body, stats):
    """Repair one bracket body. Returns (new_body, unresolved_tokens)."""
    out_segments, unresolved = [], []
    for segment in body.split("|"):
        key = segment.split(":", 1)[0] if ":" in segment else None
        if key in LABEL_KEYS:
            out_segments.append(segment)
            continue
        fixed = segment
        for zh, en in MACHINE.items():
            if zh in fixed:
                fixed = re.sub(rf"(?<![一-鿿]){re.escape(zh)}(?![一-鿿])", en, fixed)
                stats[f"{zh}->{en}"] += 1
        # Chinese comma between trait slugs is machine-hostile too
        fixed = fixed.replace("，", ",").replace("、", ",")
        if CJK.search(fixed):
            unresolved.append(fixed)
        out_segments.append(fixed)
    return "|".join(out_segments), unresolved


def repair_text(text, stats, unresolved):
    def sub(m):
        if m.group(1):
            open_tag, body, close = m.group(1), m.group(2), m.group(3)
        else:
            open_tag, body, close = m.group(4), m.group(5), m.group(6)
        if not CJK.search(body):
            return m.group(0)
        fixed, bad = repair_body(body, stats)
        unresolved.extend(bad)
        return open_tag + fixed + close
    return BRACKET.sub(sub, text)


def walk(node, path=()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from walk(value, path + (key,))
    elif isinstance(node, str):
        yield path, node


def merge_body(cn_body, en_body):
    """English is authoritative for the machine parts; Chinese wins only for labels.

    A bracket body is `a|b:c|d:e` (or a roll command with a trailing `#flavor`).
    Segments keyed by LABEL_KEYS, and the `#flavor` tail, are what the player reads;
    everything else is matched against an English key by Foundry and must be English.
    """
    if cn_body == en_body:
        return en_body

    def flavor_split(body):
        idx = body.find("#")
        return (body[:idx], body[idx:]) if idx >= 0 else (body, "")

    en_head, en_flavor = flavor_split(en_body)
    cn_head, cn_flavor = flavor_split(cn_body)

    cn_labels = {}
    for segment in cn_head.split("|"):
        if ":" in segment:
            key, _, value = segment.partition(":")
            if key in LABEL_KEYS:
                cn_labels[key] = value

    out = []
    for segment in en_head.split("|"):
        if ":" in segment:
            key, _, value = segment.partition(":")
            if key in LABEL_KEYS and key in cn_labels:
                out.append(f"{key}:{cn_labels[key]}")
                continue
        out.append(segment)
    # A translated #flavor is display text, so keep the Chinese one when there is one.
    flavor = cn_flavor if (cn_flavor and CJK.search(cn_flavor)) else en_flavor
    return "|".join(out) + flavor


def kind(opener, body):
    """A fingerprint that must match before two brackets can be treated as the same one.

    Matching by position alone is not safe: two leaves can hold the same NUMBER of
    enrichers while holding different ones, and merging by index then swaps a
    `[[/r 1d20+17 #Grapple]]` for an `@UUID[...]` - silent, total corruption. The
    opener plus (for a roll) the command word is enough to tell them apart.
    """
    if opener.startswith("[["):
        m = re.match(r"\s*(/[a-z]+)", body)
        return "[[" + (m.group(1) if m else "?")
    return opener


def kinds(text):
    out = []
    for m in BRACKET.finditer(text):
        opener = m.group(1) or m.group(4)
        body = m.group(2) if m.group(1) else m.group(5)
        out.append(kind(opener, body))
    return out


def rebuild_from_baseline(cn_text, en_text, stats):
    """Return (new_text, status). Only touches leaves whose bracket SHAPE matches."""
    cn_spans = list(BRACKET.finditer(cn_text))
    en_bodies = bodies(en_text)
    if not cn_spans:
        return cn_text, "no-brackets"
    if len(cn_spans) != len(en_bodies):
        return cn_text, "count-mismatch"
    if kinds(cn_text) != kinds(en_text):
        return cn_text, "shape-mismatch"
    out, last, changed = [], 0, 0
    for i, m in enumerate(cn_spans):
        opener = m.group(1) or m.group(4)
        body = m.group(2) if m.group(1) else m.group(5)
        closer = m.group(3) or m.group(6)
        merged = merge_body(body, en_bodies[i])
        if merged != body:
            changed += 1
            stats["bracket-rebuilt"] += 1
        out.append(cn_text[last:m.start()])
        out.append(opener + merged + closer)
        last = m.end()
    out.append(cn_text[last:])
    return "".join(out), ("rebuilt" if changed else "already-clean")


def bodies(text):
    out = []
    for m in BRACKET.finditer(text):
        out.append(m.group(2) if m.group(1) else m.group(5))
    return out


def verify(cn_dir, en_dir):
    """Every bracket body must equal the English baseline's body at the same path."""
    mismatches, checked, no_ref = [], 0, 0
    for cn_path in sorted(cn_dir.glob("*.json")):
        en_path = en_dir / cn_path.name
        if not en_path.exists():
            continue
        cn = json.loads(cn_path.read_text(encoding="utf-8")).get("entries", {})
        en = json.loads(en_path.read_text(encoding="utf-8")).get("entries", {})
        en_flat = {".".join(p): v for p, v in walk(en)}
        for path, value in walk(cn):
            cb = bodies(value)
            if not cb:
                continue
            ref = en_flat.get(".".join(path))
            if ref is None:
                no_ref += 1
                continue
            eb = bodies(ref)
            checked += 1
            if cb != eb:
                for a, b in zip(cb, eb):
                    if a != b:
                        mismatches.append(f"{cn_path.name}:{'.'.join(path[-3:])}\n"
                                          f"        cn: {a[:110]}\n        en: {b[:110]}")
                        break
                else:
                    if len(cb) != len(eb):
                        mismatches.append(f"{cn_path.name}:{'.'.join(path[-3:])}: "
                                          f"{len(cb)} brackets vs {len(eb)} in English")
    return checked, no_ref, mismatches


def run_from_baseline(args):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    grand, status_counts, needs_review = Counter(), Counter(), []
    for cn_path in sorted(args.cn_dir.glob("*.json")):
        en_path = args.from_baseline / cn_path.name
        if not en_path.exists():
            status_counts["no-baseline-file"] += 1
            continue
        data = json.loads(cn_path.read_text(encoding="utf-8"))
        en_flat = {".".join(p): v for p, v in
                   walk(json.loads(en_path.read_text(encoding="utf-8")).get("entries", {}))}
        stats = Counter()
        changed = 0

        def fix(node, path=()):
            nonlocal changed
            if isinstance(node, dict):
                return {k: fix(v, path + (k,)) for k, v in node.items()}
            if not isinstance(node, str) or "[" not in node:
                return node
            ref = en_flat.get(".".join(path))
            if ref is None:
                status_counts["no-baseline-leaf"] += 1
                return node
            new, status = rebuild_from_baseline(node, ref, stats)
            status_counts[status] += 1
            if status in ("count-mismatch", "shape-mismatch"):
                needs_review.append(f"{status}  {cn_path.name}:{'.'.join(path[-3:])}")
            if new != node:
                changed += 1
            return new

        entries = fix(data.get("entries", {}), ())
        grand.update(stats)
        if not changed:
            continue
        print(f"[{'write' if args.write else 'dry  '}] {cn_path.name[:52]:<52} "
              f"leaves={changed} brackets={stats['bracket-rebuilt']}")
        if args.write:
            backup = cn_path.parent.parent / "_backup" / f"brackets_{stamp}"
            backup.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cn_path, backup / cn_path.name)
            data["entries"] = entries
            cn_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8", newline="\n")

    print(f"\nbrackets rebuilt from the English baseline: {grand['bracket-rebuilt']}")
    for key in sorted(status_counts):
        print(f"    {key:<20} {status_counts[key]}")
    print(f"\nleaves whose bracket COUNT differs from English (not touched, need a look): "
          f"{len(needs_review)}")
    for row in needs_review[:25]:
        print(f"    {row}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"rebuilt": grand["bracket-rebuilt"],
                                           "status": dict(status_counts),
                                           "count_mismatch": needs_review},
                                          ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"report -> {args.report}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cn-dir", required=True, type=Path)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--verify-against", type=Path)
    parser.add_argument("--from-baseline", type=Path,
                        help="rebuild every bracket body from this English baseline, "
                             "keeping Chinese only in label segments")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    if args.from_baseline:
        return run_from_baseline(args)

    if args.verify_against:
        checked, no_ref, mismatches = verify(args.cn_dir, args.verify_against)
        print(f"bracket bodies compared to English: {checked} leaves "
              f"({no_ref} had no English counterpart)")
        print(f"mismatches: {len(mismatches)}")
        for row in mismatches[:40]:
            print(f"  [DIFF] {row}")
        return 1 if mismatches else 0

    stamp = time.strftime("%Y%m%d_%H%M%S")
    grand, all_unresolved = Counter(), []
    for cn_path in sorted(args.cn_dir.glob("*.json")):
        data = json.loads(cn_path.read_text(encoding="utf-8"))
        stats, unresolved = Counter(), []
        changed = 0

        def fix(node):
            nonlocal changed
            if isinstance(node, dict):
                return {k: fix(v) for k, v in node.items()}
            if not isinstance(node, str) or "[" not in node:
                return node
            new = repair_text(node, stats, unresolved)
            if new != node:
                changed += 1
            return new

        entries = fix(data.get("entries", {}))
        if not changed and not unresolved:
            continue
        print(f"[{'write' if args.write else 'dry  '}] {cn_path.name[:52]:<52} "
              f"leaves={changed} {dict(stats)}")
        grand.update(stats)
        all_unresolved.extend(f"{cn_path.name}: {u}" for u in unresolved)
        if args.write and changed:
            backup = cn_path.parent.parent / "_backup" / f"brackets_{stamp}"
            backup.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cn_path, backup / cn_path.name)
            data["entries"] = entries
            cn_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8", newline="\n")

    print(f"\ntotal machine-word repairs: {sum(grand.values())}  {dict(grand)}")
    print(f"still containing CJK after repair: {len(all_unresolved)}")
    for row in all_unresolved[:20]:
        print(f"  [?] {row[:140]}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"repairs": dict(grand),
                                           "unresolved": all_unresolved}, ensure_ascii=False,
                                          indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
